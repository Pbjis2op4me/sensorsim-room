"""Frozen numerical budgets for scalar references, separate from room accuracy."""
import numpy as np
import pytest

from sensorsim.acoustics.room import metrics
from sensorsim.acoustics.room.dataset import DryClip, read_dry
from sensorsim.acoustics.room.source_reference import SourceReference


def test_level_shape_partition_uses_original_floored_bins():
    rng = np.random.default_rng(39)
    a = rng.normal(size=800)
    b = .23*a + .04*rng.normal(size=len(a))
    d = metrics.spectral_level_shape(a, b, 24000)
    assert d['lsd'] == metrics.log_spectral_distance(a, b, 24000)
    assert d['lsd']**2 == pytest.approx(d['spectral_bias_db']**2+d['spectral_shape_db']**2, abs=1e-12)
    assert abs(metrics.si_sdr(a, b)-metrics.si_sdr(a, b*11.7)) < 1e-6


def test_direct_pool_inverse_square_robustness_and_abstention():
    rows = []
    for i, (distance, gain) in enumerate(((1., .2), (2., .2), (3., 2.))):
        h = np.zeros(500)
        h[70] = gain/distance
        rows.append(metrics.direct_level_observation(h, 24000, distance_m=distance,
            direct_sample=70, first_reflection_delay_s=.005, identity=str(i),
            position=(i, 0, 0), noise_band_power={b:0. for b in (1000,2000,4000)}))
    p = metrics.pool_direct_levels(rows)
    assert p['validity']['status'] == 'partially validated'
    assert p['amplitude_scale'] == pytest.approx(.2, abs=1e-12)
    assert p['positions'] == 3
    assert p['leave_one_position_out_db'] == pytest.approx([-10., -10., 20.])
    # Repeats of one position cannot outvote the other positions.
    assert metrics.pool_direct_levels(rows+[{**rows[2], 'identity':'repeat'}])['amplitude_scale'] == p['amplitude_scale']
    assert metrics.pool_direct_levels(rows[:1])['amplitude_scale'] is None
    bad = metrics.direct_level_observation(np.ones(500), 24000, distance_m=1.,
        direct_sample=70, first_reflection_delay_s=.005, identity='shadow', position=(9,0,0),
        obstructed=True, noise_band_power={b:0. for b in (1000,2000,4000)})
    assert not bad['eligible'] and 'obstructed' in bad['reason']
    unknown = metrics.direct_level_observation(h, 24000, distance_m=1.,
        direct_sample=70, first_reflection_delay_s=None, identity='unknown', position=(9,0,0),
        noise_band_power={b:0. for b in (1000,2000,4000)})
    assert not unknown['eligible'] and 'reflection' in unknown['reason']


def test_pa_claim_requires_calibration_even_for_float_arrays():
    with pytest.raises(ValueError, match='calibration'):
        read_dry(DryClip('tone', np.ones(16)*.01, 4000,
            source_reference=SourceReference(units='Pa')), 4000)


def test_played_digital_and_effective_reference_remain_distinct():
    digital = metrics.source_level_metadata(SourceReference(target='played_digital', units='peak_fs'))
    effective = metrics.source_level_metadata(SourceReference(target='effective_source', units='peak_fs'))
    assert not digital['effective_source']
    assert effective['effective_source']
    assert digital['source_reference']['target'] == 'played_digital'


def test_complete_mixture_export_gain_preserves_snr_and_is_reversible():
    from sensorsim.acoustics.room.augment import TrainingLevelPolicy, training_export_level
    rng = np.random.default_rng(21)
    clean = rng.normal(size=200)*.1
    noise = rng.normal(size=200)*.01
    mix = clean+noise
    policy = TrainingLevelPolicy(mode='rms', target_rms_dbfs=-24.)
    y, info = training_export_level(mix, policy, seed=12)
    assert 20*np.log10(np.sqrt(np.mean(y*y))) == pytest.approx(-24., abs=1e-12)
    scale = info['amplitude_scale']
    assert info['relative_levels_preserved'] is False
    assert 'physical_relative_validation' not in info
    np.testing.assert_allclose(y/scale, mix, atol=1e-15)
    assert np.sum((clean*scale)**2)/np.sum((noise*scale)**2) == pytest.approx(np.sum(clean**2)/np.sum(noise**2))
    unchanged, _ = training_export_level(mix, TrainingLevelPolicy(), seed=12)
    np.testing.assert_array_equal(unchanged, mix)
    random = TrainingLevelPolicy(mode='random_gain', gain_db=(-6., 6.))
    np.testing.assert_array_equal(training_export_level(mix, random, seed=12)[0], training_export_level(mix, random, seed=12)[0])
    with pytest.raises(ValueError, match='silent'):
        training_export_level(np.zeros(8), policy, seed=0)
