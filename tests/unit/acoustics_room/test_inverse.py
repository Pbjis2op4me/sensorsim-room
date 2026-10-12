"""Opt-in Tikhonov numerics and paired-comparison contracts."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import signal

from sensorsim.acoustics.room.dry import DryConfig, extract_dry
from sensorsim.acoustics.room.metrics import si_sdr


def test_tikhonov_scalar_ridge_and_gain_cap():
    x = np.random.default_rng(4).normal(size=900)
    config = DryConfig(inverse_mode='tikhonov', noise_suppression=False)
    for gain in (2., .001):
        h = np.r_[gain, np.zeros(30)]
        result = extract_dry({'M': gain*x}, {'M': h}, np.zeros(300), fs=8000,
                             case='A', config=config,
                             metadata={'M': {'direct_start_sample': 0, 'direct_end_sample': 1}})
        expected = x*gain*min(1/(gain*(1+config.ridge)), 10**(config.max_gain_db/20))
        np.testing.assert_allclose(result.samples, expected, atol=1e-10, rtol=0)
        assert result.report['max_inverse_noise_gain_db'] <= 20.00001
    with pytest.raises(ValueError, match='inverse_mode'):
        DryConfig(inverse_mode='repeat_tikhonov')


def test_tikhonov_nonminimum_phase_and_block_equivalence():
    h = np.zeros(600); h[8] = .4; h[40] = 1.; h[400] = .1
    x = np.random.default_rng(34).normal(size=1800)
    config = DryConfig(inverse_mode='tikhonov', noise_suppression=False)
    args = dict(recordings={'M': signal.convolve(x, h)}, rirs={'M': h},
                noise_floor=np.zeros(500), fs=8000, case='A',
                metadata={'M': {'direct_start_sample': 8, 'direct_end_sample': 9}})
    result = extract_dry(**args, config=config)
    blocked = extract_dry(**args, config=replace(config, block_seconds=.047))
    assert si_sdr(np.pad(x, (0, len(h)-1)), result.samples) >= 30
    np.testing.assert_allclose(result.samples, blocked.samples, atol=1e-10, rtol=0)


def test_comparison_uses_the_same_measured_response():
    from tests.validation.bench_room_kit_e2e.inverse_run import Capture, CONFIGS, array_hash
    x = np.random.default_rng(8).normal(size=1000)
    h = np.r_[1., np.zeros(29), .3]
    recordings = {'M': signal.convolve(x, h)}
    kwargs = dict(fs=8000, case='A', metadata={'M': {'direct_start_sample': 0, 'direct_end_sample': 1}})
    capture = Capture()
    result = capture.dry(recordings, {'M': h}, np.zeros(300), **kwargs, config=DryConfig())
    comparison = capture.takes[array_hash(result.samples)]['tikhonov'][0]
    expected = extract_dry(recordings, {'M': h}, np.zeros(300), **kwargs, config=CONFIGS['tikhonov'])
    np.testing.assert_array_equal(comparison, expected.samples)


def test_sidecar_missing_predictions_are_failures_not_unavailable_truth():
    from tests.validation.bench_room_closedloop.scorecard import row
    from tests.validation.bench_room_kit_e2e.inverse_run import summarize
    ctx=dict(scenario='test',tier='T0',engine='control',impairment='clean',eval_kind='real_sound',src='S1',mic='M1',band_hz=None)
    rows=[row(ctx,metric='pre',truth=-float('inf'),reconstructed=-float('inf'),gate=-30,unit='dB',upper=True),
          row(ctx,metric='missing',truth=float('inf'),reconstructed=None,gate=15,unit='dB',lower=True),
          row(ctx,metric='undefined',truth=None,reconstructed=None,gate=5,unit='%')]
    for r in rows:
        r.update(method='legacy',take='same')
    stats=summarize(rows,[])['T0']['legacy']['metrics']
    assert stats['pre']['unavailable']==0 and stats['pre']['pass_count']==1
    assert stats['missing']['unavailable']==0 and stats['missing']['missing_estimate']==1
    assert stats['undefined']['unavailable']==1
