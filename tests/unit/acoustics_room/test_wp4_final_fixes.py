"""Final WP4 regressions. All audio and room results are model-based proxies."""
import json

import numpy as np
import pytest
from scipy.io import wavfile
import yaml

from sensorsim.acoustics.room.prior import (
    RoomEnsemble, build_prior, known_source_observation, update_ensemble)
from sensorsim.acoustics.room.pipeline import run_no_sweep
from tests.unit.acoustics_room.test_room_prior import survey
from tests.unit.acoustics_room.test_wp4_review_fixes import playback


def test_f1_declared_preroll_is_not_a_hardware_delay_anchor():
    fs = 16000
    x, y = playback(pre=2.004, ppm=0)
    obs = known_source_observation(y, x, fs, source=(1, 1, 1), receiver=(3, 2, 1.4),
        source_group='noise', input_sha256='proxy', recording_offset_s=2.)
    assert obs.qualified, obs.reasons
    # Hold the acoustics exactly compatible, isolate the unaccounted 4 ms chain latency.
    predictions = [{**obs.values, 'direct_delay_s': .005} for _ in range(4)]
    updated = update_ensemble(build_prior(survey()), obs, predictions)
    assert updated.status == 'known-source', updated.evidence
    assert 'direct_delay_s' not in obs.values


def test_f2_uncalibrated_pipeline_uses_domestic_assumption(tmp_path):
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey(prior=dict(count=32)).model_dump(mode='json')))
    report = run_no_sweep(tmp_path/'room.yaml', tmp_path/'fit', fs=8000, evidence_label='proxy')
    ensemble = RoomEnsemble.model_validate_json((tmp_path/'fit'/'room_ensemble.json').read_text())
    assert report['calibration_status'] == 'assumed'
    assert min(m.late_t60_s[3] for m in ensemble.members) >= .25
    assert any('ROOM IS ASSUMED, NOT MEASURED' in w and 'played/noise.wav' in w
               and 'room_recordings/noise__room.wav' in w for w in report['warnings'])
    calibration = json.loads((tmp_path/'fit'/'calibration_ensemble.json').read_text())
    assert min(m['late_t60_s'][3] for m in calibration['members']) < .15


def test_f1_direct_api_and_pool_cannot_invent_hardware_authority():
    from sensorsim.acoustics.room.calibration_pool import pool_playback
    from sensorsim.acoustics.room.prior import CalibrationObservation
    from sensorsim.acoustics.room.render_prior import conflict_warning
    values = {'500.t30_s': .4, '1000.t30_s': .4, 'direct_delay_s': .009}
    obs = CalibrationObservation(kind='known-source', source_group='noise', input_sha256='proxy',
        source_xyz_m=(1, 1, 1), receiver_xyz_m=(3, 2, 1.4), qualified=True,
        values=values, uncertainty={k: .002 if k == 'direct_delay_s' else .1 for k in values})
    ensemble = build_prior(survey())
    predictions = [{**values, 'direct_delay_s': .005}]*4
    assert update_ensemble(ensemble, obs, predictions).status == 'known-source'
    anchored = obs.model_copy(update={'metadata': {'recording_offset_hardware_anchored': True}})
    post = update_ensemble(ensemble, pool_playback([anchored, anchored]), predictions)
    assert post.status == 'prior-conflict'
    assert 'hardware anchor' in conflict_warning(post)
    assert 'cannot repair a timing mismatch' in conflict_warning(post)
    # An unanchored contribution cannot acquire timing authority just by pooling.
    assert update_ensemble(ensemble, pool_playback([anchored, obs]), predictions).status == 'known-source'


def test_f1_hardware_anchor_requires_an_offset():
    x, y = playback(ppm=0)
    o = known_source_observation(y, x, 16000, source=(1, 1, 1), receiver=(3, 2, 1.4),
        source_group='noise', input_sha256='proxy', recording_offset_hardware_anchored=True)
    assert not o.qualified and 'requires recording_offset_s' in o.reasons[0]


def test_f1_pipeline_rejects_ambiguous_anchor_flags(tmp_path):
    import pytest
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey().model_dump(mode='json')))
    for flag in ('false', 1, None):
        (tmp_path/'calibration.yaml').write_text(yaml.safe_dump({
            'recording_offsets_s': {'noise.wav': 2.},
            'recording_offsets_hardware_anchored': {'noise.wav': flag}}))
        with pytest.raises(ValueError, match='booleans'):
            run_no_sweep(tmp_path/'room.yaml', tmp_path/'fit')
    (tmp_path/'calibration.yaml').write_text(yaml.safe_dump({
        'recording_offsets_hardware_anchored': {'noise.wav': True}}))
    with pytest.raises(ValueError, match='requires a recording_offsets_s'):
        run_no_sweep(tmp_path/'room.yaml', tmp_path/'fit')


def test_f2_only_assumed_ensembles_switch_distribution():
    import pytest
    from sensorsim.acoustics.room.render_prior import select_render_ensemble, DOMESTIC_T30_S
    e = build_prior(survey(prior=dict(count=4, t30_s=[[.1, .12]]*7)))
    domestic = select_render_ensemble(e)
    assert domestic.purpose == 'domestic-render' and domestic.status == 'assumed'
    assert select_render_ensemble(domestic) is domestic
    for member in domestic.members:
        assert all(lo <= value <= hi for value, (lo, hi) in zip(member.late_t60_s, DOMESTIC_T30_S))
        assert len(member.provenance['band_rationales']) == 7
    for status in ('known-source', 'weak-decay'):
        posterior = e.model_copy(update={'status': status})
        assert select_render_ensemble(posterior) is posterior
        assert max(m.late_t60_s[3] for m in posterior.members) < .15
    with pytest.raises(ValueError, match='prior-conflict'):
        select_render_ensemble(e.model_copy(update={'status': 'prior-conflict'}))
    with pytest.raises(ValueError, match='calibration_ensemble'):
        update_ensemble(domestic, None, [])


def test_f2_dataset_api_cannot_render_calibration_support(tmp_path):
    from sensorsim.acoustics.room.dataset import generate_dataset, DryClip, SamplingSpec
    from sensorsim.acoustics.room.augment import AugmentationConfig
    from sensorsim.acoustics.room.synth import SynthConfig
    e = build_prior(survey(prior=dict(count=4, t30_s=[[.1, .12]]*7)))
    clips = [DryClip('knock', np.random.default_rng(i).normal(0, .01, 800), 8000,
                     source_group=str(i)) for i in range(3)]
    manifest = generate_dataset(e, SamplingSpec(), clips, AugmentationConfig(), 4, 7,
        tmp_path/'dataset', fs=8000, synth_config=SynthConfig())
    rendered = RoomEnsemble.model_validate_json((manifest.parent/'ensemble.json').read_text())
    assert rendered.purpose == 'domestic-render'
    assert min(m.late_t60_s[3] for m in rendered.members) >= .25
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert {r['ensemble_member_hash'] for r in rows} <= {m.model_hash for m in rendered.members}


def _noise_and_takes(tmp_path, fs=8000):
    from tests.unit.acoustics_room.test_wp4_review_fixes import playback
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey(prior=dict(count=16, seed=4108)).model_dump(mode='json')))
    (tmp_path/'played').mkdir(); (tmp_path/'room_recordings').mkdir()
    x, y = playback(fs=fs, ppm=0)
    for name in ('noise', 'knock_01', 'doorbell_01'):
        wavfile.write(tmp_path/'played'/f'{name}.wav', fs, x.astype('float32'))
        wavfile.write(tmp_path/'room_recordings'/f'{name}__room.wav', fs, y.astype('float32'))


def _seed_that_ranks_noise_first(groups=('noise', 'knock_01', 'doorbell_01')):
    import hashlib
    rank = lambda seed: sorted(groups, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    return next(seed for seed in range(1000) if rank(seed)[0] == 'noise')


def test_automatic_holdout_never_takes_the_broadband_noise_playback(tmp_path):
    _noise_and_takes(tmp_path)
    seed = _seed_that_ranks_noise_first()                       # the old rule held out `noise` here
    report = run_no_sweep(tmp_path/'room.yaml', tmp_path/'fit', fs=8000, seed=seed, evidence_label='proxy')
    assert 'noise' in report['calibration_groups'] and len(report['holdout_groups']) == 1
    assert report['calibration_status'] == 'known-source' and report['room_warning'] is None


def test_the_assumed_room_warning_names_the_real_reason(tmp_path):
    _noise_and_takes(tmp_path)
    (tmp_path/'calibration.yaml').write_text(yaml.safe_dump(
        {'holdout_groups': ['noise', 'knock_01', 'doorbell_01'], 'evidence_label': 'proxy'}))
    with pytest.warns(RuntimeWarning, match='was held out for validation'):
        report = run_no_sweep(tmp_path/'room.yaml', tmp_path/'fit', fs=8000)
    assert report['calibration_status'] == 'assumed' and 'Add 30 s' not in report['room_warning']
    stereo = tmp_path/'stereo'; stereo.mkdir()
    (stereo/'room.yaml').write_text((tmp_path/'room.yaml').read_text())
    (stereo/'played').mkdir(); (stereo/'room_recordings').mkdir()
    from tests.unit.acoustics_room.test_wp4_review_fixes import playback
    x, y = playback(fs=8000, ppm=0)
    wavfile.write(stereo/'played'/'noise.wav', 8000, np.column_stack([x, x]).astype('float32'))
    wavfile.write(stereo/'room_recordings'/'noise__room.wav', 8000, y.astype('float32'))
    with pytest.warns(RuntimeWarning, match='No paired playback calibrated the room'):
        report = run_no_sweep(stereo/'room.yaml', stereo/'fit', fs=8000, evidence_label='proxy')
    assert 'noise: ' in report['room_warning'] and 'mono' in report['room_warning']
    empty = tmp_path/'empty'; empty.mkdir()
    (empty/'room.yaml').write_text((tmp_path/'room.yaml').read_text())
    report = run_no_sweep(empty/'room.yaml', empty/'fit', fs=8000, evidence_label='proxy')
    assert 'Add 30 s of broadband noise playback' in report['room_warning']


def test_a_holdout_shortfall_fails_with_the_counts(tmp_path):
    from tests.unit.acoustics_room.test_wp4_review_fixes import playback
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey(prior=dict(count=16, seed=4108)).model_dump(mode='json')))
    (tmp_path/'played').mkdir(); (tmp_path/'room_recordings').mkdir()
    x, y = playback(fs=8000, ppm=0)
    for name in ('noise', 'noise_2', 'noise_3', 'noise_4', 'noise_5', 'knock_01'):    # target 2, one eligible
        wavfile.write(tmp_path/'played'/f'{name}.wav', 8000, x.astype('float32'))
        wavfile.write(tmp_path/'room_recordings'/f'{name}__room.wav', 8000, y.astype('float32'))
    with pytest.raises(ValueError, match='needs 2 non-noise playback group.s. but only 1 exist'):
        run_no_sweep(tmp_path/'room.yaml', tmp_path/'fit', fs=8000, evidence_label='proxy')
    (tmp_path/'calibration.yaml').write_text(yaml.safe_dump({'holdout_groups': ['knock_01'], 'evidence_label': 'proxy'}))
    assert run_no_sweep(tmp_path/'room.yaml', tmp_path/'fit2', fs=8000)['holdout_groups'] == ['knock_01']
