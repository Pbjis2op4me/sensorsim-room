"""Generated development fixtures for empirical-bed routing, not real audio."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from sensorsim import iot_mic
from sensorsim.evaluation import fa_rate
from sensorsim.negatives.composer import Event, RenderPath
from sensorsim.negatives.procedural import generate
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig


def inputs(duration=12):
    fs = 8000
    path = RenderPath(RoomModel(shoebox_dims_m=(6, 5, 3)), (4, 3, 1.5),
        iot_mic.PROFILES['pi5_inmp441/1'], 'pi5_inmp441/1', SynthConfig(late=False, max_order=0))
    bed = np.random.default_rng(14).integers(-30, 31, fs*duration)/32768
    src = generate('vacuum', fs=fs, duration_s=2, seed=5)
    ev = Event(src, 3., (1, 1, 1), spl_db=70.)
    model = SimpleNamespace(features=SimpleNamespace(n=8000), classes=['knock'],
        meta={'features': {'fs': fs}, 'detector': dict(hop_s=.25, on={'knock': .5},
              off_ratio=.6, smooth_hops=3, min_hops=2)})
    model.scores = lambda windows: np.minimum(1., np.std(np.atleast_2d(windows), axis=1))[:, None]
    return model, bed, [ev], path


def test_overlay_uses_shared_mixer_and_keeps_recorded_bed(monkeypatch):
    from sensorsim.negatives import composer
    model, bed, events, path = inputs()
    calls = []
    original = composer.mix_events
    def spy(background, increments, rng, cfg, **kw):
        calls.append((background.copy(), len(increments), cfg['background_domain']))
        return original(background, increments, rng, cfg, **kw)
    monkeypatch.setattr(composer, 'mix_events', spy)
    context = fa_rate.overlay_context(model, bed, events, path=path, seed=7,
        capture=dict(profile_id=path.device_id, deployed_device=True, target_free=True,
                     capture_device='proxy-pi', source_kind='proxy', parent_id='recording-1',
                     recording_sha256='a'*64, segment_samples=[0, len(bed)]))
    assert calls[0][1:] == (1, 'device_recorded')
    np.testing.assert_array_equal(calls[0][0], bed)
    np.testing.assert_array_equal(context.audio[:2*8000], bed[:2*8000])
    assert not np.array_equal(context.audio, bed)
    assert context.provenance['whole_episode'] is True
    assert context.provenance['evidence_label'] == 'proxy'
    assert context.events[0]['tail_truncated_samples'] == 0
    assert context.provenance['rendered_events'][0]['active_samples'] > 0


@pytest.mark.parametrize('defect', ['short', 'empty', 'profile', 'target_free', 'constant'])
def test_overlay_refuses_bad_controls(defect):
    model, bed, events, path = inputs()
    capture=dict(profile_id=path.device_id, deployed_device=True, target_free=True,
                 capture_device='proxy-pi', source_kind='proxy', parent_id='recording-1',
                 recording_sha256='a'*64, segment_samples=[0, len(bed)])
    if defect == 'short':
        events = [replace(events[0], onset_s=11.)]
    elif defect == 'empty':
        events = []
    elif defect == 'constant':
        bed = np.zeros_like(bed)
    else:
        capture['profile_id' if defect == 'profile' else defect] = False
    with pytest.raises(ValueError):
        fa_rate.overlay_context(model, bed, events, path=path, seed=7, capture=capture)


class ScoreModel:
    classes = ['knock']
    features = SimpleNamespace(n=1)
    meta = {'features': {'fs': 1}, 'detector': dict(hop_s=1., on={'knock': .5},
        off_ratio=.6, smooth_hops=1, min_hops=1)}
    def scores(self, x):
        return np.atleast_2d(x).mean(axis=1)[:, None]


def test_bed_crossing_off_threshold_changes_whole_episode_counts():
    model = ScoreModel()
    low = np.r_[0., .8, .8, .2, .2, .8, .8, 0., 0., 0.]
    high = low.copy(); high[3:5] = .4
    assert len(fa_rate.score_audio(model, low, require_closed_context=True)) == 2
    assert len(fa_rate.score_audio(model, high, require_closed_context=True)) == 1
    diagnostic = fa_rate.bed_occupancy(model, high)
    assert diagnostic['by_class']['knock']['fraction_at_or_above_off'] == .6
    assert diagnostic['scored_hops'] == 10


def test_deployment_smoothing_and_closed_tail_are_required():
    model = ScoreModel()
    short = np.r_[0., 0., .8, 0., 0., 0.]
    assert len(fa_rate.score_audio(model, short, require_closed_context=True)) == 1
    assert not fa_rate.score_audio(model, short, thresholds={'smooth_hops': 3}, require_closed_context=True)
    with pytest.raises(ValueError, match='active tracker'):
        fa_rate.score_audio(model, np.r_[0., .8, .8], require_closed_context=True)


@pytest.fixture
def overlay_manifest(tmp_path):
    import json
    import wave
    model, _, _, path = inputs()
    def write(name, seed):
        pcm = np.random.default_rng(seed).integers(-30, 31, 8000*16, dtype=np.int16)
        with wave.open(str(tmp_path/name), 'wb') as wav:
            wav.setparams((1, 2, 8000, 0, 'NONE', 'PCM'))
            wav.writeframes(pcm.tobytes())
    for name, seed in [('floor.wav', 3), ('bed.wav', 4), ('cal.wav', 5)]:
        write(name, seed)
    (tmp_path/'room.json').write_text(path.room.model_dump_json())
    manifest = dict(source_kind='proxy', calibration_streams=[dict(path='cal.wav', parent_id='cal')],
        floor_streams=[dict(path='floor.wav', parent_id='floor')],
        overlay_path=dict(room='room.json', receiver_xyz_m=path.receiver_xyz_m,
                          device_id=path.device_id, config=dict(late=False, max_order=0)),
        challenges=[dict(family='mixed-scene', context='isolated', overlay=dict(
            background=dict(path='bed.wav', parent_id='bed', profile_id=path.device_id,
                deployed_device=True, target_free=True, capture_device='proxy-pi', source_kind='proxy'),
            duration_s=12., start_s=0., seed=9, whole_episode=True,
            events=[dict(procedural=dict(family='vacuum', duration_s=2., seed=71),
                onset_s=3., spl_db=65., source_xyz_m=[1, 1, 1], whole_episode=True)]))],
        context_prevalence_per_hour={'mixed-scene': 2}, context_weights={'isolated': 1},
        bootstrap_replicates=100)
    for entry in [*manifest['calibration_streams'], *manifest['floor_streams'],
                  manifest['challenges'][0]['overlay']['background']]:
        entry.update(recording_id=entry['parent_id'], sample_range=[0, 8000*16], sample_rate_hz=8000)
    return model, manifest, tmp_path


def test_overlay_cli_routes_real_mixture_and_incidence_sensitivity(overlay_manifest, monkeypatch):
    import json
    model, manifest, root = overlay_manifest
    monkeypatch.setattr(fa_rate.edge, 'Model', lambda _: model)
    (root/'model.npz').write_bytes(b'proxy model identifier')
    (root/'manifest.json').write_text(json.dumps(manifest))
    result = fa_rate._score_manifest(str(root/'model.npz'), str(root/'manifest.json'))
    assert result['overlays'][0]['policy'] == 'empirical-capture-overlay/1'
    assert result['detector_settings']['smooth_hops'] == 3
    assert len(result['recording_partition_audit']['calibration']) == 1
    for row in result['composite_estimates']:
        assert row['evidence_label'] == 'model-based'
        sensitivity = row['classes']['knock']['family_incidence_sensitivity']['mixed-scene']
        assert {'0.3', '3.0'} <= sensitivity.keys()
    # Small banks must stay inconclusive, even on an all-zero output control.
    assert result['challenge_results']['families']['mixed-scene']['knock']['status'].startswith('inconclusive')


@pytest.mark.parametrize('defect', ['copy', 'parent', 'undeclared', 'units', 'truncated', 'duplicate'])
def test_overlay_partition_and_episode_bad_controls(overlay_manifest, defect):
    import copy
    import shutil
    model, manifest, root = overlay_manifest
    if defect == 'copy':
        shutil.copyfile(root/'cal.wav', root/'bed.wav')
    elif defect == 'parent':
        manifest['challenges'][0]['overlay']['background'].update(parent_id='cal', recording_id='cal')
    elif defect == 'undeclared':
        del manifest['calibration_streams']
    elif defect == 'units':
        manifest['challenges'][0]['overlay']['background']['profile_id'] = 'wrong-device'
    elif defect == 'truncated':
        manifest['challenges'][0]['overlay']['events'][0]['whole_episode'] = False
    else:
        manifest['challenges'].append(copy.deepcopy(manifest['challenges'][0]))
    with pytest.raises(ValueError):
        fa_rate._prepare_overlays(model, manifest, root)


def test_reused_dry_episode_links_different_backgrounds(overlay_manifest):
    import copy
    model, manifest, root = overlay_manifest
    second = copy.deepcopy(manifest['challenges'][0])
    second['overlay']['start_s'] = 2.
    second['overlay']['seed'] = 81
    manifest['challenges'].append(second)
    rows, _ = fa_rate._prepare_overlays(model, manifest, root)
    assert rows[0]['parent_id'] == rows[1]['parent_id']
    assert rows[0]['overlay_provenance']['audio_sha256'] != rows[1]['overlay_provenance']['audio_sha256']


def test_superposition_uses_one_waveform_not_summed_detections():
    model = ScoreModel()
    first = np.r_[0., 0., .28, .28, 0., 0.]
    second = np.r_[0., 0., .34, .34, 0., 0.]
    assert not fa_rate.score_audio(model, first)
    assert not fa_rate.score_audio(model, second)
    assert len(fa_rate.score_audio(model, first+second, require_closed_context=True)) == 1
