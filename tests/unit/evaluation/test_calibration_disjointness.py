"""Disclosed development controls for calibration leakage; synthetic proxy only.

P-OV-1, P-OV-1b and P-OV-2a are regressions of the int-eval verifier's
counterexamples. Slice arithmetic and hashlib are the oracle, not the audit.
"""
import copy
import hashlib
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import edge
from sensorsim.evaluation import fa_rate

FS = 8000


def write(root, name, pcm):
    with wave.open(str(root / name), 'wb') as wav:
        wav.setparams((1, 2, FS, 0, 'NONE', 'PCM'))
        wav.writeframes(np.asarray(pcm, '<i2').tobytes())


def record(name, n, *, identity=None, start=0):
    return dict(path=name, recording_id=identity or name, parent_id=identity or name,
                sample_range=[start, start+n], sample_rate_hz=FS)


@pytest.fixture
def bank(tmp_path, monkeypatch):
    from sensorsim.acoustics.room.model import RoomModel
    rng = np.random.default_rng(5)
    pcm = rng.integers(-30, 31, 16*FS, dtype=np.int16)
    for name, data in [('cal.wav', pcm), ('floor.wav', rng.integers(-30, 31, len(pcm), dtype=np.int16)),
                       ('bed.wav', rng.integers(-30, 31, len(pcm), dtype=np.int16))]:
        write(tmp_path, name, data)
    model = SimpleNamespace(features=SimpleNamespace(n=FS), classes=['knock'],
        meta={'features': {'fs': FS}, 'detector': dict(hop_s=.25, on={'knock': .5},
              off_ratio=.6, smooth_hops=3, min_hops=2)})
    calls = []
    def scores(windows):
        calls.append(len(windows))
        return np.zeros((len(np.atleast_2d(windows)), 1), np.float32)
    model.scores = scores
    monkeypatch.setattr(edge, 'Model', lambda _: model)
    (tmp_path / 'model.npz').write_bytes(b'generated score proxy')
    (tmp_path / 'room.json').write_text(RoomModel(shoebox_dims_m=(6, 5, 3)).model_dump_json())
    manifest = dict(source_kind='proxy', calibration_streams=[record('cal.wav', len(pcm))],
        floor_streams=[record('floor.wav', len(pcm))],
        challenges=[dict(record('bed.wav', len(pcm)), family='impact', context='isolated')],
        context_prevalence_per_hour={'impact': 2}, context_weights={'isolated': 1}, bootstrap_replicates=100)
    return tmp_path, pcm, model, manifest, calls


def overlay(manifest):
    manifest['overlay_path'] = dict(room='room.json', receiver_xyz_m=[4, 3, 1.5],
        device_id='pi5_inmp441/1', config=dict(late=False, max_order=0))
    background = {k: v for k, v in manifest['challenges'][0].items() if k not in ('family', 'context')}
    background.update(profile_id='pi5_inmp441/1', deployed_device=True, target_free=True,
                      capture_device='proxy-pi', source_kind='proxy')
    manifest['challenges'] = [dict(family='impact', context='isolated', overlay=dict(
        background=background, duration_s=12., start_s=0., seed=9, whole_episode=True,
        events=[dict(procedural=dict(family='vacuum', duration_s=2., seed=71), onset_s=3.,
                     spl_db=65., source_xyz_m=[1, 1, 1], whole_episode=True)]))]


def score(bank):
    root, _, _, manifest, _ = bank
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return fa_rate._score_manifest(str(root / 'model.npz'), str(root / 'manifest.json'))


@pytest.mark.parametrize('route', ['plain', 'overlay'])
@pytest.mark.parametrize('role', ['floor', 'context'])
@pytest.mark.parametrize('offset', [0, 137])
def test_p_ov_1_and_1b_renamed_calibration_excerpts_refused(bank, route, role, offset):
    root, pcm, _, manifest, calls = bank
    # Independent oracle: literal slice equality; offset 137 is not block aligned.
    cut = pcm[FS+offset:16*FS-29]
    name = 'floor.wav' if role == 'floor' else 'bed.wav'
    write(root, name, cut)
    entry = manifest['floor_streams' if role == 'floor' else 'challenges'][0]
    entry.update(record(name, len(cut), identity='undeclared-new-recording'))
    if route == 'overlay':
        overlay(manifest)
    with pytest.raises(ValueError, match='calibration') as exc:
        score(bank)
    assert name in str(exc.value) and 'cal.wav' in str(exc.value)
    assert '[' in str(exc.value) and ('bed' if route == 'overlay' and role == 'context' else role) in str(exc.value)
    assert not calls, 'leaking audio reached the detector'


@pytest.mark.parametrize('route', ['plain', 'overlay'])
@pytest.mark.parametrize('role', ['floor', 'context'])
def test_p_ov_2a_saved_digest_checked_without_calibration_declaration(bank, route, role):
    root, pcm, model, manifest, calls = bank
    # Historical format: no provenance record, only the saved decoded digest.
    digest = hashlib.sha256((pcm.astype(np.float32)/32768).astype('<f4').tobytes()).hexdigest()
    model.meta['detector']['calibration'] = {'calibration_audio_sha256': digest}
    manifest['calibration_streams'] = []
    name = 'floor.wav' if role == 'floor' else 'bed.wav'
    write(root, name, pcm)
    if route == 'overlay':
        overlay(manifest)
    with pytest.raises(ValueError, match='calibration') as exc:
        score(bank)
    assert name in str(exc.value)
    assert not calls


@pytest.mark.parametrize('route', ['plain', 'overlay'])
def test_known_good_adjacent_recording_splits_reach_detector(bank, route):
    root, pcm, _, manifest, calls = bank
    rng = np.random.default_rng(44)
    full = np.r_[pcm, rng.integers(-30, 31, 32*FS, dtype=np.int16)]
    for i, name in enumerate(['cal.wav', 'floor.wav', 'bed.wav']):
        write(root, name, full[i*16*FS:(i+1)*16*FS])
        entry = manifest[['calibration_streams', 'floor_streams', 'challenges'][i]][0]
        entry.update(record(name, 16*FS, identity='one-session', start=i*16*FS))
    if route == 'overlay':
        overlay(manifest)
    result = score(bank)
    assert calls
    assert result['recording_partition_audit'] is not None
    assert result['composite_estimates'][0]['evidence_label'] == 'model-based'


@pytest.mark.parametrize('route', ['plain', 'overlay'])
def test_one_sample_declared_overlap_refused_even_when_content_changed(bank, route):
    _, pcm, _, manifest, calls = bank
    # No shared content: this control isolates the interval mechanism, at one sample.
    manifest['floor_streams'][0].update(record('floor.wav', len(pcm), identity='cal.wav', start=len(pcm)-1))
    if route == 'overlay':
        overlay(manifest)
    with pytest.raises(ValueError, match='overlap') as exc:
        score(bank)
    assert 'cal.wav' in str(exc.value) and 'floor.wav' in str(exc.value)
    assert not calls


@pytest.mark.parametrize('field', ['recording_id', 'sample_range', 'sample_rate_hz'])
def test_missing_record_provenance_refused(bank, field):
    del bank[3]['floor_streams'][0][field]
    with pytest.raises(ValueError, match=field):
        score(bank)


def test_legacy_model_warns_and_still_scores_disjoint_data(bank):
    with pytest.warns(UserWarning, match='calibration'):
        result = score(bank)
    assert result['recording_partition_audit']['warnings']
    assert bank[-1]


@pytest.mark.parametrize('route', ['plain', 'overlay'])
@pytest.mark.parametrize('role', ['floor', 'context'])
def test_saved_record_catches_crop_without_access_to_original(bank, route, role):
    root, pcm, model, manifest, calls = bank
    saved = edge.calibration_audio_record(pcm, FS, recording_id='original',
                                         sample_start=4000, path='cal.wav')
    model.meta['detector']['calibration'] = dict(calibration_audio_records=[saved])
    manifest['calibration_streams'] = []
    (root / 'cal.wav').unlink()  # The audit must use the model, not reopen the path.
    cut = pcm[FS+137:16*FS-29]
    name = 'floor.wav' if role == 'floor' else 'bed.wav'
    write(root, name, cut)
    entry = manifest['floor_streams' if role == 'floor' else 'challenges'][0]
    entry.update(record(name, len(cut), identity='renamed'))
    if route == 'overlay':
        overlay(manifest)
    with pytest.raises(ValueError, match='exact block content overlap') as exc:
        score(bank)
    # First intact calibration block starts at local 16000 / global 20000.
    assert 'samples [20000, 28000)' in str(exc.value)
    assert 'samples [7863, 15863)' in str(exc.value)
    assert not calls


@pytest.mark.parametrize('route', ['plain', 'overlay'])
def test_known_good_saved_record_disjoint_splits_without_original(bank, route):
    root, pcm, model, manifest, calls = bank
    saved = edge.calibration_audio_record(pcm, FS, recording_id='session', path='cal.wav')
    model.meta['detector']['calibration'] = dict(calibration_audio_records=[saved])
    manifest['calibration_streams'] = []
    (root / 'cal.wav').unlink()
    for i, entry in enumerate([manifest['floor_streams'][0], manifest['challenges'][0]], start=1):
        entry.update(record(entry['path'], len(pcm), identity='session', start=i*len(pcm)))
    if route == 'overlay':
        overlay(manifest)
    result = score(bank)
    assert calls and not result['recording_partition_audit']['warnings']


@pytest.mark.parametrize('value', [[0, 0], [0, 1], [True, 128001], [0., 128000.], [-1, 127999], None])
def test_bad_ranges_refused(bank, value):
    bank[3]['floor_streams'][0]['sample_range'] = value
    with pytest.raises(ValueError, match='sample_range'):
        score(bank)
    assert not bank[-1]


@pytest.mark.parametrize('defect', ['constant', 'empty', 'wrong_rate', 'duplicate_floor', 'duplicate_context'])
def test_invalid_or_duplicate_exposure_refused(bank, defect):
    root, pcm, _, manifest, calls = bank
    if defect in ('constant', 'empty'):
        write(root, 'floor.wav', np.zeros(0 if defect == 'empty' else len(pcm), np.int16))
    elif defect == 'wrong_rate':
        manifest['floor_streams'][0]['sample_rate_hz'] = 16000
    else:
        key = 'floor_streams' if defect == 'duplicate_floor' else 'challenges'
        manifest[key].append(copy.deepcopy(manifest[key][0]))
    with pytest.raises(ValueError):
        score(bank)
    assert not calls


def test_fingerprint_oracle_and_prefix_collision(bank):
    from sensorsim.evaluation.recordings import _block_match
    _, pcm, _, _, _ = bank
    x = pcm.astype(np.float32)/32768
    saved = edge.calibration_audio_record(x, FS)
    block = saved['fingerprint']['blocks'][0]
    assert block['sha256'] == hashlib.sha256(x[:FS].astype('<f4').tobytes()).hexdigest()
    assert saved['sample_range'] == [0, len(x)]
    # Matching the index alone must not cause refusal.
    different = -x.copy()
    different[:8] = x[:8]
    assert _block_match(different, saved) is None
    assert _block_match(x[FS+137:], saved) == (2*FS, FS-137, FS)


def test_fingerprint_chunk_boundary_and_constant_block_policy():
    from sensorsim.evaluation.recordings import _block_match
    rng = np.random.default_rng(712)
    reference = rng.integers(-200, 201, 32).astype(np.float32)/32768
    saved = edge.calibration_audio_record(reference, 32)
    for offset in (262140, 262144):
        candidate = rng.integers(-200, 201, offset+40).astype(np.float32)/32768
        candidate[offset:offset+32] = reference
        assert _block_match(candidate, saved) == (0, offset, 32)
    constant = edge.calibration_audio_record(np.zeros(65), 32)
    assert constant['fingerprint']['blocks'] == []
    assert constant['fingerprint']['constant_blocks_omitted'] == 2
    assert constant['fingerprint']['trailing_samples_omitted'] == 1


def test_documented_exact_fingerprint_limit_is_not_a_robust_claim(bank):
    from sensorsim.evaluation.recordings import _block_match
    x = bank[1].astype(np.float32)/32768
    saved = edge.calibration_audio_record(x, FS)
    assert _block_match(x[FS+137:]*.5, saved) is None


@pytest.mark.parametrize('defect', ['algorithm', 'empty_blocks', 'rate', 'missing_fingerprint', 'digest'])
def test_malformed_versioned_model_record_never_silently_falls_back(bank, defect):
    _, pcm, model, manifest, calls = bank
    saved = edge.calibration_audio_record(pcm, FS)
    if defect == 'algorithm':
        saved['fingerprint']['algorithm'] = 'unavailable'
    elif defect == 'empty_blocks':
        saved['fingerprint']['blocks'] = []
    elif defect == 'rate':
        saved['sample_rate_hz'] = FS*2
    elif defect == 'missing_fingerprint':
        del saved['fingerprint']
    else:
        saved['decoded_sha256'] = 'bad'
    model.meta['detector']['calibration'] = dict(calibration_audio_records=[saved])
    manifest['calibration_streams'] = []
    with pytest.raises(ValueError):
        score(bank)
    assert not calls


@pytest.mark.parametrize('roles', [('calibration', 'floor'), ('calibration', 'context'),
    ('calibration', 'bed'), ('floor', 'context'), ('floor', 'bed'), ('context', 'bed')])
@pytest.mark.parametrize('overlap', [0, 1])
def test_all_cross_role_range_pairs_at_one_sample_boundary(bank, roles, overlap):
    from sensorsim.evaluation.recordings import RecordingAudit
    root, pcm, model, _, _ = bank
    audit = RecordingAudit(model, root)
    first = record('cal.wav', len(pcm), identity='session')
    second = record('floor.wav', len(pcm), identity='session', start=len(pcm)-overlap)
    audit.read(first, roles[0])
    if overlap:
        with pytest.raises(ValueError, match='overlapping sample ranges'):
            audit.read(second, roles[1])
    else:
        actual, _ = audit.read(second, roles[1])
        np.testing.assert_array_equal(actual, edge.read_wav(str(root / 'floor.wav'), FS))


def test_mutant_probe_confirms_fingerprint_hook_changes_refusal(bank, monkeypatch):
    from sensorsim.evaluation import recordings
    root, pcm, _, manifest, calls = bank
    cut = pcm[FS+137:16*FS-29]
    write(root, 'floor.wav', cut)
    manifest['floor_streams'][0].update(record('floor.wav', len(cut), identity='renamed'))
    reached = []
    original = recordings._block_match
    def spy(audio, saved):
        result = original(audio, saved)
        reached.append((saved['role'], result))
        return result
    monkeypatch.setattr(recordings, '_block_match', spy)
    with pytest.raises(ValueError, match='exact block content overlap'):
        score(bank)
    assert ('calibration', (16000, 7863, 8000)) in reached and not calls
    # Deliberately broken development control. With identity concealed and the
    # whole-file digest changed, disabling block matching makes this crop pass.
    monkeypatch.setattr(recordings, '_block_match', lambda *a: None)
    assert score(bank)['recording_partition_audit']
    assert calls
