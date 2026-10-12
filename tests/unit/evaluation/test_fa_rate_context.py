"""Regression tests for counting exact tracker outputs per nuisance context."""
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim.evaluation import fa_rate


class ScoreModel:
    classes = ['knock']
    features = SimpleNamespace(n=4)
    meta = {'features': {'fs': 4}, 'detector': {
        'hop_s': .25, 'on': {'knock': .5}, 'off_ratio': .6,
        'smooth_hops': 1, 'min_hops': 2,
    }}

    def scores(self, windows):
        return np.atleast_2d(windows)[:, :1]


def wav_file(tmp_path, name, pulses):
    samples = np.zeros(48, dtype='<i2')
    for start, end in pulses:
        samples[start:end] = 30000
    path = tmp_path / name
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(4)
        wav.writeframes(samples.tobytes())
    return path


def count_row(i, count, *, parent=None, label='knock', family='impact', bound=8):
    return {'family': family, 'class': label, 'context_group': 'clustered',
            'context_id': str(i), 'parent_id': str(parent if parent is not None else i),
            'confirmed_outputs': count, 'max_confirmed_outputs': bound,
            'source_instances': 3, 'evidence_label': 'model-based'}


def test_context_scoring_counts_merged_output_once(tmp_path):
    path = wav_file(tmp_path, 'overlap.wav', [(8, 10)])
    rows = fa_rate.score_challenge_records(ScoreModel(), [{
        'path': path, 'family': 'impact', 'context': 'overlapping',
        'events': [{'start_s': 2, 'end_s': 2.5}] * 3,
    }])
    assert len(rows) == 1
    assert rows[0]['confirmed_outputs'] == 1
    assert rows[0]['matched_instances'] == 3
    assert rows[0]['evidence_label'] == 'model-based'


def test_context_scoring_keeps_multiple_outputs_from_one_source(tmp_path):
    path = wav_file(tmp_path, 'split.wav', [(8, 10), (20, 22)])
    rows = fa_rate.score_challenge_records(ScoreModel(), [{
        'path': path, 'family': 'impact', 'context': 'isolated',
        'events': [{'start_s': 2, 'end_s': 6}],
    }])
    assert rows[0]['confirmed_outputs'] == 2
    assert rows[0]['matched_instances'] == 1


def test_context_requires_a_closed_tracker_guard(tmp_path):
    path = wav_file(tmp_path, 'truncated.wav', [(8, 48)])
    with pytest.raises(ValueError, match='trailing guard'):
        fa_rate.score_challenge_records(ScoreModel(), [{
            'path': path, 'family': 'impact', 'events': [{'start_s': 2, 'end_s': 12}],
        }])


def test_count_rate_uses_context_incidence_and_all_outputs():
    rows = [count_row(i, 3 if i % 2 else 0) for i in range(100)]
    result = fa_rate.estimate_composite(
        fa_rate.summarize_floor([], 3600), rows, {'impact': 2},
        context_weights={'clustered': 1}, bootstrap_replicates=500,
    )
    assert result['classes']['knock']['rate_per_hour'] == 3
    assert result['classes']['knock']['sensitivity']['3.0']['rate_per_hour'] == 9
    assert result['prevalence_unit'] == 'whole-contexts/hour'
    assert result['dispersion_corrected']


def test_parent_bootstrap_preserves_burst_dependence_and_overdispersion():
    rows = [count_row(i, 6 if i // 5 % 2 else 0, parent=i // 5) for i in range(100)]
    kwargs = dict(context_weights={'clustered': 1}, bootstrap_replicates=2000, bootstrap_seed=42)
    clustered = fa_rate.summarize_challenges(rows, **kwargs)['families']['impact']['knock']
    independent = fa_rate.summarize_challenges(
        [{**r, 'parent_id': r['context_id']} for r in rows], **kwargs)['families']['impact']['knock']
    lo, hi = clustered['interval_95']
    ilo, ihi = independent['interval_95']
    assert lo <= 3 <= hi
    assert hi - lo > ihi - ilo
    assert clustered['by_context']['clustered']['variance_to_mean'] > 1


def test_zero_context_counts_do_not_bootstrap_to_certainty():
    result = fa_rate.summarize_challenges(
        [count_row(i, 0) for i in range(100)], context_weights={'clustered': 1},
        bootstrap_replicates=100,
    )['families']['impact']['knock']
    assert result['interval_95'][0] == 0
    assert result['interval_95'][1] > 0
    assert result['status'] == 'scored'


def test_reject_legacy_binary_rows_and_missing_incidence():
    floor = fa_rate.summarize_floor([], 3600)
    with pytest.raises(ValueError, match='rescore whole contexts'):
        fa_rate.estimate_composite(floor, [{'success': True}], {'impact': 2})
    with pytest.raises(ValueError, match='match the scored families'):
        fa_rate.estimate_composite(floor, [count_row(i, i % 2) for i in range(10)],
                                  {'impact': 2, 'missing': 5}, context_weights={'clustered': 1})


def test_reject_duplicate_or_cross_stratum_parent_contexts():
    row = count_row(0, 1)
    with pytest.raises(ValueError, match='duplicate'):
        fa_rate.summarize_challenges([row, row], context_weights={'clustered': 1})
    with pytest.raises(ValueError, match='mixed context family'):
        fa_rate.summarize_challenges([row, count_row(1, 1, parent=0, family='flow')],
                                     context_weights={'clustered': 1})


def test_reject_poisson_prediction_without_parent_arrival_model():
    with pytest.raises(ValueError, match='parent-context arrival model'):
        fa_rate.estimate_composite(fa_rate.summarize_floor([], 3600), [count_row(0, 1)],
                                  {'impact': 1}, prediction_exposure_hours=6)


def test_manifest_keeps_class_specific_floor_and_requires_measured_assumptions(tmp_path, monkeypatch):
    model = ScoreModel()
    model.classes = ['knock', 'kettle']
    monkeypatch.setattr(fa_rate.edge, 'Model', lambda _: model)
    monkeypatch.setattr(fa_rate, 'score_audio', lambda *a, **kw: [
        {'event': 'start', 'label': 'knock', 'time_s': 1},
        {'label': 'knock', 'start_s': 1, 'end_s': 2, 'confirmed': True}])
    rows = [count_row(f'{c}-{i}', i % 2, label=c) for c in model.classes for i in range(100)]
    for row in rows:
        row['evidence_label'] = 'measured'
        row.update(start_outputs=row['confirmed_outputs'], max_start_outputs=row['max_confirmed_outputs'])
    monkeypatch.setattr(fa_rate, 'score_challenge_records', lambda *a, **kw: rows)
    manifest = {'source_kind': 'measured', 'calibration_streams': [], 'floor_streams': [{'path': 'floor.wav'}],
                'challenges': [{'path': 'context.wav'}], 'context_prevalence_per_hour': {'impact': 2},
                'context_weights': {'clustered': 1}, 'prevalence_source': 'measured',
                'bootstrap_replicates': 100}
    for i, entry in enumerate([*manifest['floor_streams'], *manifest['challenges']]):
        pcm = np.random.default_rng(600+i).integers(-30, 31, 4*1800, dtype=np.int16)
        with wave.open(str(tmp_path / entry['path']), 'wb') as wav:
            wav.setparams((1, 2, 4, 0, 'NONE', 'PCM'))
            wav.writeframes(pcm.tobytes())
        entry.update(recording_id=entry['path'], sample_range=[0, len(pcm)], sample_rate_hz=4)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    model_path = tmp_path / 'model.npz'
    model_path.write_bytes(b'generated score proxy')
    report = fa_rate._score_manifest(str(model_path), str(path))
    assert [f['events'] for f in report['floor_streams']] == [1, 0]
    assert all(c['evidence_label'] == 'model-based' for c in report['composite_estimates'])
    manifest.update(context_weights_source='measured', floor_time_fraction_source='measured')
    path.write_text(json.dumps(manifest))
    model_path = tmp_path / 'model.npz'
    model_path.write_bytes(b'generated score proxy')
    report = fa_rate._score_manifest(str(model_path), str(path))
    assert all(c['evidence_label'] == 'measured' for c in report['composite_estimates'])


def test_small_context_bank_is_inconclusive_even_with_varying_counts():
    result = fa_rate.summarize_challenges(
        [count_row(i, i % 2) for i in range(10)], context_weights={'clustered': 1},
        bootstrap_replicates=100,
    )['families']['impact']['knock']
    assert result['status'].startswith('inconclusive')


def test_score_and_audio_paths_preserve_the_same_tracker_counts():
    model = ScoreModel()
    scores = np.zeros((40, 1), np.float32)
    scores[8:10] = .9
    scores[20:22] = .9
    audio = np.concatenate([scores[:, 0], np.zeros(3, np.float32)])
    assert fa_rate.confirmed_events_from_scores(model, scores) == fa_rate.score_audio(model, audio)
