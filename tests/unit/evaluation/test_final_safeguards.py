"""Regressions for the final review; procedural inputs only."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import edge, project
from sensorsim.evaluation import c2st, reporting, sensitivity, stream
from tests.unit.evaluation._fixture import save_edge_model, write_float_wav


def test_f2_record_corrects_attribution_and_preserves_failure():
    record = Path(__file__).with_name('ACCEPTANCE_RESULT.md').read_text()
    assert 'registered repair acceptance **failed**' in record
    assert '2.4%' in record
    assert 'no length effect' in record
    assert 'not a drift cause' in record
    assert 'remaining framing failure is a merge blocker' not in record


def test_below_chance_is_inconclusive_and_reported(monkeypatch):
    assert c2st.decision(.298, .124, .498) == 'inconclusive'
    assert c2st.decision(.30, .10, .50) == 'pass'
    monkeypatch.setattr(c2st, '_cluster_bootstrap', lambda *a, **k: np.full(k['replicates'], .4))
    rng = np.random.default_rng(913)
    result = c2st.compare_samples(rng.normal(size=(5, 3)), rng.normal(size=(5, 3)),
        np.arange(5), np.arange(5), bootstrap_replicates=19, permutation_replicates=9)['result']
    assert result['outcome'] == 'inconclusive'
    assert result['reason'] == 'inconclusive: CV instability'
    assert result['below_chance_cv_instability'] is True
    assert '12–14%' in result['bootstrap_limit']
    assert 'weak evidence' in result['pass_interpretation']


def test_permutation_refits_complete_grouped_cv_and_preserves_groups(monkeypatch):
    rng = np.random.default_rng(914)
    x, y = rng.normal(size=(15, 4)), rng.normal(3, size=(15, 4))
    groups = np.repeat(np.arange(5), 3)
    fits, assignments = [], []
    fit, folds = c2st._fit_logistic, c2st._group_folds
    def fit_spy(*args, **kwargs):
        fits.append(1)
        return fit(*args, **kwargs)
    def folds_spy(g, labels, count, seed):
        assignments.append(labels.copy())
        for name in np.unique(g):
            assert len(np.unique(labels[g == name])) == 1
        assert sum(labels) == 15
        splits = folds(g, labels, count, seed)
        for test in splits:
            train = np.setdiff1d(np.arange(len(g)), test)
            assert not set(g[test]) & set(g[train])
        return splits
    monkeypatch.setattr(c2st, '_fit_logistic', fit_spy)
    monkeypatch.setattr(c2st, '_group_folds', folds_spy)
    args = dict(bootstrap_replicates=19, bootstrap_seeds=(915,), fold_seed=916,
                permutation_replicates=19, permutation_seed=917)
    result = c2st.compare_samples(x, y, groups, groups, **args)['result']
    assert len(fits) == 5 * 20
    assert len(assignments) == 20
    assert any(not np.array_equal(assignments[0], a) for a in assignments[1:])
    permutation = result['group_permutation']
    assert permutation['refit_each_permutation'] is True
    assert permutation['alternative'] == 'greater'
    assert permutation['p_value'] == (1 + permutation['exceedances']) / 20
    assert .05 <= permutation['p_value'] <= 1
    assert c2st.compare_samples(x, y, groups, groups, **args)['result'] == result


def test_e2_cannot_borrow_e1_power(monkeypatch):
    model = SimpleNamespace(meta={'features': {'fs': 16000}})
    bucket = {'audio_windows': [np.ones(16000, np.float32)], 'groups': ['g'],
              'files': ['g.wav'], 'window_sampling': {}}
    monkeypatch.setattr(sensitivity, 'feature_bucket', lambda *a, **k: bucket)
    monkeypatch.setattr(reporting, 'compare_views', lambda *a, **k: ({'status': 'fail'}, {'status': 'pass'}))
    power = sensitivity.spike_in_report(bucket, bucket, model, seeds=(918, 919))
    assert sensitivity.apply_power_gate('pass', power, test='e1') == 'pass'
    assert sensitivity.apply_power_gate('pass', power, test='e2') == 'inconclusive'
    assert power['per_test']['e1']['status'] == 'detected'
    assert power['per_test']['e2']['status'] == 'underpowered'
    assert sensitivity.apply_power_gate('fail', power, test='e2') == 'fail'


def test_project_applies_power_to_each_test_and_reports_acceptance(tmp_path, monkeypatch):
    out = tmp_path/'out'; out.mkdir()
    save_edge_model(out/'model.npz', width=4, blocks=1)
    bucket = {'kind': 'proxy'}
    power = {'status': 'detected', 'per_test': {'e1': {'status': 'detected'}, 'e2': {'status': 'underpowered'}}}
    monkeypatch.setattr(project, '_feature_rows', lambda *a, **k: {'knock': bucket})
    monkeypatch.setattr(stream, 'training_features', lambda *a, **k: ({'knock': bucket}, {'knock': bucket}, {'status': 'ok'}))
    monkeypatch.setattr(reporting, 'compare_views', lambda *a, **k: ({'status': 'pass'}, {'status': 'pass'}))
    monkeypatch.setattr(sensitivity, 'spike_in_report', lambda *a, **k: power)
    from sensorsim.evaluation import recall
    monkeypatch.setattr(recall, 'evaluate_recall', lambda *a, **k: {
        'per_class': {}, 'false_events_per_hour': None, 'confusion_matrix': {'counts': {}}})
    project._evaluate_cnn([{'label': 'knock', 'path': tmp_path/'take.wav', 'data_kind': 'proxy'}], out, data_kind='proxy')
    import json
    report = json.loads((out/'evaluation.json').read_text())
    assert report['feature_space_realism']['knock']['status'] == 'pass'
    e2 = report['sim_vs_real_discriminator']['knock']
    assert e2['status'] == 'inconclusive'
    assert e2['reason'] == 'inconclusive (insufficient power): E2 did not detect every spike-in for every seed'
    assert 'gate_acceptance' in report
    assert any('weak evidence' in s for s in report['limits'])


@pytest.mark.parametrize('peak', [100, 1900])
def test_edge_clamp_is_disclosed(tmp_path, peak):
    model = save_edge_model(tmp_path/'m.npz', width=4, blocks=1)
    audio = np.zeros(32000, np.float32)
    audio[peak*16:peak*16+100] = 1
    path = tmp_path/'edge.wav'; write_float_wav(path, audio)
    bucket = stream.file_features([{'label': 'knock', 'group': 'g', 'file': 'edge.wav', 'path': path}], model, kind='proxy')['knock']
    framing = bucket['window_sampling']
    assert framing['clamped_windows'] == 1
    assert any('0.5 s' in s for s in framing['limits'])
    # Document the existing clamp; do not change its waveform semantics.
    np.testing.assert_array_equal(bucket['audio_windows'][0], stream.event_crop(edge.read_wav(str(path), 16000), model))


def test_context_exclusions_depend_on_length_and_are_disclosed(tmp_path, monkeypatch):
    from sensorsim import nn
    model = save_edge_model(tmp_path/'m.npz', width=4, blocks=1)
    model.meta['training'] = {'config': dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}))}
    generated = np.zeros(16000, np.float32); generated[1000:1100] = 1
    short = np.zeros(16000, np.float32); short[14000:14100] = 1
    long = np.r_[short, np.zeros(16000, np.float32)]
    monkeypatch.setattr(stream, 'training_example', lambda *a, **k: generated)
    monkeypatch.setattr(edge, 'read_wav', lambda path, fs: long if 'long' in path else short)
    rows = [{'label': 'background', 'file': 'bg.wav', 'split': 'train'},
            {'label': 'knock', 'file': 'source.wav', 'split': 'train', 'source_group': 'g'}]
    def run(name):
        return stream.training_features([{'label': 'knock', 'path': tmp_path/name, 'file': name, 'group': 'g'}],
                                        rows, tmp_path, model, data_kind='proxy')
    _, _, short_report = run('short.wav')
    _, _, long_report = run('long.wav')
    assert short_report['framing']['knock']['used_pairs'] == 0
    assert long_report['framing']['knock']['used_pairs'] == 1
    assert len(short_report['framing']['knock']['excluded_pairs']) == 1
    assert any('take length' in s for s in short_report['limits'])


def test_peak_can_select_background_and_warning_is_explicit(tmp_path):
    model = save_edge_model(tmp_path/'m.npz', width=4, blocks=1)
    audio = np.zeros(32000, np.float32)
    audio[18000:18300] = .01  # quiet target
    audio[10000:10300] = .2  # louder background transient
    assert 10000 <= stream.event_anchor(audio, 16000) <= 10300
    path = tmp_path/'low_snr.wav'; write_float_wav(path, audio)
    bucket = stream.file_features([{'label': 'knock', 'group': 'g', 'file': path.name, 'path': path}], model, kind='proxy')['knock']
    assert any('background transient' in s for s in bucket['window_sampling']['limits'])
