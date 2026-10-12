"""Regression controls for review F1-F10; all generated data are proxies."""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import chi2

from sensorsim import edge, project
from sensorsim.evaluation import realism, recall
from tests.unit.evaluation._fixture import save_edge_model, write_float_wav


def test_f1_decision_null_matches_all_group_counts():
    rng = np.random.default_rng(15)
    groups = np.repeat(np.arange(15), 6)
    x = (rng.normal(0, np.sqrt(.2), (15, 1, 120)) +
         rng.normal(0, np.sqrt(.8), (15, 6, 120))).reshape(-1, 120)
    y = (rng.normal(0, np.sqrt(.2), (15, 1, 120)) +
         rng.normal(0, np.sqrt(.8), (15, 6, 120))).reshape(-1, 120)
    y[:, :10] += .6
    report = realism.realism_report(x, y, reference_groups=groups, rendered_groups=groups,
                                    n_bands=40, permutations=199, seed=20261008)
    assert report['status'] == 'fail'
    assert report['decision_null']['n_groups'] == {'left': 15, 'right': 15}
    assert report['decision_null']['method'] == 'joint-group-label-permutation'
    assert report['decision_null']['permutation_p'] <= .05


def test_f2_framing_has_one_identical_window_per_group(tmp_path):
    model = save_edge_model(tmp_path/'m.npz', width=4, blocks=1)
    rng = np.random.default_rng(9)
    audio = rng.normal(0, .001, 6*16000).astype(np.float32)
    audio[7200:10400] += .2*rng.normal(size=3200)
    rows = []
    for name, x in [('long', audio), ('short', audio[:32000])]:
        path = tmp_path/f'{name}.wav'; write_float_wav(path, x)
        rows.append({'path': path, 'file': path.name, 'group': name, 'label': 'knock'})
    a = project._feature_rows(rows[:1], model, kind='proxy')['knock']
    b = project._feature_rows(rows[1:], model, kind='proxy')['knock']
    assert a['stats'].shape == b['stats'].shape == (1, 120)
    np.testing.assert_array_equal(a['stats'], b['stats'])
    np.testing.assert_array_equal(a['embedding'], b['embedding'])


def test_f2_adapter_calls_current_and_future_training_signature(monkeypatch):
    from sensorsim.evaluation.stream import training_example
    from sensorsim import nn
    cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), p_mic_chain=.42, snr_db=[2., 7.], device_split='stress')
    calls = []
    def make(corpus, rng, config, *, background_domain=None):
        calls.append((corpus, config, background_domain))
        return np.ones(16000, np.float32), np.ones(1)
    monkeypatch.setattr(nn, 'make_example', make)
    corpus = SimpleNamespace(classes=['knock'])
    out = training_example(corpus, np.random.default_rng(1), cfg)
    assert out.shape == (16000,)
    assert len(calls) == 1 and calls[0][0] is corpus
    assert calls[0][1]['p_mic_chain'] == .42
    assert calls[0][1]['snr_db'] == [2., 7.]
    assert calls[0][1]['device_split'] == 'stress'
    assert calls[0][1]['p_background_only'] == 0


def test_f3_both_e1_views_are_required():
    from sensorsim.evaluation.stream import combine_status
    assert combine_status(['pass', 'fail']) == 'fail'
    assert combine_status(['pass', 'inconclusive']) == 'inconclusive'
    assert combine_status(['pass', 'pass']) == 'pass'


def test_f4_acceptance_has_no_implicit_job_io():
    from tests.unit.evaluation import test_acceptance_truth
    assert 'os.environ["JOB"]' not in inspect.getsource(test_acceptance_truth)


def _detector(monkeypatch):
    monkeypatch.setattr(edge, 'detect', lambda model, x: [{'label': 'knock'}] if x[0] else [])
    return SimpleNamespace(classes=['knock'], meta={'features': {'fs': 10}, 'detector': {}})


def test_f5_short_negative_exposure_never_reports_a_rate(monkeypatch):
    model = _detector(monkeypatch)
    r = recall.evaluate_recall(model, [{'audio': np.ones(30), 'label': 'background'}], data_kind='proxy')
    assert r['false_events_per_hour'] is None
    assert r['false_alarm']['status'] == 'inconclusive'
    assert r['false_alarm']['exposure_seconds'] == 3.
    assert r['false_alarm']['label'] == 'smoke test, not FA/hour'


def test_f5_exact_poisson_one_sided_bound_and_exposure(monkeypatch):
    model = _detector(monkeypatch)
    r = recall.evaluate_recall(model, [{'audio': np.zeros(18000), 'label': 'background'}], data_kind='proxy')
    f = r['false_alarm']
    assert f['status'] == 'ok' and f['data_kind'] == 'proxy'
    assert f['upper_95_per_hour'] == pytest.approx(chi2.ppf(.95, 2)/2/.5)
    assert f['interval_95_per_hour'][1] == pytest.approx(chi2.ppf(.975, 2)/2/.5)


def test_f6_undetected_spike_downgrades_only_passes():
    from sensorsim.evaluation.sensitivity import apply_power_gate
    assert apply_power_gate('pass', {'status': 'underpowered'}) == 'inconclusive'
    assert apply_power_gate('fail', {'status': 'underpowered'}) == 'fail'
    assert apply_power_gate('pass', {'status': 'detected'}) == 'pass'


def test_f7_recall_respects_repeated_manifest_group(monkeypatch):
    model = _detector(monkeypatch)
    r = recall.evaluate_recall(model, [
        {'audio': np.ones(20), 'label': 'knock', 'group': 'same', 'file': 'a'},
        {'audio': np.zeros(20), 'label': 'knock', 'group': 'same', 'file': 'b'},
    ], data_kind='proxy')['per_class']['knock']
    assert r['takes'] == 1
    assert r['hits'] == 0  # conservative all-hit headline, any-hit separately
    assert r['any_hit']['hits'] == 1
    assert r['any_hit']['interval'] == pytest.approx(recall.clopper_pearson(1, 1))


def test_f9_bandwidth_and_kernel_allocations_are_bounded(monkeypatch):
    original = realism._sqdist
    def bounded(a, b):
        assert len(a) <= 1024 and len(b) <= 1024, 'unbounded dense distance matrix'
        return original(a, b)
    monkeypatch.setattr(realism, '_sqdist', bounded)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4096, 6)); y = rng.normal(size=(4096, 6))
    g = np.repeat(np.arange(16), 256)
    r = realism.realism_report(x, y, reference_groups=g, rendered_groups=g,
                               n_bands=2, permutations=19, resplit_cap=10)
    assert r['mmd']['window_subsampled']


def test_f9_status_uses_upstream_dataset_digest(tmp_path, monkeypatch):
    calls = []
    def tree(path, *args):
        calls.append(str(path))
        assert str(path) != str(tmp_path/'out'/'dataset')
        return {'x': 'digest'}
    monkeypatch.setattr(project, '_tree', tree)
    cfg = {'root': str(tmp_path), 'paths': {'evaluate': 'evaluate'}, 'out': 'out',
           'recognizer': {'type': 'cnn'}, 'code': 'code'}
    # _path takes top-level keys.
    cfg['evaluate'] = 'evaluate'
    r = project._inputs('evaluate', cfg, {'train': {'outputs': 'model'}, 'dataset': {'outputs': 'data'}})
    assert r['rendered_dataset'] == 'data'


def test_f10_group_folds_really_keep_every_parent_disjoint():
    from sensorsim.evaluation.c2st import _group_folds
    labels = np.repeat([0, 1], 60)
    groups = np.repeat([f'g{i}' for i in range(30)], 4)
    folds = _group_folds(groups, labels, 5, 13)
    assert sorted(np.concatenate(folds)) == list(range(120))
    for test in folds:
        train = np.setdiff1d(np.arange(120), test)
        assert not set(groups[test]) & set(groups[train])
        assert set(labels[test]) == set(labels[train]) == {0, 1}


def test_f10_domain_group_names_do_not_collide(monkeypatch):
    from sensorsim.evaluation import c2st
    from sensorsim.evaluation.c2st import compare_samples
    original = c2st._group_folds
    def check_groups(groups, labels, folds, seed):
        assert len(np.unique(groups)) == 10
        for group in np.unique(groups):
            assert len(np.unique(labels[groups == group])) == 1
        return original(groups, labels, folds, seed)
    monkeypatch.setattr(c2st, '_group_folds', check_groups)
    rng = np.random.default_rng(18)
    x, y = rng.normal(size=(20, 3)), rng.normal(2, size=(20, 3))
    g = np.repeat([f'g{i}' for i in range(5)], 4)
    a = compare_samples(x, y, g, g, bootstrap_replicates=19)
    b = compare_samples(x, y, g, np.char.add('other-', g), bootstrap_replicates=19)
    assert a == b


def test_f2_default_evaluate_uses_training_manifest_and_mixer(tmp_path, monkeypatch):
    import json
    from sensorsim import nn
    out = tmp_path/'out'; dataset = out/'dataset'; dataset.mkdir(parents=True)
    model_path = out/'model.npz'
    model = save_edge_model(model_path, width=4, blocks=1)
    with np.load(model_path) as archive:
        arrays = dict(archive)
    cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), snr_db=[13., 13.], p_mic_chain=0., device_split='stress')
    meta = model.meta
    meta['training'] = {'config': cfg}
    arrays['meta'] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(model_path, **arrays)
    fs = 16000
    rng = np.random.default_rng(81)
    background = rng.normal(0, .001, 2*fs)
    target = np.zeros(2*fs); target[9000:11000] = rng.normal(0, .1, 2000)
    write_float_wav(dataset/'bg.wav', background)
    write_float_wav(dataset/'target.wav', target)
    write_float_wav(tmp_path/'take.wav', target+background)
    rows = [{'label': 'background', 'file': 'bg.wav', 'split': 'train', 'fs': fs},
            {'label': 'knock', 'file': 'target.wav', 'split': 'train', 'fs': fs,
             'dry_source': {'source_group': 'source-a'}},
            {'label': 'knock', 'file': 'do-not-load.wav', 'split': 'val', 'fs': fs,
             'dry_source': {'source_group': 'source-b'}}]
    (dataset/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    calls = []
    original = nn.make_example
    def spy(corpus, rng, config):
        calls.append(config)
        return original(corpus, rng, config)
    monkeypatch.setattr(nn, 'make_example', spy)
    project._evaluate_cnn([{'label': 'knock', 'path': tmp_path/'take.wav', 'file': 'take.wav',
                            'group': 'held-out-a', 'data_kind': 'proxy'}], out, data_kind='proxy')
    report = json.loads((out/'evaluation.json').read_text())
    assert len(calls) == 1
    assert calls[0]['snr_db'] == [13., 13.]
    assert report['training_stream']['adapter'] == 'sensorsim.nn.make_example'
    assert report['raw_render_diagnostic']['status'] == 'diagnostic-only'
    assert report['spike_in_sensitivity']['knock']['status'] == 'underpowered'
    assert report['feature_space_realism']['knock']['status'] == 'inconclusive'


def test_f3_embedding_only_shift_changes_the_joint_verdict():
    from sensorsim.evaluation.reporting import compare_views
    rng = np.random.default_rng(12)
    groups = np.arange(15).astype(str)
    x = rng.normal(size=(15, 6))
    reference = {'stats': x, 'embedding': x, 'groups': groups, 'kind': 'proxy',
                 'band_centres_hz': [100., 1000.], 'window_sampling': {}}
    synthetic = dict(reference, embedding=x+4.)
    e1, _ = compare_views(reference, synthetic, permutations=99, bootstrap_replicates=39)
    assert e1['band_statistics']['status'] == 'pass'
    assert e1['pooled_embedding']['status'] == 'fail'
    assert e1['status'] == 'fail'


def test_f9_embedding_batches_are_bounded(tmp_path, monkeypatch):
    model = save_edge_model(tmp_path/'m.npz', width=4, blocks=1)
    original = model.features
    sizes = []
    class Features:
        n = original.n
        cfg = original.cfg
        bank = original.bank
        def __call__(self, windows):
            sizes.append(len(windows))
            assert len(windows) <= 16
            return original(windows)
    model.features = Features()
    result = realism.extract_audio_features(np.random.default_rng(4).normal(size=20*16000), model)
    assert len(result['embedding']) == 77
    assert len(sizes) > 1
