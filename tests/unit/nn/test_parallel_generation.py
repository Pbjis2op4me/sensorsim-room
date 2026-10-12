"""Synthetic proxy controls for versioned streams; no IPC in the default tests."""
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import edge, nn


def corpus():
    fs = 16000
    t = np.arange(fs) / fs
    targets = {'knock': [(0.03*np.sin(2*np.pi*440*t)*np.exp(-5*t)).astype(np.float32)],
               'bell': [(0.02*np.sin(2*np.pi*880*t)).astype(np.float32)]}
    return SimpleNamespace(classes=list(targets), targets=targets,
        spans={c: [(0, fs)] for c in targets}, negative_pool=None, pool_purpose='train',
        background=[np.random.default_rng(991).normal(0, .002, fs*3).astype(np.float32)],
        confuser_draws=0)


def config(**kwargs):
    return {**nn.DEFAULTS, 'batch': 5, 'steps': 3, 'seed': 77, 'workers': 0,
            'p_confuser_given_background': 0., **kwargs}


def exact(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, dict):
        assert type(left) is type(right) and left.keys() == right.keys()
        for k in left:
            exact(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert type(left) is type(right) and left == right


class InProcessLoader:
    """Pickled worker copies, reverse dispatch, ordered return; not a speed oracle."""
    last = None

    def __init__(self, data, **kwargs):
        self.data, self.kwargs = data, kwargs
        type(self).last = self

    def __iter__(self):
        import torch
        # This is the actual documented DataLoader iterator's base-seed draw.
        torch.empty((), dtype=torch.int64).random_(generator=self.kwargs.get('generator'))
        clones = [pickle.loads(pickle.dumps(self.data)) for _ in range(self.kwargs['num_workers'])]
        collate = self.kwargs.get('collate_fn', torch.utils.data.default_convert)
        result = {}
        for k in reversed(range(len(self.data))):
            result[k] = collate(clones[k % len(clones)][k])
        for k in range(len(self.data)):
            yield result[k]


@pytest.mark.parametrize('report', [False, True])
def test_legacy_batch_seed_matches_independent_assembly(report):
    cfg = config(_negative_report=report)
    expected_corpus = corpus()
    rng = np.random.default_rng([77, 2])
    examples = [nn.make_example(expected_corpus, rng, cfg) for _ in range(cfg['batch'])]
    expected = (edge.Features(nn._feature_cfg(cfg))(np.stack([x for x, _ in examples])),
                np.stack([y for _, y in examples]))
    if report:
        expected += (dict(windows=5, background_windows=int(np.sum(~expected[1].any(axis=1))), confuser_windows=0),)
    exact(nn.Batches(corpus(), cfg, 3, 77)[2], expected)


@pytest.mark.parametrize('workers', [1, 2, 8])
@pytest.mark.parametrize('version', ['stream/1', 'stream/2'])
@pytest.mark.parametrize('context', [False, True])
def test_worker_scheduling_preserves_exact_batch_and_metadata(monkeypatch, workers, version, context):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.utils.data, 'DataLoader', InProcessLoader)
    cfg = config(stream_version=version, _negative_report=True,
                 _dynamics_report=context, event_aware_masks=context)
    expected = list(nn._training_batches(corpus(), cfg, torch))
    got = list(nn._training_batches(corpus(), dict(cfg, workers=workers), torch))
    exact(expected, got)
    assert InProcessLoader.last.kwargs['num_workers'] == workers


def test_stream2_reaches_independent_per_example_seed_oracle(monkeypatch):
    calls = []
    original = nn.make_example
    def record(c, rng, cfg, **kwargs):
        calls.append(copy.deepcopy(rng.bit_generator.state))
        return original(c, rng, cfg, **kwargs)
    monkeypatch.setattr(nn, 'make_example', record)
    cfg = config(stream_version='stream/2')
    got = nn.Batches(corpus(), cfg, 3, 77)[2]
    states = copy.deepcopy(calls)
    monkeypatch.setattr(nn, 'make_example', original)
    expected_corpus = corpus()
    examples = []
    for index in range(10, 15):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([77, index, 2])))
        assert states[index-10] == rng.bit_generator.state
        examples.append(original(expected_corpus, rng, cfg))
    exact(got, (edge.Features(nn._feature_cfg(cfg))(np.stack([e[0] for e in examples])),
                np.stack([e[1] for e in examples])))


def test_stream2_example_audio_is_independent_of_batch_partition(monkeypatch):
    original = nn.make_example
    calls = []
    def record(*a, **kw):
        out = original(*a, **kw); calls.append(out); return out
    monkeypatch.setattr(nn, 'make_example', record)
    a = nn.Batches(corpus(), config(stream_version='stream/2', batch=6), 1, 77)
    a[0]; whole = calls[:]; calls.clear()
    b = nn.Batches(corpus(), config(stream_version='stream/2', batch=3), 2, 77)
    b[0]; b[1]
    exact(whole, calls)


def test_stream2_loader_isolates_model_rng_and_preserves_weight_trajectory(monkeypatch):
    torch = pytest.importorskip('torch'); torch.set_num_threads(1)
    monkeypatch.setattr(torch.utils.data, 'DataLoader', InProcessLoader)
    outputs = []
    for workers in [0, 1, 2, 8]:
        torch.manual_seed(431)
        model = nn.build_model(2, width=4, blocks=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = torch.get_rng_state().clone()
        cfg = config(stream_version='stream/2', workers=workers)
        batches = nn._training_batches(corpus(), cfg, torch)
        for x, y in batches:
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(x)), torch.from_numpy(y))
            loss.backward(); optimizer.step()
        outputs.append(({k: v.detach().numpy().copy() for k, v in model.state_dict().items()},
                        torch.get_rng_state().numpy().copy(), before.numpy()))
    for out in outputs[1:]:
        exact(outputs[0], out)


def test_stream2_loader_does_not_change_global_torch_rng(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.utils.data, 'DataLoader', InProcessLoader)
    torch.manual_seed(991)
    before = torch.get_rng_state().clone()
    list(nn._training_batches(corpus(), config(stream_version='stream/2', workers=2), torch))
    assert torch.equal(before, torch.get_rng_state())
    assert callable(InProcessLoader.last.kwargs['collate_fn'])


@pytest.mark.parametrize('wrong', ['shared_batch_rng', 'duplicate_example_index'])
def test_seed_oracle_rejects_reached_wrong_optimization(monkeypatch, wrong):
    from sensorsim import training_stream
    cfg = config(stream_version='stream/2')
    correct = nn.Batches(corpus(), cfg, 2, 77)[1]
    calls = []
    rng = np.random.default_rng([77, 1])
    def bad(seed, index):
        calls.append(index)
        return rng if wrong == 'shared_batch_rng' else np.random.default_rng([seed, 0, 2])
    monkeypatch.setattr(training_stream, 'example_rng', bad)
    changed = nn.Batches(corpus(), cfg, 2, 77)[1]
    assert calls == [5, 6, 7, 8, 9]
    with pytest.raises(AssertionError):
        exact(correct, changed)


def test_rng_oracle_rejects_unisolated_loader_mutant(monkeypatch):
    torch = pytest.importorskip('torch')
    class BadLoader(InProcessLoader):
        def __init__(self, data, **kwargs):
            kwargs.pop('generator', None); super().__init__(data, **kwargs)
    monkeypatch.setattr(torch.utils.data, 'DataLoader', BadLoader)
    before = torch.get_rng_state().clone()
    list(nn._training_batches(corpus(), config(stream_version='stream/2', workers=2), torch))
    assert not torch.equal(before, torch.get_rng_state())


def test_collation_preserves_nested_numpy_metadata_and_detects_conversion():
    torch = pytest.importorskip('torch')
    from sensorsim.training_stream import identity_collate
    value = (np.zeros((2, 3), np.float32), {'trace': [np.arange(3), None, 'cue'], 'windows': 2})
    assert identity_collate(value) is value
    exact(value, pickle.loads(pickle.dumps(identity_collate(value))))
    with pytest.raises(AssertionError):
        exact(value, torch.utils.data.default_convert(value))


@pytest.mark.parametrize('version', ['stream/0', 'stream/3', None, 2])
def test_unknown_stream_fails_closed(version):
    with pytest.raises(ValueError, match='stream_version'):
        nn.Batches(corpus(), config(stream_version=version), 3, 77)


def test_empty_schedule_constant_and_duplicate_content():
    cfg = config(stream_version='stream/2')
    c = corpus(); c.background = [np.zeros(48000, np.float32)]*2
    c.targets = {k: [np.ones(16000, np.float32)*.01]*2 for k in c.classes}
    c.spans = {k: [(0, 16000)]*2 for k in c.classes}
    data = nn.Batches(c, cfg, 0, 77)
    assert len(data) == 0
    with pytest.raises(IndexError): data[0]
    data = nn.Batches(c, cfg, 1, 77)
    exact(data[0], data[0])
    assert np.isfinite(data[0][0]).all()


@pytest.mark.parametrize('cores,expected', [(None, 0), (1, 0), (2, 0), (3, 1), (8, 6), (16, 8), (128, 8)])
def test_default_workers_reserve_two_cores(monkeypatch, cores, expected):
    from sensorsim import training_stream
    monkeypatch.setattr(training_stream, '_available_cpu_count', lambda: cores)
    assert training_stream.default_workers() == expected


def test_default_stream_is_legacy():
    assert nn.DEFAULTS['stream_version'] == 'stream/1'


def test_default_seed_config_does_not_change_saved_training_metadata():
    from sensorsim.training_stream import configure
    cfg = dict(nn.DEFAULTS)
    configure(cfg)
    assert 'stream_version' not in cfg
    cfg['stream_version'] = 'stream/2'
    configure(cfg)
    assert cfg['stream_version'] == 'stream/2'


def test_unknown_stream_is_rejected_before_reading_corpus(tmp_path):
    pytest.importorskip('torch')
    with pytest.raises(ValueError, match='stream_version'):
        nn.train(tmp_path/'absent.jsonl', tmp_path/'model.npz', stream_version='stream/3')


def test_affinity_failure_has_observed_cpu_count_fallback(monkeypatch):
    from sensorsim import training_stream
    def unavailable(_):
        raise OSError('affinity unavailable')
    monkeypatch.setattr(training_stream.os, 'sched_getaffinity', unavailable, raising=False)
    monkeypatch.setattr(training_stream.os, 'cpu_count', lambda: 6)
    assert training_stream.default_workers() == 4


def test_loader_errors_propagate_without_serial_fallback(monkeypatch):
    torch = pytest.importorskip('torch')
    def unavailable(*a, **kw):
        raise RuntimeError('worker creation unavailable')
    monkeypatch.setattr(torch.utils.data, 'DataLoader', unavailable)
    with pytest.raises(RuntimeError, match='worker creation unavailable'):
        nn._training_batches(corpus(), config(stream_version='stream/2', workers=2), torch)
