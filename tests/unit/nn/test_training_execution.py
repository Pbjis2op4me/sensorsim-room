"""Development controls for training execution; no distributional acceptance."""
import numpy as np
import pytest

from sensorsim import edge, nn
from tests.unit.nn.test_train_end_to_end import inputs, sha


def config(panel, **extra):
    return dict(steps=3, width=4, blocks=1, batch=8, workers=0, threads=1,
        val_windows=64, p_confuser_given_background=0., selection_min_groups=1,
        selection_panel=str(panel), selection_panel_sha256=sha(panel),
        selection_min_recall=0., selection_trigger_budget=1.,
        selection_grid=[dict(on=.999999, smooth_hops=1, min_hops=1)], **extra)


def assert_weights_exact(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype
        assert left[key].shape == right[key].shape
        assert left[key].tobytes() == right[key].tobytes(), key


def test_selected_checkpoint_export_is_not_recomputed(tmp_path, monkeypatch):
    manifest, _, panel = inputs(tmp_path/'inputs')
    calls = []
    original = nn.export

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result.copy())
        return result

    monkeypatch.setattr(nn, 'export', observed)
    out = tmp_path/'model.npz'
    report = nn.train(manifest, out, **config(panel))
    assert len(calls) == 1  # the selected checkpoint already passed the parity gate
    assert_weights_exact(edge.Model(str(out)).weights,
                         edge.Model(str(tmp_path/'model.checkpoint-3.npz')).weights)
    assert report['parity_max_abs_diff'] == calls[0]['parity_max_abs_diff']
    assert edge.Model(str(out)).meta['training']['steps'] == 3


def test_earlier_checkpoint_is_reused_and_stale_terminal_mutant_is_caught(tmp_path, monkeypatch):
    manifest, _, panel = inputs(tmp_path/'inputs')
    choose = nn.training_selection.choose_candidate
    calls = []

    def only_first(rows):
        calls.append(rows)
        return choose(rows) if len(calls) == 1 else None

    monkeypatch.setattr(nn.training_selection, 'choose_candidate', only_first)
    cfg = config(panel)
    cfg.update(steps=251, batch=2)
    out = tmp_path/'model.npz'
    report = nn.train(manifest, out, **cfg)
    assert len(calls) == 2 and report['selection']['selected']['step'] == 250
    selected = edge.Model(str(tmp_path/'model.checkpoint-250.npz')).weights
    terminal = edge.Model(str(tmp_path/'model.checkpoint-251.npz')).weights
    assert_weights_exact(edge.Model(str(out)).weights, selected)
    with pytest.raises(AssertionError):
        assert_weights_exact(terminal, selected)  # known-bad: discard earlier selection
    mutant = {key: value.copy() for key, value in selected.items()}
    key = 'fc.w'
    mutant[key].flat[0] = np.nextafter(mutant[key].flat[0], np.float32(np.inf))
    with pytest.raises(AssertionError):
        assert_weights_exact(mutant, selected)  # one ULP is a reachable defect


def test_infeasible_terminal_export_is_reused(tmp_path):
    manifest, _, panel = inputs(tmp_path/'inputs')
    cfg = config(panel)
    cfg['selection_min_recall'] = 1.
    out = tmp_path/'model.npz'
    with pytest.warns(UserWarning, match='fixed detector'):
        report = nn.train(manifest, out, **cfg)
    assert report['selection']['status'] == 'infeasible'
    assert_weights_exact(edge.Model(str(out)).weights,
                         edge.Model(str(tmp_path/'model.checkpoint-3.npz')).weights)


def test_explicit_cpu_keeps_default_artifact_bytes(tmp_path, monkeypatch):
    manifest, _, panel = inputs(tmp_path/'inputs')
    monkeypatch.setattr(nn.time, 'time', lambda: 1000000.)
    for name, override in [('default', {}), ('cpu', {'device': 'cpu'})]:
        folder = tmp_path/name
        folder.mkdir()
        nn.train(manifest, folder/'model.npz', **config(panel), **override)
    for path in (tmp_path/'default').iterdir():
        assert path.read_bytes() == (tmp_path/'cpu'/path.name).read_bytes()


@pytest.mark.parametrize('device', ['bad', None, 1])
def test_unknown_device_refused_before_data_access(tmp_path, device):
    with pytest.raises(ValueError, match='device must be'):
        nn.train(tmp_path/'missing.jsonl', tmp_path/'model.npz', device=device)


def test_unavailable_cuda_refused_and_auto_fallback_recorded(tmp_path, monkeypatch):
    torch = nn._torch()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA/ROCm.*unavailable'):
        nn.train(tmp_path/'missing.jsonl', tmp_path/'missing.npz', device='cuda')
    manifest, _, panel = inputs(tmp_path/'inputs')
    nn.train(manifest, tmp_path/'model.npz', **config(panel), device='auto')
    execution = edge.Model(str(tmp_path/'model.npz')).meta['training']['config']['execution']
    assert execution['device'] == 'cpu' and execution['requested_device'] == 'auto'
    assert execution['fallback'] == 'CUDA/ROCm unavailable'
