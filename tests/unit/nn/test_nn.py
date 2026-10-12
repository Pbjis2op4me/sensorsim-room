"""CNN training and export: the numpy device runtime reproduces PyTorch, the tracker settings are fitted per
class (or kept when given), and bad settings are refused before any work starts."""
import json
import pickle

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim import edge, nn, recognizer

FS = 16000


def _tone(f, k, seconds=.6):
    t = np.arange(int(seconds*FS))/FS
    return .003*np.exp(-4*t)*np.sin(2*np.pi*f*(1 + .004*k)*t)


def _write(path, x):
    wavfile.write(path, FS, np.clip(np.round(x*32767), -32768, 32767).astype(np.int16))


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    """Two tonal classes rendered at a relative level, plus background recordings for both splits."""
    root = tmp_path_factory.mktemp("dataset")
    rows = []
    for label, f in (("low", 500), ("high", 2200)):
        for k in range(6):
            _write(root/f"{label}_{k}.wav", _tone(f, k))
            rows.append({"file": f"{label}_{k}.wav", "label": label, "split": "val" if k >= 4 else "train", "fs": FS,
                         "dry_source": {"source_group": f"{label}-{k}"}})
    (root/"manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name, seed in (("room_a", 1), ("room_b", 2)):
        _write(root/f"{name}.wav", .03*np.random.default_rng(seed).normal(size=12*FS))
    recognizer.add_background(root/"manifest.jsonl", [root/"room_a.wav"], [root/"room_b.wav"])
    return root/"manifest.jsonl"


SMALL = {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final", "steps": 12, "width": 8, "blocks": 2, "batch": 16, "workers": 0, "threads": 1, "val_windows": 128}


@pytest.fixture(scope="module")
def trained(manifest, tmp_path_factory):
    pytest.importorskip("torch")
    out = tmp_path_factory.mktemp("model")/"model.npz"
    return out, nn.train(manifest, out, **SMALL)


def test_the_device_runtime_reproduces_the_trained_network(trained):
    out, report = trained
    assert report["parity_max_abs_diff"] < 1e-4                   # export refuses anything worse
    model = edge.Model(str(out))
    assert model.classes == ["high", "low"] and model.macs() == report["macs_per_window"]
    assert report["parameters"] == sum(w.size for w in model.weights.values())


def test_tracker_settings_are_fitted_per_class_and_saved(trained):
    out, report = trained
    d = edge.Model(str(out)).meta["detector"]
    assert set(d["persistence_hops"]) == set(d["smooth_hops"]) == set(d["min_hops"]) == {"high", "low"}
    assert all(1 <= d["smooth_hops"][c] <= (nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})["detector"]["smooth_hops"] for c in d["smooth_hops"])
    assert all(1 <= d["min_hops"][c] <= (nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})["detector"]["min_hops"] for c in d["min_hops"])
    assert report["detector"]["smooth_hops"] == d["smooth_hops"]


def test_per_class_settings_given_in_the_config_are_kept(trained, manifest, tmp_path):
    out, _ = trained
    copy = tmp_path/"copy.npz"
    copy.write_bytes(out.read_bytes())
    cfg = {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), **SMALL, "detector": {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})["detector"], "smooth_hops": {"high": 1, "low": 2}}}
    corpus = nn.Corpus(manifest, "val", ["high", "low"], FS)
    nn._fit_tracker(edge.Model(str(copy)), corpus, cfg, copy)
    assert edge.Model(str(copy)).meta["detector"]["smooth_hops"] == {"high": 1, "low": 2}
    partial = {**cfg, "detector": {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})["detector"], "smooth_hops": {"high": 1}, "min_hops": {"high": 1}}}
    nn._fit_tracker(edge.Model(str(copy)), corpus, partial, copy)
    d = edge.Model(str(copy)).meta["detector"]
    assert d["smooth_hops"]["high"] == 1 and d["min_hops"]["high"] == 1 and set(d["smooth_hops"]) == {"high", "low"}
    edge.detect(edge.Model(str(copy)), np.zeros(3*FS, np.float32))           # the tracker accepts the merged settings
    with pytest.raises(ValueError, match="unknown classes"):
        nn._fit_tracker(edge.Model(str(copy)), corpus, {**cfg, "detector": {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})["detector"], "smooth_hops": {"hgih": 1}}}, copy)


def test_bad_settings_are_refused_before_any_work(manifest, tmp_path):
    with pytest.raises(ValueError, match="unknown CNN settings"):
        nn.train(manifest, tmp_path/"m.npz", min_prob=.5)
    with pytest.raises(ValueError, match=".npz"):
        nn.train(manifest, tmp_path/"model.onnx")


@pytest.mark.parametrize('dynamics', [False, True])
def test_the_training_stream_pickles_and_ignores_the_worker_count(manifest, dynamics):
    torch = pytest.importorskip("torch")
    cfg = {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), **SMALL, "steps": 3, 'event_aware_masks': dynamics, '_dynamics_report': dynamics}
    corpus = nn.Corpus(manifest, "train", None, FS)
    data = nn.Batches(corpus, cfg, 3, cfg["seed"])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(data))[2][0], data[2][0])   # what "spawn" workers receive
    serial = list(nn._training_batches(corpus, {**cfg, "workers": 0}, torch))
    parallel = list(nn._training_batches(corpus, {**cfg, "workers": 2}, torch))
    assert len(serial) == len(parallel) == 3
    for a, b in zip(serial, parallel):
        assert len(a) == len(b) == (3 if dynamics else 2)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)


@pytest.mark.slow
@pytest.mark.parametrize('dynamics', [False, True])
def test_training_runs_with_spawned_workers_as_on_macos_and_windows(manifest, tmp_path, dynamics):
    pytest.importorskip("torch")
    import torch.multiprocessing as mp
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    try:
        report = nn.train(manifest, tmp_path/"m.npz", **{**SMALL, "workers": 1, "steps": 3,
                                                      'event_aware_masks': dynamics})
    finally:
        mp.set_start_method(previous, force=True)
    assert report["parity_max_abs_diff"] < 1e-4
