"""One-command pipeline: stages chain by content hash, skip when unchanged, and gates stop bad runs."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim import project
from tests.unit.room_pipeline.test_pipeline import fake_session

pytest.importorskip("sklearn")
quiet = lambda message: None


def _layout(root, *, background=2):
    root.mkdir(parents=True, exist_ok=True)
    fake_session(root/"session")                       # fast synthetic session with two labelled real takes
    (root/"background").mkdir()
    for i in range(background):
        x = np.random.default_rng(i).normal(scale=.02, size=3*24000)
        wavfile.write(root/"background"/f"room_{i}.wav", 24000, (x*32767).astype(np.int16))
    (root/"project.yaml").write_text("dataset: {count: 3, fs: 8000, seed: 1}\nrecognizer: {type: baseline, window_s: 0.5}\n")
    return root


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    root = _layout(tmp_path_factory.mktemp("project"))
    return root, project.run(root, echo=quiet)


def test_first_run_executes_every_applicable_stage(first_run):
    root, report = first_run
    assert [report[s]["status"] for s in ("fit", "dataset", "train")] == ["ran"]*3
    assert report["evaluate"]["status"].startswith("skipped")          # no evaluate/ folder
    assert (root/"out"/"model.json").is_file() and (root/"out"/"stages.json").is_file()
    assert set(report["train"]["classes"]) == {"background", "test tone 0", "test tone 1"}


def test_gates_flag_labels_that_cannot_be_validated(first_run):
    _, report = first_run
    assert sorted(report["dataset"]["labels_with_one_recording"]) == ["test tone 0", "test tone 1"]
    assert any("single recording" in n for n in report["dataset"]["notes"])


def test_unchanged_rerun_does_nothing_and_a_dataset_change_does_not_refit(first_run):
    root, _ = first_run
    again = project.run(root, echo=quiet)
    assert [again[s]["status"] for s in ("fit", "dataset", "train")] == ["up to date"]*3
    (root/"project.yaml").write_text("dataset: {count: 2, fs: 8000, seed: 1}\nrecognizer: {type: baseline, window_s: 0.5}\n")
    assert project.status(root) == {"fit": "up to date", "dataset": "needs to run",
                                    "train": "needs to run (after dataset)", "evaluate": "not applicable"}
    changed = project.run(root, echo=quiet)
    assert [changed[s]["status"] for s in ("fit", "dataset", "train")] == ["up to date", "ran", "ran"]


def test_missing_background_stops_before_training(tmp_path):
    root = _layout(tmp_path/"p", background=0)
    with pytest.raises(project.GateError, match="background"):
        project.run(root, echo=quiet)
    assert not (root/"out"/"model.json").exists()


def test_a_label_whose_every_take_is_refused_stops_with_its_name(tmp_path, monkeypatch):
    """Release walkthrough: knocks (0 of 4 takes usable) vanished and were misreported as an incomplete fit.

    The kit protocol refuses such takes (no dry clip is produced for them); emulated here by dropping
    the label's dry rows, since this fast fixture uses the legacy whole-take path.
    """
    import yaml
    from sensorsim.acoustics.room import pipeline
    root = _layout(tmp_path/"p")
    manifest = root/"session"/"manifest.yaml"
    raw = yaml.safe_load(manifest.read_text())
    raw["takes"][[t["type"] for t in raw["takes"]].index("real")]["label"] = "ghost"
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False))
    real = pipeline.run_pipeline
    def refusing(*args, **kwargs):
        report = real(*args, **kwargs)
        return {**report, "dry": [d for d in report["dry"] if d["label"] != "ghost"]}
    monkeypatch.setattr(pipeline, "run_pipeline", refusing)
    with pytest.raises(project.GateError, match=r"no usable recording of \['ghost'\]"):
        project.run(root, echo=quiet)


def test_legacy_manifests_are_warned_that_takes_are_used_whole(first_run):
    _, report = first_run
    assert any("acquisition: legacy" in n for n in report["fit"]["notes"])


def test_an_edited_install_reruns_every_stage(tmp_path, monkeypatch):
    root = _layout(tmp_path/"p")
    project.run(root, echo=quiet)
    monkeypatch.setattr(project, "implementation_fingerprint", lambda: "edited code or a dropped-in catalogue")
    assert set(project.status(root).values()) >= {"needs to run"}
    again = project.run(root, echo=quiet)
    assert [again[s]["status"] for s in ("fit", "dataset", "train")] == ["ran"]*3


def test_never_deletes_what_it_did_not_create(tmp_path):
    root = _layout(tmp_path/"p")
    mine = root/"out"/"dataset"/"user_file.txt"
    mine.parent.mkdir(parents=True)
    mine.write_text("mine")
    with pytest.raises(project.GateError, match="did not write"):           # foreign out/ without a ledger
        project.run(root, echo=quiet)
    assert mine.read_text() == "mine"
    mine.unlink(); mine.parent.rmdir()
    project.run(root, echo=quiet)                                            # pipeline-owned from here on
    (root/"out"/"dataset"/project.MARKER).unlink()                          # a stage folder the user took over
    (root/"project.yaml").write_text("dataset: {count: 2, fs: 8000, seed: 1}\nrecognizer: {type: baseline, window_s: 0.5}\n")
    with pytest.raises(project.GateError, match="not created by this pipeline"):
        project.run(root, echo=quiet)
    assert (root/"out"/"dataset"/"manifest.jsonl").is_file()


@pytest.mark.parametrize("out", ["../elsewhere", "/tmp", "."])
def test_out_must_be_inside_the_project(tmp_path, out):
    root = _layout(tmp_path/"p")
    (root/"project.yaml").write_text(f"out: {out!r}\n")
    with pytest.raises(ValueError, match="inside the project"):
        project.run(root, echo=quiet)


def test_recognizer_settings_are_checked_when_the_project_loads(tmp_path):
    root = tmp_path/"p"
    root.mkdir()
    for text, match in (("recognizer: {type: cnn, min_prob: 0.5}\n", "unknown cnn"),
                        ("recognizer: {type: baseline, width: 8}\n", "unknown baseline"),
                        ("dataset: {fs: 8000}\nrecognizer: {type: cnn}\n", "16000"),
                        ("recognizer: {type: svm}\n", "'cnn' or 'baseline'")):
        (root/"project.yaml").write_text(text)
        with pytest.raises(ValueError, match=match):
            project.load_project(root)


def _tone(f, k, fs=24000, seconds=.5):
    t = np.arange(int(seconds*fs))/fs
    return (.3*np.exp(-5*t)*np.sin(2*np.pi*f*(1 + .01*k)*t)*32767).astype(np.int16)


def _chamber_tone(f, k, fs=24000):
    rng = np.random.default_rng(800+k)
    x = rng.normal(0, 1e-5, 2*fs)
    tone = _tone(f, k, fs).astype(float)/32768
    x[fs//2:fs//2+len(tone)] += tone
    return x.astype('float32')


def _cnn_layout(root):
    """A small CNN project with approved proxy chamber sources and a labelled evaluation take."""
    import yaml
    root = _layout(root)
    for label, f in (("test tone 0", 700.), ("test tone 1", 1900.)):          # second recordings: validation
        (root/"targets"/label).mkdir(parents=True)
        for k in range(2):
            wavfile.write(root/"targets"/label/f"clean_{k}.wav", 24000, _chamber_tone(f, k))
    (root/"evaluate"/"test tone 0").mkdir(parents=True)
    wavfile.write(root/"evaluate"/"test tone 0"/"real.wav", 16000, np.r_[np.zeros(8000, np.int16), _tone(700., 3, 16000)])
    (root/"evaluate"/"background").mkdir()
    wavfile.write(root/"evaluate"/"background"/"room.wav", 16000, (np.random.default_rng(5).normal(scale=600, size=48000)).astype(np.int16))
    (root/"project.yaml").write_text("dataset: {count: 3, fs: 16000, seed: 1}\nrecognizer: {type: cnn, steps: 6, width: 8, "
                                     "blocks: 2, batch: 16, workers: 0, threads: 1, val_windows: 64}\n")
    # Target recordings now require provenance; this fixture generates proxy chamber captures.
    cfg = yaml.safe_load((root/"project.yaml").read_text())
    cfg["dataset"]["sources"] = [dict(file=f"{label}/clean_{k}.wav", label=label,
        source_group=f"{label}-take-{k}", object_id=label+"-oscillator", session="proxy-session",
        capture_route="near_dry", evidence_label="proxy")
        for label in ("test tone 0", "test tone 1") for k in range(2)]
    (root/"project.yaml").write_text(yaml.safe_dump(cfg))
    from tests.unit.sources.target_standins import approve_project_rows
    approved = approve_project_rows(project.load_project(root), root/"qa-preview")
    cfg["dataset"]["sources"] = approved["dataset"]["sources"]
    (root/"project.yaml").write_text(yaml.safe_dump(cfg))
    return root


@pytest.mark.slow
def test_a_cnn_project_trains_evaluates_and_exports_a_folder_that_runs_alone(tmp_path):
    import hashlib, json, subprocess, sys
    pytest.importorskip("torch")
    root = _cnn_layout(tmp_path/"p")
    report = project.run(root, echo=quiet)
    assert report["train"]["type"] == "cnn" and (root/"out"/"model.npz").is_file()
    assert "false_alarm_events_per_background_hour" in report["evaluate"]
    assert project.status(root) == {stage: "up to date" for stage in project.STAGES}

    folder = tmp_path/"device"
    exported = project.export(root, folder)
    assert sorted(p.name for p in folder.iterdir()) == ["DEPLOY.md", "LICENSE", "NOTICE", "SHA256SUMS", "edge.py", "model.npz"]
    assert exported["files"] == sorted(p.name for p in folder.iterdir())
    for line in (folder/"SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    assert "Apache License" in (folder/"LICENSE").read_text()
    info = subprocess.run([sys.executable, "-I", "edge.py", "info", "model.npz"], cwd=folder, capture_output=True, text=True)
    assert info.returncode == 0, info.stderr
    assert json.loads(info.stdout)["classes"] == exported["classes"]
    with pytest.raises(project.GateError, match="not an empty folder"):
        project.export(root, folder)


def test_visualize_draws_a_measured_take_with_its_response(first_run, tmp_path):
    root, _ = first_run
    result = project.visualize(root, tmp_path/"paths.html")
    assert result["by_order"] == {0: 1, 1: 6, 2: 18} and result["measured_overlay"]
    assert result["obstacle_reflections"] >= 1 and result["paths_drawn"] == 25 + result["obstacle_reflections"]  # the desk
    page = (tmp_path/"paths.html").read_text()
    assert "Sound paths to mic" in page and "https:" not in page
    custom = project.visualize(root, tmp_path/"custom.html", source=(1., 1., 1.), receiver=(3., 2., 1.5),
                               names=("doorbell", "baby monitor"), max_order=1)
    assert custom["by_order"] == {0: 1, 1: 6} and custom["measured_overlay"] is None
    with pytest.raises(ValueError, match="outside the fitted room"):
        project.visualize(root, tmp_path/"bad.html", source=(1., 1., 1.), receiver=(30., 2., 1.5))


def test_installing_an_absorption_catalogue_outside_the_package_reruns_the_fit(tmp_path, monkeypatch):
    from sensorsim.acoustics.room import fit
    from tests.unit.plugins.test_plugins import _catalogue
    monkeypatch.delenv(fit.CATALOGUE_ENV, raising=False)
    root = _layout(tmp_path/"p")
    project.run(root, echo=quiet)
    assert project.status(root)["fit"] == "up to date"
    folder = _catalogue(tmp_path/"catalogue")
    monkeypatch.setenv(fit.CATALOGUE_ENV, str(folder))
    assert project.status(root)["fit"] == "needs to run"
    project.run(root, echo=quiet)
    assert project.status(root)["fit"] == "up to date"
    (folder/"materials.yaml").write_text((folder/"materials.yaml").read_text() + "\n# edited\n")
    assert project.status(root)["fit"] == "needs to run"                    # an edited catalogue counts too


def _fake_cnn_trainer(monkeypatch, calls):
    """Stands in for nn.train with its file behaviour: the selection-run guard and the folders it creates."""
    import json
    from sensorsim import nn

    def train(manifest, out, **settings):
        out = Path(out)
        if out.exists() or out.with_suffix(".selection.json").exists():
            raise FileExistsError("use a fresh output path for a selection run")
        out.with_suffix(".negative-inputs").mkdir(exist_ok=False)
        out.with_suffix(".validation_scores.npz").write_bytes(b"scores")
        out.write_bytes(json.dumps(settings, sort_keys=True).encode())
        calls.append(settings)
        return {"classes": ["test tone 0", "test tone 1"], "parameters": 1, "macs_per_window": 1,
                "best_val_mean_ap": 0.0, "detector": {}}
    monkeypatch.setattr(nn, "train", train)
    monkeypatch.setattr(nn, "_torch", lambda: None)


def _set_steps(root, steps):
    import yaml
    cfg = yaml.safe_load((root/"project.yaml").read_text())
    cfg["recognizer"]["steps"] = steps
    (root/"project.yaml").write_text(yaml.safe_dump(cfg))


@pytest.fixture
def cnn_project(tmp_path, monkeypatch):
    import shutil
    calls = []
    _fake_cnn_trainer(monkeypatch, calls)
    root = _cnn_layout(tmp_path/"p")
    shutil.rmtree(root/"evaluate")                     # the stand-in model cannot be evaluated
    project.run(root, echo=quiet)
    assert len(calls) == 1
    return root, calls


def test_a_cnn_project_retrains_after_a_setting_change_and_on_force(cnn_project):
    root, calls = cnn_project
    _set_steps(root, 7)
    assert project.status(root)["train"] == "needs to run"
    report = project.run(root, echo=quiet)
    assert report["train"]["status"] == "ran" and calls[-1]["steps"] == 7
    assert b'"steps": 7' in (root/"out"/"model.npz").read_bytes()
    report = project.run(root, force=("train",), echo=quiet)
    assert report["train"]["status"] == "ran" and len(calls) == 3
    assert (root/"out"/"model.negative-inputs").is_dir()


def test_retraining_refuses_to_replace_a_model_it_did_not_write(cnn_project):
    root, calls = cnn_project
    model = root/"out"/"model.npz"
    model.write_bytes(b"a calibrated copy")
    _set_steps(root, 7)
    with pytest.raises(project.GateError, match="not written by this pipeline"):
        project.run(root, echo=quiet)
    assert model.read_bytes() == b"a calibrated copy" and len(calls) == 1
    model.unlink()
    (root/"out"/"model.negative-inputs"/"user.txt").write_text("mine")        # changed since training
    with pytest.raises(project.GateError, match="model.negative-inputs"):
        project.run(root, echo=quiet)
    assert (root/"out"/"model.negative-inputs"/"user.txt").read_text() == "mine"


def test_a_calibrated_model_copy_does_not_trigger_retraining(cnn_project):
    root, calls = cnn_project
    (root/"out"/"model.npz").write_bytes(b"a calibrated copy")
    assert project.status(root)["train"] == "up to date"
    assert project.run(root, echo=quiet)["train"]["status"] == "up to date" and len(calls) == 1


def test_a_missing_stage_output_reruns_that_stage(cnn_project):
    root, calls = cnn_project
    (root/"out"/"model.npz").unlink()
    assert project.status(root)["train"] == "needs to run (outputs missing or changed)"
    assert project.run(root, echo=quiet)["train"]["status"] == "ran" and len(calls) == 2


@pytest.mark.slow
def test_a_real_cnn_project_retrains_after_a_setting_change(tmp_path):
    pytest.importorskip("torch")
    root = _cnn_layout(tmp_path/"p")
    project.run(root, echo=quiet)
    first = (root/"out"/"model.npz").read_bytes()
    _set_steps(root, 7)
    report = project.run(root, echo=quiet)
    assert report["train"]["status"] == "ran" and (root/"out"/"model.npz").read_bytes() != first
    assert project.run(root, force=("train",), echo=quiet)["train"]["status"] == "ran"
    assert project.status(root) == {stage: "up to date" for stage in project.STAGES}


def test_evaluation_caps_native_thread_pools_at_recognizer_threads(tmp_path, monkeypatch):
    import yaml
    from threadpoolctl import threadpool_info
    calls, seen = [], []
    _fake_cnn_trainer(monkeypatch, calls)
    root = _cnn_layout(tmp_path/"p")
    cfg = yaml.safe_load((root/"project.yaml").read_text())
    cfg["recognizer"]["threads"] = 1
    (root/"project.yaml").write_text(yaml.safe_dump(cfg))

    def evaluate(cfg, out):
        seen.append({pool["user_api"]: pool["num_threads"] for pool in threadpool_info()})
        (out/"evaluation.json").write_text("{}")
        return {}
    monkeypatch.setattr(project, "_evaluate", evaluate)
    if all(pool["num_threads"] <= 1 for pool in threadpool_info()):
        pytest.skip("native pools already single-threaded; nothing to cap")
    assert project.run(root, echo=quiet)["evaluate"]["status"] == "ran"
    assert seen and all(n <= 1 for n in seen[0].values())
    assert any(pool["num_threads"] > 1 for pool in threadpool_info())             # restored afterwards


def test_an_old_ledger_proves_only_the_model_and_never_deletes_sidecars(cnn_project):
    import json
    root, calls = cnn_project
    ledger_file = root/"out"/"stages.json"
    ledger = json.loads(ledger_file.read_text())
    del ledger["train"]["artifacts"]                                  # as written before artifacts were recorded
    ledger_file.write_text(json.dumps(ledger))
    stray = root/"out"/"model.selection.json"
    stray.write_text("not written by training")
    _set_steps(root, 7)
    with pytest.raises(project.GateError, match="model.selection.json"):
        project.run(root, echo=quiet)
    assert stray.read_text() == "not written by training" and (root/"out"/"model.negative-inputs").is_dir()
    assert len(calls) == 1


@pytest.mark.parametrize("remove", ["dataset/manifest.jsonl", "model.negative-inputs"])
def test_a_missing_secondary_output_reruns_its_stage(cnn_project, remove):
    import shutil
    root, calls = cnn_project
    path = root/"out"/remove
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    stage = "dataset" if remove.startswith("dataset") else "train"
    assert project.status(root)[stage] == "needs to run (outputs missing or changed)"
    report = project.run(root, echo=quiet)
    assert report[stage]["status"] == "ran" and path.exists()
    # A regenerated dataset is byte-identical, so training stays up to date; a lost sidecar retrains.
    assert len(calls) == (1 if stage == "dataset" else 2)


def test_an_unrecorded_training_artifact_makes_training_stale_and_is_refused(cnn_project):
    root, calls = cnn_project
    stray = root/"out"/"model.selection.json"
    stray.write_text("not written by training")
    assert project.status(root)["train"] == "needs to run (outputs missing or changed)"
    with pytest.raises(project.GateError, match="model.selection.json"):
        project.run(root, echo=quiet)
    assert stray.read_text() == "not written by training" and len(calls) == 1


def test_the_thread_cap_never_raises_a_lower_pool_of_the_same_library(monkeypatch):
    import threadpoolctl
    seen = []
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [
        {"user_api": "blas", "num_threads": 2}, {"user_api": "blas", "num_threads": 8},
        {"user_api": "openmp", "num_threads": 3}])

    class Limits:
        def __init__(self, limits): seen.append(limits)
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", Limits)
    with project._thread_cap({"recognizer": {"threads": 4}}) as n:
        assert n == 4
    assert seen == [{"blas": 2}]                          # 8 lowered to the smaller pool, 2 kept, openmp untouched
