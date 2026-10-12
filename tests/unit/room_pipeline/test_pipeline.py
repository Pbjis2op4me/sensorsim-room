"""Frozen integration tolerances: parent-project note room-pipeline (not distributed) (synthetic only)."""
import json
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile
import yaml

from sensorsim.core.fingerprint import sha256_file
from sensorsim.acoustics.room import SweepKit, generate_kit_playback
from sensorsim.acoustics.room import pipeline
from sensorsim.acoustics.room.model import BoxObstacle, RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from sensorsim.acoustics.room.pipeline import PipelineConfig, run_pipeline


@pytest.fixture(scope="module", autouse=True)
def fast_safe_yaml():
    # Exercise the real fitter/catalogue with the equivalent libyaml safe parser.
    # Its cold catalogue parse otherwise dominates this sub-second fixture.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(yaml, "SafeLoader", getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        yield


def fake_session(folder, *, full=False):
    from tests.room_session_cache import generated_session

    return generated_session(folder, full=full, build=_build_fake_session,
                             dependencies=(synthesize, generate_kit_playback,
                                           RoomModel, BoxObstacle, SweepKit, SynthConfig,
                                           signal.fftconvolve, wavfile.write, yaml.safe_dump))


def _build_fake_session(folder, *, full=False):
    folder.mkdir()
    fs = 48000 if full else 24000
    kit = SweepKit(
        fs_hz=fs, f1_hz=40, f2_hz=20000 if full else 11500,
        sweep_s=10 if full else .5, fade_in_s=.1 if full else .01,
        fade_out_s=.01 if full else .005, peak_dbfs=-6,
        lead_silence_s=2 if full else .1, tail_silence_s=4 if full else 2.1,
        end_silence_s=2 if full else .1, repeats=4,
        period_s=14 if full else 2.6, first_sweep_start_s=2 if full else .1,
        formula="x(t)=sin(2*pi*f1*L*(exp(t/L)-1)), L=T/ln(f2/f1); half-Hann fades",
        reference="Farina 2000; shortened integration fixture" if not full else "Farina 2000; canonical kit parameters",
        dither="float fixture; no PCM quantization",
    )
    (folder / "sweep_kit.json").write_text(kit.model_dump_json())
    room = RoomModel(shoebox_dims_m=(6, 5, 3), late_t60_s=(.65,) * 7,
                     obstacles=(BoxObstacle(name="desk", min_xyz=(2.5, 2, 0),
                                            size_xyz=(1, .6, .7), material="wood"),))
    source = (1.2, 1.1, 1.25)
    # Three profiling positions exercise the installed fitter's train/holdout
    # path. Two separate real performances suffice for the recognition split.
    mics = {"REF_S1": (1.5, 1.1, 1.25), "M1": (2.2, 1.1, 1.25),
            "M2": (1.2, 2.1, 1.25), "M3": (.7, 1.9, 1.25)}
    rng = np.random.default_rng(600)
    playback = generate_kit_playback(kit)
    latency = 317
    truth, takes = {}, []
    for i, (mic, xyz) in enumerate(mics.items()):
        rir = synthesize(room, source, xyz, fs, seed=11+i,
                         config=SynthConfig(duration_s=2.0))
        wet = np.r_[np.zeros(latency), signal.fftconvolve(playback, rir.samples)]
        wet += rng.normal(0, 1e-6, len(wet))
        wavfile.write(folder / f"{mic}_sweep.wav", fs, wet.astype(np.float32))
        takes.append(dict(file=f"{mic}_sweep.wav", type="sweep", speaker="S1", mic=mic))
        if i >= 2:
            continue
        t = np.arange(round(.25*fs)) / fs
        dry = .03 * np.sin(np.pi*np.arange(len(t))/len(t))**2 * (
            np.sin(2*np.pi*731*t+i) + .4*np.sin(2*np.pi*1631*t+.3*i))
        real = np.r_[np.zeros(latency), signal.fftconvolve(dry, rir.samples)]
        real += rng.normal(0, 1e-6, len(real))
        wavfile.write(folder / f"{mic}_real.wav", fs, real.astype(np.float32))
        takes.append(dict(file=f"{mic}_real.wav", type="real", speaker="S1", mic=mic,
                          label=f"test tone {i}"))
        truth[mic] = np.r_[np.zeros(latency), dry]
    wavfile.write(folder / "noise_floor.wav", fs, rng.normal(0, 1e-6, fs//4).astype(np.float32))
    takes.append(dict(file="noise_floor.wav", type="noise", mic="M1"))
    raw = dict(room_id="fake-user", fs_hz=fs, room={"dims_m": dict(zip("xyz", room.shoebox_dims_m))},
               obstacles=[dict(name=b.name, min_xyz=b.min_xyz, size_xyz=b.size_xyz, material=b.material)
                          for b in room.obstacles],
               positions={"speakers": {"S1": source}, "mics": mics}, takes=takes,
               excitation={"type": "ess", "kit": "sweep_kit.json"})
    (folder / "manifest.yaml").write_text(yaml.safe_dump(raw))
    return folder, room, truth, latency + 64


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    return fake_session(tmp_path_factory.mktemp("session") / "input")


def config(**overrides):
    values = dict(count=2, seed=71, rir_duration_s=2.0,
                  augmentation={"gain_db": [-9, -6], "pitch_semitones": [-.2, .2],
                                "noise_snr_schedule": [[0, 35]]})
    return PipelineConfig(**(values | overrides))


def session_manifest(folder):
    raw = yaml.safe_load((folder / "manifest.yaml").read_text())
    for take in raw["takes"]:
        take["file"] = str(folder / take["file"])
    raw["excitation"]["kit"] = str(folder / "sweep_kit.json")
    return raw


@pytest.fixture(scope="module")
def completed_session(session, tmp_path_factory):
    out = tmp_path_factory.mktemp("completed") / "out"
    return run_pipeline(session[0], out, config()), out


def assert_session(report, out, room, truth, latency):
    assert report["state"] == "complete"
    model = RoomModel.from_json((out / "room_model.json").read_text())
    assert model.shoebox_dims_m == room.shoebox_dims_m
    assert [(b.min_xyz, b.size_xyz, b.material) for b in model.obstacles] == [
        (b.min_xyz, b.size_xyz, b.material) for b in room.obstacles]
    assert report["fit"]["method"] == "room.fit.fit_room"
    assert report["modal_region"] == model.modal_region()
    assert "Schroeder frequency" in (out / "report.md").read_text()
    fitted = model.provenance["pipeline_fit_report"]
    train = {tuple(r["position"]) for r in fitted["split"] if r["split"] == "train"}
    holdout = {tuple(r["position"]) for r in fitted["split"] if r["split"] == "test"}
    assert len(train) == 2 and len(holdout) == 1 and train.isdisjoint(holdout)
    assert {r["split"] for r in fitted["metrics"]} == {"train", "test"}
    assert {int(band) for band, row in report["fit"]["bands"].items()
            if row["status"] == "fitted"} == set(model.provenance["fitted_bands_hz"])
    assert json.loads((out / "run.json").read_text())["fit_backend"] == {
        "method": "room.fit.fit_room", "fallback_reason": None,
        "selected": "room.fit.fit_room", "options": {}}
    for band in (500, 1000, 2000, 4000):
        row = report["fit"]["bands"][str(band)]
        assert row["status"] == "fitted"
        assert abs(row["measured_t30_s"]/.65-1) < .15, row
    for measurement in report["measurements"]:
        assert abs(measurement["metadata"]["latency_s"][0]*report["fs"]-latency) <= 2
    assert len(report["dry"]) == 2
    for row in report["dry"]:
        fs, dry = wavfile.read(out / row["file"])
        expected = truth[row["mic_id"]]
        n = min(len(dry), len(expected))
        a, b = dry[:n].astype(float), expected[:n]
        corr = signal.correlate(a, b)
        lag = signal.correlation_lags(len(a), len(b))
        score = max(corr[abs(lag) <= round(.005*fs)]) / np.sqrt(a@a * (b@b))
        assert score > .90
        assert row["untrustworthy_bands_hz"]  # physical-source directivity is unverified
    rows = [json.loads(line) for line in (out / "dataset/manifest.jsonl").read_text().splitlines()]
    # sensorsim-room contract: each label here has ONE recording, so both stay in training (a class
    # only in validation could never be learned) and validation is empty, with a warning.
    assert len(rows) == 4 and {r["label"] for r in rows} == {"test tone 0", "test tone 1"}
    assert report["dataset"]["unique_position_pairs"] == 2
    assert report["dataset"]["split_counts"] == {"train": 4, "val": 0}
    assert report["dataset"]["val_fraction_effective"] == 0.
    assert report["dataset"]["warnings"] == [w for w in report["dataset"]["warnings"]
                                             if "no independent validation recording" in w] != []
    point_splits = {}
    take_splits = {}
    for row in rows:
        assert isinstance(row["id"], int)  # preserve room-dataset-item/1 index type
        assert "/" not in row["file"]  # one dataset, no per-take subdirectories
        source = row["dry_source"]
        take = source["provenance"]["take"]
        assert source["source_group"] == f"take-{take:04d}"
        assert take not in take_splits or take_splits[take] == row["split"]
        take_splits[take] = row["split"]
        assert (out / "dataset" / source["path"]).is_file()
        fs, samples = wavfile.read(out / "dataset" / row["file"])
        assert fs == report["fs"] and samples.dtype == np.int16
        assert np.any(samples) and np.max(np.abs(samples.astype(int))) < 32768
        assert len(samples) > round(.25*fs)
        assert sha256_file(out / "dataset" / row["file"]) == row["output_sha256"]
        # Source identity: the package version, plus a commit only when run from its own
        # checkout (never the caller's repository; see core.fingerprint.source_revision).
        environment = row["resolved_config"]["environment"]
        assert environment["package_version"] or environment["git_hash"]
        assert row["rir"]["obstacles"]["boxes"] == 1
        assert row["rir"]["obstacles"]["blocked_paths"] > 0
        for point in (row["source_xyz_m"], row["receiver_xyz_m"]):
            assert all(0 < x < limit for x, limit in zip(point, room.shoebox_dims_m))
            assert not any(b.contains(point) for b in model.obstacles)
            key = tuple(point)
            assert key not in point_splits or point_splits[key] == row["split"]
            point_splits[key] = row["split"]
    assert set(point_splits.values()) == {"train"}  # no validation recording -> no validation positions
    assert len(report["previews"]) == 3
    assert report["previews"][1]["source"] == rows[0]["dry_source"]["provenance"]["file"]
    assert (out / "report.md").is_file()


def test_fast_fake_user_session(session, completed_session):
    report, out = completed_session
    assert_session(report, out, *session[1:])


def test_pipeline_outputs_are_byte_identical(session, completed_session, tmp_path):
    _, first = completed_session
    again = tmp_path / "again"
    run_pipeline(session[0], again, config())
    files = {p.relative_to(first) for p in first.rglob("*") if p.is_file()}
    assert files == {p.relative_to(again) for p in again.rglob("*") if p.is_file()}
    for path in sorted(files):
        assert (first / path).read_bytes() == (again / path).read_bytes(), path


@pytest.mark.parametrize("recordings", ["mono", "multichannel", "identical-copies"])
def test_no_independent_validation_recording_warns(session, tmp_path, recordings):
    folder = session[0]
    raw = session_manifest(folder)
    real = [t for t in raw["takes"] if t["type"] == "real"]
    raw["takes"] = [t for t in raw["takes"] if t["type"] != "real"]
    take = dict(real[0])
    if recordings == "multichannel":
        fs, first = wavfile.read(real[0]["file"])
        _, second = wavfile.read(real[1]["file"])
        path = tmp_path / "stereo.wav"
        wavfile.write(path, fs, np.column_stack((first, second)))
        take.update(file=str(path), mic=[t["mic"] for t in real])
    raw["takes"].append(take)
    if recordings == "identical-copies":
        path = tmp_path / "renamed.wav"
        path.write_bytes((folder / "REF_S1_real.wav").read_bytes())
        raw["takes"].append({**take, "file": str(path)})
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(raw))
    out = tmp_path / "out"
    # Keep all pipeline defaults for the first real take, except the fixture's
    # required 2 s measurement window (the existing insufficient-decay gate).
    requested = PipelineConfig(rir_duration_s=2.0) if recordings == "mono" else config()
    report = run_pipeline(tmp_path, out, requested)
    warning = ("no independent validation recording: validation split disabled; "
               "record \u22652 separate real takes for a leakage-free validation set")
    assert report["state"] == "complete"
    assert warning in report["warnings"]
    assert f"**WARNING: {warning}**" in (out / "report.md").read_text()
    run = json.loads((out / "run.json").read_text())
    dataset_run = json.loads((out / "dataset/run.json").read_text())
    assert warning in run["warnings"] and warning in dataset_run["warnings"]
    assert report["dataset"]["val_fraction_requested"] == .2
    assert report["dataset"]["val_fraction_effective"] == 0
    rows = [json.loads(line) for line in (out / "dataset/manifest.jsonl").read_text().splitlines()]
    assert len(rows) == requested.count * len(report["dry"])
    assert {row["split"] for row in rows} == {"train"}
    assert report["dataset"]["split_counts"] == {"train": len(rows), "val": 0}


@pytest.mark.slow
def test_full_canonical_kit_cli(tmp_path, python_subprocess_env):
    folder, room, truth, latency = fake_session(tmp_path / "session", full=True)
    (folder / "pipeline.yaml").write_text(yaml.safe_dump(config().model_dump(mode="json")))
    out = tmp_path / "out"
    completed = subprocess.run([sys.executable, "-m", "sensorsim.acoustics.room.pipeline",
                                "--session", str(folder), "--out", str(out)],
                               env=python_subprocess_env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "4 training clips" in completed.stdout  # both single-recording labels train (see above)
    assert_session(json.loads((out / "report.json").read_text()), out, room, truth, latency)


@pytest.mark.parametrize("problem,match", [
    ("sweep", "matching sweep"), ("reference", "REF"), ("noise", "noise"),
    ("duplicate", "ambiguous"), ("label", "label"),
])
def test_rejects_incomplete_handoffs(session, tmp_path, problem, match):
    folder = session[0]
    # Reuse immutable WAVs; only the manifest is different.
    raw = session_manifest(folder)
    if problem == "sweep":
        raw["takes"] = [t for t in raw["takes"] if not (t["type"] == "sweep" and t["mic"] == "M1")]
    elif problem == "reference":
        raw["takes"] = [t for t in raw["takes"] if t["mic"] != "REF_S1"]
    elif problem == "noise":
        raw["takes"] = [t for t in raw["takes"] if t["type"] != "noise"]
    elif problem == "duplicate":
        raw["takes"].append(raw["takes"][0])
    else:
        del next(t for t in raw["takes"] if t["type"] == "real")["label"]
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=match):
        run_pipeline(tmp_path, tmp_path / "out", config())
    assert not (tmp_path / "out/report.json").exists()


def test_output_protection_and_bad_config(session, tmp_path):
    sentinel = tmp_path / "important.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        run_pipeline(session[0], tmp_path, config())
    assert sentinel.read_text() == "keep"
    with pytest.raises(ValueError):
        PipelineConfig(augmentation={"gian_db": [-3, 3]})


@pytest.mark.parametrize("coverage", ["one-position", "ref-only", "colocated", "second-speaker"])
def test_fit_requires_positions_per_speaker_before_extraction(session, tmp_path, monkeypatch, coverage):
    raw = session_manifest(session[0])
    if coverage in ("one-position", "ref-only"):
        keep = {"REF_S1", "M1"} if coverage == "one-position" else {"REF_S1"}
        raw["takes"] = [t for t in raw["takes"] if t["type"] == "noise" or t["mic"] in keep]
    elif coverage == "colocated":
        for mic in ("M2", "M3"):
            raw["positions"]["mics"][mic] = raw["positions"]["mics"]["M1"]
    else:
        raw["positions"]["speakers"]["S2"] = [4.2, 1.1, 1.25]
        take = next(t for t in raw["takes"] if t["type"] == "sweep" and t["mic"] == "M2")
        raw["takes"].append({**take, "speaker": "S2"})
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(raw))
    def no_extraction(*args):
        pytest.fail("receiver coverage must be checked before RIR extraction")
    monkeypatch.setattr(pipeline, "_measure", no_extraction)
    with pytest.raises(ValueError, match=(
            "need ≥2 non-REF mic positions per speaker for a validated room fit; "
            "see room_recording_kit/README.md")):
        run_pipeline(tmp_path, tmp_path / "out", config())
    assert not (tmp_path / "out").exists()


def test_missing_fitter_is_explicit_eyring_fallback(session, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sensorsim.acoustics.room.fit", None)
    raw = session_manifest(session[0])
    raw["takes"] = [t for t in raw["takes"] if t["mic"] not in ("M2", "M3")]
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(raw))
    out = tmp_path / "out"
    report = run_pipeline(tmp_path, out, config(fit_backend="eyring-t30"))
    assert report["state"] == "complete"
    assert report["fit"]["method"] == "eyring-t30-fallback"
    backend = json.loads((out / "run.json").read_text())["fit_backend"]
    assert backend == report["fit_backend"]
    assert backend["method"] == "eyring-t30-fallback"
    assert backend["selected"] == "eyring-t30"
    assert backend["fallback_reason"] == "explicit eyring-t30 selection"


def test_insufficient_decay_is_incomplete_not_an_invented_fit(session, tmp_path):
    # A .65 s decay at this high SNR needs >1.3 s to reach a noise-only tail. A FIXED
    # window is requested explicitly (adaptive windows are the kit-declared default).
    with pytest.raises(ValueError, match="no usable non-REF per-band T30"):
        run_pipeline(session[0], tmp_path / "out", config(rir_duration_s=1.3, adaptive_rir_window=False))
    state = json.loads((tmp_path / "out/run.json").read_text())
    assert state["state"] == "incomplete" and state["stage"] == "fit"
    assert "no usable non-REF per-band T30" in state["error"]
    assert not (tmp_path / "out/room_model.json").exists()
    assert not (tmp_path / "out/dataset").exists()
    assert not (tmp_path / "out/report.json").exists()


def test_present_fitter_is_called_and_failure_is_not_hidden(session, tmp_path, monkeypatch):
    module = ModuleType("sensorsim.acoustics.room.fit")
    def fit_room(*, manifest, rirs, initial_model):
        assert manifest.room_id == "fake-user" and len(rirs) == 4
        assert initial_model.obstacles
        raise RuntimeError("fitter failure is evidence")
    module.fit_room = fit_room
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="fitter failure"):
        run_pipeline(session[0], tmp_path / "out", config())
    assert json.loads((tmp_path / "out/run.json").read_text())["state"] == "incomplete"
    assert not (tmp_path / "out/report.json").exists()
