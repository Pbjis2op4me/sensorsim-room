"""Declared check positions: independent routing/analytic and frozen-gate oracles."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.io import wavfile
import yaml

from sensorsim.acoustics.room import fit, ingest, pipeline, rir
from sensorsim.acoustics.room.model import OCTAVE_HZ, RoomModel


@pytest.fixture
def session(tmp_path):
    source = (2., 2., 1.5)
    takes = []
    for i, receiver in enumerate(((3., 2., 1.5), (2., 3., 1.5), (4., 3., 1.5))):
        path = tmp_path / f"take{i}.wav"
        wavfile.write(path, 24000, np.zeros(2400, np.float32))
        takes.append(dict(file=str(path), source_xyz_m=source, receiver_xyz_m=receiver,
                          excitation="unknown", speaker_id="S1"))
    raw = dict(room_id="check-control", fs=24000, shoebox_dims_m=(7., 6., 4.), takes=takes)
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(raw))
    return raw


def marked(raw):
    value = deepcopy(raw)
    value["takes"][-1]["role"] = "check"
    return ingest.RecordingManifest.model_validate(value)


def records_for(manifest, folder):
    records = {}
    for i, take in enumerate(manifest.takes):
        distance = np.linalg.norm(np.subtract(take.source_xyz_m, take.receiver_xyz_m))
        onset = round(distance / 343 * manifest.fs)
        h = np.zeros(2400)
        h[onset] = .2 / distance
        h[1800] = .01
        name = f"{i}.npy"
        np.save(folder / name, h)
        records[i, 0] = dict(take=i, channel=0, mic_id=f"P{i}", source_xyz_m=take.source_xyz_m,
            receiver_xyz_m=take.receiver_xyz_m, is_reference=False, file=name,
            metadata=dict(recording_file=str(take.file), direct_samples=[onset], sound_speed_m_s=343.),
            bands={str(b):dict(excitation_supported=True, noise_power=0.,
                              t30=dict(valid=True, rt60_s=.3, r_squared=1.)) for b in OCTAVE_HZ})
    return records


def test_internal_role_defaults_serialize_exactly_and_check_is_validated(session):
    legacy = ingest.RecordingManifest.model_validate(session)
    assert all("role" not in t for t in legacy.model_dump(mode="json")["takes"])
    explicit = deepcopy(session)
    explicit["takes"][0]["role"] = "fit"
    assert ingest.RecordingManifest.model_validate(explicit).model_dump_json() == legacy.model_dump_json()
    assert marked(session).takes[-1].role == "check"
    for change in ({"role":"unknown"}, {"role":"check", "type":"real"},
                   {"role":"check", "type":"noise"}, {"role":"check", "is_reference":True}):
        value = deepcopy(session)
        value["takes"][-1].update(change)
        with pytest.raises(ValueError):
            ingest.RecordingManifest.model_validate(value)


def test_check_disjointness_uses_coordinates_and_files_not_names(session):
    for change in ({"receiver_xyz_m":session["takes"][0]["receiver_xyz_m"], "mic_ids":["renamed"]},
                   {"file":session["takes"][0]["file"]}):
        value = deepcopy(session)
        value["takes"][-1].update(role="check", **change)
        with pytest.raises(ValueError, match="check.*(position|file)"):
            ingest.RecordingManifest.model_validate(value)


def test_user_kit_preserves_check_role(session, tmp_path):
    from sensorsim.acoustics.room.session_sim import canonical_kit
    kit = canonical_kit()
    (tmp_path / "kit.json").write_text(kit.model_dump_json())
    wavfile.write(tmp_path / "long.wav", 48000, np.zeros(480000, np.int16))
    raw = dict(room_id="user-check", fs_hz=48000, room=dict(dims_m=dict(x=7,y=6,z=4)),
        positions=dict(speakers=dict(S1=[2,2,1.5]), mics=dict(P1=[3,2,1.5], C1=[4,3,1.5])),
        excitation=dict(type="ess", kit="kit.json"),
        takes=[dict(file="long.wav", type="sweep", speaker="S1", mic="C1", role="check")])
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert ingest.load_manifest(path).takes[0].role == "check"


def test_check_mutation_cannot_change_fallback_anchor_or_level(session, tmp_path):
    manifest = marked(session)
    records = records_for(manifest, tmp_path)
    cfg = pipeline.PipelineConfig(count=1, val_fraction=0, fit_backend="eyring-t30")
    initial = RoomModel(shoebox_dims_m=manifest.shoebox_dims_m)
    outputs = []
    for gain, t30 in ((1., 3.), (100., 30.)):
        check = records[2, 0]
        h = np.zeros(2400)
        h[check["metadata"]["direct_samples"][0]] = gain
        np.save(tmp_path / check["file"], h)
        for row in check["bands"].values():
            row["t30"]["rt60_s"] = t30
        model, report = pipeline._fit(manifest, records, initial, tmp_path, cfg, None)
        levels = pipeline.training_level_references(manifest, records, tmp_path)
        _, jobs = pipeline._direct_anchor_jobs(manifest, records,
            [dict(take=0, channel=0, reference=None)], tmp_path)
        outputs.append((model.to_json(), report, levels, jobs))
        assert all(b["measurement_count"] == 2 and b["measured_t30_s"] == .3
                   for b in report["bands"].values())
        reference, = levels.values()
        assert reference["amplitude_scale"] == pytest.approx(.2, abs=1e-10)
        assert {r["take"] for r in reference["observations"]} == {0, 1}
        assert jobs[0]["reference"][0] != 2
    assert outputs[0] == outputs[1]


def test_fitter_receives_all_declared_fit_positions_and_no_check_audio(session, tmp_path, monkeypatch):
    manifest = marked(session)
    records = records_for(manifest, tmp_path)
    initial = RoomModel(shoebox_dims_m=manifest.shoebox_dims_m)
    seen = []
    def optimizer(observations, model, *, config):
        seen.extend(o.id for o in observations)
        assert all(o.split == "train" for o in observations)
        return fit.FitResult(model, config, {})
    monkeypatch.setattr(fit, "fit_room", optimizer)
    monkeypatch.setattr(fit, "model_from_manifest", lambda _:initial)
    monkeypatch.setattr(fit, "_session_report", lambda *args:{})
    responses = [rir.RIR(manifest.fs, np.load(tmp_path / r["file"]), r["source_xyz_m"],
                        r["receiver_xyz_m"], r["metadata"]) for r in records.values() if r["take"] != 2]
    fit._fit_pipeline(manifest, responses, initial, fit.FitConfig(bootstrap=0))
    assert seen == ["take0:ch0", "take1:ch0"]


def test_statistical_rows_match_frozen_closedloop_gates_and_analytic_gain():
    from tests.validation.bench_room_closedloop.compare import compare_rirs
    fs = 24000
    rng = np.random.default_rng(37)
    h = rng.normal(size=12000) * np.exp(-3*np.log(10)*np.arange(12000)/fs/.15)
    h[:80] = 0
    h[80] = 3
    measured = rir.RIR(fs, h, (1,1,1), (2,1,1), {"direct_sample":80, "synthetic":True})
    predicted = rir.RIR(fs, h*2, (1,1,1), (2,1,1), {"direct_sample":80, "synthetic":True})
    rows = pipeline._check_metric_rows(measured, predicted, level_reference={"amplitude_scale":1.})
    ctx = dict(scenario="analytic",tier="T0",engine="control",impairment="none",
               eval_kind="heldout",src="S1",mic="C1",band_hz=None)
    expected = compare_rirs(h, h*2, fs, ctx)[1:]
    for oracle in expected:
        actual, = [r for r in rows if (r["band_hz"],r["metric"]) == (oracle["band_hz"],oracle["metric"])]
        assert actual["tolerance"] == oracle["gate"]["value"]
        assert actual["signed_error"] == pytest.approx(oracle["error"], abs=1e-10)
        assert actual["status"] == ("PASS" if oracle["pass"] else "FAIL")
        if actual["metric"] == "band_level_db":
            assert actual["signed_error"] == pytest.approx(20*np.log10(2), abs=1e-10)
        if actual["metric"] == "drr_db":
            assert actual["metric_policy"]["drr"]["definition"] == "drr/2"


def test_check_missing_support_is_unavailable_but_missing_prediction_fails():
    h = np.zeros(2400)
    h[80], h[1500] = 1., .1
    measured = rir.RIR(24000, h, (1,1,1), (2,1,1), dict(direct_sample=80, synthetic=True,
        excitation=dict(type="ess",f_start_hz=600.,f_end_hz=10000.)))
    rows = pipeline._check_metric_rows(measured, None, level_reference=None)
    assert all(r["status"] == "UNAVAILABLE" and "excitation" in r["reason"] for r in rows if r["band_hz"] == 500)
    level, = [r for r in rows if r["band_hz"] == 1000 and r["metric"] == "relative_band_level_db"]
    assert level["status"] == "FAIL" and level["predicted"] is None
    json.dumps(rows, allow_nan=False)


def test_pipeline_publishes_check_and_capture_qa_without_granting_accuracy(session, tmp_path, monkeypatch):
    from sensorsim.acoustics.room import capture_qa
    manifest = marked(session)
    sidecar = tmp_path / "qa.yaml"
    sidecar.write_text("test sidecar")
    manifest = manifest.model_copy(update={"capture_qa":sidecar})
    records = records_for(manifest, tmp_path)
    model = RoomModel(shoebox_dims_m=manifest.shoebox_dims_m, late_t60_s=(.15,)*7)
    monkeypatch.setattr(ingest, "load_manifest", lambda _:manifest)
    monkeypatch.setattr(pipeline, "_preflight", lambda *args, **kwargs:({}, []))
    monkeypatch.setattr(pipeline, "_measure", lambda *args:records)
    original_fit = pipeline._fit
    monkeypatch.setattr(pipeline, "_fit", lambda m,r,i,o,c,f:original_fit(m,r,model,tmp_path,c,None))
    monkeypatch.setattr(pipeline, "_dry", lambda *args:[])
    qa = dict(schema_version="capture-qa/1", status="review", takes=[], reason="pilot gain changed")
    monkeypatch.setattr(capture_qa, "build_capture_qa_report", lambda *args, **kwargs:qa)
    out = tmp_path / "out"
    # Records are small independent analytic impulse/echo controls, not fitted synth truth.
    for row in records.values():
        row["file"] = str(tmp_path / row["file"])
    report = pipeline.run_pipeline(tmp_path, out,
        pipeline.PipelineConfig(count=1,val_fraction=0,fit_backend="eyring-t30"))
    assert report["capture_qa"] == qa
    check = report["check_positions"]
    assert check["schema_version"] == "room-check-position/1"
    assert check["positions"][0]["take"] == 2
    assert check["summary"]["planned"] == 32
    assert check["summary"]["fail_count"] + check["summary"]["unavailable_count"] > 0
    assert (out / check["positions"][0]["prediction_file"]).is_file()
    markdown = (out / "report.md").read_text()
    assert markdown.index("Your held-out accuracy check") < markdown.index("Dry reliability")
    assert "Capture QA: review" in markdown
    assert "calibration residual only; no held-out measurement validation" not in markdown


def test_invalid_check_capture_does_not_abort_training_measurements(session, tmp_path, monkeypatch):
    manifest = marked(session)
    def extract(m, index, **kwargs):
        if index == 2:
            raise ValueError("clipped check take")
        samples = np.zeros((2400,1))
        samples[80] = 1
        return rir.RIR(m.fs, samples, m.takes[index].source_xyz_m,
                       m.takes[index].receiver_xyz_m, {"direct_samples":[80]})
    monkeypatch.setattr(rir, "extract_take", extract)
    records = pipeline._measure(manifest, tmp_path, pipeline.PipelineConfig())
    assert records[2,0]["error"] == "clipped check take"
    assert records[0,0]["file"] and records[1,0]["file"]


def test_templates_keep_one_explicit_check_for_every_placement_count(tmp_path):
    from scripts.room_sweep_kit import write_templates
    for count in (1, 3, 4):
        write_templates(tmp_path, placements=count)
        value = yaml.safe_load((tmp_path / "manifest_template.yaml").read_text())
        check, = [t for t in value["takes"] if t.get("role") == "check"]
        assert check == dict(file="S1_C1_sweep.wav",type="sweep",speaker="S1",mic="C1",role="check")
        assert "C1" in value["positions"]["mics"]
        assert len(value["positions"]["speakers"]) == count
