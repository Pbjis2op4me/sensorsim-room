"""Registered WP1(a-d) TRUTH-proxy acceptance run; excluded from the default suite."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from sensorsim import edge, project
from sensorsim.iot_mic import peaking
from sensorsim.acoustics.room.model import RoomModel, uniform_bands
from sensorsim.acoustics.room.session_sim import canonical_kit, simulate_session
from sensorsim.evaluation.realism import band_containing_hz
from tests.unit.evaluation._fixture import save_edge_model, write_float_wav


def _room_rir(fs: int):
    import pyroomacoustics as pra

    dims = (6.0, 5.0, 2.8)
    source = (1.2, 2.4, 1.2)
    receiver = (2.2, 2.4, 1.2)
    room = pra.ShoeBox(list(dims), fs=fs, materials=pra.Material(.25), max_order=12,
                       air_absorption=True)
    room.add_source(list(source))
    room.add_microphone_array(np.asarray(receiver, float)[:, None])
    room.compute_rir()
    h = np.asarray(room.rir[0][0], dtype=np.float64)
    truth = RoomModel(shoebox_dims_m=dims, absorption=uniform_bands(.25), room_id="truth-proxy")
    return truth, {"S1": source}, {"M1": receiver}, {("S1", "M1"): h}, h


def _procedural_take(fs: int, seed: int) -> np.ndarray:
    from sensorsim.acoustics.room.session_sim import broadband_transient

    transient = broadband_transient(fs, seed)
    audio = np.zeros(round(2.0*fs), np.float64)
    start = round(.45*fs)
    audio[start:start + len(transient)] = transient*2.0
    return audio


def _eq_3khz(audio: np.ndarray, fs: int) -> np.ndarray:
    return signal.sosfilt(peaking(3000.0, 6.0, 1.5, fs), audio).astype(np.float32)


def _mulaw_8khz(audio: np.ndarray) -> np.ndarray:
    down = signal.resample_poly(np.asarray(audio, float), 1, 2)
    encoded = np.sign(down)*np.log1p(255*np.minimum(np.abs(down), 1.0))/np.log1p(255)
    quantized = np.round(encoded*127)/127
    decoded = np.sign(quantized)*np.expm1(np.abs(quantized)*np.log1p(255))/255
    output = signal.resample_poly(decoded, 2, 1)
    if len(output) < len(audio):
        output = np.pad(output, (0, len(audio)-len(output)))
    return output[:len(audio)].astype(np.float32)


def _write_scenario(tmp_path: Path, model_path: Path, reference: list[tuple[str, np.ndarray]],
                    rendered: list[tuple[str, np.ndarray]]) -> tuple[Path, Path, list[dict]]:
    root = tmp_path
    evaluate = root/"evaluate"/"event"
    evaluate.mkdir(parents=True, exist_ok=True)
    out = root/"out"
    dataset = out/"dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    files = []
    for group, audio in reference:
        path = evaluate/f"{group}.wav"
        write_float_wav(path, audio)
        files.append({"label": "event", "path": path, "group": group,
                      "file": f"event/{path.name}", "data_kind": "proxy"})
    rows = []
    for index, (group, audio) in enumerate(rendered):
        name = f"render_{index:03d}.wav"
        path = dataset/name
        write_float_wav(path, audio)
        rows.append({"id": index, "file": name, "label": "event", "split": "train",
                     "parent_group": group, "source_xyz_m": [1.2, 2.4, 1.2],
                     "receiver_xyz_m": [2.2, 2.4, 1.2],
                     "dry_source": {"source_group": group,
                                    "provenance": {"route": "TRUTH proxy"}}})
    (dataset/"manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows))
    out.mkdir(exist_ok=True)
    (out/"model.npz").write_bytes(model_path.read_bytes())
    return root, out, files


def _run_evaluation(tmp_path: Path, model_path: Path, reference, rendered) -> tuple[dict, bytes]:
    _, out, files = _write_scenario(tmp_path, model_path, reference, rendered)
    project._evaluate_cnn(files, out, data_kind="proxy", comparison="raw-diagnostic")
    contents = (out/"evaluation.json").read_bytes()
    return json.loads(contents), contents


@pytest.mark.slow
def test_registered_wp1_acceptance_on_truth_proxy(tmp_path):
    """Run sensitivity, specificity, attribution and JSON reproducibility once, as registered."""
    pytest.importorskip("pyroomacoustics")
    fs = 48000
    truth_room, speakers, microphones, external_rirs, _ = _room_rir(fs)
    condition_ids = {"eq_ref": 1, "eq_artifact": 2, "mulaw_ref": 3, "mulaw_artifact": 4,
                     "null_a": 5, "null_b": 6}
    real_sources = {}
    for name, condition_id in condition_ids.items():
        for index in range(15):
            group = f"{name}_{index:02d}"
            seed = np.random.SeedSequence([20261008, condition_id, index]).generate_state(1)[0]
            real_sources[f"S1_M1_real_{group}.wav"] = _procedural_take(fs, int(seed))
    session_root = tmp_path/"truth-session"
    truth_root = tmp_path/"truth-evidence"
    truth = simulate_session(session_root, truth_room, speakers, microphones,
        truth_dir=truth_root, external_rirs=external_rirs, real_sources=real_sources,
        include_reference=False, kit=canonical_kit(), case="A", seed=20261008)
    capture_root = truth.manifest.parent
    model_path = tmp_path/"frozen-cnn-fixture.npz"
    model = save_edge_model(model_path, seed=20261008, width=48, blocks=4)
    groups: dict[str, list[tuple[str, np.ndarray]]] = {name: [] for name in condition_ids}
    for name in condition_ids:
        for index in range(15):
            group = f"{name}_{index:02d}"
            captured = edge.read_wav(str(capture_root/f"S1_M1_real_{group}.wav"), 16000)
            groups[name].append((group, captured))

    eq_reference = groups["eq_ref"]
    eq_artifact = [(f"eq-artifact-{group}", _eq_3khz(audio, 16000))
                   for group, audio in groups["eq_artifact"]]
    mulaw_reference = groups["mulaw_ref"]
    mulaw_artifact = [(f"mulaw-artifact-{group}", _mulaw_8khz(audio))
                      for group, audio in groups["mulaw_artifact"]]
    null_a = groups["null_a"]
    null_b = [(f"null-b-{group}", audio) for group, audio in groups["null_b"]]
    # Names are disjoint across the two classes so no take or parent can cross folds.
    eq_reference = [(f"eq-reference-{group}", audio) for group, audio in eq_reference]
    mulaw_reference = [(f"mulaw-reference-{group}", audio) for group, audio in mulaw_reference]
    null_a = [(f"null-a-{group}", audio) for group, audio in null_a]

    eq_report, _ = _run_evaluation(tmp_path/"eq", model_path, eq_reference, eq_artifact)
    mulaw_report, _ = _run_evaluation(tmp_path/"mulaw", model_path, mulaw_reference, mulaw_artifact)
    specificity_report, specificity_bytes = _run_evaluation(tmp_path/"specificity", model_path, null_a, null_b)
    # Replay the exact same project inputs and seed; evaluation.json must be byte-identical.
    _, specificity_replay = _run_evaluation(tmp_path/"specificity", model_path, null_a, null_b)

    sensitivity = {}
    for name, report in (("eq_3khz", eq_report), ("mulaw_8khz", mulaw_report)):
        reps = report["sim_vs_real_discriminator"]["event"]["representations"]
        sensitivity[name] = {}
        for representation in ("band_statistics", "pooled_embedding"):
            result = reps[representation]
            sensitivity[name][representation] = {
                "point_auc": result["point_auc"], "lower_95": result["lower_95"],
                "upper_95": result["upper_95"], "seed_intervals": result["cluster_bootstrap"],
                "pass": all(row["lower_95"] >= .70 for row in result["cluster_bootstrap"]),
            }
    specificity = {}
    reps = specificity_report["sim_vs_real_discriminator"]["event"]["representations"]
    for representation in ("band_statistics", "pooled_embedding"):
        result = reps[representation]
        specificity[representation] = {
            "point_auc": result["point_auc"], "lower_95": result["lower_95"],
            "upper_95": result["upper_95"], "seed_intervals": result["cluster_bootstrap"],
            "pass": all(row["lower_95"] <= .5 <= row["upper_95"] and row["upper_95"] <= .65
                         for row in result["cluster_bootstrap"]),
        }
    feature_report = eq_report["feature_space_realism"]["event"]["band_statistics"]
    expected_band = band_containing_hz(model.features, 3000.0)
    top3 = eq_report["sim_vs_real_discriminator"]["event"]["top5_bands_by_abs_smd"][:3]
    attribution = {"expected_3khz_band": int(expected_band), "top3": top3,
                   "pass": any(row["index"] == expected_band for row in top3)}
    reproducibility = {"byte_identical": specificity_bytes == specificity_replay,
                       "bytes": int(len(specificity_bytes))}
    parts = {
        "a_sensitivity": all(value[representation]["pass"]
                              for value in sensitivity.values()
                              for representation in ("band_statistics", "pooled_embedding")),
        "b_specificity": all(value["pass"] for value in specificity.values()),
        "c_attribution": attribution["pass"],
        "d_reproducibility": reproducibility["byte_identical"],
    }
    result = {
        "pre_registration_sha256": (Path(__file__).with_name("PREREGISTRATION.sha256").read_text().split()[0]),
        "result_label": "proxy result",
        "truth_proxy": "pyroomacoustics shoebox RIR plus session_sim captures; project-generated procedural transients",
        "embedding_fixture_note": "Frozen seeded edge-compatible CNN fixture validates pooled-feature plumbing; it is not a trained recognizer.",
        "measured_results": None,
        "source_groups_per_class": 15,
        "grouped_cv_folds": 5,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "attribution": attribution,
        "reproducibility": reproducibility,
        "parts": parts,
        "acceptance": "pass" if all(parts.values()) else "fail",
        "e3_real_take_recall": "inconclusive: no user takes or trained deployment model were supplied",
        "feature_space_realism": feature_report,
        "session_evidence": str(truth.evidence.name),
    }
    evidence = tmp_path/"acceptance_truth.json"
    evidence.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
    if os.environ.get("SENSORSIM_EVALUATION_EVIDENCE"):
        destination = Path(os.environ["SENSORSIM_EVALUATION_EVIDENCE"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination/evidence.name).write_bytes(evidence.read_bytes())
    assert result["acceptance"] == "pass", evidence.read_text()
