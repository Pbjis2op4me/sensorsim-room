"""Generated development fixtures only; P5 frozen acceptance belongs to verifier."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sensorsim.evaluation import power_audit as p
from sensorsim.iot_mic import peaking
from scipy.signal import sosfilt
from tools.power_audit import main, procedural_factory, FEATURES


def test_eq_control_independent_center_frequency_amplitude():
    # Analytic sinusoidal steady-state amplitude at the filter's center; avoids
    # using its coefficients to calculate the expected response.
    fs = 16000
    signal = np.sin(2*np.pi*3000*np.arange(2*fs)/fs)
    filtered = sosfilt(peaking(3000, 2, 1.5, fs), signal)
    ratio = np.linalg.norm(filtered[fs:])/np.linalg.norm(signal[fs:])
    assert 20*np.log10(ratio) == pytest.approx(2., abs=1e-10)
    assert abs(20*np.log10(np.linalg.norm(signal[fs:])/np.linalg.norm(signal[fs:]))) < 1e-12


def test_cli_fresh_process_retains_raw_attempt_and_low_power(tmp_path, python_subprocess_env):
    output = tmp_path/"development"
    script = Path(__file__).resolve().parents[3]/'tools/power_audit.py'
    process = subprocess.run([sys.executable, str(script), "--output", str(output)],
                             env=python_subprocess_env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    attempt = json.loads((output/"attempt.json").read_text())
    result = json.loads((output/"result.json").read_text())
    rows = [json.loads(line) for line in (output/"replicates.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert attempt["start"] <= attempt["end"]
    assert attempt["exit_code"] == 0 and attempt["run_kind"] == "development"
    assert result["cells"][0]["harness"] == "unqualified"
    assert result["cells"][0]["low_power"]
    assert result["cells"][0]["registered_outcome"] == "inconclusive"
    assert attempt["environment"]["modules"]["sensorsim.evaluation.power_audit"]["path"].endswith("evaluation/power_audit.py")
    assert Path(attempt["environment"]["modules"]["sensorsim.evaluation.power_audit"]["path"]) == Path(p.__file__).resolve()
    assert all(row["split_sha256"] and row["feature_sha256"] for row in rows)
    assert (output/"stdout.txt").read_text()
    assert not (output/"stderr.txt").read_text()


def test_verification_cannot_start_without_custody_and_signoff(tmp_path):
    output = tmp_path/"not-authorized"
    assert main(["--output", str(output), "--run-kind", "verification"]) == 1
    attempt = json.loads((output/"attempt.json").read_text())
    assert attempt["exit_code"] == 1
    assert "requires registration" in (output/"stderr.txt").read_text()
    assert not (output/"replicates.jsonl").exists()


def test_invalid_config_is_preserved_as_errored_attempt(tmp_path):
    config = tmp_path/"bad.json"
    config.write_text('{"family": {"permutations": 0}}')
    output = tmp_path/"invalid-config"
    assert main(["--output", str(output), "--config", str(config)]) == 1
    assert json.loads((output/"attempt.json").read_text())["exit_code"] == 1
    assert "invalid alpha" in (output/"stderr.txt").read_text()


def test_manifest_reuses_feature_and_inference_path(tmp_path):
    from hashlib import sha256
    from tools.power_audit import load_manifest
    # Original generated arrays stand in for future user window files; the
    # deployed feature config, declared provenance and units remain mandatory.
    bundles = procedural_factory(4, "null", 77)
    manifest = dict(features=FEATURES, evidence_domain="synthetic proxy")
    for side, bundle in zip(("reference", "candidate"), bundles):
        rng = np.random.default_rng(1 if side == "reference" else 2)
        audio = rng.normal(0, .1, (12, 16000))
        path = tmp_path/f"{side}.npy"
        np.save(path, audio, allow_pickle=False)
        records = [{k: v for k, v in row.items() if k != "audio_sha256"} for row in bundle["records"]]
        manifest[side] = dict(audio=path.name, records=records, fs=16000,
                              audio_unit="digital_full_scale_peak_1", file_sha256=sha256(path.read_bytes()).hexdigest())
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    loaded, domain = load_manifest(path)
    raw = p.run_family(*loaded, p.Family(permutations=19))
    assert raw["valid"], raw["errors"]
    assert domain == "synthetic proxy"
    manifest["reference"]["audio_unit"] = "int16"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible units"):
        load_manifest(path)


@pytest.mark.slow
def test_development_scalar_intervention_reaches_family_detector():
    # Deterministic software regression, not an independently replicated power
    # estimate. All labels receive the declared +6 dB alternative.
    family = p.Family(permutations=999)
    left, right = procedural_factory(15, "scalar_plus6db", 4801)
    result = p.run_family(left, right, family)
    assert result["valid"], result["errors"]
    assert result["difference"] == "detected"
    raw_members = [row for name, row in result["results"].items() if name.endswith("physical/max_mean")]
    assert all(row["raw_mean_difference"][0] > 4 for row in raw_members)
    assert p.infer_difference(result)["equivalence"] == "not_assessed"
