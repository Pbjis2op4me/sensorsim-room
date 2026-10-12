#!/usr/bin/env python3
"""Record development/verification attempts for the frozen physical-feature bank.

No downloads or weights. Procedural fixtures probe named mechanisms, not room
realism. `--manifest` compares user-supplied window arrays through the same code:
JSON has features (edge.Features config), evidence_domain, and reference/candidate
objects with audio (relative .npy path), fs, audio_unit=digital_full_scale_peak_1,
records and optional starts_s. No resampling, normalization or silent filtering.

Verification requires a separately held registration, custody digest and control
record. These checks are not permission separation or independent review; the
verifier alone owns acceptance receipts. Each output directory is exclusive.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np
from scipy.signal import sosfilt

from sensorsim.edge import Features
from sensorsim.evaluation import power_audit as audit
from sensorsim.iot_mic import peaking


FEATURES = dict(fs=16000, window_s=1., n_fft=512, hop=160,
                n_mels=40, fmin_hz=60., fmax_hz=7600.)


def procedural_factory(groups, defect, seed):
    """Independent generated parents/sessions per side; fixed constructed effects.

    This is a same-pipeline synthetic null, not the session/room simulator or
    deployment audio. The temporal mutation reverses contiguous 20 ms frames.
    It asserts no invariance of the complete CNN.
    """
    if defect not in ("null",)+audit.DEFECTS:
        raise ValueError("unregistered defect")
    rng = np.random.default_rng(seed)
    fs, n = FEATURES["fs"], int(FEATURES["fs"]*FEATURES["window_s"])
    t = np.arange(n)/fs
    bundles = []
    for side in range(2):
        audio, records = [], []
        for label, frequency in (("knock", 500.), ("doorbell", 1600.), ("kettle", 3600.)):
            for group in range(groups):
                identity = f"{seed}:{side}:{label}:{group}"
                envelope = np.exp(-t/rng.uniform(.18, .38))
                x = .08 * (rng.normal(size=n) + .2*np.sin(2*np.pi*frequency*t+rng.uniform(0, 6))) * envelope
                x *= 10**(rng.normal(0, .8)/20)
                if side == 1:
                    if defect == "scalar_plus6db":
                        x = x * 10**(6/20)
                    elif defect == "eq_3khz_plus2db":
                        x = sosfilt(peaking(3000., 2., 1.5, fs), x)
                    elif defect == "frame_reversal":
                        x = x.reshape(-1, 320)[::-1].reshape(-1)
                audio.append(x)
                records.append(dict(source_group=identity, independence_group=identity,
                    session_id=identity, recording_id=identity, background_groups=[],
                    parent_audio_sha256=[], label=label, route="generated", role="development",
                    condition_id="same-pipeline", framing_id="one-second-external", framing_basis="external_schedule"))
        bundles.append(audit.feature_bundle(np.array(audio), Features(FEATURES), records,
                                            qualification_scope=procedural_scope()))
    return tuple(bundles)


def procedural_scope():
    """Frozen design, independent of replicate seed, group count and treatment arm."""
    return dict(schema="qualification-scope/1", generator=dict(
        version="original-procedural-mechanisms/1",
        code_sha256=sha256(inspect.getsource(procedural_factory).encode()).hexdigest(),
        configuration=dict(features=FEATURES, frequencies_hz=[500., 1600., 3600.],
                           amplitude=.08, tone_amplitude=.2, phase_uniform_radians=[0., 6.],
                           decay_uniform_s=[.18, .38], level_std_db=.8,
                           noise="independent standard Gaussian samples")),
        null_population=dict(condition_id="same-pipeline", route="generated",
                             sources_sessions="independent by side, class and group",
                             framing="one second external schedule", windows_per_group=1,
                             audio_unit="digital_full_scale_peak_1"),
        effects=dict(eq_3khz_plus2db=dict(gain_db=2., center_hz=3000., q=1.5),
                     scalar_plus6db=dict(gain_db=6.),
                     frame_reversal=dict(frame_samples=320, fs_hz=16000, operation="reverse frame order")))


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    features = Features(manifest["features"])
    bundles = []
    for side in ("reference", "candidate"):
        row = manifest[side]
        if row["audio_unit"] != "digital_full_scale_peak_1" or row["fs"] != features.cfg["fs"]:
            raise ValueError("incompatible units or rate; no implicit conversion")
        audio_path = path.parent / row["audio"]
        actual = sha256(audio_path.read_bytes()).hexdigest()
        if actual != row["file_sha256"]:
            raise ValueError("audio file digest mismatch")
        audio = np.load(audio_path, allow_pickle=False)
        if audio.dtype.kind != "f":
            raise ValueError("explicit floating digital full-scale audio required")
        bundles.append(audit.feature_bundle(audio, features, row["records"], starts_s=row.get("starts_s"),
                                            qualification_scope=manifest.get("qualification_scope")))
    return bundles, manifest["evidence_domain"]


def environment():
    modules = {}
    for name in ("sensorsim.evaluation.power_audit", "sensorsim.evaluation.c2st",
                 "sensorsim.evaluation.realism", "sensorsim.edge", "sensorsim.iot_mic", "numpy", "scipy"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        modules[name] = dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest(),
                             version=getattr(module, "__version__", None))
    process = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                git_revision=process.stdout.strip(), modules=modules,
                tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def environment_identity(env):
    """Content/version identity portable across verifier snapshot locations.

    Actual paths and Git metadata stay in every attempt. The verifier separately
    checks the named tree; this identity binds the code that Python imports.
    """
    return {k: ({name: {field: value for field, value in module.items() if field != "path"}
                 for name, module in v.items()} if k == "modules" else v)
            for k, v in env.items() if k not in ("executable", "git_revision")}


def procedural_scope_identity():
    # No audio or prospective seeds are consumed to compute the scope binding.
    rows = [dict(label=label, route="generated", condition_id="same-pipeline",
                 framing_id="one-second-external", framing_basis="external_schedule")
            for label in ("knock", "doorbell", "kettle")]
    bundle = dict(records=rows, qualification_scope=procedural_scope())
    return audit.qualification_scope(bundle, bundle)["sha256"]


def verifier_inputs(args, spec, budget, env):
    if not all((args.registration, args.custody, args.control_record)):
        raise ValueError("verification requires registration, independent custody and signed controls")
    registration_path = Path(args.registration).resolve()
    custody_path = Path(args.custody).resolve()
    if custody_path.parent == registration_path.parent or custody_path == registration_path:
        raise ValueError("custody must be held outside the registration directory")
    registration_sha = sha256(registration_path.read_bytes()).hexdigest()
    if custody_path.read_text().split()[0] != registration_sha:
        raise ValueError("custody digest mismatch")
    # YAML is allowed, including JSON (a YAML subset).
    import yaml
    registration = yaml.safe_load(registration_path.read_text())
    if registration["family_sha256"] != spec.sha256 or registration["budget_sha256"] != audit.digest(asdict(budget)):
        raise ValueError("frozen family/budget changed")
    if set(registration["code_sha256"]) != set(env["modules"]):
        raise ValueError("registration must freeze every imported dependency")
    if registration["environment_sha256"] != audit.digest(environment_identity(env)):
        raise ValueError("frozen environment/content changed")
    if registration.get("qualification_scope_sha256") != procedural_scope_identity():
        raise ValueError("frozen generator/acquisition/effect scope changed")
    root = Path(__file__).resolve().parents[1]
    for name, expected in registration["test_file_sha256"].items():
        if sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen test changed: {name}")
    for name, expected in registration["code_sha256"].items():
        if env["modules"][name]["sha256"] != expected:
            raise ValueError(f"frozen import changed: {name}")
    if registration["tool_sha256"] != env["tool_sha256"]:
        raise ValueError("frozen harness changed")
    controls_path = Path(args.control_record)
    control_sha = sha256(controls_path.read_bytes()).hexdigest()
    if control_sha != registration["control_record_sha256"]:
        raise ValueError("frozen controls changed")
    controls = json.loads(controls_path.read_text())
    if controls.get("decision") != "qualified" or not controls.get("verifier") or controls.get("family_sha256") != spec.sha256:
        raise ValueError("reviewer has not qualified controls for this family")
    if (controls.get("qualification_scope_sha256") != registration["qualification_scope_sha256"]
            or controls.get("implementation_sha256") != audit.implementation_fingerprint()
            or controls.get("environment_sha256") != registration["environment_sha256"]
            or controls.get("test_file_sha256") != registration["test_file_sha256"]):
        raise ValueError("control signoff does not bind this scope, implementation, environment and tests")
    required = registration["required_test_node_ids"]
    if (not required or controls.get("required_test_node_ids") != required
            or sorted(controls.get("collected_test_node_ids", [])) != sorted(required)
            or controls.get("unexpected_skips_xfails", 1) != 0
            or controls.get("known_correct_accepted") is not True
            or controls.get("known_bad_rejected") is not True):
        raise ValueError("incomplete or invalid independent control record")
    controls["record_sha256"] = control_sha
    return controls, registration_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-kind", choices=("development", "verification"), default="development")
    parser.add_argument("--manifest")
    parser.add_argument("--config", help="JSON {family: {...}, budget: {...}}; frozen before any run")
    parser.add_argument("--registration")
    parser.add_argument("--custody")
    parser.add_argument("--control-record")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = environment()
    metadata = dict(command=sys.argv if argv is None else argv, start=datetime.now(timezone.utc).isoformat(),
                    run_kind=args.run_kind, environment=env,
                    fixture="original procedural mechanisms/1", retry_policy="none")
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    save("attempt.json", metadata)
    code = 0
    # Raw Python stdout/stderr are captured even on a rejected or errored attempt.
    from contextlib import redirect_stdout, redirect_stderr
    with (output/"stdout.txt").open("w") as stdout, (output/"stderr.txt").open("w") as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                config = json.loads(Path(args.config).read_text()) if args.config else {}
                family_args = config.get("family", {})
                for key in ("strata", "tests"):
                    if key in family_args:
                        family_args[key] = tuple(tuple(x) for x in family_args[key])
                budget_args = config.get("budget", {})
                for key in ("group_counts", "defects"):
                    if key in budget_args:
                        budget_args[key] = tuple(budget_args[key])
                if args.run_kind == "development" and not config:
                    family_args["permutations"] = 19
                    budget_args.update(group_counts=(4,), null_datasets=2, alternative_datasets=2)
                spec, budget = audit.Family(**family_args), audit.Budget(**budget_args)
                metadata.update(family=asdict(spec), budget=asdict(budget), family_sha256=spec.sha256,
                                input_config_sha256=audit.digest(config))
                save("attempt.json", metadata)
                controls = None
                if args.run_kind == "verification":
                    controls, metadata["preregistration_sha256"] = verifier_inputs(args, spec, budget, env)
                    if args.manifest:
                        raise ValueError("verification manifest comparison needs a separately registered data runner")
                if args.manifest:
                    metadata["fixture"] = "declared manifest windows"
                    bundles, domain = load_manifest(args.manifest)
                    metadata["manifest_sha256"] = sha256(Path(args.manifest).read_bytes()).hexdigest()
                    result = audit.infer_difference(audit.run_family(*bundles, spec), evidence_domain=domain)
                else:
                    with (output/"replicates.jsonl").open("w") as raw:
                        def record(row):
                            raw.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                            raw.flush()
                        result = audit.qualify_test(procedural_factory, spec, budget, record=record, controls=controls)
                    result.pop("attempts")  # kept verbatim in replicates.jsonl
                save("result.json", result)
                print(json.dumps(result, sort_keys=True, allow_nan=False))
            except Exception:
                code = 1
                traceback.print_exc()
    metadata.update(end=datetime.now(timezone.utc).isoformat(), exit_code=code)
    save("attempt.json", metadata)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
