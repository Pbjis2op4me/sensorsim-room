"""One-command training pipeline: room session -> fitted room -> dataset -> recognizer -> evaluation.

A project is a folder. Only ``session/`` and ``background/`` are required::

    project.yaml               optional settings (DEFAULTS below)
    session/                   the filled manifest.yaml, sweep_kit.json and recorded WAVs (USAGE.md steps 2-3).
                               Labelled ``type: real`` takes are target sounds recorded in the room.
    targets/<label>/*.wav      optional clean recordings of target sounds made elsewhere; one file = one recording
    background/*.wav           ordinary sound of the room, at least two separate recordings
                               (or background/train/*.wav and background/val/*.wav)
    evaluate/<label>/*.wav     optional REAL labelled recordings, scored separately (evaluate/background/ too)

    python -m sensorsim.project run    PROJECT [--force STAGE]
    python -m sensorsim.project status PROJECT
    python -m sensorsim.project evaluate PROJECT
    python -m sensorsim.project detect PROJECT audio.wav ...
    python -m sensorsim.project export PROJECT FOLDER     (the CNN, ready to copy to the device)
    python -m sensorsim.project visualize PROJECT paths.html   (how sound travels in the fitted room)

Stages run in order -- fit, dataset, train, evaluate -- and write to ``out/``. Each records a hash of
its inputs (files, settings and the upstream stage's outputs) in ``out/stages.json``; an up-to-date
stage is skipped, so changing the dataset size never refits the room, and changing recognizer
settings only retrains. Gates report what needs attention and stop the run when continuing would
produce a meaningless model.

``recognizer: {type: cnn}`` (the default) trains the small CNN of ``sensorsim.nn`` -- PyTorch on the
training machine, numpy only on the device -- with simulated IoT microphones in the loop; its other
keys override ``sensorsim.nn.DEFAULTS``. ``type: baseline`` is the scikit-learn logistic regression
of ``sensorsim.recognizer`` (keys: BASELINE below), which needs no PyTorch but does not survive
cheap microphones (USAGE.md).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

from sensorsim.core.fingerprint import implementation_fingerprint, sha256_file

DEFAULTS = {
    "session": "session", "targets": "targets", "background": "background", "evaluate": "evaluate",
    "out": "out",
    "fit": {},                                                    # PipelineConfig overrides, e.g. {case: B}
    "dataset": {"count": 60, "fs": 16000, "seed": 1, "val_fraction": .2},
    "recognizer": {"type": "cnn"},
}
BASELINE = {"window_s": 2.0, "snr_db": [-5., 20.], "c": 1.0, "min_prob": .6}
STAGES = ("fit", "dataset", "train", "evaluate")
AUDIO = (".wav",)


class GateError(RuntimeError):
    """A stage produced something that would make every later stage meaningless."""


def load_project(path: str | Path) -> dict:
    root = Path(path).resolve()
    cfg = copy.deepcopy(DEFAULTS)
    file = root/"project.yaml"
    if file.is_file():
        given = yaml.safe_load(file.read_text()) or {}
        unknown = set(given) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown project.yaml keys: {sorted(unknown)}")
        for key, value in given.items():
            cfg[key] = {**cfg[key], **value} if isinstance(cfg[key], dict) else value
    # A surveyed room enables no-sweep fit automatically; its complete input tree
    # is fingerprinted by the fit stage so replacing playback invalidates caches.
    room_yaml = root/"room.yaml"
    if room_yaml.is_file():
        cfg["room_yaml"] = str(room_yaml)
        # The no-sweep drop-folder contract calls chamber recordings dry/.
        # Explicit project settings always take precedence; provenance is still
        # mandatory and a filename never grants intake approval.
        if (root/"dry").is_dir() and (not file.is_file() or "targets" not in given):
            cfg["targets"] = "dry"
    cfg["root"] = str(root)
    cfg["recognizer"] = _recognizer_settings(cfg)
    out = (root/cfg["out"]).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError(f"out ({cfg['out']!r}) must be a sub-folder inside the project")
    return cfg


def _recognizer_settings(cfg: dict) -> dict:
    """The recognizer section checked against its type's settings; unknown keys are an error."""
    r = dict(cfg["recognizer"])
    kind = r.pop("type", "cnn")
    if kind == "baseline":
        unknown = set(r) - set(BASELINE)
        if unknown:
            raise ValueError(f"unknown baseline recognizer settings {sorted(unknown)}; known: {sorted(BASELINE)}")
        return {"type": "baseline", **BASELINE, **r}
    if kind != "cnn":
        raise ValueError(f"recognizer type must be 'cnn' or 'baseline', not {kind!r}")
    from sensorsim.nn import DEFAULTS as cnn_defaults
    from sensorsim.event_levels import require_legacy_training
    require_legacy_training(r)
    from sensorsim.training_dynamics import DEFAULTS as dynamics_defaults
    known = {**cnn_defaults, **dynamics_defaults}
    unknown = set(r) - set(known)
    if unknown:
        raise ValueError(f"unknown cnn recognizer settings {sorted(unknown)}; known: {sorted(known)} "
                         "(min_prob and c belong to type: baseline)")
    if r.get('proxy_screen', False):
        raise ValueError('proxy_screen_only training is forbidden in project.run and production export')
    if r.get('partial_windows', False):
        raise ValueError('partial_windows is on hold: the negatives x partial class-swap screen has no independent acceptance')
    # deployment/1 is a separate opt-in policy; default promotion needs the
    # independent nuisance/class-swap screen, not just software controls.
    from sensorsim.window_policy import validate_config
    validate_config({**known, **r})
    from sensorsim.training_negatives import validate_config as validate_negatives
    validate_negatives({**known, **r})
    if r.get('validation_mode', 'synthetic') == 'hard':
        raise ValueError('validation_mode: hard is on hold: no independent class-swap control has accepted it')
    if r.get('validation_mode', 'synthetic') != 'synthetic':
        raise ValueError("validation_mode must be 'synthetic'; hard mode is on hold")
    fs = int(r.get("fs", known["fs"]))
    if int(cfg["dataset"]["fs"]) != fs:
        raise ValueError(f"the CNN listens at {fs} Hz: set dataset fs to {fs} (it is {cfg['dataset']['fs']})")
    return {"type": "cnn", **r}                 # sensorsim.nn.DEFAULTS fill in the rest when training


def _model_file(cfg: dict) -> Path:
    return _path(cfg, "out")/("model.npz" if cfg["recognizer"]["type"] == "cnn" else "model.json")


def _version(distribution: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _path(cfg: dict, key: str) -> Path:
    return Path(cfg["root"])/cfg[key]


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _tree(path: Path, suffixes: tuple[str, ...] | None = None) -> dict:
    """{relative path: sha256} of every file below `path` (empty when absent)."""
    if not path.exists():
        return {}
    return {p.relative_to(path).as_posix(): sha256_file(p) for p in sorted(path.rglob("*"))
            if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes)}


MARKER = ".sensorsim-stage"


def _clear(target: Path) -> None:
    """Remove a stage folder, but only one this pipeline created (it carries MARKER)."""
    if not target.exists():
        return
    if not (target/MARKER).is_file():
        raise GateError(f"refusing to delete {target}: it was not created by this pipeline; move it aside")
    shutil.rmtree(target)


def _mark(target: Path, stage: str) -> None:
    (target/MARKER).write_text(f"{stage}\n")


# What training writes beside the model (nn.train and the baseline trainer); nothing else in out/ is touched.
TRAIN_ARTIFACTS = ("model.npz", "model.json", "model.selection.json", "model.validation_scores.npz",
                   "model.cue-policy.json", "model.negative-inputs", "model.panel-inputs")


def _train_artifacts(out: Path) -> dict:
    """{name: content hash} of every training artifact present in out/."""
    found = [out/name for name in TRAIN_ARTIFACTS] + sorted(out.glob("model.checkpoint-*.npz"))
    return {p.name: (_digest(_tree(p)) if p.is_dir() else sha256_file(p)) for p in found if p.exists()}


def _clear_train_outputs(cfg: dict, out: Path, entry: dict) -> None:
    """Remove the previous training run's artifacts so a retrain starts clean.

    Only artifacts recorded by the last completed training run, and still unchanged, are removed. Anything
    else at those names (a calibrated copy, a file you put there, or leftovers of an interrupted run) stops
    the run instead. A ledger written before artifacts were recorded proves only the model file, and only
    while it still has the hash training recorded; any other artifact present then stops the run.
    """
    present = _train_artifacts(out)
    recorded = entry.get("artifacts")
    if recorded is None:
        model = _model_file(cfg)
        intact = model.is_file() and sha256_file(model) == entry.get("outputs")
        recorded = {model.name: entry["outputs"]} if intact else {}
    foreign = sorted(name for name, digest in present.items() if recorded.get(name) != digest)
    if foreign:
        raise GateError(f"refusing to replace {', '.join(str(out/n) for n in foreign)}: not written by this "
                        "pipeline's last training run, or changed since (for example a calibrated copy). Move it "
                        "aside, keeping any calibrated model under another name, and rerun")
    for name in present:
        path = out/name
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def _audio(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*") if p.suffix.lower() in AUDIO) if path.is_dir() else []


# ------------------------------------------------------------------------------------------- stages
def _fit(cfg: dict, out: Path) -> dict:
    from sensorsim.acoustics.room.pipeline import PipelineConfig, run_pipeline
    target = out/"fit"
    _clear(target)
    if cfg.get("room_yaml"):
        from sensorsim.acoustics.room.pipeline import run_no_sweep
        report = run_no_sweep(cfg["room_yaml"], target, fs=int(cfg["dataset"]["fs"]), seed=int(cfg["dataset"]["seed"]))
        _mark(target, "fit")
        return {"state": report["state"], "event_detection": True, "labels_without_a_usable_take": {},
                "check_positions": {"pass": 0, "total": 0},
                "targets_in_room": {"selected_per_label": {}, "refused": [], "crops_overlapping_another_event": []},
                "labels": [], "mode": "no-sweep", "calibration_status": report["calibration_status"],
                "ensemble_file": report["ensemble_file"], "result_label": "model-based",
                "rendering_distribution": report["rendering_distribution"], "warnings": report["warnings"],
                "room_warning": report.get("room_warning")}
    # The fit stage's own starter dataset is not used (the dataset stage builds the real one): keep it minimal.
    report = run_pipeline(_path(cfg, "session"), target,
                          PipelineConfig(**{**cfg["fit"], "count": 1, "val_fraction": 0.}))
    _mark(target, "fit")
    rows = [x for p in report.get("check_positions", {}).get("positions", []) for x in p["rows"]]
    selection = report.get("event_selection", {})
    flagged = []
    sel_file = target/"dry"/"event_selection.json"
    for s in (json.loads(sel_file.read_text()) if sel_file.is_file() else []):
        if s.get("event_index") is None:
            continue
        g = s["regions"][s["event_index"]]
        others = [r for i, r in enumerate(s["regions"]) if i != s["event_index"]
                  and r["onset_sample"] < g["segment_stop"] and r["offset_sample"] > g["segment_start"]]
        if others:
            flagged.append(Path(s["file"]).name)
    labels = {row["label"] for row in report.get("dry", [])}
    # Compare against what the manifest DECLARES, not only the pipeline's coverage report: a take with
    # no detectable event never enters that accounting, and its label would vanish without a word.
    from sensorsim.acoustics.room.ingest import load_manifest
    declared = load_manifest(_path(cfg, "session")/"manifest.yaml")
    real = [t for t in declared.takes if t.type == "real" and t.label]
    lost = sorted({t.label for t in real} - labels)
    return {"state": report.get("state"),
            "event_detection": declared.acquisition == "sequential_phone" or not real,
            "labels_without_a_usable_take": {lab: [Path(t.file).name for t in real if t.label == lab] for lab in lost},
            "check_positions": {"pass": sum(r["status"] == "PASS" for r in rows), "total": len(rows)},
            "targets_in_room": {"selected_per_label": selection.get("resolved_per_label", {}),
                                "refused": [{"file": Path(r["file"]).name, "reason": r["reason"]}
                                            for r in selection.get("rejected", [])],
                                "crops_overlapping_another_event": flagged},
            "labels": sorted(labels)}


def _clips(cfg: dict, out: Path):
    """Explicit chamber/digital intake, or the existing sweep-extraction path.

    Raw target files require dataset.sources provenance; an in-room recording
    cannot acquire a dry-source declaration merely by living under targets/.
    Explicit sources replace the legacy sweep-derived source list. Dataset
    generation assigns source groups before rendering; intake never splits crops.
    """
    from sensorsim.acoustics.room.dataset import DryClip
    from sensorsim.acoustics.room.source_reference import SourceReference
    from sensorsim.sources.intake import load_project_sources, is_target_path, require_session_string
    from sensorsim.sources.project_intake import excluded_roots, require_target_sites
    targets = _path(cfg, "targets")
    excluded = excluded_roots(cfg)
    if "sources" in cfg["dataset"]:
        from sensorsim.acoustics.room.capture_qa import discover_session_reference
        rows = copy.deepcopy(cfg["dataset"]["sources"])
        if not isinstance(rows, list) or not rows:
            raise ValueError("dataset.sources must be a nonempty list of provenance rows")
        require_target_sites(cfg, rows)
        for row in rows:
            require_session_string(row.get("session"))
            if row.get("capture_route") == "near_dry" and "session_reference" not in row:
                reference = discover_session_reference(targets, session=row.get("session"),
                                                       evidence_label=row.get("evidence_label"))
                if reference is not None:
                    row["session_reference"] = reference
        return load_project_sources(rows, targets, out/"source_qa.json", excluded)
    if any(p.suffix.lower() in AUDIO and is_target_path(p, targets, excluded) for p in targets.rglob("*")):
        raise ValueError("target recordings require mandatory provenance in dataset.sources; "
                         "declare source_group, object_id, session and capture_route")
    # Backward-compatible, already deconvolved sweep sources retain their actual
    # reference contract. This is not the no-sweep WP6 intake route.
    report = json.loads((out/"fit"/"report.json").read_text())
    return [DryClip(r["label"], out/"fit"/r["dataset_source"],
                    provenance={**r, "route": "sweep-extracted source", "take": r["take"]},
                    source_group=f"take-{r['take']:04d}",
                    source_reference=SourceReference.from_dict(r["source_reference"]))
            for r in report.get("dry", [])]


def _background_split(cfg: dict) -> tuple[list[Path], list[Path]]:
    root = _path(cfg, "background")
    if (root/"train").is_dir():
        return _audio(root/"train"), _audio(root/"val")
    files = _audio(root)
    if not files:
        raise GateError(f"no background recordings in {root}: record the room's ordinary sound "
                        "(at least two separate recordings) -- without it the recognizer reads every real event as background")
    n_val = len(files)//3 if len(files) >= 3 else (1 if len(files) == 2 else 0)
    return files[:len(files) - n_val], files[len(files) - n_val:]


def _dataset(cfg: dict, out: Path) -> dict:
    from sensorsim.acoustics.room.augment import AugmentationConfig
    from sensorsim.acoustics.room.dataset import SamplingSpec, generate_dataset
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim import recognizer
    clips = _clips(cfg, out)
    labels = sorted({c.label for c in clips})
    if not clips:
        raise GateError("no target sounds: add labelled `type: real` takes to the session or files under targets/<label>/")
    train_bg, val_bg = _background_split(cfg)
    target = out/"dataset"
    _clear(target)
    d = cfg["dataset"]
    ensemble_file = out/"fit"/"room_ensemble.json"
    if ensemble_file.is_file():
        from sensorsim.acoustics.room.prior import RoomEnsemble
        from sensorsim.acoustics.room.render_prior import select_render_ensemble, conflict_warning, ASSUMED_ROOM_WARNING
        model = RoomEnsemble.model_validate_json(ensemble_file.read_text())
        if model.status == "prior-conflict":
            raise GateError(conflict_warning(model))
        model = select_render_ensemble(model)
        if model.status == "assumed":
            import warnings
            fit_report = out/"fit"/"report.json"
            reason = json.loads(fit_report.read_text()).get("room_warning") if fit_report.is_file() else None
            warnings.warn(reason or ASSUMED_ROOM_WARNING, RuntimeWarning, stacklevel=2)
    else:
        model = RoomModel.model_validate_json((out/"fit"/"room_model.json").read_text())
    manifest = generate_dataset(model, SamplingSpec(), clips, AugmentationConfig(), int(d["count"]), int(d["seed"]),
                                target, fs=int(d["fs"]), val_fraction=float(d["val_fraction"]), all_sources=True,
                                allow_train_only=True)
    _mark(target, "dataset")
    added = recognizer.add_background(manifest, train_bg, val_bg)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    recordings = {lab: len({r["dry_source"]["source_group"] for r in rows if r["label"] == lab and "dry_source" in r})
                  for lab in labels}
    val = {r["label"] for r in rows if r["split"] == "val"}
    return {"clips": len(rows), "background_clips": added, "background_files": {"train": len(train_bg), "val": len(val_bg)},
            "recordings_per_label": recordings,
            "labels_without_validation": sorted(set(labels) - val),
            "labels_with_one_recording": sorted(k for k, v in recordings.items() if v < 2)}


def _train(cfg: dict, out: Path) -> dict:
    r = cfg["recognizer"]
    manifest = out/"dataset"/"manifest.jsonl"
    if r["type"] == "baseline":
        from sensorsim import recognizer
        model = recognizer.train(manifest, c=float(r["c"]), window_s=float(r["window_s"]),
                                 snr_db=tuple(float(x) for x in r["snr_db"]))
        (out/"model.json").write_text(json.dumps(model, indent=1) + "\n")
        t = model["training"]
        return {"type": "baseline", "classes": model["classes"], "train_windows": t["train_windows"],
                "synthetic_val_accuracy (optimistic)": t.get("val_accuracy"), "warnings": t["warnings"]}
    from sensorsim import nn
    try:
        nn._torch()
    except ImportError as error:
        raise GateError(f"{error}; or set `recognizer: {{type: baseline}}` in project.yaml") from error
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    missing = sorted({x["label"] for x in rows} - {x["label"] for x in rows if x["split"] == "val"})
    if missing:
        raise GateError(f"the CNN is selected on validation recordings and {missing} have none: record at least "
                        "two takes of each sound (and at least two background recordings)")
    settings = {k: v for k, v in r.items() if k != "type"}
    for key in ("negative_manifest", "selection_panel"):
        if settings.get(key):
            settings[key] = str(Path(cfg["root"])/settings[key])
    report = nn.train(manifest, out/"model.npz", **settings)
    return {"type": "cnn", "classes": report["classes"], "parameters": report["parameters"],
            "macs_per_window": report["macs_per_window"],
            "synthetic_val_mean_ap (optimistic)": report["best_val_mean_ap"], "detector": report["detector"],
            "selection": report.get("selection", {"mode": "unrecorded"}),
            "negatives": report.get("negatives", {"enabled": False}),
            "warnings": ["synthetic window AP is diagnostic only. Development selection remains optimistic; "
                         "a separate audit is required before deployment claims (DEPLOY.md)"]}


def _evaluate(cfg: dict, out: Path) -> dict:
    root = _path(cfg, "evaluate")
    data_kind_file = root/"data_kind.txt"
    has_data_kind_marker = data_kind_file.is_file()
    data_kind = data_kind_file.read_text().strip() if has_data_kind_marker else "measured"
    if has_data_kind_marker and data_kind not in {"measured", "proxy"}:
        raise GateError(f"{data_kind_file} must contain 'measured' or 'proxy'")
    files = _evaluation_files(root, data_kind)
    if not files:
        raise GateError(f"no labelled WAV takes in {root}")
    declared_kinds = {item["data_kind"] for item in files}
    if len(declared_kinds) > 1:
        raise GateError("one evaluation run must contain a single data kind; split measured and proxy takes")
    if has_data_kind_marker and declared_kinds != {data_kind}:
        raise GateError(f"{data_kind_file} conflicts with the data_kind values in the evaluation manifest")
    if not has_data_kind_marker:
        data_kind = next(iter(declared_kinds))
    if cfg["recognizer"]["type"] == "cnn":
        return _evaluate_cnn(files, out, data_kind=data_kind)
    from sensorsim import recognizer
    model = json.loads((out/"model.json").read_text())
    min_prob = float(cfg["recognizer"]["min_prob"])
    results, background_minutes, false_alarms = [], 0., 0
    for item in files:
        label, f = item["label"], item["path"]
        windows = recognizer.predict(model, f, min_prob=min_prob)
        found = sorted({w["label"] for w in windows} - {recognizer.UNKNOWN, "background"})
        row = {"file": item["file"], "label": label, "group": item["group"],
               "data_kind": data_kind, "recognized": found}
        if _negative_label(label):
            hop = (windows[1]["start_s"] - windows[0]["start_s"]) if len(windows) > 1 else model["features"]["window_s"]
            background_minutes += len(windows)*hop/60
            row["false_alarm_windows"] = sum(w["label"] not in ("background", recognizer.UNKNOWN) for w in windows)
            false_alarms += row["false_alarm_windows"]
        else:
            row["hit"] = label in found
        results.append(row)
    report = {"schema": "sensorsim-room-evaluation/1", "recognizer": "baseline",
              "data_kind": data_kind, "recordings": results,
              "real_recordings": len(files),
              "recall_per_label": {lab: f"{sum(r['hit'] for r in results if r['label'] == lab)}/"
                                         f"{sum(r['label'] == lab for r in results)}"
                                   for lab in sorted({r["label"] for r in results if "hit" in r})},
              "false_alarm_windows_per_background_minute":
                  round(false_alarms/background_minutes, 2) if background_minutes else None}
    out.mkdir(parents=True, exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return {"real_recordings": len(files), "recall_per_label": report["recall_per_label"],
            "false_alarms_per_background_minute": report["false_alarm_windows_per_background_minute"]}


def _negative_label(label: str) -> bool:
    value = label.casefold()
    return value in {"background", "negative"} or value.startswith("negative_")


def _evaluation_files(root: Path, data_kind: str) -> list[dict]:
    """Read grouped evaluation takes from a JSONL manifest, label folders, or named WAVs."""
    manifest = next((p for p in (root/"manifest.jsonl", root/"takes.jsonl") if p.is_file()), None)
    items = []
    if manifest is not None:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row["file"])
            if not path.is_absolute():
                path = root/path
            display = Path(row.get("file", path.name))
            if display.is_absolute():
                try:
                    display = display.resolve().relative_to(root.resolve())
                except ValueError:
                    display = Path(display.name)
            label = str(row["label"])
            items.append({"label": label, "path": path, "group": str(row.get("group", row.get(
                "parent_group", row.get("source_group", path.stem)))),
                "file": display.as_posix(), "data_kind": str(row.get("data_kind", data_kind))})
        return items

    folders = sorted(p for p in root.glob("*") if p.is_dir() and p.name != "source") if root.is_dir() else []
    if folders:
        for folder in folders:
            for path in _audio(folder):
                items.append({"label": folder.name, "path": path,
                              "group": path.stem, "file": f"{folder.name}/{path.name}",
                              "data_kind": data_kind})
        return items

    if root.is_dir():
        for path in _audio(root):
            stem = path.stem
            pieces = stem.split("_")
            if stem.casefold().startswith("negative_"):
                label = "negative_" + "_".join(pieces[1:-1]) if len(pieces) > 2 and pieces[-1].isdigit() else stem
            elif len(pieces) > 1 and pieces[-1].isdigit():
                label = "_".join(pieces[:-1])
            else:
                label = stem
            items.append({"label": label, "path": path, "group": stem, "file": path.name,
                          "data_kind": data_kind})
    return items


def _dataset_rows(out: Path) -> list[dict]:
    manifest = out/"dataset"/"manifest.jsonl"
    if not manifest.is_file():
        return []
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


def _feature_rows(items: list[dict], model, *, kind: str, framing: str = "event") -> dict[str, dict]:
    from sensorsim.evaluation.stream import file_features
    return file_features(items, model, kind=kind, framing=framing)


def _rendered_feature_items(rows: list[dict], out: Path, model) -> list[dict]:
    items = []
    for row in rows:
        path = out/"dataset"/row["file"]
        if row.get("label") == "background" or not path.is_file():
            continue
        dry = row.get("dry_source") or {}
        group = dry.get("source_group") or row.get("parent_group") or row.get("source_group") or row.get("id", path.stem)
        items.append({"label": str(row["label"]), "path": path, "group": str(group),
                      "file": path.name, "data_kind": "proxy"})
    return items


def _top_bands(report: dict, centres_hz: np.ndarray, limit: int = 5) -> list[dict]:
    import numpy as np
    values = np.asarray(report.get("band_smd", {}).get("absolute_band_smd", []), float)
    order = np.argsort(-values, kind="stable")[:limit]
    return [{"index": int(index), "centre_hz": float(centres_hz[index]),
             "absolute_smd": float(values[index])} for index in order]


def _neutral_probes(fs: int) -> dict[str, np.ndarray]:
    import numpy as np
    n = 2*fs
    silence = np.zeros(n, np.float32)
    rng = np.random.default_rng(20261008)
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n, 1/fs)
    spectrum[1:] /= np.sqrt(np.maximum(freqs[1:], 1.0))
    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
    pink /= max(float(np.sqrt(np.mean(pink*pink))), 1e-12)
    gate = np.zeros(n, np.float32)
    gate[int(.5*fs):int(1.0*fs)] = 1.0
    pink *= gate * .03
    click = np.zeros(n, np.float32)
    t = np.arange(max(1, round(.03*fs)), dtype=np.float32)/fs
    click_start = int(.5*fs)
    click[click_start:click_start+len(t)] = .25*np.sin(2*np.pi*1800*t)*np.exp(-t/.006)
    return {"silence": silence, "pink_noise_burst": pink, "fixed_click": click}


def _neutral_symmetry(model, out: Path, rows: list[dict]) -> dict:
    """Diagnostic proxy paths: rendered positive source, already-in-room negative source, same mic chain."""
    import numpy as np
    from scipy.signal import oaconvolve
    from sensorsim import iot_mic
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import synthesize
    from sensorsim.evaluation.realism import score_neutral_probes

    room_file = out/"fit"/"room_model.json"
    geometry = next((row for row in rows if all(k in row for k in ("source_xyz_m", "receiver_xyz_m"))), None)
    if not room_file.is_file() or geometry is None:
        return {"status": "inconclusive", "reason": "no fitted room and rendered source/mic coordinates"}
    room = RoomModel.model_validate_json(room_file.read_text())
    fs = int(model.meta["features"]["fs"])
    rir = synthesize(room, geometry["source_xyz_m"], geometry["receiver_xyz_m"], fs,
                    seed=20261008).samples

    def device_chain(audio: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        chain = iot_mic.random_chain(rng)
        return chain.apply(np.asarray(audio, np.float32), fs, rng, ref_rms=1e-3).astype(np.float32)

    def positive_path(audio: np.ndarray) -> np.ndarray:
        wet = oaconvolve(audio, rir, mode="full").astype(np.float32)
        return device_chain(wet, 20261008)

    def negative_path(audio: np.ndarray) -> np.ndarray:
        # Negative recordings are already in-room, so they do not pass through the renderer.
        return device_chain(audio, 20261008)

    return score_neutral_probes(model, _neutral_probes(fs), positive_path=positive_path,
                                negative_path=negative_path, data_kind="proxy")


def _evaluate_cnn(files: list, out: Path, *, data_kind: str = "measured",
                  comparison: str = "training-stream") -> dict:
    """Evaluate real/proxy takes through edge.detect and compare them with rendered training data."""
    import numpy as np
    from sensorsim import edge
    from sensorsim.evaluation.recall import evaluate_recall
    from sensorsim.evaluation.realism import provenance_shares
    from sensorsim.evaluation.stream import training_features
    from sensorsim.evaluation.reporting import compare_views
    from sensorsim.evaluation.sensitivity import spike_in_report, apply_power_gate
    from sensorsim.evaluation.c2st import PASS_INTERPRETATION
    from sensorsim.evaluation.reporting import GATE_ACCEPTANCE

    model_path = out/"model.npz"
    if not model_path.is_file():
        raise GateError(f"no CNN model at {model_path}; train or export a model before evaluation")
    model = edge.Model(str(model_path))
    normalized = []
    for item in files:
        if isinstance(item, dict):
            row = dict(item)
        else:
            label, path = item
            row = {"label": label, "path": Path(path), "group": Path(path).stem,
                   "file": f"{label}/{Path(path).name}", "data_kind": data_kind}
        row["path"] = Path(row["path"])
        row.setdefault("group", row["path"].stem)
        row.setdefault("file", row["path"].name)
        normalized.append(row)

    recall = evaluate_recall(model, normalized, data_kind=data_kind)
    dataset_rows = _dataset_rows(out)
    rendered_items = _rendered_feature_items(dataset_rows, out, model)
    if comparison not in {"training-stream", "raw-diagnostic"}:
        raise ValueError("unknown evaluation comparison")
    framing = "sliding" if comparison == "raw-diagnostic" else "event"
    raw_reference = _feature_rows(normalized, model, kind=data_kind, framing=framing)
    raw_synthetic = _feature_rows(rendered_items, model, kind="proxy", framing=framing)
    raw_diagnostic = {"status": "diagnostic-only", "comparison": "raw renders before training mixer",
                      "framing": framing, "classes": {}}
    for label in sorted(raw_reference.keys() & raw_synthetic.keys()):
        e1, e2 = compare_views(raw_reference[label], raw_synthetic[label])
        raw_diagnostic['classes'][label] = {'feature_space_realism': e1, 'discriminator': e2}
    if comparison == "raw-diagnostic":
        reference_features, rendered_features = raw_reference, raw_synthetic
        training_stream = {'status': 'diagnostic-only', 'reason': 'historical raw-window tool controls; not training realism'}
    else:
        reference_features, rendered_features, training_stream = training_features(
            normalized, dataset_rows, out, model, data_kind=data_kind)
    feature_space, discriminator, sensitivity = {}, {}, {}
    labels = sorted({row['label'] for row in normalized if not _negative_label(row['label'])})
    for label in labels:
        reference, rendered = reference_features.get(label), rendered_features.get(label)
        if reference is None or rendered is None:
            reason = training_stream.get('reason', 'no complete matched training/reference windows')
            feature_space[label] = {'status': 'inconclusive', 'reason': 'inconclusive (insufficient power): ' + reason}
            discriminator[label] = {'status': 'inconclusive', 'reason': 'inconclusive (insufficient power): ' + reason}
            sensitivity[label] = {'status': 'underpowered', 'reason': reason}
            continue
        if comparison == 'raw-diagnostic':
            feature_space[label] = raw_diagnostic['classes'][label]['feature_space_realism']
            discriminator[label] = raw_diagnostic['classes'][label]['discriminator']
            sensitivity[label] = {'status': 'diagnostic-only', 'reason': 'historical controls, not a real evaluation'}
        else:
            feature_space[label], discriminator[label] = compare_views(reference, rendered)
            sensitivity[label] = spike_in_report(reference, rendered, model)
            for test, result in (('e1', feature_space[label]), ('e2', discriminator[label])):
                result['ungated_status'] = result['status']
                result['status'] = apply_power_gate(result['status'], sensitivity[label], test=test)
                result['power_status'] = sensitivity[label].get('per_test', {}).get(test, {}).get('status', 'underpowered')
                if result['power_status'] != 'detected' and result['status'] != 'fail':
                    result['power_reason'] = f'inconclusive (insufficient power): {test.upper()} did not detect every spike-in for every seed'
                    # Keep the explicit below-chance explanation alongside the power limitation.
                    if result.get('reason') != 'inconclusive: CV instability':
                        result['reason'] = result['power_reason']

    training_shares = provenance_shares(dataset_rows)
    heldout_shares = provenance_shares([{"label": row["label"], "source_kind": row["data_kind"],
        "provenance": {"route": "held-out evaluation take"}} for row in normalized])
    training_config = training_stream.get('training_config') or (model.meta.get("training") or {}).get("config") or {}
    if training_config.get('device_split') == 'in_distribution':
        chain_shares = {'data_kind': 'configured policy, not measured provenance',
                        'profile': training_config.get('device_profile'),
                        'background_domain': training_config.get('background_domain'),
                        'event_spl_db': training_config.get('event_spl_db')}
    else:
        chain_shares = {'data_kind': 'configured expectation, not observed per-example provenance',
                        'iot_mic.random_chain': training_config.get('p_mic_chain'),
                        'policy': 'legacy/stress'}
    neutral = _neutral_symmetry(model, out, dataset_rows)
    neutral["scope"] = "raw renderer diagnostic only; training mixer is evaluated by the primary comparison"
    payload = {
        "schema": "sensorsim-room-evaluation/2",
        "gate_acceptance": GATE_ACCEPTANCE,
        "comparison": comparison,
        "training_stream": training_stream,
        "raw_render_diagnostic": raw_diagnostic,
        "spike_in_sensitivity": sensitivity,
        "data_kinds": {"held_out": data_kind, "rendered_training": "proxy"},
        "source_note": "Unmarked takes are assumed measured; confirm provenance before using results. Proxy labels describe procedural or room-rendered data.",
        "recall": recall,
        "feature_space_realism": feature_space,
        "sim_vs_real_discriminator": discriminator,
        "pipeline_symmetry": {"training_provenance_shares": training_shares,
                              "held_out_provenance_shares": heldout_shares,
                              "training_device_chain_expected_shares": chain_shares,
                              "neutral_probes": neutral},
        "recordings": [{k: v for k, v in row.items() if k != "path"} for row in normalized],
        "limits": ["Proxy results are not measured room performance.",
                   "A final in-distribution decision needs independent held-out real take groups.",
                   "Recall intervals use manifest performance groups, with all-hit and any-hit outcomes.",
                   "Audio realism uses the frozen training mixer configuration before feature augmentation.",
                   "Raw renders are a diagnostic only; they do not establish training-stream realism.",
                   "Spike-in detection fractions are conditional on these groups and injected effects.",
                   PASS_INTERPRETATION],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n")
    recall_per_label = {label: f"{result['hits']}/{result['takes']}"
                        for label, result in recall["per_class"].items()}
    return {"real_recordings": len(normalized), "recall_per_label": recall_per_label,
            "wrong_class_takes_per_label": {label: row['wrong_class_takes']
                                             for label, row in recall['per_class'].items()},
            "confusion_matrix": recall['confusion_matrix'],
            "false_alarm_events_per_background_hour":
                round(recall["false_events_per_hour"], 2) if recall["false_events_per_hour"] is not None else None,
            "feature_space_status": {label: row["status"] for label, row in feature_space.items()},
            "c2st_status": {label: row["status"] for label, row in discriminator.items()},
            "data_kind": data_kind, "gate_acceptance": GATE_ACCEPTANCE}


# ---------------------------------------------------------------------------------- orchestration
def _inputs(stage: str, cfg: dict, ledger: dict) -> dict | None:
    """Everything a stage reads. None when it cannot run (missing upstream or nothing to evaluate)."""
    up = lambda s: ledger.get(s, {}).get("outputs")
    code = cfg["code"]  # installed package incl. library data: edited code or a dropped-in catalogue reruns
    if stage == "fit":
        from sensorsim.acoustics.room.fit import catalogue_fingerprint   # a plug-in catalogue changes the priors
        if cfg.get("room_yaml"):
            from sensorsim.acoustics.room.pipeline import no_sweep_input_fingerprint
            return {"code": code, "no_sweep_inputs": no_sweep_input_fingerprint(cfg["room_yaml"]),
                    "fs": cfg["dataset"]["fs"], "seed": cfg["dataset"]["seed"]}
        return {"code": code, "session": _tree(_path(cfg, "session")), "fit": cfg["fit"],
                "catalogue": catalogue_fingerprint()}
    if stage == "dataset":
        from sensorsim.sources.project_intake import target_fingerprint
        return None if up("fit") is None else {"code": code, "fit": up("fit"), "targets": target_fingerprint(cfg),
                                                "background": _tree(_path(cfg, "background"), AUDIO), "dataset": cfg["dataset"]}
    if stage == "train":
        r = {k: v for k, v in cfg["recognizer"].items() if k != "min_prob"}
        if r["type"] == "cnn":
            r["torch"] = _version("torch")      # a different PyTorch can train a different model
        return None if up("dataset") is None else {"code": code, "dataset": up("dataset"), "recognizer": r}
    evaluate_root = _path(cfg, "evaluate")
    evaluate_audio = _tree(evaluate_root, AUDIO)
    if up("train") is None or not evaluate_audio:
        return None
    thresholds = {"min_prob": cfg["recognizer"]["min_prob"]} if cfg["recognizer"]["type"] == "baseline" else {}
    return {"code": code, "model": up("train"), "evaluate": evaluate_audio,
            "evaluation_metadata": _tree(evaluate_root),
            "rendered_dataset": up("dataset"), **thresholds}


def _outputs(stage: str, cfg: dict, out: Path) -> str:
    """Content hash of what a stage produced; downstream stages key on it."""
    if stage in ("fit", "dataset"):
        return _digest(_tree(out/stage))
    return sha256_file(_model_file(cfg) if stage == "train" else out/"evaluation.json")


def _outputs_present(stage: str, cfg: dict, out: Path, entry: dict) -> bool:
    """Whether a stage's recorded outputs are still there and unchanged.

    fit/dataset: the ownership marker and the recorded folder digest. evaluate: the recorded file hash.
    train: the model file exists (a deliberately replaced model, such as the documented calibrated copy,
    still counts) and the other training artifacts present are exactly the recorded ones, unchanged. An
    unrecorded artifact (or any artifact besides the model under a ledger that recorded none) makes the
    stage stale, so the ownership check in `_clear_train_outputs` refuses it rather than deleting it.
    """
    if stage in ("fit", "dataset"):
        return (out/stage/MARKER).is_file() and _digest(_tree(out/stage)) == entry.get("outputs")
    if stage == "evaluate":
        path = out/"evaluation.json"
        return path.is_file() and sha256_file(path) == entry.get("outputs")
    model = _model_file(cfg)
    if not model.is_file():
        return False
    present = {name: digest for name, digest in _train_artifacts(out).items() if name != model.name}
    recorded = {name: digest for name, digest in entry.get("artifacts", {}).items() if name != model.name}
    return present == recorded


def _gates(stage: str, summary: dict) -> list[str]:
    """Human-readable warnings; raises GateError when later stages would be meaningless."""
    notes = []
    if stage in ("fit", "dataset") and summary.get("calibration_status") in ("assumed", "weak-decay"):
        from sensorsim.acoustics.room.render_prior import ASSUMED_ROOM_WARNING
        notes.append(summary.get("room_warning") or ASSUMED_ROOM_WARNING)
    if stage == "fit":
        lost = summary["labels_without_a_usable_take"]
        if lost:
            takes = ", ".join(f for files in lost.values() for f in files)
            raise GateError(
                f"no usable recording of {sorted(lost)}: no take of it could be used ({takes}). Add "
                "`event_window_s: [start, end]` to those takes, or `event_merge_gap_s: 0.5` if the sound has "
                "several parts (knocks, footsteps); re-record them; or remove them from the manifest. "
                "Details: out/fit/dry/event_selection.json")
        if not summary["event_detection"]:
            notes.append("manifest uses acquisition: legacy, so each real take is used WHOLE with no event "
                         "detection (noise or stray sounds in it become training data); use the kit template's "
                         "acquisition: sequential_phone")
        if summary["state"] != "complete":
            raise GateError(f"the fit did not complete (state={summary['state']}); see out/fit/report.md")
        c = summary["check_positions"]
        if c["total"]:
            notes.append(f"held-out C1 checks: {c['pass']}/{c['total']} pass (see out/fit/report.md)")
        else:
            notes.append("no held-out C1 check sweep: the room fit is unverified")
        t = summary["targets_in_room"]
        for r in t["refused"]:
            notes.append(f"target take refused: {r['file']} -- add `event_window_s: [start, end]` to that take "
                         "(or `event_merge_gap_s: 0.5` for multi-part sounds such as knocks), or re-record it")
        for f in t["crops_overlapping_another_event"]:
            notes.append(f"target take {f}: the selected sound overlaps another detected sound; listen to "
                         "out/fit/dry and re-record it quietly, or tighten its event_window_s")
    if stage == "dataset":
        for lab in summary["labels_with_one_recording"]:
            notes.append(f"'{lab}' has a single recording: it cannot be validated; record at least two")
        for lab in summary["labels_without_validation"]:
            notes.append(f"'{lab}' is absent from validation")
        if not summary["background_files"]["val"]:
            notes.append("only one background recording: validation has no background; record another")
    if stage == "train":
        notes += summary["warnings"]
    return notes


@contextmanager
def _thread_cap(cfg: dict):
    """Cap native BLAS/OpenMP pools at recognizer.threads (default 4) while evaluating; never raise one.

    Evaluation makes thousands of tiny linear-algebra calls (the C2ST logistic fits). With a BLAS pool
    sized to every core, each call waits on all of them, which is many times slower on a busy host.
    """
    from threadpoolctl import threadpool_info, threadpool_limits
    n = max(1, int(cfg.get("recognizer", {}).get("threads", 4)))
    counts: dict[str, list[int]] = {}
    for pool in threadpool_info():
        counts.setdefault(pool["user_api"], []).append(int(pool["num_threads"]))
    # threadpoolctl limits a whole user_api at once: use its smallest pool so no pool is ever raised.
    limits = {api: min(n, min(sizes)) for api, sizes in sorted(counts.items()) if max(sizes) > n}
    with threadpool_limits(limits=limits) if limits else nullcontext():
        yield n


def _progress(message: str) -> None:
    print(message, file=sys.stderr)


def run(project: str | Path, *, force: tuple[str, ...] = (), echo=_progress) -> dict:
    cfg = load_project(project)
    cfg["code"] = implementation_fingerprint()
    out = _path(cfg, "out")
    ledger_file = out/"stages.json"
    if out.is_dir() and any(out.iterdir()) and not ledger_file.is_file():
        raise GateError(f"{out} already holds files this pipeline did not write; choose another `out`")
    out.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(ledger_file.read_text()) if ledger_file.is_file() else {}
    actions = {"fit": _fit, "dataset": _dataset, "train": _train, "evaluate": _evaluate}
    report = {}
    for stage in STAGES:
        inputs = _inputs(stage, cfg, ledger)
        if inputs is None:
            report[stage] = {"status": "skipped (nothing to do)"}
            continue
        key = _digest(inputs)
        entry = ledger.get(stage, {})
        if (stage not in force and entry.get("inputs") == key and entry.get("outputs")
                and _outputs_present(stage, cfg, out, entry)):
            report[stage] = {"status": "up to date", **entry["summary"]}
            echo(f"[{stage}] up to date")
            continue
        echo(f"[{stage}] running")
        if stage == "train":
            _clear_train_outputs(cfg, out, entry)
        with _thread_cap(cfg) if stage == "evaluate" else nullcontext():
            summary = actions[stage](cfg, out)
        summary["notes"] = _gates(stage, summary)
        ledger[stage] = {"inputs": key, "outputs": _outputs(stage, cfg, out), "summary": summary}
        if stage == "train":
            ledger[stage]["artifacts"] = _train_artifacts(out)
        ledger_file.write_text(json.dumps(ledger, indent=1) + "\n")
        report[stage] = {"status": "ran", **summary}
        for note in summary["notes"]:
            echo(f"  - {note}")
    return report


def status(project: str | Path) -> dict:
    cfg = load_project(project)
    cfg["code"] = implementation_fingerprint()
    out = _path(cfg, "out")
    ledger_file = out/"stages.json"
    ledger = json.loads(ledger_file.read_text()) if ledger_file.is_file() else {}
    result, stale = {}, None
    for stage in STAGES:
        inputs = _inputs(stage, cfg, ledger)
        entry = ledger.get(stage, {})
        if inputs is None:
            result[stage] = "not applicable"
        elif stale:
            result[stage] = f"needs to run (after {stale})"
        elif entry.get("inputs") == _digest(inputs) and _outputs_present(stage, cfg, out, entry):
            result[stage] = "up to date"
        elif entry.get("inputs") == _digest(inputs):
            result[stage], stale = "needs to run (outputs missing or changed)", stage
        else:
            result[stage], stale = "needs to run", stage
    return result


def detect(project: str | Path, files, *, min_prob: float | None = None) -> list[dict]:
    cfg = load_project(project)
    if cfg["recognizer"]["type"] == "cnn":
        from sensorsim import edge
        if min_prob is not None:
            raise ValueError("min_prob applies to type: baseline; the CNN's thresholds are in the model "
                             "(set them with `edge.py calibrate`)")
        model = edge.Model(str(_model_file(cfg)))
        fs = model.meta["features"]["fs"]
        return [{"file": str(f), **e} for f in files for e in edge.detect(model, edge.read_wav(str(f), fs))]
    from sensorsim import recognizer
    model = json.loads((_path(cfg, "out")/"model.json").read_text())
    p = cfg["recognizer"]["min_prob"] if min_prob is None else min_prob
    return [row for f in files for row in recognizer.predict(model, f, min_prob=p)]


def visualize(project: str | Path, out_file: str | Path, *, take: int | None = None, source=None, receiver=None,
              names: tuple[str | None, str | None] = (None, None), max_order: int = 2) -> dict:
    """Write the 3D sound-path page for a measured take (with its measured response) or any two points."""
    import numpy as np
    from sensorsim.acoustics.room import visualize as view
    from sensorsim.acoustics.room.model import RoomModel
    cfg = load_project(project)
    fit = _path(cfg, "out")/"fit"
    if not (fit/"room_model.json").is_file():
        raise GateError(f"no fitted room in {fit}: run the project first")
    model = RoomModel.model_validate_json((fit/"room_model.json").read_text())
    row = None
    if source is None and receiver is None:
        rows = json.loads((fit/"measurements.json").read_text()) if (fit/"measurements.json").is_file() else []
        rows = [r for r in rows if (r["take"] == take if take is not None else not r.get("is_reference"))]
        if not rows:
            raise GateError("no measured take to show" + (f" with take {take}" if take is not None else "")
                            + "; give --source and --receiver positions instead")
        row = rows[0]
        source, receiver = row["source_xyz_m"], row["receiver_xyz_m"]
        names = (names[0] or f"speaker (take {row['take']})", names[1] or f"mic {row['mic_id']}")
    elif source is None or receiver is None:
        raise ValueError("give both a source and a receiver position")
    dims = np.asarray(model.shoebox_dims_m)
    for label, xyz in (("source", source), ("receiver", receiver)):
        if not np.all((np.asarray(xyz) > 0) & (np.asarray(xyz) < dims)):
            raise ValueError(f"{label} {list(xyz)} is outside the fitted room (0..{dims.tolist()} m)")
    names = (names[0] or "source", names[1] or "microphone")
    data = view.scene(model, source, receiver, max_order=max_order, names=names)
    if row is not None:
        fs = json.loads((fit/"report.json").read_text())["fs"]
        direct = float(np.linalg.norm(np.subtract(source, receiver))/model.sound_speed_m_s)
        data["measured"] = view.measured_envelope(np.load(fit/row["file"]), fs, direct, view_s=data["view_s"],
                                                  label=f"take {row['take']}, {row['mic_id']}")
    title = f"Sound paths to {names[1]}" if names[1] != "microphone" else "Sound paths in the fitted room"
    view.write(out_file, data, title=title)
    return {"file": str(out_file), **view.summary(data)}


def _licence_files() -> dict[str, Path]:
    """LICENSE and NOTICE of this distribution: installed metadata, else the source checkout."""
    from importlib.metadata import PackageNotFoundError, distribution
    found = {}
    try:
        dist = distribution("sensorsim-room")
        for f in dist.files or []:
            if f.name in ("LICENSE", "NOTICE") and f.parent.name == "licenses":
                found[f.name] = Path(dist.locate_file(f))
    except PackageNotFoundError:
        pass
    checkout = Path(__file__).resolve().parents[2]
    for name in ("LICENSE", "NOTICE"):
        if not found.get(name, Path()).is_file() and (checkout/name).is_file():
            found[name] = checkout/name
    missing = {"LICENSE", "NOTICE"} - {k for k, v in found.items() if v.is_file()}
    if missing:
        raise GateError(f"cannot find {sorted(missing)} of sensorsim-room: edge.py is Apache-2.0 and must ship with them")
    return found


def export(project: str | Path, folder: str | Path) -> dict:
    """Write what the device needs into a new folder: edge.py, model.npz, LICENSE, NOTICE, DEPLOY.md, SHA256SUMS."""
    from sensorsim.licensing import scan_for_forbidden_imports, validate_device_folder, validate_project_lineage
    scan_for_forbidden_imports(Path(__file__).resolve().parent)
    from sensorsim import edge
    cfg = load_project(project)
    if cfg["recognizer"]["type"] != "cnn":
        raise GateError("export packages the CNN for the device: set `recognizer: {type: cnn}` and run the project")
    model_file = _model_file(cfg)
    if not model_file.is_file():
        raise GateError(f"no trained model at {model_file}: run the project first")
    model = edge.Model(str(model_file))
    from sensorsim.negatives.integration import require_production_metadata
    require_production_metadata(model.meta.get("training", {}))
    validate_project_lineage(_path(cfg, "out"), model.meta)
    folder = Path(folder)
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise GateError(f"{folder} exists and is not an empty folder; give a new one")
    licences = _licence_files()
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(edge.__file__, folder/"edge.py")
    shutil.copyfile(model_file, folder/"model.npz")
    for name, path in licences.items():
        shutil.copyfile(path, folder/name)
    model = edge.Model(str(folder/"model.npz"))
    (folder/"DEPLOY.md").write_text(_deploy_notes(model))
    sums = {p.name: sha256_file(p) for p in sorted(folder.iterdir())}
    (folder/"SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sums.items()))
    files = validate_device_folder(folder)
    return {"folder": str(folder), "files": files, "classes": model.classes,
            "parameters": int(sum(w.size for w in model.weights.values())), "macs_per_window": model.macs()}


def _deploy_notes(model) -> str:
    d, f = model.meta["detector"], model.meta["features"]
    per_class = "\n".join(
        f"| {c} | {d['on'][c]} | {d['smooth_hops'][c] if isinstance(d['smooth_hops'], dict) else d['smooth_hops']} "
        f"| {d['min_hops'][c] if isinstance(d['min_hops'], dict) else d['min_hops']} |" for c in model.classes)
    params = int(sum(w.size for w in model.weights.values()))
    training_cfg = model.meta.get("training", {}).get("config", {})
    negatives = model.meta.get("training", {}).get("negatives", {})
    family_rows = "\n".join(f"| {r['family']} | {r['count']} | {r['weight']:.6g} | {r['probability']:.6g} |"
                            for r in negatives.get("families", [])) or "| none recorded | 0 | 0 | 0 |"
    return DEPLOY_TEMPLATE.format(negative_families=family_rows,
                                  negative_policy=negatives.get("policy", "disabled or legacy"),
                                  negative_probability=training_cfg.get("p_confuser_given_background", 0),
                                  procedural_cap=training_cfg.get("max_procedural_share", "unrecorded"),
                                  selection_mode=training_cfg.get("selection_mode", "legacy synthetic AP"),
                                  device_profile=training_cfg.get("device_profile", "unrecorded (legacy model)"),
                                  device_split=training_cfg.get("device_split", "unrecorded"),
                                  background_domain=training_cfg.get("background_domain", "unrecorded"),
                                  render_reference=training_cfg.get("render_reference_spl_db", "unrecorded"),
                                  event_spl=training_cfg.get("event_spl_db", "unrecorded"), classes=", ".join(model.classes), params=f"{params:,}", macs=f"{model.macs()/1e6:.1f} million",
                                  window=f["window_s"], hop=d["hop_s"], fs=f["fs"], per_class=per_class,
                                  off=d["off_ratio"])


DEPLOY_TEMPLATE = """# Deploying the sound recognizer

This folder is everything the device needs. It runs on Linux with Python 3.9 or newer and numpy;
the reference device is a Raspberry Pi (3, 4, 5 or Zero 2 W) with Raspberry Pi OS. No PyTorch, no
compiler and no network connection are needed.

| file | contents |
| --- | --- |
| `edge.py` | the recognizer: numpy and the Python standard library only |
| `model.npz` | the trained model: {classes}; {params} parameters, {macs} multiply-adds per {window} s window, one window every {hop} s |
| `LICENSE`, `NOTICE` | the Apache-2.0 terms of `edge.py`: keep them with it |
| `SHA256SUMS` | `sha256sum -c SHA256SUMS` checks the copy |

| training setting | value |
| --- | --- |
| versioned device profile | {device_profile} |
| device split | {device_split} |
| background domain | {background_domain} |
| render peak-sine reference (dB SPL) | {render_reference} |
| assumed event-active RMS SPL range (dB SPL at microphone) | {event_spl} |
| confuser probability given background | {negative_probability} |
| procedural cap per background-route opportunity | {procedural_cap} |
| checkpoint and tracker selection | {selection_mode} |

Negative admission: {negative_policy}. Sampling weights are not deployment prevalence.
Unused procedural slots remain unchanged recorded background.

| confuser family | source count | sampling weight | probability per attempted confuser draw |
| --- | --- | --- | --- |
{negative_families}

Negatives default on with an automatic ungated procedural pool. Optional `negative_manifest`,
`negative_manifest_sha256`, `p_confuser_given_background`, and `max_procedural_share` configure it
under `recognizer`. Tonal and impact inputs require verified review evidence; tones
also require a held-out doorbell check. Joint selection accepts `selection_panel`
and `selection_panel_sha256`, or builds a panel from disjoint held-out groups. Too few
groups or infeasible selection retain the terminal checkpoint at the configured fixed
detector (library defaults: on 0.5, off ratio 0.6, smooth 3 hops, confirm 2 hops, hop 0.25 s),
with an explicit note and no calibration claim. `proxy_screen_only` receipts and models
are refused in production training and export. Explicit
`selection_mode: diagnostic_final` is unqualified development output.
The selection panel and its background/confuser parents must stay out of mining.

Set these under `recognizer` in `project.yaml`: `device_profile`, `device_split`,
`background_domain`, `render_reference_spl_db`, and `event_spl_db`. The default
`in_distribution` split uses the Pi profile; `stress` explicitly enables random
processing. `device_recorded` backgrounds retain their digital levels and existing
microphone floor. Choose `spl_referenced` for synthetic or other acoustic beds that
have a known SPL reference and still need the device chain. Uncalibrated third-party
recordings have no known physical level. The 60–80 dB SPL event prior is an assumption
pending measured deployment levels; `snr_db` controls only the stress split.

The deprecated `pi5_mems_unfiltered` alias now means the absolute-SPL INMP441 response
with its 60 Hz corner, DC block and decimator. It warns on use and ignores `ref_rms`;
use the explicit versioned profile and its SPL reference instead.

## 1. Install

    sudo apt install python3-numpy alsa-utils
    sha256sum -c SHA256SUMS
    python3 edge.py info model.npz
    python3 edge.py bench model.npz        # milliseconds per window and the share of one CPU core

`arecord -l` lists the capture devices; `-D plughw:CARD,DEVICE` below picks one and converts it to
{fs} Hz mono.

### The microphone

- **Digital MEMS microphone on the Pi's I2S pins** (INMP441- or SPH0645-class breakouts): it becomes
  an ALSA capture device once its device-tree overlay is enabled. Use the overlay and wiring that the
  breakout's maker documents for your Pi model and OS release, then find the card with `arecord -l`.
  On Pi 5, use **I2S0 as the BCLK/LRCLK clock producer** for a standard I2S MEMS microphone. I2S1 is
  the clock consumer; it is not the microphone-clock path. The Pi does not provide I2S MCLK.
- **Check the PCM conversion before collecting takes.** The microphones send 24-bit samples in a
  32-bit I2S slot; the capture path must keep the upper 16 sample bits with the correct sign. Run
  `python tools/capture_check.py --self-test` from a sensorsim-room checkout to check a generated
  1 kHz tone and signed extreme values through the tool's own S32-to-S16 arithmetic.
  This does **not** check the Pi/ALSA conversion. A single S16 WAV cannot verify that
  path either: compare native and application captures of the same digital test pattern
  on the Pi before calling it verified. Then record a 1 kHz tone
  at a known acoustic level and check it against the module sensitivity:

      arecord -D plughw:1,0 -r {fs} -f S16_LE -c 1 -d 5 calibration.wav
      python tools/capture_check.py take.wav --tone-wav calibration.wav --tone-spl-db 94

  The tone check uses the part's sensitivity tolerance (INMP441: ±3 dB) and flags possible
  bit-shift or gain errors. A wrong L/R I2S slot or stereo downmix can also lose 6 dB;
  verify slot selection without averaging an empty slot. Generated extremes check software
  signed rails only. Physical wiring, clocking and conversion remain unverified by this tool.
- **Prefer native {fs} Hz mono S16_LE capture.** Check the card's supported rates with
  `arecord -D hw:CARD,DEVICE --dump-hw-params`. If {fs} Hz is unavailable, capture at a native rate
  that is an integer multiple or fraction of {fs} Hz and let `edge.py` use its integer-ratio resampler.
  `plughw` may resample before the training path and can change features; verify that choice on the
  device. The capture checker strictly requires native 16 kHz mono S16 input and rejects
  stereo and other rates; it never certifies a resampled or downmixed input. The model removes
  each window's mean, but that does not make a wrong PCM alignment harmless.
- **USB microphone or sound card**: appears in `arecord -l` without configuration.
- **"Sound sensor" boards read over I2C or a GPIO pin** report a loudness value or a threshold bit,
  not audio. They cannot feed the recognizer, which needs a PCM audio stream.

Raw MEMS capsules have no automatic gain. Check that the loudest target sound stays below full
scale: `arecord -D plughw:1,0 -r {fs} -f S16_LE -c 1 -d 10 -vv test.wav` prints a peak meter.
Keep quiet audio before and after every target take. Measure those intervals separately with
`python tools/capture_check.py take.wav --pre-quiet-s 3 --post-quiet-s 3`; they estimate the deployed
microphone floor and its drift during the session. They are not a false-alarm/hour measurement.

## 2. Calibrate on this microphone

Record at least 3 hours of target-free ordinary room sound through the deployed microphone in its
final position. The combined 1/h budget requires 3 hours even with zero alerts. This is development
threshold selection; evaluate recall and false alarms on separate untouched recordings.

    arecord -D plughw:1,0 -r {fs} -f S16_LE -c 1 -d 10800 background.wav
    python3 edge.py calibrate model.npz background.wav --budget-mode combined --fa-per-hour 1 --recording-id calibration-session --out model_calibrated.npz

With only 30 minutes and zero alerts, the one-sided 95% upper bound is about 6/h; with 1 hour it is
about 3/h. Insufficient exposure reports `status: insufficient_evidence`, `saved: false`, exits 2
and writes no calibrated model. Keep the original model and collect more disjoint development audio;
do not run the calibrated-model commands below until calibration succeeds. Three hours is necessary,
not a guarantee of feasibility if alerts remain. Convert to model-rate integer PCM before calibration,
keep original recording IDs, and declare nonoverlapping sample ranges for calibration and evaluation.

Calibration only raises thresholds, so it trades missed events for fewer false alarms. Recalibrate
when the microphone, its position or the room's usual sounds change. More background recorded
through this device, added to the project's `background/` folder before training, helps more than
calibration does.

## 3. Listen

    arecord -D plughw:1,0 -r {fs} -f S16_LE -c 1 -t raw | python3 edge.py listen model_calibrated.npz

Each line on standard output is one JSON object: `{{"event": "start", ...}}` when a sound begins, and
`{{"event": "end", "label": ..., "start_s": ..., "end_s": ..., "peak": ..., "confirmed": true}}` when it ends.
Pipe them into your own script to act on them.

## 4. Start at boot (systemd)

Copy the folder to `/opt/recognizer` and save this as `/etc/systemd/system/recognizer.service`:

    [Unit]
    Description=sound recognizer
    After=sound.target

    [Service]
    ExecStart=/bin/sh -c 'arecord -q -D plughw:1,0 -r {fs} -f S16_LE -c 1 -t raw | python3 /opt/recognizer/edge.py listen /opt/recognizer/model_calibrated.npz'
    DynamicUser=yes
    SupplementaryGroups=audio
    Restart=always

    [Install]
    WantedBy=multi-user.target

then `sudo systemctl enable --now recognizer` and `journalctl -u recognizer -f` to watch it.

## Detector settings in this model

| class | on threshold | smoothing (hops) | minimum length (hops) |
| --- | --- | --- | --- |
{per_class}

A class starts when its score, averaged over the smoothing hops, reaches its threshold, and ends when
it falls below {off} times that threshold. Events shorter than the minimum length are not reported.
`edge.py calibrate` writes new thresholds; the other settings come from training (short sounds that
the model sees in fewer windows get less smoothing).

## Limits

Validation during training uses simulated microphones and renders of the room, so it is optimistic.
Measure on the device: play or perform each target sound a few times and watch `listen`. Short
broadband sounds (knocks, taps) can resemble ordinary indoor transients. Record more takes
and background through the deployed device if they are missed or fire falsely. The Pi stream
uses a 16 kHz microphone profile with no automatic gain, noise suppression, telephone codecs
or packet loss. Its floor spectrum and event SPL prior remain model-based assumptions until
checked against real takes. Software capture checks do not establish physical conversion,
real-take recall or a measured false-alarm rate.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run every stage that is out of date")
    r.add_argument("project", type=Path)
    r.add_argument("--force", choices=STAGES, action="append", default=[], help="rerun this stage regardless")
    s = sub.add_parser("status", help="show which stages are up to date")
    s.add_argument("project", type=Path)
    d = sub.add_parser("detect", help="recognize target sounds in recordings with the project's model")
    d.add_argument("project", type=Path)
    d.add_argument("files", type=Path, nargs="+")
    d.add_argument("--min-prob", type=float, help="type: baseline only")
    e = sub.add_parser("export", help="write the CNN and its runtime into a new folder for the device")
    e.add_argument("project", type=Path)
    e.add_argument("folder", type=Path)
    q = sub.add_parser("evaluate", help="score labelled takes and compare them with rendered training data")
    q.add_argument("project", type=Path)
    v = sub.add_parser("visualize", help="write a 3D page of how sound travels from a source to a microphone")
    v.add_argument("project", type=Path)
    v.add_argument("out", type=Path, help="the .html file to write")
    v.add_argument("--take", type=int, help="a measured take, with its measured response (default: the first)")
    v.add_argument("--source", type=float, nargs=3, metavar=("X", "Y", "Z"), help="metres, in the fitted room")
    v.add_argument("--receiver", type=float, nargs=3, metavar=("X", "Y", "Z"), help="metres, in the fitted room")
    v.add_argument("--source-name")
    v.add_argument("--receiver-name")
    v.add_argument("--order", type=int, default=2, choices=range(4), help="highest reflection order drawn")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            report = run(args.project, force=tuple(args.force))
            print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "notes"} for k, v in report.items()}, indent=1, default=str))
        elif args.command == "status":
            print(json.dumps(status(args.project), indent=1))
        elif args.command == "evaluate":
            cfg = load_project(args.project)
            with _thread_cap(cfg):
                report = _evaluate(cfg, _path(cfg, "out"))
            print(json.dumps(report, indent=1, default=str))
        elif args.command == "export":
            print(json.dumps(export(args.project, args.folder), indent=1))
        elif args.command == "visualize":
            print(json.dumps(visualize(args.project, args.out, take=args.take, source=args.source, receiver=args.receiver,
                                       names=(args.source_name, args.receiver_name), max_order=args.order), indent=1))
        else:
            for row in detect(args.project, args.files, min_prob=args.min_prob):
                print(json.dumps(row))
    except (GateError, ValueError, OSError) as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
