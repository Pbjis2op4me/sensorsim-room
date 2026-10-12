"""Streaming measured campaign (parent-project bench_room_fit protocol, not distributed); no optional Python packages.

Example: .venv/bin/python tests/validation/bench_room_fit/run.py --case meshrir
 --data /path/to/data/evidence/room_acoustics --out /path/to/job/evidence
Run under nice 10, two numeric threads and the shared load guard.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

import numpy as np

from sensorsim.acoustics.room.fit import (
    FitConfig, FitResult, compare, early_diagnostics, estimate_geometry, fast_waveform_difference,
    fingerprint, fit_room, observe, predict,
)
from sensorsim.acoustics.room.model import RoomModel, uniform_bands
from sensorsim.acoustics.room.rir import RIR
from sensorsim.acoustics.room.metrics import room_metrics


def paired_drr_rows(measured, predicted, *, id, split, bands, onset, predicted_onset=None):
    """Both definitions on the same broadband RIRs; retain unavailable values."""
    rows = []
    for legacy in (True, False):
        for band in bands:
            target = room_metrics(measured.samples, measured.fs, band=band, onset_sample=onset,
                noise_power=0 if measured.metadata.get('synthetic') else None, legacy_drr=legacy)
            estimate = None if predicted is None else room_metrics(predicted.samples, predicted.fs,
                band=band, onset_sample=predicted_onset, noise_power=0, legacy_drr=legacy)
            a, b = target.drr_db, None if estimate is None else estimate.drr_db
            finite = lambda value: value is not None and np.isfinite(value)
            error = float(b-a) if finite(a) and finite(b) else None
            rows.append(dict(id=id, split=split, band_hz=band, metric='drr_db',
                drr_definition=target.drr_definition, measured=float(a) if finite(a) else None,
                predicted=float(b) if finite(b) else None, signed_error=error, tolerance=3.,
                status='UNAVAILABLE' if not finite(a) else 'FAIL' if error is None or abs(error)>3 else 'PASS',
                reason=target.decay.reason if a is None else 'nonfinite measured DRR' if not finite(a)
                else 'missing or nonfinite prediction' if not finite(b) else None))
    return rows


def rescore_drr_holdouts(name, pending, config, folder, inputs, saved):
    """Re-render saved fits; change only the reported DRR definition, never refit."""
    started = time.monotonic()
    saved = Path(saved)/name
    model_path, report_path = saved/'room_model.json', saved/'fit_report.json'
    saved_config = FitConfig.model_validate(json.loads((saved/'config.json').read_text()))
    if saved_config != config:
        raise ValueError('DRR rescore config differs from the archived fit')
    archived = json.loads((saved/'summary.json').read_text())
    if archived['inputs'] != inputs:
        raise ValueError('DRR rescore inputs differ from the archived fit')
    fit = FitResult(RoomModel.from_json(model_path.read_text()), config, json.loads(report_path.read_text()))
    expected = {(p['id'],p['split']) for p in json.loads((saved/'positions.json').read_text()) if p['split']!='train'}
    seen, rows = set(), []
    if (folder/'drr_summary.json').exists():
        raise FileExistsError(folder/'drr_summary.json')
    with (folder/'drr_metrics.jsonl').open('w') as stream:
        for rir, id, split in pending():
            if (id,split) in seen:
                raise ValueError('duplicate DRR rescore position')
            seen.add((id,split))
            obs = observe(rir, id=id, split=split, config=config)
            if list(obs.position_key) in fit.model.provenance['training_positions']:
                raise ValueError('DRR holdout overlaps training positions')
            try:
                prediction = predict(fit, obs.source_xyz_m, obs.receiver_xyz_m, obs.fs, id=id)
            except ValueError:
                prediction = None
            pair = paired_drr_rows(rir, prediction, id=id, split=split, bands=config.bands,
                onset=obs.metadata['analysis_direct_sample'],
                predicted_onset=None if prediction is None else prediction.metadata['direct_sample'])
            rows.extend(pair)
            for row in pair:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            if len(seen)%100 == 0:
                print(name, 'DRR holdouts', len(seen), flush=True)
    if seen != expected:
        raise ValueError(f'DRR rescore position coverage changed: missing={expected-seen}, extra={seen-expected}')
    summary = dict(case=name, role='metric-only paired rescore of current renders from fixed archived fits; no refit',
        fit_drr_definition='drr/1', inputs=inputs, environment=fingerprint(), positions=len(seen),
        saved={str(p):sha(p) for p in (model_path, report_path, saved/'config.json', saved/'positions.json')},
        elapsed_s=time.monotonic()-started, definitions={definition:summarize([r for r in rows if
            r['drr_definition']==definition]) for definition in ('drr/1','drr/2')})
    write(folder/'drr_summary.json',summary)
    print(name, {k:v['counts'] for k,v in summary['definitions'].items()}, flush=True)
    return summary


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=4)
def _pins(data):
    return {row["path"]: row for row in json.loads((data/"MANIFEST.json").read_text())["files"]}


def verified_sha(data, path):
    pin = _pins(data)[str(path.relative_to(data))]
    actual = sha(path)
    if actual != pin["sha256"] or path.stat().st_size != pin["size_bytes"]:
        raise ValueError(f"acquired input changed: {path}")
    return actual


def sofa_metadata(path, fields):
    cmd = ["h5dump", "-y", "-A", "0", "-m", "%.17g"]
    for field in fields:
        cmd += ["-d", field]
    text = subprocess.check_output([*cmd, str(path)], text=True)
    out = {}
    for field in fields:
        block = text.split('DATASET "'+field+'" {', 1)[1]
        data = re.search(r"\bDATA \{(.*?)\}", block, re.S).group(1)
        out[field] = np.array([float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", data)])
    return out


def sofa_ir(path, scratch, shape):
    binary = scratch/"one-sofa.bin"
    subprocess.run(["h5dump", "-b", "LE", "-d", "Data.IR", "-o", str(binary), str(path)], check=True, capture_output=True)
    data = np.fromfile(binary, dtype="<f8").reshape(shape)
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite SOFA samples: {path}")
    return data


def summarize(rows):
    groups = defaultdict(Counter)
    errors = defaultdict(list)
    joint = defaultdict(list)
    for row in rows:
        key = f"{row['split']}/{row['band_hz']}/{row['metric']}"
        groups[key][row["status"]] += 1
        error = row["relative_error"] if row["metric"] in ("edt", "t20", "t30") else row["signed_error"]
        if error is not None:
            errors[key].append(error)
        if row["metric"] in ("edt", "t20", "t30", "c50_db", "c80_db", "d50"):
            joint[(row["split"], row["id"], row["band_hz"])].append(row["status"])
    result = {}
    for key, counts in sorted(groups.items()):
        e = np.array(errors[key])
        result[key] = {s: counts[s] for s in ("PASS", "FAIL", "UNAVAILABLE")}
        if len(e):
            result[key].update(median_signed=float(np.median(e)), median_absolute=float(np.median(abs(e))),
                               p95_absolute=float(np.percentile(abs(e), 95)), rms=float(np.sqrt(np.mean(e**2))))
    joint_counts = defaultdict(Counter)
    for (split, _, _), statuses in joint.items():
        status = "FAIL" if "FAIL" in statuses else "UNAVAILABLE" if "UNAVAILABLE" in statuses else "PASS"
        joint_counts[split][status] += 1
    return {"gates": result, "all_required_jnd": {s: dict(v) for s, v in joint_counts.items()},
            "counts": dict(Counter(row["status"] for row in rows if row["split"] != "train"))}


def recover_reports(name, train, pending, config, folder, inputs):
    """Rebuild diagnostics from a completed metric stream, without refitting.

    Preserve original artifacts, including a failed campaign's completion log.
    Refuse partial streams or changed split/config/input metadata.
    """
    started = time.monotonic()
    target = folder/"summary.recovered.json"
    if target.exists():
        raise FileExistsError(target)
    if json.loads((folder/"config.json").read_text()) != config.model_dump(mode="json"):
        raise ValueError("recovery config differs from saved fit")
    old_summary = folder/"summary.json"
    if old_summary.exists() and json.loads(old_summary.read_text())["inputs"] != inputs:
        raise ValueError("recovery inputs differ from saved campaign")
    fit = FitResult(RoomModel.from_json((folder/"room_model.json").read_text()), config,
                    json.loads((folder/"fit_report.json").read_text()))
    with (folder/"metrics.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    expected = Counter((r["id"], r["split"], r["band_hz"], r["metric"]) for r in rows)
    seen, splits, early, approx = Counter(), [], [], []
    def process(obs):
        for band in config.bands:
            for metric in (*obs.bands[band].values, "spectral_db"):
                seen[(obs.id, obs.split, band, metric)] += 1
        splits.append({"id": obs.id, "split": obs.split, "position": list(obs.position_key),
                       "metadata": obs.metadata})
        early.append({"id": obs.id, "split": obs.split, **early_diagnostics(fit.model, obs)})
        if obs.split == "train":
            rendered = predict(fit, obs.source_xyz_m, obs.receiver_xyz_m, obs.fs, id=obs.id)
            prediction = observe(rendered, id=obs.id, config=config)
            approx.extend(fast_waveform_difference(fit, obs, prediction))
    for obs in train:
        process(obs)
    for count, (rir, id, split) in enumerate(pending()):
        process(observe(rir, id=id, split=split, config=config))
        if (count+1) % 500 == 0:
            print(name, "recovered", count+1, flush=True)
    if expected != seen or any(count != 1 for count in seen.values()):
        raise ValueError("saved metrics are partial, duplicate, or use a different split")
    summary = {"case": name, **summarize(rows), "recovery_elapsed_s": time.monotonic()-started,
               "environment": fingerprint(), "original_environment": json.loads((folder.parent/"run_environment.json").read_text()),
               "inputs": inputs, "position_counts": dict(Counter(s["split"] for s in splits)),
               "bootstrap": fit.model.uncertainty, "limits": fit.report["limitations"],
               "recovery": "Saved fit and metric rows unchanged; re-read measurements and rebuild diagnostics only",
               "saved_sha256": {p: sha(folder/p) for p in ("room_model.json", "fit_report.json", "config.json", "metrics.jsonl")}}
    for filename, value in (("positions", splits), ("early", early), ("fast_vs_waveform", approx), ("summary", summary)):
        write(folder/(filename+".recovered.json"), value)
    print(name, "recovery complete", summary["counts"], flush=True)
    return summary


def evaluate(name, train, pending, model, config, folder, inputs, *, recover=False, drr_models=None):
    """pending yields (RIR,id,split) only after model fit and serialization."""
    folder.mkdir(parents=True, exist_ok=True)
    if drr_models is not None:
        return rescore_drr_holdouts(name, pending, config, folder, inputs, drr_models)
    if recover:
        return recover_reports(name, train, pending, config, folder, inputs)
    if (folder/"summary.json").exists():
        raise FileExistsError(f"completed case exists: {folder}")
    started = time.monotonic()
    print(name, "fit", len(train), flush=True)
    fit = fit_room(train, model, config=config)
    (folder/"room_model.json").write_text(fit.model.to_json()+"\n")
    write(folder/"fit_report.json", fit.report)
    write(folder/"config.json", config.model_dump(mode="json"))
    print(name, "fit complete", round(time.monotonic()-started, 2), flush=True)
    all_rows, early_rows, approx, exclusions, splits = [], [], [], [], []
    with (folder/"metrics.jsonl").open("w") as stream:
        def process(obs):
            try:
                rendered = predict(fit, obs.source_xyz_m, obs.receiver_xyz_m, obs.fs, id=obs.id)
                predicted = observe(rendered, id=obs.id, split=obs.split, config=config)
                rows = compare(fit, obs, predicted)
            except ValueError as error:
                # A valid measurement with unsupported prediction is a failure,
                # not an exclusion that improves the pass denominator.
                from dataclasses import replace
                pred_bands = {b: replace(v, values={k: None for k in v.values}, energy=0.,
                                         reasons={k: str(error) for k in v.values}) for b, v in obs.bands.items()}
                predicted = replace(obs, bands=pred_bands)
                rows = compare(fit, obs, predicted)
                exclusions.append({"id": obs.id, "stage": "prediction", "reason": str(error)})
            if not config.late:
                for row in rows:
                    if row["metric"] not in ("spectral_db", "drr_db"):
                        row.update(status="UNAVAILABLE", reason="single-reflection case does not validate diffuse-room decay/clarity")
            all_rows.extend(rows)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            splits.append({"id": obs.id, "split": obs.split, "position": list(obs.position_key),
                           "metadata": obs.metadata})
            early_rows.append({"id": obs.id, "split": obs.split, **early_diagnostics(fit.model, obs)})
            if obs.split == "train":
                approx.extend(fast_waveform_difference(fit, obs, predicted))
        for obs in train:
            process(obs)
        for count, (rir, id, split) in enumerate(pending()):
            try:
                obs = observe(rir, id=id, split=split, config=config)
            except ValueError as error:
                exclusions.append({"id": id, "split": split, "stage": "measurement", "reason": str(error)})
                continue
            process(obs)
            if (count+1) % 100 == 0:
                print(name, "predicted", count+1, "elapsed", round(time.monotonic()-started, 1), flush=True)
    summary = {"case": name, **summarize(all_rows), "elapsed_s": time.monotonic()-started,
               "environment": fingerprint(), "inputs": inputs, "exclusions": exclusions,
               "position_counts": dict(Counter(s["split"] for s in splits)),
               "bootstrap": fit.model.uncertainty,
               "limits": fit.report["limitations"]}
    write(folder/"positions.json", splits)
    write(folder/"early.json", early_rows)
    write(folder/"fast_vs_waveform.json", approx)
    write(folder/"summary.json", summary)
    print(name, summary["counts"], "seconds", round(summary["elapsed_s"], 2), flush=True)
    return summary


def meshrir(data, out, scratch, config, *, recover=False, drr_models=None):
    archive = data/"meshrir/S1-M3969_npy.zip"
    archive_hash = verified_sha(data, archive)
    context_path = data/"meshrir/room_context.json"
    verified_sha(data, context_path)
    context = json.loads(context_path.read_text())
    dims_m = tuple(float(x) for x in context["dimensions_m"])
    temperature_c = float(context["temperature_c"])
    with zipfile.ZipFile(archive) as z:
        pos = np.load(io.BytesIO(z.read("S1-M3969_npy/pos_mic.npy")), allow_pickle=False)
        source = np.load(io.BytesIO(z.read("S1-M3969_npy/pos_src.npy")), allow_pickle=False)[0]
        def member(i):
            a = np.load(io.BytesIO(z.read(f"S1-M3969_npy/ir_{i}.npy")), allow_pickle=False)
            return RIR(48000, a[0], tuple(source), tuple(pos[i]), {"dataset": "MeshRIR", "index": int(i)})
        fit = np.isin(np.round(pos[:, 0], 2), [-.5, -.3, -.1]) & np.isin(np.round(pos[:, 1], 2), [-.5, 0, .5]) & np.isin(np.round(pos[:, 2], 2), [-.2, 0, .2])
        val = np.isclose(pos[:, 0], 0) & np.isin(np.round(pos[:, 1], 2), [-.5, 0, .5]) & np.isin(np.round(pos[:, 2], 2), [-.2, 0, .2])
        train = [observe(member(i), id=f"mic{i}", config=config) for i in np.flatnonzero(fit)]
        folder = out/"meshrir"; folder.mkdir(parents=True, exist_ok=True)
        try:
            geo = (json.loads((Path(drr_models)/'meshrir/summary.json').read_text())['inputs']['geometry'] if drr_models else
                   RoomModel.from_json((folder/"room_model.json").read_text()).provenance if recover else
                   estimate_geometry(train, dims_m=dims_m, sound_speed_m_s=343*np.sqrt((273.15+temperature_c)/293.15)))
        except ValueError as e:
            write(folder/"summary.json", {"case": "meshrir", "status": "UNAVAILABLE", "reason": str(e), "input_sha256": archive_hash, "geometry": "grid frame has no supplied wall transform", "training_ids": [o.id for o in train]})
            print("meshrir unavailable:", e, flush=True)
            return
        model = RoomModel(room_id="meshrir", shoebox_dims_m=dims_m, temperature_c=temperature_c, provenance=geo)
        def pending():
            for i in np.flatnonzero(~fit):
                split = "validation" if val[i] else "blocked_test" if pos[i, 0] >= .1-1e-9 else "interpolation_buffer"
                yield member(i), f"mic{i}", split
        evaluate("meshrir", train, pending, model, config, folder, {archive.name: archive_hash, "geometry": geo}, recover=recover, drr_models=drr_models)


def bras(data, out, scratch, config, *, recover=False, drr_models=None):
    archive = data/"bras/1_scene_descriptions-RS1.zip"
    archive_hash = verified_sha(data, archive)
    context_path = data/"bras/RS1_context.json"
    verified_sha(data, context_path)
    context = json.loads(context_path.read_text())
    with zipfile.ZipFile(archive) as z:
        member = "1 Scene descriptions/RS1 single reflection (infinite plate)/RIRs/RS1_RIRs_Rigid.sofa"
        path = scratch/"RS1_RIRs_Rigid.sofa"; path.write_bytes(z.read(member))
    meta = sofa_metadata(path, ["ReceiverPosition", "EmitterPosition", "Data.SamplingRate"])
    r = meta["ReceiverPosition"].reshape(3, 10).T
    s = meta["EmitterPosition"].reshape(3, 10).T
    raw = sofa_ir(path, scratch, (10, 1, 11025))[:, 0]
    def item(i):
        return RIR(44100, raw[i], tuple(s[i]), tuple(r[i]), {"dataset": "BRAS-RS1-rigid", "index": i, "source_directivity": "unmodelled Genelec orientation"})
    cfg = config.model_copy(update={"late": False, "fit_surfaces": ("z_min",)})
    train = [observe(item(i), id=f"row{i}", config=cfg) for i in range(3)]
    alpha = uniform_bands(1.); alpha["z_min"] = (.2,)*7
    model = RoomModel(room_id="bras-rs1-rigid", shoebox_dims_m=(12., 8., 5.), absorption=alpha,
                      temperature_c=float(context["temperature_c"]),
                      humidity_pct=float(context["humidity_pct"]),
                      provenance={"geometry": "z=0 hard floor; five absorbing numerical enclosure planes, no room-volume inference",
                                  "source": "User-supplied pinned RS1_context.json; no on-axis/device correction applied"})
    def pending():
        for i in range(3, 10):
            yield item(i), f"row{i}", "validation" if i < 6 else "test" if i < 9 else "extra_position"
    evaluate("bras-rs1-rigid", train, pending, model, cfg, out/"bras-rs1-rigid", {archive.name: archive_hash, "sofa_sha256": sha(path)}, recover=recover, drr_models=drr_models)
    if not recover:
        write(out/"bras-cr2-unavailable.json", {"status": "UNAVAILABLE", "reason": "CR2 roomPlan/SOFA do not supply cuboid corners; detailed SKP has no verified exporter in this environment. RS1 rigid is the predeclared alternative.", "provenance": "No CR2 geometry or JND claim; publisher processing must be checked from the user-supplied archive."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", choices=("bras", "meshrir"), required=True)
    parser.add_argument("--bootstrap", default=20, type=int)
    parser.add_argument("--recover-reports", action="store_true", help="preserve saved fits/metrics; rebuild diagnostics with current code")
    parser.add_argument("--drr-models", type=Path, help="rescore held-out DRR/1 and DRR/2 on fixed fits from this directory; no refit")
    args = parser.parse_args()
    if args.drr_models and args.recover_reports:
        parser.error('choose DRR rescore or report recovery')
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out/"tmp"; scratch.mkdir(exist_ok=True)
    config = FitConfig(bootstrap=args.bootstrap)
    if not args.recover_reports:
        write(args.out/"run_environment.json", fingerprint())
    globals()[args.case](args.data, args.out, scratch, config, recover=args.recover_reports, drr_models=args.drr_models)


if __name__ == "__main__":
    main()
