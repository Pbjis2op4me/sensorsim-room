"""Expected join energy with paired ISM references and seed-cluster intervals.

All built-in inputs are model-based proxies. Controls validate the estimator;
room-family results qualify statistical closure, not each coherent realization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time as clock
import numpy as np
from scipy import signal
from sensorsim.acoustics.room import synth
from sensorsim.acoustics.room.metrics import octave_sos
from sensorsim.acoustics.room.model import RoomModel, OCTAVE_HZ
from .run import verify_registration, precursor_trials


def decision(point, interval, threshold=1., ci_limit=1.5):
    point, interval = np.asarray(point), np.asarray(interval)
    if not np.isfinite(point).all() or not np.isfinite(interval).all():
        return "inconclusive"
    if np.all(abs(point) <= threshold) and np.all(abs(interval) <= ci_limit):
        return "pass"
    if np.any((point > threshold) & (interval[0] > threshold) |
              (point < -threshold) & (interval[1] < -threshold)):
        return "fail"
    return "inconclusive"


def expected_step(energies, reference):
    """E[power] first, log second; final axes are band and before/after."""
    e, r = np.asarray(energies, float), np.asarray(reference, float)
    if e.shape[-1] != 2 or r.shape[-1] != 2 or np.any(e <= 0) or np.any(r <= 0):
        raise ValueError("positive paired window powers required")
    return 10*np.log10(e[..., 1]/e[..., 0]) - 10*np.log10(r[..., 1]/r[..., 0])


def summarize(energy, reference, seed, repetitions=2000):
    # geometry, seed, band, window. A seed cluster spans bands and geometries.
    energy, reference = np.asarray(energy), np.asarray(reference)
    point = expected_step(energy.mean((0, 1)), reference.mean(0))
    rng = np.random.default_rng(seed)
    boot, joint = [], []
    for _ in range(repetitions):
        si = rng.integers(energy.shape[1], size=energy.shape[1])
        gi = rng.integers(energy.shape[0], size=energy.shape[0])
        e = energy[:, si].mean(1)
        boot.append(expected_step(e.mean(0), reference.mean(0)))
        joint.append(expected_step(e[gi].mean(0), reference[gi].mean(0)))
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    return dict(point_db=point.tolist(), seed_ci95_db=ci.tolist(),
                geometry_seed_ci95_db=np.percentile(joint, [2.5, 97.5], axis=0).tolist(),
                per_geometry_db=expected_step(energy.mean(1), reference).tolist(),
                decision=decision(point, ci))


def controls(spec):
    cfg, bands = spec["controls"], spec["join"]["bands_hz"]
    fs, decay = cfg["fs"], cfg["t60_s"]
    t = np.arange(round(.6*fs))/fs-.55
    env = np.exp(-3*np.log(10)*np.maximum(t, -.05)/decay)
    w = spec["criterion"]["window_s"]
    masks = [(t >= -w) & (t < 0), (t >= 0) & (t < w)]
    reference = np.array([[np.mean(env[m]**2) for m in masks]]*len(bands))
    energy = []
    for seed in range(cfg["seed_start"], cfg["seed_start"]+cfg["count"]):
        rng = np.random.default_rng(seed)
        energy.append([[np.mean(x[m]**2) for m in masks] for x in
                       [signal.sosfilt(octave_sos(fs, b), rng.standard_normal(len(t)))*env for b in bands]])
    energy = np.array(energy)
    result = {}
    for step in cfg["injection_db"]:
        x = energy.copy(); x[..., 1] *= 10**(step/10)
        row = summarize(x[None], reference[None], spec["criterion"]["bootstrap_seed"])
        row["per_band_decision"] = [decision(p, np.array(row["seed_ci95_db"])[:, i])
                                    for i, p in enumerate(row["point_db"])]
        result[str(step)] = row
    valid = all(x == "pass" for x in result["0.0"]["per_band_decision"]) and all(
        x == "fail" for k in ("1.5", "-1.5") for x in result[k]["per_band_decision"])
    return dict(label="proxy result; model-based estimator controls", controls=result,
                decision="pass" if valid else "fail", per_seed_window_power=energy.tolist())


def _late(module, bands, levels, t60, time, mixing, fade, fs, seed):
    if hasattr(module, "render_late"):
        return module.render_late(bands, levels, t60, time, mixing, fade, fs, seed)
    rng = np.random.default_rng(seed)
    out = np.zeros(len(time))
    for band, level, decay in zip(bands, levels, t60):
        if level:
            env = module.late_envelope(time, mixing, fade, decay)
            out += signal.sosfilt(band.sos, rng.standard_normal(len(time))*env*np.sqrt(level))
    return out


def geometry_trials(module, family, source, receiver, spec):
    fs = spec["join"]["fs"]
    model = RoomModel(shoebox_dims_m=family["dimensions_m"])
    cfg = module.SynthConfig()
    seeds = range(spec["join"]["seeds"]["start"], spec["join"]["seeds"]["start"]+spec["join"]["seeds"]["count"])
    first = module.synthesize(model, source, receiver, fs, seed=seeds.start)
    mixing, fade = first.metadata["mixing_time_s"], cfg.crossfade_s
    t = (np.arange(len(first.samples))-cfg.filter_half_length)/fs
    paths = module.early_paths(model, source, receiver, cfg)[0]
    early = module._render_paths(model, paths, fs, len(t), cfg)
    early *= np.cos(np.pi/2*module.crossfade_fraction(t, mixing, fade))
    bands = module.late_bands(fs)
    levels = np.array(first.metadata["late_broadband_variance_at_mixing"])
    t60 = module.late_band_t60(bands, model.decay_budget().t60_s, model, cfg.air_absorption)
    replay = (early+_late(module, bands, levels, t60, t, mixing, fade, fs, seeds.start)).astype(np.float32)
    if not np.array_equal(replay, first.samples):
        raise AssertionError("cached geometry replay differs from public renderer")
    ism = module.synthesize(model, source, receiver, fs,
        config=module.SynthConfig(late=False, early_time_s=mixing+.080))
    bank, masks, reference, trend = [], [], [], []
    w = spec["criterion"]["window_s"]
    for b in spec["join"]["bands_hz"]:
        sos = octave_sos(fs, b); bank.append(sos)
        imp = np.zeros(round(.2*fs)); imp[0] = 1
        h = signal.sosfilt(sos, imp)
        delay = np.dot(np.arange(len(h))/fs, h*h)/np.dot(h,h)
        local = t-delay
        mm = [(local >= mixing-w) & (local < mixing), (local >= mixing) & (local < mixing+w)]
        masks.append(mm)
        x = signal.sosfilt(sos, np.pad(ism.samples, (0, max(0,len(t)-len(ism.samples)))))[:len(t)]
        reference.append([np.mean(x[m]**2) for m in mm])
        env2 = np.exp(-6*np.log(10)*(local-mixing)/model.decay_budget().t60_s[OCTAVE_HZ.index(b)])
        trend.append([np.mean(env2[m]) for m in mm])
    energy = []
    for seed in seeds:
        x = (early+_late(module, bands, levels, t60, t, mixing, fade, fs, seed)).astype(np.float32)
        energy.append([[np.mean(y[m]**2) for m in mm] for y, mm in
            [(signal.sosfilt(sos,x), mm) for sos, mm in zip(bank,masks)]])
    return dict(source_m=list(source), receiver_m=list(receiver), mixing_time_s=mixing,
                energy=np.array(energy), reference=np.array(reference), trend=np.array(trend))


def join_trials(spec, module=synth):
    families = []
    rng = np.random.default_rng(spec["join"]["geometry_seed"])
    for f in spec["join"]["families"]:
        pairs = [(f["source_m"], f["receiver_m"])]
        while len(pairs) <= spec["join"]["random_pairs_per_family"]:
            s, r = rng.uniform(.5, np.array(f["dimensions_m"])-.5, (2,3))
            if np.linalg.norm(s-r) >= 1: pairs.append((s,r))
        rows = []
        for i, (s,r) in enumerate(pairs):
            rows.append(geometry_trials(module, f, s, r, spec))
            if i % 10 == 0: print(f["id"], i, flush=True)
        e, ref = np.array([r["energy"] for r in rows]), np.array([r["reference"] for r in rows])
        summary = summarize(e, ref, spec["criterion"]["bootstrap_seed"])
        summary["exponential_reference_db"] = expected_step(e.mean(1), np.array([r["trend"] for r in rows])).tolist()
        for row in rows:
            for k in ("energy", "reference", "trend"): row[k] = row[k].tolist()
        families.append(dict(id=f["id"], summary=summary, geometries=rows))
    outcomes = [f["summary"]["decision"] for f in families]
    return dict(label="proxy result; model-based expected energy versus unjoined ISM", families=families,
                decision="fail" if "fail" in outcomes else "pass" if all(x=="pass" for x in outcomes) else "inconclusive")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preregistration", type=Path, required=True)
    p.add_argument("--hash-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("controls","join","precursor"), required=True)
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError("preserve previous evidence")
    digest, spec = verify_registration(args.preregistration, args.hash_file)
    start = clock.monotonic()
    result = controls(spec) if args.mode == "controls" else join_trials(spec) if args.mode == "join" else precursor_trials(spec["precursor"])
    result.update(preregistration_sha256=digest, elapsed_s=clock.monotonic()-start,
                  renderer_sha256=hashlib.sha256(Path(synth.__file__).read_bytes()).hexdigest())
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("families","per_seed_window_power","cases")}))

if __name__ == "__main__": main()
