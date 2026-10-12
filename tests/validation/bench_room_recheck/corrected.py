"""Fresh WP8 review diagnostics; the original registration is not re-scored.

Supply --data-root, --preregistration, --prereg-sha256 and a NEW --output directory.
Only dEchorate fit/validation files are opened. Outputs contain derived metrics,
never measured RIR samples. The protocol lives outside the shipped repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from tests.validation.bench_room_recheck import run as legacy
from tests.validation.dechorate_v1 import validate_selection
from sensorsim.acoustics.room.metrics import OCTAVE_BANDS
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from sensorsim.evaluation.room_recheck import (
    direct_normalized_tail, early_peak_diagnostic, join_ensemble_summary,
    join_window_energies, octave_room_metrics,
)


def read_registration(path: Path, hash_path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = hash_path.read_text().split()[0]
    if digest != expected:
        raise ValueError("preregistration hash mismatch; no data were read")
    spec = yaml.safe_load(raw)
    if spec["protocol"] not in ("sensorsim-room-wp8-review/1", "sensorsim-room-wp8-hdf5/1"):
        raise ValueError("unsupported registration protocol")
    if spec["data"]["allowed_splits"] != ["fit_absorption", "validation_only"]:
        raise ValueError("this diagnostic permits only fit and validation splits")
    if spec["protocol"] != "sensorsim-room-wp8-hdf5/1" or spec['data'].get('edition') != 'dechorate-v1-5562386':
        raise ValueError('legacy SOFA registrations are historical; freeze a new HDF5 registration')
    if spec["join"]["geometry_count"] < 50:
        raise ValueError("at least 50 independent development geometries are required")
    return spec, digest


def write_json(path, value):
    path.write_text(json.dumps(legacy._json_value(value), indent=2, allow_nan=False)+"\n")


def provenance(digest):
    import sensorsim.acoustics.room.synth as synth
    import sensorsim.acoustics.room.fit as fit
    import sensorsim.acoustics.room.metrics as metrics
    import sensorsim.acoustics.room.model as model
    import sensorsim.evaluation.room_recheck as helpers
    paths = [Path(m.__file__) for m in (synth, fit, metrics, model, helpers, legacy)]
    paths.append(Path(__file__))
    return {"preregistration_sha256": digest,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def join_diagnostic(spec, output):
    j = spec["join"]
    dims = np.asarray(j["dimensions_m"])
    rng = np.random.default_rng(j["geometry_seed"])
    geometries = []
    for _ in range(100000):
        src, rec = rng.uniform(j["wall_margin_m"], dims-j["wall_margin_m"], (2, 3))
        if j["minimum_distance_m"] <= np.linalg.norm(src-rec) <= j["maximum_distance_m"]:
            geometries.append((src, rec))
        if len(geometries) == j["geometry_count"]:
            break
    if len(geometries) != j["geometry_count"]:
        raise ValueError("registered geometry constraints could not be sampled")
    model = legacy._model(dims, [j["absorption"]]*len(OCTAVE_BANDS))
    energies, continuations, mixing_times, seeds = [], [], [], []
    fs = j["sample_rate_hz"]
    for i, (src, rec) in enumerate(geometries):
        seed = int(np.random.SeedSequence([j["render_seed"], i]).generate_state(1)[0])
        rendered = synthesize(model, src, rec, fs, seed=seed, config=SynthConfig())
        mix = rendered.metadata["mixing_time_s"]
        config = SynthConfig(late=False, early_time_s=mix+j["continuation_extension_s"],
                             duration_s=len(rendered.samples)/fs)
        reference = synthesize(model, src, rec, fs, seed=seed, config=config)
        middle = round(mix*fs+rendered.metadata["latency_samples"])
        n = max(len(reference.samples), len(rendered.samples))
        # Pad both identically to keep zero-phase boundary conditions comparable.
        for samples, target in ((rendered.samples, energies), (reference.samples, continuations)):
            x = np.pad(samples, (0, n-len(samples)))
            target.append(join_window_energies(x, fs, middle, bands=j["bands_hz"], window_s=j["window_s"]))
        mixing_times.append(mix)
        seeds.append(seed)
    energies, continuations = np.asarray(energies), np.asarray(continuations)
    options = dict(bootstrap_draws=j["bootstrap_draws"], seed=j["bootstrap_seed"],
                   threshold_db=j["equivalence_margin_db"])
    result = join_ensemble_summary(energies, continuations, **options)
    null_control = join_ensemble_summary(continuations, continuations, **options)
    injected = continuations.copy()
    injected[:, :, 1] *= 10**(j["positive_control_step_db"]/10)
    positive_control = join_ensemble_summary(injected, continuations, **options)
    result.update(label="proxy result; model-based development geometries",
                  geometry_count=len(geometries), bands_hz=j["bands_hz"],
                  window_s=j["window_s"], geometry_seed=j["geometry_seed"],
                  no_join_control=null_control, positive_control=positive_control,
                  original_wp8_b="historical registered FAIL retained; not re-scored",
                  control_acceptance=("pass" if null_control["acceptance"] == "pass"
                                      and positive_control["acceptance"] == "fail" else "fail"))
    np.savez_compressed(output/"join_arrays.npz", geometries=np.asarray(geometries),
                        render_energy=energies, continuation_energy=continuations,
                        mixing_time_s=np.asarray(mixing_times), simulator_seed=np.asarray(seeds))
    write_json(output/"join_summary.json", result)
    return result


def finite_summary(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {"n": len(x), "median": float(np.median(x)) if len(x) else None}


def pair_seed(master, pair_id):
    token = int.from_bytes(hashlib.sha256(pair_id.encode()).digest()[:4], "little")
    return int(np.random.SeedSequence([master, token]).generate_state(1)[0])


def measured_diagnostic(spec, data_root, output):
    d = spec["data"]
    pairs, audit = legacy._load_v1_pairs(data_root, d["sample_rate_hz"],
                                     allowed_splits=tuple(d["allowed_splits"]))
    if d['temperature_c'] != audit['temperature_c'] or d['humidity_pct'] != 50.:
        raise ValueError('registered air must match logged v1 temperature and assumed 50 percent RH')
    legacy.TEMP_C, legacy.RH_PCT = audit['temperature_c'], 50.
    if any(p.source_id not in (7, 8) or p.array_id not in range(1, 6) for p in pairs):
        raise ValueError("unexpected pair outside the registered fit/validation partition")
    fitted, fit_details = legacy._fit_absorption(pairs)
    validation = [p for p in pairs if p.split == "validation_only"]
    if not validation:
        raise ValueError("no validation pairs")
    write_json(output/"fit_details.json", fit_details)
    # The allowlist is enforced before any waveform hyperslab is read.
    write_json(output/"input_manifest.json", {
        "label": "measured RIR inputs; internal proxy validation only",
        "files": [{"name": f, "sha256": digest} for f, digest in audit['input_sha256'].items()],
        "fit_pair_ids": [p.pair_id for p in pairs if p.split == "fit_absorption"],
        "validation_pair_ids": [p.pair_id for p in validation],
        "excluded_pair_waveform_reads": 0,
        "read_log": audit['read_log']})
    rows = []
    metric_names = ("edt_s", "t30_s", "c50_db", "drr_db")
    measured = {k: [] for k in metric_names}
    simulated = {k: [] for k in metric_names}
    auto = {k: [] for k in metric_names}
    tails, noise_fractions = [], []
    peaks = []
    for pair in validation:
        fs = pair.sample_rate_hz
        seed = pair_seed(d["render_seed"], pair.pair_id)
        model = legacy._model(pair.room_dims_m, fitted[pair.state])
        sim = synthesize(model, pair.source_xyz_m, pair.receiver_xyz_m, fs,
                         seed=seed, config=SynthConfig())
        on = round(sim.metadata["direct_delay_s"]*fs+sim.metadata["latency_samples"])
        clean = octave_room_metrics(sim.samples, fs, onset_sample=on, noise_power=0.0)
        prior = octave_room_metrics(sim.samples, fs, onset_sample=on)
        for k in metric_names:
            measured[k].append(pair.measured_metrics[k])
            simulated[k].append(clean[k])
            auto[k].append(prior[k])
        common = min(spec["tail"]["duration_s"], (len(pair.rir)-pair.measured_onset)/fs,
                     (len(sim.samples)-on)/fs)
        tm = direct_normalized_tail(pair.rir, fs, pair.measured_onset, duration_s=common)
        ts = direct_normalized_tail(sim.samples, fs, on, duration_s=common, noise_power=0.0)
        tails.append(tm["tail_to_direct_db"]-ts["tail_to_direct_db"])
        noise_fractions.append(tm["tail_noise_fraction"])
        # This strict subset ends at fade START, excluding the stochastic fade.
        cutoff = (sim.metadata["mixing_time_s"]-sim.metadata["crossfade_s"]/2
                  -sim.metadata["direct_delay_s"])
        p = spec["peaks"]
        peak = early_peak_diagnostic(pair.rir, sim.samples, fs=fs, measured_onset=pair.measured_onset,
            simulated_onset=on, cutoff_s=cutoff, tolerance_s=p["tolerance_s"],
            min_spacing_s=p["min_spacing_s"], threshold_db=p["threshold_db"],
            null_draws=p["null_draws"], seed=pair_seed(p["seed"], pair.pair_id))
        peaks.append(peak)
        rows.append({"pair_id": pair.pair_id, "state": pair.state, "seed": seed,
                     "cutoff_s": cutoff, "mixing_time_s": sim.metadata["mixing_time_s"],
                     "direct_delay_s": sim.metadata["direct_delay_s"],
                     "measured_onset_sample": pair.measured_onset, "simulated_onset_sample": on})
    measured, simulated, auto = ({k: np.asarray(v) for k, v in group.items()}
                                 for group in (measured, simulated, auto))
    summaries = {}
    for k in metric_names:
        summaries[k] = []
        for b in range(len(OCTAVE_BANDS)):
            summaries[k].append({"band_hz": OCTAVE_BANDS[b],
                "measured_RIR": finite_summary(measured[k][:, b]),
                "proxy_simulation_noise0": finite_summary(simulated[k][:, b]),
                "proxy_simulation_auto_noise": finite_summary(auto[k][:, b]),
                "proxy_paired_sim_minus_measured": finite_summary(simulated[k][:, b]-measured[k][:, b])})
    ratios = []
    mr = measured["edt_s"]/measured["t30_s"]
    sr = simulated["edt_s"]/simulated["t30_s"]
    for b in range(len(OCTAVE_BANDS)):
        paired = np.isfinite(mr[:, b]) & np.isfinite(sr[:, b])
        ratios.append({"band_hz": OCTAVE_BANDS[b],
            "measured_RIR_edt_over_t30": finite_summary(mr[:, b]),
            "proxy_simulated_edt_over_t30": finite_summary(sr[:, b]),
            "paired_measured_RIR_edt_over_t30": finite_summary(mr[paired, b]),
            "paired_proxy_simulated_edt_over_t30": finite_summary(sr[paired, b]),
            "proxy_paired_measured_ratio_over_simulated_ratio": finite_summary(mr[:, b]/sr[:, b])})
    fractions = [np.isfinite(simulated[k][:, 1:]).mean(axis=0) for k in metric_names]
    coverage = all(np.all(f >= spec["metrics"]["minimum_simulated_valid_fraction"]) for f in fractions)
    errors = [e for p in peaks for e in p["absolute_errors_s"]]
    losses = np.asarray([p["matching_loss_s"] for p in peaks])
    null_losses = np.asarray([p["null_matching_loss_s"] for p in peaks])
    null_medians = np.nanmedian(null_losses, axis=0)
    observed = finite_summary(losses)
    peak_summary = {
        "label": "proxy comparison; measured RIR candidates versus simulated candidates",
        "cutoff": "mixing time minus half-crossfade minus direct delay, relative to each onset",
        "matching": "maximum-cardinality then minimum-total-error; unmatched penalties equal tolerance",
        "valid_pair_count": observed["n"], "matched_count": sum(p["matched_count"] for p in peaks),
        "absolute_matched_error_s": finite_summary(errors),
        "measured_candidate_count": sum(p["measured_count"] for p in peaks),
        "simulated_candidate_count": sum(p["simulated_count"] for p in peaks),
        "unmatched_measured_count": sum(p["unmatched_measured"] for p in peaks),
        "unmatched_simulated_count": sum(p["unmatched_simulated"] for p in peaks),
        "median_matching_loss_s": observed["median"],
        "random_null_median_loss_quantiles_s": np.quantile(null_medians, [.025, .5, .975]),
         "null_lower_tail_fraction": ((1+np.count_nonzero(null_medians <= observed["median"]))/(len(null_medians)+1)
                                     if observed["median"] is not None else None),
        "decision": "inconclusive: descriptive diagnostic, no renderer acceptance threshold registered"}
    tail_summary = [{"band_hz": b, "proxy_measured_minus_simulated_db": finite_summary(np.asarray(tails)[:, i]),
                     "measured_RIR_noise_fraction": finite_summary(np.asarray(noise_fractions)[:, i])}
                    for i, b in enumerate(OCTAVE_BANDS)]
    by_state = {}
    for state in sorted(fitted):
        mask = np.asarray([p.state == state for p in validation])
        by_state[state] = [finite_summary(v) for v in (simulated["c50_db"][mask]-measured["c50_db"][mask]).T]
    result = {"label": "proxy result; measured RIR archive versus model-based renders, no user-room data",
              "input_counts": {"fit": len(pairs)-len(validation), "validation": len(validation),
                               "locked_audit_opened_or_scored": 0},
              "dataset_metadata": audit, "metrics": summaries, "edt_t30_ratios": ratios,
              "c50_sim_minus_measured_by_state": by_state,
              "b5_metric_coverage_acceptance": "pass" if coverage else "fail",
              "tail_spectrum": tail_summary, "early_peaks": peak_summary,
              "acoustic_acceptance": "inconclusive: descriptive diagnostics; no WP1 C2ST run"}
    arrays = {"pair_id": np.asarray([p.pair_id for p in validation]),
              "tail_measured_minus_simulated_db": np.asarray(tails),
              "measured_tail_noise_fraction": np.asarray(noise_fractions),
              "peak_matching_loss_s": losses, "peak_null_matching_loss_s": null_losses}
    for label, group in (("measured_RIR", measured), ("proxy_simulated_noise0", simulated),
                         ("proxy_simulated_auto_noise", auto)):
        arrays.update({f"{label}_{k}": v for k, v in group.items()})
    np.savez_compressed(output/"validation_arrays.npz", **arrays)
    write_json(output/"validation_pairs.json", rows)
    write_json(output/"early_peak_pairs.json", peaks)
    write_json(output/"measured_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "preregistration", "prereg-sha256", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    spec, digest = read_registration(args.preregistration, args.prereg_sha256)
    # Fail the incomplete measured selection before spending work on a join run.
    validate_selection((7, 8), (1, 2, 3, 4, 5))
    # Never replace an earlier acceptance result.
    args.output.mkdir(parents=True, exist_ok=False)
    legacy.TEMP_C = spec["data"]["temperature_c"]
    legacy.RH_PCT = spec["data"]["humidity_pct"]
    write_json(args.output/"provenance.json", provenance(digest))
    join = join_diagnostic(spec, args.output)
    print("proxy join:", join["acceptance"], join["step_excess_db"], flush=True)
    measured = measured_diagnostic(spec, args.data_root, args.output)
    summary = {"preregistration_sha256": digest, "proxy_join": join["acceptance"],
               "proxy_join_controls": join["control_acceptance"],
               "proxy_b5_metric_coverage": measured["b5_metric_coverage_acceptance"],
               "proxy_acoustic_acceptance": measured["acoustic_acceptance"]}
    write_json(args.output/"acceptance_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
