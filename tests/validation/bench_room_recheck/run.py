"""Historical WP8 computation helpers; the SOFA executable is retired.

Use tests.validation.bench_room_recheck.preflight_v1 before preparing a new
frozen HDF5 protocol. Neither run() nor _load_pairs() may access legacy data.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sensorsim.acoustics.room.directivity import Directivity
from sensorsim.acoustics.absorption import amplitude_factor
from sensorsim.acoustics.room.metrics import OCTAVE_BANDS, early_reflections, octave_filter
from sensorsim.acoustics.room.model import OCTAVE_HZ as ROOM_OCTAVES, RoomModel, SURFACES, uniform_bands
from sensorsim.acoustics.room.rir import detect_onset
from sensorsim.acoustics.room.synth import (EarlyScattering, SynthConfig, early_paths,
                                            _path_kernel_grid, path_kernels, synthesize)
from sensorsim.evaluation.room_recheck import (
    compare_edge_feature_pair,
    octave_room_metrics,
    rectangular_room_modes,
    standardized_mean_difference,
)
from tests.validation.dechorate_v1 import V1Archive, STATES, validate_selection


TEMP_C = 20.0
RH_PCT = 50.0


@dataclass
class Pair:
    pair_id: str
    filename: str
    state: str
    source_id: int
    array_id: int
    microphone_id: int
    room_dims_m: tuple[float, float, float]
    source_xyz_m: tuple[float, float, float]
    receiver_xyz_m: tuple[float, float, float]
    sample_rate_hz: int
    rir: np.ndarray
    raw_room_description: str
    measured_onset: int
    measured_metrics: dict
    split: str


def _split(source_id: int, array_id: int) -> str:
    if source_id in (7, 8) and 1 <= array_id <= 4:
        return "fit_absorption"
    if source_id in (7, 8) and array_id == 5:
        return "validation_only"
    if source_id == 9 and array_id == 6:
        return "locked_audit"
    return "descriptive_only"


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


_RETIRED = (
    "historical WP8 SOFA route is retired; use "
    "tests.validation.bench_room_recheck.preflight_v1 and a new frozen HDF5 "
    "protocol with independent control signoff and custody before acceptance"
)


def _load_pairs(data_root: Path, expected_sample_rate_hz: int, *, allowed_splits=None):
    """Refuse every legacy selection before opening or enumerating inputs."""
    raise RuntimeError(_RETIRED)


def _load_v1_pairs(data_root: Path, expected_sample_rate_hz: int, *,
                   allowed_splits=('fit_absorption', 'validation_only'),
                   sources=(7, 8), states=STATES) -> tuple[list[Pair], dict]:
    """Read the registered fit/validation groups without SOFA or axis fallback.

    Source 8 does not exist in processed v1, so the requested two-source
    computation fails closed. Explicit source-7 calls are development only.
    """
    if tuple(allowed_splits) != ('fit_absorption', 'validation_only'):
        raise ValueError('v1 recheck permits only fit and validation splits together')
    validate_selection(sources, (1, 2, 3, 4, 5))
    if not states or len(set(states)) != len(states) or any(s not in STATES for s in states):
        raise ValueError('states must be a nonempty unique subset of the v1 room states')
    archive = V1Archive(data_root, expected_fs=expected_sample_rate_hz)
    geometry = archive.geometry
    pairs = []
    for state in sorted(states):
        for source in sources:
            for microphone in range(1, 26):
                array = (microphone-1)//5+1
                rir = archive.read_pair(state, source, microphone)
                onset = int(detect_onset(rir))
                metrics = octave_room_metrics(rir, archive.fs, onset_sample=onset)
                pairs.append(Pair(
                    pair_id=f'state{state}-src{source}-arr{array}-mic{microphone}',
                    filename=archive.path.relative_to(data_root).as_posix(),
                    state=state, source_id=source, array_id=array, microphone_id=microphone,
                    room_dims_m=tuple(geometry['room_size']),
                    source_xyz_m=tuple(geometry['srcs_omn'][:, source-7]),
                    receiver_xyz_m=tuple(geometry['mics'][:, microphone-1]),
                    sample_rate_hz=archive.fs, rir=rir,
                    raw_room_description='v1 state key; no SOFA metadata',
                    measured_onset=onset, measured_metrics=metrics, split=_split(source, array)))
    audit = archive.metadata()
    audit.update(pairs=len(pairs), samples_per_rir=archive.count,
                 room_dimensions_m=geometry['room_size'].tolist(),
                 selected_sources=list(sources), read_log=archive.read_log,
                 split_counts={s: sum(p.split == s for p in pairs) for s in allowed_splits},
                 excluded_pair_waveform_reads=0)
    return pairs, audit


def _model(dimensions, alphas, *, scattering=0.0, source_directivity=None) -> RoomModel:
    absorption = {surface: tuple(float(a) for a in alphas) for surface in SURFACES}
    return RoomModel(
        shoebox_dims_m=tuple(float(x) for x in dimensions),
        absorption=absorption,
        scattering=uniform_bands(float(scattering)),
        temperature_c=TEMP_C,
        humidity_pct=RH_PCT,
        source_directivity=source_directivity or Directivity(pattern="omni"),
        provenance={"wp8_recheck": "effective equal-wall absorption fitted on registered training split",
                    "temperature_assumed_c": TEMP_C, "humidity_assumed_pct": RH_PCT},
    )


def _fit_absorption(pairs: list[Pair]) -> tuple[dict[str, np.ndarray], dict]:
    by_state: dict[str, list[Pair]] = {}
    for pair in pairs:
        if pair.split == "fit_absorption":
            by_state.setdefault(pair.state, []).append(pair)
    fitted, details = {}, {}
    for state, rows in sorted(by_state.items()):
        alphas = np.full(len(OCTAVE_BANDS), 0.20, dtype=float)
        state_info = {"training_pair_count": len(rows), "bands": []}
        dims = rows[0].room_dims_m
        for index, band in enumerate(OCTAVE_BANDS):
            values = np.asarray([p.measured_metrics["t30_s"][index] for p in rows], dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            target = float(np.median(values)) if len(values) else np.nan
            record = {"band_hz": band, "valid_training_t30_count": int(len(values)),
                      "median_training_t30_s": target if np.isfinite(target) else None,
                      "absorption": 0.20, "fit_status": "fallback_default"}
            if len(values) >= 10:
                low, high = 0.0, 0.99
                t_low = float(_model(dims, [low] * len(OCTAVE_BANDS)).t60_estimates()[index])
                t_high = float(_model(dims, [high] * len(OCTAVE_BANDS)).t60_estimates()[index])
                if t_high <= target <= t_low:
                    for _ in range(64):
                        if high - low <= 1e-6:
                            break
                        middle = (low + high) / 2.0
                        t_middle = float(_model(dims, [middle] * len(OCTAVE_BANDS)).t60_estimates()[index])
                        if t_middle > target:
                            low = middle
                        else:
                            high = middle
                    alphas[index] = (low + high) / 2.0
                    record["absorption"] = float(alphas[index])
                    record["fit_status"] = "fitted_equal_wall_diffuse_equivalent"
            state_info["bands"].append(record)
        fitted[state] = alphas
        details[state] = state_info
    return fitted, details


def _tail_spectrum(rir, fs: int, onset: int, duration_s: float) -> np.ndarray:
    x = np.asarray(rir, dtype=float)
    start = int(onset) + round(0.080 * fs)
    stop = min(len(x), int(onset) + round(duration_s * fs))
    if stop - start < round(0.020 * fs):
        return np.full(len(OCTAVE_BANDS), np.nan)
    energy = []
    for band in OCTAVE_BANDS:
        filtered = octave_filter(x, fs, band)
        energy.append(float(np.dot(filtered[start:stop], filtered[start:stop])))
    energy = np.asarray(energy)
    total = energy.sum()
    return energy / total if total > 0 else np.full(len(energy), np.nan)


def _peak_rows(rir, onset, fs):
    return [{"excess_delay_s": float(p.excess_delay_s), "level_db": float(p.level_db)}
            for p in early_reflections(rir, fs, onset_sample=int(onset), max_time_s=0.080,
                                       min_spacing_s=0.001, threshold_db=-30.0)]


def _pair_peak_errors(measured, simulated, tolerance_s=0.010):
    left, right = measured[1:], simulated[1:]
    unused = set(range(len(right)))
    errors = []
    for peak in left:
        if not unused:
            break
        match = min(unused, key=lambda j: abs(peak["excess_delay_s"] - right[j]["excess_delay_s"]))
        error = abs(peak["excess_delay_s"] - right[match]["excess_delay_s"])
        if error <= tolerance_s:
            errors.append(error)
            unused.remove(match)
    return errors, len(left) - len(errors), len(unused)


def _render_measured_comparison(pairs: list[Pair], fitted, output: Path,
                                simulator_seed: int) -> tuple[dict, dict]:
    feature_config = {"fs": 16000, "window_s": 1.0, "n_fft": 1024, "hop": 256,
                      "n_mels": 40, "fmin_hz": 50.0, "fmax_hz": 8000.0}
    measured_features, simulated_features = [], []
    simulated_metrics = {k: [] for k in ("edt_s", "t30_s", "c50_db", "drr_db", "edt_valid", "t30_valid")}
    measured_metrics = {k: [] for k in simulated_metrics}
    sim_tails, measured_tails = [], []
    measured_peaks, simulated_peaks = [], []
    early_errors, early_missed_measured, early_extra_simulated = [], [], []
    sim_onsets, seed_values = [], []
    all_ids, splits, states, sources, arrays, mics = [], [], [], [], [], []
    source_xyzs, receiver_xyzs = [], []
    # source probes are deterministic and shared by all pair comparisons
    early_errors_by_pair = []
    for pair_index, pair in enumerate(pairs):
        alpha = fitted[pair.state]
        model = _model(pair.room_dims_m, alpha)
        fs = pair.sample_rate_hz
        seed = int(np.random.SeedSequence([simulator_seed, pair_index]).generate_state(1)[0])
        sim = synthesize(model, pair.source_xyz_m, pair.receiver_xyz_m, fs, seed=seed, config=SynthConfig())
        sim_onset = int(round(sim.metadata["direct_delay_s"] * fs + sim.metadata["latency_samples"]))
        sim_onsets.append(sim_onset)
        seed_values.append(seed)
        stats_m, stats_s = compare_edge_feature_pair(
            pair.rir, sim.samples, input_fs=fs, feature_fs=16000,
            measured_direct_sample=pair.measured_onset, simulated_direct_sample=sim_onset,
            feature_config=feature_config, broadband_seed=simulator_seed)
        measured_features.append(stats_m)
        simulated_features.append(stats_s)
        measured_metrics["edt_s"].append(pair.measured_metrics["edt_s"])
        measured_metrics["t30_s"].append(pair.measured_metrics["t30_s"])
        measured_metrics["c50_db"].append(pair.measured_metrics["c50_db"])
        measured_metrics["drr_db"].append(pair.measured_metrics["drr_db"])
        measured_metrics["edt_valid"].append(pair.measured_metrics["edt_valid"])
        measured_metrics["t30_valid"].append(pair.measured_metrics["t30_valid"])
        sim_metric = octave_room_metrics(sim.samples, fs, onset_sample=sim_onset, noise_power=0.0)
        for key in simulated_metrics:
            simulated_metrics[key].append(sim_metric[key])
        common_duration = min(0.600,
                              (len(pair.rir) - pair.measured_onset) / fs,
                              (len(sim.samples) - sim_onset) / fs)
        measured_tails.append(_tail_spectrum(pair.rir, fs, pair.measured_onset, common_duration))
        sim_tails.append(_tail_spectrum(sim.samples, fs, sim_onset, common_duration))
        m_peaks = _peak_rows(pair.rir, pair.measured_onset, fs)
        s_peaks = _peak_rows(sim.samples, sim_onset, fs)
        measured_peaks.append(m_peaks)
        simulated_peaks.append(s_peaks)
        errors, missed, extra = _pair_peak_errors(m_peaks, s_peaks)
        early_errors_by_pair.append(errors)
        early_errors.extend(errors)
        early_missed_measured.append(missed)
        early_extra_simulated.append(extra)
        all_ids.append(pair.pair_id)
        splits.append(pair.split)
        states.append(pair.state)
        sources.append(pair.source_id)
        arrays.append(pair.array_id)
        mics.append(pair.microphone_id)
        source_xyzs.append(pair.source_xyz_m)
        receiver_xyzs.append(pair.receiver_xyz_m)
    measured_features = np.asarray(measured_features, dtype=np.float32)
    simulated_features = np.asarray(simulated_features, dtype=np.float32)
    measured_tails, sim_tails = np.asarray(measured_tails), np.asarray(sim_tails)
    split_array = np.asarray(splits)
    audit = split_array == "locked_audit"
    smd_by_source = np.stack([
        standardized_mean_difference(measured_features[audit, source_index],
                                     simulated_features[audit, source_index])
        for source_index in range(3)
    ])
    all_smd_by_source = np.stack([
        standardized_mean_difference(measured_features[:, source_index],
                                     simulated_features[:, source_index])
        for source_index in range(3)
    ])
    m_metrics = {k: np.asarray(v) for k, v in measured_metrics.items()}
    s_metrics = {k: np.asarray(v) for k, v in simulated_metrics.items()}
    metric_summary = {}
    for metric in ("edt_s", "t30_s", "c50_db", "drr_db"):
        delta = s_metrics[metric][audit] - m_metrics[metric][audit]
        bands = []
        for index, band in enumerate(OCTAVE_BANDS):
            values = delta[:, index]
            values = values[np.isfinite(values)]
            bands.append({"band_hz": band, "valid_pair_count": int(len(values)),
                          "median_signed_sim_minus_measured": float(np.median(values)) if len(values) else None,
                          "median_absolute_difference": float(np.median(np.abs(values))) if len(values) else None})
        metric_summary[metric] = bands
    tail_delta_db = 10 * np.log10(np.maximum(measured_tails, 1e-15) /
                                  np.maximum(sim_tails, 1e-15))
    audit_errors = [error for index in np.flatnonzero(audit) for error in early_errors_by_pair[index]]
    feature_top = []
    for source_index, source_name in enumerate(("impulse", "chime", "broadband")):
        order = np.argsort(np.abs(smd_by_source[source_index]))[::-1][:5]
        feature_top.append({"source": source_name, "top_5_statistic_indices": order.tolist(),
                            "top_5_abs_smd": np.abs(smd_by_source[source_index, order]).tolist()})
    arrays_out = {
        "pair_id": np.asarray(all_ids), "split": split_array, "room_state": np.asarray(states),
        "source_id": np.asarray(sources), "array_id": np.asarray(arrays), "microphone_id": np.asarray(mics),
        "source_xyz_m": np.asarray(source_xyzs, dtype=float), "receiver_xyz_m": np.asarray(receiver_xyzs, dtype=float),
        "simulator_seed": np.asarray(seed_values, dtype=np.uint32),
        "measured_onset_sample": np.asarray([p.measured_onset for p in pairs]),
        "simulated_onset_sample": np.asarray(sim_onsets),
        "measured_feature_stats": measured_features,
        "simulated_feature_stats": simulated_features,
        "audit_smd_by_source_and_statistic": smd_by_source,
        "all_pair_smd_by_source_and_statistic": all_smd_by_source,
        "measured_tail_fraction": measured_tails,
        "simulated_tail_fraction": sim_tails,
        "tail_measured_minus_simulated_db": tail_delta_db,
    }
    for key, value in m_metrics.items():
        arrays_out[f"measured_{key}"] = value
    for key, value in s_metrics.items():
        arrays_out[f"simulated_{key}"] = value
    np.savez_compressed(output / "room_recheck_arrays.npz", **arrays_out)
    summary = {
        "label": "proxy result; local measured RIR comparison, not user-room measurements",
        "dataset": "supplied RIRs; see input provenance",
        "counts": {name: int(np.count_nonzero(split_array == name)) for name in sorted(set(splits))},
        "pair_count": len(pairs),
        "absorption_fit": {state: {"bands": rows["bands"], "training_pair_count": rows["training_pair_count"]}
                           for state, rows in _fit_details_cache.items()},
        "locked_audit_metrics": metric_summary,
        "early_reflection_timing": {
            "registration": "excess delays within 10 ms are greedily matched; candidates below -30 dB excluded",
            "matched_peak_count": int(len(audit_errors)),
            "all_pair_matched_peak_count": int(len(early_errors)),
            "median_absolute_matched_delay_error_ms": float(1000 * np.median(audit_errors)) if audit_errors else None,
            "scope": "locked audit pairs",
            "median_unmatched_measured_peaks_per_pair": float(np.median(np.asarray(early_missed_measured)[audit])),
            "median_unmatched_simulated_peaks_per_pair": float(np.median(np.asarray(early_extra_simulated)[audit])),
            "locked_audit_measured_peaks": [measured_peaks[i] for i in np.flatnonzero(audit)],
            "locked_audit_simulated_peaks": [simulated_peaks[i] for i in np.flatnonzero(audit)],
        },
        "tail_spectrum": {
            "median_measured_minus_simulated_db_by_band": [
                float(np.nanmedian(tail_delta_db[audit, i])) if np.isfinite(tail_delta_db[audit, i]).any() else None
                for i in range(len(OCTAVE_BANDS))],
            "bands_hz": list(OCTAVE_BANDS),
        },
        "feature_smd": {
            "feature_layout": ["mean_by_mel_band", "std_by_mel_band", "max_by_mel_band"],
            "mel_bands": 40,
            "audit_smd_shape": list(smd_by_source.shape),
            "top_5_audit_statistics_by_abs_smd": feature_top,
            "c2st": "deferred to WP1; no C2ST score is claimed",
        },
        "decision": "inconclusive: registered WP8(a) C2ST is unavailable in this lane",
        "smd_all_pairs_saved_for_wp1": True,
    }
    with (output / "room_recheck_summary.json").open("w") as stream:
        json.dump(_json_value(summary), stream, indent=2, allow_nan=False)
    return summary, {"measured_metrics": m_metrics, "simulated_metrics": s_metrics,
                     "measured_features": measured_features, "simulated_features": simulated_features,
                     "splits": split_array, "measured_tails": measured_tails, "simulated_tails": sim_tails}


_fit_details_cache: dict = {}


def _mesh_geometry_gate(data_root: Path, output: Path) -> dict:
    archive = data_root / "meshrir" / "S1-M3969_npy.zip"
    details = {"label": "proxy result; measured dataset metadata only",
               "archive_present": archive.is_file(), "acoustic_comparison": "inconclusive"}
    if archive.is_file():
        with zipfile.ZipFile(archive, "r") as zipped:
            metadata = json.loads(zipped.read("S1-M3969_npy/data.json"))
        mic_rows = [value["pos"] for key, value in metadata.items() if key.startswith("ir_") and "pos" in value]
        positions = np.asarray([[row["x"], row["y"], row["z"]] for row in mic_rows], dtype=float)
        details.update({
            "license": "CC BY 4.0; internal validation only",
            "sample_rate_hz": int(metadata["samplerate"]),
            "rir_count": len(mic_rows),
            "source_xyz_from_data_json": [metadata["src_0"]["pos"][k] for k in ("x", "y", "z")],
            "microphone_coordinate_min": positions.min(axis=0).tolist(),
            "microphone_coordinate_max": positions.max(axis=0).tolist(),
            "reason": "The local receiver grid is specified in a grid-local frame; its transform into the room-wall frame is not verified. A matched-geometry render would require guessing an origin.",
            "next_action": "Supply a verified grid-to-room transform; rerun this same bench with that transform.",
        })
    else:
        details["reason"] = "MeshRIR archive was not found under the supplied data root."
    with (output / "mesh_geometry_gate.json").open("w") as stream:
        json.dump(_json_value(details), stream, indent=2, allow_nan=False)
    return details


def _join_continuity(output: Path, preregistration: dict) -> dict:
    dimensions = tuple(preregistration["data"]["room_dimensions_m"])
    source, receiver = (tuple(row) for row in preregistration["data"]["geometry_source_receiver_m"])
    fs = int(preregistration["data"]["sample_rate_hz"])
    absorption_spec = preregistration["data"]["absorption"]
    absorption_match = re.match(r"\s*(\d+(?:\.\d+)?)", str(absorption_spec))
    if not absorption_match:
        raise ValueError("the preregistered absorption assumption must begin with a numeric coefficient")
    absorption = float(absorption_match.group(1))
    model = _model(dimensions, [absorption] * len(OCTAVE_BANDS))
    config = SynthConfig()
    steps, tail_spectra, prerings, basis_prerings = [], [], [], []
    rng_seeds = list(preregistration["seeds"])
    step_bands = tuple(preregistration["measurements"]["bands_hz"])
    for seed in rng_seeds:
        rir = synthesize(model, source, receiver, fs, seed=seed, config=config)
        middle = int(round(rir.metadata["mixing_time_s"] * fs + rir.metadata["latency_samples"]))
        half_window = round(0.010 * fs)
        per_band = []
        for band in step_bands:
            filtered = octave_filter(rir.samples, fs, band)
            left = filtered[max(0, middle - half_window):middle]
            right = filtered[middle:min(len(filtered), middle + half_window)]
            before = float(np.mean(left * left)) if len(left) else 0.0
            after = float(np.mean(right * right)) if len(right) else 0.0
            per_band.append(10 * np.log10(max(after, 1e-30) / max(before, 1e-30)))
        steps.append(per_band)
        start = middle + round((config.crossfade_s / 2 + 0.050) * fs)
        stop = min(len(rir.samples), middle + round(0.500 * fs))
        energies = []
        for band in OCTAVE_BANDS:
            filtered = octave_filter(rir.samples, fs, band)
            segment = filtered[start:stop]
            energies.append(float(np.dot(segment, segment)) if len(segment) else 0.0)
        energies = np.asarray(energies)
        tail_spectra.append(energies / energies.sum() if energies.sum() else np.full(7, np.nan))
        paths, _, _, _ = early_paths(model, np.asarray(source), np.asarray(receiver), config)
        _, indices, kernels = next(path_kernels(model, paths, fs, config))
        delay_samples = float(paths.delay_s[0]) * fs
        prerings.append(_kernel_prering_metrics(kernels[0], indices[0], delay_samples,
                                                config.filter_half_length, fs))
        basis_prerings.append([_kernel_prering_metrics(kernel, path_indices, delay_samples,
                                                       config.filter_half_length, fs)
                               for kernel, path_indices in _direct_octave_basis_kernels(
                                   model, paths, fs, config)])
    steps = np.asarray(steps, dtype=float)
    tail_spectra = np.asarray(tail_spectra, dtype=float)
    prerings = np.asarray(prerings, dtype=float)
    basis_prerings = np.asarray(basis_prerings, dtype=float)
    med_abs = np.median(np.abs(steps), axis=0)
    passed = bool(np.all(med_abs <= 1.0))
    np.savez_compressed(output / "join_continuity_arrays.npz", seeds=np.asarray(rng_seeds),
                        band_hz=np.asarray(step_bands), band_energy_step_db=steps,
                        tail_spectrum_fraction=tail_spectra, direct_kernel_prering=prerings,
                        direct_kernel_octave_basis_prering=basis_prerings)
    summary = {
        "label": "model-based result",
        "seed_count": len(rng_seeds), "seeds": rng_seeds, "band_hz": list(step_bands),
        "median_absolute_step_db_by_band": med_abs.tolist(),
        "median_signed_step_db_by_band": np.median(steps, axis=0).tolist(),
        "acceptance_threshold_db": 1.0,
        "acceptance": "pass" if passed else "fail",
        "tail_spectrum_median_fraction_by_band": np.nanmedian(tail_spectra, axis=0).tolist(),
        "tail_spectrum_band_hz": list(OCTAVE_BANDS),
        "direct_kernel_scope": "full-band pressure path kernel; not octave-separated",
        "direct_kernel_prering_median_energy_fraction": float(np.nanmedian(prerings[:, 0])),
        "direct_kernel_prering_median_peak_db_re_peak": float(np.nanmedian(prerings[:, 1])),
        "direct_kernel_prering_note": (
            "Geometric pre-arrival includes the dominant sample when its integer index rounds down; "
            "it is not a side-lobe-only measure."),
        "direct_kernel_pre_peak_sidelobe_median_energy_fraction": float(np.nanmedian(prerings[:, 2])),
        "direct_kernel_pre_peak_sidelobe_median_peak_db_re_peak": float(np.nanmedian(prerings[:, 3])),
        "direct_kernel_octave_basis_scope": "seven current interpolation-basis kernels; not seven bandpass filters",
        "direct_kernel_octave_basis_bands_hz": list(ROOM_OCTAVES),
        "direct_kernel_octave_basis_median_prearrival_energy_fraction":
            np.nanmedian(basis_prerings[:, :, 0], axis=0).tolist(),
        "direct_kernel_octave_basis_median_prearrival_peak_db_re_peak":
            np.nanmedian(basis_prerings[:, :, 1], axis=0).tolist(),
        "direct_kernel_octave_basis_median_pre_peak_energy_fraction":
            np.nanmedian(basis_prerings[:, :, 2], axis=0).tolist(),
        "direct_kernel_octave_basis_median_pre_peak_peak_db_re_peak":
            np.nanmedian(basis_prerings[:, :, 3], axis=0).tolist(),
        "room_assumptions": {"dimensions_m": list(dimensions), "absorption": 0.20,
                             "temperature_c": TEMP_C, "humidity_pct": RH_PCT},
    }
    with (output / "join_continuity_summary.json").open("w") as stream:
        json.dump(_json_value(summary), stream, indent=2, allow_nan=False)
    return summary


def _kernel_prering_metrics(kernel, sample_indices, delay_samples, latency_samples, fs):
    values = np.asarray(kernel, dtype=float)
    indices = np.asarray(sample_indices, dtype=int)
    time_before_direct = (indices - latency_samples) / fs < delay_samples / fs
    energy = float(np.dot(values, values))
    peak = float(np.max(np.abs(values)))
    early_values = values[time_before_direct]
    early_energy = float(np.dot(early_values, early_values))
    early_peak = float(np.max(np.abs(early_values))) if len(early_values) else 0.0
    peak_index = int(np.argmax(np.abs(values)))
    before_peak = values[:peak_index]
    before_peak_energy = float(np.dot(before_peak, before_peak))
    before_peak_max = float(np.max(np.abs(before_peak))) if len(before_peak) else 0.0
    return [early_energy / energy if energy else np.nan,
            20 * np.log10(max(early_peak, 1e-30) / max(peak, 1e-30)),
            before_peak_energy / energy if energy else np.nan,
            20 * np.log10(max(before_peak_max, 1e-30) / max(peak, 1e-30))]


def _direct_octave_basis_kernels(model, paths, fs: int, config: SynthConfig):
    """Recreate the direct path's seven interpolation-basis contributions."""
    nfft, frequencies, weights, window, offsets = _path_kernel_grid(fs, config.filter_half_length)
    delay = float(paths.delay_s[0]) * fs
    integer = int(np.floor(delay))
    fractional = delay - integer
    distance = float(paths.distance_m[0])
    air = (amplitude_factor(frequencies, distance, model.temperature_c, model.humidity_pct)
           if config.air_absorption else 1.0)
    if paths.directivity_gain.ndim == 1:
        directivity = float(paths.directivity_gain[0]) / distance
    else:
        directivity = np.asarray(paths.directivity_gain[0], dtype=float) / distance
    indices = integer + config.filter_half_length + offsets
    phase = np.exp(-2j * np.pi * fractional * frequencies / fs)
    for band_index in range(len(ROOM_OCTAVES)):
        gain = directivity if np.ndim(directivity) == 0 else float(directivity[band_index])
        transfer = paths.wall_pressure[0, band_index] * weights[band_index] * air * gain
        impulse = np.fft.fftshift(np.fft.irfft(transfer * phase, nfft))
        kernel = impulse[nfft // 2 - config.filter_half_length:
                         nfft // 2 + config.filter_half_length + 1] * window
        yield kernel, indices


def _early_variants(pairs: list[Pair], fitted, comparison_data, output: Path,
                    simulator_seed: int) -> dict:
    audit_rows = [i for i, p in enumerate(pairs) if p.split == "locked_audit"]
    variants = {"baseline": [], "cardioid": [], "scattering": []}
    seeds = (810, 811, 812)
    for pair_idx in audit_rows:
        pair = pairs[pair_idx]
        base_model = _model(pair.room_dims_m, fitted[pair.state])
        direct = np.asarray(pair.receiver_xyz_m) - np.asarray(pair.source_xyz_m)
        direct /= np.linalg.norm(direct)
        cardioid = _model(pair.room_dims_m, fitted[pair.state],
                          source_directivity=Directivity(pattern="cardioid", axis=tuple(direct)))
        diffuse = _model(pair.room_dims_m, fitted[pair.state], scattering=0.20)
        fs = pair.sample_rate_hz
        seed_base = int(np.random.SeedSequence([simulator_seed, pair_idx]).generate_state(1)[0])
        base = synthesize(base_model, pair.source_xyz_m, pair.receiver_xyz_m, fs,
                          seed=seed_base, config=SynthConfig())
        cardioid_rir = synthesize(cardioid, pair.source_xyz_m, pair.receiver_xyz_m, fs,
                                  seed=seed_base, config=SynthConfig())
        base_idx = int(round(base.metadata["direct_delay_s"] * fs + base.metadata["latency_samples"]))
        cardioid_idx = int(round(cardioid_rir.metadata["direct_delay_s"] * fs + cardioid_rir.metadata["latency_samples"]))
        variants["baseline"].append(octave_room_metrics(base.samples, fs, onset_sample=base_idx, noise_power=0.0)["edt_s"])
        variants["cardioid"].append(octave_room_metrics(cardioid_rir.samples, fs, onset_sample=cardioid_idx, noise_power=0.0)["edt_s"])
        for seed in seeds:
            scatter_cfg = SynthConfig(early_scattering=EarlyScattering(rays=256, max_bounces=4))
            scatter = synthesize(diffuse, pair.source_xyz_m, pair.receiver_xyz_m, fs,
                                 seed=seed, config=scatter_cfg)
            scatter_idx = int(round(scatter.metadata["direct_delay_s"] * fs + scatter.metadata["latency_samples"]))
            variants["scattering"].append(octave_room_metrics(scatter.samples, fs,
                                                               onset_sample=scatter_idx, noise_power=0.0)["edt_s"])
    arrays = {key: np.asarray(value, dtype=float) for key, value in variants.items()}
    measured = comparison_data["measured_metrics"]["edt_s"][audit_rows]
    arrays["measured_edt_s"] = measured
    arrays["audit_pair_id"] = np.asarray([pairs[i].pair_id for i in audit_rows])
    arrays["scatter_seed"] = np.tile(np.asarray(seeds), len(audit_rows))
    arrays["scatter_pair_id"] = np.repeat(np.asarray([pairs[i].pair_id for i in audit_rows]), len(seeds))
    np.savez_compressed(output / "early_variant_arrays.npz", **arrays)
    summary = {"label": "proxy result; measured dEchorate RIRs versus model-based variants",
               "locked_audit_pair_count": len(audit_rows), "scattering_seed_count_per_pair": len(seeds),
               "measured_source_directivity": "omnidirectional source subset; no directivity ground truth",
               "bands_hz": list(OCTAVE_BANDS), "variants": {}}
    for key, values in arrays.items():
        if key in ("measured_edt_s", "audit_pair_id", "scatter_seed", "scatter_pair_id"):
            continue
        if key == "scattering":
            measured_repeated = np.repeat(measured, len(seeds), axis=0)
        else:
            measured_repeated = measured
        delta = values - measured_repeated
        paired_delta = values - (np.repeat(arrays["baseline"], len(seeds), axis=0)
                                 if key == "scattering" else arrays["baseline"])
        summary["variants"][key] = {
            "reference": "measured dEchorate EDT",
            "variant": key,
            "valid_pair_seed_count_by_band": np.isfinite(delta).sum(axis=0).tolist(),
            "median_absolute_error_vs_measured_edt_s_by_band": [
                float(np.nanmedian(np.abs(delta[:, i]))) if np.isfinite(delta[:, i]).any() else None
                for i in range(len(OCTAVE_BANDS))],
            "median_signed_error_vs_measured_edt_s_by_band": [
                float(np.nanmedian(delta[:, i])) if np.isfinite(delta[:, i]).any() else None
                for i in range(len(OCTAVE_BANDS))],
            "median_paired_delta_edt_s_vs_baseline_by_band": [
                float(np.nanmedian(paired_delta[:, i])) if np.isfinite(paired_delta[:, i]).any() else None
                for i in range(len(OCTAVE_BANDS))],
        }
    summary["decision"] = "descriptive only; no registered WP8 acceptance threshold applies"
    with (output / "early_variant_summary.json").open("w") as stream:
        json.dump(_json_value(summary), stream, indent=2, allow_nan=False)
    return summary


def _write_modes(output: Path, preregistration: dict) -> dict:
    dimensions = tuple(preregistration["mode_examples"]["dimensions_m"])
    sound_speed = float(preregistration["mode_examples"]["sound_speed_m_s"])
    maximum_frequency = float(preregistration["mode_examples"]["maximum_frequency_hz"])
    modes = rectangular_room_modes(dimensions, sound_speed_m_s=sound_speed,
                                   maximum_frequency_hz=maximum_frequency)
    with (output / "rectangular_modes_example.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("nx", "ny", "nz", "frequency_hz"))
        writer.writerows((*mode.indices, mode.frequency_hz) for mode in modes)
    plan = {
        "label": "model-based example only; not the user's room",
        "dimensions_m": list(dimensions), "sound_speed_m_s": sound_speed,
        "maximum_frequency_hz": maximum_frequency, "mode_count": len(modes),
        "lf_ablation": {"prepared": True, "run": False,
                        "reason": "trained model and WP1 evaluator are unavailable",
                        "remove_below_hz": 250,
                        "decision_rule": "open modal-sum work iff recall changes by >1 take per class or audit triggers change by >25 percent"},
    }
    with (output / "lf_ablation_plan.json").open("w") as stream:
        json.dump(plan, stream, indent=2)
    return plan


def run(data_root: Path, output: Path, preregistration_path: Path) -> dict:
    """Retired entrypoint: no input reads and no output directory creation."""
    raise RuntimeError(_RETIRED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="historical argument; command is retired")
    parser.add_argument("--preregistration", type=Path, required=True,
                        help="job-only preregistration YAML; supplies geometry, sample rate and seeds")
    parser.add_argument("--output", type=Path, required=True,
                        help="job evidence output directory")
    args = parser.parse_args()
    run(args.data_root, args.output, args.preregistration)


if __name__ == "__main__":
    main()
