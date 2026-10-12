"""Reusable measurements for the WP8 room-simulation recheck.

This module contains no dataset paths or licensed data. Dataset readers belong in
the validation bench; callers pass measured and simulated RIR arrays explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.signal import resample_poly

from sensorsim.acoustics.room.metrics import (OCTAVE_BANDS, early_reflections, filter_span,
                                               octave_filter, octave_sos, room_metrics)
from sensorsim.edge import Features


@dataclass(frozen=True)
class RectangularMode:
    indices: tuple[int, int, int]
    frequency_hz: float


def rectangular_room_modes(dimensions_m, *, sound_speed_m_s: float = 343.0,
                           maximum_frequency_hz: float = 250.0) -> list[RectangularMode]:
    """Enumerate rigid rectangular-room modes up to a declared frequency."""
    dimensions = np.asarray(dimensions_m, dtype=float)
    if (dimensions.shape != (3,) or not np.isfinite(dimensions).all() or
            np.any(dimensions <= 0) or not np.isfinite(sound_speed_m_s) or
            sound_speed_m_s <= 0 or not np.isfinite(maximum_frequency_hz) or
            maximum_frequency_hz <= 0):
        raise ValueError("dimensions, sound speed, and maximum frequency must be positive and finite")
    limits = np.floor(2.0 * maximum_frequency_hz * dimensions / sound_speed_m_s).astype(int)
    result = []
    for nx in range(limits[0] + 1):
        for ny in range(limits[1] + 1):
            for nz in range(limits[2] + 1):
                if nx == ny == nz == 0:
                    continue
                frequency = sound_speed_m_s / 2.0 * np.sqrt(
                    (nx / dimensions[0])**2 + (ny / dimensions[1])**2 + (nz / dimensions[2])**2
                )
                if frequency <= maximum_frequency_hz:
                    result.append(RectangularMode((nx, ny, nz), float(frequency)))
    return sorted(result, key=lambda mode: (mode.frequency_hz, mode.indices))


def edge_band_statistics(audio_windows, feature_config: dict) -> np.ndarray:
    """Return per-window mean/std/max of `edge.Features` log-mel values.

    Output shape is ``(window_count, 3 * mel_band_count)``. This is the 120-value
    WP1 representation when the configured feature extractor has 40 mel bands.
    """
    windows = np.asarray(audio_windows, dtype=np.float32)
    if windows.ndim == 1:
        windows = windows[None, :]
    if windows.ndim != 2 or not np.isfinite(windows).all():
        raise ValueError("audio_windows must be a finite (windows, samples) array")
    logmel = Features(feature_config)(windows)[:, 0]
    stats = np.concatenate((logmel.mean(axis=-1), logmel.std(axis=-1),
                            logmel.max(axis=-1)), axis=1)
    return np.asarray(stats, dtype=np.float32)


def standardized_mean_difference(reference, candidate, *, axis: int = 0) -> np.ndarray:
    """Per-feature SMD using the average of the two sample variances."""
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.ndim != b.ndim or a.shape[1:] != b.shape[1:] or a.shape[axis] < 2 or b.shape[axis] < 2:
        raise ValueError("SMD inputs need matching feature dimensions and at least two samples")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("SMD inputs must be finite")
    mean_delta = np.mean(a, axis=axis) - np.mean(b, axis=axis)
    pooled = np.sqrt((np.var(a, axis=axis, ddof=1) + np.var(b, axis=axis, ddof=1)) / 2.0)
    return np.divide(mean_delta, pooled, out=np.where(mean_delta == 0, 0.0, np.copysign(np.inf, mean_delta)),
                     where=pooled > 0)


def make_dry_sources(fs: int, *, broadband_seed: int = 20261008,
                     onset_s: float = 0.1, duration_s: float = 0.4) -> dict[str, np.ndarray]:
    """Generate the registered impulse, three-partial chime, and white-noise probe."""
    if (not isinstance(fs, (int, np.integer)) or fs < 1000 or not np.isfinite(onset_s) or
            onset_s < 0 or not np.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("fs, onset, and duration must be positive and finite")
    onset_sample = round(onset_s * fs)
    count = round(duration_s * fs)
    total = onset_sample + count
    impulse = np.zeros(onset_sample + 1, dtype=np.float32)
    impulse[onset_sample] = 1.0
    time = np.arange(count, dtype=float) / fs
    envelope = np.exp(-time / 0.4)
    chime = np.zeros(total, dtype=np.float32)
    chime[onset_sample:] = envelope * sum(
        gain * np.sin(2 * np.pi * frequency * time)
        for frequency, gain in ((880.0, 1.0), (1175.0, 0.6), (1568.0, 0.35)))
    broadband = np.zeros(total, dtype=np.float32)
    broadband[onset_sample:] = (
        np.random.default_rng(broadband_seed).standard_normal(count) * envelope)
    return {"impulse": impulse, "chime": chime.astype(np.float32),
            "broadband": broadband.astype(np.float32)}


def convolve_aligned_window(rir, dry_source, *, direct_sample: int, fs: int,
                            window_s: float = 1.0, direct_at_s: float = 0.1,
                            source_onset_sample: int = 0) -> np.ndarray:
    """Convolve linearly and place the declared direct arrival at `direct_at_s`.

    `direct_sample` is an array index, including any synthesis latency. This only
    aligns the onset for feature comparison; it does not alter the RIR metrics.
    """
    h = np.asarray(rir, dtype=float).reshape(-1)
    x = np.asarray(dry_source, dtype=float).reshape(-1)
    if (not np.isfinite(h).all() or not np.isfinite(x).all() or not len(h) or not len(x) or
            not isinstance(direct_sample, (int, np.integer)) or not 0 <= direct_sample < len(h) or
            not isinstance(source_onset_sample, (int, np.integer)) or
            not 0 <= source_onset_sample < len(x) or
            not isinstance(fs, (int, np.integer)) or fs <= 0 or window_s <= 0 or direct_at_s < 0):
        raise ValueError("invalid RIR, source, direct sample, sample rate, or window")
    convolved = signal.fftconvolve(x, h, mode="full")
    destination = round(direct_at_s * fs)
    shift = destination - int(source_onset_sample) - int(direct_sample)
    window = np.zeros(round(window_s * fs), dtype=np.float32)
    source_start = max(0, -shift)
    destination_start = max(0, shift)
    count = min(len(convolved) - source_start, len(window) - destination_start)
    if count > 0:
        window[destination_start:destination_start + count] = convolved[source_start:source_start + count]
    return window


def resample_rir(rir, input_fs: int, output_fs: int) -> np.ndarray:
    """Polyphase-resample a mono RIR and retain deterministic float64 intermediates."""
    x = np.asarray(rir, dtype=float).reshape(-1)
    if not np.isfinite(x).all() or input_fs <= 0 or output_fs <= 0:
        raise ValueError("RIR and sample rates must be finite and positive")
    common = int(np.gcd(input_fs, output_fs))
    return x if input_fs == output_fs else resample_poly(x, output_fs // common, input_fs // common)


def compare_edge_feature_pair(measured_rir, simulated_rir, *, input_fs: int, feature_fs: int,
                              measured_direct_sample: int, simulated_direct_sample: int,
                              feature_config: dict, broadband_seed: int = 20261008,
                              source_onset_s: float = 0.1, source_duration_s: float = 0.4) -> tuple[dict, dict]:
    """Render the same procedural sources through two RIRs and summarize edge features."""
    measured = resample_rir(measured_rir, input_fs, feature_fs)
    simulated = resample_rir(simulated_rir, input_fs, feature_fs)
    measured_direct = round(measured_direct_sample * feature_fs / input_fs)
    simulated_direct = round(simulated_direct_sample * feature_fs / input_fs)
    source_onset_sample = round(source_onset_s * feature_fs)
    sources = make_dry_sources(feature_fs, broadband_seed=broadband_seed,
                               onset_s=source_onset_s, duration_s=source_duration_s)
    result = {}
    for label, rir, direct in (("measured", measured, measured_direct),
                               ("simulated", simulated, simulated_direct)):
        windows = np.stack([convolve_aligned_window(
            rir, source, direct_sample=direct, fs=feature_fs,
            source_onset_sample=source_onset_sample)
                            for source in sources.values()])
        result[label] = edge_band_statistics(windows, feature_config)
    return result["measured"], result["simulated"]


def octave_room_metrics(rir, fs: int, *, onset_sample: int,
                        noise_power: float | None = None) -> dict[str, np.ndarray]:
    """Measure EDT, T30, C50 and DRR in every supported nominal octave.

    None estimates noise in each band; use 0.0 for noise-free simulated RIRs.
    A nonzero explicit value is in each analyzed band's power units.
    """
    output = {name: [] for name in ("edt_s", "t30_s", "c50_db", "drr_db")}
    valid = {name: [] for name in ("edt", "t30")}
    for band in OCTAVE_BANDS:
        measured = room_metrics(rir, fs, onset_sample=int(onset_sample), band=band,
                                noise_power=noise_power)
        output["edt_s"].append(measured.edt.rt60_s if measured.edt.valid else np.nan)
        output["t30_s"].append(measured.t30.rt60_s if measured.t30.valid else np.nan)
        output["c50_db"].append(measured.c50_db if measured.c50_db is not None else np.nan)
        output["drr_db"].append(measured.drr_db if measured.drr_db is not None else np.nan)
        valid["edt"].append(measured.edt.valid)
        valid["t30"].append(measured.t30.valid)
    return {**{k: np.asarray(v, dtype=float) for k, v in output.items()},
            **{f"{k}_valid": np.asarray(v, dtype=bool) for k, v in valid.items()}}


def join_window_energies(rir, fs: int, mixing_sample: int, *, bands, window_s) -> np.ndarray:
    """Zero-phase mean-square around a join, with no analysis-filter delay.

    Columns are before, after, centred on the join, and +20 to +60 ms.
    The first three windows have the caller's per-band width. Apply precisely
    the same filter and windows to the renderer and its ISM continuation.
    Forward/backward filtering squares the filter magnitude on both sides.
    """
    x = np.asarray(rir, dtype=float)
    if (x.ndim != 1 or not len(x) or not np.isfinite(x).all() or fs <= 0
            or len(bands) != len(window_s)):
        raise ValueError("invalid signal, sample rate, or window specification")
    result = []
    for band, width_s in zip(bands, window_s):
        width = round(width_s * fs)
        windows = [(mixing_sample-width, mixing_sample),
                   (mixing_sample, mixing_sample+width),
                   (mixing_sample-width//2, mixing_sample-width//2+width),
                   (mixing_sample+round(.020*fs), mixing_sample+round(.060*fs))]
        if width <= 0 or any(a < 0 or b > len(x) or b <= a for a, b in windows):
            raise ValueError("analysis window outside the available RIR")
        y = signal.sosfiltfilt(octave_sos(fs, band), x)
        result.append([float(np.mean(y[a:b]**2)) for a, b in windows])
    return np.asarray(result)


def join_ensemble_summary(render_energy, continuation_energy, *, bootstrap_draws: int,
                          seed: int, threshold_db: float) -> dict:
    """Difference of expected-energy steps, with paired geometry bootstrap.

    Average powers before taking logs, rather than averaging absolute log ratios
    of short noisy windows. The independent ISM-vs-ISM bootstrap reports the
    continuation's own finite-geometry spread. It is descriptive, not a fitted
    acceptance threshold. Confidence limits are pointwise by band.
    """
    r, c = np.asarray(render_energy, float), np.asarray(continuation_energy, float)
    if (r.shape != c.shape or r.ndim != 3 or r.shape[0] < 2 or r.shape[2] != 4
            or not np.isfinite(r).all() or not np.isfinite(c).all()
            or np.any(r <= 0) or np.any(c <= 0) or bootstrap_draws < 2 or threshold_db <= 0):
        raise ValueError("positive finite geometry-by-band-by-four energies are required")
    def step(e):
        return 10*np.log10(e[..., 1]/e[..., 0])
    def excess(a, b):
        return step(a.mean(axis=0))-step(b.mean(axis=0))
    point = excess(r, c)
    rng = np.random.default_rng(seed)
    paired, null = [], []
    for _ in range(bootstrap_draws):
        indices = rng.integers(0, len(r), len(r))
        other = rng.integers(0, len(r), len(r))
        paired.append(excess(r[indices], c[indices]))
        null.append(excess(c[indices], c[other]))
    ci = np.quantile(paired, [.025, .975], axis=0).T
    passed = np.all((ci[:, 0] >= -threshold_db) & (ci[:, 1] <= threshold_db))
    failed = np.any((ci[:, 1] < -threshold_db) | (ci[:, 0] > threshold_db))
    return {"step_excess_db": point.tolist(), "step_excess_ci95_db": ci.tolist(),
            "renderer_expected_step_db": step(r.mean(axis=0)).tolist(),
            "ism_expected_step_db": step(c.mean(axis=0)).tolist(),
            "ism_independent_null_ci95_db": np.quantile(null, [.025, .975], axis=0).T.tolist(),
            "centred_join_render_over_ism_db": (10*np.log10(r.mean(0)[:, 2]/c.mean(0)[:, 2])).tolist(),
            "late_render_over_ism_db": (10*np.log10(r.mean(0)[:, 3]/c.mean(0)[:, 3])).tolist(),
            "acceptance": "pass" if passed else "fail" if failed else "inconclusive"}


def direct_normalized_tail(rir, fs: int, onset: int, *, duration_s: float = .6,
                           noise_power: float | None = None) -> dict:
    """Band tail/direct ratios, without sum-to-one spectral normalization.

    Tail: onset+80 ms through the common duration, causal octave analysis.
    Direct: isolated +/-2.5 ms component, including its complete filter ring-out.
    None estimates tail noise from the last 10% of the recording, reporting
    its fraction explicitly; zero disables subtraction for simulated renders.
    Per-band multiplicative channel power cancels. Within-band coloration,
    directivity, and reflections in the direct window remain limitations.
    """
    x = np.asarray(rir, float)
    if (x.ndim != 1 or not np.isfinite(x).all() or fs <= 0 or not 0 <= onset < len(x)
            or not np.isfinite(duration_s) or duration_s <= .1
            or (noise_power is not None and (not np.isfinite(noise_power) or noise_power < 0))):
        raise ValueError("invalid RIR, onset, duration, or noise power")
    start, stop = onset+round(.080*fs), min(len(x), onset+round(duration_s*fs))
    half = round(.0025*fs)
    if stop-start < round(.020*fs) or onset+half >= len(x):
        raise ValueError("insufficient direct/tail window support")
    direct = x[max(0, onset-half):onset+half+1]
    ratios, noises, directs, tails = [], [], [], []
    for band in OCTAVE_BANDS:
        y = octave_filter(x, fs, band)
        raw = float(np.sum(y[start:stop]**2))
        noise = float(np.mean(y[int(.9*len(y)):]**2)) if noise_power is None else float(noise_power)
        tail = raw-noise*(stop-start)
        d = octave_filter(np.pad(direct, (0, filter_span(fs, band))), fs, band)
        direct_energy = float(d@d)
        ratios.append(10*np.log10(tail/direct_energy) if tail > 0 and direct_energy > 0 else np.nan)
        noises.append(noise*(stop-start)/raw if raw > 0 else np.nan)
        directs.append(direct_energy)
        tails.append(tail)
    return {"tail_to_direct_db": np.asarray(ratios), "tail_noise_fraction": np.asarray(noises),
            "direct_energy": np.asarray(directs), "tail_energy": np.asarray(tails)}


def _peak_match(left, right, tolerance_s):
    """Maximum-cardinality, then minimum-total-error chronological matching."""
    # Dynamic programming avoids measured-first greedy matching's order bias.
    left, right = np.sort(left), np.sort(right)
    table = [[(0, 0., ()) for _ in range(len(right)+1)] for _ in range(len(left)+1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            choices = [table[i-1][j], table[i][j-1]]
            error = abs(float(a-b))
            if error <= tolerance_s:
                n, cost, errors = table[i-1][j-1]
                choices.append((n+1, cost+error, errors+(error,)))
            table[i][j] = min(choices, key=lambda v: (-v[0], v[1]))
    count, cost, errors = table[-1][-1]
    unmatched = len(left)+len(right)-2*count
    loss = (cost+tolerance_s*unmatched)/(count+unmatched) if count+unmatched else np.nan
    return count, errors, loss


def early_peak_diagnostic(measured_rir, simulated_rir, *, fs: int, measured_onset: int,
                          simulated_onset: int, cutoff_s: float, tolerance_s: float,
                          null_draws: int, seed: int, min_spacing_s: float = .001,
                          threshold_db: float = -30.) -> dict:
    """Pre-crossfade candidates and a same-density random-time null.

    The caller supplies mixing minus half the crossfade minus direct delay as
    the cutoff relative to each onset. Delay < min_spacing is excluded on both
    sides, rather than assuming the first detected candidate is the direct peak.
    Null sets have the simulated candidate count and the same minimum spacing.
    Empty candidate sets carry no timing evidence (NaN loss).
    """
    if (not np.isfinite(cutoff_s) or cutoff_s <= min_spacing_s or tolerance_s <= 0
            or null_draws < 1):
        raise ValueError("invalid early window or null parameters")
    def pick(x, onset):
        # Crop before picking: an out-of-window peak must not set the floor.
        stop = onset + int(np.ceil(cutoff_s * fs))
        rows = early_reflections(np.asarray(x)[:stop], fs, onset_sample=onset, max_time_s=cutoff_s,
                                 min_spacing_s=min_spacing_s, threshold_db=threshold_db)
        return [p.excess_delay_s for p in rows if min_spacing_s <= p.excess_delay_s < cutoff_s]
    left, right = pick(measured_rir, measured_onset), pick(simulated_rir, simulated_onset)
    count, errors, loss = _peak_match(left, right, tolerance_s)
    rng = np.random.default_rng(seed)
    n = len(right)
    free = cutoff_s-min_spacing_s-max(0, n-1)*min_spacing_s
    if free < 0:
        raise ValueError("peak density exceeds null window capacity")
    null = []
    for _ in range(null_draws):
        times = (np.sort(rng.uniform(0, free, n)) + min_spacing_s
                 + np.arange(n)*min_spacing_s)
        null.append(_peak_match(left, times, tolerance_s)[2])
    return {"cutoff_s": cutoff_s, "measured_count": len(left), "simulated_count": n,
            "matched_count": count, "unmatched_measured": len(left)-count,
            "unmatched_simulated": n-count, "absolute_errors_s": list(errors),
            "matching_loss_s": loss, "null_matching_loss_s": null, "null_peak_count": n}
