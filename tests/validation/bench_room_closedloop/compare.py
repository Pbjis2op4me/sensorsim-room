"""Truth/prediction comparison. Production ISO implementation is reused verbatim."""
from __future__ import annotations

import numpy as np
from scipy import signal

from sensorsim.acoustics.room import metrics
from sensorsim.acoustics.room.model import OCTAVE_HZ, SURFACES
from sensorsim.acoustics.room.rir import detect_onset
from .scorecard import row

BANDS = (500, 1000, 2000, 4000)
ISO = {"edt": (5., "%"), "t20": (5., "%"), "t30": (5., "%"),
       "c50_db": (1., "dB"), "c80_db": (1., "dB"), "d50": (.05, "1"), "drr_db": (3., "dB")}


def context(scenario, engine, kind, source, mic, band=None):
    return dict(scenario=scenario.id, tier=engine.tier, engine=engine.name,
                impairment=scenario.impairments, eval_kind=kind, src=source, mic=mic, band_hz=band)


def compare_rirs(truth, reconstructed, fs, ctx, *, notes="", legacy_drr=False):
    arrival = detect_onset(truth, noise_power=0)
    predicted_arrival = None if reconstructed is None else detect_onset(reconstructed, noise_power=0)
    rows = [row({**ctx, "band_hz": None}, metric="arrival_ms", truth=arrival/fs*1000,
                reconstructed=None if predicted_arrival is None else predicted_arrival/fs*1000,
                unit="ms", gate=.2, notes=notes)]
    for band in BANDS:
        a = metrics.octave_filter(truth, fs, band)
        b = None if reconstructed is None else metrics.octave_filter(reconstructed, fs, band)
        target = metrics.room_metrics(truth, fs, band=band, onset_sample=arrival,
                                      noise_power=0, legacy_drr=legacy_drr)
        prediction = None if b is None else metrics.room_metrics(reconstructed, fs, band=band,
            onset_sample=predicted_arrival, noise_power=0, legacy_drr=legacy_drr)
        band_ctx = {**ctx, "band_hz": band}
        for name, (gate, unit) in ISO.items():
            x, y = getattr(target, name), None if prediction is None else getattr(prediction, name)
            detail = notes
            if name == "drr_db":
                detail = "; ".join(s for s in (notes, target.drr_definition) if s)
            if name in ("edt", "t20", "t30"):
                detail = "; ".join(s for s in (notes, x.reason, None if y is None else y.reason) if s)
                x, y = x.rt60_s if x.valid else None, y.rt60_s if y is not None and y.valid else None
            # Infinite clarity is not a finite usable ISO comparison.
            x = x if x is not None and np.isfinite(x) else None
            y = y if y is not None and np.isfinite(y) else None
            rows.append(row(band_ctx, metric=name, truth=x, reconstructed=y, unit=unit, gate=gate,
                            relative=unit == "%", notes=detail))
        energy = float(a @ a)
        level = 10*np.log10(energy) if energy > 0 else None
        prediction_level = None if b is None or b @ b <= 0 else float(10*np.log10(b @ b))
        rows.append(row(band_ctx, metric="band_level_db", truth=level, reconstructed=prediction_level,
                        unit="dB", gate=3., notes=notes))
    return rows


def compare_parameters(parameters, model, ctx, *, obstacles=(), notes=""):
    rows = []
    for band in BANDS:
        i = OCTAVE_HZ.index(band)
        for surface in SURFACES:
            a = parameters.get("absorption", {}).get(surface)
            rows.append(row({**ctx, "band_hz": band, "mic": surface}, metric="absorption",
                            truth=None if a is None else a[i],
                            reconstructed=None if model is None else model.absorption[surface][i],
                            unit="1", gate=.05, notes=notes))
        rt = parameters.get("t60_s")
        rows.append(row({**ctx, "band_hz": band}, metric="model_t60", truth=None if rt is None else rt[i],
                        reconstructed=None if model is None else model.late_t60()[i],
                        unit="%", gate=5., relative=True, notes=notes))
        boxes = {} if model is None else {b.name: b for b in model.obstacles}
        for name in obstacles:
            a = parameters.get("obstacles", {}).get(name)
            rows.append(row({**ctx, "band_hz": band, "mic": name}, metric="obstacle_absorption",
                            truth=None if a is None else a[i],
                            reconstructed=boxes[name].absorption[i] if name in boxes else None,
                            unit="1", gate=.05, notes=notes+"; supplied box geometry/material prior"))
    return rows


def compare_timing(expected, detected, ctx, *, notes=""):
    detected = detected or {}
    rows = [row(ctx, metric="sweep_detected", truth=1., reconstructed=float(bool(detected)),
                unit="1", gate=0., notes=notes)]
    for name, key, scale, gate, unit in (("sweep_start_ms", "start_s", 1000, .2, "ms"),
                                        ("sweep_end_ms", "end_s", 1000, .2, "ms"),
                                        ("clock_drift_ppm", "clock_ppm", 1, 20., "ppm")):
        x, y = expected.get(key), detected.get(key)
        rows.append(row(ctx, metric=name, truth=None if x is None else scale*x,
                        reconstructed=None if y is None else scale*y, unit=unit, gate=gate, notes=notes))
    return rows


def align_signals(truth, prediction, fs):
    """Bounded integer lag, no gain correction, no cropped-away tails."""
    truth = np.asarray(truth, dtype=float)
    if prediction is None:
        return truth, None, 0, None
    prediction = np.asarray(prediction, dtype=float)
    corr = signal.correlate(prediction, truth, mode="full", method="fft")
    center = len(truth)-1
    lo, hi = max(0, center-round(.5*fs)), min(len(corr), center+round(.5*fs)+1)
    lag = lo+int(np.argmax(np.abs(corr[lo:hi])))-center
    a = np.pad(truth, (max(0, lag), 0))
    b = np.pad(prediction, (max(0, -lag), 0))
    n = max(len(a), len(b))
    a, b = np.pad(a, (0, n-len(a))), np.pad(b, (0, n-len(b)))
    gain = float(a @ b/(a @ a)) if a @ a else None
    return a, b, lag, gain


def compare_sound(truth, prediction, fs, ctx, *, prefix="", notes=""):
    a, b, lag, gain = align_signals(truth, prediction, fs)
    detail = f"{notes}; lag_samples={lag}; scalar={gain}"  # reported, NOT applied for spectrum/level
    score = None if b is None else metrics.si_sdr(a, b)
    error = None if score is None else max(0., 15-score)
    rows = [row(ctx, metric=prefix+"si_sdr", truth=float("inf"), reconstructed=score,
                error=error, unit="dB", gate=15., lower=True, notes=detail)]
    distance = None if b is None else metrics.log_spectral_distance(a, b, fs)
    rows.append(row(ctx, metric=prefix+"lsd", truth=0., reconstructed=distance, gate=3., unit="dB", notes=detail))
    for band in BANDS:
        e = None if b is None else metrics.band_level_error(a, b, fs, band)
        rows.append(row({**ctx, "band_hz": band}, metric=prefix+"band_level_db", truth=0.,
                        reconstructed=e, gate=3., unit="dB", notes=notes))
    return rows


def _late_early(x, onset, fs):
    early = float(x[onset:onset+round(.05*fs)] @ x[onset:onset+round(.05*fs)])
    late = float(x[onset+round(.05*fs):] @ x[onset+round(.05*fs):])
    return float(10*np.log10(late/early)) if early > 0 and late > 0 else None


def compare_dry(truth, prediction, oracle, fs, ctx, *, notes=""):
    rows = compare_sound(truth, prediction, fs, ctx, prefix="dry_", notes=notes)
    a, b, lag, gain = align_signals(truth, prediction, fs)
    onset = int(np.flatnonzero(a)[0]) if np.any(a) else 0
    score = None if b is None else metrics.si_sdr(a, b)
    rows.append(row(ctx, metric="dry_direct_residual_db", truth=float("inf"), reconstructed=score,
                    error=None if score is None else max(0.,15-score), gate=15., lower=True,
                    unit="dB", notes=notes+"; projection proxy, not physical DRR"))
    rows.append(row(ctx, metric="dry_late_early_db", truth=_late_early(a, onset, fs),
                    reconstructed=None if b is None else _late_early(b, onset, fs),
                    gate=3., unit="dB", notes=notes))
    truth_fit = metrics.room_metrics(a, fs, onset_sample=onset, noise_power=0).edt
    pred_fit = None if b is None else metrics.room_metrics(b, fs, onset_sample=onset, noise_power=0).edt
    rows.append(row(ctx, metric="dry_edc_slope", truth=truth_fit.slope_db_s if truth_fit.valid else None,
                    reconstructed=pred_fit.slope_db_s if pred_fit is not None and pred_fit.valid else None,
                    gate=5., relative=True, unit="%", notes=notes+"; finite sound envelope, not blind room RT"))
    pre = None
    if b is not None and a @ a > 0:
        energy = float(b[:onset] @ b[:onset])
        pre = float(10*np.log10(energy/(a @ a))) if energy else -float("inf")
    rows.append(row(ctx, metric="dry_pre_echo_db", truth=-float("inf"), reconstructed=pre,
                    error=None if pre is None else max(0., pre+30.), gate=-30., upper=True,
                    unit="dB", notes=notes))
    x, o, _, _ = align_signals(truth, oracle, fs)
    oracle_score = None if o is None else metrics.si_sdr(x, o)
    rows.append(row(ctx, metric="oracle_si_sdr", truth=float("inf"), reconstructed=oracle_score,
                    error=None if oracle_score is None else max(0., 15-oracle_score), gate=15., lower=True,
                    unit="dB", notes=notes+"; conditional LTI true-RIR/noise/source-power Wiener reference"))
    gap = None if oracle_score is None or score is None else oracle_score-score
    rows.append(row(ctx, metric="oracle_gap_db", truth=0. if oracle_score is not None else None,
                    reconstructed=gap, error=gap, gate=3., upper=True, unit="dB", notes=notes))
    return rows
