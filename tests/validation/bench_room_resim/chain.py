"""Evaluator-only response reference; privileged data never enters inference."""
import numpy as np
from scipy import signal
from sensorsim.acoustics.room.session_sim import device_sos
from sensorsim.acoustics.room import metrics
from tests.validation.bench_room_closedloop.compare import align_signals

BANDS = (500, 1000, 2000, 4000)


def recorder_source(source, fs, impairments):
    """Case-B target at 1 m in the SAME recorder's relative amplitude units.

    Exactly the linear capture convention in session_sim.clean_capture:
    g M (x * h) = (g M x) * h, with the full room tail retained by convolution.
    No sweep speaker, REF spreading, timing, noise or PCM conversion here.
    The SOS output uses source-length support, matching source-based dry scoring;
    use recorder_render for exact convolution closure including the IIR tail.
    """
    return impairments.recording_gain * signal.sosfilt(
        device_sos(fs, impairments, speaker=False), np.asarray(source, dtype=float))


def recorder_render(source, rir, fs, impairments):
    """Noise-free emission-origin recorder truth; filter AFTER full convolution.

    These are evaluator-only known device settings. Never supply this reference
    to inference, and never apply the recorder twice to an extracted Case B take.
    """
    return recorder_source(signal.fftconvolve(source, rir), fs, impairments)


def score(truth, prediction, fs, *, decompose=False):
    """Frozen metrics plus per-band LSD; no gain fitting in spectral scores.

    New per-band helper is local to this job, flagged for metric consolidation.
    The common floor is determined over the full 125--8000 Hz truth spectrum.
    """
    from scipy.fft import rfft, rfftfreq
    a, b, lag, gain = align_signals(truth, prediction, fs)
    af, bf = np.abs(rfft(a)), np.abs(rfft(b))
    f = rfftfreq(len(a), 1/fs)
    headline = (f >= 125) & (f <= 8000)
    floor = max(af[headline].max()*1e-4, np.finfo(float).tiny)
    ratio = 20*np.log10(np.maximum(bf, floor)/np.maximum(af, floor))
    result = dict(si_sdr=metrics.si_sdr(a, b), lsd=metrics.log_spectral_distance(a, b, fs),
        lag_samples=lag, scalar=gain,
        band_level_db={str(c): metrics.band_level_error(a, b, fs, c) for c in BANDS},
        band_lsd={str(c): float(np.sqrt(np.mean(ratio[(f >= c/np.sqrt(2)) &
            (f < c*np.sqrt(2))]**2))) for c in BANDS})
    if decompose:
        # An algebraic partition of the SAME floored log spectrum, never a
        # waveform normalization or a gain passed back to inference.
        bias = float(np.mean(ratio[headline]))
        shape = float(np.sqrt(np.mean((ratio[headline]-bias)**2)))
        result['spectral_bias_db'] = bias
        result['spectral_shape_db'] = shape
        result['spectral_level_mse_fraction'] = bias*bias/(bias*bias+shape*shape) if bias or shape else 0.
    return result


def rir_components(model, source, receiver, fs, seed, duration):
    """Diagnostic decomposition of existing octave synthesis; no new model.

    Retain the production obstacle-adjusted direct path at index zero, subtract
    all other coherent early paths, and reuse the exact full response's tail.
    Private path helpers are used read-only here to avoid a new synthesis mode.
    """
    from dataclasses import fields, replace
    from sensorsim.acoustics.room.synth import (
        SynthConfig, synthesize, early_paths, _render_paths, crossfade_fraction)
    cfg = SynthConfig(obstacle_backend='cpu')
    source, receiver = np.asarray(source, dtype=float), np.asarray(receiver, dtype=float)
    full = synthesize(model, source, receiver, fs, seed=seed, config=cfg)
    paths, _, _, _ = early_paths(model, source, receiver, cfg)
    n = len(full.samples)
    early = _render_paths(model, paths, fs, n, cfg)
    direct_paths = replace(paths, **{f.name: getattr(paths, f.name)[:1] for f in fields(paths)})
    direct = _render_paths(model, direct_paths, fs, n, cfg)
    time = (np.arange(n)-cfg.filter_half_length)/fs
    early *= np.cos(np.pi/2*crossfade_fraction(time, model.mixing_time(), cfg.crossfade_s))
    def observed(x):
        x = np.asarray(x, dtype=float)[cfg.filter_half_length:]
        count = round(duration*fs)
        return np.pad(x[:count], (0, max(0, count-len(x))))
    h, early, direct = (observed(x) for x in (full.samples, early, direct))
    late = h-early
    return dict(full=h, early=early, direct=direct, late=late,
                no_early_reflections=late+direct)
