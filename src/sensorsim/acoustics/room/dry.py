"""Measured-RIR dry estimates; see README_DRY.md for identification limits.

No dependencies on the other room stages. Audio and RIRs are mono float arrays,
keyed by microphone id. RIR gain and time origin must match the real recording.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import platform
import os
import tempfile
from typing import Mapping

import numpy as np
import scipy
from scipy import fft, signal
from scipy.io import wavfile

from sensorsim.core.messages import ModelValidity
from sensorsim.core.fingerprint import sha256_file, source_revision
from sensorsim.core.serialization import atomic_json, load_yaml_unique, report_data, to_native
from sensorsim.acoustics.bands import BandGrid, BandSpec
from .source_reference import SourceReference
from .ingest import channel_calibration, load_manifest, read_take, read_recording_audio

BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
VALIDITY = ModelValidity(
    name="measured_rir_dry_extraction", status="partially validated",
    freq_range_hz=(125, 8000),
    conditions="Fixed linear source/mics; synchronized channels only. Numerical A/B checks; "
    "measured quality and physical-source directivity remain conditional. Not anechoic truth.",
    citation="Kodrasi et al. 2013, doi:10.1109/TASL.2013.2260743; README_DRY.md",
)
TIKHONOV_VALIDITY = ModelValidity(
    name='tikhonov_room_inverse', status='partially validated', freq_range_hz=(125, 8000),
    conditions='Fixed LTI matching measured response; octave-local ridge and noise loading. '
    'Numerical controls and bounded mono synthetic/measured-RIR convolution only; physical drift unqualified.',
    citation='Kodrasi et al. 2014 doi:10.1109/ICASSP.2014.6854590; parent-project note ROOM_DRY_INVERSE (not distributed)',
)


@dataclass(frozen=True)
class DryConfig:
    ridge: float = 1e-3
    max_gain_db: float = 20.0
    ref_weight: float = 4.0
    early_ms: float | None = None  # Full measured response by default.
    noise_suppression: bool = True
    mask_floor: float = 0.0
    late_suppression: bool = False
    mixing_ms: float = 50.0
    block_seconds: float = 8.0
    # Pydantic's PipelineConfig must retain legacy resolved-config bytes too.
    dereverb_mode: str = field(default="legacy", metadata={"exclude_if": lambda value: value == "legacy"})
    inverse_mode: str = field(default="legacy", metadata={"exclude_if": lambda value: value == "legacy"})

    def __post_init__(self):
        if self.inverse_mode not in ('legacy', 'tikhonov'):
            raise ValueError('inverse_mode must be legacy or tikhonov')
        if self.dereverb_mode not in ("legacy", "measured_residual"):
            raise ValueError("dereverb_mode must be legacy or measured_residual")
        if not (np.isfinite(self.ridge) and 0 < self.ridge <= 1):
            raise ValueError("ridge must be in (0, 1]")
        for name in ("max_gain_db", "ref_weight", "mixing_ms", "block_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.mask_floor <= 1:
            raise ValueError("mask_floor must be in [0, 1]")
        if self.early_ms is not None and (not np.isfinite(self.early_ms) or self.early_ms <= 0):
            raise ValueError("early_ms must be finite and positive")


@dataclass
class DryResult:
    samples: np.ndarray
    fs: int
    report: dict
    validity: ModelValidity = VALIDITY
    version: str = "room-dry/3"
    source_reference: SourceReference | None = None


def _array(value, name):
    x = np.asarray(value)
    if x.ndim != 1 or not len(x) or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"{name}: expected nonempty finite mono array")
    if np.iscomplexobj(x) or not all(np.isfinite(x[i:i+65536]).all() for i in range(0, len(x), 65536)):
        raise ValueError(f"{name}: expected nonempty finite mono array")
    return x


def _psd(x, fs, frequencies, fir=None):
    size = min(len(x), max(256, round(fs * .064)))
    hop = size - size//2
    total, windows = None, 0
    # Accumulate Welch frames in bounded batches, including every valid frame once.
    for start in range(0, max(1, len(x)-size+1), 64*hop):
        end = min(len(x), start+63*hop+size)
        if fir is None:
            chunk = x[start:end]
        else:
            guard = len(fir)//2+1
            left, right = max(0, start-guard), min(len(x), end+guard)
            chunk = _filter(x[left:right], fir)[start-left:end-left]
        if len(chunk) < size:
            break
        f, p = signal.welch(chunk, fs=fs, nperseg=size, detrend=False)
        weight = 1+(len(chunk)-size)//hop
        total = p*weight if total is None else total+p*weight
        windows += weight
    return np.interp(frequencies, f, total/windows)


def _db(value):
    """Dimensionless power ratio, reference 1; floor 1e-30 (-300 dB)."""
    return 10 * np.log10(np.maximum(value, 1e-30))


def _band_average(values, frequencies, *, interpolate=True):
    """Octave-local means, optionally interpolated on log frequency.

    Include sub-125 Hz bands for conditioning, even though they are outside
    the reported validity range. Do not borrow their power for the 125 Hz band.
    """
    centers = np.array([b for b in (15.625, 31.25, 62.5, *BANDS)
                        if b/np.sqrt(2) < frequencies[-1]])
    means, supported_centers = [], []
    stepped = np.empty_like(values)
    for i, center in enumerate(centers):
        lo = 0 if i == 0 else center/np.sqrt(2)
        hi = np.inf if i == len(centers)-1 else center*np.sqrt(2)
        band = (frequencies >= lo) & (frequencies < hi)
        if not band.any():
            continue
        mean = np.mean(values[..., band], axis=-1)
        means.append(mean)
        supported_centers.append(center)
        stepped[..., band] = np.expand_dims(mean, -1)
    if not interpolate:
        return stepped
    centers = np.array(supported_centers)
    means = np.array(means).T
    log_f = np.log2(np.maximum(frequencies, centers[0]))
    if values.ndim == 1:
        return np.interp(log_f, np.log2(centers), means)
    return np.array([np.interp(log_f, np.log2(centers), row) for row in means])


def _direct(h, fs, meta):
    if 'direct_start_sample' not in meta and 'first_reflection_delay_s' in meta:
        from .rir import reference_direct_window
        channel = meta.get('measurement_channel', 0)
        peak = meta.get('direct_samples', [int(np.argmax(abs(h)))])[channel]
        gate, start, stop, low_hz = reference_direct_window(
            h, fs, peak, first_reflection_delay_s=meta['first_reflection_delay_s'][channel])
        d = np.zeros_like(h)
        d[start:stop] = gate
        return d, start, stop, True, low_hz
    peak = int(np.argmax(np.abs(h)))
    start = meta.get("direct_start_sample", max(0, peak - round(.001 * fs)))
    stop = meta.get("direct_end_sample", min(len(h), peak + max(1, round(.0025 * fs))))
    if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
        raise ValueError("direct window sample indices must be integers")
    if not 0 <= start < stop <= len(h) or np.sum(h[start:stop] ** 2) == 0:
        raise ValueError("invalid or zero-energy direct window")
    d = np.zeros_like(h)
    d[start:stop] = h[start:stop]
    return d, start, stop, "direct_start_sample" in meta and "direct_end_sample" in meta, None


def clock_corrected_length(length, meta):
    """Output count on the sweep clock; sweep and real take share a stable device."""
    ppm = meta.get('clock_drift_ppm', 0.)
    if not np.isfinite(ppm) or abs(ppm) > 300:
        raise ValueError('real-take clock correction requires finite drift within +/-300 ppm')
    if abs(ppm) > .5 and not meta.get('clock_resampled', False):
        raise ValueError('sweep RIR must be clock corrected before correcting real takes')
    return max(1, round(length/(1+ppm*1e-6))) if abs(ppm) > .5 else length


def _clock_normalize(x, meta, *, output=None):
    """Constant-rate, bandlimited interpolation; no recording-start alignment.

    A 64-tap Kaiser-windowed sinc uses global sample coordinates in bounded
    blocks, with zero extension (no periodic endpoint wrap). Long outputs use
    scratch memmaps; callers can also supply a buffer for shorter recordings.
    """
    size = clock_corrected_length(len(x), meta)
    ppm = meta.get('clock_drift_ppm', 0.)
    if abs(ppm) <= .5:
        return x
    if output is None and size <= 2_000_000:
        out = np.empty(size)
    elif output is None:
        # TMPDIR is the job/output scratch directory; no whole-take RAM copy.
        with tempfile.TemporaryFile(prefix='dry-clock-') as scratch:
            out = np.memmap(scratch, mode='w+', dtype=np.float32, shape=(size,))
    else:
        out = output
    if out.shape != (size,) or out.dtype.kind != 'f' or not out.flags.writeable:
        raise ValueError('clock output buffer has wrong shape or dtype')
    ratio = 1+ppm*1e-6
    offsets = np.arange(-31, 33)
    cutoff = min(1., 1/ratio)
    for start in range(0, size, 4096):
        positions = np.arange(start, min(size, start+4096))*ratio
        indices = np.floor(positions).astype(np.int64)[:, None]+offsets
        delta = positions[:, None]-indices
        weights = cutoff*np.sinc(cutoff*delta)*np.i0(8.6*np.sqrt(np.maximum(0, 1-(delta/32)**2)))/np.i0(8.6)
        weights /= weights.sum(axis=1, keepdims=True)
        values = x[np.clip(indices, 0, len(x)-1)]
        weights *= (indices >= 0) & (indices < len(x))
        out[start:start+len(positions)] = np.sum(values*weights, axis=1)
    return out


def _measurement_band(metadata, needed, frequencies):
    bands = []
    for k in needed:
        meta = metadata.get(k, {})
        excitation = meta.get('excitation', {})
        kit = meta.get('kit', {})
        if excitation.get('type') == 'ess':
            bands.append((excitation['f_start_hz'], excitation['f_end_hz']))
        elif kit:
            bands.append((kit['f1_hz'], kit['f2_hz']))
    if not bands:
        return np.ones_like(frequencies), None
    lo, hi = max(b[0] for b in bands), min(b[1] for b in bands)
    if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
        raise ValueError('incompatible sweep measurement bands')
    # Smooth transitions remain inside the declared excitation band. Never
    # invert the spectral leakage of a finite cropped ESS impulse above f2.
    ramp = np.clip((frequencies-lo)/lo, 0, 1)*np.clip((hi-frequencies)/(.05*hi), 0, 1)
    return .5-.5*np.cos(np.pi*ramp), [float(lo), float(hi)]


def diagnostic_policy() -> dict:
    """Named quick decay/PSD policy; never a qualified measured-room estimator."""
    grid = BandGrid(tuple(BandSpec("base2", 1, float(b), float(b)) for b in BANDS),
                    "room-dry-diagnostic/1: causal Butterworth", "pressure", 3)
    return dict(method="room-dry-diagnostic/1", band_grid=grid.as_dict(),
                band_standard="diagnostic binary nominal octaves",
                input_quantity="pressure", decay_fit_db=[-5, -25], extrapolation_db=60,
                noise_correction="none", onset="uncropped RIR", settling_samples=0,
                latency="causal filter delay retained", psd="Welch; binary nominal bands",
                db_policy="power ratio re 1; floor 1e-30 (-300 dB)")


def _t60(h, fs, center):
    """Diagnostic -5..-25 dB decay fit, NOT an ISO/noise-corrected T20."""
    lo, hi = center / np.sqrt(2), center * np.sqrt(2)
    if hi >= .49 * fs or len(h) < .05 * fs:
        return None
    sos = signal.butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos")
    energy = signal.sosfilt(sos, h) ** 2
    decay = np.cumsum(energy[::-1])[::-1]
    if decay[0] <= 1e-30:
        return None
    db = _db(decay / decay[0])
    idx = np.flatnonzero((db < -5) & (db > -25))
    if len(idx) < .02 * fs:
        return None
    t = idx / fs
    slope, intercept = np.polyfit(t, db[idx], 1)
    residual = np.sum((db[idx] - (slope*t+intercept)) ** 2)
    total = np.sum((db[idx] - np.mean(db[idx])) ** 2)
    if slope >= 0 or total == 0 or 1 - residual / total < .9:
        return None
    return float(-60 / slope)


def _filter(x, fir):
    # Centered, noncausal FIR; linear convolution, never circular wraparound.
    start = len(fir) // 2
    return signal.oaconvolve(x, fir)[start:start+len(x)]


def _postfilter(x, noise_psd, frequencies, fs, config, tail_bands):
    if not config.late_suppression:
        return x
    nwin = max(32, round(.032 * fs))
    nwin += nwin % 2
    nwin = min(nwin, len(x) // 2 * 2)
    if nwin < 4:
        return x
    hop = nwin // 2
    f, _, z = signal.stft(x, fs=fs, nperseg=nwin, noverlap=nwin-hop,
                          boundary="zeros", padded=True)
    power = np.abs(z) ** 2
    # Welch one-sided density -> scipy STFT squared spectrum (Hann ENBW).
    enbw = fs * 1.5 / nwin
    n = np.interp(f, frequencies, noise_psd)[:, None] * enbw / 2
    n[0] *= 2
    n[-1] *= 2
    unwanted = np.zeros_like(power)
    if config.late_suppression:
        delay = max(1, round(config.mixing_ms * .001 * fs / hop))
        for center, t60, late_ratio in tail_bands:
            if t60 is None:
                continue
            mask = (f >= center/np.sqrt(2)) & (f < center*np.sqrt(2))
            alpha = np.exp(-6*np.log(10)*hop/fs/t60)
            # Diffuse power recursion; clipped to a conservative residual budget.
            history = np.maximum(power[mask] - n[mask], 0)
            smooth = signal.lfilter([1-alpha], [1, -alpha], history, axis=1)
            if delay < power.shape[1]:
                unwanted[mask, delay:] += min(late_ratio, 1.) * smooth[:, :-delay]
    # Subtract in power, then take the amplitude square root. A Wiener
    # amplitude mask squares the wanted-power fraction and colours weak bands.
    from scipy.ndimage import uniform_filter
    smooth_power = uniform_filter(power, size=(3, 3), mode="nearest")
    smooth_unwanted = uniform_filter(unwanted, size=(3, 3), mode="nearest")
    gain = np.sqrt(np.maximum(config.mask_floor**2,
                             1 - smooth_unwanted / np.maximum(smooth_power, 1e-30)))
    _, y = signal.istft(z * gain, fs=fs, nperseg=nwin, noverlap=nwin-hop, boundary=True)
    return y[:len(x)]


def extract_dry(recordings: Mapping[str, np.ndarray], rirs: Mapping[str, np.ndarray],
                noise_floor: Mapping[str, np.ndarray] | np.ndarray, *, fs: int,
                case: str, reference: str = "REF", synchronized: bool = False,
                metadata: Mapping[str, dict] | None = None,
                config: DryConfig | None = None,
                output_buffer: np.ndarray | None = None) -> DryResult:
    """Return one dry estimate and octave-band diagnostics.

    A targets played digital audio. B targets the physical source as heard at REF,
    retaining mic coloration and reference-distance gain. For separate performances,
    use one recording per call; all sweep RIRs including REF may still be supplied.
    Sweep metadata can normalize a constant device clock; take start offsets
    and gains are preserved. Sequential takes must use the same stable device.
    """
    config = config or DryConfig()
    measured_tail = config.dereverb_mode == 'measured_residual'
    if not isinstance(fs, (int, np.integer)) or fs < 2000:
        raise ValueError("fs must be an integer >=2000 Hz")
    if case not in ("A", "B"):
        raise ValueError("case must be A (same speaker) or B (physical source)")
    if not recordings:
        raise ValueError("no recordings")
    names = sorted(recordings)
    if len(names) > 1 and not synchronized:
        raise ValueError("joint filtering requires synchronized recordings of the SAME event")
    y = {k: _array(recordings[k], k) for k in names}
    if len({len(a) for a in y.values()}) != 1:
        raise ValueError("synchronized recordings must have equal lengths")
    metadata = metadata or {}
    surrogate = case == 'B' and reference not in rirs
    if surrogate and measured_tail:
        # One common measured target across independent performances. No actual
        # REF recording is needed; its gain/colouration must not be fabricated.
        choices = []
        for k in sorted(rirs):
            response = _array(rirs[k], f'RIR {k}')
            d, _, _, _, _ = _direct(response, fs, metadata.get(k, {}))
            ratio = np.sum(d*d)/max(float(np.sum((response-d)**2)), 1e-30)
            choices.append((float(ratio), k))
        if not choices:
            raise ValueError('missing matching RIR')
        reference = max(choices)[1]
    needed = set(names) | ({reference} if case == "B" else set())
    if not needed <= rirs.keys():
        raise ValueError("missing matching RIR or REF RIR")
    h = {k: _array(rirs[k], f"RIR {k}") for k in sorted(needed)}
    if any(not np.any(a) for a in h.values()):
        raise ValueError("zero RIR")
    source_ids = {metadata[k]["source_id"] for k in (rirs if surrogate and measured_tail else needed)
                  if "source_id" in metadata.get(k, {})}
    if len(source_ids) > 1:
        raise ValueError("RIRs belong to different source positions")
    noise = {k: _array(noise_floor[k] if isinstance(noise_floor, Mapping) else noise_floor,
                       f"noise {k}") for k in names}
    clock_report = {}
    for k in names:
        meta = metadata.get(k, {})
        before = len(y[k])
        y[k] = _clock_normalize(y[k], meta)
        noise[k] = _clock_normalize(noise[k], meta)
        clock_report[k] = dict(estimated_ppm=meta.get('clock_drift_ppm'),
                               applied=abs(meta.get('clock_drift_ppm', 0.)) > .5,
                               input_samples=before, output_samples=len(y[k]))
    if len({len(a) for a in y.values()}) != 1:
        raise ValueError('clock-corrected synchronized channels must share sample count and time origin')
    count = len(y[names[0]])
    direct = {k: _direct(h[k], fs, metadata.get(k, {})) for k in h}
    nfft = fft.next_fast_len(max(2048, 4*max(map(len, h.values()))))
    nfft += nfft % 2
    f = fft.rfftfreq(nfft, 1/fs)
    model = {}
    for k in names:
        hh = h[k].copy()
        if config.early_ms is not None:
            end = min(len(hh), direct[k][1] + round(config.early_ms*.001*fs))
            fade = min(round(.005*fs), end-direct[k][2])
            if fade > 0:
                hh[end-fade:end] *= np.linspace(1, 0, fade)
            hh[end:] = 0
        model[k] = fft.rfft(hh, nfft)
    H = np.array([model[k] for k in names])
    N = np.array([_psd(noise[k], fs, f) for k in names])
    P = np.array([_psd(y[k], fs, f) for k in names])
    # Independent-channel noise approximation. A shared floor is a PSD proxy only.
    w = 1 / np.maximum(N, np.maximum(N.max(axis=0)*1e-6, 1e-30))
    direct_power = np.array([abs(fft.rfft(direct[k][0], nfft))**2 for k in names])
    reverb_power = np.array([abs(fft.rfft(h[k]-direct[k][0], nfft))**2 for k in names])
    band_drr = _band_average(direct_power, f) / np.maximum(_band_average(reverb_power, f), 1e-30)
    reliability = np.clip(band_drr, .1, 10)
    for i, k in enumerate(names):
        reliability[i] *= config.ref_weight if k == reference else 1
    w *= reliability
    w /= w.sum(axis=0)
    energy = np.sum(w * abs(H)**2, axis=0)
    local_energy = _band_average(energy, f)
    ridge = config.ridge*local_energy
    observed = _band_average(np.sum(w*P, axis=0), f)
    noise_mix = _band_average(np.sum(w*N, axis=0), f)
    noise_fraction = noise_mix/np.maximum(observed-noise_mix, 1e-30)
    # Bias-correct the ridge inverse where the response is locally supported.
    # Loading remains at shared notches, with the joint gain cap applied below.
    denominator = np.maximum(energy, np.maximum(ridge, 1e-30))
    # A whole weak octave can lie outside the sweep's usable bandwidth. Bias
    # correction cannot recover a common notch whose recording is also noisy.
    # Keep the original robust ridge there; report that loss explicitly below.
    unsupported = (energy < config.ridge*float(energy.max())) & (noise_fraction > 10**(-6/10))
    ridge[unsupported] = np.maximum(config.ridge*float(energy.max()),
                                   energy[unsupported]*np.minimum(noise_fraction[unsupported], 100))
    denominator[unsupported] = energy[unsupported]+ridge[unsupported]
    if config.inverse_mode == 'tikhonov':
        ridge = config.ridge*local_energy + local_energy*np.minimum(noise_fraction, 100)
        denominator = np.maximum(energy+ridge, 1e-30)
    target = np.ones_like(H[0])
    if case == "B":
        d, start, stop, _, _ = direct[reference]
        target = fft.rfft(d[start:stop], nfft)
    G = target[None, :] * w * H.conj() / denominator
    gain = np.sqrt(np.sum(abs(G)**2, axis=0))
    capped = gain > 10**(config.max_gain_db/20)
    G *= np.minimum(1, 10**(config.max_gain_db/20)/np.maximum(gain, 1e-30))
    support, measurement_band = _measurement_band(metadata, needed, f)
    G *= support
    passthrough = case == 'B' and reference in names and measurement_band is not None
    if passthrough:
        # Only the actual REF recording identifies this waveform outside the
        # sweep band. Preserve it with the target's same time origin; explicitly
        # retain its reverberation instead of extrapolating an unmeasured inverse.
        G[names.index(reference)] += (1-support)*np.exp(2j*np.pi*f*direct[reference][1]/fs)
    inverse_gain_db = float(20*np.log10(max(np.sqrt(np.sum(abs(G)**2, axis=0)).max(), 1e-30)))
    shrink = energy/denominator * np.minimum(1, 10**(config.max_gain_db/20)/np.maximum(gain, 1e-30))
    shrink *= support  # passthrough outside support is not identified dry transfer
    firs = [fft.fftshift(fft.irfft(g, nfft)) for g in G]
    output_noise = np.sum(abs(G)**2*N, axis=0)
    warnings = ["Approximate estimate, not independently measured anechoic sound.",
                "RIR origins/gains must match recordings; AGC is not corrected.",
                "Sweep clock applies only to the same stable device clock; separate take start offsets are preserved."]
    if any(v['estimated_ppm'] is None for v in clock_report.values()):
        warnings.append('No sweep clock estimate for some channels; their clock drift is not corrected.')
    if measurement_band is not None:
        warnings.append('Outside the sweep band: REF passthrough retains reverberation; far-only/A output is suppressed. No dry waveform qualification there.')
    if case == "B":
        warnings.append("Physical-source/speaker directivity mismatch is unidentifiable; mic coloration retained.")
    if isinstance(noise_floor, Mapping):
        warnings.append("Noise PSDs assume independent channels; correlated noise may be underestimated.")
    else:
        warnings.append("One noise-floor recording is reused as a stationary PSD proxy at every mic.")
    if any(not direct[k][3] for k in h):
        warnings.append("Direct windows inferred around largest peaks; inspect for earlier direct arrival/obstruction.")
    rows, tail_bands = [], []
    for center in BANDS:
        if center*np.sqrt(2) >= fs/2:
            continue
        band = (f >= center/np.sqrt(2)) & (f < center*np.sqrt(2))
        snr, drrs, t60s, channel_reports = [], [], [], {}
        for i, k in enumerate(names):
            sn = float(_db(max(float(np.sum(P[i, band]-N[i, band])), 0) /
                           max(float(N[i, band].sum()), 1e-30)))
            D = fft.rfft(direct[k][0], nfft)
            R = fft.rfft(h[k]-direct[k][0], nfft)
            dr = float(_db(np.sum(abs(D[band])**2)/max(float(np.sum(abs(R[band])**2)), 1e-30)))
            t60 = metadata.get(k, {}).get("t60_s", {}).get(str(center))
            if t60 is not None and (not np.isfinite(t60) or t60 <= 0):
                raise ValueError("T60 values must be finite and positive")
            if t60 is None:
                t60 = _t60(h[k], fs, center)
            channel_reports[k] = dict(snr_db=sn, drr_db=dr, t60_s=t60,
                                      weight=float(w[i, band].mean()))
            snr.append(sn); drrs.append(dr)
            if t60 is not None:
                t60s.append(t60)
        reasons = []
        if max(snr) < 6:
            reasons.append("low_snr")
        notch = (energy[band] <= np.maximum(local_energy[band]*1e-3, 1e-30)) | unsupported[band]
        if np.any(notch):
            reasons.append("spectral_notch")
        if shrink[band].mean() < .8:
            reasons.append("inverse_transfer_loss")
        if capped[band].any():
            reasons.append("gain_cap")
        if center < 250:
            reasons.append("short_direct_window_low_frequency_uncertainty")
        if not all(direct[k][3] for k in h):
            reasons.append("unverified_direct_window")
        if case == "B":
            reasons.append("source_directivity_unverified")
            if direct[reference][4] is not None and center/np.sqrt(2) < direct[reference][4]:
                reasons.append('below_ref_window_frequency_limit')
        if np.any(support[band] < 1):
            reasons.append('outside_sweep_supported_band')
        rows.append(dict(center_hz=center, snr_db=max(snr), drr_db=max(drrs),
                         channels=channel_reports, regularization=float(np.median(ridge[band])),
                         retained_transfer=float(shrink[band].mean()),
                         unsupported_transfer_fraction=float(unsupported[band].mean()),
                         notch_fraction=float(notch.mean()), reasons=reasons,
                         trustworthy=not reasons))
        # Conservative measured tail fraction. Full inverse already addresses reverb;
        # postfilter is experimental/opt-in and can double-suppress.
        ratio = 10**(-max(drrs)/10)
        tail_bands.append((center, float(np.median(t60s)) if t60s else None, ratio/(1+ratio)))
    if config.late_suppression or measured_tail:
        warnings.append("Late suppression is a diffuse-tail approximation; may suppress sustained source energy.")
    # Bounded processing workspace. Overlap includes FIR support and postfilter decay.
    block = max(1, round(config.block_seconds*fs))
    tail_guard = max([v[1] or 0 for v in tail_bands], default=0) if config.late_suppression or measured_tail else 0
    guard = nfft//2 + round((.1+tail_guard)*fs)
    out = np.empty(count) if output_buffer is None else output_buffer
    if out.shape != (count,) or out.dtype.kind != "f" or not out.flags.writeable:
        raise ValueError("output_buffer must be writable floating array with recording length")
    # Every input that is read again after the first output write (cr-room-dsp-02):
    # aliasing the noise floor silently zeroed the default power-subtraction result.
    for what, arrays in (("recordings", y), ("noise floors", noise), ("RIRs", h)):
        if any(np.shares_memory(out, a) for a in arrays.values()):
            raise ValueError(f"output_buffer must not alias {what}")
    for begin in range(0, count, block):
        end = min(count, begin+block)
        left, right = max(0, begin-guard), min(count, end+guard)
        # Keep the postfilter STFT grid fixed across processing blocks.
        hop = max(32, round(.032*fs))
        hop = (hop+hop%2)//2
        left = left//hop*hop
        z = sum(_filter(y[k][left:right], fir) for k, fir in zip(names, firs))
        out[begin:end] = z[begin-left:end-left]
    if config.noise_suppression:
        # Average signed excess power before clipping: short-frame rectification
        # leaves a positive noise bias even when the true source band is absent.
        output_psd = _psd(out, fs, f)
        power = _band_average(output_psd, f, interpolate=False)
        # Estimate signal power from channel autospectra before combining them.
        # Random cross-channel noise terms in the sum can otherwise dominate a
        # weak band. Filter silence on the same finite Welch grid as the takes.
        excess = np.zeros_like(f)
        filtered_noise = np.zeros_like(f)
        for i, (k, fir) in enumerate(zip(names, firs)):
            channel_noise = _psd(noise[k], fs, f, fir)
            filtered_noise += channel_noise
            excess += (_psd(y[k], fs, f, fir) - channel_noise)/reliability[i]
        fractions = w*abs(H)**2/np.maximum(energy, 1e-30)
        # Power moments would otherwise square the DRR/REF prior implicit in G.
        # Use that prior once, with inverse-noise-variance power weighting.
        diagonal = np.sum(fractions**2/reliability, axis=0)
        wanted = _band_average(excess, f, interpolate=False) / np.maximum(
            _band_average(diagonal, f, interpolate=False), 1e-30)
        noise_power = _band_average(filtered_noise, f, interpolate=False)
        wanted = np.where(noise_power == 0, power, wanted)
        power_gain = np.clip(wanted/np.maximum(power, 1e-30), config.mask_floor**2, 1)
        G *= np.sqrt(power_gain)
        firs = [fft.fftshift(fft.irfft(g, nfft)) for g in G]
        output_noise *= power_gain
    else:
        power_gain = np.ones_like(f)
    residual = None
    if measured_tail:
        from .dereverb import residual_model, suppress_late
        desired = np.ones(1) if case == 'A' else direct[reference][0][direct[reference][1]:direct[reference][2]]
        residual = residual_model(firs, [h[k] for k in names], desired, fs, rows, config.mixing_ms)
    if config.noise_suppression or config.late_suppression or measured_tail:
        for begin in range(0, count, block):
            end = min(count, begin+block)
            left, right = max(0, begin-guard), min(count, end+guard)
            hop = max(32, round(.032*fs))
            hop = (hop+hop%2)//2
            left = left//hop*hop
            z = sum(_filter(y[k][left:right], fir) for k, fir in zip(names, firs))
            z = (suppress_late(z, output_noise, f, fs, residual, mixing_ms=config.mixing_ms)
                 if measured_tail else _postfilter(z, output_noise, f, fs, config, tail_bands))
            out[begin:end] = z[begin-left:end-left]
    for row in rows:
        band = (f >= row['center_hz']/np.sqrt(2)) & (f < row['center_hz']*np.sqrt(2))
        row['noise_power_gain'] = float(power_gain[band].mean())
        missing_power = bool(config.noise_suppression and np.any(wanted[band] <= 0))
        if missing_power:
            row['reasons'].append('no_resolved_signal_power')
        row['recovery_status'] = ('unresolved' if missing_power or row['retained_transfer'] < .2
                                  else 'noise_limited' if 'low_snr' in row['reasons']
                                  else 'conditioned')
        row['trustworthy'] = not row['reasons']
    report_config = asdict(config)
    if config.inverse_mode == 'legacy':
        report_config.pop('inverse_mode')
    if not measured_tail:
        report_config.pop('dereverb_mode')  # retain the legacy report format
    target_meta = metadata.get(reference if case == "B" else names[0], {})
    source_reference = SourceReference(
        target="ref_microphone" if case == "B" else "played_digital",
        units=target_meta.get("recording_units", "relative") if case == "B" else "peak_fs",
        reference_distance_m=target_meta.get("reference_distance_m") if case == "B" else None,
        calibration=target_meta.get("audio_calibration") if case == "B" else None)
    report = dict(source_reference=source_reference.as_dict(), metric_policy=diagnostic_policy(),
                  output_level_reference=dict(statistic="mean_square", reference_value=1.,
                      units="Pa^2" if source_reference.units == "Pa" else "relative_amplitude^2",
                      zero_policy="floor 1e-30 (-300 dB)"),
                  schema_version="room-dry/3", case=case, fs=fs, channels=names,
                  synchronized=synchronized, reference=reference,
                  target="played digital waveform" if case == "A" else
                  "physical-source radiated sound at REF, including mic response and REF-distance gain",
                  config=report_config, nfft=nfft, bands=rows, warnings=warnings,
                  clock_correction=clock_report,
                  measurement_band=dict(excitation_hz=measurement_band,
                                        outside_action='REF passthrough; reverberation retained' if passthrough else
                                        'suppressed' if measurement_band else 'unknown; legacy conditioning'),
                  direct_windows={k: dict(start=direct[k][1], end=direct[k][2],
                                         annotated=direct[k][3], low_frequency_hz=direct[k][4]) for k in h},
                  max_inverse_noise_gain_db=inverse_gain_db,
                  unresolved_bands_hz=[r['center_hz'] for r in rows if r['recovery_status'] == 'unresolved'],
                  untrustworthy_bands_hz=[r["center_hz"] for r in rows if not r["trustworthy"]],
                  validity=asdict(VALIDITY))
    from sensorsim.licensing import lineage_metadata
    lineage = lineage_metadata(metadata)
    if lineage:
        report['lineage'] = lineage
    if config.inverse_mode == 'tikhonov':
        from .dereverb import residual_model
        desired = np.ones(1) if case == 'A' else direct[reference][0][direct[reference][1]:direct[reference][2]]
        report['inverse_design'] = dict(mode='tikhonov', effective_response=residual_model(
            firs, [h[k] for k in names], desired, fs, rows, config.mixing_ms))
        report['validity'] = asdict(TIKHONOV_VALIDITY)
    if measured_tail:
        from .dereverb import activity_interval, VALIDITY as DEREVERB_VALIDITY
        activity = activity_interval(out, np.trapezoid(output_noise, f), fs)
        psd = (_psd(out[activity['start_sample']:activity['end_sample']], fs, f)
               if activity['status'] == 'detected' else None)
        for row, model_band in zip(rows, residual['bands']):
            band = (f >= row['center_hz']/np.sqrt(2)) & (f < row['center_hz']*np.sqrt(2))
            row['residual_reverb'] = model_band
            row['output_level_db'] = float(_db(np.trapezoid(psd[band], f[band]))) if psd is not None else None
            # A diagnostic weight, not a statistical coverage probability.
            snr = 10**(np.clip(row['snr_db'], -100, 100)/10)
            weight = snr/(1+snr)*row['retained_transfer']*(1-model_band['late_fraction'])
            if psd is None or row['recovery_status'] == 'unresolved' or any(
                    v in row['reasons'] for v in ('below_ref_window_frequency_limit', 'outside_sweep_supported_band')):
                weight = 0.
            if 'unverified_direct_window' in row['reasons']:
                weight *= .25
            row['confidence'] = float(np.clip(weight, 0, 1))
        report.update(schema_version='room-dry/4', dereverberation=residual,
                      reference_is_surrogate=surrogate, activity=activity,
                      target_response_sha256=hashlib.sha256(np.asarray(desired, dtype='<f8').tobytes()).hexdigest(),
                      source_id=next(iter(source_ids), None), validity=asdict(DEREVERB_VALIDITY))
        if surrogate:
            report['target'] = 'physical-source sound at common best-DRR sweep direct target, including mic response and target-distance gain'
            warnings.append('Optional REF absent: common best-DRR sweep direct target; no 0.3 m or anechoic calibration claimed.')
        return DryResult(out, fs, report, validity=DEREVERB_VALIDITY, version='room-dry/4', source_reference=source_reference)
    return DryResult(out, fs, report,
                     validity=TIKHONOV_VALIDITY if config.inverse_mode == 'tikhonov' else VALIDITY,
                     source_reference=source_reference)


def _read_wav(path, fs, scratch=None):
    raw, rate = read_recording_audio(Path(path))
    if rate != fs or raw.shape[1] != 1:
        raise ValueError(f"{path}: expected mono WAV at {fs} Hz")
    values = _array(raw[:, 0], str(path))
    if scratch is None:
        return values
    out = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32, shape=values.shape)
    out[:] = values
    return out


def run_manifest(folder, rir_index, *, label, case, output, source="S1",
                 synchronized=False, config=None, comparable_levels=False):
    """Calibrated RecordingManifest -> dry estimates, one sequential take at a time.

    RIR index v1 declares units (Pa/FS or FS/FS) per transfer, separately from
    audio. Unversioned indices retain the legacy same-recording-units assumption
    with a visible warning; they are not evidence of calibrated transfer units.
    """
    folder, rir_index, output = Path(folder), Path(rir_index), Path(output)
    manifest = load_manifest(folder/"manifest.yaml")
    index = load_yaml_unique(rir_index.read_text())
    fs = manifest.fs
    if index["fs_hz"] != fs or index["source"] != source:
        raise ValueError("RIR index sample rate/source does not match manifest")
    version = index.get("schema_version", "legacy-unversioned")
    if version not in ("legacy-unversioned", "room-rir-index/1"):
        raise ValueError("unsupported RIR index version")
    reference = index.get("reference", "REF")
    takes = [(i, t) for i, t in enumerate(manifest.takes)
             if t.type == "real" and t.speaker_id == source and t.label == label]
    if not takes:
        raise ValueError("no real takes for source/label")
    def mic(take):
        if take._channel_count != 1 or not take.mic_ids or len(take.mic_ids) != 1:
            raise ValueError("dry manifest CLI requires named mono takes; use the array API for synchronized channels")
        return take.mic_ids[0]
    if len({mic(t) for _, t in takes}) != len(takes):
        raise ValueError("duplicate real take for one mic; choose an unambiguous label")
    noise_takes = [(i, t) for i, t in enumerate(manifest.takes) if t.type == "noise"]
    if not noise_takes:
        raise ValueError("manifest needs a measured noise-floor take")
    if len({mic(t) for _, t in noise_takes}) != len(noise_takes):
        raise ValueError("ambiguous noise takes for one mic")
    noise_by_mic = {mic(t): (i, t) for i, t in noise_takes}
    floors_for_take = {}
    # Validate every unit boundary before allocating/writing output.
    for i, take in takes:
        name = mic(take)
        floor = noise_by_mic.get(name)
        if floor is None and len(noise_takes) == 1:
            floor = noise_takes[0]
        if floor is None:
            raise ValueError("missing or ambiguous noise-floor take")
        units = "Pa" if take.calibration else "FS"
        if bool(floor[1].calibration) != bool(take.calibration):
            raise ValueError("real and noise units must both be calibrated Pa or uncalibrated FS")
        needed = {name} | ({reference} if case == "B" and reference in index["rirs"] else set())
        for key in needed:
            entry = index["rirs"][key]
            declared = entry.get("units")
            if declared is None and version == "room-rir-index/1":
                raise ValueError("RIR index v1 requires explicit transfer units")
            if declared is not None and declared != units + "/FS":
                raise ValueError("recording/noise units do not match RIR transfer units")
        floors_for_take[i] = floor
    rirs, metadata = {}, {}
    for name, entry in index["rirs"].items():
        path = rir_index.parent/entry["file"]
        rirs[name] = np.load(path, mmap_mode="r", allow_pickle=False) if path.suffix == ".npy" else _read_wav(path, fs)
        metadata[name] = {**entry, "source_id": source}
        if entry.get("units") in ("Pa/FS", "FS/FS"):
            metadata[name]["recording_units"] = "Pa" if entry["units"] == "Pa/FS" else "relative"
        matching = [t for t in manifest.takes if t.speaker_id == source and t.mic_ids == [name]]
        if matching:
            take = matching[0]
            expected_units = "Pa/FS" if take.calibration else "FS/FS"
            if entry.get("units", expected_units) != expected_units:
                raise ValueError("RIR transfer units disagree with matching manifest take")
            metadata[name].update(recording_units="Pa" if take.calibration else "relative",
                                  audio_calibration=channel_calibration(take, 0))
            # Coordinates are measured manifest inputs, not a guessed 0.3 m REF.
            metadata[name].setdefault("reference_distance_m", float(np.linalg.norm(
                np.asarray(take.source_xyz_m)-np.asarray(take.receiver_xyz_m).reshape(-1, 3)[0])))
    results, reports = {}, {}
    groups = [takes] if synchronized else [[t] for t in takes]
    output.mkdir(parents=True, exist_ok=True)
    for group in groups:
        key = "joint" if synchronized else mic(group[0][1])
        if Path(key).name != key or key in (".", ".."):
            raise ValueError("mic id must be a simple filename component")
        audio_path, report_path = output/f"{key}_dry.wav", output/f"{key}_report.json"
        if audio_path.exists() or report_path.exists():
            raise FileExistsError("refusing to overwrite a dry extraction")
        with tempfile.TemporaryDirectory(prefix="dry-", dir=output) as temp:
            recordings = {mic(t): read_take(manifest, i)[:, 0] for i, t in group}
            floors = {mic(t): read_take(manifest, floors_for_take[i][0])[:, 0] for i, t in group}
            buffer = np.lib.format.open_memmap(Path(temp)/"estimate.npy", mode="w+", dtype=np.float32,
                shape=(clock_corrected_length(len(next(iter(recordings.values()))), metadata[mic(group[0][1])]),))
            result = extract_dry(recordings, rirs, floors, fs=fs, case=case, reference=reference,
                                 synchronized=synchronized, metadata=metadata, config=config, output_buffer=buffer)
            wavfile.write(audio_path, fs, result.samples)
        result.report["input_units"] = {mic(t): "Pa" if t.calibration else "FS" for _, t in group}
        result.report["rir_unit_policy"] = version
        if version == "legacy-unversioned":
            result.report["warnings"].append("Legacy unversioned RIR index: transfer units assumed to match calibrated recordings; declare Pa/FS or FS/FS in room-rir-index/1.")
        if any(mic(floors_for_take[i][1]) != mic(t) for i, t in group):
            result.report["warnings"].append("One measured noise take is a proxy for a different microphone position.")
        inputs = [folder/"manifest.yaml", rir_index] + [t.file for _, t in group+noise_takes]
        inputs += [rir_index.parent/e["file"] for e in index["rirs"].values()]
        result.report["provenance"] = {
            "files": {str(p.resolve()): sha256_file(p) for p in inputs},
            "git": source_revision()["commit"],  # this package's checkout only, never the caller's cwd
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
            "numeric_threads": {k: os.environ.get(k) for k in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS")},
        }
        atomic_json(report_path, report_data(to_native(result.report)))
        results[key] = str(audio_path)
        reports[key] = result.report
    recommended = "joint" if synchronized else reference if reference in results else next(iter(results))
    summary = dict(outputs=results, recommended=recommended,
                   note="Independent takes are separate performances; REF preferred when available.")
    if config and config.dereverb_mode == 'measured_residual' and not synchronized:
        from .dereverb import combine_take_reports
        summary.update(combine_take_reports(reports, comparable_levels=comparable_levels))
        summary['note'] = 'Independent performances: combined envelope diagnostics only; selected waveform remains one take.'
    atomic_json(output/"summary.json", report_data(to_native(summary)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rir-index", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case", choices=["A", "B"], required=True)
    parser.add_argument("--source", default="S1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synchronized", action="store_true", help="same event and common sample clock ONLY")
    parser.add_argument("--dereverb-mode", choices=['legacy', 'measured_residual'], default='legacy')
    parser.add_argument("--early-ms", type=float, help='optional partial inverse duration; full measured RIR otherwise')
    parser.add_argument("--comparable-levels", action='store_true', help='assert comparable performance levels, device gain and orientation')
    args = parser.parse_args(argv)
    options = vars(args)
    config = DryConfig(dereverb_mode=options.pop('dereverb_mode'), early_ms=options.pop('early_ms'))
    print(json.dumps(run_manifest(**options, config=config), indent=2))


if __name__ == "__main__":
    main()
