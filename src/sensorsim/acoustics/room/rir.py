"""Known-excitation RIR extraction. Farina AES 5093 (2000), 7121 (2007).

ESS inverse: time-reversed sweep times exp(-t/L), L=T/log(f2/f1).
Packets at -L*log(k) separate harmonic orders in time, not Volterra kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve, windows

from sensorsim.core.messages import ModelValidity
from .ingest import ESSExcitation, KnownExcitation, RecordingManifest, SweepKit, read_recording_audio, read_take

VALIDITY = ModelValidity(
    name="room known-excitation RIR extraction", status="partially validated",
    conditions="Synthetic round trips only; stationary source/receiver. Kit takes "
    "may have independent offsets and constant clocks within +/-300 ppm; earliest "
    "supported linear packet is assumed direct. Generic ESS requires a known offset. ESS band "
    "limited; harmonic packets must not overlap. Blind extraction unsupported.",
    citation="Farina AES 5093 (2000), AES 7121 (2007)",
)


def _samples(value, *, mono=False):
    a = np.asarray(value)
    if np.iscomplexobj(a):
        raise ValueError("pressure samples must be real")
    a = np.asarray(a, dtype=np.float64)
    if a.ndim not in ((1,) if mono else (1, 2)) or not a.size or not np.all(np.isfinite(a)):
        raise ValueError("expected nonempty finite samples, shape (n,) or (n, channels)")
    return a


def _fs(fs):
    if isinstance(fs, bool) or not isinstance(fs, (int, np.integer)) or fs <= 0:
        raise ValueError("fs must be a positive integer")


@dataclass
class RIR:
    fs: int
    samples: np.ndarray
    source_xyz_m: tuple[float, float, float] | None = None
    receiver_xyz_m: tuple[float, float, float] | list[tuple[float, float, float]] | None = None
    metadata: dict = field(default_factory=lambda: {"method": "unspecified", "model_version": "1", "seed": None})

    def __post_init__(self):
        _fs(self.fs)
        self.samples = _samples(self.samples).astype(np.float32)
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("RIR exceeds float32 range")
        for xyz in (self.source_xyz_m, self.receiver_xyz_m):
            if xyz is not None:
                coords = np.asarray(xyz)
                if coords.ndim not in (1, 2) or coords.shape[-1] != 3 or not np.all(np.isfinite(coords)):
                    raise ValueError("RIR coordinates must be finite XYZ")
        if self.receiver_xyz_m is not None and np.asarray(self.receiver_xyz_m).ndim == 2:
            channels = 1 if self.samples.ndim == 1 else self.samples.shape[1]
            if len(self.receiver_xyz_m) != channels:
                raise ValueError("receiver coordinates must match channels")
        self.metadata = {"method": "unspecified", "model_version": "1", "seed": None, **self.metadata}


def generate_ess(fs: int, spec: ESSExcitation) -> np.ndarray:
    """Generate (never play) a sweep, using rounded duration n/fs in the phase."""
    _fs(fs)
    n = round(spec.duration_s * fs)
    if n < 2 or spec.f_end_hz >= fs / 2:
        raise ValueError("ESS needs >=2 samples and f_end below Nyquist")
    t = np.arange(n) / fs
    length = n / fs / np.log(spec.f_end_hz / spec.f_start_hz)
    sweep = spec.amplitude * np.sin(2 * np.pi * spec.f_start_hz * length * (np.exp(t / length) - 1))
    if spec.fade_in_s is not None or spec.fade_out_s is not None:
        n_in, n_out = int((spec.fade_in_s or 0) * fs), int((spec.fade_out_s or 0) * fs)
        if n_in:
            sweep[:n_in] *= windows.hann(2 * n_in)[:n_in]
        if n_out:
            sweep[-n_out:] *= windows.hann(2 * n_out)[n_out:]
        return sweep
    fade = round(spec.fade_s * fs)
    if fade:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        sweep[:fade] *= ramp
        sweep[-fade:] *= ramp[::-1]
    return sweep


def ess_inverse(fs: int, spec: ESSExcitation) -> np.ndarray:
    """Amplitude-compensated inverse with unit in-band transfer gain.

    Stationary-phase spectrum |X(f)|^2=A^2*fs^2*L/(4*f), compensated
    by f/f2. exp(1/(fs*L)) accounts for reversing sample n-1, not n.
    A band-limited unit transfer has a peak below one: forcing its time-domain
    peak to unity would incorrectly amplify every recovered in-band RIR.
    Edge fades deliberately remain part of the measurement bandwidth.
    """
    sweep = generate_ess(fs, spec)
    length = len(sweep) / fs / np.log(spec.f_end_hz / spec.f_start_hz)
    inverse = sweep[::-1] * np.exp(-np.arange(len(sweep)) / fs / length)
    gain = spec.amplitude ** 2 * fs ** 2 * length / (4 * spec.f_end_hz) * np.exp(1 / (fs * length))
    if gain <= np.finfo(float).tiny:
        raise ValueError("degenerate sweep inverse")
    return inverse / gain


@dataclass
class ESSResult:
    linear: np.ndarray
    harmonics: dict[int, np.ndarray]
    harmonic_advance_s: dict[int, float]
    linear_start_sample: int  # in full deconvolution, including inverse delay
    harmonic_start_samples: dict[int, int]
    packet_pre_s: float


def deconvolve_ess(recording, fs: int, spec: ESSExcitation, *, rir_duration_s: float,
                   orders: tuple[int, ...] = (), recording_delay_s: float = 0,
                   packet_pre_s: float = 0.0025) -> ESSResult:
    """Return linear RIR and optional isolated distortion packets.

    recording_delay_s is a KNOWN acquisition offset, not inferred by peak alignment.
    Linear RIR starts at that offset (propagation delay is preserved). Each harmonic
    packet starts packet_pre_s before its predicted order arrival. Requested RIR
    duration plus pre-window must fit between ALL orders through max(orders)+1;
    otherwise raises instead of silently mixing distortion with the linear RIR.
    Requires the complete sweep and recorded decay, no circular convolution.
    """
    _fs(fs)
    y = _samples(recording)
    parameters = [rir_duration_s, recording_delay_s, packet_pre_s]
    if not np.all(np.isfinite(parameters)) or rir_duration_s <= 0 or min(parameters[1:]) < 0:
        raise ValueError("invalid extraction durations")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 2 for k in orders):
        raise ValueError("harmonic orders must be integers >=2")
    inverse = ess_inverse(fs, spec)
    count = round(rir_duration_s * fs)
    offset = round(recording_delay_s * fs)
    if count < 1 or len(y) < offset + len(inverse) + count - 1:
        raise ValueError("recording must include full sweep and requested decay")
    length = len(inverse) / fs / np.log(spec.f_end_hz / spec.f_start_hz)
    if orders and rir_duration_s + packet_pre_s >= length * np.log((max(orders) + 1) / max(orders)):
        raise ValueError("harmonic packets overlap: use a longer sweep or shorter RIR window")
    full = fftconvolve(y, inverse if y.ndim == 1 else inverse[:, None], axes=0)
    start = len(inverse) - 1 + offset
    advances = {k: float(length * np.log(k)) for k in orders}
    starts = {k: start - round(advances[k] * fs) - round(packet_pre_s * fs) for k in orders}
    if any(i < 0 for i in starts.values()):
        raise ValueError("harmonic packet precedes recording support")
    packet_count = count + round(packet_pre_s * fs)
    return ESSResult(full[start:start + count].astype(np.float32),
                     {k: full[i:i + packet_count].astype(np.float32) for k, i in starts.items()},
                     advances, start, starts, packet_pre_s)


def correct_known_clock(recording, excitation, fs: int, *, max_ppm: float = 300.) -> tuple[np.ndarray, dict]:
    """Estimate a constant capture clock from whole-record FIR concentration.

    Resampling maximizes the joint response of direct sound and reflections;
    no block independently chooses a single echo. Short, periodic or ambiguous
    signals abstain. The offset includes acquisition and propagation and stays
    in the corrected signal. It only places an unanchored extraction window.
    This is a broadband LTI estimator, not an arbitrary-audio synchronizer.
    """
    from .known_clock import fit_clock
    _fs(fs)
    y, x = _samples(recording), _samples(excitation, mono=True)
    if not np.isfinite(max_ppm) or not 0 < max_ppm <= 2000:
        raise ValueError("max_ppm must lie in (0,2000]")
    return fit_clock(y, x, fs, max_ppm)


def deconvolve_known(recording, excitation, *, regularization: float = 1e-8,
                     rir_samples: int | None = None, fs: int | None = None,
                     correct_clock_drift: bool = False, timing: dict | None = None) -> np.ndarray:
    """Wiener-style H=Y*conj(X)/(|X|^2 + lambda*max(|X|^2)).

    lambda is dimensionless relative spectral power (not an estimated noise PSD).
    Frequencies absent from X cannot be recovered. Full linear convolution must
    be recorded; maximum identifiable FIR length is len(y)-len(x)+1. Zero-padding
    both arrays to at least len(y)+len(x)-1 prevents time-domain wrap at the seam.
    By default no gain/delay fitting or resampling (legacy output unchanged).
    With correct_clock_drift=True, fs is required; constant clock drift is estimated
    and corrected first, retaining the total acquisition/propagation delay. Ambiguous
    timing raises. Optional timing receives the correction diagnostics.
    Returns float32, preserving channels.
    """
    y, x = _samples(recording), _samples(excitation, mono=True)
    if correct_clock_drift:
        if fs is None:
            raise ValueError("fs is required for known-signal clock correction")
        y, diagnostics = correct_known_clock(y, x, fs)
        if timing is not None:
            timing.update(diagnostics)
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    maximum = len(y) - len(x) + 1
    count = maximum if rir_samples is None else rir_samples
    if not isinstance(count, (int, np.integer)) or count < 1 or count > maximum:
        raise ValueError("requested FIR length exceeds full-convolution recording support")
    nfft = next_fast_len(len(y) + len(x) - 1)
    spectrum = rfft(x, n=nfft)
    power = np.abs(spectrum) ** 2
    if not np.any(power):
        raise ValueError("excitation has zero energy")
    inverse = spectrum.conj() / (power + regularization * power.max())
    if y.ndim == 2:
        inverse = inverse[:, None]
    return irfft(rfft(y, n=nfft, axis=0) * inverse, n=nfft, axis=0)[:count].astype(np.float32)


def estimate_noise_power(samples, tail_fraction: float = 0.1) -> np.ndarray:
    """Mean square of final fraction, per channel; assumes stationary noise-only tail."""
    x = _samples(samples)
    if not np.isfinite(tail_fraction) or not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    return np.mean(x[-max(1, int(len(x) * tail_fraction)):] ** 2, axis=0)


def detect_onset(samples, *, threshold_db: float = -20, noise_margin_db: float = 20,
                 noise_power: float | None = None) -> int:
    """First sample above BOTH peak-relative and noise-relative power thresholds.

    Heuristic on broadband mono RIR (not a geometry proof); weak direct paths can
    be missed. Supply manual arrival for occluded/low-SNR/multiple-peak cases.
    """
    x = _samples(samples, mono=True)
    noise = float(estimate_noise_power(x)) if noise_power is None else float(noise_power)
    if not np.all(np.isfinite([noise, threshold_db, noise_margin_db])) or noise < 0 or threshold_db > 0:
        raise ValueError("invalid onset threshold/noise")
    power = x * x
    threshold = max(power.max() * 10 ** (threshold_db / 10), noise * 10 ** (noise_margin_db / 10))
    indices = np.flatnonzero((power >= threshold) & (power > 0))
    if not len(indices):
        raise ValueError("no onset distinguishable from noise")
    return int(indices[0])


def trim_rir(rir: RIR, *, onset_sample: int | None = None, pre_s: float = 0.0025,
             duration_s: float | None = None) -> RIR:
    """Common trim across microphones, preserving origin shift in metadata."""
    if not np.isfinite(pre_s) or pre_s < 0 or (duration_s is not None and
            (not np.isfinite(duration_s) or duration_s <= 0)):
        raise ValueError("invalid trim duration")
    x = rir.samples
    mono = x if x.ndim == 1 else x[:, np.argmax(np.max(np.abs(x), axis=0))]
    onset = detect_onset(mono) if onset_sample is None else onset_sample
    if not isinstance(onset, (int, np.integer)) or not 0 <= onset < len(x):
        raise ValueError("invalid onset sample")
    start = max(0, onset - round(pre_s * rir.fs))
    stop = len(x) if duration_s is None else min(len(x), onset + max(1, round(duration_s * rir.fs)))
    metadata = {**rir.metadata, "trim_start_sample": start,
                "time_origin_s": rir.metadata.get("time_origin_s", 0) + start / rir.fs,
                "onset_sample": onset - start}
    if "direct_samples" in metadata:
        # These indices refer to the RIR array, unlike recording repeat_peak_samples.
        # A negative/out-of-range value explicitly locates an arrival outside the crop.
        metadata["direct_samples"] = [int(sample) - start for sample in metadata["direct_samples"]]
    if "direct_samples_fractional" in metadata:
        metadata["direct_samples_fractional"] = [
            float(sample) - start for sample in metadata["direct_samples_fractional"]]
    return RIR(rir.fs, x[start:stop], rir.source_xyz_m, rir.receiver_xyz_m, metadata)


def deconvolve_blind(*args, **kwargs):
    raise NotImplementedError("blind RIR extraction is not supported yet")


def extract_take(manifest: RecordingManifest, index: int, *, rir_duration_s: float,
                 regularization: float = 1e-8, sound_speed_m_s: float = 343.,
                 sound_speed_model: str = "fixed-reference/343", adaptive_window: bool = False) -> RIR:
    """Convenience path; use deconvolve_ess directly to retrieve distortion packets."""
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0:
        raise ValueError("sound speed must be finite and positive")
    if adaptive_window and manifest.sweep_kit is None:
        raise ValueError('adaptive RIR windows require a sweep kit with a known quiet tail')
    take = manifest.takes[index]
    if take.type != "sweep":
        raise ValueError("RIR extraction requires a sweep take; real/noise are separate inputs")
    if take.excitation.type == "unknown":
        return deconvolve_blind()
    samples = read_take(manifest, index)
    kit_meta = {}
    if manifest.sweep_kit is not None:
        result = deconvolve_repetitions(samples, manifest.sweep_kit,
                                        rir_duration_s=rir_duration_s,
                                        min_repeats=min(3, manifest.sweep_kit.repeats),
                                        source_xyz_m=take.source_xyz_m,
                                        receiver_xyz_m=take.receiver_xyz_m, sound_speed_m_s=sound_speed_m_s,
                                        adaptive_window=adaptive_window)
        extracted, kit_meta = result.samples, result.metadata
        if manifest.obstacles:
            from sensorsim.core.schema import SceneBox
            from sensorsim.scene.geometry import Box
            receivers = np.atleast_2d(take.receiver_xyz_m)
            sources = np.broadcast_to(take.source_xyz_m, receivers.shape)
            obstructed = np.zeros(len(receivers), dtype=bool)
            for obstacle in manifest.obstacles:
                lower, size = np.array(obstacle.min_xyz), np.array(obstacle.size_xyz)
                box = Box(SceneBox(id=obstacle.name, base_center_enu_m=tuple(lower+size*[.5,.5,0]),
                                   size_m=tuple(size), heading_deg=90., material=obstacle.material))
                obstructed |= box.segment_slab(sources, receivers)[0]
            kit_meta['direct_path_obstructed'] = obstructed.tolist()
            if np.any(obstructed):
                message = "obstacle intersects the direct path: first supported packet may be a reflection/diffraction; geometry-aligned timing is not a measured direct arrival"
                warnings.warn(message, UserWarning, stacklevel=2)
                kit_meta['quality']['warnings'].append(message)
    elif isinstance(take.excitation, ESSExcitation):
        extracted = deconvolve_ess(samples, manifest.fs, take.excitation,
                                  rir_duration_s=rir_duration_s).linear
    elif isinstance(take.excitation, KnownExcitation):
        excitation, fs = read_recording_audio(take.excitation.path)
        if fs != manifest.fs or excitation.shape[1] != 1:
            raise ValueError("known excitation changed since manifest validation")
        if not np.isfinite(rir_duration_s) or rir_duration_s <= 0:
            raise ValueError("rir_duration_s must be positive")
        extracted = deconvolve_known(samples, excitation[:, 0], regularization=regularization,
                                     rir_samples=round(rir_duration_s * fs))
    if take.is_reference and manifest.shoebox_dims_m is not None:
        # Image distances give the earliest boundary return, independent of its
        # unknown absorption. Box distance/halfspace bounds are conservative:
        # they may shorten the gate, never certify a later reflection-free time.
        kit_meta['first_reflection_delay_s'] = first_reflection_delays(
            take.source_xyz_m, take.receiver_xyz_m, manifest.shoebox_dims_m,
            manifest.obstacles, sound_speed_m_s)
        kit_meta['reflection_bound_method'] = 'shoebox images and conservative box-distance bound'
        if manifest.obstacles:
            kit_meta['reflection_bound_method'] = 'shoebox images and conservative box-distance/halfspace bound'
    from sensorsim.licensing import lineage_metadata
    lineage = lineage_metadata(manifest.lineage, {"room_id": manifest.room_id, "recording_file": str(take.file)})
    return RIR(manifest.fs, extracted, take.source_xyz_m, take.receiver_xyz_m,
               {"method": take.excitation.type, "model_version": "1", "seed": None,
                "room_id": manifest.room_id, "recording_file": str(take.file),
                **({"lineage": lineage} if lineage else {}),
                "file_fs": take._file_fs or manifest.fs, "analysis_fs": manifest.fs,
                "excitation": take.excitation.model_dump(mode="json"),
                "calibration": take.calibration.model_dump() if take.calibration else None,
                "regularization": regularization if take.excitation.type == "known_signal" else None,
                "is_reference": take.is_reference, "mic_ids": take.mic_ids, **kit_meta,
                "units": "Pa/FS" if take.calibration else "FS/FS",
                "sound_speed_m_s": sound_speed_m_s, "sound_speed_model": sound_speed_model,
                "time_grid_policy": "sample index / analysis_fs; acoustic delay separate from acquisition offset"})


def generate_kit_playback(kit: SweepKit) -> np.ndarray:
    """Exact undithered float playback: lead + repeats*(sweep+tail) + end."""
    fs = kit.fs_hz
    sweep = generate_ess(fs, kit.excitation)
    period = np.r_[sweep, np.zeros(round(kit.tail_silence_s * fs))]
    return np.r_[np.zeros(round(kit.lead_silence_s * fs)), np.tile(period, kit.repeats),
                 np.zeros(round(kit.end_silence_s * fs))]


def _packet_repeat_alignment(records, fs, count, reference_channel, direct_fractional,
                             geometry, min_repeats, detection):
    """Experimental broadband registration, followed by one mean-onset rule.

    Reuse SciPy correlation and the detector's fractional interpolation. Gamper,
    HSCMA 2017 motivates response registration; the 50 ms / +/-1 ms windows,
    0.9 correlation and 0.1 residual floors are engineering choices, not a
    calibrated quality certificate. A majority must describe the same LTI path.
    """
    from scipy.signal import correlate
    from .sweep_detect import fractional_crop, fractional_peak

    radius = max(2, round(.001*fs))
    pre = max(round(.03*fs), int(np.ceil(np.max(direct_fractional))) + 2*radius)
    early_count = pre + round(.05*fs)
    width = max(count + 2*pre, early_count + 2*radius)
    packets, usable = [], []
    for record in records:
        full, origin, step, rep, coarse = record
        start = coarse - pre
        available = (rep.segment_stop-origin)/step
        # The FFT tail is not recorded decay. Retain only observed support.
        length = min(width, int(np.floor(available-start)))
        if start < radius or length < early_count+radius:
            detection.dropped.append(dict(slot=rep.slot, reason="repeat alignment lacks early-response support"))
            continue
        packets.append(fractional_crop(full[:, reference_channel], start-radius, early_count+2*radius))
        usable.append(record)
    if len(usable) < min_repeats:
        raise ValueError("too few repeats with alignment support")

    def register(packet, template):
        corr = correlate(packet, template, mode="valid", method="fft")
        power = np.r_[0., np.cumsum(packet*packet)]
        norm = np.sqrt(np.maximum((power[len(template):]-power[:-len(template)])*
                                 np.sum(template*template), 1e-300))
        score = corr/norm
        index = int(np.argmax(score))
        if index in (0, len(corr)-1):
            return 0., 0.
        location = fractional_peak(corr, index)
        registered = fractional_crop(packet, location, len(template))
        quality = float(registered @ template / max(
            np.linalg.norm(registered)*np.linalg.norm(template), 1e-300))
        return location-radius, float(np.clip(quality, -1, 1))

    n = len(usable)
    scores = np.eye(n)
    for i in range(n):
        template = packets[i][radius:-radius]
        for j in range(i+1, n):
            _, scores[i, j] = register(packets[j], template)
            scores[j, i] = scores[i, j]
    # A medoid survives a bad first repeat; ties are deterministic by slot order.
    medoid = max(range(n), key=lambda i: (np.count_nonzero(scores[i] >= .9),
                                        np.median(scores[i])))
    template = packets[medoid][radius:-radius]
    shifts, retained, correlations = [], [], []
    for i, record in enumerate(usable):
        shift, score = (0., 1.) if i == medoid else register(packets[i], template)
        if score < .9:
            detection.dropped.append(dict(slot=record[3].slot,
                reason="repeat early-response correlation below 0.9", correlation=score))
            continue
        shifts.append(shift)
        correlations.append(score)
        retained.append(record)
    if len(retained) < min_repeats or len(retained)*2 <= n:
        raise ValueError("no majority of consistent repeat responses")

    # Compare the whole observed decay too: a transient outside the early
    # alignment packet must not enter the mean. No gain or polarity correction.
    lengths = [int(np.floor((r[3].segment_stop-r[1])/r[2] - (r[4]-pre+s)))
               for r, s in zip(retained, shifts)]
    common_count = min(width, *lengths)
    aligned = np.array([np.column_stack([
        fractional_crop(r[0][:, ch], r[4]-pre+s, common_count)
        for ch in range(r[0].shape[1])]) for r, s in zip(retained, shifts)])
    center = np.median(aligned, axis=0)
    residuals = np.sqrt(np.sum((aligned-center)**2, axis=1) /
                        np.maximum(np.sum(center**2, axis=0), 1e-30)).max(axis=1)
    limit = max(.1, 4*float(np.median(residuals)))
    good = residuals <= limit
    for i in np.flatnonzero(~good):
        detection.dropped.append(dict(slot=retained[i][3].slot,
            reason="repeat full-response residual outlier", relative_residual=float(residuals[i]),
            residual_limit=limit))
    if np.count_nonzero(good) < min_repeats:
        raise ValueError("too few repeats after response outlier rejection")
    mean = np.mean(aligned[good], axis=0)

    # A threshold crossing of the mean, never the strongest of nearby peaks.
    # Smooth over 0.1 ms, then first crossing of -20 dB peak-relative power AND
    # 20 dB above the median pre-arrival noise. Interpolate the power crossing.
    onsets = []
    smooth = np.ones(max(1, round(.0001*fs))) / max(1, round(.0001*fs))
    # cr-room-dsp-01: onsets are found only in the early window, from `pre` before to 50 ms after
    # the reference arrival. A channel arriving later (by geometry, or with no supported packet
    # there by the same -20 dB smoothed-power rule) is rejected, never aligned from sidelobes.
    if geometry and np.any(np.rint(direct_fractional) >= count):
        raise ValueError("RIR duration ends before assumed direct path")
    for channel in range(mean.shape[1]):
        if channel == reference_channel:
            continue
        late = direct_fractional[channel]-direct_fractional[reference_channel] if geometry else 0.
        power = np.convolve(mean[:, channel]**2, smooth, mode="same")
        if pre+late >= early_count or not power[:early_count].max() >= .01*power.max():
            evidence = (f"geometric delay {late/fs*1e3:+.1f} ms" if geometry else
                        f"strongest packet {(int(np.argmax(power))-pre)/fs*1e3:+.1f} ms")
            raise ValueError(
                f"packet-medoid/1 cannot align channel {channel}: its arrival lies outside the method's early "
                f"window (-{pre/fs*1e3:.0f} to +50 ms around reference channel {reference_channel}; {evidence}); "
                "use repeat_alignment='legacy'" + ("" if geometry else " with source/receiver geometry"))
    for channel in range(mean.shape[1]):
        power = np.convolve(mean[:early_count, channel]**2, smooth, mode="same")
        noise = float(np.median(power[:max(1, pre//2)]))
        threshold = max(float(power.max())*.01, noise*100)
        found = np.flatnonzero((power >= threshold) & (power > 0))
        if not len(found) or found[0] == 0:
            raise ValueError("mean response has no supported onset inside alignment window")
        k = int(found[0])
        onsets.append(k-1 + (threshold-power[k-1])/(power[k]-power[k-1]))
    onsets = np.asarray(onsets)
    if not geometry:
        direct_fractional = onsets-onsets.min()
    if np.any(np.rint(direct_fractional) >= count):
        raise ValueError("RIR duration ends before assumed direct path")
    local_offsets = onsets-direct_fractional if geometry else np.full(len(onsets), onsets.min())
    windows, arrivals, kept, corrections = [], [], [], []
    for i in np.flatnonzero(good):
        full, origin, step, rep, coarse = retained[i]
        offsets = coarse-pre+shifts[i]+local_offsets
        available = (rep.segment_stop-origin)/step
        if np.any(offsets < 0) or np.any(offsets+count > available):
            detection.dropped.append(dict(slot=rep.slot, reason="aligned repeat lacks complete requested decay"))
            continue
        windows.append(np.column_stack([fractional_crop(full[:, ch], offset, count)
                                        for ch, offset in enumerate(offsets)]))
        arrivals.append((origin+(coarse-pre+shifts[i]+onsets)*step).tolist())
        # Inverse-filter delay is subtracted by the caller, in recording units.
        kept.append(rep)
        corrections.append(dict(slot=rep.slot, shift_samples=float(shifts[i]),
            correlation=correlations[i], relative_residual=float(residuals[i])))
    if len(kept) < min_repeats:
        raise ValueError("too few aligned repeats with complete requested decay")
    return windows, arrivals, kept, direct_fractional, dict(
        method="packet-medoid/1", template_slot=usable[medoid][3].slot,
        origin_rule="mean first 0.1ms-smoothed power crossing: max(-20dB peak, +20dB noise)",
        early_window_s=.05, search_radius_s=.001, corrections=corrections,
        status="unvalidated", majority_required=True)


def _channel_arrivals(full, fs, expected, count, reference_channel, shifts, *, bandwidth_hz, geometry):
    """First supported linear packet of every channel, each searched near its own arrival.

    The reference channel keeps the +/-30 ms search around the detector's arrival.
    Another channel is searched +/-30 ms around ``expected+shift``: its geometric delay
    relative to the reference, or zero without geometry. Its search window must hold an
    actually supported packet: by linear_arrival's own 0.1 ms-smoothed power and -20 dB
    onset convention, relative to that channel's strongest packet wherever an arrival
    could lie (|TDOA| < RIR length + 30 ms). Otherwise the take is rejected instead of
    aligned from sidelobes (cr-room-dsp-01). Returns None when a predicted arrival lies
    beyond this repeat's compressed support.
    """
    from scipy.signal import convolve
    from .sweep_detect import linear_arrival
    radius = round(.03*fs)
    width = max(1, round(.0001*fs))
    span = slice(max(0, int(expected-count-radius)), min(len(full), int(expected+count+radius+1)))
    peaks = []
    for channel in range(full.shape[1]):
        center = expected+shifts[channel]
        lo, hi = max(0, int(center-radius)), min(len(full), int(center+radius+1))
        if hi <= lo:
            return None
        peak, _ = linear_arrival(full[:, channel], fs, lo, hi, bandwidth_hz=bandwidth_hz)
        if channel != reference_channel:
            power = convolve(full[span, channel]**2, np.ones(width)/width, mode="same")
            window = power[max(0, lo-span.start):max(0, hi-span.start)]
            if not (len(window) and window.max() >= .01*power.max()):
                strongest = span.start+int(np.argmax(power))
                away = f"strongest compressed packet {(strongest-center)/fs*1e3:+.1f} ms away"
                raise ValueError(
                    f"channel {channel}: no supported linear packet within 30 ms of its geometric arrival "
                    f"({away}); check source/receiver coordinates, channel order, sound speed or occlusion"
                    if geometry else
                    f"channel {channel}: its arrival lies outside the 30 ms search around reference channel "
                    f"{reference_channel} ({away}); supply source/receiver geometry or extract it separately")
        peaks.append(peak)
    return peaks


def deconvolve_repetitions(recording, kit: SweepKit, *, rir_duration_s: float = 1,
                           source_xyz_m=None, receiver_xyz_m=None, sound_speed_m_s: float = 343,
                           max_clock_drift_ppm: float = 300,
                           correct_clock_drift: bool = True,
                           min_repeats: int = 3, adaptive_window: bool = False,
                           repeat_alignment: str = "legacy") -> RIR:
    """Use sweep_detect's sole detector; average only accepted isolated packets.

    Fractional packet shifts preserve the compressed pulse. Geometry supplies
    fractional distance/c as time origin; otherwise a common relative shift
    preserves TDOA. ``direct_samples`` contains rounded window indices only;
    ``direct_samples_fractional`` retains the sub-sample arrivals on that same
    array grid. Cropping shifts both fields; ``time_origin_s`` retains the origin.
    Absolute emission time remains unobservable. First supported packet = direct is an
    explicit assumption, not an independent range measurement.

    ``repeat_alignment="packet-medoid/1"`` is an experimental opt-in. It
    registers broadband early responses, rejects response outliers and chooses
    the first supported onset of their mean. This changes the RIR origin;
    see parent-project note ROOM_RIR_ALIGN (not distributed) for qualification failures and default policy.
    """
    from .sweep_detect import (DetectionConfig, clipping_present, compress_repeat, detect_sweeps,
                               linear_arrival, fractional_crop)
    y = _samples(recording)
    if repeat_alignment not in ("legacy", "packet-medoid/1"):
        raise ValueError("unsupported repeat alignment method")
    was_mono = y.ndim == 1
    if was_mono:
        y = y[:, None]
    fs = kit.fs_hz
    if (not np.all(np.isfinite([rir_duration_s, sound_speed_m_s, max_clock_drift_ppm])) or
            min(rir_duration_s, sound_speed_m_s, max_clock_drift_ppm) <= 0):
        raise ValueError("invalid duration, sound speed or clock bound")
    if rir_duration_s > kit.tail_silence_s:
        raise ValueError("RIR duration exceeds quiet gap between sweeps")
    minimum_duration = rir_duration_s
    if adaptive_window:
        # Keep a small separation from the following sweep; never manufacture
        # a noise-only tail beyond the samples the kit actually provides.
        rir_duration_s = max(minimum_duration,kit.tail_silence_s-.05)
    count = round(rir_duration_s*fs)
    if count < 1:
        raise ValueError("RIR needs at least one sample")
    reference_channel = int(np.argmax(np.mean(y*y, axis=0)))
    detection = detect_sweeps(y[:, reference_channel], fs, kit,
                             config=DetectionConfig(min_repeats=min_repeats,
                                                    max_clock_drift_ppm=max_clock_drift_ppm))
    if detection.clipped:
        raise ValueError("clipping detected in sweep take; lower fixed gain and re-record before linear RIR extraction")
    # The detector sees only the reference channel. Admit every channel by the same digital-rail
    # rule (cr-room-dsp-04): a clipped quieter channel was returned as a linear RIR.
    clipped = [channel for channel in range(y.shape[1]) if clipping_present(y[:, channel])]
    if clipped:
        raise ValueError(f"clipping detected in sweep take channel(s) {clipped}; lower fixed gain "
                         "and re-record before linear RIR extraction")
    ppm = detection.clock_drift_ppm
    applied = correct_clock_drift and abs(ppm) > .5
    channels = y.shape[1]
    geometry = source_xyz_m is not None and receiver_xyz_m is not None
    direct_samples = np.zeros(channels, dtype=int)
    direct_fractional = np.zeros(channels)
    if geometry:
        receivers = np.atleast_2d(receiver_xyz_m)
        source = np.asarray(source_xyz_m)
        if (len(receivers) not in (1, channels) or receivers.shape[1] != 3 or
                source.shape != (3,) or not np.all(np.isfinite(source)) or not np.all(np.isfinite(receivers))):
            raise ValueError("geometry must be finite XYZ and receiver coordinates must match channels")
        direct_fractional[:] = np.linalg.norm(receivers-source, axis=1)/sound_speed_m_s*fs
        direct_samples[:] = np.rint(direct_fractional).astype(int)
    inverse_delay = round(kit.sweep_s*fs)-1
    # Search centre of each channel relative to the reference arrival (cr-room-dsp-01).
    shifts = direct_fractional-direct_fractional[reference_channel] if geometry else np.zeros(channels)
    repeat_windows, arrivals, kept = [], [], []
    records, alignment_metadata = [], {}
    for rep in detection.repeats:
        full, origin, step = compress_repeat(y, kit, start_sample=rep.segment_start,
                                             stop_sample=rep.segment_stop,
                                             drift_ppm=ppm if applied else 0.)
        expected = (rep.start_sample-origin)/step+inverse_delay
        radius = round(.03*fs)
        lo, hi = max(0,int(expected-radius)), min(len(full),int(expected+radius+1))
        if repeat_alignment == "packet-medoid/1":
            coarse = linear_arrival(full[:, reference_channel], fs, lo, hi, bandwidth_hz=kit.f2_hz)[0]
            records.append((full, origin, step, rep, coarse))
            continue
        if geometry and np.any(direct_samples >= count):  # known before searching: keep this error first
            raise ValueError("RIR duration ends before geometric direct path")
        channel_peaks = _channel_arrivals(full, fs, expected, count, reference_channel, shifts,
                                          bandwidth_hz=kit.f2_hz, geometry=geometry)
        if channel_peaks is None:
            detection.dropped.append(dict(slot=rep.slot, reason="a channel's predicted arrival lies beyond this repeat's recorded support; retain a longer post-roll"))
            continue
        if not geometry and not repeat_windows:
            direct_samples[:] = np.rint(np.array(channel_peaks)-min(channel_peaks)).astype(int)
            direct_fractional[:] = (np.array(channel_peaks)-channel_peaks[reference_channel]
                                    + direct_samples[reference_channel])
        if np.any(direct_samples >= count):
            raise ValueError("RIR duration ends before geometric direct path")
        # Rounding the geometric delay independently at each position changes
        # RIR phase by up to half a sample. That phase does not cancel against
        # the real take or a separately measured REF during dry inversion.
        offsets = [channel_peaks[ch]-direct_fractional[ch] if geometry else
                   channel_peaks[reference_channel]-direct_samples[reference_channel]
                   for ch in range(channels)]
        available = (rep.segment_stop-origin)/step
        if any(start < 0 or start+count > available for start in offsets):
            detection.dropped.append(dict(slot=rep.slot,reason="recording lacks the requested decay after this repeat; retain a longer post-roll"))
            continue
        current = np.zeros((count,channels))
        for channel in range(channels):
            current[:,channel] = fractional_crop(full[:,channel],offsets[channel],count)
        repeat_windows.append(current)
        kept.append(rep)
        arrivals.append((origin+(np.array(channel_peaks)-inverse_delay)*step).tolist())
    if repeat_alignment == "packet-medoid/1":
        repeat_windows, arrivals, kept, direct_fractional, detail = _packet_repeat_alignment(
            records, fs, count, reference_channel, direct_fractional, geometry, min_repeats, detection)
        step = 1+ppm*1e-6 if applied else 1.
        arrivals = (np.asarray(arrivals)-inverse_delay*step).tolist()
        direct_samples = np.rint(direct_fractional).astype(int)
        alignment_metadata = {"repeat_alignment": detail}
    if len(kept) < min_repeats:
        raise ValueError(f"only {len(kept)} repeats include the requested decay; need {min_repeats}; keep recording through the reverb tail")
    detection.repeats = kept
    samples = np.mean(repeat_windows,axis=0).astype(np.float32)
    energy = np.array([np.sum(w*w,axis=0) for w in repeat_windows])
    gain_range = 10*np.log10(np.maximum(energy.max(axis=0),1e-30)/np.maximum(energy.min(axis=0),1e-30))
    residual = np.max([np.sqrt(np.sum((w-samples)**2,axis=0)/np.maximum(np.sum(samples**2,axis=0),1e-30))
                       for w in repeat_windows],axis=0)
    # This legacy field is an offset diagnostic; it does not observe emission.
    first_slot = detection.repeats[0].slot
    latency = ((np.asarray(arrivals[0])-direct_fractional)/fs -
               (kit.first_sweep_start_s+first_slot*kit.period_s)*(1+ppm*1e-6))
    snr = []
    for channel in range(channels):
        stop = max(0,int(arrivals[0][channel])-round(.01*fs))
        # A median of short pre-sweep powers resists isolated handling sounds.
        block = max(1,round(.02*fs))
        noise = None
        if stop >= block:
            quiet = y[:stop//block*block,channel].reshape(-1,block)
            noise = float(np.median(np.mean(quiet*quiet,axis=1)))
        powers = []
        for rep in detection.repeats:
            lo = round(rep.start_sample+kit.fade_in_s*fs*(1+ppm*1e-6))
            hi = round(rep.end_sample-kit.fade_out_s*fs*(1+ppm*1e-6))
            powers.append(float(np.mean(y[lo:hi,channel]**2)))
        total = float(np.mean(powers))
        snr.append(float(10*np.log10(max(total-noise,1e-30)/max(noise,1e-30))) if noise is not None else None)
    messages = ["AGC/noise suppression cannot be ruled out by equal-level sweeps; verify raw mic and the two-level self-check before trusting a fixed linear RIR",
                *detection.warnings]
    if geometry:
        messages.append("geometry alignment places the first supported packet at distance/c; this is not independent arrival-time evidence")
    if np.max(np.abs(latency)) >= .1:
        messages.append(f"recording start latency/offset {np.max(np.abs(latency)):.3f} s; absolute emission time is unobservable")
    if detection.dropped:
        messages.append(f"dropped {len(detection.dropped)} unconfirmed/corrupted repeats before averaging; see sweep_detection")
    if np.max(gain_range) > 1 or np.max(residual) > .1:
        messages.append("repeat gain/waveform inconsistency: possible AGC, motion, low SNR or multipath packet misalignment; fixed linear RIR is unreliable")
    if any(value is not None and value < 30 for value in snr):
        messages.append("sweep SNR below 30 dB; quiet bands/decay may be unreliable; reduce background or increase unclipped fixed level")
    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=2)
    window_metadata = {}
    if adaptive_window:
        stop, info = adaptive_decay_window(samples,fs,minimum_s=minimum_duration,
            onsets=direct_samples, f_start_hz=kit.f1_hz, f_end_hz=kit.f2_hz)
        samples = samples[:stop]
        window_metadata['observation_window'] = info
    return RIR(fs, samples[:,0] if was_mono else samples, source_xyz_m, receiver_xyz_m,
               {"method":"ess_repetition_average", "model_version":"3", "seed":None,
                "repeats":len(repeat_windows), "repeat_peak_samples":arrivals,
                "direct_samples":direct_samples.tolist(), "latency_s":latency.tolist(),
                "direct_samples_fractional":direct_fractional.tolist(),
                "clock_drift_ppm":ppm, "clock_resampled":bool(applied),
                "processing_options": {"correct_clock_drift": correct_clock_drift,
                    "clock_correction_applied": bool(applied), "clock_threshold_ppm": .5,
                    "max_clock_drift_ppm": max_clock_drift_ppm, "min_repeats": min_repeats,
                    "alignment": "geometry" if geometry else "relative",
                    "detector": "room.sweep_detect", "validity_id": VALIDITY.name},
                "clock_estimation_method":"raw_repeat_correlation",
                "clock_repeat_correlations":detection.repeat_correlations,
                "chirp_drift_ppm":detection.chirp_drift_ppm,
                **window_metadata, **alignment_metadata,
                "sweep_detection":detection.metadata(), "sound_speed_m_s":sound_speed_m_s,
                "geometry_aligned":geometry, "time_origin":"geometry_distance_over_c" if geometry else "relative",
                "absolute_emission_time_observed":False,
                "channel_alignment":"geometry" if geometry else "common_shift_preserving_tdoa",
                "alignment_assumption":"first supported linear packet is direct sound; verify for occluded paths",
                "quality":{"repeat_gain_range_db":gain_range.tolist(),
                           "repeat_relative_residual":residual.tolist(),
                           "active_sweep_snr_db":snr,
                           "active_sweep_snr_method":"median pre-sweep block power subtraction",
                           "processing_verified":False, "warnings":messages},
                "kit":kit.model_dump(mode="json")})


@dataclass
class ReferenceEQ:
    """Frequency-domain device estimate usable by fit/dry without importing them."""
    fs: int
    nfft: int
    frequency_hz: np.ndarray
    response: np.ndarray            # complex device transfer, propagation delay removed
    correction: np.ndarray          # bounded regularized complex inverse; 0 outside mask
    supported: np.ndarray
    metadata: dict


def adaptive_decay_window(samples, fs, *, minimum_s=1.5, onsets=None,
                          f_start_hz=40., f_end_hz=20000.):
    """Smallest progressive window with usable in-band T30/noise evidence.

    Observation-window/2 is opt-in. It reuses the existing octave filters and
    Chu/Lundeby validity checks; no T30 threshold is relaxed. Every channel must
    qualify. If the available tail is exhausted, return it with the unresolved
    bands and ``tail_limited=True``. This is censoring, not an invented decay.
    The caller must pass only the acquired quiet tail, never zero padding.
    """
    from .metrics import OCTAVE_BANDS, exact_center, room_metrics
    x = _samples(samples)
    if x.ndim == 1:
        x = x[:,None]
    _fs(fs)
    if not np.isfinite(minimum_s) or minimum_s <= 0:
        raise ValueError('minimum observation duration must be positive')
    onsets = np.zeros(x.shape[1],dtype=int) if onsets is None else np.asarray(onsets)
    if len(onsets) != x.shape[1]:
        raise ValueError('one onset is required per observation channel')
    bands = [band for band in OCTAVE_BANDS
             if (f_start_hz <= exact_center(band)*10**-.15
                 and exact_center(band)*10**.15 <= f_end_hz
                 and exact_center(band)*10**.15 < fs/2)]
    stop = min(len(x),max(1,round(minimum_s*fs)))
    steps=[]
    while True:
        unresolved=[]; supported=0
        for channel,onset in enumerate(onsets):
            for band in bands:
                m = room_metrics(x[:stop,channel], fs, onset_sample=int(onset), band=band)
                supported+=1
                if not m.t30.valid or m.t30.r_squared < .9:
                    unresolved.append(dict(channel=channel,band_hz=band,reason=m.t30.reason))
        complete=bool(supported) and not unresolved
        steps.append(dict(duration_s=stop/fs,supported_bands=supported,unresolved=unresolved))
        if complete or stop==len(x):
            return stop,dict(version='observation-window/2',minimum_s=minimum_s,
                available_s=len(x)/fs,selected_s=stop/fs,tail_limited=not complete,
                reason='all supported bands have usable T30 and noise tail' if complete else
                       'available tail exhausted; unresolved bands remain unqualified',steps=steps)
        stop=min(len(x),max(stop+1,round(stop*1.5)))


def first_reflection_delays(source_xyz_m, receiver_xyz_m, dims_m, obstacles, sound_speed_m_s):
    """Lower bound on reflected excess travel, shared by REF and training anchors.

    A box is inside each supporting halfspace. If both endpoints are outside
    the same face, the infinite-plane mirror path bounds every path touching
    the box, including edge paths. Combine valid lower bounds by taking their
    maximum. This retains lateral travel omitted by the endpoint-distance sum
    without pretending a finite face necessarily contains the specular point.
    """
    source = np.asarray(source_xyz_m, dtype=float)
    delays = []
    for receiver in np.atleast_2d(receiver_xyz_m):
        distance = np.linalg.norm(receiver-source)
        paths = []
        for axis, length in enumerate(dims_m):
            for plane in (0., length):
                image = source.copy()
                image[axis] = 2*plane-source[axis]
                paths.append(np.linalg.norm(receiver-image))
        for box in obstacles:
            lo, hi = np.array(box.min_xyz), np.array(box.min_xyz)+box.size_xyz
            bound = sum(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0))
                        for p in (source, receiver))
            for axis in range(3):
                for plane, side in ((lo[axis], -1), (hi[axis], 1)):
                    if (side*(source[axis]-plane) >= 0 and
                            side*(receiver[axis]-plane) >= 0):
                        image = source.copy()
                        image[axis] = 2*plane-source[axis]
                        bound = max(bound, np.linalg.norm(receiver-image))
            paths.append(bound)
        delays.append(float(max(0., (min(paths)-distance)/sound_speed_m_s)))
    return delays


def reference_direct_window(samples, fs, direct, *, pre_s=.0025, post_s=.01,
                            first_reflection_delay_s=None):
    """Quasi-anechoic gate, with an explicit inverse-duration LF limit.

    REW impulse-response windows / Klippel Training 8: stop before the first
    reflection; zero padding cannot restore low-frequency resolution. A 0.25 ms
    guard and cosine end taper reduce reflection onset/ringing contamination.
    This is a geometry-conditioned bound, not measured anechoic calibration.
    """
    if not isinstance(direct, (int, np.integer)) or not 0 <= direct < len(samples):
        raise ValueError('invalid REF direct sample')
    post = round(post_s*fs)
    if first_reflection_delay_s is not None:
        if not np.isfinite(first_reflection_delay_s) or first_reflection_delay_s <= .00025:
            raise ValueError('REF has no resolved direct/reflection separation')
        post = min(post, int(np.floor((first_reflection_delay_s-.00025)*fs)))
    if post < 1 or direct+post >= len(samples):
        raise ValueError('REF post-arrival window must be fully recorded')
    start, stop = max(0, direct-round(pre_s*fs)), direct+post+1
    gate = np.asarray(samples[start:stop], float).copy()
    left = min(round(.001*fs), max(0, direct-start)//4)
    right = min(round(.001*fs), post//4)
    if left:
        gate[:left] *= .5-.5*np.cos(np.linspace(0, np.pi, left))
    if right:
        gate[-right:] *= .5+.5*np.cos(np.linspace(0, np.pi, right))
    return gate, start, stop, fs/max(1, post-right)


def estimate_reference_eq(rir: RIR, *, channel: int = 0, direct_sample: int | None = None,
                           pre_s: float = .0025, post_s: float = .01,
                           reference_distance_m: float = .3, regularization: float = 1e-4,
                           max_gain_db: float = 12, band_hz: tuple[float, float] = (200, 8000),
                           nfft: int = 4096) -> ReferenceEQ:
    """Direct-gated REF speaker+mic estimate and regularized complex correction.

    Removes 1/r spreading relative to 1 m, NOT calibrated absolute SPL; retains
    device gain. It cannot separate speaker from mic. 0.3 m may be near field,
    on-axis response differs off-axis, gating limits LF resolution, reflections
    in the gate contaminate EQ. Apply only supported bins; don't deconvolve a real
    source with speaker EQ a second time. Speaker vs real-source response remains
    a physical ambiguity for the dry-source owner to handle explicitly.
    """
    options = [pre_s, post_s, reference_distance_m, regularization, max_gain_db, *band_hz]
    if not np.all(np.isfinite(options)) or pre_s < 0 or min(post_s, reference_distance_m, regularization) <= 0 or max_gain_db < 0:
        raise ValueError("invalid REF options")
    if not 0 < band_hz[0] < band_hz[1] < rir.fs / 2 or not isinstance(nfft, int) or nfft < 2:
        raise ValueError("invalid REF band or FFT size")
    x = rir.samples
    if x.ndim == 2:
        if not 0 <= channel < x.shape[1]:
            raise ValueError("invalid REF channel")
        x = x[:, channel]
    elif channel != 0:
        raise ValueError("mono REF only has channel 0")
    direct = int(np.argmax(np.abs(x))) if direct_sample is None else direct_sample
    if not isinstance(direct, (int, np.integer)) or not 0 <= direct < len(x):
        raise ValueError("invalid REF direct sample")
    delays = rir.metadata.get('first_reflection_delay_s')
    reflection = None if delays is None else delays[channel]
    gate, start, stop, low_hz = reference_direct_window(
        x, rir.fs, direct, pre_s=pre_s, post_s=post_s, first_reflection_delay_s=reflection)
    if len(gate) > nfft:
        raise ValueError("nfft must contain full REF direct window")
    if not np.any(gate):
        raise ValueError("REF gate has zero energy")
    freq = np.fft.rfftfreq(nfft, 1 / rir.fs)
    response = rfft(gate, nfft) * reference_distance_m * np.exp(2j * np.pi * freq * (direct - start) / rir.fs)
    power = np.abs(response) ** 2
    correction = response.conj() / (power + regularization * power.max())
    correction *= np.minimum(1, 10 ** (max_gain_db / 20) / np.maximum(np.abs(correction), np.finfo(float).tiny))
    supported = (freq >= max(band_hz[0], low_hz)) & (freq <= band_hz[1]) & (power >= 10 * regularization * power.max())
    correction[~supported] = 0
    return ReferenceEQ(rir.fs, nfft, freq, response, correction, supported,
                       {"method": "reflection_bounded_regularized_complex_inverse", "model_version": "2",
                        "regularization": regularization, "max_gain_db": max_gain_db,
                        "direct_sample": int(direct), "gate_samples": [start, stop],
                        "first_reflection_delay_s": reflection,
                        "reflection_bound_verified": reflection is not None,
                        "window_low_frequency_hz": low_hz,
                        "supported_band_hz": [float(freq[supported][0]), float(freq[supported][-1])] if supported.any() else None,
                        "window_duration_s": (stop-start)/rir.fs,
                        "reference_distance_m": reference_distance_m,
                        "limits": "combined speaker+mic only; near-field and on-axis; low-frequency gate/SNR limits; no absolute calibration"})
