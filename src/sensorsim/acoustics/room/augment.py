"""Seeded, explicitly labelled source and receiver perturbations (mono signals).

Pitch and duration controls are intentionally coupled resampling, not a phase vocoder.
They approximate playback/motor speed variation, not an independent pitch-preserving edit.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy import signal

from sensorsim.acoustics.room.model import OCTAVE_HZ
from sensorsim.core.messages import ModelValidity

VALIDITY = ModelValidity(
    name="Room training source and microphone augmentation",
    status="partially validated", freq_range_hz=(125., 8000.),
    conditions="Numerical transforms only; no measured source/microphone population model. "
    "Resampling couples pitch and duration; linear time-varying delay is a motion proxy "
    "without time-varying reflected paths or convective amplification. Noise SNR uses "
    "whole clean-clip RMS, not a physical calibrated sound level.",
    citation="SciPy resample_poly; RBJ Audio EQ Cookbook (W3C Audio EQ Cookbook); "
    "white/pink synthetic noise only in this distribution",
)


class EQBand(BaseModel):
    """Peaking EQ approximates one source/body or microphone resonance."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    frequency_hz: float = Field(gt=0.)
    gain_db: float = Field(ge=-36., le=36.)
    q: float = Field(default=1., gt=0., le=50.)


class TrainingLevelPolicy(BaseModel):
    """Opt-in complete-mixture export gain, separate from physical source gain.

    RMS is whole rendered mixture RMS including the retained room tail. It is
    digital full scale, never SPL. SNR is selected before this common gain.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    mode: Literal['preserve', 'rms', 'random_gain'] = 'preserve'
    target_rms_dbfs: float = Field(default=-24., ge=-100., le=0.)
    gain_db: tuple[float, float] = (-6., 6.)

    @model_validator(mode='after')
    def _bounds(self):
        if not -60 <= self.gain_db[0] <= self.gain_db[1] <= 60:
            raise ValueError('training gain must be ordered within +/-60 dB')
        return self


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    gain_db: tuple[float, float] = (0., 0.)
    tilt_db_octave: tuple[float, float] = (0., 0.)
    eq: tuple[EQBand, ...] = ()
    pitch_semitones: tuple[float, float] = (0., 0.)
    stretch_rate: tuple[float, float] = (1., 1.)
    doppler_delay_s: tuple[float, float] | None = None
    mic_response_std_db: float = Field(default=0., ge=0., le=12.)
    noise_snr_schedule: tuple[tuple[float, float], ...] = ()  # (seconds, dB)
    noise_spectrum: Literal["white", "pink"] = "white"
    training_level: TrainingLevelPolicy = Field(default_factory=TrainingLevelPolicy,
        exclude_if=lambda value: value == TrainingLevelPolicy())

    @field_validator("noise_spectrum", mode="before")
    @classmethod
    def _withheld_spectra(cls, value):
        if value in ("traffic", "park"):
            raise ValueError(
                f"noise_spectrum={value!r} is not available in this distribution: its spectrum came "
                "from an ISO 717-1 table / the CC BY-SA DEMAND dataset and was withheld (THIRD_PARTY.md). "
                "Use 'white' or 'pink'. For realistic background, record your own room and add it "
                "with `python -m sensorsim.recognizer add-background` (USAGE.md step 7).")
        return value

    @model_validator(mode="after")
    def _ranges(self) -> "AugmentationConfig":
        for name, lo, hi in (("gain_db", -60, 60), ("tilt_db_octave", -12, 12),
                             ("pitch_semitones", -24, 24), ("stretch_rate", .25, 4)):
            a, b = getattr(self, name)
            if not lo <= a <= b <= hi:
                raise ValueError(f"{name} must be ordered within [{lo},{hi}]")
        if self.doppler_delay_s is not None and min(self.doppler_delay_s) < 0:
            raise ValueError("Doppler delays must be nonnegative")
        if self.noise_snr_schedule:
            t, snr = np.array(self.noise_snr_schedule).T
            if t[0] != 0 or np.any(np.diff(t) <= 0) or np.any(np.abs(snr) > 120):
                raise ValueError("SNR schedule must start at 0, increase in time, and stay within ±120 dB")
        return self


def _audio(samples, fs: int) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("audio must be a finite mono array with at least two samples")
    if not isinstance(fs, (int, np.integer)) or fs < 1000:
        raise ValueError("fs must be an integer >=1000 Hz")
    return x


def gain(samples, db: float) -> np.ndarray:
    """Source level/electronic gain; dB is a pressure ratio (20 log10)."""
    if not np.isfinite(db) or abs(db) > 120:
        raise ValueError("gain must be finite within ±120 dB")
    return np.asarray(samples, float) * 10**(db/20)


def training_export_level(samples, policy: TrainingLevelPolicy, *, seed: int) -> tuple[np.ndarray, dict]:
    """Common post-mixture scalar, reversible before PCM quantization/clipping.

    Random gain uses a separate seed namespace supplied by the dataset writer;
    enabling this must not change source, room, noise or sample-selection RNGs.
    """
    x = np.asarray(samples, float)
    if not isinstance(policy, TrainingLevelPolicy):
        raise TypeError('typed TrainingLevelPolicy required')
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('finite mono complete mixture required')
    rms = float(np.sqrt(np.mean(x*x)))
    if policy.mode == 'rms' and rms == 0:
        raise ValueError('silent mixture has no RMS normalization gain')
    db = (policy.target_rms_dbfs-20*np.log10(rms) if policy.mode == 'rms' else
          float(np.random.default_rng(seed).uniform(*policy.gain_db)) if policy.mode == 'random_gain' else 0.)
    y = gain(x, db)
    return y, dict(version='room-training-level/1', policy=policy.model_dump(mode='json'),
        seed=int(seed), amplitude_scale=float(10**(db/20)), gain_db=float(db),
        pre_level_rms=rms, pre_level_peak=float(np.max(np.abs(x))),
        domain='peak_fs', applied_to='complete rendered signal plus receiver noise',
        snr_definition='whole clean clip unweighted RMS; selected before common export gain',
        relative_levels_preserved=policy.mode == 'preserve')


def parametric_eq(samples, fs: int, bands: tuple[EQBand, ...]) -> np.ndarray:
    """Causal RBJ peaking biquads; phase and start-up transient are part of the edit."""
    x = _audio(samples, fs).copy()
    for band in bands:
        if band.frequency_hz >= fs/2:
            raise ValueError("EQ centre must lie below Nyquist")
        w = 2*np.pi*band.frequency_hz/fs
        a, alpha = 10**(band.gain_db/40), np.sin(w)/(2*band.q)
        b = [1+alpha*a, -2*np.cos(w), 1-alpha*a]
        denominator = [1+alpha/a, -2*np.cos(w), 1-alpha/a]
        x = signal.lfilter(b, denominator, x)
    return x


def _response(samples, fs, frequencies, db) -> np.ndarray:
    x = _audio(samples, fs)
    frequencies, db = np.asarray(frequencies), np.asarray(db)
    grid = np.linspace(0, fs/2, 513)
    response = 10**(np.interp(np.log2(np.maximum(grid, 1)), np.log2(frequencies), db)/20)
    fir = signal.firwin2(257, grid, response, fs=fs)
    return signal.fftconvolve(x, fir, mode="full")[128:128+len(x)]


def spectral_tilt(samples, fs: int, db_per_octave: float) -> np.ndarray:
    """Zero-phase source colour about 1 kHz; response held flat outside 125–8000 Hz."""
    if not np.isfinite(db_per_octave) or abs(db_per_octave) > 12:
        raise ValueError("tilt must be finite within ±12 dB/octave")
    if db_per_octave == 0:
        return _audio(samples, fs).copy()
    return _response(samples, fs, OCTAVE_HZ, db_per_octave*np.log2(np.array(OCTAVE_HZ)/1000))


def resample_speed(samples, fs: int, rate: float) -> np.ndarray:
    """Playback/motor speed: pitch multiplies by rate, duration divides by rate.

    Polyphase anti-aliasing removes bandwidth on speed-up. Rational approximation
    (denominator <=1000), edge transients and sample rounding are documented artefacts.
    """
    x = _audio(samples, fs)
    if not np.isfinite(rate) or not .0625 <= rate <= 16:
        raise ValueError("resampling rate must be within [0.0625,16]")
    ratio = Fraction(1/rate).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def pitch_shift(samples, fs: int, semitones: float) -> np.ndarray:
    """Resample-based pitch shift: also changes duration by 2**(-semitones/12)."""
    if not np.isfinite(semitones) or abs(semitones) > 48:
        raise ValueError("pitch shift must be finite within ±48 semitones")
    return resample_speed(samples, fs, 2**(semitones/12))


def time_stretch(samples, fs: int, rate: float) -> np.ndarray:
    """Rate>1 shortens the event AND raises pitch; no independent pitch preservation."""
    return resample_speed(samples, fs, rate)


def doppler(samples, fs: int, delay_start_s: float, delay_end_s: float) -> np.ndarray:
    """Receiver-time varying delay x(t-tau(t)); linear interpolation, zero outside clip.

    Constant range velocity proxy v/c=d(tau)/dt. No gain change, updated room
    reflections or convective amplification. Output length includes the final delay.
    Approaching motion lowpasses the input to the largest local playback rate's
    Nyquist limit (257-tap Kaiser FIR); this also limits the held-delay tail's
    bandwidth. Linear interpolation still has frequency-dependent droop.
    """
    x = _audio(samples, fs)
    if not np.isfinite([delay_start_s, delay_end_s]).all() or min(delay_start_s, delay_end_s) < 0:
        raise ValueError("delays must be finite and nonnegative")
    if max(delay_start_s, delay_end_s) > 10:
        raise ValueError("delay exceeds bounded ten-second augmentation budget")
    delay_samples = np.array([delay_start_s, delay_end_s]) * fs
    # Decimal seconds for an integer sample delay can round just above that
    # integer. Avoid an extra sample and losing the last real sample to `right=0`.
    rounded = np.rint(delay_samples)
    delay_samples = np.where(abs(delay_samples-rounded) < 1e-9, rounded, delay_samples)
    slope = (delay_samples[1]-delay_samples[0])/(len(x)-1)
    if abs(slope) >= 1:
        raise ValueError("delay slope must have magnitude <1 (subsonic proxy)")
    if slope < 0:
        # x(t-tau(t)) advances at 1-tau' samples per receiver sample. Linear
        # interpolation alone is not an anti-alias filter for a rate above one.
        fir = signal.firwin(257, 1/(1-slope), window=("kaiser", 8.6))
        x = signal.fftconvolve(x, fir, mode="full")[128:128+len(x)]
    # Hold end delay after the original clip duration; preserve the delayed tail.
    sample = np.arange(len(x)+int(np.ceil(delay_samples[1])))
    delay = np.interp(sample, [0, len(x)-1], delay_samples)
    return np.interp(sample-delay, np.arange(len(x)), x, left=0., right=0.)


def mic_response(samples, fs: int, std_db: float, *, seed: int) -> tuple[np.ndarray, list[float]]:
    """Independent octave sensitivity offsets; no directivity, AGC or head shadow."""
    if not np.isfinite(std_db) or not 0 <= std_db <= 12:
        raise ValueError("microphone response std must lie within [0,12] dB")
    db = np.random.default_rng(seed).normal(0, std_db, 7)
    y = _response(samples, fs, OCTAVE_HZ, db) if std_db else _audio(samples, fs).copy()
    return y, db.tolist()


def additive_noise(samples, fs: int, schedule, *, seed: int, spectrum: str = "white"
                   ) -> tuple[np.ndarray, dict]:
    """Ambient/electronic noise; schedule dB refers to the entire clean clip's RMS.

    Linear interpolation in dB, not local window SNR. Silence with requested SNR
    is undefined and rejected. White or pink (1/f) spectra only in this distribution.
    """
    x = _audio(samples, fs)
    checked = AugmentationConfig(noise_snr_schedule=schedule, noise_spectrum=spectrum)
    if not checked.noise_snr_schedule:
        return x.copy(), {"enabled": False}
    rms = np.sqrt(np.mean(x*x))
    if rms == 0:
        raise ValueError("SNR relative to a silent clip is undefined")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x))
    f = np.fft.rfftfreq(len(x), 1/fs)
    if checked.noise_spectrum == "pink":
        psd = 1/np.maximum(f, fs/len(x))
        noise = np.fft.irfft(np.fft.rfft(noise)*np.sqrt(psd), len(x))
    noise -= noise.mean()
    noise /= np.sqrt(np.mean(noise**2))
    times, levels = np.array(checked.noise_snr_schedule).T
    snr = np.interp(np.arange(len(x))/fs, times, levels)
    noise *= rms*10**(-snr/20)
    return x+noise, {"enabled": True, "spectrum": spectrum, "schedule": list(schedule),
        "reference": "whole clean clip unweighted RMS",
        "clean_rms": float(rms), "noise_rms": float(np.sqrt(np.mean(noise**2))),
        "realised_global_snr_db": float(20*np.log10(rms/np.sqrt(np.mean(noise**2))))}


def augment_source(samples, fs: int, config: AugmentationConfig, *, seed: int
                    ) -> tuple[np.ndarray, dict]:
    """Source transforms BEFORE room convolution, with every realised choice returned."""
    rng = np.random.default_rng(seed)
    chosen = {key: float(rng.uniform(*getattr(config, key))) for key in
              ("gain_db", "tilt_db_octave", "pitch_semitones", "stretch_rate")}
    rate = 2**(chosen["pitch_semitones"]/12)*chosen["stretch_rate"]
    x = gain(_audio(samples, fs), chosen["gain_db"])
    x = spectral_tilt(parametric_eq(x, fs, config.eq), fs, chosen["tilt_db_octave"])
    x = resample_speed(x, fs, rate) if rate != 1 else x
    if config.doppler_delay_s is not None:
        x = doppler(x, fs, *config.doppler_delay_s)
    chosen.update(seed=int(seed), eq=[b.model_dump() for b in config.eq],
        doppler_delay_s=config.doppler_delay_s, coupled_speed_rate=rate,
        rng_policy="numpy-PCG64/1",
        resampling_policy="scipy.resample_poly; rational denominator<=1000; centered FIR; zero boundary",
        delay_policy="receiver grid t=n/fs; x(t-delay); linear interpolation; zero outside; noncircular",
        doppler_filter="257-tap Kaiser anti-alias for decreasing delay; linear interpolation",
        meaning="gain/colour/speed at source; pitch and duration coupled; motion proxy before static RIR")
    return x, chosen


def augment_receiver(samples, fs: int, config: AugmentationConfig, *, seed: int
                      ) -> tuple[np.ndarray, dict]:
    """Microphone response then additive receiver noise, AFTER room convolution."""
    seeds = np.random.SeedSequence(seed).generate_state(2)
    x, db = mic_response(samples, fs, config.mic_response_std_db, seed=int(seeds[0]))
    x, noise = additive_noise(x, fs, config.noise_snr_schedule, seed=int(seeds[1]),
                              spectrum=config.noise_spectrum)
    return x, {"rng_policy": "numpy-SeedSequence-positional-PCG64/1", "seed": int(seed), "mic_seed": int(seeds[0]), "noise_seed": int(seeds[1]),
               "mic_octave_db": db, "noise": noise}
