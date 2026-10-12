"""Simulated capture chains for Pi MEMS and other IoT microphones.

The versioned Pi profiles model a single absolute-level deployment chain. ``random_chain(rng)``
is retained only for an explicitly selected stress split. Other hand-specified profiles remain
available for evaluation. ``out_of_range`` sits outside the random-chain ranges on purpose, to
show where robustness ends.

A legacy chain can model a capsule and housing response, noise relative to ``ref_rms``, ADC
overload, AGC, noise suppression, codec and packet loss. The versioned Pi profiles use datasheet
sensitivity and an absolute A-weighted noise floor instead; they do not route through those
processing stages. Grounding:

* MEMS capsule, e.g. TDK InvenSense INMP441: flat 60 Hz-15 kHz, SNR 61 dBA, sensitivity
  -26 dBFS at 94 dB SPL (datasheet). Cheap electret capsules: SNR around 58 dBA.
* Housing: the cavity behind a small sound hole forms a Helmholtz resonator; cheap
  enclosures and long gasket ports pull its peak down into the kHz band (e.g.
  STMicroelectronics app. note on gasket design for MEMS microphones).
* Codecs: DECT baby monitors use ITU-T G.726 ADPCM at 32 kbit/s (8 kHz, narrowband) or G.722
  (16 kHz, 50-7000 Hz); most IP cameras send G.711 PCMU/PCMA at 8 kHz.

Proxies, stated plainly: G.711 is modelled as continuous mu-law (mu = 255) quantized to 8 bits;
G.726 as a project ADPCM-like stand-in, not IMA bit-exact, at 8 kHz (step-size
adaptation and slope overload on sharp onsets); G.722 as a 50-7000 Hz band limit. AGC and
noise suppression are generic designs, not any vendor's algorithm. Opus/AAC are not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import warnings

import numpy as np
from scipy import ndimage, signal

from sensorsim.acoustics.weighting import a_weighted_mean_square

CODECS = ("pcm16", "mulaw8k", "adpcm8k", "bandlimit7k")


@dataclass(frozen=True)
class MicChain:
    """One device's capture chain.

    Legacy chains reference self-noise to ``ref_rms``. Versioned Pi MEMS profiles instead
    interpret an input sine with peak 1.0 as the declared SPL reference and use an absolute
    output sensitivity and A-weighted noise floor.
    """
    hp_hz: float = 60.                         # capsule / DC-blocking high-pass corner
    shelf_hz: float = 4000.                    # grille or fabric: treble shelf corner
    shelf_db: float = 0.                       # negative = muffled
    resonances: tuple = ()                     # ((freq_hz, gain_db, q), ...) acoustic-port peaks
    self_noise_db: float = -40.                # capsule + preamp noise, dB relative to ref_rms
    drive_db: float = 0.                       # gain before the ADC; > 0 makes loud sounds clip
    dc_offset: float = 0.                      # constant offset, fraction of full scale (raw MEMS / I2S)
    agc: dict | None = None                    # {target_dbfs, max_gain_db, attack_s, release_s}
    noise_suppression: dict | None = None      # {alpha, floor_db}
    codec: str | None = "pcm16"
    packet_loss: float = 0.                    # fraction of 20 ms frames lost
    concealment: str = "zero"                  # "zero" or "repeat"
    name: str = "custom"
    profile_id: str | None = None
    sensitivity_dbfs_at_94db_spl: float | None = None
    self_noise_dbfs_a: float | None = None
    snr_db_a: float | None = None
    acoustic_corner_hz: float | None = None
    dc_block_hz_at_48k: float | None = None
    passband_fraction_fs: float | None = None
    max_spl_peak_db: float | None = None
    adc_bits: int | None = None
    capture_bits: int | None = None
    evidence_note: str | None = None

    def apply(self, x: np.ndarray, fs: int, rng: np.random.Generator, *, ref_rms: float | None = None,
              input_reference_spl_db: float = 94.0,
              include_self_noise: bool = True, quantize: bool = True) -> np.ndarray:
        """Apply this chain to mono float audio; return same-length 16-bit PCM as float.

        For absolute profiles, input amplitude is pressure-relative: a peak-1.0 sine has
        ``input_reference_spl_db`` dB SPL. For legacy chains, ``ref_rms`` continues to set
        the relative self-noise reference. ``include_self_noise=False, quantize=False``
        gives the linear event path for mixing with an already captured background;
        quantize the combined signal once. These options apply to absolute profiles only.
        """
        if self.name == "pi5_mems_unfiltered":
            warnings.warn("pi5_mems_unfiltered is deprecated: it now means the absolute-SPL "
                          "pi5_inmp441/1 response; ref_rms is ignored. Set input_reference_spl_db.",
                          DeprecationWarning, stacklevel=2)
        x = np.asarray(x, dtype=float)
        if self.sensitivity_dbfs_at_94db_spl is not None:
            return _apply_absolute_profile(self, x, fs, rng, input_reference_spl_db,
                                           include_self_noise=include_self_noise, quantize=quantize)
        ref = float(ref_rms) if ref_rms else float(np.sqrt(np.mean(x*x)) or 1e-4)
        y = signal.sosfilt(_capsule_sos(self, fs), x)
        y = y + ref*10**(self.self_noise_db/20)*rng.standard_normal(y.size)
        y = np.clip(y*10**(self.drive_db/20) + self.dc_offset, -1., 1.)    # ADC overload; offset
        if self.agc:
            y = agc(y, fs, **self.agc)
        if self.noise_suppression:
            y = noise_suppress(y, fs, **self.noise_suppression)
        if self.codec is not None:
            y = codec(y, fs, self.codec)
        if self.packet_loss > 0:
            y = packet_loss(y, fs, self.packet_loss, rng, concealment=self.concealment)
        return np.clip(np.round(y*32767)/32767, -1., 1.)                   # stored as 16-bit PCM

    def with_variation(self, rng: np.random.Generator) -> "MicChain":
        """Return a bounded profile draw; sensitivity and the SPL reference stay fixed."""
        if self.profile_id is None:
            return self
        updates = {"self_noise_dbfs_a": float(self.self_noise_dbfs_a + rng.uniform(-2.0, 2.0))}
        if self.profile_id == "pi5_inmp441/1":
            corner = float(rng.uniform(50.0, 80.0))
            updates.update(hp_hz=corner, acoustic_corner_hz=corner)
        elif self.profile_id == "pi5_sph0645/1":
            corner = float(rng.uniform(40.0, 50.0))
            updates.update(hp_hz=corner, acoustic_corner_hz=corner)
        return replace(self, **updates)


@lru_cache(maxsize=256)
def _butter_sos(order, cutoff, kind, designer) -> np.ndarray:
    """Reuse exact designs, never filter state; include the active design hook.

    A varied capsule still shares its DC block with other examples, and its
    acoustic section with its own calibration and noise paths. Cutoffs are the
    original normalized values, without rounding. Callers copy/stack these
    private coefficients before exposing them or passing them to a filter.
    """
    sos = designer(order, cutoff, kind, output="sos")
    sos.setflags(write=False)
    return sos


def _capsule_sos(chain: MicChain, fs: int) -> np.ndarray:
    nyq = fs/2
    sections = []
    highpass_hz = chain.acoustic_corner_hz if chain.acoustic_corner_hz is not None else chain.hp_hz
    if highpass_hz > 0:
        sections.append(_butter_sos(2, min(highpass_hz, .45*fs)/nyq, "highpass", signal.butter))
    if chain.dc_block_hz_at_48k:
        dc_corner = chain.dc_block_hz_at_48k*fs/48000.0
        sections.append(_butter_sos(2, min(dc_corner, .45*fs)/nyq, "highpass", signal.butter))
    if chain.shelf_db:
        sections.append(high_shelf(chain.shelf_hz, chain.shelf_db, fs))
    for f0, gain_db, q in chain.resonances:
        if f0 < .45*fs:
            sections.append(peaking(f0, gain_db, q, fs))
    return np.vstack(sections) if sections else np.array([[1., 0., 0., 1., 0., 0.]])


def a_weighting_sos(fs: int) -> np.ndarray:
    """Legacy bilinear approximation, retained for diagnostic comparisons only.

    Inaccurate near Nyquist: use :func:`a_weighted_rms` for level measurements.
    """
    w1, w2, w3, w4 = 2*np.pi*np.array([20.598997, 107.65265, 737.86223, 12194.217])
    zeros = np.zeros(4)
    poles = np.array([-w1, -w1, -w2, -w3, -w4, -w4])
    b, a = signal.zpk2tf(zeros, poles, w4**2)
    bz, az = signal.bilinear(b, a, fs=fs)
    _, at_1k = signal.freqz(bz, az, worN=[2*np.pi*1000/fs])
    bz = bz/abs(at_1k[0])
    return signal.tf2sos(bz, az)


def a_weighted_rms(x: np.ndarray, fs: int) -> float:
    """A-weighted RMS of a digital signal, in its input amplitude units."""
    x = np.asarray(x, dtype=float)
    if not x.size:
        return 0.0
    return float(np.sqrt(a_weighted_mean_square(x, fs)))


def _truncate_24_to_16(x: np.ndarray) -> np.ndarray:
    """Quantize to signed 24-bit PCM, discard eight low bits, and return S16 units."""
    q24 = np.rint(np.clip(x, -1.0, (2**23 - 1)/2**23)*2**23).astype(np.int32)
    q16 = (q24 >> 8).astype(np.int16)
    return q16.astype(np.float64)/32768.0


@lru_cache(maxsize=16)
def _decimation_fir(passband_fraction_fs: float) -> np.ndarray:
    """Model-based linear-phase decimator: flat passband, -6 dB at Nyquist.

    InvenSense DS-INMP441-00 Rev 1.1 specifies these magnitudes, not the taps or
    phase. A 129-tap interpolation models the transition without copying a curve.
    """
    return signal.firwin2(129, [0., 2*passband_fraction_fs, 1.],
                          [1., 1., 10**(-6/20)], window="hamming")


@lru_cache(maxsize=32)
def _calibration_fir_response(tap_bytes, fs, freqz) -> np.ndarray:
    """The fixed FIR's 1 kHz bin, independent of each varied capsule corner."""
    taps = np.frombuffer(tap_bytes, dtype=np.float64)
    _, response = freqz(taps, worN=np.array([1000.]), fs=fs)
    response.setflags(write=False)
    return response


def profile_response(chain: MicChain, fs: int, frequencies: np.ndarray) -> np.ndarray:
    """Complex response including the modelled decimator (before 1 kHz calibration)."""
    _, h = signal.sosfreqz(_capsule_sos(chain, fs), worN=frequencies, fs=fs)
    if chain.passband_fraction_fs:
        taps = _decimation_fir(chain.passband_fraction_fs)
        # Preserve the general frequency API (including float32 arithmetic).
        # Only the exact float64 calibration query shares this bounded cache.
        if (isinstance(frequencies, np.ndarray) and frequencies.dtype == np.float64
                and frequencies.shape == (1,) and frequencies[0] == 1000.
                and taps.ndim == 1 and taps.dtype == np.float64):
            # Include the actual coefficients: diagnostic callers can replace
            # the decimator or mutate its existing cached tap array.
            hf = _calibration_fir_response(taps.tobytes(), fs, signal.freqz)
        else:
            _, hf = signal.freqz(taps, worN=frequencies, fs=fs)
        h *= hf
    return h


def _filter_profile(chain: MicChain, x: np.ndarray, fs: int) -> np.ndarray:
    y = signal.sosfilt(_capsule_sos(chain, fs), x)
    if chain.passband_fraction_fs:
        # Causal FIR delay (64 samples) is tiny compared with event windows; lead-in
        # settles the filter. Keep the delay rather than inventing pre-event audio.
        y = signal.lfilter(_decimation_fir(chain.passband_fraction_fs), [1.], y)
    return y


def _apply_absolute_profile(chain: MicChain, x: np.ndarray, fs: int, rng: np.random.Generator,
                            input_reference_spl_db: float, *, include_self_noise: bool = True,
                            quantize: bool = True) -> np.ndarray:
    if not np.isfinite(input_reference_spl_db):
        raise ValueError("input_reference_spl_db must be finite")
    if not x.size:
        return x.copy()
    sensitivity_bin = profile_response(chain, fs, np.array([1000.]))
    response_scale = 1.0/max(float(abs(sensitivity_bin[0])), 1e-15)
    reference_dbfs = chain.sensitivity_dbfs_at_94db_spl + input_reference_spl_db - 94.0
    y = _filter_profile(chain, x, fs)*10**(reference_dbfs/20)*response_scale
    if include_self_noise:
        # The spectral shape remains an explicit proxy, not a measured device PSD.
        noise = _filter_profile(chain, rng.standard_normal(x.size), fs)*response_scale
        noise_rms_a = a_weighted_rms(noise, fs)
        if noise_rms_a:
            noise *= 10**(chain.self_noise_dbfs_a/20)/(np.sqrt(2)*noise_rms_a)
        y += noise
    if not quantize:
        return y
    return quantize_profile(chain, y)


def quantize_profile(chain: MicChain, audio: np.ndarray) -> np.ndarray:
    """Apply the profile's overload limit and 24-to-16 capture conversion once."""
    clip_level = (10**((chain.sensitivity_dbfs_at_94db_spl + chain.max_spl_peak_db - 94.0)/20)
                  if chain.max_spl_peak_db is not None else 1.0)
    clip_level = min(float(clip_level), 1.0)
    return _truncate_24_to_16(np.clip(audio, -clip_level, clip_level))


def peaking(f0: float, gain_db: float, q: float, fs: int) -> np.ndarray:
    """RBJ Audio EQ Cookbook peaking biquad as one SOS row."""
    a, w = 10**(gain_db/40), 2*np.pi*f0/fs
    alpha = np.sin(w)/(2*q)
    b = [1 + alpha*a, -2*np.cos(w), 1 - alpha*a]
    d = [1 + alpha/a, -2*np.cos(w), 1 - alpha/a]
    return np.array([[*(np.array(b)/d[0]), 1., d[1]/d[0], d[2]/d[0]]])


def high_shelf(f0: float, gain_db: float, fs: int, slope: float = 1.) -> np.ndarray:
    """RBJ Audio EQ Cookbook high shelf as one SOS row."""
    a, w = 10**(gain_db/40), 2*np.pi*min(f0, .45*fs)/fs
    alpha = np.sin(w)/2*np.sqrt((a + 1/a)*(1/slope - 1) + 2)
    c, sa = np.cos(w), 2*np.sqrt(a)*alpha
    b = [a*((a + 1) + (a - 1)*c + sa), -2*a*((a - 1) + (a + 1)*c), a*((a + 1) + (a - 1)*c - sa)]
    d = [(a + 1) - (a - 1)*c + sa, 2*((a - 1) - (a + 1)*c), (a + 1) - (a - 1)*c - sa]
    return np.array([[*(np.array(b)/d[0]), 1., d[1]/d[0], d[2]/d[0]]])


def agc(x: np.ndarray, fs: int, *, target_dbfs: float = -18., max_gain_db: float = 24., attack_s: float = .01,
        release_s: float = .4) -> np.ndarray:
    """Block AGC: fast gain cuts on loud onsets, slow recovery ("breathing" background); limiter at full scale."""
    block = max(1, round(.01*fs))
    n = -(-x.size//block)
    level = 20*np.log10(np.sqrt(np.mean(np.pad(x, (0, n*block - x.size)).reshape(n, block)**2, axis=1)) + 1e-9)
    desired = np.clip(target_dbfs - level, -24., max_gain_db)
    up, down = 1 - np.exp(-.01/release_s), 1 - np.exp(-.01/attack_s)
    gains, g = np.empty(n), 0.
    for i, want in enumerate(desired):
        g += (want - g)*(down if want < g else up)
        gains[i] = g
    per_sample = np.interp(np.arange(x.size), np.arange(n)*block + block/2, gains)
    return np.clip(x*10**(per_sample/20), -1., 1.)


def noise_suppress(x: np.ndarray, fs: int, *, alpha: float = 2., floor_db: float = -15.) -> np.ndarray:
    """Spectral subtraction with a minimum-statistics noise estimate (musical-noise artifacts included)."""
    n = 1 << int(np.ceil(np.log2(.032*fs)))
    hop = n//2
    window = np.sqrt(np.hanning(n + 1)[:n])
    pad = np.pad(x, (n, n + (-x.size) % hop))
    frames = np.lib.stride_tricks.sliding_window_view(pad, n)[::hop]*window
    spec = np.fft.rfft(frames, axis=1)
    power = np.abs(spec)**2
    smooth = signal.lfilter([.2], [1, -.8], power, axis=0)
    size = 2*max(1, round(.75*fs/hop)) + 1                     # trailing ~1.5 s window [i - size + 1, i]
    noise = ndimage.minimum_filter1d(smooth, size, axis=0, mode="nearest", origin=(size - 1)//2)
    gain = np.maximum(1 - alpha*1.5*noise/np.maximum(power, 1e-20), 10**(floor_db/20))
    out = np.fft.irfft(spec*gain, n, axis=1)*window
    y = np.zeros(pad.size)
    for i, frame in enumerate(out):
        y[i*hop:i*hop + n] += frame
    return y[n:n + x.size]


def mulaw(x: np.ndarray, mu: float = 255.) -> np.ndarray:
    """G.711-style: mu-law compand, quantize to 8 bits, expand."""
    y = np.sign(x)*np.log1p(mu*np.minimum(np.abs(x), 1.))/np.log1p(mu)
    y = np.round(y*127)/127
    return np.sign(y)*np.expm1(np.abs(y)*np.log1p(mu))/mu


def _adpcm_steps() -> np.ndarray:
    """Project geometric quantizer scale: round(7 * 1.1**i), strictly increasing."""
    steps = []
    for i in range(128):
        step = min(32767, max(round(7 * 1.1**i), steps[-1] + 1 if steps else 1))
        steps.append(step)
        if step == 32767:
            break
    return np.array(steps, dtype=np.int32)


_ADPCM_STEPS = _adpcm_steps()


def adpcm_like(x: np.ndarray) -> np.ndarray:
    """Project ADPCM-like stand-in, not IMA bit-exact or a G.726 implementation.

    IMA-style adaptation, project-generated step sizes, not bit-exact. A signed
    4-bit innovation (0, ±1..7, in quarter steps) updates a clipped predictor.
    Steps are round(7·1.1^i), strictly increasing, clipped at 32767. The index
    adaptation (-1 below code 4, else +2·(code-3)) gives the same increments as
    IMA/DVI ADPCM. It is written as a formula; no codec table is copied.
    Stress augmentation only.
    """
    pcm = np.clip(np.round(np.asarray(x)*32767), -32767, 32767).astype(int)
    out = np.empty(pcm.size)
    pred, index = 0, 0
    for i, sample in enumerate(pcm):
        step = int(_ADPCM_STEPS[index])
        innovation = int(np.clip(round((int(sample) - pred) * 4 / step), -7, 7))
        pred = max(-32767, min(32767, pred + round(innovation * step / 4)))
        magnitude = abs(innovation)
        change = -1 if magnitude < 4 else 2 * (magnitude - 3)
        index = max(0, min(len(_ADPCM_STEPS) - 1, index + change))
        out[i] = pred
    return out/32767


def ima_adpcm(x: np.ndarray) -> np.ndarray:
    """Compatibility name for adpcm_like: project stand-in, NOT IMA bit-exact."""
    return adpcm_like(x)


def codec(x: np.ndarray, fs: int, name: str) -> np.ndarray:
    """Pass through a transmission codec and return at the original rate."""
    if name == "pcm16":
        return x
    if name == "bandlimit7k":                                  # G.722-style wideband
        sos = _butter_sos(6, (50/(fs/2), min(7000., .45*fs)/(fs/2)), "bandpass", signal.butter)
        return signal.sosfilt(sos.copy(), x)
    if name not in ("mulaw8k", "adpcm8k"):
        raise ValueError(f"unknown codec {name!r}; known: {CODECS}")
    g = int(np.gcd(fs, 8000))
    narrow = signal.resample_poly(x, 8000//g, fs//g)            # anti-aliased down to telephone rate
    narrow = mulaw(narrow) if name == "mulaw8k" else adpcm_like(narrow)
    return signal.resample_poly(narrow, fs//g, 8000//g)[:x.size]


def packet_loss(x: np.ndarray, fs: int, rate: float, rng: np.random.Generator, *, concealment: str = "zero",
                frame_ms: float = 20.) -> np.ndarray:
    """Drop whole frames (Wi-Fi monitors), zero-filled or repeated from the previous frame."""
    n = max(1, round(frame_ms/1000*fs))
    y = x.copy()
    for start in np.flatnonzero(rng.random(-(-x.size//n)) < rate)*n:
        stop = min(start + n, x.size)
        y[start:stop] = 0. if concealment == "zero" or start < n else y[start - n:start - n + stop - start]
    return y


def random_chain(rng: np.random.Generator) -> MicChain:
    """TRAINING draw: broad ranges covering MEMS and electret IoT devices (not the eval archetypes)."""
    # Housing and port resonances, up to 7.5 kHz: a MEMS port behind a gasket can sit high in the band.
    resonances = tuple((float(rng.uniform(1500, 7500)), float(rng.uniform(2, 12)), float(rng.uniform(1, 4)))
                       for _ in range(int(rng.integers(0, 3))))
    codec_name = rng.choice(CODECS, p=[.35, .25, .25, .15])
    return MicChain(
        hp_hz=float(np.exp(rng.uniform(np.log(15), np.log(300)))),
        shelf_hz=float(rng.uniform(2000, 5000)), shelf_db=float(rng.uniform(-10, 3)),
        resonances=resonances,
        self_noise_db=float(rng.uniform(-30, -2)),
        drive_db=float(rng.uniform(0, 14)) if rng.random() < .15 else 0.,
        dc_offset=float(rng.uniform(-.05, .05)) if rng.random() < .3 else 0.,
        agc=dict(target_dbfs=float(rng.uniform(-24, -12)), max_gain_db=float(rng.uniform(6, 30)),
                 attack_s=float(rng.uniform(.003, .05)), release_s=float(rng.uniform(.1, 1.))) if rng.random() < .5 else None,
        noise_suppression=dict(alpha=float(rng.uniform(1, 3)), floor_db=float(rng.uniform(-20, -8)))
        if rng.random() < .3 else None,
        codec=str(codec_name),
        packet_loss=float(rng.uniform(.005, .03)) if rng.random() < .2 else 0.,
        concealment="zero" if rng.random() < .5 else "repeat",
        name="random")


_PI_INMP441 = MicChain(
    hp_hz=60.0, self_noise_dbfs_a=-87.0, codec=None, packet_loss=0.0,
    name="pi5_inmp441/1", profile_id="pi5_inmp441/1",
    sensitivity_dbfs_at_94db_spl=-26.0, snr_db_a=61.0, acoustic_corner_hz=60.0,
    dc_block_hz_at_48k=3.7, passband_fraction_fs=0.423, max_spl_peak_db=120.0,
    adc_bits=24, capture_bits=16,
    evidence_note="InvenSense DS-INMP441-00 Rev 1.1; FIR transition and white self-noise spectrum are model-based",
)

_PI_ICS43434 = MicChain(
    hp_hz=60.0, acoustic_corner_hz=60.0, self_noise_dbfs_a=-91.0, codec=None, packet_loss=0.0,
    name="pi5_ics43434/1", profile_id="pi5_ics43434/1",
    sensitivity_dbfs_at_94db_spl=-26.0, snr_db_a=65.0, max_spl_peak_db=120.0,
    adc_bits=24, capture_bits=16,
    evidence_note="TDK ICS-43434 product specification DS-000069 v1.2: 60 Hz LF cutoff. Separate DC block and decimation response omitted: coefficients and rate-specific corners unverified; acoustic HPF rejects DC. High-band response is not validated",
)

_PI_SPH0645 = MicChain(
    hp_hz=45.0, self_noise_dbfs_a=-91.0, codec=None, packet_loss=0.0,
    name="pi5_sph0645/1", profile_id="pi5_sph0645/1",
    sensitivity_dbfs_at_94db_spl=-26.0, snr_db_a=65.0, acoustic_corner_hz=45.0,
    max_spl_peak_db=120.0, adc_bits=24, capture_bits=16,
    evidence_note="Knowles SPH0645LM4H-B brief KAS-700-0137: 45 Hz roll-off, SNR and AOP; nominal sensitivity assumed at -26 dBFS/94 dB SPL. Separate DC block and decimation response omitted: coefficients and rate-specific corners unverified; acoustic HPF rejects DC. High-band response is not validated",
)


PROFILES: dict[str, MicChain] = {
    # A Pi with an INMP441-class I2S MEMS mic: flat, quiet, no processing, 16 kHz PCM.
    "pi_mems_i2s": MicChain(hp_hz=60., self_noise_db=-12., name="pi_mems_i2s"),
    "pi5_inmp441/1": _PI_INMP441,
    "pi5_ics43434/1": _PI_ICS43434,
    "pi5_sph0645/1": _PI_SPH0645,
    # Deprecated spelling retained as a corrected alias for source compatibility. This is not
    # an unfiltered capsule: it now uses the INMP441's 60 Hz acoustic corner and DC block.
    "pi5_mems_unfiltered": replace(_PI_INMP441, name="pi5_mems_unfiltered"),
    # DECT audio baby monitor: electret in a plastic housing, port resonance, AGC, G.726-class ADPCM.
    "baby_monitor_dect": MicChain(hp_hz=150., resonances=((3400., 7., 2.),), shelf_hz=4000., shelf_db=-3.,
                                  self_noise_db=-7., agc=dict(target_dbfs=-18., max_gain_db=24., attack_s=.008,
                                                              release_s=.5), codec="adpcm8k", name="baby_monitor_dect"),
    # Wi-Fi camera / video baby monitor: MEMS behind a grille, AGC + noise suppression, G.711 PCMU, packet loss.
    "wifi_camera_g711": MicChain(hp_hz=100., shelf_hz=3000., shelf_db=-6., self_noise_db=-10.,
                                 agc=dict(target_dbfs=-20., max_gain_db=18., attack_s=.02, release_s=.3),
                                 noise_suppression=dict(alpha=2., floor_db=-14.), codec="mulaw8k",
                                 packet_loss=.01, concealment="zero", name="wifi_camera_g711"),
    # Wideband (G.722-class) monitor with a cheap electret and a strong enclosure peak.
    "wideband_monitor": MicChain(hp_hz=120., resonances=((2600., 9., 1.5), (5200., 4., 3.)), self_noise_db=-5.,
                                 codec="bandlimit7k", name="wideband_monitor"),
    # Cheap USB electret, gain set too high: loud sounds clip, high self-noise.
    "cheap_usb_electret": MicChain(hp_hz=250., resonances=((2800., 9., 2.),), self_noise_db=-3., drive_db=12.,
                                   name="cheap_usb_electret"),
    # OUTSIDE the training ranges on purpose: mic buried in fabric (1.2 kHz muffling), noise above the room.
    "out_of_range": MicChain(hp_hz=80., shelf_hz=1200., shelf_db=-24., self_noise_db=4., codec="mulaw8k",
                             name="out_of_range"),
}


def describe(chain: MicChain) -> dict:
    """JSON-ready parameters, for evaluation reports."""
    return {k: v for k, v in chain.__dict__.items()}


__all__ = ["MicChain", "PROFILES", "random_chain", "agc", "noise_suppress", "mulaw", "adpcm_like", "ima_adpcm", "codec",
           "packet_loss", "peaking", "high_shelf", "a_weighted_rms", "a_weighting_sos", "describe"]
