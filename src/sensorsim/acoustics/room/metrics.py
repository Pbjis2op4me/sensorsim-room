"""Room-acoustic parameters: ISO 3382-1 ranges, Schroeder/Chu-Lundeby decay.

Numerical CPU reference, not an IEC-certified measurement instrument. 6th-order
Butterworth prototype octave filters (12 poles). How a band is analysed is the metric
policy (METRIC_POLICY, room-measured-metrics/2):

* decay (EDT, T20, T30): the band filter runs over the time-reversed response, so its
  ringing falls before the onset instead of into the decay (time-reversed analysis,
  Jacobsen and Rindel 1987); exact for exponential decays at every band;
* energy (C50, C80, D50, Ts): zero-phase band filtering, integrated from the onset minus
  the filter's span, so neither group delay nor pre-ringing moves energy across the
  50/80 ms boundary;
* DRR: the broadband response is split at +-2.5 ms before filtering each part (drr/2).

room-measured-metrics/1 (causal filter, energy integrated from the broadband onset)
remains selectable for frozen reproductions. Its low-band bias is in MATH.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt

from sensorsim.core.messages import ModelValidity
from sensorsim.acoustics.bands import BandGrid, BandSpec
from .rir import _fs, _samples, detect_onset, estimate_noise_power

OCTAVE_BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
METRIC_POLICY = "room-measured-metrics/2"
LEGACY_METRIC_POLICY = "room-measured-metrics/1"
DRR_DEFINITION = "drr/2"
LEGACY_DRR_DEFINITION = "drr/1"
DRR_VALIDITY = ModelValidity(
    name="window-before-filter octave direct-to-reverberant ratio", status="partially validated",
    freq_range_hz=(125, 8000),
    conditions="Analytic isolated impulse, reflection and finite diffuse-tail energy controls. "
    "Broadband input; resolved direct packet inside +/-2.5 ms and reflections outside. "
    "Window overlap, colored boundary noise and measured-room separation remain unvalidated.",
    citation="Eaton et al. 2016 doi:10.1109/TASLP.2016.2577502; "
    "Meyer-Kahlen and Deppisch, EUSIPCO 2025, pp.321-325; parent-project note ROOM_DRR_METRIC (not distributed)",
)
VALIDITY = ModelValidity(
    name="room octave decay and energy metrics", status="partially validated",
    freq_range_hz=(125, 8000), conditions="Synthetic exponential decays here; IEC periodic frequency-response "
    "checks were in the parent project and the standard table is not included here; stationary noise, resolved monotonic late decay. "
    "room-measured-metrics/2: time-reversed decay analysis is exact for exponential decays in "
    "expectation at every band (/1, causal: +4.32% T20 at 125 Hz in the parent's ensemble, up to "
    "+64% EDT); zero-phase energy metrics within 1.1 dB C50 at 125 Hz and 0.2 dB above (MATH.md). "
    "Single short low-band decays keep their statistical scatter (about 20% EDT at 125 Hz). "
    "No full IEC class-1 certification or measured-room validation.",
    citation="ISO 3382-1:2009; IEC 61260-3:2016 Table 1; Lundeby et al., Acustica 81 (1995) 344-355; Chu (1978)",
)


def exact_center(band: int) -> float:
    if band not in OCTAVE_BANDS:
        raise ValueError("band must be a nominal octave 125..8000 Hz")
    return float(1000 * 10 ** (0.3 * (OCTAVE_BANDS.index(band) - 3)))


def metric_policy(band=None, *, onset_sample=None, legacy_drr=False) -> dict:
    """Report both the unchanged decay convention and the selected DRR definition."""
    bands = () if band in (None, "broadband") else (BandSpec("base10", 1, exact_center(band), float(band)),)
    grid = BandGrid(bands, f"{METRIC_POLICY}: Butterworth, time-reversed (decay) and zero-phase (energy)",
                    "pressure", 6 if bands else None)
    return dict(method=METRIC_POLICY, input_quantity="pressure",
                decay_filter="time-reversed band filter; decay curve from the broadband onset",
                energy_filter="zero-phase band filter; integrated from onset minus the filter span",
                band_standard="base10 IEC nominal octaves; periodic response points only", band_grid=grid.as_dict(),
                onset_sample=onset_sample, noise_policy="Schroeder/Chu-Lundeby; explicit zero disables correction",
                fit_db={"edt": [0, -10], "t20": [-5, -25], "t30": [-5, -35]},
                extrapolation_db=60, settling_samples=0, latency="no filter delay in decay or energy metrics",
                drr=dict(definition=LEGACY_DRR_DEFINITION if legacy_drr else DRR_DEFINITION,
                         direct_half_width_s=.0025, endpoints="inclusive; round(0.0025*fs)",
                         input="broadband pressure before octave filtering",
                         ordering="filter then window (construct-invalid at low frequency)" if legacy_drr
                         else "window then independently filter direct/reverberant components",
                         filter_ringout="not retained separately" if legacy_drr else "pole amplitude settling to 1e-12",
                         pre_window="excluded", noise="full-band Lundeby/Chu; zero disables correction"))


@lru_cache(maxsize=64)
def octave_sos(fs: int, band: int) -> np.ndarray:
    """Base-10 octave edges; rejects incomplete bands at/above Nyquist."""
    _fs(fs)
    center = exact_center(band)
    edges = center * np.array([10 ** -0.15, 10 ** 0.15])
    if edges[1] >= fs / 2:
        raise ValueError(f"{band} Hz octave upper edge exceeds Nyquist at fs={fs}")
    return butter(6, edges, btype="bandpass", fs=fs, output="sos")


def octave_filter(samples, fs: int, band: int) -> np.ndarray:
    """Causal phase, preserves sample origin; input includes pre-arrival silence."""
    return sosfilt(octave_sos(fs, band), _samples(samples), axis=0)


@lru_cache(maxsize=64)
def filter_span(fs: int, band: int) -> int:
    """Samples holding all but 1e-9 of the band filter's impulse-response energy."""
    impulse = np.zeros(max(16, round(.5*fs))); impulse[0] = 1.
    h = sosfilt(octave_sos(fs, band), impulse)
    return int(np.searchsorted(np.cumsum(h*h), (1 - 1e-9)*np.sum(h*h))) + 1


def _tail_pad(x: np.ndarray, count: int) -> np.ndarray:
    """Mirror of the last `count` samples (noise statistics) so filtering backwards starts settled."""
    count = min(count, len(x))
    return x[len(x) - count:][::-1]


def reverse_filter(samples, fs: int, band: int) -> np.ndarray:
    """Band filter applied to the time-reversed response, then restored: ringing precedes events."""
    x = _samples(samples, mono=True)
    span = filter_span(fs, band)
    padded = np.r_[x, _tail_pad(x, span)]
    return sosfilt(octave_sos(fs, band), padded[::-1])[::-1][:len(x)]


def zero_phase_filter(samples, fs: int, band: int) -> tuple[np.ndarray, int]:
    """Forward then backward band filtering (zero phase), with `span` zeros prepended so pre-ringing
    before the first sample is kept. Returns (signal, offset of the input's first sample)."""
    x = _samples(samples, mono=True)
    span = filter_span(fs, band)
    padded = np.r_[np.zeros(span), x, _tail_pad(x, span)]
    sos = octave_sos(fs, band)
    y = sosfilt(sos, sosfilt(sos, padded)[::-1])[::-1]
    return y[:span + len(x)], span


@dataclass
class DecayCurve:
    time_s: np.ndarray
    energy: np.ndarray  # unnormalised, sample-energy sum (scale cancels in metrics)
    db: np.ndarray
    noise_power: float
    truncation_sample: int
    tail_energy: float
    tail_moment: float
    slope_db_s: float | None
    valid: bool
    reason: str | None
    iterations: int


def _regress(t, y):
    centered = t - np.mean(t)
    slope = float(np.dot(centered, y - np.mean(y)) / np.dot(centered, centered))
    intercept = float(np.mean(y) - slope * np.mean(t))
    return slope, intercept


def schroeder_decay(samples, fs: int, *, noise_power: float | None = None,
                    max_iterations: int = 30) -> DecayCurve:
    """Iterative Lundeby intersection, Chu subtraction, exponential missing tail.

    10 ms block means initialize the fit; refine to ~5 blocks per 10 dB decay.
    Fit signal power 5 dB below the block peak and >=10 dB above noise. Re-estimate
    noise >=10 dB of fitted decay after the crossing, never earlier than last 10%.
    Integrate SIGNED (p^2-noise) up to crossing, plus fitted exponential tail.
    Individual samples are not clipped (that would bias the energy upward).
    Requires a recorded noise-only tail. Failure returns invalid diagnostics.
    Explicit noise_power=0 selects ordinary finite-record Schroeder integration.
    """
    _fs(fs)
    x = _samples(samples, mono=True)
    p = x * x
    noise = float(estimate_noise_power(x)) if noise_power is None else float(noise_power)
    if not np.isfinite(noise) or noise < 0 or max_iterations < 1:
        raise ValueError("noise power must be finite/nonnegative; iterations >=1")
    cut, tail, moment, slope, iterations = len(x), 0.0, 0.0, None, 0
    reason = None
    if not np.any(p):
        reason = "zero energy"
    elif noise > 0:
        block = max(1, round(0.01 * fs))
        previous = None
        for iterations in range(1, max_iterations + 1):
            blocks = len(p) // block
            if blocks < 8:
                reason = "insufficient duration for decay/noise regression"
                break
            power = p[:blocks * block].reshape(blocks, block).mean(axis=1)
            times = (np.arange(blocks) * block + (block - 1) / 2) / fs
            peak = int(np.argmax(power))
            signal = power - noise
            usable = ((np.arange(blocks) >= peak) & (signal > 10 * noise) &
                      (signal < max(signal[peak], 0) * 10 ** (-5 / 10)))
            if np.count_nonzero(usable) < 5:
                reason = "insufficient decay above noise"
                break
            slope, intercept = _regress(times[usable], 10 * np.log10(signal[usable]))
            if slope >= 0:
                reason = "nondecaying response"
                break
            crossing = (10 * np.log10(noise) - intercept) / slope
            noise_start = max(round((crossing - 10 / slope) * fs), int(0.9 * len(x)))
            if crossing <= 0 or noise_start >= len(x) - max(block, round(.01 * fs)):
                reason = "recording ends before a usable noise-only tail"
                break
            if noise_power is None:
                noise = float(np.mean(p[noise_start:]))
                if noise <= 0:
                    reason = "noise estimate vanished"
                    break
            # Refine averaging ONCE. Repartitioning samples every iteration makes
            # narrow-band random-power bins jump and the crossing oscillate.
            new_block = max(1, round((-60 / slope) * fs / 30)) if iterations == 1 else block
            if previous is not None and abs(crossing - previous) < max(.001, .001 * crossing):
                cut = min(len(x), max(1, round(crossing * fs)))
                alpha = -slope * np.log(10) / (10 * fs)
                q = np.exp(-alpha)
                tail = float(10 ** ((intercept + slope * cut / fs) / 10) / (-np.expm1(-alpha)))
                moment = tail * (cut + q / (-np.expm1(-alpha))) / fs
                break
            previous, block = crossing, new_block
        else:
            reason = "Lundeby iteration did not converge"
    corrected = p[:cut] - noise
    energy = np.cumsum(corrected[::-1])[::-1] + tail
    if energy[0] <= 0:
        reason = reason or "no energy above noise"
    db = np.full(len(energy), -np.inf)
    if energy[0] > 0:
        positive = energy > 0
        db[positive] = 10 * np.log10(energy[positive] / energy[0])
    return DecayCurve(np.arange(cut) / fs, energy, db, noise, cut, tail, moment,
                      slope, reason is None, reason, iterations)


@dataclass
class DecayFit:
    rt60_s: float | None
    slope_db_s: float | None
    r_squared: float | None
    range_db: tuple[float, float]
    valid: bool
    reason: str | None


def fit_decay(curve: DecayCurve, upper_db: float, lower_db: float) -> DecayFit:
    """ISO ranges EDT 0..-10, T20 -5..-25, T30 -5..-35 dB; return T60."""
    def invalid(reason):
        return DecayFit(None, None, None, (upper_db, lower_db), False, reason)
    if not lower_db < upper_db <= 0:
        raise ValueError("invalid regression range")
    if not curve.valid:
        return invalid(curve.reason)
    reached = np.flatnonzero(curve.db <= lower_db)
    if not len(reached):
        return invalid("requested decay range not observed")
    end = int(reached[0])
    if curve.noise_power > 0 and curve.slope_db_s is not None:
        dt = curve.time_s[1] - curve.time_s[0] if len(curve.time_s) > 1 else 0
        margin = -curve.slope_db_s * (curve.truncation_sample * dt - curve.time_s[end])
        if margin < 10:
            return invalid("regression endpoint less than 10 dB above noise")
    mask = (curve.db >= lower_db) & (curve.db <= upper_db) & (np.arange(len(curve.db)) <= end)
    if np.count_nonzero(mask) < 3:
        return invalid("too few decay samples")
    t, y = curve.time_s[mask], curve.db[mask]
    slope, intercept = _regress(t, y)
    if slope >= 0:
        return invalid("nonnegative decay slope")
    residual = np.sum((y - (intercept + slope * t)) ** 2)
    variance = np.sum((y - np.mean(y)) ** 2)
    return DecayFit(-60 / slope, slope, float(1 - residual / variance),
                    (upper_db, lower_db), True, None)


@dataclass
class BandMetrics:
    edt: DecayFit
    t20: DecayFit
    t30: DecayFit
    c50_db: float | None
    c80_db: float | None
    d50: float | None
    ts_s: float | None
    drr_db: float | None
    onset_sample: int
    decay: DecayCurve
    drr_definition: str = DRR_DEFINITION

    def __post_init__(self):
        if self.drr_definition not in (DRR_DEFINITION, LEGACY_DRR_DEFINITION):
            raise ValueError("unsupported DRR definition")


def _ratio_db(numerator, denominator):
    if numerator < 0 or denominator < 0 or numerator + denominator <= 0:
        return None
    if denominator == 0:
        return float("inf")
    if numerator == 0:
        return float("-inf")
    return float(10 * np.log10(numerator / denominator))


@lru_cache(maxsize=64)
def _drr_settling_samples(fs: int, band: int) -> int:
    # Derive support from the slowest pole, not a fixed millisecond crop that
    # truncates low-band energy. Roots of each SOS avoid high-order conditioning.
    radius = max(abs(p) for section in octave_sos(fs, band)
                 for p in np.roots(section[3:]))
    return max(1, int(np.ceil(np.log(1e-12)/np.log(radius))))


def _component_energy(samples, fs, band):
    if not len(samples):
        return 0.
    if band is not None:
        samples = octave_filter(np.pad(samples, (0, _drr_settling_samples(fs, band))), fs, band)
    return float(np.dot(samples, samples))


def _partition_drr(broadband, fs, band, onset, curve):
    """Complete energies of disjoint input components, including filter ring-out."""
    half = round(.0025*fs)
    start, stop = max(0, onset-half), onset+half+1
    end = min(len(broadband), onset+curve.truncation_sample)
    if end < stop:
        return None  # no inferred direct-window support from an extrapolated tail
    direct = _component_energy(broadband[start:stop], fs, band)
    reverb = _component_energy(broadband[stop:end], fs, band)
    direct -= (stop-start)*curve.noise_power
    reverb += curve.tail_energy - (end-stop)*curve.noise_power
    return _ratio_db(direct, reverb)


def room_metrics(samples, fs: int, *, onset_sample: int | None = None,
                 noise_power: float | None = None, band: int | None = None,
                 legacy_drr: bool = False, policy: str = METRIC_POLICY) -> BandMetrics:
    """Broadband mono pressure input; optional octave analysis at broadband arrival.

    Clarity includes exponential missing tail. Ts includes its first moment.
    drr/2 splits input at +-2.5 ms BEFORE filtering its components, retaining
    filter ring-out. Never prefilter the input for a drr/2 octave measurement.
    noise_power is in the analyzed band's units. Explicit legacy_drr=True
    selects drr/1 (filter first, gate later), invalid for low-band direct energy.
    DRR is a window convention, not an ISO 3382 parameter or path separator.
    Missing decay/noise support yields None energy metrics, invalid RT fits.
    """
    _fs(fs)
    broadband = _samples(samples, mono=True)
    onset = detect_onset(broadband, noise_power=noise_power if band is None else None) if onset_sample is None else onset_sample
    if not isinstance(onset, (int, np.integer)) or not 0 <= onset < len(broadband):
        raise ValueError("onset_sample out of range")
    if policy not in (METRIC_POLICY, LEGACY_METRIC_POLICY):
        raise ValueError(f"unknown metric policy {policy!r}")
    if band is not None and policy == METRIC_POLICY:
        result = _band_metrics_v2(broadband, fs, band, onset, noise_power)
        if legacy_drr:                       # the DRR definition is versioned on its own
            legacy = room_metrics(broadband, fs, onset_sample=onset, noise_power=noise_power, band=band,
                                  legacy_drr=True, policy=LEGACY_METRIC_POLICY)
            from dataclasses import replace
            result = replace(result, drr_db=legacy.drr_db, drr_definition=LEGACY_DRR_DEFINITION)
        return result
    x = broadband if band is None else octave_filter(broadband, fs, band)
    curve = schroeder_decay(x[onset:], fs, noise_power=noise_power)
    edt, t20, t30 = (fit_decay(curve, *bounds) for bounds in [(0, -10), (-5, -25), (-5, -35)])
    c50 = c80 = d50 = ts = drr = None
    if curve.valid and curve.energy[0] > 0:
        p = x[onset:onset + curve.truncation_sample] ** 2 - curve.noise_power
        total = curve.energy[0]
        # A missing early boundary is not supplied by extrapolation.
        if len(p) >= int(np.ceil(.05 * fs)):
            e50 = float(np.sum(p[:int(np.ceil(.05 * fs))]))
            c50 = _ratio_db(e50, total - e50)
            d50 = float(e50 / total) if 0 <= e50 <= total else None
        if len(p) >= int(np.ceil(.08 * fs)):
            e80 = float(np.sum(p[:int(np.ceil(.08 * fs))]))
            c80 = _ratio_db(e80, total - e80)
        ts = float((np.dot(np.arange(len(p)) / fs, p) + curve.tail_moment) / total)
        if legacy_drr:
            half = round(.0025 * fs)
            direct = float(np.sum(x[max(0, onset - half):onset + half + 1] ** 2 - curve.noise_power))
            reverb = float(np.sum(p[half + 1:]) + curve.tail_energy)
            drr = _ratio_db(direct, reverb)
        else:
            drr = _partition_drr(broadband, fs, band, onset, curve)
    return BandMetrics(edt, t20, t30, c50, c80, d50, ts, drr, onset, curve,
                       LEGACY_DRR_DEFINITION if legacy_drr else DRR_DEFINITION)


def _band_metrics_v2(broadband, fs, band, onset, noise_power) -> BandMetrics:
    """room-measured-metrics/2 for one band (module docstring)."""
    curve = schroeder_decay(reverse_filter(broadband, fs, band)[onset:], fs, noise_power=noise_power)
    edt, t20, t30 = (fit_decay(curve, *bounds) for bounds in [(0, -10), (-5, -25), (-5, -35)])
    c50 = c80 = d50 = ts = drr = None
    zero_phase, offset = zero_phase_filter(broadband, fs, band)
    span = filter_span(fs, band)
    start = onset + offset - span                                        # >= 0: span zeros were prepended
    # Zero-phase filtering weights the band by |H|^4, not |H|^2, so its noise is re-estimated
    # unless the caller declared a noise-free response.
    energy_curve = schroeder_decay(zero_phase[start:], fs, noise_power=0. if noise_power == 0 else None)
    if energy_curve.valid and energy_curve.energy[0] > 0:
        p = zero_phase[start:start + energy_curve.truncation_sample] ** 2 - energy_curve.noise_power
        total = energy_curve.energy[0]
        for ms in (50, 80):
            split = span + int(np.ceil(ms/1000*fs))
            if len(p) < split:
                continue
            early = float(np.sum(p[:split]))
            if ms == 50:
                c50 = _ratio_db(early, total - early)
                d50 = float(early/total) if 0 <= early <= total else None
            else:
                c80 = _ratio_db(early, total - early)
        times = (np.arange(len(p)) - span)/fs
        ts = float((np.dot(times, p) + energy_curve.tail_moment - span/fs*energy_curve.tail_energy)/total)
    if curve.valid and curve.energy[0] > 0:
        drr = _partition_drr(broadband, fs, band, onset, curve)
    return BandMetrics(edt, t20, t30, c50, c80, d50, ts, drr, onset, curve, DRR_DEFINITION)


def analyze_octaves(samples, fs: int, *, onset_sample: int | None = None,
                    noise_power: float | None = None, legacy_drr: bool = False) -> dict:
    """Broadband + all supported octaves. Unsupported bands raise, never vanish.

    Mono API: call separately for each microphone. Onset found once broadband; bands follow
    room-measured-metrics/2 (module docstring). DRR defaults to window-first drr/2.
    noise_power=0 is useful for noise-free controls; None estimates each band.
    """
    x = _samples(samples, mono=True)
    onset = detect_onset(x) if onset_sample is None else onset_sample
    options = dict(onset_sample=onset, noise_power=noise_power, legacy_drr=legacy_drr)
    result = {"broadband": room_metrics(x, fs, **options)}
    for band in OCTAVE_BANDS:
        result[band] = room_metrics(x, fs, band=band, **options)
    return result


@dataclass(frozen=True)
class ReflectionPeak:
    sample: int
    toa_s: float             # absolute time from sample zero
    excess_delay_s: float    # relative to direct arrival
    level_db: float          # relative to strongest early peak


def early_reflections(samples, fs: int, *, onset_sample: int | None = None,
                      max_time_s: float = .08, min_spacing_s: float = .001,
                      threshold_db: float = -20) -> list[ReflectionPeak]:
    """Absolute-waveform local maxima; candidates, not identified wall paths.

    Includes direct arrival explicitly. Unfiltered broadband avoids octave
    ringing; finite-bandwidth real data can still create spurious candidates.
    """
    _fs(fs)
    x = _samples(samples, mono=True)
    if not np.all(np.isfinite([max_time_s, min_spacing_s, threshold_db])) or min(max_time_s, min_spacing_s) <= 0 or threshold_db > 0:
        raise ValueError("invalid reflection picker options")
    onset = detect_onset(x) if onset_sample is None else onset_sample
    if not isinstance(onset, (int, np.integer)) or not 0 <= onset < len(x):
        raise ValueError("onset_sample out of range")
    envelope = np.abs(x[onset:min(len(x), onset + round(max_time_s * fs) + 1)])
    maximum = envelope.max()
    if maximum == 0:
        return []
    peaks, _ = find_peaks(np.r_[0, envelope, 0], height=maximum * 10 ** (threshold_db / 20),
                          distance=max(1, round(min_spacing_s * fs)))
    return [ReflectionPeak(int(onset + p - 1), (onset + p - 1) / fs, (p - 1) / fs,
                            float(20 * np.log10(envelope[p - 1] / maximum))) for p in peaks]


def _comparison_pair(reference, estimate):
    a, b = _samples(reference, mono=True), _samples(estimate, mono=True)
    if a.shape != b.shape:
        raise ValueError("comparison signals need equal length and explicit alignment")
    return a, b


def si_sdr(reference, estimate) -> float | None:
    """Zero-mean scalar-projection SI-SDR (Le Roux et al., 2019, arXiv:1811.02508).

    No time alignment here. Silent reference/estimate is undefined; perfect
    nonzero agreement is +inf, orthogonality -inf. No arbitrary epsilon ceiling.
    """
    a, b = _comparison_pair(reference, estimate)
    a, b = a-a.mean(), b-b.mean()
    aa, bb = float(a @ a), float(b @ b)
    if aa == 0 or bb == 0:
        return None
    target = (float(a @ b)/aa)*a
    energy, residual = float(target @ target), b-target
    noise = float(residual @ residual)
    if energy == 0:
        return -float("inf")
    return float(10*np.log10(energy/noise)) if noise else float("inf")


def log_spectral_distance(reference, estimate, fs: int, *, low_hz=125., high_hz=8000., floor_db=80.) -> float | None:
    """RMS whole-record log-magnitude ratio; shared reference-relative floor.

    No gain fitting. Frozen room dry benchmark convention, not an ISO metric.
    A common 80 dB floor limits silent-bin instability without normalizing away
    level errors. Signals must already share their sample origin and support.
    """
    return spectral_level_shape(reference, estimate, fs, low_hz=low_hz,
                                high_hz=high_hz, floor_db=floor_db)['lsd']


def spectral_level_shape(reference, estimate, fs: int, *, low_hz=125., high_hz=8000., floor_db=80.) -> dict:
    """Expose the existing LSD partition on exactly the original bins and floor.

    LSD squared = signed mean error squared + centered error mean square.
    This is a diagnostic, never an evaluation-fitted correction to a waveform.
    Reuses the bench_room_resim.chain convention; no new score or tolerance.
    """
    from scipy.fft import rfft, rfftfreq
    _fs(fs)
    if not (0 <= low_hz < high_hz <= fs/2 and np.isfinite(floor_db) and floor_db > 0):
        raise ValueError("invalid spectral comparison band/floor")
    a, b = _comparison_pair(reference, estimate)
    a, b = np.abs(rfft(a)), np.abs(rfft(b))
    f = rfftfreq(len(reference), 1/fs)
    mask = (f >= low_hz) & (f <= high_hz)
    if not mask.any() or np.max(a[mask]) == 0:
        return dict(lsd=None, spectral_bias_db=None, spectral_shape_db=None,
                    spectral_level_mse_fraction=None)
    floor = np.max(a[mask])*10**(-floor_db/20)
    ratio = 20*np.log10(np.maximum(b[mask], floor)/np.maximum(a[mask], floor))
    bias = float(np.mean(ratio))
    shape = float(np.sqrt(np.mean((ratio-bias)**2)))
    return dict(lsd=float(np.sqrt(np.mean(ratio**2))), spectral_bias_db=bias,
                spectral_shape_db=shape,
                spectral_level_mse_fraction=bias*bias/(bias*bias+shape*shape) if bias or shape else 0.)


DIRECT_LEVEL_BANDS = (1000, 2000, 4000)
DIRECT_LEVEL_VALIDITY = ModelValidity(
    name='training direct-path relative level', status='partially validated',
    freq_range_hz=(707.9457843841379, 5623.413251903491),
    conditions='Fixed 1/2/4 kHz octaves; resolved reflection-bounded direct path; '
    'stable same-source and recorder gain; at least two eligible training positions. '
    'Analytic inverse-square control and conditional measured-RIR reconstruction only; '
    'no phone repeatability, directivity correction or independent Pa/SPL validation.',
    citation='Dammerud 2011 direct-reference distance compensation; '
    'parent-project note room_gap/level_reference (not distributed); parent-project note ROOM_LEVEL_CONVENTION (not distributed)',
)


def direct_level_observation(samples, fs, *, distance_m, direct_sample,
                             first_reflection_delay_s, identity, position,
                             noise_band_power, obstructed=False, clipped=False,
                             excitation_band_hz=(0., float('inf'))) -> dict:
    """Training direct energy at 1 m, relative to a unit digital impulse.

    Amplitude spreads as 1/r, energy as 1/r². The reflection-bounded gate is
    reused from rir, then filtered (never filter first and gate ringing). The
    fixed mid/high bands avoid interpreting a short gate as low-frequency EQ.
    This is a scalar recorder/sweep-chain reference, not device calibration.
    See parent-project note room_gap/level_reference (not distributed) (Dammerud 2011; Katz 2015).
    """
    from .rir import reference_direct_window
    h = _samples(samples, mono=True)
    _fs(fs)
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError('positive finite direct distance required')
    if not identity or not np.isfinite(position).all():
        raise ValueError('explicit identity and finite position required')
    result = dict(identity=identity, position=list(position), distance_m=float(distance_m),
                  eligible=False, reason=None, level_db_1m=None,
                  bands_hz=list(DIRECT_LEVEL_BANDS), excluded_bands_hz=[500])
    if obstructed or clipped:
        return {**result, 'reason': 'obstructed direct path' if obstructed else 'clipped recording'}
    if first_reflection_delay_s is None:
        return {**result, 'reason': 'resolved first-reflection bound required'}
    try:
        gate, start, stop, low = reference_direct_window(h, fs, direct_sample,
            first_reflection_delay_s=first_reflection_delay_s)
    except ValueError as exc:
        return {**result, 'reason': str(exc)}
    result.update(start_sample=start, stop_sample=stop, low_frequency_hz=low)
    edges = [(exact_center(b)*10**-.15, exact_center(b)*10**.15) for b in DIRECT_LEVEL_BANDS]
    if any(lo < max(low, excitation_band_hz[0]) or hi > min(fs/2, excitation_band_hz[1]) for lo, hi in edges):
        return {**result, 'reason': 'unsupported scalar-estimation band (window/excitation/Nyquist)'}
    if (set(noise_band_power) != set(DIRECT_LEVEL_BANDS)
            or any(not np.isfinite(v) or v < 0 for v in noise_band_power.values())):
        return {**result, 'reason': 'missing finite nonnegative band noise proxy'}
    # Pad equally to preserve causal-filter tails for short direct gates.
    gated = np.pad(gate, (0, max(0, round(.05*fs)-len(gate))))
    unit = np.zeros_like(gated)
    unit[direct_sample-start] = 1.
    energy = sum(float((y := octave_filter(gated, fs, b)) @ y) for b in DIRECT_LEVEL_BANDS)
    unit_energy = sum(float((y := octave_filter(unit, fs, b)) @ y) for b in DIRECT_LEVEL_BANDS)
    noise_energy = len(gate)*sum(noise_band_power.values())
    if energy <= 0 or energy < 10*noise_energy:
        return {**result, 'reason': 'direct band energy below 10 dB noise-proxy gate'}
    level = float(10*np.log10(distance_m**2*energy/unit_energy))
    direct = np.zeros_like(h)
    direct[start:stop] = gate
    residual = h-direct
    return {**result, 'eligible': True, 'level_db_1m': level,
            'direct_band_energy': energy, 'unit_band_energy': unit_energy,
            'direct_residual_ratio': float((direct@direct)/max(float(residual@residual),1e-30)),
            'noise_energy_proxy': noise_energy}


def pool_direct_levels(observations) -> dict:
    """Median dB intercept, equal weight per distinct delivered position.

    Leave-one-position-out residuals diagnose common-gain inconsistency; they
    are not confidence intervals for phone repeatability or physical accuracy.
    """
    observations = list(observations)
    groups = {}
    for item in observations:
        if item['eligible']:
            level = item['level_db_1m']
            if level is None or not np.isfinite(level):
                raise ValueError('eligible direct level must be finite')
            groups.setdefault(tuple(item['position']), []).append(level)
    levels = np.array([np.median(values) for values in groups.values()])
    available = len(levels) >= 2
    center = float(np.median(levels)) if available else None
    return dict(version='room-direct-level/1', status='relative' if available else 'unavailable',
        validity=asdict(DIRECT_LEVEL_VALIDITY),
        reference_distance_m=1., units='recorder_relative', reference_kind='training_direct_pool',
        method='median distance-corrected direct energy in dB; equal weight per position',
        level_db_1m=center, amplitude_scale=10**(center/20) if available else None,
        positions=len(levels), position_keys=[list(p) for p in groups],
        position_levels_db=levels.tolist(),
        residual_db=(levels-center).tolist() if available else [],
        leave_one_position_out_db=[float(levels[i]-np.median(np.delete(levels,i))) for i in range(len(levels))] if available else [],
        observations=list(observations),
        reason=None if available else 'at least two eligible distinct training positions required',
        assumptions='stable source/recorder gain and orientation; resolved direct path; no device EQ or Pa inference')


def source_level_metadata(reference, *, kind='relative_only') -> dict:
    """Describe, never upgrade, the existing typed SourceReference carrier."""
    from .source_reference import SourceReference
    if not isinstance(reference, SourceReference):
        raise TypeError('typed SourceReference required')
    if reference.units == 'Pa' and reference.calibration is None:
        raise ValueError('Pa claims require independent audio calibration')
    return dict(version='room-level-metadata/1', reference_kind=kind,
        source_reference=reference.as_dict(), intended_convention='relative source at 1 m in RECORDER units',
        calibrated_pressure=reference.units == 'Pa',
        effective_source=reference.target == 'effective_source',
        frequency_response='retained and unresolved; scalar reference is not equalisation',
        absolute_spl_claim=False)


def band_level_error(reference, estimate, fs: int, band: int) -> float | None:
    """Signed octave energy error in dB, reusing the production causal filter."""
    a, b = _comparison_pair(reference, estimate)
    a, b = octave_filter(a, fs, band), octave_filter(b, fs, band)
    energy, predicted = float(a @ a), float(b @ b)
    if energy == 0:
        return None
    return float(10*np.log10(predicted/energy)) if predicted > 0 else -float("inf")
