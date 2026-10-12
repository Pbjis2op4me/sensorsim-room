"""Identify asynchronous ESS trains and isolate separate real-sound events.

Farina, AES 5093 (2000): https://www.angelofarina.it/Public/papers/134-AES00.PDF
Matched filtering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html

Uses the existing exact kit phase/fades and ESS inverse, never a sync marker.
Arrival means the first supported compressed *linear packet*, not emission time or
an independently identified direct ray. A stationary, constant-clock LTI path
is assumed. Ambiguous trains and insufficient evidence raise DetectionError.
Scores are diagnostics, NOT calibrated probabilities of correct detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import combinations, product

import numpy as np
from scipy import signal
from scipy.fft import ifft, irfft, next_fast_len, rfft
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar

from sensorsim.core.messages import ModelValidity
from .ingest import SweepKit

VALIDITY = ModelValidity(
    name="asynchronous room sweep identification", status="partially validated",
    conditions="Stationary synthetic mono captures, including measured-RIR convolution; constant clock within +/-300 ppm; "
    "not field validated. First supported linear packet may be reflected. "
    "Real-event label consistency is conditional, not semantic identification.",
    citation="Farina AES 5093 (2000); SciPy signal.correlate; parent-project note ROOM_SWEEP_DETECT (not distributed)",
)


class DetectionError(ValueError):
    """Evidence is insufficient; the message describes the required corrective action."""


@dataclass(frozen=True)
class DetectionConfig:
    min_repeats: int = 3
    max_clock_drift_ppm: float = 300.
    min_correlation: float = .035
    min_peak_snr_db: float = 12.
    spacing_tolerance_s: float = .003
    guard_s: float = .05
    max_candidates: int = 128

    def __post_init__(self):
        if (isinstance(self.min_repeats, bool) or not isinstance(self.min_repeats,int) or self.min_repeats < 2 or
                not np.all(np.isfinite([self.max_clock_drift_ppm, self.min_correlation,
                                       self.min_peak_snr_db, self.spacing_tolerance_s, self.guard_s])) or
                not 0 < self.max_clock_drift_ppm <= 1000 or
                not 0 < self.min_correlation < 1 or self.min_peak_snr_db <= 0 or
                self.spacing_tolerance_s <= 0 or self.guard_s <= 0 or self.max_candidates < 4):
            raise ValueError("invalid sweep detection configuration")


@dataclass
class RepeatDetection:
    slot: int
    start_sample: float
    end_sample: float
    confidence: float
    peak_snr_db: float
    polarity: int
    segment_start: int
    segment_stop: int
    clipped: bool = False
    linear_peak_amplitude: float = 0.


@dataclass
class SweepDetection:
    fs: int
    repeats: list[RepeatDetection]
    dropped: list[dict]
    clock_drift_ppm: float
    chirp_drift_ppm: float
    spacing_residual_samples: float
    repeat_correlations: list[float]
    clipped: bool
    warnings: list[str] = field(default_factory=list)
    time_origin: str = "relative first supported linear arrival; absolute emission unobservable"
    chirp_estimation_method: str = "strict_ridge_then_spectral_factor_entropy/1"

    def metadata(self) -> dict:
        return asdict(self)


def _mono(samples, fs):
    x = np.asarray(samples)
    if (isinstance(fs, bool) or not isinstance(fs, (int, np.integer)) or fs <= 0 or
            x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x))):
        raise ValueError("expected finite nonempty mono samples and a positive integer rate")
    return x


def clipping_present(samples) -> bool:
    """Digital full-scale rails; analog/pre-export clipping cannot be certified."""
    x = np.asarray(samples)
    # Arbitrary floating-point pressure/FS signals may pass THROUGH one; only
    # repeated exact digital rail values establish the flag without format info.
    rails = (x == 32767/32768) | (x == 1.) | (x == -1.)
    return bool(np.count_nonzero(rails) >= 3)


def fractional_peak(values, peak: int) -> float:
    """Windowed-sinc interpolation of a signed compressed peak (sample coordinates)."""
    if peak < 2 or peak >= len(values)-2:
        return float(peak)
    radius = min(32, peak, len(values)-1-peak)
    offsets = np.arange(-radius, radius+1)
    local = np.array(values[peak-radius:peak+radius+1], dtype=float, copy=True)
    local *= np.sign(local[radius]) or 1
    window = np.kaiser(len(local), 6)
    def objective(t):
        weights = np.sinc(t-offsets)*window
        return -float(np.dot(local, weights)/weights.sum())
    result = minimize_scalar(objective, bounds=(-.6, .6), method="bounded",
                             options={"xatol": 1e-5})
    return float(peak+result.x)


def linear_arrival(values, fs, lo, hi, *, bandwidth_hz):
    """First supported linear packet, then its fractional signed peak.

    The -20 dB relative power criterion is the existing RIR onset convention;
    a 20 dB noise margin rejects compressed background. Resolve the peak within
    half a millisecond (or eight inverse bandwidths) so inverse-filter sidelobes
    do not become separate arrivals. A weaker physical direct path can remain
    unobservable; geometry supplies the time origin, not an emission timestamp.
    """
    packet = values[lo:hi]
    width = max(1,round(.0001*fs))
    power = signal.convolve(packet*packet,np.ones(width)/width,mode='same')
    noise = float(np.median(power[:max(1,len(power)//4)]))
    found = np.flatnonzero(power >= max(float(power.max())*.01,noise*100))
    first = int(found[0]) if len(found) else int(np.argmax(power))
    end = min(len(packet),first+max(3,round(max(.0005,8/bandwidth_hz)*fs)))
    peak = lo+first+int(np.argmax(abs(packet[first:end])))
    return fractional_peak(values,peak),peak


def _coarse_template(kit, fs, ppm=0.):
    """Exact kit phase on a low-rate grid; truncate before Nyquist, no aliased chirp.

    The full-band inverse is applied only after the clock/structure is found.
    This bounded-band search keeps arbitrarily long lead/trail FFTs small.
    """
    length = round(kit.sweep_s*kit.fs_hz)/kit.fs_hz / np.log(kit.f2_hz/kit.f1_hz)
    stop = min(kit.sweep_s, length*np.log(min(.4*fs, kit.f2_hz)/kit.f1_hz))
    if stop <= 0:
        raise ValueError("file rate does not support the kit's starting frequency")
    ratio = 1+ppm*1e-6
    t = np.arange(int(np.ceil(stop*fs*ratio)))/(fs*ratio)
    x = np.sin(2*np.pi*kit.f1_hz*length*np.expm1(t/length))
    # Match the kit half-Hann; early fade is long relative to the search grid.
    n_in = int(kit.fade_in_s*kit.fs_hz)
    if n_in:
        x *= np.where(t < n_in/kit.fs_hz,
                      .5-.5*np.cos(2*np.pi*np.minimum(t*kit.fs_hz, n_in)/(2*n_in-1)), 1.)
    fade = min(.02, stop*.05)
    x *= np.sin(np.clip((stop-t)/fade, 0, 1)*np.pi/2)**2
    # Pre-emphasis shortens the compressed pulse without another sensor model.
    return np.diff(x, prepend=0)


def _normalized_search(y, template):
    corr = signal.oaconvolve(y, template[::-1], mode="valid")
    power = np.r_[0., np.cumsum(np.square(y, dtype=float))]
    energy = power[len(template):]-power[:-len(template)]
    norm = np.sqrt(np.maximum(energy*np.dot(template, template), 1e-30))
    return np.where(energy > max(float(energy.max())*1e-12, 1e-30), corr/norm, 0.)


def _train_candidates(peaks, fs, kit, config):
    period = kit.period_s*fs
    tolerance = max(config.spacing_tolerance_s*fs, .02*fs)
    groups = set()
    for a, b in combinations(range(len(peaks)), 2):
        gap = int(round((peaks[b]-peaks[a])/period))
        if not 1 <= gap < kit.repeats:
            continue
        slope = (peaks[b]-peaks[a])/gap
        if abs(slope-period) > period*config.max_clock_drift_ppm*1e-6 + tolerance:
            continue
        slots = np.rint((peaks-peaks[a])/slope).astype(int)
        good = np.abs(peaks-(peaks[a]+slots*slope)) < tolerance
        for first in range(-(kit.repeats-1), 1):
            ids = tuple(np.flatnonzero(good & (slots >= first) & (slots < first+kit.repeats)))
            if len(ids) >= config.min_repeats and len(set(slots[list(ids)])) == len(ids):
                groups.add(ids)
    # Subsets of a confirmed four-repeat train are not independent detections.
    return [g for g in groups if not any(set(g) < set(other) for other in groups)]


def _repeat_clock(y, starts, slots, period, fs, config, reference=0):
    width = max(8, round(min(period/fs*.65, (len(y)-max(starts))/fs)*fs))
    margin = max(4, round(.025*fs))
    anchor = int(starts[reference])
    ref = y[anchor:anchor+width]
    times, correlations = [], []
    for start in starts:
        if start == starts[reference]:
            times.append(float(anchor))
            correlations.append(1.)
            continue
        lo = max(0, int(start)-margin)
        window = y[lo:min(len(y), int(start)+width+margin)]
        corr = signal.correlate(window, ref, mode="valid", method="fft")
        peak = int(np.argmax(np.abs(corr)))
        segment = window[peak:peak+width]
        norm = np.sqrt(np.dot(ref, ref)*np.dot(segment, segment))
        correlations.append(float(abs(corr[peak])/max(norm, 1e-30)))
        times.append(lo+fractional_peak(corr, peak))
    times = np.array(times)
    slopes = [(times[j]-times[i])/(slots[j]-slots[i]) for i,j in combinations(range(len(times)), 2)]
    slope = float(np.median(slopes))
    ppm = (slope/period-1)*1e6
    residual = times-(np.median(times-slots*slope)+slots*slope)
    # One sample across the baseline accounts for deliberately tiny regression kits.
    resolution = 1e6/(period*max(np.ptp(slots), 1))
    if (abs(ppm) > config.max_clock_drift_ppm+resolution or
            max(abs(residual)) > max(2., config.spacing_tolerance_s*fs)):
        raise DetectionError("repeat spacing is inconsistent or clock exceeds bounds; check rate headers, motion and playback dropouts")
    return times, float(np.clip(ppm, -config.max_clock_drift_ppm, config.max_clock_drift_ppm)), correlations, float(max(abs(residual)))


def _room_train(y, score, fs, kit, config):
    """Fit one repetition lattice even when different echoes win each peak.

    Propose nearby matches at the known period from each anchor, then verify
    with raw repeat correlation. The correlation/clock bounds are unchanged.
    Overlapping hypotheses within the kit's decay gap represent the same room
    response; separated trains still require explicit selection.
    """
    period = kit.period_s*fs
    anchors, _ = signal.find_peaks(score, height=config.min_correlation,
                                  distance=max(1, round(.7*period)))
    if len(anchors) > config.max_candidates:
        raise DetectionError("too many sweep candidates; select one playback take")
    proposals = set()
    for anchor in anchors:
        nearby = {}
        for slot in range(1-kit.repeats, kit.repeats):
            predicted = anchor+slot*period
            radius = max(2, round(config.spacing_tolerance_s*fs+abs(slot)*period*config.max_clock_drift_ppm*1e-6))
            lo, hi = max(0,round(predicted)-radius), min(len(score),round(predicted)+radius+1)
            if lo >= hi:
                continue
            peak = lo+int(np.argmax(score[lo:hi]))
            if score[peak] >= config.min_correlation:
                nearby[slot] = peak
        for first in range(1-kit.repeats, 1):
            proposal = tuple((s,nearby[s]) for s in range(first, first+kit.repeats) if s in nearby)
            if len(proposal) >= config.min_repeats:
                proposals.add(proposal)
    candidates = []
    for proposal in sorted(proposals):
        slots, starts = np.array(proposal).T
        slots = slots-slots[0]
        reference = int(np.argmax(score[starts]))
        try:
            times, ppm, corr, residual = _repeat_clock(y,starts,slots,period,fs,config,reference)
        except DetectionError:
            continue
        if min(corr) < config.min_correlation:
            continue
        quality = len(starts)+float(np.mean(score[starts])*np.mean(corr))
        candidates.append((quality, starts, slots, times, ppm, corr, residual))
    candidates.sort(key=lambda c:c[0], reverse=True)
    distinct = []
    for candidate in candidates:
        times = candidate[3]
        if not any(np.min(abs(times[:,None]-other[3][None,:])) < kit.tail_silence_s*fs
                   for other in distinct):
            distinct.append(candidate)
    if len(distinct) != 1:
        if not distinct:
            raise DetectionError(f"fewer than {config.min_repeats} coherent sweep repeats; check the kit, fixed gain and completeness")
        raise DetectionError("ambiguous sweep trains; supply one playback take or explicitly crop the intended train")
    return distinct[0][1:]


def _chirp_clock(samples, start, fs, kit, config):
    """Independent single-sweep clock from the earliest compressed wavefront.

    Sweep clock error is an allpass phase curvature, approximately
    -2*pi*ppm*1e-6*f*L*log(f/f1) (Bryan/Kolar/Abel, AES 129, 2010).
    For the full kit, remove minimum-phase coloration before phase focusing.
    Tiny regression kits use phase focusing over an early packet because their
    short durations cannot resolve a ppm-scale ridge. No repetition spacing or
    estimated ppm enters either estimator.
    """
    margin = max(.2, 8*kit.sweep_s*config.max_clock_drift_ppm*1e-6)
    full, origin, _ = compress_repeat(samples, kit,
        start_sample=round(start-margin*fs),
        stop_sample=round(start+(kit.sweep_s+2*margin)*fs))
    center = round(start-origin)+round(kit.sweep_s*fs)-1
    radius = round(margin*fs)
    packet = full[max(0, center-radius):center+radius]
    if kit.sweep_s >= 2.:
        return _ridge_clock(packet,fs,kit,config)
    width = max(1, round(max(.001, kit.sweep_s*config.max_clock_drift_ppm*1e-6)*fs))
    power = signal.convolve(packet*packet, np.ones(width)/width, mode="same")
    threshold = max(float(power.max())*.01,
                    float(np.median(power[:max(1,len(power)//4)]))*10**(config.min_peak_snr_db/10))
    supported = np.flatnonzero(power > threshold)
    if not len(supported):
        raise DetectionError("no early sweep response above noise; increase unclipped sweep level or reduce background")
    first = int(supported[0])
    last = min(len(packet), first+round(max(.015, 5*kit.sweep_s*config.max_clock_drift_ppm*1e-6)*fs))
    nfft = next_fast_len(4*len(packet))
    frequency = np.fft.rfftfreq(nfft, 1/fs)
    spectrum = rfft(packet*signal.windows.tukey(len(packet), .25), nfft)
    length = kit.sweep_s/np.log(kit.f2_hz/kit.f1_hz)
    phase = 2*np.pi*frequency*length*np.log(np.maximum(frequency,kit.f1_hz)/kit.f1_hz)*1e-6
    def score(ppm):
        response = irfft(spectrum*np.exp(1j*ppm*phase), nfft)[first:last]
        peak = int(np.argmax(abs(response)))
        loc = fractional_peak(response, peak)
        indices = np.arange(max(0,peak-32), min(len(response),peak+33))
        weights = np.sinc(loc-indices)*np.kaiser(len(indices), 6)
        return -abs(float(response[indices]@weights/weights.sum()))
    bound = config.max_clock_drift_ppm
    # Resolve sub-sample phase curvature before local refinement. Nine coarse
    # guesses can miss a narrow focus and select another reflection's maximum.
    intervals = max(8, int(np.ceil(2*bound*kit.sweep_s*kit.f2_hz/2e6)))
    grid = np.linspace(-bound, bound, intervals+1)
    best = int(np.argmin([score(p) for p in grid]))
    result = minimize_scalar(score, bounds=(grid[max(0,best-1)], grid[min(intervals,best+1)]),
                             method="bounded", options={"xatol": .2})
    return float(result.x)


def _minimum_phase_angle(spectrum, nfft):
    magnitude=abs(spectrum)
    cepstrum=irfft(np.log(np.maximum(magnitude,magnitude.max()*1e-4)),nfft)
    cepstrum[1:(nfft+1)//2]*=2
    cepstrum[nfft//2+1:]=0
    return np.imag(rfft(cepstrum,nfft))


@lru_cache(maxsize=8)
def _compression_phase(fs, duration, f1, f2, fade_in, fade_out, nfft):
    """Remove the exact inverse/reference's own phase factor from clock focus."""
    from .ingest import ESSExcitation
    from .rir import generate_ess, ess_inverse
    spec=ESSExcitation(duration_s=duration,f_start_hz=f1,f_end_hz=f2,
                       amplitude=1.,fade_in_s=fade_in,fade_out_s=fade_out)
    sweep=generate_ess(fs,spec)
    response=signal.fftconvolve(sweep,ess_inverse(fs,spec))
    center=len(sweep)-1
    offsets=np.arange(nfft)-nfft//2
    packet=np.zeros(nfft)
    good=(center+offsets>=0)&(center+offsets<len(response))
    packet[good]=response[center+offsets[good]]
    spectrum=rfft(packet,nfft)
    return _minimum_phase_angle(spectrum,nfft)-np.angle(spectrum)


def _ridge_clock(packet,fs,kit,config):
    """Independent phase focus, refined on an early packet only when dominant.

    A diffuse response uses all wavefronts. If the first supported 20 ms contains
    a majority of packet energy, remove later reflections and refit. The first
    estimate only locates this gate; the second clock search keeps its original
    bounds and receives no repeat-spacing prior. Two-times analytic interpolation
    in the objective limits sampling-grid modulation of the entropy.
    """
    # A normalized peak exists even in numerical stopband leakage. Require the
    # same actual spectral support as the entropy fallback BEFORE attempting a
    # ridge, or a low-band-only response can spuriously certify zero drift.
    support_nfft=next_fast_len(len(packet))
    _usable_clock_band(rfft(packet,support_nfft),fs,kit,support_nfft)
    resolved=_resolved_clock_ridge(packet,fs,kit,config)
    if resolved is not None:
        return resolved
    initial=_phase_focus_clock(packet,fs,kit,config)
    nfft=next_fast_len(len(packet))
    frequency=np.fft.rfftfreq(nfft,1/fs)
    length=kit.sweep_s/np.log(kit.f2_hz/kit.f1_hz)
    phase=2*np.pi*frequency*length*np.log(np.maximum(frequency,kit.f1_hz)/kit.f1_hz)*1e-6
    corrected=irfft(rfft(packet,nfft)*np.exp(1j*initial*phase),nfft)
    power=gaussian_filter1d(corrected*corrected,max(1.,fs*.0001))
    first=int(np.flatnonzero(power>.01*power.max())[0])
    time=(np.arange(nfft)-first)/fs
    window=np.clip((.020-time)/.005,0,1)
    early=corrected*window
    if np.sum(early*early)<.5*np.sum(corrected*corrected):
        return initial
    nominal=irfft(rfft(early,nfft)*np.exp(-1j*initial*phase),nfft)
    return _phase_focus_clock(nominal,fs,kit,config)


def _resolved_clock_ridge(packet,fs,kit,config):
    """Use directly resolved wavefronts only with strict majority consensus.

    Seven of twelve bands must agree within half an ADC sample and span an
    octave. This is stronger evidence than the former four-band/two-sample
    rule. It avoids imposing minimum phase on a resolved diffraction/reflection
    ridge. Colored or unresolved responses use spectral phase focusing instead;
    failed consensus never becomes a rejection by itself. Neither path sees
    repeat clocks, geometry, dataset identity or inferred acquisition truth.
    """
    nfft=next_fast_len(4*len(packet))
    frequency=np.fft.rfftfreq(nfft,1/fs)
    spectrum=rfft(packet,nfft)
    high=min(15000.,kit.f2_hz/1.3,fs*.38)
    low=min(1800.,high/4)
    times,centers=[],[]
    for center in np.geomspace(low,high,12):
        left,right=center/1.3,center*1.3
        mask=np.sin(np.pi*np.clip((frequency-left)/(right-left),0,1))**2
        analytic=ifft(np.r_[spectrum*mask,np.zeros(nfft-len(spectrum))])
        envelope=abs(analytic[:len(packet)])
        supported=np.flatnonzero(envelope>.1*envelope.max())
        if not len(supported):
            return None
        first=int(supported[0])
        last=min(len(envelope),first+max(3,round(fs*(4/(right-left)+.001))))
        peak=first+int(np.argmax(envelope[first:last]))
        times.append(fractional_peak(envelope,peak)/fs)
        centers.append((left+right)/2)
    u,y=np.log(centers),np.asarray(times)
    length=kit.sweep_s/np.log(kit.f2_hz/kit.f1_hz)
    fits=[]
    for i,j in combinations(range(len(y)),2):
        slope=(y[j]-y[i])/(u[j]-u[i])
        if abs(slope/length*1e6)>4*config.max_clock_drift_ppm:
            continue
        intercept=y[i]-slope*u[i]
        good=abs(y-(intercept+slope*u))<.5/fs
        if good.sum()<7 or np.ptp(u[good])<np.log(2):
            continue
        fit=np.polyfit(u[good],y[good],1)
        residual=float(np.mean((np.polyval(fit,u[good])-y[good])**2))
        fits.append((int(good.sum()),float(np.ptp(u[good])),-residual,float(fit[0])))
    return max(fits)[3]/length*1e6 if fits else None


def _usable_clock_band(spectrum,fs,kit,nfft):
    """Existing independent-clock support criterion, shared by both estimators.

    This is an abstention boundary, not a clock estimate from a narrow passband.
    See parent-project note ROOM_SWEEP_CLOCK (not distributed) for the 89--178 Hz counterexample.
    """
    magnitude=abs(spectrum)
    if not np.any(magnitude):
        raise DetectionError('no within-sweep phase evidence; record a clearer stationary sweep')
    frequency=np.fft.rfftfreq(nfft,1/fs)
    smooth=gaussian_filter1d(magnitude,max(1.,100/(fs/nfft)))
    usable=(frequency>=max(kit.f1_hz,500.)) & (frequency<=kit.f2_hz) & (smooth>.1*smooth.max())
    if not np.any(usable):
        raise DetectionError('insufficient usable sweep bandwidth for an independent clock; check the playback and device response')
    high=min(float(frequency[np.flatnonzero(usable)[-1]]),fs*.45,kit.f2_hz)
    low=min(max(500.,kit.f1_hz),high/4)
    return low,high


def _phase_focus_clock(packet,fs,kit,config):
    """Independent ESS phase focus after regularized minimum-phase removal.

    A transducer's group delay is not clock drift. Remove the minimum-phase
    spectral factor before testing the ESS clock's allpass phase curvature.
    A flat usable passband avoids weighting one resonant subband as the clock.
    Minimize analytic-envelope energy entropy at fixed spectral energy. Unlike
    a peak/fourth-moment objective, it includes dispersed wavefronts without
    overweighting one dominant reflection; unlike an L1 cost it suppresses the
    influence of numerous weak noise samples. Analytic magnitude also avoids
    fitting an arbitrary carrier phase. This is one-dimensional phase focusing,
    not unconstrained blind RIR inversion.
    Room excess phase remains a conditional limitation, not a calibrated clock
    confidence interval. Repeat timing never supplies an estimate or prior.
    """
    nfft=next_fast_len(len(packet))
    frequency=np.fft.rfftfreq(nfft,1/fs)
    spectrum=rfft(packet,nfft)
    magnitude=abs(spectrum)
    low,high=_usable_clock_band(spectrum,fs,kit,nfft)
    floor=float(magnitude.max())*.01
    edge=min(500.,(high-low)/4)
    # Phase estimation must retain weak stopbands of a causal device filter.
    # Its log floor is separate from the conservative whitening-gain bound.
    minimum_phase=_minimum_phase_angle(spectrum,nfft)
    reference=_compression_phase(fs,kit.sweep_s,kit.f1_hz,kit.f2_hz,
                                  kit.fade_in_s,kit.fade_out_s,nfft)
    whitened=spectrum/np.maximum(magnitude,floor)*np.exp(-1j*minimum_phase+1j*reference)
    whitened*=np.clip((frequency-low)/edge,0,1)*np.clip((high-frequency)/edge,0,1)
    length=kit.sweep_s/np.log(kit.f2_hz/kit.f1_hz)
    phase=2*np.pi*frequency*length*np.log(np.maximum(frequency,kit.f1_hz)/kit.f1_hz)*1e-6
    def score(ppm):
        positive=whitened*np.exp(1j*ppm*phase)
        focused=ifft(np.r_[positive,np.zeros(2*nfft-len(positive))])
        power=abs(focused)**2
        return -float(np.sum(power*np.log(np.maximum(power,1e-30))))
    bound=4*config.max_clock_drift_ppm
    grid=np.linspace(-bound,bound,max(16,int(np.ceil(2*bound/40)))+1)
    best=int(np.argmin([score(ppm) for ppm in grid]))
    optimum=minimize_scalar(score,bounds=(grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]),
                            method='bounded',options={'xatol':.1})
    return float(optimum.x)


def compress_repeat(samples, kit, *, start_sample, stop_sample, drift_ppm=0.):
    """Shared isolation/deconvolution primitive for detector and RIR extraction.

    Returns (linear convolution, recording crop start, original samples per
    corrected sample). Subtract len(inverse)-1 to map its peak to recording time.
    Only one repeat plus guards/tail is transformed, never minutes of silence.
    """
    from .rir import ess_inverse
    lo, hi = max(0, int(start_sample)), min(len(samples), int(stop_sample))
    segment = np.asarray(samples[lo:hi], dtype=float)
    step = 1+drift_ppm*1e-6
    if len(segment) < 2 or not np.isfinite(step) or step <= 0:
        raise DetectionError("incomplete sweep segment; record before playback and through its decay")
    if drift_ppm:
        segment = clock_resample(segment, step)
    inverse = ess_inverse(kit.fs_hz, kit.excitation)
    full = signal.fftconvolve(segment, inverse if segment.ndim == 1 else inverse[:, None], axes=0)
    return full, lo, step


def clock_resample(samples, step: float) -> np.ndarray:
    """Sample at exactly j*step, without rounding the clock ratio to FFT lengths.

    Reuse SciPy's 4x polyphase interpolator followed by cubic Lagrange evaluation
    on that fine grid. Processing is blocked with 32 input samples of context.
    This avoids enormous rational-rate FIRs for ppm-sized ratios. Validated on
    analytic tones through 20 kHz at 48 kHz to 0.01 absolute sample error.
    See J.O. Smith, Physical Audio Signal Processing, Lagrange interpolation:
    https://www.dsprelated.com/freebooks/pasp/Lagrange_Interpolation.html
    """
    x = np.asarray(samples,dtype=float)
    if x.ndim not in (1,2) or len(x)<2 or not np.isfinite(step) or not .999 <= step <= 1.001:
        raise ValueError("clock interpolation requires samples and a finite near-unity step")
    count = int(np.floor((len(x)-1)/step))+1
    out = np.empty((count,)+x.shape[1:])
    for begin in range(0,count,65536):
        positions = np.arange(begin,min(count,begin+65536))*step
        lo,hi = max(0,int(positions[0])-32),min(len(x),int(np.ceil(positions[-1]))+33)
        up = signal.resample_poly(x[lo:hi],4,1,axis=0)
        indices = (positions-lo)*4
        base = np.floor(indices).astype(int)
        u = indices-base
        coefficients = (-u*(u-1)*(u-2)/6, (u+1)*(u-1)*(u-2)/2,
                        -(u+1)*u*(u-2)/2, (u+1)*u*(u-1)/6)
        values = np.zeros((len(positions),)+x.shape[1:])
        for offset,weight in zip((-1,0,1,2),coefficients):
            index=base+offset
            weight=weight*((index>=0)&(index<len(up)))
            values += (weight if x.ndim==1 else weight[:,None])*up[np.clip(index,0,len(up)-1)]
        out[begin:begin+len(positions)]=values
    return out


def fractional_crop(samples, start: float, count: int) -> np.ndarray:
    """Bandlimited crop on a fractional grid, with padded linear FFT support.

    This numerical primitive is shared by RIR alignment and the known-timing
    benchmark oracle; it does not identify any sweep or choose an arrival.
    """
    integer = int(np.floor(start))
    if integer < 0 or integer+count > len(samples):
        raise ValueError("fractional crop exceeds sample support")
    fraction = start-integer
    pad = 64
    a,b = max(0,integer-pad), min(len(samples),integer+count+pad)
    packet = samples[a:b]
    nfft = next_fast_len(len(packet)+2*pad)
    shifted = irfft(rfft(packet,nfft)*np.exp(2j*np.pi*np.fft.rfftfreq(nfft)*fraction),nfft)
    return shifted[integer-a:integer-a+count]


def detect_sweeps(samples, fs: int, kit: SweepKit, *, config: DetectionConfig | None = None) -> SweepDetection:
    """Find >=N repeats at any offset, returning fractional *file-rate* coordinates.

    44.1 kHz input is correctly resampled for the exact 48 kHz reference. Both
    spacing and chirp mismatch clocks use positive ppm for longer recordings.
    Dropout slots are relative: with a missing edge repeat, its absolute kit
    ordinal is unobservable. Never infer an emission timestamp from file start.
    """
    x = _mono(samples, fs)
    try:
        return _detect_sweeps(x,fs,kit,config=config)
    except DetectionError as exc:
        # Clipping can destroy the clock ridge before a result is constructed.
        # Preserve the acquisition flag on EVERY rejection path, too.
        if clipping_present(x):
            raise DetectionError(f'clipping detected; lower fixed gain and re-record. {exc}') from exc
        raise


def _detect_sweeps(x,fs,kit,*,config):
    """Implementation behind the public rejection/quality-report boundary."""
    config = config or DetectionConfig()
    if kit.repeats < config.min_repeats:
        raise DetectionError(f"kit has fewer than {config.min_repeats} repeats; choose a compatible minimum or record the four-repeat kit")
    if min(fs, kit.fs_hz) <= 2*kit.f1_hz:
        raise DetectionError("recording sample rate cannot represent the kit")
    if fs != kit.fs_hz:
        g = np.gcd(fs, kit.fs_hz)
        converted = signal.resample_poly(x, kit.fs_hz//g, fs//g)
        result = detect_sweeps(converted, kit.fs_hz, kit, config=config)
        scale = fs/kit.fs_hz
        result.fs = fs
        result.spacing_residual_samples *= scale
        for r in result.repeats:
            r.start_sample *= scale
            r.end_sample *= scale
            r.segment_start = int(np.floor(r.segment_start*scale))
            r.segment_stop = min(len(x), int(np.ceil(r.segment_stop*scale)))
        result.clipped = clipping_present(x)
        for r in result.repeats:
            r.clipped = clipping_present(x[r.segment_start:r.segment_stop])
        return result
    decimation = max(1, fs//6000)
    coarse_fs = fs/decimation
    coarse_audio = signal.resample_poly(x, 1, decimation) if decimation > 1 else np.asarray(x, dtype=float)
    y = np.diff(coarse_audio, prepend=0)
    template = _coarse_template(kit, coarse_fs)
    if len(y) < len(template)+(config.min_repeats-1)*kit.period_s*coarse_fs:
        raise DetectionError(f"recording too short for {config.min_repeats} repeats; capture the whole playback")
    score = _normalized_search(y, template)
    peaks, _ = signal.find_peaks(abs(score), height=config.min_correlation,
                                 distance=max(1, round(.7*kit.period_s*coarse_fs)))
    if len(peaks) > config.max_candidates:
        raise DetectionError("too many sweep candidates; select one playback take and remove unrelated sessions")
    groups = _train_candidates(peaks, coarse_fs, kit, config)
    clock_fit = None
    if len(groups) == 1 and len(groups[0]) == kit.repeats:
        starts = peaks[list(groups[0])]
        slots = np.rint((starts-starts[0])/(kit.period_s*coarse_fs)).astype(int)
        strongest = int(np.argmax(abs(score[starts])))
        try:
            clock_fit = _repeat_clock(y, starts, slots, kit.period_s*coarse_fs, coarse_fs, config, strongest)
        except DetectionError:
            pass  # A different reflection may have won each coarse peak.
    if clock_fit is not None:
        times, ppm, correlations, residual = clock_fit
    else:
        # Keep the same normalized evidence threshold. A bounded scale bank
        # restores coherent peaks smeared by clock error in diffuse rooms.
        score = abs(score)
        for clock in np.linspace(-config.max_clock_drift_ppm, config.max_clock_drift_ppm, 7):
            if clock == 0:
                continue
            other = abs(_normalized_search(y, _coarse_template(kit, coarse_fs, clock)))
            n = min(len(score),len(other))
            score[:n] = np.maximum(score[:n],other[:n])
        starts, slots, times, ppm, correlations, residual = _room_train(y,score,coarse_fs,kit,config)
    # Compare aligned raw sweep blocks to a majority, without averaging bad takes.
    width = len(template)
    aligned = []
    for t in times:
        lo = max(0, int(t)-32)
        # Pre-emphasis is useful for locating a narrow peak, but it suppresses
        # the low-frequency half of an ESS relative to broadband noise. Quality
        # checks must use the original bandlimited audio, not that search filter.
        segment = signal.resample_poly(coarse_audio[lo:min(len(y),int(t)+width+32)], 8, 1)
        aligned.append(np.interp((t-lo+np.arange(width))*8, np.arange(len(segment)), segment))
    aligned = np.array(aligned)
    bad = np.zeros(len(times), dtype=bool)
    incoherent_blocks = np.zeros(len(times), dtype=int)
    for block in np.array_split(aligned, 8, axis=1):
        energy = np.sum(block*block, axis=1)
        norms = np.sqrt(np.maximum(energy[:,None]*energy[None,:], 1e-30))
        corr = (block@block.T)/norms
        np.fill_diagonal(corr, 0)
        support = np.sum(np.abs(corr), axis=1)/(len(times)-1)
        median = np.median(support)
        incoherent_blocks += (support < .4*median) & (median > .15)
        bad |= energy > 4*np.median(energy)+1e-20
    bad |= incoherent_blocks >= 2
    dropped = [dict(slot=int(slots[i]), reason="repeat waveform inconsistent with the majority (masked, dropout or unrelated sound)")
               for i in np.flatnonzero(bad)]
    for slot in range(int(slots[-1])+1):
        if slot not in slots:
            dropped.append(dict(slot=slot, reason="no confident sweep at the expected repeat spacing"))
    while len(dropped)+len(times)-int(bad.sum()) < kit.repeats:
        dropped.append(dict(slot=None, reason="missing edge repeat; absolute kit ordinal is unobservable"))
    healthy = np.flatnonzero(~bad)
    if len(healthy) < config.min_repeats:
        raise DetectionError(f"only {len(healthy)} coherent sweep repeats; need {config.min_repeats}; remove handling/dropouts and re-record")
    # Independent estimates on a majority resist a nuisance event in one
    # wavefront. Neither spacing ppm nor a bound centred on it is supplied.
    selected = healthy[np.argsort(-abs(score[starts[healthy]]))[:3]]
    chirp_estimates = np.array([_chirp_clock(x,times[i]*decimation,fs,kit,config) for i in selected])
    chirp_ppm = float(np.median(chirp_estimates))
    clock_resolution = 1e6/(kit.sweep_s*coarse_fs)
    clock_tolerance = max(60., 4*clock_resolution)
    # A median is robust only when a majority measures the same quantity.
    # Unrelated noise optima can have a lucky middle value near the repeat
    # clock, even for a completely different ESS. Require majority support
    # at the existing clock tolerance before consulting repeat spacing.
    if np.count_nonzero(abs(chirp_estimates-chirp_ppm) <= clock_tolerance) <= len(selected)//2:
        raise DetectionError("independent chirp clocks lack repeat consensus; check the exact playback file and record clearer sweeps")
    if abs(chirp_ppm-ppm) > clock_tolerance:
        raise DetectionError("chirp-rate and repeat-spacing clocks disagree; check the exact playback file, rate conversion or variable clock")
    repeats = []
    ratio = 1+ppm*1e-6
    # Low-frequency room peaks can lag the broadband arrival. This search
    # context is distinct from the user-visible isolation guard.
    guard = max(config.guard_s, .2)*fs
    for i, (slot, start) in enumerate(zip(slots, times*decimation)):
        if bad[i]:
            continue
        lo = max(0, int(np.floor(start-guard)))
        hi = min(len(x), int(np.floor(start+kit.period_s*fs*ratio)))
        full, origin, step = compress_repeat(x, kit, start_sample=lo, stop_sample=hi, drift_ppm=ppm)
        expected = (start-origin)/step + round(kit.sweep_s*fs)-1
        a, b = max(0, int(expected-guard)), min(len(full), int(expected+guard+1))
        refined,p = linear_arrival(full,fs,a,b,bandwidth_hz=kit.f2_hz)
        local = np.arange(max(0,p-32),min(len(full),p+33))
        weights = np.sinc(refined-local)*np.kaiser(len(local),6)
        amplitude = float(abs(np.dot(full[local],weights)/weights.sum()))
        arrival = origin+(refined-(round(kit.sweep_s*fs)-1))*step
        # Local sidelobes provide a robust compressed-noise diagnostic.
        flank = np.r_[full[max(0,p-round(.2*fs)):max(0,p-round(.02*fs))],
                      full[p+round(.02*fs):p+round(.2*fs)]]
        floor = np.median(abs(flank))/.67449 if len(flank) else 0.
        snr = float(20*np.log10(max(abs(full[p]), 1e-30)/max(floor, 1e-30)))
        end = arrival+kit.sweep_s*fs*ratio
        if snr < config.min_peak_snr_db or arrival < 0 or end > len(x):
            dropped.append(dict(slot=int(slot), reason="weak compressed peak or incomplete sweep; re-record at a clear fixed level"))
            continue
        repeats.append(RepeatDetection(int(slot), float(arrival), float(end),
                       float(np.clip(1-10**(-snr/20), 0, 1)), snr, 1 if full[p] >= 0 else -1,
                       max(0, int(np.floor(arrival-config.guard_s*fs))),
                       min(len(x), int(np.floor(arrival+kit.period_s*fs*ratio))),
                       clipping_present(x[lo:hi]),amplitude))
    if repeats:
        # The low-rate search cannot see masking above its Nyquist frequency.
        # The full-band Farina packet supplies that missing integrity check.
        median_peak = np.median([r.linear_peak_amplitude for r in repeats])
        weak = [r for r in repeats if r.linear_peak_amplitude < .5*median_peak]
        for r in weak:
            dropped.append(dict(slot=r.slot,reason="full-band linear peak lost more than half its majority amplitude; partial sweep masking or gain dropout"))
        repeats = [r for r in repeats if r not in weak]
        majority = 1 if sum(r.polarity for r in repeats) >= 0 else -1
        for r in repeats:
            if r.polarity != majority:
                dropped.append(dict(slot=r.slot,reason="repeat polarity disagrees with the majority; inverted or corrupted packet"))
        repeats = [r for r in repeats if r.polarity == majority]
    if len(repeats) < config.min_repeats:
        raise DetectionError(f"only {len(repeats)} confident healthy sweep repeats; need {config.min_repeats}. Re-record the complete kit without handling noise/dropouts")
    messages = ["absolute emission time is unobservable; first supported linear packet may be a reflection"]
    clipped = clipping_present(x)
    if clipped:
        messages.append("clipping detected; lower fixed gain and re-record before using a linear RIR")
    return SweepDetection(fs, repeats, dropped, ppm, chirp_ppm, residual*decimation,
                          correlations, clipped, messages,
                          chirp_estimation_method="strict_ridge_then_spectral_factor_entropy/1" if kit.sweep_s>=2. else
                                                  "early_peak_phase_focus/1")


def isolate_sweeps(samples, detection: SweepDetection):
    """Yield (repeat metadata, sample view): guard, full sweep, available decay to next slot.

    No copies, clock correction or polarity correction. A dropped slot is never
    appended to its neighbour's tail. Coordinates refer to the original file.
    """
    x = _mono(samples, detection.fs)
    for repeat in detection.repeats:
        yield repeat, x[repeat.segment_start:repeat.segment_stop]


@dataclass(frozen=True)
class EventRegion:
    onset_sample: int
    offset_sample: int
    segment_start: int
    segment_stop: int
    noise_rms: float
    peak_snr_db: float
    tail_truncated: bool


@dataclass(frozen=True)
class EventCandidates:
    """Small, gain-independent descriptors; recording samples are not retained."""
    fs: int
    regions: list[EventRegion]
    features: np.ndarray
    envelopes: list[np.ndarray] = field(default_factory=list)
    response_spectrum: np.ndarray | None = None
    response_envelope: np.ndarray | None = None
    spectral_envelopes: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class EventSelection:
    event_index: int | None
    method: str
    reason: str


def describe_events(samples, fs, noise_samples, *, response=None, **kwargs) -> EventCandidates:
    """Describe energy regions without assigning meaning to a sound or label.

    Welch band powers and energy quantile durations tolerate independent source
    performances, level changes and modest room coloration. They do not identify
    speech content. Silence and the requested decay guard are excluded. Matching
    deliberately abstains when several recurring motifs explain a label.
    """
    regions = detect_events(samples, fs, noise_samples, **kwargs)
    edges = np.geomspace(40., min(fs*.49, 16000.), 25)
    def spectrum(x, *, impulse=False):
        nperseg = min(len(x), max(32, round(.064*fs)))
        if impulse:
            nfft = next_fast_len(max(len(x),round(.128*fs)))
            f, p = np.fft.rfftfreq(nfft,1/fs), abs(rfft(x,nfft))**2
        else:
            f, p = signal.welch(x, fs=fs, nperseg=nperseg, nfft=next_fast_len(max(nperseg,round(.128*fs))))
        # Integrate PSD (including narrow tones); interpolation gives each band
        # support even for deliberately short, low-rate test signals.
        cumulative = np.r_[0., np.cumsum(p)]
        return np.diff(np.interp(edges, np.r_[0.,f+fs/(2*(len(f)-1))], cumulative))
    noise = spectrum(np.asarray(noise_samples))
    width = max(1, round(.01*fs))
    features, envelopes, spectral_envelopes = [], [], []
    for r in regions:
        x = np.asarray(samples[r.onset_sample:r.offset_sample], dtype=float)
        power = np.maximum(spectrum(x)-noise, 0.)
        relative = np.maximum(power/max(float(power.sum()), 1e-30), 1e-5)
        spectral = 10*np.log10(relative)
        spectral -= spectral.mean()
        energy = np.maximum(np.add.reduceat(x*x, np.arange(0,len(x),width))-width*r.noise_rms**2, 0.)
        envelopes.append(energy/max(float(energy.sum()),1e-30))
        features.append(np.r_[spectral, _energy_quantiles(energy)])
        spectral_envelopes.append(_spectral_envelope(x, fs, edges, r.noise_rms))
    color, decay = None, None
    if response is not None:
        h = _mono(response,fs)
        color = 10*np.log10(np.maximum(spectrum(h,impulse=True), 1e-30))
        color -= color.mean()
        decay = np.add.reduceat(h*h,np.arange(0,len(h),width))
        decay /= max(float(decay.sum()),1e-30)
    return EventCandidates(fs, regions, np.asarray(features).reshape((-1,28)),
                           envelopes,color,decay,spectral_envelopes)


def _spectral_envelope(x, fs, edges, noise_rms):
    """At most 64 log-band frames; silence/decay export boundaries stay intact.

    Analyze 32 ms Hann windows at 10 ms spacing, uniformly subsampling longer
    regions. Overall level is removed, while frequency order and modulation
    remain observable. Unlike waveform fingerprints, phase is discarded.
    """
    width, hop = max(8, round(.032*fs)), max(1, round(.01*fs))
    centers = np.arange(0, len(x), hop)
    if len(centers) > 64:
        centers = centers[np.round(np.linspace(0, len(centers)-1, 64)).astype(int)]
    padded = np.pad(x, (width//2, width))
    frames = np.array([padded[c:c+width] for c in centers])*np.hanning(width)
    nfft = next_fast_len(max(width, round(.128*fs)))
    power = abs(rfft(frames, nfft, axis=1))**2
    freq = np.fft.rfftfreq(nfft, 1/fs)
    # Non-overlapping bands; white noise is subtracted in the same FFT units.
    power = np.maximum(power-noise_rms**2*np.sum(np.hanning(width)**2), 0.)
    cumulative = np.pad(np.cumsum(power, axis=1), ((0,0),(1,0)))
    bands = np.array([np.diff(np.interp(edges, np.r_[0.,freq+fs/nfft], row))
                      for row in cumulative])
    bands /= max(float(np.mean(bands.sum(axis=1))), 1e-30)
    return 10*np.log10(np.maximum(bands, 1e-5))


def _bounded_dtw(cost):
    """Whole-sequence, weighted-path DTW; no subsequence/unchecked time skips.

    Standard recurrence with (1,1)/(1,2)/(2,1) steps and a 25% normalized
    Sakoe-Chiba band. The step weights give all complete paths the same total
    weight, so minimizing cumulative cost also minimizes normalized cost.
    See parent-project note ROOM_EVENT_MATCH_PROTOCOL (not distributed) and librosa.sequence.dtw documentation.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not cost.size or not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError('DTW requires a finite nonnegative cost matrix')
    n, m = cost.shape
    if max(n,m) > 2*min(n,m) or max(n,m) > 64:
        return float('inf')
    if min(n,m) == 1:
        return float(cost[0,0]) if n == m else float('inf')
    accumulated = np.full((n,m), np.inf)
    accumulated[0,0] = cost[0,0]
    for i in range(1,n):
        for j in range(max(1, int(np.ceil((i/(n-1)-.25)*(m-1)))),
                       min(m, int(np.floor((i/(n-1)+.25)*(m-1)))+1)):
            accumulated[i,j] = min(
                accumulated[i-1,j-1]+cost[i,j],
                accumulated[i-1,j-2]+1.5*cost[i,j] if j>=2 else np.inf,
                accumulated[i-2,j-1]+1.5*cost[i,j] if i>=2 else np.inf)
    return float(accumulated[-1,-1]/((n+m)/2))


def _energy_quantiles(energy):
    cumulative = np.cumsum(energy)
    q = np.interp(np.array([.1,.25,.5,.75,.9])*cumulative[-1], cumulative,
                  (np.arange(len(energy))+.5)*.01)
    return np.log2(np.maximum([q[4]-q[0], q[2]-q[0], q[4]-q[2], q[3]-q[1]], .01))


def _spectral_distance(a,b,correction=0.):
    # Silent bins do not acquire fictitious evidence when room coloration is
    # removed. A band present in EITHER event retains weight (different tones
    # must remain different); remove only a common gain.
    weight = np.maximum(.05,10**(np.maximum(a-a.max(),b-b.max())/20))
    delta = a-b+correction
    delta -= np.average(delta,weights=weight)
    return float(np.average((delta/8.)**2,weights=weight))


def select_labeled_events(candidates: dict, labels: dict, *, hints: dict | None = None) -> dict:
    """Find a unique recurring spectral/temporal motif across same-label takes.

    Labels are opaque identifiers. Matching uses neither filename/order, clocks,
    truth sources nor held-out positions. A match needs a peer recording, with
    RMS spectral disagreement <=8 dB and energy durations within roughly a
    factor of two (joint normalized distance <=1). Motifs recurring across two
    *other* labels are non-discriminating background, not label evidence.
    More than one supported motif is ambiguous, regardless of loudness. A
    unique motif may anchor other performances with a 1.5x distance margin.
    A bounded joint spectral-envelope DTW stage can resolve remaining ambiguity.
    Hints map take keys to {event_index: ...} or {event_window_s: ...}; they
    override only that take AFTER automatic inference, never becoming anchors.
    """
    if set(candidates) != set(labels):
        raise ValueError("candidate and label keys must agree")
    hints = hints or {}
    selected = {}
    vertices = [(key,i) for key,c in candidates.items() for i in range(len(c.regions))]
    if len(vertices) > 256:
        selected = {key: EventSelection(None, 'ambiguous',
                    'candidate budget exceeded (256); supply event_window_s or re-record isolated performances')
                    for key in candidates}
        return _apply_event_hints(candidates, selected, hints)
    features = np.array([candidates[key].features[i] for key,i in vertices])
    raw_features=features.copy()
    band_power = 10**((features[:,:24]-features[:,:24].max(axis=1,keepdims=True))/10) if len(features) else np.empty((0,24))
    band_power /= np.maximum(band_power.sum(axis=1,keepdims=True),1e-30)
    common_bands = np.zeros(24,dtype=bool)
    for a,(key,i) in enumerate(vertices):
        tone_labels = {labels[key]:band_power[a]>.01*band_power[a].max()}
        for b,(other,j) in enumerate(vertices):
            if labels[other] == labels[key]:
                continue
            tonal = min(band_power[a].max(),band_power[b].max()) > .4
            overlap = float(np.sum(np.sqrt(band_power[a]*band_power[b])))
            if tonal and overlap > .95:
                label = labels[other]
                tone_labels[label] = tone_labels.get(label,np.zeros(24,dtype=bool)) | (band_power[b]>.01*band_power[b].max())
        if len(tone_labels)>=3:
            common_bands |= np.sum(list(tone_labels.values()),axis=0)>=3
    # One neighbouring band covers finite-window leakage when the nuisance's
    # duration changes; otherwise its sidelobe can become a spurious new motif.
    common_bands=np.convolve(common_bands.astype(int),[1,1,1],mode='same')>0
    # A recurring narrow-band nuisance may overlap the intended event. Ignore
    # its learned bands for SIGNATURE comparison; never alter delivered audio.
    # Keep only bands shared by three labels so one mixed target cannot teach
    # its own identifying frequencies as background.
    remaining = band_power[:,~common_bands].sum(axis=1)
    band_power[:,common_bands]=0.
    band_power /= np.maximum(remaining[:,None],1e-30)
    if len(features):
        delta = features[:,None,:]-features[None,:,:]
        distances = np.sqrt(np.array([[_spectral_distance(a[:24],b[:24]) for b in features] for a in features])+
                            np.mean(delta[:,:,24:]**2,axis=2))
    else:
        distances = np.empty((0,0))
    background = {}
    for a,(key,i) in enumerate(vertices):
        # Broad impulses have uncertain finite-sample spectra. Match their
        # envelope too; a shared tone's duration need not be invariant.
        shared = {labels[other] for b,(other,j) in enumerate(vertices)
                  if labels[other]!=labels[key] and
                  ((np.mean((raw_features[a,24:]-raw_features[b,24:])**2)<=.5**2 and
                    _spectral_distance(raw_features[a,:24],raw_features[b,:24])<=.25**2) or
                   (max(2**raw_features[a,24],2**raw_features[b,24])<=.03 and
                    max(band_power[a].max(),band_power[b].max())<.4))}
        background[key,i] = remaining[a]<.01 or len(shared)>=2
    for a,(key,i) in enumerate(vertices):
        ca = candidates[key]
        for b in range(a+1,len(vertices)):
            other,j = vertices[b]
            cb = candidates[other]
            if labels[key] != labels[other]:
                continue
            # Energy cross-convolution: x_i*h_j and x_j*h_i have the same
            # expected envelope/spectrum for a shared source family. This
            # accounts for different decay tails without inverting room notches.
            if ca.response_spectrum is not None and cb.response_spectrum is not None:
                correction=cb.response_spectrum-ca.response_spectrum
                spectral = _spectral_distance(features[a,:24],features[b,:24],correction)
                temporal = (_energy_quantiles(np.convolve(ca.envelopes[i],cb.response_envelope))-
                            _energy_quantiles(np.convolve(cb.envelopes[j],ca.response_envelope)))
                distances[a,b] = distances[b,a] = np.sqrt(spectral+np.mean(temporal**2))
            # A steady tone's identifying frequencies survive room coloration;
            # independently performed durations need not match. Broadband
            # events still require temporal agreement.
            overlap=float(np.sum(np.sqrt(band_power[a]*band_power[b])))
            if min(band_power[a].max(),band_power[b].max())>.4 and overlap>.95:
                distances[a,b]=distances[b,a]=min(distances[a,b],np.sqrt(max(0.,1-overlap)/.05))
    positions = {v:i for i,v in enumerate(vertices)}
    for key, c in candidates.items():
        peers = [other for other in candidates if other != key and labels[other] == labels[key]]
        supported = []
        for i in range(len(c.regions)):
            row = distances[positions[key,i]]
            matches = [(other,j) for n,(other,j) in enumerate(vertices)
                       if other != key and row[n] <= 1. and
                       not background[other,j]]
            peer_distance = min((row[n] for n,(other,j) in enumerate(vertices) if other in peers),default=np.inf)
            support = {other for other,j in matches if other in peers}
            consistent = (bool(support) and
                          peer_distance <= 1. and not background[key,i])
            if consistent:
                supported.append(i)
        if len(supported) == 1:
            selected[key] = EventSelection(supported[0],'cross-take signature',
                                          'unique same-label spectral and temporal agreement')
        elif not peers and len(c.regions) == 1:
            selected[key] = EventSelection(0,'unique energy region','only one region above measured noise')
        else:
            selected[key] = EventSelection(None,'ambiguous',
                f'{len(supported)} recurring candidates among {len(c.regions)} detected events; '
                'supply event_window_s or event_index for this take, or re-record an audible isolated performance')
    # A uniquely supported motif can identify its less clear performances.
    # Use the original decisions as anchors once, never recursively amplify
    # weak matches. Competing coherent motifs generate no anchor in the first
    # place. Require a 1.5x separation from the next candidate, in addition to
    # the unchanged compatibility bound.
    seeds = {key:s.event_index for key,s in selected.items() if s.event_index is not None}
    for key,c in candidates.items():
        if selected[key].event_index is not None or not c.regions:
            continue
        peers = [(other,i) for other,i in seeds.items() if labels[other]==labels[key] and other!=key]
        if not peers:
            continue
        if any(distances[positions[a],positions[b]]>1. for a,b in combinations(peers,2)):
            continue
        scores = np.array([max(distances[positions[key,i],positions[p]] for p in peers) for i in range(len(c.regions))])
        for i in range(len(c.regions)):
            if background[key,i]:
                scores[i] = np.inf
        ranked = np.argsort(scores)
        if scores[ranked[0]] <= 1. and (len(ranked)==1 or scores[ranked[1]] > 1.5*scores[ranked[0]]):
            selected[key] = EventSelection(int(ranked[0]),'cross-take signature',
                'unique match to independently supported same-label motif; nearest alternative >1.5x farther')
    selected = _joint_event_matches(candidates, labels, selected, background, common_bands)
    return _apply_event_hints(candidates, selected, hints)


def _apply_event_hints(candidates, selected, hints):
    """Annotation authority is local to a take, including invalid annotations."""
    for key, c in candidates.items():
        hint = hints.get(key, {})
        if hint.get('event_index') is None and hint.get('event_window_s') is None:
            continue
        try:
            region = select_event_region(c.regions, c.fs, **hint)
        except DetectionError as error:
            selected[key] = EventSelection(None, 'invalid manifest hint', str(error))
        else:
            selected[key] = EventSelection(c.regions.index(region), 'manifest hint',
                'explicit recorded-time annotation; applies only to this take, never an automatic anchor')
    return selected


def _envelope_distance(ca, i, cb, j, common_bands):
    """Level-independent spectral evolution and modulation; not source separation."""
    if not ca.spectral_envelopes or not cb.spectral_envelopes:
        return np.inf  # Legacy descriptors alone cannot justify additional coverage.
    left, right = ca.spectral_envelopes[i].copy(), cb.spectral_envelopes[j].copy()
    def band_power_grid(sequence, candidate, index):
        frames = len(candidate.envelopes[index])
        power = np.array([np.interp(np.linspace(0.,1.,frames),
                                   np.linspace(0.,1.,len(sequence)),band)
                          for band in (10**(sequence/10)).T]).T
        # Use Welch's better-resolved static spectrum for band weights. A short
        # STFT window leaks a strong 75 Hz tone into adjacent low bands; that
        # leakage must not define the target's duration after masking the tone.
        power *= 10**(candidate.features[index,:24]/10)/np.maximum(power.mean(axis=0),1e-30)
        return power
    left_power, right_power = band_power_grid(left,ca,i), band_power_grid(right,cb,j)
    usable = ~common_bands
    if not np.any(usable):
        return np.inf
    ea,eb = (left_power[:,usable].sum(axis=1),right_power[:,usable].sum(axis=1)) if np.any(common_bands) else (ca.envelopes[i],cb.envelopes[j])
    duration = _energy_quantiles(ea)[0]-_energy_quantiles(eb)[0]
    if ca.response_envelope is not None and cb.response_envelope is not None:
        duration = (_energy_quantiles(np.convolve(ea,cb.response_envelope))[0]-
                    _energy_quantiles(np.convolve(eb,ca.response_envelope))[0])
        def cross_envelope(power, response):
            power = signal.fftconvolve(power,response[:,None],axes=0)
            total = np.maximum(power[:,usable].sum(axis=1),0.)
            cumulative = np.cumsum(total)/max(float(total.sum()),1e-30)
            lo,hi = np.searchsorted(cumulative,[.01,.99])
            power = power[lo:hi+1]
            if len(power)>64:
                power = np.array([np.interp(np.linspace(0.,1.,64),
                                           np.linspace(0.,1.,len(power)),band)
                                  for band in power.T]).T
            power /= max(float(power.mean()),1e-30)
            return 10*np.log10(np.maximum(power,1e-5))
        left = cross_envelope(left_power,cb.response_envelope)
        right = cross_envelope(right_power,ca.response_envelope)
    if abs(duration) > 1.:
        return np.inf
    # Remove static channel coloration, retaining the absolute spectral evidence
    # in the independent coarse score. Per-band centering keeps spectral changes
    # and amplitude modulation, without equalizing or changing any output audio.
    left -= np.mean(left, axis=0)
    right -= np.mean(right, axis=0)
    weight = 10**(np.maximum(ca.features[i,:24]-ca.features[i,:24].max(),
                            cb.features[j,:24]-cb.features[j,:24].max())/20)
    weight[common_bands] = 0.
    if weight.sum() <= 1e-12:
        return np.inf
    weight /= weight.sum()
    cost = np.sum(((left[:,None,:]-right[None,:,:])/6.)**2*weight, axis=2)
    dl, dr = np.diff(left,axis=0,prepend=left[:1]), np.diff(right,axis=0,prepend=right[:1])
    cost += .25*np.sum(((dl[:,None,:]-dr[None,:,:])/6.)**2*weight, axis=2)
    return _bounded_dtw(cost)


def _joint_event_matches(candidates, labels, selected, background, common_bands):
    """Score complete hypotheses including abstentions, with no anchor cascade.

    Previously unique AUTO decisions are fixed observations in this extension.
    New decisions are all derived from the SAME hypothesis table, never used as
    seeds for another iteration. Missing takes incur one unit divided by group
    size; every included pair must independently meet the compatibility bound.
    """
    result = dict(selected)
    for label in dict.fromkeys(labels.values()):
        keys = [k for k in candidates if labels[k] == label]
        unresolved = [k for k in keys if selected[k].event_index is None]
        if not unresolved:
            continue
        choices = [[selected[k].event_index] if selected[k].event_index is not None else
                   [None, *(i for i in range(len(candidates[k].regions)) if not background[k,i])]
                   for k in keys]
        budget = 1
        for options in choices:
            budget *= len(options)
        reason = 'insufficient compatible peer evidence'
        hypotheses = []
        # A new two-file cluster cannot establish an otherwise unresolved
        # three-take label. It may be a nuisance with an inaudible third target.
        # Missing-take hypotheses remain available when AUTO already identified
        # the label independently, or only two takes were actually supplied.
        minimum_support = 2 if any(selected[k].event_index is not None for k in keys) else min(3,len(keys))
        if budget > 4096 or len(keys) > 8:
            reason = 'joint hypothesis budget exceeded (4096 assignments / 8 takes)'
        else:
            pair_cost = {}
            for ka,kb in combinations(range(len(keys)),2):
                a,b = keys[ka],keys[kb]
                for i,j in product(choices[ka],choices[kb]):
                    if i is None or j is None:
                        continue
                    ca,cb = candidates[a],candidates[b]
                    correction = (cb.response_spectrum-ca.response_spectrum
                                  if ca.response_spectrum is not None and cb.response_spectrum is not None else np.zeros(24))
                    # Coarse filtering is SPECTRAL only. Quantile-time distances
                    # cannot preempt DTW: an energy median can jump between two
                    # lobes when their relative energies change only slightly.
                    usable = ~common_bands
                    coarse = (min(_spectral_distance(ca.features[i,:24][usable],cb.features[j,:24][usable],correction[usable]),
                                  _spectral_distance(ca.features[i,:24][usable],cb.features[j,:24][usable]))
                              if np.any(usable) else np.inf)
                    fine = _envelope_distance(candidates[a],i,candidates[b],j,common_bands) if coarse<=1. else np.inf
                    pair_cost[ka,i,kb,j] = .5*(coarse+fine) if fine<=1. else np.inf
            for assignment in product(*choices):
                members = [(n,i) for n,i in enumerate(assignment) if i is not None]
                if len(members)<max(2,minimum_support):
                    continue
                costs = [pair_cost[ka,i,kb,j] for (ka,i),(kb,j) in combinations(members,2)]
                if max(costs)>1.:
                    continue
                score = float(np.mean(costs)+(len(keys)-len(members))/len(keys))
                hypotheses.append((score,assignment))
        # Two disjoint, fully recurring motifs cannot acquire semantic identity
        # from a small cost difference (e.g. two separately repeated tones).
        complete = [h for _,h in hypotheses if None not in h]
        competing = any(all(i != j for i,j in zip(a,b))
                        for a,b in combinations(complete,2))
        for n,key in enumerate(keys):
            if key not in unresolved:
                continue
            if not candidates[key].regions:
                detail = 'no event above measured noise floor'
            elif competing:
                detail = 'competing coherent same-label motifs'
            elif not hypotheses:
                detail = reason
            else:
                scores = {i:min((s for s,h in hypotheses if h[n]==i),default=np.inf)
                          for i in choices[n]}
                # None participates in the ranking; an equal score never resolves
                # by index, file order, loudness, or how many tied hypotheses exist.
                ordered = sorted(scores, key=scores.get)
                best = ordered[0]
                first = scores[best]
                second = scores[ordered[1]] if len(ordered)>1 else np.inf
                if best is not None and first<=1. and second>max(1.5*first,first+.1):
                    result[key] = EventSelection(best, 'joint spectral-envelope DTW',
                        f'joint same-label match score={first:.4g}; alternative={second:.4g}; no hint or recursive anchor')
                    continue
                detail = f'competing or missing motifs; joint score={first:.4g}, alternative={second:.4g}'
            result[key] = EventSelection(None, 'ambiguous', detail+
                '; supply event_window_s or event_index for this take, or re-record an audible isolated performance')
    return result


def detect_events(samples, fs: int, noise_samples, *, pre_roll_s=.1, post_roll_s=.5,
                  threshold_db=10., release_db=3., frame_s=.01, min_event_s=.01,
                  merge_gap_s=.15) -> list[EventRegion]:
    """Energy onset/offset against a measured, same-gain noise recording.

    Returns ALL event regions. Cannot tell intended sound from speech, doors or
    handling. Use isolate_event with an explicit index when multiple events exist.
    post_roll_s should cover the measured reverberant decay (e.g. the largest T60).
    Noise-floor uncertainty and frame resolution bound onset/offset accuracy.
    """
    x, noise = _mono(samples, fs), _mono(noise_samples, fs)
    if (not np.all(np.isfinite([pre_roll_s, post_roll_s, threshold_db, release_db, frame_s, min_event_s, merge_gap_s])) or
            min(pre_roll_s, post_roll_s, min_event_s, merge_gap_s) < 0 or
            frame_s <= 0 or not 0 <= release_db < threshold_db):
        raise ValueError("invalid event segmentation parameters")
    size = max(1, round(frame_s*fs))
    def powers(a):
        count = (len(a)+size-1)//size
        power = np.add.reduceat(np.square(a, dtype=float), np.arange(count)*size)
        return power/np.minimum(size, len(a)-np.arange(count)*size)
    floor = max(float(np.median(powers(noise))), np.finfo(float).tiny)
    p = powers(x)
    active = p > floor*10**(release_db/10)
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    merged = []
    for a,b in zip(starts, stops):
        if merged and (a-merged[-1][1])*size/fs <= merge_gap_s:
            merged[-1][1] = int(b)
        else:
            merged.append([int(a),int(b)])
    result = []
    for a,b in merged:
        if (b-a)*size/fs < min_event_s or max(p[a:b]) < floor*10**(threshold_db/10):
            continue
        onset, offset = a*size, min(len(x), b*size)
        stop = offset+round(post_roll_s*fs)
        result.append(EventRegion(onset, offset, max(0,onset-round(pre_roll_s*fs)),
                      min(len(x),stop), float(np.sqrt(floor)),
                      float(10*np.log10(max(p[a:b])/floor)), stop > len(x)))
    return result


def select_event_region(regions, fs, *, event_index=None, event_window_s=None):
    """Resolve explicit recorded-time hints, retaining the full region and tail.

    Windows identify an event by intersection; they do not crop reverberation.
    A window touching several regions is ambiguous, even if one is louder.
    """
    if event_index is not None and event_window_s is not None:
        raise DetectionError("supply event_index or event_window_s, not both")
    if not regions:
        raise DetectionError("no real-sound event above the measured noise floor; check gain, noise take and source level")
    if event_window_s is not None:
        window = np.asarray(event_window_s, dtype=float)
        if window.shape != (2,) or not np.all(np.isfinite(window)) or not 0 <= window[0] < window[1]:
            raise DetectionError("event_window_s needs finite 0 <= start < end in recording seconds")
        selected = [i for i, r in enumerate(regions)
                    if r.onset_sample < window[1]*fs and r.offset_sample > window[0]*fs]
        if len(selected) != 1:
            raise DetectionError(f"event_window_s overlaps {len(selected)} detected events; narrow the interval around the intended event or check its level")
        event_index = selected[0]
    if event_index is None and len(regions) != 1:
        raise DetectionError(f"{len(regions)} real-sound events are plausible; add event_window_s or event_index to the take manifest, or record matching labelled performances at other positions")
    index = 0 if event_index is None else event_index
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(regions):
        raise DetectionError("event_index is outside the detected event list; inspect detect_events output")
    return regions[index]


def isolate_event(samples, fs, noise_samples, *, event_index: int | None = None,
                  event_window_s=None, **kwargs):
    """Documented dry-job handoff: (sample view, EventRegion), no clock transfer.

    Each real take is an independent performance with an unmeasured clock. A
    sweep clock must NEVER be copied to it without independent calibration.
    """
    regions = detect_events(samples, fs, noise_samples, **kwargs)
    region = select_event_region(regions, fs, event_index=event_index, event_window_s=event_window_s)
    return np.asarray(samples)[region.segment_start:region.segment_stop], region


def real_take_metadata(rir_metadata: dict, *, measured_clock_ppm: float | None = None) -> dict:
    """Dry-job adapter: preserve RIR provenance, explicitly prevent clock inheritance.

    An unknown sound supplies no known clock reference. Only independent,
    per-take calibration may populate measured_clock_ppm. Zero here means no
    correction, NOT a measurement that the real take has a perfect clock.
    """
    if measured_clock_ppm is not None and (not np.isfinite(measured_clock_ppm) or abs(measured_clock_ppm) > 300):
        raise ValueError("independent real-take clock must be finite within +/-300 ppm")
    return {**rir_metadata, "sweep_clock_drift_ppm":rir_metadata.get("clock_drift_ppm"),
            "clock_drift_ppm": measured_clock_ppm or 0., "clock_resampled":True,
            "real_clock_observed":measured_clock_ppm is not None,
            "real_clock_source":"independent take calibration" if measured_clock_ppm is not None else "unobservable from an unknown event"}
