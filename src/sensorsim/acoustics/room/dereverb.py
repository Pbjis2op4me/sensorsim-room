"""Measured residual-tail processing and dry diagnostics (CPU reference).

Kirkeby/Nelson 1999: bounded offline inverse; Lebart/Boucher/Denbigh 2001
and Habets/Gannot/Cohen 2009: statistical late power. This is a measured-RIR
adaptation, not an implementation or validation of their speech experiments.
Metric helpers here are owned by dry-dereverb pending metrics consolidation.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter

from sensorsim.core.messages import ModelValidity


VALIDITY = ModelValidity(
    name="measured_residual_dereverberation", status="partially validated", freq_range_hz=(125, 8000),
    conditions="Offline mono, fixed LTI matching sweep, stationary noise; diffuse late tail. "
    "Analytic controls only; physical-source directionality and phone processing unqualified.",
    citation="Habets et al. 2009 doi:10.1109/LSP.2009.2024791; parent-project note ROOM_DEREVERB (not distributed)",
)


def _db_ratio(a, b):
    return float(10*np.log10(max(float(a), 1e-30)/max(float(b), 1e-30)))


def tail_metrics(samples, fs, onset, *, early_ms=50., direct_end=None):
    """Energy/EDC diagnostics with an explicit time origin, never peak aligned.

    The slope is a descriptive -5..-25 dB residual EDC regression, not ISO T20.
    Direct-to-residual uses the specified direct interval; source-signal callers
    instead get early-to-late. Empty/zero tails have no estimable decay slope.
    """
    x = np.asarray(samples, float)
    onset = int(np.clip(onset, 0, len(x)))
    end = min(len(x), onset+max(1, round(early_ms*.001*fs)))
    early = float(x[onset:end]@x[onset:end])
    late_x = x[end:]
    late = float(late_x@late_x)
    pre = float(x[:onset]@x[:onset])
    slope, r2 = None, None
    if late > max(early, 1e-30)*1e-12 and len(late_x) >= round(.02*fs):
        edc = np.cumsum(late_x[::-1]**2)[::-1]
        db = 10*np.log10(np.maximum(edc/late, 1e-30))
        idx = np.flatnonzero((db < -5) & (db > -25))
        if len(idx) >= round(.02*fs):
            fit = np.polyfit(idx/fs, db[idx], 1)
            total = float(np.sum((db[idx]-np.mean(db[idx]))**2))
            r2 = 1-float(np.sum((db[idx]-np.polyval(fit, idx/fs))**2))/max(total, 1e-30)
            if fit[0] < 0 and r2 >= .9:
                slope = float(fit[0])
    report = dict(early_energy=early, residual_tail_energy=late,
                  late_to_early_db=_db_ratio(late, early),
                  pre_origin_energy=pre, edc_slope_db_s=slope, edc_fit_r2=r2,
                  edc_t60_s=None if slope is None else -60/slope,
                  early_end_sample=end, onset_sample=onset)
    if direct_end is not None:
        direct_end = int(np.clip(direct_end, onset, len(x)))
        direct = float(x[onset:direct_end]@x[onset:direct_end])
        rest = pre+float(x[direct_end:]@x[direct_end:])
        report['direct_to_residual_db'] = _db_ratio(direct, rest)
    return report


def residual_model(firs, responses, target, fs, band_rows, mixing_ms):
    """Calibrate only on measured sweeps and the actual, noise-masked inverse.

    FIRs have a modelling delay len(fir)//2. Keep that delay here so negative
    time energy is measured rather than discarded. Linear convolution avoids
    folded tails. Ratios use the effective response, not the original wet DRR.
    """
    delay = len(firs[0])//2
    size = max(len(g)+len(h)-1 for g, h in zip(firs, responses))
    effective = np.zeros(size)
    for g, h in zip(firs, responses):
        v = signal.oaconvolve(g, h)
        effective[:len(v)] += v
    desired = np.zeros(size)
    desired[delay:delay+len(target)] = target
    scale = float(effective@desired/max(desired@desired, 1e-30))
    residual = effective-scale*desired
    summary = tail_metrics(effective, fs, delay, early_ms=mixing_ms,
                           direct_end=delay+len(target))
    summary.update(target_to_residual_db=_db_ratio(np.sum((scale*desired)**2), residual@residual),
                   modelling_delay_samples=delay, target_scale=scale,
                   postfilter_parameters=dict(mixing_ms=mixing_ms, stft_ms=32., hop_fraction=.5,
                                              smoothing_bins_frames=[3, 3], amplitude_floor=.1),
                   basis='measured effective RIR before nonlinear late mask; not independent truth')
    bands = []
    for row in band_rows:
        center = row['center_hz']
        sos = signal.butter(3, [center/np.sqrt(2), center*np.sqrt(2)],
                            fs=fs, btype='bandpass', output='sos')
        band = signal.sosfilt(sos, effective)
        stats = tail_metrics(band, fs, delay, early_ms=mixing_ms)
        # The known sweep supplies decay. The effective RIR supplies amplitude.
        ts = [v['t60_s'] for v in row['channels'].values() if v['t60_s'] is not None]
        t60 = float(np.median(ts)) if ts else None
        ratio = stats['residual_tail_energy']/max(stats['early_energy'], 1e-30)
        bands.append(dict(center_hz=center, t60_s=t60,
                          late_to_early_ratio=float(ratio),
                          late_fraction=float(ratio/(1+ratio)), **stats,
                          status='measured_decay' if t60 is not None else 'missing_decay_no_subtraction'))
    return dict(**summary, bands=bands)


def suppress_late(samples, noise_psd, frequencies, fs, model, *, mixing_ms=50., floor=.1):
    """Delayed exponential-power predictor, with bounded smooth power subtraction.

    Estimate early power recursively after subtracting predicted late power.
    The kernel's integrated gain is the measured late/early energy ratio.
    The measured residual ratio tends to zero after exact inversion.
    Model validity requires a diffuse stationary tail; tones are not predicted.
    """
    if all(b['t60_s'] is None or b['late_fraction'] < 1e-12 for b in model['bands']):
        return samples
    nwin = max(32, round(.032*fs))
    nwin += nwin % 2
    nwin = min(nwin, len(samples)//2*2)
    if nwin < 4:
        return samples
    hop = nwin//2
    f, _, z = signal.stft(samples, fs=fs, nperseg=nwin, noverlap=hop,
                          boundary='zeros', padded=True)
    power = abs(z)**2
    # One-sided Welch density to two-sided STFT power; DC/Nyquist exceptions.
    noise = np.interp(f, frequencies, noise_psd)[:, None]*(fs*1.5/nwin)/2
    noise[[0, -1]] *= 2
    unwanted = np.zeros_like(power)
    supported = np.zeros(len(f), dtype=bool)
    delay = max(1, round(mixing_ms*.001*fs/hop))
    for band in model['bands']:
        if band['t60_s'] is None:
            continue
        c = band['center_hz']
        mask = (f >= c/np.sqrt(2)) & (f < c*np.sqrt(2))
        supported |= mask
        alpha = np.exp(-6*np.log(10)*hop/fs/band['t60_s'])
        history = np.maximum(uniform_filter(power[mask], size=(3, 3), mode='nearest')-noise[mask], 0)
        decay = np.zeros_like(history)
        for frame in range(delay, power.shape[1]):
            decay[:, frame] = (alpha*decay[:, frame-1]
                               + band['late_to_early_ratio']*(1-alpha)*history[:, frame-delay])
            history[:, frame] = np.maximum(history[:, frame]-decay[:, frame], 0)
        unwanted[mask] = decay
    smooth = uniform_filter(power, size=(3, 3), mode='nearest')
    unwanted = uniform_filter(unwanted, size=(3, 3), mode='nearest')
    # Smoothing may not invent a decay estimate in an unsupported neighbour.
    unwanted[~supported] = 0
    gain = np.sqrt(np.clip(1-unwanted/np.maximum(smooth, 1e-30), floor**2, 1))
    _, out = signal.istft(z*gain, fs=fs, nperseg=nwin, noverlap=hop, boundary=True)
    return out[:len(samples)]


def activity_interval(samples, noise_power, fs):
    """10 ms frame detector; offsets preserved, no emission-time assertion.

    Six dB above stationary noise and -40 dB below peak frame energy. This
    isolates recorded activity, which can include noise/extraneous sounds and
    residual reverb. No blind single-channel method can identify an arbitrary
    desired event among unrelated sounds without further labels.
    """
    hop = max(1, round(.01*fs))
    peaks = []
    for start in range(0, len(samples), hop*1024):
        x = np.asarray(samples[start:start+hop*1024], float)
        n = len(x)//hop
        if n:
            peaks.extend(np.mean(x[:n*hop].reshape(n, hop)**2, axis=1).tolist())
        if len(x) % hop:
            peaks.append(float(np.mean(x[n*hop:]**2)))
    energy = np.asarray(peaks)
    threshold = max(4*float(noise_power), float(energy.max(initial=0))*1e-4, 1e-30)
    active = np.flatnonzero(energy > threshold)
    return dict(start_sample=int(active[0]*hop) if len(active) else None,
                end_sample=min(len(samples), int((active[-1]+1)*hop)) if len(active) else None,
                threshold_power=threshold, status='detected' if len(active) else 'unresolved',
                basis='recorded activity; may include residual tail or extraneous events; not emission time')


def combine_take_reports(reports, *, comparable_levels=False):
    """Combine independent performances' envelopes only; select one waveform.

    Inputs are reports from a common case/reference/source and unchanged gain.
    Per-band weights are diagnostics, not posterior probabilities. Dispersion
    includes performance variation and measurement/model error.
    """
    if not reports:
        raise ValueError('no take reports')
    for field in ('case', 'fs', 'reference', 'target', 'source_id', 'target_response_sha256'):
        if len({r.get(field) for r in reports.values()}) > 1:
            raise ValueError(f'independent takes require a common {field}')
    if any(r.get('synchronized') or len(r['channels']) != 1 for r in reports.values()):
        raise ValueError('expected independent single-channel take reports')
    usable = {k: r for k, r in reports.items()
              if any(b.get('confidence', 0) > 0 for b in r['bands'])}
    def rank(k):
        return (float(np.median([b['drr_db'] for b in usable[k]['bands']
                                 if b.get('confidence', 0) > 0])), k)
    selected = max(usable, key=rank) if usable else None
    offsets = {}
    for k, r in reports.items():
        levels = [b['output_level_db'] for b in r['bands']
                  if b.get('confidence', 0) > 0 and b.get('output_level_db') is not None]
        offsets[k] = float(np.mean(levels)) if levels else 0.
    rows = []
    for c in sorted({b['center_hz'] for r in reports.values() for b in r['bands']}):
        entries = [(k, b) for k, r in reports.items() for b in r['bands']
                   if b['center_hz'] == c and b.get('confidence', 0) > 0
                   and b.get('output_level_db') is not None]
        weights = np.array([b['confidence'] for _, b in entries])
        values = np.array([b['output_level_db']-(0 if comparable_levels else offsets[k])
                           for k, b in entries])
        mean = float(np.average(values, weights=weights)) if len(values) else None
        spread = float(np.sqrt(np.average((values-mean)**2, weights=weights))) if len(values) else None
        rows.append(dict(center_hz=c, level_db=mean, dispersion_db=spread,
                         confidence=float(np.mean(weights)) if len(weights) else 0.,
                         contributors=[k for k, _ in entries],
                         residual_reverb={k: b.get('residual_reverb') for k, b in entries},
                         status='combined' if len(values) > 1 else 'single_take' if len(values) else 'unresolved'))
    return dict(schema_version='room-dry-independent/1', recommended=selected, bands=rows,
                combination='absolute level' if comparable_levels else 'mean-centred octave envelope',
                waveform_policy='select highest measured DRR among takes with resolved bands; never sum samples',
                assumptions='same source class, device gain/orientation and target; confidence is not calibration')


def artifact_metrics(actual, expected, fs, *, lag_samples=0):
    """Evaluation only: gain-aligned TF LSD, pre-echo, late energy, isolation.

    Caller supplies the frozen evaluation lag. Source-silent tails include all
    output noise, and are never cropped away to improve SI-SDR. The isolation
    fraction is an artifact proxy, not a musical-noise perceptual score.
    """
    lag = int(lag_samples)
    origin = max(0, -lag)
    n = max(origin+len(actual), origin+lag+len(expected))
    y, t = np.zeros(n), np.zeros(n)
    y[origin:origin+len(actual)] = actual
    t[origin+lag:origin+lag+len(expected)] = expected
    scale = float(y@t/max(t@t, 1e-30))
    target = scale*t
    indices = np.flatnonzero(abs(t) > max(float(np.max(abs(t))), 1e-30)*1e-3)
    if not len(indices) or abs(scale) < 1e-15:
        return dict(status='unresolved_zero_target')
    on, off = int(indices[0]), int(indices[-1]+1)
    half = max(1, round(.016*fs))
    pre = y[:max(0, on-half)]
    late = y[min(n, off+half):]
    early = y[on:min(n, off+half)]
    nwin = min(max(32, round(.032*fs)), n)
    nwin -= nwin % 2
    f, times, z = signal.stft(y, fs, nperseg=nwin, noverlap=nwin//2)
    _, _, tz = signal.stft(target, fs, nperseg=nwin, noverlap=nwin//2)
    p, q = abs(z)**2, abs(tz)**2
    floor = max(float(q.max())*1e-6, 1e-30)
    active = (q > floor) & (f[:, None] >= 125) & (f[:, None] <= min(8000, fs*.45))
    delta = 10*np.log10(np.maximum(p, floor)/np.maximum(q, floor))
    lsd = float(np.sqrt(np.mean(delta[active]**2))) if active.any() else None
    silent = (times*fs < on-half) | (times*fs > off+half)
    median = median_filter(p, size=(3, 3), mode='nearest')
    isolated = (p > 10*np.maximum(median, floor)) & silent[None, :]
    isolation = float(p[isolated].sum()/max(p[:, silent].sum(), 1e-30))
    return dict(status='evaluated', log_spectral_distance_db=lsd,
                pre_echo_energy_ratio_db=_db_ratio(pre@pre, target@target) if on-half > origin else None,
                pre_echo_status='evaluated' if on-half > origin else 'insufficient_recorded_lead',
                late_to_early_db=_db_ratio(late@late, early@early),
                residual_tail_energy=float(late@late), spectral_isolation_fraction=isolation,
                tail=tail_metrics(y, fs, off, early_ms=16.),
                target_to_residual_db=_db_ratio(target@target, np.sum((y-target)**2)))
