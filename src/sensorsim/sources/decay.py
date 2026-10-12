"""Noise-limited late-decay diagnostics, not DRR or source/room separation.

The thresholds and framing are project QA heuristics. A broadband session
impulse can qualify a capture setup; an unknown ringing source cannot do so.
"""
import numpy as np
from scipy import signal

from .catalog import mono, content_hash

BANDS = (125, 500, 1000, 2000)
# Tuned on the development generator, not a published door-mode range.
LATE_LIMIT_S = .20
KNOCK_DECAY_REVIEW = ('late decay above the dry-knock heuristic, which may be a resonant door or a wet '
                      'room; record a session clap to decide automatically')


def impact_onsets(samples, fs):
    """Detect fresh energy rises, not local maxima in a ringing envelope.

    Five-ms frames and a threefold power rise are detector choices, not a
    population prior on hit spacing. Return only observed rises; never fill gaps.
    """
    x = np.asarray(samples)
    size = max(1, round(.005*fs))
    p = np.add.reduceat(x*x, np.arange(0, len(x), size))/size
    baseline = np.maximum.reduce([np.r_[np.zeros(k), p[:-k]] if len(p) > k else np.zeros_like(p)
                                  for k in (1, 2, 3)])
    rises = (p > 3*baseline) & (p > max(float(p.max())*.002, 1e-20))
    starts = np.flatnonzero(rises & ~np.r_[False, rises[:-1]])
    # A strike spanning adjacent frames is a single rise. This 20-ms resolution
    # limit is a signal-processing choice; faster patterns need annotations.
    selected = []
    for k in starts:
        if not selected or k-selected[-1] > 4:
            selected.append(int(k))
    return [max(0, k*size) for k in selected]


def late_decay(samples, fs, noise_samples, *, start=None, stop=None):
    """T20-extrapolated T60 after the final detected excitation (seconds).

    Causal octave filters, 10-ms mean powers, subtract the quiet band floor,
    and stop at the first sustained floor crossing. Noise is subtracted BEFORE
    reverse integration; no positive rectification of individual noise samples.
    Lack of 20 dB usable range is unavailable, never a many-second noise fit.
    """
    x, noise = mono(samples, fs), mono(noise_samples, fs)
    stop = len(x) if stop is None else int(stop)
    onsets = impact_onsets(x[:stop], fs)
    if start is None:
        start = onsets[-1] if onsets else int(np.argmax(abs(x[:stop])))
    start = int(start)
    result = dict(method='octave-late-t20/1', start_sample=start, bands={},
                  evidence_label='proxy', limitation='Source plus room decay; not DRR. Unavailable fits need review.')
    frame = max(1, round(.01*fs))
    for center in BANDS:
        row = dict(t20_s=None, status='unavailable')
        result['bands'][str(center)] = row
        if center*2**.5 >= fs/2 or stop-start < 6*frame:
            continue
        sos = signal.butter(4, [center/2**.5, center*2**.5], btype='bandpass', fs=fs, output='sos')
        y = signal.sosfilt(sos, x)
        q = signal.sosfilt(sos, noise)
        floor = float(np.mean(q[min(frame, len(q)//4):]**2))
        segment = y[start:stop]
        n = len(segment)//frame
        p = np.mean(segment[:n*frame].reshape(n, frame)**2, axis=1)
        peak = int(np.argmax(p[:min(n, 5)]))
        # Three quiet frames end the decay; later isolated noise never extends it.
        low = p <= max(4*floor, float(p[peak])*1e-8, 1e-30)
        end = n
        for i in range(peak+1, n-2):
            if low[i:i+3].all():
                end = i
                break
        energy = np.maximum(p[peak:end]-floor, 0.)
        row.update(noise_power=floor, stop_sample=start+end*frame)
        if len(energy) < 4 or not np.any(energy):
            continue
        edc = np.cumsum(energy[::-1])[::-1]
        db = 10*np.log10(np.maximum(edc/edc[0], 1e-30))
        # Account for the fitted tail beyond truncation by requiring floor margin.
        mask = (db <= -5) & (db >= -25)
        if db[-1] > -25 or np.count_nonzero(mask) < 3:
            continue
        t = np.arange(len(db))*frame/fs
        slope, intercept = np.polyfit(t[mask], db[mask], 1)
        predicted = slope*t[mask]+intercept
        residual = float(np.sum((db[mask]-predicted)**2))
        total = float(np.sum((db[mask]-np.mean(db[mask]))**2))
        r2 = 1-residual/max(total, 1e-30)
        row['r_squared'] = r2
        if slope < 0 and r2 >= .9:
            row.update(t20_s=float(-60/slope), status='available')
    return result


def qualify_session(reference, *, session):
    """Qualify an optional single clap/pop from the SAME chamber session.

    Retain its input hash and evidence label. No windowed DRR is calculated.
    Failure or missing dynamic range never certifies a session as dry.
    """
    if reference['session'] != session or reference['evidence_label'] not in ('proxy', 'measured'):
        raise ValueError('session_reference must identify the same session and its evidence_label')
    fs = reference['fs']
    x = mono(reference['samples'], fs)
    q = reference.get('quiet_window_s', (0., .2))
    if len(q) != 2 or not 0 <= q[0] < q[1] <= len(x)/fs or q[1]-q[0] < .1:
        raise ValueError('session_reference needs at least 0.1 s of quiet inside the recording')
    noise = x[round(q[0]*fs):round(q[1]*fs)]
    onsets = impact_onsets(x, fs)
    diagnostic = late_decay(x, fs, noise)
    diagnostic['evidence_label'] = reference['evidence_label']
    # Preserve unavailable T20 as None. A short impulse gets an explicitly
    # labelled comparison bound instead of an invented fitted decay rate.
    for band in diagnostic['bands'].values():
        fitted = band['t20_s']
        short = (len(onsets) == 1 and
                 0 < (band.get('stop_sample', len(x))-diagnostic['start_sample'])/fs <= .1)
        band['comparison_decay_s'] = fitted if fitted is not None else .1 if short else None
        band['comparison_method'] = 't20' if fitted is not None else 'short_decay_upper_bound' if short else 'unavailable'
    mid = [diagnostic['bands'][str(b)] for b in (500, 1000, 2000)]
    valid = [b['comparison_decay_s'] for b in mid if b['comparison_decay_s'] is not None]
    status = ('not_dry' if any(v > LATE_LIMIT_S for v in valid) else
              'qualified' if len(valid) >= 2 else 'unavailable')
    if any(b['comparison_method'] == 'short_decay_upper_bound' for b in mid):
        diagnostic['short_decay_upper_bound_s'] = .1
    peak = int(np.argmax(abs(x)))
    if (len(onsets) != 1 or peak < round(q[1]*fs) or len(x)-peak < round(.5*fs)
            or np.max(abs(x)) >= 32767/32768 or np.mean(x*x) <= 10*np.mean(noise*noise)):
        status = 'unavailable'
    return dict(status=status, content_sha256=content_hash(x, fs), fs=int(fs), session=session,
                evidence_label=reference['evidence_label'], quiet_window_s=list(q), decay=diagnostic)
