"""Opt-in phone-chain QA; observational checks, no signal repair or accuracy claim.

See parent-project note ROOM_CAPTURE_QA (not distributed) for the frozen protocol and report hook.
Reuse the existing Farina inverse, detector, causal octave filters and metrics.
All acquisition decisions precede constant-clock correction and repeat averaging.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations

import numpy as np
from scipy import signal

from sensorsim.core.fingerprint import sha256_file
from sensorsim.core.messages import ModelValidity
from .ingest import (CaptureIdentity, load_capture_qa, load_qa_assets, load_sweep_kit,
                     read_recording_audio)
from .metrics import OCTAVE_BANDS, exact_center, octave_filter, room_metrics, metric_policy
from .sweep_detect import (DetectionConfig, DetectionError, compress_repeat, detect_sweeps,
                           fractional_crop, clipping_present)

VALIDITY = ModelValidity(
    name="phone capture QA", status="unvalidated", freq_range_hz=(125, 8000),
    conditions="Opt-in engineering diagnostics; synthetic controls only. Stationary mono "
    "ESS chain, declared matching geometry and gains. Repeatability is not accuracy; "
    "identically repeated clock warps and stable processing can evade consistency checks.",
    citation="Farina AES 5093 (2000); ISO 3382-1:2009; parent-project note room_gap/smartphone_measurement (not distributed)",
)
VERSION = "capture-qa/2"  # /2: two-level input_status/consistency gates, matching H2/H3 gates
PRIMARY_BANDS = (500, 1000, 2000, 4000)


def discover_session_reference(targets, *, session, evidence_label):
    """Find the drop-folder clap for an explicitly declared chamber session.

    No nearest-date or single-file fallback: another session's clap cannot
    qualify this session. The intake loader reads, hashes and qualifies the WAV;
    discovery itself makes no dryness or listening claim. Explicit reference
    declarations remain the route for other names or quiet-window bounds.
    """
    from pathlib import Path
    if (not isinstance(session, str) or not session.strip()
            or session in (".", "..") or any(c in session for c in ("/", "\\", "\x00"))):
        raise ValueError("session must be a nonempty filename component for reference discovery")
    if evidence_label not in ("proxy", "measured"):
        raise ValueError("session reference evidence_label must be proxy or measured")
    root = Path(targets).resolve()
    name = f"session_reference_{session}.wav"
    path = root/name
    if not path.is_file():
        return None
    if not path.resolve().is_relative_to(root):
        raise ValueError("session reference must stay inside the hashed targets directory")
    return dict(file=name, session=session, evidence_label=evidence_label)


def _status(states):
    states = list(states)
    if "reject" in states:
        return "reject"
    if "review" in states or ("pass" in states and "unavailable" in states):
        return "review"
    return "pass" if states and all(s == "pass" for s in states) else "unavailable"


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _check(values, planned, *, relative=False, minimum=3, limit=1.):
    valid = [float(v) for v in values if v is not None and np.isfinite(v) and (not relative or v > 0)]
    differences = [2*abs(a-b)/(a+b) if relative else abs(a-b) for a,b in combinations(valid, 2)]
    spread = max(differences) if differences else None
    state = ("review" if spread is not None and spread > limit else
             "unavailable" if len(valid) < minimum or len(valid) != planned else "pass")
    return dict(status=state, valid_count=len(valid), planned_count=planned,
                max_pairwise_difference=spread, limit=limit,
                statistic="symmetric_relative" if relative else "absolute",
                reason="insufficient valid observations" if state == "unavailable" else None)


def _mono(samples):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("capture QA needs finite mono samples")
    return x


def _edges(band):
    return exact_center(band)*10**-.15, exact_center(band)*10**.15


def _supported(band, fs, excitation=None, order=1):
    lo, hi = _edges(band)
    return hi < fs/2 and (excitation is None or
        (lo >= order*excitation.f_start_hz and hi <= excitation.f_end_hz))


def _metric_row(metric, fs):
    curve = metric.decay
    fits = {}
    for name in ("edt", "t20", "t30"):
        fit = getattr(metric, name)
        end = np.flatnonzero(curve.db <= fit.range_db[1])
        end = int(end[0]) if len(end) else None
        mask = ((curve.db >= fit.range_db[1]) & (curve.db <= fit.range_db[0]) &
                (np.arange(len(curve.db)) <= end)) if end is not None else np.zeros(len(curve.db), bool)
        indices = np.flatnonzero(mask)
        margin = (-curve.slope_db_s*(curve.truncation_sample/fs-end/fs)
                  if end is not None and curve.slope_db_s is not None else None)
        fits[name] = dict(status="pass" if fit.valid else "unavailable", value=fit.rt60_s,
            reason=fit.reason, range_db=list(fit.range_db), r_squared=fit.r_squared,
            fit_start_s=float(indices[0]/fs) if len(indices) else None,
            fit_end_s=float(indices[-1]/fs) if len(indices) else None,
            endpoint_margin_db=_finite(margin), required_endpoint_margin_db=10.)
    fits['c50'] = dict(status="pass" if curve.valid and _finite(metric.c50_db) is not None else "unavailable",
        value=_finite(metric.c50_db), reason=curve.reason or
        ("clarity boundary/finite energy ratio unavailable" if _finite(metric.c50_db) is None else None))
    crossings = {}
    for db in (5, 10, 15, 20, 25, 30, 35):
        reached = np.flatnonzero(curve.db <= -db)
        if not curve.valid or not len(reached):
            continue
        t = float(reached[0]/fs)
        if curve.slope_db_s is not None and -curve.slope_db_s*(curve.truncation_sample/fs-t) >= 10:
            crossings[str(db)] = t
    return dict(metrics=fits, noise_power=curve.noise_power,
        noise_crossing_s=curve.truncation_sample/fs if curve.valid and curve.noise_power > 0 else None,
        truncation_sample=curve.truncation_sample, decay_valid=curve.valid,
        decay_reason=curve.reason, usable_decay_db=max(map(int, crossings), default=None),
        supported_decay_crossings_s=crossings)


def assess_repeat_rirs(repeats, fs, *, onset_sample=0, excitation=None, planned_repeats=None):
    """Per-repeat metrics BEFORE averaging. Keep every octave and failed count.

    Responses must share sampling, window, onset and nominal inverse-filter gain.
    In particular, do not normalize the quiet RIR by its quieter excitation.
    SNR here is response-window mean signal/noise power, not detection peak SNR.
    """
    arrays = [_mono(x) for x in repeats]
    if not arrays or any(len(x) != len(arrays[0]) for x in arrays):
        raise ValueError("repeat RIR windows must be nonempty and have identical lengths")
    if (isinstance(fs,bool) or not isinstance(fs,(int,np.integer)) or fs <= 0 or
            isinstance(onset_sample,bool) or not isinstance(onset_sample,(int,np.integer)) or
            not 0 <= onset_sample < len(arrays[0])):
        raise ValueError("positive integer fs and an in-window integer onset are required")
    planned = len(arrays) if planned_repeats is None else planned_repeats
    if not isinstance(planned, int) or planned < len(arrays):
        raise ValueError("planned repeats must include every supplied repeat")
    bands = {}
    for band in OCTAVE_BANDS:
        if not _supported(band, fs, excitation):
            bands[str(band)] = dict(status="unavailable", reason="octave outside acquired excitation/Nyquist",
                planned_repeats=planned, repeats=[], consistency={})
            continue
        rows = []
        for x in arrays:
            filtered = octave_filter(x, fs, band)
            metric = room_metrics(filtered, fs, onset_sample=onset_sample)
            row = _metric_row(metric, fs)
            power = max(float(np.mean(filtered[onset_sample:]**2))-metric.decay.noise_power, 0.)
            row['level_db'] = float(10*np.log10(power)) if power > 0 else None
            row['snr_db'] = (float(10*np.log10(power/metric.decay.noise_power))
                             if power > 0 and metric.decay.noise_power > 0 else None)
            rows.append(row)
        checks = {name: _check([r['metrics'][name]['value'] for r in rows], planned,
                    relative=name != 'c50', limit=.05 if name != 'c50' else 1.)
                  for name in ('edt', 't20', 't30', 'c50')}
        checks['level_db'] = _check([r['level_db'] for r in rows], planned)
        bands[str(band)] = dict(status=_status(c['status'] for c in checks.values()),
            planned_repeats=planned, retained_repeats=len(rows), repeats=rows, consistency=checks)
    return dict(schema_version=VERSION, fs=fs, window_samples=len(arrays[0]), onset_sample=onset_sample,
        metric_policy=metric_policy(onset_sample=onset_sample), bands=bands,
        status=_status(b['status'] for b in bands.values()),
        primary_status=_status(bands[str(b)]['status'] for b in PRIMARY_BANDS),
        physical_accuracy="unverified")


def _median(rows, key):
    values = [r.get(key) for r in rows if _finite(r.get(key)) is not None]
    return float(np.median(values)) if values else None


def _consistency(name, *bands):
    """Each band's own within-take check of `name`, including planned-versus-retained repeats."""
    return [band.get('consistency', {}).get(name, {}).get('status', 'unavailable') for band in bands]


def compare_levels(nominal, quiet, *, after=None):
    """Compare same-geometry captures on the SAME nominal inverse/gain convention.

    Reject incompatible windows/policies. Low quiet support is unavailable;
    repeatability and declaration of raw processing cannot confer accuracy.
    Level medians are statistics only: every compared band keeps its own
    within-take consistency and planned-versus-retained repeat status.
    """
    for other in (quiet, after):
        if other is not None and any(nominal[k] != other[k] for k in ('fs', 'window_samples', 'onset_sample', 'metric_policy')):
            raise ValueError("level comparison needs matching metric/filter/onset/window definitions")
    bands = {}
    for band in OCTAVE_BANDS:
        key = str(band)
        n, q = nominal['bands'][key], quiet['bands'][key]
        nr, qr = n['repeats'], q['repeats']
        a, b = _median(nr, 'level_db'), _median(qr, 'level_db')
        snrs = [r.get('snr_db') for r in qr]
        sufficient = (len(nr) >= 3 and len(qr) >= 3 and all(s is not None and s >= 20 for s in snrs)
                      and a is not None and b is not None)
        delta = a-b if a is not None and b is not None else None
        row = dict(status="unavailable", nominal_minus_quiet_db=delta, expected_db=6., tolerance_db=1.,
            input_normalized_delta_db=delta-6 if delta is not None else None,
            quiet_min_snr_db=min(snrs) if snrs and all(s is not None for s in snrs) else None,
            nominal_retained=len(nr), quiet_retained=len(qr),
            reason="need three responses per level and quiet-band SNR >=20 dB", metrics={})
        if sufficient:
            # cr-room-capture-02: a median level must not erase a within-take level spread or a
            # missing planned repeat; each level's own consistency check gates the transfer decision.
            transfer = 'pass' if abs(delta-6) <= 1. else 'review'
            row['linearity_status'] = _status([transfer, *_consistency('level_db', n, q)])
            row['status'] = row['linearity_status']
            row['reason'] = (None if row['status'] == 'pass' else
                             'level-dependent combined speaker/recorder response' if transfer == 'review' else
                             'within-take repeat levels inconsistent or planned repeats missing')
            for name in ('edt','t20','t30','c50'):
                values = [_median([{'v': r['metrics'][name]['value']} for r in rows], 'v')
                          if all(r['metrics'][name]['status'] == 'pass' for r in rows) else None
                          for rows in (nr,qr)]
                row['metrics'][name] = _check(values, 2, minimum=2, relative=name != 'c50',
                                              limit=.05 if name != 'c50' else 1.)
            # Only compare normalized decay crossings supported at BOTH levels.
            common = set.intersection(*(set(r['supported_decay_crossings_s']) for r in nr+qr))
            row['common_decay_range_db'] = [0, -max(map(int,common))] if common else None
            row['common_decay_crossings_s'] = {
                db: dict(nominal=float(np.median([r['supported_decay_crossings_s'][db] for r in nr])),
                         quiet=float(np.median([r['supported_decay_crossings_s'][db] for r in qr])))
                for db in sorted(common, key=int)}
            if any(v['status'] == 'review' for v in row['metrics'].values()):
                row['status'], row['reason'] = 'review', 'level-dependent decay/clarity on common metric support'
            elif any(v['status'] == 'unavailable' for v in row['metrics'].values()):
                row['status'], row['reason'] = 'review', 'level transfer observed but some decay/clarity metrics are unavailable'
            inputs = dict(nominal=n.get('status', 'unavailable'), quiet=q.get('status', 'unavailable'))
            if after is not None:
                returned = after['bands'][key]
                inputs['after'] = returned.get('status', 'unavailable')
                c = _median(returned['repeats'], 'level_db')
                row['nominal_return'] = _check([a,c], 2, minimum=2)
                row['nominal_return']['status'] = _status([row['nominal_return']['status'],
                                                          *_consistency('level_db', n, returned)])
                row['status'] = _status([row['status'], row['nominal_return']['status']])
            # Every compared band must itself be consistent and fully supported at each level.
            row['input_status'] = inputs
            folded = _status([row['status'], *inputs.values()])
            if folded != row['status']:
                row['status'] = folded
                row['reason'] = row['reason'] or 'within-take repeat inconsistency or missing planned repeats at a compared level'
        bands[key] = row
    return dict(schema_version=VERSION, bands=bands, status=_status(b['status'] for b in bands.values()),
                primary_status=_status(bands[str(b)]['status'] for b in PRIMARY_BANDS),
                warning="Consistency cannot prove processing is disabled or attribute it to speaker versus phone.")


def _chunk_lag(a, b, maximum):
    """Bounded raw cross-correlation; no resampling, correction or phase model."""
    a, b = a-np.mean(a), b-np.mean(b)
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    if norm <= 1e-25:
        return None, None
    corr = signal.correlate(a, b, mode='full', method='fft')/norm
    center = len(b)-1
    lo, hi = max(0,center-maximum), min(len(corr),center+maximum+1)
    peak = lo+int(np.argmax(np.abs(corr[lo:hi])))
    delta = 0.
    if 0 < peak < len(corr)-1:
        left, top, right = abs(corr[peak-1:peak+2])
        curvature = left-2*top+right
        if curvature < 0:
            delta = float(np.clip(.5*(left-right)/curvature, -.5, .5))
    return peak-center+delta, float(abs(corr[peak]))


def assess_timing(samples, kit, detection):
    """Audit detector outputs + local raw-repeat lags before any QA correction.

    Absolute offset is harmless/unobservable. Adjacent slot gaps use their actual
    ordinals (a dropped repeat isn't a doubled clock). Every raw pair is examined
    so one corrupted reference cannot hide a step. Identical intra-sweep warps
    across all repeats remain unobservable and are explicitly not certified.
    """
    x = _mono(samples)
    fs = detection.fs
    reps = sorted(detection.repeats, key=lambda r: r.slot)
    reasons = []
    if len(reps) < 3:
        reasons.append('fewer than three retained repeats')
    if detection.clipped:
        reasons.append('digital clipping; re-record')
    coarse_fs = kit.fs_hz/max(1, kit.fs_hz//6000)
    tolerance = max(60., 4e6/(kit.sweep_s*coarse_fs))
    if abs(detection.clock_drift_ppm) > 300:
        reasons.append('clock outside +/-300 ppm')
    if abs(detection.chirp_drift_ppm-detection.clock_drift_ppm) > tolerance:
        reasons.append('chirp and spacing clocks disagree')
    if detection.spacing_residual_samples/fs > .003:
        reasons.append('repeat-spacing residual exceeds 3 ms')
    rates = []
    for a,b in zip(reps, reps[1:]):
        if b.slot <= a.slot:
            raise ValueError('detector repeat slots must be distinct')
        ppm = ((b.start_sample-a.start_sample)/(fs*kit.period_s*(b.slot-a.slot))-1)*1e6
        rates.append(dict(slots=[a.slot,b.slot], ppm=float(ppm)))
        if abs(ppm-detection.clock_drift_ppm) > tolerance:
            reasons.append('adjacent-repeat rate/latency disagrees with the constant clock')
    pairs = []
    ratio = 1+detection.clock_drift_ppm*1e-6
    # Avoid faded endpoints, compare corresponding raw phase chunks.
    edges = np.linspace(max(kit.fade_in_s,.03*kit.sweep_s),
                        min(kit.sweep_s-kit.fade_out_s,.97*kit.sweep_s), 9)
    for a,b in combinations(reps, 2):
        lags, correlations, centers = [], [], []
        for left,right in zip(edges,edges[1:]):
            size = round((right-left)*ratio*fs)
            ia, ib = (round(r.start_sample+left*ratio*fs) for r in (a,b))
            if size < 8 or min(ia,ib) < 0 or max(ia,ib)+size > len(x):
                lag, corr = None, None
            else:
                lag, corr = _chunk_lag(x[ia:ia+size],x[ib:ib+size], max(2,round(.003*fs)))
                if lag is not None:
                    lag += (ia-a.start_sample)-(ib-b.start_sample)
            lags.append(lag/fs if lag is not None else None)
            correlations.append(corr)
            centers.append((left+right)/2)
        good = [i for i,c in enumerate(correlations) if c is not None and c >= .8]
        variable = any(abs(lags[j]-lags[i]) > max(2/fs,60e-6*(centers[j]-centers[i]))
                       for i,j in zip(good,good[1:]))
        state = 'review' if variable else 'pass' if len(good) == 8 else 'unavailable'
        pairs.append(dict(slots=[a.slot,b.slot], status=state, lag_s=lags, correlations=correlations,
                          chunk_centers_s=centers))
    if any(p['status'] == 'review' for p in pairs):
        reasons.append('within-sweep relative lag changes; changing clock, dropout or path suspected')
    state = 'reject' if detection.clipped or len(reps) < 3 else 'review' if reasons else _status(p['status'] for p in pairs)
    return dict(status=state, correction_applied=False, reasons=list(dict.fromkeys(reasons)),
        clock_drift_ppm=detection.clock_drift_ppm, chirp_drift_ppm=detection.chirp_drift_ppm,
        agreement_tolerance_ppm=tolerance, spacing_residual_s=detection.spacing_residual_samples/fs,
        adjacent_rates=rates, within_sweep_pairs=pairs, dropped=deepcopy(detection.dropped),
        limitation="Relative lags cannot rule out the same warp in every repeat; transducer/path change can mimic clock change.")


def assess_harmonics(compressed, fs, excitation, *, linear_start_sample, decay_support_s):
    """Bounded H2/H3 packets from the existing inverse, with conservative overlap.

    Per-band linear noise crossings are a proxy for harmonic-tail length, not a
    measurement of it. Unknown/long tails preclude a ratio; frequency holes and
    finite windows preclude whole-band claims. No calibrated THD is returned.
    """
    full = _mono(compressed)
    length = round(excitation.duration_s*fs)/fs/np.log(excitation.f_end_hz/excitation.f_start_hz)
    pre = .0025
    separation = length*np.log(4/3)  # includes unrequested H4, as deconvolve_ess does
    duration = max(0., separation-pre-2/fs)
    count = int(duration*fs)
    size = count+round(pre*fs)
    # The fundamental gate starts `pre` before its peak, exactly as each harmonic gate does
    # (cr-room-capture-03): starting it at the peak dropped pre-peak energy from the denominator.
    fundamental_begin = linear_start_sample-pre*fs
    bands = {}
    for band in OCTAVE_BANDS:
        orders = {}
        support = decay_support_s.get(band)
        for order in (2,3):
            advance = length*np.log(order)
            begin = linear_start_sample-(advance+pre)*fs
            stop = begin+(duration+pre)*fs
            reason = None
            state = 'pass'
            if not _supported(band,fs,excitation,order) or duration*exact_center(band) < 6:
                state, reason = 'unavailable', 'insufficient harmonic octave bandwidth/window'
            elif (begin < 0 or stop > len(full) or fundamental_begin < 0 or
                  int(np.floor(fundamental_begin))+size > len(full)):
                state, reason = 'unavailable', 'compressed packet outside recorded support'
            elif support is None:
                state, reason = 'unavailable', 'harmonic-tail overlap unresolved: no supported band decay'
            elif support+pre >= separation:
                state, reason = 'review', 'potential harmonic-tail overlap; linear decay exceeds guarded separation'
            ratio = peak = None
            if begin >= 0 and stop <= len(full) and count > 0:
                packet = fractional_crop(full,begin,size)
                peak = float(begin+np.argmax(abs(packet)))
                if state == 'pass':
                    h = octave_filter(packet,fs,band)
                    fundamental = octave_filter(fractional_crop(full,fundamental_begin,size),fs,band)
                    a,b = float(np.sum(h*h)),float(np.sum(fundamental*fundamental))
                    ratio = float(10*np.log10(a/b)) if min(a,b) > 0 else None
            orders[str(order)] = dict(status=state, reason=reason, advance_s=float(advance),
                window_samples=[float(begin),float(stop)],
                fundamental_window_samples=[float(fundamental_begin),float(fundamental_begin+(duration+pre)*fs)],
                peak_sample=peak, packet_ratio_db=ratio,
                supported_frequency_hz=[order*excitation.f_start_hz,min(excitation.f_end_hz,fs/2)])
        bands[str(band)] = dict(status=_status(o['status'] for o in orders.values()), orders=orders,
            linear_decay_support_s=support, conservative_separation_s=float(separation))
    return dict(status=_status(b['status'] for b in bands.values()), bands=bands,
        primary_status=_status(bands[str(b)]['status'] for b in PRIMARY_BANDS),
        warning="Packet ratios are not THD or independently separated nonlinear kernels; linear decay only bounds a proxy for harmonic support.")


def _capture_snr(x, noise, kit, detection):
    """Band energy on raw active ESS windows, measured against same-gain quiet.

    Uses the interval exciting each octave, including causal filter settling from
    the start of that repeat. Unknown/zero floors stay unavailable, not infinite.
    """
    fs = kit.fs_hz
    length = kit.sweep_s/np.log(kit.f2_hz/kit.f1_hz)
    result = {}
    ratio = 1+detection.clock_drift_ppm*1e-6
    for band in OCTAVE_BANDS:
        snrs = []
        if noise is not None and len(noise) >= round(.05*fs) and _supported(band,fs,kit.excitation):
            floor = octave_filter(noise,fs,band)
            # Discard six cycles of initial filter settling for the floor only.
            trim = round(6*fs/exact_center(band))
            power = float(np.mean(floor[trim:]**2)) if len(floor) > trim else 0.
            for rep in detection.repeats:
                start = max(0,int(rep.start_sample))
                stop = min(len(x),int(np.ceil(rep.end_sample)))
                active = octave_filter(x[start:stop],fs,band)
                low,high = _edges(band)
                a,b = (round((rep.start_sample-start)+length*np.log(f/kit.f1_hz)*fs*ratio)
                       for f in (low,high))
                total = float(np.mean(active[max(0,a):min(len(active),b)]**2)) if b > a else 0.
                s = max(total-power,0.)
                snrs.append(float(10*np.log10(s/power)) if min(s,power) > 0 else None)
        if len(snrs) != len(detection.repeats):
            snrs = [None]*len(detection.repeats)
        result[str(band)] = dict(status='pass' if all(s is not None and s >= 20 for s in snrs) else 'unavailable',
            per_repeat_db=snrs, minimum_db=20.,
            reason='raw active octave versus same-gain quiet; missing/zero/low floor support is unavailable')
    return result


def assess_capture(samples, kit, *, rir_duration_s=1.5, noise_samples=None):
    """One mono take at kit.fs_hz. No repair after timing review/unavailability."""
    x = _mono(samples)
    if not 0 < rir_duration_s <= kit.tail_silence_s:
        raise ValueError('QA RIR window must fit the acquired quiet tail')
    result = dict(schema_version=VERSION, physical_accuracy='unverified', planned_repeats=kit.repeats)
    try:
        detection = detect_sweeps(x,kit.fs_hz,kit,config=DetectionConfig())
    except DetectionError as exc:
        return {**result, 'status':'reject', 'reason':str(exc), 'retained_repeats':0,
                'timing':dict(status='reject', correction_applied=False, reason=str(exc))}
    timing = assess_timing(x,kit,detection)
    result.update(timing=timing, retained_repeats=len(detection.repeats), detection=detection.metadata())
    if timing['status'] != 'pass':
        return {**result, 'status':timing['status'], 'reason':'timing unresolved; no QA correction or averaging performed'}
    fs = kit.fs_hz
    # Quiet lead is a fallback only; a supplied physical quiet recording is preferred.
    noise = (_mono(noise_samples) if noise_samples is not None else
             x[:max(0,int(detection.repeats[0].start_sample-.05*fs))])
    snrs = _capture_snr(x,noise,kit,detection)
    count = round(rir_duration_s*fs)
    windows, compressed, kept_indices = [], [], []
    for index,rep in enumerate(detection.repeats):
        full,origin,step = compress_repeat(x,kit,start_sample=rep.segment_start,
            stop_sample=rep.segment_stop,drift_ppm=detection.clock_drift_ppm)
        peak = (rep.start_sample-origin)/step+round(kit.sweep_s*fs)-1
        # Convolution padding isn't measured decay. Apply the same support boundary as rir.py.
        if peak+count > (rep.segment_stop-origin)/step:
            continue
        windows.append(fractional_crop(full,peak,count))
        compressed.append((full,peak))
        kept_indices.append(index)
    if len(windows) < 3:
        return {**result, 'status':'reject', 'reason':'fewer than three complete decay windows',
                'retained_repeats':len(windows)}
    metrics = assess_repeat_rirs(windows,fs,excitation=kit.excitation,planned_repeats=kit.repeats)
    for band in OCTAVE_BANDS:
        for row,index in zip(metrics['bands'][str(band)]['repeats'],kept_indices):
            row['response_window_snr_db'] = row['snr_db']
            row['snr_db'] = snrs[str(band)]['per_repeat_db'][index]
    harmonics = []
    for i,(full,peak) in enumerate(compressed):
        support = {b:metrics['bands'][str(b)]['repeats'][i]['noise_crossing_s']
                   for b in OCTAVE_BANDS if metrics['bands'][str(b)]['repeats']}
        harmonics.append(assess_harmonics(full,fs,kit.excitation,linear_start_sample=peak,decay_support_s=support))
    result.update(status=_status([metrics['status'],*[h['status'] for h in harmonics],
                                  *[s['status'] for s in snrs.values()]]),
                  repeat_metrics=metrics, harmonics=harmonics, retained_repeats=len(windows),
                  retained_slots=[detection.repeats[i].slot for i in kept_indices],
                  incomplete_decay_slots=[r.slot for i,r in enumerate(detection.repeats) if i not in kept_indices],
                  band_capture_snr=snrs, noise_source='supplied quiet' if noise_samples is not None else 'quiet lead',
                  constant_clock_correction_ppm=detection.clock_drift_ppm)
    return result


def _read_mono(path, fs):
    samples, actual_fs = read_recording_audio(path)
    if samples.shape[1] != 1:
        raise ValueError('QA needs mono recordings')
    x = samples[:,0]
    if clipping_present(x):
        raise ValueError('digital clipping; re-record at lower fixed gain')
    if actual_fs != fs:
        if {actual_fs,fs} != {44100,48000}:
            raise ValueError('QA recording rate must match kit or supported native 44.1/48 kHz conversion')
        divisor = np.gcd(actual_fs,fs)
        x = signal.resample_poly(x,fs//divisor,actual_fs//divisor)
    return x, actual_fs


def summarize_pilot(rows):
    """A/B/C fixed-position repeatability and D versus their median, separately.

    P2/P3 remain coverage takes, never extra fixed-position observations. Within-
    take failures are retained even when a take median happens to match its peers.
    """
    by_id = {r['id']:r for r in rows}
    nominal, returned = {}, {}
    group = [by_id.get(k) for k in ('A','B','C','D')]
    fixed_position = all(group[:3]) and len({r['position'] for r in group[:3]}) == 1
    return_position = fixed_position and group[3] is not None and group[3]['position'] == group[0]['position']
    for band in OCTAVE_BANDS:
        key = str(band)
        base, back = {}, {}
        for name in ('edt','t20','t30','c50','level_db'):
            values, states = [], []
            for row in group:
                data = row.get('qa',{}).get('repeat_metrics',{}).get('bands',{}).get(key,{}) if row else {}
                repeat_rows = data.get('repeats',[])
                check = data.get('consistency',{}).get(name,{})
                states.append(check.get('status','unavailable'))
                values.append(_median([{'v':r['level_db'] if name == 'level_db' else r['metrics'][name]['value']}
                                       for r in repeat_rows], 'v') if repeat_rows and
                              all(_finite(r['level_db'] if name == 'level_db' else r['metrics'][name]['value'])
                                  is not None for r in repeat_rows) else None)
            relative = name in ('edt','t20','t30')
            base[name] = _check(values[:3],3,relative=relative,limit=.05 if relative else 1.)
            base[name]['status'] = _status([base[name]['status'], *states[:3]])
            median = float(np.median(values[:3])) if all(v is not None for v in values[:3]) else None
            back[name] = _check([median,values[3]],2,minimum=2,relative=relative,limit=.05 if relative else 1.)
            back[name]['status'] = _status([back[name]['status'],*states])
            if not fixed_position or any(r and r['level'] != 'nominal' for r in group[:3]):
                base[name].update(status='unavailable',reason='need matching-position nominal A/B/C')
            if not return_position or any(r and r['level'] != 'nominal' for r in group):
                back[name].update(status='unavailable',reason='need matching-position nominal A/B/C/D')
            # Missing planned values must not be promoted by other passing states.
            if base[name]['valid_count'] < 3 and 'review' not in states[:3]:
                base[name]['status'] = 'unavailable'
            if back[name]['valid_count'] < 2 and 'review' not in states:
                back[name]['status'] = 'unavailable'
        nominal[key], returned[key] = base, back
    def summary(bands):
        return dict(bands=bands,status=_status(c['status'] for band in bands.values() for c in band.values()),
                    primary_status=_status(c['status'] for b in PRIMARY_BANDS for c in bands[str(b)].values()))
    return dict(nominal_repeatability=summary(nominal),return_to_P1=summary(returned))


def build_capture_qa_report(manifest, *, rir_duration_s=1.5):
    """Documented integration seam: load opt-in manifest.capture_qa and assess it.

    Returns strict-JSON-compatible metadata, never modifies pipeline/manifest or
    production takes. Bad sidecar/hash declarations raise; bad captures become
    explicit reject rows. Keeps all seven pilot slots even when a file is absent
    from the sidecar. The pipeline persists this under report['capture_qa'].
    """
    identity = manifest.capture_identity or CaptureIdentity()
    base = dict(schema_version=VERSION, capture_identity=identity.model_dump(),
                processing_verified=False, physical_accuracy='unverified',
                validity=asdict(VALIDITY), planned_pilot_takes=7)
    if manifest.capture_qa is None:
        return {**base, 'status':'unavailable', 'reason':'no capture_qa sidecar supplied', 'takes':[]}
    session = load_capture_qa(manifest.capture_qa)
    assets = load_qa_assets(session.assets)
    kit = load_sweep_kit(assets.kit.file)
    if manifest.sweep_kit is not None and kit != manifest.sweep_kit:
        raise ValueError('QA and production manifest must use the same sweep kit')
    paths = {manifest.capture_qa, session.assets, assets.kit.file, assets.nominal.file, assets.quiet.file}
    paths.update(t.file for t in session.takes)
    paths.update(t.noise_file for t in session.takes if t.noise_file is not None)
    inputs = {str(p):sha256_file(p) for p in sorted(paths)}
    rows = []
    by_id = {}
    for take in session.takes:
        row = dict(id=take.id, level=take.level, position=take.position, file=str(take.file),
                   sha256=sha256_file(take.file), source_xyz_m=list(take.source_xyz_m),
                   receiver_xyz_m=list(take.receiver_xyz_m))
        try:
            x, native_fs = _read_mono(take.file,kit.fs_hz)
            noise = None
            if take.noise_file is not None:
                noise, _ = _read_mono(take.noise_file,kit.fs_hz)
                row.update(noise_file=str(take.noise_file),noise_sha256=sha256_file(take.noise_file))
            row.update(native_fs=native_fs, qa=assess_capture(x,kit,rir_duration_s=rir_duration_s,noise_samples=noise))
        except ValueError as exc:
            row['qa'] = dict(status='reject', reason=str(exc))
        rows.append(row)
        by_id[take.id] = row
    comparisons = summarize_pilot(rows)
    # A/L/C brackets the lower level; B is retained as an independent nominal take.
    group = [by_id.get(k) for k in ('A','L','C')]
    if all(group) and len({r['position'] for r in group}) == 1:
        if [r['level'] for r in group] != ['nominal','quiet','nominal']:
            raise ValueError('pilot A/L/C must be nominal/quiet/nominal')
        if all('repeat_metrics' in r['qa'] for r in group):
            comparisons['two_level'] = compare_levels(group[0]['qa']['repeat_metrics'],
                group[1]['qa']['repeat_metrics'],after=group[2]['qa']['repeat_metrics'])
    if 'two_level' not in comparisons:
        comparisons['two_level'] = dict(status='unavailable', reason='need usable matching-geometry A/L/C')
    missing = [key for key in ('A','B','L','C','P2','P3','D') if key not in by_id]
    states = [r['qa']['status'] for r in rows]+[v['status'] for v in comparisons.values()]
    if missing:
        states.append('unavailable')
    return {**base, 'status':_status(states), 'takes':rows, 'comparisons':comparisons,
        'input_sha256':inputs,
        'missing_pilot_takes':missing, 'take_status_counts':dict(Counter(r['qa']['status'] for r in rows)),
        'assets':session.assets.name, 'assets_sha256':sha256_file(session.assets),
        'sidecar_sha256':sha256_file(manifest.capture_qa)}


def attach_capture_qa(pipeline_report, qa_report):
    """Return a copy with report['capture_qa']; preserve the production state.

    Call build_capture_qa_report before extraction if timing is an admission gate.
    This hook deliberately grants no new acceptance to fitting/dry stages. A
    capture-qa/1 report still attaches unchanged and keeps its own label: its
    two-level statuses and packet ratios predate cr-room-capture-02/03.
    """
    if qa_report.get('schema_version') not in ('capture-qa/1', VERSION):
        raise ValueError('unsupported capture QA report version')
    return {**pipeline_report, 'capture_qa':deepcopy(qa_report)}


def assess_source_crop(samples, fs, region, regions, noise_samples, *,
                       contaminated=False, sir_db=None, dryness_window_s=None, label=None,
                       session_reference=None, kind=None, heterogeneous_hits=False,
                       incompatible_rises_samples=()):
    """WP6 crop gate, separate from sweep QA; all acoustic ratios are proxies.

    A single unknown excitation does not identify physical DRR or interference.
    Explicit direct/post-excitation windows estimate an energy ratio only. A
    listener must resolve semantic isolation and unavailable dryness/SIR. The
    intake layer binds that approval to the recording, crop and this report.
    """
    from sensorsim.sources.catalog import mono
    from sensorsim.sources.kinds import source_kind
    kind = source_kind(label or '', kind)
    if type(heterogeneous_hits) is not bool:
        raise ValueError('heterogeneous_hits must be a boolean observation')
    if heterogeneous_hits and kind != 'knock':
        raise ValueError('heterogeneous_hits is restricted to knock sequences')
    if type(contaminated) is not bool:
        raise ValueError('contaminated must be a boolean observation')
    x, noise = mono(samples, fs), mono(noise_samples, fs)
    a, b = region.segment_start, region.segment_stop
    crop = x[a:b]
    if not 0 <= a < b <= len(x):
        raise ValueError('crop bounds outside recording')
    rejected, review = [], ['semantic isolation requires listening approval']
    floor = float(np.mean(noise**2))
    active = x[region.onset_sample:region.offset_sample]
    power = float(np.mean(active**2)) if len(active) else 0.
    snr = float(10*np.log10(max(power-floor, 1e-30)/floor)) if floor > 0 else None
    peak = float(np.max(abs(crop)))
    clipped = bool(np.any(crop >= 32767/32768) or np.any(crop <= -1))
    overlaps = [i for i, other in enumerate(regions) if other != region
                and other.onset_sample < b and other.offset_sample > a]
    if contaminated or overlaps or (incompatible_rises_samples and not heterogeneous_hits):
        rejected.append('contaminated')
    if heterogeneous_hits:
        review.append('heterogeneous_knock_hits_require_listening')
    if clipped:
        rejected.append('clipped')
    if peak < 1/32768:
        rejected.append('silent_or_sub_lsb')
    if region.tail_truncated or region.onset_sample < round(.1*fs):
        rejected.append('truncated')
    if len(noise) < round(.1*fs) or floor == 0:
        review.append('noise_floor_unavailable')
    if snr is None:
        review.append('snr_proxy_unavailable')
    elif snr < 10:
        rejected.append('low_snr_proxy')
    elif snr < 20:
        review.append('low_snr_proxy')
    if sir_db is None:
        review.append('sir_unavailable')
    elif not np.isfinite(sir_db):
        raise ValueError('SIR must be finite when supplied')
    elif sir_db < 10:
        rejected.append('low_sir')
    elif sir_db < 20:
        review.append('low_sir')
    drr, tail_fraction = None, None
    if dryness_window_s is None:
        review.append('dryness_unavailable')
    else:
        w = np.asarray(dryness_window_s, dtype=float)
        if w.shape != (3,) or not np.isfinite(w).all() or not 0 <= w[0] < w[1] < w[2] <= len(x)/fs:
            raise ValueError('dryness windows require direct start < source stop < tail stop inside recording')
        start, stop, end = np.round(w*fs).astype(int)
        if not start < stop < end or not start <= region.onset_sample < stop or end < region.offset_sample:
            raise ValueError('dryness windows must cover the selected onset and complete event tail')
        direct = max(0., float(np.sum(np.square(x[start:stop]))) - (stop-start)*floor)
        tail = max(0., float(np.sum(np.square(x[stop:end]))) - (end-stop)*floor)
        if direct <= 0 or floor == 0:
            review.append('dryness_unavailable')
        else:
            # Bound the denominator by the observed quiet energy; never claim infinite DRR.
            drr = float(10*np.log10(direct/max(tail, (end-stop)*floor)))
            tail_fraction = float(tail/(direct+tail))
            if drr < 10 or tail_fraction > .1:
                review.append('reverberant_or_source_decay')
    from sensorsim.sources.decay import late_decay, LATE_LIMIT_S, KNOCK_DECAY_REVIEW
    decay = late_decay(x, fs, noise, stop=b)
    comparisons, review_messages = {}, {}
    mid = decay['bands']['500']['t20_s']
    if mid is None:
        review.append('late_decay_unavailable')
    elif mid > LATE_LIMIT_S:
        review.append('late_decay_exceeds_limit')
        if kind == 'knock':
            review_messages['late_decay_exceeds_limit'] = (
                'late decay above the dry-knock heuristic with a qualified dry session; '
                'review intrinsic door ringing and semantic isolation'
                if session_reference is not None and session_reference['status'] == 'qualified'
                else KNOCK_DECAY_REVIEW)
    if session_reference is not None:
        if session_reference['status'] == 'not_dry':
            rejected.append('session_not_dry')
        elif session_reference['status'] != 'qualified':
            review.append('session_decay_unavailable')
        elif kind == 'knock':
            for band, take_band in decay['bands'].items():
                ref_band = session_reference['decay']['bands'][band]
                ref = ref_band.get('comparison_decay_s', ref_band['t20_s'])
                value = take_band['t20_s']
                slower = ref is not None and value is not None and value > 1.5*ref
                comparisons[band] = dict(take_t20_s=value, reference_decay_s=ref,
                    reference_method=ref_band.get('comparison_method', 't20'),
                    limit_s=1.5*ref if ref is not None else None, slower=bool(slower))
            if any(c['slower'] for c in comparisons.values()):
                review.append('take_slower_than_session')
    return dict(schema_version='source-crop-qa/1',
        status='reject' if rejected else 'review', rejected=rejected, review=review,
        crop_samples=[a, b], event_samples=[region.onset_sample, region.offset_sample],
        noise_rms=float(np.sqrt(floor)), noise_floor_dbfs=float(10*np.log10(floor)) if floor > 0 else None,
        peak=peak, clipped=clipped, snr_proxy_db=snr, sir_db=sir_db,
        estimated_drr_db=None, early_tail_ratio_db=drr, tail_energy_fraction=tail_fraction,
        late_decay=decay, session_reference=session_reference,
        session_decay_comparison=comparisons, review_messages=review_messages, kind=kind,
        dryness_window_s=list(dryness_window_s) if dryness_window_s is not None else None,
        overlapping_event_indices=overlaps, heterogeneous_hits=heterogeneous_hits,
        incompatible_rises_samples=list(incompatible_rises_samples),
        limitation='Early/tail energy and noise ratios are proxies, not physical DRR or separated interference. '
                   'Intrinsic source decay can resemble reverberation; unresolved crops require listening.')
