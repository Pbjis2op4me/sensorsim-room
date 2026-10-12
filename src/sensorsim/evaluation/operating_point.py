# SPDX-License-Identifier: Apache-2.0
"""Finite operating-point search and time-aware replay of the deployed tracker.

Score controls establish software behavior only. A selected development operating
point is not a deployment false-alarm or natural-event recall certification.
"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np

from sensorsim import edge


def average_precision(scores, labels):
    """Non-interpolated AP, grouped by distinct scores; -1 labels are unknown.

    No known positives (including an empty/fully filtered panel) returns NaN.
    Invalid labels or nonfinite scores are errors, never silently filtered.
    """
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    if s.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError('AP requires finite equal-length one-dimensional arrays')
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('AP labels must be -1, 0 or 1')
    s, y = s[y >= 0], y[y >= 0]
    if not y.sum():
        return float('nan')
    order = np.argsort(-s, kind='stable')
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    hits = np.cumsum(y)[last]
    return float(np.sum(np.diff(np.r_[0., hits]) * hits/(last+1))/y.sum())


def validate_policy(classes, policy, *, window_s, sample_rate_hz):
    """Canonicalize per-class settings and the device's integer-sample hop."""
    if not classes or len(set(classes)) != len(classes) or any(not isinstance(c,str) or not c for c in classes):
        raise ValueError('unique nonempty class names required')
    fs = sample_rate_hz
    if isinstance(fs, bool) or not np.isfinite(fs) or int(fs) != fs or fs < 1:
        raise ValueError('sample_rate_hz must be a positive integer')
    if not np.isfinite(window_s) or window_s <= 0 or not np.isclose(window_s*fs, round(window_s*fs), rtol=0, atol=1e-8):
        raise ValueError('window_s must represent whole samples')
    if round(window_s*fs) < 1:
        raise ValueError('window must contain samples')
    result = {}
    for key in ('on', 'off_ratio', 'smooth_hops', 'min_hops'):
        if key not in policy:
            raise ValueError(f'missing policy {key}')
        value = policy[key]
        if isinstance(value, dict) and set(value) != set(classes):
            raise ValueError('policy class mapping must be complete')
        values = {c: value[c] if isinstance(value, dict) else value for c in classes}
        for c, v in values.items():
            if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v):
                raise ValueError('finite numeric policy required')
            if key in ('on', 'off_ratio'):
                if not 0 < v <= 1:
                    raise ValueError('thresholds and ratios must be in (0,1]')
                values[c] = float(v)
            else:
                if v < 1 or int(v) != v:
                    raise ValueError('persistence must be positive integer hops')
                values[c] = int(v)
        result[key] = values
    hop = policy.get('hop_s', float('nan'))
    if not np.isfinite(hop) or hop <= 0 or not 1 <= round(hop*fs) <= round(window_s*fs):
        raise ValueError('hop_s must represent 1..window_samples samples')
    result['hop_s'] = round(hop*fs)/fs
    return result


def latency_allowance(classes, policy, window_s):
    """Window plus smoothing history plus on-confirmation duration, in seconds.

    Tracker has no separate off-confirmation duration. Qualification is the
    decision at which min_hops is attained, not the later end notification.
    """
    return {c: window_s + (policy['smooth_hops'][c]-1 + policy['min_hops'][c]-1)*policy['hop_s']
            for c in classes}


def replay_policy(scores, classes, policy, *, window_s, sample_rate_hz, sample_count):
    """Replay complete probability sequences (or chunks) with one final flush.

    All endpoints come from edge.Tracker. The adapter observes its active state
    to timestamp activation and qualification, without changing runtime logic.
    Missing observation hooks fail closed. Constant probabilities are legitimate
    controls; constant *audio* must not certify exposure (handled by the bank).
    """
    cfg = validate_policy(classes, policy, window_s=window_s, sample_rate_hz=sample_rate_hz)
    fs, n = int(sample_rate_hz), round(window_s*sample_rate_hz)
    hop = round(cfg['hop_s']*fs)
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < n:
        raise ValueError('a complete recording in sample units is required')
    chunks = [scores] if isinstance(scores, np.ndarray) else scores
    tracker = edge.Tracker(list(classes), cfg, n/fs)
    if not hasattr(tracker, 'active') or not hasattr(tracker, 'min_hops'):
        raise RuntimeError('missing tracker qualification observation hooks')
    pending, events, index, serial = {}, [], 0, 0

    def annotate(outputs, time_s, flushed=False):
        nonlocal serial
        for raw in outputs:
            e = dict(raw, decision_s=time_s, flushed=flushed)
            c = e['label']
            if e['event'] == 'start':
                if c in pending:
                    raise RuntimeError('tracker started an already active class')
                serial += 1
                pending[c] = dict(event_id=serial, activation_s=time_s, qualification_s=None)
            elif e['event'] != 'end' or c not in pending:
                raise RuntimeError('unobserved tracker branch')
            e.update(pending[c])
            if e['event'] == 'end':
                if bool(e['confirmed']) != (e['qualification_s'] is not None):
                    raise RuntimeError('tracker qualification observer disagrees with output')
                del pending[c]
            events.append(e)

    for chunk in chunks:
        x = np.asarray(chunk, float)
        if x.ndim != 2 or x.shape[1] != len(classes) or not len(x) or not np.isfinite(x).all() or np.any((x < 0)|(x > 1)):
            raise ValueError('nonempty finite probability chunks with one column per class required')
        for score in x:
            time_s = (n+index*hop)/fs
            if time_s > sample_count/fs:
                raise ValueError('score sequence exceeds recording extent')
            annotate(tracker.step(score, time_s), time_s)
            for i, active in tracker.active.items():
                c = classes[i]
                if c not in pending or 'hops' not in active:
                    raise RuntimeError('missing tracker activation/qualification observation')
                if active['hops'] >= tracker.min_hops[i] and pending[c]['qualification_s'] is None:
                    pending[c]['qualification_s'] = time_s
            index += 1
    if index != 1+(sample_count-n)//hop:
        raise ValueError('incomplete score sequence for recording extent')
    annotate(tracker.flush(), sample_count/fs, True)
    if pending:
        raise RuntimeError('tracker failed to flush active events')
    starts = {c: sum(e['event']=='start' and e['label']==c for e in events) for c in classes}
    confirmed = {c: sum(e['event']=='end' and e['confirmed'] and e['label']==c for e in events) for c in classes}
    return dict(events=events, starts=sum(starts.values()), confirmed=sum(confirmed.values()),
                starts_by_class=starts, confirmed_by_class=confirmed, score_rows=index,
                sample_count=int(sample_count), exposure_s=sample_count/fs, policy=cfg,
                latency_s=latency_allowance(classes, cfg, window_s), tracker='sensorsim.edge.Tracker')


def match_events(events, annotations, *, latency_s):
    """Maximum one-to-one correct-class matching using actual decision times.

    Both activation and qualification must fall in [onset, offset + allowance].
    In particular a pre-existing alert cannot earn a hit by spanning the target.
    A deterministic augmenting-path matcher handles overlapping annotations.
    """
    targets = list(annotations)
    for a in targets:
        if (a.get('label') not in latency_s or not np.isfinite([a['start_s'],a['end_s']]).all()
                or not 0 <= a['start_s'] < a['end_s']):
            raise ValueError('invalid annotated target interval/class')
    if any(not np.isfinite(v) or v < 0 for v in latency_s.values()):
        raise ValueError('latency allowances must be finite nonnegative seconds')
    outputs = [e for e in events if e['event']=='end' and e.get('confirmed')]
    ids = [e['event_id'] for e in outputs]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate output event identity')
    for e in outputs:
        times = [e.get('activation_s'),e.get('qualification_s')]
        if any(v is None for v in times) or not np.isfinite(times).all() or times[1] < times[0]:
            raise ValueError('actual activation and qualification decision times required')
    edges = [[j for j,e in enumerate(outputs) if e['label']==a['label'] and
              a['start_s'] <= e['activation_s'] <= e['qualification_s'] <= a['end_s']+latency_s[a['label']]]
             for a in targets]
    assigned = {}
    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    for i in range(len(targets)):
        visit(i, set())
    pairs = sorted((i, ids[j]) for j,i in assigned.items())
    return dict(matched=len(pairs), total=len(targets), all_matched=bool(targets) and len(pairs)==len(targets),
                pairs=[dict(annotation_index=i,event_id=j) for i,j in pairs],
                unmatched_annotation_indices=sorted(set(range(len(targets)))-{i for i,_ in pairs}))


def threshold_policies(classes, base, grid, *, max_candidates=4096):
    """Enumerate the finite Cartesian threshold grid in frozen class order."""
    values = tuple(float(v) for v in grid)
    if not values or len(set(values)) != len(values) or any(not 0 < v <= 1 for v in values):
        raise ValueError('unique finite thresholds in (0,1] required')
    if len(values)**len(classes) > max_candidates:
        raise ValueError('threshold grid exceeds registered resource budget')
    return [dict(copy.deepcopy(base), on=dict(zip(classes,v))) for v in product(values, repeat=len(classes))]


def search_operating_point(candidate, bank, policies, constraints, *, score_cache=None,
                           max_candidates=4096, max_replay_rows=50_000_000):
    """Replay every declared policy; search exhaustion has no fallback candidate."""
    from .selection import candidate_identity, score_candidate, select_candidate, infer_scores
    if bank.role != 'development':
        raise ValueError('operating-point search may only consume development banks')
    policies = list(policies)
    if not policies or len(policies) > max_candidates:
        raise ValueError('empty or over-budget policy search')
    cache = score_cache if score_cache is not None else infer_scores(candidate, bank)
    if sum(len(x) for x in cache['sequences'].values())*len(policies) > max_replay_rows:
        raise ValueError('finite search exceeds registered replay-row budget; do not shorten the bank')
    results = []
    for policy in policies:
        changed = candidate_identity(model_sha256=candidate['model_sha256'], features=candidate['features'],
            classes=candidate['classes'], policy=policy, step=candidate['step'], seed=candidate['seed'],
            code_revision=candidate['code_revision'], model_path=candidate.get('model_path'))
        results.append(score_candidate(changed, bank, constraints, score_cache=cache))
    return dict(select_candidate(results), attempted=results,
                search_rule='all declared policies replayed; no monotonicity assumption')
