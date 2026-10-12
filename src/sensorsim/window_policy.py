# SPDX-License-Identifier: Apache-2.0
"""Experimental deployment-window supervision, version deployment/1.

States: 1 = cue-supported positive, 0 = verified absent, -1 = unknown (no class
loss). Single-target inputs and target-free beds/confusers retain the corpus's
absence contract for other classes. This is not an automatic annotation oracle.

Knock/chime defaults assume the identifying attack occupies the first 30/120 ms
of the supplied audible span. A row may instead declare ``window_cues`` with
``units: samples`` and ``identifying: [[start, end], ...]`` in rendered-file
coordinates. Kettle requires ``boil`` intervals; missing annotation is unknown,
never an inferred boil stage. Intervals must be reviewed for real sources.

The policy is opt-in. Time/frequency masking and MixStyle are refused until a
cue-preservation oracle qualifies them. Energy is a necessary support check,
never sufficient class evidence. Thresholds below are modelling assumptions.
"""
from __future__ import annotations

import numpy as np

POLICY = 'deployment/1'
RULES = {
    'knock': {'attack_s': .030, 'min_energy_fraction': .20},
    'doorbell': {'attack_s': .120, 'min_energy_fraction': .20},
    'kettle': {'min_boil_s': .500, 'min_energy_fraction': .25},
}


def validate_config(cfg):
    policy = cfg.get('window_policy', 'legacy')
    if policy not in ('legacy', POLICY):
        raise ValueError('window_policy must be legacy or deployment/1')
    if policy == 'legacy':
        return False
    if cfg.get('partial_windows') or cfg.get('low_event_spl_tail'):
        raise ValueError('deployment/1 cannot be combined with legacy partial/low-level policies')
    if cfg.get('p_specaugment', 0.) or cfg.get('p_freq_mixstyle', 0.) or cfg.get('event_aware_masks'):
        raise ValueError('deployment/1 requires feature augmentation off until cue preservation is qualified')
    if (cfg.get('device_split', 'in_distribution') != 'in_distribution' or
            cfg.get('background_domain', 'device_recorded') != 'device_recorded'):
        raise ValueError('deployment/1 requires in_distribution device_recorded audio for audibility')
    fs = cfg['fs']
    n = round(cfg['window_s']*fs)
    hop = round(cfg.get('detector', {}).get('hop_s', .25)*fs)
    if not 1 <= hop <= n:
        raise ValueError('deployment/1 requires 1 <= detector hop samples <= window samples')
    return True


def _intervals(cues, key, size):
    if cues is None:
        return None
    if not isinstance(cues, dict) or cues.get('units') != 'samples':
        raise ValueError('window_cues units must be samples in rendered-file coordinates')
    intervals = cues.get(key)
    if intervals is None:
        return None
    if not isinstance(intervals, list) or not intervals:
        raise ValueError('cue intervals must be a nonempty list')
    previous = -1
    for pair in intervals:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in pair) or
                not 0 <= pair[0] < pair[1] <= size or pair[0] < previous):
            raise ValueError('cue intervals must be ordered disjoint sample ranges inside the file')
        previous = pair[1]
    return intervals


def supervision(class_name, source, span, start, n, fs, cues=None):
    """Declare cue support before capture; caller additionally checks audibility.

    A fragment never becomes negative just because its cue is absent. Absence
    comes from the separately admitted target-free corpus/pool, not this rule.
    """
    from sensorsim.training_dynamics import best_window_energy
    x = np.asarray(source, float)
    a, b = span
    if (x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.ptp(x) == 0 or
            not 0 <= a < b <= len(x) or n < 1 or fs < 1):
        raise ValueError('window source QA: require finite nonconstant mono audio and a valid span')
    if class_name not in RULES:
        raise ValueError(f'no identifying-cue rule for class {class_name!r}')
    rule = RULES[class_name]
    lo, hi = max(a, start), min(b, start+n)
    total = best_window_energy(x, span, n)
    if total <= 0:
        raise ValueError('window source QA: empty event energy')
    fraction = float(np.sum(x[lo:hi]**2)/total) if hi > lo else 0.
    key = 'boil' if class_name == 'kettle' else 'identifying'
    intervals = _intervals(cues, key, len(x))
    assumption = intervals is None
    if class_name == 'kettle':
        support = max((max(0, min(v, start+n)-max(u, start)) for u, v in intervals or []), default=0)
        cue_ok = support >= round(rule['min_boil_s']*fs)
    else:
        intervals = intervals if intervals is not None else [[a, a+round(rule['attack_s']*fs)]]
        cue_ok = any(start <= u and v <= start+n and v <= b for u, v in intervals)
        support = max((max(0, min(v, start+n)-max(u, start)) for u, v in intervals), default=0)
    positive = cue_ok and fraction >= rule['min_energy_fraction']
    return dict(state='positive' if positive else 'unknown', retained_fraction=fraction,
                cue_samples=int(support), cue_rule=key, cue_assumption=assumption,
                cue_intervals=[] if intervals is None else intervals,
                rule_version=POLICY)


def masked_bce(logits, labels):
    """Mean BCE over known class cells; an all-unknown batch has zero gradient."""
    import torch
    if logits.shape != labels.shape or not torch.all((labels == -1) | (labels == 0) | (labels == 1)):
        raise ValueError('labels must match logits and contain only -1 (unknown), 0 or 1')
    known = labels >= 0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.clamp_min(0), reduction='none')
    return (loss*known).sum()/known.sum().clamp_min(1)


def validate_corpus(corpus, cfg):
    """Refuse a training class with no annotated positive support, before SGD."""
    n = round(cfg['window_s']*cfg['fs'])
    for name in corpus.classes:
        if name not in RULES:
            raise ValueError(f'no identifying-cue rule for class {name!r}')
        for k, x in enumerate(corpus.targets[name]):
            cues = corpus.window_cues[name][k]
            span = corpus.spans[name][k]
            supervision(name, x, span, span[0], n, cfg['fs'], cues)
            if name == 'kettle':
                intervals = _intervals(cues, 'boil', len(x))
                if intervals is None or not any(b-a >= round(.5*cfg['fs']) for a, b in intervals):
                    raise ValueError('deployment/1 kettle training requires window_cues.boil '
                                     'with at least 0.5 s of annotated support for each source')


def draw_example(corpus, rng, cfg, mixer):
    """No rejection sampling; preserve history and full-take gain before capture."""
    from sensorsim import event_level, iot_mic, training_dynamics as td, training_negatives as tn
    from sensorsim.training_device import ReceiverEvent
    matched = tn.validate_config(cfg)
    from sensorsim.recognizer import _crop, _padded_window, _sliding_event_window
    validate_config(cfg)
    fs, n = cfg['fs'], round(cfg['fs']*cfg['window_s'])
    hop = round(cfg.get('detector', {}).get('hop_s', .25)*fs)
    y = np.zeros(len(corpus.classes), np.float32)
    bed = _crop(corpus.background[int(rng.integers(len(corpus.background)))], 2*n, rng).astype(float)
    selected = rng.random() >= cfg['p_background_only']
    event, active = np.zeros(2*n), (n, 2*n)
    name, index, start, info = None, -1, None, dict(state='negative', retained_fraction=0.)
    local = dict(cfg)
    if selected:
        index = int(rng.integers(len(corpus.classes)))
        name = corpus.classes[index]
        k = int(rng.integers(len(corpus.targets[name])))
        source, span = corpus.targets[name][k], corpus.spans[name][k]
        _, _, start = _sliding_event_window(source, span, n, hop, rng)
        cues = getattr(corpus, 'window_cues', {}).get(name, [None]*len(corpus.targets[name]))[k]
        info = supervision(name, source, span, start, n, fs, cues)
        # Preceding source samples warm the device filter; mid-event windows
        # must not manufacture an attack at their left edge.
        event = _padded_window(source, start-n, 2*n)
        active = (max(0, n+span[0]-start), min(2*n, n+span[1]-start))
        whole = event_level.level_power(source, fs, span)
        crop = event_level.level_power(event, fs, active)
        shift = 10*np.log10(max(crop, 1e-30)/max(whole, 1e-30))
        local['event_spl_db'] = [float(v+shift) for v in cfg['event_spl_db']]
        y[index] = 1. if info['state'] == 'positive' else -1.
    elif not matched:
        from sensorsim.negatives.integration import draw_confuser
        confuser = draw_confuser(corpus, rng, cfg, n)
        if confuser is not None:
            window, (a, b) = confuser
            event[n:], active = window, (n+a, n+b)
    probe = td.clone_rng(rng)
    iot_mic.PROFILES[cfg.get('device_profile', 'pi5_inmp441/1')].with_variation(probe)
    requested = float(probe.uniform(*cfg['event_spl_db'])) if selected else None
    negative_draw = (tn.prepare_context(corpus, rng, cfg, n, class_name=name,
                     target_index=k if selected else None) if matched else None)
    if matched:
        mixed, counterfactual = tn.draw_training_context(bed, ReceiverEvent(event, active), rng, local,
            draw=negative_draw, counterfactual=selected)
        heard = mixed[n:].astype(np.float32)
        floor = counterfactual[n:] if selected else mixed[n:]
        if negative_draw.provenance['injected']:
            corpus.confuser_draws += 1
    else:
        heard = mixer(bed, event, active, rng, local)[n:].astype(np.float32)
        floor = bed[n:]
    event_power = td.power(heard.astype(float)-floor, cfg) if selected else td.power(heard, cfg)
    floor_power = td.power(floor, cfg)
    audible_mask = td.audible_frames(event_power, floor_power)
    if selected:
        # At least 10 ms of cue in each counted STFT frame. Whole-window energy
        # elsewhere cannot stand in for an inaudible attack or boil fragment.
        frame_starts = np.arange(len(audible_mask))*cfg['hop']
        cue_frames = np.zeros(len(audible_mask), bool)
        for a, b in info['cue_intervals']:
            overlap = np.minimum(frame_starts+cfg['n_fft'], b-start)-np.maximum(frame_starts, a-start)
            cue_frames |= overlap >= max(1, round(.010*fs))
        audible_mask &= cue_frames
    audible = int(audible_mask.sum())
    if selected and audible < 3:
        y[index] = -1.
        info['state'] = 'unknown'
    ctx = dict(info, window_policy=POLICY, class_name=name, class_index=index,
        positive=bool(selected and y[index] == 1), target_selected=bool(selected),
        source_start=start, grid_phase=None if start is None else start % hop, hop_samples=hop,
        attempts=1, rejections=0, requested_level_db=requested, level_kind='event_spl_db',
        loss_mask=(y >= 0).astype(float).tolist(), audible_frames=audible,
        event=event_power, floor=floor_power)
    if matched:
        ctx['negative_context'] = negative_draw.provenance
    return heard, y, ctx
