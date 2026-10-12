# SPDX-License-Identifier: Apache-2.0
"""Opt-in training policies. Defaults bypass this module's per-example work.

All audio goes through the caller's production mixer. Retry randomness is separate
from the production stream: selection and level draws never change on rejection.
These policies are experimental and require held-out deployment evaluation.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json

import numpy as np
from sensorsim import edge, iot_mic, event_level, training_take_cache
from sensorsim.recognizer import _crop, _event_window

DEFAULTS = dict(cue_validity=False, cue_masks=False, cue_pairing='same_draw_no_partial_no_masks/1', audibility_gate=False, partial_windows=False, event_aware_masks=False,
                window_policy='legacy',
                low_event_spl_tail=False, mixstyle_same_label=False,
                positive_fraction=None, validation_mode='synthetic', validation_background_gain_db=15.,
                dynamics_max_attempts=32)


def enabled(cfg):
    return cfg.get('cue_validity', False) or cfg.get('window_policy', 'legacy') != 'legacy' or any(
        cfg.get(k, False) for k in ('audibility_gate', 'partial_windows', 'low_event_spl_tail'))


def clone_rng(rng):
    result = np.random.Generator(type(rng.bit_generator)())
    result.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return result


def auxiliary_rng(rng, purpose):
    state = json.dumps(rng.bit_generator.state, sort_keys=True, default=lambda x: np.asarray(x).tolist())
    seed = int.from_bytes(hashlib.sha256((state + purpose).encode()).digest()[:8], 'little')
    return np.random.default_rng(seed)


@lru_cache(maxsize=16)
def _features(items):
    return edge.Features(dict(items))


def power(x, cfg):
    keys = ('fs', 'window_s', 'n_fft', 'hop', 'n_mels', 'fmin_hz', 'fmax_hz')
    f = _features(tuple((k, cfg[k]) for k in keys))
    x = np.asarray(x, float)
    x = x - x.mean()
    frames = np.lib.stride_tricks.sliding_window_view(x, cfg['n_fft'])[::cfg['hop']]
    return (np.abs(np.fft.rfft(frames*f.window, axis=-1))**2 @ f.bank.T).T


def audible_frames(e, f):
    energy = e.sum(0)
    active = energy > max(1e-30, energy.max()*10**(-2.5))
    return active & np.any(e >= 10**.3*np.maximum(f, 1e-30), axis=0)


def best_window_energy(source, span, n):
    a, b = span
    energy = np.asarray(source[a:b], float)**2
    if len(energy) <= n:
        return float(energy.sum())
    cumulative = np.r_[0., np.cumsum(energy)]
    return float(np.max(cumulative[n:] - cumulative[:-n]))


def partial_window(source, span, n, rng):
    """Uniform sample phase, including clipped beginnings/ends; no fixed hop grid."""
    a, b = span
    start = int(rng.integers(a - n + 1, b))
    lo, hi = max(0, start), min(len(source), start + n)
    window = np.zeros(n)
    window[lo-start:hi-start] = source[lo:hi]
    return window, (max(0, a-start), min(n, b-start)), start


def eligible(context, cfg):
    if cfg.get('cue_validity'):
        if 'cue_states' not in context:
            raise ValueError('missing cue validity hook')
        return True  # unknown class losses are masked; never forced into background
    return (not cfg.get('partial_windows') or context['retained_fraction'] >= .10) and (
        not cfg.get('audibility_gate') or audible_frames(context['event'], context['floor']).sum() >= 3)


def draw_example(corpus, rng, cfg, mixer, *, context=False):
    """Lock source/label/level before retrying placement; exhaustion is a QA error.

    Partial crops inherit the full take's active-sample gain (span SNR in stress mode), so a quiet tail stays
    quiet. The fraction is relative to the best achievable window, not take length.
    Audibility/context decomposition currently requires device-recorded Pi audio;
    unsupported nonlinear chains fail explicitly rather than invent a separation.
    """
    from sensorsim import window_policy, training_negatives as tn
    from sensorsim.training_device import ReceiverEvent
    matched = tn.validate_config(cfg)
    if window_policy.validate_config(cfg):
        return window_policy.draw_example(corpus, rng, cfg, mixer)
    fs, n = cfg['fs'], round(cfg['window_s']*cfg['fs'])
    need_power = context or cfg.get('audibility_gate', False) or cfg.get('cue_validity', False)
    if need_power and (cfg.get('device_split', 'in_distribution') != 'in_distribution' or
                       cfg.get('background_domain', 'device_recorded') != 'device_recorded'):
        raise ValueError('dynamics context requires in_distribution device_recorded audio')
    label = np.zeros(len(corpus.classes), np.float32)
    bed_index = int(rng.integers(len(corpus.background)))
    bed = _crop(corpus.background[bed_index], 2*n, rng).astype(float)
    selected = rng.random() >= cfg['p_background_only']
    source, span, class_name, index = None, None, None, -1
    window, active = np.zeros(n), (0, n)
    if selected:
        index = int(rng.integers(len(corpus.classes)))
        class_name = corpus.classes[index]
        k = int(rng.integers(len(corpus.targets[class_name])))
        source, span = corpus.targets[class_name][k], corpus.spans[class_name][k]
        window, active = _event_window(source, span, n, rng)  # exactly base RNG consumption
        if active[1] <= active[0] or not np.any(window):
            raise ValueError(f'dynamics source QA: empty event for {class_name}, item {k}')
        label[index] = 1.
    if not selected and not matched:
        from sensorsim.negatives.integration import draw_confuser
        confuser = draw_confuser(corpus, rng, cfg, n)
        if confuser is not None:
            window, active = confuser
    negative_draw = (tn.prepare_context(corpus, rng, cfg, n, class_name=class_name,
                     target_index=k if selected else None) if matched else None)
    mix_rng = clone_rng(rng)
    placement_rng = auxiliary_rng(rng, 'placement')
    local_cfg = dict(cfg)
    if cfg.get('low_event_spl_tail'):
        if cfg.get('device_split', 'in_distribution') != 'in_distribution':
            raise ValueError('low_event_spl_tail requires the SPL device contract')
        if auxiliary_rng(rng, 'low-tail').random() < .25:
            local_cfg['event_spl_db'] = [45., 60.]
    # Record the actual requested level using an independent copy of mixer RNG.
    probe = clone_rng(mix_rng)
    if cfg.get('device_split', 'in_distribution') == 'in_distribution':
        profile = iot_mic.PROFILES[cfg.get('device_profile', 'pi5_inmp441/1')].with_variation(probe)
        requested_level = float(probe.uniform(*local_cfg.get('event_spl_db', [60., 80.]))) if selected else None
        # Main's capture adds the rendered increment to the recording without requantizing,
        # so the event-free counterfactual is the recorded bed itself.
        floor = bed[n:].copy() if selected and need_power else None
        level_key = 'event_spl_db'
    else:
        requested_level = float(probe.uniform(*cfg['snr_db'])) if selected else None
        floor, level_key = None, 'snr_db'
    best = best_window_energy(source, span, n) if selected and cfg.get('partial_windows') else 1.
    split = cfg.get('device_split', 'in_distribution')
    full_power = (training_take_cache.take_level_power(source, fs, span, split=split)
                  if selected and (cfg.get('partial_windows') or cfg.get('cue_validity')) else 0.)
    if cfg.get('cue_validity'):
        from sensorsim import training_policy as cue
        cue.validate_config(cfg)
        annotation = (corpus.cue_annotations[class_name][k] if selected
                      else corpus.cue_background_annotations[bed_index])
    max_attempts = int(cfg.get('dynamics_max_attempts', 32))
    if max_attempts < 1:
        raise ValueError('dynamics_max_attempts must be positive')
    for attempt in range(max_attempts):
        start, retained = None, 1.
        if selected and cfg.get('partial_windows'):
            window, active, start = partial_window(source, span, n, placement_rng)
            retained = float(np.sum(window**2)/max(best, 1e-30))
        elif selected and attempt:
            window, active = _event_window(source, span, n, placement_rng)
        if cfg.get('cue_validity'):
            from sensorsim.recognizer import _padded_window
            if selected and cfg.get('partial_windows'):
                # Geometry-only proposals, one draw; ambiguity becomes unknown.
                start = span[0]+cue.propose_crop_starts(placement_rng, span[1]-span[0], n,
                    round(cfg.get('detector', {}).get('hop_s', .25)*fs), 1)[0]
                window = _padded_window(source, start, n)
                active = (max(0, span[0]-start), min(n, span[1]-start))
                retained = float(np.sum(window**2)/max(best, 1e-30))
            else:
                start = span[0]-active[0] if selected else 0
        event = np.r_[np.zeros(n), window]
        mix_active = (n+active[0], n+active[1])
        if cfg.get('cue_validity') and selected:
            # Preserve real source history at middle/late crops.
            event = _padded_window(source, start-n, 2*n)
            mix_active = (max(0, n+span[0]-start), min(2*n, n+span[1]-start))
        attempt_cfg = dict(local_cfg)
        if selected and (cfg.get('partial_windows') or cfg.get('cue_validity')):
            a, b = active
            crop_power = (event_level.level_power(event, fs, mix_active, split=split)
                          if cfg.get('cue_validity') else event_level.level_power(window, fs, active, split=split))
            # Mixer normalizes over this crop; compensate to inherit whole-take gain.
            shift = 10*np.log10(max(crop_power, 1e-30)/max(full_power, 1e-30))
            attempt_cfg[level_key] = [float(v + shift) for v in local_cfg[level_key]]
        replay = clone_rng(mix_rng)
        if matched:
            mixed, counterfactual = tn.draw_training_context(bed,
                ReceiverEvent(event, mix_active), replay, attempt_cfg,
                draw=negative_draw, counterfactual=need_power and selected)
            heard = mixed[n:].astype(np.float32)
            if counterfactual is not None:
                floor = counterfactual[n:]
        else:
            heard = mixer(bed, event, mix_active, replay, attempt_cfg)[n:].astype(np.float32)
        ctx = dict(class_name=class_name, class_index=index, positive=selected, source_start=start,
                   retained_fraction=retained, attempts=attempt+1, rejections=attempt,
                   requested_level_db=requested_level, level_kind=level_key)
        if need_power:
            base = heard if not selected else floor
            ctx.update(event=power(heard.astype(float)-base, cfg) if selected else power(heard, cfg),
                       floor=power(base, cfg))
        if cfg.get('cue_validity'):
            ctx['cue_pair_context'] = cue.pair_context(bed, mix_rng, fs)
            label = cue.annotate_context(ctx, annotation, n, cfg)
        if not selected or eligible(ctx, cfg):
            if matched:
                ctx['negative_context'] = negative_draw.provenance
                if negative_draw.provenance['injected']:
                    corpus.confuser_draws += 1
            rng.bit_generator.state = copy.deepcopy(replay.bit_generator.state)
            return heard, label, ctx
    raise ValueError(f'dynamics source QA: {class_name} item {k}, level {requested_level}, '
                     f'{max_attempts}/{max_attempts} rejected; class and level were not redrawn')


def event_aware_masks(x, rng, p, contexts):
    """One width distribution for all labels; protect event cues (bed cues for negatives)."""
    if p <= 0:
        return x
    for i, ctx in enumerate(contexts):
        if rng.random() >= p:
            continue
        e, f = np.asarray(ctx.get('event')), np.asarray(ctx.get('floor'))
        if (i >= len(x) or min(x.shape[2:]) < 2 or e.shape != x.shape[2:] or
                f.shape != e.shape or not np.isfinite(e).all() or not np.isfinite(f).all()):
            ctx['mask_qa_skips'] = ctx.get('mask_qa_skips', 0) + 1
            continue
        energy = e.sum(0)
        active = energy > max(1e-30, energy.max()*10**(-2.5))
        width_f = int(rng.integers(1, min(5, x.shape[2]-1)+1))
        width_t = int(rng.integers(1, min(3, x.shape[3]-1)+1))
        # Candidate widths never depend on class. The final candidate masks the
        # least energetic bands/frames, rather than omitting the mask on a positive.
        candidates = [(int(rng.integers(x.shape[2]-width_f+1)),
                       int(rng.integers(x.shape[3]-width_t+1))) for _ in range(16)]
        candidates.append((int(np.argmin(np.convolve(e.sum(1), np.ones(width_f), 'valid'))),
                           int(np.argmin(np.convolve(energy, np.ones(width_t), 'valid')))))
        for f0, t0 in candidates:
            mask = np.zeros_like(e, dtype=bool)
            mask[f0:f0+width_f] = True
            mask[:, t0:t0+width_t] = True
            if np.sum(active[t0:t0+width_t]) > .5*active.sum():
                continue
            if ctx['positive'] and audible_frames(e, f).sum() >= 3 and audible_frames(e*(~mask), f).sum() < 3:
                continue
            x[i, 0][mask] = 0.
            break
        else:
            # Pathological narrow cues: try a non-event region. Count an
            # impossible mask and leave this example intact; training continues.
            valid = np.convolve(active.astype(int), np.ones(width_t), 'valid') == 0
            if not valid.any():
                ctx['mask_qa_skips'] = ctx.get('mask_qa_skips', 0) + 1
                continue
            t0 = int(np.flatnonzero(valid)[0])
            x[i, 0, :, t0:t0+width_t] = 0.
    return x


def validation_config(cfg):
    result = dict(cfg)
    if cfg.get('negative_context_policy') == 'matched/1':
        result['negative_context_policy'] = 'legacy'
        result['p_confuser_given_background'] = 0.
    for key in ('audibility_gate', 'partial_windows', 'event_aware_masks', 'low_event_spl_tail'):
        result[key] = False
    if cfg.get('cue_validity'):
        result.update(cue_masks=False, p_specaugment=0.)
    result['positive_fraction'] = None
    mode = cfg.get('validation_mode', 'synthetic')
    if mode == 'hard':
        result.update(event_spl_db=[45., 65.], p_background_only=.8)
    elif mode != 'synthetic':
        raise ValueError("validation_mode must be 'synthetic' or 'hard'")
    return result


def validation_corpus(corpus, cfg):
    """Hard validation: an explicit louder-bed stress context, never a calibrated room.

    Apply the same gain to both labels and copy the corpus to leave training and
    ordinary validation untouched. This scales the recorded floor too; it is an
    evaluation stress condition, not a new microphone model.
    """
    if cfg.get('validation_mode', 'synthetic') != 'hard':
        return corpus
    gain_db = float(cfg.get('validation_background_gain_db', 15.))
    if not np.isfinite(gain_db):
        raise ValueError('validation_background_gain_db must be finite')
    result = copy.copy(corpus)
    result.background = [x * 10**(gain_db/20) for x in corpus.background]
    if any(not np.isfinite(x).all() or np.max(np.abs(x)) >= 1 for x in result.background):
        raise ValueError('hard-validation bed would clip; reduce its declared gain')
    return result
