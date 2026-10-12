"""Bounded sampling of the actual training audio stream and matched held-out takes.

No training or microphone policy is reimplemented here. Each synthetic sample comes
from nn.make_example, conditioned on the selected positive source group. Feature
augmentation is intentionally excluded: this tests the audio seen by edge.Features.
"""
from collections import defaultdict
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import uniform_filter1d

from sensorsim import edge
from .realism import extract_audio_features

MAX_GROUPS = 256


def combine_status(statuses):
    values = list(statuses)
    return 'fail' if 'fail' in values else ('pass' if values and all(v == 'pass' for v in values) else 'inconclusive')


def training_example(corpus, rng, config):
    """One thin, signature-stable adapter; optional new make_example kwargs keep defaults.

    All level, SNR, background and device policy lives in the trainer. Only the
    positive-only condition changes, because E1/E2 are per target class.
    """
    from sensorsim import nn
    audio, label = nn.make_example(corpus, rng, dict(config, p_background_only=0.))
    if not np.any(label > 0):
        raise ValueError('training example has no cue-supported audible target (absent or unknown)')
    return np.asarray(audio, np.float32)


def event_anchor(audio, fs):
    """Strongest 10 ms energy region; a background transient may win at low SNR."""
    audio = np.asarray(audio, np.float64)
    if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
        raise ValueError('empty, nonfinite or silent event audio')
    return int(np.argmax(uniform_filter1d(audio*audio, max(1, round(.01*fs)), mode='constant')))


def event_crop(audio, model, *, anchor_offset=None, return_info=False):
    """One full device window around the event, with no zero padding or tiling.

    For training-stream comparisons the reference uses the synthetic example's
    peak offset. Thus the synthetic window is untouched and event placement is
    matched, while unused context from a long reference take cannot dominate.
    """
    fs, n = int(model.meta['features']['fs']), model.features.n
    audio = np.asarray(audio, np.float32).reshape(-1)
    if len(audio) < n:
        raise ValueError('take shorter than one full device window')
    peak = event_anchor(audio, fs)
    offset = n//2 if anchor_offset is None else int(anchor_offset)
    start = peak-offset
    requested_start = start
    if anchor_offset is None:
        start = max(0, min(start, len(audio)-n))
    if start < 0 or start+n > len(audio):
        raise ValueError('insufficient context for event-aligned window without padding')
    cropped = audio[start:start+n]
    info = {'anchor_sample': peak, 'window_start_sample': start,
            'anchor_offset_sample': peak-start, 'clamped': start != requested_start}
    return (cropped, info) if return_info else cropped


def feature_bucket(windows, groups, files, model, *, kind, sampling):
    # Windows are bounded before CNN inference. Individual extraction also bounds activations.
    windows = [np.array(w, dtype=np.float32, copy=True) for w in windows]
    rows = [extract_audio_features(w, model) for w in windows]
    if not rows:
        return None
    return {'stats': np.concatenate([r['band_statistics'] for r in rows]),
            'embedding': np.concatenate([r['embedding'] for r in rows]),
            'groups': np.asarray(groups).astype(str), 'files': files,
            'band_centres_hz': rows[0]['band_centres_hz'], 'band_3khz': rows[0]['band_3khz'],
            'kind': kind, 'window_sampling': sampling, 'audio_windows': windows}


def _seed(seed, *parts):
    digest = sha256('|'.join(map(str, parts)).encode()).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:4], 'little')])


class _LazyAudio:
    """Sequence interface expected by Corpus, without eagerly loading the dataset."""
    def __init__(self, rows, root, fs):
        self.rows, self.root, self.fs = rows, root, fs
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return edge.read_wav(str(self.root/self.rows[index]['file']), self.fs)


def training_features(items, rows, out, model, *, data_kind, seed=20261008):
    from sensorsim import nn
    from sensorsim.recognizer import _audible_span

    metadata = (model.meta.get('training') or {}).get('config')
    if metadata is None:
        return {}, {}, {'status': 'inconclusive', 'reason': 'frozen training configuration absent from model metadata'}
    cfg = dict(nn.DEFAULTS, **metadata)
    # Models exported before the Pi profile existed used the legacy stress policy.
    # New defaults must not silently reinterpret their stored training configuration.
    if 'device_split' not in metadata and 'device_split' in nn.DEFAULTS:
        cfg['device_split'] = 'stress'
    cfg.update(model.meta['features'])
    fs = int(cfg['fs'])
    train_rows = [r for r in rows if r.get('split') == 'train']
    background_rows = [r for r in train_rows if r['label'] == 'background']
    if not background_rows:
        return {}, {}, {'status': 'inconclusive', 'reason': 'no training backgrounds; cannot reproduce nn.make_example'}
    references, targets = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for item in items:
        references[item['label']][str(item['group'])].append(item)
    for row in train_rows:
        if row['label'] == 'background':
            continue
        dry = row.get('dry_source') or {}
        group = str(dry.get('source_group') or row.get('parent_group') or row.get('source_group') or row.get('id', row['file']))
        targets[row['label']][group].append(row)
    ref_features, syn_features, sampling = {}, {}, {}
    background = _LazyAudio(background_rows, out/'dataset', fs)
    for label in sorted(references.keys() & targets.keys()):
        ref_names, syn_names = sorted(references[label]), sorted(targets[label])
        rng = _seed(seed, label, 'groups')
        count = min(len(ref_names), len(syn_names), MAX_GROUPS)
        ref_names = rng.permutation(ref_names)[:count]
        syn_names = rng.permutation(syn_names)[:count]
        real, synthetic, rg, sg, rf, sf, excluded = [], [], [], [], [], [], []
        for index, (ref_name, syn_name) in enumerate(zip(ref_names, syn_names)):
            rng = _seed(seed, label, index)
            ref_rows, syn_rows = references[label][ref_name], targets[label][syn_name]
            ref = ref_rows[int(rng.integers(len(ref_rows)))]
            row = syn_rows[int(rng.integers(len(syn_rows)))]
            try:
                target = edge.read_wav(str(out/'dataset'/row['file']), fs)
                corpus = SimpleNamespace(classes=[label], background=background,
                    targets={label: [target]}, spans={label: [_audible_span(target, fs)]},
                    window_cues={label: [row.get('window_cues')]})
                generated = training_example(corpus, rng, cfg)
                if len(generated) != model.features.n:
                    raise ValueError('trainer returned a different window length')
                offset = event_anchor(generated, fs)
                measured = event_crop(edge.read_wav(str(ref['path']), fs), model, anchor_offset=offset)
            except (ValueError, OSError) as error:
                excluded.append({'reference_group': str(ref_name), 'synthetic_group': str(syn_name), 'reason': str(error)})
                continue
            real.append(measured); synthetic.append(generated)
            rg.append(ref_name); sg.append(syn_name); rf.append(ref['file']); sf.append(row['file'])
        sample = {'strategy': 'one event-aligned window per group; paired peak offsets; no padding',
                  'seed': seed, 'selected_pairs': count, 'used_pairs': len(real),
                  'available_groups': {'reference': len(references[label]), 'synthetic': len(targets[label])},
                  'maximum_groups_per_class': MAX_GROUPS, 'excluded_pairs': excluded}
        sampling[label] = sample
        if real:
            ref_features[label] = feature_bucket(real, rg, rf, model, kind=data_kind, sampling=sample)
            syn_features[label] = feature_bucket(synthetic, sg, sf, model, kind='proxy', sampling=sample)
    return ref_features, syn_features, {'status': 'ok' if ref_features else 'inconclusive',
        'adapter': 'sensorsim.nn.make_example', 'conditioning': 'positive class and source group',
        'training_config': cfg, 'framing': sampling,
        'limits': ['Peak alignment assumes the target is the strongest event; annotate/crop contaminated takes.',
                   'Context exclusions depend on take length and can select a different subset; inspect excluded_pairs.',
                   'At low SNR the peak anchor can select a background transient; no target-span agreement is inferred.',
                   'Paired offsets condition on event placement; this is not continuous-stream false-alarm validation.']}


def file_features(items, model, *, kind, framing='event', seed=20261008):
    """Bound group/window counts before CNN extraction; report excluded short takes."""
    from .realism import window_starts
    grouped = defaultdict(lambda: defaultdict(list))
    for item in items:
        label = str(item['label'])
        if label == 'background' or label == 'negative' or label.startswith('negative_'):
            continue
        grouped[label][str(item['group'])].append(item)
    result = {}
    fs, n = int(model.meta['features']['fs']), model.features.n
    for label, groups in sorted(grouped.items()):
        names = sorted(groups)
        rng = _seed(seed, label, 'file-groups')
        if len(names) > MAX_GROUPS:
            names = sorted(rng.choice(names, MAX_GROUPS, replace=False))
        windows, ids, files, excluded, framing_details = [], [], [], [], []
        per_group_cap = max(1, 1024//len(names))
        for name in names:
            # One file/window per performance group for primary event diagnostics.
            members = sorted(groups[name], key=lambda r: r['file'])
            if framing == 'event':
                members = [members[int(rng.integers(len(members)))]]
            used = 0
            for item in members:
                audio = edge.read_wav(str(item['path']), fs)
                if framing == 'event':
                    try:
                        window, info = event_crop(audio, model, return_info=True)
                        candidates = [window]
                        framing_details.append({'group': name, 'file': item['file'], **info})
                    except ValueError as error:
                        excluded.append({'group': name, 'file': item['file'], 'reason': str(error)})
                        continue
                else:
                    starts = window_starts(len(audio), n, round(fs*model.meta['detector']['hop_s']))
                    if not len(starts):
                        excluded.append({'group': name, 'file': item['file'], 'reason': 'shorter than device window'})
                        continue
                    remain = per_group_cap-used
                    if len(starts) > remain:
                        starts = np.sort(rng.choice(starts, remain, replace=False))
                    candidates = [audio[i:i+n] for i in starts]
                windows.extend(candidates); ids.extend([name]*len(candidates)); files.extend([item['file']]*len(candidates))
                used += len(candidates)
                if used >= per_group_cap:
                    break
        sampling = {'strategy': framing, 'used_windows': len(windows), 'used_groups': len(set(ids)),
                    'available_groups': len(groups), 'excluded': excluded,
                    'maximum_groups': MAX_GROUPS, 'maximum_windows': 1024,
                    'clamped_windows': sum(r['clamped'] for r in framing_details),
                    'event_framing': framing_details,
                    'limits': [
                        'For a 1 s window, peaks within 0.5 s of a take edge trigger the event-crop clamp; placement can depend on length.',
                        'At low SNR the peak anchor can select a background transient; no target-span agreement is inferred.'] if framing == 'event' else []}
        bucket = feature_bucket(windows, ids, files, model, kind=kind, sampling=sampling)
        if bucket:
            result[label] = bucket
    return result
