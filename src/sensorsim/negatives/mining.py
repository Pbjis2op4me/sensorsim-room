"""Frozen-model development mining, listening review and capped training weights.

Audit identities are rejected before inference, including mixed pools. Scores
select review candidates only: they never certify that an audio window lacks a
missed target. Review IDs bind the model, split, source samples and time window.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .composer import audio_hash, digest
from .splits import Item, SplitPlan


class FrozenModel:
    """Trusted inference callback bound to immutable serialized checkpoint bytes.

    Use from_edge() for the shipped runtime. A custom scorer must be inference-
    only and return probabilities; its checkpoint hash is checked before/after
    mining. The callback is application code, never loaded from a manifest.
    """
    def __init__(self, checkpoint, scorer, *, classes, fs: int, window_s: float = 1.):
        self.checkpoint = Path(checkpoint)
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.scorer, self.classes, self.fs = scorer, tuple(classes), fs
        self.n = round(window_s*fs)
        if not self.classes or len(set(self.classes)) != len(self.classes) or self.n < 1 or fs < 1:
            raise ValueError('invalid model classes, rate or window')

    def verify(self):
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.sha256:
            raise ValueError('frozen model checkpoint changed')

    @classmethod
    def from_edge(cls, checkpoint):
        from sensorsim.edge import Model
        before = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        runtime = Model(str(checkpoint))
        for value in runtime.weights.values():
            value.setflags(write=False)
        frozen = cls(checkpoint, runtime.scores, classes=runtime.classes,
                     fs=runtime.meta['features']['fs'], window_s=runtime.features.n/runtime.meta['features']['fs'])
        if frozen.sha256 != before:
            raise ValueError('checkpoint changed during model loading')
        return frozen


@dataclass(frozen=True)
class PoolItem:
    item: Item
    audio: np.ndarray
    fs: int
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    review_id: str
    item: Item
    start_sample: int
    end_sample: int
    fs: int
    scores: tuple[float, ...]
    classes: tuple[str, ...]
    model_sha256: str
    split_sha256: str
    audio_sha256: str


def mine(pool, *, model: FrozenModel, split: str | Path, expected_split_sha256: str,
         threshold: float = .5, hop_s: float = .25, max_per_item: int = 8,
         max_total: int = 1000) -> tuple[Candidate, ...]:
    """Rank nonoverlapping high-score windows; no incomplete/padded tail windows.

    split is the persisted audit-lock PATH; expected_split_sha256 is its raw
    FILE hash frozen in the preregistration (not a rebuilt plan payload hash).
    Both mining and bank admission reopen and validate that same file.
    Item.sha256 binds decoded pool audio (float64 little-endian), not a WAV
    container. source_group must retain the original take/item identity across
    renders. Freeze selection parameters in the run's preregistration.
    """
    split = SplitPlan.from_audit_lock(split, expected_file_sha256=expected_split_sha256)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or not np.isfinite(hop_s) or hop_s <= 0:
        raise ValueError('invalid mining threshold or hop')
    if not isinstance(max_per_item, int) or not isinstance(max_total, int) or min(max_per_item, max_total) < 1:
        raise ValueError('positive integer candidate caps required')
    hop = round(hop_s*model.fs)
    if hop < 1:
        raise ValueError('hop rounds to zero samples')
    pool = tuple(pool)
    if len({row.item.item_id for row in pool}) != len(pool):
        raise ValueError('duplicate pool item')
    for row in pool:  # validate entire pool before ANY scoring
        if row.provenance.get('role') in ('calibration', 'selection', 'audit'):
            raise ValueError('calibration and audit audio are never mined')
        split.require(row.item, purpose='mine')
        x = np.asarray(row.audio)
        if row.fs != model.fs or x.ndim != 1 or not np.isfinite(x).all() or len(x) < model.n:
            raise ValueError('invalid or too-short pool audio / sample rate')
        if audio_hash(x) != row.item.sha256:
            raise ValueError('pool content no longer matches split identity')
    model.verify()
    selected = []
    for row in pool:
        windows = []
        x = np.asarray(row.audio, dtype=float)
        for start in range(0, len(x)-model.n+1, hop):
            window = x[start:start+model.n].copy()
            score = np.asarray(model.scorer(window), dtype=float).reshape(-1)
            if score.shape != (len(model.classes),) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
                raise ValueError('frozen scorer must return finite per-class probabilities')
            if score.max() >= threshold:
                windows.append((float(score.max()), start, tuple(map(float, score)), audio_hash(window)))
        chosen = []
        for _, start, score, content_hash in sorted(windows, key=lambda v: (-v[0], v[1])):
            if any(abs(start-prev) < model.n for prev in chosen):
                continue
            chosen.append(start)
            identity = {'item': asdict(row.item), 'start': start, 'end': start+model.n,
                        'model': model.sha256, 'split': split.sha256, 'audio': content_hash}
            selected.append(Candidate(digest(identity), row.item, start, start+model.n, row.fs,
                                      score, model.classes, model.sha256, split.sha256, content_hash))
            if len(chosen) == max_per_item:
                break
    model.verify()
    return tuple(sorted(selected, key=lambda c: (-max(c.scores), c.review_id))[:max_total])


def export_review(candidates, pool, folder: str | Path):
    """Write project/user audio to a caller-owned review directory, never the repo.

    Float WAV preserves listening levels; no normalization. Existing files are
    refused. reviews.json is a blank decision template, never auto-approval.
    """
    candidates = tuple(candidates)
    rows = {row.item.item_id: row for row in pool}
    prepared = []
    for candidate in candidates:
        row = rows[candidate.item.item_id]
        x = np.asarray(row.audio[candidate.start_sample:candidate.end_sample], dtype=float)
        if row.item != candidate.item or audio_hash(row.audio) != row.item.sha256 or audio_hash(x) != candidate.audio_sha256:
            raise ValueError('review audio does not match the mined window')
        prepared.append((candidate, x))
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=False)
    reviews = []
    for candidate, x in prepared:
        name = candidate.review_id + '.wav'
        wavfile.write(root/name, candidate.fs, x)
        reviews.append({**asdict(candidate), 'file': name, 'decision': 'pending', 'reviewer': '', 'notes': ''})
    (root/'reviews.json').write_text(json.dumps(reviews, indent=2)+'\n', encoding='utf-8')


def reviewed_bank(candidates, reviews: dict, *, split: str | Path, expected_split_sha256: str,
                  base_training_weight: float, max_fraction: float = .2,
                  max_source_weight: float = .05, max_family_weight: float = .1) -> tuple[dict, ...]:
    """Admit only explicit human negative decisions; cap source, family AND total.

    All caps are fractions of base_training_weight (including total).
    Returned additions keep development origin; they are not a new independent
    validation set. Never tune on development after adding it to training.
    """
    split = SplitPlan.from_audit_lock(split, expected_file_sha256=expected_split_sha256)
    if (not np.isfinite([base_training_weight, max_fraction, max_source_weight, max_family_weight]).all()
            or base_training_weight <= 0 or not 0 < max_fraction <= .5 or min(max_source_weight, max_family_weight) <= 0):
        raise ValueError('invalid training-weight caps')
    candidates = tuple(candidates)
    if len({c.model_sha256 for c in candidates}) > 1:
        raise ValueError('one frozen model checkpoint per bank is required')
    windows = [(c.item.sha256, c.start_sample, c.end_sample) for c in candidates]
    if len(set(windows)) != len(windows):
        raise ValueError('duplicate mined content window')
    if len({c.review_id for c in candidates}) != len(candidates):
        raise ValueError('duplicate mined candidate')
    approved = []
    for c in candidates:
        split.require(c.item, purpose='mine')
        if c.split_sha256 != split.sha256:
            raise ValueError('candidate belongs to another split')
        identity = {'item': asdict(c.item), 'start': c.start_sample, 'end': c.end_sample,
                    'model': c.model_sha256, 'split': c.split_sha256, 'audio': c.audio_sha256}
        if digest(identity) != c.review_id:
            raise ValueError('candidate identity changed since mining')
        review = reviews.get(c.review_id, {})
        if review.get('decision') != 'confirmed_negative':
            continue
        if not all(isinstance(review.get(k), str) and review[k].strip() for k in ('reviewer', 'notes')):
            raise ValueError('listening reviewer and notes required')
        approved.append(c)
    if not approved:
        return ()
    sources = Counter(c.item.source_group for c in approved)
    families = Counter(c.item.family for c in approved)
    total_budget = base_training_weight*max_fraction
    return tuple({**asdict(c), 'label': 'background', 'origin_split': 'development',
                  'review': dict(reviews[c.review_id]),
                  'weight': min(base_training_weight*max_source_weight/sources[c.item.source_group],
                                base_training_weight*max_family_weight/families[c.item.family], total_budget/len(approved))}
                 for c in approved)
