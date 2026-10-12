"""Take identities survive cropping, fitting and rendering; split before synthesis."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from scipy.signal import fftconvolve


def mono(samples, fs):
    if isinstance(fs, bool) or not isinstance(fs, (int, np.integer)) or not 1000 <= fs <= 192000:
        raise ValueError('sample rate must be an integer in [1000, 192000]')
    x = np.asarray(samples)
    if x.ndim != 1 or len(x) < 2 or x.dtype.kind != 'f' or not np.isfinite(x).all():
        raise ValueError('source audio must be finite mono floating point, decoded in peak-FS units')
    return np.array(x, dtype='<f8', copy=True)


def content_hash(samples, fs):
    x = mono(samples, fs)
    # Treat signed zero identically; container names and dtypes are not identities.
    x[x == 0] = 0
    h = hashlib.sha256(int(fs).to_bytes(8, 'little'))
    h.update(x.tobytes())
    return h.hexdigest()


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{name} must be a nonempty, trimmed identifier')


@dataclass(frozen=True)
class SourceTake:
    label: str
    samples: np.ndarray
    fs: int
    source_group: str
    object_id: str
    session: str
    capture_route: str
    evidence_label: str
    parent_content_sha256: str | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ('label', 'source_group', 'object_id', 'session'):
            _identifier(getattr(self, name), name)
        if self.capture_route not in ('in_situ', 'near_dry', 'known_digital'):
            raise ValueError('capture_route must be in_situ, near_dry or known_digital')
        if self.evidence_label not in ('measured', 'proxy'):
            raise ValueError('evidence_label must be measured or proxy')
        x = mono(self.samples, self.fs)
        x.setflags(write=False)
        object.__setattr__(self, 'samples', x)
        parent = self.parent_content_sha256 or content_hash(x, self.fs)
        if len(parent) != 64 or any(c not in '0123456789abcdef' for c in parent):
            raise ValueError('parent_content_sha256 must be lowercase SHA256')
        object.__setattr__(self, 'parent_content_sha256', parent)
        # Copy nested provenance, rejecting NaN and non-serializable metadata.
        object.__setattr__(self, 'details', json.loads(json.dumps(self.details, allow_nan=False)))

    def metadata(self):
        return dict(label=self.label, fs=int(self.fs), samples=len(self.samples),
            source_group=self.source_group, object_id=self.object_id,
            session=self.session, capture_route=self.capture_route, evidence_label=self.evidence_label,
            parent_content_sha256=self.parent_content_sha256,
            content_sha256=content_hash(self.samples, self.fs), details=self.details)


@dataclass(frozen=True)
class AssignedSource:
    take: SourceTake
    split: str

    def __post_init__(self):
        if not isinstance(self.take, SourceTake) or self.split not in ('train', 'val', 'RH'):
            raise ValueError('an assigned source needs a SourceTake and train, val or RH split')


def _keys(take):
    row = take.metadata()
    keys = [('group', row['source_group']), ('hash', row['content_sha256']),
            ('hash', row['parent_content_sha256'])]
    keys += [('group', v) for v in take.details.get('fit_groups', [])]
    keys += [('hash', v) for v in take.details.get('fit_hashes', [])]
    return keys


def audit_splits(assigned):
    """Hard error for any group/content/parent/fit lineage crossing train/val/RH."""
    assigned = list(assigned)
    seen, labels = {}, {}
    for item in assigned:
        if not isinstance(item, AssignedSource):
            raise ValueError('split audit requires AssignedSource rows')
        row = item.take.metadata()
        key = row['content_sha256']
        old = labels.setdefault(key, row['label'])
        if old != row['label']:
            raise ValueError('conflicting labels for identical source content')
        for key in _keys(item.take):
            old = seen.setdefault(key, item.split)
            if old != item.split:
                raise ValueError(f'split leakage: {key[0]} {key[1]} crosses {old}/{item.split}')
        if item.take.details.get('synthetic') and item.split != 'train':
            raise ValueError('synthetic children are training-only')
    return dict(status='pass', clips=len(assigned), groups_crossing=0, hashes_crossing=0,
                counts={s: sum(a.split == s for a in assigned) for s in ('train', 'val', 'RH')})


def split_sources(takes, *, seed=610, heldout_fraction=1/3, val_fraction=1/3):
    """Reserve RH, then validation from the remainder, by connected identities.

    val_fraction is a fraction of non-RH groups. Objects and sessions remain
    reported, not silently conflated with independent performances. An object-
    disjoint evaluation requires a separate explicit assignment and audit.
    """
    from sensorsim.acoustics.room.dataset import _connected_groups
    takes = list(takes)
    if not takes or not 0 < heldout_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError('nonempty sources and split fractions in (0,1) required')
    audit_splits([AssignedSource(t, 'train') for t in takes])
    groups = _connected_groups([_keys(t) for t in takes])
    by_label = {}
    rng = np.random.default_rng(seed)
    for key in rng.permutation(list(groups)):
        by_label.setdefault(tuple(sorted({takes[i].label for i in groups[key]})), []).append(key)
    assignment = {}
    for labels, keys in by_label.items():
        if len(keys) < 3:
            raise ValueError(f'need at least three independent groups for train/val/RH: {labels}')
        rh = min(len(keys)-2, max(1, int(np.ceil(len(keys)*heldout_fraction))))
        val = min(len(keys)-rh-1, max(1, round((len(keys)-rh)*val_fraction)))
        for j, key in enumerate(keys):
            for i in groups[key]:
                assignment[i] = 'RH' if j < rh else 'val' if j < rh+val else 'train'
    result = [AssignedSource(t, assignment[i]) for i, t in enumerate(takes)]
    audit_splits(result)
    return result


def require_renderable(take):
    if take.capture_route == 'in_situ':
        raise ValueError('in_situ takes cannot enter the room renderer (double reverberation)')
    qa = take.details.get('qa', {})
    if qa.get('status') != 'pass':
        raise ValueError('source crop QA must pass before rendering')
    if qa.get('crop_content_sha256') != content_hash(take.samples, take.fs):
        raise ValueError('source samples changed after crop QA')


def render_source(source, rir, *, fs, loudspeaker=None, source_position=None):
    """Apply optional speaker then a caller-supplied WP4 RIR, preserving full tails.

    RIRs must be supplied at the source rate; room/device calibration and ensemble
    selection belong to WP4/WP5. Output remains relative-level model-based audio.
    """
    if not isinstance(source, AssignedSource):
        raise ValueError('split before rendering: pass an AssignedSource')
    take = source.take
    require_renderable(take)
    if fs != take.fs:
        raise ValueError('source and RIR sample rates must match explicitly')
    h = mono(rir, fs)
    if not np.any(h):
        raise ValueError('silent RIR')
    from .speaker import playback_contract, validate_source_position
    validate_source_position(take.details, source_position)
    x = take.samples
    speaker = dict(take.details.get('loudspeaker_response', {'status': 'unknown', 'applied': False}))
    if loudspeaker is not None:
        if take.details.get('synthetic'):
            raise ValueError('synthetic children cannot receive a loudspeaker response')
        if take.details.get('loudspeaker_response', {}).get('applied'):
            raise ValueError('loudspeaker response already applied')
        playback_contract(take.capture_route, take.details.get('loudspeaker_played', False),
                          take.details.get('loudspeaker_position'), response=loudspeaker)
        x = loudspeaker.apply(x, fs)
        speaker = loudspeaker.metadata()
    y = fftconvolve(x, h)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite source rendering')
    return y, dict(source=take.metadata(), split=source.split,
                   loudspeaker_response=speaker, result_label='model-based',
                   pitch_semitones=0., stretch_factor=1.)


def as_dry_clip(take, *, split=None):
    """Adapter for WP4; dataset must retain provenance and split BEFORE rendering.

    Without a split, the existing dataset group splitter owns the assignment.
    An explicit split must be honoured by the WP4 route-metadata integration.
    """
    from sensorsim.acoustics.room.dataset import DryClip
    from sensorsim.acoustics.room.source_reference import SourceReference
    require_renderable(take)
    if take.details.get('synthetic') and split != 'train':
        raise ValueError('synthetic children are training-only; assign split=train')
    provenance = take.metadata()
    if split is not None:
        if split not in ('train', 'val'):
            raise ValueError('RH sources cannot enter a training dataset')
        provenance['split'] = split
    return DryClip(take.label, take.samples, take.fs, provenance, take.source_group,
                   SourceReference(target='played_digital' if take.capture_route == 'known_digital' and not take.details.get('synthetic')
                                   and not take.details.get('loudspeaker_response', {}).get('applied')
                                   else 'effective_source', units='relative'))
