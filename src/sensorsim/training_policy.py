# SPDX-License-Identifier: Apache-2.0
"""Experimental cue validity; no automatic source annotation or default activation.

A reviewed annotation lists sufficient cues in rendered-source sample coordinates.
Each alternative is a conjunction of time/mel rectangles; any complete alternative
is sufficient. Crops must contain it and masks must preserve it in the *combined*
time/frequency mask. Energy is never used to infer class identity. Annotations are
assumptions, even when reviewed; deployment efficacy needs independent evaluation.

Labels use the existing convention: present=1, verified_absent=0, unknown=-1.
Unknowns have no loss; removing evidence never establishes absence. Masking is
training-only. Missing hooks, malformed provenance and empty loss batches fail
explicitly. This module does not choose model size, levels, negatives or MixStyle.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import numpy as np

POLICY = 'cue_valid/1'
PRESENT, VERIFIED_ABSENT, UNKNOWN = 1, 0, -1
STATES = {'present': PRESENT, 'verified_absent': VERIFIED_ABSENT, 'unknown': UNKNOWN}
DOMAINS = ('synthetic proxy', 'measured-RIR proxy', 'deployment recording')


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return int(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} is required')
    return value


def _states(values):
    values = np.asarray(values)
    if values.ndim not in (1, 2) or not values.size or not np.isin(values, [-1, 0, 1]).all():
        raise ValueError('nonempty class states must be -1, 0 or 1')
    return values.astype(np.float32)


def audio_digest(audio, fs):
    """Content identity includes rate and canonical finite mono float64 samples."""
    fs = _integer(fs, 'fs', 1)
    x = np.asarray(audio, dtype='<f8')
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.ptp(x) == 0:
        raise ValueError('source must be finite nonconstant nonempty mono audio')
    return hashlib.sha256(str(fs).encode() + b'\0' + x.tobytes()).hexdigest()


@dataclass(frozen=True)
class FeatureGrid:
    fs: int
    n_fft: int
    hop: int
    n_mels: int
    fmin_hz: float
    fmax_hz: float

    def __post_init__(self):
        for name in ('fs', 'n_fft', 'hop', 'n_mels'):
            _integer(getattr(self, name), name, 1)
        if self.hop > self.n_fft or not 0 <= self.fmin_hz < self.fmax_hz <= self.fs/2:
            raise ValueError('incompatible feature grid')

    def shape(self, samples):
        samples = _integer(samples, 'crop samples', self.n_fft)
        return self.n_mels, 1 + (samples-self.n_fft)//self.hop

    @classmethod
    def from_config(cls, cfg):
        return cls(**{key: cfg[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Region:
    start: int
    stop: int
    mel_bins: tuple[int, ...]


@dataclass(frozen=True)
class Annotation:
    source_id: str
    source_sha256: str
    annotation_id: str
    reviewer: str
    evidence_domain: str
    grid: FeatureGrid
    samples: int
    classes: tuple[str, ...]
    states: tuple[int, ...]
    # One tuple per class; each alternative is a tuple of required rectangles.
    cues: tuple[tuple[tuple[Region, ...], ...], ...]

    @classmethod
    def from_dict(cls, data, audio, grid, classes):
        """Require explicit review and exact waveform/grid binding; no onset defaults."""
        if not isinstance(data, dict) or data.get('schema') != POLICY or data.get('units') != 'samples/mel_bins':
            raise ValueError('cue annotation requires cue_valid/1 and units samples/mel_bins')
        if data.get('reviewed') is not True:
            raise ValueError('independently reviewed support is required')
        for key in ('source_id', 'annotation_id', 'reviewer'):
            _text(data.get(key), key)
        if data.get('evidence_domain') not in DOMAINS:
            raise ValueError('explicit evidence domain is required')
        if data.get('source_sha256') != audio_digest(audio, grid.fs):
            raise ValueError('annotation source content/rate digest mismatch')
        if data.get('feature_grid') != vars(grid):
            raise ValueError('annotation feature grid mismatch')
        if not classes or len(set(classes)) != len(classes) or any(not isinstance(c, str) or not c for c in classes):
            raise ValueError('unique nonempty class names required')
        entries = data.get('classes')
        if not isinstance(entries, dict) or set(entries) != set(classes):
            raise ValueError('every class needs an explicit label state')
        states, cues = [], []
        for name in classes:
            row = entries[name]
            if not isinstance(row, dict) or row.get('state') not in STATES:
                raise ValueError('unrecognized label state')
            state = STATES[row['state']]
            alternatives = []
            for alternative in row.get('alternatives', []):
                if not isinstance(alternative, list) or not alternative:
                    raise ValueError('empty sufficient cue alternative')
                regions = []
                for region in alternative:
                    a = _integer(region['start'], 'cue start', 0)
                    b = _integer(region['stop'], 'cue stop', 1)
                    bins = tuple(_integer(v, 'mel bin', 0) for v in region['mel_bins'])
                    if not a < b <= len(audio) or not bins or len(set(bins)) != len(bins) or max(bins) >= grid.n_mels:
                        raise ValueError('cue rectangle outside source/grid or duplicate bins')
                    regions.append(Region(a, b, bins))
                alternatives.append(tuple(regions))
            if (state == PRESENT) != bool(alternatives):
                raise ValueError('present needs sufficient cues; absent/unknown cannot declare cues')
            states.append(state)
            cues.append(tuple(alternatives))
        return cls(data['source_id'], data['source_sha256'], data['annotation_id'], data['reviewer'],
                   data['evidence_domain'], grid, len(audio), tuple(classes), tuple(states), tuple(cues))


def validate_bank(annotations):
    """Duplicate content cannot masquerade as independent source identities."""
    if not annotations:
        raise ValueError('empty/filtered annotation bank')
    for key in ('source_sha256',):
        values = [getattr(a, key) for a in annotations]
        if len(values) != len(set(values)):
            raise ValueError(f'duplicate annotation {key}')
    first = annotations[0]
    if any(a.classes != first.classes or a.grid != first.grid for a in annotations):
        raise ValueError('incompatible annotation classes/grid')
    if any(not any(a.states[i] == PRESENT for a in annotations) for i in range(len(first.classes))):
        raise ValueError('training bank has a class without reviewed present support')


def _supports(annotation, start, samples):
    start = _integer(start, 'crop start')
    shape = annotation.grid.shape(samples)
    starts = start + np.arange(shape[1])*annotation.grid.hop
    # The unused rightmost samples of a feature window cannot establish a cue.
    analysed_stop = int(starts[-1]) + annotation.grid.n_fft
    result = []
    for alternatives in annotation.cues:
        local = []
        for alternative in alternatives:
            if any(r.start < start or r.stop > analysed_stop for r in alternative):
                continue
            cells = np.zeros(shape, bool)
            for r in alternative:
                frames = (starts < r.stop) & (starts+annotation.grid.n_fft > r.start)
                cells[np.ix_(r.mel_bins, np.flatnonzero(frames))] = True
            if cells.any():
                local.append(cells)
        result.append(local)
    return result


def class_validity(annotation, start, samples, *, erased=None, available=None):
    """Return reviewed states after a crop/mask; support must also be observable.

    ``available`` is a boolean feature-cell audibility hook from the actual
    rendered event/floor path. It may remove support but never invent identity.
    Without it this is a structural constraint, not post-device audibility proof.
    """
    shape = annotation.grid.shape(samples)
    keep = np.ones(shape, bool)
    for array, name in ((erased, 'erased'), (available, 'available')):
        if array is not None:
            array = np.asarray(array)
            if array.shape != shape or array.dtype != bool:
                raise ValueError(f'{name} must be a boolean mel/frame array on the annotated grid')
            keep &= ~array if name == 'erased' else array
    states = np.array(annotation.states, dtype=np.float32)
    for i, alternatives in enumerate(_supports(annotation, start, samples)):
        if states[i] == PRESENT and not any(np.all(keep[cells]) for cells in alternatives):
            states[i] = UNKNOWN
    return states


@dataclass(frozen=True)
class CropDecision:
    start: int
    samples: int
    states: tuple[float, ...]
    phase: int
    preserves_present: bool


def constrain_crop(annotation, start, samples, *, hop_samples):
    """Assess exactly the proposed crop; never redraw source, class or level."""
    hop_samples = _integer(hop_samples, 'deployment hop samples', 1)
    if hop_samples > samples:
        raise ValueError('deployment hop exceeds crop')
    states = class_validity(annotation, start, samples)
    present = np.asarray(annotation.states) == PRESENT
    return CropDecision(int(start), int(samples), tuple(states.tolist()), int(start % hop_samples),
                        bool(np.all(states[present] == PRESENT)))


def propose_crop_starts(rng, source_samples, samples, hop_samples, count):
    """Class-neutral proposals: geometry only, one uniform deployment-grid phase."""
    for value, name in ((source_samples, 'source samples'), (samples, 'window samples'),
                        (hop_samples, 'hop samples'), (count, 'proposal count')):
        _integer(value, name, 1)
    if hop_samples > samples:
        raise ValueError('deployment hop exceeds crop')
    phase = int(rng.integers(hop_samples))
    first = ((-samples+1-phase + hop_samples-1)//hop_samples)*hop_samples+phase
    starts = np.arange(first, source_samples, hop_samples)
    return tuple(int(v) for v in rng.choice(starts, size=count))


@dataclass(frozen=True)
class MaskProposal:
    frequency: tuple[int, int]
    time: tuple[int, int]

    def cells(self, shape):
        result = np.zeros(shape, bool)
        for axis, pair in enumerate((self.frequency, self.time)):
            if len(pair) != 2:
                raise ValueError('mask interval must have two endpoints')
            a, b = (_integer(v, 'mask endpoint', 0) for v in pair)
            if not a <= b <= shape[axis]:
                raise ValueError('mask interval outside feature grid')
            if axis == 0:
                result[a:b, :] = True
            else:
                result[:, a:b] = True
        return result


def propose_masks(rng, shape, count=16, *, max_frequency=5, max_time=3):
    """No class, label, energy or support input; no special fallback proposal."""
    if len(shape) != 2:
        raise ValueError('mask shape must be mel/frame')
    for value in (*shape, count, max_frequency, max_time):
        _integer(value, 'mask dimension/count/width', 1)
    result = []
    for _ in range(count):
        f = int(rng.integers(1, min(max_frequency, shape[0])+1))
        t = int(rng.integers(1, min(max_time, shape[1])+1))
        f0 = int(rng.integers(shape[0]-f+1))
        t0 = int(rng.integers(shape[1]-t+1))
        result.append(MaskProposal((f0, f0+f), (t0, t0+t)))
    return tuple(result)


@dataclass
class MaskDecision:
    features: np.ndarray
    states: np.ndarray
    attempts: list[dict]
    accepted: bool
    reason: str


def constrain_masks(features, annotation, start, samples, proposals, *, training, available=None):
    """Reject combined cue erasure; exhausted proposals keep the original example.

    No validation call, missing proposal hook, constant input or identity transform
    is silently counted as successful augmentation. Input arrays are never mutated.
    """
    if training is not True:
        raise ValueError('validation masks are forbidden')
    x = np.asarray(features)
    if x.shape != annotation.grid.shape(samples) or not np.isfinite(x).all() or np.ptp(x) == 0:
        raise ValueError('mask features must be finite nonconstant and match the annotated grid')
    proposals = tuple(proposals)
    if not proposals:
        raise ValueError('missing/empty mask proposal hook')
    before = class_validity(annotation, start, samples, available=available)
    attempts = []
    for proposal in proposals:
        erased = proposal.cells(x.shape)
        after = class_validity(annotation, start, samples, erased=erased, available=available)
        candidate = np.where(erased, 0., x)
        changed = bool(np.any(candidate != x))
        preserves = bool(np.all(after[before == PRESENT] == PRESENT))
        # Unknown-only examples must not inflate the useful-augmentation numerator.
        useful = bool(np.any(before >= 0))
        reason = 'accepted' if changed and preserves and useful else (
            'identity' if not changed else 'cue_erased' if not preserves else 'all_unknown')
        attempts.append(dict(frequency=list(proposal.frequency), time=list(proposal.time),
                             changed=changed, states=after.tolist(), reason=reason))
        if reason == 'accepted':
            return MaskDecision(candidate, after, attempts, True, reason)
    return MaskDecision(x.copy(), before, attempts, False, 'exhausted')


def masked_bce_weights(states):
    """Return BCE targets/weights; unknowns use a safe placeholder with zero weight."""
    states = _states(states)
    weights = (states >= 0).astype(np.float32)
    if not weights.any():
        raise ValueError('all-unknown loss batch is invalid')
    return np.maximum(states, 0), weights


def masked_bce(logits, states):
    """Unreduced BCE, normalized over known cells; optional torch import is explicit."""
    import torch
    if logits.shape != states.shape or not torch.isfinite(logits).all():
        raise ValueError('finite logits must match class states')
    targets, weights = masked_bce_weights(states.detach().cpu().numpy())
    targets = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    weights = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss*weights).sum()/weights.sum()


class PolicyLedger:
    """Per-example proposal records and P6 guardrails, never an acceptance receipt.

    Baselines are the registered paired unaugmented states, not an inferred
    denominator from accepted examples. Record once per final example, including
    skipped proposals, to avoid counting retries as independent loss mass. Every
    attempted candidate still enters the augmentation-success denominator.
    """
    def __init__(self, classes):
        if not classes or len(classes) != len(set(classes)):
            raise ValueError('unique classes required')
        self.classes = tuple(classes)
        self.records = []
        self._example_ids = set()

    def record(self, *, example_id, source_id, source_sha256, phase, level_bin,
               baseline, final, eligible, augmented, attempts, skipped, details=()):
        before, after = _states(baseline), _states(final)
        if before.shape != (len(self.classes),) or after.shape != before.shape:
            raise ValueError('ledger needs one state per class')
        if example_id in self._example_ids:
            raise ValueError('duplicate example ID would inflate loss mass')
        for value, name in ((example_id, 'example ID'), (source_id, 'source ID'),
                            (source_sha256, 'source digest'), (level_bin, 'level bin')):
            _text(value, name)
        _integer(phase, 'crop phase', 0)
        _integer(attempts, 'attempts', 0)
        if any(type(v) is not bool for v in (eligible, augmented, skipped)):
            raise ValueError('ledger flags must be explicit booleans')
        if eligible and attempts < 1:
            raise ValueError('eligible proposals need at least one attempt')
        if augmented and (not eligible or skipped or attempts < 1):
            raise ValueError('inconsistent augmentation accounting')
        if np.any((before == VERIFIED_ABSENT) & (after != VERIFIED_ABSENT)):
            raise ValueError('verified-absence losses must be retained')
        if np.any((before != VERIFIED_ABSENT) & (after == VERIFIED_ABSENT)):
            raise ValueError('present/unknown cannot become verified absent through augmentation')
        self._example_ids.add(example_id)
        self.records.append(dict(example_id=example_id, source_id=source_id,
            source_sha256=source_sha256, phase=phase, level_bin=level_bin,
            baseline=before.tolist(), final=after.tolist(), eligible=eligible,
            augmented=augmented, attempts=attempts, skipped=skipped, details=list(details)))

    def summary(self):
        if not self.records:
            return dict(status='inconclusive', reasons=['empty/filtered ledger'])
        before = np.array([r['baseline'] for r in self.records])
        after = np.array([r['final'] for r in self.records])
        base = (before == PRESENT).sum(0)
        present = (after == PRESENT).sum(0)
        retained = ((before == PRESENT) & (after == PRESENT)).sum(0)
        eligible_examples = sum(r['eligible'] for r in self.records)
        eligible = sum(r['attempts'] for r in self.records if r['eligible'])
        augmented = sum(r['augmented'] for r in self.records)
        ratios = [float(n/d) if d else None for n, d in zip(retained, base)]
        reasons = []
        if not np.any(after >= 0):
            reasons.append('all_unknown')
        if np.any(present == 0):
            reasons.append('class_without_present_loss')
        if any(v is not None and v < .80 for v in ratios):
            reasons.append('present_loss_mass_below_0.80')
        if eligible and augmented/eligible < .25:
            reasons.append('nontrivial_augmentation_below_0.25')
        insufficient = not eligible or np.any(base == 0)
        strata = Counter()
        for r in self.records:
            # Both states and level/phase are observable, including skipped cases.
            key = json.dumps([r['baseline'], r['level_bin'], r['phase'], r['augmented'],
                              r['skipped'], r['attempts']], separators=(',', ':'))
            strata[key] += 1
        return dict(status='fail' if reasons else 'inconclusive' if insufficient else 'pass',
            claim_type='deterministic training guardrails only', reasons=reasons,
            examples=len(self.records), unique_contents=len({r['source_sha256'] for r in self.records}),
            parent_groups=len({r['source_id'] for r in self.records}),
            present_mass_fraction=dict(zip(self.classes, ratios)), eligible=eligible,
            eligible_examples=eligible_examples, denominator='all eligible attempts including retries',
            augmented=augmented, nontrivial_fraction=augmented/eligible if eligible else None,
            verified_absence_cells=int(np.sum(before == VERIFIED_ABSENT)),
            attempts=sum(r['attempts'] for r in self.records),
            retries=sum(max(0, r['attempts']-1) for r in self.records),
            skipped=sum(r['skipped'] for r in self.records), distributions=dict(strata))


def validate_config(cfg):
    """Validate the proposed opt-in trainer seam; existing defaults remain untouched."""
    for key in ('cue_validity', 'cue_masks'):
        if type(cfg.get(key, False)) is not bool:
            raise ValueError(f'{key} must be an explicit boolean')
    if not cfg.get('cue_validity'):
        if cfg.get('cue_masks'):
            raise ValueError('cue masks require cue validity')
        return False
    if cfg.get('window_policy', 'legacy') != 'legacy':
        raise ValueError('cue_validity is a separate experimental arm from deployment/1')
    if any(cfg.get(k) for k in ('low_event_spl_tail', 'audibility_gate', 'event_aware_masks', 'p_freq_mixstyle')):
        raise ValueError('cue validity must isolate crop/mask changes from other augmentation policies')
    probability = cfg.get('p_specaugment', 0.)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('mask probability must be finite and in [0,1]')
    if probability and not cfg.get('cue_masks'):
        raise ValueError('unconstrained masks are forbidden in the cue-valid arm')
    if (cfg.get('device_split', 'in_distribution') != 'in_distribution' or
            cfg.get('background_domain', 'device_recorded') != 'device_recorded'):
        raise ValueError('cue observability requires in_distribution device_recorded audio')
    FeatureGrid.from_config(cfg).shape(round(cfg['fs']*cfg['window_s']))
    return True


def bind_corpus(corpus, cfg):
    """Load per-row cue_support before training; never infer background absence.

    The confuser composer retains its separate reviewed absence/admission contract.
    Its source provenance is reported by that owner; a background annotation here
    attests only the background component, not a confuser's identity.
    """
    grid = FeatureGrid.from_config(cfg)
    target_annotations = {}
    all_annotations = []
    for name in corpus.classes:
        rows = [r for r in corpus.rows if r['label'] == name]
        sources = corpus.targets[name]
        if len(rows) != len(sources):
            raise ValueError('cue source/row alignment mismatch')
        annotations = [Annotation.from_dict(r.get('cue_support'), x, grid, corpus.classes)
                       for r, x in zip(rows, sources)]
        for a, row in zip(annotations, rows):
            if row.get('source_group') != a.source_id or a.states[corpus.classes.index(name)] != PRESENT:
                raise ValueError('cue source parent/target label mismatch')
        target_annotations[name] = annotations
        all_annotations.extend(annotations)
    validate_bank(all_annotations)
    rows = [r for r in corpus.rows if r['label'] == 'background']
    if len(rows) != len(corpus.background) or not rows:
        raise ValueError('missing/alignment mismatch for background annotation rows')
    backgrounds = [Annotation.from_dict(r.get('cue_support'), x, grid, corpus.classes)
                   for r, x in zip(rows, corpus.background)]
    if any(row.get('source_group') != a.source_id for row, a in zip(rows, backgrounds)):
        raise ValueError('cue background parent mismatch or missing parent')
    if any(any(state != VERIFIED_ABSENT for state in a.states) for a in backgrounds):
        raise ValueError('background requires reviewed absence for every target class')
    identities = [a.source_sha256 for a in all_annotations+backgrounds]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate target/background content')
    corpus.cue_annotations = target_annotations
    corpus.cue_background_annotations = backgrounds


def annotate_context(context, annotation, samples, cfg):
    """Actual mixer observability hook for the integration patch; no inferred cues."""
    from dataclasses import asdict
    e, floor = np.asarray(context['event']), np.asarray(context['floor'])
    expected = annotation.grid.shape(samples)
    if (e.shape != expected or floor.shape != expected or not np.isfinite(e).all() or
            not np.isfinite(floor).all() or np.any(e < 0) or np.any(floor < 0)):
        raise ValueError('post-device cue observability hook missing or incompatible')
    available = e >= 10**.3*np.maximum(floor, 1e-30)
    start = context['source_start']
    decision = constrain_crop(annotation, start, samples,
        hop_samples=round(cfg.get('detector', {}).get('hop_s', .25)*cfg['fs']))
    states = class_validity(annotation, start, samples, available=available)
    annotation_sha256 = hashlib.sha256(json.dumps(asdict(annotation), sort_keys=True).encode()).hexdigest()
    context.update(cue_annotation=annotation, cue_available=available,
        cue_states=states, positive=bool(np.any(states == PRESENT)),
        cue_trace=dict(policy=POLICY, source_id=annotation.source_id,
            source_sha256=annotation.source_sha256, annotation_sha256=annotation_sha256,
            evidence_domain=annotation.evidence_domain, source_start=start,
            phase=decision.phase, structural_states=list(decision.states), states=states.tolist(),
            crop_attempts=1, crop_unknown=bool(np.any(states == UNKNOWN)),
            mask_eligible=False, mask_accepted=False, mask_attempts=[], mask_reason='disabled',
            background_component_only=context['class_index'] < 0,
            level_db=context['requested_level_db'],
            pair_context=context.get('cue_pair_context')))
    return states


def mask_batch(x, rng, probability, contexts, *, training):
    """Proposed Batches hook; routes every attempted mask through constrain_masks."""
    if training is not True:
        raise ValueError('validation masks are forbidden')
    if (not np.isfinite(probability) or not 0 <= probability <= 1 or
            x.ndim != 4 or x.shape[1] != 1 or len(contexts) != len(x)):
        raise ValueError('invalid cue mask batch/probability/context count')
    result = x.copy()
    for i, ctx in enumerate(contexts):
        if not all(k in ctx for k in ('cue_trace', 'cue_annotation', 'cue_available')):
            raise ValueError('missing annotated cue context; no legacy fallback')
        trace = ctx['cue_trace']
        if rng.random() >= probability:
            trace['mask_reason'] = 'not_selected'
            continue
        trace['mask_eligible'] = True
        proposals = propose_masks(rng, x.shape[2:])
        annotation = ctx['cue_annotation']
        # Features determine the analysed sample count; no unrepresented tail cues.
        samples = (x.shape[-1]-1)*annotation.grid.hop+annotation.grid.n_fft
        decision = constrain_masks(x[i, 0], annotation, ctx['source_start'], samples,
            proposals, training=training, available=ctx['cue_available'])
        result[i, 0] = decision.features
        trace.update(mask_accepted=decision.accepted, mask_attempts=decision.attempts,
                     mask_reason=decision.reason)
        ctx['mask_qa_skips'] = int(not decision.accepted)
    return result


def training_diagnostics(records, classes, *, require_masks):
    """Evaluate consumed examples against separately rendered registered comparators."""
    if not records:
        raise ValueError('cue training trace missing/empty')
    states = np.asarray([r['states'] for r in records])
    _states(states)
    if states.shape[1] != len(classes) or np.any(np.sum(states == PRESENT, axis=0) == 0):
        raise ValueError('cue training has a class without valid-present loss')
    eligible = sum(r['mask_eligible'] for r in records)
    accepted = sum(r['mask_accepted'] for r in records)
    attempts = sum(len(r['mask_attempts']) for r in records)
    if require_masks and (not eligible or not accepted):
        raise ValueError('all-skipped/nontrivial-mask-free cue training is invalid')
    if require_masks and (attempts < eligible or accepted/attempts < .25):
        raise ValueError('nontrivial masks below 0.25 of all attempted proposals')
    return dict(policy=POLICY, examples=len(records), mask_eligible=eligible,
        mask_proposals=attempts, nontrivial_fraction=accepted/attempts if attempts else None,
        mask_accepted=accepted, mask_skipped=eligible-accepted,
        mask_retries=sum(max(0, len(r['mask_attempts'])-1) for r in records),
        present_counts=dict(zip(classes, (states == PRESENT).sum(0).tolist())),
        paired_guardrails=paired_ledger(records, classes).summary(),
        acceptance='not run', default_promotion=False,
        scientific_adoption='blocked: requires independent downstream P6 receipt')


PAIRING = 'same_draw_no_partial_no_masks/1'


def pair_context(bed, mix_rng, fs):
    """Bind the shared recorded bed and pre-mixer random state, before cropping."""
    return dict(background_window_sha256=audio_digest(bed, fs),
        mixer_rng_sha256=hashlib.sha256(json.dumps(mix_rng.bit_generator.state,
            sort_keys=True).encode()).hexdigest())


def paired_example(draw, corpus, rng, cfg, *, example_id):
    """Render a separate unaugmented comparator using the same source/mixer draw.

    The versioned protocol is frozen with the executable registration. Source,
    recorded bed, requested parent level and pre-mixer RNG must match. Baseline
    rendering uses a copied corpus so its telemetry cannot count as training.
    Confuser pairing needs the negative owner's per-draw identity hook; refuse
    that route until it exists, rather than treating bed identity as its proof.
    """
    import copy
    from sensorsim.training_dynamics import clone_rng
    from sensorsim import training_negatives
    if cfg.get('cue_pairing') != PAIRING:
        raise ValueError('registered cue pairing protocol is required')
    if (cfg.get('p_confuser_given_background', 0.) != 0 or
            (training_negatives.enabled(cfg) and cfg.get('p_confuser_given_target',
                training_negatives.DEFAULTS['p_confuser_given_target']) != 0)):
        raise ValueError('paired cue training requires the confuser identity hook; use diagnostic confusers off')
    baseline_cfg = dict(cfg, partial_windows=False, cue_masks=False, p_specaugment=0.)
    baseline = draw(copy.copy(corpus), clone_rng(rng), baseline_cfg, return_context=True)
    candidate = draw(corpus, rng, cfg, return_context=True)
    ref, final = baseline[2]['cue_trace'], candidate[2]['cue_trace']
    for key in ('source_id', 'source_sha256', 'annotation_sha256', 'pair_context', 'level_db'):
        if key not in ref or key not in final or ref[key] != final[key]:
            raise ValueError(f'paired comparator {key} mismatch or missing hook')
    for trace in (ref, final):
        _validate_pair_context(trace['pair_context'])
    reference = {key: ref[key] for key in ('source_id', 'source_sha256', 'annotation_sha256',
        'pair_context', 'level_db', 'states', 'source_start')}
    reference.update(protocol=PAIRING, audio_sha256=audio_digest(baseline[0], cfg['fs']))
    changed = not np.array_equal(baseline[0], candidate[0])
    eligible = bool(cfg.get('partial_windows') and candidate[2]['class_index'] >= 0)
    before, after = _states(ref['states']), _states(final['states'])
    accepted = bool(eligible and changed and np.any(before >= 0) and
                    np.all(after[before == PRESENT] == PRESENT))
    final.update(example_id=_text(example_id, 'example ID'), paired_baseline=reference,
        crop_eligible=eligible, crop_accepted=accepted,
        crop_proposals=[dict(source_start=final['source_start'], changed=changed,
            states=final['states'], accepted=accepted)] if eligible else [])
    return candidate


def _validate_pair_context(value):
    if not isinstance(value, dict) or set(value) != {'background_window_sha256', 'mixer_rng_sha256'}:
        raise ValueError('paired comparator context hook missing')
    for digest in value.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('paired comparator context digest invalid')


def paired_ledger(records, classes):
    """Count each consumed example once; never infer baselines from final labels."""
    ledger = PolicyLedger(classes)
    for row in records:
        baseline = row.get('paired_baseline')
        if not isinstance(baseline, dict) or baseline.get('protocol') != PAIRING:
            raise ValueError('registered paired comparator missing from training trace')
        for key in ('source_id', 'source_sha256', 'annotation_sha256', 'pair_context', 'level_db'):
            if key not in row or key not in baseline or row[key] != baseline[key]:
                raise ValueError(f'paired comparator {key} mismatch or missing hook')
        _validate_pair_context(row['pair_context'])
        if not baseline.get('audio_sha256'):
            raise ValueError('paired comparator rendered audio digest is required')
        crop = row['crop_proposals']
        masks = row['mask_attempts']
        if len(crop) != int(row['crop_eligible']) or (row['mask_eligible'] and not masks):
            raise ValueError('paired proposal trace missing or inconsistent')
        # A mask proposal completes the already drawn crop candidate. Its first
        # try is not a retry; every subsequent mask candidate is still counted.
        eligible = row['crop_eligible'] or row['mask_eligible']
        augmented = row['crop_accepted'] or row['mask_accepted']
        ledger.record(example_id=row['example_id'], source_id=row['source_id'],
            source_sha256=row['source_sha256'], phase=row['phase'],
            level_bin=json.dumps(row['level_db']), baseline=baseline['states'], final=row['states'],
            eligible=eligible, augmented=augmented, attempts=max(len(crop), len(masks)),
            skipped=eligible and not augmented, details=[dict(crop=crop, masks=masks)])
    return ledger


def require_paired_guardrails(summary):
    """Fail before model export; a software guard pass still cannot promote P6."""
    paired = summary.get('paired_guardrails')
    if not isinstance(paired, dict) or paired.get('status') != 'pass':
        raise ValueError(f'paired 0.80 retention / absence / 0.25 augmentation guard failed: {paired}')


def validate_consumed_records(records, states):
    """The ledger must describe exactly the class cells passed to the loss."""
    actual = _states(states)
    if (actual.ndim != 2 or len(records) != len(actual) or
            any('states' not in row for row in records) or
            not np.array_equal(np.asarray([r['states'] for r in records]), actual)):
        raise ValueError('cue trace disagrees with consumed loss labels')
