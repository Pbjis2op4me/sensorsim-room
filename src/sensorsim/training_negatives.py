# SPDX-License-Identifier: Apache-2.0
"""Opt-in parent-bound negative contexts, without an efficacy claim.

``negative-training/2`` stores source parents, not arbitrary rendered descendants.
Dry parents are rendered on the selected target room draw. Deployed recordings
replace the bed; they are never re-rendered, re-levelled or captured a second time.
The existing mixer owns level, device, clipping and quantization conventions.

Locks and file hashes provide integrity, not independent custody. The caller must
freeze their expected hashes outside the training lane before acceptance. No
procedural-only fallback can satisfy the conditional procedural-share cap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from sensorsim.negatives.composer import audio_hash, digest
from sensorsim.negatives.integration import LockedEventPool, verified_file
from sensorsim.negatives.mining import PoolItem
from sensorsim.negatives.splits import Item, SplitPlan

DEFAULTS = dict(negative_context_policy='legacy', p_confuser_given_target=.5,
                negative_source_cap=.25, negative_family_cap=.5)
POLICY = 'matched/1'


def enabled(cfg):
    policy = cfg.get('negative_context_policy', 'legacy')
    if policy not in ('legacy', POLICY):
        raise ValueError('negative_context_policy must be legacy or matched/1')
    return policy == POLICY


def validate_config(cfg):
    if not enabled(cfg):
        return False
    if (cfg.get('device_split', 'in_distribution') != 'in_distribution' or
            cfg.get('background_domain', 'device_recorded') != 'device_recorded'):
        raise ValueError('matched/1 requires in_distribution device_recorded backgrounds')
    for name in ('p_confuser_given_target', 'p_confuser_given_background', 'max_procedural_share',
                 'negative_source_cap', 'negative_family_cap'):
        value = cfg.get(name, DEFAULTS.get(name))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a finite probability')
    if cfg.get('proxy_screen'):
        raise ValueError('matched/1 requires semantic review, not proxy_screen_only permission')
    return True


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _path(root, value):
    value = _text(value, 'relative path')
    rel = Path(value)
    path = (root / rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or root.resolve() not in path.parents:
        raise ValueError('manifest path must remain inside its root')
    return path


def _json(root, ref):
    return json.loads(verified_file(_path(root, ref['file']), ref['sha256']))


def _review(review, item, *, scope):
    if (review.get('parent') != asdict(item) or review.get('target_scope') != scope or
            review.get('target_free') is not True or review.get('ambiguous') is not False or
            review.get('listened') is not True or review.get('rights_cleared') is not True):
        raise ValueError('parent-bound target-free, unambiguous listening/rights review required')
    for key in ('reviewer', 'notes'):
        _text(review.get(key), key)


def _probabilities(rows, *, procedural_cap, source_cap, family_cap):
    """Cap conditional *emitted* mass; insufficient recorded support is an error.

    Base weights are sampling masses, never loss weights or deployment incidence.
    Descendants keep one parent group and cannot enlarge its aggregate budget.
    """
    weights = np.array([r.provenance.get('weight', 1.) for r in rows], float)
    if not len(weights) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('nonempty finite positive sampling weights required')
    procedural = np.array([r.provenance['provenance'] == 'synthetic' for r in rows])
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError('sampling weight sum must be finite and positive')
    mass = float(weights[procedural].sum()/total)
    if mass > procedural_cap:
        if procedural.all():
            raise ValueError('conditional procedural cap needs eligible recorded confusers; no fallback')
        weights[procedural] *= procedural_cap*(1-mass)/(mass*(1-procedural_cap))
    if np.any(weights == 0):
        raise ValueError('zero-mass filtered entries must be removed before freezing the pool')
    probabilities = weights/weights.sum()
    mined = np.array([r.provenance.get('context_role') == 'training_mining' for r in rows])
    base = float(probabilities[~mined].sum())
    if base <= 0 or probabilities[mined].sum() > .2*base + 1e-12:
        raise ValueError('mined total mass exceeds 0.20 of base mass')
    for field, cap, mined_cap in (('source_group', source_cap, .05), ('family', family_cap, .10)):
        for identity in {getattr(r.item, field) for r in rows}:
            mask = np.array([getattr(r.item, field) == identity for r in rows])
            if probabilities[mask].sum() > cap + 1e-12:
                raise ValueError(f'{field} sampling mass exceeds cap')
            if probabilities[mask & mined].sum() > mined_cap*base + 1e-12:
                raise ValueError(f'mined {field} mass exceeds cap')
    return weights, probabilities


def corpus_room_draws(manifest, corpus):
    """Recover actual target rooms; no direct-field or unknown-room substitution."""
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import SynthConfig
    from sensorsim.training_device import RoomDraw
    result = {}
    for row in corpus.rows:
        if row['label'] == 'background':
            continue
        resolved = row.get('resolved_config', {})
        model = resolved.get('model')
        if row.get('ensemble_member_index') is not None:
            ensemble = json.loads((Path(manifest).parent/'ensemble.json').read_text())
            model = ensemble['members'][row['ensemble_member_index']]
        if model is None:
            raise ValueError('matched/1 needs target room geometry; no direct-field fallback')
        room = RoomModel.model_validate(model)
        key = row.get('room_draw_id', row.get('model_hash'))
        if not key or room.model_hash != row.get('model_hash'):
            raise ValueError('target room draw identity mismatch')
        receiver = tuple(row['receiver_xyz_m'])
        config = SynthConfig.model_validate(resolved.get('synthesis', {}))
        draw = RoomDraw(room, receiver, config, int(row.get('seed', 0)))
        if key in result and (result[key].room != room or result[key].receiver_xyz_m != receiver or result[key].config != config):
            raise ValueError('one room draw ID names incompatible physical paths')
        result[key] = draw
    if not result:
        raise ValueError('nonempty target room draw set required')
    return result


class TrainingPool:
    """Strict source pool; every generated crop/render retains its locked parent."""
    def __init__(self, rows, *, lock, lock_sha256, room_draws, scope,
                 max_procedural_share=.2, source_cap=.25, family_cap=.5, bindings=(), purpose="train",
                 _plan=None):
        from sensorsim.training_negative_lock import PinnedSplitPlan
        if purpose not in ("train", "mine"):
            raise ValueError("pool purpose must be train or mine")
        self.purpose = purpose
        self.lock, self.lock_sha256 = Path(lock), lock_sha256
        self.room_draws, self.scope = dict(room_draws), _text(scope, 'target_scope')
        self.bindings = tuple(bindings)
        self.rows = tuple(rows)
        self._plan = PinnedSplitPlan.open(self.lock, lock_sha256) if _plan is None else _plan
        plan = self.verify()
        for row in self.rows:
            p = row.provenance
            if p.get('context_role') not in ('training_base', 'training_mining'):
                raise ValueError('only separate training-role pools may train or mine')
            if p.get('mined'):
                raise ValueError('legacy development mining is not training-role mining')
            _review(p.get('semantic_review', {}), row.item, scope=scope)
            if p.get('acoustic_stage') not in ('dry', 'device_recorded'):
                raise ValueError('only dry parents or deployed-device contexts are eligible')
            if p['acoustic_stage'] == 'dry':
                if set(p.get('positions', {})) != set(self.room_draws):
                    raise ValueError('every dry parent needs a position on every target room draw')
                for key, position in p['positions'].items():
                    xyz = np.asarray(position, float)
                    if (xyz.shape != (3,) or not np.isfinite(xyz).all() or
                            np.any(xyz < 0) or np.any(xyz > self.room_draws[key].room.shoebox_dims_m)):
                        raise ValueError('finite source position in metres required')
            elif (p.get('deployed_device') is not True or not p.get('profile_id') or
                  set(p.get('room_draw_ids', [])) != set(self.room_draws)):
                raise ValueError('recorded context must match deployed device and every eligible room draw')
            if purpose == 'mine' and (p['context_role'] != 'training_mining' or p['acoustic_stage'] != 'device_recorded'):
                raise ValueError('mining pool must contain only separate recorded training_mining contexts')
            if 'mining_addition' in p and (purpose != 'train' or p['context_role'] != 'training_mining'):
                raise ValueError('mining bank additions require training_mining role in the training pool')
            if purpose == 'train' and p['context_role'] == 'training_mining':
                _validate_addition(p.get('mining_addition', {}), row, self.lock_sha256)
        # Reuse current family realism, content, duplicate and partition gates.
        self.admission = LockedEventPool(self.rows, plan=plan, expected_sha256=plan.sha256,
            purpose='train', max_procedural_share=1.)
        self.fs = self.admission.fs
        for value in (max_procedural_share, source_cap, family_cap):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('sampling caps must lie in [0,1]')
        if purpose == 'train':
            self.weights, self.probabilities = _probabilities(self.rows, procedural_cap=max_procedural_share,
                source_cap=source_cap, family_cap=family_cap)
        else:
            self.weights = np.ones(len(self.rows))
            self.probabilities = self.weights/self.weights.sum()
        self.caps = dict(procedural=max_procedural_share, source=source_cap, family=family_cap)
        # Admission gets an immutable snapshot in the builder. Pin in-memory state
        # too: callers must rebuild, not mutate weights, reviews or source arrays.
        self._state = self._state_digest()

    def _state_digest(self):
        return digest(dict(purpose=self.purpose, fs=self.fs, scope=self.scope, caps=self.caps, rows=[dict(item=asdict(r.item), provenance=r.provenance,
                                     audio=audio_hash(r.audio)) for r in self.rows],
                           probabilities=self.probabilities.tolist(), weights=self.weights.tolist(),
                           rooms={k:dict(model=v.room.model_dump(mode='json'), receiver=v.receiver_xyz_m,
                                         config=v.config.model_dump(mode='json') if v.config else None,
                                         seed=v.seed) for k,v in self.room_draws.items()}))

    def verify(self):
        for path, expected in self.bindings:
            verified_file(path, expected)
        plan = self._plan
        if (str(self.lock.resolve()) != plan._pinned_path or self.lock_sha256 != plan._file_sha256):
            raise ValueError('frozen pool lock changed')
        plan.verify_lock(plan._plan_sha256)
        if hasattr(self, '_state') and self._state_digest() != self._state:
            raise ValueError('frozen pool state changed')
        return plan

    def choose(self, rng):
        if self.purpose != 'train':
            raise ValueError('unreviewed mining pool cannot supply training examples')
        self.verify()
        index = int(np.searchsorted(np.cumsum(self.probabilities), rng.random(), side='right'))
        row = self.rows[min(index, len(self.rows)-1)]
        self.admission._authorize(row)
        return row

    def metadata(self):
        banks = {r.provenance['mining_addition']['bank_sha256']: r.provenance['mining_addition']['bank']
                 for r in self.rows if 'mining_addition' in r.provenance}
        return dict(policy=POLICY, evidence_domain='synthetic proxy', scope=self.scope,
            split_file_sha256=self.lock_sha256, caps=self.caps,
            mining_banks=[dict(bank_sha256=key, **banks[key]) for key in sorted(banks)],
            procedural_cap_basis='conditional emitted confusers on both target and empty routes',
            items=[dict(item_id=r.item.item_id, source_group=r.item.source_group, family=r.item.family,
                        audio_sha256=r.item.sha256, weight=float(w), probability=float(p),
                        provenance=r.provenance['provenance'], context_role=r.provenance['context_role'])
                   for r,w,p in zip(self.rows,self.weights,self.probabilities)],
            families=[dict(family=f, count=sum(r.item.family == f for r in self.rows))
                      for f in sorted({r.item.family for r in self.rows})])


def build_training_pool(path, expected_sha256, *, fs, room_draws, max_procedural_share=.2,
                        source_cap=.25, family_cap=.5, purpose="train"):
    """Read negative-training/2; verify every parent before any training/inference.

    Non-project parents use the existing per-item WP9 gate via load_clip. Original
    procedural sources are regenerated from their recipe and content-checked.
    Every row and semantic review binds the exact parent in the persisted split.
    No acceptance status is inferred from a self-authored manifest or receipt.
    """
    from sensorsim.negatives.clips import load_clip
    from sensorsim.negatives.procedural import generate
    from sensorsim.licensing import require_commercial_lineage
    path = Path(path).resolve()
    data = json.loads(verified_file(path, expected_sha256))
    if data.get('schema') != 'negative-training/2':
        raise ValueError('matched/1 requires negative-training/2')
    root = path.parent
    lock = _path(root, data['split_lock'])
    from sensorsim.training_negative_lock import PinnedSplitPlan
    plan = PinnedSplitPlan.open(lock, data['split_file_sha256'])
    rows, bindings = [], [(path, expected_sha256)]
    def document(ref):
        p = _path(root, ref['file'])
        result = _json(root, ref)
        bindings.append((p, ref['sha256']))
        return result
    scope = _text(data.get('target_scope'), 'target_scope')
    for entry in data['rows']:
        item = Item(**entry['item'])
        plan.require(item, purpose='train')  # before opening any source/review
        require_commercial_lineage(entry, context='training negative parent')
        review = document(entry['semantic_review'])
        _review(review, item, scope=scope)
        source_spec = entry['source']
        if source_spec['kind'] == 'procedural':
            source = generate(fs=fs, **source_spec['recipe'])
            provenance = dict(source.provenance)
        elif source_spec['kind'] == 'recorded':
            register = _path(root, source_spec['register']['file'])
            verified_file(register, source_spec['register']['sha256'])
            bindings.append((register, source_spec['register']['sha256']))
            source = load_clip(root, source_spec['asset'], register_path=register,
                               review=source_spec['review'], fs=fs)
            if source.provenance['original_fs'] != fs:
                raise ValueError('freeze sources at model sample rate; implicit resampling is forbidden')
            bindings.append((_path(root, source_spec['asset']['path']), source_spec['asset']['sha256']))
            provenance = dict(source.provenance)
            if provenance['acoustic_stage'] == 'captured':
                provenance['acoustic_stage'] = 'device_recorded'
            provenance['family'] = source.family
            if source_spec['review']['source_group'] != item.source_group:
                raise ValueError('recorded source group differs from locked parent')
        else:
            raise ValueError('source kind must be procedural or recorded')
        if source.family != item.family or audio_hash(source.audio) != item.sha256:
            raise ValueError('source content or family differs from locked parent')
        provenance.update(training_eligible=True, semantic_review=review, weight=entry.get('weight', 1.),
                          context_role=entry['context_role'], whole_context=entry.get('whole_context', False),
                          target_free=review['target_free'], positions=entry.get('positions', {}),
                          room_draw_ids=entry.get('room_draw_ids', []))
        for field in ('realism_gate', 'doorbell_recall'):
            if field in entry:
                value = document(entry[field])
                for key in ('evidence_file', 'reference_file', 'file'):
                    if key in value:
                        value[key] = str(_path(root, value[key]))
                        hash_key = key.replace('_file', '_sha256') if key != 'file' else 'sha256'
                        bindings.append((Path(value[key]), value[hash_key]))
                provenance[field] = value
        if 'mining_addition' in entry:
            addition = document(entry['mining_addition'])
            provenance['mining_addition'] = addition
            provenance['weight'] = addition['weight']
        x = np.array(source.audio, dtype=float, copy=True)
        x.setflags(write=False)
        rows.append(PoolItem(item, x, fs, provenance))
    return TrainingPool(rows, lock=lock, lock_sha256=data['split_file_sha256'], room_draws=room_draws,
        scope=scope, max_procedural_share=max_procedural_share, source_cap=source_cap,
        family_cap=family_cap, bindings=bindings, purpose=purpose, _plan=plan)


@dataclass(frozen=True)
class ContextDraw:
    event: object | None = None
    background: np.ndarray | None = None
    provenance: dict | None = None


def prepare_context(corpus, rng, cfg, n, *, class_name=None, target_index=None):
    """Draw once before dynamics retries; nuisance probability is label-conditional.

    Ordinary backgrounds retain mass 1-p. All pool entries support all target room
    draws, so conditioning on a room never silently changes the sampling caps.
    """
    from sensorsim.training_dynamics import auxiliary_rng
    from sensorsim.training_device import ReceiverEvent
    from sensorsim.recognizer import _event_window, _crop
    validate_config(cfg)
    pool = getattr(corpus, 'negative_pool', None)
    if not isinstance(pool, TrainingPool) or corpus.pool_purpose != 'train':
        raise ValueError('matched/1 needs a strict training pool; validation is not training')
    if pool.fs != cfg['fs']:
        raise ValueError('pool sample rate differs from trainer units')
    targets = [r for r in corpus.rows if r['label'] != 'background']
    choose = auxiliary_rng(rng, 'matched-confuser')
    if class_name is None:
        if not targets:
            raise ValueError('target room rows missing')
        room_class = corpus.classes[int(choose.integers(len(corpus.classes)))]
        room_rows = [r for r in targets if r['label'] == room_class]
        reference = room_rows[int(choose.integers(len(room_rows)))]
        probability = cfg['p_confuser_given_background']
    else:
        reference = [r for r in targets if r['label'] == class_name][target_index]
        probability = cfg['p_confuser_given_target']
    room_id = reference.get('room_draw_id', reference.get('model_hash'))
    if room_id not in pool.room_draws:
        raise ValueError('selected target room is absent from locked pool')
    provenance = dict(policy=POLICY, room_draw_id=room_id, split_file_sha256=pool.lock_sha256,
                      target_present=class_name is not None, injected=False)
    if class_name is not None:
        provenance['target_parent'] = dict(source_group=reference.get('source_group') or
            reference.get('dry_source', {}).get('source_group'),
            audio_sha256=audio_hash(corpus.targets[class_name][target_index]),
            dry_source=reference.get('dry_source', {}))
    if choose.random() >= probability:
        pool.verify()
        return ContextDraw(provenance=provenance)
    row = pool.choose(choose)
    provenance.update(injected=True, parent=asdict(row.item), context_role=row.provenance['context_role'])
    if row.provenance['acoustic_stage'] == 'device_recorded':
        if row.provenance['profile_id'] != cfg['device_profile']:
            raise ValueError('confuser capture profile differs from target device')
        if len(row.audio) < 2*n:
            raise ValueError('captured context shorter than warmup plus training window')
        # One existing context/floor replaces the bed, rather than summing floors.
        bed = _crop(row.audio, 2*n, choose).astype(float)
        provenance.update(route='deployed context; no room or device recapture', audio_sha256=audio_hash(bed))
        return ContextDraw(background=bed, provenance=provenance)
    draw = pool.room_draws[room_id]
    # Same RIR seed convention and convolution as render_room_event, but keep the
    # whole parent before taking a window: never guess the RIR duration.
    from sensorsim.training_context_cache import render_parent
    wet_audio, wet_span, rir_sha256, rir_seed = render_parent(row.audio, draw=draw,
        position=row.provenance['positions'][room_id], fs=cfg['fs'])
    wet = ReceiverEvent(wet_audio, metadata=dict(parent=asdict(row.item),
        split_file_sha256=pool.lock_sha256, room_hash=draw.room.model_hash,
        receiver_xyz_m=list(draw.receiver_xyz_m), source_xyz_m=row.provenance['positions'][room_id],
        rir_sha256=rir_sha256, rir_seed=rir_seed))
    window, (a,b) = _event_window(wet.audio, wet_span, n, choose)
    level = float(choose.uniform(*cfg['event_spl_db']))
    event = ReceiverEvent(np.r_[np.zeros(n), window], (n+a,n+b), level, wet.metadata)
    provenance.update(route='dry parent through target room/device', render=wet.metadata,
                      audio_sha256=audio_hash(event.audio))
    return ContextDraw(event=event, provenance=provenance)


def draw_training_context(background, target, rng, cfg, *, draw, counterfactual=False):
    """Compose target/confuser/empty through one shared device capture.

    Explicit confuser SPL draws do not shift target-level/device randomness. The
    target-free counterfactual includes the nuisance for audibility supervision.
    This route-equivalence contract is not a distributional-realism oracle.
    """
    from sensorsim.training_device import ReceiverEvent, mix_events
    from sensorsim.training_dynamics import clone_rng
    bed = background if draw.background is None else draw.background
    events = [target] + ([] if draw.event is None else [draw.event])
    replay = clone_rng(rng)
    result = mix_events(bed, events, rng, cfg)
    floor = None
    if counterfactual:
        empty = ReceiverEvent(np.zeros_like(target.audio), target.span, target.spl_db)
        floor = mix_events(bed, [empty]+events[1:], replay, cfg).audio
    if draw.provenance is not None:
        draw.provenance.update(output_audio_sha256=audio_hash(result.audio),
                               background_audio_sha256=audio_hash(bed))
    return result.audio, floor


def _candidate_identity(candidate):
    return digest({k:v for k,v in candidate.items() if k != 'review_id'})


def mine_context_candidates(pool, *, model, detector, max_total=1000):
    """Mine whole device contexts from a separate TRAIN-role pool, once per parent.

    Uses the actual edge Tracker including start alerts and final flush. Windows
    score the detector, but the review unit is the full original context. Never
    uses threshold-development, calibration or audit audio. No candidates is a
    recorded empty result, not a claim of detector performance.
    """
    from sensorsim.edge import Tracker
    if isinstance(max_total, bool) or not isinstance(max_total, int) or max_total < 1:
        raise ValueError('positive candidate budget required')
    if not isinstance(pool, TrainingPool) or pool.purpose != 'mine':
        raise ValueError('a separately manifested eligible mining pool is required')
    pool.verify()
    rows = pool.rows
    lock, lock_sha256 = pool.lock, pool.lock_sha256
    if not rows:
        raise ValueError('separate nonempty training_mining pool required')
    plan = SplitPlan.from_audit_lock(lock, expected_file_sha256=lock_sha256)
    hop_s = detector.get('hop_s', 0.)
    if not np.isfinite(hop_s) or not 1 <= round(hop_s*model.fs) <= model.n:
        raise ValueError('detector hop must be in model sample units')
    for key in ('on','off_ratio','smooth_hops','min_hops'):
        vals = detector.get(key)
        vals = [vals.get(c) for c in model.classes] if isinstance(vals, dict) else [vals]
        for v in vals:
            if not isinstance(v, (float,int)) or not np.isfinite(v):
                raise ValueError('complete finite detector policy required')
            if key in ('on','off_ratio') and not 0 < v <= 1:
                raise ValueError('detector thresholds must lie in (0,1]')
            if key in ('smooth_hops','min_hops') and (v < 1 or int(v) != v):
                raise ValueError('detector persistence must use positive integer hops')
    seen, groups = set(), set()
    for row in rows:
        plan.require(row.item, purpose='train')
        p = row.provenance
        if p.get('context_role') != 'training_mining' or p.get('acoustic_stage') != 'device_recorded':
            raise ValueError('only whole device-recorded training_mining contexts may be mined')
        if p.get('whole_context') is not True or p.get('target_free') is not True:
            raise ValueError('whole target-free context declaration required')
        if row.fs != model.fs or len(row.audio) < model.n or np.asarray(row.audio).ndim != 1:
            raise ValueError('context rate/shape/length incompatible with model')
        if not np.isfinite(row.audio).all() or np.ptp(row.audio) == 0 or audio_hash(row.audio) != row.item.sha256:
            raise ValueError('context content changed, constant or nonfinite')
        if row.item.sha256 in seen or row.item.source_group in groups:
            raise ValueError('duplicate context content or independent source parent')
        seen.add(row.item.sha256); groups.add(row.item.source_group)
    model.verify()
    output = []
    hop = round(hop_s*model.fs)
    for row in rows:
        tracker = Tracker(list(model.classes), detector, model.n/model.fs)
        events = []
        for start in range(0, len(row.audio)-model.n+1, hop):
            scores = np.asarray(model.scorer(np.array(row.audio[start:start+model.n], copy=True)), float).reshape(-1)
            if scores.shape != (len(model.classes),) or not np.isfinite(scores).all() or np.any((scores < 0)|(scores > 1)):
                raise ValueError('scorer must return finite per-class probabilities')
            events.extend(tracker.step(scores, (start+model.n)/model.fs))
        events.extend(tracker.flush())
        starts = sum(e['event'] == 'start' for e in events)
        confirmed = sum(e['event'] == 'end' and e['confirmed'] for e in events)
        if starts:
            candidate = dict(schema='negative-context-candidate/1', parent=asdict(row.item),
                fs=row.fs, sample_range=[0,len(row.audio)], units='samples',
                model_sha256=model.sha256, detector=detector, detector_sha256=digest(detector),
                split_file_sha256=lock_sha256, starts=starts, confirmed=confirmed, events=events)
            output.append(dict(candidate, review_id=_candidate_identity(candidate)))
    model.verify()
    pool.verify()
    SplitPlan.from_audit_lock(lock, expected_file_sha256=lock_sha256)
    return tuple(sorted(output, key=lambda c:(-c['starts'],-c['confirmed'],c['review_id']))[:max_total])


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _mining_binding(candidate, lock_sha256):
    """Identity shared by every candidate in one frozen mining bank."""
    if (candidate.get('schema') != 'negative-context-candidate/1' or
            candidate.get('review_id') != _candidate_identity(candidate) or
            candidate.get('split_file_sha256') != lock_sha256):
        raise ValueError('mining candidate or bank split identity changed')
    if not _sha256(candidate.get('model_sha256')):
        raise ValueError('mining bank requires a valid model checkpoint SHA-256')
    detector = candidate.get('detector')
    if (not isinstance(detector, dict) or
            not {'hop_s', 'on', 'off_ratio', 'smooth_hops', 'min_hops'} <= detector.keys() or
            candidate.get('detector_sha256') != digest(detector)):
        raise ValueError('mining bank requires a complete frozen detector policy')
    return dict(model_sha256=candidate['model_sha256'], detector=detector,
                detector_sha256=candidate['detector_sha256'], split_file_sha256=lock_sha256)


def _validate_bank_addition(addition, lock_sha256):
    binding = _mining_binding(addition.get('candidate', {}), lock_sha256)
    bank = addition.get('bank')
    if (not isinstance(bank, dict) or bank.get('schema') != 'negative-context-bank/1' or
            addition.get('bank_sha256') != digest(bank) or
            any(bank.get(key) != value for key, value in binding.items())):
        raise ValueError('mining bank checkpoint, detector, split or digest binding changed')
    members = bank.get('candidate_review_ids')
    if (not isinstance(members, list) or not members or not all(_sha256(m) for m in members) or
            members != sorted(set(members)) or addition['candidate']['review_id'] not in members):
        raise ValueError('mining candidate is not a member of the frozen bank')


def _validate_addition(addition, row, lock_sha256):
    _validate_bank_addition(addition, lock_sha256)
    candidate = addition.get('candidate', {})
    review = addition.get('review', {})
    if (candidate.get('schema') != 'negative-context-candidate/1' or
            candidate.get('review_id') != _candidate_identity(candidate) or
            candidate.get('parent') != asdict(row.item) or candidate.get('split_file_sha256') != lock_sha256 or
            candidate.get('sample_range') != [0,len(row.audio)] or candidate.get('fs') != row.fs or
            candidate.get('units') != 'samples' or candidate.get('detector_sha256') != digest(candidate.get('detector'))):
        raise ValueError('mined whole-context parent, detector, units or lock identity changed')
    if (review.get('review_id') != candidate['review_id'] or review.get('decision') != 'confirmed_negative' or
            review.get('ambiguous') is not False or review.get('listened') is not True):
        raise ValueError('mined context needs explicit unambiguous listening review')
    _text(review.get('reviewer'), 'reviewer'); _text(review.get('notes'), 'notes')
    if addition.get('weight') != row.provenance.get('weight'):
        raise ValueError('mined sampling weight changed')


def reviewed_context_bank(candidates, reviews, *, base_weight, lock, lock_sha256):
    """Return capped additions bound to one checkpoint, detector and locked split.

    Each persisted addition carries the same bank descriptor and its digest, so
    training can recheck membership and identity without an implicit sidecar.
    These integrity checks do not confer independent custody or listening review.
    An empty or fully filtered result remains an explicit empty tuple.
    """
    plan = SplitPlan.from_audit_lock(lock, expected_file_sha256=lock_sha256)
    if not np.isfinite(base_weight) or base_weight <= 0:
        raise ValueError('positive base sampling mass required')
    # Snapshot nested policies and reviews; changing a caller's dictionaries must
    # not silently alter an issued candidate, listening decision or bank identity.
    candidates = tuple(json.loads(json.dumps(tuple(candidates), allow_nan=False)))
    reviews = json.loads(json.dumps(reviews, allow_nan=False))
    ids = [c['review_id'] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate mining candidates')
    approved = []
    binding = None
    for candidate in candidates:
        plan.require(Item(**candidate['parent']), purpose='train')
        current = _mining_binding(candidate, lock_sha256)
        if binding is not None and current != binding:
            raise ValueError('one model checkpoint and frozen detector policy required per mining bank')
        binding = current
        review = reviews.get(candidate['review_id'], {})
        if review.get('decision') == 'confirmed_negative':
            if review.get('ambiguous') is not False or review.get('listened') is not True:
                raise ValueError('ambiguous or unheard candidate cannot become a negative')
            _text(review.get('reviewer'), 'reviewer'); _text(review.get('notes'), 'notes')
            approved.append((candidate,review))
    if not approved:
        return ()
    bank = dict(schema='negative-context-bank/1', **binding,
                candidate_review_ids=sorted(c['review_id'] for c, _ in approved))
    bank_sha256 = digest(bank)
    result = []
    for candidate,review in approved:
        parent = candidate['parent']
        by_source = sum(c['parent']['source_group'] == parent['source_group'] for c,_ in approved)
        by_family = sum(c['parent']['family'] == parent['family'] for c,_ in approved)
        weight = base_weight*min(.2/len(approved), .05/by_source, .1/by_family)
        result.append(dict(candidate=candidate,review=dict(review,review_id=candidate['review_id']),weight=weight,
                           bank=json.loads(json.dumps(bank)), bank_sha256=bank_sha256))
    return tuple(result)


def validate_overlay_sources(events, *, lock, lock_sha256, purpose, parent_items, scope, reviews,
                             source_records, root):
    """WP2 adapter: validate original dry parents before its existing overlay path.

    An audit caller must supply the custodian's audit lock and parents. Training
    authority cannot be relabelled as audit. Overlay descendants remain grouped
    by the original dry parents and by the background recording in WP2.
    """
    plan = SplitPlan.from_audit_lock(lock, expected_file_sha256=lock_sha256)
    from sensorsim.licensing import require_commercial_lineage
    from sensorsim.negatives.clips import load_clip
    from sensorsim.negatives.procedural import generate
    root = Path(root).resolve()
    events, parent_items, reviews = tuple(events), tuple(parent_items), tuple(reviews)
    source_records = tuple(source_records)
    if (not events or len(events) != len(parent_items) or len(events) != len(reviews)
            or len(events) != len(source_records)):
        raise ValueError('one locked parent and semantic review per overlay event required')
    for event,item,review,record in zip(events,parent_items,reviews,source_records):
        plan.require(item,purpose=purpose)
        require_commercial_lineage(record, event.source.provenance, context='negative overlay parent')
        fs = event.source.provenance['fs']
        if 'procedural' in record:
            original = generate(fs=fs, **record['procedural'])
        else:
            clearance = record.get('clearance')
            if not clearance:
                raise ValueError('overlay recording requires per-item licence clearance')
            register = _path(root, clearance['register']['file'])
            verified_file(register, clearance['register']['sha256'])
            original = load_clip(root, clearance['asset'], register_path=register,
                                 review=clearance['review'], fs=fs)
            if original.provenance['review']['source_group'] != item.source_group:
                raise ValueError('overlay source group differs from cleared parent')
        if original.family != item.family or audio_hash(original.audio) != item.sha256:
            raise ValueError('overlay licence or procedural recipe does not bind this parent')
        _review(review,item,scope=scope)
        if event.source.family != item.family or audio_hash(event.source.audio) != item.sha256:
            raise ValueError('overlay source differs from locked parent')
        if event.source.provenance.get('acoustic_stage') != 'dry':
            raise ValueError('recorded device contexts cannot be room-rendered as dry overlays')
    return dict(split_file_sha256=lock_sha256, parents=[asdict(p) for p in parent_items], purpose=purpose)


def context_counts(examples):
    """Report target-present and target-free denominators separately, including unknown labels."""
    contexts = [e[2]['negative_context'] for e in examples]
    target = sum(c['target_present'] for c in contexts)
    target_confuser = sum(c['target_present'] and c['injected'] for c in contexts)
    background_confuser = sum(not c['target_present'] and c['injected'] for c in contexts)
    return dict(windows=len(contexts), background_windows=len(contexts)-target,
                confuser_windows=target_confuser+background_confuser, target_windows=target,
                target_confuser_windows=target_confuser, background_confuser_windows=background_confuser)
