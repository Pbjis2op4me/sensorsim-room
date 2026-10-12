# SPDX-License-Identifier: Apache-2.0
"""Role-bound continuous banks and complete-detector selection (P0/P2).

The evaluation protocol owns allocation and custody. This module verifies a
frozen allocation; it neither invents a split nor opens audit audio for search.
Manifests contain identities for *all* roles before development scoring. Audit
rows need no accessible files until explicitly loaded for a frozen outer screen.
No supplied identity inventory proves independent acquisition or honest lineage.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import copy
import json
from pathlib import Path
import re
import wave

import numpy as np
from scipy.stats import binom, chi2

from sensorsim import edge
from .intervals import clopper_pearson
from .operating_point import match_events, replay_policy, validate_policy

SCHEMA = 'operational-bank/1'
DOMAINS = ('synthetic proxy', 'measured-RIR proxy', 'deployment recording')
ROLES = ('train', 'development', 'audit')
PANELS = ('nominal', 'diagnostic', 'playback', 'stress')


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def audio_digest(audio):
    return sha256(np.ascontiguousarray(audio, dtype='<f4').tobytes()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('lowercase SHA-256 identity required')
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('nonempty identity required')
    return value


def _family_allocation(data):
    """Separate mandatory support floors from explicitly allowed extra families."""
    allocation = {}
    for key in ('required_families', 'optional_families'):
        declared = data.get(key, [])
        if isinstance(declared, dict):
            if set(declared) != set(ROLES):
                raise ValueError('family allocation must name every role')
            per_role = declared
        else:
            per_role = dict.fromkeys(ROLES, declared)
        for values in per_role.values():
            if not isinstance(values, list) or any(not isinstance(f, str) or not f.strip() for f in values):
                raise ValueError('family allocation requires lists of nonempty names')
            if len(set(values)) != len(values) or (key == 'required_families' and not values):
                raise ValueError('unique families and nonempty required family lists required')
        allocation[key] = per_role
    required, optional = allocation['required_families'], allocation['optional_families']
    if any(set(required[r]) & set(optional[r]) for r in ROLES):
        raise ValueError('required and optional family allocations must be disjoint per role')
    return required, optional


@dataclass(frozen=True)
class ValidationBank:
    manifest: dict
    manifest_sha256: str
    content_sha256: str
    role: str
    rows: tuple
    audio: tuple
    bindings: tuple = ()

    def verify(self):
        for path, expected in self.bindings:
            if sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('context schema or split dependency changed')
        if digest(self.manifest) != self.content_sha256:
            raise ValueError('bank manifest changed after loading')
        selected = [r for r in self.manifest['rows'] if r['role']==self.role]
        if digest(selected) != digest(list(self.rows)) or len(self.audio) != len(self.rows):
            raise ValueError('bank rows changed or were filtered')
        for row, x in zip(self.rows, self.audio):
            if audio_digest(x) != row['audio_sha256']:
                raise ValueError('bank audio changed after loading')


def build_validation_bank(manifest, *, expected_sha256, role='development', root=None):
    """Load a frozen JSON allocation and only the requested role's whole audio.

    Each row binds: id, role, kind (target/nuisance/background), parent_id,
    session_id, lineage_ids (including mixed-in source/background ancestry),
    parent_audio_sha256 (including original/normalized identities), audio,
    file_sha256, audio_sha256 (decoded float32 LE), sample_rate_hz, sample_count,
    route, evidence_domain, panel and annotations. Nuisances also bind family.
    Only nominal rows enter the primary gates; other panels remain diagnostics.
    All rows, including diagnostic descendants, participate in overlap checks.

    Decoded hashes catch renaming/lossless re-encoding; lossy transforms require
    retained original lineage/hashes. This is not an acoustic near-duplicate
    detector. Missing ancestry cannot be repaired from filenames.
    """
    if role not in ('development','audit'):
        raise ValueError('only development or explicitly opened audit banks can be scored')
    path = Path(manifest)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != _sha(expected_sha256):
        raise ValueError('bank manifest digest mismatch')
    data = json.loads(raw)
    if data.get('schema') != SCHEMA or data.get('allocation_frozen') is not True:
        raise ValueError('frozen operational-bank/1 allocation required')
    classes = data.get('classes', [])
    required, optional = _family_allocation(data)
    if not classes or len(set(classes))!=len(classes):
        raise ValueError('nonempty unique required classes and families required')
    for name in classes:
        _text(name)
    rows = data.get('rows', [])
    if not rows or {r.get('role') for r in rows} != set(ROLES):
        raise ValueError('allocate train, development and sealed audit before selection')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate context identity')
    identities, content, nominal_targets, families_by_parent = {}, {}, set(), {}
    bindings, training_families = [], set()
    # Consume lane A's versioned source/context inventory without importing or
    # editing its implementation. Training/mining ancestry remains excluded.
    if data.get('negative_training') is not None:
        ref = data['negative_training']
        rel = Path(ref['file'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('negative context inventory must be relative')
        source = Path(root or path.parent)/rel
        source_raw = source.read_bytes()
        if sha256(source_raw).hexdigest() != _sha(ref['sha256']):
            raise ValueError('negative context inventory digest mismatch')
        inventory = json.loads(source_raw)
        if inventory.get('schema') != 'negative-training/2':
            raise ValueError('lane A negative-training/2 context schema required')
        _text(inventory.get('target_scope'))
        if not isinstance(inventory.get('rows'),list) or not inventory['rows']:
            raise ValueError('nonempty lane A context inventory required')
        from sensorsim.negatives.splits import Item, SplitPlan
        lock_rel = Path(inventory['split_lock'])
        if lock_rel.is_absolute() or '..' in lock_rel.parts:
            raise ValueError('negative split lock must be relative')
        lock = source.parent/lock_rel
        plan = SplitPlan.from_audit_lock(lock, expected_file_sha256=inventory['split_file_sha256'])
        bindings.extend(((str(source), ref['sha256']), (str(lock), inventory['split_file_sha256'])))
        for entry in inventory['rows']:
            if not isinstance(entry.get('source'),dict) or not isinstance(entry.get('semantic_review'),dict):
                raise ValueError('lane A source and semantic-review references required')
            _text(entry['semantic_review'].get('file'))
            _sha(entry['semantic_review'].get('sha256'))
            if entry.get('context_role') not in ('training_base', 'training_mining'):
                raise ValueError('unknown lane A context role')
            item = Item(**entry['item'])
            plan.require(item, purpose='train')
            training_families.add(item.family)
            identities[item.source_group] = identities[item.item_id] = 'train'
            content[item.sha256] = 'train'
    selected, audio = [], []
    for row in rows:
        for key in ('id','parent_id','session_id','route'):
            _text(row.get(key))
        if row.get('kind') not in ('target','nuisance','background') or row.get('panel') not in PANELS:
            raise ValueError('explicit kind and panel required')
        if row.get('evidence_domain') not in DOMAINS:
            raise ValueError('explicit evidence domain required')
        stage = row.get('acoustic_stage')
        if stage not in ('dry_target', 'known_playback', 'deployed_device_background',
                         'room_response', 'rendered_context', 'actual_source_capture'):
            raise ValueError('explicit recording acoustic_stage required')
        if row['role'] == role and stage in ('dry_target', 'room_response'):
            raise ValueError('dry targets and room responses are not continuous evaluation contexts')
        if stage == 'rendered_context' and row['evidence_domain'] == 'deployment recording':
            raise ValueError('rendered proxy contexts cannot be deployment recordings')
        if stage == 'known_playback' and row['panel'] != 'playback':
            raise ValueError('playback comparisons require their separate diagnostic panel')
        if (not isinstance(row.get('lineage_ids'),list) or not row['lineage_ids']
                or not isinstance(row.get('parent_audio_sha256'),list) or not row['parent_audio_sha256']):
            raise ValueError('complete source/background lineage and parent content identities required')
        for ident in [row['parent_id'],row['session_id'],*row['lineage_ids']]:
            _text(ident)
            if ident in identities and identities[ident] != row['role']:
                raise ValueError('parent/session/lineage crosses roles')
            identities[ident] = row['role']
        for h in [row.get('audio_sha256'),row.get('file_sha256'),*row['parent_audio_sha256']]:
            _sha(h)
            if h in content and content[h] != row['role']:
                raise ValueError('duplicate content crosses roles')
            content[h] = row['role']
        fs, count = row.get('sample_rate_hz'), row.get('sample_count')
        if any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in (fs,count)):
            raise ValueError('sample rate and extent require positive integer sample units')
        annotations = row.get('annotations')
        if not isinstance(annotations,list):
            raise ValueError('explicit annotation list required')
        if row['kind']=='target':
            if not annotations:
                raise ValueError('target occurrences must be annotated')
            for a in annotations:
                if a.get('label') not in classes or not np.isfinite([a['start_s'],a['end_s']]).all() or not 0 <= a['start_s'] < a['end_s'] <= count/fs:
                    raise ValueError('invalid target annotation in seconds')
            if row['panel']=='nominal':
                key=(row['role'],row['parent_id'])
                if key in nominal_targets:
                    raise ValueError('only one frozen nominal target context per source parent')
                nominal_targets.add(key)
        elif annotations:
            raise ValueError('target-free rows cannot contain target annotations')
        if row['kind']=='nuisance':
            if row['role'] != 'train' and row.get('family') in training_families:
                raise ValueError('training confuser family cannot enter a held-out bank')
            if row.get('family') not in required[row['role']] + optional[row['role']]:
                raise ValueError('nuisance family is not allocated to its own role')
            parent=row['parent_id']
            if parent in families_by_parent and families_by_parent[parent] != row['family']:
                raise ValueError('one parent cannot manufacture multiple independent families')
            families_by_parent[parent]=row['family']
        if row['role'] != role:
            continue
        # Repeated contexts for one nuisance parent are allowed, but cloned
        # waveforms cannot manufacture additional parents or background exposure.
        for old in selected:
            if old['audio_sha256']==row['audio_sha256']:
                raise ValueError('duplicate decoded context content')
            if set(old['parent_audio_sha256']) & set(row['parent_audio_sha256']) and old['parent_id'] != row['parent_id']:
                raise ValueError('duplicate parent content under different parent identities')
            if row['kind']=='background' and old['kind']=='background' and row['session_id']==old['session_id']:
                raise ValueError('background exposure must use one complete recording per session')
        audio_path = Path(row['audio'])
        if audio_path.is_absolute() or '..' in audio_path.parts:
            raise ValueError('bank audio paths must be relative to the manifest root')
        audio_path = Path(root or path.parent)/audio_path
        if sha256(audio_path.read_bytes()).hexdigest() != row['file_sha256']:
            raise ValueError('audio file digest mismatch')
        with wave.open(str(audio_path)) as wav:
            if wav.getframerate()!=fs or wav.getnchannels()!=1 or wav.getnframes()!=count:
                raise ValueError('audio sample rate/channel/extent mismatch')
        x = edge.read_wav(str(audio_path), fs)
        if not len(x) or not np.isfinite(x).all() or np.ptp(x)==0:
            raise ValueError('empty, nonfinite or constant audio is not exposure evidence')
        if audio_digest(x) != row['audio_sha256']:
            raise ValueError('decoded audio digest mismatch')
        x.setflags(write=False)
        selected.append(copy.deepcopy(row)); audio.append(x)
    if not selected:
        raise ValueError('empty selected role bank')
    bank = ValidationBank(data, expected_sha256, digest(data), role, tuple(selected), tuple(audio), tuple(bindings))
    bank.verify()
    return bank


def candidate_identity(*, model_sha256, features, classes, policy, step, seed, code_revision, model_path=None):
    """Bind model bytes, features, class order, tracker and training identity."""
    _sha(model_sha256); _text(code_revision)
    if any(isinstance(v,bool) or not isinstance(v,int) or v < 0 for v in (step,seed)):
        raise ValueError('nonnegative integer checkpoint and seed required')
    features = copy.deepcopy(features)
    cfg = validate_policy(classes, policy, window_s=features['window_s'], sample_rate_hz=features['fs'])
    identity = dict(schema='operational-candidate/1', model_sha256=model_sha256,
        features=features, features_sha256=digest(features), classes=list(classes),
        policy=cfg, step=step, seed=seed, code_revision=code_revision)
    identity['candidate_sha256'] = digest(identity)
    if model_path is not None:
        identity['model_path'] = str(model_path)
    return identity


def _verify_candidate(candidate):
    payload = {k:v for k,v in candidate.items() if k not in ('candidate_sha256','model_path')}
    if digest(payload) != candidate['candidate_sha256'] or digest(candidate['features']) != candidate['features_sha256']:
        raise ValueError('candidate identity changed')
    if candidate.get('model_path') and sha256(Path(candidate['model_path']).read_bytes()).hexdigest() != candidate['model_sha256']:
        raise ValueError('actual model file changed after candidate creation')


def evaluate_checkpoint_selector(callback, checkpoint, *, step, seed):
    """Bind a callback to the emitted artifact before allowing trainer selection.

    Capture bytes, full features and class order *before* invoking external
    code, and check bytes even if no feasible candidate is returned. This guards
    callback identity/policy routing; it is not an oracle for reported efficacy.
    """
    path = Path(checkpoint)
    emitted_sha256 = sha256(path.read_bytes()).hexdigest()
    model = edge.Model(str(path))
    features, classes = copy.deepcopy(model.meta['features']), list(model.classes)
    if sha256(path.read_bytes()).hexdigest() != emitted_sha256:
        raise ValueError('checkpoint changed while capturing emitted identity')
    result = copy.deepcopy(callback(path, step))
    if sha256(path.read_bytes()).hexdigest() != emitted_sha256:
        raise ValueError('checkpoint callback changed the emitted artifact')
    if not isinstance(result, dict) or 'selected' not in result:
        raise ValueError('checkpoint callback must return an explicit selection result')
    selected = result['selected']
    if selected is None:
        if result.get('status') not in ('infeasible', 'insufficient_evidence'):
            raise ValueError('missing candidate cannot be feasible')
        return result
    try:
        identity = selected['candidate']
        _verify_candidate(identity)
        canonical = candidate_identity(model_sha256=identity['model_sha256'],
            features=identity['features'], classes=identity['classes'], policy=identity['policy'],
            step=identity['step'], seed=identity['seed'], code_revision=identity['code_revision'])
        public = {k: v for k, v in identity.items() if k != 'model_path'}
        if (public != canonical or identity['model_sha256'] != emitted_sha256
                or identity['features'] != features or identity['features_sha256'] != digest(features)
                or identity['classes'] != classes or identity['step'] != step or identity['seed'] != seed
                or selected['role'] != 'development' or selected['status'] != 'feasible'
                or result.get('status') != 'feasible'):
            raise ValueError('checkpoint callback returned an unbound candidate')
    except (KeyError, TypeError) as exc:
        raise ValueError('checkpoint callback returned an incomplete candidate') from exc
    selected['candidate'] = canonical
    return result


def cache_scores(candidate, bank, sequences, *, origin='synthetic_control'):
    """Bind complete cached scores to model/features and every context.

    The synthetic-control route is explicit and may not acquire a measured
    evidence label. Production callers should use infer_scores, which verifies
    the actual artifact and computes scores through edge.Model/window_scores.
    """
    _verify_candidate(candidate); bank.verify()
    if origin not in ('synthetic_control','edge.window_scores'):
        raise ValueError('unknown score provenance')
    if origin=='synthetic_control' and any(r['evidence_domain']!='synthetic proxy' for r in bank.rows):
        raise ValueError('synthetic score controls cannot stand in for measured banks')
    if set(sequences) != {r['id'] for r in bank.rows}:
        raise ValueError('missing or extra score contexts; filtering is forbidden')
    arrays={k:np.array(v,dtype='<f8',copy=True) for k,v in sequences.items()}
    for x in arrays.values():
        x.setflags(write=False)
    return dict(model_sha256=candidate['model_sha256'], features_sha256=candidate['features_sha256'],
                classes=candidate['classes'], bank_sha256=bank.manifest_sha256, role=bank.role,
                hop_s=candidate['policy']['hop_s'],
                origin=origin, sequences=arrays,
                score_sha256={k:sha256(x.tobytes()).hexdigest() for k,x in arrays.items()})


def infer_scores(candidate, bank):
    _verify_candidate(candidate); bank.verify()
    path = candidate.get('model_path')
    if not path or sha256(Path(path).read_bytes()).hexdigest() != candidate['model_sha256']:
        raise ValueError('actual model file and digest required; no inference fallback')
    model=edge.Model(str(path))
    if model.classes != candidate['classes'] or digest(model.meta['features'])!=candidate['features_sha256']:
        raise ValueError('imported model features/classes differ from candidate')
    fs=candidate['features']['fs']
    if any(r['sample_rate_hz']!=fs for r in bank.rows):
        raise ValueError('bank/model sample-rate units differ; no silent resampling')
    sequences={r['id']:edge.window_scores(model,x,candidate['policy']['hop_s']) for r,x in zip(bank.rows,bank.audio)}
    return cache_scores(candidate,bank,sequences,origin='edge.window_scores')


def _constraints(value):
    required={'min_parent_recall','max_family_trigger_fraction','min_target_parents',
              'min_family_parents','min_background_hours','max_background_starts','max_background_confirmed'}
    if set(value) != required:
        raise ValueError('complete explicit operating constraints required')
    for k,v in value.items():
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v < 0:
            raise ValueError('finite nonnegative constraints required')
        if k in ('min_parent_recall','max_family_trigger_fraction') and v > 1:
            raise ValueError('recall and trigger limits are fractions')
        if k in ('min_target_parents','min_family_parents') and (int(v)!=v or v<1):
            raise ValueError('positive independent group counts required')
        if k in ('max_background_starts','max_background_confirmed') and int(v)!=v:
            raise ValueError('background budgets count total outputs, not rates')
    if value['min_parent_recall']<=0 or value['min_background_hours']<=0:
        raise ValueError('positive recall and exposure requirements are mandatory')
    return copy.deepcopy(value)


def score_candidate(candidate, bank, constraints, *, score_cache=None):
    """Score all continuous contexts, retaining starts, ends and parent outcomes."""
    _verify_candidate(candidate); bank.verify(); constraints=_constraints(constraints)
    if candidate['classes'] != bank.manifest['classes']:
        raise ValueError('candidate/bank class order mismatch')
    cache=score_cache if score_cache is not None else infer_scores(candidate,bank)
    for k,v in dict(model_sha256=candidate['model_sha256'], features_sha256=candidate['features_sha256'],
                    bank_sha256=bank.manifest_sha256,role=bank.role,classes=candidate['classes'],
                    hop_s=candidate['policy']['hop_s']).items():
        if cache.get(k)!=v:
            raise ValueError(f'stale score cache: {k}')
    if cache.get('origin') not in ('synthetic_control','edge.window_scores'):
        raise ValueError('missing score provenance')
    if cache['origin']=='synthetic_control' and any(r['evidence_domain']!='synthetic proxy' for r in bank.rows):
        raise ValueError('synthetic controls are proxy only')
    if set(cache['sequences'])!={r['id'] for r in bank.rows}:
        raise ValueError('missing/filtered score contexts')
    outcomes=[]
    for row in bank.rows:
        x=cache['sequences'][row['id']]
        if not isinstance(x,np.ndarray) or x.dtype != np.dtype('<f8'):
            raise ValueError('cached probabilities must retain float64 units')
        if sha256(x.tobytes()).hexdigest()!=cache['score_sha256'][row['id']]:
            raise ValueError('cached scores changed')
        if row['sample_rate_hz'] != candidate['features']['fs']:
            raise ValueError('bank/model sample rate mismatch')
        replay=replay_policy(x,candidate['classes'],candidate['policy'],
            window_s=candidate['features']['window_s'], sample_rate_hz=row['sample_rate_hz'],sample_count=row['sample_count'])
        matches=match_events(replay['events'],row['annotations'],latency_s=replay['latency_s'])
        per_class={c: all(i not in matches['unmatched_annotation_indices'] for i,a in enumerate(row['annotations']) if a['label']==c)
                   for c in {a['label'] for a in row['annotations']}}
        outcomes.append(dict(id=row['id'],parent_id=row['parent_id'],session_id=row['session_id'],
            kind=row['kind'],family=row.get('family'),panel=row['panel'],route=row['route'],
            acoustic_stage=row['acoustic_stage'],
            evidence_domain=row['evidence_domain'],matches=matches,parent_success=per_class,**replay))
    primary=[r for r in outcomes if r['panel']=='nominal']
    recall, families, missing, violations = {}, {}, [], []
    parent_success={}
    for c in candidate['classes']:
        parents={r['parent_id']:r['parent_success'][c] for r in primary if r['kind']=='target' and c in r['parent_success']}
        parent_success[c]=parents
        k,n=sum(parents.values()),len(parents)
        recall[c]=dict(hits=k,parents=n,fraction=k/n if n else None,
                       interval_95=list(clopper_pearson(k,n)) if n else None)
        if n < constraints['min_target_parents']:
            missing.append('target parents: '+c)
        if n and k/n < constraints['min_parent_recall']:
            violations.append('positive recall: '+c)
    nuisances = [r for r in primary if r['kind']=='nuisance']
    # Every nominal row contributes, regardless of mandatory support floors.
    # Multiple contexts count outputs separately, but each parent triggers once.
    nuisance_parents={}
    for r in nuisances:
        nuisance_parents[r['parent_id']] = nuisance_parents.get(r['parent_id'], False) or r['starts'] > 0
    required, optional = _family_allocation(bank.manifest)
    required_families, optional_families = required[bank.role], optional[bank.role]
    for f in required_families + optional_families:
        rows=[r for r in nuisances if r['family']==f]
        parents={}
        for r in rows:
            parents[r['parent_id']]=parents.get(r['parent_id'],False) or r['starts']>0
        k,n=sum(parents.values()),len(parents)
        families[f]=dict(required=f in required_families,triggered=k,parents=n,fraction=k/n if n else None,
                         interval_95=list(clopper_pearson(k,n)) if n else None,
                         starts=sum(r['starts'] for r in rows),confirmed=sum(r['confirmed'] for r in rows))
        if f in required_families and n < constraints['min_family_parents']:
            missing.append('nuisance parents: '+f)
        if n and k/n > constraints['max_family_trigger_fraction']:
            violations.append('nuisance trigger fraction: '+f)
    backgrounds=[r for r in primary if r['kind']=='background']
    hours=sum(r['exposure_s'] for r in backgrounds)/3600
    if hours < constraints['min_background_hours']:
        missing.append('complete background exposure')
    bg={}
    for unit in ('starts','confirmed'):
        k=sum(r[unit] for r in backgrounds)
        if k > constraints['max_background_'+unit]:
            violations.append('total background '+unit)
        bg[unit]=dict(total=k,by_class={c:sum(r[unit+'_by_class'][c] for r in backgrounds) for c in candidate['classes']},
            rate_per_hour=k/hours if hours else None,
            poisson_upper_95_per_hour=float(chi2.ppf(.95,2*(k+1))/(2*hours)) if hours else None)
    # Missing support always prevents qualification, even if observed cells fail.
    status='insufficient_evidence' if missing else 'infeasible' if violations else 'feasible'
    public_identity = {k:copy.deepcopy(v) for k,v in candidate.items() if k != 'model_path'}
    return dict(schema='operational-score/1',candidate=public_identity,
        model_file=Path(candidate['model_path']).name if candidate.get('model_path') else None,
        bank_sha256=bank.manifest_sha256,
        role=bank.role,criteria=constraints,criteria_sha256=digest(constraints),status=status,
        screen='INCONCLUSIVE' if missing else 'FAIL_SCREEN' if violations else 'PASS_SCREEN',
        missing=missing,violations=violations,recall=recall,families=families,
        parent_success=parent_success,nuisance_parent_trigger=nuisance_parents,
        nuisance_starts=sum(r['starts'] for r in nuisances),
        nuisance_confirmed=sum(r['confirmed'] for r in nuisances),
        background=dict(hours=hours,sessions=[r['session_id'] for r in backgrounds],**bg),
        outcomes=outcomes,score_origin=cache['origin'],score_sha256=cache['score_sha256'],
        runtime=dict(module=edge.__name__,file=Path(edge.__file__).name,
                     sha256=sha256(Path(edge.__file__).read_bytes()).hexdigest()),
        evidence_domains=sorted({r['evidence_domain'] for r in outcomes}),
        deployment_validated=False,
        interval_scope='Descriptive parent CP and conditional homogeneous-Poisson bounds; development selection is optimistic; session dependence and activity coverage are not certified.')


def candidate_key(row):
    c=row['candidate']; recalls=[v['fraction'] for v in row['recall'].values()]
    if row['status']!='feasible' or not recalls or any(v is None for v in recalls):
        raise ValueError('only feasible complete candidates can be ranked')
    return (-min(recalls),-sum(recalls)/len(recalls),row['nuisance_starts'],row['nuisance_confirmed'],
            c['step'],tuple(c['policy']['on'][label] for label in c['classes']),c['candidate_sha256'])


def select_candidate(results):
    """Deterministic P0 order, with explicit exhaustion/insufficiency states."""
    rows=list(results)
    if any(r['role']!='development' for r in rows):
        raise ValueError('audit scores cannot select or reselect a candidate')
    if len({(r['bank_sha256'],r['criteria_sha256']) for r in rows})>1:
        raise ValueError('candidates must share the frozen bank and criteria')
    if len({r['candidate']['seed'] for r in rows})>1:
        raise ValueError('select within a fixed seed; never select the best training seed')
    feasible=[r for r in rows if r['status']=='feasible']
    selected=min(feasible,key=candidate_key) if feasible else None
    status='feasible' if selected else 'insufficient_evidence' if not rows or any(r['status']=='insufficient_evidence' for r in rows) else 'infeasible'
    return dict(status=status,screen='PASS_SCREEN' if selected else 'INCONCLUSIVE' if status=='insufficient_evidence' else 'FAIL_SCREEN',
                selected=selected,candidate_count=len(rows))


def calibrate_ap_selector(candidates, ap_scores, bank, policies, constraints, *, score_caches=None):
    """Earliest highest-raw-AP checkpoint, then the SAME finite P0 calibration.

    An AP winner with no feasible policy remains infeasible; no replacement AP
    checkpoint is chosen after inspecting its operating-point result.
    """
    from .operating_point import search_operating_point
    if len(candidates)!=len(ap_scores) or not candidates or not np.isfinite(ap_scores).all():
        raise ValueError('finite complete raw AP metrics required')
    if len({c['seed'] for c in candidates})!=1:
        raise ValueError('AP baseline must use the fixed deployment seed')
    i=min(range(len(candidates)), key=lambda i:(-ap_scores[i],candidates[i]['step'],candidates[i]['candidate_sha256']))
    chosen=candidates[i]
    return search_operating_point(chosen,bank,policies,constraints,
        score_cache=None if score_caches is None else score_caches[i])


def outer_screen(candidate, comparator, *, max_poisson_upper_per_hour=1.5):
    """Finite P0 outer screen on an already frozen pair; never performs selection.

    This API does not own the audit seal: the independent runner must enforce
    single use and artifact custody. The 95% bound is conditional on homogeneous
    Poisson exposure. Proxy audio never becomes measured deployment evidence.
    """
    if any(r['role']!='audit' for r in (candidate,comparator)):
        raise ValueError('outer screen requires explicitly opened audit results')
    if candidate['bank_sha256']!=comparator['bank_sha256'] or candidate['criteria_sha256']!=comparator['criteria_sha256']:
        raise ValueError('outer pair must use identical audit allocation and criteria')
    if candidate['candidate']['seed']!=comparator['candidate']['seed']:
        raise ValueError('outer comparison must retain the paired frozen seed')
    if not np.isfinite(max_poisson_upper_per_hour) or max_poisson_upper_per_hour <= 0:
        raise ValueError('positive finite rate bound required')
    lost={}
    for c, reference in comparator['parent_success'].items():
        proposed=candidate['parent_success'].get(c,{})
        if set(proposed)!=set(reference):
            raise ValueError('outer pair has different source parent denominators')
        lost[c]=[p for p in sorted(reference) if reference[p] and not proposed[p]]
    violations=list(candidate['violations'])
    if any(lost.values()):
        violations.append('comparator-successful parent lost')
    missing=list(candidate['missing'])
    for unit in ('starts','confirmed'):
        b=candidate['background'][unit]
        if b['total'] or b['total'] > comparator['background'][unit]['total']:
            violations.append('outer background '+unit)
        if b['poisson_upper_95_per_hour'] is None or b['poisson_upper_95_per_hour'] > max_poisson_upper_per_hour:
            missing.append('outer rate resolution: '+unit)
        if candidate['nuisance_'+unit] > comparator['nuisance_'+unit]:
            violations.append('additional nuisance '+unit)
    # An observed breach (in particular a nonzero alert under the zero-alert
    # gate) remains a failure even when a separate rate bound is unresolved.
    status='fail' if violations else 'inconclusive' if missing else 'pass'
    return dict(outcome=status,missing=missing,violations=violations,lost_parents=lost,
        candidate_sha256=candidate['candidate']['candidate_sha256'],comparator_sha256=comparator['candidate']['candidate_sha256'],
        evidence_domains=candidate['evidence_domains'],deployment_validated=False,
        claim='finite-panel outer screen, conditional rate bound; no natural-event rank-transfer claim')


def paired_parent_improvement(candidate, comparator):
    """P2 one-sided exact McNemar on each nuisance parent counted once."""
    if candidate['bank_sha256']!=comparator['bank_sha256'] or candidate['candidate']['seed']!=comparator['candidate']['seed']:
        raise ValueError('paired bank and fixed seed required')
    a,b=candidate['nuisance_parent_trigger'],comparator['nuisance_parent_trigger']
    if not a or set(a)!=set(b):
        raise ValueError('complete matching nuisance-parent denominators required')
    improved=sum(b[p] and not a[p] for p in a)
    worsened=sum(a[p] and not b[p] for p in a)
    n=improved+worsened
    return dict(improved=improved,worsened=worsened,parents=len(a),discordant=n,
        one_sided_exact_p=float(binom.sf(improved-1,n,.5)) if n else 1.,
        smallest_reachable_p=2.**(-n),minimum_discordant_for_p_below_005=5,
        claim='paired finite-panel start-trigger comparison; no rate or population transfer claim')


def checkpoint_callback(bank, constraints, policies, *, seed, code_revision):
    """Create the integration-owner trainer callback before any experiment.

    The callback takes ``(checkpoint_path, step)`` and returns a complete finite
    search record. It only consumes development audio. The trainer must retain
    checkpoints, compare candidate_key with MIN ordering, and never fit tracker
    persistence after this callback. Construct one callback per registered seed.
    """
    from .operating_point import search_operating_point
    if bank.role != 'development':
        raise ValueError('checkpoint callback requires a development bank')
    bank.verify()
    frozen_constraints = _constraints(constraints)
    frozen_policies = copy.deepcopy(list(policies))
    if not frozen_policies:
        raise ValueError('nonempty frozen policy search required')
    def evaluate(path, step):
        model = edge.Model(str(path))
        candidate = candidate_identity(model_sha256=sha256(Path(path).read_bytes()).hexdigest(),
            features=model.meta['features'],classes=model.classes,policy=frozen_policies[0],
            step=step,seed=seed,code_revision=code_revision,model_path=path)
        result = search_operating_point(candidate,bank,frozen_policies,frozen_constraints)
        result['policy_grid_sha256'] = digest(frozen_policies)
        return result
    return evaluate
