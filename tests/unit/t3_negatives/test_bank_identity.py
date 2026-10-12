# SPDX-License-Identifier: Apache-2.0
"""Disclosed verifier counterexamples, reused as synthetic development regressions."""
import copy
import json
from dataclasses import replace

import pytest
from sensorsim import training_negatives as tn
from sensorsim.negatives.mining import FrozenModel
from .conftest import document

DETECTOR=dict(hop_s=.25,on=.5,off_ratio=.6,smooth_hops=1,min_hops=2)

@pytest.mark.parametrize('mixed_checkpoint',[False,True])
def test_one_checkpoint_per_mining_bank(factory,mixed_checkpoint):
    f=factory(count=2,role='training_mining',stage='device_recorded')
    pool=f.build(purpose='mine')
    def mine(name):
        path=f.root/name;path.write_bytes(name.encode())
        model=FrozenModel(path,lambda audio:[.9],classes=['knock'],fs=f.fs)
        return tn.mine_context_candidates(pool,model=model,detector=DETECTOR)
    first=mine('checkpoint-A');second=mine('checkpoint-B')
    a={c['parent']['item_id']:c for c in first}
    b={c['parent']['item_id']:c for c in second}
    candidates=[a['parent-0'],b['parent-1'] if mixed_checkpoint else a['parent-1']]
    assert len({c['parent']['source_group'] for c in candidates})==2
    reviews={c['review_id']:dict(decision='confirmed_negative',ambiguous=False,listened=True,
        reviewer='independent generated control',notes='synthetic fixture only') for c in candidates}
    def build():return tn.reviewed_context_bank(candidates,reviews,base_weight=4.,lock=pool.lock,lock_sha256=pool.lock_sha256)
    print(json.dumps(dict(probe='checkpoint_identity',mixed_checkpoint=mixed_checkpoint,
        candidate_hashes=[c['model_sha256'] for c in candidates],starts=[c['starts'] for c in candidates])))
    if mixed_checkpoint:
        with pytest.raises(ValueError,match='checkpoint|model|bank'):
            build()
    else:
        bank=build();assert len(bank)==2
        assert sum(a['weight'] for a in bank)==pytest.approx(.4)


def candidates_for(f, *, checkpoint='checkpoint-A', detector=None):
    pool = f.build(purpose='mine')
    path = f.root/checkpoint
    path.write_bytes(checkpoint.encode())
    model = FrozenModel(path, lambda audio: [.9], classes=['knock'], fs=f.fs)
    candidates = tn.mine_context_candidates(pool, model=model, detector=detector or DETECTOR)
    return pool, {c['parent']['item_id']: c for c in candidates}


def bank_for(pool, candidates, reviews=None):
    if reviews is None:
        reviews = {c['review_id']: dict(decision='confirmed_negative', ambiguous=False, listened=True,
            reviewer='development control', notes='generated whole context') for c in candidates}
    return tn.reviewed_context_bank(candidates, reviews, base_weight=4.,
                                    lock=pool.lock, lock_sha256=pool.lock_sha256)


@pytest.mark.parametrize('mixed_detector', [False, True])
def test_bank_binds_one_detector_policy(factory, mixed_detector):
    f = factory(count=2, role='training_mining', stage='device_recorded')
    pool, a = candidates_for(f)
    _, b = candidates_for(f, detector=dict(DETECTOR, on=.6))
    candidates = [a['parent-0'], (b if mixed_detector else a)['parent-1']]
    assert len({c['parent']['source_group'] for c in candidates}) == 2
    if mixed_detector:
        with pytest.raises(ValueError, match='detector|bank'):
            bank_for(pool, candidates)
    else:
        assert len(bank_for(pool, candidates)) == 2


@pytest.mark.parametrize('checkpoint', [None, '', 'x'*64, 'a'*63, 123])
def test_missing_or_invalid_checkpoint_cannot_be_rehashed_into_bank(factory, checkpoint):
    f = factory(count=1, role='training_mining', stage='device_recorded')
    pool, candidates = candidates_for(f)
    c = copy.deepcopy(candidates['parent-0'])
    c['model_sha256'] = checkpoint
    c['review_id'] = tn._candidate_identity(c)
    with pytest.raises(ValueError, match='checkpoint|model'):
        bank_for(pool, [c])


def test_bank_empty_and_filtered_results_are_explicit(factory):
    f = factory(count=1, role='training_mining', stage='device_recorded')
    pool, candidates = candidates_for(f)
    assert bank_for(pool, []) == ()
    assert bank_for(pool, list(candidates.values()), reviews={}) == ()


def test_persisted_bank_identity_binds_checkpoint_detector_split_and_members(factory):
    import hashlib
    f = factory(count=2, role='training_mining', stage='device_recorded')
    pool, candidates = candidates_for(f)
    additions = bank_for(pool, list(candidates.values()))
    restored = json.loads(json.dumps(additions))
    bank = restored[0].get('bank')
    assert bank is not None, 'persist a bank binding with every mining addition'
    assert all(a['bank'] == bank and a['bank_sha256'] == restored[0]['bank_sha256'] for a in restored)
    assert bank['model_sha256'] == hashlib.sha256(b'checkpoint-A').hexdigest()
    assert bank['detector'] == DETECTOR
    assert bank['split_file_sha256'] == pool.lock_sha256
    assert bank['candidate_review_ids'] == sorted(c['review_id'] for c in candidates.values())
    # Independent canonical JSON SHA-256, not the production digest helper.
    encoded = json.dumps(bank, sort_keys=True, allow_nan=False).encode()
    assert restored[0]['bank_sha256'] == hashlib.sha256(encoded).hexdigest()
    for a in restored:
        row = next(r for r in pool.rows if r.item.item_id == a['candidate']['parent']['item_id'])
        tn._validate_addition(a, replace(row, provenance=dict(row.provenance, weight=a['weight'])), pool.lock_sha256)


@pytest.mark.parametrize('change', ['none', 'model', 'detector', 'bank_digest', 'membership', 'split', 'missing_bank', 'relabel_base'])
def test_training_admission_revalidates_persisted_bank(factory, change):
    f = factory(count=8, stage='device_recorded')
    for entry in f.data['rows'][4:]:
        entry['context_role'] = 'training_mining'
    ref = document(f.root, 'mining-pool.json', dict(f.data, rows=f.data['rows'][4:]))
    pool = tn.build_training_pool(f.root/ref['file'], ref['sha256'], fs=f.fs, room_draws=f.draws, purpose='mine')
    path = f.root/'checkpoint-A'; path.write_bytes(b'checkpoint-A')
    model = FrozenModel(path, lambda audio: [.9], classes=['knock'], fs=f.fs)
    candidates = tn.mine_context_candidates(pool, model=model, detector=DETECTOR)
    additions = list(bank_for(pool, candidates))
    # On old code no binding exists: do not manufacture a new API for the good control.
    bad = additions[-1]
    if change == 'model':
        bad['candidate']['model_sha256'] = 'f'*64
    elif change == 'detector':
        bad['candidate']['detector'] = dict(DETECTOR, on=.6)
        bad['candidate']['detector_sha256'] = tn.digest(bad['candidate']['detector'])
    elif change == 'bank_digest':
        bad['bank_sha256'] = 'f'*64
    elif change == 'membership':
        bad['bank'] = dict(bad.get('bank', {}), candidate_review_ids=['f'*64])
        bad['bank_sha256'] = tn.digest(bad['bank'])
    elif change == 'split':
        bad['bank'] = dict(bad.get('bank', {}), split_file_sha256='f'*64)
        bad['bank_sha256'] = tn.digest(bad['bank'])
    elif change == 'missing_bank':
        bad.pop('bank', None); bad.pop('bank_sha256', None)
    if change in ('model', 'detector'):
        # Preserve the previous candidate/review checks so the new bank gate must reject.
        bad['candidate']['review_id'] = tn._candidate_identity(bad['candidate'])
        bad['review']['review_id'] = bad['candidate']['review_id']
    for entry in f.data['rows'][4:]:
        addition = next(a for a in additions if a['candidate']['parent'] == entry['item'])
        entry['mining_addition'] = document(f.root, 'addition-'+entry['item']['item_id']+'.json', addition)
    if change == 'relabel_base':
        f.data['rows'][-1]['context_role'] = 'training_base'
    document(f.root, 'pool.json', f.data)
    if change == 'none':
        training = f.build()
        assert training.rows[-1].provenance['mining_addition']['candidate']['model_sha256'] == model.sha256
        assert len(training.metadata()['mining_banks']) == 1
        assert training.metadata()['mining_banks'][0]['model_sha256'] == model.sha256
        assert training.probabilities[-1] == pytest.approx(.2/4.8)
    else:
        with pytest.raises(ValueError, match='bank|checkpoint|model|detector'):
            f.build()
