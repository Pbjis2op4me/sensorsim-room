"""Whole-context tracker controls, using a deterministic independent score trace."""
import json
import numpy as np
import pytest
from sensorsim import training_negatives as tn
from sensorsim.negatives.mining import FrozenModel
from .conftest import document, sha

DETECTOR=dict(hop_s=.25,on=.5,off_ratio=.6,smooth_hops=1,min_hops=2)


def model(f, scores):
    checkpoint=f.root/'model';checkpoint.write_bytes(b'original deterministic score fixture')
    iterator=iter(scores)
    return FrozenModel(checkpoint,lambda x:[next(iterator)],classes=['knock'],fs=f.fs)


def test_tracker_mines_whole_context_and_flushes_not_isolated_peak(factory):
    f=factory(count=1,role='training_mining',stage='device_recorded')
    pool=f.build(purpose='mine')
    # 9 windows: start/end unconfirmed at 0; then start at 1.5, flush confirmed.
    candidates=tn.mine_context_candidates(pool,model=model(f,[.9,0,0,0,0,0,.9,.9,.9]),detector=DETECTOR)
    assert len(candidates)==1
    c=candidates[0]
    assert (c['starts'],c['confirmed'])==(2,1)
    assert c['sample_range']==[0,3*f.fs] and c['units']=='samples'
    assert c['events'][-1]['confirmed'] is True
    clean=tn.mine_context_candidates(pool,model=model(f,[0.]*9),detector=DETECTOR)
    assert clean==()
    with pytest.raises(ValueError,match='unreviewed mining'):
        pool.choose(np.random.default_rng(1))


def test_mining_rejects_wrong_role_before_any_inference(factory):
    f=factory();pool=f.build();called=[]
    checkpoint=f.root/'model';checkpoint.write_bytes(b'fixture')
    frozen=FrozenModel(checkpoint,lambda x:called.append(1) or [.9],classes=['knock'],fs=f.fs)
    with pytest.raises(ValueError,match='eligible mining pool'):
        tn.mine_context_candidates(pool,model=frozen,detector=DETECTOR)
    assert called==[]


def test_bank_review_weight_caps_and_actual_training_consumption(factory):
    f=factory(count=4,role='training_mining',stage='device_recorded');pool=f.build(purpose='mine')
    candidates=tn.mine_context_candidates(pool,model=model(f,[.9]*36),detector=DETECTOR)
    reviews={c['review_id']:dict(decision='confirmed_negative',ambiguous=False,listened=True,
        reviewer='generated control',notes='whole context inspected in deterministic test') for c in candidates}
    additions=tn.reviewed_context_bank(candidates,reviews,base_weight=4.,lock=pool.lock,lock_sha256=pool.lock_sha256)
    assert len(additions)==4 and sum(a['weight'] for a in additions)==pytest.approx(.8)
    assert all(a['weight']==pytest.approx(.2) for a in additions)
    # A candidate is admitted only with its exact whole-parent review and mass.
    for a in additions:
        row=next(r for r in pool.rows if r.item.item_id==a['candidate']['parent']['item_id'])
        p=dict(row.provenance,mining_addition=a,weight=a['weight'])
        from dataclasses import replace
        tn._validate_addition(a,replace(row,provenance=p),pool.lock_sha256)
        with pytest.raises(ValueError,match='weight'):
            tn._validate_addition(a,replace(row,provenance=dict(p,weight=99.)),pool.lock_sha256)
    reviews[candidates[0]['review_id']]['ambiguous']=True
    with pytest.raises(ValueError,match='ambiguous'):
        tn.reviewed_context_bank(candidates,reviews,base_weight=4.,lock=pool.lock,lock_sha256=pool.lock_sha256)


def test_incompatible_hops_constant_scores_and_empty_outputs(factory):
    f=factory(count=1,role='training_mining',stage='device_recorded');pool=f.build(purpose='mine')
    for detector in (dict(DETECTOR,hop_s=0),dict(DETECTOR,min_hops=1.5),dict(DETECTOR,on=float('nan'))):
        with pytest.raises(ValueError):
            tn.mine_context_candidates(pool,model=model(f,[.9]*9),detector=detector)
    with pytest.raises(ValueError,match='probabilities'):
        tn.mine_context_candidates(pool,model=model(f,[float('nan')]*9),detector=DETECTOR)


def test_reviewed_weights_reach_training_pool_and_overweight_refused(factory):
    f=factory(count=8,stage='device_recorded')
    for entry in f.data['rows'][4:]:entry['context_role']='training_mining'
    mining_data=dict(f.data,rows=f.data['rows'][4:])
    ref=document(f.root,'mining-pool.json',mining_data)
    mining_pool=tn.build_training_pool(f.root/ref['file'],ref['sha256'],fs=f.fs,room_draws=f.draws,purpose='mine')
    candidates=tn.mine_context_candidates(mining_pool,model=model(f,[.9]*36),detector=DETECTOR)
    reviews={c['review_id']:dict(decision='confirmed_negative',ambiguous=False,listened=True,
        reviewer='fixture',notes='generated test context') for c in candidates}
    additions=tn.reviewed_context_bank(candidates,reviews,base_weight=4.,lock=mining_pool.lock,lock_sha256=mining_pool.lock_sha256)
    for entry in f.data['rows'][4:]:
        addition=next(a for a in additions if a['candidate']['parent']==entry['item'])
        entry['mining_addition']=document(f.root,'addition-'+entry['item']['item_id']+'.json',addition)
    document(f.root,'pool.json',f.data)
    training=f.build()
    np.testing.assert_allclose(training.weights,[1]*4+[.2]*4)
    np.testing.assert_allclose(training.probabilities,np.array([1]*4+[.2]*4)/4.8)
    class Draw:
        def random(self):return .99
    assert training.choose(Draw()).provenance['context_role']=='training_mining'
    ref=f.data['rows'][-1]['mining_addition'];bad=json.loads((f.root/ref['file']).read_text());bad['weight']=2.
    f.data['rows'][-1]['mining_addition']=document(f.root,ref['file'],bad);document(f.root,'pool.json',f.data)
    with pytest.raises(ValueError,match='mined total'):
        f.build()
