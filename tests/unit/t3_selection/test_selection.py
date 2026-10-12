"""Declared synthetic development fixtures; no physical or acceptance evidence."""
import copy
import json
from hashlib import sha256

import numpy as np
import pytest
from scipy.io import wavfile


def make_bank(tmp_path, *, role='development', families=('fan','dishes'), classes=('knock',), duration_s=3., background_s=10., fs=100, target_parents=1):
    from sensorsim.evaluation.selection import audio_digest, build_validation_bank
    rows=[]
    rng=np.random.default_rng(401)
    for r in ('train','development','audit'):
        specs=[('target',c) for c in classes for _ in range(target_parents)]+[('nuisance',f) for f in families]+[('background','bed')]
        for i,(kind,label) in enumerate(specs):
            name=f'{r}-{kind}-{label}' + (f'-{i}' if kind=='target' and target_parents>1 else '')
            count=round((background_s if kind=='background' else duration_s)*fs)
            audio=rng.integers(-15000,15000,count,dtype=np.int16)
            wavfile.write(tmp_path/(name+'.wav'),fs,audio)
            rows.append(dict(id=name,role=r,kind=kind,parent_id=name,session_id=name+'-session',
                lineage_ids=[name+'-original'], parent_audio_sha256=[sha256(name.encode()).hexdigest()],
                audio=name+'.wav',file_sha256=sha256((tmp_path/(name+'.wav')).read_bytes()).hexdigest(),
                audio_sha256=audio_digest(audio.astype(np.float32)/32768),sample_rate_hz=fs,sample_count=count,
                acoustic_stage='rendered_context',route='generated-development-fixture',evidence_domain='synthetic proxy',panel='nominal',
                family=label if kind=='nuisance' else None,
                annotations=[dict(label=label,start_s=1.4,end_s=1.8)] if kind=='target' else []))
    data=dict(schema='operational-bank/1',allocation_frozen=True,classes=list(classes),required_families=list(families),rows=rows)
    path=tmp_path/'bank.json'
    path.write_text(json.dumps(data))
    bank=build_validation_bank(path,expected_sha256=sha256(path.read_bytes()).hexdigest(),role=role)
    return bank,path


def candidate(*, on=.5, step=250, seed=1103, classes=('knock',)):
    from sensorsim.evaluation.selection import candidate_identity
    return candidate_identity(model_sha256=sha256(f'control-{step}'.encode()).hexdigest(),
        features=dict(fs=100,window_s=1.),classes=list(classes),
        policy=dict(on=on,off_ratio=.6,smooth_hops=1,min_hops=2,hop_s=.25),
        step=step,seed=seed,code_revision='declared-synthetic-development-control')


def constraints(**kw):
    return dict(dict(min_parent_recall=.75,max_family_trigger_fraction=.1,min_target_parents=1,
        min_family_parents=1,min_background_hours=10/3600,max_background_starts=0,max_background_confirmed=0),**kw)


def scores(bank, c, *, negative=False):
    from sensorsim.evaluation.selection import cache_scores
    result={}
    for r in bank.rows:
        x=np.zeros((1+(r['sample_count']-100)//25,len(c['classes'])))
        if r['kind']=='target' and not negative:
            x[2:4,c['classes'].index(r['annotations'][0]['label'])]=.9
        result[r['id']]=x
    return cache_scores(c,bank,result)


def resave(path,data):
    from sensorsim.evaluation.selection import build_validation_bank
    path.write_text(json.dumps(data))
    return build_validation_bank(path,expected_sha256=sha256(path.read_bytes()).hexdigest())


def test_positive_feasibility_rejects_always_negative(tmp_path):
    from sensorsim.evaluation.selection import score_candidate
    bank,_=make_bank(tmp_path); c=candidate()
    good=score_candidate(c,bank,constraints(),score_cache=scores(bank,c))
    bad=score_candidate(c,bank,constraints(),score_cache=scores(bank,c,negative=True))
    assert good['status']=='feasible' and good['recall']['knock']['fraction']==1.
    assert bad['status']=='infeasible' and bad['recall']['knock']['fraction']==0.
    assert bad['violations']==['positive recall: knock']
    assert good['score_origin']==bad['score_origin']=='synthetic_control'
    assert good['outcomes'][0]['starts']==1 and bad['outcomes'][0]['starts']==0


def test_exhaustion_and_combined_starts_across_classes(tmp_path):
    from sensorsim.evaluation.selection import cache_scores,score_candidate
    from sensorsim.evaluation.operating_point import search_operating_point
    classes=('knock','doorbell','kettle')
    bank,_=make_bank(tmp_path,classes=classes); c=candidate(classes=classes)
    seq={k:v.copy() for k,v in scores(bank,c)['sequences'].items()}
    bg=next(r['id'] for r in bank.rows if r['kind']=='background')
    seq[bg][2,:]=1.
    cache=cache_scores(c,bank,seq)
    limit=constraints(max_background_starts=2)
    result=search_operating_point(c,bank,[c['policy'],dict(c['policy'],on=.8)],limit,score_cache=cache)
    assert result['status']=='infeasible' and result['selected'] is None
    assert len(result['attempted'])==2
    for row in result['attempted']:
        assert row['background']['starts']['total']==3
        assert row['background']['confirmed']['total']==0
        assert row['background']['starts']['by_class']==dict.fromkeys(classes,1)


def test_required_families_cannot_be_pooled_or_disappear(tmp_path):
    from sensorsim.evaluation.selection import cache_scores,score_candidate
    bank,path=make_bank(tmp_path); c=candidate()
    seq={k:v.copy() for k,v in scores(bank,c)['sequences'].items()}
    seq['development-nuisance-fan'][2,0]=.9
    result=score_candidate(c,bank,constraints(max_family_trigger_fraction=.5),score_cache=cache_scores(c,bank,seq))
    assert result['status']=='infeasible' and result['families']['fan']['fraction']==1.
    data=json.loads(path.read_text());data['required_families'].append('missing')
    bank=resave(path,data)
    result=score_candidate(c,bank,constraints(),score_cache=scores(bank,c))
    assert result['status']=='insufficient_evidence' and result['families']['missing']['fraction'] is None


def test_deterministic_ties_earliest_checkpoint_and_threshold_tuple(tmp_path):
    from sensorsim.evaluation.selection import score_candidate,select_candidate
    bank,_=make_bank(tmp_path)
    rows=[]
    for step,on in ((500,.5),(250,.8),(250,.5)):
        c=candidate(step=step,on=on)
        rows.append(score_candidate(c,bank,constraints(),score_cache=scores(bank,c)))
    a=select_candidate(rows)['selected']['candidate']
    b=select_candidate(rows[::-1])['selected']['candidate']
    assert a==b and a['step']==250 and a['policy']['on']=={'knock':.5}
    assert a['model_sha256'] and a['features_sha256'] and a['candidate_sha256']
    assert a['policy']['min_hops']=={'knock':2}


def test_raw_ap_baseline_gets_same_search_and_cannot_replace_infeasible_winner(tmp_path):
    from sensorsim.evaluation.selection import calibrate_ap_selector
    bank,_=make_bank(tmp_path); a,b=candidate(),candidate(step=500)
    result=calibrate_ap_selector([a,b],[.999975124,1.],bank,[a['policy']],constraints(),
        score_caches=[scores(bank,a),scores(bank,b,negative=True)])
    assert result['status']=='infeasible' and len(result['attempted'])==1
    assert result['attempted'][0]['candidate']['step']==500
    result=calibrate_ap_selector([b,a],[1.,1.],bank,[a['policy']],constraints(),
        score_caches=[scores(bank,b),scores(bank,a)])
    assert result['selected']['candidate']['step']==250


@pytest.mark.parametrize('key',['parent_id','session_id','lineage_ids','parent_audio_sha256','audio_sha256'])
def test_renamed_reencoded_parent_inventory_cannot_cross_roles(tmp_path,key):
    bank,path=make_bank(tmp_path)
    data=json.loads(path.read_text())
    a=next(r for r in data['rows'] if r['role']=='train')
    b=next(r for r in data['rows'] if r['role']=='audit')
    b[key]=a[key]
    with pytest.raises(ValueError,match='crosses roles'):
        resave(path,data)


def test_duplicate_nominal_parent_and_constant_audio_rejected(tmp_path):
    from sensorsim.evaluation.selection import audio_digest
    bank,path=make_bank(tmp_path)
    data=json.loads(path.read_text());a=copy.deepcopy(next(r for r in data['rows'] if r['role']=='development' and r['kind']=='target'))
    a['id']='renamed';data['rows'].append(a)
    with pytest.raises(ValueError,match='one frozen nominal'):resave(path,data)
    data=json.loads(json.dumps(bank.manifest));a=next(r for r in data['rows'] if r['role']=='development')
    wavfile.write(tmp_path/a['audio'],100,np.zeros(a['sample_count'],np.int16))
    a['audio_sha256']=audio_digest(np.zeros(a['sample_count'],np.float32))
    a['file_sha256']=sha256((tmp_path/a['audio']).read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='constant audio'):resave(path,data)


def test_bank_caches_and_candidate_cannot_be_mutated_or_filtered(tmp_path):
    from sensorsim.evaluation.selection import score_candidate
    bank,_=make_bank(tmp_path);c=candidate();cache=scores(bank,c)
    c['policy']['min_hops']['knock']=1
    with pytest.raises(ValueError,match='candidate identity changed'):score_candidate(c,bank,constraints(),score_cache=cache)
    c=candidate();cache['sequences'].pop(next(iter(cache['sequences'])))
    with pytest.raises(ValueError,match='filtered'):score_candidate(c,bank,constraints(),score_cache=cache)
    bank.manifest['required_families'].clear()
    with pytest.raises(ValueError,match='manifest changed'):score_candidate(c,bank,constraints())


def test_search_refuses_audit_and_cross_seed_selection(tmp_path):
    from sensorsim.evaluation.selection import score_candidate,select_candidate
    from sensorsim.evaluation.operating_point import search_operating_point
    bank,_=make_bank(tmp_path,role='audit');c=candidate()
    with pytest.raises(ValueError,match='development'):search_operating_point(c,bank,[c['policy']],constraints())
    scored=score_candidate(c,bank,constraints(),score_cache=scores(bank,c))
    with pytest.raises(ValueError,match='audit scores'):select_candidate([scored])
    bank,_=make_bank(tmp_path)
    rows=[score_candidate(c,bank,constraints(),score_cache=scores(bank,c)) for c in (candidate(seed=1),candidate(seed=2))]
    with pytest.raises(ValueError,match='fixed seed'):select_candidate(rows)


def test_audit_bytes_are_never_opened_during_development(tmp_path):
    from sensorsim.evaluation.selection import build_validation_bank
    bank,path=make_bank(tmp_path)
    for r in bank.manifest['rows']:
        if r['role']=='audit':(tmp_path/r['audio']).unlink()
    bank=build_validation_bank(path,expected_sha256=sha256(path.read_bytes()).hexdigest())
    assert bank.role=='development'


def test_unrelated_pre_event_success_does_not_credit_parent(tmp_path):
    from sensorsim.evaluation.selection import cache_scores,score_candidate
    bank,_=make_bank(tmp_path);c=candidate()
    seq={k:v.copy() for k,v in scores(bank,c)['sequences'].items()}
    seq['development-target-knock'][:]=.9
    r=score_candidate(c,bank,constraints(),score_cache=cache_scores(c,bank,seq))
    assert r['recall']['knock']['hits']==0 and r['outcomes'][0]['confirmed']==1


def test_outer_screen_preserves_each_parent_and_rate_resolution(tmp_path):
    from sensorsim.evaluation.selection import score_candidate,outer_screen
    bank,_=make_bank(tmp_path,role='audit',background_s=7200.)
    c=candidate();control=score_candidate(c,bank,constraints(),score_cache=scores(bank,c))
    r=outer_screen(control,control)
    assert r['outcome']=='pass'
    upper=control['background']['starts']['poisson_upper_95_per_hour']
    assert upper==pytest.approx(-np.log(.05)/2)
    assert outer_screen(control,control,max_poisson_upper_per_hour=1.)['outcome']=='inconclusive'
    bad=score_candidate(c,bank,constraints(),score_cache=scores(bank,c,negative=True))
    r=outer_screen(bad,control)
    assert r['outcome']=='fail' and r['lost_parents']=={'knock':['audit-target-knock']}
    assert r['deployment_validated'] is False


def test_mcnemar_reachability_and_correct_null():
    from sensorsim.evaluation.selection import paired_parent_improvement
    def row(values):
        return dict(bank_sha256='same',candidate={'seed':1103},nuisance_parent_trigger=dict(enumerate(values)))
    assert paired_parent_improvement(row([False]*4),row([True]*4))['one_sided_exact_p']==.0625
    assert paired_parent_improvement(row([False]*5),row([True]*5))['one_sided_exact_p']==.03125
    assert paired_parent_improvement(row([False]*5),row([False]*5))['one_sided_exact_p']==1.


def test_cache_binds_actual_hop_and_missing_inference_never_falls_back(tmp_path):
    from sensorsim.evaluation.selection import candidate_identity,score_candidate
    bank,_=make_bank(tmp_path);c=candidate();cache=scores(bank,c)
    changed=candidate_identity(model_sha256=c['model_sha256'],features=c['features'],classes=c['classes'],
        policy=dict(c['policy'],hop_s=.2),step=c['step'],seed=c['seed'],code_revision=c['code_revision'])
    with pytest.raises(ValueError,match='hop_s'):score_candidate(changed,bank,constraints(),score_cache=cache)
    with pytest.raises(ValueError,match='actual model file'):score_candidate(c,bank,constraints())


def test_actual_edge_feature_model_and_tracker_route(tmp_path):
    from sensorsim import edge
    from sensorsim.evaluation.selection import candidate_identity,score_candidate
    bank,_=make_bank(tmp_path)
    features=dict(fs=100,window_s=1.,n_fft=20,hop=10,n_mels=3,fmin_hz=1.,fmax_hz=45.)
    policy=candidate()['policy']
    meta=dict(format=edge.FORMAT,classes=['knock'],features=features,detector=policy,
              layers=[dict(type='pool'),dict(type='fc',name='head')])
    path=tmp_path/'model.npz'
    np.savez(path,meta=np.frombuffer(json.dumps(meta).encode(),np.uint8),
             **{'head.w':np.zeros((1,2),np.float32),'head.b':np.array([np.log(9.)],np.float32)})
    c=candidate_identity(model_sha256=sha256(path.read_bytes()).hexdigest(),features=features,classes=['knock'],
                         policy=policy,step=250,seed=1103,code_revision='development-fixture',model_path=path)
    r=score_candidate(c,bank,constraints())
    assert r['score_origin']=='edge.window_scores' and r['runtime']['sha256']
    assert r['background']['starts']['total']==1 and r['background']['confirmed']['total']==1
    assert r['status']=='infeasible'
    assert all(o['starts']==o['confirmed']==1 for o in r['outcomes'])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='actual model file'):score_candidate(c,bank,constraints())


def test_lane_a_inventory_is_consumed_and_mining_cannot_leak(tmp_path):
    from dataclasses import asdict
    from sensorsim.negatives.splits import Item,SplitPlan
    bank,path=make_bank(tmp_path)
    source=Item('training-source','training-family','training-source',sha256(b'training-source').hexdigest())
    audit=Item('audit-source','audit-family','audit-source',sha256(b'audit-source').hexdigest())
    plan=SplitPlan(((source,'train'),(audit,'audit')),1103)
    lock=tmp_path/'negative-lock.json';plan.save(lock)
    inventory=dict(schema='negative-training/2',target_scope='development fixture targets',split_lock=lock.name,
        split_file_sha256=sha256(lock.read_bytes()).hexdigest(),
        rows=[dict(item=asdict(source),context_role='training_base',
                   source=dict(kind='procedural',recipe=dict(family=source.family,seed=1,duration_s=3.)),
                   semantic_review=dict(file='review.json',sha256=sha256(b'review').hexdigest()))])
    inv=tmp_path/'negative-contexts.json';inv.write_text(json.dumps(inventory))
    data=copy.deepcopy(bank.manifest)
    data['negative_training']=dict(file=inv.name,sha256=sha256(inv.read_bytes()).hexdigest())
    clean=resave(path,data)
    assert len(clean.bindings)==2
    bad=copy.deepcopy(data)
    next(r for r in bad['rows'] if r['role']=='development')['lineage_ids'].append(source.source_group)
    with pytest.raises(ValueError,match='crosses roles'):resave(path,bad)
    inv.write_text('{}')
    with pytest.raises(ValueError,match='dependency changed'):clean.verify()


def test_diagnostic_descendants_do_not_enlarge_parent_recall(tmp_path):
    from sensorsim.evaluation.selection import score_candidate
    bank,path=make_bank(tmp_path)
    data=copy.deepcopy(bank.manifest)
    row=next(r for r in data['rows'] if r['role']=='development' and r['kind']=='target')
    row['panel']='diagnostic'
    bank=resave(path,data);c=candidate()
    result=score_candidate(c,bank,constraints(),score_cache=scores(bank,c))
    assert result['status']=='insufficient_evidence' and result['recall']['knock']['parents']==0


def test_empty_candidate_set_and_insufficient_exposure_are_explicit(tmp_path):
    from sensorsim.evaluation.selection import score_candidate,select_candidate
    assert select_candidate([])['status']=='insufficient_evidence'
    bank,_=make_bank(tmp_path);c=candidate()
    r=score_candidate(c,bank,constraints(min_background_hours=1.),score_cache=scores(bank,c))
    assert r['status']=='insufficient_evidence' and r['background']['hours']==10/3600


def test_equal_recall_cannot_offset_a_lost_comparator_parent(tmp_path):
    from sensorsim.evaluation.selection import cache_scores,score_candidate,outer_screen
    bank,_=make_bank(tmp_path,role='audit',target_parents=4,background_s=7200.)
    c=candidate();results=[]
    for missing in (0,1):
        seq={k:v.copy() for k,v in scores(bank,c)['sequences'].items()}
        seq[f'audit-target-knock-{missing}'][:]=0.
        results.append(score_candidate(c,bank,constraints(min_target_parents=4),score_cache=cache_scores(c,bank,seq)))
    assert all(r['status']=='feasible' and r['recall']['knock']['fraction']==.75 for r in results)
    assert outer_screen(*results)['outcome']=='fail'
    assert outer_screen(*results)['lost_parents']=={'knock':['audit-target-knock-0']}


def test_low_class_recall_is_not_hidden_by_macro_average(tmp_path):
    from sensorsim.evaluation.selection import cache_scores,score_candidate
    bank,_=make_bank(tmp_path,classes=('knock','doorbell'),target_parents=4)
    c=candidate(classes=('knock','doorbell'))
    seq={k:v.copy() for k,v in scores(bank,c)['sequences'].items()}
    for i in (0,1):seq[f'development-target-knock-{i}'][:]=0.
    result=score_candidate(c,bank,constraints(min_target_parents=4),score_cache=cache_scores(c,bank,seq))
    assert result['recall']['knock']['fraction']==.5 and result['recall']['doorbell']['fraction']==1.
    assert result['status']=='infeasible'


def test_over_budget_search_refuses_to_filter_data(tmp_path):
    from sensorsim.evaluation.operating_point import search_operating_point
    bank,_=make_bank(tmp_path);c=candidate()
    with pytest.raises(ValueError,match='replay-row budget'):
        search_operating_point(c,bank,[c['policy']],constraints(),score_cache=scores(bank,c),max_replay_rows=1)


def test_cache_byte_reinterpretation_cannot_change_probability_units(tmp_path):
    from sensorsim.evaluation.selection import score_candidate
    bank,_=make_bank(tmp_path);c=candidate();cache=scores(bank,c)
    key=next(iter(cache['sequences']))
    cache['sequences'][key]=cache['sequences'][key].view(np.int64)
    with pytest.raises(ValueError,match='float64 units'):
        score_candidate(c,bank,constraints(),score_cache=cache)


def test_rendered_context_cannot_be_relabelled_deployment_recording(tmp_path):
    bank,path=make_bank(tmp_path)
    data=copy.deepcopy(bank.manifest)
    next(r for r in data['rows'] if r['role']=='development')['evidence_domain']='deployment recording'
    with pytest.raises(ValueError,match='cannot be deployment'):resave(path,data)


def test_observed_outer_alert_failure_is_not_erased_by_rate_uncertainty(tmp_path):
    from sensorsim.evaluation.selection import cache_scores,score_candidate,outer_screen
    bank,_=make_bank(tmp_path,role='audit',background_s=7200.)
    c=candidate();cache=scores(bank,c)
    correct=score_candidate(c,bank,constraints(),score_cache=cache)
    seq={k:v.copy() for k,v in cache['sequences'].items()}
    seq['audit-background-bed'][2,0]=.9
    bad=score_candidate(c,bank,constraints(),score_cache=cache_scores(c,bank,seq))
    assert bad['background']['starts']['total']==1
    result=outer_screen(bad,correct)
    assert 'outer rate resolution: starts' in result['missing']
    assert result['outcome']=='fail'
