"""Explicit development patch checks. Run on a candidate with the handoff applied.

Filename intentionally avoids default collection: the owned module alone does
not wire the shared trainer. This is required acceptance work for the verifier,
not an xfail or implicit skip in the module suite.
"""
import json
import numpy as np
import pytest
from scipy.io import wavfile
from sensorsim import nn, training_negatives as tn, training_device as device
from tests.unit.t3_negatives.conftest import factory, sha


@pytest.mark.parametrize('knobs',[{},dict(partial_windows=True),dict(audibility_gate=True),
    dict(low_event_spl_tail=True),dict(event_aware_masks=True),dict(window_policy='deployment/1',
        p_specaugment=0.,p_freq_mixstyle=0.)])
@pytest.mark.parametrize('positive',[True,False])
def test_actual_make_example_reaches_context_on_both_routes(factory,monkeypatch,knobs,positive):
    f=factory(stage='device_recorded');f.corpus.negative_pool=f.build()
    cfg=dict(f.cfg,**knobs,p_background_only=0. if positive else 1.,event_spl_db=[85.,85.])
    visits=[];original=tn.draw_training_context
    def observe(*args,**kwargs):
        visits.append(kwargs['draw']);return original(*args,**kwargs)
    monkeypatch.setattr(tn,'draw_training_context',observe)
    heard,label,ctx=nn.make_example(f.corpus,np.random.default_rng(89),cfg,return_context=True)
    assert visits and visits[0].provenance['injected'], 'matched confuser composition hook was bypassed'
    assert ctx['negative_context']['target_present'] is positive
    assert f.corpus.confuser_draws==1
    baseline,label0=nn.make_example(f.corpus,np.random.default_rng(89),dict(cfg,p_confuser_given_target=0.,
        p_confuser_given_background=0.))
    assert not np.array_equal(heard,baseline)
    np.testing.assert_array_equal(label,label0)
    if not positive:assert not label.any()


def test_real_shared_mixer_receives_two_events_and_independent_counterfactual(factory,monkeypatch):
    f=factory();f.corpus.negative_pool=f.build();visits=[];original=device.mix_events
    def record(bed,events,*args,**kwargs):
        visits.append([np.array(e.audio,copy=True) for e in events]);return original(bed,events,*args,**kwargs)
    monkeypatch.setattr(device,'mix_events',record)
    _,y,ctx=nn.make_example(f.corpus,np.random.default_rng(3),dict(f.cfg,p_background_only=0.),return_context=True)
    assert y[0]==1 and ctx['negative_context']['injected']
    assert len(visits)==2 and all(len(v)==2 for v in visits)
    assert np.any(visits[0][0]) and np.any(visits[0][1])
    assert not np.any(visits[1][0])
    np.testing.assert_array_equal(visits[1][1],visits[0][1])


def test_nn_train_constructs_strict_pool_and_reports_consumed_draws(factory,monkeypatch):
    f=factory(stage='device_recorded');manifest=f.root/'training.jsonl';rows=[]
    room=f.draws['room-A'].room
    for split in ['train','val']:
        for label,x in [('knock',f.corpus.targets['knock'][0]),('background',f.corpus.background[0])]:
            path=f.root/f'{split}-{label}.wav';wavfile.write(path,f.fs,x.astype(np.float32))
            row=dict(file=path.name,label=label,split=split,fs=f.fs,source_group=f'{split}-{label}')
            if label!='background':row.update(room_draw_id='room-A',model_hash=room.model_hash,
                receiver_xyz_m=[2.,2.,1.],resolved_config=dict(model=room.model_dump(mode='json'),
                    synthesis=f.draws['room-A'].config.model_dump(mode='json')))
            rows.append(row)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    seen=[];original=tn.build_training_pool
    def construct(*args,**kwargs):
        result=original(*args,**kwargs);seen.append(result);return result
    monkeypatch.setattr(tn,'build_training_pool',construct)
    report=nn.train(manifest,f.root/'result.npz',negative_context_policy='matched/1',
        negative_manifest=str(f.manifest),negative_manifest_sha256=sha(f.manifest),
        p_confuser_given_target=1.,p_confuser_given_background=1.,p_background_only=.35,
        selection_mode='diagnostic_final',steps=2,batch=8,width=4,blocks=1,workers=0,threads=2,val_windows=16,
        p_specaugment=0.,p_freq_mixstyle=0.)
    assert len(seen)==1
    assert report['negatives']['policy']=='matched/1'
    assert report['negatives']['sampling']['confuser_windows']==16
    sampling=report['negatives']['sampling']
    assert sampling['background_windows'] < 16
    assert sampling['target_confuser_windows']==sampling['target_windows']
    assert sampling['background_confuser_windows']==sampling['background_windows']
    assert sampling['target_windows']+sampling['background_windows']==16
    assert sampling['effective_share_of_background']==1.
    assert sampling['effective_share_of_targets']==1.


def test_project_accepts_config_and_rejects_stress(factory):
    from sensorsim.project import _recognizer_settings
    f=factory()
    cfg=dict(recognizer=dict(type='cnn',negative_context_policy='matched/1'),dataset=dict(fs=f.fs))
    assert _recognizer_settings(cfg)['negative_context_policy']=='matched/1'
    cfg['recognizer']['device_split']='stress'
    with pytest.raises(ValueError,match='in_distribution'):_recognizer_settings(cfg)


def test_wp2_overlay_invokes_parent_gate_before_render(factory,monkeypatch):
    from types import SimpleNamespace
    from sensorsim.evaluation import fa_rate
    from sensorsim.negatives.splits import Item,SplitPlan
    from tests.unit.t3_negatives.conftest import document
    f=factory(procedural=True,count=1);entry=f.data['rows'][0]
    # Independent audit fixture, never reuse the training-role split as authority.
    plan=SplitPlan.create(f.parents,audit_families=[f.parents[0].family],development_families=[],seed=31)
    plan.save(f.root/'audit.json')
    (f.root/'room.json').write_text(f.draws['room-A'].room.model_dump_json())
    background=np.random.default_rng(411).normal(0,.001,8*f.fs)
    bed=dict(path='bed.wav',sample_range=[0,len(background)],deployed_device=True,target_free=True,
             capture_device='generated test control',profile_id='pi5_inmp441/1',recording_id='bed-parent')
    source=dict(procedural=entry['source']['recipe'],whole_episode=True,onset_s=2.,source_xyz_m=[1.,1.,1.],
                spl_db=70.,locked_parent=entry['item'],semantic_review=entry['semantic_review'])
    manifest=dict(overlay_path=dict(room='room.json',receiver_xyz_m=[2.,2.,1.],device_id='pi5_inmp441/1',
                  config=dict(max_order=0,late=False)),
        negative_parent_lock=dict(file='audit.json',sha256=sha(f.root/'audit.json'),target_scope=f.data['target_scope']),
        challenges=[dict(family=f.parents[0].family,overlay=dict(background=bed,events=[source],whole_episode=True,
                              duration_s=8.,seed=412,start_s=0.))])
    audit=SimpleNamespace(cache={(f.root/'bed.wav').resolve():background},report=lambda:{'development_fixture':True})
    model=SimpleNamespace(meta=dict(features=dict(fs=f.fs),detector=dict(hop_s=.25,smooth_hops=1,on=.5,off_ratio=.6,min_hops=2)),
        features=SimpleNamespace(n=f.fs),classes=['knock'],scores=lambda x:np.zeros((len(x),1)))
    visits=[];original=tn.validate_overlay_sources
    def spy(*args,**kwargs):
        result=original(*args,**kwargs);visits.append(result);return result
    monkeypatch.setattr(tn,'validate_overlay_sources',spy)
    rows,_=fa_rate._prepare_overlays(model,manifest,f.root,audit=audit)
    assert len(visits)==1 and rows[0]['overlay_provenance']['negative_parent_lock']['purpose']=='audit'
    manifest['negative_parent_lock']['sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):
        fa_rate._prepare_overlays(model,manifest,f.root,audit=audit)
