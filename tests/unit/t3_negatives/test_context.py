"""Path controls, not a realism comparison or efficacy measurement."""
from dataclasses import replace
import numpy as np
import pytest
from sensorsim import training_negatives as tn, training_device as device
from sensorsim.negatives.composer import audio_hash


def test_both_label_branches_prepare_confusers_and_disabled_control(factory):
    f=factory();f.corpus.negative_pool=f.build()
    for name,index in [('knock',0),(None,None)]:
        draw=tn.prepare_context(f.corpus,np.random.default_rng(101),f.cfg,f.fs,class_name=name,target_index=index)
        assert draw.provenance['injected'] and draw.event is not None
        assert draw.provenance['parent']['source_group'].startswith('group-')
        assert draw.event.metadata['room_hash']==f.draws['room-A'].room.model_hash
        assert np.any(draw.event.audio[f.fs:])
        off=tn.prepare_context(f.corpus,np.random.default_rng(101),dict(f.cfg,p_confuser_given_target=0.,
            p_confuser_given_background=0.),f.fs,class_name=name,target_index=index)
        assert not off.provenance['injected'] and off.event is None


def test_capture_context_replaces_bed_and_keeps_native_level(factory):
    f=factory(stage='device_recorded');f.corpus.negative_pool=f.build()
    draw=tn.prepare_context(f.corpus,np.random.default_rng(11),f.cfg,f.fs,class_name='knock',target_index=0)
    assert draw.event is None and draw.background is not None
    parent=next(r for r in f.corpus.negative_pool.rows if r.item.item_id==draw.provenance['parent']['item_id'])
    # Exact original contiguous samples; no gain, second floor, or renderer.
    start=np.flatnonzero(parent.audio==draw.background[0])[0]
    np.testing.assert_array_equal(draw.background,parent.audio[start:start+2*f.fs])
    target=device.ReceiverEvent(np.zeros(2*f.fs),(f.fs,2*f.fs),70.)
    heard,_=tn.draw_training_context(np.ones(2*f.fs)*.01,target,np.random.default_rng(5),f.cfg,draw=draw)
    np.testing.assert_array_equal(heard,draw.background)
    with pytest.raises(ValueError,match='profile'):
        tn.prepare_context(f.corpus,np.random.default_rng(11),dict(f.cfg,device_profile='wrong'),f.fs,
                           class_name='knock',target_index=0)


def test_independent_linear_capture_oracle_and_bad_double_capture(monkeypatch):
    # Independent active-power oracle: full-span alternating +/-1 has power 1.
    # This is a route arithmetic control, not the current mixer used as its oracle.
    calls=[]
    class Linear:
        profile_id='linear-test';sensitivity_dbfs_at_94db_spl=-6.;max_spl_peak_db=None
        def with_variation(self,rng):return self
        def apply(self,x,fs,rng,**kw):
            calls.append(kw);return np.asarray(x)*.5
    monkeypatch.setitem(device.iot_mic.PROFILES,'linear-test',Linear())
    monkeypatch.setattr(device,'_check_recording',lambda *args:None)
    n=32;x=np.tile([-1.,1.],n//2);bed=np.arange(n)*1e-5
    target=device.ReceiverEvent(x,(0,n),94.)
    nuisance=device.ReceiverEvent(-x,(0,n),88.)
    cfg=dict(fs=16000,device_profile='linear-test',event_spl_db=[60.,80.])
    heard,floor=tn.draw_training_context(bed,target,np.random.default_rng(1),cfg,
        draw=tn.ContextDraw(event=nuisance),counterfactual=True)
    expected=bed+.5*x*(1-10**(-6/20))/np.sqrt(2)
    np.testing.assert_allclose(heard,expected,atol=1e-15)
    np.testing.assert_allclose(floor,bed-.5*x*10**(-6/20)/np.sqrt(2),atol=1e-15)
    assert len(calls)==2 and all(c['include_self_noise'] is False and c['quantize'] is False for c in calls)
    bad=.5*expected # designated erroneous second device capture
    assert np.max(np.abs(bad-expected)) > .08


def test_actual_target_room_not_merely_any_allowed_room(factory):
    f=factory();f.corpus.negative_pool=f.build()
    f.corpus.rows[0]['room_draw_id']='other-room'
    with pytest.raises(ValueError,match='selected target room'):
        tn.prepare_context(f.corpus,np.random.default_rng(1),f.cfg,f.fs,class_name='knock',target_index=0)


def test_dry_path_matches_production_room_render(factory):
    f=factory();pool=f.build();f.corpus.negative_pool=pool
    draw=tn.prepare_context(f.corpus,np.random.default_rng(91),f.cfg,f.fs,class_name='knock',target_index=0)
    row=next(r for r in pool.rows if r.item.item_id==draw.provenance['parent']['item_id'])
    wet=device.render_room_event(row.audio,fs=f.fs,draw=f.draws['room-A'],position=[1.,1.,1.],
                                 onset_s=0.,n=len(row.audio)+f.fs)
    assert draw.event.metadata['rir_sha256']==wet.metadata['rir_sha256']
    assert draw.event.metadata['rir_seed']==wet.metadata['rir_seed']
