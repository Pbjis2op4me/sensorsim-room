# SPDX-License-Identifier: Apache-2.0
import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim import nn, iot_mic, recognizer
from tools import dynamics_sweep as sw


def corpus(long=False):
    r = np.random.default_rng(4)
    names = ['knock', 'doorbell', 'kettle']
    lengths = [1600, 32000, 960000 if long else 32000]
    xs = [r.normal(0, .03, n) for n in lengths]
    return SimpleNamespace(classes=names, targets={k:[x] for k,x in zip(names,xs)},
                           spans={k:[(0,len(x))] for k,x in zip(names,xs)},
                           background=[r.normal(0, 1e-5, 48000)])


def test_t1_observes_real_floor_and_device():
    c=corpus(); cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),p_background_only=0,p_mic_chain=1)
    a, y, t=sw.trace_example(c,np.random.default_rng(42),cfg)
    assert 'floor_full' in t
    assert t['hook']=='mix_device_audio'
    assert t['chain']['profile_id']=='pi5_inmp441/1'
    from test_integration_oracles import decomposition_error
    assert decomposition_error(42) < 1e-3


def test_t1_missing_hook_raises():
    with patch.object(nn,'make_example',return_value=(np.zeros(16000),np.ones(3))):
        with pytest.raises(RuntimeError,match='hook'):
            sw.trace_example(corpus(),np.random.default_rng(2),dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),snr_db=[3,3]))


def test_t2_clicks_are_floor_relative_at_quiet_levels():
    x=np.zeros(16000); x[8000:]=1e-4
    assert sw.fingerprints(x,16000,[8000],baseline_difference_rms=1e-6)['boundary_click_candidates']==1
    assert sw.fingerprints(x*.01,16000,[8000],baseline_difference_rms=1e-8)['boundary_click_candidates']==1


def test_t3_dc_is_separate_from_acoustic_level():
    x=.001*np.sin(2*np.pi*1000*np.arange(16000)/16000)
    assert sw.levels(x+.05,16000)['rms_dbfs']==pytest.approx(sw.levels(x,16000)['rms_dbfs'])
    assert sw.levels(x+.05,16000)['dc_offset_fs']==pytest.approx(.05)


def test_t4_input_qa_reports_clipped_input():
    qa=sw.input_qa(np.array([0.,1.,-1.]),16000)
    assert qa['clipped_samples']==2
    assert qa['peak_fs']==1.


def test_quantization_error_is_measured_after_device_chain():
    c=corpus(); cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),p_background_only=0,snr_db=[3,3])
    x=c.targets['knock'][0]; ideal=x+1e-6*np.sin(np.arange(len(x)))
    c.classes=['knock']
    r=sw.post_chain_quantization(c,ideal,43,cfg)
    a,_,_=sw.trace_example(c,np.random.default_rng(43),cfg)
    ci=copy.deepcopy(c);ci.targets['knock']=[ideal]
    b,_,_=sw.trace_example(ci,np.random.default_rng(43),cfg)
    assert r['post_capture_source_difference_a_sine_dbfs']==pytest.approx(sw.levels(a.astype(float)-b,16000)['a_sine_dbfs'])


def test_partial_60s_kettle_preserves_labels_and_device_path():
    c=corpus(True); cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),p_background_only=0)
    draws=[]
    def mixer(bg,event,span,rng,cfg):
        draws.append((cfg.get('device_split'),cfg.get('event_spl_db')))
        return bg+event
    counts=np.zeros(3,int)
    with patch.object(nn,'mix_device_audio',side_effect=mixer):
        for seed in range(40):
            a,y,ctx=nn.make_example(c,np.random.default_rng(seed),dict(cfg,partial_windows=True),return_context=True)
            assert ctx['retained_fraction']>=.1
            assert np.isfinite(a).all()
            counts+=y.astype(int)
    assert (counts>0).all()
    assert all(d[0]=='in_distribution' for d in draws)


def test_retries_pin_class_source_and_mixer_rng_and_report_rejections():
    from sensorsim import training_dynamics as td
    c=corpus(True); cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),p_background_only=0,partial_windows=True,audibility_gate=True)
    states=[]
    original=nn.mix_device_audio
    def mix(bg,event,span,rng,cfg):
        states.append(copy.deepcopy(rng.bit_generator.state))
        return original(bg,event,span,rng,cfg)
    with patch.object(nn,'mix_device_audio',side_effect=mix), patch.object(td,'eligible',side_effect=[False,False,True]):
        _,y,ctx=nn.make_example(c,np.random.default_rng(19),cfg,return_context=True)
    assert ctx['rejections']==2 and ctx['attempts']==3
    assert ctx['class_name']==c.classes[int(y.argmax())]
    assert all(s==states[0] for s in states)


def test_validation_batches_never_mask_or_use_training_knobs():
    c=corpus(); cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),batch=4)
    base=nn.Batches(c,cfg,1,99,training=False)[0]
    changed=nn.Batches(c,dict(cfg,event_aware_masks=True,partial_windows=True,low_event_spl_tail=True),1,99,training=False)[0]
    for a,b in zip(base,changed):np.testing.assert_array_equal(a,b)


def test_mask_width_label_independent_and_zero_probability_no_rng():
    r=np.random.default_rng(1); state=copy.deepcopy(r.bit_generator.state)
    nn._spec_augment(np.ones((4,1,40,97)),r,0)
    assert r.bit_generator.state==state
    context={'event':np.ones((40,97)), 'floor':np.ones((40,97))*.01,'positive':True}
    contexts=[dict(context,class_name=c) for c in ['knock','kettle','background']]
    x=nn._spec_augment(np.ones((3,1,40,97)),np.random.default_rng(2),1,contexts=contexts)
    assert all(np.all(v[0]==0,axis=0).sum()<=3 for v in x)


def test_mixstyle_restricts_label_peers():
    torch=nn._torch(); torch.manual_seed(0)
    x=torch.from_numpy(np.random.default_rng(2).normal(size=(3,1,4,10)).astype(np.float32))
    y=np.eye(3,dtype=np.float32)
    out=nn._freq_mixstyle(x,torch,.4,labels=y)
    np.testing.assert_allclose(out.numpy(),x.numpy(),atol=5e-7)


def test_background_float_storage_provenance(tmp_path):
    import json
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps({'fs':16000,'label':'knock','split':'train','file':'x.wav'})+'\n')
    x=np.full(16000,1e-7,np.float32); wavfile.write(tmp_path/'raw.wav',16000,x)
    recognizer.add_background(manifest,[tmp_path/'raw.wav'],storage='float32')
    row=json.loads(manifest.read_text().splitlines()[-1]); _,out=wavfile.read(tmp_path/row['file'])
    np.testing.assert_array_equal(out,x)
    assert row['storage']=='float32' and row['quantization_error_bound_fs'] is None


def test_truth_proxy_has_no_extra_geometric_gain_and_refuses_clipping():
    from tools.prepare_dynamics_proxy import truth_rir, require_unclipped
    pra=pytest.importorskip('pyroomacoustics')
    room=pra.ShoeBox([5,4,3],fs=16000,max_order=0)
    room.add_source([1,2,1]);room.add_microphone_array(np.array([[2],[2],[1]]));room.compute_rir()
    h=truth_rir(room)
    assert .7 < np.max(np.abs(h)) < 1.1
    with pytest.raises(ValueError,match='clipped'):
        require_unclipped(np.array([1.2]),'test')


def test_partial_phase_and_retry_rng_do_not_shift_later_class_or_level_draws():
    c=corpus(True); cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}))
    r0,r1=np.random.default_rng(55),np.random.default_rng(55)
    phases=set()
    for _ in range(60):
        _,y0=nn.make_example(c,r0,cfg)
        _,y1,ctx=nn.make_example(c,r1,dict(cfg,partial_windows=True),return_context=True)
        np.testing.assert_array_equal(y0,y1)
        assert r0.bit_generator.state==r1.bit_generator.state
        if ctx['source_start'] is not None:phases.add(ctx['source_start']%4000)
    assert len(phases)>20


def test_exhausted_gate_is_source_qa_error_not_a_new_label():
    from sensorsim import training_dynamics as td
    with patch.object(td,'eligible',return_value=False):
        with pytest.raises(ValueError,match='class and level were not redrawn'):
            nn.make_example(corpus(),np.random.default_rng(3),dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),p_background_only=0,audibility_gate=True))


def test_hard_validation_and_balance_are_opt_in():
    from sensorsim import training_dynamics as td
    cfg=td.validation_config(dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),validation_mode='hard'))
    assert cfg['event_spl_db']==[45.,65.] and cfg['p_background_only']==.8
    batches=nn.Batches(corpus(),dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),positive_fraction=.3),1,7)
    assert batches.cfg['p_background_only']==pytest.approx(.7)


def test_rejected_zero_overlap_cannot_shift_next_selection_rng():
    x=np.zeros(8);x[4]=.1
    c=SimpleNamespace(classes=['knock'],targets={'knock':[x]},spans={'knock':[(0,8)]},
                      background=[np.full(100,1e-5)])
    cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),window_s=.001,p_background_only=0)
    for seed in range(50):
        a,b=np.random.default_rng(seed),np.random.default_rng(seed)
        nn.make_example(c,a,cfg)
        nn.make_example(c,b,dict(cfg,partial_windows=True))
        assert a.bit_generator.state==b.bit_generator.state


def test_hard_validation_raises_bed_level_without_mutating_training():
    c=corpus();old=c.background[0].copy()
    cfg=dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}),batch=4,validation_mode='hard')
    val=nn.Batches(c,cfg,1,99,training=False)
    np.testing.assert_allclose(val.corpus.background[0],old*10**(15/20))
    np.testing.assert_array_equal(c.background[0],old)
    train=nn.Batches(c,cfg,1,99,training=True)
    np.testing.assert_array_equal(train.corpus.background[0],old)
    assert val.cfg['event_aware_masks'] is False


def test_component_level_ratio_removes_dc_too():
    x=.001*np.sin(2*np.pi*1000*np.arange(16000)/16000)
    f=sw.edge.Features(nn._feature_cfg((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})))
    plain=sw.audibility(x,x*.1,f)
    offset=sw.audibility(x+.03,x*.1+.02,f)
    assert offset['event_to_floor_db']==pytest.approx(plain['event_to_floor_db'])


@pytest.mark.parametrize('license_text', ['Creative Commons Attribution NonCommercial 4.0 International',
                                        'Creative Commons Attribution ShareAlike 4.0 International',
                                        'Attribution 4.0 terms unclear'])
def test_proxy_preparation_rejects_unapproved_licenses_before_reading_data(tmp_path,license_text):
    from tools.prepare_dynamics_proxy import prepare
    record=tmp_path/'LICENSE';record.write_text(license_text)
    with pytest.raises(ValueError,match='CC BY 4.0'):
        prepare(tmp_path/'out',tmp_path/'not-read.zip',record)


def test_pi_sweep_cell_names_the_actual_spl_not_an_unused_snr(tmp_path):
    import csv,hashlib,json
    rng=np.random.default_rng(7)
    wavfile.write(tmp_path/'target.wav',16000,rng.normal(0,.01,16000).astype(np.float32))
    wavfile.write(tmp_path/'floor.wav',16000,rng.normal(0,1e-5,48000).astype(np.float32))
    cfg={'cases':[{'id':'p','source_group':'p','label':'knock','evidence_kind':'model-based',
                   'target':'target.wav','floor':'floor.wav'}],
         'profiles':['default_pi'],'seeds':[7],'snr_db':[3],'event_spl_db':[60],'augmentations':['off']}
    (tmp_path/'config.json').write_text(json.dumps(cfg));(tmp_path/'prereg').write_text('{}')
    (tmp_path/'hash').write_text(hashlib.sha256(b'{}').hexdigest())
    sw.run(tmp_path/'config.json',tmp_path/'prereg',tmp_path/'hash',tmp_path/'output')
    row=next(csv.DictReader((tmp_path/'output/cells.csv').open()))
    assert float(row['level_cell'])==60.


def test_proxy_tools_ship_no_background_levels(tmp_path):
    from tools.prepare_dynamics_proxy import prepare as prepare_proxy
    from tools.prepare_dynamics_screen import prepare as prepare_screen
    with pytest.raises(ValueError, match='ships no measured household levels'):
        prepare_proxy(tmp_path/'proxy')
    with pytest.raises(ValueError, match='supply 3 A-weighted'):
        prepare_screen(tmp_path/'screen', bed_dba=(40., 45.))
    assert not (tmp_path/'proxy').exists() and not (tmp_path/'screen').exists()
