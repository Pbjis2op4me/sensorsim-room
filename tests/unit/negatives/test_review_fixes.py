"""Review regressions; all sounds and ownership records are generated controls."""
from pathlib import Path
from dataclasses import replace
from datetime import date
import hashlib
import json
import numpy as np
import pytest
import yaml
from sensorsim import iot_mic
from sensorsim.negatives.procedural import generate, EVENT_FAMILIES, BED_FAMILIES
from sensorsim.negatives.composer import Event, RenderPath, compose, captured_context, render_context
from sensorsim.negatives.clips import load_clip
from sensorsim.negatives.splits import SplitPlan, Item
from sensorsim.negatives.mining import FrozenModel, PoolItem, mine, reviewed_bank
from sensorsim.training_device import mix_device_audio
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig
from sensorsim.negatives.composer import audio_hash


def path():
    return RenderPath(RoomModel(shoebox_dims_m=(6, 5, 3)), (4, 3, 1.5),
                      iot_mic.PROFILES['pi5_inmp441/1'], 'pi5_inmp441/1',
                      SynthConfig(late=False, max_order=1), input_reference_spl_db=94.)


def test_no_family_aliases():
    families = EVENT_FAMILIES + BED_FAMILIES
    for seed in (1, 2, 3, 77):
        hashes = [audio_hash(generate(f, fs=16000, duration_s=1, seed=seed).audio) for f in families]
        assert len(set(hashes)) == len(families)


def test_impacts_have_broadband_body_and_jitter():
    for family in ('door_closure', 'hammering'):
        src = generate(family, fs=16000, duration_s=1, seed=6)
        power = abs(np.fft.rfft(src.audio*np.hanning(len(src.audio)), 32768))**2
        f = np.fft.rfftfreq(32768, 1/16000)
        p = power[(f >= 60) & (f <= 7600)] + 1e-20
        assert np.exp(np.log(p).mean())/p.mean() > .1
    onsets = generate('hammering', fs=16000, duration_s=1, seed=6).provenance['parameters']['onsets_s']
    assert np.std(np.diff(onsets))/np.mean(np.diff(onsets)) > .01


@pytest.mark.parametrize('family', ['tv_jingle', 'appliance_beeps', 'phone_notification', 'timer', 'alarm_chirp'])
def test_all_doorbell_like_families_require_review(family):
    with pytest.raises(ValueError, match='reviewed'):
        compose(path=path(), family_rates_per_hour={family: 3600}, positions={}, seed=1)


def test_padding_does_not_change_active_level():
    t = np.arange(4000)/16000
    short = np.sin(2*np.pi*700*t)*np.exp(-t/.03)
    cfg = dict(fs=16000, event_spl_db=[70,70], background_domain='device_recorded')
    # Same lead-in, same physical source; arbitrary trailing zeros have no level effect.
    outs=[]
    for size in (8000, 24000):
        ev=np.zeros(size); ev[2000:6000]=short
        bg=np.full(size, 32/32768)
        outs.append(mix_device_audio(bg,ev,(0,size),np.random.default_rng(42),cfg))
    np.testing.assert_array_equal(outs[0],outs[1][:8000])


def test_conditional_capture_is_unbiased_and_empty_event_exact():
    from sensorsim.training_device import conditional_capture
    rng=np.random.default_rng(1)
    bg=rng.integers(-100,101,200000)/32768
    inc=np.full(len(bg),.3/32768)
    profile=iot_mic.PROFILES['pi5_inmp441/1']
    y=conditional_capture(bg,inc,np.random.default_rng(2),profile)
    assert abs(np.mean((y-bg)*32768)-.3)<.004
    np.testing.assert_array_equal(conditional_capture(bg,inc*0,rng,profile),bg)
    np.testing.assert_array_equal(y, bg+inc)  # No second quantization or dither.


@pytest.mark.parametrize('kind', ['silent','below','format'])
def test_bad_device_recordings_warn(kind):
    bg={'silent':np.zeros(16000),'below':np.full(16000,1/32768),
        'format':np.full(16000,.0012345)}[kind]
    with pytest.warns(UserWarning):
        mix_device_audio(bg,np.zeros_like(bg),(0,len(bg)),np.random.default_rng(1),{'fs':16000})


def test_recorded_background_accepts_events_without_recapture():
    p=path(); b=generate('room_tone',fs=16000,duration_s=10,seed=1)
    pcm=iot_mic.PROFILES[p.device_id].apply(b.audio*.0001,16000,np.random.default_rng(1),input_reference_spl_db=94)
    src=replace(b,audio=pcm,provenance=b.provenance|{'acoustic_stage':'captured',
                'capture_device':'my-pi','profile_id':p.device_id,'deployed_device':True,'native_s16':True})
    np.testing.assert_array_equal(captured_context(src,fs=16000,path=p,seed=9).audio,pcm)
    event=Event(generate('hammering',fs=16000,duration_s=1,seed=8),2,(1,1,1),reviewed_label_boundary=True)
    ctx=captured_context(src,fs=16000,path=p,seed=9,events=[event])
    assert ctx.provenance['background_domain']=='device_recorded'
    assert not np.array_equal(ctx.audio,pcm)
    np.testing.assert_array_equal(ctx.audio[:32000],pcm[:32000])


def test_external_capture_requires_provenance_and_chain():
    p=path(); s=generate('room_tone',fs=16000,duration_s=10,seed=1)
    s=replace(s,audio=s.audio*.001,provenance=s.provenance|{'acoustic_stage':'captured'})
    with pytest.raises(ValueError,match='capture'):
        captured_context(s,fs=16000)
    s=replace(s,provenance=s.provenance|{'capture_device':'external-recorder','profile_id':'external/1','deployed_device':False})
    with pytest.raises(ValueError,match='reference'):
        captured_context(s,fs=16000,path=p,seed=1)
    ctx=captured_context(s,fs=16000,path=p,seed=1,cfg={'captured_reference_spl_db':70.})
    assert ctx.provenance['device_applied']
    assert ctx.provenance['evidence_label']=='proxy'
    assert ctx.provenance['domain']=='external_capture_stress'


def test_continuous_kettle_confuser_and_assumption_provenance():
    ctx=compose(path=path(),family_rates_per_hour={'running_tap':3600},
                positions={'room_tone':((1,1,1),),'running_tap':((2,2,1),)},seed=2)
    durations=[r['source']['parameters']['duration_s'] for r in ctx.events if r['role']=='event']
    assert max(durations)>1.5
    assert all('assumption' in r['source']['range_basis'] for r in ctx.events)
    s=generate('fridge_hum',fs=16000,duration_s=1,seed=1)
    assert 'assumption' in s.provenance['parameters']['energy_split_basis']


def mining_fixture(tmp_path):
    x=np.sin(np.arange(48000)*.1)*.1
    dev=Item('dev','timer','take-dev',audio_hash(x)); au=Item('audit','vacuum','take-audit','a'*64)
    tr=Item('train','hammering','take-train','b'*64)
    p=SplitPlan.create([dev,au,tr],audit_families=['vacuum'],development_families=['timer'],seed=1)
    p.save(tmp_path/'split.json')
    p=SplitPlan.load(tmp_path/'split.json',expected_sha256=p.sha256)
    ck=tmp_path/'ck';ck.write_text('model')
    m=FrozenModel(ck,lambda x:[.9],classes=['knock'],fs=16000)
    return p,PoolItem(dev,x,16000),m


def test_rebuilt_plan_cannot_unlock_audit(tmp_path):
    p,pool,m=mining_fixture(tmp_path)
    items=[i for i,_ in p.rows]
    rebuilt=SplitPlan.create(items,audit_families=['hammering'],development_families=['timer','vacuum'],seed=1)
    with pytest.raises(ValueError,match='persist|lock'):
        mine([pool],model=m,split=rebuilt,expected_split_sha256=rebuilt.sha256)


def test_caps_scale_and_one_checkpoint_per_bank(tmp_path):
    p,pool,m=mining_fixture(tmp_path)
    candidates=mine([pool],model=m,**mining_lock(p))
    reviews={c.review_id:{'decision':'confirmed_negative','reviewer':'fixture','notes':'fixture'} for c in candidates}
    totals=[]
    for base in (1.,1000.):
        bank=reviewed_bank(candidates,reviews,**mining_lock(p),base_training_weight=base)
        totals.append(sum(r['weight'] for r in bank)/base)
    assert totals[0]==pytest.approx(totals[1])
    m.checkpoint.write_text('other model')
    m2=FrozenModel(m.checkpoint,lambda x:[.9],classes=['knock'],fs=16000)
    other=mine([pool],model=m2,**mining_lock(p))
    with pytest.raises(ValueError,match='checkpoint|model'):
        reviewed_bank(candidates+other,reviews,**mining_lock(p),base_training_weight=1)


def test_user_ownership_does_not_expire(tmp_path):
    from scipy.io import wavfile
    from sensorsim.negatives.clips import user_register_row
    file=tmp_path/'own.wav';wavfile.write(file,16000,np.ones(16000,dtype=np.int16))
    item=dict(path='own.wav',sha256=hashlib.sha256(file.read_bytes()).hexdigest(),registry_id='own',
              license_id='user-owned',verified_on='2026-10-08',owner='fixture',ownership_statement='generated')
    row=user_register_row(item,name='Own recording',evidence_url='https://example.test/ownership')
    assert {'name','url'}<=row.keys()
    reg=tmp_path/'register.yaml';reg.write_text(yaml.safe_dump({'assets':[row]}))
    review=dict(family='timer',source_group='take',acoustic_stage='dry',listened=True,target_free=True,
                rights_cleared=True,reviewer='fixture',notes='generated')
    src=load_clip(tmp_path,item,register_path=reg,review=review,as_of=date(2027,10,8))
    assert src.provenance['license_id']=='user-owned'


def test_integration_checks_content_not_self_declared_split(tmp_path):
    from sensorsim.negatives.integration import authorize_audio, LockedEventPool
    p,pool,m=mining_fixture(tmp_path)
    authorize_audio(pool.audio,pool.item,plan=p,expected_sha256=p.sha256,purpose='tune')
    with pytest.raises(ValueError):
        authorize_audio(pool.audio,pool.item,plan=p,expected_sha256=p.sha256,purpose='train')
    with pytest.raises(ValueError,match='content'):
        authorize_audio(pool.audio*.5,pool.item,plan=p,expected_sha256=p.sha256,purpose='tune')
    pool=replace(pool,provenance={**pool.provenance,'family':pool.item.family,'training_eligible':True,'realism_gate':{
        'decision':'pass','family':pool.item.family,'audio_sha256':pool.item.sha256,
        'reference_kind':'recorded_family','reviewer':'fixture control only',
        'evidence_sha256':'a'*64,'reference_sha256':'b'*64,'reviewed_label_boundary':True}})
    for key in ("evidence", "reference"):
        evidence = tmp_path/key
        evidence.write_bytes((key+" fixture").encode())
        pool.provenance["realism_gate"][key+"_file"] = str(evidence)
        pool.provenance["realism_gate"][key+"_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    recall = tmp_path/"recall.json"
    recall_data=dict(claim="heldout-doorbell-recall/1", units="independent_source_parent",
        source_groups=[f"held-{i}" for i in range(10)], training_source_groups=["training"],
        baseline_detected=[True]*10, candidate_detected=[True]*10,
        confuser_audio_sha256=[pool.item.sha256], baseline_model_sha256="a"*64, candidate_model_sha256="b"*64)
    for key in ("baseline_model", "candidate_model"):
        model_file=tmp_path/key;model_file.write_bytes((key+" fixture").encode())
        recall_data[key+"_file"]=model_file.name;recall_data[key+"_sha256"]=hashlib.sha256(model_file.read_bytes()).hexdigest()
    recall.write_text(json.dumps(recall_data))
    pool.provenance["doorbell_recall"] = dict(review_decision="accept",reviewer="fixture",
        file=str(recall),sha256=hashlib.sha256(recall.read_bytes()).hexdigest())
    bank=LockedEventPool([pool],plan=p,expected_sha256=p.sha256,purpose='tune')
    window,span=bank.window(np.random.default_rng(0),16000,fs=16000,purpose='tune')
    assert window.shape==(16000,) and span[1]>span[0]


@pytest.mark.parametrize('family', ['tv_jingle', 'other_doorbell', 'timer'])
def test_low_level_negative_render_cannot_bypass_boundary(family):
    source=generate(family,fs=16000,duration_s=1,seed=8)
    with pytest.raises(ValueError,match='reviewed'):
        render_context([Event(source,1,(1,1,1))],path=path(),seed=9)


def mining_lock(plan):
    # The study pins these arguments at registration, before mining any scores.
    path = Path(plan._lock_path)
    return dict(split=path, expected_split_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
