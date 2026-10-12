"""Development controls: original procedural fixtures, never realism evidence."""
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from sensorsim import nn, training_dynamics
from sensorsim.negatives import integration, procedural
from sensorsim.negatives.composer import audio_hash
from sensorsim.negatives.splits import Item, SplitPlan


def row(family="hvac_fan", seed=1, weight=1., synthetic=True):
    source=procedural.generate(family,fs=16000,duration_s=1,seed=seed)
    x=source.audio
    return SimpleNamespace(audio=x,fs=16000,item=Item(str(seed),family,str(seed),audio_hash(x)),
        provenance=dict(source.provenance, training_eligible=True, provenance="synthetic" if synthetic else "user-owned",
                        acoustic_stage="receiver",weight=weight))


def plan_for(tmp_path, rows):
    p=SplitPlan.create([r.item for r in rows]+[Item("audit","sealed","sealed","f"*64)],
        audit_families=["sealed"],development_families=[],seed=1)
    p.save(tmp_path/"lock.json")
    return SplitPlan.load(tmp_path/"lock.json",expected_sha256=p.sha256)


def pool_for(tmp_path, rows, **kw):
    p=plan_for(tmp_path,rows)
    return integration.LockedEventPool(rows,plan=p,expected_sha256=p.sha256,purpose="train",**kw)


def receipt(tmp_path, r):
    values=dict(decision="pass",audio_sha256=r.item.sha256,family=r.item.family,
                reference_kind="recorded_family",reviewer="test control",reviewed_label_boundary=True)
    for key in ("reference","evidence"):
        path=tmp_path/key; path.write_bytes((key+" fixture").encode())
        values[key+"_file"]=str(path); values[key+"_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    r.provenance["realism_gate"]=values
    return values


def test_non_gated_families_are_admitted_without_review(tmp_path):
    pool=pool_for(tmp_path,[row()])
    assert np.any(pool.window(np.random.default_rng(1),16000,fs=16000,purpose="train")[0])


def test_receipts_bind_actual_files_and_recheck(tmp_path):
    r=row("hammering"); gate=receipt(tmp_path,r); pool=pool_for(tmp_path,[r])
    pool.window(np.random.default_rng(1),16000,fs=16000,purpose="train")
    from pathlib import Path
    Path(gate["evidence_file"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash"):
        pool.window(np.random.default_rng(1),16000,fs=16000,purpose="train")


def test_mining_weights_and_procedural_share_are_consumed(tmp_path):
    rows=[row(seed=1,weight=1,synthetic=False),row(seed=2,weight=9,synthetic=False),row(seed=3,weight=100)]
    pool=pool_for(tmp_path,rows,max_procedural_share=.2)
    np.testing.assert_allclose(pool.probabilities,[.08,.72,.2])
    assert pool.metadata()["families"][0]["count"]==3


def test_pure_procedural_pool_keeps_unused_share_as_plain_background(tmp_path):
    pool=pool_for(tmp_path,[row()],max_procedural_share=.2)
    assert pool.probabilities.sum()==pytest.approx(.2)
    assert pool.metadata()["plain_background_probability"]==pytest.approx(.8)


@pytest.mark.parametrize("knobs",[{},{"partial_windows":True},{"audibility_gate":True},{"low_event_spl_tail":True}])
def test_zero_label_confuser_reaches_shared_mixer_in_all_routes(tmp_path,monkeypatch,knobs):
    pool=pool_for(tmp_path,[row()],max_procedural_share=1.)
    c=SimpleNamespace(classes=["knock"],background=[np.arange(48000)*1e-10],targets={"knock":[np.ones(4000)]},
        spans={"knock":[(0,4000)]},negative_pool=pool,pool_purpose="train")
    seen=[]
    def mix(b,e,span,rng,cfg):
        seen.append((b.copy(),e.copy(),span)); return b+e
    monkeypatch.setattr(nn,"mix_device_audio",mix)
    cfg=dict(nn.DEFAULTS,p_background_only=1.,p_confuser_given_background=1.,**knobs)
    out,y=nn.make_example(c,np.random.default_rng(5),cfg)
    assert len(seen)==1 and np.any(seen[0][1]) and not y.any()
    np.testing.assert_allclose(out,(seen[0][0]+seen[0][1])[16000:].astype(np.float32))
    seen.clear()
    nn.make_example(c,np.random.default_rng(5),dict(cfg,p_confuser_given_background=0.))
    assert not seen[0][1].any()


def test_tonal_exclusion_is_fifteen_percent():
    for seed in range(20):
        x=procedural.generate("appliance_beeps",fs=16000,duration_s=1,seed=seed,target_partials_hz=[880.])
        assert not 880*.85 <= x.provenance["parameters"]["base_hz"] <= 880*1.15


def test_tonal_admission_needs_separate_heldout_recall(tmp_path):
    r=row("appliance_beeps"); receipt(tmp_path,r)
    r.provenance["parameters"].update(base_hz=650.,target_partials_hz=[880.])
    with pytest.raises(ValueError,match="doorbell"):
        pool_for(tmp_path,[r])


@pytest.mark.parametrize("audio",[np.zeros(16000),np.ones(16000),np.array([])])
def test_empty_constant_confusers_cannot_pass_a_gate(tmp_path,audio):
    r=row(); r.audio=audio; r.item=Item("1",r.item.family,"1",audio_hash(audio))
    with pytest.raises(ValueError,match="constant|empty|silent"):
        pool_for(tmp_path,[r])


def test_reviewed_mining_window_weight_and_calibration_exclusion(tmp_path):
    from sensorsim.negatives import mining
    from dataclasses import replace
    from sensorsim.negatives.composer import digest
    base=row("room_tone",1); mined=row("hvac_fan",2)
    p=SplitPlan.create([base.item,mined.item,Item("audit","vacuum","audit","f"*64)],
        audit_families=["vacuum"],development_families=["hvac_fan"],seed=1)
    lock=tmp_path/"lock.json";p.save(lock);file_hash=hashlib.sha256(lock.read_bytes()).hexdigest()
    p=SplitPlan.load(lock,expected_sha256=p.sha256)
    model_file=tmp_path/"checkpoint";model_file.write_bytes(b"test scorer control")
    called=[]
    model=mining.FrozenModel(model_file,lambda x:called.append(len(x)) or [.9],classes=["knock"],fs=16000)
    source=mining.PoolItem(mined.item,mined.audio,16000,dict(role="calibration"))
    with pytest.raises(ValueError,match="calibration"):
        mining.mine([source],model=model,split=lock,expected_split_sha256=file_hash)
    assert not called
    candidates=mining.mine([replace(source,provenance={"role":"mining"})],model=model,split=lock,expected_split_sha256=file_hash)
    reviews={c.review_id:dict(decision="confirmed_negative",reviewer="fixture",notes="generated negative control") for c in candidates}
    additions=mining.reviewed_bank(candidates,reviews,split=lock,expected_split_sha256=file_hash,base_training_weight=1.)
    assert len(additions)==1 and additions[0]["weight"]==.05
    mined.provenance.update(mined=True,role="mining",mining_candidate=additions[0],
        weight=additions[0]["weight"],mining_review=additions[0]["review"])
    pool=integration.LockedEventPool([base,mined],plan=p,expected_sha256=p.sha256,purpose="train")
    np.testing.assert_allclose(pool.weights,[1.,.05])
    # Changing the reviewed interval/weight cannot leave the claim intact.
    mined.provenance["mining_candidate"]["end_sample"]-=1
    with pytest.raises(ValueError,match="identity"):
        pool._authorize(mined)


def test_doorbell_known_correct_and_known_bad_recall_receipts(tmp_path):
    import json
    r=row("appliance_beeps"); receipt(tmp_path,r)
    r.provenance["parameters"].update(base_hz=650.,target_partials_hz=[880.])
    path=tmp_path/"recall.json"
    data=dict(claim="heldout-doorbell-recall/1",units="independent_source_parent",
        source_groups=[f"held-{i}" for i in range(20)],training_source_groups=["train"],
        baseline_detected=[True]*20,candidate_detected=[True]*20,confuser_audio_sha256=[r.item.sha256],
        baseline_model_sha256="a"*64,candidate_model_sha256="b"*64)
    for key in ("baseline_model", "candidate_model"):
        model_file=tmp_path/key;model_file.write_bytes((key+" fixture").encode())
        data[key+"_file"]=model_file.name;data[key+"_sha256"]=hashlib.sha256(model_file.read_bytes()).hexdigest()
    def bind():
        path.write_text(json.dumps(data));r.provenance["doorbell_recall"]=dict(file=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reviewer="fixture",review_decision="accept")
    bind();integration.require_realism(r)
    data["candidate_detected"]=[True]*18+[False]*2;bind()
    with pytest.raises(ValueError,match="doorbell"):integration.require_realism(r)
    data["candidate_detected"]=[True]*20;data["source_groups"]=["duplicate"]*20;bind()
    with pytest.raises(ValueError,match="doorbell"):integration.require_realism(r)


def test_locked_calibration_partition_cannot_be_mined_even_with_forged_role():
    item=Item("cal","hvac_fan","cal","a"*64)
    p=SplitPlan.create([item,Item("audit","vacuum","audit","b"*64)],audit_families=["vacuum"],
        development_families=[],calibration_families=["hvac_fan"],seed=1)
    p.require(item,purpose="calibrate")
    for purpose in ("mine","train","tune"):
        with pytest.raises(ValueError,match="not authorized"):p.require(item,purpose=purpose)


def test_negative_ablation_keeps_positive_draws_and_caller_rng_paired(tmp_path):
    pool=pool_for(tmp_path,[row()],max_procedural_share=1.)
    c=SimpleNamespace(classes=["knock"],background=[np.random.default_rng(1).normal(0,1e-4,48000)],
        targets={"knock":[np.sin(np.arange(8000)*.3)]},spans={"knock":[(0,8000)]},negative_pool=pool,pool_purpose="train")
    a=np.random.default_rng(48);b=np.random.default_rng(48);changed=0
    for _ in range(40):
        off,yo=nn.make_example(c,a,dict(nn.DEFAULTS,p_confuser_given_background=0.))
        on,yn=nn.make_example(c,b,dict(nn.DEFAULTS,p_confuser_given_background=1.))
        np.testing.assert_array_equal(yo,yn)
        if yo.any(): np.testing.assert_array_equal(off,on)
        else: changed += int(not np.array_equal(off,on))
    assert changed>0
    assert repr(a.bit_generator.state)==repr(b.bit_generator.state)
