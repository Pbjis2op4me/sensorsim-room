"""Small development-only file/pool/panel integration; no acoustic acceptance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from sensorsim import nn, edge, training_selection
from sensorsim.negatives import procedural, integration
from sensorsim.negatives.composer import audio_hash
from sensorsim.negatives.splits import Item, SplitPlan


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root):
    root.mkdir(exist_ok=True)
    fs=16000; rng=np.random.default_rng(448)
    def write(name,x):
        x=np.asarray(x,np.float32);path=root/(name+".wav");wavfile.write(path,fs,x)
        return path.name,sha(path),audio_hash(x)
    rows=[]
    for split in ("train","val"):
        for label in ("doorbell","background"):
            f,_,_=write(split+label,rng.normal(0,.002 if label=="doorbell" else 1e-4,3*fs))
            rows.append(dict(file=f,label=label,split=split,fs=fs,source_group=split+label,room_draw_id="room1"))
    manifest=root/"manifest.jsonl";manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    source=procedural.generate("hvac_fan",fs=fs,duration_s=2,seed=223)
    x=source.audio*.01
    f,raw,h=write("fan",x);item=Item("fan","hvac_fan","train-fan",h)
    audit_item=Item("sealed","vacuum","sealed-fan","f"*64)
    plan=SplitPlan.create([item,audit_item],audit_families=["vacuum"],development_families=[],seed=1)
    plan.save(root/"split.json")
    negatives=root/"negative.json";negatives.write_text(json.dumps(dict(schema="negative-training/1",
        split_lock="split.json",split_file_sha256=sha(root/"split.json"),rows=[dict(file=f,file_sha256=raw,
        item=item.__dict__,room_draw_id="room1",provenance=dict(source.provenance,training_eligible=True,provenance="synthetic",
            acoustic_stage="receiver",license_id="Apache-2.0",weight=1.))])))
    panel_rows=[]
    for kind in ("target","nuisance","background"):
        f,raw,h=write("panel-"+kind,rng.normal(0,.001,3*fs))
        panel_rows.append(dict(audio=f,file_sha256=raw,audio_sha256=h,source_group="dev-"+kind,
            background_groups=["dev-bg-"+kind],parent_audio_sha256=[h],kind=kind,label="doorbell" if kind=="target" else None,
            family="water" if kind=="nuisance" else None,start_s=.5,end_s=2.5,evidence_domain="synthetic proxy"))
    panel=root/"panel.json";panel.write_text(json.dumps(dict(schema="training-panel/1",role="development",rows=panel_rows,
        sealed_audit=[dict(source_group="audit-target",audio_sha256="e"*64),dict(source_group="sealed-fan",audio_sha256="f"*64)])))
    return manifest,negatives,panel


def test_actual_train_builds_pool_selects_and_preserves_exact_detector(tmp_path,monkeypatch):
    manifest,negatives,panel=inputs(tmp_path)
    reached=[];original=integration.LockedEventPool.window
    def observed(self,*a,**kw):
        result=original(self,*a,**kw);reached.append(result is not None);return result
    monkeypatch.setattr(integration.LockedEventPool,"window",observed)
    def no_refit(*a,**kw): raise AssertionError("selected tracker must not be refitted")
    monkeypatch.setattr(nn,"_fit_tracker",no_refit)
    cfg=dict(steps=3,width=4,blocks=1,batch=8,workers=0,threads=1,val_windows=64,
        p_background_only=.8,p_confuser_given_background=1.,max_procedural_share=1.,
        negative_manifest=str(negatives),negative_manifest_sha256=sha(negatives),
        selection_panel=str(panel),selection_panel_sha256=sha(panel),selection_min_groups=1,
        selection_min_recall=0.,selection_trigger_budget=1.,
        selection_grid=[dict(on=.999999,smooth_hops=1,min_hops=1)],p_specaugment=0.,p_freq_mixstyle=0.)
    out=tmp_path/"model.npz";report=nn.train(manifest,out,**cfg)
    assert any(reached)
    model=edge.Model(str(out))
    assert model.classes==["doorbell"]
    assert model.meta["detector"]["on"]=={"doorbell":.999999}
    assert model.meta["detector"]["smooth_hops"]==1
    assert model.meta["training"]["negatives"]["families"][0]["family"]=="hvac_fan"
    assert report["selection"]["selected"]["step"]==3
    assert len(json.loads(out.with_suffix(".selection.json").read_text())["checkpoints"])==1
    assert model.meta["training"]["config"]["negative_manifest"]=="negative.json"


def test_panel_rejects_duplicate_content_parent_leak_and_missing_strata(tmp_path):
    manifest,negative,panel=inputs(tmp_path)
    original=json.loads(panel.read_text())
    excluded=training_selection.corpus_identities(manifest,16000,split="train")
    def load(): return training_selection.load_panel(panel,sha(panel),classes=["doorbell"],fs=16000,excluded=excluded,min_groups=1)
    assert len(load()["rows"])==3
    for mutate in (
        lambda d:d["rows"].append(dict(d["rows"][0],source_group="duplicate-other-name")),
        lambda d:d["rows"][0].update(background_groups=["trainbackground"]),
        lambda d:d["rows"].pop(1),
        lambda d:d["rows"][0].update(source_group="audit-target"),
        lambda d:d["rows"][0].update(role="audit"),
    ):
        data=json.loads(json.dumps(original));mutate(data);panel.write_text(json.dumps(data))
        with pytest.raises(ValueError): load()


def test_pool_loader_rejects_wrong_room_hash_licence_and_audit(tmp_path):
    _,path,_=inputs(tmp_path)
    original=json.loads(path.read_text())
    def load(): return integration.build_pool(path,sha(path),fs=16000,max_procedural_share=.2,allowed_room_draws={"room1"})
    assert load().metadata()["families"][0]["count"]==1
    for mutate in (lambda d:d["rows"][0].update(room_draw_id="wrong"),
                   lambda d:d["rows"][0].update(file_sha256="a"*64),
                   lambda d:d["rows"][0]["provenance"].update(license_id="CC-BY-NC-4.0"),
                   lambda d:d["rows"][0].update(item=dict(item_id="sealed",family="vacuum",source_group="sealed-fan",sha256="f"*64))):
        data=json.loads(json.dumps(original));mutate(data);path.write_text(json.dumps(data))
        with pytest.raises(ValueError): load()


def test_negative_pool_stream_is_serializable_without_changing_draws(tmp_path):
    import pickle
    manifest,negative,_=inputs(tmp_path)
    corpus=nn.Corpus(manifest,"train",None,16000)
    corpus.negative_pool=integration.build_pool(negative,sha(negative),fs=16000,max_procedural_share=.2,allowed_room_draws={"room1"})
    cfg=dict(nn.DEFAULTS,batch=8,event_aware_masks=True,_dynamics_report=True)
    data=nn.Batches(corpus,cfg,3,890)
    restored=pickle.loads(pickle.dumps(data))
    for a,b in zip(data[2],restored[2]):np.testing.assert_array_equal(a,b)


@pytest.mark.slow
def test_spawned_negative_pool_stream_matches_serial(tmp_path):
    import torch.multiprocessing as mp
    manifest,negative,_=inputs(tmp_path)
    corpus=nn.Corpus(manifest,"train",None,16000)
    corpus.negative_pool=integration.build_pool(negative,sha(negative),fs=16000,max_procedural_share=.2,allowed_room_draws={"room1"})
    cfg=dict(nn.DEFAULTS,batch=8,steps=2,event_aware_masks=True,_dynamics_report=True,_negative_report=True)
    serial=list(nn._training_batches(corpus,dict(cfg,workers=0),nn._torch()))
    previous=mp.get_start_method(allow_none=True);mp.set_start_method("spawn",force=True)
    try:parallel=list(nn._training_batches(corpus,dict(cfg,workers=1),nn._torch()))
    finally:mp.set_start_method(previous,force=True)
    for sa,sb in zip(serial,parallel):
        for a,b in zip(sa,sb):
            if isinstance(a,dict):assert a==b
            else:np.testing.assert_array_equal(a,b)
