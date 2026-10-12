# SPDX-License-Identifier: Apache-2.0
"""Independent review probes; original generated controls, never deployment evidence."""
import json
from dataclasses import asdict,replace
from types import SimpleNamespace
import numpy as np
import pytest
from sensorsim import training_negatives as tn
from sensorsim.negatives.splits import Item,SplitPlan
from sensorsim.negatives.mining import FrozenModel
from sensorsim.evaluation import fa_rate
from tests.unit.t3_negatives.conftest import factory,document,sha

DETECTOR=dict(hop_s=.25,on=.5,off_ratio=.6,smooth_hops=1,min_hops=2)

@pytest.mark.parametrize('shared_parent,shared_bed',[(False,False),(True,False),(False,True)])
def test_wp2_locked_source_group_controls_independent_cluster(factory,shared_parent,shared_bed,monkeypatch):
    f=factory(procedural=True,count=5)
    # Two different generated contents from the same supported procedural family.
    entries=[f.data['rows'][0],f.data['rows'][4]]
    parents=[]
    for index,entry in enumerate(entries):
        item=Item(**entry['item'])
        if shared_parent:item=replace(item,source_group='one-acquisition-parent')
        entry['item']=asdict(item);parents.append(item)
        review=json.loads((f.root/entry['semantic_review']['file']).read_text())
        review['parent']=asdict(item)
        entry['semantic_review']=document(f.root,'independent-review-'+str(index)+'.json',review)
    plan=SplitPlan.create(parents,audit_families=['hvac_fan'],development_families=[],seed=819)
    plan.save(f.root/'overlay-audit.json')
    (f.root/'room.json').write_text(f.draws['room-A'].room.model_dump_json())
    cache={};challenges=[]
    for index,entry in enumerate(entries):
        background=np.random.default_rng(600+index).normal(0,.001,4*f.fs)
        path=f'bed-{index}.wav';cache[(f.root/path).resolve()]=background
        bed=dict(path=path,sample_range=[0,len(background)],deployed_device=True,target_free=True,
            capture_device='generated verifier control',profile_id='pi5_inmp441/1',recording_id='same-bed' if shared_bed else f'bed-parent-{index}')
        source=dict(procedural=entry['source']['recipe'],whole_episode=True,onset_s=2.,source_xyz_m=[1.,1.,1.],
            spl_db=70.,locked_parent=entry['item'],semantic_review=entry['semantic_review'])
        challenges.append(dict(family='hvac_fan',overlay=dict(background=bed,events=[source],whole_episode=True,
            duration_s=4.,seed=820+index,start_s=0.)))
    manifest=dict(overlay_path=dict(room='room.json',receiver_xyz_m=[2.,2.,1.],device_id='pi5_inmp441/1',
            config=dict(max_order=0,late=False)),
        negative_parent_lock=dict(file='overlay-audit.json',sha256=sha(f.root/'overlay-audit.json'),target_scope=f.data['target_scope']),
        challenges=challenges)
    audit=SimpleNamespace(cache=cache,report=lambda:{'domain':'synthetic proxy'})
    model=SimpleNamespace(meta=dict(features=dict(fs=f.fs),detector=DETECTOR),features=SimpleNamespace(n=f.fs),
        classes=['knock'],scores=lambda audio:np.zeros((len(audio),1)))
    called=[]
    original=tn.validate_overlay_sources
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        called.append(result)
        return result
    monkeypatch.setattr(tn,'validate_overlay_sources',observed)
    rows,_=fa_rate._prepare_overlays(model,manifest,f.root,audit=audit)
    assert len(called)==len(rows)==2
    assert rows[0]['overlay_provenance'] and rows[1]['overlay_provenance']
    locked=[r['overlay_provenance']['negative_parent_lock']['parents'][0]['source_group'] for r in rows]
    observed=[r['parent_id'] for r in rows]
    print(json.dumps(dict(probe='source_parent_clustering',shared_parent=shared_parent,locked_groups=locked,
        output_clusters=observed,dependencies=[r['overlay_provenance']['dependency_ids'] for r in rows])))
    assert len(set(locked))==(1 if shared_parent else 2)
    assert len(set(observed))==(1 if shared_parent or shared_bed else 2), 'locked parent descendants counted as independent clusters'
