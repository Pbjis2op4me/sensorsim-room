"""Development fixtures: original generated samples, no actual user/media assets."""
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim import nn, training_negatives as tn
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig
from sensorsim.negatives.composer import audio_hash
from sensorsim.negatives.splits import Item, SplitPlan
from sensorsim.training_device import RoomDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(root, name, value):
    path = root/name
    path.write_text(json.dumps(value, indent=2)+'\n')
    return dict(file=name, sha256=sha(path))


@pytest.fixture
def factory(tmp_path):
    counter = 0
    def make(*, count=4, procedural=False, role='training_base', stage='dry', weight=None):
        nonlocal counter
        root=tmp_path/str(counter);root.mkdir();counter+=1
        fs=16000
        room=RoomModel.model_validate(dict(shoebox_dims_m=[4.,5.,3.]))
        draws={'room-A':RoomDraw(room,(2.,2.,1.),SynthConfig(max_order=0,late=False),3)}
        entries=[];parents=[]
        for i in range(count):
            family=('hvac_fan','running_tap','fridge_hum','extractor_hood')[i%4]
            if procedural:
                from sensorsim.negatives.procedural import generate
                recipe=dict(family=family,seed=i+1,duration_s=.15)
                x=generate(fs=fs,**recipe).audio
                source=dict(kind='procedural',recipe=recipe)
            else:
                # Exercise the recording loader using generated development controls.
                # The grants below authorize only these project-created test samples.
                x=np.random.default_rng(3100+i).normal(0,4e-4,fs*3).astype(np.float32)
                filename=f'parent-{i}.wav';wavfile.write(root/filename,fs,x)
                asset=dict(path=filename,registry_id=f'asset-{i}',sha256=sha(root/filename),
                    license_id='user-owned',verified_on=date.today().isoformat(),owner='test fixture author',
                    ownership_statement='Original generated development fixture; not a deployment recording')
                register=document(root,f'licence-{i}.json',dict(assets=[dict(id=f'asset-{i}',name='Generated test audio',
                    license_id='user-owned',disposition='plug-in',version_or_hash=asset['sha256'],
                    url='https://example.test/original-fixture',checked_on=date.today().isoformat(),
                    evidence_quote=asset['ownership_statement'])]))
                source=dict(kind='recorded',asset=asset,register=register,review=dict(family=family,
                    source_group=f'group-{i}',listened=True,target_free=True,rights_cleared=True,
                    reviewer='development control',notes='generated samples only',
                    acoustic_stage='captured' if stage=='device_recorded' else 'dry',
                    deployed_device=True,capture_device='synthetic capture fixture',profile_id='pi5_inmp441/1'))
            item=Item(f'parent-{i}',family,f'group-{i}',audio_hash(x));parents.append(item)
            review=document(root,f'review-{i}.json',dict(parent=asdict(item),target_scope='three household targets',
                target_free=True,ambiguous=False,listened=True,rights_cleared=True,
                reviewer='development control',notes='mechanism fixture, no realism conclusion'))
            entries.append(dict(item=asdict(item),source=source,semantic_review=review,context_role=role,
                positions={'room-A':[1.,1.,1.]},room_draw_ids=['room-A'],whole_context=True,
                weight=1. if weight is None else weight[i]))
        plan=SplitPlan.create(parents+[Item('audit','sealed-family','sealed-group','e'*64)],
            audit_families=['sealed-family'],development_families=[],seed=31)
        lock=root/'split.json';plan.save(lock)
        data=dict(schema='negative-training/2',target_scope='three household targets',split_lock='split.json',
                  split_file_sha256=sha(lock),rows=entries)
        manifest=root/'pool.json';document(root,'pool.json',data)
        def build(**kw):
            return tn.build_training_pool(manifest,sha(manifest),fs=fs,room_draws=draws,**kw)
        target=np.sin(np.arange(fs//2)*.27)*.1
        corpus=SimpleNamespace(classes=['knock'],background=[np.random.default_rng(61).normal(0,.001,fs*3)],
            targets={'knock':[target]},spans={'knock':[(0,len(target))]},window_cues={'knock':[None]},
            rows=[dict(label='knock',room_draw_id='room-A',model_hash=room.model_hash)],
            pool_purpose='train',negative_pool=None,confuser_draws=0)
        cfg=dict({**nn.DEFAULTS,**tn.DEFAULTS},negative_context_policy=tn.POLICY,
                 p_confuser_given_target=1.,p_confuser_given_background=1.,threads=2,workers=0)
        return SimpleNamespace(root=root,data=data,manifest=manifest,build=build,draws=draws,
                               corpus=corpus,cfg=cfg,parents=parents,fs=fs)
    return make
