"""Deterministic development proxies for intake boundaries and reruns."""
import json
from datetime import date
import numpy as np
import pytest
import yaml
from scipy.io import wavfile
from sensorsim import project
from sensorsim.sources import intake
from tests.unit.sources import final_standins as F
from tests.unit.acoustics_room.test_rehearsal_intake import survey


def approved_project(root):
    (root/'dry').mkdir()
    wavfile.write(root/'dry/take.wav', F.FS, F.door().astype('float32'))
    row=dict(file='dry/take.wav', **F.meta('door_knock'))
    report=root/'qa.json'
    with pytest.raises(ValueError, match='review'):
        intake.load_project_sources([row],root,report)
    qa=json.loads(report.read_text())['clips'][0]
    row['approval']=dict(reviewer='development proxy',decision='approve',approval_key=qa['approval_key'])
    cfg=project.load_project(root)
    cfg.update(targets='.')
    cfg['dataset']['sources']=[row]
    return cfg


@pytest.mark.parametrize('custom', [False,True])
def test_root_targets_rerun_ignores_generated_and_evaluation(tmp_path,custom):
    cfg=approved_project(tmp_path)
    if custom:cfg.update(out='generated',evaluate='audit-takes')
    before=project._clips(cfg,tmp_path/cfg['out'])
    cfg['code']='development'; ledger={'fit':{'outputs':'same fit'}}
    fingerprint=project._inputs('dataset',cfg,ledger)
    for name in (cfg['out']+'/dataset/train',cfg['evaluate']+'/door_knock'):
        (tmp_path/name).mkdir(parents=True,exist_ok=True)
        wavfile.write(tmp_path/name/'take.wav',F.FS,F.door().astype('float32'))
    after=project._clips(cfg,tmp_path/cfg['out'])
    assert len(before)==len(after)==1
    assert project._inputs('dataset',cfg,ledger)==fingerprint
    wavfile.write(tmp_path/'dry/undeclared.wav',F.FS,F.door().astype('float32'))
    with pytest.raises(ValueError,match='undeclared.wav'):
        project._clips(cfg,tmp_path/cfg['out'])


@pytest.mark.parametrize('folder',['out','evaluate'])
def test_excluded_audio_cannot_be_declared_as_target(tmp_path,folder):
    cfg=approved_project(tmp_path)
    (tmp_path/folder).mkdir(exist_ok=True)
    wavfile.write(tmp_path/folder/'copy.wav',F.FS,F.door().astype('float32'))
    cfg['dataset']['sources'][0]['file']=folder+'/copy.wav'
    with pytest.raises(ValueError,match='reserved|generated|evaluation'):
        project._clips(cfg,tmp_path/'out')


def test_session_yaml_date_is_actionable_before_discovery(tmp_path):
    cfg=approved_project(tmp_path)
    cfg['dataset']['sources'][0]['session']=yaml.safe_load('2026-10-09')
    assert isinstance(cfg['dataset']['sources'][0]['session'],date)
    with pytest.raises(ValueError,match='quote the session string'):
        project._clips(cfg,tmp_path/'out')


def test_sites_checked_before_listening_and_all_missing_named(tmp_path,monkeypatch):
    cfg=approved_project(tmp_path); room=tmp_path/'room.yaml'
    room.write_text(yaml.safe_dump(survey()));cfg['room_yaml']=str(room)
    cfg['dataset']['sources'].append(dict(cfg['dataset']['sources'][0],label='kettle'))
    calls=[]
    monkeypatch.setattr(intake,'load_project_sources',lambda *a,**k: calls.append(a) or [])
    with pytest.raises(ValueError,match='source_sites') as error:project._clips(cfg,tmp_path/'out')
    assert 'door_knock' in str(error.value) and 'kettle' in str(error.value)
    assert calls==[]
    data=survey();data['source_sites']={label:[dict(x=.08,y=1.2,z=1.2)] for label in ('door_knock','kettle')}
    room.write_text(yaml.safe_dump(data));project._clips(cfg,tmp_path/'out')
    assert len(calls)==1
