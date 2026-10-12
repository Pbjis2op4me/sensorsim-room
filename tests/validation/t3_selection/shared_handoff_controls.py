# SPDX-License-Identifier: Apache-2.0
"""Development probes of proposed shared hunks, executed in a fresh process."""
import importlib.util
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.unit.nn.test_nn import manifest
from tests.unit.t3_selection.test_selection import make_bank, constraints


def module(name):
    return __import__('sensorsim.'+name, fromlist=[name])


def test_ap_tie_and_unrounded_final_metadata():
    nn=module('nn')
    assert nn._average_precision(np.ones(4),np.array([1,1,0,0]))==.5
    assert '"best_val_mean_ap": round(best[0], 4)' not in inspect.getsource(nn.train)


def test_proposed_callback_reaches_real_model_and_skips_postfit(manifest,tmp_path,monkeypatch):
    from sensorsim.evaluation.selection import checkpoint_callback
    nn=module('nn')
    root=tmp_path/'bank';root.mkdir()
    bank,_=make_bank(root,classes=('high','low'),fs=16000)
    policy=dict(on=1.,smooth_hops=1,min_hops=2,off_ratio=.6,hop_s=.25)
    frozen=checkpoint_callback(bank,constraints(),[policy],seed=1103,code_revision='development-overlay')
    seen=[]
    def observe(path,step):
        result=frozen(path,step);seen.append(result);return result
    def forbidden(*a,**kw):raise AssertionError('post-selection tracker fit reached')
    monkeypatch.setattr(nn,'_fit_tracker',forbidden)
    report=nn.train(manifest,tmp_path/'out.npz',checkpoint_selector=observe,selection_mode='operational',
        steps=3,width=4,blocks=1,batch=8,workers=0,threads=1,val_windows=64,
        p_confuser_given_background=0.,seed=1103)
    assert len(seen)==1 and seen[0]['status']=='infeasible'
    assert seen[0]['attempted'][0]['score_origin']=='edge.window_scores'
    assert report['selection']['status']=='infeasible'
    assert report['tracker_order']=='policy frozen before scoring; no post-selection fitting'
    assert report['artifact_sha256']
    assert (tmp_path/'out.checkpoint-3.npz').exists()


def test_saving_changed_policy_marks_selection_stale(tmp_path):
    edge=module('edge')
    p=dict(on={'knock':.5},smooth_hops=1,min_hops=2,off_ratio=.6,hop_s=.25)
    meta=dict(detector=p,training=dict(selection=dict(status='feasible',selected=dict(detector=p))))
    path=tmp_path/'old.npz';out=tmp_path/'new.npz'
    np.savez(path,meta=np.frombuffer(json.dumps(meta).encode(),np.uint8))
    edge.save_detector(str(path),str(out),on={'knock':.9})
    with np.load(out) as loaded:changed=json.loads(bytes(loaded['meta']))
    assert changed['training']['selection']['status']=='requires_reselection'
    assert changed['training']['selection']['integration']=='stale'
    edge.save_detector(str(path),str(out),on={'knock':.5})
    with np.load(out) as loaded:same=json.loads(bytes(loaded['meta']))
    assert same['training']['selection']['status']=='feasible'


def test_callback_selected_policy_is_retained_without_postfit(manifest,tmp_path,monkeypatch):
    from hashlib import sha256
    from sensorsim import edge
    from sensorsim.evaluation.selection import candidate_identity,cache_scores,score_candidate,select_candidate
    nn=module('nn');root=tmp_path/'bank';root.mkdir()
    bank,_=make_bank(root,classes=('high','low'),fs=16000)
    policy=dict(on=.8,smooth_hops=1,min_hops=2,off_ratio=.6,hop_s=.25)
    def control_callback(path,step):
        runtime=edge.Model(str(path))
        c=candidate_identity(model_sha256=sha256(path.read_bytes()).hexdigest(),features=runtime.meta['features'],
            classes=runtime.classes,policy=policy,step=step,seed=1103,code_revision='synthetic-callback-control',model_path=path)
        seq={}
        for r in bank.rows:
            x=np.zeros((1+(r['sample_count']-16000)//4000,2))
            if r['kind']=='target':x[2:4,runtime.classes.index(r['annotations'][0]['label'])]=.9
            seq[r['id']]=x
        row=score_candidate(c,bank,constraints(),score_cache=cache_scores(c,bank,seq))
        return dict(select_candidate([row]),attempted=[row])
    def forbidden(*a,**kw):raise AssertionError('post-selection fit')
    monkeypatch.setattr(nn,'_fit_tracker',forbidden)
    report=nn.train(manifest,tmp_path/'out.npz',checkpoint_selector=control_callback,selection_mode='operational',
        steps=3,width=4,blocks=1,batch=8,workers=0,threads=1,val_windows=64,p_confuser_given_background=0.,seed=1103)
    assert report['selection']['status']=='feasible'
    runtime=edge.Model(str(tmp_path/'out.npz'))
    assert runtime.meta['detector']['on']=={'high':.8,'low':.8}
    assert runtime.meta['detector']['min_hops']=={'high':2,'low':2}
    assert report['selected_checkpoint_sha256']==sha256((tmp_path/'out.checkpoint-3.npz').read_bytes()).hexdigest()
    # This accepts callback/persistence routing only, not classifier quality.
    assert report['selection']['selected']['score_origin']=='synthetic_control'
    assert 'model_path' not in report['selection']['selected']['candidate']


def test_calibration_metadata_tracks_changed_policy_and_unchecked_positives(monkeypatch):
    from types import SimpleNamespace
    edge=module('edge')
    model=SimpleNamespace(classes=['knock'],features=SimpleNamespace(n=100),
        meta=dict(features=dict(fs=100),detector=dict(on=.5,off_ratio=.6,smooth_hops=1,min_hops=2,hop_s=.25)))
    x=np.linspace(-.1,.1,100,dtype=np.float32)
    for score,changed in ((0.,False),(.9,True)):
        monkeypatch.setattr(edge,'window_scores',lambda *a:np.array([[score]]))
        result=edge.calibrate(model,x,fa_per_hour=1.)
        assert result['selection_invalidated'] is changed
        assert result['requires_reselection'] is changed
        assert result['positive_feasibility']=='not_checked'
