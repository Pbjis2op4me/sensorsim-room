"""Development replay of disclosed verifier routes. Synthetic proxy, no measured data."""
import hashlib,json,os,pathlib
from unittest.mock import patch
import numpy as np
import pytest
import yaml
from scipy.io import wavfile
from scipy.signal import fftconvolve
from sensorsim.acoustics.room import prior,pipeline,calibration_discrepancy as cd
from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.dataset import DryClip,SamplingSpec,generate_dataset

def survey(near=False):
    return prior.RoomSurvey.model_validate(dict(room=dict(length_x=4.,width_y=3.,height_z=2.5),
        microphone=dict(x=1.,y=1.,z=1.,tolerance_m=.05),loudspeaker=dict(x=2.,y=2.,z=1.),
        source_sites={'knock':[dict(x=2.,y=2.,z=1.)],'bell':[dict(x=1.26 if near else 2.5,y=1.,z=1.)]},prior=dict(count=4,seed=300)))

def test_real_playback_route_and_dataset_block(tmp_path):
    fs=16000;rng=np.random.default_rng(101);x=rng.normal(0,.02,fs*2)
    h=np.zeros(fs);h[80]=.5;h[81:]=rng.normal(0,.008,len(h)-81)*np.exp(-np.arange(len(h)-81)/(fs*.035))
    y=np.r_[fftconvolve(x,h),np.zeros(fs)]
    room=tmp_path/'room.yaml';room.write_text(yaml.safe_dump(survey().model_dump(mode='json')))
    for folder,name,samples in [('played','noise.wav',x),('room_recordings','noise__room.wav',y)]:
        (tmp_path/folder).mkdir();wavfile.write(tmp_path/folder/name,fs,samples.astype(np.float32))
    keys=['1000.t30_s','1000.edt_s','1000.c50_db'];n=len(keys)
    components=dict(keys=keys,units={k:cd.metric_unit(k) for k in keys},measurement_covariance=np.diag([.16,.16,1.]).tolist(),
        playback_covariance=np.zeros((n,n)).tolist(),simulator_covariance=np.zeros((n,n)).tolist(),
        discrepancy_covariance=np.diag([.09,.09,1.]).tolist(),policy_id='verifier-route-synthetic-v1')
    (tmp_path/'calibration.yaml').write_text(yaml.safe_dump(dict(holdout_groups=[],late_block=dict(uncertainty_components=components,early_tolerances={'1000.edt_s':.2,'1000.c50_db':1.}))))
    with patch.object(prior,'known_source_observation',wraps=prior.known_source_observation) as observation,patch.object(prior,'predict_metrics',wraps=prior.predict_metrics) as predictions,patch.object(cd,'late_block_update',wraps=cd.late_block_update) as update:
        report=pipeline.run_no_sweep(room,tmp_path/'fit',fs=fs,seed=761,evidence_label='proxy')
        assert observation.call_count==predictions.call_count==update.call_count==1
    data=json.loads((tmp_path/'fit/room_ensemble.json').read_text());ensemble=cd.ResearchRoomEnsemble.model_validate(data)
    with pytest.raises(ValueError):
        prior.RoomEnsemble.model_validate(data)
    evidence=ensemble.evidence[-1];obs=evidence['observation']['metadata']['observations'][0]
    assert obs['qualified']
    assert obs['metadata']['method']=='known-whole-response-clock/2'
    assert report['rendering_distribution']=='research-only modular candidate; training blocked'
    clip=DryClip('knock',np.random.default_rng(40).normal(0,.03,1600),fs,source_group='independent-dry')
    with pytest.raises(ValueError,match='P4'):
        generate_dataset(ensemble,SamplingSpec(),[clip],AugmentationConfig(),1,700,tmp_path/'dataset',fs=fs,val_fraction=0,allow_train_only=True)
    out=dict(scope='proxy end-to-end signal/extraction/pooling/update/report/training-refusal route; not acoustic acceptance',
        calls=dict(known_source_observation=observation.call_count,predict_metrics=predictions.call_count,late_block_update=update.call_count),
        calibration_status=ensemble.status,late_block=evidence['late_block'],clock_metadata=obs['metadata'],report=report)
    (tmp_path/'real_playback_route.json').write_text(json.dumps(out,indent=2)+'\n')

def test_generated_scene_manifest_is_coherent(tmp_path):
    e=prior.build_prior(survey());rng=np.random.default_rng(987)
    clips=[DryClip(label,rng.normal(0,.03,1600),16000,source_group=label+'-v') for label in ('knock','bell')]
    path=generate_dataset(e,SamplingSpec(),clips,AugmentationConfig(),1,1,tmp_path/'generated',fs=16000,
        val_fraction=0,allow_train_only=True,all_sources=True)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==2
    result=cd.audit_scene_states(rows)
    assert result['status']=='pass'
    (tmp_path/'generated_scene_route.json').write_text(json.dumps({'scope':'synthetic generated WAV and manifest route','scene_audit':result,'rows':rows},indent=2)+'\n')
