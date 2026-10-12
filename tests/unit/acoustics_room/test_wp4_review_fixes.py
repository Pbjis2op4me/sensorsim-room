"""Review regressions. All waveforms and acoustic claims here are proxy results."""
import json
import numpy as np
import pytest
import yaml
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import map_coordinates

from sensorsim.acoustics.room.prior import (
    RoomSurvey, build_prior, CalibrationObservation, update_ensemble,
    known_source_observation, predictive_intervals, predict_metrics)
from sensorsim.acoustics.room.rir import correct_known_clock
from sensorsim.acoustics.room.dataset import _anchored_position, SamplingSpec
from tests.unit.acoustics_room.test_room_prior import survey


def playback(fs=16000, pre=0., ppm=150., pink=False, normalized=False):
    rng = np.random.default_rng(732)
    x = rng.normal(size=fs*4)
    if pink:
        f = np.fft.rfftfreq(len(x), 1/fs)
        x = np.fft.irfft(np.fft.rfft(x)/np.sqrt(np.maximum(f, 20)), len(x))
    x *= (1. if normalized else .1)/max(abs(x))
    h = np.zeros(round(1.5*fs)); h[80] = .15; h[193] = .16; h[376] = -.15
    h[81:] += rng.normal(0, .007, len(h)-81)*np.exp(-3*np.log(10)*np.arange(len(h)-81)/(fs*.45))
    y = np.r_[np.zeros(round(pre*fs)), signal.fftconvolve(x, h), np.zeros(fs)]
    ratio = 1+ppm*1e-6
    y = map_coordinates(y, [np.arange(int(len(y)*ratio))/ratio], order=3, mode='constant')
    y += rng.normal(0, 1e-7, len(y))
    return x, y


def observation(x, y, fs=16000):
    return known_source_observation(y, x, fs, source=(1,1,1), receiver=(3,2,1.4),
                                   source_group='noise', input_sha256='proxy')


@pytest.mark.parametrize('pink', [False, True])
@pytest.mark.parametrize('ppm', [-200., 150.])
def test_h1_equal_early_reflections_track_one_response(pink, ppm):
    x, y = playback(pink=pink, ppm=ppm)
    _, timing = correct_known_clock(y, x, 16000)
    assert abs(timing['drift_ppm']-ppm) < 2


@pytest.mark.parametrize('kind', ['known-source', 'impulsive-decay'])
def test_h2_one_contradictory_metric_cannot_be_averaged_away(kind):
    ens = build_prior(survey())
    values = {f'{b}.t30_s': .15 if b == 500 else .4 for b in (500,1000,2000,4000)}
    obs = CalibrationObservation(kind=kind, source_group='a', input_sha256='proxy',
        source_xyz_m=(1,1,1), receiver_xyz_m=(3,2,1.4), values=values,
        uncertainty={k:.025 for k in values}, qualified=True)
    pred = [{**values, '500.t30_s': .27+i*.001} for i in range(4)]
    post = update_ensemble(ens, obs, pred)
    assert post.status == 'prior-conflict'
    assert post.weights == ens.weights and not post.evidence[-1]['accepted']
    assert '500.t30_s' in post.evidence[-1]['conflicts']
    # Later valid evidence cannot erase a conflict and authorize rendering.
    assert update_ensemble(post, obs.model_copy(update={'values': pred[0]}), pred).status == 'prior-conflict'


@pytest.mark.parametrize('pre', [.5, 2.])
def test_h3_preroll_retains_500hz_decay_without_offset(pre):
    x, y = playback(pre=pre, ppm=0)
    o = observation(x, y)
    assert o.qualified, o.reasons
    assert '500.t30_s' in o.values
    assert 'direct_delay_s' not in o.values
    assert o.values['500.t30_s'] == pytest.approx(observation(*playback(ppm=0)).values['500.t30_s'], rel=.05)


def test_m5_isolated_full_scale_peak_is_not_clipping():
    o = observation(*playback(ppm=0, normalized=True))
    assert o.qualified, o.reasons
    x, y = playback(ppm=0); y[100:108] = 1.
    assert not observation(x,y).qualified


def test_m2_m7_intervals_include_discrepancy_and_survive_missing_member():
    constant = predictive_intervals([{'500.t30_s':.4}]*4, [.25]*4)['500.t30_s']
    assert constant[2]-constant[0] >= .328  # spread alone is zero, discrepancy is not
    info = {}
    pi = predictive_intervals([{'500.t30_s':.4}]*3+[{}], [.25]*4, diagnostics=info)
    lo, mid, hi = pi['500.t30_s']
    assert mid == pytest.approx(.4)
    assert lo <= .4-1.64*.1 and hi >= .4+1.64*.1
    assert info['500.t30_s']['missing_weight'] == .25


def test_m3_holdout_within_each_site_with_default_mic_tolerance():
    s = survey(microphone=dict(x=3,y=2,z=1.4), source_sites={'knock':[
        dict(x=.02,y=1,z=1.25,spread_m=[0,.2,.35]),
        dict(x=4.98,y=1,z=1.25,spread_m=[0,.2,.35])]})
    model=build_prior(s).members[0]
    for subset in ('train','val'):
        rows=[_anchored_position(s,model,'knock',subset,.2,np.random.default_rng(i),SamplingSpec()) for i in range(80)]
        assert {round(p.source_xyz_m[0],2) for p,r in rows} == {.02,4.98}
        for p,r in rows:
            assert abs(p.receiver_xyz_m[0]-3) <= .05
            coordinate=p.source_xyz_m[r['axis']]
            assert coordinate < r['boundary_m']-r['guard_m'] if subset=='train' else coordinate >= r['boundary_m']+r['guard_m']


def test_l1_point_site_has_content_holdout_without_fake_spread():
    s=survey(source_sites={'knock':[dict(x=.02,y=1,z=1)]})
    p,r=_anchored_position(s,build_prior(s).members[0],'knock','val',.2,np.random.default_rng(1),SamplingSpec())
    assert p.source_xyz_m == (.02,1,1) and r['policy']=='content/group only; no spatial holdout'


@pytest.mark.parametrize('site', ['microphone','loudspeaker'])
def test_l5_wall_sites_fail_at_survey_loading(site):
    with pytest.raises(ValueError,match='strictly inside'):
        survey(**{site:dict(x=0,y=1,z=1)})


def test_l4_prediction_duration_covers_widened_prior():
    e=build_prior(survey(prior=dict(count=4,t30_s=[[1.25,1.3]]*7)))
    assert len(predict_metrics(e,(1,1,1),(3,2,1.4),8000))==4


def test_l3_l7_stereo_and_unpaired_files_report_abstention(tmp_path):
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey().model_dump(mode='json')))
    (tmp_path/'played').mkdir(); (tmp_path/'room_recordings').mkdir()
    x,y=playback(fs=8000,ppm=0)
    wavfile.write(tmp_path/'played'/'noise.wav',8000,np.column_stack([x,x]).astype('float32'))
    wavfile.write(tmp_path/'room_recordings'/'noise__room.wav',8000,y.astype('float32'))
    wavfile.write(tmp_path/'room_recordings'/'orphan.wav',8000,y.astype('float32'))
    r=run_no_sweep(tmp_path/'room.yaml',tmp_path/'fit',fs=8000,evidence_label='proxy')
    assert r['calibration_status']=='assumed'
    assert any('mono' in str(e) for e in r['evidence'])
    assert any('orphan.wav' in w for w in r['warnings'])


def test_l8_bad_survey_does_not_block_project_status(tmp_path):
    from sensorsim import project
    (tmp_path/'room.yaml').write_text('invalid: survey')
    cfg=project.load_project(tmp_path)
    with pytest.raises(ValueError): project._fit(cfg,tmp_path/'out')


def test_h4_project_dataset_uses_ensemble(tmp_path):
    from sensorsim import project
    root=tmp_path
    (root/'room.yaml').write_text(yaml.safe_dump(survey().model_dump(mode='json')))
    # WP6: raw target WAVs need dataset.sources provenance and bound listening approval (proxy).
    rows=[]
    for i in range(3):
        folder=root/'targets'/'dry'; folder.mkdir(parents=True,exist_ok=True)
        t=np.arange(1600)/8000
        take=np.random.default_rng(i).normal(0,1e-4,16000)
        take[4000:5600]+=.2*np.sin(2*np.pi*(600+40*i)*t)*np.exp(-t/.01)
        wavfile.write(folder/f'knock_{i}.wav',8000,take.astype('float32'))
        rows.append(dict(file=f'dry/knock_{i}.wav',label='knock',source_group=f'knock-{i}',object_id='door',
                         session='s1',capture_route='near_dry',evidence_label='proxy'))
        bg=root/'background'; bg.mkdir(exist_ok=True)
        wavfile.write(bg/f'{i}.wav',8000,np.random.default_rng(i+10).normal(0,.001,8000).astype('float32'))
    cfg=project.load_project(root); cfg['dataset'].update(count=4,fs=8000,sources=rows)
    out=root/'out'; project._fit(cfg,out)
    with pytest.raises(ValueError,match='source intake stopped'):
        project._clips(cfg,out)
    for row,clip in zip(rows,json.loads((out/'source_qa.json').read_text())['clips']):
        row['approval']=dict(reviewer='proxy fixture',decision='approve',approval_key=clip['approval_key'])
    summary=project._dataset(cfg,out)
    rows=[json.loads(l) for l in (out/'dataset'/'manifest.jsonl').read_text().splitlines()]
    assert summary['clips']>0 and all('ensemble_hash' in r for r in rows if r['label']=='knock')


def test_m1_sweep_pipeline_pins_legacy_split(tmp_path, monkeypatch):
    from sensorsim.acoustics.room import pipeline
    from sensorsim.acoustics.room.model import RoomModel
    def generate(*args,**kwargs):
        assert kwargs.get('split_policy')=='legacy-both'
        raise RuntimeError('checked before rendering')
    monkeypatch.setattr(pipeline,'generate_dataset',generate)
    with pytest.raises(RuntimeError,match='checked before rendering'):
        pipeline._dataset(RoomModel(shoebox_dims_m=(5,4,3)),[],tmp_path,pipeline.PipelineConfig(),8000)


def test_m4_repeat_observations_share_discrepancy():
    e=build_prior(survey())
    values={'500.t30_s':.3,'1000.t30_s':.4}
    o=CalibrationObservation(kind='known-source',source_group='a',input_sha256='proxy',
        source_xyz_m=(1,1,1),receiver_xyz_m=(3,2,1.4),values=values,
        uncertainty={k:.15 for k in values},qualified=True)
    p=[{k:v+i*.05 for k,v in values.items()} for i in range(4)]
    first=update_ensemble(e,o,p); post=first
    for i in range(8): post=update_ensemble(post,o.model_copy(update={'source_group':str(i)}),p)
    assert post.weights==first.weights


def test_h2_conflict_blocks_dataset_even_with_valid_clips(tmp_path):
    from sensorsim.acoustics.room.dataset import generate_dataset, DryClip
    from sensorsim.acoustics.room.augment import AugmentationConfig
    e=build_prior(survey()).model_copy(update={'status':'prior-conflict'})
    clips=[DryClip('knock',np.random.default_rng(i).normal(0,.01,800),8000,source_group=str(i)) for i in range(3)]
    with pytest.raises(ValueError,match='prior-conflict'):
        generate_dataset(e,SamplingSpec(),clips,AugmentationConfig(),4,0,tmp_path/'bad',fs=8000)
    assert not (tmp_path/'bad').exists()


def test_identifiable_sweep_packets_and_ambiguous_single_sweep():
    from tests.validation.wp4_ensemble.review_acceptance import excitation
    fs=16000
    h=np.zeros(fs);h[80]=.1;h[193]=.1;h[376]=-.1
    for kind in ('sweep_packets','sweep'):
        x=excitation(kind,fs,320,10);y=np.r_[signal.fftconvolve(x,h),np.zeros(fs)]
        ratio=1.00015;y=map_coordinates(y,[np.arange(int(len(y)*ratio))/ratio],order=3,mode='constant')
        if kind=='sweep':
            with pytest.raises(ValueError,match='ambiguous'):correct_known_clock(y,x,fs)
        else:
            assert abs(correct_known_clock(y,x,fs)[1]['drift_ppm']-150)<2


def test_h2_pipeline_warns_and_does_not_render_wrong_prior(tmp_path):
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    from sensorsim.acoustics.room.dataset import DryClip
    s=survey(prior=dict(count=8,t30_s=[[1.,1.2]]*7))
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(s.model_dump(mode='json')))
    (tmp_path/'played').mkdir();(tmp_path/'room_recordings').mkdir()
    x,y=playback(ppm=0)
    wavfile.write(tmp_path/'played'/'noise.wav',16000,x.astype('float32'))
    wavfile.write(tmp_path/'room_recordings'/'noise__room.wav',16000,y.astype('float32'))
    clips=[DryClip('knock',np.random.default_rng(i).normal(0,.01,800),16000,source_group=str(i)) for i in range(3)]
    r=run_no_sweep(tmp_path/'room.yaml',tmp_path/'fit',fs=16000,clips=clips,evidence_label='proxy')
    assert r['calibration_status']=='prior-conflict'
    assert any('prior-conflict' in w for w in r['warnings'])
    assert not (tmp_path/'fit'/'dataset').exists()


def test_l9_weak_fallback_requires_explicit_review(tmp_path):
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey().model_dump(mode='json')))
    wavfile.write(tmp_path/'knock.wav',8000,np.ones(8000,dtype='float32')*.01)
    settings={'impulses':[dict(file='knock.wav',source_group='knock',source_xyz_m=[.02,1,1],
        excitation_end_s=.1,floor_end_s=.04,source_tail_upper_s=.01)]}
    (tmp_path/'calibration.yaml').write_text(yaml.safe_dump(settings))
    r=run_no_sweep(tmp_path/'room.yaml',tmp_path/'fit',fs=8000,evidence_label='proxy')
    assert r['calibration_status']=='assumed' and not r['evidence']
    assert any('reviewed: true' in w for w in r['warnings'])


def test_m3_insufficient_site_training_capacity_fails_early(tmp_path):
    from sensorsim.acoustics.room.dataset import generate_dataset,DryClip
    from sensorsim.acoustics.room.augment import AugmentationConfig
    s=survey(source_sites={'knock':[dict(x=.02,y=1,z=1),dict(x=4.98,y=1,z=1)]})
    clips=[DryClip('knock',np.random.default_rng(i).normal(0,.01,800),8000,source_group=str(i)) for i in range(2)]
    with pytest.raises(ValueError,match='cover every site'):
        generate_dataset(build_prior(s),SamplingSpec(),clips,AugmentationConfig(),2,0,tmp_path/'bad',fs=8000,all_sources=True)
    assert not (tmp_path/'bad').exists()
