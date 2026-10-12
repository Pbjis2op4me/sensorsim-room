"""Surveyed ensemble contracts; all audio and geometry are model-based proxies."""
import json

import numpy as np
import pytest
import yaml
from scipy.io import wavfile

from sensorsim.acoustics.room.prior import (RoomSurvey, RoomEnsemble, build_prior,
    CalibrationObservation, update_ensemble, known_source_observation, impulsive_decay_observation)
from sensorsim.acoustics.room.dataset import DryClip, SamplingSpec, generate_dataset
from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.synth import SynthConfig


def survey(**changes):
    raw = dict(units="m", room=dict(length_x=5, width_y=4, height_z=3),
        microphone=dict(x=3, y=2, z=1.4, tolerance_m=0), loudspeaker=dict(x=1, y=1, z=1),
        source_sites={"knock": [dict(x=.02, y=1.0, z=1.25, spread_m=[0,.2,.35])],
                      "doorbell": [dict(x=2, y=2, z=2.5, spread_m=[.05, .05, .05])]},
        prior=dict(count=4, seed=4108))
    raw.update(changes)
    return RoomSurvey.model_validate(raw)


def test_prior_reproducible_assumed_and_decay_consistent():
    a, b = build_prior(survey()), build_prior(survey())
    assert a.model_dump_json() == b.model_dump_json()
    assert len({m.model_hash for m in a.members}) == 4
    assert RoomEnsemble.model_validate_json(a.model_dump_json()).ensemble_hash == a.ensemble_hash
    for m in a.members:
        assert m.provenance["status"] == m.uncertainty["status"] == "assumed"
        assert m.t60_estimates() == pytest.approx(m.late_t60_s, rel=1e-8)
        assert m.sound_speed_policy == "humid-air-cramer/1"
    assert sorted(a.member_schedule(4, 4)) == list(range(4))


def test_wp4c_fixed_mic_dataset(tmp_path):
    from sensorsim.acoustics.room.render_prior import select_render_ensemble
    ensemble = select_render_ensemble(build_prior(survey()))
    rng = np.random.default_rng(4108)
    clips = [DryClip(label, rng.normal(0,.02,128), 8000, source_group=f"{label}-{i}")
             for label in ("knock", "doorbell") for i in range(4)]
    # A renamed/container-changed copy must remain on the same side as its source.
    clips.append(DryClip("knock", clips[0].audio.copy(), 8000, source_group="renamed"))
    paths = [generate_dataset(ensemble, SamplingSpec(), clips, AugmentationConfig(), 6, 4108, tmp_path/f"run{i}",
        fs=8000, all_sources=True, synth_config=SynthConfig(late=False, max_order=0)) for i in range(2)]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    rows = [json.loads(s) for s in paths[0].read_text().splitlines()]
    groups = {key: [r for r in rows if r["split"] == key] for key in ("train", "val")}
    assert all(groups.values())
    for key in ("source_group", "content_sha256", "normalized_content_sha256"):
        assert {r["dry_source"][key] for r in groups["train"]}.isdisjoint(r["dry_source"][key] for r in groups["val"])
    for row in rows:
        region = row["source_subregion"]
        coordinate = row["source_xyz_m"][region["axis"]]
        if row["split"] == "train":
            assert coordinate < region["boundary_m"]-region["guard_m"]
        else:
            assert coordinate >= region["boundary_m"]+region["guard_m"]
    assert {r["output_sha256"] for r in groups["train"]}.isdisjoint(r["output_sha256"] for r in groups["val"])
    assert {tuple(r["receiver_xyz_m"]) for r in rows} == {(3.,2.,1.4)}
    assert {r["ensemble_member_hash"] for r in rows} == {m.model_hash for m in ensemble.members}
    for r in rows:
        assert r["ensemble_member_hash"] == r["model_hash"]
        assert r["result_label"] == "model-based"
        if r["label"] == "knock":
            assert r["source_xyz_m"][0] == .02  # global wall clearance would reject this
        else:
            assert r["source_xyz_m"][2] > 2.4  # global source height would reject this


def test_survey_validation_and_point_site_holdout(tmp_path):
    with pytest.raises(ValueError, match="outside room"):
        survey(microphone=dict(x=50,y=2,z=1))
    with pytest.raises(ValueError, match="spread extends"):
        survey(source_sites={"knock": [dict(x=.01,y=1,z=1,spread_m=[.2,0,0])]})
    ensemble = build_prior(survey(source_sites={"knock": [dict(x=.02,y=1,z=1)]}))
    clips=[DryClip("knock", np.array([.01, .02+i*.01]), 8000, source_group=str(i)) for i in range(2)]
    point_manifest = generate_dataset(ensemble, SamplingSpec(), clips, AugmentationConfig(), 4, 0, tmp_path/"point", fs=8000)
    point_rows = [json.loads(line) for line in point_manifest.read_text().splitlines()]
    assert {r["split"] for r in point_rows} == {"train", "val"}
    assert all(r["source_subregion"]["policy"] == "content/group only; no spatial holdout" for r in point_rows)
    with pytest.raises(ValueError, match="requires RoomEnsemble"):
        generate_dataset(ensemble.members[0], SamplingSpec(), clips, AugmentationConfig(), 4, 0, tmp_path/"bad", fs=8000)


def test_weak_evidence_abstains_on_unknown_tail_and_censoring():
    ensemble = build_prior(survey())
    for censored, tail in [(True, .01), (False, None)]:
        obs = impulsive_decay_observation(np.ones(8000)*.01, 8000, source=(1,1,1),receiver=(3,2,1.4),
            source_group="knock-1",input_sha256="synthetic", excitation_end_s=.1, floor_end_s=.04,
            source_tail_upper_s=tail,censored=censored)
        updated = update_ensemble(ensemble, obs, [{}]*4)
        assert updated.weights == ensemble.weights and updated.status == "assumed"
        assert not updated.evidence[-1]["accepted"]


def test_likelihood_weights_and_unsupported_observation_abstains():
    ensemble = build_prior(survey())
    obs = CalibrationObservation(kind="known-source",source_group="a",input_sha256="synthetic",
        source_xyz_m=(1,1,1),receiver_xyz_m=(3,2,1.4),qualified=True,
        values={"500.t30_s": .3,"1000.t30_s":.4},uncertainty={"500.t30_s":.15,"1000.t30_s":.15})
    predictions=[{"500.t30_s":.3+i*.05,"1000.t30_s":.4+i*.05} for i in range(4)]
    updated=update_ensemble(ensemble,obs,predictions)
    assert updated.status=="known-source" and updated.weights[0]>updated.weights[-1]
    assert sum(updated.weights)==pytest.approx(1.)
    unsupported=update_ensemble(ensemble,obs,[{"500.t30_s":10.,"1000.t30_s":10.}]*4)
    assert unsupported.weights==ensemble.weights


def test_known_tone_and_short_signal_are_rejected():
    x=.1*np.sin(2*np.pi*1000*np.arange(16000)/8000)
    for excitation in (x, x[:100]):
        obs=known_source_observation(np.r_[excitation,np.zeros(10000)],excitation,8000,
            source=(1,1,1),receiver=(3,2,1.4),source_group="a",input_sha256="synthetic")
        assert not obs.qualified and obs.reasons


def test_no_sweep_prior_and_fallback_pipeline(tmp_path):
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    cfg=survey().model_dump(mode="json")
    (tmp_path/"room.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path/"played").mkdir(); (tmp_path/"room_recordings").mkdir()
    x=np.sin(np.arange(16000)/8000*2*np.pi*1000).astype(np.float32)*.1
    wavfile.write(tmp_path/"played"/"tone.wav",8000,x)
    wavfile.write(tmp_path/"room_recordings"/"tone__room.wav",8000,np.r_[x,np.zeros(10000)].astype(np.float32))
    report=run_no_sweep(tmp_path/"room.yaml",tmp_path/"fit",fs=8000,evidence_label="proxy")
    assert report["calibration_status"]=="assumed" and report["evidence"]
    assert report["state"]=="complete" and report["result_label"]=="model-based"
    assert (tmp_path/"fit"/"room_ensemble.json").is_file()
    with pytest.raises(FileExistsError):
        run_no_sweep(tmp_path/"room.yaml",tmp_path/"fit")


def test_project_no_sweep_fit_and_cache_inputs(tmp_path):
    from sensorsim import project
    (tmp_path/"room.yaml").write_text(yaml.safe_dump(survey().model_dump(mode="json")))
    cfg=project.load_project(tmp_path);cfg["code"]="test"
    before=project._inputs("fit",cfg,{})
    out=tmp_path/"out"
    summary=project._fit(cfg,out)
    assert summary["mode"]=="no-sweep" and summary["result_label"]=="model-based"
    (tmp_path/"played").mkdir();(tmp_path/"played"/"new.wav").write_bytes(b"new input")
    assert project._inputs("fit",cfg,{}) != before
    assert project._gates("fit",summary)


def test_fixed_mic_explicit_positions_and_stress_split(tmp_path):
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.dataset import PositionPair
    model=RoomModel(shoebox_dims_m=(5,4,3))
    clips=[DryClip('knock',np.array([.02,.03+i*.01]),8000,source_group=str(i)) for i in range(3)]
    pairs=[PositionPair(source_xyz_m=(1+i*.4,1,1),receiver_xyz_m=(4,3,1)) for i in range(5)]
    path=generate_dataset(model,SamplingSpec(),clips,AugmentationConfig(),5,4108,tmp_path/'positions',
        target_positions=pairs,fs=8000,all_sources=True,synth_config=SynthConfig(late=False,max_order=0))
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert {r['split'] for r in rows}=={'train','val'}
    path=generate_dataset(model,SamplingSpec(),clips,AugmentationConfig(),2,4108,tmp_path/'stress',
        fs=8000,val_fraction=0,stress=True,synth_config=SynthConfig(late=False,max_order=0))
    assert {json.loads(x)['split'] for x in path.read_text().splitlines()}=={'stress'}


def test_custom_impulse_dependency_is_fingerprinted(tmp_path):
    from sensorsim.acoustics.room.pipeline import no_sweep_input_fingerprint
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey().model_dump(mode='json')))
    (tmp_path/'takes').mkdir();audio=tmp_path/'takes'/'a.wav';audio.write_bytes(b'first')
    (tmp_path/'calibration.yaml').write_text(yaml.safe_dump({'impulses':[{'file':'takes/a.wav'}]}))
    before=no_sweep_input_fingerprint(tmp_path/'room.yaml')
    audio.write_bytes(b'changed')
    assert no_sweep_input_fingerprint(tmp_path/'room.yaml')!=before


def test_known_broadband_observation_retains_clock_and_delay_authority():
    from scipy.signal import fftconvolve
    fs=16000;rng=np.random.default_rng(101)
    x=rng.normal(0,.02,fs*2)
    h=np.zeros(fs);h[80]=.5
    h[81:]=rng.normal(0,.008,len(h)-81)*np.exp(-np.arange(len(h)-81)/(fs*.035))
    y=np.r_[fftconvolve(x,h),np.zeros(fs)]
    common=dict(source=(1,1,1),receiver=(3,2,1.4),source_group='a',input_sha256='synthetic')
    free=known_source_observation(y,x,fs,**common)
    anchored=known_source_observation(y,x,fs,recording_offset_s=0.,recording_offset_hardware_anchored=True,**common)
    assert free.qualified and anchored.qualified
    assert 'direct_delay_s' not in free.values
    assert abs(anchored.values['direct_delay_s']-.005)<2/fs
    assert set(free.metadata['excited_bands_hz'])=={500,1000,2000,4000}
    assert abs(anchored.metadata['drift_ppm'])<20


def test_real_comparison_rejects_calibration_leakage():
    from tests.validation.wp4_ensemble.real import compare
    takes=[dict(source_group='held-'+label,label=label,hit=True) for label in ('knock','doorbell','kettle')]
    row=dict(steps_a=5,steps_b=5,training_rows_a=10,training_rows_b=10,
        takes_a=takes,takes_b=takes,auc_a=.6,auc_b=.5,c2st_b=dict(point=.5,lower=.4,upper=.6),wp2_audit_ratio_upper=1.1)
    payload=dict(label='measured',calibration_groups=['held-knock'],training_groups=[],seeds=[dict(row,seed=s) for s in [4108,4109,4110]])
    with pytest.raises(ValueError,match='leaked'):
        compare(payload)
    payload['calibration_groups']=[]
    assert compare(payload)['acceptance']=='pass'
    payload['label']='proxy'
    with pytest.raises(ValueError,match='measured'):
        compare(payload)
