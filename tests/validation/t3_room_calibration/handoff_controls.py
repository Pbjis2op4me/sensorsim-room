"""Development regressions for the owner-only handoff patch; run explicitly.

These intentionally fail on the unpatched lane. They are not in default pytest
collection, because this lane cannot edit shared room modules.
"""
import hashlib

import numpy as np
import pytest
import yaml

from sensorsim.acoustics.room.prior import RoomSurvey, build_prior, CalibrationObservation, update_ensemble
from sensorsim.acoustics.room.calibration_pool import pool_playback
from sensorsim.acoustics.room.dataset import DryClip, SamplingSpec, _ensemble_examples, generate_dataset
from sensorsim.acoustics.room.augment import AugmentationConfig


def survey():
    return RoomSurvey.model_validate(dict(room=dict(length_x=4.,width_y=3.,height_z=2.5),
        microphone=dict(x=1.,y=1.,z=1.,tolerance_m=.05),loudspeaker=dict(x=2.,y=2.,z=1.),
        source_sites={"knock":[dict(x=2.,y=1.,z=1.)]},prior=dict(count=4,seed=330)))


def test_shared_installation_across_whole_scene():
    ensemble=build_prior(survey())
    clips=[DryClip("knock",np.ones(16),8000,source_group=f"source-{i}") for i in range(2)]
    rows=list(_ensemble_examples(ensemble,2,330,["train","val"],np.random.SeedSequence(1).spawn(2),
        {"train":[0,1],"val":[0,1]},True,clips,SamplingSpec(),.2))
    for scene in (0,1):
        selected=[r for r in rows if r[6]==scene]
        assert len(selected)==2
        assert len({r[0].receiver_xyz_m for r in selected})==1
        assert len({r[4].model_hash for r in selected})==1


def test_pool_late_path_preserves_lineage_and_blocks_dataset(tmp_path):
    keys=["1000.t30_s","1000.c50_db"]
    observations=[CalibrationObservation(kind="known-source",source_group=f"repeat-{i}",
        input_sha256=hashlib.sha256(str(i).encode()).hexdigest(),values=dict(zip(keys,(.5,0.))),
        uncertainty=dict.fromkeys(keys,.1),source_xyz_m=(2.,2.,1.),receiver_xyz_m=(1.,1.,1.),
        qualified=True,metadata={"lineage":{"use":"internal-validation"}}) for i in range(2)]
    components=dict(keys=keys,units=dict(zip(keys,("s","dB"))),measurement_covariance=np.eye(2).tolist(),
        playback_covariance=np.zeros((2,2)).tolist(),simulator_covariance=np.zeros((2,2)).tolist(),
        discrepancy_covariance=np.zeros((2,2)).tolist(),policy_id="dev-handoff")
    pooled=pool_playback(observations,uncertainty_components=components)
    assert pooled.metadata["observations"][0]["metadata"]["lineage"]["use"]=="internal-validation"
    ensemble=build_prior(survey())
    updated=update_ensemble(ensemble,pooled,[dict(zip(keys,(.5,0.)))]*4,
        modular_uncertainty=pooled.metadata["modular_uncertainty"],early_tolerances={keys[1]:1.})
    assert updated.evidence[-1]["late_block"]["early_status"]=="compatible"
    assert updated.evidence[-1]["accepted"]
    with pytest.raises(ValueError,match="P4"):
        generate_dataset(updated,SamplingSpec(),
            [DryClip("knock",np.linspace(-.1,.1,128),8000,source_group="dry-control")],
            AugmentationConfig(),2,1,tmp_path/"dataset")


def test_opt_in_pipeline_without_playback_cannot_fall_back(tmp_path):
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    room=tmp_path/"room.yaml"
    room.write_text(yaml.safe_dump(survey().model_dump(mode="json")))
    (tmp_path/"calibration.yaml").write_text(yaml.safe_dump(dict(late_block=dict(uncertainty_components={},early_tolerances={}))))
    with pytest.raises(ValueError,match="no qualified playback"):
        run_no_sweep(room,tmp_path/"out")


def test_low_ess_modular_pipeline_stays_research_only(tmp_path,monkeypatch):
    import sensorsim.acoustics.room.prior as prior
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    from scipy.io import wavfile
    # Test only the integration route; signal-extraction qualification is stubbed
    # and cannot supply acoustic evidence. The real update and report path run.
    configured=survey().model_copy(update={"prior":survey().prior.model_copy(update={"count":8})})
    room=tmp_path/"room.yaml"; room.write_text(yaml.safe_dump(configured.model_dump(mode="json")))
    keys=["1000.t30_s","1000.c50_db"]
    components=dict(keys=keys,units=dict(zip(keys,("s","dB"))),measurement_covariance=[[1e-6,0.],[0.,1.]],
        playback_covariance=np.zeros((2,2)).tolist(),simulator_covariance=np.zeros((2,2)).tolist(),
        discrepancy_covariance=np.zeros((2,2)).tolist(),policy_id="dev-low-ess")
    (tmp_path/"calibration.yaml").write_text(yaml.safe_dump(dict(holdout_groups=[],late_block=dict(
        uncertainty_components=components,early_tolerances={keys[1]:1.}))))
    for folder,name in (("played","noise.wav"),("room_recordings","noise__room.wav")):
        (tmp_path/folder).mkdir()
        wavfile.write(tmp_path/folder/name,8000,np.arange(100,dtype=np.int16))
    def observation(*args,**kwargs):
        return CalibrationObservation(kind="known-source",source_group="noise",input_sha256="a"*64,
            values=dict(zip(keys,(.5,0.))),uncertainty=dict.fromkeys(keys,.1),qualified=True,
            source_xyz_m=(2.,2.,1.),receiver_xyz_m=(1.,1.,1.))
    monkeypatch.setattr(prior,"known_source_observation",observation)
    monkeypatch.setattr(prior,"predict_metrics",lambda ensemble,*a,**k:
        [dict(zip(keys,(.5 if i==0 else 1.,0.))) for i in range(len(ensemble.members))])
    import warnings
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        run_no_sweep(room,tmp_path/"out",fs=8000,evidence_label="proxy")
    assert not any("rendering uses domestic-room-assumption" in str(w.message) for w in observed)
    import json
    result=json.loads((tmp_path/"out/room_ensemble.json").read_text())
    assert result["purpose"]=="calibration"
    assert result["evidence"][-1]["late_block"]["status"]=="needs-refinement"
    report=json.loads((tmp_path/"out/report.json").read_text())
    assert report["rendering_distribution"]=="research-only modular candidate; training blocked"


@pytest.mark.parametrize('seed', [1, 2, 3, 4])
@pytest.mark.parametrize('near', [True, False])
def test_complete_scene_jointly_retries_common_receiver(seed, near):
    """F2 disclosed failing seeds 1/3, passing seeds 2/4, ordinary controls."""
    configured = survey().model_dump(mode='json')
    configured['source_sites'] = {'knock': [dict(x=2., y=2., z=1.)],
        'bell': [dict(x=1.26 if near else 2.5, y=1., z=1.)]}
    ensemble = build_prior(RoomSurvey.model_validate(configured))
    clips = [DryClip(label, np.ones(16), 8000, source_group=label) for label in ('knock', 'bell')]
    common = np.array([.96, 1., 1.])
    # Independent geometric witness, not the sampler as its own oracle.
    assert np.max(abs(common - np.array([1., 1., 1.]))) <= .05
    for label in ('knock', 'bell'):
        source = ensemble.survey.source_sites[label][0].xyz
        assert np.linalg.norm(source-common) >= .25
    def sample():
        return list(_ensemble_examples(ensemble, 1, 330, ['train'],
            np.random.SeedSequence(seed).spawn(1), {'train': [0, 1]}, True, clips, SamplingSpec(), 0.))
    rows = sample()
    assert len(rows) == 2
    assert rows == sample()
    assert rows[0][0].receiver_xyz_m == rows[1][0].receiver_xyz_m
    assert [row[2] for row in rows] == [0, 1]
    assert {row[7] for row in rows} == set(ensemble.member_schedule(1, 330))
    for row in rows:
        assert row[1] == 'train' and row[5]['site_index'] == 0 and row[6] == 0
        assert np.linalg.norm(np.array(row[0].source_xyz_m)-row[0].receiver_xyz_m) >= .25


def test_impossible_scene_yields_no_partial_descendants():
    configured = survey().model_dump(mode='json')
    configured['microphone']['tolerance_m'] = 0.
    configured['source_sites'] = {'knock': [dict(x=2., y=2., z=1.)],
                                  'bell': [dict(x=1.1, y=1., z=1.)]}
    ensemble = build_prior(RoomSurvey.model_validate(configured))
    clips = [DryClip(label, np.ones(16), 8000, source_group=label) for label in ('knock', 'bell')]
    rows = []
    with pytest.raises(ValueError, match='no legal'):
        for row in _ensemble_examples(ensemble, 1, 330, ['train'],
                np.random.SeedSequence(1).spawn(1), {'train': [0, 1]}, True, clips, SamplingSpec(), 0.):
            rows.append(row)
    assert rows == []


def test_rejected_receiver_is_replaced_before_any_scene_yield(monkeypatch):
    """Force the verifier's invalid receiver, then its independent legal witness."""
    import sensorsim.acoustics.room.dataset as dataset
    configured = survey().model_dump(mode='json')
    configured['source_sites'] = {'knock': [dict(x=2., y=2., z=1.)],
                                  'bell': [dict(x=1.26, y=1., z=1.)]}
    ensemble = build_prior(RoomSurvey.model_validate(configured))
    clips = [DryClip(label, np.ones(16), 8000, source_group=label) for label in ('knock', 'bell')]
    bad, good = np.array([1.04, 1., 1.]), np.array([.96, 1., 1.])
    assert np.linalg.norm(np.array([1.26, 1., 1.])-bad) < .25
    assert np.linalg.norm(np.array([1.26, 1., 1.])-good) >= .25
    receivers = []
    class ReceiverDraws:
        def uniform(self, lo, hi):
            receiver = bad if not receivers else good
            assert np.all(receiver >= lo) and np.all(receiver <= hi)
            receivers.append(receiver.copy())
            return receiver.copy()
    original_rng = np.random.default_rng
    def controlled_rng(seed=None):
        # The dedicated scene receiver stream has a two-word seed. All other
        # source/member streams execute normally; missing hook fails below.
        return ReceiverDraws() if isinstance(seed, np.ndarray) and seed.shape == (2,) else original_rng(seed)
    monkeypatch.setattr(dataset.np.random, 'default_rng', controlled_rng)
    iterator = _ensemble_examples(ensemble, 1, 330, ['train'], np.random.SeedSequence(1).spawn(1),
                                  {'train': [0, 1]}, True, clips, SamplingSpec(), 0.)
    first = next(iterator)
    assert len(receivers) == 2  # neither a missing hook nor a partial first yield
    rows = [first, *iterator]
    assert len(rows) == 2
    assert all(row[0].receiver_xyz_m == tuple(good) for row in rows)
