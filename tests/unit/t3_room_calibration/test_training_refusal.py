"""Disclosed F1 counterexample and ordinary-room control; synthetic proxy only."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from sensorsim.acoustics.room import calibration_discrepancy as cd
from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.dataset import DryClip, SamplingSpec, generate_dataset
from sensorsim.acoustics.room.prior import RoomSurvey, RoomEnsemble, build_prior, CalibrationObservation


def fixture():
    values = {'1000.t30_s': .5, '1000.c50_db': 0.}
    survey = RoomSurvey.model_validate(dict(room=dict(length_x=4., width_y=3., height_z=2.5),
        microphone=dict(x=1., y=1., z=1.), loudspeaker=dict(x=2., y=2., z=1.),
        source_sites={'knock': [dict(x=2., y=1., z=1.)]}, prior=dict(count=4, seed=300)))
    budget = cd.decompose_uncertainty([values], keys=list(values),
        units={'1000.t30_s': 's', '1000.c50_db': 'dB'}, repeat_ids=['guard-control'],
        content_sha256=['a'*64], measurement_covariance=np.eye(2),
        playback_covariance=np.zeros((2, 2)), simulator_covariance=np.zeros((2, 2)),
        discrepancy_covariance=np.zeros((2, 2)), policy_id='disclosed-verifier-f1')
    obs = CalibrationObservation(kind='known-source', source_group='guard-control',
        input_sha256='a'*64, values=values, uncertainty=dict.fromkeys(values, .1),
        source_xyz_m=(2., 2., 1.), receiver_xyz_m=(1., 1., 1.), qualified=True)
    return build_prior(survey), obs, budget


def generate(ensemble, path):
    clip = DryClip('knock', np.random.default_rng(156).normal(0, .04, 1600), 16000,
                   source_group='guard-dry')
    return generate_dataset(ensemble, SamplingSpec(), [clip], AugmentationConfig(),
        1, 3, path, fs=16000, val_fraction=0, allow_train_only=True)


def test_ordinary_room_still_generates_training_row(tmp_path):
    ensemble, _, _ = fixture()
    manifest = generate(ensemble, tmp_path/'ordinary')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 1
    assert len(list(manifest.parent.glob('*.wav'))) == 1


@pytest.mark.parametrize('branch', ['updated', 'early-mismatch', 'low-ess', 'unqualified', 'prior-conflict'])
def test_modular_result_cannot_enter_existing_dataset_api(tmp_path, branch):
    ensemble, obs, budget = fixture()
    if branch == 'low-ess':
        ensemble = build_prior(ensemble.survey.model_copy(update={
            'prior': ensemble.survey.prior.model_copy(update={'count': 8})}))
    predictions = [dict(obs.values) for _ in ensemble.members]
    if branch == 'early-mismatch':
        predictions = [dict(row, **{'1000.c50_db': 3.}) for row in predictions]
    elif branch == 'low-ess':
        predictions = [dict(row, **{'1000.t30_s': .5 if i == 0 else 1e10})
                       for i, row in enumerate(predictions)]
    elif branch == 'unqualified':
        obs = obs.model_copy(update={'qualified': False})
    elif branch == 'prior-conflict':
        ensemble = ensemble.model_copy(update={'status': 'prior-conflict'})
    with patch.object(cd, 'late_block_update', wraps=cd.late_block_update) as update:
        result = cd.update_room_ensemble(ensemble, obs, predictions, uncertainty=budget,
                                        early_tolerances={'1000.c50_db': 1.})
    assert update.call_count == (0 if branch in ('unqualified', 'prior-conflict') else 1)
    if branch == 'early-mismatch':
        assert result.evidence[-1]['late_block']['early_status'] == 'mismatch'
    if branch == 'low-ess':
        assert result.evidence[-1]['late_block']['status'] == 'needs-refinement'
    if branch == 'updated':
        assert result.evidence[-1]['accepted']
        assert result.evidence[-1]['training_promotion'] is False
    with pytest.raises(ValueError, match='P4'):
        generate(result, tmp_path/branch)
    assert not (tmp_path/branch).exists()
    # Serialization must not quietly turn the research result into a trainable
    # RoomEnsemble in another process or a legacy JSON-loading consumer.
    with pytest.raises(ValueError):
        RoomEnsemble.model_validate_json(result.model_dump_json())


def test_research_roundtrip_and_repeated_update_remain_blocked(tmp_path):
    ensemble, obs, budget = fixture()
    result = cd.update_room_ensemble(ensemble, obs, [obs.values]*4, uncertainty=budget,
                                    early_tolerances={'1000.c50_db': 1.})
    assert hasattr(cd, 'ResearchRoomEnsemble')
    restored = cd.ResearchRoomEnsemble.model_validate_json(result.model_dump_json())
    assert restored.ensemble_hash == result.ensemble_hash
    again = cd.update_room_ensemble(restored, obs, [obs.values]*4, uncertainty=budget,
                                   early_tolerances={'1000.c50_db': 1.})
    assert not again.evidence[-1]['accepted']
    assert again.weights == result.weights
    with pytest.raises(ValueError, match='P4'):
        generate(again, tmp_path/'again')
