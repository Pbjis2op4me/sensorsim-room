"""Development controls for assumed-room notes and weak-knock rendering."""
import json
import numpy as np
import pytest
import yaml
from sensorsim import project
from sensorsim.acoustics.room import pipeline, prior
from sensorsim.acoustics.room.render_prior import ASSUMED_ROOM_WARNING, DOMESTIC_T30_S
from tests.unit.acoustics_room.test_room_prior import survey


@pytest.mark.parametrize('status', ['assumed', 'weak-decay', 'known-source'])
def test_project_notes_disclose_assumed_room(status):
    summary = dict(labels_without_a_usable_take={}, event_detection=True, state='complete',
        check_positions={'total': 0}, targets_in_room={'refused': [], 'crops_overlapping_another_event': []},
        calibration_status=status, warnings=[])
    notes = project._gates('fit', summary)
    assert (ASSUMED_ROOM_WARNING in notes) == (status != 'known-source')


def test_weak_pipeline_renders_domestic_keeps_broad_evidence(tmp_path, monkeypatch):
    # Deliberately extreme broad posterior makes the wrong branch observable.
    broad = prior.build_prior(survey(prior=dict(count=4, t30_s=[[.1, .12]]*7)))
    broad = broad.model_copy(update={'status': 'weak-decay', 'evidence': ({'test': 'weak proxy'},)})
    original = prior.build_prior
    monkeypatch.setattr(prior, 'build_prior', lambda s, **kw: original(s, **kw) if kw else broad)
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(broad.survey.model_dump(mode='json')))
    with pytest.warns(RuntimeWarning, match='ROOM IS ASSUMED'):
        report = pipeline.run_no_sweep(tmp_path/'room.yaml', tmp_path/'out', fs=8000)
    rendered = prior.RoomEnsemble.model_validate_json((tmp_path/'out/room_ensemble.json').read_text())
    saved = prior.RoomEnsemble.model_validate_json((tmp_path/'out/calibration_ensemble.json').read_text())
    assert saved.ensemble_hash == broad.ensemble_hash
    assert rendered.purpose == 'domestic-render'
    assert rendered.status == 'assumed'
    assert report['calibration_status'] == 'weak-decay'
    assert report['rendering_distribution'] == 'domestic-room-assumption/1'
    assert all(lo <= t <= hi for m in rendered.members for t, (lo, hi) in zip(m.late_t60_s, DOMESTIC_T30_S))


def test_frame_flag_only_changes_sequence_contamination_review():
    from sensorsim.acoustics.room.capture_qa import assess_source_crop
    from sensorsim.acoustics.room.sweep_detect import detect_events
    from tests.unit.sources.final_standins import door, FS
    x = door(); noise = x[:FS//5]
    regions = detect_events(x, FS, noise, merge_gap_s=.7, post_roll_s=.5)
    kwargs = dict(label='knock', incompatible_rises_samples=[FS], heterogeneous_hits=True)
    good = assess_source_crop(x, FS, regions[0], regions, noise, **kwargs)
    assert good['status'] == 'review'
    assert 'heterogeneous_knock_hits_require_listening' in good['review']
    bad = assess_source_crop(x, FS, regions[0], regions, noise, **{**kwargs, 'heterogeneous_hits': False})
    assert 'contaminated' in bad['rejected']
    explicit = assess_source_crop(x, FS, regions[0], regions, noise, contaminated=True, **kwargs)
    assert 'contaminated' in explicit['rejected']
    with pytest.raises(ValueError, match='knock'):
        assess_source_crop(x, FS, regions[0], regions, noise, **{**kwargs, 'label': 'kettle'})
    with pytest.raises(ValueError, match='boolean'):
        assess_source_crop(x, FS, regions[0], regions, noise, **{**kwargs, 'heterogeneous_hits': 'true'})
