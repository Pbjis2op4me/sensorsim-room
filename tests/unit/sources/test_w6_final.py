"""Final target-intake regression contracts; all audio and results are proxy."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
import yaml

from sensorsim import project
from sensorsim.sources import intake_chamber, intake_known_digital
from sensorsim.sources.decay import qualify_session
from sensorsim.sources.intake import load_project_sources
from tests.unit.sources import final_standins as F


@pytest.mark.parametrize('panel', range(3))
@pytest.mark.parametrize('with_reference', [False, True])
def test_n1_independent_resonant_door_is_reviewable(panel, with_reference):
    kwargs = {'session_reference': F.reference()} if with_reference else {}
    first = intake_chamber(F.door(panel), F.FS, metadata=F.meta(), **kwargs)
    assert first.report['late_decay']['bands']['500']['t20_s'] > .20
    assert first.report['status'] == 'review', first.report
    if not with_reference:
        assert 'record a session clap to decide automatically' in first.report['review_messages']['late_decay_exceeds_limit']
    approved = intake_chamber(F.door(panel), F.FS, metadata=F.meta(), **kwargs,
        approval=dict(reviewer='proxy listener', decision='approve', approval_key=first.report['approval_key']))
    assert approved.take is not None


def test_n1_short_clap_provides_band_bounds_and_comparison_without_fixed_floor():
    ref = qualify_session(F.reference(), session='final-session')
    assert ref['status'] == 'qualified'
    assert all(ref['decay']['bands'][str(b)]['comparison_decay_s'] > 0 for b in (500, 1000, 2000))
    r = intake_chamber(F.door(2), F.FS, metadata=F.meta(), session_reference=F.reference()).report
    assert 'take_slower_than_session' in r['review']
    assert r['session_decay_comparison']['500']['limit_s'] == pytest.approx(
        1.5*ref['decay']['bands']['500']['comparison_decay_s'])
    assert r['session_decay_comparison']['500']['limit_s'] < .20


def test_n1_wet_session_is_hard_gate_and_no_reference_is_review():
    x = F.wet(F.door())
    r = intake_chamber(x, F.FS, metadata=F.meta(), session_reference=F.reference(F.wet(F.clap()))).report
    assert 'session_not_dry' in r['rejected']
    no_ref = intake_chamber(x, F.FS, metadata=F.meta()).report
    assert no_ref['status'] == 'review'
    assert 'late_decay_exceeds_limit' in no_ref['review']


@pytest.mark.parametrize('delay', [.6, .9])
def test_n3_stray_after_final_knock_is_not_merged_as_a_hit(delay):
    x = F.door(gap=.24)
    t = np.arange(round(.07*F.FS))/F.FS
    a = round((.55+.48+delay)*F.FS)
    x[a:a+len(t)] += .07*np.sin(2*np.pi*1381*t)*np.exp(-t/.012)
    r = intake_chamber(x, F.FS, metadata=F.meta(), event_index=0).report
    assert 'contaminated' in r['rejected'], r
    assert r['event_samples'][1] < a
    assert r['premerge_regions'] and len(r['impact_onsets_samples']) >= 4


@pytest.mark.parametrize('label,kind', [('door_knock', None), ('knocking', None), ('front entrance', 'knock')])
def test_n7_knock_kind_independent_of_label(label, kind):
    kwargs = {} if kind is None else {'kind': kind}
    r = intake_chamber(F.door(gap=.65), F.FS, metadata=F.meta(label), **kwargs).report
    assert r['detection']['merge_gap_s'] == .7
    assert r['kind'] == 'knock'
    assert r['event_count'] == 1
    assert 'contaminated' not in r['rejected']
    assert r['event_samples'][1] > 1.85*F.FS


def test_n2_project_three_failed_runs_never_leave_dataset(tmp_path):
    from tests.unit.project.test_project import _layout, _tone, quiet
    root = _layout(tmp_path/'p')
    targets = root/'targets'
    targets.mkdir(exist_ok=True)
    wavfile.write(targets/'playback.wav', 24000, _tone(700., 0))
    row = dict(file='playback.wav', **F.meta())
    row.update(capture_route='known_digital', digital_origin='proxy tone', loudspeaker_played=True,
               loudspeaker_position=[1., 1., 1.])
    (root/'project.yaml').write_text(yaml.safe_dump(dict(dataset=dict(count=3, fs=16000, seed=1, sources=[row]))))
    for _ in range(3):
        with pytest.raises(ValueError, match='playback recordings are for room calibration and comparison'):
            project.run(root, echo=quiet)
        assert not (root/'out'/'dataset').exists()


def test_n2_dataset_positions_preflight_and_corrected_retry(tmp_path):
    from sensorsim.acoustics.room.dataset import generate_dataset, SamplingSpec, PositionPair
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.augment import AugmentationConfig
    from sensorsim.acoustics.room.synth import SynthConfig
    from sensorsim.sources.catalog import as_dry_clip
    # Use a short digital knock accepted on the base so N1 cannot mask N2.
    from tests.unit.sources import target_standins as L
    x = L.knock_take([0, .22, .44], fs=F.FS)[1]
    metadata = dict(F.meta(), capture_route='known_digital')
    options = dict(metadata=metadata, loudspeaker_played=True, loudspeaker_position=[1., 1., 1.])
    first = intake_known_digital(x, F.FS, **options)
    second = intake_known_digital(x, F.FS, **options, approval=dict(reviewer='proxy', decision='approve', approval_key=first.report['approval_key']))
    # Even the LAST of two positions must be checked before output is created.
    good = PositionPair(source_xyz_m=(1.,1.,1.), receiver_xyz_m=(3.,2.,1.))
    bad = PositionPair(source_xyz_m=(2.,1.,1.), receiver_xyz_m=(3.,2.,1.))
    out = tmp_path/'dataset'
    args = (RoomModel(shoebox_dims_m=(6,5,3)), SamplingSpec(), [as_dry_clip(second.take, split='train')],
            AugmentationConfig(), 2, 88, out)
    kwargs = dict(val_fraction=0, fs=F.FS, synth_config=SynthConfig(late=False, max_order=0))
    for _ in range(2):
        with pytest.raises(ValueError, match='position'):
            generate_dataset(*args, target_positions=[good, bad], **kwargs)
        assert not out.exists()
    generate_dataset(*args, target_positions=[good, good], **kwargs)
    assert (out/'manifest.jsonl').is_file()
    assert not (out/'manifest.partial.jsonl').exists()


def test_n5_drop_folder_ignores_non_targets_but_not_unlisted_dry(tmp_path):
    for d in ('dry', 'room_recordings', 'background', 'played'):
        (tmp_path/d).mkdir()
        wavfile.write(tmp_path/d/'take.wav', F.FS, F.door().astype('float32'))
    row = dict(file='dry/take.wav', **F.meta('door_knock'))
    report = tmp_path/'qa.json'
    with pytest.raises(ValueError, match='review'):
        load_project_sources([row], tmp_path, report)
    q = json.loads(report.read_text())['clips'][0]
    row['approval'] = dict(reviewer='proxy', decision='approve', approval_key=q['approval_key'])
    clips = load_project_sources([row], tmp_path, report)
    assert len(clips) == 1
    (tmp_path/'dry'/'unlisted.flac').write_bytes(b'unlisted')
    with pytest.raises(ValueError, match='unlisted.flac'):
        load_project_sources([row], tmp_path, report)


def test_n4_replay_cannot_be_claimed_as_procedural_or_c2st():
    from sensorsim.sources.procedural import fit_procedural
    with pytest.raises(ValueError, match='replay'):
        fit_procedural([], 'knock', enabled=True)

    from sensorsim.sources.replay import fit_replay, generate_replay
    from sensorsim.sources.catalog import AssignedSource
    from sensorsim.sources.evaluation import prepare_acceptance
    x = F.door()
    first = intake_chamber(x, F.FS, metadata=F.meta())
    accepted = intake_chamber(x, F.FS, metadata=F.meta(), approval=dict(
        reviewer='proxy', decision='approve', approval_key=first.report['approval_key']))
    parent = AssignedSource(accepted.take, 'train')
    child = generate_replay(fit_replay([parent], 'knock', enabled=True), seed=801, enabled=True)
    assert child.take.details['mode'] == 'replay'
    assert child.take.details['c2st_eligible'] is False
    np.testing.assert_array_equal(child.take.samples, parent.take.samples*child.take.details['amplitude_scale'])
    with pytest.raises(ValueError, match='replay.*excluded.*C2ST'):
        prepare_acceptance([parent, child], training_steps=10, examples_per_run=20)
