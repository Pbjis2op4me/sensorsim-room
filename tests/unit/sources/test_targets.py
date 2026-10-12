"""WP6 proxy contracts; no physical source-fidelity or recognition claim.

Pre-registration: bf17ed12d1e23db533d98ba12b57ec3f7b9e3a6adfaa22d7865ea4a3fc772b2c
"""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from sensorsim.sources import (SourceTake, AssignedSource, audit_splits, split_sources,
    intake_chamber, intake_known_digital, render_source, LoudspeakerResponse)
from sensorsim.sources.catalog import content_hash, as_dry_clip
from sensorsim.sources.replay import fit_replay, generate_replay

from tests.unit.sources.target_standins import approved_digital, approve_project_rows

FS = 8000


def metadata(group='take-1', label='knock', route='near_dry'):
    return dict(label=label, source_group=group, object_id=label+'-object', session='session-1',
                capture_route=route, evidence_label='proxy')


def chamber(seed=610, *, wet=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1e-5, 2*FS)
    t = np.arange(round(.12*FS))/FS
    event = .2*np.sin(2*np.pi*700*t)*np.exp(-t/.009)
    if wet:
        h = rng.normal(size=round(.4*FS))*np.exp(-np.arange(round(.4*FS))/FS/.12)
        h[0] += 1
        event = signal.fftconvolve(event, h)
        event *= .3/np.max(abs(event))
    x[round(.4*FS):round(.4*FS)+len(event)] += event
    return x


def approved(x=None, **kwargs):
    x = chamber() if x is None else x
    options = dict(metadata=metadata(), dryness_window_s=(.39, .45, 1.1), **kwargs)
    first = intake_chamber(x, FS, **options)
    approval = dict(reviewer='proxy-fixture-review', decision='approve', approval_key=first.report['approval_key'])
    return intake_chamber(x, FS, approval=approval, **options)


def digital(group='g1', label='knock', phase=0):
    x = .1*np.sin(np.arange(800)/FS*2*np.pi*731+phase)
    return approved_digital(x, FS, metadata=metadata(group, label, 'known_digital')).take


@pytest.mark.parametrize('name', ['source_group', 'object_id', 'session', 'capture_route'])
def test_mandatory_provenance(name):
    m = metadata()
    m[name] = ''
    with pytest.raises(ValueError, match=name):
        SourceTake(samples=chamber(), fs=FS, **m)


@pytest.mark.parametrize('seed', [610, 611, 612])
def test_proxy_chamber_requires_bound_listening_approval(seed):
    x = chamber(seed)
    first = intake_chamber(x, FS, metadata=metadata(), dryness_window_s=(.39, .45, 1.1))
    assert first.take is None and first.report['status'] == 'review'
    assert first.report['early_tail_ratio_db'] >= 10
    assert first.report['tail_energy_fraction'] <= .1
    result = approved(x)
    assert result.report['status'] == 'pass'
    assert result.take.parent_content_sha256 == content_hash(x, FS)
    assert result.take.samples.shape[0] < len(x)
    assert json.loads(json.dumps(result.report, allow_nan=False)) == result.report
    wrong = dict(reviewer='tester', decision='approve', approval_key=first.report['approval_key'])
    x[0] += 1e-6
    with pytest.raises(ValueError, match='approval_key'):
        intake_chamber(x, FS, metadata=metadata(), dryness_window_s=(.39, .45, 1.1), approval=wrong)


@pytest.mark.parametrize('defect', ['clipped', 'contaminated', 'low_sir', 'truncated', 'wet'])
def test_hard_crop_rejects_cannot_be_overridden(defect):
    x, opts = chamber(), {}
    if defect == 'clipped':
        x[round(.41*FS)] = 1.
    elif defect == 'contaminated':
        opts['contaminated'] = True
    elif defect == 'low_sir':
        opts['sir_db'] = 5.
    elif defect == 'truncated':
        x = x[:round(.7*FS)]
    else:
        x = chamber(wet=True)
        from tests.unit.sources import final_standins as F
        opts['session_reference'] = dict(F.reference(F.wet(F.clap())), session='session-1')
    dryness = None if defect == 'truncated' else (.39, .45, 1.1)
    first = intake_chamber(x, FS, metadata=metadata(), dryness_window_s=dryness, **opts)
    assert first.report['status'] == 'reject', first.report
    second = intake_chamber(x, FS, metadata=metadata(), dryness_window_s=dryness,
        approval=dict(reviewer='tester', decision='approve', approval_key=first.report['approval_key']), **opts)
    assert second.take is None and second.report['status'] == 'reject'


def test_overlapping_crop_is_contamination():
    x = chamber()
    t = np.arange(160)/FS
    x[round(.8*FS):round(.8*FS)+160] += .1*np.sin(2*np.pi*1300*t)
    result = intake_chamber(x, FS, metadata=metadata(), event_index=0, merge_gap_s=.15)
    assert result.report['status'] == 'reject'
    assert result.report['overlapping_event_indices'] == [1]


def test_ambiguous_detection_and_silence_do_not_become_sources():
    x = chamber()
    x[round(1.3*FS):round(1.42*FS)] += x[round(.4*FS):round(.52*FS)]
    result = intake_chamber(x, FS, metadata=metadata())
    assert result.take is None and result.report['event_count'] == 2
    result = intake_chamber(np.zeros(FS), FS, metadata=metadata())
    assert result.take is None
    assert result.report['status'] == 'reject'
    assert result.report['rejected'] == ['silent_or_sub_lsb']


def test_dryness_unavailable_is_not_physical_pass():
    result = intake_chamber(chamber(), FS, metadata=metadata())
    assert result.report['status'] == 'review'
    assert result.report['estimated_drr_db'] is None
    assert 'dryness_unavailable' in result.report['review']


@pytest.mark.parametrize('amplitude', [0., 1e-7, 1.])
def test_bad_digital_audio_rejected(amplitude):
    result = intake_known_digital(np.full(200, amplitude), FS,
        metadata=metadata(route='known_digital'))
    assert result.take is None and result.report['status'] == 'reject'


def test_renderer_requires_split_and_rejects_in_situ():
    take = digital()
    with pytest.raises(ValueError, match='split before'):
        render_source(take, np.array([1., .1]), fs=FS)
    room_take = replace(take, capture_route='in_situ')
    with pytest.raises(ValueError, match='in_situ'):
        render_source(AssignedSource(room_take, 'train'), np.array([1., .1]), fs=FS)
    with pytest.raises(ValueError, match='in_situ'):
        as_dry_clip(room_take)
    changed = replace(take, samples=take.samples*.5)
    with pytest.raises(ValueError, match='changed after crop QA'):
        render_source(AssignedSource(changed, 'train'), np.array([1., .1]), fs=FS)


@pytest.mark.parametrize('seed', [610, 611, 612])
def test_zero_group_or_hash_leakage_with_renamed_duplicate_and_sibling(seed):
    # Distinct labels cannot share exact audio, so offset the phase per class.
    takes = [digital(f'{label}-{i}', label, i*.17+c*.013) for c, label in enumerate(['knock', 'doorbell', 'kettle']) for i in range(9)]
    duplicate = replace(takes[0], source_group='renamed-duplicate')
    sibling = replace(takes[0], samples=chamber()[:800], source_group='renamed-sibling')
    takes += [duplicate, sibling]
    assigned = split_sources(takes, seed=seed)
    assert audit_splits(assigned)['groups_crossing'] == 0
    assert audit_splits(assigned)['hashes_crossing'] == 0
    assert assigned[0].split == assigned[-1].split == assigned[-2].split
    for split, count in [('RH', 9), ('val', 6), ('train', 12)]:
        assert sum(a.split == split for a in assigned[:27]) == count
    changed = list(assigned)
    changed[-1] = AssignedSource(changed[-1].take, 'val' if changed[0].split != 'val' else 'train')
    with pytest.raises(ValueError, match='split leakage'):
        audit_splits(changed)


def test_duplicate_labels_and_insufficient_groups_are_hard_errors():
    a = digital()
    with pytest.raises(ValueError, match='conflicting labels'):
        split_sources([a, replace(a, label='kettle')])
    with pytest.raises(ValueError, match='three independent'):
        split_sources([a])


def test_speaker_response_applies_once_and_retains_full_tail():
    take = digital()
    take = replace(take, details={**take.details, 'loudspeaker_played': True, 'loudspeaker_position': [1.,2.,1.]})
    h = np.array([.5, .25, -.1])
    speaker = LoudspeakerResponse(h, FS, 'measured', 'user-owned response')
    rir = np.array([1., 0., .2])
    y, info = render_source(AssignedSource(take, 'train'), rir, fs=FS, loudspeaker=speaker, source_position=[1.,2.,1.])
    np.testing.assert_allclose(y, np.convolve(np.convolve(take.samples, h), rir), atol=1e-15)
    assert len(y) == len(take.samples)+len(h)+len(rir)-2
    assert info['loudspeaker_response']['applied'] and info['result_label'] == 'model-based'
    with pytest.raises(ValueError, match='sample rate'):
        speaker.apply(take.samples, FS*2)
    with pytest.raises(ValueError, match='loudspeaker response requires'):
        render_source(AssignedSource(approved().take, 'train'), rir, fs=FS, loudspeaker=speaker, source_position=[1.,2.,1.])
    applied = approved_digital(take.samples, FS, metadata=metadata(route='known_digital'), loudspeaker=speaker,
        loudspeaker_played=True, loudspeaker_position=[1.,2.,1.]).take
    _, inherited = render_source(AssignedSource(applied, 'train'), rir, fs=FS, source_position=[1.,2.,1.])
    assert inherited['loudspeaker_response']['response_sha256'] == speaker.metadata()['response_sha256']
    with pytest.raises(ValueError, match='already applied'):
        render_source(AssignedSource(applied, 'train'), rir, fs=FS, loudspeaker=speaker, source_position=[1.,2.,1.])


def test_datasheet_hook_is_labelled_approximation_without_gain_normalization():
    speaker = LoudspeakerResponse.from_magnitude([0, 2000, 4000], [-6, -6, -6], FS,
                                                reference='user datasheet', taps=65)
    assert speaker.phase == 'linear-phase approximation'
    assert np.sum(speaker.impulse) == pytest.approx(10**(-6/20), rel=.01)
    with pytest.raises(ValueError, match='DC to Nyquist'):
        LoudspeakerResponse.from_magnitude([100, 3000], [0, 0], FS, reference='incomplete')


@pytest.mark.parametrize('family', ['knock', 'chime', 'kettle'])
def test_replay_is_opt_in_train_only_bounded_and_labelled(family):
    fs = FS
    x = np.zeros(fs)
    if family == 'kettle':
        x = np.random.default_rng(610).normal(size=fs)*.02
        x[:fs//3] = signal.lfilter([.1], [1, -.9], x[:fs//3])
        label = 'kettle'
    else:
        for onset in ([800, 2600, 4500] if family == 'knock' else [800]):
            t = np.arange(fs-onset)/fs
            x[onset:] += .1*np.sin(2*np.pi*730*t)*np.exp(-t/(.015 if family == 'knock' else .08))
            x[onset:] += .05*np.sin(2*np.pi*1270*t)*np.exp(-t/(.008 if family == 'knock' else .04))
        label = 'knock' if family == 'knock' else 'doorbell'
    take = approved_digital(x, fs, metadata=metadata('fit-1', label, 'known_digital')).take
    row = AssignedSource(take, 'train')
    opts = dict(kettle_stages={'fit-1': [[0, 2000], [2000, 6000], [6000, 8000]]}) if family == 'kettle' else {}
    with pytest.raises(ValueError, match='opt-in'):
        fit_replay([row], family)
    with pytest.raises(ValueError, match='training groups only'):
        fit_replay([AssignedSource(take, 'RH')], family, enabled=True, **opts)
    fit = fit_replay([row], family, enabled=True, **opts)
    with pytest.raises(ValueError, match='opt-in'):
        generate_replay(fit, seed=610)
    generated = generate_replay(fit, seed=610, enabled=True)
    again = generate_replay(fit, seed=610, enabled=True)
    assert np.array_equal(generated.take.samples, again.take.samples)
    assert np.max(abs(generated.take.samples)) <= np.max(abs(x))+1e-12
    assert generated.take.details['synthetic'] is True
    assert generated.take.details['pitch_semitones'] == 0.
    assert generated.take.details['stretch_factor'] == 1.
    assert generated.take.details['fit_groups'] == ['fit-1']
    assert audit_splits([row, generated])['status'] == 'pass'
    with pytest.raises(ValueError, match='split leakage|training-only'):
        audit_splits([AssignedSource(generated.take, 'val'), row])


def test_project_intake_qa_report_and_metadata_cache_key(tmp_path):
    from sensorsim import project
    root = tmp_path/'targets'
    root.mkdir()
    x = chamber()
    wavfile.write(root/'take.wav', FS, x.astype(np.float32))
    cfg = project.load_project(tmp_path)
    cfg['dataset']['sources'] = [dict(file='take.wav', **metadata())]
    approve_project_rows(cfg, tmp_path/'out')
    clips = project._clips(cfg, tmp_path/'out')
    assert clips[0].provenance['object_id'] == 'knock-object'
    assert clips[0].source_reference.target == 'effective_source'
    report = json.loads((tmp_path/'out'/'source_qa.json').read_text())
    assert report['status'] == 'pass'
    cfg['code'] = 'test'
    before = project._digest(project._inputs('dataset', cfg, {'fit': {'outputs': 'test'}}))
    cfg['dataset']['sources'][0]['session'] = 'different-session'
    after = project._digest(project._inputs('dataset', cfg, {'fit': {'outputs': 'test'}}))
    assert before != after


def test_project_chamber_writes_review_report_then_accepts(tmp_path):
    from sensorsim import project
    root = tmp_path/'targets'
    root.mkdir()
    wavfile.write(root/'take.wav', FS, chamber().astype(np.float32))
    cfg = project.load_project(tmp_path)
    cfg['dataset']['sources'] = [dict(file='take.wav', **metadata(), dryness_window_s=[.39, .45, 1.1])]
    out = tmp_path/'out'
    with pytest.raises(ValueError, match='source intake stopped'):
        project._clips(cfg, out)
    report = json.loads((out/'source_qa.json').read_text())
    cfg['dataset']['sources'][0]['approval'] = dict(reviewer='proxy tester', decision='approve',
        approval_key=report['clips'][0]['approval_key'])
    clips = project._clips(cfg, out)
    assert len(clips) == 1 and clips[0].source_reference.target == 'effective_source'


def test_real_data_acceptance_plan_stays_pending_and_reserves_RH():
    from sensorsim.sources.evaluation import prepare_acceptance
    takes = [digital(f'{label}-{i}', label, i*.17+c*.013)
             for c, label in enumerate(['knock', 'doorbell', 'kettle']) for i in range(9)]
    assigned = split_sources(takes)
    plan = prepare_acceptance(assigned, training_steps=100, examples_per_run=1000)
    assert plan['status'] == plan['b']['status'] == plan['c']['status'] == 'pending'
    assert len(plan['jobs']) == 18 and len(plan['RH']) == 9
    assert len(plan['shortages']) == 3
    heldout = {r['source_group'] for r in plan['RH']}
    for job in plan['jobs']:
        assert not heldout.intersection(g for groups in job['source_groups'].values() for g in groups)
        assert job['training_steps'] == 100 and job['examples_per_run'] == 1000
    json.dumps(plan, allow_nan=False)


def test_noise_floor_gate_and_short_quiet_remain_explicit():
    from sensorsim.acoustics.room.capture_qa import assess_source_crop
    from sensorsim.acoustics.room.sweep_detect import EventRegion
    noise = np.full(800, .02)
    x = np.full(8000, .02)
    x[1600:2400] = .035
    region = EventRegion(1600, 2400, 800, 6400, .02, 10., False)
    qa = assess_source_crop(x, FS, region, [region], noise)
    assert qa['status'] == 'reject' and 'low_snr_proxy' in qa['rejected']
    assert qa['noise_floor_dbfs'] == pytest.approx(20*np.log10(.02))
    qa = assess_source_crop(chamber(), FS, region, [region], np.zeros(80))
    assert 'noise_floor_unavailable' in qa['review']


def test_learning_curve_counts_connected_performances_not_renamed_files():
    from sensorsim.sources.evaluation import prepare_acceptance
    take = digital()
    rows = [AssignedSource(replace(take, source_group=f'alias-{i}'), 'train') for i in range(9)]
    plan = prepare_acceptance(rows, training_steps=100, examples_per_run=1000)
    assert len(plan['shortages']) == 6
    for job in plan['jobs']:
        assert job['independent_counts'] == {'knock': 1}
        assert len(job['source_groups']['knock']) == 9


def test_project_rejects_untracked_paths_and_multichannel(tmp_path):
    from sensorsim import project
    root = tmp_path/'targets'
    root.mkdir()
    wavfile.write(tmp_path/'elsewhere.wav', FS, digital().samples.astype(np.float32))
    cfg = project.load_project(tmp_path)
    cfg['dataset']['sources'] = [dict(file='../elsewhere.wav', **metadata())]
    with pytest.raises(ValueError, match='hashed targets'):
        project._clips(cfg, tmp_path/'out')
    wavfile.write(root/'stereo.wav', FS, np.tile(digital().samples[:,None], (1,2)).astype(np.float32))
    cfg['dataset']['sources'][0]['file'] = 'stereo.wav'
    with pytest.raises(ValueError, match='requires mono'):
        project._clips(cfg, tmp_path/'out')


def test_wp4_route_patch_hooks_reject_in_situ_and_frozen_split_leaks():
    from sensorsim.acoustics.room.dataset import read_dry
    from sensorsim.sources.dataset_routes import route_metadata, route_keys, declared_source_pools
    a, b = digital('group-a'), digital('group-b', phase=.1)
    clips = [as_dry_clip(a, split='train'), as_dry_clip(b, split='val')]
    catalog = [read_dry(c, FS)[1] for c in clips]
    assert route_metadata(clips[0])['split'] == 'train'
    assert declared_source_pools(catalog) == {'train': [0], 'val': [1]}
    assert ('content', a.parent_content_sha256) in route_keys(catalog[0])
    bad = replace(clips[0], provenance={**clips[0].provenance, 'capture_route': 'in_situ'})
    with pytest.raises(ValueError, match='in_situ'):
        route_metadata(bad)
    leaked = dict(catalog[1], provenance={**catalog[1]['provenance'],
                                      'parent_content_sha256': a.parent_content_sha256})
    with pytest.raises(ValueError, match='split leakage'):
        declared_source_pools([catalog[0], leaked])


def test_wp4_route_patch_checks_audio_against_approval():
    from sensorsim.sources.dataset_routes import validate_route_audio
    take = digital()
    clip = as_dry_clip(take)
    validate_route_audio(clip, take.samples, take.fs)
    with pytest.raises(ValueError, match='changed after crop QA'):
        validate_route_audio(clip, take.samples*.9, take.fs)


@pytest.mark.parametrize('label', ['doorbell', 'kettle'])
def test_nonimpulsive_chamber_sources_keep_unknown_dryness_until_listening(label):
    rng = np.random.default_rng(611)
    x = rng.normal(0, 1e-5, FS*2)
    t = np.arange(4000)/FS
    event = (.1*np.sin(2*np.pi*960*t)*np.exp(-t/.1) if label == 'doorbell'
             else rng.normal(0, .025, len(t))*np.sin(np.pi*np.arange(len(t))/len(t))**2)
    x[3200:7200] += event
    m = metadata(label=label)
    first = intake_chamber(x, FS, metadata=m)
    assert first.report['status'] == 'review' and first.report['estimated_drr_db'] is None
    accepted = intake_chamber(x, FS, metadata=m, approval=dict(reviewer='proxy listener',
        decision='approve', approval_key=first.report['approval_key']))
    assert accepted.take.label == label
    assert accepted.report['status'] == 'pass'
    assert 'dryness_unavailable' in accepted.report['review']
