"""Review regressions using project-generated proxy signals, never measured audio.

Acceptance preregistration SHA256: 1f780baf50b7f220c476cb4dd6752d097e68aa6a685ede805d782ef7e35e388f
"""
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from tests.unit.sources import target_standins as L
from sensorsim.sources import intake_chamber, intake_known_digital, LoudspeakerResponse, render_source
from sensorsim.sources.catalog import AssignedSource, as_dry_clip, content_hash
from sensorsim.sources.intake import load_project_sources
from sensorsim.sources.replay import fit_replay, generate_replay


def speaker(fs=L.FS):
    return LoudspeakerResponse(np.array([.6, .3, -.2, .05]), fs, 'measured', 'proxy FIR')


def digital(x=None, **kw):
    if x is None:
        x = L.knock_take([0, .22, .44])[1]
    return intake_known_digital(x, L.FS, metadata=L.meta('knock', 'digital', 'known_digital'), **kw)


@pytest.mark.parametrize('onsets', [[0, .22, .44], [0, .18, .36, .7]])
@pytest.mark.parametrize('t60,drr', [(.3, 10), (.5, 0)])
def test_h1_wet_sequences_flagged_without_windows(onsets, t60, drr):
    _, ev = L.knock_take(onsets, dur=3.)
    x = L.wet(ev, L.FS, t60, drr, 10**(-12/20), 1e-4, 11)
    r = intake_chamber(x, L.FS, metadata=L.meta('knock', 'wet')).report
    assert 'late_decay_exceeds_limit' in r['rejected'] + r['review']
    assert r['late_decay']['bands']['500']['t20_s'] > .20


def test_h1_intrinsic_chime_decay_is_reviewable():
    x, _ = L.chime_take()
    r = intake_chamber(x, L.FS, metadata=L.meta('doorbell', 'ding-dong'),
                       dryness_window_s=(.49, 1.18, 11.5)).report
    assert r['status'] == 'review', r
    assert 'reverberant_or_source_decay' in r['review']
    assert r['late_decay']['bands']['500']['t20_s'] is not None


def test_h1_session_reference_is_content_bound_and_wet_reference_refused():
    x, _ = L.knock_take([0, .22, .44])
    ref = np.random.default_rng(981).normal(0, 1e-5, L.FS*2)
    pulse = np.random.default_rng(982).normal(size=500)*np.exp(-np.arange(500)/50)
    ref[L.FS//2:L.FS//2+500] += pulse*.05
    args = dict(metadata=L.meta('knock', 'session'), session_reference={'samples': ref, 'fs': L.FS,
                'session': 's1', 'evidence_label': 'proxy'})
    r = intake_chamber(x, L.FS, **args).report
    assert r['session_reference']['status'] == 'qualified'
    key = r['approval_key']
    ref[0] += 1e-6
    assert intake_chamber(x, L.FS, **args).report['approval_key'] != key
    wet_ref = np.zeros(L.FS*2)
    h = L.room_rir(L.FS, .5, 0.)*.15
    wet_ref[L.FS//2:L.FS//2+len(h)] = h
    args['session_reference']['samples'] = wet_ref + np.random.default_rng(983).normal(0, 1e-5, len(wet_ref))
    r = intake_chamber(x, L.FS, **args).report
    assert 'session_not_dry' in r['rejected']


def test_m1_no_implicit_speaker_and_chamber_cannot_opt_in():
    assert digital().take.details['loudspeaker_played'] is False
    x, _ = L.knock_take([0, .22, .44])
    r = intake_chamber(x, L.FS, metadata=L.meta('knock', 'physical'))
    with pytest.raises(ValueError, match='known_digital'):
        intake_chamber(x, L.FS, metadata=L.meta('knock', 'physical'), loudspeaker_played=True,
            loudspeaker=speaker(), approval=dict(reviewer='proxy', decision='approve', approval_key=r.report['approval_key']))


def test_m1_speaker_requires_playback_and_position_at_intake_and_render():
    with pytest.raises(ValueError, match='loudspeaker_played'):
        digital(loudspeaker=speaker())
    take = digital(loudspeaker_played=True, loudspeaker_position=[1., 2., 1.]).take
    with pytest.raises(ValueError, match='position'):
        render_source(AssignedSource(take, 'train'), np.array([1., .1]), fs=L.FS, loudspeaker=speaker())
    with pytest.raises(ValueError, match='position'):
        render_source(AssignedSource(take, 'train'), np.array([1., .1]), fs=L.FS,
                      loudspeaker=speaker(), source_position=[2., 2., 1.])
    y, _ = render_source(AssignedSource(take, 'train'), np.array([1., .1]), fs=L.FS,
                        loudspeaker=speaker(), source_position=[1., 2., 1.])
    np.testing.assert_allclose(y, np.convolve(np.convolve(take.samples, speaker().impulse), [1., .1]), atol=1e-14)


def test_m2_children_inherit_route_and_response_and_never_reapply():
    x, _ = L.knock_take([0, .22, .44])
    _, accepted = L.approve(x, L.FS, L.meta('knock', 'chamber'))
    fit = fit_replay([AssignedSource(accepted.take, 'train')], 'knock', enabled=True)
    child = generate_replay(fit, seed=31, enabled=True)
    assert child.take.capture_route == 'near_dry'
    with pytest.raises(ValueError, match='synthetic'):
        render_source(child, np.array([1., .1]), fs=L.FS, loudspeaker=speaker())
    assert as_dry_clip(child.take, split='train').source_reference.target == 'effective_source'
    with pytest.raises(ValueError, match='training-only'):
        as_dry_clip(child.take)
    parent = digital(loudspeaker_played=True, loudspeaker_position=[1., 2., 1.], loudspeaker=speaker()).take
    child = generate_replay(fit_replay([AssignedSource(parent, 'train')], 'knock', enabled=True), seed=32, enabled=True)
    assert child.take.details['loudspeaker_response'] == parent.details['loudspeaker_response']
    with pytest.raises(ValueError, match='synthetic|already applied'):
        render_source(child, np.array([1., .1]), fs=L.FS, loudspeaker=speaker(), source_position=[1., 2., 1.])


@pytest.mark.parametrize('gap,scale', [(.6, 1.), (.3, .3), (.25, .3)])
@pytest.mark.parametrize('selector', [{}, {'event_index': 0}, {'event_window_s': [.49, .52]}])
def test_m3_knock_sequence_keeps_all_hits(gap, scale, selector, monkeypatch):
    monkeypatch.setattr(L, 'DOOR', [(f, a, tau*scale) for f, a, tau in L.DOOR])
    x, _ = L.knock_take([0, gap, 2*gap], dur=3.)
    r = intake_chamber(x, L.FS, metadata=L.meta('knock', 'slow'), **selector).report
    assert r['event_count'] == 1, r
    assert 'contaminated' not in r['rejected']
    assert r['crop_samples'][1] > (.5+2*gap)*L.FS


def test_m4_digital_noisy_wet_capture_gets_acoustic_qa():
    _, ev = L.knock_take([0, .22, .44])
    r = digital(L.wet(ev, L.FS, .5, 0., .25, .003, 17))
    assert r.take is None
    assert r.report['noise_floor_dbfs'] > -90
    assert any(k in r.report['rejected'] + r.report['review'] for k in
               ('late_decay_exceeds_limit', 'late_decay_unavailable'))


@pytest.mark.parametrize('file', ['room_recordings/knock.wav', 'dry/knock__room.wav'])
@pytest.mark.parametrize('route', ['known_digital', 'near_dry'])
def test_m4_room_paths_are_never_dry(tmp_path, file, route):
    path = tmp_path/file
    path.parent.mkdir(parents=True)
    wavfile.write(path, L.FS, L.knock_take([0, .22, .44])[0].astype('float32'))
    with pytest.raises(ValueError, match='room_recordings|__room'):
        load_project_sources([dict(file=file, **L.meta('knock', 'room', route))], tmp_path, tmp_path/'qa.json')


@pytest.mark.parametrize('onsets', [[0, .22, .44], [0, .18, .36, .70]])
def test_m5_replay_preserves_impacts_and_varies_gain(onsets):
    x, _ = L.knock_take(onsets, dur=3.)
    _, r = L.approve(x, L.FS, L.meta('knock', 'pattern'))
    fit = fit_replay([AssignedSource(r.take, 'train')], 'knock', enabled=True)
    assert len(fit.templates[0]['onsets']) == len(onsets)
    outs = [generate_replay(fit, seed=s, enabled=True).take.samples for s in range(10)]
    assert len({o.tobytes() for o in outs}) == 10


def test_m5_chime_keeps_ding_dong():
    x, _ = L.chime_take()
    _, r = L.approve(x, L.FS, L.meta('doorbell', 'pattern'))
    fit = fit_replay([AssignedSource(r.take, 'train')], 'chime', enabled=True)
    assert len(fit.templates[0]['onsets']) == 2
    y = generate_replay(fit, seed=4, enabled=True).take.samples
    start = r.report['crop_samples'][0]
    a, b = round(.55*L.FS)-start, round(1.1*L.FS)-start
    def amp(z):
        return abs(np.mean(z[a:b]*np.exp(-2j*np.pi*523*np.arange(b-a)/L.FS)))
    assert amp(y) < .001, amp(y)
    assert not np.array_equal(y, generate_replay(fit, seed=6, enabled=True).take.samples)


def test_m5_kettle_retains_click_and_rising_rumble():
    x, _, fs = L.kettle_take()
    _, r = L.approve(x, fs, L.meta('kettle', 'kettle'), event_index=1)
    a = r.report['crop_samples'][0]
    b, c, n = round(50*fs)-a, round(64.9*fs)-a, len(r.take.samples)
    fit = fit_replay([AssignedSource(r.take, 'train')], 'kettle', enabled=True,
                         kettle_stages={'kettle': [[0, b], [b, c], [c, n]]})
    y = generate_replay(fit, seed=5, enabled=True).take.samples
    real = r.take.samples
    def crest(z): return 20*np.log10(np.max(abs(z))/np.sqrt(np.mean(z*z)))
    assert abs(crest(y[c:])-crest(real[c:])) < 1.
    assert abs(np.argmax(abs(y[c:]))-np.argmax(abs(real[c:]))) <= 1
    def rise(z): return 10*np.log10(np.mean(z[b-5*fs:b]**2)/np.mean(z[:5*fs]**2))
    assert abs(rise(y)-rise(real)) < 1.
    assert not np.array_equal(y, generate_replay(fit, seed=6, enabled=True).take.samples)


def test_l3_isolated_normalized_peak_is_review_not_clipping():
    x = L.knock_take([0, .22, .44])[1]
    x[np.argmax(x)] = 32767/32768
    r = digital(x).report
    assert r['status'] == 'review' and 'clipped' not in r['rejected']
    i = np.argmax(x)
    x[i:i+3] = 32767/32768
    assert 'clipped' in digital(x).report['rejected']


def test_l4_datasheet_spl_needs_relative_reference():
    with pytest.raises(ValueError, match='relative|reference_db'):
        LoudspeakerResponse.from_magnitude([0, 2000, 24000], [70, 88, 70], L.FS, reference='proxy')
    spk = LoudspeakerResponse.from_magnitude([0, 2000, 24000], [70, 88, 70], L.FS,
                                            reference='proxy', reference_db=88.)
    assert np.max(abs(spk.impulse)) < 1.


def test_h2_real_read_dry_refuses_in_situ_and_honors_split():
    from sensorsim.acoustics.room.dataset import read_dry, _source_pools
    take = digital().take
    clip = as_dry_clip(take, split='train')
    bad = replace(clip, provenance={**clip.provenance, 'capture_route': 'in_situ'})
    with pytest.raises(ValueError, match='in_situ'):
        read_dry(bad, L.FS)
    _, info = read_dry(clip, L.FS)
    assert info['capture_route'] == 'known_digital'
    assert _source_pools([info], 0., 43) == {'train': [0], 'val': []}


def test_l2_approval_keys_ignore_blas_thread_count(python_subprocess_env):
    code = '''from tests.unit.sources import target_standins as L
x, _ = L.knock_take([0, .22, .44])
r = L.intake_chamber(x, L.FS, metadata=L.meta('knock', 'key'), dryness_window_s=(.49, 1., 2.))
print(r.report['approval_key'])
'''
    keys = [subprocess.check_output([sys.executable, '-c', code], text=True,
        env={**python_subprocess_env, 'OPENBLAS_NUM_THREADS': str(n), 'OMP_NUM_THREADS': str(n)}, timeout=30).strip()
        for n in (1, 2)]
    assert keys[0] == keys[1]


def test_m3_alias_is_documented_and_equivalent():
    x, _ = L.knock_take([0, .6, 1.2], dur=3.)
    a = intake_chamber(x, L.FS, metadata=L.meta('knock', 'alias'), merge_gap_s=.7)
    b = intake_chamber(x, L.FS, metadata=L.meta('knock', 'alias'), event_merge_gap_s=.7)
    assert a.report['approval_key'] == b.report['approval_key']
    with pytest.raises(ValueError, match='not both'):
        intake_chamber(x, L.FS, metadata=L.meta('knock', 'alias'), event_merge_gap_s=.7, merge_gap_s=.7)


def test_l5_unlisted_flac_is_refused(tmp_path):
    wavfile.write(tmp_path/'take.wav', L.FS, L.knock_take([0, .22, .44])[0].astype('float32'))
    (tmp_path/'extra.flac').write_bytes(b'not decoded: must still be listed')
    row = dict(file='take.wav', **L.meta('knock', 'chamber'))
    with pytest.raises(ValueError, match='extra.flac'):
        load_project_sources([row], tmp_path, tmp_path/'qa.json')


def test_n2_digital_targets_are_refused_even_outside_played(tmp_path):
    wavfile.write(tmp_path/'take.wav', L.FS, L.knock_take([0, .22, .44])[1].astype('float32'))
    with pytest.raises(ValueError, match='playback recordings are for room calibration'):
        load_project_sources([dict(file='take.wav', **L.meta('knock', 'digital', 'known_digital'))], tmp_path, tmp_path/'qa.json')


def test_m1_dataset_refuses_speaker_audio_at_another_position(tmp_path):
    from sensorsim.acoustics.room.dataset import generate_dataset, PositionPair
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.dataset import SamplingSpec
    from sensorsim.acoustics.room.augment import AugmentationConfig
    from sensorsim.acoustics.room.synth import SynthConfig
    parent = digital().take
    audio = speaker().apply(parent.samples, L.FS)
    take = replace(parent, samples=audio, details={**parent.details,
        'qa': {'status': 'pass', 'crop_content_sha256': content_hash(audio, L.FS)},
        'loudspeaker_played': True, 'loudspeaker_position': [1.,2.,1.],
        'loudspeaker_response': speaker().metadata()})
    position = PositionPair(source_xyz_m=(2., 2., 1.), receiver_xyz_m=(3., 2., 1.))
    with pytest.raises(ValueError, match='position'):
        generate_dataset(RoomModel(shoebox_dims_m=(6,5,3)), SamplingSpec(),
                         [as_dry_clip(take, split='train')], AugmentationConfig(), 1, 88, tmp_path,
                         target_positions=[position], val_fraction=0, fs=L.FS,
                         synth_config=SynthConfig(late=False, max_order=0))
