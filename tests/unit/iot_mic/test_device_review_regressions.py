"""Model-based regressions for the independent device-chain review (D1-D8)."""
from dataclasses import replace
from types import SimpleNamespace
import inspect
import json
import wave

import numpy as np
import pytest
from scipy import signal

from sensorsim import edge, iot_mic, nn, project
from sensorsim.acoustics.weighting import a_weighted_mean_square, a_weighting_exact_db
from tools import capture_check

FS = 16000


def write_pcm(path, audio, fs=FS, channels=1):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(np.clip(np.rint(audio*32768), -32768, 32767).astype('<i2').tobytes())


def corpus(bg=None):
    t = np.arange(FS)/FS
    tone = np.sin(2*np.pi*1000*t)
    if bg is None:
        bg = np.tile(np.array([1, -1, 2, -2], dtype=float)/32768, FS)
    return SimpleNamespace(background=[bg], classes=['tone'], targets={'tone': [tone]}, spans={'tone': [(0, FS)]})


def test_d1_floor_uses_independent_iec_meter_and_full_scale_sine_reference():
    y = iot_mic.PROFILES['pi5_inmp441/1'].apply(np.zeros(3*FS), FS, np.random.default_rng(37))
    floor = 10*np.log10(2*a_weighted_mean_square(y, FS))
    assert abs(floor + 87) <= .5


@pytest.mark.parametrize('frequency', [6000, 7000])
def test_d1_a_weighted_meter_matches_iec_at_high_frequencies(frequency):
    x = np.sin(2*np.pi*frequency*np.arange(FS)/FS)
    expected = 10**(a_weighting_exact_db(np.array([frequency]))[0]/20)/np.sqrt(2)
    assert abs(20*np.log10(iot_mic.a_weighted_rms(x, FS)/expected)) < .01


def test_d2_recorded_negative_is_preserved_sample_for_sample():
    bg = np.tile(np.array([3, -2, 4, -3], dtype=np.float32)/32768, FS)
    c = corpus(bg)
    cfg = {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), 'p_background_only': 1.0}
    y, labels = nn.make_example(c, np.random.default_rng(39), cfg)
    # Periodic background avoids depending on crop offset.
    assert np.array_equal(np.sort(y), np.sort(bg[:FS]))
    assert not labels.any()


def test_d2_positive_has_physical_level_even_in_silent_background():
    cfg = {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), 'p_background_only': 0.0, 'event_spl_db': [65., 65.]}
    y, label = nn.make_example(corpus(np.zeros(4*FS)), np.random.default_rng(4), cfg)
    peak = capture_check.fit_sine_peak_dbfs(y, FS)
    assert label[0] == 1 and abs(peak - (-55.)) < .2


def test_d2_spl_bed_is_explicit_and_passes_through_device():
    bg = .01*np.sin(2*np.pi*1000*np.arange(4*FS)/FS)
    cfg = {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), 'p_background_only': 1., 'background_domain': 'spl_referenced'}
    y, _ = nn.make_example(corpus(bg), np.random.default_rng(4), cfg)
    assert abs(capture_check.fit_sine_peak_dbfs(y, FS) + 66) < .2
    with pytest.raises(ValueError, match='background_domain'):
        nn.make_example(corpus(bg), np.random.default_rng(4), {**cfg, 'background_domain': 'typo'})


def test_d3_persistence_and_training_share_chain_and_never_draw_stress(monkeypatch):
    def forbidden(*args, **kw):
        pytest.fail('in-distribution fitting reached random_chain')
    monkeypatch.setattr(iot_mic, 'random_chain', forbidden)
    monkeypatch.setattr(edge, 'window_scores', lambda *args: np.ones((16, 1)))
    cfg = {**(nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), 'p_mic_chain': 1.}
    nn._persistence(SimpleNamespace(features=SimpleNamespace(n=FS)), corpus(), cfg, seed=40, per_class=2)
    assert 'random IoT mic chains' not in inspect.getsource(nn)
    assert 'mix_device_audio(' in inspect.getsource(nn.make_example)
    assert 'mix_device_audio(' in inspect.getsource(nn._persistence)


def test_d4_project_settings_accept_and_forward_device_contract():
    r = {'type': 'cnn', 'device_split': 'stress', 'device_profile': 'pi5_ics43434/1',
         'render_reference_spl_db': 110., 'background_domain': 'spl_referenced', 'event_spl_db': [55., 70.]}
    checked = project._recognizer_settings({'recognizer': r, 'dataset': {'fs': FS}})
    assert checked == r
    assert all(k in (nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}) for k in r if k != 'type')


def test_d5_nominal_floor_and_in_spec_tone_are_accepted(tmp_path):
    noise = np.random.default_rng(41).standard_normal(3*FS)
    noise *= 10**(-87/20)/np.sqrt(2*a_weighted_mean_square(noise, FS))
    take, tone = tmp_path/'take.wav', tmp_path/'tone.wav'
    write_pcm(take, noise)
    write_pcm(tone, 10**(-28/20)*np.sin(2*np.pi*1000*np.arange(2*FS)/FS))
    r = capture_check.inspect_capture(take, pre_quiet_s=1, post_quiet_s=1,
                                     tone_wav=tone, tone_spl_db=94, evidence_label='proxy result')
    assert r['quiet_floor']['within_3db_of_profile_floor']
    assert r['calibration_tone']['pass_within_sensitivity_tolerance']
    assert r['pcm_conversion']['status'] == 'not_checked'
    assert r['capture_quality_status'] == 'inconclusive'


def test_d5_feature_parity_uses_independent_training_reader(tmp_path, monkeypatch):
    from sensorsim import recognizer
    take = tmp_path/'take.wav'
    x = np.random.default_rng(42).standard_normal(2*FS)*.01
    write_pcm(take, x)
    monkeypatch.setattr(recognizer, 'read_audio', lambda *args: np.zeros_like(x))
    r = capture_check.inspect_capture(take, evidence_label='proxy result')
    assert not r['feature_parity']['pass']


@pytest.mark.parametrize('rate,channels', [(48000, 1), (16000, 2), (48000, 2)])
def test_d5_wrong_format_fails_before_resampling_or_downmix(tmp_path, rate, channels):
    p = tmp_path/'wrong.wav'
    write_pcm(p, np.zeros((rate, channels)), fs=rate, channels=channels)
    r = capture_check.inspect_capture(p, evidence_label='proxy result')
    assert r['s16_capture_format'].startswith('fail')
    assert r['capture_quality_status'] == 'fail'


def test_d5_cli_does_not_override_selected_profile_sensitivity(tmp_path, monkeypatch, capsys):
    p = tmp_path/'tone.wav'
    write_pcm(p, 10**(-30/20)*np.sin(2*np.pi*1000*np.arange(FS)/FS))
    monkeypatch.setitem(iot_mic.PROFILES, 'pi5_ics43434/1',
                        replace(iot_mic.PROFILES['pi5_ics43434/1'], sensitivity_dbfs_at_94db_spl=-30))
    capture_check.main([str(p), '--tone-wav', str(p), '--tone-spl-db', '94', '--profile', 'pi5_ics43434/1', '--proxy'])
    r = json.loads(capsys.readouterr().out)
    assert r['calibration_tone']['expected_peak_dbfs'] == -30


def test_d6_ics_corner_and_other_profiles_document_missing_filter_evidence():
    p = iot_mic.PROFILES['pi5_ics43434/1']
    _, h = signal.sosfreqz(iot_mic._capsule_sos(p, FS), worN=[60], fs=FS)
    assert abs(20*np.log10(abs(h[0])) + 3.01) < .2
    for pid in ['pi5_ics43434/1', 'pi5_sph0645/1']:
        note = iot_mic.PROFILES[pid].evidence_note
        assert 'decimation' in note and 'DC block' in note and 'unreported response left unfiltered' not in note


def test_d7_decimation_passband_and_nyquist_are_datasheet_consistent():
    p = iot_mic.PROFILES['pi5_inmp441/1']
    # Measure actual processing, independent of the filter representation.
    p = replace(p, self_noise_dbfs_a=-300)
    levels = []
    for f in [4000., .423*FS, 7600., FS/2]:
        t = np.arange(2*FS)/FS
        x = np.cos(2*np.pi*f*t)
        y = p.apply(x, FS, np.random.default_rng(43))[FS:]
        amplitude = abs(np.dot(y, x[FS:])/np.dot(x[FS:], x[FS:]))
        # A linear-phase implementation may compensate delay or use sine+cos fit.
        if f != FS/2:
            basis = np.column_stack((np.cos(2*np.pi*f*t[FS:]), np.sin(2*np.pi*f*t[FS:])))
            amplitude = np.linalg.norm(np.linalg.lstsq(basis, y, rcond=None)[0])
        levels.append(20*np.log10(amplitude) + 26)
    assert abs(levels[1]) <= .5
    assert -6.1 <= levels[2] <= .5
    assert abs(levels[3] + 6) <= .5


def test_d8_alias_warns_and_public_sources_are_named():
    with pytest.warns(DeprecationWarning, match='absolute'):
        iot_mic.PROFILES['pi5_mems_unfiltered'].apply(np.zeros(FS), FS, np.random.default_rng(0))
    titles = ['InvenSense DS-INMP441-00 Rev 1.1', 'TDK ICS-43434 product specification',
              'Knowles SPH0645LM4H-B brief KAS-700-0137']
    for pid, title in zip(capture_check.PI_PROFILE_IDS, titles):
        assert title in iot_mic.PROFILES[pid].evidence_note
    limits = project.DEPLOY_TEMPLATE.split('## Limits')[1]
    assert '8 kHz' not in limits and 'gain-controlled' not in limits


def test_d4_trained_archive_and_deploy_record_effective_device_contract(tmp_path):
    pytest.importorskip('torch')
    rows = []
    for split in ['train', 'val']:
        for label in ['tone', 'background']:
            path = tmp_path/f'{split}_{label}.wav'
            x = (.01*np.sin(2*np.pi*1000*np.arange(2*FS)/FS) if label == 'tone'
                 else np.random.default_rng(len(split)).standard_normal(2*FS)*3e-5)
            write_pcm(path, x)
            rows.append({'file': path.name, 'label': label, 'split': split, 'fs': FS})
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    model_path = tmp_path/'model.npz'
    settings = {'device_split': 'in_distribution', 'device_profile': 'pi5_ics43434/1',
                'render_reference_spl_db': 102., 'event_spl_db': [60., 65.],
                'background_domain': 'device_recorded'}
    nn.train(manifest, model_path, steps=3, width=4, blocks=1, batch=4, workers=0,
             threads=1, val_windows=64, p_confuser_given_background=0., selection_mode='diagnostic_final', **settings)
    model = edge.Model(str(model_path))
    cfg = model.meta['training']['config']
    assert all(cfg[k] == v for k, v in settings.items())
    notes = project._deploy_notes(model)
    assert '| versioned device profile | pi5_ics43434/1 |' in notes
    assert '| device split | in_distribution |' in notes
    assert '| render peak-sine reference (dB SPL) | 102.0 |' in notes
