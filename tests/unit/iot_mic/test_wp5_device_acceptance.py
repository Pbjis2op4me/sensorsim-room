from types import SimpleNamespace
from dataclasses import replace
import wave

import numpy as np
from scipy import signal

from sensorsim import iot_mic, nn

FS = 16000
SEED = 20261008


def _peak_dbfs_at_1khz(x):
    x = np.asarray(x, dtype=float)
    x = x[FS//2:-FS//2]
    t = np.arange(x.size)/FS
    design = np.column_stack((np.sin(2*np.pi*1000*t), np.cos(2*np.pi*1000*t)))
    sine, cosine = np.linalg.lstsq(design, x, rcond=None)[0]
    return 20*np.log10(np.hypot(sine, cosine))


def test_wp5_a_01_94_db_spl_sine_maps_to_minus_26_dbfs_peak():
    profile = iot_mic.PROFILES["pi5_inmp441/1"]
    t = np.arange(2*FS)/FS
    y = profile.apply(np.sin(2*np.pi*1000*t), FS, np.random.default_rng(SEED))
    assert abs(_peak_dbfs_at_1khz(y) - (-26.0)) <= 0.1


def test_wp5_a_02_silence_has_absolute_minus_87_dbfs_a_noise():
    profile = iot_mic.PROFILES["pi5_inmp441/1"]
    y = profile.apply(np.zeros(3*FS), FS, np.random.default_rng(SEED))
    measured = 20*np.log10(np.sqrt(2)*iot_mic.a_weighted_rms(y, FS))
    assert abs(measured - (-87.0)) <= 0.5


def test_wp5_a_03_pi_profiles_do_not_reach_codec_agc_ns_or_packet_loss(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pi profile reached a non-deployment processing stage")

    monkeypatch.setattr(iot_mic, "codec", forbidden)
    monkeypatch.setattr(iot_mic, "agc", forbidden)
    monkeypatch.setattr(iot_mic, "noise_suppress", forbidden)
    monkeypatch.setattr(iot_mic, "packet_loss", forbidden)
    for profile_id in ("pi5_inmp441/1", "pi5_ics43434/1", "pi5_sph0645/1"):
        profile = iot_mic.PROFILES[profile_id]
        assert profile.codec is None and profile.agc is None and profile.noise_suppression is None
        assert profile.packet_loss == 0
        profile.apply(np.zeros(FS), FS, np.random.default_rng(SEED))


def test_inmp441_response_has_its_acoustic_corner_dc_block_and_passband():
    profile = iot_mic.PROFILES["pi5_inmp441/1"]
    sos = iot_mic._capsule_sos(profile, FS)
    frequencies = np.array([60.0, 1000.0, .423*FS])
    response = iot_mic.profile_response(profile, FS, frequencies)
    levels = 20*np.log10(np.maximum(np.abs(response), 1e-12))
    assert abs(levels[0] - (-3.01)) < .2
    assert abs(levels[1]) < .1
    assert abs(levels[2]) < .5
    dc_only = replace(profile, hp_hz=0.0, acoustic_corner_hz=0.0, passband_fraction_fs=None)
    _, dc_response = signal.sosfreqz(iot_mic._capsule_sos(dc_only, FS), worN=[3.7*FS/48000], fs=FS)
    assert abs(20*np.log10(abs(dc_response[0])) - (-3.01)) < .2


def test_inmp441_clips_at_120_db_spl_peak_and_truncates_24_to_16():
    profile = iot_mic.PROFILES["pi5_inmp441/1"]
    t = np.arange(FS)/FS
    below_limit = profile.apply(10**(25/20)*np.sin(2*np.pi*1000*t), FS,
                                np.random.default_rng(SEED))
    assert below_limit.max() < .9
    y = profile.apply(10**(26/20)*np.sin(2*np.pi*1000*t), FS, np.random.default_rng(SEED))
    assert y.max() == 32767/32768 and y.min() == -1.0
    np.testing.assert_array_equal(y*32768, np.round(y*32768))


def _empty_target_corpus():
    background = 0.1*np.sin(2*np.pi*440*np.arange(4*FS)/FS)
    return SimpleNamespace(background=[background], classes=[], targets={}, spans={})


def test_make_example_uses_pi_profile_by_default_and_applies_the_spl_reference(monkeypatch):
    corpus = _empty_target_corpus()
    calls = []
    original_apply = iot_mic.MicChain.apply

    def observe_apply(self, x, fs, rng, *, ref_rms=None, input_reference_spl_db=94.0):
        calls.append((self.profile_id, ref_rms, input_reference_spl_db))
        return original_apply(self, x, fs, rng, ref_rms=ref_rms,
                              input_reference_spl_db=input_reference_spl_db)

    monkeypatch.setattr(iot_mic.MicChain, "apply", observe_apply)
    monkeypatch.setattr(iot_mic, "random_chain", lambda rng: (_ for _ in ()).throw(
        AssertionError("random_chain must not be used by the in-distribution split")))
    cfg = {"fs": FS, "window_s": 0.5, "p_background_only": 1.0, "p_mic_chain": 0.0,
           "render_reference_spl_db": 88.0, "background_domain": "spl_referenced"}
    audio, labels = nn.make_example(corpus, np.random.default_rng(SEED), cfg)
    assert audio.shape == (FS//2,) and not labels.any()
    assert calls == [("pi5_inmp441/1", None, 88.0)]


def test_random_chain_is_only_selected_by_the_explicit_stress_split(monkeypatch):
    calls = []

    class StressChain:
        def apply(self, x, fs, rng, *, ref_rms=None):
            calls.append(ref_rms)
            return x

    monkeypatch.setattr(iot_mic, "random_chain", lambda rng: StressChain())
    cfg = {"fs": FS, "window_s": 0.5, "p_background_only": 1.0, "p_mic_chain": 1.0,
           "device_split": "stress"}
    nn.make_example(_empty_target_corpus(), np.random.default_rng(SEED), cfg)
    assert len(calls) == 1 and calls[0] > 0


def test_wp5_d_01_capture_self_test_checks_tone_extremes_and_feature_parity():
    from tools.capture_check import self_test

    result = self_test(seed=SEED, fs=FS)
    assert result["evidence_label"] == "proxy result"
    assert result["s32_extremes_pass"] and result["tone_conversion_pass"]
    assert abs(result["tone_conversion_error_db"]) <= 0.01
    assert result["feature_parity_pass"] and result["feature_max_abs_difference"] <= 1e-5


def _write_s16(path, samples):
    samples = np.clip(np.rint(samples*32768), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(FS)
        output.writeframes(samples.tobytes())


def test_capture_check_reads_pre_post_quiet_and_known_level_tone_as_proxy(tmp_path):
    from tools.capture_check import inspect_capture

    rng = np.random.default_rng(SEED)
    quiet = iot_mic.PROFILES["pi5_inmp441/1"].apply(np.zeros(FS), FS, rng)
    t = np.arange(FS)/FS
    take = np.concatenate((quiet, .1*np.sin(2*np.pi*800*t), quiet))
    take_path, tone_path = tmp_path/"take.wav", tmp_path/"tone.wav"
    _write_s16(take_path, take)
    _write_s16(tone_path, 10**(-26/20)*np.sin(2*np.pi*1000*np.arange(2*FS)/FS))
    result = inspect_capture(take_path, fs=FS, pre_quiet_s=1, post_quiet_s=1,
                             tone_wav=tone_path, tone_spl_db=94,
                             evidence_label="proxy result")
    assert result["evidence_label"] == "proxy result"
    assert result["s16_capture_format"] == "pass" and result["below_0_99_fs"]
    assert result["quiet_floor"]["pre_take_dbfs_a"] is not None
    assert result["quiet_floor"]["post_take_dbfs_a"] is not None
    assert result["quiet_floor"]["within_3db_of_profile_floor"]
    assert result["calibration_tone"]["pass_within_sensitivity_tolerance"]
    assert result["feature_parity"]["max_abs_feature_difference"] == 0.0
    assert result["capture_quality_status"] == "inconclusive"
    assert result["pcm_conversion"]["status"] == "not_checked"
    assert result["wp5b_per_band_smd_and_c2st"].startswith("pending")
    assert result["wp5c_heldout_recall_and_audit_ratio"].startswith("pending")
