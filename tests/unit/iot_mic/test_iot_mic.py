"""Simulated IoT microphone chains: valid audio out, codecs band-limit and quantize as specified,
training draws stay inside their documented ranges, and the out-of-range profile is outside them."""
import numpy as np
import pytest

from sensorsim import iot_mic

FS = 16000


def _program(seconds=2.0, seed=0):
    """Speech-like test signal: noise bursts and tones at a room level, plus a quiet floor."""
    r = np.random.default_rng(seed)
    t = np.arange(int(seconds*FS))/FS
    tone = .1*np.sin(2*np.pi*440*t)*(np.sin(2*np.pi*3*t) > 0)
    return tone + .02*r.normal(size=t.size)


def _band_fraction(y, lo_hz):
    spectrum = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(y.size, 1/FS)
    return spectrum[freqs > lo_hz].sum()/spectrum.sum()


@pytest.mark.parametrize("name", sorted(iot_mic.PROFILES))
def test_every_profile_returns_valid_16_bit_audio_of_the_same_length(name):
    x = _program()
    y = iot_mic.PROFILES[name].apply(x, FS, np.random.default_rng(1), ref_rms=.02)
    assert y.shape == x.shape and np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0
    scale = 32768 if iot_mic.PROFILES[name].profile_id else 32767
    np.testing.assert_allclose(y*scale, np.round(y*scale), atol=1e-6)


@pytest.mark.parametrize("codec", ["mulaw8k", "adpcm8k"])
def test_narrowband_codecs_remove_everything_above_4_khz(codec):
    noise = .1*np.random.default_rng(2).normal(size=FS)
    assert _band_fraction(noise, 4200) > .4
    assert _band_fraction(iot_mic.codec(noise, FS, codec), 4200) < .01


def test_wideband_codec_keeps_speech_band_and_cuts_above_7_khz():
    noise = .1*np.random.default_rng(3).normal(size=FS)
    y = iot_mic.codec(noise, FS, "bandlimit7k")
    assert _band_fraction(y, 7300) < .01 and _band_fraction(y, 4200) > .2


@pytest.mark.parametrize("quantizer, min_snr_db", [(iot_mic.mulaw, 33.), (iot_mic.ima_adpcm, 24.)])
def test_quantizers_keep_their_expected_fidelity(quantizer, min_snr_db):
    t = np.arange(FS)/8000
    x = .3*np.sin(2*np.pi*300*t) + .1*np.sin(2*np.pi*1100*t)
    y = quantizer(x)
    snr = 10*np.log10(np.sum(x**2)/np.sum((y - x)**2))
    assert snr > min_snr_db


def test_agc_raises_quiet_audio_within_its_maximum_gain():
    x = .003*np.sin(2*np.pi*500*np.arange(2*FS)/FS)
    y = iot_mic.agc(x, FS, target_dbfs=-18., max_gain_db=20., attack_s=.01, release_s=.3)
    gain_db = 20*np.log10(np.std(y[FS:])/np.std(x[FS:]))
    assert 15. < gain_db <= 20.01 and np.max(np.abs(y)) <= 1.0


def test_same_seed_same_output():
    x = _program()
    a = iot_mic.random_chain(np.random.default_rng(5))
    b = iot_mic.random_chain(np.random.default_rng(5))
    assert a == b
    np.testing.assert_array_equal(a.apply(x, FS, np.random.default_rng(6)), b.apply(x, FS, np.random.default_rng(6)))


def test_training_draws_stay_in_their_documented_ranges():
    rng = np.random.default_rng(7)
    chains = [iot_mic.random_chain(rng) for _ in range(300)]
    assert all(15. <= c.hp_hz <= 300. for c in chains)
    assert all(1500. <= f <= 7500. and 2. <= g <= 12. and 1. <= q <= 4. for c in chains for f, g, q in c.resonances)
    assert any(c.dc_offset for c in chains) and all(abs(c.dc_offset) <= .05 for c in chains)
    assert all(-10. <= c.shelf_db <= 3. for c in chains)
    assert all(-30. <= c.self_noise_db <= -2. for c in chains)
    assert all(c.codec in iot_mic.CODECS for c in chains)
    assert {c.codec for c in chains} == set(iot_mic.CODECS)
    assert any(c.agc for c in chains) and any(c.noise_suppression for c in chains) and any(c.packet_loss for c in chains)


def test_the_out_of_range_profile_lies_outside_the_training_ranges():
    p = iot_mic.PROFILES["out_of_range"]
    assert p.shelf_db < -10. and p.self_noise_db > -2.


def test_pi5_inmp441_profile_uses_the_datasheet_response_and_absolute_levels():
    p = iot_mic.PROFILES["pi5_inmp441/1"]
    assert p.profile_id == "pi5_inmp441/1"
    assert p.sensitivity_dbfs_at_94db_spl == -26.0
    assert p.self_noise_dbfs_a == -87.0 and p.snr_db_a == 61.0
    assert p.acoustic_corner_hz == 60.0 and p.dc_block_hz_at_48k == 3.7
    assert p.passband_fraction_fs == .423 and p.max_spl_peak_db == 120.0
    assert p.adc_bits == 24 and p.capture_bits == 16
    assert p.dc_offset == 0 and not p.resonances
    assert p.codec is None and p.agc is None and p.noise_suppression is None and p.packet_loss == 0
    assert iot_mic.PROFILES["pi5_mems_unfiltered"].profile_id == p.profile_id  # warns when applied


def test_pi_mems_profiles_are_anchored_to_their_documented_specifications():
    ics = iot_mic.PROFILES["pi5_ics43434/1"]
    sph = iot_mic.PROFILES["pi5_sph0645/1"]
    assert ics.sensitivity_dbfs_at_94db_spl == sph.sensitivity_dbfs_at_94db_spl == -26.0
    assert ics.self_noise_dbfs_a == sph.self_noise_dbfs_a == -91.0
    assert ics.snr_db_a == sph.snr_db_a == 65.0
    assert sph.acoustic_corner_hz == 45.0
    assert all(p.max_spl_peak_db == 120.0 and p.adc_bits == 24 and p.capture_bits == 16
               and p.codec is None and p.agc is None and p.noise_suppression is None and p.packet_loss == 0
               for p in (ics, sph))


def test_pi_profile_variation_stays_small_and_keeps_sensitivity_fixed():
    base = iot_mic.PROFILES["pi5_inmp441/1"]
    draws = [base.with_variation(np.random.default_rng(k)) for k in range(300)]
    assert all(d.sensitivity_dbfs_at_94db_spl == -26.0 for d in draws)
    assert all(-89.0 <= d.self_noise_dbfs_a <= -85.0 for d in draws)
    assert all(50.0 <= d.acoustic_corner_hz <= 80.0 and not d.resonances for d in draws)
