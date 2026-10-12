"""Adversarial room measurement contracts; see parent-project note room-audit (not distributed)."""
import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import fftconvolve

from sensorsim.acoustics.room.ingest import RecordingManifest
from sensorsim.acoustics.room.rir import (RIR, deconvolve_repetitions,
    estimate_reference_eq, generate_kit_playback, trim_rir)
from sensorsim.acoustics.room.metrics import room_metrics
from sensorsim.acoustics.room import load_manifest, read_take
from tests.unit.acoustics_room.test_room import kit, generic_manifest


def test_replaced_mono_take_cannot_broadcast_into_two_calibrated_mics(tmp_path):
    path, _ = generic_manifest(tmp_path)
    manifest = load_manifest(path)
    wavfile.write(tmp_path / 'take.wav', 8000, np.ones(400, dtype=np.float32))
    with pytest.raises(ValueError, match='channel'):
        read_take(manifest, 0)


def test_kit_cannot_silently_override_take_excitation(tmp_path):
    path, raw = generic_manifest(tmp_path)
    k = kit(duration=.025).model_copy(update={
        'lead_silence_s': 0., 'first_sweep_start_s': 0., 'repeats': 1})
    raw['sweep_kit'] = k.model_dump()
    raw['takes'][0]['excitation'] = {**k.excitation.model_dump(), 'f_start_hz': 100.}
    with pytest.raises(ValueError, match='kit.*excitation|excitation.*kit'):
        RecordingManifest.model_validate(raw, context={'base_dir': path.parent})


def test_repetition_alignment_preserves_interchannel_delay_without_geometry():
    k = kit()
    h = np.zeros((480, 2))
    h[8, 0], h[40, 1] = 1., .7
    playback = generate_kit_playback(k)
    recording = np.stack([fftconvolve(playback, h[:, i]) for i in range(2)], axis=1)
    rir = deconvolve_repetitions(recording, k, rir_duration_s=.06)
    arrivals = np.argmax(np.abs(rir.samples), axis=0)
    assert arrivals[1] - arrivals[0] == pytest.approx(32, abs=1)
    assert rir.metadata['direct_samples'] == arrivals.tolist()


def test_clock_correction_option_has_applied_and_disabled_witnesses():
    from scipy.signal import resample
    k = kit()
    playback = generate_kit_playback(k)
    captured = resample(playback, round(len(playback)*1.0002))
    for enabled in (True, False):
        with pytest.warns(UserWarning):
            result = deconvolve_repetitions(captured, k, rir_duration_s=.06,
                                            correct_clock_drift=enabled)
        options = result.metadata['processing_options']
        assert options['correct_clock_drift'] is enabled
        assert options['clock_correction_applied'] is enabled
        assert result.metadata['clock_resampled'] is enabled
        assert .5 < abs(result.metadata['clock_drift_ppm']) < 300


def test_trim_updates_channel_arrivals_in_the_returned_sample_frame():
    h = np.zeros((300, 2))
    h[100, 0], h[120, 1] = 1., .7
    rir = RIR(10000, h, metadata={'direct_samples': [100, 120]})
    cropped = trim_rir(rir, onset_sample=100, pre_s=.0025)
    assert cropped.metadata['direct_samples'] == [25, 45]
    assert cropped.metadata['time_origin_s'] == .0075
    assert rir.metadata['direct_samples'] == [100, 120]


def test_ref_rejects_truncated_post_arrival_window():
    h = np.zeros(510)
    h[480] = 1 / .3
    with pytest.raises(ValueError, match='REF.*window|post.*window'):
        estimate_reference_eq(RIR(48000, h), direct_sample=480)


def test_c50_does_not_require_c80_integration_support():
    metrics = room_metrics(np.ones(70), 1000, onset_sample=0, noise_power=0)
    assert metrics.c50_db == pytest.approx(10 * np.log10(50 / 20), abs=1e-12)
    assert metrics.d50 == pytest.approx(50 / 70, abs=1e-12)
    assert metrics.c80_db is None


def test_regression_ranges_on_piecewise_analytic_energy_decay():
    fs = 20000
    t = np.arange(6001) / fs
    db = -100*t - 100*np.maximum(t-.1, 0) - 200*np.maximum(t-.175, 0)
    energy = 10 ** (db/10)
    # Telescope to the prescribed EDC exactly, including finite-record tail energy.
    h = np.sqrt(np.r_[energy[:-1] - energy[1:], energy[-1]])
    m = room_metrics(h, fs, onset_sample=0, noise_power=0)
    # Continuous least squares slopes are -100, -164.8 and -1700/9 dB/s.
    for fit, expected in zip((m.edt, m.t20, m.t30), (.6, 60/164.8, 540/1700)):
        assert fit.valid and fit.rt60_s == pytest.approx(expected, rel=.002)


def test_ref_inverse_gain_cap_is_engaged_for_weak_transfer():
    h = np.zeros(1024)
    h[200] = .001/.3
    eq = estimate_reference_eq(RIR(48000, h), direct_sample=200)
    assert eq.supported.any()
    np.testing.assert_allclose(abs(eq.correction[eq.supported]), 10**(12/20), rtol=1e-12)
