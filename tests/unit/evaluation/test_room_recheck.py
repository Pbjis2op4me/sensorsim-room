import numpy as np
import pytest

from sensorsim.evaluation.room_recheck import (
    convolve_aligned_window,
    edge_band_statistics,
    make_dry_sources,
    rectangular_room_modes,
    standardized_mean_difference,
)


def test_rectangular_room_modes_are_sorted_and_exclude_zero_mode():
    modes = rectangular_room_modes((5.0, 4.0, 2.5), sound_speed_m_s=340.0,
                                   maximum_frequency_hz=100.0)
    assert modes
    assert modes[0].indices == (1, 0, 0)
    assert modes[0].frequency_hz == pytest.approx(34.0)
    assert all(mode.indices != (0, 0, 0) for mode in modes)
    assert [mode.frequency_hz for mode in modes] == sorted(mode.frequency_hz for mode in modes)


@pytest.mark.parametrize("dimensions", [(0, 1, 1), (1, -1, 1), (1, 2), (np.nan, 1, 1)])
def test_rectangular_room_modes_reject_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        rectangular_room_modes(dimensions)


def test_smd_reports_zero_for_equal_constant_features_and_signed_infinity_otherwise():
    same = np.ones((4, 2))
    assert np.array_equal(standardized_mean_difference(same, same), np.zeros(2))
    shifted = np.full((4, 2), 3.0)
    result = standardized_mean_difference(shifted, same)
    assert np.all(np.isposinf(result))


def test_smd_uses_pooled_sample_variance():
    left = np.array([[0.0], [2.0], [4.0]])
    right = np.array([[1.0], [3.0], [5.0]])
    assert standardized_mean_difference(left, right)[0] == pytest.approx(-0.5)


def test_edge_statistics_match_registered_120_value_layout():
    fs = 16000
    config = {"fs": fs, "window_s": 1.0, "n_fft": 1024, "hop": 256,
              "n_mels": 40, "fmin_hz": 50.0, "fmax_hz": 8000.0}
    x = np.zeros(fs, dtype=np.float32)
    x[fs // 10] = 1.0
    stats = edge_band_statistics(np.stack((x, x)), config)
    assert stats.shape == (2, 120)
    np.testing.assert_allclose(stats[0], stats[1], atol=1e-5, rtol=0)


def test_convolution_places_the_declared_direct_arrival_in_a_one_second_window():
    rir = np.zeros(64)
    rir[20] = 1.0
    dry = np.array([1.0])
    result = convolve_aligned_window(rir, dry, direct_sample=20, fs=1000,
                                     window_s=1.0, direct_at_s=0.1)
    assert result.shape == (1000,)
    assert result[100] == pytest.approx(1.0)
    assert np.count_nonzero(result) == 1


def test_convolution_keeps_prering_when_alignment_shifts_left():
    rir = np.zeros(32)
    rir[10:13] = [0.25, 1.0, 0.25]
    result = convolve_aligned_window(rir, [1.0], direct_sample=10, fs=1000,
                                     window_s=0.02, direct_at_s=0.002)
    np.testing.assert_allclose(result[2:5], [0.25, 1.0, 0.25])


def test_registered_dry_sources_have_leading_silence_and_declared_durations():
    fs = 16000
    sources = make_dry_sources(fs, broadband_seed=4)
    onset = round(0.1 * fs)
    assert np.flatnonzero(sources["impulse"]).tolist() == [onset]
    assert len(sources["chime"]) == round(0.5 * fs)
    assert len(sources["broadband"]) == round(0.5 * fs)
    assert not np.any(sources["chime"][:onset])
    assert not np.any(sources["broadband"][:onset])
    assert np.any(sources["chime"][onset:])
    assert np.any(sources["broadband"][onset:])


def test_source_onset_translation_keeps_aligned_rir_window_identical():
    rir = np.zeros(64)
    rir[20:23] = [0.25, 1.0, 0.25]
    zero_onset = np.array([1.0])
    delayed_onset = np.zeros(11)
    delayed_onset[10] = 1.0
    direct_zero = convolve_aligned_window(rir, zero_onset, direct_sample=20, fs=1000,
                                          direct_at_s=0.05)
    direct_delayed = convolve_aligned_window(rir, delayed_onset, direct_sample=20, fs=1000,
                                             direct_at_s=0.05, source_onset_sample=10)
    np.testing.assert_allclose(direct_zero, direct_delayed, atol=1e-6, rtol=0)
