"""Exact cache controls on generated development fixtures (synthetic proxy)."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import signal

from sensorsim import iot_mic


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def uncached_capsule(chain, fs):
    """Pre-optimization calculation; does not call the cache under test."""
    sections = []
    corner = chain.acoustic_corner_hz if chain.acoustic_corner_hz is not None else chain.hp_hz
    if corner > 0:
        sections.append(signal.butter(2, min(corner, .45*fs)/(fs/2), 'highpass', output='sos'))
    if chain.dc_block_hz_at_48k:
        dc = chain.dc_block_hz_at_48k*fs/48000.
        sections.append(signal.butter(2, min(dc, .45*fs)/(fs/2), 'highpass', output='sos'))
    if chain.shelf_db:
        sections.append(iot_mic.high_shelf(chain.shelf_hz, chain.shelf_db, fs))
    for f, gain, q in chain.resonances:
        if f < .45*fs:
            sections.append(iot_mic.peaking(f, gain, q, fs))
    return np.vstack(sections) if sections else np.array([[1., 0., 0., 1., 0., 0.]])


@pytest.mark.parametrize('fs', [8000, 16000, 44100, 48000])
@pytest.mark.parametrize('name', sorted(iot_mic.PROFILES))
def test_capsule_matches_uncached_calculation(name, fs):
    chain = iot_mic.PROFILES[name].with_variation(np.random.default_rng(431))
    expected = uncached_capsule(chain, fs)
    assert_bytes(iot_mic._capsule_sos(chain, fs), expected)
    assert_bytes(iot_mic._capsule_sos(chain, fs), expected)


def test_design_reused_across_response_audio_noise_and_profile_variation(monkeypatch):
    original = signal.butter
    calls = []
    def observe(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(signal, 'butter', observe)
    p = iot_mic.PROFILES['pi5_inmp441/1']
    x = np.random.default_rng(432).normal(0, .01, 2048)
    p.apply(x, 16000, np.random.default_rng(433))
    assert len(calls) == 2  # acoustic HP and DC block, reused for audio/noise/calibration
    q = replace(p, acoustic_corner_hz=71.2345, self_noise_dbfs_a=-88.)
    q.apply(x, 16000, np.random.default_rng(433))
    assert len(calls) == 3  # only the changed acoustic corner is redesigned


def test_returned_sos_mutation_does_not_poison_cached_design():
    p = iot_mic.PROFILES['pi5_inmp441/1']
    expected = uncached_capsule(p, 16000)
    first = iot_mic._capsule_sos(p, 16000)
    first[:] = 0
    assert_bytes(iot_mic._capsule_sos(p, 16000), expected)


def test_changed_rate_and_exact_corner_reach_designer(monkeypatch):
    original = signal.butter
    calls = []
    def observe(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(signal, 'butter', observe)
    p = iot_mic.MicChain(hp_hz=63.12345)
    iot_mic._capsule_sos(p, 16000)
    iot_mic._capsule_sos(p, 48000)
    iot_mic._capsule_sos(replace(p, hp_hz=np.nextafter(p.hp_hz, np.inf)), 16000)
    assert len(calls) == 3  # never round cache parameters


def test_replaced_and_missing_design_hook_are_not_hidden(monkeypatch):
    p = iot_mic.MicChain(hp_hz=62.5)
    expected = iot_mic._capsule_sos(p, 16000)
    original = signal.butter
    reached = []
    def changed(*args, **kwargs):
        reached.append(True)
        sos = original(*args, **kwargs)
        sos[0, 0] = np.nextafter(sos[0, 0], np.inf)
        return sos
    monkeypatch.setattr(signal, 'butter', changed)
    different = iot_mic._capsule_sos(p, 16000)
    assert reached and different.tobytes() != expected.tobytes()
    monkeypatch.setattr(signal, 'butter', None)
    with pytest.raises(TypeError):
        iot_mic._capsule_sos(p, 16000)


def test_one_ulp_and_constant_corner_mutants_are_rejected(monkeypatch):
    p = iot_mic.MicChain(hp_hz=63.12345)
    expected = uncached_capsule(p, 16000)
    assert_bytes(iot_mic._capsule_sos(p, 16000), expected)  # known correct
    actual = iot_mic._capsule_sos(p, 16000).copy()
    actual[0, 0] = np.nextafter(actual[0, 0], np.inf)
    with pytest.raises(AssertionError):
        assert_bytes(actual, expected)
    original = iot_mic._capsule_sos
    reached = []
    def wrong(chain, fs):
        reached.append(chain.hp_hz)
        return original(replace(chain, hp_hz=60.), fs)
    monkeypatch.setattr(iot_mic, '_capsule_sos', wrong)
    x = np.random.default_rng(431).normal(size=1003)
    bad = iot_mic._filter_profile(p, x, 16000)
    correct = signal.sosfilt(expected, x)
    assert reached == [p.hp_hz]
    assert bad.tobytes() != correct.tobytes()
    with pytest.raises(AssertionError):
        assert_bytes(bad, correct)


def test_wideband_codec_design_is_reused(monkeypatch):
    original = signal.butter
    calls = []
    def observe(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    x = np.random.default_rng(434).normal(size=1003)
    expected = signal.sosfilt(original(6, [50/8000, 7000/8000], 'bandpass', output='sos'), x)
    monkeypatch.setattr(signal, 'butter', observe)
    assert_bytes(iot_mic.codec(x, 16000, 'bandlimit7k'), expected)
    assert_bytes(iot_mic.codec(x.copy(), 16000, 'bandlimit7k'), expected)
    assert len(calls) == 1


@pytest.mark.parametrize('fs', [8000, 16000, 48000])
def test_profile_response_matches_uncached_fir_calculation(fs):
    p = iot_mic.PROFILES['pi5_inmp441/1']
    for frequencies in (np.array([1000.]), np.array([0., 1000., fs/2]), np.array([])):
        _, expected = signal.sosfreqz(uncached_capsule(p, fs), worN=frequencies, fs=fs)
        _, fir = signal.freqz(iot_mic._decimation_fir(p.passband_fraction_fs), worN=frequencies, fs=fs)
        expected *= fir
        assert_bytes(iot_mic.profile_response(p, fs, frequencies), expected)


@pytest.mark.parametrize('kind', ['silence', 'constant', 'impulse', 'duplicate', 'empty'])
def test_filter_array_edge_cases_match_uncached(kind):
    p = iot_mic.PROFILES['pi5_inmp441/1']
    x = {'silence': np.zeros(257), 'constant': np.ones(257), 'impulse': np.eye(1, 257)[0],
         'duplicate': np.tile([.1, -.1], 128), 'empty': np.array([])}[kind]
    if not x.size:
        rng = np.random.default_rng(435); state = rng.bit_generator.state
        assert_bytes(p.apply(x, 16000, rng), x)
        assert rng.bit_generator.state == state
        return
    expected = signal.sosfilt(uncached_capsule(p, 16000), x)
    expected = signal.lfilter(iot_mic._decimation_fir(p.passband_fraction_fs), [1.], expected)
    assert_bytes(iot_mic._filter_profile(p, x, 16000), expected)


def test_identity_filtered_resonance_and_invalid_rate_branches():
    p = iot_mic.MicChain(hp_hz=0, resonances=((10000., 5., 2.),))
    assert_bytes(iot_mic._capsule_sos(p, 16000), uncached_capsule(p, 16000))
    with pytest.raises(ZeroDivisionError):
        iot_mic._capsule_sos(iot_mic.MicChain(), 0)
    # Preserve the existing negative-rate behavior; validation is not this change.
    assert_bytes(iot_mic._capsule_sos(iot_mic.MicChain(), -16000),
                 uncached_capsule(iot_mic.MicChain(), -16000))


def test_calibration_fir_reused_but_rate_passband_and_hook_changes_recompute(monkeypatch):
    original = signal.freqz
    fir_calls = []
    def observe(b, *args, **kwargs):
        if len(b) == 129:
            fir_calls.append((tuple(b), kwargs['fs']))
        return original(b, *args, **kwargs)
    monkeypatch.setattr(signal, 'freqz', observe)
    p = iot_mic.PROFILES['pi5_inmp441/1']
    frequencies = np.array([1000.])
    first = iot_mic.profile_response(p, 16000, frequencies)
    assert_bytes(iot_mic.profile_response(p, 16000, frequencies), first)
    iot_mic.profile_response(replace(p, acoustic_corner_hz=73.), 16000, frequencies)
    assert len(fir_calls) == 1
    iot_mic.profile_response(p, 48000, frequencies)
    iot_mic.profile_response(replace(p, passband_fraction_fs=.4), 16000, frequencies)
    assert len(fir_calls) == 3
    reached = []
    def changed(b, *args, **kwargs):
        w, response = original(b, *args, **kwargs)
        if len(b) == 129:
            reached.append(True)
            response *= 1.01
        return w, response
    monkeypatch.setattr(signal, 'freqz', changed)
    different = iot_mic.profile_response(p, 16000, frequencies)
    assert reached and different.tobytes() != first.tobytes()
    with pytest.raises(AssertionError):
        assert_bytes(different, first)


def test_calibration_changed_decimator_is_observed(monkeypatch):
    p = iot_mic.PROFILES['pi5_inmp441/1']
    frequencies = np.array([1000.])
    expected = iot_mic.profile_response(p, 16000, frequencies)
    original = iot_mic._decimation_fir
    reached = []
    def wrong(passband):
        reached.append(passband)
        return original(passband)*.99
    monkeypatch.setattr(iot_mic, '_decimation_fir', wrong)
    actual = iot_mic.profile_response(p, 16000, frequencies)
    assert reached and actual.tobytes() != expected.tobytes()


@pytest.mark.parametrize('frequencies', [np.array([1000.], np.float32), np.array([1000]),
                                        np.array([1000., 1000.]), np.array([np.nan]), 16])
def test_general_frequency_query_fallback_is_unchanged(frequencies):
    p = iot_mic.PROFILES['pi5_inmp441/1']
    _, expected = signal.sosfreqz(uncached_capsule(p, 16000), worN=frequencies, fs=16000)
    _, fir = signal.freqz(iot_mic._decimation_fir(p.passband_fraction_fs), worN=frequencies, fs=16000)
    expected *= fir
    assert_bytes(iot_mic.profile_response(p, 16000, frequencies), expected)


def test_cache_storage_is_bounded_and_read_only():
    iot_mic._butter_sos.cache_clear()
    for k in range(300):
        iot_mic._capsule_sos(iot_mic.MicChain(hp_hz=50.+k/10), 16000)
    assert iot_mic._butter_sos.cache_info().currsize == 256
    sos = iot_mic._butter_sos(2, .01, 'highpass', signal.butter)
    assert not sos.flags.writeable
    for fs in range(16000, 16040):
        iot_mic.profile_response(iot_mic.PROFILES['pi5_inmp441/1'], fs, np.array([1000.]))
    assert iot_mic._calibration_fir_response.cache_info().currsize == 32


def test_dependency_error_is_propagated_without_fallback(monkeypatch):
    p = iot_mic.MicChain(hp_hz=62.5)
    iot_mic._capsule_sos(p, 16000)
    def unavailable(*args, **kwargs):
        raise ImportError('development missing dependency control')
    monkeypatch.setattr(signal, 'butter', unavailable)
    with pytest.raises(ImportError, match='missing dependency control'):
        iot_mic._capsule_sos(p, 16000)


def test_mutable_fir_taps_cannot_leave_a_stale_calibration(monkeypatch):
    p = iot_mic.PROFILES['pi5_inmp441/1']
    taps = iot_mic._decimation_fir(p.passband_fraction_fs).copy()
    monkeypatch.setattr(iot_mic, '_decimation_fir', lambda _: taps)
    frequencies = np.array([1000.])
    first = iot_mic.profile_response(p, 16000, frequencies)
    taps[64] *= .99
    _, expected = signal.sosfreqz(uncached_capsule(p, 16000), worN=frequencies, fs=16000)
    _, fir = signal.freqz(taps, worN=frequencies, fs=16000)
    expected *= fir
    actual = iot_mic.profile_response(p, 16000, frequencies)
    assert actual.tobytes() != first.tobytes()
    assert_bytes(actual, expected)
