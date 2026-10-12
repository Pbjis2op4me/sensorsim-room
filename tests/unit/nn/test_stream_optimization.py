"""Exact route controls on generated development audio, never realism evidence."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import iot_mic, nn, training_device as device, event_level


def _scalar_recording_check(x, fs, profile, native):
    """Frozen pre-optimization scalar QA; independent of the candidate helper."""
    messages = []
    if not native or np.max(x) > 32767/32768 or not np.array_equal(x*32768, np.rint(x*32768)):
        messages.append('device-recorded background is not native S16; review capture provenance')
    if not np.any(x):
        messages.append('device-recorded background is silent')
    else:
        nominal = iot_mic.PROFILES.get(profile.profile_id, profile)
        frame = max(1, round(.25*fs))
        run = 0
        for start in range(0, len(x)-frame+1, frame):
            rms = iot_mic.a_weighted_rms(x[start:start+frame], fs)*np.sqrt(2)
            below = 20*np.log10(max(rms, 1e-30)) < nominal.self_noise_dbfs_a-6.
            run = run+frame if below else 0
            if run >= fs:
                messages.append('device-recorded background is below the profile noise floor '
                                'by more than 6 dB for at least 1 s')
                break
    return messages


def _messages(x, fs, profile, native=True):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        device._check_recording(x, fs, profile, native)
    return [str(w.message) for w in seen]


@pytest.mark.parametrize('fs', [8000, 16000, 44100, 48000])
@pytest.mark.parametrize('kind', ['empty', 'silence', 'constant', 'short', 'native', 'float', 'gap', 'low'])
def test_recording_qa_matches_scalar_oracle(fs, kind):
    rng = np.random.default_rng(491)
    profile = iot_mic.PROFILES['pi5_inmp441/1']
    x = np.rint(rng.normal(0, .002, 2*fs)*32768)/32768
    if kind == 'empty':
        with pytest.raises(ValueError): device._check_recording(x[:0], fs, profile, True)
        return
    if kind == 'silence': x[:] = 0
    if kind == 'constant': x[:] = 1/32768
    if kind == 'short': x = x[:max(1, fs//8)]
    if kind == 'float': x += 1e-9
    if kind == 'gap': x[fs//4:fs] = 0
    if kind == 'low': x[fs//4:] = 0
    assert _messages(x, fs, profile) == _scalar_recording_check(x, fs, profile, True)


def test_recording_qa_batches_the_meter(monkeypatch):
    from sensorsim.acoustics import weighting
    old = weighting.a_weighted_mean_square
    calls = []
    def meter(x, fs):
        calls.append(x.shape)
        return old(x, fs)
    monkeypatch.setattr(weighting, 'a_weighted_mean_square', meter)
    x = np.random.default_rng(62).normal(0, .002, 32000)
    _messages(x, 16000, iot_mic.PROFILES['pi5_inmp441/1'], False)
    assert calls == [(8, 4000)]


def test_recording_qa_known_bad_single_frame_alarm():
    fs = 16000
    profile = iot_mic.PROFILES['pi5_inmp441/1']
    x = np.rint(np.random.default_rng(21).normal(0, .002, 2*fs)*32768)/32768
    x[fs//4:fs//2] = 0
    assert _messages(x, fs, profile) == []  # correct: only one quiet quarter-second
    bad_alarm = any(iot_mic.a_weighted_rms(x[k:k+fs//4], fs) == 0
                    for k in range(0, len(x), fs//4))
    assert bad_alarm  # designated wrong optimization would issue an alarm


def test_zero_event_skips_convolution_and_keeps_nonzero_control(monkeypatch):
    real_convolve = np.convolve
    calls = []
    def spy(*args, **kwargs):
        calls.append(True)
        return real_convolve(*args, **kwargs)
    monkeypatch.setattr(np, 'convolve', spy)
    assert device.active_power(np.zeros(32000), 16000, (16000, 32000)) == (0., 0)
    assert not calls
    x = np.tile([-1., 1.], 16000)
    assert device.active_power(x, 16000) == (1., 32000)
    assert len(calls) == 1
    assert (0., 0) != device.active_power(x, 16000)  # known-bad silence shortcut


def test_unused_full_take_power_is_not_recomputed(monkeypatch):
    c = SimpleNamespace(classes=['knock'], background=[np.ones(32000)/32768],
        targets={'knock':[np.sin(np.arange(8000)*.2)]}, spans={'knock':[(0,8000)]},
        negative_pool=None)
    calls = []
    real_power = event_level.level_power
    def spy(*args, **kwargs):
        calls.append(True)
        return real_power(*args, **kwargs)
    monkeypatch.setattr(event_level, 'level_power', spy)
    cfg = dict(nn.DEFAULTS, p_background_only=0., p_confuser_given_background=0.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nn.make_example(c, np.random.default_rng(71), cfg, return_context=True)
    assert not calls
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nn.make_example(c, np.random.default_rng(71), dict(cfg, partial_windows=True))
    assert calls  # required gain compensation still reaches full-take power


def test_recording_qa_rejects_reusing_first_frame_power(monkeypatch):
    from sensorsim.acoustics import weighting
    real_meter = weighting.a_weighted_mean_square
    fs = 16000
    profile = iot_mic.PROFILES['pi5_inmp441/1']
    x = np.rint(np.random.default_rng(81).normal(0, .002, 2*fs)*32768)/32768
    x[:fs//4] = 0
    expected = _scalar_recording_check(x, fs, profile, True)
    assert _messages(x, fs, profile) == expected == []
    reached = []
    def wrong_cached_frame(frames, rate):
        reached.append(frames.shape)
        powers = real_meter(frames, rate)
        return np.full_like(powers, powers[0])
    monkeypatch.setattr(weighting, 'a_weighted_mean_square', wrong_cached_frame)
    assert _messages(x, fs, profile) != expected
    assert reached == [(8, 4000)]


def test_example_oracle_rejects_treating_every_event_as_silent(monkeypatch):
    c = SimpleNamespace(classes=['knock'], background=[np.full(32000, .001)],
        targets={'knock':[np.sin(np.arange(8000)*.2)]}, spans={'knock':[(0, 8000)]},
        negative_pool=None)
    cfg = dict(nn.DEFAULTS, p_background_only=0., p_confuser_given_background=0.)
    good_rng = np.random.default_rng(75)
    bad_rng = np.random.default_rng(75)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        good, label = nn.make_example(c, good_rng, cfg)
        reached = []
        def wrong_power(event, fs, span):
            reached.append(bool(np.any(event)))
            return 0., 0
        monkeypatch.setattr(device, 'active_power', wrong_power)
        bad, bad_label = nn.make_example(c, bad_rng, cfg)
    assert reached == [True]
    assert label.tobytes() == bad_label.tobytes()
    assert good_rng.bit_generator.state == bad_rng.bit_generator.state
    assert good.tobytes() != bad.tobytes()
