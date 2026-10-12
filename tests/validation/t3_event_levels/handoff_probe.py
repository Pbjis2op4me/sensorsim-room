# SPDX-License-Identifier: Apache-2.0
"""Explicit shared-handoff development checks; requires proposed mixer hunks.

Invoke explicitly on a scratch candidate; this is not in default collection.
Missing hooks fail, never skip or fall back. No acceptance receipt is emitted.
"""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from sensorsim import event_levels as el, iot_mic, nn, training_device as td
from sensorsim.acoustics.room.source_reference import SourceReference


def fixture(mode, source=None):
    fs = 16000
    tone = .01*np.sin(2*np.pi*1000*np.arange(fs)/fs)
    x = np.r_[tone, .1*tone] if source is None else source
    units = "Pa" if mode == el.CALIBRATED else "relative"
    ref = el.LevelReference("rms", units, 20e-6 if units == "Pa" else 1., "Z", (0, len(x)), fs, "source-1m")
    p = el.ParentEvent("p", "g", "s", "kettle", "k", "cycle", "train", "synthetic_proxy", "source_only",
                       x, SourceReference(units=units), ref)
    roles = {"parent:p": "train", "group:g": "train", "session:s": "train", "content:"+p.content_id: "train"}
    prior = el.fit_level_prior([p], mode=mode, role_bindings=roles,
        relative_anchors={"s": el.RelativeAnchor(.2, "assumed")} if mode == el.RELATIVE else {})
    draw = el.sample_parent_gain(prior, np.random.default_rng(0), condition=p.condition)
    cfg = dict(fs=fs, event_level_mode=mode, event_spl_db=[60, 80], device_profile="pi5_inmp441/1",
               device_split="in_distribution", background_domain="device_recorded", render_reference_spl_db=94.)
    return x, draw, cfg


def mix(x, draw, cfg):
    assert "parent_gain" in td.ReceiverEvent.__dataclass_fields__, "missing shared mixer hook"
    with warnings.catch_warnings(record=True) as caught:
        result = td.mix_events(np.zeros(len(x)), [td.ReceiverEvent(x, (0, len(x)), parent_gain=draw)],
            np.random.default_rng(123), cfg, device_draw=iot_mic.PROFILES[cfg["device_profile"]])
    assert any("silent" in str(w.message) for w in caught)
    assert result.events[0]["parent_level"]["mode"] == draw.mode
    assert result.events[0]["receiver_gain"] == draw.gain
    assert result.events[0]["event_spl_db"] is None
    return result.audio


@pytest.mark.parametrize("mode", [el.RELATIVE, el.CALIBRATED])
def test_handoff_actual_device_path(mode, record_property):
    x, d, cfg = fixture(mode)
    y, louder = mix(x, d, cfg), mix(x*10, d, cfg)
    scalar = 10*np.log10(np.dot(louder, louder)/np.dot(y, y))
    assert abs(scalar-20.) < .2
    # Inspect steady portions of full-history capture; no crop is normalized.
    loud, quiet = y[2000:14000], y[18000:30000]
    relation = 10*np.log10(np.dot(loud, loud)/np.dot(quiet, quiet))
    assert abs(relation-20.) < .2
    bad_x = np.r_[x[:16000], x[16000:]*10]
    bad = mix(bad_x, d, cfg)
    mutant = 10*np.log10(np.dot(bad[2000:14000], bad[2000:14000])/np.dot(bad[18000:30000], bad[18000:30000]))
    assert abs(mutant-20.) > .2
    record_property("actual_scalar_db", scalar)
    record_property("actual_cycle_relation_db", relation)
    record_property("crop_normalization_mutant_error_db", abs(mutant-20.))


def test_handoff_missing_draw_cannot_fall_back():
    x, d, cfg = fixture(el.RELATIVE)
    with pytest.raises(ValueError, match="matching typed"):
        td.mix_events(np.zeros(len(x)), [td.ReceiverEvent(x)], np.random.default_rng(0), cfg)
    with pytest.raises(ValueError, match="not integrated"):
        nn.make_example(None, np.random.default_rng(0), cfg)
    with pytest.raises(ValueError, match="not integrated"):
        nn._persistence(None, None, cfg, 0)


def test_handoff_legacy_profile_and_draw_identity():
    x, _, cfg = fixture(el.RELATIVE)
    cfg["event_level_mode"] = el.LEGACY
    profile = iot_mic.PROFILES[cfg["device_profile"]]
    cfg["background_domain"] = "spl_referenced"
    one = td.mix_device_audio(np.zeros(len(x)), x, (0, len(x)), np.random.default_rng(123), cfg, device_draw=profile)
    cfg.pop("event_level_mode")
    two = td.mix_device_audio(np.zeros(len(x)), x, (0, len(x)), np.random.default_rng(123), cfg, device_draw=profile)
    np.testing.assert_array_equal(one, two)


def independent_device_history(samples, fs):
    """Direct recurrences for the declared nominal INMP441 numerical model.

    No production filter, MicChain.apply or captured output supplies the oracle.
    Bilinear second-order Butterworth HP coefficients are calculated explicitly;
    the model's FIR specification is built with SciPy and applied by direct
    convolution. This checks numerical history, not measured microphone physics.
    """
    from scipy.signal import firwin2
    y = np.array(samples, dtype=float, copy=True)
    z = np.exp(-2j*np.pi*1000/fs)
    response = 1.+0j
    for corner in (60., 3.7*fs/48000):
        k = np.tan(np.pi*corner/fs)
        scale = 1/(1+np.sqrt(2)*k+k*k)
        b = np.array([1., -2., 1.])*scale
        a = np.array([1., 2*(k*k-1)*scale, (1-np.sqrt(2)*k+k*k)*scale])
        response *= sum(b[i]*z**i for i in range(3))/sum(a[i]*z**i for i in range(3))
        out = np.empty_like(y)
        x1 = x2 = y1 = y2 = 0.
        for i, x0 in enumerate(y):
            y0 = b[0]*x0+b[1]*x1+b[2]*x2-a[1]*y1-a[2]*y2
            out[i] = y0
            x2, x1, y2, y1 = x1, x0, y1, y0
        y = out
    taps = firwin2(129, [0., .846, 1.], [1., 1., 10**(-6/20)], window="hamming")
    response *= sum(v*z**i for i, v in enumerate(taps))
    return np.convolve(y, taps)[:len(y)]*10**(-26/20)/abs(response)


def history_boundary_control(mode, monkeypatch, record_property):
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import SynthConfig
    from sensorsim.acoustics.room import synth

    assert "parent_gain" in td.ReceiverEvent.__dataclass_fields__, "missing shared mixer hook"
    fs = 16000
    source = .02*np.sin(2*np.pi*137*np.arange(2*fs)/fs+.7)
    _, d, cfg = fixture(mode, source)
    rir_calls, device_calls = [], []
    synthesize, apply = synth.synthesize, iot_mic.MicChain.apply

    def room_spy(*args, **kwargs):
        rir = synthesize(*args, **kwargs)
        rir_calls.append(rir.samples.copy())
        return rir

    def device_spy(self, x, rate, rng, **kwargs):
        output = apply(self, x, rate, rng, **kwargs)
        device_calls.append((np.array(x), output.copy(), rate, kwargs))
        return output

    monkeypatch.setattr(synth, "synthesize", room_spy)
    monkeypatch.setattr(iot_mic.MicChain, "apply", device_spy)
    room = RoomModel(shoebox_dims_m=(6, 5, 3))
    event = td.render_room_event(source, fs=fs,
        draw=td.RoomDraw(room, (3., 2., 1.5), SynthConfig(late=False, max_order=0), 21),
        position=(2., 2., 1.5), onset_s=.05, n=2*fs+1600)
    assert len(rir_calls) == 1, "room hook was not reached exactly once"
    wet = np.zeros_like(event.audio)
    conv = np.convolve(source, rir_calls[0])
    take = min(len(conv), len(wet)-800)
    wet[800:800+take] = conv[:take]
    # The room path convolves a float32 RIR; its FFT roundoff is included in
    # the registered one-step output bound, not mistaken for float64 identity.
    room_error = float(np.max(np.abs(event.audio-wet))*32768)
    record_property("room_convolution_error_pcm16_steps", room_error)
    assert room_error < 1.
    event = replace(event, parent_gain=d)
    profile = iot_mic.PROFILES["pi5_inmp441/1"]
    assert (profile.acoustic_corner_hz, profile.dc_block_hz_at_48k, profile.passband_fraction_fs,
            profile.sensitivity_dbfs_at_94db_spl) == (60., 3.7, .423, -26.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = td.mix_events(np.zeros(len(wet)), [event], np.random.default_rng(123), cfg, device_draw=profile)
    assert len(device_calls) == 1, "device hook was not reached exactly once"
    incoming, captured, rate, kwargs = device_calls[0]
    gain = .2 if mode == el.RELATIVE else 1/(np.sqrt(2)*20e-6*10**4.7)
    np.testing.assert_allclose(incoming, event.audio*gain, atol=1e-15)
    assert rate == fs and kwargs == dict(input_reference_spl_db=94., include_self_noise=False, quantize=False)
    oracle = independent_device_history(wet*gain, fs)
    # Startup exclusion is ONLY the start of the entire stream. Include the
    # interior one-second boundary and its first 256 samples in both crops.
    errors = []
    for start, stop in ((fs-16, fs+256), (2*fs-16, 2*fs+256)):
        errors.append(float(np.max(np.abs(output.audio[start:stop]-oracle[start:stop]))*32768))
    record_property("interior_history_errors_pcm16_steps", errors)
    record_property("room_calls", len(rir_calls))
    record_property("device_calls", len(device_calls))
    record_property("captured_to_mixer_error_pcm16_steps", float(np.max(np.abs(output.audio-captured))*32768))
    assert max(errors) < 1., f"interior history error {errors} exceeds one PCM16 step"
    return oracle


@pytest.mark.parametrize("mode", [el.RELATIVE, el.CALIBRATED])
def test_handoff_full_history_oracle(mode, monkeypatch, record_property):
    history_boundary_control(mode, monkeypatch, record_property)


@pytest.mark.parametrize("mode", [el.RELATIVE, el.CALIBRATED])
def test_handoff_full_history_rejects_reset_mutant(mode, monkeypatch, record_property):
    apply = iot_mic.MicChain.apply
    reached = []

    def reset_history(self, x, fs, rng, **kwargs):
        reached.append((len(x), fs))
        return np.concatenate([apply(self, x[a:a+fs], fs, rng, **kwargs) for a in range(0, len(x), fs)])

    monkeypatch.setattr(iot_mic.MicChain, "apply", reset_history)
    with pytest.raises(AssertionError, match="interior history error"):
        history_boundary_control(mode, monkeypatch, record_property)
    assert reached == [(33600, 16000)]


@pytest.mark.parametrize("mode", [el.RELATIVE, el.CALIBRATED])
@pytest.mark.parametrize("steps", [.99, 1.01])
def test_handoff_history_boundary_sensitivity(mode, steps, monkeypatch, record_property):
    """Exercise sensitivity at the registered one-step numerical boundary."""
    apply = iot_mic.MicChain.apply
    changes = []

    def perturb_boundary(self, x, fs, rng, **kwargs):
        output = apply(self, x, fs, rng, **kwargs)
        before = output[fs+7]
        output[fs+7] += steps/32768
        changes.append(float((output[fs+7]-before)*32768))
        return output

    monkeypatch.setattr(iot_mic.MicChain, "apply", perturb_boundary)
    if steps < 1:
        history_boundary_control(mode, monkeypatch, record_property)
    else:
        with pytest.raises(AssertionError, match="interior history error"):
            history_boundary_control(mode, monkeypatch, record_property)
    assert changes == pytest.approx([steps], abs=1e-12)
    record_property("injected_boundary_error_pcm16_steps", changes)


@pytest.mark.parametrize("mode", [el.RELATIVE, el.CALIBRATED])
def test_handoff_guards_and_forged_draw(mode, monkeypatch):
    from sensorsim import project
    x, d, cfg = fixture(mode)
    for entry in (lambda: project._recognizer_settings(dict(recognizer={"type": "cnn", "event_level_mode": mode})),
                  lambda: nn.make_example(None, np.random.default_rng(0), cfg),
                  lambda: nn._persistence(None, None, cfg, 0)):
        with pytest.raises(ValueError, match="not integrated"):
            entry()
    # Correct control reaches the actual mixer and device.
    mix(x, d, cfg)
    reached = []
    original = el.receiver_gain

    def spy(*args, **kwargs):
        reached.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(el, "receiver_gain", spy)
    if mode == el.RELATIVE:
        object.__setattr__(d, "mode", el.CALIBRATED)
        cfg["event_level_mode"] = el.CALIBRATED
    else:
        object.__setattr__(d, "gain", d.gain*10**(.21/20))
    with pytest.raises(ValueError):
        mix(x, d, cfg)
    assert reached == [True]
