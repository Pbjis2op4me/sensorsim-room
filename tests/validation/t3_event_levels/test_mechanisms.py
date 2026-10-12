# SPDX-License-Identifier: Apache-2.0
"""P3 engineering development controls; no held-out physical-prior inference.

Oracles use analytic energy ratios and direct FIR sums. The gain sampler is the
system under test, not the oracle. A verifier must review and freeze its own
required collection and run the actual merged training route separately.
"""
from dataclasses import replace

import numpy as np
import pytest
from scipy import signal

from sensorsim import event_levels as el, iot_mic, training_device as td
from sensorsim.acoustics.room.source_reference import SourceReference


def draw(mode, x):
    units = "Pa" if mode == el.CALIBRATED else "relative"
    ref = el.LevelReference("rms", units, 20e-6 if units == "Pa" else 1., "Z", (0, len(x)), 16000, "source-1m")
    p = el.ParentEvent("cycle", "group", "session", "kettle", "kettle-1", "whole-cycle", "train",
                       "synthetic_proxy", "source_only", x, SourceReference(units=units), ref)
    roles = {"parent:cycle": "train", "group:group": "train", "session:session": "train", "content:"+p.content_id: "train"}
    prior = el.fit_level_prior([p], mode=mode, role_bindings=roles,
        relative_anchors={"session": el.RelativeAnchor(.2, "development-assumption")} if mode == el.RELATIVE else {})
    return el.sample_parent_gain(prior, np.random.default_rng(310), condition=p.condition)


def energy_db(x):
    return 10*np.log10(float(np.dot(x, x)))


@pytest.mark.parametrize("mode", [el.RELATIVE, el.CALIBRATED])
def test_scalar_spacing_propagation_and_cycle_crop_controls(mode, record_property):
    tone = np.sin(2*np.pi*1000*np.arange(1600)/16000)
    cycle = np.r_[tone, tone*.1]
    d = draw(mode, cycle)
    full = el.apply_parent_gain(cycle, d)
    louder = el.apply_parent_gain(cycle*10, d)
    scalar_error = energy_db(louder)-energy_db(full)-20.
    near, far = full/2, full/4
    propagation_error = energy_db(near)-energy_db(far)-20*np.log10(2)
    short = np.r_[tone, np.zeros(100), tone*.5]
    spaced = np.r_[tone, np.zeros(1000), tone*.5]
    spacing_error = energy_db(el.apply_parent_gain(short, d))-energy_db(el.apply_parent_gain(spaced, d))
    crop_error = energy_db(full[:1600])-energy_db(full[1600:])-20.
    for error in (scalar_error, propagation_error, spacing_error, crop_error):
        assert abs(error) < .2
    # Mutation is reached and changes the intended output, not an unrelated error.
    bad = [piece/np.sqrt(np.mean(piece**2)) for piece in (full[:1600], full[1600:])]
    mutant_error = abs(energy_db(bad[0])-energy_db(bad[1])-20.)
    assert mutant_error > .2
    assert not np.array_equal(bad[1], full[1600:])
    record_property("errors_db", [scalar_error, propagation_error, spacing_error, crop_error])
    record_property("crop_normalization_mutant_error_db", mutant_error)


@pytest.mark.parametrize("mode", [el.RELATIVE, el.CALIBRATED])
def test_full_filter_history_and_crop_mutant(mode, record_property):
    x = np.sin(2*np.pi*170*np.arange(5000)/16000)
    d = draw(mode, x)
    scaled = el.apply_parent_gain(x, d)
    h = np.array([.5, .25, .125, .0625])
    whole = signal.lfilter(h, [1.], scaled)
    start, stop = 201, 501
    # Direct four-tap summation is independent of scipy's filter implementation.
    oracle = np.array([sum(h[k]*scaled[i-k] for k in range(4)) for i in range(start, stop)])
    error_steps = float(np.max(np.abs(whole[start:stop]-oracle))*32768)
    assert error_steps < 1.
    cropped_first = signal.lfilter(h, [1.], scaled[start:stop])
    mutant_steps = float(np.max(np.abs(cropped_first-oracle))*32768)
    assert mutant_steps > 1.
    record_property("full_history_max_pcm16_steps", error_steps)
    record_property("missing_history_mutant_pcm16_steps", mutant_steps)


def test_reviewed_legacy_mixer_route_and_analytic_gain():
    fs = 16000
    x = .2*np.sin(2*np.pi*1000*np.arange(fs)/fs)
    profile = iot_mic.PROFILES['pi5_inmp441/1']
    cfg = dict(fs=fs, device_profile=profile.profile_id, device_split="in_distribution",
               background_domain="spl_referenced", event_spl_db=[70., 70.], render_reference_spl_db=94.)
    ev = td.ReceiverEvent(x, (0, fs))
    mixed = td.mix_events(np.zeros(fs), [ev], np.random.default_rng(21), cfg, device_draw=profile)
    direct = td.mix_device_audio(np.zeros(fs), x, (0, fs), np.random.default_rng(21), cfg, device_draw=profile)
    np.testing.assert_array_equal(mixed.audio, direct)  # Declared route identity only.
    # Every sample of this uninterrupted sine exceeds the envelope gate, including boundaries.
    assert mixed.events[0]["active_samples"] == fs
    expected_gain = 10**((70.-94.)/20.)/.2
    assert mixed.events[0]["receiver_gain"] == pytest.approx(expected_gain, abs=1e-12)
    scaled = td.mix_device_audio(np.zeros(fs), x*10, (0, fs), np.random.default_rng(21), cfg, device_draw=profile)
    np.testing.assert_allclose(direct, scaled, atol=1e-15)  # Intentional legacy receiver-level authority.


def test_noise_correct_and_bad_controls(record_property):
    x = np.tile([1., -1., 1., -1.], 100)
    noise = np.tile([1., 1., -1., -1.], 100)
    ref = el.LevelReference("rms", "relative", 1., "Z", (0, len(x)), 16000, "receiver")
    good = el.level_diagnostics(x, ref, background=x, source_noise=noise*.1)
    bad = el.level_diagnostics(x, ref, background=x, source_noise=noise)
    assert good["background_rise_db"] == pytest.approx(10*np.log10(1.01))
    assert good["background_rise_db"] <= 1.
    assert bad["background_rise_db"] == pytest.approx(10*np.log10(2))
    assert bad["background_rise_db"] > 1.
    record_property("correct_noise_rise_db", good["background_rise_db"])
    record_property("bad_noise_rise_db", bad["background_rise_db"])
