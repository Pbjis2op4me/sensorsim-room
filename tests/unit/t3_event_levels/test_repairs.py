# SPDX-License-Identifier: Apache-2.0
"""Disclosed verifier counterexamples reused as deterministic development controls."""
from dataclasses import replace

import numpy as np
import pytest

from sensorsim import event_levels as el
from sensorsim.acoustics.room.source_reference import SourceReference
from sensorsim.core.audio_units import AudioCalibration


def fitted(mode=el.RELATIVE, units=None):
    units = units or ("Pa" if mode == el.CALIBRATED else "relative")
    x = .02*np.sin(2*np.pi*1000*np.arange(32000)/16000)
    calibration = AudioCalibration(2., "synthetic control", "fixed session") if units == "peak_fs" else None
    p = el.ParentEvent("p", "g", "s", "kettle", "unit", "cycle", "train", "synthetic_proxy",
        "source_only", x, SourceReference(units=units, calibration=calibration),
        el.LevelReference("rms", units, 20e-6 if units == "Pa" else 1., "Z", (0, len(x)), 16000, "source-1m"))
    roles = {"parent:p": "train", "group:g": "train", "session:s": "train", "content:"+p.content_id: "train"}
    prior = el.fit_level_prior([p], mode=mode, role_bindings=roles,
        relative_anchors={"s": el.RelativeAnchor(.25, "assumed")} if mode == el.RELATIVE else {})
    return prior, el.sample_parent_gain(prior, np.random.default_rng(310), condition=p.condition)


def test_mode_only_forgery_cannot_upgrade_relative_calibration():
    _, d = fitted()
    assert el.receiver_gain(d, {"event_level_mode": el.RELATIVE}) == .25
    assert d.metadata()["absolute_calibration"] == "unknown"
    with pytest.raises(ValueError):
        replace(d, mode=el.CALIBRATED)  # V2: only this field changed in the verifier probe.


@pytest.mark.parametrize("field", ["gain", "parent", "prior_digest", "anchor_id", "render_reference_spl_db"])
def test_draw_is_bound_to_fitted_prior(field):
    _, d = fitted()
    changes = dict(gain=d.gain*10**(.21/20), parent=replace(d.parent, parent_id="alias"),
                   prior_digest="0"*64, anchor_id="different", render_reference_spl_db=80.)
    assert el.receiver_gain(d, {"event_level_mode": d.mode}) == d.gain
    with pytest.raises(ValueError):
        replace(d, **{field: changes[field]})


@pytest.mark.parametrize("mode,units", [(el.RELATIVE, "relative"), (el.CALIBRATED, "Pa"), (el.CALIBRATED, "peak_fs")])
def test_valid_source_semantics_and_consumer_revalidation(mode, units):
    _, d = fitted(mode, units)
    expected = .25 if units == "relative" else (2. if units == "peak_fs" else 1.)/(np.sqrt(2)*20e-6*10**4.7)
    assert el.receiver_gain(d, {"event_level_mode": mode}) == pytest.approx(expected)
    assert d.metadata()["absolute_calibration"] == ("unknown" if units == "relative" else "known")
    # Simulates malformed deserialization bypassing the dataclass constructor.
    object.__setattr__(d, "gain", d.gain*10**(.21/20))
    for consume in (lambda: el.receiver_gain(d, {"event_level_mode": mode}),
                    lambda: el.apply_parent_gain(d.parent.samples, d), d.metadata):
        with pytest.raises(ValueError):
            consume()


@pytest.mark.parametrize("field,value", [("mode", el.CALIBRATED), ("gains", (.5,)), ("digest", "0"*64)])
def test_prior_cannot_be_replaced_under_unchanged_fit_identity(field, value):
    prior, d = fitted()
    assert el.sample_parent_gain(prior, np.random.default_rng(0), condition=d.parent.condition).gain == .25
    with pytest.raises(ValueError):
        el.sample_parent_gain(replace(prior, **{field: value}), np.random.default_rng(0), condition=d.parent.condition)


def a_power(frequency):
    """Independent continuous A-curve arithmetic at an exact Fourier bin."""
    f2 = float(frequency)**2
    # Precise Annex E poles match the meter definition. The original reviewer
    # used rounded poles; retain that result in history, not as an exact oracle.
    ra = 12194.217**2*f2*f2/((f2+20.598997**2)*np.sqrt((f2+107.65265**2)*(f2+737.86223**2))*(f2+12194.217**2))
    return (ra*10**(2/20))**2


@pytest.mark.parametrize("quantity", ["rms", "exposure", "peak"])
@pytest.mark.parametrize("weighting", ["A", "Z"])
def test_spectral_noise_rise_uses_declared_interval_and_weighting(quantity, weighting, record_property):
    if quantity == "peak" and weighting == "A":
        with pytest.raises(ValueError):
            el.LevelReference(quantity, "relative", 1., weighting, (0, 16000), 16000, "mic")
        return
    t = np.arange(16000)/16000
    bg = np.r_[np.full(8000, 100.), np.sin(2*np.pi*1000*t), np.full(8000, 100.)]
    noise = np.r_[np.full(8000, 100.), np.sin(2*np.pi*50*t), np.full(8000, 100.)]
    ref = el.LevelReference(quantity, "relative", 1., weighting, (8000, 24000), 16000, "mic")
    expected = 10*np.log10(1+(a_power(50)/a_power(1000) if weighting == "A" else 1.))
    row = el.level_diagnostics(bg, ref, background=bg, source_noise=noise)
    assert row["background_rise_db"] == pytest.approx(expected, abs=1e-10)
    assert row["noise_reference"] == replace(ref, quantity="rms").as_dict()
    assert row["noise_diagnostic_version"] == "matched-source-noise/2"
    assert el.source_noise_gate([row])["outcome"] == ("pass" if weighting == "A" else "fail")
    record_property("spectral_noise_rise_db", row["background_rise_db"])
    # Spectrally distinct control crosses the same unchanged 1 dB gate.
    bad_noise = noise.copy()
    bad_noise[8000:24000] *= 30 if weighting == "A" else .1
    other = el.level_diagnostics(bg, ref, background=bg, source_noise=bad_noise)
    assert el.source_noise_gate([other])["outcome"] == ("fail" if weighting == "A" else "pass")


def test_weighted_dc_and_missing_reference_do_not_pass():
    x = np.ones(16000)
    ref = el.LevelReference("rms", "relative", 1., "A", (0, len(x)), 16000, "mic")
    row = el.level_diagnostics(x, ref, background=x, source_noise=x)
    assert row["background_power"] == 0.
    assert row["background_rise_db"] is None
    assert el.source_noise_gate([row])["outcome"] == "inconclusive"
    assert el.source_noise_gate([{"background_rise_db": 0.}])["outcome"] == "inconclusive"


@pytest.mark.parametrize("rise,outcome", [(.99, "pass"), (1.01, "fail")])
def test_a_weighted_noise_boundary(rise, outcome):
    t = np.arange(16000)/16000
    bg = np.sin(2*np.pi*1000*t)
    noise = np.sin(2*np.pi*50*t)*np.sqrt((10**(rise/10)-1)*a_power(1000)/a_power(50))
    ref = el.LevelReference("rms", "relative", 1., "A", (0, len(t)), 16000, "mic")
    row = el.level_diagnostics(bg, ref, background=bg, source_noise=noise)
    assert row["background_rise_db"] == pytest.approx(rise, abs=1e-10)
    assert el.source_noise_gate([row])["outcome"] == outcome


def test_noise_gate_rejects_mismatched_meters_and_missing_weighting_dependency(monkeypatch):
    import sensorsim.acoustics.weighting as weighting
    x = np.sin(2*np.pi*1000*np.arange(16000)/16000)
    ref = el.LevelReference("rms", "relative", 1., "A", (0, len(x)), 16000, "mic")
    good = el.level_diagnostics(x, ref, background=x, source_noise=x*.01)
    z_row = el.level_diagnostics(x, replace(ref, weighting="Z"), background=x, source_noise=x*.01)
    assert el.source_noise_gate([good])["outcome"] == "pass"
    with pytest.raises(ValueError, match="incompatible"):
        el.source_noise_gate([good, z_row])
    changed = dict(good, noise_reference=z_row["noise_reference"])
    with pytest.raises(ValueError, match="mismatch"):
        el.source_noise_gate([changed])
    monkeypatch.delattr(weighting, "a_weighted_mean_square")
    with pytest.raises(ImportError):
        el.level_diagnostics(x, ref, background=x, source_noise=x*.01)
