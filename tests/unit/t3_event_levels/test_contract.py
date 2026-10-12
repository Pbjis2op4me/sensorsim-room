# SPDX-License-Identifier: Apache-2.0
"""Synthetic proxy development fixtures, not a frozen acceptance collection."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from sensorsim import event_levels as el
from sensorsim.acoustics.room.source_reference import SourceReference
from sensorsim.core.audio_units import AudioCalibration


def parent(name="p", *, amplitude=1., units="relative", session="session", role="train", **kw):
    x = amplitude*np.sin(2*np.pi*np.arange(1000)/10)
    sr = kw.pop("source_reference", SourceReference(units=units))
    ref = el.LevelReference("rms", units, 20e-6 if units == "Pa" else 1., "Z", (0, len(x)), 1000, "source-1m")
    return el.ParentEvent(name, name, session, "kettle", "kettle-1", "cycle", role,
                          "synthetic_proxy", "source_only", x, sr, ref, **kw)


def bindings(parents):
    return {key: p.role for p in parents for key in (f"parent:{p.parent_id}", f"group:{p.source_group}",
            f"session:{p.session_id}", f"content:{p.content_id}")}


def fit(rows, **kw):
    return el.fit_level_prior(rows, mode=kw.pop("mode", el.RELATIVE),
        role_bindings=kw.pop("role_bindings", bindings(rows)),
        relative_anchors=kw.pop("relative_anchors", {p.session_id: el.RelativeAnchor(.1, "assumed-A") for p in rows}), **kw)


def test_level_independent_analytic_units_and_interval():
    p = parent(units="Pa")
    ref = p.level_reference
    assert el.measure_level(p.samples, ref) == pytest.approx(20*np.log10(1/np.sqrt(2)/20e-6))
    assert el.measure_level(p.samples, replace(ref, quantity="exposure")) == pytest.approx(el.measure_level(p.samples, ref))
    half = replace(ref, interval_samples=(0, 500), quantity="exposure")
    assert el.measure_level(p.samples, half) == pytest.approx(el.measure_level(p.samples, ref)-10*np.log10(2))
    assert el.measure_level(np.array([0., 2., -1.]), replace(ref, quantity="peak", interval_samples=(0, 3))) == pytest.approx(100.)
    assert el.validate_level_reference(ref.as_dict(), p.source_reference, sample_count=1000) == ref


@pytest.mark.parametrize("field,value", [("quantity", "sound_power"), ("quantity", "LUFS"), ("weighting", "Fast"),
    ("units", "dBA"), ("reference_value", 1.), ("interval_samples", (0, 0)), ("interval_samples", (-1, 2)),
    ("interval_samples", (0., 2.)), ("fs", True), ("position_id", ""), ("version", "future")])
def test_bad_level_metadata(field, value):
    with pytest.raises(ValueError):
        replace(parent(units="Pa").level_reference, **{field: value})


def test_reference_mismatch_and_unsupported_weighted_peak():
    p = parent()
    with pytest.raises(ValueError, match="units"):
        el.validate_level_reference(p.level_reference, SourceReference(units="Pa"))
    with pytest.raises(ValueError, match="interval"):
        el.validate_level_reference(p.level_reference, p.source_reference, sample_count=20)
    with pytest.raises(ValueError, match="weighting"):
        replace(p.level_reference, quantity="peak", weighting="A")


def test_a_weighted_constant_and_zero_are_explicit():
    p = parent()
    assert el.measure_level(np.zeros(1000), p.level_reference) is None
    assert el.measure_level(np.ones(1000), replace(p.level_reference, weighting="A")) is None
    d = el.level_diagnostics(np.ones(1000), p.level_reference)
    assert d["constant"] and not d["silent"]
    assert d["source_noise_status"] == "unavailable"


@pytest.mark.parametrize("samples", [[], [[1., 2.]], [np.nan, 0.], [np.inf, 0.], [1j, 0j], ["1", "2"]])
def test_bad_waveforms(samples):
    with pytest.raises(ValueError):
        el.content_sha256(samples, 1000)


def test_constant_silence_and_censoring():
    p = parent()
    for x in (np.ones(1000), np.zeros(1000)):
        with pytest.raises(ValueError, match="constant"):
            replace(p, samples=x)
    censored = replace(p, samples=np.zeros(1000), censored=True)
    prior = fit([censored])
    draw = el.sample_parent_gain(prior, np.random.default_rng(1), condition=censored.condition)
    assert draw.parent.censored and not el.apply_parent_gain(censored.samples, draw).any()
    assert draw.parent.metadata()["observed_level_db"] is None


def test_joint_waveform_level_draw_and_session_offset():
    quiet, loud = parent("quiet", amplitude=.1), parent("loud")
    prior = fit([quiet, loud], session_offsets_db={"session": 6.})
    rng = np.random.default_rng(10)
    draws = [el.sample_parent_gain(prior, rng, condition=quiet.condition) for _ in range(40)]
    assert {d.parent.parent_id for d in draws} == {"quiet", "loud"}
    assert all(d.gain == pytest.approx(.1*10**(.3)) for d in draws)
    levels = {d.parent.parent_id: el.measure_level(d.parent.samples, d.parent.level_reference) for d in draws}
    assert levels["loud"]-levels["quiet"] == pytest.approx(20.)
    assert all(d.metadata()["absolute_calibration"] == "unknown" for d in draws)
    assert prior.digest == fit([quiet, loud], session_offsets_db={"session": 6.}).digest
    assert prior.digest != fit([quiet, loud]).digest
    json.dumps(prior.metadata(), allow_nan=False)


def test_no_mutable_waveform_aliases():
    p = parent()
    with pytest.raises(ValueError):
        p.samples[0] = 8.
    with pytest.raises(ValueError):
        p.samples.setflags(write=True)


def test_exact_conditions_and_group_weighting():
    rows = [parent("a"), parent("b", amplitude=.2), parent("c", amplitude=.3)]
    rows[1] = replace(rows[1], source_group="a")
    prior = fit(rows)
    class ControlledRng(np.random.Generator):
        def __init__(self):
            super().__init__(np.random.PCG64(0)); self.sizes = []
        def integers(self, high, *args, **kwargs):
            self.sizes.append(high); return 0
    rng = ControlledRng()
    el.sample_parent_gain(prior, rng, condition=rows[0].condition)
    assert rng.sizes == [2, 2]  # Group first; repeats never add group probability.
    with pytest.raises(ValueError, match="fallback"):
        el.sample_parent_gain(prior, rng, condition=("kettle", "another", "cycle"))


@pytest.mark.parametrize("prefix", ["parent", "group", "session", "content"])
@pytest.mark.parametrize("role", ["audit", "development", None])
def test_role_aliases_and_missing_bindings_fail(prefix, role):
    p = parent()
    roles = bindings([p])
    key = next(k for k in roles if k.startswith(prefix+":"))
    roles[key] = role
    with pytest.raises(ValueError, match="train"):
        fit([p], role_bindings=roles)


def test_empty_duplicate_and_nontraining_parents():
    with pytest.raises(ValueError, match="empty"):
        fit([])
    p = parent()
    with pytest.raises(ValueError, match="duplicate"):
        fit([p, replace(p, parent_id="alias", source_group="other")])
    with pytest.raises(ValueError, match="train"):
        fit([replace(p, role="audit")])
    with pytest.raises(ValueError, match="complete"):
        replace(p, level_reference=replace(p.level_reference, interval_samples=(0, 500)))


def test_absolute_calibration_and_peak_sine_reference():
    pa = parent(units="Pa")
    prior = fit([pa], mode=el.CALIBRATED, relative_anchors={})
    draw = el.sample_parent_gain(prior, np.random.default_rng(2), condition=pa.condition)
    assert draw.gain == pytest.approx(1/(np.sqrt(2)*20e-6*10**(94/20)))
    calibration = AudioCalibration(2., "capture", "fixed gain calibration")
    fs = parent(units="peak_fs", source_reference=SourceReference(units="peak_fs", calibration=calibration))
    fsprior = fit([fs], mode=el.CALIBRATED, relative_anchors={})
    assert fsprior.gains[0] == pytest.approx(2*prior.gains[0])
    assert draw.metadata()["absolute_calibration"] == "known"
    with pytest.raises(ValueError, match="unknown"):
        fit([parent()], mode=el.CALIBRATED, relative_anchors={})


@pytest.mark.parametrize("change", [{"capture_route": "known_playback"}, {"response_domain": "device_recording"},
    {"response_domain": "effective_playback"}, {"response_domain": "receiver_pre_device"},
    {"source_reference": SourceReference(target="effective_source")},
    {"source_reference": SourceReference(target="ref_microphone", reference_distance_m=.3)}])
def test_no_double_room_or_device_or_distance(change):
    with pytest.raises(ValueError, match="separate adapter"):
        fit([replace(parent(), **change)])


def test_session_anchor_and_reference_consistency():
    with pytest.raises(ValueError, match="anchor"):
        fit([parent()], relative_anchors={})
    with pytest.raises(ValueError, match="unused"):
        fit([parent()], session_offsets_db={"unseen": 2.})
    a, b = parent("a"), parent("b", amplitude=.2)
    b = replace(b, source_reference=replace(b.source_reference, time_origin="changed"))
    with pytest.raises(ValueError, match="session calibration"):
        fit([a, b])


def test_render_recovery_and_explicit_fallbacks():
    sr = SourceReference(units="Pa", calibration=AudioCalibration(2., "capture", "fixed"))
    row = dict(dry_source=dict(source_reference=sr.as_dict()),
               training_level=dict(relative_units="Pa", total_export_scale=.25, relative_file=None),
               audio_boundary=dict(source_units="Pa", pa_per_peak_fs=2.))
    x = np.array([.125, -.25])
    recovered, info = el.recover_render(x, row)
    np.testing.assert_array_equal(recovered, [1., -2.])
    assert info["quantization_error_bound"] == .5/32768*8
    row["training_level"].update(relative_file="relative.wav", relative_sha256="digest")
    with pytest.raises(ValueError, match="no silent"):
        el.recover_render(x, row)
    exact, info = el.recover_render(x, row, relative_samples=[1., -2.])
    np.testing.assert_array_equal(exact, recovered)
    assert info["recovery"] == "lossless_relative"
    row["dry_source"]["level_reference"] = {"effective_source": True}
    with pytest.raises(ValueError, match="effective"):
        el.recover_render(x, row, relative_samples=[1., -2.])


def test_source_noise_diagnostic_retains_rise_and_zero_denominator():
    p = parent()
    bg = np.tile([1., -1., 1., -1.], 250)
    noise = np.tile([1., 1., -1., -1.], 250)
    d = el.level_diagnostics(p.samples, p.level_reference, background=bg, source_noise=noise)
    assert d["background_rise_db"] == pytest.approx(10*np.log10(2))
    zero = el.level_diagnostics(p.samples, p.level_reference, background=bg*0, source_noise=noise)
    assert zero["background_rise_db"] is None
    assert zero["source_noise_status"] == "undefined_zero_power"
    assert el.source_noise_gate([d])["outcome"] == "fail"
    assert el.source_noise_gate([d, zero])["outcome"] == "inconclusive"
    assert el.source_noise_gate([])["outcome"] == "inconclusive"
    with pytest.raises(ValueError, match="both"):
        el.level_diagnostics(p.samples, p.level_reference, background=bg)


def test_playback_gate_scope_controls_and_missing_data():
    ref = parent().level_reference.as_dict()
    rows = [dict(route="known_playback", role="audit", matched_acquisition=True, group_id=f"g{i}",
        pair_id=f"pair{i}", measured_reference=ref, predicted_reference=ref, measured_db=-20., predicted_db=-18.,
        measured_content_sha256=f"{i:064x}", predicted_content_sha256=f"{i+5:064x}") for i in range(5)]
    result = el.playback_level_gate(rows)
    assert result["outcome"] == "pass" and result["physical_prior_outcome"] == "inconclusive"
    assert result["median_absolute_error_db"] == 2.
    assert el.playback_level_gate([r | {"predicted_db": -10.} for r in rows])["outcome"] == "fail"
    assert el.playback_level_gate([])["outcome"] == "inconclusive"
    assert el.playback_level_gate([rows[0] | {"measured_db": None}])["outcome"] == "inconclusive"
    assert el.playback_level_gate([rows[0] | {"route": "actual_source"}])["outcome"] == "inconclusive"
    with pytest.raises(ValueError, match="group"):
        el.playback_level_gate([rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate playback"):
        el.playback_level_gate([rows[0], rows[0] | {"group_id": "alias"}])
    with pytest.raises(ValueError, match="mismatch"):
        el.playback_level_gate([rows[0] | {"predicted_reference": ref | {"weighting": "A"}}])


def test_operational_bank_provenance_contract_and_aliases():
    p = parent()
    rows = [dict(id=role, role=role, parent_id=p.parent_id if role == "train" else role,
        session_id=p.session_id if role == "train" else role+"-session",
        lineage_ids=[p.source_group if role == "train" else role+"-group"],
        parent_audio_sha256=[p.content_id if role == "train" else f"{i:064x}"],
        audio_sha256=f"{i+10:064x}", file_sha256=f"{i+20:064x}")
        for i, role in enumerate(("train", "development", "audit"))]
    bank = dict(schema="operational-bank/1", allocation_frozen=True, rows=rows)
    digest = lambda: hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    roles = el.role_bindings_from_bank(bank, expected_sha256=digest())
    assert fit([p], role_bindings=roles).parents[0] is p
    with pytest.raises(ValueError, match="digest"):
        el.role_bindings_from_bank(bank, expected_sha256="0"*64)
    rows[-1]["parent_audio_sha256"] = [p.content_id]
    with pytest.raises(ValueError, match="content crosses"):
        el.role_bindings_from_bank(bank, expected_sha256=digest())


def test_missing_integration_cannot_enable_parent_training():
    assert el.validate_mode({}) == el.LEGACY
    el.require_legacy_training({})
    with pytest.raises(ValueError, match="not integrated"):
        el.require_legacy_training({"event_level_mode": el.RELATIVE})
    with pytest.raises(ValueError, match="versionless"):
        el.validate_mode({"event_level_mode": "relative_parent_anchor"})
    with pytest.raises(ValueError, match="stress"):
        el.validate_mode({"event_level_mode": el.RELATIVE, "device_split": "stress"})
    p = parent(); d = el.sample_parent_gain(fit([p]), np.random.default_rng(0), condition=p.condition)
    assert el.receiver_gain(d, {"event_level_mode": el.RELATIVE}) == d.gain
    with pytest.raises(ValueError, match="references"):
        el.receiver_gain(d, {"event_level_mode": el.RELATIVE, "render_reference_spl_db": 80.})
    with pytest.raises(ValueError, match="matching"):
        el.receiver_gain(None, {"event_level_mode": el.RELATIVE})
