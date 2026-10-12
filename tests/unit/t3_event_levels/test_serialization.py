# SPDX-License-Identifier: Apache-2.0
"""F1 development regressions from the independent verifier's JSON counterexample.

Coherent source noise has the analytic rise 20 log10(1 + amplitude), for
both A and Z. These synthetic proxies test persistence, not physical realism.
"""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from sensorsim import event_levels as el


def diagnostic(weighting, rise, quantity="rms"):
    fs = 16000
    background = np.sin(2*np.pi*1000*np.arange(fs)/fs)
    noise = background*(10**(rise/20)-1)
    ref = el.LevelReference(quantity, "relative", 1., weighting, (0, fs), fs, "receiver")
    return el.level_diagnostics(background, ref, background=background, source_noise=noise)


@pytest.mark.parametrize("weighting", ["A", "Z"])
@pytest.mark.parametrize("rise,expected", [(.99, "pass"), (1.01, "fail")])
def test_noise_reference_json_roundtrip(tmp_path, weighting, rise, expected, record_property):
    row = diagnostic(weighting, rise)
    direct = el.source_noise_gate([row])
    assert direct["outcome"] == expected
    assert direct["median_rise_db"] == pytest.approx(rise, abs=1e-12)
    path = tmp_path/"noise.json"
    path.write_text(json.dumps(row, allow_nan=False))
    loaded = json.loads(path.read_text())
    result = el.source_noise_gate([loaded])
    assert result["outcome"] == expected
    assert result["median_rise_db"] == direct["median_rise_db"]
    assert result["maximum_rise_db"] == direct["maximum_rise_db"]
    assert result["rows"] == [loaded]
    record_property("analytic_rise_db", rise)
    record_property("persisted_rise_db", result["median_rise_db"])


@pytest.mark.parametrize("quantity,weighting", [("rms", "A"), ("exposure", "A"), ("peak", "Z")])
def test_reference_wire_schema_and_legacy_tuple_compatibility(quantity, weighting):
    row = diagnostic(weighting, .99, quantity)
    ref = el.LevelReference(**row["reference"])
    assert ref.interval_samples == (0, 16000)  # Immutable in memory.
    assert ref.as_dict()["interval_samples"] == [0, 16000]  # JSON wire array.
    assert json.loads(json.dumps(row, allow_nan=False)) == row
    for target_tuple, noise_tuple in [(False, False), (True, False), (False, True), (True, True)]:
        variant = copy.deepcopy(row)
        for field, use_tuple in [("reference", target_tuple), ("noise_reference", noise_tuple)]:
            if use_tuple:
                variant[field]["interval_samples"] = tuple(variant[field]["interval_samples"])
        before = copy.deepcopy(variant)
        assert el.source_noise_gate([variant])["outcome"] == "pass"
        assert variant == before  # Validation does not rewrite the caller's evidence.
        assert el.LevelReference(**variant["noise_reference"]) == replace(ref, quantity="rms")


@pytest.mark.parametrize("field,value", [
    ("weighting", "Z"), ("interval_samples", [1, 16000]),
    ("fs", 8000), ("position_id", "elsewhere"), ("units", "peak_fs"),
    ("quantity", "exposure"), ("reference_value", 2.), ("version", "future"),
    ("interval_samples", [0., 16000.]), ("interval_samples", [False, 16000]),
    ("interval_samples", [0, 0]), ("interval_samples", [0]),
    ("fs", True), ("position_id", ""),
])
def test_persisted_invalid_reference_is_refused(field, value):
    good = json.loads(json.dumps(diagnostic("A", .99), allow_nan=False))
    assert el.source_noise_gate([good])["outcome"] == "pass"
    bad = copy.deepcopy(good)
    bad["noise_reference"][field] = value
    # Python equates False with 0 and 0.0 with 0; JSON preserves these types.
    assert json.dumps(bad["noise_reference"]) != json.dumps(good["noise_reference"])
    with pytest.raises(ValueError):
        el.source_noise_gate([bad])


@pytest.mark.parametrize("weighting", ["A", "Z"])
def test_genuinely_incompatible_meter_rejected(weighting):
    row = json.loads(json.dumps(diagnostic(weighting, .99), allow_nan=False))
    assert el.source_noise_gate([row])["outcome"] == "pass"
    bad = copy.deepcopy(row)
    bad["noise_reference"]["weighting"] = "Z" if weighting == "A" else "A"
    with pytest.raises(ValueError, match="mismatch"):
        el.source_noise_gate([bad])


def test_persisted_mixed_missing_and_undefined_rows_do_not_pass():
    row = json.loads(json.dumps(diagnostic("A", .99), allow_nan=False))
    with pytest.raises(ValueError, match="incompatible"):
        el.source_noise_gate([row, diagnostic("Z", .99)])
    for field in ("noise_reference", "noise_diagnostic_version"):
        bad = copy.deepcopy(row)
        del bad[field]
        assert el.source_noise_gate([row, bad])["outcome"] == "inconclusive"
    ref = el.LevelReference(**row["reference"])
    zero = np.zeros(ref.fs)
    undefined = el.level_diagnostics(zero, ref, background=zero, source_noise=zero)
    loaded = json.loads(json.dumps(undefined, allow_nan=False))
    assert el.source_noise_gate([row, loaded])["outcome"] == "inconclusive"
    assert el.source_noise_gate([])["outcome"] == "inconclusive"


def test_persisted_noise_reference_requires_explicit_version():
    row = json.loads(json.dumps(diagnostic("A", .99), allow_nan=False))
    assert el.source_noise_gate([row])["outcome"] == "pass"
    del row["noise_reference"]["version"]
    with pytest.raises(ValueError, match="version"):
        el.source_noise_gate([row])
