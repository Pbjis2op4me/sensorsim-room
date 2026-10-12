"""Declared development controls. These tests do not emit acceptance receipts."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from math import log10

import numpy as np
import pytest

from sensorsim.edge import Features
from sensorsim.evaluation import power_audit as p
from tools.power_audit import FEATURES, procedural_factory


def pair(groups=4, effect="null", seed=8):
    return procedural_factory(groups, effect, seed)


def spec(**kw):
    return p.Family(permutations=19, **kw)


def test_frozen_physical_bank_against_sine_and_time_arithmetic():
    fs = 16000
    t = np.arange(fs)/fs
    x = .5*np.sin(2*np.pi*1000*t)
    out = p.build_physical_features(x, fs, starts_s=[7.25])
    assert out["version"] == p.BANK_VERSION
    assert out["starts_s"] == [7.25]
    assert out["values"][0, 0] == pytest.approx(20*log10(.5/np.sqrt(2)), abs=1e-10)
    assert out["values"][0, 1] == pytest.approx(20*log10(.5), abs=1e-10)
    assert out["values"][0, 4] == pytest.approx(np.dot(x*x, t)/np.dot(x, x), abs=1e-12)
    # Deliberate incompatible sine-reference mutant is 3.0103 dB away.
    assert abs(out["values"][0, 0] - 20*log10(.5)) > 3
    assert not out["constant_audio"][0]


def test_hidden_level_control_reaches_raw_and_unchanged_edge_path(monkeypatch):
    rng = np.random.default_rng(4)
    audio = rng.normal(0, .1, (4, 16000))
    features = Features(FEATURES)
    observed = []
    original = p.build_physical_features
    def spy(audio, *args, **kw):
        observed.append(np.asarray(audio).copy())
        return original(audio, *args, **kw)
    monkeypatch.setattr(p, "build_physical_features", spy)
    a = p.feature_bundle(audio, features, [{}]*4)
    b = p.feature_bundle(audio*10**(.3), features, [{}]*4)
    assert len(observed) == 2
    np.testing.assert_allclose(observed[1]/observed[0], 10**(.3))
    np.testing.assert_allclose(b["views"]["physical"][:, 0]-a["views"]["physical"][:, 0], 6, atol=1e-12)
    np.testing.assert_allclose(a["views"]["deployed"], b["views"]["deployed"], atol=3e-6)


def test_order_sensitive_diagnostic_and_summary_blind_spot():
    x = np.random.default_rng(20).normal(size=16000)*np.exp(-np.arange(16000)/3000)
    reverse = x.reshape(-1, 320)[::-1].reshape(-1)
    a = p.build_physical_features(x, 16000)["values"][0]
    b = p.build_physical_features(reverse, 16000)["values"][0]
    assert b[4]-a[4] > .5
    assert b[5] == pytest.approx(-a[5], abs=.001)
    frames = Features(FEATURES)(x)
    np.testing.assert_allclose(p.band_statistics(frames), p.band_statistics(frames[..., ::-1]), atol=3e-6)


@pytest.mark.parametrize("audio,fs", [(np.empty((0, 10)), 16000), ([np.nan, 2], 16000),
                                      (np.ones(16000), 4000), (np.ones(10), 16.0)])
def test_invalid_audio_units_and_shapes(audio, fs):
    with pytest.raises(ValueError):
        p.build_physical_features(audio, fs)


def test_constant_audio_is_flagged_not_fabricated_as_measurement():
    out = p.build_physical_features(np.ones(16000), 16000)
    assert out["constant_audio"] == [True]
    assert out["level_floor_used"] == [True]
    assert np.isfinite(out["values"]).all()


def test_aliases_collapse_transitively_and_cross_domain_duplicate_abstains():
    left, right = pair()
    a, b = deepcopy(left["records"]), deepcopy(right["records"])
    a[1]["source_group"] = a[0]["source_group"]
    a[2]["session_id"] = a[1]["session_id"]
    result = p.validate_pairing(a, b, null_justification="development independent control")
    assert result["valid"]
    assert len(set(result["groups_left"][:3])) == 1
    b[0]["audio_sha256"] = a[0]["audio_sha256"]
    result = p.validate_pairing(a, b, null_justification="development independent control")
    assert not result["valid"]
    assert any("route identity" in e for e in result["errors"])


def paired_rows():
    a, b = pair()
    for left, right in zip(a["records"], b["records"]):
        for key in ("source_group", "session_id", "independence_group", "condition_id"):
            right[key] = left[key]
        for row in (left, right):
            row.update(pair_id=left["source_group"], acquisition_id=left["source_group"],
                       source_position=[1, 2, 1], microphone_position=[0, 0, 1],
                       framing_basis="external_schedule", framing_id="declared-onset")
    return a, b


def test_valid_pair_and_deliberate_invalid_peak_pair():
    a, b = paired_rows()
    result = p.validate_pairing(a["records"], b["records"], mode="paired", null_justification="prospective symmetric paired acquisition")
    assert result["valid"]
    assert result["retained_groups"] == [12, 12]
    b["records"][0]["framing_basis"] = "peak_offset"
    result = p.validate_pairing(a["records"], b["records"], mode="paired", null_justification="same class")
    assert not result["valid"]
    assert any("class/peak" in e for e in result["errors"])
    # Never change the declared null to recover a p-value.
    raw = p.run_family(a, b, spec(mode="paired"))
    assert raw["difference"] == "not_assessed"


@pytest.mark.parametrize("mutation", ["missing", "units", "constant", "empty", "hook", "rate", "fallback"])
def test_bad_family_inputs_never_yield_acceptance(mutation, monkeypatch):
    a, b = pair()
    if mutation == "missing":
        a["records"][0]["background_groups"] = None
    elif mutation == "units":
        a["units"]["physical"] = b["units"]["physical"] = ["SPL"]*7
    elif mutation == "constant":
        a["views"]["deployed"][:] = b["views"]["deployed"][:] = 0
    elif mutation == "empty":
        a["records"] = []
        a["physical"]["audio_sha256"] = []
    elif mutation == "hook":
        del b["views"]["deployed"]
    elif mutation == "rate":
        b["feature_config"]["fs"] = 8000
    elif mutation == "fallback":
        original = p.mmd_permutation_test
        def fallback(*args, **kw):
            return dict(original(*args, **kw), window_subsampled=True)
        monkeypatch.setattr(p, "mmd_permutation_test", fallback)
    result = p.infer_difference(p.run_family(a, b, spec()))
    assert result["harness"] == "invalid"
    assert result["registered_outcome"] == "inconclusive"
    assert result["equivalence"] == "not_assessed"
    assert result["raw"]["errors"]


def test_complete_family_reaches_existing_baselines_and_holm(monkeypatch):
    calls = []
    for name in ("mmd_permutation_test", "refit_group_permutation"):
        original = getattr(p, name)
        def spy(*args, _original=original, _name=name, **kw):
            calls.append((_name, np.asarray(args[0]).shape))
            return _original(*args, **kw)
        monkeypatch.setattr(p, name, spy)
    raw = p.run_family(*pair(), spec())
    assert raw["valid"], raw["errors"]
    assert len(raw["results"]) == 12
    assert sum(name == "mmd_permutation_test" for name, _ in calls) == 6
    assert sum(name == "refit_group_permutation" for name, _ in calls) == 3
    assert raw["retained_groups"] == {f"{label}/generated": [4, 4] for label in ("knock", "doorbell", "kettle")}
    pvals = sorted(r["p_value"] for r in raw["results"].values())
    assert min(r["holm_p"] for r in raw["results"].values()) == min(1., 12*pvals[0])
    result = p.infer_difference(raw)
    assert result["equivalence"] == "not_assessed"
    assert result["harness"] == "unqualified"


def test_holm_against_hand_calculation():
    assert p._holm(dict(a=.01, b=.025, c=.8)) == dict(a=.03, b=.05, c=.8)


def test_paired_mmd_uses_boolean_masks_and_whole_cluster_swaps():
    from sensorsim.evaluation.realism import _kernel_matrix
    a, b = paired_rows()
    x, y = a["views"]["physical"][:4], b["views"]["physical"][:4]
    groups = ["a", "b", "c", "d"]
    result = p._paired_baseline(x, y, groups, groups, "mmd", spec(mode="paired"), 8)
    k = _kernel_matrix(x, y, 8)[0]
    expected = k[:4, :4].mean()+k[4:, 4:].mean()-2*k[:4, 4:].mean()
    assert result["statistic"] == pytest.approx(expected)
    assert result["p_value"] >= 1/20


def test_paired_mean_permutation_against_enumerated_signs():
    x = np.array([[0.], [2.], [4.], [6.]])
    y = x+np.array([[.1], [.2], [.3], [.4]])
    groups = list("abcd")
    actual = p._max_mean(x, y, groups, groups, paired=True, permutations=1999, seed=14)
    # Enumerate the entire sign orbit independently: both all-positive and
    # all-negative assignments tie the observed maximum, 2/16 = .125.
    differences = (y-x).ravel()
    count = sum(abs(np.dot(sign, differences)) >= differences.sum()-1e-12 for sign in product([-1, 1], repeat=4))
    assert count == 2
    assert abs(actual["p_value"]-count/16) < .03


def test_one_sided_bounds_have_correct_units_and_reachability():
    zero = p.binomial_bounds(0, 1000)
    all_hits = p.binomial_bounds(300, 300)
    assert zero["upper"] == pytest.approx(1-.05**(1/1000))
    assert all_hits["lower"] == pytest.approx(.05**(1/300))
    assert zero["upper"] < .075 and all_hits["lower"] > .8
    assert p.binomial_bounds(2, 2)["lower"] < .8


def test_reduced_budget_missing_controls_and_surviving_defects_stay_visible(monkeypatch):
    # Exercise summary mechanics independently of statistical success; the
    # intentionally insensitive family is the designated surviving mutant.
    calls = []
    real = p.run_family
    def insensitive(*args, **kw):
        out = real(*args, **kw)
        out["family_rejected"] = False
        return out
    monkeypatch.setattr(p, "run_family", insensitive)
    result = p.qualify_test(procedural_factory, spec(), p.Budget(group_counts=(4,), null_datasets=1,
                            alternative_datasets=1), record=calls.append)
    assert len(calls) == 4
    cell = result["cells"][0]
    assert cell["harness"] == "unqualified"
    assert cell["registered_outcome"] == "inconclusive"
    assert cell["low_power"] and cell["qualification_failed"]
    assert cell["effects"]["scalar_plus6db"]["bounds"]["count"] == 0
    assert not result["p5_budget"] and not result["control_signoff_present"]
    assert result["independent_receipt"] is None


def test_duplicate_replicates_and_generator_error_are_retained():
    fixed = pair()
    def repeated(groups, effect, seed):
        if effect == "scalar_plus6db":
            raise ImportError("development unavailable dependency control")
        return fixed
    result = p.qualify_test(repeated, spec(), p.Budget(group_counts=(4,), null_datasets=1, alternative_datasets=1))
    assert len(result["attempts"]) == 4
    assert result["attempts"][2]["execution"] == "errored"
    assert "content reused" in str(result["attempts"][1])
    assert result["cells"][0]["harness"] == "invalid"
    assert result["cells"][0]["effects"]["eq_3khz_plus2db"]["missing_or_invalid"] == 1


def test_no_margin_means_no_equivalence_even_with_power():
    s = spec()
    raw = p.run_family(*pair(), s)
    qualification = dict(family_sha256=s.sha256, implementation_sha256=raw["implementation_sha256"],
                         feature_configuration_sha256=raw["feature_configuration_sha256"],
                         qualification_scope_sha256=raw["qualification_scope_sha256"], evidence_domain="synthetic proxy",
                         cells=[dict(groups=4, harness="qualified")])
    result = p.infer_difference(raw, qualification=qualification)
    assert result["harness"] == "qualified"
    assert result["equivalence"] == "not_assessed"
    stale = dict(qualification, family_sha256=replace(s, seed=1).sha256)
    assert p.infer_difference(raw, qualification=stale)["harness"] == "unqualified"
    assert p.infer_difference(raw, qualification=qualification, evidence_domain="deployment recording")["harness"] == "unqualified"
    assert p.legacy_status_adapter({"status": "pass"}, {"status": "pass"})["difference"] == "not_assessed"


def test_missing_pooled_hook_raises_without_fallback():
    class Model:
        meta = {"layers": []}
    with pytest.raises(ValueError, match="no pooled embedding"):
        p.feature_bundle(np.random.default_rng(2).normal(size=(1, 16000)), Features(FEATURES), [{}], model=Model())


def test_retained_group_count_cannot_be_replaced_by_row_count():
    def aliased(groups, effect, seed):
        a, b = procedural_factory(groups, effect, seed)
        a["records"][1]["source_group"] = a["records"][0]["source_group"]
        return a, b
    out = p.qualify_test(aliased, spec(), p.Budget(group_counts=(4,), null_datasets=1, alternative_datasets=1))
    assert out["cells"][0]["harness"] == "invalid"
    assert all("actual retained group count" in str(r["result"]["errors"]) for r in out["attempts"])
    assert out["attempts"][0]["result"]["retained_groups"]["knock/generated"] == [3, 4]


def test_wall_budget_records_every_unrun_replicate(monkeypatch):
    times = iter(range(20))
    monkeypatch.setattr(p.time, "monotonic", lambda: next(times))
    def never_called(*args):
        raise AssertionError("budget should stop before fixture generation")
    out = p.qualify_test(never_called, spec(), p.Budget(group_counts=(4,), null_datasets=1,
                         alternative_datasets=1, max_seconds=.5))
    assert len(out["attempts"]) == 4
    assert all(r["execution"] == "not run" for r in out["attempts"])
    assert out["cells"][0]["registered_outcome"] == "inconclusive"


def test_nonfinite_baseline_pvalue_is_invalid(monkeypatch):
    monkeypatch.setattr(p, "refit_group_permutation", lambda *a, **kw: {"p_value": float("nan")})
    out = p.run_family(*pair(), spec())
    assert not out["valid"]
    assert "invalid p-value" in str(out["errors"])
