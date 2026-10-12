"""Deterministic regressions on disclosed procedural development fixtures."""

import json

import numpy as np
import pytest

from sensorsim.evaluation import permutation_decision as decision
from sensorsim.evaluation import power_audit
from tools.power_audit import procedural_factory


@pytest.mark.parametrize("alpha", [0.05, np.float64(0.05)],
                         ids=["python-float", "numpy-float64"])
def test_valid_alpha_preserves_complete_family(alpha, monkeypatch):
    # Verifier reproduction: four groups, 39 draws, seed 88031.
    bundles = procedural_factory(4, "null", 88031)
    family = power_audit.Family(permutations=39, alpha=alpha)
    full = power_audit.run_family(*bundles, family, seed=88031)
    reached = []
    original = decision.FixedBudgetDecision

    def traced(*args, **kwargs):
        reached.append(kwargs["family_already_rejected"])
        return original(*args, **kwargs)

    monkeypatch.setattr(decision, "FixedBudgetDecision", traced)
    bounded = power_audit.run_family(*bundles, family, seed=88031,
                                     execution="decision")
    assert full["valid"], full["errors"]
    assert bounded["valid"], bounded["errors"]
    assert len(full["results"]) == len(bounded["results"]) == 12
    assert len(reached) == 3 and all(type(value) is bool for value in reached)
    assert type(bounded["family_rejected"]) is bool
    assert full["family_rejected"] == bounded["family_rejected"]
    for name, result in bounded["results"].items():
        low, high = result["p_value_interval"]
        assert low <= full["results"][name]["p_value"] <= high
    json.dumps(bounded, allow_nan=False)


@pytest.mark.parametrize("alpha", [0.05, np.float64(0.05)])
@pytest.mark.parametrize("scalar", [float, np.float64])
@pytest.mark.parametrize("p_value, expected", [(0.001, True), (0.1, False)])
def test_first_holm_returns_native_bool(alpha, scalar, p_value, expected):
    assert decision.first_holm_rejects(scalar(p_value), 12, alpha) is expected


def test_uncast_holm_mutant_reaches_and_invalidates_family(monkeypatch):
    reached = []

    def uncast(p_value, members, alpha):
        value = min(1.0, members * p_value) <= alpha
        reached.append(value)
        return value

    monkeypatch.setattr(decision, "first_holm_rejects", uncast)
    bundles = procedural_factory(4, "null", 88031)
    report = power_audit.run_family(
        *bundles, power_audit.Family(permutations=39, alpha=np.float64(0.05)),
        seed=88031, execution="decision")
    assert reached and all(isinstance(value, np.bool_) for value in reached)
    assert not report["valid"] and report["family_rejected"] is None
    assert len(report["results"]) == 3
    assert any("invalid fixed-budget decision contract" in error
               for error in report["errors"])
