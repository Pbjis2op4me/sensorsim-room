"""Exact bounds for a fixed-budget Monte Carlo family decision.

These intervals enclose the original finite-draw p-value, not a population
parameter. They are neither confidence intervals nor sequential p-values.
"""
from dataclasses import dataclass
import math


def first_holm_rejects(p_value: float, members: int, alpha: float) -> bool:
    """Use the original Holm multiplication, including its floating boundary."""
    # NumPy scalars must not leak a numpy.bool_ into the decision state contract.
    return bool(min(1., members * p_value) <= alpha)


@dataclass
class FixedBudgetDecision:
    permutations: int
    members: int
    alpha: float
    family_already_rejected: bool = False

    def __post_init__(self):
        if (type(self.permutations) is not int or self.permutations < 1
                or type(self.members) is not int or self.members < 1
                or not math.isfinite(self.alpha) or not 0 < self.alpha < 1
                or type(self.family_already_rejected) is not bool):
            raise ValueError("invalid fixed-budget decision contract")
        self.evaluated = 0
        self.exceedances = 0

    @property
    def interval(self):
        denominator = 1 + self.permutations
        return ((1 + self.exceedances) / denominator,
                (1 + self.exceedances + self.permutations - self.evaluated) / denominator)

    @property
    def rejected(self):
        lower, upper = self.interval
        if first_holm_rejects(upper, self.members, self.alpha):
            return True
        if not first_holm_rejects(lower, self.members, self.alpha):
            return False
        return None

    @property
    def resolved(self):
        return self.family_already_rejected or self.rejected is not None

    def observe(self, exceeds: bool):
        if self.resolved or self.evaluated >= self.permutations:
            raise ValueError("cannot evaluate after the decision is determined")
        if type(exceeds) is not bool:
            raise ValueError("one boolean exceedance required")
        self.evaluated += 1
        self.exceedances += int(exceeds)

    def report(self):
        if not self.resolved:
            raise ValueError("unfinished fixed-budget decision")
        return dict(p_value_interval=list(self.interval),
                    first_step_rejected=self.rejected,
                    permutations_evaluated=self.evaluated,
                    permutations_drawn=self.permutations,
                    exceedances_evaluated=self.exceedances,
                    stop_reason=("family_already_rejected" if self.family_already_rejected
                                 else "fixed_budget_bound"),
                    interpretation="bounds on original fixed-draw p-value; no point estimate")
