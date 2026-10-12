"""Shared statistical intervals for evaluation; no data or model assets."""
import math

from scipy.stats import beta


def constant_count_interval(count: int, trials: int, cap: int,
                            confidence: float = 0.95) -> tuple[float, float]:
    """Bound an IID bounded count mean when all observations equal ``count``.

    For n > 1 take q = 1 - alpha**(1/(n-1)). If P(X != c) <= q,
    the mean lies in [c(1-q), c+(cap-c)q]. Across all data-selected c,
    P(constant sample with P(X != c) > q) = sum_bad p_c**n
    <= (1-q)**(n-1) * sum_bad p_c <= alpha. Thus the whole
    component alpha suffices, with no count-value or two-tail correction.
    This bounds failure on the constant-sample branch only; it makes no
    guarantee for a separate bootstrap used on nonconstant samples.
    The cap is a declared population support bound, never a sample maximum.
    """
    if any(not math.isfinite(v) or int(v) != v for v in (count, trials, cap)):
        raise ValueError("count, trials and cap must be finite integers")
    if not 0 <= count <= cap or trials < 1 or not 0 < confidence < 1:
        raise ValueError("require 0 <= count <= cap, trials >= 1 and 0 < confidence < 1")
    if trials == 1:
        return 0.0, float(cap)
    q = -math.expm1(math.log1p(-confidence) / (trials - 1))
    return count * (1 - q), count + (cap - count) * q


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided exact Clopper-Pearson interval for a Bernoulli probability."""
    if any(not math.isfinite(v) or int(v) != v for v in (successes, trials)):
        raise ValueError("successes and trials must be finite integers")
    successes, trials = int(successes), int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper

