"""Model validity statements shared by the acoustic and recognition models."""
from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Literal

ValidationStatus = Literal["validated", "partially validated", "unvalidated"]


def validate_validity_bounds(*ranges):
    """Shared in-memory/persistence domain policy; explicit infinity is supported."""
    for bounds in ranges:
        if bounds is not None and (len(bounds) != 2 or any(isnan(x) for x in bounds)
                                   or bounds[0] > bounds[1]):
            raise ValueError("validity bounds must be ordered pairs and cannot contain NaN")


@dataclass(frozen=True)
class ModelValidity:
    """Validity statement documenting each model's evidence and limits."""
    name: str
    status: ValidationStatus
    freq_range_hz: tuple[float, float] | None = None
    range_m: tuple[float, float] | None = None
    conditions: str = ""
    citation: str = ""

    def __post_init__(self):
        if self.status not in ("validated", "partially validated", "unvalidated"):
            raise ValueError(f"unsupported validation status {self.status!r}")
        validate_validity_bounds(self.freq_range_hz, self.range_m)
