"""Acoustic gain conventions at adapter boundaries, independent of solver choice.

These records describe bookkeeping, not physical validation. ModelValidity stays
with each model. ndarray DSP and the independent validation solvers remain separate.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class GainBudget:
    model_id: str
    quantity: Literal["pressure", "power"]
    phase_convention: Literal["exp(-iwt)", "exp(+iwt)", "magnitude", "retarded_time", "none"]
    reference_distance_m: float
    spreading_included: bool
    air_included: bool
    air_distance_policy: Literal["none", "path_length", "horizontal_range"] = "none"

    def __post_init__(self):
        if (not self.model_id or self.quantity not in ("pressure", "power")
                or self.phase_convention not in ("exp(-iwt)", "exp(+iwt)", "magnitude", "retarded_time", "none")
                or not np.isfinite(self.reference_distance_m) or self.reference_distance_m <= 0
                or self.air_distance_policy not in ("none", "path_length", "horizontal_range")
                or self.air_included != (self.air_distance_policy != "none")):
            raise ValueError("explicit acoustic quantity, phase, reference distance and air policy required")


def relative_pressure(pressure, distance_m, wavenumber_rad_m, *, budget: GainBudget):
    """Remove included 1/R spreading and the direct free-field travel phase.

    Air loss, ground and all other physics remain in the result. This is not an
    absorption compensator. Refuse magnitudes/powers or an already-relative gain.
    """
    if not budget.spreading_included:
        raise ValueError("relative pressure requires included spreading")
    if budget.quantity != "pressure" or budget.phase_convention not in ("exp(-iwt)", "exp(+iwt)"):
        raise ValueError("complex pressure with a declared harmonic phase convention required")
    distance = np.asarray(distance_m, dtype=float)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError("finite positive reference path lengths required")
    sign = -1 if budget.phase_convention == "exp(-iwt)" else 1
    return pressure * (distance/budget.reference_distance_m) * np.exp(sign*1j*wavenumber_rad_m*distance)


def path_length_compensation(path_length_m, direct_distance_m):
    """L/d pressure factor when a direct-distance gain rides on a 1/L route."""
    path, direct = np.broadcast_arrays(np.asarray(path_length_m, float), np.asarray(direct_distance_m, float))
    if np.any(~np.isfinite(path)) or np.any(~np.isfinite(direct)) or np.any(path <= 0) or np.any(direct <= 0):
        raise ValueError("finite positive path and direct distances required")
    return path/direct
