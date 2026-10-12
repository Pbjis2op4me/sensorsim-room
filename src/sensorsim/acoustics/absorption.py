"""ISO 9613-1:1993 atmospheric absorption (pure-tone attenuation coefficient).

Equations (ISO 9613-1:1993 eqs. 3-5 and Annex B for the saturation vapour pressure), as also
written out in parent-project note acoustic_propagation (not distributed) §4:

    C     = -6.8346 (T01/T)^1.261 + 4.6151,         T01 = 273.16 K (triple point)
    h     = RH * (p_sat/p_r) / (p_a/p_r),            p_sat/p_r = 10^C       [molar conc., %]
    f_rO  = (p_a/p_r) (24 + 4.04e4 h (0.02 + h)/(0.391 + h))
    f_rN  = (p_a/p_r) (T/T0)^-1/2 (9 + 280 h exp(-4.170 ((T/T0)^-1/3 - 1)))
    alpha = 8.686 f^2 [1.84e-11 (p_r/p_a)(T/T0)^1/2
            + (T/T0)^-5/2 (0.01275 e^{-2239.1/T}/(f_rO + f^2/f_rO)
                          + 0.1068  e^{-3352.0/T}/(f_rN + f^2/f_rN))]      [dB/m]

with T0 = 293.15 K and p_r = 101.325 kPa. Checked in the parent project against brief Appendix B and ISO 9613-2:1996
Table 2; the standard's table and those checks are not included in this repository.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from sensorsim.core.messages import ModelValidity
from sensorsim.acoustics.gain_budget import GainBudget

GAIN_BUDGET = GainBudget("iso9613-1/1993", "pressure", "magnitude", 1., False, True, "path_length")

T0_K = 293.15
T01_K = 273.16
P_REF_KPA = 101.325

VALIDITY = ModelValidity(
    name="ISO 9613-1 air absorption",
    status="validated",
    freq_range_hz=(50.0, 10_000.0),
    range_m=(0.0, 3_000.0),
    conditions=("ISO 9613-1 section 7.1 accuracy +/-10 % requires water-vapour molar concentration "
                "0.05..5 %, -20..50 C, p < 200 kPa and f/p = 4e-4..10 Hz/Pa (RH <=100 %); "
                "RH alone does not define this range. Drier air has wider section 7.2/7.3 error bounds. "
                "Frequency range here is the tabulated validation band, not a hard formula cutoff; "
                "homogeneous atmosphere along the path (scenario T/RH/p). Applied as a zero-phase magnitude "
                "filter per path, re-designed per 256 ms frame from the path length (dispersion ignored)."),
    citation="ISO 9613-1:1993; parent-project checks against brief Appendix B and ISO 9613-2:1996 Table 2 (not included here).",
)


def molar_concentration_h(temperature_c: float, relative_humidity_pct: float,
                          pressure_kpa: float = P_REF_KPA) -> float:
    """Molar concentration of water vapour h [%] (ISO 9613-1 Annex B)."""
    if (not np.isfinite([temperature_c, relative_humidity_pct, pressure_kpa]).all()
            or temperature_c <= -273.15 or not 0 <= relative_humidity_pct <= 100
            or pressure_kpa <= 0):
        raise ValueError("finite temperature above absolute zero, RH in [0,100] and positive pressure required")
    t = temperature_c + 273.15
    c = -6.8346 * (T01_K / t) ** 1.261 + 4.6151
    return relative_humidity_pct * 10.0 ** c / (pressure_kpa / P_REF_KPA)


@dataclass(frozen=True)
class IsoAtmosphere:
    """ISO-native state: total static pressure and molar water-vapour percent.

    Construct via the named adapters: RH percent is not molar percent, and
    total pressure is not dry-air partial pressure. This retains ISO Annex B;
    it does not substitute another modality's saturation-vapour model.
    """
    temperature_k: float
    total_pressure_kpa: float
    water_vapour_molar_pct: float

    @classmethod
    def from_iso_inputs(cls, temperature_c, relative_humidity_pct, pressure_kpa=P_REF_KPA):
        h = molar_concentration_h(temperature_c, relative_humidity_pct, pressure_kpa)
        return cls(temperature_c+273.15, pressure_kpa, h)


def alpha_db_per_m(f_hz: np.ndarray, temperature_c: float, relative_humidity_pct: float,
                   pressure_kpa: float = P_REF_KPA) -> np.ndarray:
    """Pure-tone attenuation [dB/m]; DC is zero. See VALIDITY for accuracy limits.

    Reject nonphysical inputs; finite extrapolation outside the tabulated band
    remains available for FFT filters and is not an accuracy claim.
    """
    f = np.asarray(f_hz, dtype=np.float64)
    if not np.isfinite(f).all() or np.any(f < 0):
        raise ValueError("finite nonnegative frequencies required")
    state = IsoAtmosphere.from_iso_inputs(temperature_c, relative_humidity_pct, pressure_kpa)
    t = state.temperature_k
    pa = state.total_pressure_kpa / P_REF_KPA
    tr = t / T0_K
    h = state.water_vapour_molar_pct
    fro = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frn = pa * tr ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * (tr ** (-1.0 / 3.0) - 1.0)))
    f2 = f * f
    return 8.686 * f2 * (1.84e-11 / pa * tr ** 0.5
                         + tr ** -2.5 * (0.01275 * np.exp(-2239.1 / t) / (fro + f2 / fro)
                                         + 0.1068 * np.exp(-3352.0 / t) / (frn + f2 / frn)))


def amplitude_factor(f_hz: np.ndarray, distance_m: np.ndarray | float, temperature_c: float,
                     relative_humidity_pct: float, pressure_kpa: float = P_REF_KPA) -> np.ndarray:
    """Linear pressure-amplitude factor 10^(-alpha*r/20); broadcasts f against distance."""
    distance = np.asarray(distance_m, dtype=np.float64)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("finite nonnegative path length required")
    a = alpha_db_per_m(f_hz, temperature_c, relative_humidity_pct, pressure_kpa)
    return 10.0 ** (-a * distance / 20.0)
