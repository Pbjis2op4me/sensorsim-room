"""Shared diffuse decay budget (D03); renderer energy normalization stays outside.

Eyring/Sabine and ISO 9613-1 retain RoomModel's analytic validity limits. This
calculator is not a measured decay estimator; those belong to room.metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sensorsim.acoustics.absorption import alpha_db_per_m, molar_concentration_h

DecayPolicy = Literal["room-decay/1", "legacy-0.161/1"]


@dataclass(frozen=True)
class RoomDecayBudget:
    frequency_hz: np.ndarray
    volume_m3: float
    surface_area_m2: float
    mean_surface_loss: np.ndarray
    surface_loss_m2: np.ndarray
    air_energy_loss_per_m: np.ndarray
    contents_absorption_area_m2: np.ndarray
    sound_speed_m_s: float
    temperature_c: float
    humidity_pct: float
    pressure_kpa: float
    method: str
    policy: DecayPolicy
    air_policy: str
    decay_source: str
    t60_s: np.ndarray

    @property
    def air_loss_area_m2(self) -> np.ndarray:
        return 4*self.volume_m3*self.air_energy_loss_per_m

    @property
    def decay_area_m2(self) -> np.ndarray:
        return self.surface_loss_m2 + self.air_loss_area_m2 + self.contents_absorption_area_m2

    def metadata(self) -> dict:
        """Finite report values, with explicit reasons for unavailable loss/decay."""
        from sensorsim.core.serialization import report_data
        return report_data({
            "policy": self.policy, "method": self.method, "air_policy": self.air_policy,
            "decay_source": self.decay_source, "frequency_hz": self.frequency_hz.tolist(),
            "sound_speed_m_s": self.sound_speed_m_s, "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct, "pressure_kpa": self.pressure_kpa,
            "surface_loss_m2": self.surface_loss_m2.tolist(),
            "air_energy_loss_per_m": self.air_energy_loss_per_m.tolist(),
            "contents_absorption_area_m2": self.contents_absorption_area_m2.tolist(),
            "t60_s": self.t60_s.tolist(),
        })


def room_decay_budget(*, volume_m3: float, surface_area_m2: float, mean_surface_loss,
                      frequency_hz, sound_speed_m_s: float, method: str = "eyring",
                      air_absorption: bool = True, temperature_c: float = 20.,
                      humidity_pct: float = 50., pressure_kpa: float = 101.325,
                      contents_absorption_area_m2=0., measured_t60_s=None,
                      policy: DecayPolicy = "room-decay/1") -> RoomDecayBudget:
    """T60 = 24 ln(10) V / [c (A_surface + 4 m V + A_contents)].

    m is the ENERGY exponent per metre, alpha_dB*ln(10)/10. A measured
    override already includes every decay loss: air is never added to its slope.
    The legacy coefficient is available only by name for frozen reproductions.
    Frequency zero is legal for scene FFT grids; extrapolation is not validation.
    """
    f = np.asarray(frequency_hz, float)
    am = np.broadcast_to(np.asarray(mean_surface_loss, float), f.shape).copy()
    contents = np.broadcast_to(np.asarray(contents_absorption_area_m2, float), f.shape).copy()
    if (f.ndim != 1 or not f.size or not np.isfinite(f).all() or np.any(f < 0)
            or not np.isfinite([volume_m3, surface_area_m2, sound_speed_m_s]).all()
            or min(volume_m3, surface_area_m2, sound_speed_m_s) <= 0
            or not np.isfinite(am).all() or np.any((am < 0) | (am > 1))
            or not np.isfinite(contents).all() or np.any(contents < 0)):
        raise ValueError("finite room geometry, frequencies, energy fractions and contents required")
    molar_concentration_h(temperature_c, humidity_pct, pressure_kpa)
    if policy not in ("room-decay/1", "legacy-0.161/1"):
        raise ValueError("unknown room decay policy")
    if method == "sabine":
        surface = surface_area_m2*am
    elif method == "eyring":
        with np.errstate(divide="ignore"):
            surface = -surface_area_m2*np.log1p(-am)
    else:
        raise ValueError("method must be eyring or sabine")
    m = (alpha_db_per_m(f, temperature_c, humidity_pct, pressure_kpa)*np.log(10)/10
         if air_absorption else np.zeros_like(f))
    loss = surface + 4*volume_m3*m + contents
    if measured_t60_s is None:
        with np.errstate(divide="ignore"):
            t60 = (24*np.log(10)*volume_m3/(sound_speed_m_s*loss)
                   if policy == "room-decay/1" else .161*volume_m3/loss)
        source = "inferred"
    else:
        t60 = np.broadcast_to(np.asarray(measured_t60_s, float), f.shape).copy()
        if not np.isfinite(t60).all() or np.any(t60 <= 0):
            raise ValueError("measured T60 must be finite and positive")
        source = "measured-override"
    return RoomDecayBudget(f.copy(), float(volume_m3), float(surface_area_m2), am, surface, m,
                           contents, float(sound_speed_m_s), float(temperature_c), float(humidity_pct),
                           float(pressure_kpa), method, policy,
                           "iso9613-1/1993" if air_absorption else "disabled", source, t60)
