"""Versioned shoebox inputs; coefficients are energy fractions, not pressure gains."""
from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sensorsim.acoustics.room.decay import room_decay_budget
from sensorsim.acoustics.room.directivity import Directivity
from sensorsim.core.geometry import CRAMER_RANGE, HUMID_SOUND_SPEED_MODEL, sound_speed
from sensorsim.core.messages import ModelValidity
from sensorsim.core.serialization import canonical_json

OCTAVE_HZ = (125, 250, 500, 1000, 2000, 4000, 8000)
SURFACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
Bands = tuple[float, float, float, float, float, float, float]

VALIDITY = ModelValidity(
    name="Shoebox room Eyring/Sabine and mixing-time estimates",
    status="partially validated", freq_range_hz=(125., 8000.),
    conditions="Analytic equations only; diffuse mild absorption, dimensions above wavelength. "
    "Mixing time is a three-mean-free-flight heuristic, not a measured estimator. "
    "Supplied surface coefficients/provenance are not independently validated.",
    citation="Eyring 1930, JASA 1:217; ISO 9613-1:1993; parent-project note ROOM_SYNTH_PROTOCOL (not distributed)",
)


def uniform_bands(value: float) -> dict[str, Bands]:
    """Construct six identical seven-octave energy-coefficient rows."""
    return {s: (value,) * 7 for s in SURFACES}


class EarlyScattering(BaseModel):
    """Opt-in Lambertian energy transport, ISO 17497-1 coefficient convention.

    Frequency-dependent coefficients live on walls/boxes, never in this numerical
    configuration. rays/max_bounces control convergence, not fitted material data.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    method: Literal['diffuse-rain/1'] = 'diffuse-rain/1'
    rays: int = Field(default=2048, ge=64, le=131072)
    max_bounces: int = Field(default=32, ge=1, le=128)
    receiver_radius_m: float = Field(default=.1, gt=0., le=.5)

    @model_validator(mode='after')
    def _count(self):
        if self.rays & (self.rays-1): raise ValueError('scattering rays must be a power of two')
        return self


class BoxObstacle(BaseModel):
    """Manifest-compatible solid axis-aligned box; material is an informational label.

    Default coefficients are assumptions, never inferred from the material name.
    A box has one seven-band coefficient row shared by all its exposed faces.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    name: str
    min_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    material: str = "unspecified"
    absorption: Bands = (.2,) * 7
    scattering: Bands = (.1,) * 7

    @model_validator(mode="after")
    def _box(self) -> "BoxObstacle":
        if min(self.size_xyz) <= 0 or min(self.min_xyz) < 0:
            raise ValueError("box sizes must be positive, corners nonnegative")
        if any(not 0 <= x <= 1 for x in self.absorption + self.scattering):
            raise ValueError("obstacle coefficients must lie in [0,1]")
        return self

    @property
    def max_xyz(self) -> np.ndarray:
        return np.array(self.min_xyz) + self.size_xyz

    def contains(self, xyz, clearance_m: float = 0.) -> np.ndarray:
        """Closed expanded box (conservative axis-wise clearance, including faces)."""
        p = np.asarray(xyz)
        return np.all((p >= np.array(self.min_xyz)-clearance_m) &
                      (p <= self.max_xyz+clearance_m), axis=-1)


class RoomModel(BaseModel):
    """Local Cartesian metres: lower corner (0,0,0), z vertical.

    Geometry is tagged for future polyhedral schemas; v1 accepts only shoeboxes.
    Optional measured T60 overrides affect the late tail, never early wall losses.
    Uncertainty stores caller-supplied bounds/units/methods, not invented distributions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["room-model/1"] = "room-model/1"
    geometry: Literal["shoebox"] = "shoebox"
    room_id: str = "synthetic-room"
    shoebox_dims_m: tuple[float, float, float]
    absorption: dict[str, Bands] = Field(default_factory=lambda: uniform_bands(.2))
    scattering: dict[str, Bands] = Field(default_factory=lambda: uniform_bands(0.))
    obstacles: tuple[BoxObstacle, ...] = ()
    source_directivity: Directivity | None = Field(default=None, exclude_if=lambda v: v is None)
    receiver_directivity: Directivity | None = Field(default=None, exclude_if=lambda v: v is None)
    early_scattering: EarlyScattering | None = Field(default=None, exclude_if=lambda v: v is None)
    temperature_c: float = Field(default=20., ge=-20., le=50.)
    humidity_pct: float = Field(default=50., ge=10., le=100.)
    # Omitted when unused so existing finite canonical model bytes stay stable.
    reference_sound_speed_m_s: float | None = Field(default=None, gt=0., exclude_if=lambda v: v is None)
    # humid-air-cramer/1 (Cramer 1993, temperature and humidity); dry-air-core/1 reproduces earlier fits.
    sound_speed_model: Literal["humid-air-cramer/1", "dry-air-core/1"] = Field(
        default="humid-air-cramer/1", exclude_if=lambda v: v == "humid-air-cramer/1")
    late_t60_s: Bands | None = None
    mixing_time_s: float | None = Field(default=None, gt=0.)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    uncertainty: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _physical(self) -> "RoomModel":
        if min(self.shoebox_dims_m) <= 0:
            raise ValueError("shoebox dimensions must be positive")
        for name in ("absorption", "scattering"):
            table = getattr(self, name)
            if set(table) != set(SURFACES):
                raise ValueError(f"{name} must contain exactly {SURFACES}")
            a = np.asarray([table[s] for s in SURFACES])
            if not np.all(np.isfinite(a) & (a >= 0) & (a <= 1)):
                raise ValueError(f"{name} coefficients must lie in [0, 1]")
        if self.late_t60_s is not None and min(self.late_t60_s) <= 0:
            raise ValueError("late_t60_s must be positive")
        if len({b.name for b in self.obstacles}) != len(self.obstacles):
            raise ValueError("obstacle names must be unique")
        for i, box in enumerate(self.obstacles):
            if np.any(box.max_xyz > self.shoebox_dims_m):
                raise ValueError("obstacles must fit inside the room")
            for other in self.obstacles[:i]:
                if np.all(np.minimum(box.max_xyz, other.max_xyz) >=
                          np.maximum(box.min_xyz, other.min_xyz)):
                    raise ValueError("overlapping or touching obstacle boxes are unsupported")
        if self.volume_m3 <= 0:
            raise ValueError("obstacles leave no free room volume")
        lo, hi = CRAMER_RANGE["temperature_c"]
        if (self.reference_sound_speed_m_s is None and self.sound_speed_model == HUMID_SOUND_SPEED_MODEL
                and not lo <= self.temperature_c <= hi):
            raise ValueError(f"{HUMID_SOUND_SPEED_MODEL} covers {lo:g}-{hi:g} C; at {self.temperature_c:g} C set "
                             "sound_speed_model='dry-air-core/1' or reference_sound_speed_m_s")
        # JSON provenance must be portable, including its arbitrary numeric leaves.
        canonical_json([self.provenance, self.uncertainty])
        return self

    @property
    def sound_speed_m_s(self) -> float:
        """sound_speed_model at the room's temperature and humidity, unless a reference speed is pinned."""
        if self.reference_sound_speed_m_s is not None:
            return self.reference_sound_speed_m_s
        return float(sound_speed(self.sound_speed_model, self.temperature_c, self.humidity_pct))

    @property
    def sound_speed_policy(self) -> str:
        return "fixed-reference/1" if self.reference_sound_speed_m_s is not None else self.sound_speed_model

    @property
    def volume_m3(self) -> float:
        """Free acoustic volume, subtracting solid boxes."""
        return float(np.prod(self.shoebox_dims_m) - sum(np.prod(b.size_xyz) for b in self.obstacles))

    @property
    def surface_areas_m2(self) -> np.ndarray:
        x, y, z = self.shoebox_dims_m
        return np.array([y*z, y*z, x*z, x*z, x*y, x*y])

    def diffuse_surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        """Exposed areas and corresponding absorption rows, excluding wall contacts."""
        room_area = self.surface_areas_m2.copy()
        areas, absorption = [], []
        for box in self.obstacles:
            x, y, z = box.size_xyz
            exposed = np.array([y*z, y*z, x*z, x*z, x*y, x*y])
            for axis in range(3):
                for side, contact in enumerate((abs(box.min_xyz[axis]) < 1e-9,
                        abs(box.max_xyz[axis]-self.shoebox_dims_m[axis]) < 1e-9)):
                    j = 2*axis+side
                    if contact:
                        room_area[j] -= exposed[j]
                        exposed[j] = 0.
            areas.append(exposed.sum())
            absorption.append(box.absorption)
        return (np.r_[room_area, areas], np.array([self.absorption[s] for s in SURFACES] + absorption))

    def t60_estimates(self, method: Literal["eyring", "sabine"] = "eyring",
                      *, air_absorption: bool = True,
                      room_absorption: np.ndarray | None = None) -> np.ndarray:
        """Seven per-band seconds, excluding measured overrides; zero loss -> inf.

        T60 = 24 ln(10) V / [c (A + 4 m V)], where m is the ISO energy-loss
        exponent per metre and A is S*alpha (Sabine) or -S*ln(1-alpha) (Eyring).
        """
        return self.decay_budget(method, air_absorption=air_absorption,
                                 room_absorption=room_absorption, measured_override=False).t60_s

    def decay_budget(self, method: Literal["eyring", "sabine"] = "eyring", *,
                     air_absorption: bool = True, room_absorption: np.ndarray | None = None,
                     measured_override: bool = True):
        """Shared per-band physics; measured overrides include air loss already."""
        area, absorption = self.diffuse_surfaces()
        if room_absorption is not None:
            candidate = np.asarray(room_absorption, float)
            if candidate.shape != (6, 7) or not np.all(np.isfinite(candidate) &
                                                     (candidate >= 0) & (candidate <= 1)):
                raise ValueError("room_absorption must be six by seven energy fractions")
            absorption[:6] = candidate
        mean = area @ absorption / area.sum()
        return room_decay_budget(volume_m3=self.volume_m3, surface_area_m2=float(area.sum()),
            mean_surface_loss=mean, frequency_hz=OCTAVE_HZ, sound_speed_m_s=self.sound_speed_m_s,
            method=method, air_absorption=air_absorption, temperature_c=self.temperature_c,
            humidity_pct=self.humidity_pct, measured_t60_s=self.late_t60_s if measured_override else None)

    def late_t60(self) -> np.ndarray:
        return self.decay_budget().t60_s

    def schroeder_frequency_hz(self) -> float:
        """2000 sqrt(T/V) (Schroeder): below it, individual room modes dominate, and the diffuse
        late-tail model and octave-band decay metrics stop describing the room. T is the mean late
        T60 at 500 Hz and 1 kHz, V the free volume."""
        t = self.late_t60()
        return float(2000*np.sqrt(np.mean(t[2:4])/self.volume_m3))

    def modal_region(self) -> dict:
        """Octave bands whose centre lies below the Schroeder frequency (the modal region)."""
        f = self.schroeder_frequency_hz()
        return {"schroeder_frequency_hz": round(f, 1), "bands_below_hz": [b for b in OCTAVE_HZ if b < f],
                "meaning": "below the Schroeder frequency room modes dominate; the image-source early part "
                           "still applies, but the diffuse late tail and band decay metrics are approximations"}

    def mixing_time(self) -> float:
        """Override or max(three mean free paths, room diagonal)/c [s]."""
        if self.mixing_time_s is not None:
            return self.mixing_time_s
        return float(max(12 * self.volume_m3 / self.diffuse_surfaces()[0].sum(),
                         np.linalg.norm(self.shoebox_dims_m)) / self.sound_speed_m_s)

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, value: str) -> "RoomModel":
        return cls.model_validate_json(value)

    @property
    def model_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()
