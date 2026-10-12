"""Band definitions only: diagnostic filters and measured-room metrics stay distinct (D05).

The binary meter deliberately uses nominal centres, including 63 Hz. The offline
PSD integrator uses exact base-10 centres. Equal labels do not imply equal bands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

def renard_nominal_hz(index: int) -> float:
    """R10 preferred label at index relative to 1000 Hz; not an exact centre.

    NBS Technical Note 990 (H. J. Milton, 1978) credits Renard's 1877--1879
    work with R5 (§2.5.5, p. 29). R10's intermediate values are listed in
    Table 7 (§2.5.6, p. 31) and §2.5.12, p. 37:
    https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote990.pdf
    This project rounds 10**(k/10) to twentieths at k=1,5, a whole unit at k=9,
    and tenths elsewhere to reproduce R10; powers of ten scale those labels.
    Exact base-10 centres are 1000 * 10**(index/10), not these labels.
    """
    decade, k = divmod(index + 30, 10)
    units = 20 if k in (1, 5) else 1 if k == 9 else 10
    # Scale the integer numerator first to avoid 6.3*10 becoming 63.00000000000001.
    return float(round(10.**(k/10)*units) * 10.**decade / units)


OCTAVE_NOMINAL_HZ = tuple(renard_nominal_hz(n) for n in range(-12, 10, 3))
OCTAVE_NOMINAL_EXTENDED_HZ = tuple(renard_nominal_hz(n) for n in range(-15, 13, 3))
THIRD_NOMINAL_HZ = tuple(renard_nominal_hz(n) for n in range(-17, 14))
THIRD_NOMINAL_100_3150_HZ = tuple(f for f in THIRD_NOMINAL_HZ if 100 <= f <= 3150)
OCTAVE_NOMINAL_125_8000_HZ = tuple(int(f) for f in OCTAVE_NOMINAL_HZ if f >= 125)
# Exact binary rendering grids remain distinct from nominal 63 Hz meter labels
# and from the exact base-10 measurement grid. Preserve historical arithmetic.
OCTAVE_BINARY_EXACT_HZ = tuple(1000. * 2. ** n for n in range(-5, 5))
THIRD_BINARY_PROPAGATION_HZ = tuple(float(1000. * 2. ** (n / 3.)) for n in range(-13, 7))
# Historical density proxies, subsequently normalized in A-weighted power.
# They do not define integration edges; retaining these literals preserves FIR bytes.
LEGACY_OCTAVE_PSD_WIDTH = .707
LEGACY_INDOOR_OCTAVE_PSD_WIDTH = .7071
LEGACY_THIRD_PSD_WIDTH = .2316


def _edge_ratio(base: str, fraction: int) -> float:
    if base not in ("base2", "base10") or fraction < 1 or fraction != int(fraction):
        raise ValueError("named base and positive integer fraction required")
    return 2. ** (1. / (2 * fraction)) if base == "base2" else 10. ** (3. / (20 * fraction))


BINARY_OCTAVE_EDGE_RATIO = _edge_ratio("base2", 1)
# Keep the diagnostic white-band arithmetic order (one-ulp difference from hi-lo).
BINARY_OCTAVE_WIDTH = BINARY_OCTAVE_EDGE_RATIO - 2. ** -.5
BASE10_OCTAVE_EDGE_FACTORS = (1. / _edge_ratio("base10", 1), _edge_ratio("base10", 1))
BASE10_THIRD_EDGE_FACTORS = (1. / _edge_ratio("base10", 3), _edge_ratio("base10", 3))


def fractional_octave_edges(center_hz, *, base: Literal["base2", "base10"], fraction: int = 1):
    """Edges about the supplied actual centre; never convert a nominal label silently."""
    ratio = _edge_ratio(base, fraction)
    center = np.asarray(center_hz)
    if np.any(~np.isfinite(center)) or np.any(center <= 0):
        raise ValueError("positive finite band centres required")
    return center / ratio, center * ratio


def room_octave_center(band: int) -> float:
    """IEC base-10 exact centre for the room measurement labels 125..8000 Hz."""
    if band not in OCTAVE_NOMINAL_125_8000_HZ:
        raise ValueError("band must be a nominal octave 125..8000 Hz")
    return float(1000 * 10 ** (.3 * (OCTAVE_NOMINAL_125_8000_HZ.index(band) - 3)))


def binary_third_octave_midpoints(bands_hz, sub: int) -> np.ndarray:
    """Legacy fixed log-midpoint quadrature for frozen reference adapters.

    This sampling rule alone supplies no band-integration accuracy guarantee;
    runtime propagation uses phase-resolved band_integration.band_quadrature.
    """
    b = np.asarray(bands_hz, dtype=np.float64)
    off = (np.arange(sub) - (sub - 1) / 2.) / (3. * sub)
    return b[:, None] * 2. ** off[None, :]


@dataclass(frozen=True)
class BandSpec:
    base: Literal["base2", "base10"]
    fraction: int
    exact_center_hz: float
    nominal_hz: float

    def __post_init__(self):
        if (self.base not in ("base2", "base10") or self.fraction < 1 or self.fraction != int(self.fraction)
                or not np.isfinite([self.exact_center_hz, self.nominal_hz]).all()
                or min(self.exact_center_hz, self.nominal_hz) <= 0):
            raise ValueError("positive finite band centres, integer fraction and named base required")

    @property
    def edges_hz(self) -> tuple[float, float]:
        lo, hi = fractional_octave_edges(self.exact_center_hz, base=self.base, fraction=self.fraction)
        return float(lo), float(hi)

    def as_dict(self) -> dict:
        lo, hi = self.edges_hz
        return {**asdict(self), "lower_hz": lo, "upper_hz": hi}


@dataclass(frozen=True)
class BandGrid:
    bands: tuple[BandSpec, ...]
    method: str
    input_quantity: str
    filter_order: int | None = None

    def as_dict(self) -> dict:
        return {"method": self.method, "input_quantity": self.input_quantity,
                "filter_order": self.filter_order, "bands": [band.as_dict() for band in self.bands]}

    def require_same_grid(self, other: BandGrid) -> None:
        """Reject a comparison through different bands/methods instead of aligning labels."""
        if self != other:
            raise ValueError("band grids or metric policies differ")


BINARY_OCTAVE_BANDS = tuple(BandSpec("base2", 1, f, f) for f in OCTAVE_NOMINAL_HZ)
THIRD_OCTAVE_BANDS = tuple(BandSpec("base10", 3, float(f), label)
                          for f, label in zip(10. ** (np.arange(13, 44) / 10.), THIRD_NOMINAL_HZ))


def third_octave_edges(fc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base-10 third-octave edges, retaining the offline integrator's arithmetic."""
    return fractional_octave_edges(fc, base="base10", fraction=3)
