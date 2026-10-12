"""Barrier (edge-diffraction) attenuation, fast tier: Maekawa's chart in the Kurze-Anderson closed form,
with ISO 9613-2's thick-barrier factor C3 and attenuation caps.

Fresnel number of a diffracted path: N = 2 delta / lambda, delta = (path via the edge(s)) - (direct
distance) >= 0 in the shadow; in the lit zone near the shadow boundary N = -2 delta / lambda with delta the
excess of the path via the nearest edge.

Kurze & Anderson (1971), "Sound attenuation by barriers", Applied Acoustics 4:35-53 (closed-form fit of
Maekawa 1968, Applied Acoustics 1:157, for a thin semi-infinite screen, point source):
    A(N) = 5 + 20 lg( sqrt(2 pi N) / tanh(sqrt(2 pi N)) )          N > 0
    A(0) = 5
    A(N) = 5 + 20 lg( sqrt(2 pi |N|) / tan(sqrt(2 pi |N|)) )       -0.1926 < N < 0  (A = 0 at N = -0.1926)
    A(N) = 0                                                        N <= -0.1926
A is the attenuation relative to the free field at the DIRECT distance d (so a path of length L carrying it
needs the gain (L/d) 10^(-A/20) on top of its own 1/L spreading).

ISO 9613-2:1996 §7.4 eq. (14): D_z = 10 lg(3 + (C2/lambda) C3 z K_met); for double diffraction (two edges
e apart) C3 = [1 + (5 lambda/e)^2] / [1/3 + (5 lambda/e)^2], else C3 = 1; D_z is limited to 20 dB (single)
and 25 dB (double diffraction). Here: A_route = min(A_KA(C3 N), cap) (C2 = 40 = pure Maekawa, because the
receiver-side ground reflection is modelled explicitly; K_met = 1).

VALIDITY: see VALIDITY_BARRIER.
"""
from __future__ import annotations

import math

import numpy as np

from sensorsim.core.messages import ModelValidity
from sensorsim.acoustics.gain_budget import GainBudget, path_length_compensation as path_length_compensation

GAIN_BUDGET = GainBudget("kurze-anderson-iso-cap/1", "pressure", "magnitude", 1., False, False)

N_LIT_ZERO = -0.1926          # K-A lit-zone attenuation reaches 0 dB here
CAP_SINGLE_DB = 20.0          # ISO 9613-2 §7.4
CAP_DOUBLE_DB = 25.0

VALIDITY_BARRIER = ModelValidity(
    name="Edge diffraction (Maekawa / Kurze-Anderson, ISO 9613-2 C3 and caps) over and around scene boxes",
    status="partially validated",
    freq_range_hz=(50.0, 10000.0),
    range_m=(1.0, 1000.0),
    conditions=("Engineering barrier model: magnitude only (zero-phase filter) on a path through the exact "
                "Fermat edge point(s), so delays/Doppler are geometric. Routes per box: over the roof (1-2 roof "
                "edges, vertical-plane topology refined in 3-D) and around each side (1-2 vertical edges); "
                "3-corner detours and multi-object diffraction chains are not modelled; a route whose legs are "
                "blocked by another object is dropped. Source-side ground reflection of diffracted paths is not "
                "modelled (ISO's C2 = 20 would add up to +3 dB). Lit-zone (N < 0) attenuation uses the "
                "thin-screen form for every edge (grazing along a building face may be under-predicted by up to "
                "5 dB). Implementation checked against the K-A closed form (thin wall in the node: top route "
                "within 0.6 dB, all routes within 0.9 dB of the coherent K-A route sum). Model error vs the exact "
                "rigid half-plane (plane wave, scene_check barrier_exact): K-A under-predicts the attenuation by "
                "1.9 dB on average for N < 1 (range -3.0..+0.2 dB), -0.7 dB for 1 < N < 5, +0.2 dB (-1.9..+1.4) "
                "for N > 5."),
    citation="Kurze & Anderson 1971 Appl. Acoust. 4:35; Maekawa 1968 Appl. Acoust. 1:157; ISO 9613-2:1996 §7.4.",
)


def kurze_anderson_db(n) -> np.ndarray:
    """Attenuation [dB] vs signed Fresnel number N (module docstring)."""
    n = np.asarray(n, dtype=np.float64)
    if not np.isfinite(n).all():
        raise ValueError("finite signed Fresnel numbers required")
    out = np.zeros(n.shape)
    pos = n > 1e-12
    x = np.sqrt(2.0 * math.pi * np.where(pos, n, 1.0))
    out = np.where(pos, 5.0 + 20.0 * np.log10(x / np.tanh(x)), out)
    lit = (n < -1e-12) & (n > N_LIT_ZERO)
    y = np.sqrt(2.0 * math.pi * np.where(lit, -n, 0.01))
    out = np.where(lit, np.maximum(0.0, 5.0 + 20.0 * np.log10(y / np.tan(y))), out)
    out = np.where(np.abs(n) <= 1e-12, 5.0, out)
    return out


def iso_c3(lam, e) -> np.ndarray:
    """ISO 9613-2 eq. (14) C3 for edge separation e [m] (e <= 0 -> 1)."""
    lam = np.asarray(lam, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    q = (5.0 * lam / np.maximum(e, 1e-9)) ** 2
    return np.where(e > 1e-9, (1.0 + q) / (1.0 / 3.0 + q), 1.0)


def _wavelength(f_hz, c):
    """Physical inputs; retain the legacy 1 Hz DC/sub-Hz engineering extension."""
    f = np.asarray(f_hz, dtype=np.float64)
    if not np.isfinite(f).all() or np.any(f < 0) or not np.isfinite(c) or c <= 0:
        raise ValueError("finite nonnegative frequencies and positive sound speed required")
    return c / np.maximum(f, 1.0)[None, :]


def route_attenuation_db(delta, f_hz, c: float, e_sep, n_edges) -> np.ndarray:
    """Shadow-zone attenuation of a diffracted route: delta (P,), e_sep (P,), n_edges (P,) -> (P, F)."""
    lam = _wavelength(f_hz, c)
    delta = np.asarray(delta, dtype=np.float64)[:, None]
    e = np.asarray(e_sep, dtype=np.float64)[:, None]
    n_fr = 2.0 * delta / lam
    c3 = np.where(np.asarray(n_edges)[:, None] >= 2, iso_c3(lam, e), 1.0)
    a = kurze_anderson_db(c3 * n_fr)
    cap = np.where(np.asarray(n_edges)[:, None] >= 2, CAP_DOUBLE_DB, CAP_SINGLE_DB)
    return np.minimum(a, cap)


# ------------------------------------------------------------------------------------------ accurate tier


