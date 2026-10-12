"""Rigid first-order edge scattering, with an analytic shadow-boundary limit.

Svensson, Fred & Vanderkooy (1999), JASA 106:2331, doi:10.1121/1.428071,
eqs. (2), (3), (18), (21); pressure convention exp(-ikr)/r.

This implementation integrates in the *signed hyperbolic angle*, not in edge
position. It subtracts each angular pole analytically before quadrature. The
independent benchmark integrates secondary sources in physical edge position.
Finite edge sums are first order; they are not a full-wave finite-body solution.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

from sensorsim.core.messages import ModelValidity
from .bands import BINARY_OCTAVE_EDGE_RATIO

VALIDITY = ModelValidity(
    name="Rigid BTM edge integral with analytic pole subtraction",
    status="partially validated", freq_range_hz=(125., 8000.),
    conditions="Point monopoles, straight rigid convex edges. Finite-edge first order; "
    "no multiple scattering or impedance wedge. Room API reduces coherent edge sums "
    "to octave energy and earliest delay; phase is not retained between path groups.",
    citation="Svensson, Fred & Vanderkooy 1999, doi:10.1121/1.428071",
)


@lru_cache(maxsize=1)
def _rule():
    return roots_legendre(12)


def edge_geometry(source, receiver, origin, axis):
    """Cylindrical distances and stationary finite-edge coordinate."""
    s, r = np.asarray(source)-origin, np.asarray(receiver)-origin
    zs, zr = s @ axis, r @ axis
    rs, rr = np.linalg.norm(s-zs*axis), np.linalg.norm(r-zr*axis)
    if min(rs, rr) <= 1e-12:
        raise ValueError("source and receiver must be off the diffracting edge")
    return rs, rr, zs, zr, (rr*zs+rs*zr)/(rs+rr)


def edge_response(frequency_hz, source, receiver, origin, axis, face, normal,
                  bounds, *, opening=2*np.pi, sound_speed=343., refinement=1):
    """Complex finite-edge pressure and shortest edge-path length.

    ``face`` points into the material along theta=0, ``normal`` into air.
    ``axis`` runs along the edge. Opening is the exterior angle, pi..2*pi.
    At an exact optical boundary the pole is assigned its symmetric (zero)
    limit; the caller must assign half of the corresponding geometrical ray.
    This is a distributional limiting prescription, not a denominator floor.
    """
    f = np.asarray(frequency_hz, float)
    origin, axis, face, normal = (np.asarray(x, float) for x in (origin, axis, face, normal))
    basis = np.stack((axis, face, normal))
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-10):
        raise ValueError("edge frame must be orthonormal")
    if (not np.isfinite(f).all() or np.any(f < 0) or sound_speed <= 0
            or not np.pi <= opening <= 2*np.pi or bounds[1] <= bounds[0]):
        raise ValueError("invalid frequency, wedge, sound speed or edge bounds")
    rs, rr, zs, zr, apex = edge_geometry(source, receiver, origin, axis)
    s, r = np.asarray(source)-origin, np.asarray(receiver)-origin
    ts, tr = (np.arctan2(x @ normal, x @ face) % (2*np.pi) for x in (s, r))
    ts, tr = (0. if abs(t-2*np.pi) < 1e-12 else t for t in (ts, tr))
    if max(ts, tr) > opening+1e-10:
        raise ValueError("source or receiver inside solid wedge")
    nu = np.pi/opening
    angles = nu*(np.pi+np.array([ts+tr, ts-tr, -ts+tr, -ts-tr]))
    # u=asinh((z-zs)/rs)+asinh((z-zr)/rr), du/dz=1/m+1/l.
    # Thus dz/(m*l)=du/L and L^2=(rs+rr)^2+(zs-zr)^2+4*rs*rr*sinh(u/2)^2.
    u = np.arcsinh((np.asarray(bounds)-zs)/rs)+np.arcsinh((np.asarray(bounds)-zr)/rr)
    base = np.hypot(rs+rr, zs-zr)
    split = np.unique(np.r_[u, np.clip(0., *u)])
    nodes, weights = _rule()
    positions, quadrature = [], []
    for low, high in zip(split[:-1], split[1:]):
        # Resolve the steepest phase slope in this interval with 12-node panels.
        end = max(abs(low), abs(high))
        slope = rs*rr*np.sinh(end)/np.sqrt(base*base+4*rs*rr*np.sinh(end/2)**2)
        panels = max(1, int(np.ceil((high-low)*slope*max(float(f.max()), 1.)/
                                    sound_speed/2*refinement)))
        step = (high-low)/panels
        positions.append((low+(np.arange(panels)[:, None]+.5)*step+nodes*step/2).ravel())
        quadrature.append(np.tile(weights*step/2, panels))
    x, w = np.concatenate(positions), np.concatenate(quadrature)
    lengths = np.sqrt(base*base+4*rs*rr*np.sinh(x/2)**2)
    beta, mass = np.zeros_like(x), 0.
    for angle in angles:
        sine, half = np.sin(angle), np.sin(angle/2)
        if abs(half) < 1e-12:
            continue  # symmetric boundary value; no clipped singularity
        beta += sine/(2*(np.sinh(nu*x/2)**2+half*half))
        primitive = 2/nu*np.arctan(np.tanh(nu*u/2)/np.tan(angle/2))
        mass += primitive[1]-primitive[0]
    out = np.empty(f.size, complex)
    for i in range(0, f.size, 32):
        k = (2*np.pi/sound_speed*f.ravel()[i:i+32])[:, None]
        # expm1 avoids loss of the regular remainder near the subtracted pole.
        regular = (np.expm1(-1j*k*(lengths-base))/lengths
                   + (1/lengths-1/base)) @ (w*beta)
        out[i:i+32] = -nu/(4*np.pi)*np.exp(-1j*k[:, 0]*base)*(mass/base+regular)
    z = np.clip(apex, *bounds)
    earliest = np.hypot(rs, z-zs)+np.hypot(rr, z-zr)
    return out.reshape(f.shape), float(earliest)


def octave_grid(centres, samples=33):
    """Uniform frequency, including both octave edges (energy, not mean dB)."""
    return np.asarray(centres)[:, None]*np.linspace(1/BINARY_OCTAVE_EDGE_RATIO, BINARY_OCTAVE_EDGE_RATIO, samples)


def octave_amplitude(response, frequencies):
    return np.sqrt(np.trapezoid(abs(response)**2, frequencies, axis=-1)/
                   (frequencies[:, -1]-frequencies[:, 0]))


def rectangular_response(frequency_hz, source, receiver, face, *, sound_speed=343.):
    """Isolated rigid rectangular face: visible image plus four thin-plate edges.

    This face approximation neglects the adjoining faces of a thick box. The
    thin-plate limit has independent analytic and measured BRAS comparisons.
    """
    source, receiver = np.asarray(source), np.asarray(receiver)
    ds, dr = (source-face.center) @ face.normal, (receiver-face.center) @ face.normal
    if min(ds, dr) <= 0:
        raise ValueError("reflection requires both points on the outward side")
    image = face.mirror(source)
    point = receiver+dr/(ds+dr)*(image-receiver)
    uv = face.uv(point)
    margins = np.array([face.half_u, face.half_v])-abs(uv)
    weight = np.prod(np.where(abs(margins) < 1e-10, .5, margins > 0))
    length = np.linalg.norm(image-receiver)
    response = weight*np.exp(-2j*np.pi*np.asarray(frequency_hz)*length/sound_speed)/length
    arrivals = [length] if weight else []
    for across, along, half_across, half_along in (
            (face.u, face.v, face.half_u, face.half_v),
            (face.v, face.u, face.half_v, face.half_u)):
        for sign in (-1, 1):
            origin = face.center+sign*half_across*across
            value, arrival = edge_response(frequency_hz, source, receiver, origin,
                along, -sign*across, face.normal, (-half_along, half_along),
                sound_speed=sound_speed)
            response += value
            arrivals.append(arrival)
    return response, min(arrivals), point, float(weight)
