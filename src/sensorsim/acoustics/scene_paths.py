"""Diffraction routes around one box obstacle, used by the room renderer (`acoustics.room.obstacles`).

`top_route` takes the taut string over the roof in the vertical plane through S and R (0, 1 or 2 roof
edges, upper convex hull); `side_route` takes the 2-D taut string around the footprint via the tangent
corners (1 or 2 vertical edges). Diffraction points are refined in 3-D with the exact Fermat edge point
(`edge_point_v`); two coplanar edges are handled by unfolding with finite-endpoint checks. Both return
the route's first and last diffraction points, its length `ell`, the edge count and whether the direct
segment is blocked; the caller applies the barrier attenuation.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


# ----------------------------------------------------------------------------------- geometry helpers
def edge_point_v(src: np.ndarray, rx: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point version of `Scene.edge_point` (same formula): src, rx, a, b all (N, 3); returns E (N, 3),
    the point of segment a-b minimising |src - E| + |E - rx|."""
    e = b - a
    le = np.linalg.norm(e, axis=-1)
    eh = e / np.maximum(le, _EPS)[:, None]
    ps = np.sum((src - a) * eh, axis=-1)
    pr = np.sum((rx - a) * eh, axis=-1)
    ds = np.linalg.norm((src - a) - ps[:, None] * eh, axis=-1)
    dr = np.linalg.norm((rx - a) - pr[:, None] * eh, axis=-1)
    tot = ds + dr
    w = np.where(tot > _EPS, ds / np.maximum(tot, _EPS), 0.5)
    par = ps + (pr - ps) * w
    s = np.clip(par / np.maximum(le, _EPS), 0.0, 1.0)
    return a + s[:, None] * e


def _two_edge_points(src, rx, a1, b1, a2, b2):
    """Shortest path through two distinct coplanar box edges, each (N, 3).

    Unfold each endpoint into the edge plane, on the far side of its edge.
    Distances to every point on that edge are unchanged. A straight segment
    crossing both finite edges is the interior Fermat solution. Otherwise the
    convex length minimum is on a segment endpoint; check all four boundaries
    using the exact single-edge solution. This avoids slowly converging
    alternating projections (and their nonreciprocal travel phase).
    """
    e1, e2 = b1-a1, b2-a2
    l1, l2 = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
    u1, u2 = e1/l1[:, None], e2/l2[:, None]
    normal = np.cross(u1, u2)
    parallel = np.linalg.norm(normal, axis=1) < _EPS
    normal[parallel] = np.cross(u1[parallel], (a2-a1)[parallel])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    into1, into2 = np.cross(normal, u1), np.cross(normal, u2)
    between = (a2+b2-a1-b1)/2
    into1 *= np.where(np.sum(into1*between, axis=1) >= 0, 1., -1.)[:, None]
    into2 *= np.where(np.sum(into2*between, axis=1) <= 0, 1., -1.)[:, None]
    ps, pr = np.sum((src-a1)*u1, axis=1), np.sum((rx-a2)*u2, axis=1)
    ds = np.linalg.norm(src-a1-ps[:, None]*u1, axis=1)
    dr = np.linalg.norm(rx-a2-pr[:, None]*u2, axis=1)
    s = a1+ps[:, None]*u1-ds[:, None]*into1
    r = a2+pr[:, None]*u2-dr[:, None]*into2
    direction = r-s
    den1, den2 = np.sum(direction*into1, axis=1), np.sum(direction*into2, axis=1)
    t1 = np.divide(ds, den1, out=np.zeros_like(ds), where=abs(den1) > _EPS)
    t2 = np.divide(np.sum((a2-s)*into2, axis=1), den2,
                   out=np.zeros_like(dr), where=abs(den2) > _EPS)
    first, last = s+t1[:, None]*direction, s+t2[:, None]*direction
    q1, q2 = np.sum((first-a1)*u1, axis=1), np.sum((last-a2)*u2, axis=1)
    interior = ((abs(den1) > _EPS) & (abs(den2) > _EPS) & (0 <= t1) & (t1 <= t2)
                & (t2 <= 1) & (0 <= q1) & (q1 <= l1) & (0 <= q2) & (q2 <= l2))
    # Project onto the original segments to remove only roundoff normal to them.
    first, last = a1+q1[:, None]*u1, a2+q2[:, None]*u2
    pairs = [(first, last)]
    pairs += [(end, edge_point_v(end, rx, a2, b2)) for end in (a1, b1)]
    pairs += [(edge_point_v(src, end, a1, b1), end) for end in (a2, b2)]
    lengths = np.stack([np.linalg.norm(src-v1, axis=1)+np.linalg.norm(v2-v1, axis=1)
                        + np.linalg.norm(rx-v2, axis=1) for v1, v2 in pairs])
    lengths[0, ~interior] = np.inf
    choice, rows = np.argmin(lengths, axis=0), np.arange(len(src))
    return np.stack([p[0] for p in pairs])[choice, rows], np.stack([p[1] for p in pairs])[choice, rows]


def _norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def _slab2d(s2: np.ndarray, r2: np.ndarray, hx: float, hy: float):
    """2-D slab test of segments s2 -> r2 against [-hx, hx] x [-hy, hy] (local footprint). Returns
    t_near, t_far, face_near, face_far (local side index 0..3, -1 when that end is inside), crossing."""
    d = r2 - s2
    n = s2.shape[0]
    t_near = np.full(n, -np.inf)
    t_far = np.full(n, np.inf)
    f_near = np.full(n, -1, dtype=np.int64)
    f_far = np.full(n, -1, dtype=np.int64)
    miss = np.zeros(n, dtype=bool)
    lo, hi = (-hx, -hy), (hx, hy)
    low_face, high_face = (0, 2), (1, 3)
    for ax in range(2):
        da = d[:, ax]
        par = np.abs(da) < _EPS
        miss |= par & ((s2[:, ax] < lo[ax]) | (s2[:, ax] > hi[ax]))
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo[ax] - s2[:, ax]) / da
            t2 = (hi[ax] - s2[:, ax]) / da
        pos = da > 0
        tn = np.where(par, -np.inf, np.where(pos, t1, t2))
        tf = np.where(par, np.inf, np.where(pos, t2, t1))
        fn = np.where(pos, low_face[ax], high_face[ax])
        ff = np.where(pos, high_face[ax], low_face[ax])
        un = tn > t_near
        t_near = np.where(un, tn, t_near)
        f_near = np.where(un, fn, f_near)
        uf = tf < t_far
        t_far = np.where(uf, tf, t_far)
        f_far = np.where(uf, ff, f_far)
    cross = ~miss & (t_near < t_far) & (t_far > 0.0) & (t_near < 1.0)
    f_near = np.where(t_near <= 0.0, -1, f_near)
    f_far = np.where(t_far >= 1.0, -1, f_far)
    return t_near, t_far, f_near, f_far, cross


def _box_frame(box):
    hx, hy, hz = box.half
    h = 2.0 * hz
    roof = {0: ((-hx, -hy, h), (-hx, hy, h)), 1: ((hx, -hy, h), (hx, hy, h)),
            2: ((-hx, -hy, h), (hx, -hy, h)), 3: ((-hx, hy, h), (hx, hy, h))}
    roof_a = np.array([roof[k][0] for k in range(4)])
    roof_b = np.array([roof[k][1] for k in range(4)])
    corners = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])      # CCW seen from above
    return hx, hy, h, roof_a, roof_b, corners


def _inside_local(q: np.ndarray, hx, hy, h) -> np.ndarray:
    return (np.abs(q[:, 0]) < hx) & (np.abs(q[:, 1]) < hy) & (q[:, 2] > 0.0) & (q[:, 2] < h)


def top_route(box, S: np.ndarray, R: np.ndarray) -> dict:
    """Over-the-roof route of box for segments S -> R (N, 3) (module docstring)."""
    s, r = box.to_local(S), box.to_local(R)
    hx, hy, h, roof_a, roof_b, _ = _box_frame(box)
    tn, tf, fn, ff, cross = _slab2d(s[:, :2], r[:, :2], hx, hy)
    inside = _inside_local(s, hx, hy, h) | _inside_local(r, hx, hy, h)
    valid = cross & ~inside
    a_ex = valid & (tn > 0.0)
    b_ex = valid & (tf < 1.0)
    dz = r[:, 2] - s[:, 2]
    za = s[:, 2] + np.where(a_ex, tn, 0.0) * dz
    zb = s[:, 2] + np.where(b_ex, tf, 1.0) * dz
    blk_a, blk_b = a_ex & (za < h), b_ex & (zb < h)
    blocked = blk_a | blk_b
    hull_a = blk_a & (~b_ex | (s[:, 2] < h))
    hull_b = blk_b & (~a_ex | (r[:, 2] < h))
    fa, fb = np.clip(fn, 0, 3), np.clip(ff, 0, 3)
    ea = edge_point_v(s, r, roof_a[fa], roof_b[fa])
    eb = edge_point_v(s, r, roof_a[fb], roof_b[fb])
    dbl = hull_a & hull_b
    if dbl.any():
        i = np.nonzero(dbl)[0]
        e1, e2 = _two_edge_points(s[i], r[i], roof_a[fa[i]], roof_b[fa[i]],
                                  roof_a[fb[i]], roof_b[fb[i]])
        ea[i], eb[i] = e1, e2
    v1 = np.where(hull_a[:, None], ea, np.where(hull_b[:, None], eb, r))
    last = np.where(hull_b[:, None], eb, np.where(hull_a[:, None], ea, s))
    mid = np.where(dbl, _norm(eb - ea), 0.0)
    ell = _norm(v1 - s) + mid + np.where(hull_a | hull_b, _norm(r - last), 0.0)
    return {"valid": valid, "blocked": blocked, "V1": box.to_world(v1), "last": box.to_world(last),
            "ell": ell, "n_edges": hull_a.astype(np.int64) + hull_b, "e_sep": mid}


def side_route(box, S: np.ndarray, R: np.ndarray, side: int) -> dict:
    """Route around the footprint on the left (side=+1) or right (-1) of S -> R (module docstring)."""
    s, r = box.to_local(S), box.to_local(R)
    hx, hy, h, _, _, corners = _box_frame(box)
    tn, tf, fn, ff, cross = _slab2d(s[:, :2], r[:, :2], hx, hy)
    s_in2 = (np.abs(s[:, 0]) < hx) & (np.abs(s[:, 1]) < hy)
    r_in2 = (np.abs(r[:, 0]) < hx) & (np.abs(r[:, 1]) < hy)
    hit, t_in, t_out, _, _ = box.segment_slab(S, R)
    blocked = hit & (t_out - t_in > 1e-9)
    d = r[:, :2] - s[:, :2]
    vs = corners[None, :, :] - s[:, None, :2]
    vr = corners[None, :, :] - r[:, None, :2]
    ang_s = np.arctan2(d[:, None, 0] * vs[..., 1] - d[:, None, 1] * vs[..., 0], np.sum(d[:, None, :] * vs, -1))
    ang_r = np.arctan2(-d[:, None, 0] * vr[..., 1] + d[:, None, 1] * vr[..., 0], -np.sum(d[:, None, :] * vr, -1))
    if side > 0:
        ts, tr = np.argmax(ang_s, axis=1), np.argmin(ang_r, axis=1)
    else:
        ts, tr = np.argmin(ang_s, axis=1), np.argmax(ang_r, axis=1)
    diff = (tr - ts) % 4
    single = diff == 0
    adj = (diff == 1) | (diff == 3)
    valid = cross & ~s_in2 & ~r_in2 & (single | adj)
    va = np.column_stack([corners, np.zeros(4)])
    vb = np.column_stack([corners, np.full(4, h)])
    e1 = edge_point_v(s, r, va[ts], vb[ts])
    e2 = e1.copy()
    dbl = valid & adj
    if dbl.any():
        i = np.nonzero(dbl)[0]
        f1, f2 = _two_edge_points(s[i], r[i], va[ts[i]], vb[ts[i]], va[tr[i]], vb[tr[i]])
        e1[i], e2[i] = f1, f2
    mid = np.where(dbl, _norm(e2 - e1), 0.0)
    ell = _norm(e1 - s) + mid + _norm(r - e2)
    return {"valid": valid, "blocked": blocked & valid, "V1": box.to_world(e1), "last": box.to_world(e2),
            "ell": ell, "n_edges": 1 + dbl.astype(np.int64), "e_sep": mid}
