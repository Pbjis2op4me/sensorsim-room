"""Oriented boxes on a flat ground plane, and the geometric queries propagation models need.

Conventions (see core.schema.SceneBox):
- Box-local frame: origin at the centre of the bottom face; x_l along `heading_deg` (compass; the box
  "length"), y_l to the left of x_l, z_l up (right-handed). World ENU = R @ local + base_center.
- Faces: front (+x_l), back (-x_l), left (+y_l), right (-y_l), roof (+z_l), floor (z_l = 0, normal down).
- Face-local (u, v) coordinates, relative to the face centre: side faces: u horizontal and to the RIGHT as
  seen from outside, v up; roof: u = +x_l, v = +y_l; floor: u = +x_l, v = -y_l.

All queries are vectorised over N segments/points (arrays (..., 3)) and loop only over boxes and faces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sensorsim.core.schema import SceneBox, SceneOpening

FACES = ("front", "back", "left", "right", "roof", "floor")
_EPS = 1e-9


@dataclass(frozen=True)
class Face:
    """One rectangular face in world coordinates."""
    name: str
    center: np.ndarray      # (3,)
    normal: np.ndarray      # (3,) outward unit normal
    u: np.ndarray           # (3,) unit face axis
    v: np.ndarray           # (3,) unit face axis
    half_u: float
    half_v: float
    material: str
    openings: tuple[SceneOpening, ...] = ()

    @property
    def area(self) -> float:
        return 4.0 * self.half_u * self.half_v

    def uv(self, p: np.ndarray) -> np.ndarray:
        """(..., 2) face-local coordinates of points (assumed on or near the face plane)."""
        d = np.asarray(p, dtype=np.float64) - self.center
        return np.stack([d @ self.u, d @ self.v], axis=-1)

    def contains_uv(self, uv: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        return (np.abs(uv[..., 0]) <= self.half_u + tol) & (np.abs(uv[..., 1]) <= self.half_v + tol)

    def opening_index(self, p: np.ndarray) -> np.ndarray:
        """(...,) index of the opening containing each point on the face, -1 if none (first match wins)."""
        uv = self.uv(p)
        out = np.full(uv.shape[:-1], -1, dtype=np.int64)
        for k in range(len(self.openings) - 1, -1, -1):
            o = self.openings[k]
            du = np.abs(uv[..., 0] - o.center_uv_m[0]) <= 0.5 * o.size_uv_m[0]
            dv = np.abs(uv[..., 1] - o.center_uv_m[1]) <= 0.5 * o.size_uv_m[1]
            out = np.where(du & dv, k, out)
        return out

    def mirror(self, p: np.ndarray) -> np.ndarray:
        """Mirror image of points across the face plane."""
        p = np.asarray(p, dtype=np.float64)
        d = (p - self.center) @ self.normal
        return p - 2.0 * d[..., None] * self.normal

    def corners(self) -> np.ndarray:
        """(4, 3) corners, counter-clockwise seen from outside."""
        cu, cv = self.half_u * self.u, self.half_v * self.v
        return np.stack([self.center - cu - cv, self.center + cu - cv, self.center + cu + cv, self.center - cu + cv])


@dataclass
class Box:
    """An oriented box built from a `SceneBox` spec."""
    spec: SceneBox
    R: np.ndarray = field(init=False)          # (3, 3) local -> world rotation (columns x_l, y_l, z_l)
    origin: np.ndarray = field(init=False)     # (3,) base centre
    half: np.ndarray = field(init=False)       # (3,) (L/2, W/2, H/2)
    faces: dict[str, Face] = field(init=False)

    def __post_init__(self):
        h = np.deg2rad(self.spec.heading_deg)
        x = np.array([np.sin(h), np.cos(h), 0.0])
        y = np.array([-np.cos(h), np.sin(h), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        self.R = np.stack([x, y, z], axis=1)
        self.origin = np.asarray(self.spec.base_center_enu_m, dtype=np.float64)
        L, W, H = (float(v) for v in self.spec.size_m)
        self.half = np.array([L / 2, W / 2, H / 2])
        c = self.origin + z * (H / 2)
        mats = {f: self.spec.face_materials.get(f, self.spec.material) for f in FACES}
        ops = {f: tuple(o for o in self.spec.openings if o.face == f) for f in FACES}
        # (name, outward normal, u, v, half_u, half_v, centre offset)
        defs = {
            "front": (x, y, z, W / 2, H / 2, x * L / 2),
            "back": (-x, -y, z, W / 2, H / 2, -x * L / 2),
            "left": (y, -x, z, L / 2, H / 2, y * W / 2),
            "right": (-y, x, z, L / 2, H / 2, -y * W / 2),
            "roof": (z, x, y, L / 2, W / 2, z * H / 2),
            "floor": (-z, x, -y, L / 2, W / 2, -z * H / 2),
        }
        self.faces = {name: Face(name, c + off, n, u, v, hu, hv, mats[name], ops[name])
                      for name, (n, u, v, hu, hv, off) in defs.items()}

    @property
    def id(self) -> str:
        return self.spec.id


    def to_local(self, p: np.ndarray) -> np.ndarray:
        """World ENU -> box-local, with z_l measured from the bottom face."""
        return (np.asarray(p, dtype=np.float64) - self.origin) @ self.R

    def to_world(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float64) @ self.R.T + self.origin

    def contains(self, p: np.ndarray, tol: float = 0.0) -> np.ndarray:
        q = self.to_local(p)
        return ((np.abs(q[..., 0]) < self.half[0] - tol) & (np.abs(q[..., 1]) < self.half[1] - tol)
                & (q[..., 2] > tol) & (q[..., 2] < 2 * self.half[2] - tol))

    def edges(self) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
        """12 edges as (a, b, face1, face2) in world coordinates."""
        L, W, H = self.half[0], self.half[1], 2 * self.half[2]
        xs, ys, zs = (-L, L), (-W, W), (0.0, H)
        fx = {-L: "back", L: "front"}
        fy = {-W: "right", W: "left"}
        fz = {0.0: "floor", H: "roof"}
        out = []
        for y in ys:
            for z in zs:
                out.append((self.to_world(np.array([-L, y, z])), self.to_world(np.array([L, y, z])), fy[y], fz[z]))
        for x in xs:
            for z in zs:
                out.append((self.to_world(np.array([x, -W, z])), self.to_world(np.array([x, W, z])), fx[x], fz[z]))
        for x in xs:
            for y in ys:
                out.append((self.to_world(np.array([x, y, 0.0])), self.to_world(np.array([x, y, H])), fx[x], fy[y]))
        return out

    def segment_slab(self, p0: np.ndarray, p1: np.ndarray):
        """Slab test for segments p0 -> p1 (arrays (N, 3)).

        Returns (hit, t_in, t_out, face_in, face_out): hit (N,) bool if the segment overlaps the box
        interior; t_in/t_out in [0, 1] (clamped: t_in = 0 when p0 is inside, t_out = 1 when p1 is inside);
        face_in/face_out: index into FACES of the boundary crossed at t_in/t_out, or -1 when that end is
        inside the box (no crossing).
        """
        a, b = self.to_local(p0), self.to_local(p1)
        d = b - a
        lo = np.array([-self.half[0], -self.half[1], 0.0])
        hi = np.array([self.half[0], self.half[1], 2 * self.half[2]])
        n = a.shape[0]
        t_near = np.full(n, -np.inf)
        t_far = np.full(n, np.inf)
        f_near = np.full(n, -1, dtype=np.int64)
        f_far = np.full(n, -1, dtype=np.int64)
        miss = np.zeros(n, dtype=bool)
        # face index for (axis, entering through the low side) / (axis, high side)
        low_face = {0: FACES.index("back"), 1: FACES.index("right"), 2: FACES.index("floor")}
        high_face = {0: FACES.index("front"), 1: FACES.index("left"), 2: FACES.index("roof")}
        for ax in range(3):
            da = d[:, ax]
            par = np.abs(da) < _EPS
            miss |= par & ((a[:, ax] < lo[ax]) | (a[:, ax] > hi[ax]))
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lo[ax] - a[:, ax]) / da
                t2 = (hi[ax] - a[:, ax]) / da
            pos = da > 0
            tn = np.where(pos, t1, t2)
            tf = np.where(pos, t2, t1)
            fn = np.where(pos, low_face[ax], high_face[ax])
            ff = np.where(pos, high_face[ax], low_face[ax])
            tn = np.where(par, -np.inf, tn)
            tf = np.where(par, np.inf, tf)
            upd_n = tn > t_near
            t_near = np.where(upd_n, tn, t_near)
            f_near = np.where(upd_n, fn, f_near)
            upd_f = tf < t_far
            t_far = np.where(upd_f, tf, t_far)
            f_far = np.where(upd_f, ff, f_far)
        hit = ~miss & (t_near < t_far) & (t_far > 0.0) & (t_near < 1.0)
        start_inside = t_near <= 0.0
        end_inside = t_far >= 1.0
        t_in = np.where(start_inside, 0.0, t_near)
        t_out = np.where(end_inside, 1.0, t_far)
        face_in = np.where(start_inside, -1, f_near)
        face_out = np.where(end_inside, -1, f_far)
        return hit, t_in, t_out, face_in, face_out


@dataclass
class Crossing:
    """One boundary crossing of a segment through a box face (for transmission / opening lookups)."""
    seg: np.ndarray       # (K,) segment indices
    box: int
    face: str
    t: np.ndarray         # (K,) segment parameter of the crossing point
    point: np.ndarray     # (K, 3)
    entering: bool        # True: from outside into the box; False: leaving the box
    opening: np.ndarray   # (K,) opening index on that face, -1 = solid face


class Scene:
    """All obstacle boxes in a room scene."""

    def __init__(self, boxes: list[Box]):
        self.boxes = boxes
        self._raycore = None

    @property
    def raycore(self):
        """Lazy immutable query snapshot; call invalidate_rays after editing boxes."""
        if self._raycore is None:
            from .raycore import RayCore
            self._raycore = RayCore(self)
        return self._raycore

    def invalidate_rays(self):
        if self._raycore is not None:
            self._raycore.close()
        self._raycore = None

    def __len__(self) -> int:
        return len(self.boxes)

    def box(self, box_id: str) -> Box:
        for b in self.boxes:
            if b.id == box_id:
                return b
        raise KeyError(box_id)

    def containing(self, p: np.ndarray) -> np.ndarray:
        """(N,) index of the box containing each point, -1 if outside every box."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        out = np.full(p.shape[0], -1, dtype=np.int64)
        for i, b in enumerate(self.boxes):
            out = np.where((out < 0) & b.contains(p), i, out)
        return out

    def crossings(self, p0: np.ndarray, p1: np.ndarray, ignore_box: int | None = None) -> list[Crossing]:
        """Every face crossing of the segments p0[i] -> p1[i] (arrays (N, 3)), grouped by (box, face,
        direction). A segment that passes through a box yields an entering and a leaving crossing.
        `ignore_box` omits one box inside the ray query (e.g. the box an endpoint lies on)."""
        return self.raycore.legacy_crossings(p0, p1, ignore_box=ignore_box)

    def blocked(self, p0: np.ndarray, p1: np.ndarray, ignore_box: int | None = None) -> np.ndarray:
        """(N,) True where the segment passes through any solid box (openings are NOT considered open here;
        use `crossings` for transmission through openings). `ignore_box` skips one box (e.g. the reflector)."""
        return self.raycore.any_hit(p0, p1, ignore_box=ignore_box)

    def reflection(self, src: np.ndarray, rx: np.ndarray, box: int, face: str):
        """Specular reflection of src -> rx off one face. Returns (valid (N,), point (N, 3), image_rx (N, 3)).
        Valid when both ends are on the outward side, the reflection point lies on the face, and neither leg
        is blocked by another box (the reflecting box's own interior is excluded by the side test)."""
        src = np.atleast_2d(np.asarray(src, dtype=np.float64))
        rx = np.atleast_2d(np.asarray(rx, dtype=np.float64))
        src, rx = np.broadcast_arrays(src, rx)
        f = self.boxes[box].faces[face]
        ds = (src - f.center) @ f.normal
        dr = (rx - f.center) @ f.normal
        img = f.mirror(rx)
        den = (img - src) @ f.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(den) > _EPS, -ds / den, np.nan)
        q = src + t[:, None] * (img - src)
        valid = (ds > _EPS) & (dr > _EPS) & (t > 0) & (t < 1) & f.contains_uv(f.uv(q))
        if valid.any():
            idx = np.nonzero(valid)[0]
            blk = self.blocked(src[idx], q[idx], ignore_box=box) | self.blocked(q[idx], rx[idx], ignore_box=box)
            valid[idx[blk]] = False
        return valid, q, img

    @staticmethod
    def edge_point(src: np.ndarray, rx: np.ndarray, a: np.ndarray, b: np.ndarray):
        """Point E on segment a-b minimising |src-E| + |E-rx| (the diffraction point; exact for a line:
        unfold rx about the edge into src's half-plane). Returns (E (N, 3), s (N,) in [0, 1], interior (N,)
        True when the unconstrained optimum lies strictly inside the segment)."""
        src = np.atleast_2d(np.asarray(src, dtype=np.float64))
        rx = np.atleast_2d(np.asarray(rx, dtype=np.float64))
        e = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
        le = float(np.linalg.norm(e))
        eh = e / le
        ps = (src - a) @ eh
        pr = (rx - a) @ eh
        ds = np.linalg.norm((src - a) - ps[:, None] * eh, axis=-1)
        dr = np.linalg.norm((rx - a) - pr[:, None] * eh, axis=-1)
        tot = ds + dr
        w = np.where(tot > _EPS, ds / np.maximum(tot, _EPS), 0.5)
        par = ps + (pr - ps) * w
        s = par / le
        interior = (s > 0.0) & (s < 1.0)
        s = np.clip(s, 0.0, 1.0)
        return np.asarray(a) + s[:, None] * e, s, interior
