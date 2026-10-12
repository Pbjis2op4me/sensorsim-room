"""Batched, backend-neutral finite-segment geometry (ENU metres).

Boxes retain Box.segment_slab's exact float64 rules: parallel |d_local| < 1e-9,
strict near < far, endpoints excluded from face crossings, axis x/y/z tie order.
any_hit/visibility test volume overlap > 1e-9 in segment parameter, including a
zero-length segment inside a box; closest_hit reports a *boundary*, not occupancy.
Openings annotate crossings but do not remove geometry. Materials are identifiers,
never attenuation. Reflection/image construction stays with the existing models.

Panels are planar polygons given by a centre, a normal and in-plane axes. Generic
triangles are double-sided, exclude t <= 1e-9 and t >= 1-1e-9, use a 1e-9
barycentric margin, and reject parallel/coplanar rays. Coincident triangles with
the same face id produce one crossing; different faces remain separate. Closest
ties use (t, face_id, object_id), with no approximate distance bucketing.

Endpoint surfaces are declared by ID, not inferred from an epsilon (the
OptiX/Vulkan self-intersection rule). `start_face`/`end_face` (int or (N,),
-1 none) name the planar panel or mesh face on which a segment's start/end
point lies, e.g. an image-source reflection point. That face is excluded for
that ray: no crossing, occupancy or closest hit. A segment cannot otherwise
cross the plane containing one of its endpoints, so this equals endpoint
exclusion without the float32 ambiguity. Box faces are rejected: use
`ignore_box` (int or (N,) box index, -1 none), which excludes a whole box from
any segment query.

CPU is the vectorised NumPy reference. Backend selection accepts cpu|auto;
`auto` resolves to CPU. The incomplete Vulkan backend has been removed.
`ray_backend_manifest()` records CPU usage without probing devices.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Protocol

import numpy as np

from .geometry import Box, Crossing, FACES

class Panel(Protocol):
    """A planar polygon panel: centre, normal and in-plane axes u, v (metres), polygon in (u, v)."""
    center: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    polygon: tuple


@dataclass(frozen=True)
class QueryContract:
    """Geometry semantics, independent of the selected numerical backend."""
    id: str
    parameter: str
    boundary_policy: str


QUERY_CONTRACTS = MappingProxyType({
    "any_hit": QueryContract("segment-occupancy/1", "finite segment fraction",
        "volume overlap > 1e-9; zero-length inside is occupied; openings annotate geometry"),
    "segment_crossings": QueryContract("segment-boundary/1", "finite segment fraction",
        "strict near < far; endpoints and declared endpoint faces excluded; x/y/z box ties"),
})


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    face_ids: np.ndarray
    material_ids: np.ndarray
    object_ids: np.ndarray
    materials: tuple[str, ...] = ()

    def __post_init__(self):
        arrays = {"vertices": (np.float64, (-1, 3)), "triangles": (np.int64, (-1, 3)),
                  "face_ids": (np.int64, (-1,)), "material_ids": (np.int64, (-1,)),
                  "object_ids": (np.int64, (-1,))}
        for name, (dtype, shape) in arrays.items():
            a = np.asarray(getattr(self, name))
            if a.ndim != len(shape) or (len(shape) == 2 and a.shape[1] != 3):
                raise ValueError(f"invalid {name} shape")
            if name != "vertices":
                if not np.issubdtype(a.dtype, np.integer):
                    raise ValueError(f"{name} must contain nonnegative integers")
                limits = np.iinfo(dtype)
                if a.size and (int(a.min()) < 0 or int(a.max()) > limits.max):
                    raise ValueError(f"{name} values must fit nonnegative {np.dtype(dtype)}")
            a = np.array(a, dtype=dtype, copy=True)
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if not np.isfinite(self.vertices).all():
            raise ValueError("finite mesh vertices required")
        n = len(self.triangles)
        if any(len(getattr(self, k)) != n for k in ("face_ids", "material_ids", "object_ids")):
            raise ValueError("one face/material/object id per triangle required")
        if n and (self.triangles.max() >= len(self.vertices) or
                  (self.materials and self.material_ids.max() >= len(self.materials))):
            raise ValueError("mesh index out of range")
        if n:
            _,first,inverse=np.unique(self.face_ids,return_index=True,return_inverse=True)
            if (np.any(self.material_ids != self.material_ids[first][inverse]) or
                    np.any(self.object_ids != self.object_ids[first][inverse])):
                raise ValueError("one physical face id must have one material and object owner")
        object.__setattr__(self, "materials", tuple(self.materials))


@dataclass(frozen=True)
class Crossings:
    """Flat crossings sorted by (ray, t, face_id, object_id); t is in (0,1)."""
    ray: np.ndarray
    object_id: np.ndarray
    face_id: np.ndarray
    material_id: np.ndarray
    t: np.ndarray
    distance: np.ndarray
    point: np.ndarray
    entering: np.ndarray
    opening: np.ndarray

    def __len__(self):
        return len(self.ray)


@dataclass(frozen=True)
class ClosestHit:
    hit: np.ndarray
    t: np.ndarray             # inf on miss
    distance: np.ndarray      # metres, inf on miss
    face_id: np.ndarray       # -1 on miss
    material_id: np.ndarray   # -1 on miss
    object_id: np.ndarray     # -1 on miss


class RayQueries(Protocol):
    def any_hit(self, start, end, *, ignore_box=None) -> np.ndarray: ...
    def closest_hit(self, start, end) -> ClosestHit: ...
    def segment_crossings(self, start, end) -> Crossings: ...
    def visibility(self, start, end, *, ignore_box=None) -> np.ndarray: ...


@dataclass(frozen=True)
class RayBackendInput:
    """Versioned geometry and CPU-reference handoff; no backend owns its caller."""
    mesh: TriangleMesh
    boxes: tuple[Box, ...]
    panels: tuple[Panel, ...]
    standalone: bool
    batch_size: int
    reference: RayCore
    version: str = "ray-backend/1"

    def __post_init__(self):
        if self.version != "ray-backend/1":
            raise ValueError(f"unsupported ray backend version: {self.version}")
        if not isinstance(self.mesh, TriangleMesh) or not isinstance(self.reference, RayCore):
            raise ValueError("ray backend requires typed mesh and CPU reference")
        if (self.reference.mesh is not self.mesh or self.reference.boxes is not self.boxes
                or self.reference.panels is not self.panels or self.reference.standalone != self.standalone):
            raise ValueError("ray backend geometry must match its CPU reference")
        if self.batch_size < 1:
            raise ValueError("invalid ray backend resource budget")


def segments(start, end):
    a, b = np.broadcast_arrays(np.atleast_2d(np.asarray(start, np.float64)),
                               np.atleast_2d(np.asarray(end, np.float64)))
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite (N,3) or (3,) ENU segment endpoints required")
    return a, b


def triangle_intersection(vertices, start, end):
    """Vectorized double-sided Moller-Trumbore, one triangle against N segments."""
    v0, v1, v2 = vertices
    e1, e2, d = v1-v0, v2-v0, end-start
    p = np.cross(d, e2)
    det = p @ e1
    # Relative parallel threshold makes scaling a triangle/ray well defined.
    scale = np.linalg.norm(e1)*np.linalg.norm(e2)*np.linalg.norm(d, axis=-1)
    good = np.abs(det) > 1e-14*scale
    inv = np.divide(1., det, out=np.zeros_like(det), where=good)
    s = start-v0
    u = np.sum(s*p, axis=-1)*inv
    q = np.cross(s, e1)
    v = np.sum(d*q, axis=-1)*inv
    t = (q @ e2)*inv
    hit = good & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (t > 1e-9) & (t < 1-1e-9)
    return hit, t, det > 0


DEFAULT_BACKEND = "auto"
AUTO_RULE = "auto resolves to the CPU NumPy reference"
PRECISION_NOTE = "CPU float64 finite-segment queries"
_USAGE = {"cpu": [0, 0]}  # process-wide [queries, rays]


def requested_backend():
    """(selection, source): an explicit SENSORSIM_RAY_BACKEND, else DEFAULT_BACKEND."""
    env = os.environ.get("SENSORSIM_RAY_BACKEND")
    return (env, "env SENSORSIM_RAY_BACKEND") if env else (DEFAULT_BACKEND, "code default")


def usage():
    """Snapshot of process-wide backend usage (for per-run differences)."""
    return {k: list(v) for k, v in _USAGE.items()}


def ray_backend_manifest(since=None):
    """Run-manifest record of what decides geometry float bits in this process.

    Never initialises a device: `device` is None unless a query already probed it.
    `since` (a `usage()` snapshot) restricts counts to one run.
    """
    selection, source = requested_backend()
    counts = {k: {"queries": v[0]-(since or {}).get(k, [0, 0])[0], "rays": v[1]-(since or {}).get(k, [0, 0])[1]}
              for k, v in _USAGE.items()}
    return dict(requested=selection, source=source, rule=AUTO_RULE,
                device=None, usage=counts, usage_scope="run" if since is not None else "process",
                precision=PRECISION_NOTE,
                query_contracts={name: contract.id for name, contract in QUERY_CONTRACTS.items()})


class RayCore:
    """Immutable geometry snapshot. Rebuild after editing a scene.

    Use scene boxes, panels, OR a general TriangleMesh. The mesh export always
    carries stable physical face/material/object ids, including aperture faces.
    No automatic terrain/ground plane is added. The default is DEFAULT_BACKEND
    ('auto', which resolves to CPU).
    """
    def __init__(self, scene=None, *, panels=(), mesh: TriangleMesh | None = None,
                 backend=None, batch_size=65536):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("positive batch_size required")
        if mesh is not None and (scene is not None or panels):
            raise ValueError("use a scene/panels or a standalone mesh")
        self.boxes = tuple(Box(b.spec) for b in scene.boxes) if scene is not None else ()
        self.panels = tuple(panels)
        if self.panels:
            self._centers,self._normals,self._u,self._v = (
                np.stack([getattr(p, name) for p in self.panels]) for name in ("center", "normal", "u", "v"))
            corners=max(len(p.polygon) for p in self.panels)
            self._polygons=np.zeros((len(self.panels),corners,2))
            self._edges=np.zeros_like(self._polygons)
            for i,p in enumerate(self.panels):
                poly=np.asarray(p.polygon)
                self._polygons[i,:len(poly)]=poly
                self._edges[i,:len(poly)]=np.roll(poly,-1,axis=0)-poly
        self.batch_size = batch_size
        self._standalone = mesh is not None
        self.mesh = mesh if mesh is not None else self._export()
        planar = self.mesh.face_ids if self._standalone else self.mesh.face_ids[self.mesh.face_ids >= 6*len(self.boxes)]
        self._planar_faces = np.unique(planar)
        self._closed = False
        self.backend = "cpu"
        self.fallback_reason = None
        self._record = True  # internal CPU recheck copies set False
        selection, self.selection_source = (backend, "argument") if backend else requested_backend()
        if selection == "vulkan":
            raise ValueError("Vulkan backend was removed from sensorsim-room; use cpu or auto")
        if selection not in ("cpu", "auto"):
            raise ValueError("SENSORSIM_RAY_BACKEND must be cpu or auto")
        self._selection = "cpu"

    def _record_query(self, n):
        if self._closed: raise RuntimeError("raycore is closed")
        if getattr(self, "_candidate_rows", None) is not None and n > self.batch_size:
            raise ValueError("CPU reference candidate queries must fit one batch")
        if self._record and n:
            _USAGE["cpu"][0] += 1
            _USAGE["cpu"][1] += int(n)

    @property
    def standalone(self) -> bool:
        return self._standalone

    def cpu_reference(self, candidate_rows: dict[int, np.ndarray] | None = None) -> RayCore:
        """Independent CPU query lifetime, optionally restricted to local ray rows.

        Candidates are conservative primitive IDs, not physical object IDs.
        Scene primitives enumerate boxes followed by panels; absent IDs have
        no candidate rays. Standalone meshes use triangle indices instead.
        Restricted references accept a single batch.
        """
        from copy import copy
        if self._closed:
            raise RuntimeError("raycore is closed")
        cpu = copy(self)
        cpu._selection, cpu.backend, cpu._record = "cpu", "cpu", False
        cpu._candidate_rows = None
        if candidate_rows is not None:
            count = len(self.mesh.triangles) if self.standalone else len(self.boxes) + len(self.panels)
            candidates = {}
            for key, rows in candidate_rows.items():
                rows = np.asarray(rows)
                if (not isinstance(key, (int, np.integer)) or not 0 <= key < count or rows.ndim != 1
                        or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0)):
                    raise ValueError("invalid CPU reference candidate IDs/rows")
                candidates[int(key)] = rows.copy()
            cpu._candidate_rows = candidates
        return cpu

    def backend_input(self) -> RayBackendInput:
        return RayBackendInput(self.mesh, self.boxes, self.panels, self.standalone,
                               self.batch_size, self.cpu_reference())

    def panel_face_id(self, index):
        """Physical face id of panel `index` (for start_face/end_face)."""
        if not 0 <= index < len(self.panels):
            raise IndexError("panel index out of range")
        return 6*len(self.boxes)+index

    def _exclusions(self, n, ignore_box=None, start_face=None, end_face=None):
        """(N,3) int64 [ignore box, start face, end face], -1 for none; None if unused."""
        if ignore_box is None and start_face is None and end_face is None:
            return None
        ids=np.full((n,3),-1,np.int64)
        for col,value in enumerate((ignore_box,start_face,end_face)):
            if value is None:
                continue
            v=np.asarray(value)
            if v.dtype == bool or not np.issubdtype(v.dtype,np.integer) or v.ndim > 1:
                raise ValueError("exclusion ids must be integers or (N,) integer arrays")
            if v.size and int(v.max()) > np.iinfo(np.int64).max:  # uint64 would wrap to the -1 "none" sentinel
                raise ValueError("exclusion ids must fit a signed int64")
            ids[:,col]=np.broadcast_to(v,(n,))
        box,faces=ids[:,0],ids[:,1:]
        if np.any((box < -1) | (box >= len(self.boxes))):  # standalone meshes have no boxes
            raise ValueError("ignore_box must be -1 or a scene box index")
        declared=faces[faces >= 0]
        if np.any(faces < -1) or not np.all(np.isin(declared,self._planar_faces)):
            raise ValueError("start_face/end_face must be -1 or a panel/mesh face id; "
                             "for box surfaces use ignore_box")
        return ids

    @staticmethod
    def _rows_kept(ids, rows, obj=None, face=None):
        """Rows not excluded for this box `obj` or planar `face`."""
        if ids is None:
            return rows
        if obj is not None:
            return rows[ids[rows,0] != obj]
        return rows[(ids[rows,1] != face) & (ids[rows,2] != face)]

    def _export(self):
        vertices, triangles, faces, mats, objects, materials = [], [], [], [], [], []
        def polygon(corners, face_id, object_id, material):
            if material not in materials:
                materials.append(material)
            off = len(vertices)
            vertices.extend(corners)
            for j in range(1, len(corners)-1):
                triangles.append((off, off+j, off+j+1))
                faces.append(face_id)
                mats.append(materials.index(material))
                objects.append(object_id)
        for bi, box in enumerate(self.boxes):
            for fi, name in enumerate(FACES):
                face = box.faces[name]
                polygon(face.corners(), bi*6+fi, bi, face.material)
        for pi, panel in enumerate(self.panels):
            corners = panel.center + np.asarray(panel.polygon) @ np.array([panel.u, panel.v])
            polygon(corners, 6*len(self.boxes)+pi, len(self.boxes)+pi, panel.assembly_id)
        return TriangleMesh(np.array(vertices, float).reshape(-1,3), np.array(triangles, int).reshape(-1,3),
                            np.array(faces,int), np.array(mats,int), np.array(objects,int), tuple(materials))

    def _groups(self, a, b):
        if self._closed:
            raise RuntimeError("raycore is closed")
        if getattr(self,"_candidate_rows",None) is not None:
            if len(a) > self.batch_size or any(np.any(rows >= len(a)) for rows in self._candidate_rows.values()):
                raise ValueError("CPU reference candidate rows exceed the query batch")
            return sorted(self._candidate_rows.items())
        # None means all rows; no rays x objects allocation.
        n = len(self.mesh.triangles) if self._standalone else len(self.boxes)+len(self.panels)
        return ((i, None) for i in range(n))

    def _events(self, a, b, *, occupancy=False, ids=None):
        """Yield rows, t, object, face, material, entering, opening, blocked.

        `ids` is this chunk's (N,3) exclusion array from `_exclusions`, or None.
        """
        all_rows = np.arange(len(a))
        materials = {name: i for i, name in enumerate(self.mesh.materials)}
        groups = self._groups(a, b)  # Validate candidate rows before panel work too.
        if self.panels:
            yield from self._panel_events_cpu(a,b,materials,ids)
        for obj, rows in groups:
            if not self._standalone and obj >= len(self.boxes):
                continue
            rows = all_rows if rows is None else rows
            if ids is not None:
                rows = (self._rows_kept(ids,rows,face=self.mesh.face_ids[obj]) if self._standalone
                        else self._rows_kept(ids,rows,obj=obj))
                if not len(rows):
                    continue
            aa, bb = (a, b) if rows is all_rows else (a[rows], b[rows])
            if self._standalone:
                tri = obj
                hit, t, entering = triangle_intersection(self.mesh.vertices[self.mesh.triangles[tri]], aa, bb)
                sel = np.flatnonzero(hit)
                if len(sel):
                    yield (rows[sel], t[sel], self.mesh.object_ids[tri], self.mesh.face_ids[tri],
                           self.mesh.material_ids[tri], entering[sel], np.full(len(sel), -1), hit[sel])
            elif obj < len(self.boxes):
                box = self.boxes[obj]
                hit, tin, tout, fin, fout = box.segment_slab(aa, bb)
                if occupancy:
                    sel = np.flatnonzero(hit & (tout-tin > 1e-9))
                    if len(sel):
                        yield (rows[sel], None, obj, None, None, None, None, None)
                    continue
                for t, fids, enter in ((tin, fin, True), (tout, fout, False)):
                    for fi, name in enumerate(FACES):
                        sel = np.flatnonzero(hit & (fids == fi))
                        if not len(sel):
                            continue
                        face = box.faces[name]
                        p = aa[sel]+t[sel,None]*(bb[sel]-aa[sel])
                        yield (rows[sel], t[sel], obj, obj*6+fi, materials[face.material],
                               np.full(len(sel),enter), face.opening_index(p), None)

    def _panel_events_cpu(self, a, b, materials, ids=None):
        """Vectorize scalar-heavy E3 across panels as well as receiver rays."""
        # Bound polygon work arrays, including richer convex polygons.
        tile=max(1,262144//(len(self.panels)*self._polygons.shape[1]))
        normal=self._normals[:,None,:]
        center=self._centers[:,None,:]
        edges=self._edges[:,None,:,:]
        direction=b-a
        candidates=getattr(self,"_candidate_rows",None)
        for lo in range(0,len(a),tile):
            aa=a[None,lo:lo+tile,:]; d=direction[None,lo:lo+tile,:]
            den=np.sum(d*normal,axis=-1)
            with np.errstate(divide="ignore",invalid="ignore"):
                t=np.divide(np.sum((center-aa)*normal,axis=-1),den,
                            out=np.full_like(den,np.nan),where=np.abs(den)>1e-12)
            q=aa+t[...,None]*d
            delta=q-center
            uv=np.stack([np.sum(delta*self._u[:,None,:],axis=-1),
                         np.sum(delta*self._v[:,None,:],axis=-1)],axis=-1)
            delta=uv[:,:,None,:]-self._polygons[:,None,:,:]
            inside=np.all(edges[...,0]*delta[...,1]-edges[...,1]*delta[...,0]>=-1e-9,axis=-1)
            hit=inside&(t>1e-9)&(t<1-1e-9)
            if ids is not None:
                face=6*len(self.boxes)+np.arange(len(self.panels))[:,None]
                hit&=(face != ids[None,lo:lo+tile,1]) & (face != ids[None,lo:lo+tile,2])
            for pi in np.flatnonzero(np.any(hit,axis=1)):
                rows=np.flatnonzero(hit[pi])
                if candidates is not None:
                    rows=rows[np.isin(lo+rows,candidates.get(len(self.boxes)+pi,()))]
                    if not len(rows):
                        continue
                p=self.panels[pi]
                yield (lo+rows,t[pi,rows],len(self.boxes)+pi,6*len(self.boxes)+pi,
                       materials[p.assembly_id],den[pi,rows]<0,p.aperture_index(q[pi,rows]),None)

    def any_hit(self, start, end, *, ignore_box=None, start_face=None, end_face=None):
        a,b = segments(start,end)
        ids = self._exclusions(len(a),ignore_box,start_face,end_face)
        self._record_query(len(a))
        out = np.zeros(len(a),bool)
        for lo in range(0,len(a),self.batch_size):
            hi = lo+self.batch_size
            for rows, *_ in self._events(a[lo:hi], b[lo:hi], occupancy=True,
                                         ids=None if ids is None else ids[lo:hi]):
                out[lo+rows] = True
        return out

    def visibility(self, start, end, *, ignore_box=None, start_face=None, end_face=None):
        return ~self.any_hit(start,end,ignore_box=ignore_box,start_face=start_face,end_face=end_face)

    def segment_crossings(self, start, end, *, ignore_box=None, start_face=None, end_face=None):
        a,b = segments(start,end)
        ids = self._exclusions(len(a),ignore_box,start_face,end_face)
        self._record_query(len(a))
        return self._cpu_crossings(a,b,ids)

    def _cpu_crossings(self, a, b, ids):
        fields = [[] for _ in range(7)]
        for lo in range(0,len(a),self.batch_size):
            hi = lo+self.batch_size
            for rows, t, obj, face, mat, enter, opening, _ in self._events(
                    a[lo:hi],b[lo:hi],ids=None if ids is None else ids[lo:hi]):
                for col,val in zip(fields,(rows+lo,t,np.full(len(rows),obj),np.full(len(rows),face),
                                            np.full(len(rows),mat),enter,opening)):
                    col.append(val)
        types = (np.int64,float,np.int64,np.int64,np.int64,bool,np.int64)
        ray,t,obj,face,mat,enter,opening = [np.concatenate(c).astype(ty,copy=False) if c else np.empty(0,ty)
                                         for c,ty in zip(fields,types)]
        order = np.lexsort((obj,face,t,ray))
        if self._standalone and len(order):
            # A fan's diagonal must not stack the same physical face twice.
            by_face=np.lexsort((t,face,ray))
            r, f, tt = ray[by_face], face[by_face], t[by_face]
            keep = np.r_[True, (r[1:] != r[:-1]) | (f[1:] != f[:-1]) | (np.abs(tt[1:]-tt[:-1]) > 1e-12)]
            allowed=np.zeros(len(ray),bool)
            allowed[by_face[keep]]=True
            order=order[allowed[order]]
        ray,t,obj,face,mat,enter,opening = (x[order] for x in (ray,t,obj,face,mat,enter,opening))
        point = a[ray]+t[:,None]*(b[ray]-a[ray])
        distance = t*np.linalg.norm(b[ray]-a[ray],axis=-1)
        return Crossings(ray,obj,face,mat,t,distance,point,enter,opening)

    def closest_hit(self, start, end, *, ignore_box=None, start_face=None, end_face=None):
        a,b = segments(start,end)
        ids = self._exclusions(len(a),ignore_box,start_face,end_face)
        self._record_query(len(a))
        n=len(a)
        t=np.full(n,np.inf)
        face=np.full(n,-1,int); mat=face.copy(); obj=face.copy()
        # Bounded crossing temporaries even for ten million rays.
        for lo in range(0,n,self.batch_size):
            hi=lo+self.batch_size
            h=self._cpu_crossings(a[lo:hi],b[lo:hi],None if ids is None else ids[lo:hi])
            if len(h):
                _,ix=np.unique(h.ray,return_index=True)
                rows=lo+h.ray[ix]
                t[rows],face[rows],mat[rows],obj[rows]=h.t[ix],h.face_id[ix],h.material_id[ix],h.object_id[ix]
        distance=np.full(n,np.inf)
        hit=face >= 0
        distance[hit]=t[hit]*np.linalg.norm(b[hit]-a[hit],axis=-1)
        return ClosestHit(hit,t,distance,face,mat,obj)

    def legacy_crossings(self, start, end, *, ignore_box=None):
        """Retain Scene's (box, entering/leaving, face, row) summation order."""
        h=self.segment_crossings(start,end,ignore_box=ignore_box)
        out=[]
        for bi in range(len(self.boxes)):
            for entering in (True,False):
                for fi,name in enumerate(FACES):
                    ix=np.flatnonzero((h.object_id == bi) & (h.face_id == bi*6+fi) & (h.entering == entering))
                    if len(ix):
                        ix=ix[np.argsort(h.ray[ix],kind="stable")]
                        out.append(Crossing(h.ray[ix],bi,name,h.t[ix],h.point[ix],entering,h.opening[ix]))
        return out

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
