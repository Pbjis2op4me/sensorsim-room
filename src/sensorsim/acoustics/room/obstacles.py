"""Finite-face BTM scattering and folded shoebox image-path visibility.

Coherent first-order edge sums are reduced to the existing octave path API.
Thick shadows/higher image orders retain a documented K-A detour approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from sensorsim.acoustics.absorption import amplitude_factor
from sensorsim.acoustics.barrier import route_attenuation_db
from sensorsim.acoustics.gain_budget import path_length_compensation
from sensorsim.acoustics.edge_diffraction import (edge_geometry, edge_response, octave_amplitude,
                                                 octave_grid, rectangular_response)
from sensorsim.acoustics.room.model import OCTAVE_HZ, SURFACES, RoomModel
from sensorsim.acoustics.scene_paths import side_route, top_route
from sensorsim.core.messages import ModelValidity
from sensorsim.core.schema import SceneBox
from sensorsim.scene.geometry import Box, Scene
from sensorsim.scene.raycore import RayCore

VALIDITY = ModelValidity(
    name="Room box obstacles: finite-face BTM, floor images and barrier detours",
    status="partially validated", freq_range_hz=(125., 8000.),
    conditions="Frozen BTM/BRAS gates partially pass; measured residual failures retained. "
    "Isolated finite faces: coherent image plus first-order rigid edges. Single-body "
    "direct/floor groups include lit edges and floor images. Octave RMS reduction loses "
    "phase between groups and the edge IR; no transmission or all-orders body solution. "
    "Thick shadows, multiple boxes and higher images use K-A detours on folded legs. "
    "Thin-screen thickness <=5% of both face dimensions is an approximation. Diffuse tail approximate.",
    citation="Svensson, Fred & Vanderkooy 1999, doi:10.1121/1.428071; "
    "Kurze & Anderson 1971 Applied Acoustics 4:35; ISO 9613-2:1996 section 7.4; "
    "parent-project note ROOM_OBSTACLES_BENCH (not distributed)",
)


@dataclass(frozen=True)
class ObstacleSpectrum:
    """Pressure per 1 m reference, exp(+i*omega*t); earliest arrival removed.

    Uniform real-FFT grid includes DC and Nyquist. This optional transfer is
    separate from JSON diagnostics and replaces the listed real image paths.
    max_delay_s bounds response AFTER time_origin_s, not a 129-tap FIR.
    """
    frequency_hz: np.ndarray
    transfer: np.ndarray
    time_origin_s: float
    replaces_image_indices: np.ndarray
    max_delay_s: float


def _band_pressure(values, frequencies):
    """Octave-constant fast quadrature; complex tier uses continuous log-f gains."""
    values = np.asarray(values)
    if frequencies.ndim == 2:
        return values[:, None]
    return np.interp(np.log2(np.maximum(frequencies, 1.)), np.log2(OCTAVE_HZ), values)


def _spectrum(model, frequencies, field, length, indices, config):
    origin = float(length/model.sound_speed_m_s)
    # Conservative bound for finite box edges, face images and floor images.
    # Zero-phase ISO ringing remains a declared finite-FFT approximation.
    support = 8*float(np.linalg.norm(model.shoebox_dims_m))/model.sound_speed_m_s-origin
    if config.air_absorption:
        field = field*amplitude_factor(frequencies, length, model.temperature_c, model.humidity_pct)
    return ObstacleSpectrum(frequencies.copy(), field*np.exp(2j*np.pi*frequencies*origin),
                             origin, np.asarray(indices, int), max(0., support))


def obstacle_scene(model: RoomModel) -> Scene:
    return Scene([Box(SceneBox(id=b.name, base_center_enu_m=(
        b.min_xyz[0]+b.size_xyz[0]/2, b.min_xyz[1]+b.size_xyz[1]/2, b.min_xyz[2]),
        size_m=b.size_xyz, heading_deg=90., material=b.material)) for b in model.obstacles])


def transport_hits(model, scene, rays, start, direction):
    """Next boundary for energy rays: shoebox analytic planes + shared box queries.

    Return distance, inward-to-air normal, absorption, scattering. Boundary
    coefficients are ISO energy fractions; geometric edges do not add a second
    empirical scattering term. Same RayCore backend switch as obstacle ISM.
    """
    with np.errstate(divide='ignore',invalid='ignore'):
        lengths=np.where(direction>0,(np.asarray(model.shoebox_dims_m)-start)/direction,-start/direction)
    lengths=np.where(np.abs(direction)>1e-14,lengths,np.inf)
    axis=np.argmin(lengths,axis=1);row=np.arange(len(start))
    distance=lengths[row,axis]
    wall=2*axis+(direction[row,axis]>0)
    normal=np.zeros_like(start);normal[row,axis]=-np.sign(direction[row,axis])
    alpha=np.array([model.absorption[s] for s in SURFACES])[wall]
    scatter=np.array([model.scattering[s] for s in SURFACES])[wall]
    if model.obstacles:
        from sensorsim.scene.geometry import FACES
        hit=rays.closest_hit(start,start+distance[:,None]*direction)
        use=np.flatnonzero(hit.hit & (hit.distance<distance))
        for i in use:
            obj=int(hit.object_id[i]);face=FACES[int(hit.face_id[i])%6]
            normal[i]=scene.boxes[obj].faces[face].normal
            alpha[i]=model.obstacles[obj].absorption;scatter[i]=model.obstacles[obj].scattering
        distance[use]=hit.distance[use]
    return distance,normal,alpha,scatter


def folded_segments(images_m: np.ndarray, receiver_xyz_m, dims_m
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (path index, segment start, segment end), receiver to original source.

    Cross every lattice plane k*L in the unfolded line, then triangular-wave fold.
    Arrays are batched across images. Coincident corner-plane crossings are deduplicated.
    """
    images, receiver, dims = np.asarray(images_m), np.asarray(receiver_xyz_m), np.asarray(dims_m)
    if len(images) == 0:
        return np.empty(0, int), np.empty((0, 3)), np.empty((0, 3))
    vector = images-receiver
    crossings = [np.zeros((len(images), 1)), np.ones((len(images), 1))]
    for axis in range(3):
        low = int(np.floor(min(images[:, axis].min(), receiver[axis])/dims[axis]))
        high = int(np.ceil(max(images[:, axis].max(), receiver[axis])/dims[axis]))
        planes = np.arange(low, high+1)*dims[axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (planes[None, :]-receiver[axis])/vector[:, axis, None]
        crossings.append(np.where((t > 1e-12) & (t < 1-1e-12), t, 1.))
    time = np.sort(np.concatenate(crossings, axis=1), axis=1)
    points = receiver + time[:, :, None]*vector[:, None, :]
    folded = dims-np.abs(np.mod(points, 2*dims)-dims)
    keep = np.diff(time, axis=1) > 1e-12
    owner = np.broadcast_to(np.arange(len(images))[:, None], keep.shape)
    return owner[keep], folded[:, :-1][keep], folded[:, 1:][keep]


def _clear_leg(rays: RayCore, a: np.ndarray, b: np.ndarray, ignore_box=None) -> np.ndarray:
    # Nudge endpoints off exact box/room edge contacts; do not ignore intervening boxes.
    delta = b-a
    return rays.visibility(a+1e-8*delta, b-1e-8*delta, ignore_box=ignore_box)


def shortest_detour(scene: Scene, rays: RayCore, start: np.ndarray, end: np.ndarray,
                     box_index: int, sound_speed: float, *, room_dims_m=None,
                     reciprocal: bool = False) -> dict:
    """Batched reused box routes; a blocked-by-another-box route is invalid."""
    start, end = np.atleast_2d(start), np.atleast_2d(end)
    swap = np.zeros(len(start), bool)
    if reciprocal:
        # The legacy alternating edge solve takes four iterations. Canonical
        # endpoint traversal gives a reciprocal scalar approximation without
        # changing that default solver or claiming a more accurate path length.
        undecided = np.ones(len(start), bool)
        for axis in range(3):
            swap |= undecided & (start[:, axis] > end[:, axis])
            undecided &= start[:, axis] == end[:, axis]
        start, end = np.where(swap[:, None], end, start), np.where(swap[:, None], start, end)
    n = len(start)
    distance = np.linalg.norm(end-start, axis=1)
    result = {"length_m": np.full(n, np.inf), "attenuation_db": np.zeros((n, 7)),
              "first": np.zeros((n, 3)), "last": np.zeros((n, 3))}
    box = scene.boxes[box_index]
    routes = [top_route(box, start, end), side_route(box, start, end, 1),
              side_route(box, start, end, -1)]
    # RayCore includes travel along a closed box face. The outdoor top_route
    # helper uses strict below-roof shadow tests, so it has no edge at exact
    # grazing. Supply the N=0 roof limit instead of taking a longer side path.
    roof = box.faces["roof"]
    grazing = ((np.abs((start-roof.center) @ roof.normal) < 1e-12) &
               (np.abs((end-roof.center) @ roof.normal) < 1e-12))
    hit, entry, leave, _, _ = box.segment_slab(start, end)
    grazing &= hit & (leave-entry > 1e-9)
    if grazing.any():
        top = routes[0]
        top["valid"][grazing] = top["blocked"][grazing] = True
        top["V1"][grazing] = (start+entry[:, None]*(end-start))[grazing]
        top["last"][grazing] = (start+leave[:, None]*(end-start))[grazing]
        top["ell"][grazing] = distance[grazing]
        top["n_edges"][grazing] = 2
        top["e_sep"][grazing] = ((leave-entry)*distance)[grazing]
    for route in routes:
        valid = route["valid"] & route["blocked"] & (route["n_edges"] > 0)
        if room_dims_m is not None:
            # A box/wall junction is not a free diffracting edge. Scene routes
            # describe exterior boxes and otherwise allow passage through this
            # zero-width gap (including across an entire sealed partition).
            for vertex in (route["V1"], route["last"]):
                valid &= np.all((vertex > 1e-9) &
                                (vertex < np.asarray(room_dims_m)-1e-9), axis=1)
        valid &= _clear_leg(rays, start, route["V1"], box_index)
        valid &= _clear_leg(rays, route["V1"], route["last"], box_index)
        valid &= _clear_leg(rays, route["last"], end, box_index)
        use = valid & (route["ell"] < result["length_m"])
        loss = route_attenuation_db(np.maximum(route["ell"]-distance, 0.),
                                   np.array(OCTAVE_HZ), sound_speed,
                                   route["e_sep"], route["n_edges"])
        result["length_m"][use] = route["ell"][use]
        result["attenuation_db"][use] = loss[use]
        result["first"][use], result["last"][use] = route["V1"][use], route["last"][use]
    if reciprocal:
        first, last = result['first'], result['last']
        result['first'] = np.where(swap[:, None], last, first)
        result['last'] = np.where(swap[:, None], first, last)
    return result


def first_reflections(model: RoomModel, source, receiver, *, rays: RayCore,
                       scene: Scene, config, frequency_hz=None, spectra=None):
    """Coherent image/BTM face response reduced to octave RMS and earliest delay.

    The existing real-valued octave path interface cannot retain dispersive phase
    between groups. Edge interference *within* each face is evaluated before this
    reduction, including faces with no geometrical specular hit.
    """
    from sensorsim.acoustics.room.synth import ImagePaths
    source, receiver = np.asarray(source), np.asarray(receiver)
    rows, records = [], []
    if config.max_order == 0:
        return None, records
    frequencies = octave_grid(OCTAVE_HZ) if frequency_hz is None else frequency_hz
    for index, box in enumerate(scene.boxes):
        for name, face in box.faces.items():
            ds, dr = (source-face.center) @ face.normal, (receiver-face.center) @ face.normal
            if ds <= 1e-9 or dr <= 1e-9:
                continue
            image = face.mirror(source)
            fraction = dr/(ds+dr)
            point = receiver + fraction*(image-receiver)
            # Test an actual point on the finite face, including outside the
            # specular zone. Full partial-face occlusion remains approximate.
            uv = np.clip(face.uv(point), [-face.half_u, -face.half_v],
                         [face.half_u, face.half_v])
            visible_point = face.center+uv[0]*face.u+uv[1]*face.v
            if not (_clear_leg(rays, source[None], visible_point[None], index)[0] and
                    _clear_leg(rays, visible_point[None], receiver[None], index)[0]):
                continue
            response, length, point, specular = rectangular_response(
                frequencies, source, receiver, face, sound_speed=model.sound_speed_m_s)
            if config.early_time_s is not None and length/model.sound_speed_m_s > config.early_time_s:
                continue
            b = model.obstacles[index]
            wall = np.sqrt((1-np.array(b.absorption))*(1-np.array(b.scattering)))
            if frequency_hz is not None:
                spectra.append(_spectrum(model, frequencies, response*_band_pressure(wall, frequencies),
                                          length, [], config))
                records.append({"box":b.name,"face":name,"point_m":point.tolist(),
                                "distance_m":float(length),"delay_s":float(length/model.sound_speed_m_s),
                                "specular_weight":specular,"edge_model":"BTM isolated finite face; complex"})
                continue
            air = (amplitude_factor(np.array(OCTAVE_HZ), length, model.temperature_c,
                                    model.humidity_pct) if config.air_absorption else np.ones(7))
            measured = config.source_directivity.pattern=='measured' or config.receiver_directivity.pattern=='measured'
            sg = (config.source_directivity.octave_gain if measured else config.source_directivity.gain)(
                ((point-source)/np.linalg.norm(point-source))[None])[0]
            rg = (config.receiver_directivity.octave_gain if measured else config.receiver_directivity.gain)(
                ((point-receiver)/np.linalg.norm(point-receiver))[None])[0]
            wall = wall*octave_amplitude(response, frequencies)*length
            gain = wall*air/length*sg*rg
            rows.append((image, length, wall, gain, sg*rg))
            records.append({"box": b.name, "face": name, "point_m": point.tolist(),
                            "distance_m": float(length), "delay_s": float(length/model.sound_speed_m_s),
                            "specular_weight": specular, "edge_model": "BTM isolated finite face; octave RMS"})
    if not rows:
        return None, records
    images, lengths, walls, gains, directivity = (np.array(x) for x in zip(*rows))
    return ImagePaths(images, np.zeros((len(rows), 6), int), lengths,
                      lengths/model.sound_speed_m_s, walls, gains, gains,
                      directivity), records


def _screen_edges(model, scene, rays, source, receiver, index, frequencies):
    """Thin-box silhouette, excluding edges welded to room walls/floor.

    The two close parallel edges collapse onto the median plane (thin-screen
    approximation). Exterior thick boxes retain the declared detour fallback.
    """
    box = model.obstacles[index]
    size = np.asarray(box.size_xyz)
    thin = int(np.argmin(size))
    axes = [j for j in range(3) if j != thin]
    if size[thin] > .05*min(size[axes]):
        return None
    center = np.asarray(box.min_xyz)+size/2
    normal = np.eye(3)[thin]
    if (source-center) @ normal < 0:
        normal = -normal
    # A same-side field belongs to first_reflections; do not double count it.
    if ((source-center) @ normal)*((receiver-center) @ normal) > 0:
        return None
    response = np.zeros_like(frequencies, complex)
    arrivals = []
    for across, along in (axes, axes[::-1]):
        for sign in (-1, 1):
            origin = center+sign*size[across]/2*np.eye(3)[across]
            if origin[across] <= 1e-9 or origin[across] >= model.shoebox_dims_m[across]-1e-9:
                continue  # a wall contact is not a free edge
            axis, face = np.eye(3)[along], -sign*np.eye(3)[across]
            bounds = (-size[along]/2, size[along]/2)
            *_, apex = edge_geometry(source, receiver, origin, axis)
            point = origin+np.clip(apex, *bounds)*axis
            if not (_clear_leg(rays, source[None], point[None], index)[0] and
                    _clear_leg(rays, receiver[None], point[None], index)[0]):
                continue
            value, length = edge_response(frequencies, source, receiver, origin,
                axis, face, normal, bounds, sound_speed=model.sound_speed_m_s)
            response += value
            arrivals.append(length)
    if not arrivals:
        return None
    return response, min(arrivals)


def _thin_screen_paths(model, scene, rays, paths, source, receiver, counts,
                       lengths, scale, drop, *, frequency_hz=None, spectra=None, config=None):
    """First-order screen field, including lit-side and floor-image edge terms.

    The four source/receiver floor images implement the Neumann image
    construction. Single-floor terms remain one path group; double-floor edge
    terms join the direct group. Thick/multiple-body chains use the fallback.
    """
    if len(model.obstacles) != 1:
        return 0
    source, receiver = np.asarray(source, float), np.asarray(receiver, float)
    frequencies = octave_grid(OCTAVE_HZ) if frequency_hz is None else frequency_hz
    response = _screen_edges(model, scene, rays, source, receiver, 0, frequencies)
    if response is None:
        return 0
    c = model.sound_speed_m_s
    direct_ids = np.flatnonzero(paths.wall_hits.sum(axis=1) == 0)
    floor_ids = np.flatnonzero((paths.wall_hits[:, 4] == 1) &
                               (paths.wall_hits.sum(axis=1) == 1))
    edges, onset = response
    direct_length = np.linalg.norm(source-receiver)
    source_image, receiver_image = source.copy(), receiver.copy()
    source_image[2] *= -1
    receiver_image[2] *= -1
    floor = _band_pressure(np.sqrt((1-np.asarray(model.absorption['z_min']))*
                          (1-np.asarray(model.scattering['z_min']))), frequencies)
    extra = None
    if len(floor_ids) and np.any(floor):
        extra = [_screen_edges(model, scene, rays, a, b, 0, frequencies) for a, b in
                 ((source_image, receiver), (source, receiver_image),
                  (source_image, receiver_image))]
        if extra[2] is not None:
            edges = edges+floor**2*extra[2][0]
    changed = 0
    box = model.obstacles[0]
    size = np.asarray(box.size_xyz)
    thin = int(np.argmin(size))
    axes = [j for j in range(3) if j != thin]
    center = np.asarray(box.min_xyz)+size/2
    fraction = (center[thin]-source[thin])/(receiver[thin]-source[thin])
    crossing = source+fraction*(receiver-source)
    margins = size[axes]/2-abs((crossing-center)[axes])
    boundary = np.any(abs(margins) < 1e-10) and np.all(margins >= -1e-10)
    for i in direct_ids:
        visible = float(counts[i] == 0)
        if boundary:
            visible = 1-np.prod(np.where(abs(margins) < 1e-10, .5, 1.))
        field = edges+visible*np.exp(-2j*np.pi*frequencies*direct_length/c)/direct_length
        lengths[i] = direct_length if visible else onset
        if frequency_hz is None:
            scale[i] = octave_amplitude(field, frequencies)*paths.distance_m[i]
        else:
            spectra.append(_spectrum(model, frequencies, field, lengths[i], [i], config))
            scale[i] = 0.  # replacement, never an additional magnitude path
        drop[i] = False
        changed += 1
    for i in floor_ids:
        if extra is None or extra[0] is None or extra[1] is None:
            continue
        # floor coefficient is already in paths.wall_pressure, once per term.
        length = paths.distance_m[i]
        field = extra[0][0]+extra[1][0]
        if counts[i] == 0:
            field = field+np.exp(-2j*np.pi*frequencies*length/c)/length
        lengths[i] = min(length if counts[i] == 0 else np.inf, extra[0][1], extra[1][1])
        if frequency_hz is None:
            scale[i] = octave_amplitude(field, frequencies)*length
        else:
            spectra.append(_spectrum(model, frequencies, field*floor, lengths[i], [i], config))
            scale[i] = 0.
        drop[i] = False
        changed += 1
    return changed


def _folded_floor_clear(rays, a, b):
    """Visibility of a virtual leg with zero or one floor crossing."""
    a, b = np.asarray(a).copy(), np.asarray(b).copy()
    if a[2]*b[2] < 0:
        point = a+a[2]/(a[2]-b[2])*(b-a)
        a[2], b[2] = abs(a[2]), abs(b[2])
        return bool(_clear_leg(rays, a[None], point[None])[0] and
                    _clear_leg(rays, point[None], b[None])[0])
    a[2], b[2] = abs(a[2]), abs(b[2])
    return bool(_clear_leg(rays, a[None], b[None])[0])


def _body_scattered(model, scene, rays, source, receiver, frequencies):
    """One convex box: visible 270-degree edges and geometrical face images.

    A thick-body shadow requires higher-order edges. Retain the explicitly
    approximate K-A detour there; no fitted correction is used.
    """
    obstacle, box = model.obstacles[0], scene.boxes[0]
    lower, size = np.asarray(obstacle.min_xyz), np.asarray(obstacle.size_xyz)
    center = lower+size/2
    c = model.sound_speed_m_s
    response = np.zeros_like(frequencies, complex)
    arrivals = []
    visible = _folded_floor_clear(rays, source, receiver)
    if not visible:
        # First- and second-order terms cancel in a thick-body shadow (Svensson
        # 1999 section I.C). Do not promote an incomplete first-order sum here.
        route = shortest_detour(scene, rays, source, receiver, 0, c,
                                room_dims_m=model.shoebox_dims_m, reciprocal=frequencies.ndim == 1)
        length = route['length_m'][0]
        if np.isfinite(length) and _folded_floor_clear(rays, source, route['first'][0]) and _folded_floor_clear(
                rays, receiver, route['last'][0]):
            gain = 10**(-route['attenuation_db'][0]/20)/np.linalg.norm(source-receiver)
            response = _band_pressure(gain, frequencies)*np.exp(-2j*np.pi*frequencies*length/c)
            return response, float(length), False, True
        return response, np.inf, False, True
    material = _band_pressure(np.sqrt((1-np.asarray(obstacle.absorption))*
                              (1-np.asarray(obstacle.scattering))), frequencies)
    for face in box.faces.values():
        ds, dr = (source-face.center) @ face.normal, (receiver-face.center) @ face.normal
        if min(ds, dr) <= 1e-9:
            continue
        image = face.mirror(source)
        point = receiver+dr/(ds+dr)*(image-receiver)
        margins = np.array([face.half_u, face.half_v])-abs(face.uv(point))
        weight = np.prod(np.where(abs(margins) < 1e-10, .5, margins > 0))
        if not weight or not (_folded_floor_clear(rays, source, point) and
                                  _folded_floor_clear(rays, receiver, point)):
            continue
        length = np.linalg.norm(image-receiver)
        response += material*weight*np.exp(-2j*np.pi*frequencies*length/c)/length
        arrivals.append(length)
    for along in range(3):
        a, b = [j for j in range(3) if j != along]
        for sa in (-1, 1):
            for sb in (-1, 1):
                origin = center+sa*size[a]/2*np.eye(3)[a]+sb*size[b]/2*np.eye(3)[b]
                if any(origin[j] <= 1e-9 or origin[j] >= model.shoebox_dims_m[j]-1e-9 for j in (a, b)):
                    continue
                axis, face, normal = np.eye(3)[along], -sb*np.eye(3)[b], sa*np.eye(3)[a]
                angles = [np.arctan2((p-origin) @ normal, (p-origin) @ face) % (2*np.pi)
                          for p in (source, receiver)]
                if any(1.5*np.pi+1e-10 < t < 2*np.pi-1e-10 for t in angles):
                    continue  # endpoint-only visibility needs corner diffraction
                bounds = (-size[along]/2, size[along]/2)
                *_, apex = edge_geometry(source, receiver, origin, axis)
                point = origin+np.clip(apex, *bounds)*axis
                if not (_folded_floor_clear(rays, source, point) and
                        _folded_floor_clear(rays, receiver, point)):
                    continue
                value, length = edge_response(frequencies, source, receiver, origin, axis,
                    face, normal, bounds, opening=1.5*np.pi, sound_speed=c)
                response += value
                arrivals.append(length)
    return response, min(arrivals, default=np.inf), visible, False


def _body_paths(model, scene, rays, paths, source, receiver, lengths, scale, drop, *,
                frequency_hz=None, spectra=None, config=None):
    """Coherent near-body field and floor images for a single separated box.

    Cross-body paths need the lit-edge field to interfere with the direct and
    roof image; treating the edge magnitude as another positive reflection
    loses that cancellation. Same-side isolated echoes use first_reflections.
    Multi-box and higher-order image interactions keep the bounded fallback.
    """
    if len(model.obstacles) != 1:
        return None
    source, receiver = np.asarray(source, float), np.asarray(receiver, float)
    obstacle = model.obstacles[0]
    size = np.asarray(obstacle.size_xyz)
    thin = np.argmin(size)
    if size[thin] <= .05*min(np.delete(size, thin)):
        return None
    center = np.asarray(obstacle.min_xyz)+size/2
    if not np.any((source[:2]-center[:2])*(receiver[:2]-center[:2]) < 0):
        return None
    frequencies = octave_grid(OCTAVE_HZ) if frequency_hz is None else frequency_hz
    direct = _body_scattered(model, scene, rays, source, receiver, frequencies)
    floor_ids = np.flatnonzero((paths.wall_hits[:, 4] == 1) & (paths.wall_hits.sum(axis=1) == 1))
    floor = _band_pressure(np.sqrt((1-np.asarray(model.absorption['z_min']))*
                          (1-np.asarray(model.scattering['z_min']))), frequencies)
    field, onset, visible, fallback = direct
    extra = None
    if len(floor_ids) and np.any(floor):
        si, ri = source.copy(), receiver.copy()
        si[2], ri[2] = -si[2], -ri[2]
        extra = [_body_scattered(model, scene, rays, a, b, frequencies)
                 for a, b in ((si, receiver), (source, ri), (si, ri))]
        field = field+floor**2*extra[2][0]
        onset = min(onset, extra[2][1])
    c = model.sound_speed_m_s
    for i in np.flatnonzero(paths.wall_hits.sum(axis=1) == 0):
        length = paths.distance_m[i]
        value = field.copy()
        if visible:
            value += np.exp(-2j*np.pi*frequencies*length/c)/length
            onset = min(onset, length)
        if np.isfinite(onset):
            lengths[i], drop[i] = onset, False
            if frequency_hz is None:
                scale[i] = octave_amplitude(value, frequencies)*length
            else:
                spectra.append(_spectrum(model, frequencies, value, onset, [i], config))
                scale[i] = 0.
    for i in floor_ids:
        if extra is None:
            continue
        length = paths.distance_m[i]
        value = extra[0][0]+extra[1][0]
        first = min(extra[0][1], extra[1][1])
        if extra[0][2] and extra[1][2]:
            value += np.exp(-2j*np.pi*frequencies*length/c)/length
            first = min(first, length)
        if np.isfinite(first):
            lengths[i], drop[i] = first, False
            if frequency_hz is None:
                scale[i] = octave_amplitude(value, frequencies)*length
            else:
                spectra.append(_spectrum(model, frequencies, value*floor, first, [i], config))
                scale[i] = 0.
    return {'model': 'visible 270-degree BTM edges + images; octave energy groups',
            'two_edge_KA_fallback': bool(fallback or (extra and any(e[3] for e in extra)))}


def apply_obstacles(model: RoomModel, paths, source, receiver, config, *, frequency_hz=None):
    """Fast octave paths, or additionally phase-preserving isolated-group spectra.

    Default: (paths, report). With a uniform DC..Nyquist grid: (paths, report,
    spectra), with replaced real paths zeroed. Other image groups retain their
    K-A/octave approximation. Directional complex requests fail explicitly until
    per-secondary-source directivity is supported by the edge owner.
    """
    from sensorsim.acoustics.room.synth import resolve_config
    config = resolve_config(model, config)
    spectra = []
    if frequency_hz is not None:
        frequency_hz = np.asarray(frequency_hz, float)
        if (frequency_hz.ndim != 1 or not 3 <= len(frequency_hz) <= 131073
                or not np.isfinite(frequency_hz).all() or frequency_hz[0] != 0
                or frequency_hz[-1] <= 0 or not np.allclose(np.diff(frequency_hz), frequency_hz[1],
                                                                          rtol=1e-12, atol=0)):
            raise ValueError("obstacle spectrum requires a bounded uniform DC..Nyquist grid")
        if config.source_directivity.pattern != 'omni' or config.receiver_directivity.pattern != 'omni':
            raise ValueError("complex obstacle spectra currently require omni directivity on every leg")
    if not model.obstacles:
        result = paths, {"boxes": 0, "blocked_paths": 0, "dropped_paths": 0}
        return result if frequency_hz is None else (*result, spectra)
    scene = obstacle_scene(model)
    with RayCore(scene, backend=config.obstacle_backend) as rays:
        owner, starts, ends = folded_segments(paths.images_m, receiver, model.shoebox_dims_m)
        blocked = ~_clear_leg(rays, starts, ends)
        lengths = paths.distance_m.copy()
        scale = np.ones_like(paths.wall_pressure)
        drop = np.zeros(len(lengths), bool)
        counts = np.bincount(owner[blocked], minlength=len(lengths))
        rows = np.flatnonzero(blocked & ~drop[owner])
        if len(rows):
            hit = rays.closest_hit(starts[rows], ends[rows])
            for index in np.unique(hit.object_id):
                selected = rows[hit.object_id == index]
                if index < 0:
                    drop[owner[selected]] = True
                    continue
                route = shortest_detour(scene, rays, starts[selected], ends[selected],
                                         int(index), model.sound_speed_m_s,
                                         room_dims_m=model.shoebox_dims_m, reciprocal=frequency_hz is not None)
                valid = np.isfinite(route["length_m"])
                drop[owner[selected[~valid]]] = True
                selected_valid = selected[valid]
                ids = owner[selected_valid]
                delta = route["length_m"][valid]-np.linalg.norm(ends[selected_valid]-starts[selected_valid], axis=1)
                # A floor path may have TWO folded blocked legs belonging to
                # the same image. Fancy-index +=/*= silently loses repeats.
                np.add.at(lengths, ids, delta)
                np.multiply.at(scale, ids, 10**(-route["attenuation_db"][valid]/20))
        screen_paths = _thin_screen_paths(model, scene, rays, paths, source, receiver,
                                           counts, lengths, scale, drop, frequency_hz=frequency_hz,
                                           spectra=spectra, config=config)
        body = _body_paths(model, scene, rays, paths, source, receiver, lengths, scale, drop,
                           frequency_hz=frequency_hz, spectra=spectra, config=config)
        # Preserve original-distance spreading: K-A insertion loss is relative to free field.
        wall = paths.wall_pressure * scale * path_length_compensation(lengths, paths.distance_m)[:, None]
        wall[drop] = 0.
        air = (amplitude_factor(np.array(OCTAVE_HZ), lengths[:, None], model.temperature_c,
                                model.humidity_pct) if config.air_absorption else 1.)
        adjusted = replace(paths, distance_m=lengths, delay_s=lengths/model.sound_speed_m_s,
                           wall_pressure=wall,
                           gain_octave=wall*air/lengths[:, None]*(paths.directivity_gain[:, None]
                               if paths.directivity_gain.ndim==1 else paths.directivity_gain))
        reflections, reflection_records = ((None, []) if body is not None else
            first_reflections(model, source, receiver, rays=rays, scene=scene, config=config,
                               frequency_hz=frequency_hz, spectra=spectra))
        if reflections is not None:
            adjusted = type(paths)(**{f.name: np.concatenate([getattr(adjusted, f.name),
                getattr(reflections, f.name)]) for f in fields(paths)})
        report = {"boxes": len(model.obstacles), "blocked_paths": int(np.count_nonzero(counts)),
                  "dropped_paths": int(drop.sum()), "first_reflections": reflection_records,
                  "ray_backend": rays.backend, "ray_fallback_reason": rays.fallback_reason,
                  "ray_query_policy": "closed finite segments; relative 1e-8 endpoint inset; intervening boxes retained",
                  "btm_screen_paths": screen_paths,
                  "btm_body": body,
                  "diffraction": "shortest clear top/side K-A routes on every blocked folded leg",
                  "spectral_reduction": "coherent BTM within groups; octave RMS + earliest delay between groups",
                  "multi_box_limit": "independent finite faces and folded detours; coherent inter-box scattering omitted",
                  "directivity_limit": "diffracted paths retain original image departure/arrival pattern"}
    if frequency_hz is not None:
        report.update(phase_policy="complex-isolated-groups/1", complex_groups=len(spectra),
            waveform_validity="unvalidated physically; phase/onset/energy adapter checks only",
            spectral_reduction="complex isolated groups; remaining folded images keep octave K-A",
            detour_policy="canonical endpoint traversal of legacy four-iteration K-A routes",
            directivity_limit="omni only; directional complex requests rejected",
            air_policy="ISO magnitude once at each group's earliest length; within-edge air dispersion unresolved")
        return adjusted, report, spectra
    return adjusted, report
