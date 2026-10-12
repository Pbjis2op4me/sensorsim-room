"""Vectorised specular images + prescribed stochastic octave decay (CPU reference).

See README_SYNTH.md for phase, crossfade, scattering and fidelity limitations.
All geometric times exclude DSP padding; returned samples include a declared latency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterator, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator
from scipy import signal

from sensorsim.acoustics.absorption import alpha_db_per_m, amplitude_factor
from sensorsim.acoustics.bands import BandGrid, BandSpec
from sensorsim.acoustics.gain_budget import GainBudget
from sensorsim.acoustics.room.metrics import exact_center, octave_sos
from sensorsim.acoustics.room.directivity import Directivity
from sensorsim.acoustics.room.model import OCTAVE_HZ, SURFACES, EarlyScattering, RoomModel
from sensorsim.core.messages import ModelValidity
from .late import byte_cache, stationary_noise

LATE_BANDS = BandGrid(tuple(BandSpec("base10", 1, exact_center(b), float(b)) for b in OCTAVE_HZ),
                      "room-synthesis-output-envelope/3", "pressure", filter_order=12)

VALIDITY = ModelValidity(
    name="Shoebox image RIR with stochastic octave late reverberation",
    status="partially validated", freq_range_hz=(125., 8000.),
    conditions="Synthetic geometric/decay comparisons only; no measured-room validation. "
    "Diffuse mildly absorptive rooms above modal region. Flat-phase wall reflection; "
    "scattering removes specular energy, diffuse energy uses absorption-only paths. "
    "Independent receiver tails have no validated spatial coherence. CPU reference.",
    citation="Allen & Berkley 1979, doi:10.1121/1.382599; Valimaki et al. 2012, "
    "doi:10.1109/TASL.2012.2189567; parent-project note ROOM_SYNTH_PROTOCOL (not distributed)",
)

SCATTER_VALIDITY = ModelValidity(
    name='Early Lambertian room scattering transport', status='unvalidated',
    freq_range_hz=(125.,8000.),
    conditions='Geometric energy, randomized pressure without measured phase. ISO 17497-1 '
    'random-incidence coefficients used at all incidence angles; Lambert angular prior. '
    'Finite receiver aperture for scattered-then-specular histories; bounded rays/bounces. '
    'No material or measured-room qualification; late field remains a diffuse closure.',
    citation='Savioja and Svensson 2015 doi:10.1121/1.4926438; ISO 17497-1:2004',
)


class SynthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    max_order: int | None = Field(default=None, ge=0, le=60)
    early_time_s: float | None = Field(default=None, gt=0.)
    late: bool = True
    duration_s: float | None = Field(default=None, gt=0.)
    crossfade_s: float = Field(default=.012, gt=0.)
    filter_half_length: int = Field(default=64, ge=32, le=1024)
    air_absorption: bool = True
    source_directivity: Directivity = Field(default_factory=Directivity)
    receiver_directivity: Directivity = Field(default_factory=Directivity)
    early_scattering: EarlyScattering | None = None
    backend: Literal["numpy"] = "numpy"
    obstacle_backend: Literal["auto", "cpu"] = "cpu"
    obstacle_fidelity: Literal["octave", "complex"] = "octave"
    obstacle_fft_length: int | None = Field(default=None, ge=128, le=262144)

    @field_validator("obstacle_backend", mode="before")
    @classmethod
    def _supported_obstacle_backend(cls, value):
        if value == "vulkan":
            raise ValueError("Vulkan backend was removed from sensorsim-room; use cpu or auto")
        return value

    @model_validator(mode="after")
    def _obstacle_grid(self):
        if self.obstacle_fft_length is not None:
            if self.obstacle_fidelity != "complex" or self.obstacle_fft_length % 2:
                raise ValueError("an even obstacle FFT length requires complex fidelity")
        return self

    @model_serializer(mode="wrap")
    def _inheritance(self, handler):
        """Preserve which inheritable fields were omitted versus explicitly set."""
        data = handler(self)
        for key in ("source_directivity", "receiver_directivity", "early_scattering"):
            if key not in self.model_fields_set:
                data.pop(key, None)
        return data


@dataclass(frozen=True)
class ImagePaths:
    images_m: np.ndarray          # (P,3)
    wall_hits: np.ndarray         # (P,6), SURFACES order
    distance_m: np.ndarray        # (P,)
    delay_s: np.ndarray           # (P,)
    wall_pressure: np.ndarray     # (P,7), includes specular scattering loss
    gain_octave: np.ndarray       # (P,7), includes 1/r, air and directivity
    total_gain_octave: np.ndarray # (P,7), absorption only, omnidirectional
    directivity_gain: np.ndarray  # (P,) analytic; (P,7) measured octave pressure


def resolve_config(model: RoomModel, config: SynthConfig) -> SynthConfig:
    """Explicit call fields override declared model devices/transport.

    An explicitly supplied omni (or early_scattering=None) disables the model
    setting. Omitted fields inherit it. Legacy config/model bytes stay unchanged.
    Public seam for the fitter; no device/material parameter is inferred here.
    """
    updates = {key: getattr(model, key) if getattr(model, key) is not None else getattr(config, key)
               for key in ('source_directivity', 'receiver_directivity')
               if key not in config.model_fields_set}
    if 'early_scattering' not in config.model_fields_set and model.early_scattering is not None:
        updates['early_scattering'] = model.early_scattering
    return config.model_copy(update=updates) if updates else config


@dataclass(frozen=True)
class SyntheticRIR:
    fs: int
    samples: np.ndarray
    source_xyz_m: tuple[float, float, float]
    receiver_xyz_m: tuple[float, float, float]
    metadata: dict


def _position(model: RoomModel, xyz) -> np.ndarray:
    p = np.asarray(xyz, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p <= 0) or np.any(
            p >= model.shoebox_dims_m):
        raise ValueError("positions must be finite xyz strictly inside the room")
    if any(box.contains(p) for box in model.obstacles):
        raise ValueError("positions must be outside obstacle volumes and surfaces")
    return p


@byte_cache(32)
def _image_lattice(max_order, order_type):
    """Immutable integer geometry, bounded to 32 MiB of retained cache storage.

    Preserve the original enumeration order for stable distance ties. The scalar
    type is part of the key because NumPy integer types can affect arange's dtype.
    No room, position, material or random state is retained.
    """
    k = np.arange(-max_order, max_order + 1)
    grid = np.stack(np.meshgrid(k, k, k, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.abs(grid).sum(axis=1) <= max_order]
    parity = grid % 2
    n = (grid + parity) // 2
    hits = np.stack([np.abs(n-parity), np.abs(n)], axis=-1).reshape(-1, 6)
    for array in (n, parity, hits):
        array.setflags(write=False)
    return n, parity, hits


def image_paths(model: RoomModel, source_xyz_m, receiver_xyz_m, *, max_order: int,
                time_limit_s: float | None = None, air_absorption: bool = True,
                source_directivity: Directivity | None = None,
                receiver_directivity: Directivity | None = None) -> ImagePaths:
    """Shoebox lattice 2*n*L + (-1)**p*s, order=sum(abs(2*n-p)).

    Vectorised arrays are the future torch seam. No per-ray Python traversal.
    Surface scattering s leaves coherent pressure sqrt((1-alpha)*(1-s)).
    Total energy used by the tail excludes this loss because scattering redirects it.
    """
    source, receiver = _position(model, source_xyz_m), _position(model, receiver_xyz_m)
    if np.linalg.norm(source - receiver) < 1e-6:
        raise ValueError("source and receiver must be at least 1 micrometre apart")
    if not isinstance(max_order, (int, np.integer)) or not 0 <= max_order <= 60:
        raise ValueError("max_order must be an integer in [0,60]")
    if time_limit_s is not None and (not np.isfinite(time_limit_s) or time_limit_s <= 0):
        raise ValueError("time_limit_s must be finite and positive")
    n, parity, hits = _image_lattice(max_order, type(max_order))
    images = 2 * n * np.array(model.shoebox_dims_m) + (1 - 2*parity) * source
    distance = np.linalg.norm(images - receiver, axis=1)
    keep = np.ones(len(distance), dtype=bool) if time_limit_s is None else (
        distance <= time_limit_s * model.sound_speed_m_s)
    images, hits, parity, distance = images[keep], hits[keep], parity[keep], distance[keep]
    order = np.argsort(distance, kind="stable")
    images, hits, parity, distance = images[order], hits[order], parity[order], distance[order]
    absorption = np.array([model.absorption[s] for s in SURFACES])
    scattering = np.array([model.scattering[s] for s in SURFACES])
    # pow(0,0)=1 correctly includes perfectly absorbing walls not hit by a path.
    wall_total = np.prod((1-absorption)[None, :, :] ** hits[:, :, None], axis=1)**.5
    wall = wall_total * np.prod((1-scattering)[None, :, :] ** hits[:, :, None], axis=1)**.5
    to_image = (images - receiver) / distance[:, None]
    src = source_directivity or model.source_directivity or Directivity()
    rec = receiver_directivity or model.receiver_directivity or Directivity()
    if src.pattern == 'measured' or rec.pattern == 'measured':
        directivity = src.octave_gain(-to_image * (1-2*parity)) * rec.octave_gain(to_image)
    else:
        directivity = src.gain(-to_image * (1-2*parity)) * rec.gain(to_image)
    air = (amplitude_factor(np.array(OCTAVE_HZ), distance[:, None], model.temperature_c,
                            model.humidity_pct) if air_absorption else 1.)
    total = wall_total * air / distance[:, None]
    return ImagePaths(images, hits, distance, distance/model.sound_speed_m_s, wall,
                      wall * air / distance[:, None] * (directivity[:, None] if directivity.ndim==1 else directivity), total, directivity)


def _render_paths(model: RoomModel, paths: ImagePaths, fs: int, length: int,
                  config: SynthConfig) -> np.ndarray:
    """Coherent sum of the same linear path kernels cached by room.fit."""
    out = np.zeros(length)
    for sl, indices, kernel in path_kernels(model, paths, fs, config):
        valid = (indices >= 0) & (indices < length)
        np.add.at(out, indices[valid], kernel[valid])
    return out


@lru_cache(maxsize=16)
def _path_kernel_grid(fs: int, half: int):
    """Bounded immutable DSP grid; no room/path/RNG values are cached."""
    nfft = 2 ** int(np.ceil(np.log2(8 * (2*half+1))))
    freq = np.fft.rfftfreq(nfft, 1/fs)
    # Linear pressure interpolation on log frequency; flat outside octave centres.
    weights = np.stack([np.interp(np.log2(np.maximum(freq, 1)), np.log2(OCTAVE_HZ), row)
                        for row in np.eye(7)])
    window = np.kaiser(2*half+1, 8.6)
    offset = np.arange(-half, half+1)
    arrays = (freq, weights, window, offset)
    for array in arrays:
        array.setflags(write=False)
    return nfft, *arrays


def path_kernels(model: RoomModel, paths: ImagePaths, fs: int, config: SynthConfig,
                 *, wall_pressure: np.ndarray | None = None):
    """Chunked linear pressure kernels, including air, spreading and fractional delay.

    The fitter caches these kernels, retaining early interference and octave
    interpolation. Windowing is linear; spectrum-dependent DC rescaling would
    break that identity and colour every band according to an unrelated DC bin.
    The finite Kaiser kernel's flat-path error is covered by the free-field gate.
    """
    half = config.filter_half_length
    nfft, freq, weights, window, offset = _path_kernel_grid(fs, half)
    pressure = paths.wall_pressure if wall_pressure is None else wall_pressure
    for start in range(0, len(paths.distance_m), 128):
        sl = slice(start, start+128)
        delay = paths.delay_s[sl] * fs
        integer = np.floor(delay).astype(int)
        transfer = pressure[sl] @ weights
        if config.air_absorption:
            transfer *= amplitude_factor(freq, paths.distance_m[sl, None],
                                         model.temperature_c, model.humidity_pct)
        if paths.directivity_gain.ndim == 1:
            transfer *= (paths.directivity_gain[sl] / paths.distance_m[sl])[:, None]
        else:
            transfer *= (paths.directivity_gain[sl] @ weights)/paths.distance_m[sl,None]
        impulse = np.fft.fftshift(np.fft.irfft(
            transfer * np.exp(-2j*np.pi*(delay-integer)[:, None]*freq/fs), nfft), axes=-1)
        kernel = impulse[:, nfft//2-half:nfft//2+half+1] * window
        indices = integer[:, None] + half + offset
        yield sl, indices, kernel


def crossfade_fraction(time, mixing: float, fade: float):
    return np.clip((np.asarray(time) - (mixing - fade/2))/fade, 0., 1.)


def late_envelope(time, mixing: float, fade: float, t60):
    """Pressure envelope AFTER stationary causal synthesis filtering.

    The same deterministic envelope is used by the inverse covariance model.
    Hidden noise warmup never appears before the physical handoff.
    """
    exponent, fade_gain, _ = _late_envelope_parts(time, mixing, fade)
    with np.errstate(over="raise", invalid="raise"):
        return np.exp(exponent/t60) * fade_gain


def _late_envelope_parts(time, mixing, fade):
    time = np.asarray(time)
    fraction = crossfade_fraction(time, mixing, fade)
    with np.errstate(over="raise", invalid="raise"):
        return (-3*np.log(10)*np.maximum(time-mixing, -fade/2),
                np.sin(np.pi/2*fraction), fraction)


def active_octaves(fs: int) -> np.ndarray:
    """Octaves analysed at this rate: complete bands below Nyquist."""
    return np.array([exact_center(b)*10**.15 < fs/2 for b in OCTAVE_HZ])


@dataclass(frozen=True)
class LateBand:
    """One late-tail synthesis band: its filter, the octave whose decay and level it follows, and the
    frequency at which its air absorption is taken."""
    index: int
    air_hz: float
    kind: Literal["lowpass", "bandpass", "highpass", "extension"]
    sos: np.ndarray


@lru_cache(maxsize=16)
def late_bands(fs: int) -> tuple[LateBand, ...]:
    """Bands that tile the late tail from DC to Nyquist (late-band-coverage/2).

    The lowest octave is a low-pass: below 125 Hz the tail decays like 125 Hz, as early wall gains
    are flat below it. A top octave cut by Nyquist becomes a high-pass. Above a complete top octave,
    an extension band reaches Nyquist with the top octave's wall absorption and the air absorption
    of the next octave up, where it is much stronger. All skirts are 6th-order Butterworth, like
    the octave band-passes, so neighbouring bands cross at -3 dB and their noise powers add flat.
    """
    nyquist = fs/2
    centres = [exact_center(b) for b in OCTAVE_HZ]
    usable = [i for i, c in enumerate(centres) if c*10**-.15 < .95*nyquist]
    bands = []
    for i in usable:
        lower, upper = centres[i]*10**-.15, centres[i]*10**.15
        if i == usable[0]:
            band = LateBand(i, float(OCTAVE_HZ[i]), "lowpass", signal.butter(6, upper, "lowpass", fs=fs, output="sos"))
        elif i == usable[-1] and upper >= .9*nyquist:
            band = LateBand(i, float(OCTAVE_HZ[i]), "highpass", signal.butter(6, lower, "highpass", fs=fs, output="sos"))
        else:
            band = LateBand(i, float(OCTAVE_HZ[i]), "bandpass", octave_sos(fs, OCTAVE_HZ[i]))
        bands.append(band)
    edge = centres[usable[-1]]*10**.15
    if bands[-1].kind == "bandpass" and edge < .9*nyquist:
        bands.append(LateBand(usable[-1], 2.*OCTAVE_HZ[usable[-1]], "extension",
                              signal.butter(6, edge, "highpass", fs=fs, output="sos")))
    return tuple(bands)


def late_band_t60(bands, t60, model: RoomModel, air_absorption: bool) -> np.ndarray:
    """Decay time of each late band. An extension adds the extra air loss above its octave:
    1/T = 1/T_octave + c dm / (6 ln 10), with dm the ISO 9613-1 energy attenuation difference per metre."""
    out = []
    for band in bands:
        value = float(t60[band.index])
        if band.kind == "extension" and air_absorption:
            extra = (alpha_db_per_m(band.air_hz, model.temperature_c, model.humidity_pct)
                     - alpha_db_per_m(OCTAVE_HZ[band.index], model.temperature_c, model.humidity_pct))*np.log(10)/10
            value = 1/(1/value + model.sound_speed_m_s*float(extra)/(6*np.log(10)))
        out.append(value)
    return np.array(out)


def extension_gain_ratio(band: LateBand, distance_m, model: RoomModel, air_absorption: bool):
    """Squared pressure ratio of an extension band to its octave for paths of these lengths."""
    if band.kind != "extension" or not air_absorption:
        return np.ones_like(np.asarray(distance_m, float))
    ratio = (amplitude_factor(band.air_hz, distance_m, model.temperature_c, model.humidity_pct)
             / amplitude_factor(OCTAVE_HZ[band.index], distance_m, model.temperature_c, model.humidity_pct))
    return ratio**2


def late_band_levels(paths: ImagePaths, mixing: float, fade: float, bands, model: RoomModel,
                     air_absorption: bool) -> np.ndarray:
    """Absorption-only squared-pressure density at mixing for each late band."""
    mask = (abs(paths.delay_s-mixing) <= fade/2) & (paths.wall_hits.sum(axis=1) > 0)
    if not mask.any():
        raise ValueError("no image arrivals near mixing; widen crossfade or adjust mixing time")
    return np.array([np.sum(paths.total_gain_octave[mask, b.index]**2
                            * extension_gain_ratio(b, paths.distance_m[mask], model, air_absorption))/fade
                     for b in bands])


def render_late(bands, levels, t60, time, mixing, fade, fs, seed):
    """Independent warmed causal bands, followed by the output envelope."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(time))
    parts = None
    for band, level, decay in zip(bands, levels, t60):
        if level == 0:
            continue
        try:
            # Time and crossfade are shared by every band. Keep evaluation lazy
            # for all-zero levels, and preserve the per-band arithmetic/RNG order.
            if parts is None:
                parts = _late_envelope_parts(time, mixing, fade)
            exponent, fade_gain, _ = parts
            with np.errstate(over="raise", invalid="raise"):
                envelope = np.exp(exponent/decay) * fade_gain
            out += stationary_noise(band.sos, len(time), fs, rng)*envelope*np.sqrt(level)
        except FloatingPointError as exc:
            raise ValueError("late decay across the crossfade exceeds numerical dynamic range") from exc
    return out


def scatter_energy_split(incoming, absorption, scattering):
    """ISO 17497-1 boundary ledger: absorbed, specular, scattered energy.

    Directional diffusion (ISO 17497-2) is not interchangeable with scattering.
    This exact boundary identity does not assert conserved coherent RIR energy.
    """
    e,a,s=np.asarray(incoming,float),np.asarray(absorption,float),np.asarray(scattering,float)
    if (not all(np.isfinite(x).all() for x in (e,a,s)) or np.any(e<0) or
            np.any((a<0)|(a>1)) or np.any((s<0)|(s>1))):
        raise ValueError('energy must be nonnegative; absorption/scattering must lie in [0,1]')
    reflected=e*(1-a)
    return e*a,reflected*(1-s),reflected*s


def scattering_energy(model: RoomModel, source, receiver, fs: int, *,
                      config: SynthConfig, seed: int=0):
    """Bounded ISM/ray hybrid: seven prefilter squared-pressure echograms.

    All-specular histories belong exclusively to ISM. Diffuse-rain connections
    sample histories whose LAST bounce is diffuse; receiver-sphere intersections
    sample scattered histories whose last bounce is specular. Those disjoint
    estimators avoid counting a diffuse ray both as rain and as a sphere hit.
    Importance-weighted branch selection carries each frequency's own coefficient.
    Uniform solid-angle source sampling has energy 4*pi/N for 1/r pressure.

    Only geometry before early_time (or mixing+fade/2) is transported. Synthesis
    applies the existing cos/sin power crossfade to this field and late closure;
    it never adds these arrivals to the absorption-only late normalization.
    """
    from scipy.stats import qmc
    from sensorsim.acoustics.room.obstacles import obstacle_scene, transport_hits, _clear_leg
    from sensorsim.scene.raycore import RayCore
    config=resolve_config(model,config)
    settings=config.early_scattering
    if settings is None: raise ValueError('declare early_scattering explicitly')
    if not isinstance(fs,(int,np.integer)) or not 1000<=fs<=192000: raise ValueError('invalid scattering sample rate')
    source,receiver=_position(model,source),_position(model,receiver)
    if np.linalg.norm(source-receiver)<1e-6: raise ValueError('source and receiver must differ')
    cutoff=config.early_time_s or model.mixing_time()+config.crossfade_s/2
    if cutoff*fs>5_000_000: raise ValueError('scattering exceeds bounded sample budget')
    count=int(np.ceil(cutoff*fs))+1
    histogram=np.zeros((count,7))
    report={'method':settings.method,'rays':settings.rays,'max_bounces':settings.max_bounces,
        'receiver_radius_m':settings.receiver_radius_m,'cutoff_s':cutoff,
        'boundary_closure_max_relative':0.,'connections':0,'specular_receiver_hits':0,
        'bounce_limit_remaining_energy': [0.]*7,
        'status':SCATTER_VALIDITY.status,'validity':SCATTER_VALIDITY.conditions,
        'citation':SCATTER_VALIDITY.citation,
        'partition':'ISM pure specular; diffuse rain last-diffuse; sphere last-specular after scattering',
        'transition':'early cos^2 and legacy late sin^2; unchanged absorption-only late density',
        'diffraction':'geometric occlusion only in diffuse transport; finite-edge field belongs to ISM/BTM'}
    if not any(np.any(row) for row in model.scattering.values()) and not any(np.any(b.scattering) for b in model.obstacles):
        report['prefilter_energy_octave']=[0.]*7
        return histogram,report
    radius=settings.receiver_radius_m
    if min(np.min(receiver),np.min(np.asarray(model.shoebox_dims_m)-receiver))<=radius or any(
            b.contains(receiver,radius) for b in model.obstacles):
        raise ValueError('scattering receiver sphere must fit in free room volume')
    scene=obstacle_scene(model)
    rng=np.random.default_rng(np.random.SeedSequence([seed,0x52415953]))
    uv=qmc.Sobol(2,scramble=True,seed=int(rng.integers(2**32))).random_base2(int(np.log2(settings.rays)))
    z=1-2*uv[:,0];az=2*np.pi*uv[:,1]
    directions=np.stack([np.sqrt(1-z*z)*np.cos(az),np.sqrt(1-z*z)*np.sin(az),z],axis=1)
    energy=config.source_directivity.octave_gain(directions)**2*(4*np.pi/settings.rays)
    starts=np.broadcast_to(source,(settings.rays,3)).copy()
    travelled=np.zeros(settings.rays)
    diffuse_seen=np.zeros(settings.rays,bool);last_diffuse=diffuse_seen.copy()
    limit=cutoff*model.sound_speed_m_s
    def air(distance):
        return amplitude_factor(np.array(OCTAVE_HZ),distance[:,None],model.temperature_c,
            model.humidity_pct)**2 if config.air_absorption else np.ones((len(distance),7))
    def deposit(distance,values):
        sample=np.floor(distance/model.sound_speed_m_s*fs+.5).astype(int)
        inside=(sample>=0)&(sample<count)&(distance<=limit)
        np.add.at(histogram,sample[inside],values[inside])
        return int(np.count_nonzero(inside & np.any(values>0,axis=1)))
    with RayCore(scene,backend=config.obstacle_backend) as rays:
        report['ray_backend']=rays.backend
        for bounce in range(settings.max_bounces):
            if not len(energy): break
            length,normal,alpha,scatter=transport_hits(model,scene,rays,starts,directions)
            # Aperture estimator only for histories absent from diffuse rain.
            delta=receiver-starts;projection=np.sum(delta*directions,axis=1)
            perpendicular=np.sum(delta*delta,axis=1)-projection**2
            near=(diffuse_seen & ~last_diffuse & (projection>0) & (projection<length) & (perpendicular<radius**2))
            ids=np.flatnonzero(near)
            if len(ids):
                to_receiver=np.linalg.norm(delta[ids],axis=1)
                values=energy[ids]*air(to_receiver)*config.receiver_directivity.octave_gain(-directions[ids])**2/(np.pi*radius**2)
                report['specular_receiver_hits']+=deposit(travelled[ids]+to_receiver,values)
            travelled+=length
            live=(travelled<limit)&np.any(energy>0,axis=1)
            if not live.any(): break
            point=(starts+length[:,None]*directions)[live]
            energy=energy[live]*air(length[live]);normal=normal[live]
            alpha,scatter=alpha[live],scatter[live]
            travelled=travelled[live];directions=directions[live];diffuse_seen=diffuse_seen[live]
            absorbed,specular,diffuse=scatter_energy_split(energy,alpha,scatter)
            closure=np.max(abs(absorbed+specular+diffuse-energy)/np.maximum(energy,1e-300))
            report['boundary_closure_max_relative']=max(report['boundary_closure_max_relative'],float(closure))
            toward=receiver-point;distance=np.linalg.norm(toward,axis=1)
            outgoing=toward/distance[:,None];cosine=np.maximum(0,np.sum(outgoing*normal,axis=1))
            visible=_clear_leg(rays,point, np.broadcast_to(receiver,point.shape))
            values=diffuse*(cosine*visible/(np.pi*distance**2))[:,None]*air(distance)
            values*=config.receiver_directivity.octave_gain(-outgoing)**2
            report['connections']+=deposit(travelled+distance,values)
            reflected=specular+diffuse
            total=reflected.sum(axis=1)
            q=np.divide(diffuse.sum(axis=1),total,out=np.zeros_like(total),where=total>0)
            choose=rng.random(len(q))<q
            energy=np.where(choose[:,None],
                np.divide(diffuse,q[:,None],out=np.zeros_like(diffuse),where=q[:,None]>0),
                np.divide(specular,1-q[:,None],out=np.zeros_like(specular),where=q[:,None]<1))
            directions=directions-2*np.sum(directions*normal,axis=1)[:,None]*normal
            if choose.any():
                n=normal[choose];helper=np.eye(3)[np.argmin(abs(n),axis=1)]
                tangent=np.cross(n,helper);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
                u,v=rng.random((2,int(choose.sum())))
                directions[choose]=(np.sqrt(1-u)[:,None]*n+np.sqrt(u)[:,None]*(
                    np.cos(2*np.pi*v)[:,None]*tangent+np.sin(2*np.pi*v)[:,None]*np.cross(n,tangent)))
            diffuse_seen|=choose;last_diffuse=choose
            starts=point+1e-7*normal
        else:
            report['bounce_limit_remaining_energy']=energy.sum(axis=0).tolist()
    report['prefilter_energy_octave']=histogram.sum(axis=0).tolist()
    return histogram,report


def late_level_density(paths: ImagePaths, mixing: float, fade: float) -> np.ndarray:
    """Absorption-only squared-pressure impulse density per second at mixing."""
    mask = (abs(paths.delay_s-mixing) <= fade/2) & (paths.wall_hits.sum(axis=1) > 0)
    if not mask.any():
        raise ValueError("no image arrivals near mixing; widen crossfade or adjust mixing time")
    return np.sum(paths.total_gain_octave[mask]**2, axis=0)/fade


def effective_mixing_time(model: RoomModel, source, receiver, fade: float, *, late: bool = True
                          ) -> tuple[float, dict | None]:
    """Resolve the pair handoff; only the default estimate may receive a margin."""
    original = model.mixing_time()
    if not late:
        return original, None
    direct = float(np.linalg.norm(np.asarray(source)-np.asarray(receiver))/model.sound_speed_m_s)
    fade_start = direct + fade/2
    if model.mixing_time_s is not None:
        if original <= fade_start:
            raise ValueError("crossfade must start after the direct arrival; increase mixing_time_s")
        return original, None
    effective = max(original, fade_start + 1/model.sound_speed_m_s)
    adjustment = ({"applied": True, "original_mixing_time_s": original,
                   "effective_mixing_time_s": effective} if effective != original else None)
    return effective, adjustment


def early_paths(model: RoomModel, source, receiver, config: SynthConfig, *, frequency_hz=None):
    """One path-coverage and obstacle convention for fitting and synthesis."""
    config = resolve_config(model, config)
    mixing, _ = effective_mixing_time(model, source, receiver, config.crossfade_s, late=config.late)
    fade = config.crossfade_s
    end = config.early_time_s or (mixing + fade/2)
    required_order = int(np.ceil(end * model.sound_speed_m_s * np.linalg.norm(
        1/np.array(model.shoebox_dims_m)))) + 3
    order = config.max_order if config.max_order is not None else required_order
    if config.late and (end < mixing+fade/2 or order < required_order):
        raise ValueError("late synthesis requires complete crossfade image coverage; "
                         "use max_order=None/early_time_s=None or late=False")
    limit = end if config.late or config.early_time_s is not None or config.max_order is None else None
    anchor = image_paths(model, source, receiver, max_order=order, time_limit_s=limit,
                         air_absorption=config.air_absorption,
                         source_directivity=config.source_directivity,
                         receiver_directivity=config.receiver_directivity)
    if not len(anchor.delay_s):
        raise ValueError("early time limit excludes the direct arrival")
    from sensorsim.acoustics.room.obstacles import apply_obstacles
    result = apply_obstacles(model, anchor, source, receiver, config, frequency_hz=frequency_hz)
    paths, report = result[:2]
    base = paths, anchor, report, order
    return base if frequency_hz is None else (*base, result[2])


def _render_obstacle_spectra(spectra, fs, length, latency):
    """Linear placement of complex group responses with explicit finite support.

    The full transfer, including its imaginary part, goes through the real IFFT.
    The same finite pre-arrival guard as ordinary paths carries bandlimited
    ringing; no magnitude projection, short FIR, or circular room convolution.
    """
    out = np.zeros(length)
    diagnostics = []
    for spectrum in spectra:
        nfft = 2*(len(spectrum.frequency_hz)-1)
        support = int(np.ceil(spectrum.max_delay_s*fs))+2*latency+1
        if support > nfft:
            raise ValueError("obstacle FFT grid is too short for physical response support")
        onset = spectrum.time_origin_s*fs
        integer = int(np.floor(onset))
        shifted = spectrum.transfer*np.exp(-2j*np.pi*spectrum.frequency_hz*(onset-integer+latency)/fs)
        impulse = np.fft.irfft(shifted, nfft)
        count = min(support, length-integer)
        if count < support:
            raise ValueError("output would truncate obstacle response support")
        out[integer:integer+count] += impulse[:count]
        diagnostics.append({"time_origin_s":spectrum.time_origin_s,
            "max_delay_s":spectrum.max_delay_s,"fft_length":nfft,
            "replaces_image_indices":spectrum.replaces_image_indices.tolist(),
            "discarded_fft_energy_fraction":float(np.sum(impulse[count:]**2)/max(np.sum(impulse**2),1e-300))})
    return out, diagnostics


def synthesize(model: RoomModel, source_xyz_m, receiver_xyz_m, fs: int, *, seed: int = 0,
               config: SynthConfig | None = None) -> SyntheticRIR:
    """Render one RIR; sample time is n/fs minus metadata['latency_samples']/fs.

    With late=True, order/time must cover the entire crossfade. Auto order is a
    conservative geometric bound; explicit truncation that leaves gaps is rejected.
    The returned length includes the prescribed decay and short DSP padding.
    """
    config = resolve_config(model, config or SynthConfig())
    model = RoomModel.model_validate(model.model_dump())  # recheck mutable dict leaves
    if not isinstance(fs, (int, np.integer)) or not 1000 <= fs <= 192000:
        raise ValueError("fs must be an integer in [1000,192000]")
    source, receiver = _position(model, source_xyz_m), _position(model, receiver_xyz_m)
    mixing, mixing_adjustment = effective_mixing_time(model, source, receiver, config.crossfade_s,
                                                       late=config.late)
    fade = config.crossfade_s
    spectra = []
    if config.obstacle_fidelity == 'complex' and model.obstacles:
        minimum = int(np.ceil(8*np.linalg.norm(model.shoebox_dims_m)/model.sound_speed_m_s*fs))
        nfft = config.obstacle_fft_length or 2**int(np.ceil(np.log2(minimum+2*config.filter_half_length+1)))
        if nfft > 262144:
            raise ValueError("obstacle spectrum exceeds the bounded 262144-sample FFT budget")
        frequency_hz = np.fft.rfftfreq(nfft,1/fs)
        paths, anchor_paths, obstacle_report, order, spectra = early_paths(
            model, source, receiver, config, frequency_hz=frequency_hz)
    else:
        paths, anchor_paths, obstacle_report, order = early_paths(model, source, receiver, config)
    decay = model.decay_budget(air_absorption=config.air_absorption)
    t60 = decay.t60_s
    active = active_octaves(fs)
    bands = late_bands(fs)
    if config.late and (not np.isfinite(t60).all() or np.any(t60 <= 0)):
        raise ValueError("late decay must be finite and positive; supply late_t60_s or disable late")
    band_t60 = late_band_t60(bands, t60, model, config.air_absorption)
    if config.late and 3*np.log(10)*fade/(2*min(band_t60)) > np.log(np.finfo(float).max)-1:
        raise ValueError("late decay across the crossfade exceeds numerical dynamic range")
    duration = (mixing + 1.2 * max(band_t60) if config.late else float(paths.delay_s.max()))
    if config.early_scattering is not None:
        duration = max(duration, config.early_time_s or mixing+fade/2)
    if spectra:
        duration = max(duration, max(s.time_origin_s+s.max_delay_s for s in spectra))
    if config.duration_s is not None:
        if config.duration_s < duration:
            raise ValueError("duration_s would truncate the required RIR decay/arrivals")
        duration = config.duration_s
    if not np.isfinite(duration*fs) or duration*fs + 2*config.filter_half_length + 1 > 5_000_000:
        raise ValueError("RIR exceeds bounded five-million-sample budget")
    length = int(np.ceil(duration*fs)) + 2*config.filter_half_length + 1
    early = _render_paths(model, paths, fs, length, config)
    scatter_report = None
    if config.early_scattering is not None:
        energy, scatter_report = scattering_energy(model, source, receiver, fs, seed=seed, config=config)
        # Independent stream: enabling scattering cannot move the legacy late RNG.
        rng_scatter = np.random.default_rng(np.random.SeedSequence([seed, 0x53434154]))
        latency = config.filter_half_length
        for b, centre in enumerate(OCTAVE_HZ):
            if not active[b] or not energy[:,b].any(): continue
            pressure = np.zeros(length)
            n = min(len(energy), length-latency)
            pressure[latency:latency+n] = rng_scatter.standard_normal(n)*np.sqrt(energy[:n,b])
            early += signal.sosfilt(octave_sos(fs,centre), pressure)
    if spectra:
        complex_early, spectrum_report = _render_obstacle_spectra(spectra, fs, length, config.filter_half_length)
        early += complex_early
        obstacle_report['spectra'] = spectrum_report
        obstacle_report['spectral_endpoint_policy'] = 'real DC/Nyquist projection for sampled real pressure'
    time = (np.arange(length)-config.filter_half_length)/fs
    levels = np.zeros(len(bands))
    late = np.zeros(length)
    if config.late:
        levels = late_band_levels(anchor_paths, mixing, fade, bands, model, config.air_absorption)/fs
        power = np.broadcast_to(np.asarray(config.source_directivity.diffuse_power, float) *
                                config.receiver_directivity.diffuse_power, (7,))
        levels *= power[[band.index for band in bands]]
        late = render_late(bands, levels, band_t60, time, mixing, fade, fs, seed)
        fraction = crossfade_fraction(time, mixing, fade)
        early *= np.cos(np.pi/2*fraction)
    metadata = {
        "method": "shoebox-vector-ism+stochastic-octaves" if config.late else "shoebox-vector-ism",
        "model_version": model.version, "model_hash": model.model_hash, "seed": int(seed),
        "config": config.model_dump(mode="json"), "resolved_max_order": order,
        "sound_speed_m_s": model.sound_speed_m_s, "sound_speed_policy": model.sound_speed_policy,
        "decay_budget": decay.metadata(), "late_band_grid": LATE_BANDS.as_dict(),
        "late_normalization": "absorption-only image pressure-squared density at mixing; no draw normalization",
        "rng_policy": "numpy-pcg64-stationary-warmup/2",
        "gain_budget": asdict(GainBudget("room-path-kernel/2", "pressure", "exp(+iwt)", 1., True,
                                        config.air_absorption, "path_length" if config.air_absorption else "none")),
        "time_grid": "absolute sample n/fs minus DSP latency; linear zero-padded finite Kaiser kernels",
        "path_count": len(paths.delay_s), "latency_samples": config.filter_half_length,
        "direct_delay_s": float(np.linalg.norm(source-receiver)/model.sound_speed_m_s),
        "first_path_delay_s": float(paths.delay_s.min()), "mixing_time_s": mixing,
        "crossfade_s": fade, "t60_target_s": [float(x) if np.isfinite(x) else None for x in t60],
        "modal_region": model.modal_region(),
        "active_late_octave_hz": np.array(OCTAVE_HZ)[active].tolist(),
        "late_bands": [{"kind": b.kind, "decay_octave_hz": OCTAVE_HZ[b.index], "air_hz": b.air_hz,
                        "t60_s": float(t)} for b, t in zip(bands, band_t60)],
        "late_broadband_variance_at_mixing": levels.tolist(),
        "forward_physics": "linear-path-kernels+output-envelope-covariance/3",
        "status": VALIDITY.status, "validity": VALIDITY.conditions,
        "obstacles": obstacle_report,
        "outside_band_policy": ("early wall gains flat below 125/above 8000 Hz; late tail DC..Nyquist "
                                "(late-band-coverage/2): lowest octave low-pass, a top octave cut by Nyquist "
                                "high-pass, above a complete top octave an extension band with its air loss"),
    }
    from sensorsim.licensing import lineage_metadata
    lineage = lineage_metadata(model, config)
    if lineage:
        metadata['lineage'] = lineage
    if mixing_adjustment is not None:
        metadata["mixing_time_adjustment"] = mixing_adjustment
    if scatter_report is not None:
        metadata['early_scattering'] = scatter_report
        metadata['status'] = SCATTER_VALIDITY.status
        metadata['validity'] += ' Opt-in scattering: '+SCATTER_VALIDITY.conditions
    if config.source_directivity.pattern=='measured' or config.receiver_directivity.pattern=='measured':
        metadata['directivity_transfer'] = 'octave magnitude; on-axis colour removed; no measured phase/delay'
    samples = early+late
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > np.finfo(np.float32).max:
        raise ValueError("RIR early/late decay exceeds finite float32 audio range")
    return SyntheticRIR(int(fs), samples.astype(np.float32), tuple(source), tuple(receiver), metadata)


def iter_synthesize(model: RoomModel, sources_xyz_m, receivers_xyz_m, fs: int, *, seed: int = 0,
                    config: SynthConfig | None = None) -> Iterator[SyntheticRIR]:
    """Paired batch; one source or receiver may broadcast. O(one RIR) output memory."""
    sources, receivers = np.asarray(sources_xyz_m, float), np.asarray(receivers_xyz_m, float)
    sources, receivers = np.atleast_2d(sources), np.atleast_2d(receivers)
    if sources.shape[1:] != (3,) or receivers.shape[1:] != (3,):
        raise ValueError("positions must have shape (N,3) or (3,)")
    n = max(len(sources), len(receivers))
    if min(len(sources), len(receivers)) == 0 or len(sources) not in (1, n) or len(receivers) not in (1, n):
        raise ValueError("paired batch lengths must match or broadcast one position")
    children = np.random.SeedSequence(seed).spawn(n)
    for i, child in enumerate(children):
        yield synthesize(model, sources[i % len(sources)], receivers[i % len(receivers)], fs,
                         seed=int(child.generate_state(1)[0]), config=config)


def synthesize_batch(model: RoomModel, sources_xyz_m, receivers_xyz_m, fs: int, **kwargs
                     ) -> list[SyntheticRIR]:
    """Convenience materialised paired batch; use iter_synthesize for long jobs."""
    return list(iter_synthesize(model, sources_xyz_m, receivers_xyz_m, fs, **kwargs))
