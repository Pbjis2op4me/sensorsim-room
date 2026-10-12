"""Measured-position room inversion; see README_FIT.md and the frozen protocol.

The optimizer caches room.synth's coherent early kernels and the expected
filtered-noise tail energy. Final predictions use room.synth. Parameters are effective
and conditional on geometry, fixed scattering and the device response.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
from typing import Iterable, Literal
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal
from scipy.fft import next_fast_len, irfft
from scipy.optimize import least_squares, linear_sum_assignment
import yaml

from sensorsim.core.messages import ModelValidity
from .ingest import load_manifest
from .late import ByteCache, byte_cache, prepare_tail
from .metrics import exact_center, octave_filter, octave_sos, room_metrics, early_reflections
from .model import OCTAVE_HZ, SURFACES, BoxObstacle, RoomModel, uniform_bands
from .rir import RIR, detect_onset, estimate_reference_eq, extract_take
from .synth import (SynthConfig, active_octaves, crossfade_fraction, early_paths, effective_mixing_time, extension_gain_ratio,
                    image_paths, late_band_t60, late_bands, late_envelope, path_kernels, resolve_config,
                    synthesize)

VALIDITY = ModelValidity(
    name="Measured-position effective shoebox inversion", status="unvalidated",
    freq_range_hz=(500., 4000.),
    conditions="Position-held-out accuracy is dataset/band dependent; see room-fit audit. "
    "Cached coherent early/expected late energy, fixed scattering/obstacle geometry, conditional material "
    "priors and bootstrap. REF identifies combined devices only. No unique geometry, "
    "physical wall identification, phase coherence or recognition guarantee.",
    citation="ISO 3382-1:2009; Allen & Berkley 1979 doi:10.1121/1.382599; "
    "Brinkmann et al. 2019 doi:10.1121/1.5096178; parent-project bench_room_fit protocol (not distributed)",
)

METRICS = ("edt", "t20", "t30", "c50_db", "c80_db", "d50", "drr_db")
GATES = dict(edt=.05, t20=.05, t30=.05, c50_db=1., c80_db=1., d50=.05, drr_db=3., spectral_db=3.)


class FitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    backend: Literal["shared-energy-lsq/2"] = "shared-energy-lsq/2"
    bands: tuple[int, ...] = (500, 1000, 2000, 4000)
    # Analysis filters overlap: measured neighboring decays constrain nuisance
    # synthesis bands, while `bands` alone defines the evaluation scorecard.
    neighbor_bands: bool = True
    seed: int = Field(default=20260930, ge=0)
    bootstrap: int = Field(default=20, ge=0, le=1000)
    max_nfev: int = Field(default=80, ge=1, le=1000)
    prior_sigma: float = Field(default=.3, gt=0.)
    late_override: bool = True
    late: bool = True
    fit_surfaces: tuple[str, ...] = SURFACES
    energy_dt_s: float = Field(default=.001, ge=.0001, le=.002)
    rir_duration_s: float = Field(default=1.5, gt=.1, le=10.)
    geometry_max_m: float = Field(default=30., gt=1., le=100.)

    @model_validator(mode="after")
    def _supported(self):
        if not self.bands or len(set(self.bands)) != len(self.bands) or any(b not in OCTAVE_HZ for b in self.bands):
            raise ValueError("bands must be unique supported octaves")
        if not self.fit_surfaces or len(set(self.fit_surfaces)) != len(self.fit_surfaces) or any(s not in SURFACES for s in self.fit_surfaces):
            raise ValueError("fit_surfaces must be unique room surfaces")
        return self


def _neighbor_bands(config, fs):
    if not config.neighbor_bands or not config.late:
        return ()
    neighbors = {j for b in config.bands for j in
                 (OCTAVE_HZ.index(b)-1, OCTAVE_HZ.index(b)+1) if 0 <= j < len(OCTAVE_HZ)}
    return tuple(OCTAVE_HZ[j] for j in sorted(neighbors)
                 if OCTAVE_HZ[j] not in config.bands and active_octaves(fs)[j])


@dataclass
class BandObservation:
    values: dict[str, float | None]
    reasons: dict[str, str | None]
    edc_time_s: np.ndarray
    edc_db: np.ndarray
    energy: float
    direct_energy: float
    early_energy_ratio: np.ndarray


@dataclass
class Observation:
    id: str
    source_xyz_m: tuple[float, float, float]
    receiver_xyz_m: tuple[float, float, float]
    fs: int
    bands: dict[int, BandObservation]
    echoes_s: np.ndarray
    metadata: dict = field(default_factory=dict)
    split: str = "train"

    @property
    def position_key(self):
        # Repeats/orientations at the same pair cannot become independent samples.
        return tuple(float(x) for x in np.round((*self.source_xyz_m, *self.receiver_xyz_m), 6))


@dataclass
class FitResult:
    model: RoomModel
    config: FitConfig
    report: dict


def _finite(x):
    return float(x) if x is not None and np.isfinite(x) else None


def _onset(rir: RIR) -> int:
    value = rir.metadata.get("direct_sample")
    if value is None and "direct_samples" in rir.metadata:
        value = rir.metadata["direct_samples"][0]
    if value is not None:
        if isinstance(value, bool) or int(value) != value or not 0 <= value < len(rir.samples):
            raise ValueError("direct_sample must be an in-range integer")
        return int(value)
    first = detect_onset(rir.samples)
    return first + int(np.argmax(np.abs(rir.samples[first:first + max(1, round(.001*rir.fs))])))


def correct_reference(rir: RIR, reference: RIR) -> RIR:
    """Linear bounded REF inverse; retain a pre-roll, never wrap late energy.

    The full exact-grid inverse is centred as a finite FIR and linearly
    convolved. A matching support-only FIR is applied to synthetic predictions.
    This intentionally retains pre-ringing, with an explicit sample origin.
    """
    if rir.fs != reference.fs or rir.samples.ndim != 1 or reference.samples.ndim != 1:
        raise ValueError("REF and measurement must be mono at the same sample rate")
    nfft = int(next_fast_len(max(4096, round(.1*rir.fs))))
    source, receiver = reference.source_xyz_m, reference.receiver_xyz_m
    if source is None or receiver is None:
        raise ValueError("REF requires known source/receiver coordinates")
    distance = float(np.linalg.norm(np.asarray(source)-receiver))
    eq = estimate_reference_eq(reference, direct_sample=_onset(reference),
                               reference_distance_m=distance, nfft=nfft,
                               band_hz=(200., min(8000., .49*rir.fs)))
    prediction = bool(rir.metadata.get("synthetic"))
    transfer = eq.supported.astype(float) if prediction else eq.correction
    kernel = np.fft.fftshift(irfft(transfer, nfft))
    corrected = signal.fftconvolve(rir.samples, kernel)
    delay = nfft//2
    complete = []
    for band in OCTAVE_HZ:
        centre = exact_center(band)
        bins = (eq.frequency_hz >= centre*10**-.15) & (eq.frequency_hz <= centre*10**.15)
        if np.any(bins) and np.all(eq.supported[bins]):
            complete.append(band)
    from sensorsim.licensing import lineage_metadata
    lineage = lineage_metadata(rir.metadata, reference.metadata)
    return RIR(rir.fs, corrected, rir.source_xyz_m, rir.receiver_xyz_m,
               {**rir.metadata, **({'lineage': lineage} if lineage else {}), "direct_sample": _onset(rir)+delay,
                "reference": {**eq.metadata, "nfft": nfft, "complete_bands_hz": complete,
                              "sample_origin_s": -delay/rir.fs,
                              "reference_id": reference.metadata.get("id"),
                              "support_only": prediction}})


def observe(rir: RIR, *, id: str, config: FitConfig | None = None,
            split: str = "train", reference: RIR | None = None) -> Observation:
    """Reduce one measured/synthetic RIR; no collection of waveforms is retained."""
    config = config or FitConfig()
    if rir.samples.ndim != 1 or rir.source_xyz_m is None or rir.receiver_xyz_m is None or np.asarray(rir.receiver_xyz_m).shape != (3,):
        raise ValueError("fit requires mono RIRs and one known source/receiver XYZ")
    onset_raw = _onset(rir)
    if reference is not None:
        rir = correct_reference(rir, reference)
    onset = _onset(rir)
    peaks = early_reflections(rir.samples, rir.fs, onset_sample=onset,
                              min_spacing_s=.0005, threshold_db=-25)
    echoes = np.array([p.excess_delay_s for p in peaks if p.excess_delay_s >= .0005])
    result = {}
    # Auxiliary measurements are collected only at training positions. They
    # must pass the same excitation/REF/noise checks as the scored bands.
    bands = (*config.bands, *(_neighbor_bands(config, rir.fs) if split == "train" else ()))
    for band in bands:
        x = octave_filter(rir.samples, rir.fs, band)
        noise = 0. if rir.metadata.get("synthetic") else None
        # Compatibility: observations and the expected-power optimizer remain
        # a matched drr/1 pair until both are migrated; evaluation uses drr/2.
        metric = room_metrics(x, rir.fs, onset_sample=onset, noise_power=noise, legacy_drr=True)
        values = {k: _finite(getattr(metric, k).rt60_s if k in ("edt", "t20", "t30") else getattr(metric, k)) for k in METRICS}
        reasons = {k: getattr(metric, k).reason if k in ("edt", "t20", "t30") else metric.decay.reason for k in METRICS}
        curve = metric.decay
        keep = np.flatnonzero(np.isfinite(curve.db) & (curve.db >= -35)) if curve.valid else np.array([], int)
        idx = keep[np.unique(np.linspace(0, len(keep)-1, min(24, len(keep))).astype(int))] if len(keep) else keep
        half = round(.0025*rir.fs)
        direct = float(np.sum(x[max(0, onset-half):onset+half+1]**2))
        # Causal octave-filter delay is retained consistently in these windows.
        impulse = np.zeros(round(.04*rir.fs)); impulse[0] = 1
        peak_delay = int(np.argmax(np.abs(octave_filter(impulse, rir.fs, band))))
        width = max(1, round(.00075*rir.fs))
        def pulse_energy(delay):
            center = onset+peak_delay+round(delay*rir.fs)
            return float(np.sum(x[max(0, center-width):center+width+1]**2))
        denominator = max(pulse_energy(0), 1e-30)
        ratios = np.array([pulse_energy(e)/denominator for e in echoes])
        energy = float(curve.energy[0]) if curve.valid else float(np.sum(x[onset:]**2))
        unsupported = None
        if reference is not None and band not in rir.metadata["reference"]["complete_bands_hz"]:
            unsupported = "octave not fully supported by REF"
        excitation = rir.metadata.get("excitation", {})
        if excitation.get("type") == "ess" and not (
                excitation["f_start_hz"] <= exact_center(band)*10**-.15 and
                exact_center(band)*10**.15 <= excitation["f_end_hz"]):
            unsupported = "octave not fully supported by declared excitation"
        if unsupported:
            values = {k: None for k in METRICS}
            reasons = {k: unsupported for k in METRICS}
            idx = np.array([], int)
            energy, direct = 0., 0.
        result[band] = BandObservation(values, reasons, curve.time_s[idx], curve.db[idx],
                                        energy, direct, ratios)
    return Observation(id, tuple(rir.source_xyz_m), tuple(rir.receiver_xyz_m), rir.fs,
                       result, echoes, {**rir.metadata, "raw_direct_sample": onset_raw,
                                       "analysis_direct_sample": onset, "drr_definition": "drr/1",
                                       "device_response": "REF-corrected" if reference else "retained/no REF"}, split)


def check_split(observations: Iterable[Observation]) -> None:
    seen, ids = {}, set()
    for obs in observations:
        if obs.id in ids:
            raise ValueError(f"duplicate observation id: {obs.id}")
        ids.add(obs.id)
        old = seen.setdefault(obs.position_key, obs.split)
        if old != obs.split:
            raise ValueError(f"position leakage between {old} and {obs.split}: {obs.id}")


# The absorption catalogue is an OPTIONAL plug-in. It is compiled from third-party handbook sources
# whose redistribution rights were not verified, so it does not ship. Install one you have the rights
# to (THIRD_PARTY.md, "Absorption catalogue"): point CATALOGUE_ENV or use_catalogue() at a folder
# holding CATALOGUE_FILES, or drop the files into the package folder below. Without it,
# material_prior() falls through to the explicit broad 0.2 prior and records that in its evidence.
CATALOGUE_ENV = "SENSORSIM_ACOUSTIC_CATALOGUE"
CATALOGUE_FILES = ("materials.yaml", "assemblies_measured.yaml")
_PACKAGE_CATALOGUE = Path(__file__).parents[2]/"library/data/acoustic"
_catalogue_override: Path | None = None
_warned_missing: list = []           # the missing-catalogue warning is given once per process


def use_catalogue(folder: str | Path | None) -> None:
    """Use the absorption catalogue in `folder` (it must hold CATALOGUE_FILES); None restores the default."""
    global _catalogue_override
    _catalogue_override = None if folder is None else Path(folder)


def catalogue_location() -> tuple[Path, bool]:
    """(folder, explicitly chosen): use_catalogue(), then CATALOGUE_ENV, then the package folder."""
    if _catalogue_override is not None:
        return _catalogue_override, True
    if os.environ.get(CATALOGUE_ENV):
        return Path(os.environ[CATALOGUE_ENV]), True
    return _PACKAGE_CATALOGUE, False


def catalogue_fingerprint() -> dict:
    """Where the catalogue is read from and the sha256 of its files (pipelines key their fit on it)."""
    folder, _ = catalogue_location()
    files = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
             for name in CATALOGUE_FILES if (folder/name).is_file()}
    return {"folder": str(folder.resolve()), "files": files}


def _material_tables():
    # Safe, bounded catalogue read; avoid constructing the unrelated RF/EO library.
    path, chosen = catalogue_location()
    missing = [name for name in CATALOGUE_FILES if not (path/name).is_file()]
    if missing and chosen:
        raise FileNotFoundError(f"absorption catalogue at {path} lacks {missing} (set by use_catalogue() or "
                                f"{CATALOGUE_ENV}); see THIRD_PARTY.md, 'Absorption catalogue'")
    if missing:
        if _warned_missing:
            return {}, {}
        _warned_missing.append(True)
        warnings.warn(
            "absorption catalogue not installed; using a uniform 0.2 absorption prior for "
            "every surface. Fitted absorption will be less accurate than the published "
            "benchmark figures, which were produced with the catalogue. To install one, see "
            f"THIRD_PARTY.md ('Absorption catalogue'): set {CATALOGUE_ENV} to its folder.",
            UserWarning, stacklevel=3)
        return {}, {}
    stamp = tuple((path/name).stat().st_mtime_ns for name in CATALOGUE_FILES)
    return _read_catalogue(str(path.resolve()), stamp)


@lru_cache(maxsize=4)
def _read_catalogue(folder: str, stamp: tuple) -> tuple[dict, dict]:
    """Parsed once per folder and file version (`stamp`), so an edited or newly chosen catalogue is reread."""
    path = Path(folder)
    materials = yaml.safe_load((path/"materials.yaml").read_text())["materials"]
    profiles = yaml.safe_load((path/"assemblies_measured.yaml").read_text())["absorption_profiles"]
    return materials, profiles


def material_prior(description: str) -> tuple[tuple, dict]:
    """Exact IDs or conservative text aliases; mounting caveats stay in provenance."""
    materials, profiles = _material_tables()
    text = description.strip().lower()
    aliases = (("carpet", "carpet_with_underlay"), ("drywall", "plasterboard"),
               ("plasterboard", "plasterboard"), ("concrete", "concrete"),
               ("glass", "glass_float"), ("window", "glass_float"), ("wood", "wood_generic"))
    candidates = [key for word, key in aliases if word in text]
    key = text if text in materials else (candidates[0] if len(set(candidates)) == 1 else None)
    # "carpet over concrete" is a carpet mounting description, not two independent walls.
    if text.startswith("carpet"):
        key = "carpet_with_underlay"
    record = materials.get(key, {})
    refs = record.get("absorption_profile_refs", [])
    profile = profiles.get(text) or (profiles[refs[0]] if refs else None)
    if profile is not None:
        freq = profile["frequency_hz"]["nominal"]
        alpha = profile["alpha"]["nominal"]
        if alpha is not None:
            row = np.interp(np.log2(OCTAVE_HZ), np.log2(freq), alpha)
            return tuple(row), {"description": description, "material_id": key,
                                "profile": text if text in profiles else refs[0],
                                "source": profile["alpha"]["source"],
                                "validity": profile["alpha"]["validity"],
                                "extrapolated_bands_hz": [b for b in OCTAVE_HZ if b < min(freq) or b > max(freq)],
                                "population_uncertainty": "unknown; regularization scale is not measured spread"}
    return (.2,)*7, {"description": description, "material_id": key, "profile": None,
                      "reason": "no unambiguous non-null library profile; explicit broad 0.2 prior"}


def model_from_manifest(manifest) -> RoomModel:
    if manifest.shoebox_dims_m is None:
        raise ValueError("estimate geometry from training observations before constructing model")
    descriptions = {s: manifest.surfaces.get(s, manifest.surfaces.get(
        "floor" if s == "z_min" else "ceiling" if s == "z_max" else "walls", "unspecified")) for s in SURFACES}
    priors = {s: material_prior(d) for s, d in descriptions.items()}
    boxes = []
    for box in manifest.obstacles:
        alpha, evidence = material_prior(box.material)
        boxes.append(BoxObstacle(**box.model_dump(), absorption=alpha))
        priors["obstacle:"+box.name] = (alpha, evidence)
    return RoomModel(room_id=manifest.room_id, shoebox_dims_m=manifest.shoebox_dims_m,
                     absorption={s: priors[s][0] for s in SURFACES}, obstacles=tuple(boxes),
                     provenance={"surface_descriptions": descriptions,
                                 "material_priors": {s: v[1] for s, v in priors.items()},
                                 "scattering": "fixed prior; not identified",
                                 "obstacles": "provided boxes; fixed material priors, not reconstructed"})


def _first_echoes(source, receiver, lower, upper, speed):
    direct = np.linalg.norm(source-receiver)
    images = np.tile(np.asarray(source, dtype=float), (6, 1))  # float: images are continuous
    for j in range(6):
        axis = j//2
        images[j, axis] = 2*(lower[axis] if j % 2 == 0 else upper[axis])-source[axis]
    return (np.linalg.norm(images-receiver, axis=1)-direct)/speed


def estimate_geometry(observations: list[Observation], *, dims_m=None,
                      sound_speed_m_s: float = 343., max_dimension_m: float = 30.) -> dict:
    """Training-only echo reconstruction; explicit ambiguity, no wall-label claim.

    With no dimensions, coordinates must already use the lower room corner.
    `metadata['wall_echoes_s']` maps wall names to excess TOAs when labelled.
    Otherwise bounded multistart assignment estimates three upper planes.
    Known dimensions plus a local frame estimates the lower-corner translation.
    """
    if not observations or any(o.split != "train" for o in observations):
        raise ValueError("geometry requires training observations only")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0:
        raise ValueError("sound speed must be finite and positive")
    # float64 on entry: integer XYZ (e.g. [1, 1, 1] in a YAML manifest) otherwise truncates
    # every wall image and zeroes the finite-difference derivatives (cr-room-model-04).
    sources = np.array([o.source_xyz_m for o in observations], dtype=float)
    receivers = np.array([o.receiver_xyz_m for o in observations], dtype=float)
    allpos = np.vstack([sources, receivers])
    labelled = all("wall_echoes_s" in o.metadata for o in observations)
    if dims_m is None:
        if np.any(allpos <= 0):
            raise ValueError("unknown dimensions require a known corner frame with positive coordinates")
        lo = allpos.max(0)+.01
        hi = np.full(3, max_dimension_m)
        if np.any(lo >= hi):
            raise ValueError("positions exceed geometry dimension bound")
        select = np.array([1, 3, 5])
    else:
        dims_m = np.asarray(dims_m, float)
        if dims_m.shape != (3,) or not np.isfinite(dims_m).all() or np.any(dims_m <= 0):
            raise ValueError("invalid known dimensions")
        lo, hi = allpos.max(0)-dims_m+.01, allpos.min(0)-.01
        if np.any(lo >= hi):
            raise ValueError("positions do not fit known room dimensions")
        select = np.arange(6)
    def residual(p):
        lower, upper = (np.zeros(3), p) if dims_m is None else (p, p+dims_m)
        out = []
        for obs, source, receiver in zip(observations, sources, receivers):
            predicted = _first_echoes(source, receiver, lower, upper, sound_speed_m_s)[select]
            if labelled:
                for j, value in zip(select, predicted):
                    if SURFACES[j] in obs.metadata["wall_echoes_s"]:
                        out.append((value-obs.metadata["wall_echoes_s"][SURFACES[j]])/.0002)
            else:
                cost = np.abs(predicted[:, None]-obs.echoes_s[None, :])
                # Dummy candidates explicitly penalize missing echoes.
                cost = np.c_[cost, np.full((len(select), len(select)), .004)]
                rows, cols = linear_sum_assignment(cost)
                out.extend(cost[rows, cols]/.0002)
        return np.array(out)
    if labelled:
        for j in select:
            if not any(SURFACES[j] in o.metadata["wall_echoes_s"] for o in observations):
                raise ValueError("all fitted planes require labelled echo support")
        starts = [(lo+hi)/2]
    else:
        if len({o.position_key for o in observations}) < 3 or any(len(o.echoes_s) < 3 for o in observations):
            raise ValueError("too few independent positions/echoes for geometry")
        # Echo-derived candidates seed each axis; no waveform-dependent holdout choice.
        choices = []
        for axis in range(3):
            candidates = []
            for obs, s, r in zip(observations, sources, receivers):
                distance = np.linalg.norm(s-r)+sound_speed_m_s*obs.echoes_s
                transverse = np.sum((s-r)**2)-(s[axis]-r[axis])**2
                root = np.sqrt(np.maximum(distance**2-transverse, 0))
                values = (s[axis]+r[axis]+root)/2 if dims_m is None else (s[axis]+r[axis]-root)/2
                candidates.extend(values[(values > lo[axis]) & (values < hi[axis])])
            if not candidates:
                raise ValueError("no feasible echo-derived wall plane")
            values = np.array(candidates)
            counts = np.sum(abs(values[:, None]-values[None, :]) < .06, axis=1)
            picked = []
            for i in np.argsort(-counts, kind="stable"):
                if all(abs(values[i]-v) > .12 for v in picked):
                    picked.append(float(values[i]))
                if len(picked) == 3:
                    break
            choices.append(picked)
        starts = list(itertools.product(*choices))
    fits = [least_squares(residual, p, bounds=(lo, hi), loss="soft_l1", max_nfev=80,
                           diff_step=1e-4) for p in starts]
    best = min(fits, key=lambda r: r.cost)
    if np.linalg.matrix_rank(best.jac) < 3:
        raise ValueError("echo geometry is rank deficient")
    rms = float(np.sqrt(np.mean(residual(best.x)**2))*.0002)
    if rms > .002:
        raise ValueError(f"echo geometry unsupported: excess-TOA RMS {rms:.6f} s > 0.002 s")
    lower, upper = (np.zeros(3), best.x) if dims_m is None else (best.x, best.x+dims_m)
    alternatives = [f.x.tolist() for f in fits if f.cost <= best.cost+max(1., .05*best.cost)]
    return {"shoebox_dims_m": (upper-lower).tolist(), "coordinate_translation_m": (-lower).tolist(),
            "rms_excess_toa_s": rms, "labelled": labelled, "solutions": alternatives,
            "status": "conditional echo estimate; competing solutions are not a confidence interval"}


def _position(model, xyz):
    return np.array(xyz)+np.array(model.provenance.get("coordinate_translation_m", [0, 0, 0]))


@dataclass
class _EnergyCache:
    hits: np.ndarray
    gains: np.ndarray
    times: np.ndarray
    anchor_hits: np.ndarray
    anchor_gains: np.ndarray
    anchor_mask: np.ndarray
    anchor_extension: tuple
    echo_indices: np.ndarray
    echo_matches: np.ndarray
    basis_hits: np.ndarray
    early_basis: np.ndarray
    onset: int
    synth: SynthConfig                 # the renderer settings the basis and tail mirror
    last_key: bytes | None = None
    last_features: dict = field(default_factory=dict)
    early_key: bytes | None = None
    early_components: dict = field(default_factory=dict)
    early_data: dict = field(default_factory=dict)
    late_components: dict = field(default_factory=dict)
    device_power: np.ndarray = field(default_factory=lambda: np.ones(7))
    floor_groups: np.ndarray | None = None
    floor_terms: np.ndarray | None = None
    floor_reference: np.ndarray | None = None


def _floor_terms(model, s, r, synth_config, reference):
    """Fit the quadratic floor-image energy embedded in obstacle path groups."""
    floor = SURFACES.index("z_min")
    hits = reference.wall_hits[:, floor, None]/2

    def trace(alpha):
        probe = RoomModel.model_validate({**model.model_dump(),
            "absorption": {**uniform_bands(0.), "z_min": (alpha,)*7}})
        paths = early_paths(probe, s, r, synth_config)[0]
        if not (np.array_equal(paths.wall_hits, reference.wall_hits) and
                np.array_equal(paths.delay_s, reference.delay_s) and
                np.array_equal(paths.directivity_gain, reference.directivity_gain)):
            raise ValueError("obstacle paths change with floor absorption; cached fit cannot represent them")
        return paths.wall_pressure/(1-alpha)**hits

    probes = (.5, .75)
    pressure = np.stack([reference.wall_pressure, *(trace(a) for a in probes)])
    change = np.abs(pressure-pressure[0]).max(axis=(0, 2))
    embedded = np.flatnonzero(change > 1e-9*np.abs(pressure).max(axis=(0, 2)))
    if not len(embedded):
        return None
    energy = pressure[:, embedded]**2
    if np.any((pressure[0, embedded] == 0) & (energy.max(axis=0) > 0)):
        raise ValueError("floor-dependent obstacle group vanishes in neutral trace")
    t = 1-np.array([0., *probes])
    coefficients = np.linalg.solve(np.vander(t[:3], 3, increasing=True), energy[:3].reshape(3, -1))
    coefficients = coefficients.reshape(3, len(embedded), 7).transpose(1, 2, 0)
    check, t = trace(.25)[embedded]**2, .75
    predicted = coefficients[..., 0]+t*(coefficients[..., 1]+t*coefficients[..., 2])
    if np.any(np.abs(predicted-check) > 1e-9*np.maximum(energy.max(axis=0), check)+1e-300):
        raise ValueError("floor-image obstacle term is not quadratic in floor coefficient")
    return embedded, coefficients, pressure[0, embedded]


def _cache(model, obs, config):
    s, r = _position(model, obs.source_xyz_m), _position(model, obs.receiver_xyz_m)
    synth_config = SynthConfig(late=config.late, max_order=None if config.late else 1)
    mixing, _ = effective_mixing_time(model, s, r, synth_config.crossfade_s, late=config.late)
    fade = synth_config.crossfade_s
    # Trace with neutral room absorption. Dividing out the starting coefficients
    # loses paths irreversibly when a prior is exactly one (0/0).
    neutral = RoomModel.model_validate({**model.model_dump(), "absorption": uniform_bands(0.)})
    paths, anchor, _, _ = early_paths(neutral, s, r, synth_config)
    anchor_gains = anchor.total_gain_octave**2
    gains = paths.gain_octave**2
    direct = float(np.linalg.norm(s-r)/model.sound_speed_m_s)
    candidates = np.flatnonzero(paths.wall_hits.sum(1) == 1)
    if len(obs.echoes_s) and len(candidates):
        cost = abs((paths.delay_s[candidates]-direct)[:, None]-obs.echoes_s[None])
        rows, cols = linear_sum_assignment(cost)
        ok = cost[rows, cols] <= .002
        ii, jj = candidates[rows[ok]], cols[ok]
    else:
        ii = jj = np.array([], int)
    # Each coefficient multiplies one group of identical wall-hit counts. The
    # basis includes the actual spectral interpolation, air absorption, obstacle
    # loss, fractional delay and early crossfade from the waveform renderer.
    floor = _floor_terms(model, s, r, synth_config, paths) if model.obstacles else None
    if floor is None:
        hits, groups = np.unique(paths.wall_hits, axis=0, return_inverse=True)
    else:
        tag = np.full(len(paths.delay_s), -1)
        tag[floor[0]] = floor[0]
        keys, groups = np.unique(np.c_[paths.wall_hits, tag], axis=0, return_inverse=True)
        hits = keys[:, :6]
    length = int(np.ceil(paths.delay_s.max()*obs.fs)) + 2*synth_config.filter_half_length + 1
    if len(hits)*7*length*8 > 64*1024**2:
        raise ValueError("early fitting basis exceeds 64 MiB per position; shorten the mixing horizon")
    basis = np.zeros((len(hits), 7, length))
    for b in range(7):
        pressure = np.zeros_like(paths.wall_pressure)
        pressure[:, b] = paths.wall_pressure[:, b]
        for sl, index, kernel in path_kernels(neutral, paths, obs.fs, synth_config,
                                               wall_pressure=pressure):
            valid = (index >= 0) & (index < length)
            group = np.broadcast_to(groups[sl, None], index.shape)
            np.add.at(basis[:, b], (group[valid], index[valid]), kernel[valid])
    if config.late:
        time = (np.arange(length)-synth_config.filter_half_length)/obs.fs
        basis *= np.cos(np.pi/2*crossfade_fraction(time, mixing, fade))
    onset = round(direct*obs.fs)+synth_config.filter_half_length
    # Per late band, the squared air-loss ratio that turns its octave's anchor gains into its own.
    extension = tuple(extension_gain_ratio(b, anchor.distance_m, model, synth_config.air_absorption)
                      for b in late_bands(obs.fs))
    devices = resolve_config(model, synth_config)
    device_power = np.broadcast_to(np.asarray(devices.source_directivity.diffuse_power, float) *
                                   devices.receiver_directivity.diffuse_power, (7,)).copy()
    extra = {} if floor is None else dict(floor_groups=groups[floor[0]], floor_terms=floor[1],
                                          floor_reference=floor[2])
    return _EnergyCache(paths.wall_hits, gains, paths.delay_s-direct, anchor.wall_hits,
                        anchor_gains, (abs(anchor.delay_s-mixing) <= fade/2) &
                        (anchor.wall_hits.sum(1) > 0), extension, ii, jj, hits, basis, onset, synth_config,
                        device_power=device_power, **extra)


def _power_features(power, dt):
    energy = np.cumsum(power[::-1])[::-1]
    if not len(energy) or energy[0] <= 0:
        raise ValueError("zero expected model energy")
    db = 10*np.log10(np.maximum(energy/energy[0], 1e-30))
    t = np.arange(len(power))*dt
    result = {}
    for name, (upper, lower) in zip(("edt", "t20", "t30"), ((0, -10), (-5, -25), (-5, -35))):
        mask = (db <= upper) & (db >= lower)
        if mask.sum() < 3:
            result[name] = None
        else:
            x, y = t[mask], db[mask]
            slope = float(np.dot(x-x.mean(), y-y.mean())/np.sum((x-x.mean())**2))
            result[name] = -60/slope if slope < 0 else None
    for name, cut in (("c50_db", .05), ("c80_db", .08), ("drr_db", .0025)):
        early = power[t < cut].sum()
        result[name] = float(10*np.log10(max(early, 1e-30)/max(energy[0]-early, 1e-30)))
    result["d50"] = float(power[t < .05].sum()/energy[0])
    return result, t, db, float(energy[0])


@lru_cache(maxsize=64)
def _analysis_peak(fs, band):
    x = np.zeros(round(.04*fs)); x[0] = 1.
    return int(np.argmax(abs(octave_filter(x, fs, band))))


@byte_cache(4)
def _tail_edges(fs, dt, n):
    """One native-sample bin grid, shared by every band and observation."""
    length = int(np.ceil(n*dt*fs))+1
    index = np.floor(np.arange(length)/fs/dt).astype(int)
    edges = np.searchsorted(index, np.arange(n+1))
    edges.setflags(write=False)
    return edges


@byte_cache(4)
def _tail_envelope(fs, stop, latency, mixing, fade, decay):
    """Output envelope independent of the receiver's analysis-bin origin."""
    time = (np.arange(stop)-latency)/fs
    envelope = late_envelope(time, mixing, fade, decay)
    amplitude = float(late_envelope((stop-latency)/fs, mixing, fade, decay))
    ratio = float(np.exp(-3*np.log(10)/fs/decay))
    envelope.setflags(write=False)
    return envelope, amplitude, ratio


@byte_cache(4)
def _prepared_tail(fs, latency, mixing, fade, decay, synthesis_blob, analysis_blob):
    # Keep the complete physical fade, independent of duration and bin onset.
    stop = max(0, int(np.ceil((mixing+fade/2)*fs+latency)))
    envelope, amplitude, ratio = _tail_envelope(fs, stop, latency, mixing, fade, decay)
    return prepare_tail(np.frombuffer(synthesis_blob, float).reshape(-1, 6),
                        np.frombuffer(analysis_blob, float).reshape(-1, 6),
                        envelope, amplitude, ratio)


@byte_cache(64)
def _cached_tail_power(fs, dt, n, onset, latency, mixing, fade, decay, synthesis_blob, analysis_blob):
    state = _prepared_tail(fs, latency, mixing, fade, decay, synthesis_blob, analysis_blob)
    out = state.integrate_edges(onset+_tail_edges(fs, dt, n))
    out.setflags(write=False)
    return out


def _tail_power(fs, late_band, analysis_band, dt, n, onset, latency, mixing, fade, decay):
    """Integer-edge integration with separate physical state and grid reuse.

    Exact decay, sample rate, latency, fade/mixing phase and coefficient bytes
    participate in identity. Wall levels remain outside this shape cache.
    """
    synthesis = np.asarray(late_bands(fs)[late_band].sos, dtype=np.float64).tobytes()
    analysis = np.asarray(octave_sos(fs, analysis_band), dtype=np.float64).tobytes()
    return _cached_tail_power(fs, dt, n, onset, latency, mixing, fade, decay, synthesis, analysis)


_tail_power.cache_info = _cached_tail_power.cache_info
_tail_power.cache_clear = _cached_tail_power.cache_clear


def _expected_all(model, obs, cache, alpha, t60, config):
    """Cached coherent early pressure + exact expected filtered-noise tail.

    No waveform realization is drawn by the optimizer. Only the slow tail is
    stochastic in synth; early wall and air physics are the same linear kernels.
    """
    key = alpha.tobytes()+t60.tobytes()
    if key == cache.last_key:
        return cache.last_features
    fs = obs.fs
    # A few isolated pulses can decay within one 1 ms bin. There is no long
    # stochastic tail to accelerate in that case; retain the native sample grid.
    dt = config.energy_dt_s if config.late else 1/fs
    fade = cache.synth.crossfade_s
    mixing, _ = effective_mixing_time(model, _position(model, obs.source_xyz_m),
                                      _position(model, obs.receiver_xyz_m), fade, late=config.late)
    bands = late_bands(fs)
    band_t60 = late_band_t60(bands, t60, model, cache.synth.air_absorption)
    duration = max(1., mixing+1.2*max(band_t60)) if config.late else max(.12, cache.times.max()+.08)
    n = int(np.ceil(duration/dt))
    alpha_key = alpha.tobytes()
    if cache.early_key != alpha_key:
        for j in range(7):
            column_key = alpha[:, j].tobytes()
            previous = cache.early_components.get(j)
            if previous is None or previous[0] != column_key:
                factor = np.prod((1-alpha[:, j])[None]**(cache.basis_hits/2), axis=1)
                if cache.floor_groups is not None:
                    t, c = 1-alpha[SURFACES.index("z_min"), j], cache.floor_terms[:, j]
                    pressure = np.sqrt(np.maximum(c[:, 0]+t*(c[:, 1]+t*c[:, 2]), 0.))
                    reference = cache.floor_reference[:, j]
                    factor[cache.floor_groups] *= np.divide(pressure, reference,
                        out=np.ones_like(reference), where=reference > 0)
                cache.early_components[j] = (column_key, factor @ cache.early_basis[:, j])
        early = sum(cache.early_components[j][1] for j in range(7))
        # Let the causal analysis filters ring out beyond the finite early waveform.
        early = np.pad(early, (0, round(.08*fs)))
        for band in config.bands:
            filtered = octave_filter(early, fs, band)
            samples = filtered[cache.onset:]
            indices = np.floor(np.arange(len(samples))/fs/dt).astype(int)
            power = np.bincount(indices, weights=samples*samples)
            half = round(.0025*fs)
            direct = float(np.sum(filtered[max(0, cache.onset-half):cache.onset+half+1]**2))
            reverb = float(np.sum(filtered[cache.onset+half+1:]**2))
            # Match observe's finite windows, including overlapping reflections.
            peak, width = _analysis_peak(fs, band), max(1, round(.00075*fs))
            centers = cache.onset+peak+np.rint(np.r_[0., obs.echoes_s[cache.echo_matches]]*fs).astype(int)
            cumulative = np.r_[0., np.cumsum(filtered*filtered)]
            pulses = cumulative[np.clip(centers+width+1, 0, len(filtered))] - cumulative[np.clip(centers-width, 0, len(filtered))]
            ratios = pulses[1:]/max(pulses[0], 1e-30)
            cache.early_data[band] = power, direct, reverb, ratios
        cache.early_key = alpha_key
    levels = np.zeros(len(bands))
    if config.late:
        anchor = cache.anchor_gains*np.prod((1-alpha)[None]**cache.anchor_hits[:, :, None], axis=1)
        mask = cache.anchor_mask
        levels = np.array([np.sum(anchor[mask, b.index]*cache.anchor_extension[k][mask])/fade
                           for k, b in enumerate(bands)])
        levels *= cache.device_power[[b.index for b in bands]]
    out = {}
    for band in config.bands:
        power, early_direct, early_reverb, ratios = cache.early_data[band]
        p = np.zeros(n)
        p[:min(n, len(power))] = power[:n]
        late_power = np.zeros(n)
        if config.late:
            for k, b in enumerate(bands):
                column_key = (alpha[:, b.index].tobytes(), float(band_t60[k]), n, fs, dt,
                              cache.onset, cache.synth.filter_half_length, mixing, fade, float(levels[k]),
                              bands[k].sos.tobytes(), octave_sos(fs, band).tobytes())
                previous = cache.late_components.get((band, k))
                if previous is None or previous[0] != column_key:
                    value = levels[k]/fs*_tail_power(fs, k, band, dt, n, cache.onset,
                        cache.synth.filter_half_length, mixing, fade, float(band_t60[k]))
                    cache.late_components[band, k] = column_key, value
                late_power += cache.late_components[band, k][1]
        p += late_power
        values, t, db, energy = _power_features(p, dt)
        # The +/-2.5 ms direct gate is not an integer number of 1 ms bins.
        # Use actual early samples, retaining the pre-onset half of the window;
        # integrate the smooth expected tail fraction at the same sample edge.
        half = round(.0025*fs)
        in_direct = np.clip(((half+1)/fs-t)/dt, 0., 1.)
        tail_direct = float(late_power @ in_direct)
        direct = early_direct+tail_direct
        reverb = early_reverb+float(late_power.sum())-tail_direct
        values['drr_db'] = float(10*np.log10(direct/reverb)) if min(direct, reverb) > 0 else None
        out[band] = values, t, db, energy, ratios
    cache.last_key, cache.last_features = key, out
    return out


def _expected(model, obs, cache, band, alpha, t60, config):
    coefficients = np.array([model.absorption[s] for s in SURFACES])
    decay = model.late_t60().copy()
    b = OCTAVE_HZ.index(band)
    coefficients[:, b], decay[b] = alpha, t60
    return _expected_all(model, obs, cache, coefficients, decay, config)[band]


@dataclass
class _FitProblem:
    """One deterministic objective and its byte-bounded forward table.

    Forward predictions do not depend on bootstrap multiplicities or priors.
    Retain exact parameter-byte keys across the main fit and all refits; each
    spawned worker receives the main fit's warm table and private mutable state.
    This warms evaluations, not optimizer starting points or stopping rules.
    """
    model: RoomModel
    observations: list
    caches: list
    config: FitConfig
    specs: dict
    mapping: dict
    alpha0: np.ndarray
    late0: np.ndarray
    low: list
    high: list
    starts: list

    def __post_init__(self):
        self.predictions = ByteCache(self._prediction, 64*1024**2)

    def _prediction(self, index, alpha_blob, rt_blob):
        return _expected_all(self.model, self.observations[index], self.caches[index],
                             np.frombuffer(alpha_blob, float).reshape(6, 7),
                             np.frombuffer(rt_blob, float), self.config)

    def decode(self, params):
        model, specs, mapping = self.model, self.specs, self.mapping
        alpha0, late0 = self.alpha0, self.late0
        alpha = alpha0.copy()
        for spec in specs.values():
            for surface, group in mapping.items():
                alpha[SURFACES.index(surface), spec['b']] = params[spec['coefficients']][group]
        # Same Eyring/obstacle/ISO-air implementation as the waveform model.
        rt = (np.array(model.late_t60_s) if model.late_t60_s is not None else
              model.t60_estimates(room_absorption=alpha))
        rt = np.where(np.isfinite(rt), rt, late0)
        for spec in specs.values():
            if spec['rt_index'] is not None:
                rt[spec['b']] = np.exp(params[spec['rt_index']])
        return alpha, rt

    def residual(self, params, counts, late_priors):
        observations, caches, config = self.observations, self.caches, self.config
        specs = self.specs
        alpha, rt = self.decode(params)
        out = []
        predictions = {i: self.predictions(i, alpha.tobytes(), rt.tobytes())
                       for i, obs in enumerate(observations) if counts[i]}
        for band, spec in specs.items():
            for i in spec['usable']:
                if not counts[i]:
                    continue
                obs, cache = observations[i], caches[i]
                data = obs.bands[band]
                values, t, db, _, echoes = predictions[i][band]
                row = []
                for metric in ("edt", "t20", "t30", "c50_db", "drr_db"):
                    target = data.values[metric]
                    if target is not None:
                        scale = .05*target if metric in ("edt", "t20", "t30") else GATES[metric]
                        row.append((values[metric]-target)/scale if values[metric] is not None else 20.)
                if len(data.edc_db):
                    row.extend((np.interp(data.edc_time_s, t, db)-data.edc_db)/np.sqrt(len(data.edc_db)))
                if len(echoes):
                    measured = data.early_energy_ratio[cache.echo_matches]
                    row.extend(10*np.log10(np.maximum(echoes, 1e-20)/np.maximum(measured, 1e-20))/3)
                    row.extend((cache.times[cache.echo_indices]-obs.echoes_s[cache.echo_matches])/.0002)
                # Repeat residual blocks: sqrt(count) is wrong for soft-L1.
                out.extend(row*int(counts[i]))
            out.extend((params[spec['coefficients']]-spec['prior'])/config.prior_sigma)
            if spec['rt_index'] is not None:
                out.append((params[spec['rt_index']]-np.log(late_priors[band]))/.2)
        return np.array(out)

    def optimize(self, counts, late_priors):
        specs, starts, low, high, config = self.specs, self.starts, self.low, self.high, self.config
        guesses = [x.copy() for x in starts]
        for band, spec in specs.items():
            if spec['rt_index'] is not None:
                for x in guesses:
                    x[spec['rt_index']] = np.log(late_priors[band])
        return [least_squares(self.residual, x, args=(counts, late_priors), bounds=(low, high),
                              loss="soft_l1", max_nfev=config.max_nfev, diff_step=1e-3)
                for x in guesses]

    def bootstrap(self, counts):
        observations, specs = self.observations, self.specs
        priors = {b: spec['target'] for b, spec in specs.items()}
        resampled_priors, valid = {}, True
        for band, spec in specs.items():
            rt_values = [o.bands[band].values[k] for i, o in enumerate(observations)
                         for k in ("t30", "t20") for _ in range(int(counts[i]))
                         if o.bands[band].values[k] is not None]
            valid &= any(counts[i] for i in spec['usable']) and (bool(rt_values) or spec['rt_index'] is None)
            resampled_priors[band] = float(np.clip(np.median(rt_values), .03, 5)) if rt_values else priors[band]
        if not valid:
            return None
        candidates = self.optimize(counts, resampled_priors)
        boot = min(candidates, key=lambda f: f.cost)
        if not boot.success:
            return None
        return self.decode(boot.x)


_BOOTSTRAP_PROBLEM = None


def _init_bootstrap_worker(problem):
    global _BOOTSTRAP_PROBLEM
    _BOOTSTRAP_PROBLEM = problem


def _bootstrap_worker(counts):
    if _BOOTSTRAP_PROBLEM is None:
        raise RuntimeError("bootstrap worker was not initialized")
    return _BOOTSTRAP_PROBLEM.bootstrap(counts)


def _bootstrap_results(problem, boot_weights, workers):
    # Repeated cluster draws are the same objective. Fit each unique draw once,
    # but restore its original position and multiplicity in the report.
    unique = {tuple(map(int, counts)): counts for counts in boot_weights}
    if workers == 1 or len(unique) < 2:
        results = map(problem.bootstrap, unique.values())
        by_counts = dict(zip(unique, results))
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        # Spawn avoids inheriting active BLAS threads or another fit's state.
        # Errors propagate; there is no unreported serial fallback.
        with ProcessPoolExecutor(max_workers=min(workers, len(unique)),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_bootstrap_worker, initargs=(problem,)) as pool:
            by_counts = dict(zip(unique, pool.map(_bootstrap_worker, unique.values())))
    return [by_counts[tuple(map(int, counts))] for counts in boot_weights]


def fit_room(observations: Iterable[Observation] | None = None, model: RoomModel | None = None, *,
             config: FitConfig | None = None, manifest=None, rirs=None,
             initial_model: RoomModel | None = None, bootstrap_workers: int = 1) -> FitResult | RoomModel:
    """Fit training features, or adapt main's keyword-only pipeline contract.

    The feature API returns FitResult. The manifest/rirs/initial_model pipeline
    API returns RoomModel, carrying its held-out report in provenance.
    bootstrap_workers is an execution option only: ordered draws, objectives,
    starts and output metadata are unchanged. The caller budgets native threads
    per process. Spawned callers must use a guarded Python entry point.
    """
    if isinstance(bootstrap_workers, bool) or not isinstance(bootstrap_workers, int) or bootstrap_workers < 1:
        raise ValueError("bootstrap_workers must be a positive integer")
    config = config or FitConfig()
    if manifest is not None:
        if observations is not None or model is not None or rirs is None or initial_model is None:
            raise ValueError("pipeline fitting requires only manifest, rirs and initial_model")
        return _fit_pipeline(manifest, rirs, initial_model, config, bootstrap_workers)
    if observations is None or model is None or rirs is not None or initial_model is not None:
        raise ValueError("feature fitting requires observations and model")
    observations = list(observations)
    check_split(observations)
    if not observations or any(o.split != "train" for o in observations):
        raise ValueError("fit_room accepts training positions only")
    if any(any(b not in o.bands for b in config.bands) for o in observations):
        raise ValueError("observation bands do not match fit configuration")
    requested_config = config
    support = tuple(b for b in OCTAVE_HZ if
                    all(b in _neighbor_bands(config, o.fs) and b in o.bands for o in observations) and
                    any(any(o.bands[b].values[k] is not None for k in ("t20", "t30")) for o in observations))
    config = config.model_copy(update={"bands": (*config.bands, *support)})
    caches, cache_bytes = [], 0
    for obs in observations:
        cache = _cache(model, obs, config)
        cache_bytes += cache.early_basis.nbytes
        if cache_bytes > 1536*1024**2:
            raise ValueError("early fitting bases exceed the 1.5 GiB cache budget")
        caches.append(cache)
    descriptions = model.provenance.get("surface_descriptions", {})
    # Identical described surfaces share a coefficient. Unspecified surfaces share one broad prior.
    keys = [descriptions.get(s, "unspecified") for s in config.fit_surfaces]
    groups = list(dict.fromkeys(keys))
    mapping = {s: groups.index(k) for s, k in zip(config.fit_surfaces, keys)}
    alpha0 = np.array([model.absorption[s] for s in SURFACES])
    late0 = model.late_t60().copy()
    late0 = np.where(np.isfinite(late0), late0, 1.)
    outcomes, gain_db, bootstrap_rows = {}, {}, {}
    keys_positions = list(dict.fromkeys(o.position_key for o in observations))
    cluster = np.array([keys_positions.index(o.position_key) for o in observations])
    rng = np.random.default_rng(config.seed)
    boot_weights = [np.bincount(rng.integers(len(keys_positions), size=len(keys_positions)),
                               minlength=len(keys_positions))[cluster]
                    for _ in range(config.bootstrap)] if len(keys_positions) >= 3 else []
    specs, initial, low, high = {}, [], [], []
    weights = np.ones(len(observations), int)
    for band in config.bands:
        b = OCTAVE_HZ.index(band)
        usable = [i for i, o in enumerate(observations) if o.bands[band].energy > 0 and
                  (len(o.bands[band].edc_db) or len(caches[i].echo_indices)) and
                  (band not in support or any(o.bands[band].values[k] is not None for k in ("t20", "t30")))]
        if not usable:
            outcomes[str(band)] = {"status": "unavailable", "reason": "no supported decay or matched early echoes"}
            continue
        prior = np.array([np.mean([alpha0[SURFACES.index(s), b] for s, g in mapping.items() if g == j]) for j in range(len(groups))])
        rt_values = [o.bands[band].values[k] for o in observations for k in ("t30", "t20") if o.bands[band].values[k] is not None]
        target = float(np.clip(np.median(rt_values), .03, 5)) if rt_values else float(np.clip(late0[b], .03, 5))
        override = config.late and config.late_override and bool(rt_values)
        offset = len(initial)
        initial.extend(np.clip(prior, .0011, .9899))
        low.extend([.001]*len(groups)); high.extend([.99]*len(groups))
        rt_index = len(initial) if override else None
        if override:
            initial.append(np.log(target)); low.append(np.log(.03)); high.append(np.log(5.))
        specs[band] = dict(b=b, usable=usable, prior=prior, target=target,
                           coefficients=slice(offset, offset+len(groups)), rt_index=rt_index)
    if not specs:
        raise ValueError("no usable fitted bands; no model can be claimed")

    initial = np.array(initial)
    other = initial.copy()
    for spec in specs.values():
        other[spec['coefficients']] = .5
    starts = [initial, other]
    priors = {b: s['target'] for b, s in specs.items()}

    # Bands are coupled by interpolation and filters. Independent band optima
    # describe different rooms and cease to be optima when put into one model.
    problem = _FitProblem(model, observations, caches, config, specs, mapping,
                          alpha0, late0, low, high, starts)
    fits = problem.optimize(weights, priors)
    best = min(fits, key=lambda f: f.cost)
    fitted, late = problem.decode(best.x)
    final_predictions = [_expected_all(model, o, c, fitted, late, config)
                         for o, c in zip(observations, caches)]
    for band, spec in specs.items():
        outcomes[str(band)] = {"status": "fit" if best.success else "iteration_limit",
                               "joint_cost": float(best.cost), "nfev": int(best.nfev),
                               "late_override_fitted": spec['rt_index'] is not None,
                               "starts": [{"cost": float(f.cost), "parameters": f.x.tolist(),
                                           "success": bool(f.success)} for f in fits],
                               "usable_training_observations": len(spec['usable'])}
        ratios = [10*np.log10(max(observations[i].bands[band].energy, 1e-30)/
                             max(final_predictions[i][band][3], 1e-30)) for i in spec['usable']]
        gain_db[str(band)] = float(np.median(ratios))
    samples, failures = [], 0
    for sample in _bootstrap_results(problem, boot_weights, bootstrap_workers):
        if sample is None:
            failures += 1
        else:
            samples.append(sample)
    for band, spec in specs.items():
        b = spec['b']
        rows = [np.r_[aa[:, b], tt[b]] for aa, tt in samples]
        bootstrap_rows[str(band)] = {"successful": len(rows), "failed": failures,
            "replicates": [row.tolist() for row in rows],
            "percentiles_2_5_97_5": np.percentile(rows, [2.5, 97.5], axis=0).tolist() if len(rows) >= 2 else None,
            "columns": [*SURFACES, "late_t60_s"], "units": [*["energy_fraction"]*6, "s"]}
    uncertainty = {"method": "bootstrap whole training source/receiver positions", "requested": config.bootstrap,
                   "independent_positions": len(keys_positions), "bands": bootstrap_rows,
                   "limits": "conditional on geometry, associations, grouping, device, fixed scattering and obstacle priors; excludes model discrepancy"}
    data = model.model_dump()
    data.update(absorption={s: fitted[j].tolist() for j, s in enumerate(SURFACES)},
                late_t60_s=late.tolist() if config.late and config.late_override else model.late_t60_s,
                uncertainty=uncertainty,
                provenance={**model.provenance, "fit_method": "bounded-soft-l1-shared-forward-energy/2",
                            "fit_backend": config.backend,
                            "seed": config.seed, "training_ids": [o.id for o in observations],
                            "training_positions": [list(k) for k in keys_positions],
                            "fitted_bands_hz": [int(b) for b in gain_db],
                            "evaluation_bands_hz": list(requested_config.bands),
                            "support_bands_hz": list(support),
                            "unvalidated_bands_hz": [b for b in OCTAVE_HZ if str(b) not in gain_db],
                            "spectral_gain_db": gain_db, "scattering": "fixed; absorption/scattering are not jointly identified",
                            "validity": json.loads(json.dumps(asdict(VALIDITY)))})
    from sensorsim.licensing import lineage_metadata
    lineage = lineage_metadata(model, [o.metadata for o in observations])
    if lineage:
        data['provenance']['lineage'] = lineage
    result = RoomModel.model_validate(data)
    return FitResult(result, requested_config, {"version": "room-fit/2", "backend": config.backend, "drr_definition": "drr/1",
                                     "support_bands_hz": list(support),
                                     "support_policy": "adjacent complete octaves with valid training T20/T30; no added evaluation gates",
                                     "bands": outcomes, "surface_groups": mapping,
                                     "uncertainty": uncertainty, "spectral_gain_db": gain_db,
                                     "training_ids": [o.id for o in observations],
                                     "limitations": VALIDITY.conditions})


def predict(result: FitResult, source_xyz_m, receiver_xyz_m, fs: int, *, id: str = "prediction") -> RIR:
    """Full waveform prediction with a stable ID-derived seed, independent of batching."""
    seed = int.from_bytes(hashlib.sha256(f"{result.config.seed}:{id}".encode()).digest()[:4], "little")
    out = synthesize(result.model, _position(result.model, source_xyz_m), _position(result.model, receiver_xyz_m), fs,
                     seed=seed, config=SynthConfig(late=result.config.late, max_order=None if result.config.late else 1))
    direct = round(out.metadata["direct_delay_s"]*fs)+out.metadata["latency_samples"]
    return RIR(fs, out.samples, tuple(source_xyz_m), tuple(receiver_xyz_m),
               {**out.metadata, "id": id, "synthetic": True, "direct_sample": direct})


def compare(result: FitResult, observed: Observation, predicted: Observation) -> list[dict]:
    """All requested gates, including unavailable measurements and invalid predictions."""
    if observed.split != "train" and list(observed.position_key) in result.model.provenance["training_positions"]:
        raise ValueError("held-out coordinate was used for fitting")
    rows = []
    for band in result.config.bands:
        truth, pred = observed.bands[band], predicted.bands[band]
        for metric in (*METRICS, "spectral_db"):
            if metric == "spectral_db":
                supported = str(band) in result.report["spectral_gain_db"]
                value = 10*np.log10(pred.energy/truth.energy)+result.report["spectral_gain_db"][str(band)] if supported and min(pred.energy, truth.energy) > 0 else None
                target, estimate = 0. if truth.energy > 0 else None, value
            else:
                target, estimate = truth.values[metric], pred.values[metric]
            error = estimate-target if target is not None and estimate is not None else None
            scaled = error/target if error is not None and metric in ("edt", "t20", "t30") else error
            status = "UNAVAILABLE" if target is None else "FAIL" if scaled is None or abs(scaled) > GATES[metric] else "PASS"
            rows.append({"id": observed.id, "split": observed.split, "band_hz": band, "metric": metric,
                         "measured": target, "predicted": estimate, "signed_error": error,
                         "relative_error": scaled if metric in ("edt", "t20", "t30") else None,
                         "tolerance": GATES[metric], "status": status,
                         "reason": truth.reasons.get(metric) if target is None else pred.reasons.get(metric) if estimate is None else None})
    return rows


def early_diagnostics(model: RoomModel, observed: Observation) -> dict:
    source, receiver = _position(model, observed.source_xyz_m), _position(model, observed.receiver_xyz_m)
    paths = image_paths(model, source, receiver, max_order=1)
    room_paths = len(paths.delay_s)
    if model.obstacles:
        from .obstacles import apply_obstacles
        paths, _ = apply_obstacles(model, paths, source, receiver, SynthConfig(max_order=1, late=False))
    reflective = (paths.wall_hits.sum(1) > 0) | (np.arange(len(paths.delay_s)) >= room_paths)
    bands = [OCTAVE_HZ.index(b) for b in observed.bands] or list(range(7))
    reflective &= np.any(paths.gain_octave[:, bands] > 0, axis=1)
    times = paths.delay_s[reflective]-np.linalg.norm(source-receiver)/model.sound_speed_m_s
    contaminated = bool(np.any(times < .0025))
    times = times[(times >= .0005) & (times <= .08)]
    base = {"drr_contaminated": contaminated, "predicted_excess_s": times.tolist(),
            "observed_excess_s": observed.echoes_s.tolist(),
            "association": "positive-gain first-order path candidates, not labelled truth"}
    if not len(observed.echoes_s) or not len(times):
        return {**base, "matched": 0, "unmatched_predicted": len(times),
                "unmatched_observed": len(observed.echoes_s), "errors_s": [], "passes_0_2_ms": 0}
    cost = abs(times[:, None]-observed.echoes_s[None])
    i, j = linear_sum_assignment(cost)
    keep = cost[i, j] <= .002
    error = times[i[keep]]-observed.echoes_s[j[keep]]
    return {**base, "matched": len(error), "unmatched_predicted": len(times)-len(error), "unmatched_observed": len(observed.echoes_s)-len(error),
            "errors_s": error.tolist(), "passes_0_2_ms": int(np.sum(abs(error) <= .0002)),
            }


def fast_waveform_difference(result: FitResult, observed: Observation, predicted: Observation) -> list[dict]:
    cache = _cache(result.model, observed, result.config)
    out = []
    for band in result.config.bands:
        b = OCTAVE_HZ.index(band)
        alpha = np.array([result.model.absorption[s][b] for s in SURFACES])
        fast = _expected(result.model, observed, cache, band, alpha, result.model.late_t60()[b], result.config)[0]
        for key in METRICS:
            slow = predicted.bands[band].values[key]
            out.append({"id": observed.id, "band_hz": band, "metric": key,
                        "fast": fast[key], "waveform": slow,
                        "signed_difference": fast[key]-slow if fast[key] is not None and slow is not None else None})
    return out


def fingerprint() -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pydantic", "PyYAML")}
    # Source identity of this package only. The previous check=True git call raised in any
    # installed copy, and walked up into whichever repository contained the venv.
    from sensorsim.core.fingerprint import source_revision
    revision = source_revision()
    return {"git": revision["commit"], "dirty": revision["dirty"], "package_version": revision["package_version"],
            "python": platform.python_version(), "platform": platform.platform(), "dependencies": versions,
            "backend": "numpy/scipy CPU; no AMD inverse backend", "thread_limit": 2}


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def _holdout_receivers(manifest):
    # Explicit checks replace the historical automatic split only when supplied.
    # Their audio is excluded; all ordinary profiling positions may train.
    if any(t.role == "check" for t in manifest.takes):
        return set()
    positions = sorted({tuple(r) for t in manifest.takes if t.type == "sweep" and not t.is_reference
                        for r in np.atleast_2d(t.receiver_xyz_m)})
    if len(positions) < 2:
        raise ValueError("at least two receiver positions are needed for a held-out comparison")
    return set(positions[-max(1, int(np.ceil(len(positions)*.25))):])


def _session_report(result, features, source_refs):
    rows, approximations, early = [], [], []
    for obs in features:
        prediction = observe(predict(result, obs.source_xyz_m, obs.receiver_xyz_m, obs.fs, id=obs.id),
                             id=obs.id, config=result.config, reference=source_refs[obs.id], split=obs.split)
        rows.extend(compare(result, obs, prediction))
        early.append({"id": obs.id, "split": obs.split, **early_diagnostics(result.model, obs)})
        if obs.split == "train":
            approximations.extend(fast_waveform_difference(result, obs, prediction))
    return {**result.report, "environment": fingerprint(), "config": result.config.model_dump(mode="json"),
            "split": [{"id": o.id, "split": o.split, "position": list(o.position_key)} for o in features],
            "observations": [{"id": o.id, "metadata": o.metadata} for o in features],
            "metrics": rows, "early": early, "fast_vs_waveform": approximations}


def _fit_pipeline(manifest, rirs, initial_model, config, bootstrap_workers=1):
    """Adapt pipeline.py mono RIRs in recording/channel order without re-extraction."""
    from sensorsim.realdata.audio_io import probe
    holdout = _holdout_receivers(manifest)
    check_files = {str(t.file.resolve()) for t in manifest.takes if t.role == "check"}
    by_file = {}
    for response in rirs:
        filename = response.metadata.get("recording_file")
        if filename is None or response.samples.ndim != 1 or response.fs != manifest.fs:
            raise ValueError("pipeline RIR requires recording_file, mono samples and manifest sample rate")
        if str(Path(filename).resolve()) in check_files:
            continue
        by_file.setdefault(str(Path(filename).resolve()), []).append(response)
    references, prepared, used = {}, [], set()
    for i, take in enumerate(manifest.takes):
        if take.type != "sweep" or take.role == "check":
            continue
        filename = str(take.file.resolve())
        if filename in used:
            raise ValueError("pipeline recording_file must identify a unique sweep take")
        used.add(filename)
        responses = by_file.get(filename, [])
        if len(responses) != probe(take.file)["channels"]:
            raise ValueError("pipeline RIR channels do not match the sweep recording")
        key = take.speaker_id if take.speaker_id is not None else tuple(take.source_xyz_m)
        receivers = np.atleast_2d(take.receiver_xyz_m)
        for channel, response in enumerate(responses):
            receiver = tuple(receivers[channel % len(receivers)])
            if (not np.array_equal(response.source_xyz_m, take.source_xyz_m) or
                    not np.array_equal(response.receiver_xyz_m, receiver)):
                raise ValueError("pipeline RIR coordinates or channel order differ from manifest")
            meta = dict(response.metadata)
            if "direct_samples" in meta:
                if len(meta["direct_samples"]) != len(responses):
                    raise ValueError("pipeline direct_samples count differs from recording channels")
                meta["direct_sample"] = meta["direct_samples"][channel]
            one = RIR(response.fs, response.samples, tuple(take.source_xyz_m), receiver, meta)
            if take.is_reference:
                if key in references:
                    raise ValueError("select one matched REF channel/take per source")
                references[key] = one
            else:
                prepared.append((one, f"take{i}:ch{channel}", key))
    if set(by_file) != used:
        raise ValueError("pipeline RIR contains an unknown or non-sweep recording")
    features, source_refs = [], {}
    for response, id, key in prepared:
        ref = references.get(key)
        features.append(observe(response, id=id, config=config, reference=ref,
                                split="test" if tuple(response.receiver_xyz_m) in holdout else "train"))
        source_refs[id] = ref
    check_split(features)
    if tuple(initial_model.shoebox_dims_m) != tuple(manifest.shoebox_dims_m or ()):
        raise ValueError("pipeline initial geometry must match the manifest; use CLI/API echo inference for absent dimensions")
    prior = model_from_manifest(manifest)
    model = RoomModel.model_validate({**initial_model.model_dump(), "absorption": prior.absorption,
                                     "obstacles": prior.obstacles,
                                     "provenance": {**initial_model.provenance, **prior.provenance}})
    execution = {"bootstrap_workers": bootstrap_workers} if bootstrap_workers != 1 else {}
    result = fit_room([o for o in features if o.split == "train"], model, config=config, **execution)
    report = _session_report(result, features, source_refs)
    return RoomModel.model_validate({**result.model.model_dump(),
                                    "provenance": {**result.model.provenance, "pipeline_fit_report": report}})


def run_manifest(manifest_path, out, *, config: FitConfig | None = None) -> dict:
    """Canonical recording-kit CLI. REF/noise/real takes never enter the fit split."""
    config = config or FitConfig()
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    manifest = load_manifest(manifest_path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output directory must be empty; preserve previous evidence")
    out.mkdir(parents=True, exist_ok=True)
    references = {}
    def source_key(take):
        return take.speaker_id if take.speaker_id is not None else tuple(take.source_xyz_m)
    for i, take in enumerate(manifest.takes):
        if take.type == "sweep" and take.is_reference:
            if source_key(take) in references:
                raise ValueError("multiple REF takes for one source require an explicit selection")
            rir = extract_take(manifest, i, rir_duration_s=config.rir_duration_s)
            if rir.samples.ndim == 2 and rir.samples.shape[1] != 1:
                raise ValueError("select one matched REF channel per source")
            rir.samples = rir.samples.reshape(-1)
            rir.receiver_xyz_m = tuple(np.atleast_2d(rir.receiver_xyz_m)[0])
            references[source_key(take)] = rir
    candidates = [(i, t) for i, t in enumerate(manifest.takes) if t.type == "sweep" and not t.is_reference]
    holdout = _holdout_receivers(manifest)
    all_features = []
    source_refs = {}
    # Extraction is measured preprocessing; train/test labels are fixed before reading audio.
    for i, take in candidates:
        rir = extract_take(manifest, i, rir_duration_s=config.rir_duration_s)
        samples = rir.samples[:, None] if rir.samples.ndim == 1 else rir.samples
        receivers = np.atleast_2d(rir.receiver_xyz_m)
        for channel in range(samples.shape[1]):
            receiver = tuple(receivers[channel % len(receivers)])
            meta = dict(rir.metadata)
            if "direct_samples" in meta:
                meta["direct_sample"] = meta["direct_samples"][channel]
            one = RIR(rir.fs, samples[:, channel], rir.source_xyz_m, receiver, meta)
            id = f"take{i}:ch{channel}"
            ref = references.get(source_key(take))
            split = "check" if take.role == "check" else "test" if receiver in holdout else "train"
            all_features.append(observe(one, id=id, config=config, split=split, reference=ref))
            source_refs[id] = ref
    check_split(all_features)
    train = [o for o in all_features if o.split == "train"]
    if manifest.shoebox_dims_m is None:
        geometry = estimate_geometry(train, max_dimension_m=config.geometry_max_m)
        manifest = manifest.model_copy(update={"shoebox_dims_m": tuple(geometry["shoebox_dims_m"])})
        model = model_from_manifest(manifest)
        model = RoomModel.model_validate({**model.model_dump(), "provenance": {**model.provenance, **geometry}})
    else:
        model = model_from_manifest(manifest)
    result = fit_room(train, model, config=config)
    session_report = _session_report(result, all_features, source_refs)
    input_paths = {t.file for t in manifest.takes}
    input_paths.update(t.excitation.path for t in manifest.takes if t.excitation.type == "known_signal")
    source_hashes = {}
    for path in sorted(input_paths):
        with path.open("rb") as stream:
            source_hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {**session_report,
              "source_sha256": source_hashes, "resolved_manifest": manifest.model_dump(mode="json"),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "manifest": str(manifest_path)}
    (out/"room_model.json").write_text(result.model.to_json()+"\n")
    _json(out/"fit_report.json", report)
    _json(out/"resolved_config.json", config.model_dump(mode="json"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--rir-duration", type=float, default=1.5)
    args = parser.parse_args(argv)
    report = run_manifest(args.manifest, args.out, config=FitConfig(bootstrap=args.bootstrap, seed=args.seed, rir_duration_s=args.rir_duration))
    heldout = [r for r in report["metrics"] if r["split"] != "train"]
    print(json.dumps({"model": str(args.out/"room_model.json"), "heldout": {s: sum(r["status"] == s for r in heldout) for s in ("PASS", "FAIL", "UNAVAILABLE")}}))


if __name__ == "__main__":
    main()
