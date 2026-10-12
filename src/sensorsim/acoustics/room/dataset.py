"""Streaming mono WAV augmentation datasets, including externally extracted dry sounds.

CLI: python -m sensorsim.acoustics.room.dataset --help
One clip at a time; full RIR convolution tails retained; no waveform in the manifest.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal
from scipy.io import wavfile

from sensorsim.acoustics.room.augment import AugmentationConfig, augment_receiver, augment_source, training_export_level
from sensorsim.acoustics.room.model import OCTAVE_HZ, RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, SyntheticRIR, synthesize
from sensorsim.acoustics.bands import BandGrid, BandSpec
from sensorsim.acoustics.audio_boundary import encode_pcm16
from sensorsim.core.fingerprint import sha256_file, source_revision
from sensorsim.core.serialization import atomic_json, canonical_json
from .source_reference import SourceReference, source_at_one_m
from .metrics import source_level_metadata


class SamplingSpec(BaseModel):
    """Independent uniform volume sampling, truncated by walls and solid boxes."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    wall_clearance_m: float = Field(default=.25, gt=0.)
    obstacle_clearance_m: float = Field(default=.1, ge=0.)
    source_height_m: tuple[float, float] = (.5, 1.8)
    receiver_height_m: tuple[float, float] = (.5, 1.8)
    min_separation_m: float = Field(default=.25, gt=0.)

    @model_validator(mode="after")
    def _heights(self) -> "SamplingSpec":
        if any(not 0 < low <= high for low, high in (self.source_height_m, self.receiver_height_m)):
            raise ValueError("height bounds must be positive and ordered")
        return self


class PositionPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    source_xyz_m: tuple[float, float, float]
    receiver_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class DryClip:
    """Declared relative source at 1 m, or an explicit extracted-source reference.

    source_group ties excerpts of one recording/event for splitting. Use
    from_result for DryResult; never discard its Case-B REF reference.
    """
    label: str
    audio: Path | str | np.ndarray
    fs: int | None = None
    provenance: dict | None = None
    source_group: str | None = None

    source_reference: SourceReference = field(default_factory=SourceReference)

    @classmethod
    def from_result(cls, label, result, *, provenance=None, source_group=None):
        if not isinstance(result.source_reference, SourceReference):
            raise ValueError("DryResult needs explicit source reference metadata")
        from sensorsim.licensing import lineage_metadata
        lineage = lineage_metadata(result.report, provenance or {})
        if lineage:
            provenance = {**(provenance or {}), "lineage": lineage}
        return cls(label, result.samples, result.fs, provenance, source_group, result.source_reference)


# Compatibility names retain the canonical providers, never another implementation.
_canonical = canonical_json
_hash_file = sha256_file


def read_dry(clip: DryClip, target_fs: int) -> tuple[np.ndarray, dict]:
    from sensorsim.licensing import require_commercial_lineage
    require_commercial_lineage(clip.provenance or {},
        {"path": str(clip.audio)} if isinstance(clip.audio, (str, Path)) else {}, context="dry source")
    from sensorsim.sources.dataset_routes import route_metadata, validate_route_audio
    route = route_metadata(clip)
    level_metadata = source_level_metadata(clip.source_reference,
        kind=(clip.provenance or {}).get('reference_kind', 'relative_only'))
    # The once-only distance adapter sets target=source_1m, so keep the earlier
    # effective-source qualification from the extraction provenance as well.
    if (clip.provenance or {}).get('level_reference', {}).get('effective_source'):
        level_metadata['effective_source'] = True
    if isinstance(target_fs, bool) or not isinstance(target_fs, int) or target_fs < 1000:
        raise ValueError("target_fs must be an integer >=1000")
    if not isinstance(clip.label, str) or not clip.label.strip():
        raise ValueError("dry clip requires a nonempty class label")
    if clip.source_group is not None and (not isinstance(clip.source_group, str) or
                                          not clip.source_group.strip()):
        raise ValueError("source_group must be a nonempty recording/event identifier")
    if isinstance(clip.audio, (str, Path)):
        path = Path(clip.audio).resolve()
        fs, audio = wavfile.read(path)
        identity = {"path": str(path), "sha256": _hash_file(path)}
    else:
        fs, audio = clip.fs, np.asarray(clip.audio)
        if audio.dtype.kind not in "biuf":
            raise ValueError(f"dry clip {clip.label!r} has unsupported audio dtype {audio.dtype}")
        raw = np.ascontiguousarray(audio)
        identity_hash = hashlib.sha256()
        identity_hash.update(raw.dtype.str.encode("ascii"))
        identity_hash.update(raw.tobytes())
        identity = {"path": None, "sha256": identity_hash.hexdigest(),
                    "hash_encoding": f"dtype-tagged:{raw.dtype.str}:c-order"}
    if not isinstance(fs, (int, np.integer)) or fs < 1000:
        raise ValueError("dry array requires an integer fs >=1000")
    if audio.ndim != 1 or len(audio) < 2:
        raise ValueError("dry audio must be mono with at least two samples (no implicit channel mix)")
    if audio.dtype.kind not in "biuf":
        raise ValueError(f"dry clip {clip.label!r} has unsupported audio dtype {audio.dtype}")
    integer_pcm = audio.dtype.kind in "ui"
    if audio.dtype.kind == "u":
        audio = (audio.astype(float)-128)/128 if audio.dtype.itemsize == 1 else audio.astype(float)
    elif audio.dtype.kind == "i":
        audio = audio.astype(float)/2**(8*audio.dtype.itemsize-1)
    else:
        audio = audio.astype(float)
    if integer_pcm and clip.source_reference.units == "Pa":
        calibration = clip.source_reference.calibration
        if calibration is None:
            raise ValueError("Pa source in integer PCM requires explicit peak-FS calibration")
        audio *= calibration.pa_per_peak_fs
    if not np.isfinite(audio).all():
        raise ValueError("nonfinite dry audio")
    # File/container hashes do not identify duplicate decoded audio. Include the
    # sample rate, and canonicalise both PCM and floating WAV/array representations.
    content = hashlib.sha256(int(fs).to_bytes(8, "little"))
    content.update(memoryview(np.ascontiguousarray(audio, dtype="<f8")).cast("B"))
    validate_route_audio(clip, audio, fs)
    audio, reference, source_scale = source_at_one_m(audio, clip.source_reference)
    normalized_content = hashlib.sha256(int(fs).to_bytes(8, "little"))
    normalized_content.update(memoryview(np.ascontiguousarray(audio, dtype="<f8")).cast("B"))
    original_n = len(audio)
    if fs != target_fs:
        divisor = math.gcd(int(fs), target_fs)
        audio = signal.resample_poly(audio, target_fs//divisor, int(fs)//divisor)
        reference = replace(reference, latency_samples=round(reference.latency_samples*target_fs/fs))
    return audio, {**identity, **route, "label": clip.label, "original_fs": int(fs),
        "content_sha256": content.hexdigest(), "normalized_content_sha256": normalized_content.hexdigest(),
        "source_group": clip.source_group, "source_reference": reference.as_dict(),
        "input_source_reference": clip.source_reference.as_dict(), "source_scale": source_scale,
        "level_reference": level_metadata,
        "resampling_policy": "scipy.resample_poly; zero boundary; centered FIR; full ceil(n*ratio) support",
        "original_samples": original_n, "resampled_fs": target_fs,
        "provenance": clip.provenance or {},
        "dryness": "caller-supplied; residual recording/speaker/mic coloration may remain"}


def _read_dataset_source(clip: DryClip, fs: int, directory: Path) -> tuple[np.ndarray, dict]:
    """Keep sources inside a run relocatable without changing external input paths."""
    audio, info = read_dry(clip, fs)
    if info["path"] is not None:
        path = Path(info["path"])
        run_root = directory.resolve().parent
        if path.is_relative_to(run_root):
            info["path"] = str(Path("..") / path.relative_to(run_root))
    return audio, info


def _bounds(model: RoomModel, spec: SamplingSpec, heights) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(3, spec.wall_clearance_m)
    high = np.array(model.shoebox_dims_m)-spec.wall_clearance_m
    low[2], high[2] = max(low[2], heights[0]), min(high[2], heights[1])
    if np.any(low > high) or np.any(low[:2] == high[:2]):
        raise ValueError("clearance/height bounds leave no sampling volume")
    return low, high


def validate_positions(model: RoomModel, spec: SamplingSpec, pair: PositionPair) -> None:
    for xyz, heights in ((pair.source_xyz_m, spec.source_height_m),
                         (pair.receiver_xyz_m, spec.receiver_height_m)):
        low, high = _bounds(model, spec, heights)
        if np.any(np.asarray(xyz) < low) or np.any(np.asarray(xyz) > high):
            raise ValueError("target position violates wall/height bounds")
        if any(box.contains(xyz, spec.obstacle_clearance_m) for box in model.obstacles):
            raise ValueError("target position violates obstacle clearance")
    if np.linalg.norm(np.array(pair.source_xyz_m)-pair.receiver_xyz_m) < spec.min_separation_m:
        raise ValueError("source/receiver separation below sampling minimum")


def sample_positions(model: RoomModel, spec: SamplingSpec, count: int, seed: int) -> list[PositionPair]:
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    bounds = [_bounds(model, spec, h) for h in (spec.source_height_m, spec.receiver_height_m)]
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(count):
        for attempt in range(10000):
            source, receiver = [rng.uniform(low, high) for low, high in bounds]
            pair = PositionPair(source_xyz_m=tuple(source), receiver_xyz_m=tuple(receiver))
            try:
                validate_positions(model, spec, pair)
            except ValueError:
                continue
            pairs.append(pair)
            break
        else:
            raise ValueError("no legal source/receiver pair after 10000 attempts; relax sampling bounds")
    return pairs


def _connected_groups(keys_per_item) -> dict[int, list[int]]:
    """Transitive identity components in input order; shared by both split axes.

    Preserve root labels/insertion order because split RNGs permute those labels.
    Keys are supplied by each policy; position and recording identity stay distinct.
    """
    parent = list(range(len(keys_per_item)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, keys in enumerate(keys_per_item):
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i in range(len(keys_per_item)):
        groups.setdefault(root(i), []).append(i)
    return groups


def split_positions(pairs: Sequence[PositionPair], val_fraction: float = .2, *, seed: int = 0,
                     policy: Literal["legacy-both", "source-region"] = "legacy-both", region_size_m: float = .25
                     ) -> list[Literal["train", "val"]]:
    """Split source subregions, or retain the historical both-role policy.

    source-region groups fixed-size source cells and ignores receiver coordinates;
    legacy-both groups connected positions across BOTH roles before splitting.

    Equal positions (rounded to 1 micrometre) can never leak through a shared
    source or a receiver that becomes a source. This does not guarantee spatial
    separation of neighbouring points or source-clip/room disjointness.
    """
    if policy not in ("legacy-both", "source-region") or not np.isfinite(region_size_m) or region_size_m <= 0:
        raise ValueError("invalid split policy or source region size")
    if not 0 <= val_fraction < 1 or not pairs:
        raise ValueError("need positions and val_fraction in [0,1)")
    if val_fraction == 0:
        return ["train"]*len(pairs)
    keys = ([[tuple(np.floor(np.asarray(pair.source_xyz_m)/region_size_m).astype(int))] for pair in pairs]
            if policy == "source-region" else
            [[tuple(np.round(point, 6)) for point in (pair.source_xyz_m, pair.receiver_xyz_m)] for pair in pairs])
    groups = _connected_groups(keys)
    if len(groups) < 2:
        raise ValueError("all pairs share positions or source subregions; cannot make nonempty spatial train/val")
    order = np.random.default_rng(seed).permutation(list(groups))
    target = max(1, round(len(pairs)*val_fraction))
    labels, chosen = ["train"]*len(pairs), 0
    for key in order[:-1]:  # reserve at least one component for training
        for i in groups[key]:
            labels[i] = "val"
        chosen += len(groups[key])
        if chosen >= target:
            break
    return labels


def _anchored_position(survey, model, label, subset, fraction, rng, sampling, *, site_index=None,
                       scene_receiver=None):
    """Hold out a guarded upper slab WITHIN each site; keep every site in train.

    Point sites retain their exact source coordinate with a group/content split;
    they explicitly have no spatial holdout. Receiver jitter cannot define a split.
    """
    sites = survey.source_sites.get(label)
    if not sites:
        raise ValueError(f"room.yaml needs source_sites for {label!r}")
    # Select the site once, independent of split and geometry-rejection draws.
    site_index = int(rng.integers(len(sites))) if site_index is None else site_index
    site = sites[site_index]
    lo, hi = site.xyz-np.array(site.spread_m), site.xyz+np.array(site.spread_m)
    axis = int(np.argmax(hi-lo))
    spatial = bool(fraction and hi[axis]-lo[axis] >= 1e-6)
    boundary = lo[axis]+(1-fraction)*(hi[axis]-lo[axis])
    guard = min(.025, (hi[axis]-lo[axis])*min(fraction,1-fraction)/4) if spatial else 0.
    tol = survey.microphone.tolerance_m
    mic_lo = np.maximum(1e-6, survey.microphone.xyz-tol)
    mic_hi = np.minimum(np.array(model.shoebox_dims_m)-1e-6, survey.microphone.xyz+tol)
    draw_lo, draw_hi = lo.copy(), hi.copy()
    if spatial:
        if subset == "val": draw_lo[axis] = boundary+guard
        else: draw_hi[axis] = boundary-guard
    # A fixed receiver gets one proposal; the caller retries the whole scene.
    for _ in range(10000 if scene_receiver is None else 1):
        source = rng.uniform(draw_lo, draw_hi)
        receiver = (rng.uniform(mic_lo, mic_hi) if scene_receiver is None
                    else np.asarray(scene_receiver, dtype=float))
        if any(box.contains(p, sampling.obstacle_clearance_m) for box in model.obstacles for p in (source, receiver)):
            continue
        if np.linalg.norm(source-receiver) < sampling.min_separation_m:
            continue
        pair = PositionPair(source_xyz_m=tuple(source), receiver_xyz_m=tuple(receiver))
        interval = [float(draw_lo[axis]), float(draw_hi[axis])]
        region = {"id": f"{label}:site{site_index}:axis{axis}:{interval}", "site_index": site_index,
                  "axis": axis, "boundary_m": float(boundary), "guard_m": float(guard), "interval_m": interval,
                  "policy": "within-site held-out upper slab" if spatial else "content/group only; no spatial holdout"}
        return pair, region
    if scene_receiver is not None:
        return None
    raise ValueError(f"no legal {subset} position for {label!r} site {site_index}; check sites/obstacles")


def _ensemble_examples(ensemble, count, seed, split, children, source_pools, all_sources, clips, sampling, fraction):
    schedule = ensemble.member_schedule(count, seed)
    site_counts = {}
    for scene, (subset, child, member_index) in enumerate(zip(split, children, schedule)):
        pool = source_pools[subset]
        selected = pool if all_sources else [int(np.random.default_rng(child.generate_state(1)).choice(pool))]
        if not selected:
            raise ValueError("scene needs at least one source")
        member = ensemble.members[member_index]
        planned = []
        # Keep site, split, member and descendant seeds fixed across retries.
        for index, draw in zip(selected, child.spawn(len(selected))):
            seeds = tuple(map(int, draw.generate_state(4)))
            key = (subset, clips[index].label)
            sites = ensemble.survey.source_sites.get(clips[index].label)
            if not sites:
                raise ValueError(f"room.yaml needs source_sites for {clips[index].label!r}")
            site_index = site_counts.get(key, 0) % len(sites)
            site_counts[key] = site_counts.get(key, 0)+1
            planned.append((index, seeds, site_index, np.random.default_rng(seeds[0])))
        microphone = ensemble.survey.microphone
        mic_lo = np.maximum(1e-6, microphone.xyz-microphone.tolerance_m)
        mic_hi = np.minimum(np.array(member.shoebox_dims_m)-1e-6,
                            microphone.xyz+microphone.tolerance_m)
        receiver_rng = np.random.default_rng(child.generate_state(2))
        for _ in range(10000):
            receiver = receiver_rng.uniform(mic_lo, mic_hi)
            complete = []
            for index, seeds, site_index, rng in planned:
                proposed = _anchored_position(ensemble.survey, member, clips[index].label,
                    subset, fraction, rng, sampling, site_index=site_index,
                    scene_receiver=receiver)
                if proposed is None:
                    break
                pair, region = proposed
                complete.append((pair, subset, index, seeds, member, region, scene, member_index))
            else:
                # No descendant becomes visible until every source is legal.
                yield from complete
                break
        else:
            raise ValueError(f"no legal complete {subset} scene {scene} in 10000 joint draws; "
                             "check shared microphone, sites and obstacles")


def rir_summary(rir: SyntheticRIR) -> dict:
    """Bounded synthetic diagnostics via installed PRA (not stage-1 ISO metrics).

    Reports late T20-derived T60 per supported octave, after the crossfade;
    no noise correction or measured-data validity claim. Missing decay -> null.
    """
    from pyroomacoustics.experimental import measure_rt60
    start = int(np.ceil((rir.metadata["mixing_time_s"]+rir.metadata["crossfade_s"]/2)*rir.fs))
    start += rir.metadata["latency_samples"]
    bands = {}
    for centre in OCTAVE_HZ:
        if centre*np.sqrt(2) >= rir.fs/2 or start >= len(rir.samples)-32:
            bands[str(centre)] = None
            continue
        sos = signal.butter(4, [centre/np.sqrt(2), centre*np.sqrt(2)],
                            btype="bandpass", fs=rir.fs, output="sos")
        x = signal.sosfilt(sos, rir.samples.astype(float))[start:]
        if np.sum(x*x) <= 1e-30:
            bands[str(centre)] = None
            continue
        # PRA may reduce the fit span on insufficient decay: require 25 dB ourselves.
        energy = np.cumsum((x*x)[::-1])[::-1]
        db = 10*np.log10(np.maximum(energy/energy[0], 1e-300))
        if np.count_nonzero((db < -5) & (db > -25)) < 8:
            bands[str(centre)] = None
            continue
        value = float(measure_rt60(x, fs=rir.fs, decay_db=20))
        bands[str(centre)] = value if np.isfinite(value) and value > 0 else None
    grid = BandGrid(tuple(BandSpec("base2", 1, float(b), float(b)) for b in OCTAVE_HZ),
                    "room-rir-quick-summary/1: causal Butterworth", "pressure", 4)
    return {"metric_policy": {"method": "room-rir-quick-summary/1", "band_grid": grid.as_dict(), "band_standard": "diagnostic binary nominal octaves",
            "onset_sample": start, "noise_correction": "none", "decay_fit_db": [-5, -25],
            "extrapolation_db": 60, "settling_samples": 0, "latency": "causal filter delay retained"},
        "method": "PRA measure_rt60 decay_db=20 after crossfade; scipy Butterworth order4 octaves",
        "status": "synthetic diagnostic; not ISO measured-room validation",
        "late_t20_extrapolated_t60_s": bands,
        "energy": float(np.sum(rir.samples.astype(float)**2)),
        "peak": float(np.max(np.abs(rir.samples))),
        "samples": len(rir.samples), "direct_geometric_delay_s": rir.metadata["direct_delay_s"]}


def _source_pools(catalog: list[dict], val_fraction: float, seed: int, *,
                  allow_train_only: bool = False) -> dict[str, list[int]]:
    """Split connected content identities AND caller-declared recording groups.

    Neither renaming a file/label nor assigning a duplicate a different group
    permits leakage. Unknown overlapping crops require a caller source_group;
    equality hashing cannot establish recording, performer or class independence.
    """
    from sensorsim.sources.dataset_routes import declared_source_pools, route_keys
    declared = declared_source_pools(catalog)
    if declared is not None:
        if declared["val"] and val_fraction == 0:
            raise ValueError("preassigned validation sources require validation positions")
        return declared
    if val_fraction == 0:
        return {"train": list(range(len(catalog))), "val": []}
    keys_per_item = []
    for item in catalog:
        keys = [("content", item["content_sha256"])]
        if item.get("normalized_content_sha256") is not None:
            keys.append(("normalized", item["normalized_content_sha256"]))
        if item["source_group"] is not None:
            keys.append(("recording", item["source_group"]))
        keys.extend(route_keys(item))
        keys_per_item.append(keys)
    groups = _connected_groups(keys_per_item)
    if len(groups) < 2:
        if allow_train_only:
            return {"train": list(range(len(catalog))), "val": []}
        raise ValueError("train/val needs at least two independent dry source groups; "
                         "supply separate recordings or set val_fraction=0")
    order = np.random.default_rng(seed).permutation(list(groups))
    # Stratify by label (ML02): every class with >= 2 independent recording groups gets at
    # least one validation group, so validation can score every such class. A class with a
    # single group always stays in training: it cannot be in both splits without leakage, and
    # a class present only in validation could never be learned. With no class of two groups,
    # validation is empty and generate_dataset warns that no independent recording exists.
    by_label: dict[frozenset, list] = {}
    for key in order:
        by_label.setdefault(frozenset(catalog[i]["label"] for i in groups[key]), []).append(key)
    chosen = {key for keys in by_label.values() if len(keys) >= 2
              for key in keys[:min(len(keys)-1, max(1, round(len(keys)*val_fraction)))]}
    return {"val": [i for key in order if key in chosen for i in groups[key]],
            "train": [i for key in order if key not in chosen for i in groups[key]]}


def _dataset_examples(pairs, split, children, source_pools, all_sources):
    """Stream selections from the shared split; never cross source/position pools."""
    for pair, subset, child in zip(pairs, split, children):
        pool = source_pools[subset]
        if all_sources:
            for index, source_child in zip(pool, child.spawn(len(pool))):
                yield pair, subset, index, tuple(map(int, source_child.generate_state(4)))
        else:
            seeds = tuple(map(int, child.generate_state(4)))
            index = int(np.random.default_rng(seeds[0]).choice(pool))
            yield pair, subset, index, seeds


def environment_fingerprint() -> dict:
    # Source identity of this package only; never the caller's repository (cr-room-capture-04).
    revision = source_revision()
    return {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pyroomacoustics", "pydantic")},
        "package_version": revision["package_version"],
        "git_hash": revision["commit"], "git_dirty": revision["dirty"]}


def generate_dataset(model: RoomModel | "RoomEnsemble", sampling: SamplingSpec, clips: Sequence[DryClip],
                     augmentation: AugmentationConfig, count: int, seed: int, output_dir: str | Path,
                     *, fs: int = 32000, synth_config: SynthConfig | None = None,
                     target_positions: Sequence[PositionPair] | None = None,
                     val_fraction: float = .2, peak_policy: Literal["error", "scale"] = "error",
                     all_sources: bool = False, allow_train_only: bool = False,
                     split_policy: Literal["auto", "legacy-both", "source-region"] = "auto",
                     source_region_size_m: float = .25, stress: bool = False,
                     ) -> Path:
    """WAV PCM16 + manifest.jsonl. Existing nonempty output dirs are never overwritten.

    Real-sound mode: supply DryClip(array, fs) or a WAV and explicit target_positions.
    The caller owns dry extraction/quality; no implicit inverse-room processing here.
    Ensemble mode uses surveyed class patches and a fixed mic (within tolerance).
    Explicit target_positions use source-region splits by default. Legacy sampled
    scenes retain both-role splits. All modes keep decoded content/source_group
    disjoint; stress=True writes only a separate stress split (val_fraction=0).
    all_sources renders every eligible clip at each position in its split; otherwise
    draw one eligible clip per position. count always counts position pairs.
    For one source group, allow_train_only explicitly disables validation and records
    a warning; the default still requires val_fraction=0 or independent recordings.
    Augmentations are not independent validation examples. Class balance and distinct
    rooms are not guaranteed.
    Source paths inside the output's parent are relative to the dataset directory;
    external source paths stay absolute. This keeps generated sibling dry files portable.
    Completion is manifest-last. A failed generation retains a labelled partial run.
    """
    if not clips or not isinstance(count, int) or count < 1:
        raise ValueError("provide at least one clip and a positive integer count")
    if peak_policy not in ("error", "scale"):
        raise ValueError("peak_policy must be error or scale")
    if not isinstance(fs, int) or not 1000 <= fs <= 192000:
        raise ValueError("fs must be an integer in [1000,192000]")
    from .prior import RoomEnsemble
    ensemble = model if isinstance(model, RoomEnsemble) else None
    if ensemble is not None:
        from .calibration_discrepancy import require_discrepancy_review
        require_discrepancy_review(ensemble)
        from .render_prior import select_render_ensemble
        ensemble = select_render_ensemble(ensemble)
    if ensemble is None and model.provenance.get("method") == "surveyed-room-prior/1":
        raise ValueError("no-sweep rendering requires RoomEnsemble from room_ensemble.json, not its representative room_model.json")
    if ensemble is not None:
        if target_positions is not None or stress:
            raise ValueError("ensemble uses surveyed class sites; explicit positions/stress are separate modes")
        model = ensemble.members[0]
    if stress and val_fraction:
        raise ValueError("stress data is a separate split; set val_fraction=0")
    if split_policy not in ("auto", "legacy-both", "source-region"):
        raise ValueError("unknown position split policy")
    out = Path(output_dir)
    synth_config = synth_config or SynthConfig()
    from sensorsim.licensing import require_commercial_lineage
    require_commercial_lineage(model, synth_config, context="dataset generation")
    children = np.random.SeedSequence(seed).spawn(count+2)
    position_seed, split_seed = (int(s.generate_state(1)[0]) for s in children[:2])
    pairs = ([] if ensemble else list(target_positions) if target_positions is not None
             else sample_positions(model, sampling, count, position_seed))
    if not ensemble and len(pairs) != count:
        raise ValueError("target_positions length must equal count")
    for pair in pairs:
        validate_positions(model, sampling, pair)
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0,1)")
    catalog = []
    for clip in clips:
        audio, info = _read_dataset_source(clip, fs, out)
        reference = SourceReference.from_dict(info["source_reference"])
        catalog.append(info)
        del audio
    labels_by_content = {}
    for info in catalog:
        for key in (info["content_sha256"], info["normalized_content_sha256"]):
            prior = labels_by_content.setdefault(key, info["label"])
            if prior != info["label"]:
                raise ValueError(f"conflicting labels {prior!r} and {info['label']!r} for identical dry clips")
    source_split_seed = int(np.random.SeedSequence([seed, 0x534F5552]).generate_state(1)[0])
    source_pools = _source_pools(catalog, val_fraction, source_split_seed,
                                 allow_train_only=allow_train_only)
    effective_fraction = val_fraction if source_pools["val"] else 0.
    warnings = (["no independent validation recording: validation split disabled; "
                 "record \u22652 separate real takes for a leakage-free validation set"]
                if effective_fraction != val_fraction else [])
    if ensemble:
        nval = min(count-1, max(1, round(count*effective_fraction))) if effective_fraction else 0
        if effective_fraction and count < 2:
            raise ValueError("nonempty validation needs at least two scenes")
        split = ["val"]*nval+["train"]*(count-nval)
        np.random.default_rng(split_seed).shuffle(split)
        # With all_sources, every declared site must get a training example.
        # Too few scenes is an explicit error, never a silently missing door.
        if all_sources:
            for label in {clips[i].label for i in source_pools["train"]}:
                available = split.count("train")*sum(clips[i].label == label for i in source_pools["train"])
                if available < len(ensemble.survey.source_sites.get(label, ())):
                    raise ValueError(f"too few training examples to cover every site for {label!r}; increase count")
        # Validate every class/subset before creating any output artifacts.
        for subset in set(split):
            for index in source_pools[subset]:
                sites = ensemble.survey.source_sites.get(clips[index].label, ())
                for site_index in range(max(1, len(sites))):
                    _anchored_position(ensemble.survey, model, clips[index].label, subset,
                        effective_fraction, np.random.default_rng(position_seed), sampling, site_index=site_index)
    else:
        policy = ("source-region" if target_positions is not None else "legacy-both") if split_policy == "auto" else split_policy
        split = split_positions(pairs, effective_fraction, seed=split_seed, policy=policy,
                                region_size_m=source_region_size_m)
    split_counts = {name: split.count(name) * (len(source_pools[name]) if all_sources else 1)
                    for name in ("train", "val")}
    # WP6 playback guard: validate every eligible source/position BEFORE writes.
    # No RNG is consumed here, so the generation selection remains unchanged.
    from sensorsim.sources.speaker import validate_source_position
    for pair, subset in zip(pairs, split):
        for index in source_pools[subset]:
            validate_source_position(catalog[index].get('provenance', {}).get('details', {}), pair.source_xyz_m)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty dataset: {out}")
    out.mkdir(parents=True, exist_ok=True)
    env = environment_fingerprint()
    resolved = {"version": "room-dataset/1", "model": model.model_dump(mode="json"),
        "model_hash": model.model_hash, "sampling": sampling.model_dump(mode="json"),
        "augmentation": augmentation.model_dump(mode="json"), "synthesis": synth_config.model_dump(mode="json"),
        "count": count, "seed": seed, "rng_policy": "numpy-SeedSequence-positional-PCG64/1", "fs": fs, "pcm_bits": 16, "peak_policy": peak_policy,
        "val_fraction_requested": val_fraction, "val_fraction_effective": effective_fraction,
        "val_count_actual": split_counts["val"], "split_counts": split_counts,
        "all_sources": all_sources, "allow_train_only": allow_train_only, "warnings": warnings,
        "position_seed": position_seed, "split_seed": split_seed, "environment": env,
        "source_catalog": catalog, "source_split_seed": source_split_seed,
        "source_path_base": "dataset directory for relative paths",
        "source_split_pools": source_pools,
        "split_policy": "disjoint position components and dry content/recording groups",
        "mode": "real-sound-target-positions" if target_positions is not None else "sampled-positions"}
    if ensemble:
        resolved.update(mode="anchored-ensemble", ensemble_hash=ensemble.ensemble_hash,
            ensemble_status=ensemble.status, result_label="model-based",
            split_policy="disjoint source content/groups and class source subregions; fixed mic",
            member_hashes=[m.model_hash for m in ensemble.members], member_weights=list(ensemble.weights),
            model=None, model_hash=None)
    elif policy == "source-region":
        resolved.update(split_policy="disjoint source subregions and dry content/recording groups",
                        source_region_size_m=source_region_size_m)
    if stress:
        resolved.update(mode="stress", split_counts={"stress": split_counts["train"]}, val_count_actual=0)
    # Keep the complete position list once, not in every row's resolved_config.
    run_config = {**resolved, "target_positions": [p.model_dump(mode="json") for p in pairs]}
    atomic_json(out/"run.json", {**run_config, "state": "incomplete"}, sort_keys=True)
    if ensemble:
        atomic_json(out/"ensemble.json", ensemble.model_dump(mode="json"), sort_keys=True)
    partial = out/"manifest.partial.jsonl"
    output_splits = {}
    with partial.open("x") as manifest:
        if ensemble:
            examples = _ensemble_examples(ensemble, count, position_seed, split, children[2:],
                source_pools, all_sources, clips, sampling, effective_fraction)
        else:
            examples = ((pair, subset, index, seeds, model, None, scene, None)
                        for scene, (pair, subset, index, seeds) in enumerate(
                            _dataset_examples(pairs, split, children[2:], source_pools, all_sources)))
        for i, (pair, subset, index, seeds, member, region, scene, member_index) in enumerate(examples):
            clip_seed, src_seed, rir_seed, mic_seed = seeds
            dry, dry_info = _read_dataset_source(clips[index], fs, out)
            if dry_info != catalog[index]:
                raise ValueError("dry source changed after input fingerprinting")
            from sensorsim.sources.speaker import validate_source_position
            validate_source_position(dry_info.get('provenance', {}).get('details', {}), pair.source_xyz_m)
            transformed, source_info = augment_source(dry, fs, augmentation, seed=src_seed)
            rir = synthesize(member, pair.source_xyz_m, pair.receiver_xyz_m, fs,
                             seed=rir_seed, config=synth_config)
            require_commercial_lineage(rir.metadata, dry_info, context="dataset render")
            wet = signal.oaconvolve(transformed, rir.samples, mode="full")
            wet, receiver_info = augment_receiver(wet, fs, augmentation, seed=mic_seed)
            if not np.isfinite(wet).all():
                raise ValueError("nonfinite output cannot be exported to PCM")
            reference = SourceReference.from_dict(dry_info["source_reference"])
            calibration = reference.calibration if reference.units == "Pa" else None
            if reference.units == "Pa" and calibration is None:
                raise ValueError("Pa source needs explicit peak-FS calibration for PCM export")
            export_fs = wet/calibration.pa_per_peak_fs if calibration else wet
            relative_file = None
            if augmentation.training_level.mode != 'preserve' or dry_info['provenance'].get('parent_event_level'):
                # Parent-level consumers need the lossless unnormalized render.
                relative_file = f"clip_{i:06d}_relative.wav"
                # Preserve the unnormalized product losslessly in its original
                # relative/Pa domain, independently of the PCM training export.
                wavfile.write(out/relative_file, fs, wet.astype(np.float64))
            level_seed = int(np.random.SeedSequence([seed, i, 0x4C455645]).generate_state(1)[0])
            export_fs, level_info = training_export_level(export_fs, augmentation.training_level, seed=level_seed)
            peak = float(np.max(np.abs(export_fs)))
            scale = 1.
            if peak > 32767/32768:
                if peak_policy == "error":
                    raise ValueError(f"clip {i} peak {peak:g} would clip; reduce gain or explicitly select peak_policy='scale'")
                scale = .98/peak
            pcm = (encode_pcm16(wet*level_info['amplitude_scale']*scale, calibration, clipping="raise") if calibration else
                   np.rint(export_fs*scale*32768).astype(np.int16))
            if not np.any(pcm):
                raise ValueError(f"clip {clips[index].label!r} (row {i}) encodes to all-zero PCM16; "
                                 "adjust source level or training augmentation")
            filename = f"clip_{i:06d}.wav"
            wavfile.write(out/filename, fs, pcm)
            row = {"version": "room-dataset-item/1", "id": i, "file": filename,
                "label": clips[index].label, "split": "stress" if stress else subset, "fs": fs, "pcm_bits": 16,
                **pair.model_dump(mode="json"), "model_hash": member.model_hash,
                "resolved_config": resolved, "dry_source": dry_info,
                "source_catalog_index": index,
                "selection_seed": clip_seed, "source_augmentation": source_info,
                "receiver_augmentation": receiver_info, "rir": rir.metadata,
                "rir_metrics": rir_summary(rir), "output_samples": len(pcm),
                "pre_export_peak": peak, "export_scale": scale,
                "parent_event_level": dry_info["provenance"].get("parent_event_level"),
                "training_level": {**level_info, 'relative_file': relative_file,
                    'relative_units': reference.units,
                    'relative_sha256': _hash_file(out/relative_file) if relative_file else None,
                    'total_export_scale': level_info['amplitude_scale']*scale,
                    'recovery': 'PCM/32768 divided by total_export_scale; multiply Pa/FS if calibrated',
                    'quantization_error_bound_fs': .5/32768/(level_info['amplitude_scale']*scale)},
                "audio_boundary": {"units": "peak_fs", "statistic": "sample_peak",
                    "source_units": reference.units, "pa_per_peak_fs": calibration.pa_per_peak_fs if calibration else None,
                    "gain_provenance": calibration.gain_provenance if calibration else "relative source amplitude at 1 m",
                    "axes": ["sample"], "start_sample": 0, "latency_samples": rir.metadata["latency_samples"]},
                "output_sha256": _hash_file(out/filename)}
            prior_split = output_splits.setdefault(row["output_sha256"], row["split"])
            if prior_split != row["split"]:
                raise ValueError("identical exported content would cross dataset splits")
            if ensemble:
                row.update(ensemble_member_hash=member.model_hash, ensemble_hash=ensemble.ensemble_hash,
                    ensemble_member_index=member_index, ensemble_weight=ensemble.weights[member_index],
                    source_subregion=region, render_group=f"scene-{scene}:source-{index}",
                    scene_id=f"{ensemble.ensemble_hash}:scene-{scene}",
                    installation_id=hashlib.sha256(_canonical({"ensemble": ensemble.ensemble_hash,
                        "scene": scene, "receiver_xyz_m": list(pair.receiver_xyz_m)}).encode()).hexdigest(),
                    result_label="model-based")
            elif policy == "source-region":
                row["source_subregion"] = {"id": list(map(int, np.floor(np.asarray(pair.source_xyz_m)/source_region_size_m)))}
            manifest.write(_canonical(row)+"\n")
            manifest.flush()
    partial.rename(out/"manifest.jsonl")
    atomic_json(out/"run.json", {**run_config, "state": "complete"}, sort_keys=True)
    return out/"manifest.jsonl"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--room", required=True, type=Path, help="RoomModel JSON")
    parser.add_argument("--clips", required=True, type=Path, help=(
        "JSON list of {path,label,provenance?,source_group?} rows; each needs exactly one of "
        "source_reference (raw mono WAV, e.g. {\"target\":\"source_1m\",\"units\":\"relative\"}) "
        "or dry_report (room.dry extraction report); paths are relative to the list"))
    parser.add_argument("--sampling", type=Path, help="SamplingSpec JSON (defaults: uniform volume)")
    parser.add_argument("--augmentation", type=Path, help="AugmentationConfig JSON")
    parser.add_argument("--synthesis", type=Path, help="SynthConfig JSON")
    parser.add_argument("--positions", type=Path, help="Real-sound mode: JSON list of source_xyz_m/receiver_xyz_m pairs")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs", type=int, default=32000)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--peak-policy", choices=("error", "scale"), default="error")
    parser.add_argument("--all-sources", action="store_true",
                        help="render every recording in the split's pool at each position "
                             "(every class at every position, in proportion to its recordings; "
                             "recommended for training a recognizer)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    def load(path, cls):
        return cls.model_validate_json(path.read_text()) if path else cls()
    entries = json.loads(args.clips.read_text())
    clips = []
    for row in entries:
        if set(row)-{"path", "label", "provenance", "source_group", "source_reference", "dry_report"}:
            raise ValueError("unknown dry-clip manifest keys")
        if "source_reference" in row and "dry_report" in row:
            raise ValueError("choose source_reference or dry_report, not both")
        ref = row.get("source_reference")
        if "dry_report" in row:
            report = json.loads((args.clips.parent / row["dry_report"]).read_text())
            from sensorsim.licensing import require_commercial_lineage
            require_commercial_lineage(report, context="CLI dry report")
            ref = report["source_reference"]
        if ref is None:
            raise ValueError("raw CLI sources need source_reference; extracted files need dry_report")
        clips.append(DryClip(row["label"], args.clips.parent/row["path"],
                             provenance=row.get("provenance"), source_group=row.get("source_group"),
                             source_reference=SourceReference.from_dict(ref)))
    positions = ([PositionPair.model_validate(p) for p in json.loads(args.positions.read_text())]
                 if args.positions else None)
    path = generate_dataset(RoomModel.from_json(args.room.read_text()), load(args.sampling, SamplingSpec),
        clips, load(args.augmentation, AugmentationConfig), args.count, args.seed, args.output,
        fs=args.fs, synth_config=load(args.synthesis, SynthConfig), target_positions=positions,
        val_fraction=args.val_fraction, peak_policy=args.peak_policy, all_sources=args.all_sources)
    print(path)


if __name__ == "__main__":
    main()
