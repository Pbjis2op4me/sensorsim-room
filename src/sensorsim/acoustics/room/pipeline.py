"""Record -> measured RIRs -> effective room -> dry takes -> training WAVs.

python -m sensorsim.acoustics.room.pipeline --session SESSION --out runs/room
See USAGE.md. This is an offline, relative-level,
partially validated synthetic workflow, not an anechoic or geometry estimator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.io import wavfile
from scipy.optimize import brentq

from sensorsim.acoustics.room import ingest, metrics, rir
from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.dataset import (
    DryClip, SamplingSpec, environment_fingerprint, generate_dataset, sample_positions,
)
from sensorsim.acoustics.room.dry import DryConfig, extract_dry
from sensorsim.acoustics.room.model import BoxObstacle, OCTAVE_HZ, RoomModel, SURFACES
from sensorsim.acoustics.room.synth import SynthConfig
from sensorsim.core.messages import ModelValidity
from sensorsim.core.geometry import sound_speed
from sensorsim.core.fingerprint import sha256_file
from sensorsim.core.serialization import atomic_json, load_yaml_unique, report_data, to_native
from .source_reference import SourceReference, source_at_one_m
from sensorsim.realdata.audio_io import probe

VALIDITY = ModelValidity(
    name="Measured-session room training pipeline",
    status="partially validated", freq_range_hz=(125., 8000.),
    conditions="Synthetic integration only; stationary linear room/device response, "
    "known shoebox and box geometry, sufficient sweep bandwidth and decay SNR. "
    "Eyring fallback identifies uniform effective absorption, not geometry/materials. "
    "Physical-source estimates retain REF mic coloration; directivity unverified.",
    citation="Eyring 1930 doi:10.1121/1.1915175; parent-project note room-pipeline (not distributed); "
    "room/README.md; room/README_DRY.md; room/README_SYNTH.md",
)


class PipelineConfig(BaseModel):
    """Strict YAML/JSON config; count is shared positions before train/val splitting."""
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    count: int = Field(default=10, strict=True, ge=1)
    seed: int = Field(default=0, strict=True, ge=0)
    case: Literal["A", "B"] = "B"
    # Explicit opt-in: keep historical default waveforms and config identities.
    no_ref_target: Literal["effective_source", "direct_anchor", "pooled_scalar"] = Field(
        default="effective_source", exclude_if=lambda value: value == "effective_source")
    fit_backend: Literal["room.fit.fit_room", "eyring-t30", "legacy-auto"] = "room.fit.fit_room"
    fit_options: dict = Field(default_factory=dict)
    timing_policy: Literal["humid-air-cramer/1", "dry-air-core/1", "fixed-reference/343"] = "humid-air-cramer/1"
    rir_duration_s: float = Field(default=1.5, gt=0.)
    # None = auto: adaptive whenever a sweep kit (known quiet tail) is declared. Evidence:
    # parent-project note ROOM_SWEEP_DETECT (not distributed) observation-window/2 (typical rooms 54/63 -> 63/63 usable T30s, all within 5%).
    adaptive_rir_window: bool | None = None
    minimum_t30_r_squared: float = Field(default=.9, ge=0., le=1.)
    missing_band_absorption: float = Field(default=.2, gt=0., lt=1.)
    temperature_c: float = Field(default=20., ge=-20., le=50.)
    humidity_pct: float = Field(default=50., ge=10., le=100.)
    dry: DryConfig = Field(default_factory=DryConfig)
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    synthesis: SynthConfig = Field(default_factory=SynthConfig)
    val_fraction: float = Field(default=.2, ge=0., lt=1.)
    peak_policy: Literal["error", "scale"] = "error"
    preview_seconds: float = Field(default=3., gt=0., le=30.)
    # Coverage affects session completion, never whether an ambiguous take is used.
    event_min_resolved_per_label: int = Field(default=1, strict=True, ge=0)
    # Detector FFTs are bounded to one repeat; allow minutes of extra lead/trail.
    max_take_sample_values: int = Field(default=32_000_000, strict=True, ge=1, le=32_000_000)

    @model_validator(mode="after")
    def split_possible(self):
        if self.count == 1 and self.val_fraction:
            raise ValueError("count=1 requires val_fraction=0; one position cannot be split")
        if self.fit_backend == "eyring-t30" and self.fit_options:
            raise ValueError("eyring-t30 uses pipeline T30 settings, not fitter options")
        return self


def _timing_speed(config) -> float:
    """Sound speed of the arrival anchor (distance / c): the same policy as the fitted room model."""
    if config.timing_policy == "fixed-reference/343":
        return 343.
    return sound_speed(config.timing_policy, config.temperature_c, config.humidity_pct)


def _model_speed(config) -> dict:
    if config.timing_policy == "fixed-reference/343":
        return {"reference_sound_speed_m_s": 343.}
    return {"sound_speed_model": config.timing_policy}


def _write_json(path, value, *, report=False):
    """Strict identities; unavailable report numbers carry null plus a reason."""
    value = to_native(value)
    if report:
        value = ([report_data(row) for row in value] if isinstance(value, list) else report_data(value))
    atomic_json(path, value, sort_keys=True)


_hash = sha256_file


def _xyz(take, channel):
    positions = np.atleast_2d(take.receiver_xyz_m)
    return tuple(positions[channel if len(positions) > 1 else 0])


def _mic(take, channel):
    return take.mic_ids[channel] if take.mic_ids else f"channel-{channel}"


def _source(take):
    return take.speaker_id, take.source_xyz_m


def _key(take, channel):
    orientation = take.receiver_orientation_xyz
    if orientation is not None:
        rows = np.atleast_2d(orientation)
        orientation = tuple(rows[channel if len(rows) > 1 else 0])
    return _source(take), _mic(take, channel), _xyz(take, channel), orientation


def _load_fitter(backend="room.fit.fit_room"):
    """Resolve the optional backend once; broken installs are not absence."""
    if backend == "eyring-t30":
        return None
    if backend not in ("room.fit.fit_room", "legacy-auto"):
        raise ValueError("unsupported fit backend")
    module_name = "sensorsim.acoustics.room.fit"
    try:
        fitter = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name or backend != "legacy-auto":
            raise
        return None
    fit_room = getattr(fitter, "fit_room", None)
    if not callable(fit_room):
        raise ValueError("room.fit exists but lacks fit_room(*, manifest, rirs, initial_model); API gap")
    return fit_room


def _adaptive(config, manifest) -> bool:
    """Resolve the adaptive-RIR-window policy (None = auto: on when a sweep kit is declared)."""
    if config.adaptive_rir_window is None:
        return manifest.sweep_kit is not None
    return bool(config.adaptive_rir_window)


def _preflight(manifest, config, *, require_fit_positions):
    """Resolve every real channel before writing any output; no arbitrary matches."""
    if manifest.shoebox_dims_m is None:
        raise ValueError("room dimensions are required; the pipeline cannot infer geometry")
    if manifest.fs < 24000:
        raise ValueError("pipeline requires fs >= 24000 for all 125..8000 Hz octave metrics")
    if manifest.sweep_kit and config.rir_duration_s > manifest.sweep_kit.tail_silence_s:
        raise ValueError("rir_duration_s exceeds the sweep kit quiet gap")
    if config.adaptive_rir_window and manifest.sweep_kit is None:
        raise ValueError('adaptive RIR windows require a sweep kit with a known quiet tail')
    adaptive = _adaptive(config, manifest)
    observation_s = (max(config.rir_duration_s, manifest.sweep_kit.tail_silence_s-.05)
                     if adaptive else config.rir_duration_s)
    if observation_s * manifest.fs > min(1_000_000, config.max_take_sample_values):
        raise ValueError("RIR duration exceeds the sample budget")
    sweeps, refs, noise, real = {}, {}, [], []
    channels, receivers = {}, {}
    for index, take in enumerate(manifest.takes):
        info = probe(take.file)
        channels[index] = info["channels"]
        values = int(np.ceil(info["duration_s"] * manifest.fs)) * info["channels"]
        if values > config.max_take_sample_values:
            raise ValueError(f"{take.file.name}: whole-take sample budget exceeded; split the session recording")
        if take.type == "real":
            speed = 2**(config.augmentation.pitch_semitones[0]/12) * config.augmentation.stretch_rate[0]
            if values / speed > config.max_take_sample_values:
                raise ValueError("augmented real take exceeds the sample budget; shorten the recording")
            if not take.label or not take.label.strip():
                raise ValueError(f"real take {index} needs a nonempty label")
        for channel in range(info["channels"]):
            item = (index, channel)
            if take.type == "sweep":
                if take.role == "check":
                    continue
                positions = receivers.setdefault(_source(take), set())
                if not take.is_reference:
                    positions.add(_xyz(take, channel))
                key = _key(take, channel)
                if key in sweeps:
                    raise ValueError(f"ambiguous matching sweeps for {key}; select one measurement in the manifest")
                sweeps[key] = item
                if take.is_reference:
                    if _source(take) in refs:
                        raise ValueError("ambiguous REF sweeps for one source; select one REF")
                    refs[_source(take)] = item
            elif take.type == "noise":
                noise.append(item)
            else:
                real.append(item)
    if not real:
        raise ValueError("session has no real takes to augment")
    if not noise:
        raise ValueError("session needs a noise take (same-gain silence)")
    jobs = []
    for index, channel in real:
        take = manifest.takes[index]
        match = sweeps.get(_key(take, channel))
        if match is None:
            raise ValueError(f"real take {index}/{channel}: missing matching sweep (source, mic, XYZ, orientation)")
        reference = refs.get(_source(take))
        if config.case == "B" and reference is None and manifest.acquisition != "sequential_phone":
            raise ValueError(f"real take {index}: case B needs a matching source REF sweep")
        matched_noise = [n for n in noise if
                         _mic(manifest.takes[n[0]], n[1]) == _mic(take, channel) and
                         _xyz(manifest.takes[n[0]], n[1]) == _xyz(take, channel)]
        if len(matched_noise) == 1:
            floor, proxy = matched_noise[0], False
        elif not matched_noise and len(noise) == 1:
            floor, proxy = noise[0], True
        else:
            raise ValueError(f"real take {index}: missing or ambiguous noise; supply one matching floor")
        # Calibrated sweeps are Pa/input-FS; real/noise must use the same Pa or FS convention.
        required = [match, floor] + ([reference] if config.case == "B" and reference else [])
        if any(bool(manifest.takes[n[0]].calibration) != bool(take.calibration) for n in required):
            raise ValueError("real, matching sweep, REF and noise must share calibrated Pa or uncalibrated FS units")
        jobs.append(dict(take=index, channel=channel, sweep=match, reference=reference,
                         noise=floor, noise_proxy=proxy))
    deficient = [speaker or str(xyz) for (speaker, xyz), positions in receivers.items()
                 if len(positions) < 2]
    if require_fit_positions and deficient:
        raise ValueError("need ≥2 non-REF mic positions per speaker for a validated room fit; "
                         "see room_recording_kit/README.md (insufficient: " + ", ".join(deficient) + ")")
    return channels, jobs


def _metric_row(value, band):
    return {"metric_policy": metrics.metric_policy(band, onset_sample=value.onset_sample,
                    legacy_drr=value.drr_definition == "drr/1"),
            **{name: asdict(getattr(value, name)) for name in ("edt", "t20", "t30")},
            **{name: getattr(value, name) for name in ("c50_db", "c80_db", "d50", "ts_s", "drr_db")},
            "decay_valid": value.decay.valid, "decay_reason": value.decay.reason,
            "noise_power": value.decay.noise_power, "onset_sample": value.onset_sample}


def _measure(manifest, out, config):
    records = {}
    (out / "rirs").mkdir()
    for index, take in enumerate(manifest.takes):
        if take.type != "sweep":
            continue
        try:
            response = rir.extract_take(manifest, index, rir_duration_s=config.rir_duration_s,
                                        sound_speed_m_s=_timing_speed(config),
                                        sound_speed_model=config.timing_policy,
                                        adaptive_window=_adaptive(config, manifest))
        except ValueError as exc:
            if take.role != "check":
                raise
            for channel in range(probe(take.file)["channels"]):
                records[index, channel] = dict(take=index, channel=channel, role="check",
                    mic_id=_mic(take, channel), source_xyz_m=take.source_xyz_m,
                    receiver_xyz_m=_xyz(take, channel), is_reference=False,
                    file=None, bands={}, metadata={}, error=str(exc))
            continue
        for channel in range(response.samples.shape[1]):
            samples = response.samples[:, channel]
            onset = response.metadata.get("direct_samples", [None] * response.samples.shape[1])[channel]
            measured = metrics.analyze_octaves(samples, manifest.fs, onset_sample=onset)
            name = f"rirs/take_{index:04d}_ch_{channel:02d}.npy"
            np.save(out / name, samples, allow_pickle=False)
            bands = {str(k): _metric_row(v, k) for k, v in measured.items()}
            # T30 outside the excitation passband must not drive absorption fitting.
            for band in OCTAVE_HZ:
                spec = take.excitation
                low, high = metrics.exact_center(band) * np.array([10**-.15, 10**.15])
                bands[str(band)]["excitation_supported"] = (
                    spec.type != "ess" or spec.f_start_hz <= low < high <= spec.f_end_hz)
            records[index, channel] = dict(
                take=index, channel=channel, mic_id=_mic(take, channel),
                source_xyz_m=take.source_xyz_m, receiver_xyz_m=_xyz(take, channel),
                is_reference=take.is_reference, file=name, metadata=response.metadata, bands=bands,
                **({"role":"check"} if take.role == "check" else {}))
    return records


def _uniform_model(base, values):
    return RoomModel.model_validate({**base.model_dump(),
        "absorption": {s: tuple(values) for s in SURFACES},
        "obstacles": [{**b.model_dump(), "absorption": tuple(values)} for b in base.obstacles]})


def _fit(manifest, records, initial, out, config, fit_room):
    records = {k:r for k,r in records.items() if manifest.takes[r["take"]].role != "check"}
    bands = {}
    for band in OCTAVE_HZ:
        eligible = [r["bands"][str(band)]["t30"] for r in records.values()
                    if not r["is_reference"] and r["bands"][str(band)]["excitation_supported"] and
                    r["bands"][str(band)]["t30"]["valid"] and
                    r["bands"][str(band)]["t30"]["r_squared"] >= config.minimum_t30_r_squared]
        values = [v["rt60_s"] for v in eligible]
        bands[str(band)] = dict(
            measured_t30_s=float(np.median(values)) if values else None,
            measurement_count=len(values), min_t30_s=min(values) if values else None,
            max_t30_s=max(values) if values else None,
            minimum_r_squared=min(v["r_squared"] for v in eligible) if values else None,
            status="fitted" if values else "unmeasured prior")
    # Early reflections alone cannot establish the usable decay this workflow
    # requires. Keep that admission gate even when an inverse fitter is installed.
    if not any(b["measurement_count"] for b in bands.values()):
        raise ValueError("no usable non-REF per-band T30; record longer/quieter sweeps or increase rir_duration_s")
    if fit_room is not None:
        measured_rirs = [rir.RIR(manifest.fs, np.load(out / r["file"], mmap_mode="r"),
                                r["source_xyz_m"], r["receiver_xyz_m"], r["metadata"])
                         for r in records.values()]
        options = {}
        if config.fit_options:
            from .fit import FitConfig
            fit_options = dict(config.fit_options)
            if "bootstrap_workers" in fit_options:
                options["bootstrap_workers"] = fit_options.pop("bootstrap_workers")
            options["config"] = FitConfig(**fit_options)
        try:
            model = fit_room(manifest=manifest, rirs=measured_rirs, initial_model=initial, **options)
        except ValueError as error:
            if str(error) == "REF has no resolved direct/reflection separation":
                raise ValueError(
                    "REF has no resolved direct/reflection separation: a nearby surface "
                    "(for example the laptop table) overlaps its direct window. Keep the "
                    "clock and reflection gates unchanged. Re-record all affected takes "
                    "with more surface clearance, or explicitly omit REF for the "
                    "effective-source workflow; physical-source coloration then remains unresolved."
                ) from error
            raise
        if not isinstance(model, RoomModel):
            raise TypeError("room.fit.fit_room must return RoomModel; API gap")
        model = RoomModel.model_validate(model.model_dump())
        method, geometry = "room.fit.fit_room", "provided by room.fit; consult its provenance"
        fitted_bands = model.provenance.get("fitted_bands_hz")
        for band, row in bands.items():
            row["status"] = ("consult fitter provenance" if fitted_bands is None else
                             "fitted" if int(band) in fitted_bands else "unfitted prior")
        quality = ("inspect room.provenance.pipeline_fit_report for held-out metrics and gates; "
                   "completion does not establish validation")
    else:
        coefficients = [config.missing_band_absorption] * 7
        for i, band in enumerate(OCTAVE_HZ):
            row = bands[str(band)]
            if row["measured_t30_s"] is None:
                continue
            def residual(alpha):
                values = coefficients.copy()
                values[i] = alpha
                return _uniform_model(initial, values).t60_estimates()[i] - row["measured_t30_s"]
            if residual(0.) < 0 or residual(1.-1e-9) > 0:
                raise ValueError(f"{band} Hz T30 is outside the physical Eyring/air-loss range")
            coefficients[i] = brentq(residual, 0., 1.-1e-9, xtol=1e-12)
        model = _uniform_model(initial, coefficients)
        method, geometry = "eyring-t30-fallback", "supplied, not inferred"
        quality = "calibration residual only; no held-out measurement validation"
        model = model.model_copy(update={
            "provenance": {**model.provenance, "fit_method": method,
                           "geometry_status": geometry, "t30_bands": bands,
                           "coefficients": "same effective absorption on every exposed wall/box; material labels unused"},
            "uncertainty": {"unmeasured_bands_hz": [b for b in OCTAVE_HZ if not bands[str(b)]["measurement_count"]],
                            "missing_band_absorption": config.missing_band_absorption,
                            "scope": "no per-surface/material/geometry identification or measured holdout validation"}})
    predicted = model.late_t60()
    area, absorption = model.diffuse_surfaces()
    for i, band in enumerate(OCTAVE_HZ):
        row = bands[str(band)]
        row["model_t60_s"] = float(predicted[i])
        row["residual_fraction"] = (row["model_t60_s"]/row["measured_t30_s"]-1
                                    if row["measured_t30_s"] is not None else None)
        row["effective_absorption"] = float(area @ absorption[:, i] / area.sum())
    return model, dict(method=method, geometry_status=geometry, bands=bands,
                      quality=quality)


def _direct_anchor_jobs(manifest, records, jobs, out):
    """Choose a common, training-only measured target per source; no EQ fitting.

    Uses the existing Case-B inverse, REF window and geometric distance adapter.
    Direct-window isolation follows dEchorate section 2.3.2 / REW IR windows.
    This does not identify an anechoic response or source angular radiation.
    """
    from .dry import _direct
    anchors, updated = {}, dict(records)
    for job in jobs:
        source = _source(manifest.takes[job['take']])
        if job['reference'] is not None or source in anchors:
            continue
        candidates = []
        for key, record in sorted(records.items()):
            take = manifest.takes[record['take']]
            if record['is_reference'] or getattr(take, 'role', 'fit') == 'check' or _source(take) != source:
                continue
            if bool(take.calibration) != bool(manifest.takes[job['take']].calibration):
                continue
            channel = record['channel']
            meta = record['metadata']
            if (meta.get('direct_path_obstructed', [False]*(channel+1))[channel]
                    or 'direct_samples' not in meta):
                continue
            # Store a mono view of the bound; original measurement metadata is
            # untouched and remains the exact fitter/detector evidence.
            speed = meta['sound_speed_m_s']
            delay = rir.first_reflection_delays(record['source_xyz_m'], record['receiver_xyz_m'],
                manifest.shoebox_dims_m, manifest.obstacles, speed)[0]
            bound = [delay]*(channel+1)
            metadata = {**meta, 'first_reflection_delay_s': bound,
                        'measurement_channel': channel}
            samples = np.load(out/record['file'], mmap_mode='r')
            try:
                direct, start, stop, _, low_hz = _direct(samples, manifest.fs, metadata)
            except ValueError:
                continue
            direct_energy = float(direct@direct)
            if not np.isfinite(direct_energy) or direct_energy <= 0:
                continue
            ratio = direct_energy/max(float((samples-direct)@(samples-direct)), 1e-30)
            candidates.append((ratio, key, metadata, low_hz, start, stop))
        if not candidates:
            raise ValueError('no REF: no unobstructed training sweep with resolved direct/reflection separation')
        # Stable lexical key wins an exact tie; no evaluation values enter here.
        best = max(candidates, key=lambda row: (row[0], tuple(-v for v in row[1])))
        ratio, key, metadata, low_hz, start, stop = best
        anchors[source] = key
        updated[key] = {**records[key], 'metadata': metadata,
                       'direct_anchor': dict(take=key[0], channel=key[1],
                           direct_residual_ratio=ratio, low_frequency_hz=low_hz,
                           start_sample=start, stop_sample=stop,
                           selection='maximum direct/residual energy among delivered same-source sweeps')}
    planned = [{**job, 'reference': anchors[_source(manifest.takes[job['take']])],
                'reference_kind': 'training_direct_anchor'} if job['reference'] is None else job
               for job in jobs]
    return updated, planned


def _level_key(take, channel):
    calibration = ingest.channel_calibration(take, channel)
    return json.dumps([_source(take), asdict(calibration) if calibration else None], sort_keys=True)


def training_level_references(manifest, records, out):
    """Training-only recorder reference; never uses fitted or evaluation audio.

    Caller delivers the training manifest/records. Different source identities,
    positions of a source, or calibration factors are never silently pooled.
    The same moved recorder and stable playback/gain are explicit assumptions;
    ordinary position-named mic IDs cannot establish device repeatability.
    """
    grouped, identities = {}, {}
    for record in sorted(records.values(), key=lambda r: (r['take'],r['channel'])):
        take, channel = manifest.takes[record['take']], record['channel']
        if record['is_reference'] or getattr(take, 'role', 'fit') == 'check':
            continue
        key = _level_key(take, channel)
        source, receiver = record['source_xyz_m'], record['receiver_xyz_m']
        identities[key] = dict(source_id=take.speaker_id, source_xyz_m=list(source),
            calibration=asdict(ingest.channel_calibration(take,channel)) if take.calibration else None)
        meta = record['metadata']
        identity = f"take-{record['take']}:channel-{channel}"
        evidence = dict(identity=identity, position=[*source,*receiver], eligible=False,
                        level_db_1m=None, reason='missing direct sample or band noise proxy')
        bands = record.get('bands', {})
        if 'direct_samples' in meta and all(str(b) in bands for b in metrics.DIRECT_LEVEL_BANDS):
            speed = meta['sound_speed_m_s']
            delay = rir.first_reflection_delays(source, receiver,
                manifest.shoebox_dims_m, manifest.obstacles, speed)[0]
            spec = take.excitation
            passband = (spec.f_start_hz,spec.f_end_hz) if spec.type == 'ess' else (0.,manifest.fs/2)
            if not all(bands[str(b)]['excitation_supported'] for b in metrics.DIRECT_LEVEL_BANDS):
                passband = (0.,0.)
            evidence = metrics.direct_level_observation(np.load(out/record['file'], mmap_mode='r'),
                manifest.fs, distance_m=float(np.linalg.norm(np.subtract(source,receiver))),
                direct_sample=meta['direct_samples'][channel], first_reflection_delay_s=delay,
                identity=identity, position=[*source,*receiver], excitation_band_hz=passband,
                noise_band_power={b:bands[str(b)]['noise_power'] for b in metrics.DIRECT_LEVEL_BANDS},
                obstructed=meta.get('direct_path_obstructed',[False]*(channel+1))[channel])
        evidence.update(take=record['take'],channel=channel,rir_file=record['file'],
            input_sha256=_hash(take.file),
            receiver_orientation_xyz=getattr(take,'receiver_orientation_xyz',None),
            direction_from_source=np.subtract(receiver,source).tolist())
        grouped.setdefault(key,[]).append(evidence)
    return {key:{**metrics.pool_direct_levels(rows), **identities[key]} for key,rows in grouped.items()}


def _check_metric_rows(measured, predicted, *, level_reference, fs=None, unavailable_reason=None):
    """Frozen closed-loop statistical gates, in the measured sweep-chain domain.

    Reuse production ISO/drr/2 metrics and fit's frozen tolerances. Unlike the
    noise-free benchmark, physical captures keep Chu/Lundeby support checks.
    Level uses I3's training-only scalar once; raw level remains diagnostic.
    No gain, spectrum, onset or model parameter is fitted to a check response.
    """
    from .fit import GATES, METRICS
    fs = measured.fs if measured is not None else fs
    if predicted is not None and predicted.fs != fs:
        raise ValueError("check RIR sample rates must agree")
    scalar = (level_reference or {}).get("amplitude_scale")
    if scalar is not None and (not np.isfinite(scalar) or scalar <= 0):
        raise ValueError("check level reference must be finite and positive")
    reference_reason = (level_reference or {}).get("reason") or "no training level reference for this source/calibration"
    rows = []
    for band in (500, 1000, 2000, 4000):
        target = estimate = None
        support_reason = unavailable_reason if measured is None else None
        onset = None if measured is None else measured.metadata.get("direct_sample")
        if measured is not None:
            spec = measured.metadata.get("excitation", {})
            lo, hi = metrics.exact_center(band)*np.array([10**-.15, 10**.15])
            if spec.get("type") == "ess" and not spec["f_start_hz"] <= lo < hi <= spec["f_end_hz"]:
                support_reason = "octave not fully supported by declared excitation"
            if support_reason is None:
                target = metrics.room_metrics(measured.samples, fs, band=band, onset_sample=onset,
                    noise_power=0. if measured.metadata.get("synthetic") else None)
        if predicted is not None:
            estimate = metrics.room_metrics(predicted.samples, fs, band=band,
                onset_sample=predicted.metadata.get("direct_sample"), noise_power=0.)
        def value(metric, name):
            if metric is None:
                return None
            v = getattr(metric, name)
            if name in ("edt", "t20", "t30"):
                v = v.rt60_s if v.valid else None
            return float(v) if v is not None and np.isfinite(v) else None
        def energy_level(response):
            if response is None:
                return None
            y = metrics.octave_filter(response.samples, fs, band)
            energy = float(y @ y)
            return float(10*np.log10(energy)) if energy > 0 else None
        measured_level, predicted_level = energy_level(measured), energy_level(predicted)
        for name in (*METRICS, "band_level_db", "relative_band_level_db"):
            relative = name in ("edt", "t20", "t30")
            level = name in ("band_level_db", "relative_band_level_db")
            a = measured_level if level else value(target, name)
            b = predicted_level if level else value(estimate, name)
            reason = support_reason
            if name == "relative_band_level_db":
                b = b + 20*np.log10(scalar) if b is not None and scalar is not None else None
                if scalar is None:
                    reason = reason or reference_reason
            if support_reason:
                a = None
            if relative and a is not None and a <= 0:
                a = None
            if a is None:
                reason = reason or (getattr(target, name).reason if relative and target else
                                    target.decay.reason if target else "check measurement unavailable")
            elif b is None:
                reason = reason or (getattr(estimate, name).reason if relative and estimate else
                                    estimate.decay.reason if estimate else "check prediction unavailable")
            error = (100*(b/a-1) if relative else b-a) if a is not None and b is not None else None
            tolerance = 3. if level else GATES[name]*(100 if relative else 1)
            status = "UNAVAILABLE" if a is None else "FAIL" if error is None or abs(error) > tolerance else "PASS"
            rows.append(dict(band_hz=band, metric=name, measured=a, predicted=b,
                signed_error=error, unit="%" if relative else "1" if name == "d50" else "dB",
                tolerance=tolerance, status=status, reason=reason,
                primary=name != "band_level_db", metric_policy=metrics.metric_policy(band, onset_sample=onset)))
    return rows


def _check_positions(manifest, records, model, out, config):
    """Evaluate declared checks only after fitting; never feeds inference back."""
    from .synth import synthesize
    selected = [r for r in records.values() if manifest.takes[r["take"]].role == "check"]
    if not selected:
        return None
    references = training_level_references(manifest, records, out)
    directory = out / "checks"
    directory.mkdir()
    positions = []
    for record in selected:
        take, channel = manifest.takes[record["take"]], record["channel"]
        reference = references.get(_level_key(take, channel))
        measured = predicted = None
        error, prediction_file = record.get("error"), None
        seed_key = json.dumps([config.seed, record["source_xyz_m"], record["receiver_xyz_m"]])
        seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], "little")
        if error is None:
            meta = {**record["metadata"], "excitation":take.excitation.model_dump(mode="json")}
            if "direct_samples" in meta:
                meta["direct_sample"] = meta["direct_samples"][channel]
            measured = rir.RIR(manifest.fs, np.load(out / record["file"], mmap_mode="r"),
                record["source_xyz_m"], record["receiver_xyz_m"], meta)
        try:
            # Let the renderer retain its complete decay. The evaluation window
            # below is the acquired duration, not a newly optimized model window.
            rendered = synthesize(model, record["source_xyz_m"], record["receiver_xyz_m"],
                manifest.fs, seed=seed, config=config.synthesis.model_copy(update={"duration_s":None}))
            samples = rendered.samples[rendered.metadata["latency_samples"]:]
            count = len(measured.samples) if measured is not None else round(config.rir_duration_s*manifest.fs)
            samples = np.pad(samples[:count], (0, max(0, count-len(samples))))
            predicted = rir.RIR(manifest.fs, samples, record["source_xyz_m"], record["receiver_xyz_m"],
                {**rendered.metadata, "synthetic":True,
                 "direct_sample":round(rendered.metadata["direct_delay_s"]*manifest.fs)})
            prediction_file = f"checks/take_{record['take']:04d}_ch_{channel:02d}.npy"
            np.save(out / prediction_file, samples, allow_pickle=False)
        except ValueError as exc:
            error = f"prediction unavailable: {exc}"
        rows = _check_metric_rows(measured, predicted, level_reference=reference, fs=manifest.fs,
                                  unavailable_reason=record.get("error"))
        positions.append(dict(take=record["take"], channel=channel, mic_id=record["mic_id"],
            source_xyz_m=record["source_xyz_m"], receiver_xyz_m=record["receiver_xyz_m"],
            measured_file=record["file"], prediction_file=prediction_file, seed=seed,
            acquisition_error=record.get("error"), prediction_error=error if predicted is None else None,
            level_reference=reference, rows=rows))
    primary = [r for p in positions for r in p["rows"] if r["primary"]]
    counts = {f"{status.lower()}_count":sum(r["status"] == status for r in primary)
              for status in ("PASS", "FAIL", "UNAVAILABLE")}
    summary = dict(**counts, planned=len(primary),
        status="FAIL" if counts["fail_count"] else "INCONCLUSIVE" if counts["unavailable_count"] else "PASS")
    return dict(schema_version="room-check-position/1", summary=summary, positions=positions,
        protocol="parent-project bench_room_closedloop protocol (not distributed); statistical gates + I3 scalar reference; drr/2",
        scope="Your held-out recorded positions and fixed laptop/phone chain only; not general physical or recognition validation.",
        level_convention="Recorder-relative sweep transfer; common training-only scalar, no check gain fit or device EQ. Raw band level is diagnostic.",
        timing="Geometry-aligned onset is not an independent arrival/localization accuracy check.")


def _dry(manifest, records, jobs, out, config):
    (out / "dry").mkdir()
    level_references = training_level_references(manifest, records, out)
    _write_json(out/'level_references.json', level_references, report=True)
    if config.case == 'B' and config.no_ref_target == 'direct_anchor':
        records, jobs = _direct_anchor_jobs(manifest, records, jobs, out)
    candidates, selections = {}, {}
    if manifest.acquisition == "sequential_phone":
        from .sweep_detect import describe_events, select_labeled_events
        labels, hints = {}, {}
        for job in jobs:
            index, channel = job['take'], job['channel']
            key = (index,channel)
            take = manifest.takes[index]
            floor_index, floor_channel = job['noise']
            response = np.load(out/records[job['sweep']]['file'],mmap_mode='r')
            # Keep small descriptors only; minutes-long recordings are read one
            # at a time, then re-read for the selected dry conversion below.
            candidates[key] = describe_events(
                ingest.read_take(manifest,index)[:,channel],manifest.fs,
                ingest.read_take(manifest,floor_index)[:,floor_channel],
                response=response,
                pre_roll_s=take.event_pre_roll_s,
                post_roll_s=take.event_post_roll_s if take.event_post_roll_s is not None else len(response)/manifest.fs,
                merge_gap_s=take.event_merge_gap_s)
            labels[key] = (_source(take),take.label)
            hints[key] = dict(event_index=take.event_index,event_window_s=take.event_window_s)
        # Hints (including invalid ones) affect only their own output. Do not
        # remove a peer from AUTO inference merely because its annotation fails.
        selections = select_labeled_events(candidates, labels, hints=hints)
        _write_json(out/'dry/event_selection.json',[
            dict(take=key[0],channel=key[1],file=str(manifest.takes[key[0]].file),
                 **asdict(selections[key]),regions=[asdict(r) for r in c.regions])
            for key,c in candidates.items()], report=True)
        summary = _event_coverage(selections, {key:manifest.takes[key[0]].label for key in selections},
                                  config.event_min_resolved_per_label)
        summary['automatic_with_peer_hints']=0
        summary['automatic_without_hints']=summary['automatic']
        summary['hint_policy']='take-local annotation; automatic decisions computed without hints'
        summary['rejected'] = [dict(take=key[0], channel=key[1],
            file=str(manifest.takes[key[0]].file), reason=s.reason)
            for key,s in selections.items() if s.event_index is None]
        _write_json(out/'dry/event_summary.json', summary)
    rows = []
    for job in jobs:
        index, channel = job["take"], job["channel"]
        if selections and selections[index,channel].event_index is None:
            continue
        take = manifest.takes[index]
        response = records[job["sweep"]]
        ref = records[job["reference"]] if job["reference"] else None
        name = "REF" if job["reference"] == job["sweep"] else "mic"
        responses = {name: np.load(out / response["file"], mmap_mode="r")}
        if config.case == "B" and ref is not None:
            responses["REF"] = np.load(out / ref["file"], mmap_mode="r")
        floor_index, floor_channel = job["noise"]
        sound = ingest.read_take(manifest, index)[:, channel]
        floor = ingest.read_take(manifest, floor_index)[:, floor_channel]
        event = None
        if manifest.acquisition == "sequential_phone":
            selection = selections[index,channel]
            event = candidates[index,channel].regions[selection.event_index]
            sound = sound[event.segment_start:event.segment_stop]
        metadata = {name: {**response['metadata'], 'measurement_channel': response['channel'],
                           "source_id": str(_source(take))}}
        if config.case == 'B' and ref is not None:
            metadata['REF'] = {**ref['metadata'], 'measurement_channel': ref['channel'],
                               'source_id': str(_source(take))}
        if manifest.acquisition == "sequential_phone":
            from .sweep_detect import real_take_metadata
            metadata[name] = real_take_metadata(metadata[name],measured_clock_ppm=take.clock_drift_ppm)
        target_record = ref if config.case == "B" and ref is not None else response
        target_key = "REF" if config.case == "B" and ref is not None else name
        target_take = manifest.takes[target_record["take"]]
        metadata[target_key].update(
            reference_distance_m=float(np.linalg.norm(np.array(target_record["source_xyz_m"])-target_record["receiver_xyz_m"])),
            recording_units="Pa" if target_take.calibration else "relative",
            audio_calibration=ingest.channel_calibration(target_take, target_record["channel"]))
        # Do not annotate an automatically selected direct window as verified.
        result = extract_dry({name: sound}, responses, floor, fs=manifest.fs,
                             case=config.case if ref is not None else "A", reference="REF", metadata=metadata, config=config.dry)
        if event is not None:
            result.report['event_region'] = asdict(event)
            result.report['event_selection'] = asdict(selection)
            result.report['independent_take_clock'] = metadata[name]['real_clock_source']
            if event.tail_truncated:
                result.report['warnings'].append('requested event post-roll reaches file end; reverberant tail may be truncated')
        if config.case == 'B' and ref is None:
            result.source_reference = replace(result.source_reference, target="effective_source")
            result.report["source_reference"] = result.source_reference.as_dict()
            result.report['target'] = 'effective source after matched sweep inversion; speaker/mic coloration inseparable without REF'
            result.report['warnings'].append('optional REF absent: physical-source coloration cannot be separated; no anechoic source claim')
        if job.get('reference_kind') == 'training_direct_anchor':
            result.report['reference_kind'] = job['reference_kind']
            result.report['direct_anchor'] = ref['direct_anchor']
            result.report['target'] = 'same-recorder relative source at the training direct anchor; geometry-distance conversion follows once'
            result.report['warnings'].append('No close REF: training direct-window target retains directional device response and window limits; not an anechoic or Pa calibration.')
        reference_kind = ('measured_REF' if ref is not None else 'relative_only')
        if job.get('reference_kind') == 'training_direct_anchor':
            reference_kind = 'training_direct_anchor'
        if config.case == 'B' and ref is None and config.no_ref_target == 'pooled_scalar':
            pooled = level_references.get(_level_key(take,channel))
            scalar = pooled['amplitude_scale'] if pooled else None
            correction = dict(stage='post_inverse', applied=scalar is not None,
                amplitude_scale=scalar, reference_key=_level_key(take,channel),
                equalisation=False, inverse_cap_changed=False,
                inverse_diagnostics_domain='before post-inverse scalar; original dry report retained',
                reason=pooled['reason'] if pooled else 'no compatible training source reference')
            result.report['scalar_level_correction'] = correction
            if scalar is not None:
                result.samples = result.samples*scalar
                calibration = ingest.channel_calibration(take,channel)
                result.source_reference = replace(result.source_reference,
                    units='Pa' if calibration else 'relative', calibration=calibration)
                result.report['source_reference'] = result.source_reference.as_dict()
                reference_kind = 'training_direct_pool'
                result.report['target'] = 'effective source at 1 m with recorder-referenced scalar; source/device spectral shape unresolved'
            else:
                result.report['warnings'].append('Pooled scalar unavailable: effective source retained without a recorder-level claim.')
        result.report['reference_kind'] = reference_kind
        result.report['level_reference'] = metrics.source_level_metadata(result.source_reference, kind=reference_kind)
        result.report['level_reference']['training_reference_key'] = _level_key(take,channel)
        stem = f"dry/take_{index:04d}_ch_{channel:02d}"
        wavfile.write(out / f"{stem}.wav", manifest.fs, result.samples.astype(np.float32))
        _write_json(out / f"{stem}.json", result.report, report=True)
        # One shared adapter owns the REF-distance conversion for both CLI and API.
        source_audio, source_reference, scale = source_at_one_m(result.samples, result.source_reference)
        source_file = f"{stem}_dataset_source.wav"
        wavfile.write(out / source_file, manifest.fs, source_audio.astype(np.float32))
        rows.append(dict(take=index, channel=channel, mic_id=_mic(take, channel), label=take.label,
                         file=f"{stem}.wav", report=f"{stem}.json", dataset_source=source_file,
                         dataset_source_scale=scale, source_reference=source_reference.as_dict(),
                         noise_proxy=job["noise_proxy"],
                         target=result.report["target"], warnings=result.report["warnings"],
                         untrustworthy_bands_hz=result.report["untrustworthy_bands_hz"],
                         source_sha256=_hash(take.file), sweep_file=response["file"],
                         reference_rir=ref["file"] if ref else None,
                         reference_kind=reference_kind, level_reference=result.report['level_reference']))
        if 'scalar_level_correction' in result.report:
            rows[-1]['scalar_level_correction'] = result.report['scalar_level_correction']
        if job.get('reference_kind') == 'training_direct_anchor':
            rows[-1]['reference_kind'] = job['reference_kind']
            rows[-1]['direct_anchor'] = ref['direct_anchor']
    return rows


def _event_coverage(selections, labels, minimum):
    """Report accepted evidence per label, independently of per-take decisions."""
    counts = {label:0 for label in labels.values()}
    automatic = hinted = 0
    for key, selection in selections.items():
        if selection.event_index is not None:
            counts[labels[key]] += 1
            if selection.method == 'manifest hint':
                hinted += 1
            else:
                automatic += 1
    missing = [label for label,count in counts.items() if count < minimum]
    return dict(automatic=automatic, hinted=hinted,
                unresolved=len(selections)-automatic-hinted,
                resolved_per_label=counts, minimum_per_label=minimum,
                insufficient_labels=missing,
                coverage_satisfied=bool(automatic+hinted) and not missing)


def _dataset(model, dry, out, config, fs):
    directory = out / "dataset"
    positions = sample_positions(model, config.sampling, config.count, config.seed)
    clips = [DryClip(row["label"], out / row["dataset_source"], provenance=row,
                     source_group=f"take-{row['take']:04d}",
                     source_reference=SourceReference.from_dict(row["source_reference"])) for row in dry]
    # Dataset owns content/group unions, both splits, and the single-group fallback.
    # Every channel of a take stays in one source group, even if its dry audio differs.
    manifest = generate_dataset(
        model, config.sampling, clips, config.augmentation, config.count, config.seed,
        directory, fs=fs, synth_config=config.synthesis, target_positions=positions,
        val_fraction=config.val_fraction, peak_policy=config.peak_policy,
        all_sources=True, allow_train_only=True, split_policy="legacy-both")
    run = json.loads(manifest.with_name("run.json").read_text())
    splits = run["split_counts"]
    return dict(manifest="dataset/manifest.jsonl", examples=sum(splits.values()),
                unique_position_pairs=config.count, real_channels=len(dry), split_counts=splits,
                val_fraction_requested=run["val_fraction_requested"],
                val_fraction_effective=run["val_fraction_effective"], warnings=run["warnings"],
                interpretation="train recording groups x train positions; val recording groups x val positions; "
                               "all channels per group, disjoint decoded content and positions; "
                               "no independent-room or recognition-accuracy validation")


def _previews(manifest, out, config):
    directory = out / "previews"
    directory.mkdir()
    with (out / "dataset/manifest.jsonl").open() as stream:
        first = json.loads(next(stream))
    row = first["dry_source"]["provenance"]
    take = manifest.takes[row["take"]]
    original = ingest.read_take(manifest, row["take"])[:, row["channel"]]
    _, estimate = wavfile.read(out / row["file"], mmap=True)
    _, augmented = wavfile.read(out / "dataset" / first["file"], mmap=True)
    previews = []
    for label, audio, source in (("recorded", original, str(take.file)),
                                 ("dry", estimate, row["file"]),
                                 ("augmented", augmented, "dataset/"+first["file"])):
        clip = audio[:round(config.preview_seconds * manifest.fs)].astype(float)
        if audio.dtype.kind == "i":
            clip /= 2**(8*audio.dtype.itemsize-1)
        gain = min(1., .9/max(float(np.max(np.abs(clip))), 1e-30))
        path = f"previews/{label}.wav"
        wavfile.write(out / path, manifest.fs, (clip*gain).astype(np.float32))
        previews.append(dict(file=path, source=source, playback_scale=gain,
                             seconds=len(clip)/manifest.fs))
    return previews


def _modal_region_lines(modal):
    if modal is None:
        return []
    bands = modal["bands_below_hz"]
    below = ("no octave band lies below it" if not bands else
             f"the {', '.join(map(str, bands))} Hz octave band{'s lie' if len(bands) > 1 else ' lies'} below it, in the "
             "modal region, where single room modes dominate and band decay values are approximations")
    return [f"Schroeder frequency: {modal['schroeder_frequency_hz']} Hz; {below}."]


def _report_markdown(report):
    model, fit, dataset = report["room"], report["fit"], report["dataset"]
    lines = [f"# Room session: {report['room_id']}", "",
             f"Status: {report['state']}; partially validated synthetic workflow. Physical accuracy remains unvalidated.", "",
             f"Room dimensions (m): {model['shoebox_dims_m']}; boxes: {len(model['obstacles'])}.",
             *_modal_region_lines(report.get("modal_region")),
             f"Geometry: **{fit['geometry_status']}**. Fit: `{fit['method']}`.",
             ("Uniform effective absorption is shared by walls and boxes; material names are descriptive."
              if fit["method"] == "eyring-t30-fallback" else
              "The installed fitter supplies the model; consult room_model.json provenance and uncertainty."),
             fit["quality"] + ".",
             "T30 summaries combine measured positions; these table residuals are not held-out accuracy.", "",
             "| Hz | Measured T30 (s, extrapolated T60) | Model T60 (s) | Effective absorption | Relative residual | Status |",
             "|---|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "unavailable" if value is None else f"{value:.4g}"
    # Insert opt-in results before the calibration table; preserve legacy bytes.
    checks = report.get("check_positions")
    if checks:
        summary = checks["summary"]
        details = ["## Your held-out accuracy check", "",
            f"**{summary['status']}**: {summary['pass_count']} PASS / {summary['fail_count']} FAIL / "
            f"{summary['unavailable_count']} UNAVAILABLE of {summary['planned']} planned statistical checks.",
            "These positions were excluded from fitting, REF, direct-anchor selection and level pooling.",
            checks["scope"], checks["level_convention"], checks["timing"], "",
            "Decay error is percent; clarity/DRR/level error is dB. A missing prediction fails an evaluable measurement.",
            "| Take/channel | Hz | Metric | Measured | Predicted | Signed error | Limit | Result / reason |",
            "|---|---:|---|---:|---:|---:|---:|---|"]
        for position in checks["positions"]:
            for row in position["rows"]:
                if not row["primary"]:
                    continue
                reason = str(row["reason"] or "").replace("|", "\\|").replace("\n", " ")
                details.append(f"| {position['take']}/{position['channel']} | {row['band_hz']} | {row['metric']} | "
                    f"{fmt(row['measured'])} | {fmt(row['predicted'])} | {fmt(row['signed_error'])} | "
                    f"{row['tolerance']:g} {row['unit']} | {row['status']} {reason} |")
            if position["prediction_error"]:
                details.append(f"Prediction: {position['prediction_error']}")
        details += ["", "## Calibration positions", ""]
        lines[4:4] = details
        lines[2] = (f"Status: {report['state']}; computation completion is separate from check-position accuracy. "
                    "General physical and recognition accuracy remain unvalidated.")
    qa = report.get("capture_qa")
    if qa:
        details = ["## Capture QA", "", f"**Capture QA: {qa['status']}**.",
            "Pilot repeatability is separate from held-out accuracy and does not change pipeline completion.",
            "Per-take, repeat and band support/reasons are in report.json → capture_qa.", ""]
        if qa.get("reason"):
            details.insert(-1, qa["reason"])
        for take in qa.get("takes", []):
            details.insert(-1, f"- {take['id']}: {take['qa']['status']} — {take['qa'].get('reason') or 'see per-band diagnostics'}")
        lines[4:4] = details
    for band, row in fit["bands"].items():
        lines.append(f"| {band} | {fmt(row['measured_t30_s'])} | {fmt(row['model_t60_s'])} | "
                     f"{fmt(row['effective_absorption'])} | {fmt(row['residual_fraction'])} | {row['status']} |")
    lines += ["", "## Dry reliability", "",
              "| Take/channel | Label | Untrustworthy bands (Hz) | Noise proxy | Details |",
              "|---|---|---|---|---|"]
    for row in report["dry"]:
        label = row["label"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['take']}/{row['channel']} | {label} | {row['untrustworthy_bands_hz']} | "
                     f"{row['noise_proxy']} | [{row['file']}]({row['file']}), [diagnostics]({row['report']}) |")
    if report.get('event_selection'):
        events = report['event_selection']
        lines += ["", f"Event selection: {events['automatic']} automatic, {events['hinted']} hinted, "
                  f"{events['unresolved']} rejected. Required per-label coverage: {events['coverage_satisfied']}.",
                  "[Per-take decisions and reasons](dry/event_selection.json)."]
    lines += ["", "Separate performances are processed independently. Automatically selected REF windows remain unverified.",
              "Case B retains REF mic coloration and source-directivity uncertainty. The dataset source is scaled by REF distance",
              "to remove the geometric reference gain once; this does not establish absolute Pa or correct near-field effects.",
              "Case A targets played digital audio; synthesized clips do not reapply speaker coloration.", "",
              "## Dataset", "",
              f"{dataset['examples']} WAVs from {dataset['unique_position_pairs']} new position pairs split across recording groups.",
              f"Split counts: {dataset['split_counts']}. " +
              (f"[Manifest]({dataset['manifest']})." if dataset['manifest'] else "No dataset manifest: no resolved events."),
              f"Validation fraction requested: {dataset['val_fraction_requested']}; effective: {dataset['val_fraction_effective']}.",
              *[f"**WARNING: {warning}**" for warning in dataset["warnings"]],
              dataset["interpretation"] + ".", "",
              "## Previews", "", "Playback-only attenuation is recorded; dry and dataset files retain their documented scales.", ""]
    lines += [f"- [{p['file']}]({p['file']}), playback scale {p['playback_scale']:.4g}" for p in report["previews"]]
    lines += ["", "## Limits", ""] + [f"- {warning}" for warning in report["warnings"]]
    lines += ["", "Full metrics (EDT/T20/T30, clarity, D50, DRR), fit sample counts, input hashes,",
              "resolved configuration, git/environment and all reliability flags: [report.json](report.json).", ""]
    return "\n".join(lines)


def no_sweep_input_fingerprint(room_yaml: str | Path) -> dict:
    """All no-sweep dependencies, including explicitly located impulse evidence."""
    room_yaml = Path(room_yaml).resolve()
    root = room_yaml.parent
    paths = [room_yaml, *sorted((root/"played").glob("*.wav")),
             *sorted((root/"room_recordings").glob("*.wav"))]
    config = root/"calibration.yaml"
    if config.exists():
        paths.append(config)
        raw = load_yaml_unique(config.read_text())
        for event in raw.get("impulses", []):
            path = (root/event["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("impulse file must be inside the room input folder")
            paths.append(path)
    return {str(p.relative_to(root)): _hash(p) for p in paths}


def run_no_sweep(room_yaml: str | Path, output: str | Path, *, fs: int = 16000,
                 clips=None, count: int = 10, seed: int = 4108, val_fraction: float = .2,
                 synth_config: SynthConfig | None = None,
                 evidence_label: Literal["measured", "proxy"] | None = None) -> dict:
    """Survey -> prior -> known playback / weak impulse evidence -> anchored renders.

    `played/name.wav` pairs with `room_recordings/name__room.wav`. An optional
    calibration.yaml declares playback offsets in `recording_offsets_s` (by played
    filename). Only `recording_offsets_hardware_anchored: {name.wav: true}`
    declares that an offset includes all DAC/ADC/buffer latency and leaves acoustic
    propagation time. Otherwise delay is a nuisance, even with a declared offset.
    Qualified impulsive takes go in `impulses`. Weak impulse entries
    require `reviewed: true` after independent review of their excitation/floor
    endpoints and source-tail bound; those quantities are not auto-estimated.
    Direct delay is never aligned to the assumed geometry.
    Every calibration pair must be a development group, separate from real holdout.
    No dry sources are inferred from in-room recordings. Pass DryClip objects from
    independently qualified dry/known-digital inputs to produce a dataset.
    """
    from .prior import (RoomSurvey, build_prior, known_source_observation,
                        impulsive_decay_observation, predict_metrics, update_ensemble)
    from .dataset import read_dry
    room_yaml, out = Path(room_yaml).resolve(), Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {out}")
    survey = RoomSurvey.load(room_yaml)
    ensemble = build_prior(survey)
    root = room_yaml.parent
    settings_file = root/"calibration.yaml"
    settings = load_yaml_unique(settings_file.read_text()) if settings_file.exists() else {}
    if not isinstance(settings, dict) or set(settings)-{
            "recording_offsets_s", "recording_offsets_hardware_anchored", "impulses", "holdout_groups", "evidence_label", "late_block"}:
        raise ValueError("calibration.yaml accepts recording_offsets_s, recording_offsets_hardware_anchored, impulses, holdout_groups, evidence_label, late_block")
    late_block = settings.get("late_block")
    if late_block is not None and (not isinstance(late_block, dict) or
            set(late_block) != {"uncertainty_components", "early_tolerances"}):
        raise ValueError("late_block requires explicit uncertainty_components and early_tolerances")
    offsets = settings.get("recording_offsets_s", {})
    anchors = settings.get("recording_offsets_hardware_anchored", {})
    if not isinstance(offsets, dict) or not isinstance(anchors, dict):
        raise ValueError("recording offsets and hardware anchors must be filename mappings")
    if any(type(v) is not bool for v in anchors.values()):
        raise ValueError("recording_offsets_hardware_anchored values must be booleans")
    if any(v and (k not in offsets or offsets[k] is None) for k, v in anchors.items()):
        raise ValueError("each hardware anchor requires a recording_offsets_s entry")
    if evidence_label is None:
        evidence_label = settings.get("evidence_label", "measured")
    if evidence_label not in ("measured", "proxy"):
        raise ValueError("evidence_label must be measured or proxy")
    groups = {p.stem for p in (root/"played").glob("*.wav") if (root/"room_recordings"/(p.stem+"__room.wav")).exists()}
    groups.update(e["source_group"] for e in settings.get("impulses", []))
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    from .render_prior import calibration_only
    target = len(ranked)//3
    automatic = [group for group in ranked if not calibration_only(group)][:target]
    holdout = set(settings.get("holdout_groups", automatic))
    if "holdout_groups" not in settings and len(automatic) < target:
        raise ValueError(f"automatic holdout needs {target} non-noise playback group(s) but only {len(automatic)} "
                         "exist (broadband noise playback is kept for calibration); add non-noise playback groups "
                         "or declare holdout_groups in calibration.yaml")
    inputs = no_sweep_input_fingerprint(room_yaml)
    if settings_file.exists():
        inputs["calibration.yaml"] = _hash(settings_file)
    def read(path):
        audio, info = read_dry(DryClip("calibration", path), fs)
        inputs[str(path.relative_to(root))] = info["sha256"]
        return audio, info["sha256"]
    predicted = None
    playback = []
    warnings = []
    rejected = {}
    paired = {p.stem+"__room.wav" for p in (root/"played").glob("*.wav")}
    warnings += [f"Unpaired room recording: {p.name}" for p in sorted((root/"room_recordings").glob("*.wav")) if p.name not in paired]
    for played in sorted((root/"played").glob("*.wav")):
        recording = root/"room_recordings"/(played.stem+"__room.wav")
        if not recording.exists():
            warnings.append(f"Unpaired played file: {played.name}")
            continue
        if played.stem in holdout:
            continue
        try:
            x, xhash = read(played)
            y, yhash = read(recording)
            obs = known_source_observation(y, x, fs, source=survey.loudspeaker.xyz, receiver=survey.microphone.xyz,
                source_group=played.stem, input_sha256=hashlib.sha256((xhash+yhash).encode()).hexdigest(),
                recording_offset_s=offsets.get(played.name),
                recording_offset_hardware_anchored=anchors.get(played.name, False))
        except ValueError as exc:
            from .prior import CalibrationObservation
            obs = CalibrationObservation(kind="known-source", source_group=played.stem,
                input_sha256=hashlib.sha256((_hash(played)+_hash(recording)).encode()).hexdigest(),
                source_xyz_m=tuple(survey.loudspeaker.xyz), receiver_xyz_m=tuple(survey.microphone.xyz),
                values={}, uncertainty={}, reasons=(str(exc),))
        obs = obs.model_copy(update={"metadata": {**obs.metadata, "result_label": evidence_label}})
        if not obs.qualified:
            rejected[played.stem] = list(obs.reasons)
        if predicted is None and obs.qualified:
            predicted = predict_metrics(ensemble, survey.loudspeaker.xyz, survey.microphone.xyz, fs, seed=seed)
        if late_block is not None and obs.qualified:
            # Pool first: legacy early-field conflict checks must not feed back
            # into the opt-in late cut update.
            playback.append(obs)
            continue
        candidate = update_ensemble(ensemble, obs, predicted or [{} for _ in ensemble.members])
        if not obs.qualified or candidate.status == "prior-conflict":
            ensemble = candidate
        else:
            playback.append(obs)
    if playback:
        from .calibration_pool import pool_playback
        pooled = pool_playback(playback, uncertainty_components=(
            late_block["uncertainty_components"] if late_block is not None else None))
        ensemble = update_ensemble(ensemble, pooled, predicted,
            modular_uncertainty=pooled.metadata.get("modular_uncertainty"),
            early_tolerances=late_block["early_tolerances"] if late_block is not None else None)
    # Weak event evidence is a fallback only. Censored/unknown-source observations
    # record a rejection and leave the prior weights unchanged.
    if late_block is not None and not playback:
        raise ValueError("late_block requested but no qualified playback; no weak/default fallback")
    if late_block is None and ensemble.status not in ("known-source", "prior-conflict"):
        seen = set()
        for event in settings.get("impulses", []):
            group = event["source_group"]
            if group in holdout:
                continue
            if group in seen:
                raise ValueError("one weak-decay observation per independent source_group")
            seen.add(group)
            if event.get("reviewed") is not True:
                warnings.append(f"Weak-decay group {group} ignored: endpoints and source-tail bound require reviewed: true")
                continue
            path = (root/event["file"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("impulse file must be inside the room input folder")
            try:
                audio, sha = read(path)
            except ValueError as exc:
                warnings.append(f"Impulse evidence {path.name} abstained: {exc}")
                continue
            obs = impulsive_decay_observation(audio, fs, source=event["source_xyz_m"], receiver=survey.microphone.xyz,
                source_group=group, input_sha256=sha, excitation_end_s=event["excitation_end_s"],
                floor_end_s=event["floor_end_s"], source_tail_upper_s=event.get("source_tail_upper_s"),
                censored=event.get("censored", False))
            obs = obs.model_copy(update={"metadata": {**obs.metadata, "result_label": evidence_label}})
            pred = (predict_metrics(ensemble, event["source_xyz_m"], survey.microphone.xyz, fs, seed=seed)
                    if obs.qualified else [{} for _ in ensemble.members])
            ensemble = update_ensemble(ensemble, obs, pred)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run.json", {"state": "incomplete", "inputs": inputs, "method": "no-sweep/1"})
    from .render_prior import select_render_ensemble, conflict_warning, assumed_room_warning
    calibration_ensemble = ensemble
    _write_json(out/"calibration_ensemble.json", calibration_ensemble.model_dump(mode="json"))
    paired_groups = {p.stem for p in (root/"played").glob("*.wav") if (root/"room_recordings"/(p.stem+"__room.wav")).exists()}
    for item in ensemble.evidence:                     # qualified observations the ensemble update did not accept
        if not item.get("accepted") and item.get("reason"):
            why = rejected.setdefault(item.get("observation", {}).get("source_group", "pooled playback"), [])
            why.append(item["reason"]) if item["reason"] not in why else None
    room_warning = assumed_room_warning(paired=paired_groups, held_out=holdout, rejected=rejected)
    if late_block is None and ensemble.status in ("assumed", "weak-decay"):
        warnings.append(room_warning)
    if late_block is None and ensemble.status == "weak-decay":
        # An unknown knock is weak evidence, not room identification. Keep its
        # broad posterior for inspection; render the explicit domestic assumption.
        # Do not interpolate likelihood weights between unrelated member draws.
        from .prior import build_prior
        ensemble = build_prior(ensemble.survey, purpose="domestic-render").model_copy(
            update={"evidence": ensemble.evidence})
    if late_block is None and ensemble.status != "prior-conflict":
        ensemble = select_render_ensemble(ensemble)
    _write_json(out/"room_ensemble.json", ensemble.model_dump(mode="json"))
    # Compatibility point model is explicitly a representative, not a fitted truth.
    representative = ensemble.members[int(np.argmax(ensemble.weights))]
    _write_json(out/"room_model.json", representative.model_dump(mode="json"))
    dataset = {"examples": 0, "manifest": None}
    if ensemble.status == "prior-conflict":
        warnings.append(conflict_warning(ensemble))
    if late_block is not None:
        warnings.append("Modular late calibration is research-only; independently verified P4/P0 training promotion is unavailable.")
    if clips and ensemble.status != "prior-conflict":
        from .calibration_discrepancy import require_discrepancy_review
        require_discrepancy_review(ensemble)
        manifest = generate_dataset(ensemble, SamplingSpec(), clips, AugmentationConfig(), count, seed,
            out/"dataset", fs=fs, val_fraction=val_fraction, all_sources=True, synth_config=synth_config)
        dataset = {"examples": sum(1 for _ in manifest.open()), "manifest": "dataset/manifest.jsonl"}
    report = {"state": "complete", "method": "no-sweep/1", "result_label": "model-based", "fs": fs,
        "calibration_status": calibration_ensemble.status, "ensemble_hash": ensemble.ensemble_hash,
        "calibration_ensemble_file": "calibration_ensemble.json",
        "calibration_ensemble_hash": calibration_ensemble.ensemble_hash,
        "rendering_distribution": ("research-only modular candidate; training blocked" if late_block is not None else
            "blocked" if ensemble.status == "prior-conflict" else
            "domestic-room-assumption/1" if ensemble.status == "assumed" else "calibrated posterior"),
        "holdout_groups": sorted(holdout), "calibration_groups": sorted(groups-holdout),
        "split_policy": ("declared holdout groups, or a seed-hashed third of source groups (broadband noise "
                         "playback is never held out automatically); before calibration"),
        "observation_label": evidence_label,
        "ensemble_members": len(ensemble.members), "ensemble_file": "room_ensemble.json",
        "room_warning": room_warning if ensemble.status in ("assumed", "weak-decay") else None,
        "representative_member_hash": representative.model_hash, "inputs": inputs,
        "evidence": list(ensemble.evidence), "dry": [], "dataset": dataset,
        "warnings": warnings + (["fewer than three source groups: no automatic real holdout; WP4(b) pending"] if len(groups)<3 else [])+
        ["Room ensemble assumptions are not measured material properties.",
            "Known playback observes the speaker-room-mic chain, not the room alone.",
            "No held-out real recognition or false-alarm validation; WP4(b) pending real data.",
            "Open/closed door tags are descriptive; no open-wall propagation model."]}
    # Emit the actionable room warning for both API and CLI callers.
    if ensemble.status == "prior-conflict" or (late_block is None and ensemble.status == "assumed"):
        import warnings as runtime_warnings
        runtime_warnings.warn(room_warning if ensemble.status == "assumed" else
                              conflict_warning(ensemble), RuntimeWarning, stacklevel=2)
    _write_json(out/"report.json", report)
    (out/"report.md").write_text("# No-sweep room ensemble (model-based)\n\n"+
        f"Calibration route: {ensemble.status}; {len(ensemble.members)} members.\n\n"+
        "\n".join("- "+s for s in report["warnings"])+"\n")
    _write_json(out/"run.json", {"state": "complete", "inputs": inputs, "method": "no-sweep/1"})
    return report


def run_pipeline(session: str | Path, output: str | Path,
                 config: PipelineConfig | None = None) -> dict:
    """Process sequential takes; report complete or partial per-label coverage.

    The selected room.fit module must expose
    fit_room(*, manifest: RecordingManifest, rirs: list[RIR], initial_model: RoomModel)
    -> RoomModel. Eyring fallback is explicit; fitter failures propagate.
    legacy-auto is retained only for explicit reproduction of old dispatch.
    """
    config = PipelineConfig.model_validate((config or PipelineConfig()).model_dump())
    session, out = Path(session).resolve(), Path(output).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output: {out}")
    manifest_path = session / "manifest.yaml"
    manifest = ingest.load_manifest(manifest_path)
    fit_room = _load_fitter(config.fit_backend)
    _, jobs = _preflight(manifest, config, require_fit_positions=fit_room is not None)
    fit_backend = dict(selected=config.fit_backend, options=config.fit_options, method="room.fit.fit_room" if fit_room is not None else "eyring-t30-fallback",
                       fallback_reason=None if fit_room is not None else
                       "explicit eyring-t30 selection" if config.fit_backend == "eyring-t30" else
                       "sensorsim.acoustics.room.fit is not importable (legacy-auto)")
    initial = RoomModel(
        room_id=manifest.room_id, shoebox_dims_m=manifest.shoebox_dims_m,
        obstacles=tuple(BoxObstacle(**b.model_dump()) for b in manifest.obstacles),
        temperature_c=config.temperature_c, humidity_pct=config.humidity_pct, **_model_speed(config),
        provenance={**({"lineage": manifest.lineage} if manifest.lineage else {}),
                    "geometry": "user manifest", "surface_descriptions": manifest.surfaces,
                    "scattering": "unfitted model defaults: walls 0, obstacles 0.1"})
    paths = {manifest_path, *(t.file for t in manifest.takes)}
    paths.update(t.excitation.path for t in manifest.takes if t.excitation.type == "known_signal")
    # The normalized ingest model retains kit parameters but not its source path.
    raw = load_yaml_unique(manifest_path.read_text())
    if "fs_hz" in raw:
        kit_path = Path(raw["excitation"]["kit"]).expanduser()
        paths.add((kit_path if kit_path.is_absolute() else session / kit_path).resolve())
    qa_report = None
    if manifest.capture_qa is not None:
        from .capture_qa import build_capture_qa_report
        qa_report = build_capture_qa_report(manifest, rir_duration_s=config.rir_duration_s)
        paths.add(manifest.capture_qa)
    inputs = {str(p): _hash(p) for p in sorted(paths)}
    if qa_report is not None:
        inputs.update(qa_report.get("input_sha256", {}))
    environment = environment_fingerprint()
    resolved = dict(config=config.model_dump(mode="json"), manifest=manifest.model_dump(mode="json"),
                    input_sha256=inputs, environment=environment, fit_backend=fit_backend)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "resolved_config.json", resolved)
    state = dict(version="room-pipeline/1", state="incomplete", stage="rir", **resolved)
    _write_json(out / "run.json", state)
    try:
        records = _measure(manifest, out, config)
        _write_json(out / "measurements.json", list(records.values()), report=True)
        state["stage"] = "fit"
        model, fit = _fit(manifest, records, initial, out, config, fit_room)
        (out / "room_model.json").write_text(model.to_json()+"\n")
        state["stage"] = "check_positions"
        checks = _check_positions(manifest, records, model, out, config)
        if checks:
            fit["quality"] = "calibration residuals below; see the separate held-out check-position accuracy result"
        state["stage"] = "dry"
        dry = _dry(manifest, records, jobs, out, config)
        state["stage"] = "dataset"
        event_summary_path = out/'dry/event_summary.json'
        event_summary = json.loads(event_summary_path.read_text()) if event_summary_path.exists() else None
        dataset = _dataset(model, dry, out, config, manifest.fs) if dry else dict(
            manifest=None, examples=0, unique_position_pairs=0, real_channels=0,
            split_counts={}, val_fraction_requested=config.val_fraction,
            val_fraction_effective=0., warnings=['No real events resolved; no training audio produced.'],
            interpretation='No real events resolved; add event hints or record clearer performances')
        state["dataset"] = dataset
        state["warnings"] = dataset["warnings"]
        state["stage"] = "previews"
        previews = _previews(manifest, out, config) if dry else []
        complete = event_summary is None or event_summary['coverage_satisfied']
        event_warnings = []
        if event_summary and event_summary['unresolved']:
            event_warnings.append(f"{event_summary['unresolved']} real takes rejected; only resolved performances were converted. "
                                  "See dry/event_selection.json and add event_window_s/event_index or re-record.")
        if not complete:
            event_warnings.append('Required per-label event coverage is incomplete; available dry and re-simulated outputs are retained.')
        # Detect a session edited during a run before publishing a success marker.
        if any(_hash(path) != digest for path, digest in inputs.items()):
            raise ValueError("session input changed during the run; output is incomplete")
        report = report_data(to_native(dict(
            version="room-pipeline/1", state="complete" if complete else "partial", room_id=manifest.room_id, fs=manifest.fs,
            validity=asdict(VALIDITY), room=model.model_dump(mode="json"), modal_region=model.modal_region(), fit=fit,
            measurements=list(records.values()), dry=dry, dataset=dataset, previews=previews,
            **({"check_positions":checks} if checks is not None else {}),
            **({'event_selection':event_summary} if event_summary is not None else {}),
            **resolved, warnings=[*dataset["warnings"], *event_warnings,
                ("Check-position results apply only to the recorded positions and device chain; recognition accuracy remains unvalidated."
                 if checks else "No measured-room or recognition-accuracy validation; inspect weak bands before training."),
                "Geometry-aligned sweep timing assumes the first supported packet is direct; obstacles can violate this.",
                "Only non-REF, valid, in-band T30 with sufficient regression R-squared drives fallback fitting; missing bands are explicit priors.",
                "Air defaults to 20 C / 50% RH unless configured; scattering, source directivity and spatial coherence are not fitted.",
                "A single noise take is a flagged proxy at other microphones; AGC, changed gains or moving sources break the transfer assumptions.",
                "Augmentation is synthetic: no head shadow or validated source population; pitch and duration changes are coupled.",
                "Undefined/infinite clarity or energy ratios are JSON null; consult each metric's validity/reason.",
            ])))
        if qa_report is not None:
            from .capture_qa import attach_capture_qa
            report = attach_capture_qa(report, qa_report)
        (out / "report.md").write_text(_report_markdown(report))
        _write_json(out / "report.json", report)
        _write_json(out / "run.json", {**state, "stage": "complete", "state": report['state'],
                                      **({'event_selection':event_summary} if event_summary is not None else {})})
        return report
    except Exception as error:
        _write_json(out / "run.json", {**state, "error": f"{type(error).__name__}: {error}"})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--session", type=Path)
    entry.add_argument("--room-yaml", type=Path, help="no-sweep survey; writes a model-based ensemble")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="YAML/JSON; defaults to SESSION/pipeline.yaml if present")
    parser.add_argument("--augmentation", type=Path, help="AugmentationConfig YAML/JSON override")
    parser.add_argument("--count", type=int, help="new position pairs before train/val splitting")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--case", choices=("A", "B"), help="B physical source (default); A same-speaker playback")
    args = parser.parse_args(argv)
    try:
        if args.room_yaml:
            if any(v is not None for v in (args.config, args.augmentation, args.count, args.case)):
                raise ValueError("no-sweep CLI uses room.yaml priors; rendering uses run_no_sweep(clips=...)")
            report = run_no_sweep(args.room_yaml, args.out, seed=args.seed if args.seed is not None else 4108)
            print(f"{args.out / 'report.md'} (model-based ensemble)")
            return 0
        path = args.config or args.session / "pipeline.yaml"
        raw = load_yaml_unique(path.read_text()) if args.config or path.exists() else {}
        if not isinstance(raw, dict):
            raise ValueError("pipeline config must be a mapping")
        for name in ("count", "seed", "case"):
            value = getattr(args, name)
            if value is not None:
                raw[name] = value
        if args.augmentation:
            raw["augmentation"] = load_yaml_unique(args.augmentation.read_text())
        report = run_pipeline(args.session, args.out, PipelineConfig.model_validate(raw))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"room pipeline: {error}\n")
    print(f"{args.out / 'report.md'} ({report['dataset']['examples']} training clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
