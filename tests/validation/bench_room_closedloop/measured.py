"""T2 measured-pressure adapter and bounded public-pipeline campaign.

No room solver or metric is implemented here. See parent-project note ROOM_CLOSEDLOOP_T2 (not distributed) for the
frozen selection, native device/geometry limits and geometry-defined time origin.
Data are read one HDF5/SOFA hyperslab or ZIP member at a time, never as a room tensor.
The dEchorate route uses processed v1 only and refuses unavailable source 8.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gzip
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time
import warnings
import zipfile

import numpy as np
from scipy import signal
from scipy.io import wavfile
import yaml

from sensorsim.acoustics.room import pipeline
from sensorsim.acoustics.room.dry import DryConfig
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.rir import detect_onset
from sensorsim.acoustics.room.session_sim import Impairments, simulate_session
from sensorsim.acoustics.room.session_timing import TakeTiming
from sensorsim.acoustics.room.synth import synthesize
from sensorsim.core.geometry import HUMID_SOUND_SPEED_MODEL, sound_speed
from sensorsim.core.messages import ModelValidity
from tests.validation.bench_room_fit.run import sofa_metadata, verified_sha
from .compare import compare_dry, compare_parameters, compare_rirs, compare_sound, compare_timing, context
from .engines import AssembledSession, _event_support, kit_for, source_realization, stable_seed, synthesis_config
from .runner import (_legacy_timing, file_hash, optional_hash, oracle_dry, public_outcome,
                     window_options, apply_event_hints, begin_attempt, recover_dry_takes)
from .scenario import Acquisition, Render, SOUNDS
from tests.validation.dechorate_v1 import V1Archive, validate_selection, STATES as V1_STATES
from .scorecard import check_baseline, summarize, write_json, write_scorecard

# Measured-room data (dEchorate etc.) is not distributed; download it yourself and
# point SENSORSIM_ROOM_DATA at it, or pass data= explicitly.
DEFAULT_DATA = Path("data/evidence/room_acoustics")
STATES = ("020002", "011111", "000000", "000001", "000010", "000100", "001000", "010000", "011000", "011100", "011110")
DRY_MODES = ("legacy", "measured_residual")
EXCLUSIONS = {
    "meshrir": "Grid-local coordinates do not establish wall translation. Room-fit's training-only echo-inferred geometry is estimated, outside this measured-geometry campaign.",
    "bras_cr2": "Complex seminar room; verified cuboid wall corners and the local-to-wall transform are unavailable. RS1/RS2 are reflector scenes, not substitute room measurements.",
}
CR2_MEMBER = "1 Scene descriptions/CR2 small room (seminar room)/RIRs/CR2_RIRs_Genelec8020c.sofa"
VALIDITY = ModelValidity(
    name="Closed-loop measured RIR acquisition adapter", status="partially validated",
    freq_range_hz=(500., 4000.),
    conditions="Measured LTI loudspeaker/microphone chain; synthetic independent performances, "
    "40 dB added sweep-anchor noise and sequential clock/event layer. Geometry-derived onset "
    "is not emission-time ground truth. No measured close REF in the selected subsets. "
    "MeshRIR/BRAS CR2 wall transform unavailable; no room relocation qualification.",
    citation="parent-project note validation_cards/room_{dechorate,meshrir,bras} (not distributed); parent-project note ROOM_CLOSEDLOOP_T2 (not distributed)",
)


@dataclass(frozen=True)
class PairRecord:
    path: Path
    fs: int
    count: int
    start: tuple[int, ...] = ()
    shape: tuple[int, ...] = ()
    member: str | None = None
    hdf5_state: str | None = None
    source_id: int | None = None
    microphone_id: int | None = None


@dataclass
class MeasuredScenario:
    """Measured coordinates/splits cannot be expressed by a synthetic Cartesian grid.

    Presents the engine/runner Scenario interface without bypassing its synthetic
    Pydantic validator. Native positions and all evaluation IDs remain explicit.
    """
    id: str
    dataset: str
    positions: dict
    records: dict
    heldout_pairs: list
    novel_pairs: list
    dims_m: tuple | None
    metadata: dict
    render: Render = field(default_factory=Render)
    seed: int = 20260930
    impairments: str = "measured_40db"
    acquisition: Acquisition = field(default_factory=lambda: Acquisition(reference=False))
    real_sounds: tuple = SOUNDS
    obstacles: tuple = ()

    def __post_init__(self):
        if set(self.mics) != {"M1", "M2", "M3"} or not 1 <= len(self.speakers) <= 3:
            raise ValueError("measured protocol requires exactly three mics and one to three speakers")
        if self.acquisition.reference:
            raise ValueError("no verified close on-axis REF in the selected measured subsets")
        selected = [*self.train_pairs, *self.heldout_pairs, *self.novel_pairs]
        if len(set(selected)) != len(selected) or set(selected) != set(self.records):
            raise ValueError("every measured pair must belong to exactly one split")
        for s, m in selected:
            a, b = np.asarray(self.positions[s]), np.asarray(self.positions[m])
            if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
                raise ValueError("invalid measured XYZ")
            if np.linalg.norm(a-b) < .05:
                raise ValueError("source/receiver separation below 5 cm")
            if self.dims_m and (np.any(a < 0) or np.any(b < 0) or
                               np.any(a > self.dims_m) or np.any(b > self.dims_m)):
                raise ValueError("surveyed position outside room")

    @property
    def speakers(self):
        return {k: v for k, v in self.positions.items() if k.startswith("S")}

    @property
    def mics(self):
        return {k: v for k, v in self.positions.items() if k.startswith("M")}

    @property
    def all_positions(self):
        return self.positions

    @property
    def train_pairs(self):
        return [(s, m) for s in self.speakers for m in self.mics]

    def model_dump(self, **_):
        return dict(version="room-closedloop-measured/1", id=self.id, dataset=self.dataset,
                    positions=self.positions, train_pairs=self.train_pairs,
                    heldout_pairs=self.heldout_pairs, novel_pairs=self.novel_pairs,
                    dims_m=self.dims_m, metadata=self.metadata,
                    render=self.render.model_dump(mode="json"), seed=self.seed,
                    impairments=self.impairments, acquisition=self.acquisition.model_dump(mode="json"),
                    real_sounds=self.real_sounds)


class MeasuredEngine:
    name = "measured-rir-lti"
    tier = "T2"

    def __init__(self, data=None, scratch=None):
        self.data = Path(data or os.environ.get("SENSORSIM_ROOM_DATA", DEFAULT_DATA))
        self.scratch = Path(scratch or tempfile.gettempdir())
        self.input_hashes = {}
        self.origins = {}

    def verify(self, relative):
        path = self.data/relative
        if relative not in self.input_hashes:
            self.input_hashes[relative] = verified_sha(self.data, path)
        return path

    def sofa_pair(self, record):
        self.scratch.mkdir(parents=True, exist_ok=True)
        # h5dump is the already installed HDF5 reader; a single float64 channel.
        with tempfile.TemporaryDirectory(prefix="rir-", dir=self.scratch) as folder:
            out = Path(folder)/"samples.bin"
            subprocess.run(["h5dump", "-b", "LE", "-d", "Data.IR", "-s",
                            ",".join(map(str, record.start)), "-c", ",".join(map(str, record.shape)),
                            "-o", str(out), str(record.path)], check=True, capture_output=True)
            h = np.fromfile(out, dtype="<f8")
        if h.size != record.count:
            raise ValueError("SOFA hyperslab does not match declared sample count")
        return h

    def native_rir(self, record):
        if record.hdf5_state is not None:
            archive = getattr(self, "_v1_archive", None)
            if archive is None or record.path != archive.path or record.fs != archive.fs or record.count != archive.count:
                raise ValueError("HDF5 record is not bound to this verified v1 archive")
            h = archive.read_pair(record.hdf5_state, record.source_id, record.microphone_id)
        elif record.member is None:
            h = self.sofa_pair(record)
        else:
            with zipfile.ZipFile(record.path) as archive:
                h = np.load(io.BytesIO(archive.read(record.member)), allow_pickle=False)
            if h.shape != (1, record.count):
                raise ValueError("MeshRIR channel/sample shape changed")
            h = h[0]
        if not np.isfinite(h).all() or not np.any(h):
            raise ValueError("missing, silent or nonfinite measured RIR")
        return np.asarray(h, dtype=np.float64)

    def render_rirs(self, scenario, pairs, fs):
        if fs not in (24000, 48000):
            raise ValueError("T2 supports the core 24/48 kHz profiles only")
        if not isinstance(scenario, MeasuredScenario):
            raise ValueError("T2 requires measured positions; synthetic scenario interpolation is forbidden")
        result = {}
        for pair in pairs:
            if pair not in scenario.records:
                raise ValueError(f"unmeasured source/mic pair: {pair}")
            record = scenario.records[pair]
            h = self.native_rir(record)
            if fs != record.fs:
                divisor = int(np.gcd(fs, record.fs))
                h = signal.resample_poly(h, fs//divisor, record.fs//divisor)
            # Unknown acquisition delay is not independent range truth. Place the onset at
            # distance/c for the measurement's air, the speed the pipeline is configured with.
            onset = detect_onset(h, noise_power=0)
            s, m = pair
            air = measurement_air(scenario)
            speed = sound_speed(HUMID_SOUND_SPEED_MODEL, air["temperature_c"], air["humidity_pct"])
            geometric = round(np.linalg.norm(np.subtract(scenario.positions[s], scenario.positions[m]))/speed*fs)
            shift = geometric-onset
            removed = float(h[:max(0, -shift)] @ h[:max(0, -shift)] / (h @ h))
            h = np.pad(h, (shift, 0)) if shift >= 0 else h[-shift:]
            n = round(scenario.render.rir_duration_s*fs)
            cropped = float(h[n:] @ h[n:] / (h @ h)) if h @ h else 0.
            result[pair] = np.asarray(np.pad(h[:n], (0, max(0, n-len(h)))), dtype=np.float64)
            self.origins[f"{scenario.id}:{s}:{m}"] = dict(native_fs=record.fs,
                raw_onset_at_output_fs=onset, geometry_onset=geometric, shift_samples=shift,
                discarded_prefix_energy_fraction=removed, cropped_tail_energy_fraction=cropped)
        return result

    def truth_parameters(self, scenario):
        # Do not turn the paper's scalar room descriptor or fitted holdout model
        # into per-band measured absorption/T60 truth.
        return dict(validity=asdict(VALIDITY), dataset=scenario.dataset,
                    reference="unavailable: no verified close on-axis measured mic",
                    noise="no separate acquired ambient file; added white noise at 40 dB S1/M1 active-sweep SNR",
                    time_origin="measured onset translated to round(distance/c*fs), c for the logged temperature "
                                "and 50% humidity (humid-air-cramer/1); not independent TOA truth",
                    air=measurement_air(scenario),
                    limitations=scenario.metadata["limits"])

    def dechorate(self, state, profile):
        """Restricted T2 selection; never silently omit unavailable source 8."""
        return self.dechorate_v1(state, profile)

    def dechorate_v1(self, state, profile, *, sources=(7, 8)):
        """Native v1 loader; a source-7-only call is a separate development study.

        Preserve the T2 training microphones 1/11/21. Array-5 evaluation is
        therefore microphones 22-25; mic 21 must not appear on both sides.
        The WP8 split is different (arrays 1-4 fit, all of array 5 validate).
        """
        validate_selection(sources, (1, 3, 5))
        archive = getattr(self, "_v1_archive", None)
        if archive is None:
            archive = self._v1_archive = V1Archive(self.data)
        if state not in V1_STATES:
            raise ValueError("unknown room state")
        self.input_hashes.update(archive.input_hashes)
        g = archive.geometry
        positions, records, heldout, mapping = {}, {}, [], {}
        selected = {1: "M1", 11: "M2", 21: "M3"}
        for source in sources:
            s = f"S{source-6}"
            positions[s] = g['srcs_omn'][:, source-7].tolist()
            for native in (1, 11, 21, 22, 23, 24, 25):
                m = selected.get(native, f"H{native}")
                positions[m] = g['mics'][:, native-1].tolist()
                records[s, m] = PairRecord(archive.path, archive.fs, archive.count,
                    hdf5_state=state, source_id=source, microphone_id=native)
                mapping[f"{s}:{m}"] = dict(source=source, mic=native,
                    file=archive.path.relative_to(self.data).as_posix(),
                    dataset=state, source_column_zero_based=source-1,
                    microphone_column_zero_based=native-1)
                if m.startswith('H'):
                    heldout.append((s, m))
        meta = archive.metadata()
        meta.update(native_mapping=mapping, state=state,
            csv_temperature_c=archive.temperature_c,
            temperature_basis="v1 publisher geometry script, logged room_temperature",
            requested_sources=list(sources), restricted_evaluation_array=5,
            split_note="T2 mics 1/11/21 train; 22-25 heldout; no novel split; differs from WP8")
        return MeasuredScenario("dechorate_"+state, "dEchorate processed v1 HDF5",
            positions, records, heldout, [], tuple(g['room_size']), meta,
            render=render_config(profile))

    def meshrir(self, profile):
        relative = "meshrir/S1-M3969_npy.zip"
        path = self.verify(relative)
        with zipfile.ZipFile(path) as archive:
            xyz = np.load(io.BytesIO(archive.read("S1-M3969_npy/pos_mic.npy")), allow_pickle=False)
            sources = np.load(io.BytesIO(archive.read("S1-M3969_npy/pos_src.npy")), allow_pickle=False)
            details = json.loads(archive.read("S1-M3969_npy/data.json"))
        if xyz.shape != (3969, 3) or sources.shape != (1, 3) or details["samplerate"] != 48000 or details["ir length"] != 32768:
            raise ValueError("MeshRIR acquired schema changed")
        chosen = ((-.5, -.5, -.2), (-.3, 0., 0.), (-.1, .5, .2))
        selected = {}
        for j, value in enumerate(chosen):
            indices = np.flatnonzero(np.all(np.isclose(xyz, value, atol=1e-10, rtol=0), axis=1))
            if len(indices) != 1:
                raise ValueError("MeshRIR frozen training position missing/duplicated")
            selected[int(indices[0])] = f"M{j+1}"
        positions, records, heldout, novel, mapping = {"S1": sources[0].tolist()}, {}, [], [], {}
        for i, point in enumerate(xyz):
            m = selected.get(i, f"N{i}" if point[0] >= .1-1e-9 else f"H{i}")
            positions[m] = point.tolist()
            records["S1", m] = PairRecord(path, 48000, 32768, member=f"S1-M3969_npy/ir_{i}.npy")
            mapping[m] = i
            if m.startswith("H"):
                heldout.append(("S1", m))
            elif m.startswith("N"):
                novel.append(("S1", m))
        meta = dict(native_mapping=mapping,
                    geometry="grid-local coordinates; verified wall-frame transform unavailable; public dimensions omitted",
                    limits=["Single DIATONE DS-7 source; no novel-source evidence or measured on-axis REF.",
                            "Approximate room size does not locate walls in the supplied local grid frame.",
                            "Robot-scanned 5 cm grid, microphone/loudspeaker responses retained, furnishings unknown.",
                            "32768-sample native record; no separate acquired ambient recording."])
        return MeasuredScenario("meshrir", "MeshRIR S1-M3969", positions, records, heldout, novel,
                                None, meta, render=render_config(profile))

    def bras(self, profile):
        relative = "bras/1_scene_descriptions-CR2.zip"
        archive = self.verify(relative)
        self.scratch.mkdir(parents=True, exist_ok=True)
        path = self.scratch/"CR2_RIRs_Genelec8020c.sofa"
        # Copy just one 43 MB member; no 616 MB expansion, no binaural data.
        with zipfile.ZipFile(archive) as z, z.open(CR2_MEMBER) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)
        meta = sofa_metadata(path, ["EmitterPosition", "ReceiverPosition", "SourcePosition", "MeasurementView", "Data.SamplingRate", "Data.Delay"])
        if meta["Data.SamplingRate"].tolist() != [44100.] or np.any(meta["Data.Delay"]) or np.any(meta["SourcePosition"]):
            raise ValueError("unrecognized BRAS origin/rate/delay")
        if not np.array_equal(meta["MeasurementView"].reshape(4, 3)[0], [1, 0, 0]):
            raise ValueError("BRAS positive-X orientation changed")
        positions = {f"S{i+1}": xyz.tolist() for i, xyz in enumerate(meta["EmitterPosition"].reshape(2, 3))}
        receivers = meta["ReceiverPosition"].reshape(5, 3)
        records, heldout, novel = {}, [], []
        for i, xyz in enumerate(receivers):
            m = f"M{i+1}" if i < 3 else "H4" if i == 3 else "N5"
            positions[m] = xyz.tolist()
            for source in range(2):
                pair = (f"S{source+1}", m)
                records[pair] = PairRecord(path, 44100, 154350, (0, i, source, 0), (1, 1, 1, 154350))
                if i == 3:
                    heldout.append(pair)
                elif i == 4:
                    novel.append(pair)
        details = dict(archive=relative, member=CR2_MEMBER, member_sha256=file_hash(path), orientation="positiveX, index 0",
                       native_mapping="S1/S2 = Emitter 0/1; M1-M3/H4/N5 = Receiver 0-4",
                       geometry="complex seminar room; no verified cuboid corners/wall-frame transform; public dimensions omitted",
                       limits=["Directional Genelec 8020c and measured mic response retained; no calibrated close REF.",
                               "Publisher's existing 6 dB CR2 correction retained once; no additional correction.",
                               "Native 3.5 s response cropped to core's 1.5 s horizon; discarded energy recorded.",
                               "Positive-Y missing LS1/MP5 is outside the frozen positive-X orientation.",
                               "BRAS RS1/RS2 are isolated reflectors, not measured rectangular rooms; not substituted for CR2.",
                               "Detailed furnishings/noncuboid geometry and absolute source emission delay are unverified."])
        return MeasuredScenario("bras_cr2", "BRAS CR2 Genelec positiveX", positions, records, heldout, novel,
                                None, details, render=render_config(profile))

    def scenario(self, name, profile="full"):
        if name.startswith("dechorate_") and name[10:] in STATES:
            return self.dechorate(name[10:], profile)
        if name == "meshrir":
            return self.meshrir(profile)
        if name == "bras_cr2":
            return self.bras(profile)
        raise ValueError(f"unknown T2 dataset scenario: {name}")


def measurement_air(scenario):
    """The air the pipeline is told about: the dataset's logged temperature (dEchorate v1: 24 C), and 50%
    relative humidity, which no dataset logs. The geometric time origin uses the same speed."""
    return dict(temperature_c=float(scenario.metadata.get("csv_temperature_c", 20.)), humidity_pct=50.)


def render_config(profile):
    return Render(profile=profile, rir_duration_s=.22 if profile == "smoke" else 1.5)


def make_engine():
    return MeasuredEngine()


def assemble_measured_session(scenario, engine, work):
    """Exactly three sequential takes per source; evaluation never enters capture."""
    kit = kit_for(scenario)
    fs = kit.fs_hz
    p = Impairments.clean(fs, recording_gain=1., snr_db=40.)
    a = scenario.acquisition
    if a.jitter_s[0] != -a.jitter_s[1]:
        raise ValueError("shared timing API requires symmetric trigger jitter")
    timing_config = TakeTiming(lead_s=a.lead_s, trail_s=a.trail_s, clock_ppm=a.clock_ppm,
        start_jitter_s=a.jitter_s[1], stop_jitter_s=0., extraneous_events=len(a.extraneous),
        event_peak=10**(a.event_level_dbfs/20))
    responses = engine.render_rirs(scenario, scenario.train_pairs, fs)
    seconds = .06 if scenario.render.profile == "smoke" else .4
    sources = {f"S1_{m}_real_{label}.wav": source_realization(label, fs,
               stable_seed(scenario.seed, f"performance:{m}:{label}"), seconds=seconds)
               for m in scenario.mics for label in scenario.real_sounds}
    capture = work/"capture"
    # Required by the existing acquisition API only. External measured RIRs
    # bypass room synthesis. Unknown geometry is removed from the delivered
    # manifest; this placeholder is never supplied as truth or fitted geometry.
    carrier = RoomModel(room_id=scenario.id, shoebox_dims_m=scenario.dims_m or (1., 1., 1.),
                        provenance={"scope": "acquisition carrier; external RIRs only"})
    truth = simulate_session(capture, carrier, scenario.speakers, scenario.mics,
        truth_dir=work/"truth", kit=kit, seed=scenario.seed, impairments=p,
        external_rirs=responses, external_rir_provenance={"dataset": scenario.id},
        purpose="internal-validation", real_sources=sources, include_reference=False,
        noise_seconds=.1 if scenario.render.profile == "smoke" else 15., timing=timing_config)
    # Translate shared acquisition truth into the core evaluator's schema.
    # No waveform changes occur here; session_sim owns every offset/clock/event.
    timing = {}
    for name, info in truth.takes.items():
        if name == "noise_floor.wav":
            continue
        d = info["timing"]
        s, m = name.split("_")[:2]
        direct = detect_onset(truth.rirs[f"{s}:{m}"], noise_power=0)/fs
        ratio = 1+d["clock_drift_ppm"]*1e-6
        start, end, kind = _event_support(name, truth, kit)
        timing[name] = {**d, "clock_ratio": ratio, "clock_ppm": d["clock_drift_ppm"],
                       "support_kind": kind, "start_s": d["lead_samples"]/fs+(start+direct)*ratio,
                       "end_s": d["lead_samples"]/fs+(end+direct)*ratio}
    raw = yaml.safe_load(truth.manifest.read_text())
    raw["speaker"] = dict(model=scenario.dataset+" measured source chain", facing="dataset orientation")
    raw["recorder"] = dict(device="measured receiver chain plus PCM16 synthetic acquisition", processing="unprocessed")
    raw["notes"] = "Measured-RIR LTI convolution; independent sequential performances; unknown per-file clock and offset; no REF."
    folder = work/"session"
    folder.mkdir()
    for name in ["sweep_kit.json", *(t["file"] for t in raw["takes"])]:
        shutil.copyfile(capture/name, folder/name)
    delivered = raw
    if scenario.dims_m is None:
        # Public generic manifest supports unknown geometry and arbitrary local
        # coordinates. Never invent a wall translation to make preflight pass.
        delivered = dict(lineage=raw["lineage"], room_id=scenario.id, fs=fs, shoebox_dims_m=None,
            acquisition="sequential_phone",
            recorder=raw["recorder"], speaker=raw["speaker"], sweep_kit=kit.model_dump(mode="json"),
            named_positions=raw["positions"], notes=raw["notes"], takes=[dict(
                file=t["file"], type=t["type"], speaker_id=t.get("speaker"),
                source_xyz_m=scenario.speakers.get(t.get("speaker")),
                receiver_xyz_m=scenario.mics[t["mic"]], mic_ids=[t["mic"]],
                label=t.get("label"), excitation=kit.excitation.model_dump(mode="json")
                if t["type"] == "sweep" else dict(type="unknown")) for t in raw["takes"]])
    (folder/"manifest.yaml").write_text(yaml.safe_dump(delivered, sort_keys=False))
    return AssembledSession(folder, capture, truth, kit, p, timing, raw,
                            "session_sim + session_timing.TakeTiming")


class ScoreSink:
    """Use core serializers/summaries; bound verbose scorecard memory by shards."""
    def __init__(self, folder, limit=4096):
        self.folder, self.limit, self.buffer, self.summary_rows, self.files = folder, limit, [], [], []

    def add(self, rows):
        self.buffer.extend(rows)
        keys = ("tier", "scenario", "eval_kind", "band_hz", "metric", "unit", "pass", "error", "reconstructed")
        self.summary_rows.extend({k: r[k] for k in keys} for r in rows)
        if len(self.buffer) >= self.limit:
            self.flush()

    def flush(self):
        if self.buffer:
            name = f"scorecard-{len(self.files):04d}.jsonl.gz"
            write_scorecard(self.folder/name, self.buffer)
            self.files.append(name)
            self.buffer.clear()


def run_measured_scenario(scenario, engine, work, result_dir, *, dry_mode="legacy", assembled=None,
                          window_policy="auto", event_hints=None):
    """Public pipeline only; streaming evaluation uses core comparison functions."""
    started = time.perf_counter()
    work.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    if dry_mode not in DRY_MODES:
        raise ValueError("unknown public dry mode")
    assembled = assembled or assemble_measured_session(scenario, engine, work)
    apply_event_hints(assembled, event_hints)
    fs = assembled.kit.fs_hz
    config = pipeline.PipelineConfig(count=1, val_fraction=0., seed=scenario.seed, case="B",
        dry=DryConfig(dereverb_mode=dry_mode),
        rir_duration_s=scenario.render.rir_duration_s, synthesis=synthesis_config(scenario), peak_policy="scale", preview_seconds=.1,
        **measurement_air(scenario), **window_options(window_policy))
    output = work/"pipeline"
    attempt = begin_attempt(assembled.folder, output)
    report, model, error = None, None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            report = pipeline.run_pipeline(assembled.folder, output, config)
            model = RoomModel.from_json((output/"room_model.json").read_text())
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}".replace(str(work), "WORK")
    # The public pipeline writes this only after a successful fit. A subsequent
    # dry/event/dataset failure must not erase the completed room-stage result.
    if model is None and (output/"room_model.json").exists():
        model = RoomModel.from_json((output/"room_model.json").read_text())
    sink = ScoreSink(result_dir)
    note = engine.name+f"; dry_mode={dry_mode}; measured source/mic response retained; geometry-defined onset"
    if error:
        note += "; pipeline="+error
    dry_by_take = (recover_dry_takes(attempt, assembled.manifest["takes"], fs) if report is None
                   else {r["take"]: r for r in report["dry"]})
    real = []
    measurements = [] if report is None else report["measurements"]
    if not measurements and (output/"measurements.json").exists():
        measurements = json.loads((output/"measurements.json").read_text())
    by_take = {r["take"]: r for r in measurements}
    for i, take in enumerate(assembled.manifest["takes"]):
        s, m = take.get("speaker", ""), take["mic"]
        if take["type"] == "sweep":
            record = by_take.get(i)
            detected = None if record is None else _legacy_timing(record["metadata"], fs, assembled.kit)
            sink.add(compare_timing(assembled.timing[take["file"]], detected,
                context(scenario, engine, "timing", s, m), notes=note))
        if take["type"] != "real":
            continue
        dry = assembled.truth.dry_sources[take["file"]]
        recovered = None
        if i in dry_by_take:
            rate, recovered = wavfile.read(output/dry_by_take[i]["dataset_source"])
            if rate != fs:
                raise ValueError("pipeline changed dry sample rate")
        oracle = oracle_dry(assembled, take)
        detail = note+"; take="+take["file"]
        sink.add(compare_dry(dry, recovered, oracle, fs, context(scenario, engine, "real_sound", s, m), notes=detail))
        real.append((dry, recovered, detail))
    for kind, pairs in (("heldout", scenario.heldout_pairs), ("novel", scenario.novel_pairs)):
        for number, pair in enumerate(pairs):
            h = engine.render_rirs(scenario, [pair], fs)[pair]
            prediction = None
            s, m = pair
            if model is not None:
                rendered = synthesize(model, scenario.positions[s], scenario.positions[m], fs,
                    seed=stable_seed(scenario.seed, f"reconstruction:{s}:{m}"), config=synthesis_config(scenario))
                values = np.asarray(rendered.samples[rendered.metadata["latency_samples"]:], dtype=np.float64)
                n = round(scenario.render.rir_duration_s*fs)
                prediction = np.pad(values[:n], (0, max(0, n-len(values))))
            sink.add(compare_rirs(h, prediction, fs, context(scenario, engine, kind, s, m), notes=note))
            for dry, recovered, detail in real:
                wet = signal.fftconvolve(dry, h)
                estimate = None if recovered is None or prediction is None else signal.fftconvolve(recovered, prediction)
                sink.add(compare_sound(wet, estimate, fs, context(scenario, engine, "real_sound", s, m),
                    notes=detail+'; claim=sample-realisation-diagnostic; reference=room_only'))
            if (number+1) % 500 == 0:
                print(json.dumps(dict(scenario=scenario.id, split=kind, evaluated=number+1)), flush=True)
    sink.add(compare_parameters(engine.truth_parameters(scenario), model,
                               context(scenario, engine, "parameter", "room", "room"), notes=note))
    sink.flush()
    write_json(result_dir/"summary.json", summarize(sink.summary_rows, profile=scenario.render.profile))
    write_json(result_dir/"scenario.json", scenario.model_dump())
    write_json(result_dir/"index.json", dict(scorecards=sink.files))
    outcome = public_outcome(report, output, len(dry_by_take))
    provenance = dict(version="room-closedloop-measured/2", scenario=scenario.id, tier="T2", engine=engine.name,
        dry_mode=dry_mode, window_policy=window_policy,
        event_variant='manifest_hints' if event_hints else 'auto', event_hints=event_hints or {},
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        engine_source_sha256=file_hash(Path(__file__)), pipeline_source_sha256=file_hash(pipeline.__file__),
        core_sources_sha256={name: file_hash(Path(__file__).with_name(name+".py")) for name in ("engines", "compare", "scorecard", "runner")},
        protocol_sha256=optional_hash(Path(__file__).with_name("PROTOCOL.md")),
        t2_protocol_sha256=optional_hash(Path(__file__).parents[3]/"parent-project note ROOM_CLOSEDLOOP_T2 (not distributed)"),
        environment=dict(python=platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pydantic")}),
        acquisition_layer=assembled.timing_layer, impairments=assembled.impairments.model_dump(mode="json"),
        pipeline_config=config.model_dump(mode="json"), pipeline_error=error,
        fitted_model_available=model is not None, **outcome,
        input_sha256=dict(engine.input_hashes), truth_parameters=engine.truth_parameters(scenario),
        origin_transforms={k: v for k, v in engine.origins.items() if k.startswith(scenario.id+":")},
        timing_truth=assembled.timing, capture_snr_and_clipping=assembled.truth.takes,
        session_sha256={p.name: file_hash(p) for p in sorted(assembled.folder.iterdir())},
        warnings=sorted({str(w.message).replace(str(work), "WORK") for w in caught}),
        reference_ablation=dict(without_ref="executed", with_ref="UNAVAILABLE: no verified measured close on-axis reference"),
        elapsed_s=time.perf_counter()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(result_dir/"provenance.json", provenance)
    if model is not None:
        (result_dir/"room_model.json").write_text(model.to_json()+"\n")
        write_json(result_dir/"pipeline_diagnostics.json", dict(
            fit=report["fit"] if report else model.provenance.get("pipeline_fit_report"),
            dry=list(dry_by_take.values()), **outcome))
    return sink.summary_rows, provenance, [str(result_dir/name) for name in sink.files]


def run_campaign(args):
    started = time.perf_counter()
    hint_path = getattr(args, 'event_hints', None)
    hints = json.loads(hint_path.read_text()) if hint_path else {}
    window_policy = getattr(args, 'window_policy', 'auto')
    names = [n for group in args.scenarios for n in group.split(",")]
    # The parent project's HOMULA engine (measured_homula.py) is not part of this distribution.
    registry = [*("dechorate_"+state for state in STATES), "meshrir", "bras_cr2"]
    if names == ["all"]:
        names = list(registry)
    if len(set(names)) != len(names) or any(n not in registry for n in names):
        raise ValueError("T2 selects dataset IDs: "+", ".join(registry))
    # Fixed furnished and highly reverberant stress states first, then full set.
    names = sorted(names, key=registry.index)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("output must be empty; never overwrite evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    all_rows, runs, cards, metadata = [], [], [], {}
    mode_rows = {mode: [] for mode in DRY_MODES}
    excluded = {name: EXCLUSIONS[name] for name in names if name in EXCLUSIONS}
    included = [name for name in names if name not in excluded]
    with tempfile.TemporaryDirectory(prefix="measured-reader-", dir=args.scratch) as reader_tmp:
        engine = MeasuredEngine(scratch=reader_tmp)
        for name in included:
            scenario = engine.scenario(name, args.profile)
            with tempfile.TemporaryDirectory(prefix=name+"-", dir=args.scratch) as folder:
                work = args.scratch/(args.out.name+"_"+name) if args.keep_audio else Path(folder)/"work"
                assembled = assemble_measured_session(scenario, engine, work)
                for mode in DRY_MODES:
                    relative = name+"/"+mode
                    rows, provenance, files = run_measured_scenario(scenario, engine, work/mode,
                        args.out/relative, dry_mode=mode, assembled=assembled,
                        window_policy=window_policy, event_hints=hints.get(name))
                    all_rows.extend(rows)
                    mode_rows[mode].extend(rows)
                    cards.extend(str(Path(p).relative_to(args.out)) for p in files)
                    runs.append(dict(scenario=name, dry_mode=mode, seconds=provenance["elapsed_s"],
                        peak_rss_mib=provenance["peak_rss_mib"], error=provenance["pipeline_error"],
                        stage=provenance["pipeline_stage"],
                        pipeline_state=provenance['pipeline_state'], pipeline_complete=provenance['pipeline_complete'],
                        session_status=provenance['session_status'], event_selection=provenance['event_selection'],
                        fitted_model_available=provenance["fitted_model_available"],
                        completed_dry_takes=provenance["completed_dry_takes"]))
                    metadata.update(pack_metadata(args.out, [relative]))
                    write_json(args.out/"progress.json", runs)
                    print(json.dumps(runs[-1]), flush=True)
    summary = summarize(all_rows, profile=args.profile)
    write_json(args.out/"summary.json", summary)
    write_json(args.out/"index.json", dict(tier="T2", profile=args.profile, scorecards=cards, runs=runs,
                                          window_policy=window_policy, event_hints=str(hint_path) if hint_path else None,
                                          metadata=metadata, exclusions=excluded))
    # One compact publication artifact; all verbose per-session evidence stays
    # in the requested output directory (normally the job's evidence folder).
    compact = dict(version="room-closedloop-measured/2", tier="T2", profile=args.profile,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        evidence=str(args.out.resolve()), included=included, exclusions=excluded, runs=runs,
        elapsed_s=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        reference_ablation={"without_ref": "executed", "with_ref": "unavailable: no measured close on-axis REF"},
        modes={mode: detailed_summary(rows, args.profile) for mode, rows in mode_rows.items()},
        artifact_sha256={name: file_hash(args.out/name) for name in [*cards, "index.json", "summary.json"]})
    write_json(args.out/"T2_summary.json", compact)
    print(json.dumps(summary["overall"]), flush=True)
    if args.baseline:
        checked = check_baseline(args.out, args.baseline)
        write_json(args.out/"baseline_check.json", checked)
        return 0 if checked["pass_check"] else 1
    return 0


def detailed_summary(rows, profile):
    """Group core score rows; core summarize remains the only aggregator."""
    from .compare import BANDS
    return dict(summary=summarize(rows, profile=profile),
        per_split_band={kind: {str(band): summarize(
            [r for r in rows if r["eval_kind"] == kind and r["band_hz"] == band], profile=profile)
            for band in (None, *BANDS)} for kind in ("heldout", "novel", "real_sound", "timing")})


def pack_metadata(root, names):
    """Compress verbose native coordinates/provenance without dropping evidence.

    Only consumes metadata created by this campaign; scorecard bytes stay exact.
    The index names both compressed JSON resources for each scenario.
    """
    packed = {}
    for name in names:
        files = []
        for kind in ("scenario", "provenance"):
            source = root/name/(kind+".json")
            target = source.with_suffix(".json.gz")
            if target.exists():
                raise FileExistsError(target)
            payload = source.read_bytes()
            target.write_bytes(gzip.compress(payload, mtime=0))
            if gzip.decompress(target.read_bytes()) != payload:
                raise ValueError("metadata compression verification failed")
            source.unlink()  # this invocation's redundant, newly written JSON
            files.append(str(target.relative_to(root)))
        packed[name] = files
    return packed
