"""Engine contract and same-model T0 truth. Never pass truth into the fitter."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import signal
import yaml

from sensorsim.acoustics.room.ingest import SweepKit
from sensorsim.acoustics.room.session_sim import Impairments, canonical_kit, simulate_session, device_sos
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from .scenario import Scenario


@runtime_checkable
class TruthEngine(Protocol):
    name: str
    tier: str

    def render_rirs(self, scenario: Scenario, pairs: list[tuple[str, str]], fs: int) -> dict[tuple[str, str], np.ndarray]: ...

    def truth_parameters(self, scenario) -> dict: ...


def stable_seed(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:4], "little")


def synthesis_config(scenario: Scenario) -> SynthConfig:
    # Production refuses a too-short requested decay. Render its complete
    # response, then apply the protocol's common observation horizon explicitly.
    return SynthConfig(obstacle_backend="cpu")


class T0Engine:
    name = "room-synth-same-model"
    tier = "T0"

    def render_rirs(self, scenario: Scenario, pairs: list[tuple[str, str]], fs: int) -> dict[tuple[str, str], np.ndarray]:
        model, positions = scenario.room_model(), scenario.all_positions
        results = {}
        for source, mic in pairs:
            result = synthesize(model, positions[source], positions[mic], fs,
                                seed=stable_seed(scenario.seed, f"truth:{source}:{mic}"),
                                config=synthesis_config(scenario))
            h = np.asarray(result.samples[result.metadata["latency_samples"]:], dtype=np.float64)
            n = round(scenario.render.rir_duration_s*fs)
            results[source, mic] = np.pad(h[:n], (0, max(0, n-len(h))))
        return results

    def truth_parameters(self, scenario) -> dict:
        model = scenario.room_model()
        return dict(absorption=model.absorption, t60_s=model.late_t60().tolist(),
                    obstacles={b.name: b.absorption for b in model.obstacles},
                    validity="same-model inverse crime; finite 1.5 s RIRs; no measured accuracy")


def kit_for(scenario):
    kit = canonical_kit()
    if scenario.render.profile == "full":
        return kit
    smoke = scenario.render.profile == "smoke"
    sweep, tail = (.12, .25) if smoke else (.4, 1.6)
    return SweepKit.model_validate({**kit.model_dump(), "fs_hz": 24000, "f2_hz": 11500,
        "sweep_s": sweep, "tail_silence_s": tail, "lead_silence_s": .05,
        "end_silence_s": .05, "first_sweep_start_s": .05, "repeats": 2,
        "period_s": sweep+tail, "fade_in_s": .005, "fade_out_s": .002})


def source_realization(label, fs, seed, *, seconds=.4):
    """Related but independent nonverbal performances, with 20 ms known silence."""
    rng = np.random.default_rng(seed)
    n = round(seconds*fs*rng.uniform(.95, 1.05))
    t = np.arange(n)/fs
    f = 730*2**(rng.uniform(-.4, .4)/12)
    noise = rng.standard_normal(n)
    if label == "transient":
        x = noise*np.minimum(t/.003, 1)*np.exp(-t/.035)
    elif label == "speech_like":
        x = signal.sosfilt(signal.butter(2, [250, min(3500, fs*.4)], fs=fs, btype="bandpass", output="sos"), noise)
        x *= (.5+.5*np.sin(2*np.pi*5*t))**2*np.sin(np.pi*np.arange(n)/n)**2
    elif label == "tonal_machine":
        x = (np.sin(2*np.pi*f*t)+.3*np.sin(2*np.pi*2.7*f*t)) * np.sin(np.pi*np.arange(n)/n)**2
    elif label == "broadband":
        x = noise*np.sin(np.pi*np.arange(n)/n)**2
    else:
        raise ValueError(label)
    x *= .025*10**(rng.uniform(-1, 1)/20)/max(np.sqrt(np.mean(x*x)), 1e-12)
    return np.r_[np.zeros(round(.02*fs)), x]


def _event_support(name, truth, kit):
    if name in truth.dry_sources:
        support = np.flatnonzero(truth.dry_sources[name])
        if not len(support):
            raise ValueError("silent source has no event timing truth")
        return support[0]/kit.fs_hz, (support[-1]+1)/kit.fs_hz, "real_source_arrival_support"
    return (kit.first_sweep_start_s,
            kit.first_sweep_start_s+(kit.repeats-1)*kit.period_s+kit.sweep_s, "sweep_train_arrival")


@dataclass
class AssembledSession:
    folder: Path
    capture: Path
    truth: object
    kit: SweepKit
    impairments: Impairments
    timing: dict
    manifest: dict
    timing_layer: str = "session_sim + session_timing.TakeTiming"


def assemble_session(scenario: Scenario, engine: TruthEngine, root: Path) -> AssembledSession:
    """Render acquisition, then physically deliver only the training files."""
    kit = kit_for(scenario)
    fs = kit.fs_hz
    pairs = [(s, m) for s in scenario.speakers
             for m in (([f"REF_{s}"] if scenario.acquisition.reference else []) + list(scenario.mics))]
    responses = engine.render_rirs(scenario, pairs, fs)
    if set(responses) != set(pairs):
        raise ValueError("truth engine omitted acquisition RIRs")
    for h in responses.values():
        if np.asarray(h).dtype != np.float64:
            raise ValueError("TruthEngine pressure RIR must be float64")
    seconds = .06 if scenario.render.profile == "smoke" else .4
    sources = {f"S1_{m}_real_{label}.wav": source_realization(label, fs,
               stable_seed(scenario.seed, f"performance:{m}:{label}"), seconds=seconds)
               for m in ((["REF"] if scenario.acquisition.reference else []) + list(scenario.mics))
               for label in scenario.real_sounds}
    p = Impairments.clean(fs) if scenario.impairments == "clean" else Impairments()
    changes = dict(clock_drift_ppm=0., latency_s=(0., 0.))  # timing layer owns these
    if scenario.impairments == "poor":
        changes.update(snr_db=20., coordinate_error_m=.08)
    p = Impairments.model_validate({**p.model_dump(), **changes})
    from sensorsim.acoustics.room.session_timing import TakeTiming
    a = scenario.acquisition
    if a.jitter_s[0] != -a.jitter_s[1]:
        raise ValueError("shared timing API requires symmetric trigger jitter")
    timing_config = TakeTiming(lead_s=a.lead_s, trail_s=a.trail_s, clock_ppm=a.clock_ppm,
        start_jitter_s=a.jitter_s[1], stop_jitter_s=0., extraneous_events=len(a.extraneous),
        event_peak=10**(a.event_level_dbfs/20))
    capture = root/"capture"
    truth = simulate_session(capture, scenario.room_model(), scenario.speakers, scenario.mics,
        truth_dir=root/"truth", kit=kit, seed=scenario.seed, impairments=p,
        synth_config=synthesis_config(scenario), external_rirs=responses, real_sources=sources,
        noise_seconds=.1 if scenario.render.profile == "smoke" else 15.,
        include_reference=scenario.acquisition.reference, timing=timing_config)
    from sensorsim.acoustics.room.rir import detect_onset
    timing, timing_layer = {}, "session_sim + session_timing.TakeTiming"
    for name, info in truth.takes.items():
        if name == "noise_floor.wav":
            continue
        d = info["timing"]
        s, token = name.split("_")[:2]
        mic = f"REF_{s}" if token == "REF" else token
        h = signal.sosfilt(device_sos(fs, p, speaker=False), truth.rirs[f"{s}:{mic}"])
        if name.endswith("_sweep.wav"):
            h = signal.sosfilt(device_sos(fs, p, speaker=True), h)
        direct = detect_onset(h, noise_power=0)/fs
        ratio = 1+d["clock_drift_ppm"]*1e-6
        start, end, kind = _event_support(name, truth, kit)
        timing[name] = {**d, "clock_ratio":ratio, "clock_ppm":d["clock_drift_ppm"],
            "support_kind":kind, "start_s":d["lead_samples"]/fs+(start+direct)*ratio,
            "end_s":d["lead_samples"]/fs+(end+direct)*ratio}
    raw = yaml.safe_load(truth.manifest.read_text())
    if raw.get("acquisition") != "sequential_phone":
        raise ValueError("shared acquisition did not declare sequential_phone")
    training = set(scenario.train_pairs)
    train_s = {s for s, _ in training}
    takes = [t for t in raw["takes"] if t["type"] == "noise" or
             (t["speaker"], t["mic"]) in training or
             (scenario.acquisition.reference and t["speaker"] in train_s and t["mic"] == f"REF_{t['speaker']}")]
    raw["takes"] = takes
    raw["positions"] = dict(speakers={k:v for k,v in raw["positions"]["speakers"].items() if k in train_s},
                            mics={k:v for k,v in raw["positions"]["mics"].items() if k in {t["mic"] for t in takes}})
    raw["room"]["surfaces"] = {s: v.material or "unspecified" for s, v in scenario.room.surfaces.items()}
    raw["notes"] = "Sequential mono takes, independent performances; unknown recorder offsets. Physical source case B."
    folder = root/"session"
    folder.mkdir()
    for name in ["sweep_kit.json", *(t["file"] for t in takes)]:
        shutil.copyfile(capture/name, folder/name)
    (folder/"manifest.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return AssembledSession(folder, capture, truth, kit, p, timing, raw, timing_layer)
