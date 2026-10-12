"""Deterministic *synthetic* rehearsal of the phone recording-kit workflow.

This is an acquisition adapter, not another room solver. Physics comes solely
from room.synth/obstacles. Filter/clock/nonlinearity defaults are scenario
assumptions, NOT measured Samsung Galaxy S26 or Bluetooth population parameters.
See parent-project bench_room_kit_e2e protocol (not distributed) and parent-project note ROOM_KIT_E2E (not distributed).
No sound is played or recorded. One take at a time; truth is kept outside delivery.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal
from scipy.io import wavfile
import yaml

from sensorsim.core.messages import ModelValidity
from .ingest import SweepKit
from .model import BoxObstacle, RoomModel, uniform_bands
from .rir import generate_kit_playback
from .synth import SynthConfig, synthesize
from .session_timing import TakeTiming, apply_take_timing
from .devices import DeviceChain, apply_linear, speaker_signal, validate_table

VALIDITY = ModelValidity(
    name="Synthetic phone room recording session", status="unvalidated",
    freq_range_hz=(125., 8000.),
    conditions="Engineering impairment scenarios only, not a measured Galaxy S26, "
    "speaker or Bluetooth codec model. Stationary omnidirectional sources and fixed "
    "room responses; inherits room.synth/obstacles validity. AGC/NS are stress proxies.",
    citation="Android MediaRecorder raw-source documentation; Superpowered Android "
    "audio latency primer; SciPy resample_poly; parent-project note ROOM_KIT_E2E (not distributed)",
)


class Impairments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    speaker_response: bool = True
    harmonic2: float = Field(default=.02, ge=0, le=1)
    harmonic3: float = Field(default=.01, ge=0, le=1)
    mic_response: bool = True
    mic_highpass_hz: float = Field(default=80., gt=0)
    clock_drift_ppm: float = Field(default=100., ge=-1000, le=1000)
    device_fs: int = Field(default=44100, ge=1000, le=192000)
    rate_mode: Literal["resample", "wrong_header", "mislabeled_payload"] = "resample"
    latency_s: tuple[float, float] = (.1, .3)
    snr_db: float | None = 40.  # active clean S1/M1 RMS / fixed stationary noise RMS
    recording_gain: float = Field(default=.1, gt=0, le=100)
    clip_drive: float = Field(default=1., ge=1, le=1000)
    agc: bool = False
    noise_suppression: bool = False
    declare_processing: bool = True
    coordinate_error_m: float = Field(default=.03, ge=0, le=.1)

    @model_validator(mode="after")
    def _bounds(self):
        if not 0 <= self.latency_s[0] <= self.latency_s[1] <= 10:
            raise ValueError("latency bounds must be ordered inside [0,10] seconds")
        return self

    @classmethod
    def clean(cls, fs=48000, **changes):
        return cls(**(dict(speaker_response=False, harmonic2=0, harmonic3=0,
                           mic_response=False, clock_drift_ppm=0, device_fs=fs,
                           latency_s=(0, 0), snr_db=None, coordinate_error_m=0) | changes))


def canonical_kit() -> SweepKit:
    return SweepKit(fs_hz=48000, f1_hz=40, f2_hz=20000, sweep_s=10,
                    fade_in_s=.1, fade_out_s=.01, peak_dbfs=-6,
                    lead_silence_s=2, tail_silence_s=4, end_silence_s=2,
                    repeats=4, period_s=14, first_sweep_start_s=2,
                    formula="x(t)=sin(2*pi*f1*L*(exp(t/L)-1)), L=T/ln(f2/f1); half-Hann fades",
                    reference="Farina 2000 AES preprint 5093; Farina 2007 AES 122nd Conv.",
                    dither="Synthetic recording PCM16 TPDF; per-take stable seed")


def rehearsal_room() -> RoomModel:
    return RoomModel(shoebox_dims_m=(6, 5, 2.8), absorption=uniform_bands(.25),
                     obstacles=(BoxObstacle(name="desk", min_xyz=(2.5, 2, 0),
                                            size_xyz=(1, .8, .9), material="wood"),))


def protocol_positions():
    """Named example in rehearsal_room; caller supplies positions for another room."""
    speakers = {"S1": (1.2, 2.4, .5), "S2": (1.1, 1.1, 1.2), "S3": (4.2, 3.5, 1.7)}
    mics = {"M1": (2.2, 2.4, .5), "M2": (1.2, 3.8, 1.2), "M3": (3.5, 1.2, 1.7),
            "M4": (5.3, 4.2, 1.2), "M5": (4.8, 3.2, 1.7), "M6": (1.1, 1.1, .5),
            "M7": (4.3, 2.4, .5), "M8": (3.9, 2.4, .5)}
    return speakers, mics


def _seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "little")


def device_sos(fs: int, impairments: Impairments, *, speaker: bool) -> np.ndarray:
    """Causal two-pole HP/LP stress response; frequencies are in Hz.

    Engineering assumptions: speaker HP 120 Hz / LP 12 kHz; recorder HP is
    mic_highpass_hz / LP 16 kHz. Clamp HP to 0.1*fs and LP to 0.45*fs so
    reduced-rate rehearsal fixtures remain below Nyquist. These are synthetic
    stress filters, not measured phone/speaker responses or codec models.
    """
    enabled = impairments.speaker_response if speaker else impairments.mic_response
    if not enabled:
        return np.array([[1., 0, 0, 1, 0, 0]])
    hp = 120. if speaker else impairments.mic_highpass_hz
    lp = 12000. if speaker else 16000.
    return np.vstack([signal.butter(2, min(hp, fs*.1), "highpass", fs=fs, output="sos"),
                      signal.butter(2, min(lp, fs*.45), "lowpass", fs=fs, output="sos")])


def _speaker(x, fs, p):
    z = signal.sosfilt(device_sos(fs, p, speaker=True), x)
    if not (p.harmonic2 or p.harmonic3):
        return z
    # Generate polynomial harmonics above file Nyquist before anti-aliasing the
    # acquisition. Fourfold oversampling covers the third-order bandwidth.
    antialias = signal.firwin(257, .25, window=('kaiser', 8.6))
    hi = signal.resample_poly(z, 4, 1, window=antialias)
    hi = hi + p.harmonic2*hi*hi + p.harmonic3*hi*hi*hi
    return signal.resample_poly(hi, 1, 4, window=antialias)[:len(z)]


def _clock(x, fs, p):
    # Sample through a device at its nominal rate plus a fixed oscillator error,
    # then correctly convert nominal 44.1 kHz to file rate unless testing a mistake.
    g = int(np.gcd(fs, p.device_fs))
    z = signal.resample_poly(x, p.device_fs//g, fs//g) if p.device_fs != fs else x
    if p.clock_drift_ppm:
        z = signal.resample(z, round(len(z)*(1+p.clock_drift_ppm*1e-6)))
    if p.rate_mode in ("wrong_header", "mislabeled_payload"):
        return z, p.device_fs if p.rate_mode == "wrong_header" else fs
    if p.device_fs != fs:
        z = signal.resample_poly(z, fs//g, p.device_fs//g)
    return z, fs


def _processing(x, fs, p, noise_rms):
    if not (p.agc or p.noise_suppression):
        return x
    # 10 ms block envelope, attack 50 ms / release 600 ms compressor proxy.
    # Does not claim to reproduce a proprietary Android/speaker algorithm.
    block = max(1, round(.01*fs))
    count = (len(x)+block-1)//block
    padded = np.pad(x, (0, count*block-len(x))).reshape(count, block)
    level = np.sqrt(np.mean(padded*padded, axis=1))
    gains = np.ones(count)
    if p.agc:
        env = 0.
        for i, rms in enumerate(level):
            a = np.exp(-.01/(.05 if rms > env else .6))
            env = a*env + (1-a)*rms
            gains[i] = np.clip(.045/max(env, 1e-6), .25, 6.)
    if p.noise_suppression:
        gains *= np.where(level < max(3*noise_rms, .004), .03, 1.)
    return x * np.interp(np.arange(len(x)), np.arange(count)*block, gains)


def _active_rms(x, kit, latency, p):
    fs = kit.fs_hz
    ratio = (1+p.clock_drift_ppm*1e-6)
    if p.rate_mode != "resample":
        ratio *= p.device_fs/fs
    powers = []
    for i in range(kit.repeats):
        start = (latency + kit.first_sweep_start_s + i*kit.period_s + kit.fade_in_s)*fs*ratio
        stop = start + (kit.sweep_s-kit.fade_in_s-kit.fade_out_s)*fs*ratio
        a = x[round(start):round(stop)]
        powers.append(float(np.mean(a*a)))
    return float(np.sqrt(np.mean(powers)))


def broadband_transient(fs: int, seed: int) -> np.ndarray:
    """Known, independent 0.4 s broadband test burst; no speech content."""
    rng = np.random.default_rng(seed)
    n = round(.4*fs)
    time = np.arange(n)/fs
    return .06*rng.standard_normal(n)*np.minimum(time/.005, 1)*np.exp(-time/.07)


def _git_head() -> str | None:
    """This package's own source commit, or None; absence is not an error.

    The original ran an unguarded `git rev-parse HEAD` in the caller's cwd: it raised
    after session files were already written when run outside a checkout, and inside
    any other repository it recorded that repository's HEAD (cr-room-capture-04).
    """
    from sensorsim.core.fingerprint import source_revision
    return source_revision()["commit"]


@dataclass
class SessionTruth:
    manifest: Path
    evidence: Path
    rirs: dict[str, np.ndarray]
    rir_metadata: dict[str, dict]
    takes: dict[str, dict]
    dry_sources: dict[str, np.ndarray]
    noise_rms: float


def simulate_session(output: str | Path, model: RoomModel, speakers: dict, mics: dict, *,
                     truth_dir: str | Path, impairments: Impairments | None = None,
                     kit: SweepKit | None = None, seed: int = 20260930,
                     synth_config: SynthConfig | None = None, case: Literal["A", "B"] = "B",
                     take_order: list[str] | None = None, reverse_order: bool = False,
                     noise_seconds: float = 15, rir_cache: dict | None = None,
                     external_rirs: dict[tuple[str, str], np.ndarray] | None = None,
                     external_rir_provenance: dict | None = None,
                     purpose: Literal["commercial", "internal-validation"] = "commercial",
                     real_sources: dict[str, np.ndarray] | None = None,
                     include_reference: bool = True,
                     timing: TakeTiming | None = None,
                     device_chain: DeviceChain | None = None) -> SessionTruth:
    """Write a user's folder, with no embedded truth or simulator-only manifest keys.

    Mic names exclude REF: those are generated 0.3 m on +x for each speaker.
    Caller-provided take_order lists all output filenames exactly once. Per-file
    seeds keep waveforms invariant to order; no per-take peak/SNR normalization.
    Optional rir_cache is keyed by complete model/config/position/fs/seed identity.
    Opt-in external_rirs must cover every (source, mic), including REF_source,
    with finite mono pressure responses at kit.fs_hz and emission sample zero.
    Opt-in real_sources maps S1_<mic|REF>_real_<label>.wav to the separate dry
    performance. Neither option changes legacy synthesis, filenames or RNGs.
    include_reference=False omits REF rendering/capture entirely, so a measured
    truth adapter need not invent an unavailable close-reference response.
    external_rir_provenance declares the ancestry of supplied arrays. The explicit
    internal-validation purpose permits restricted inputs and stamps every output;
    those outputs remain prohibited for training and export. Missing provenance
    is not licence clearance. Existing nonempty outputs are never overwritten. Raw WAVs are PCM16.
    timing opts into independent clocks, arbitrary padding/jitter/events, and the
    sequential_phone protocol. None preserves historical RNG streams and bytes.
    device_chain replaces legacy speaker/mic/harmonic/AGC processing explicitly.
    Use Impairments.clean(...) for remaining acquisition settings. A uses the
    speaker for real takes; B bypasses it. Table reflection is rendered by synth.
    """
    from sensorsim.licensing import lineage_metadata, require_commercial_lineage
    if purpose not in {"commercial", "internal-validation"}:
        raise ValueError("purpose must be commercial or internal-validation")
    lineage = lineage_metadata(model, synth_config, device_chain, external_rir_provenance)
    if purpose == "commercial":
        require_commercial_lineage(lineage, context="session render")
    else:
        lineage = lineage_metadata(lineage, {"internal_validation_only": True})
    output, truth_dir = Path(output).resolve(), Path(truth_dir).resolve()
    if truth_dir == output or output in truth_dir.parents:
        raise ValueError("truth_dir must be outside the delivered session")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if truth_dir.exists() and any(truth_dir.iterdir()):
        raise FileExistsError(truth_dir)
    anchor_mic = "M1" if "M1" in mics else next(iter(mics), None)
    if not speakers or "S1" not in speakers or anchor_mic is None or case not in ("A", "B"):
        raise ValueError("session needs S1, a mic position and case A or B")
    if any(name.startswith("REF") for name in mics):
        raise ValueError("REF microphones are generated from the speaker positions")
    if not np.isfinite(noise_seconds) or noise_seconds <= 0:
        raise ValueError("noise_seconds must be positive")
    if not all(str(name).isalnum() for name in [*speakers, *mics]):
        raise ValueError("position IDs must be alphanumeric")
    kit = kit or canonical_kit()
    p = impairments or (Impairments.clean(kit.fs_hz) if device_chain is not None else Impairments())
    if device_chain is not None:
        if not isinstance(device_chain, DeviceChain):
            raise TypeError("typed DeviceChain required")
        # Revalidate even a model_copy update before crossing the ingest boundary.
        device_chain = DeviceChain.model_validate(device_chain.model_dump())
        if p.speaker_response or p.mic_response or p.harmonic2 or p.harmonic3 or p.agc or p.noise_suppression:
            raise ValueError("device_chain cannot stack legacy speaker/mic/nonlinear processing")
        for response in (device_chain.speaker, device_chain.phone):
            apply_linear(np.zeros(2), kit.fs_hz, response)  # fail before creating output
        if device_chain.table is not None:
            validate_table(model, speakers, device_chain.table)
    if timing is not None:
        p = p.model_copy(update={"clock_drift_ppm":0.})  # timing owns this per take
    config = synth_config or SynthConfig(duration_s=1.5)
    fs = kit.fs_hz
    playback = generate_kit_playback(kit)
    if len(playback) > 8_000_000:
        raise ValueError("session take exceeds the bounded eight-million-sample budget")
    positions = {str(k): list(v) for k, v in mics.items()}
    for source, xyz in speakers.items():
        if include_reference:
            positions[f"REF_{source}"] = (np.asarray(xyz) + [.3, 0, 0]).tolist()
    def receivers(source):
        return ([f"REF_{source}"] if include_reference else []) + list(mics)
    required = {(s, m) for s in speakers for m in receivers(s)}
    if external_rirs is not None:
        if set(external_rirs) != required:
            raise ValueError("external_rirs must cover exactly all source/mic/REF pairs")
        for h in external_rirs.values():
            a = np.asarray(h)
            if a.ndim != 1 or not a.size or a.size > fs*10 or np.iscomplexobj(a) or not np.isfinite(a).all():
                raise ValueError("external_rirs need bounded finite real mono pressure samples")
    if real_sources is not None:
        allowed = [f"S1_{m}_real_" for m in (["REF"] if include_reference else []) + list(mics)]
        if not real_sources:
            raise ValueError("real_sources must not be empty")
        for name, source in real_sources.items():
            prefix = next((p for p in allowed if name.startswith(p)), None)
            label = name[len(prefix):-4] if prefix and name.endswith('.wav') else ''
            a = np.asarray(source)
            if not label or not label.replace('_', '').isalnum() or a.ndim != 1 or not a.size or a.size > fs*30 or np.iscomplexobj(a) or not np.isfinite(a).all():
                raise ValueError("real_sources require S1 mic/REF filenames and finite mono dry samples")
    rirs, metadata = {}, {}
    for source, xyz in speakers.items():
        for mic in receivers(source):
            identity = f"{source}:{mic}"
            if external_rirs is not None:
                rirs[identity] = np.asarray(external_rirs[source, mic], dtype=np.float64).copy()
                metadata[identity] = dict(method="externally_supplied", time_origin_s=0., latency_samples=0)
                continue
            key = (model.model_hash, tuple(xyz), tuple(positions[mic]), fs,
                   config.model_dump_json(), _seed(seed, identity))
            cached = None if rir_cache is None else rir_cache.get(key)
            if cached is None:
                result = synthesize(model, xyz, positions[mic], fs,
                                    seed=_seed(seed, identity), config=config)
                h = result.samples[result.metadata['latency_samples']:].copy()
                cached = h, result.metadata
                if rir_cache is not None:
                    rir_cache[key] = cached
            rirs[identity], metadata[identity] = cached
    lineage = lineage_metadata(lineage, metadata)
    if purpose == "commercial":
        require_commercial_lineage(lineage, context="session RIRs")
    if lineage:
        metadata = {key: {**value, "lineage": lineage} for key, value in metadata.items()}
    def play(x):
        if device_chain is None:
            return _speaker(x, fs, p)
        # Bound the offline LTI tail explicitly; no circular shift or cropped delay.
        tail = round(fs*(.25+device_chain.speaker.delay_s))
        return speaker_signal(np.pad(x, (0, tail)), fs, device_chain)
    speaker_playback = play(playback)
    def clean_capture(x, h, latency):
        wet = p.recording_gain * signal.fftconvolve(x, h)
        if device_chain is None:
            wet = signal.sosfilt(device_sos(fs, p, speaker=False), wet)
        else:
            tail = round(fs*(.25+device_chain.phone.delay_s))
            wet = apply_linear(np.pad(wet, (0, tail)), fs, device_chain.phone)
        return _clock(np.r_[np.zeros(round(latency*fs)), wet], fs, p)
    anchor, _ = clean_capture(speaker_playback, rirs[f'S1:{anchor_mic}'], 0)
    anchor_rms = _active_rms(anchor, kit, 0, p)
    noise_rms = 0. if p.snr_db is None else anchor_rms*10**(-p.snr_db/20)
    del anchor
    rows, dry, info = [], {}, {}
    for source in speakers:
        for mic in receivers(source):
            token = 'REF' if mic.startswith('REF') else mic
            rows.append(dict(file=f"{source}_{token}_sweep.wav", type="sweep", speaker=source, mic=mic))
            if source == 'S1':
                if real_sources is None:
                    rows.append(dict(file=f"S1_{token}_real_transient.wav", type="real", speaker=source,
                                     mic=mic, label="synthetic broadband transient"))
                else:
                    prefix = f"S1_{token}_real_"
                    rows.extend(dict(file=name, type="real", speaker=source, mic=mic,
                                     label=name[len(prefix):-4]) for name in sorted(real_sources)
                                if name.startswith(prefix))
    rows.append(dict(file='noise_floor.wav', type='noise', mic=anchor_mic))
    if take_order is not None:
        if len(take_order) != len(rows) or set(take_order) != {r['file'] for r in rows}:
            raise ValueError("take_order must name every output file exactly once")
        by_file = {r['file']: r for r in rows}
        rows = [by_file[name] for name in take_order]
    elif reverse_order:
        rows.reverse()
    output.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row['file']
        rng = np.random.default_rng(_seed(seed, name))
        noise_rng = np.random.default_rng(_seed(seed, name+':noise'))
        dither_rng = np.random.default_rng(_seed(seed, name+':dither'))
        latency = float(rng.uniform(*p.latency_s))
        if row['type'] == 'noise':
            clean, header_fs = _clock(np.zeros(round(noise_seconds*fs)), fs, p)
        else:
            identity = f"{row['speaker']}:{row['mic']}"
            x = speaker_playback
            if row['type'] == 'real':
                source = (broadband_transient(fs, _seed(seed, name)) if real_sources is None
                          else np.asarray(real_sources[name], dtype=np.float64))
                dry[name] = source
                x = play(source) if case == 'A' else source
            clean, header_fs = clean_capture(x, rirs[identity], latency)
        known_rms = _active_rms(clean, kit, latency, p) if row['type'] == 'sweep' else float(np.sqrt(np.mean(clean**2)))
        timing_truth = None
        if timing is not None:
            clean, timing_truth = apply_take_timing(clean,header_fs,timing,
                                                   seed=_seed(seed,name+':timing'),
                                                   inject_events=row['type'] != 'noise')
        wet = clean + noise_rng.normal(0, noise_rms, len(clean))
        processing = p if device_chain is None else p.model_copy(update={"agc": device_chain.phone_agc})
        wet = _processing(wet, header_fs, processing, noise_rms) * p.clip_drive
        clipped = float(np.mean((wet < -1) | (wet > 32767/32768)))
        wet = np.clip(wet, -1, 32767/32768)
        wet += (dither_rng.random(len(wet))-dither_rng.random(len(wet)))/32768
        pcm = np.clip(np.rint(wet*32768), -32768, 32767).astype(np.int16)
        wavfile.write(output/name, header_fs, pcm)
        info[name] = dict(latency_s=latency, active_clean_rms=known_rms,
                          known_snr_db=float(20*np.log10(max(known_rms, 1e-30)/noise_rms)) if noise_rms else None,
                          clipping_fraction=clipped, peak_dbfs=float(20*np.log10(max(np.max(np.abs(pcm.astype(float)))/32768,1e-30))),
                          sha256=hashlib.sha256((output/name).read_bytes()).hexdigest())
        if timing_truth is not None:
            info[name]['timing'] = timing_truth
    position_rng = np.random.default_rng(_seed(seed, 'positions'))
    def measured(table):
        return {name: (np.asarray(xyz) + position_rng.uniform(-p.coordinate_error_m, p.coordinate_error_m, 3)).tolist()
                for name, xyz in sorted(table.items())}
    recorder = dict(device="Synthetic phone (not measured Galaxy S26)", app="session_sim",
                    mic_hole="bottom", processing="unprocessed, NS/AGC off")
    if p.declare_processing:
        recorder.update(agc=p.agc, noise_suppression=p.noise_suppression)
        if p.agc or p.noise_suppression:
            recorder['processing'] = 'AGC on' if p.agc else 'noise suppression ON'
    manifest = dict(room_id=model.room_id, fs_hz=fs, recorder=recorder,
                    speaker=dict(model="synthetic consumer speaker", eq_off=True, facing="+x"),
                    room=dict(dims_m=dict(zip('xyz', model.shoebox_dims_m))),
                    obstacles=[b.model_dump(include={'name','min_xyz','size_xyz','material'}, mode='json') for b in model.obstacles],
                    positions=dict(speakers=measured(speakers), mics=measured(positions)), takes=rows,
                    excitation=dict(type='ess', kit='sweep_kit.json'),
                    notes=f"Synthetic rehearsal; separate performances; real takes case {case}.")
    if timing is not None or not include_reference:
        manifest['acquisition'] = 'sequential_phone'
    if device_chain is not None:
        manifest['speaker'].update(model='synthetic laptop mono equivalent (unvalidated)',
            eq_off=False, compressor=device_chain.compressor.model_dump() if device_chain.compressor else None,
            processing='declared static compression' if device_chain.compressor else 'fixed linear response')
        manifest['recorder'].update(agc=device_chain.phone_agc, noise_suppression=False,
            processing='AGC on' if device_chain.phone_agc else 'unprocessed, NS/AGC off')
    if lineage:
        manifest['lineage'] = lineage
    (output/'sweep_kit.json').write_text(kit.model_dump_json(indent=2)+'\n')
    (output/'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    np.savez_compressed(truth_dir/'rirs.npz', **rirs)
    np.savez_compressed(truth_dir/'dry_sources.npz', **dry)
    payload = dict(seed=seed, case=case, model=model.model_dump(mode='json'),
                   random_streams='independent source, latency, noise and dither namespaces v2',
                   synthesis=config.model_dump(mode='json'), impairments=p.model_dump(mode='json'),
                   kit=kit.model_dump(mode='json'), speakers=speakers, mics=positions,
                   takes=info, rir_metadata=metadata, noise_rms=noise_rms,
                   validity=VALIDITY.model_dump() if hasattr(VALIDITY, 'model_dump') else str(VALIDITY),
                   git=_git_head(),
                   environment=dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in ('numpy','scipy','pydantic')}),
                   source_sha256={str(Path(__file__).name): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if timing is not None:
        payload['timing'] = timing.model_dump(mode='json')
    if device_chain is not None:
        from . import devices
        payload['device_chain'] = device_chain.model_dump(mode='json')
        payload['source_sha256']['devices.py'] = hashlib.sha256(Path(devices.__file__).read_bytes()).hexdigest()
    if lineage:
        payload['lineage'] = lineage
    evidence = truth_dir/'truth.json'
    evidence.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return SessionTruth(output/'manifest.yaml', evidence, rirs, metadata, info, dry, noise_rms)
