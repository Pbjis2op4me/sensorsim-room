"""Strict recording manifests; coordinates in one room-local Cartesian frame (metres)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from scipy.signal import resample_poly
from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, FilePath, PositiveFloat, PrivateAttr,
                      ValidationInfo, field_validator, model_validator)

from sensorsim.realdata.audio_io import probe, read_audio
from sensorsim.core.audio_units import AudioCalibration
from sensorsim.core.serialization import load_yaml_unique
from sensorsim.core.fingerprint import sha256_file

XYZ = tuple[float, float, float]
Positions = XYZ | list[XYZ]


def _event_window(value):
    if not 0 <= value[0] < value[1] or not np.all(np.isfinite(value)):
        raise ValueError("event_window_s needs finite 0 <= start < end in recording seconds")
    return value


EventWindow = Annotated[tuple[float, float], AfterValidator(_event_window)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ESSExcitation(StrictModel):
    type: Literal["ess"] = "ess"
    duration_s: PositiveFloat
    f_start_hz: PositiveFloat
    f_end_hz: PositiveFloat
    amplitude: Annotated[float, Field(gt=0, le=1)] = 0.5
    fade_s: Annotated[float, Field(ge=0)] = 0.0
    fade_in_s: Annotated[float, Field(ge=0)] | None = None
    fade_out_s: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.f_end_hz <= self.f_start_hz or 2 * self.fade_s >= self.duration_s:
            raise ValueError("ESS requires increasing frequencies and 2*fade_s < duration_s")
        if (self.fade_in_s or self.fade_s) + (self.fade_out_s or self.fade_s) >= self.duration_s:
            raise ValueError("ESS fades must fit inside sweep")
        return self


def _relative_path(value, info: ValidationInfo):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path((info.context or {}).get("base_dir", Path.cwd())) / path
    return path.resolve()


class KnownExcitation(StrictModel):
    type: Literal["known_signal"] = "known_signal"
    path: FilePath
    _resolve = field_validator("path", mode="before")(_relative_path)


class UnknownExcitation(StrictModel):
    type: Literal["unknown"] = "unknown"


Excitation = Annotated[ESSExcitation | KnownExcitation | UnknownExcitation,
                       Field(discriminator="type")]


class Calibration(StrictModel):
    pa_per_fs: PositiveFloat | list[PositiveFloat]


class UnprocessedSource(StrictModel):
    """A support-query record is not independent proof of unprocessed audio."""
    requested: str = "unknown"
    actual: str = "unknown"
    support_query: Literal["unknown", "supported", "unsupported"] = "unknown"
    evidence: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def evidenced_query(self):
        if self.support_query != "unknown" and not (self.evidence or "").strip():
            raise ValueError("unprocessed support query requires evidence; an app label is not an API result")
        return self


class CaptureIdentity(StrictModel):
    device: str = "unknown"
    os_version: str = "unknown"
    app: str = "unknown"
    app_version: str = "unknown"
    unprocessed_source: UnprocessedSource = Field(default_factory=UnprocessedSource)
    agc: Literal["unknown", "off", "on"] = "unknown"
    noise_suppression: Literal["unknown", "off", "on"] = "unknown"
    echo_cancellation: Literal["unknown", "off", "on"] = "unknown"
    gain: str = "unknown"
    microphone_route: str = "unknown"
    mic_hole: str = "unknown"
    orientation: str = "unknown"
    case_and_stand: str = "unknown"
    native_format: str = "unknown"

    @model_validator(mode="after")
    def fixed_chain(self):
        if "on" in (self.agc, self.noise_suppression, self.echo_cancellation):
            raise ValueError("AGC/noise suppression/echo cancellation must be OFF; re-record with fixed gain")
        return self


class RecordingTake(StrictModel):
    _channel_count: int | None = PrivateAttr(default=None)
    _file_fs: int | None = PrivateAttr(default=None)
    file: FilePath
    source_xyz_m: XYZ | None = None
    receiver_xyz_m: Positions
    excitation: Excitation
    calibration: Calibration | None = None
    mic_ids: list[str] | None = None
    receiver_orientation_xyz: Positions | None = None  # unit look vectors; descriptive only
    notes: str | None = None
    type: Literal["sweep", "real", "noise"] = "sweep"
    is_reference: bool = False
    # Omit the legacy default from serialized manifests and identity hashes.
    role: Literal["fit", "check"] = Field(default="fit", exclude_if=lambda v: v == "fit")
    speaker_id: str | None = None
    label: str | None = None
    event_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    event_window_s: EventWindow | None = None
    event_pre_roll_s: Annotated[float, Field(ge=0)] = .1
    event_post_roll_s: Annotated[float, Field(ge=0)] | None = None
    # Bursts closer than this form one event. Raise it (e.g. 0.5) for multi-part sounds such as
    # knocks or footsteps, which the default 0.15 s splits into separate events.
    event_merge_gap_s: Annotated[float, Field(ge=0, le=5)] = .15
    clock_drift_ppm: Annotated[float, Field(ge=-300, le=300)] | None = None
    _resolve = field_validator("file", mode="before")(_relative_path)

    @field_validator("excitation", mode="before")
    @classmethod
    def unknown_shorthand(cls, value):
        return {"type": "unknown"} if value == "unknown" else value

    @model_validator(mode="after")
    def nonempty_positions(self):
        if self.role == "check" and (self.type != "sweep" or self.is_reference):
            raise ValueError("check role requires a non-REF sweep take")
        if self.event_index is not None and self.event_window_s is not None:
            raise ValueError("supply event_index or event_window_s, not both")
        if self.event_window_s is not None and self.type != 'real':
            raise ValueError('event_window_s applies only to real-sound takes')
        if self.type != "noise" and self.source_xyz_m is None:
            raise ValueError("source coordinates required except for noise takes")
        if len(self.receiver_xyz_m) == 0:
            raise ValueError("receiver positions cannot be empty")
        if self.receiver_orientation_xyz is not None:
            vectors = np.atleast_2d(self.receiver_orientation_xyz)
            if not len(vectors) or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6):
                raise ValueError("receiver orientations must be unit look vectors")
        return self


def _audio_info(path: Path) -> dict:
    if path.suffix.lower() not in {".wav", ".flac"}:
        raise ValueError(f"only WAV/FLAC supported: {path}")
    info = probe(path)
    if not info or info["fs"] <= 0 or info["channels"] <= 0 or info["duration_s"] <= 0:
        raise ValueError(f"empty/invalid audio or missing FLAC decoder: {path}")
    return info


class RecordingManifest(StrictModel):
    lineage: dict = Field(default_factory=dict, exclude_if=lambda v: not v)
    schema_version: Literal["1"] = "1"
    room_id: Annotated[str, Field(min_length=1)]
    fs: Annotated[int, Field(strict=True, gt=0)]
    acquisition: Literal["legacy", "sequential_phone"] = "legacy"
    shoebox_dims_m: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None
    takes: Annotated[list[RecordingTake], Field(min_length=1)]
    notes: str | None = None
    obstacles: list["BoxObstacle"] = Field(default_factory=list)
    surfaces: dict[str, str] = Field(default_factory=dict)
    recorder: dict = Field(default_factory=dict)
    capture_identity: CaptureIdentity | None = Field(default=None, exclude_if=lambda v: v is None)
    capture_qa: FilePath | None = Field(default=None, exclude_if=lambda v: v is None)
    _resolve_qa = field_validator("capture_qa", mode="before")(
        lambda value, info: _relative_path(value, info) if value is not None else None)
    speaker: dict = Field(default_factory=dict)
    sweep_kit: "SweepKit | None" = None
    named_positions: dict[str, dict[str, XYZ]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_takes(self):
        # A renamed mic, rotated phone or repeated file is not a new holdout.
        # Round at the same micrometre resolution as fit.Observation.position_key.
        checks = [t for t in self.takes if t.role == "check"]
        if checks:
            other = [t for t in self.takes if t.role != "check"]
            if {t.file for t in checks} & {t.file for t in other}:
                raise ValueError("check file must not also be used by a fitting/real/noise take")
            def pairs(take):
                return {tuple(np.round((*take.source_xyz_m, *r), 6))
                        for r in np.atleast_2d(take.receiver_xyz_m)} if take.source_xyz_m else set()
            used = set().union(*(pairs(t) for t in other if t.type != "noise"))
            if any(pairs(t) & used for t in checks):
                raise ValueError("check position must be disjoint from fitting, REF and real-sound positions")
        # Acquisition processing is not invertible with a single fixed linear RIR.
        enabled = {"true", "on", "yes", "enabled", "1"}
        for key in ("agc", "auto_gain", "noise_suppression", "ns", "echo_cancellation"):
            if str(self.recorder.get(key, "")).lower() in enabled:
                raise ValueError("AGC/noise suppression/echo cancellation must be OFF; re-record with fixed gain")
        processing = str(self.recorder.get("processing", "")).lower()
        if re.search(r"\b(agc|auto(?:matic)?[- ]gain(?: control)?|noise suppression|ns)\s*[:=]?\s*(on|enabled)\b", processing):
            raise ValueError("AGC/noise suppression must be OFF; re-record with fixed gain")
        if self.sweep_kit is not None and self.sweep_kit.fs_hz != self.fs:
            raise ValueError("kit and recording sample rates must agree")
        if self.shoebox_dims_m is not None:
            for box in self.obstacles:
                if np.any(np.asarray(box.min_xyz) + box.size_xyz > self.shoebox_dims_m):
                    raise ValueError("obstacle must fit inside room")
        for take in self.takes:
            if (self.sweep_kit is not None and take.type == "sweep" and
                    take.excitation != self.sweep_kit.excitation):
                raise ValueError("sweep take excitation must match the declared kit excitation")
            receivers = np.atleast_2d(take.receiver_xyz_m)
            if self.shoebox_dims_m is not None:
                positions = receivers if take.source_xyz_m is None else np.vstack([take.source_xyz_m, receivers])
                if np.any(positions < 0) or np.any(positions > self.shoebox_dims_m):
                    raise ValueError("source/receiver coordinates must be inside [0, shoebox_dims_m]")
            info = _audio_info(take.file)
            if info["fs"] != self.fs and not (self.sweep_kit and {info["fs"], self.fs} <= {44100,48000}):
                raise ValueError(f"sample rate mismatch for {take.file}: {info['fs']} != {self.fs}")
            channels = info["channels"]
            take._channel_count = channels
            take._file_fs = info["fs"]
            if self.acquisition == "sequential_phone" and channels != 1:
                raise ValueError("sequential_phone takes require one mono phone channel")
            # A single XYZ denotes co-located channels. A list must map every channel.
            if isinstance(take.receiver_xyz_m, list) and len(receivers) != channels:
                raise ValueError("receiver_xyz_m list must match recording channel count")
            if take.mic_ids is not None and (len(take.mic_ids) != channels or
                                           len(set(take.mic_ids)) != channels):
                raise ValueError("mic_ids must be unique and match channel count")
            if isinstance(take.receiver_orientation_xyz, list) and len(take.receiver_orientation_xyz) != channels:
                raise ValueError("orientation list must match channel count")
            if take.calibration and isinstance(take.calibration.pa_per_fs, list):
                if len(take.calibration.pa_per_fs) != channels:
                    raise ValueError("calibration list must match channel count")
            if isinstance(take.excitation, ESSExcitation):
                if take.excitation.f_end_hz >= self.fs / 2:
                    raise ValueError("ESS upper frequency must be below Nyquist")
                if round(take.excitation.duration_s * self.fs) < 2:
                    raise ValueError("ESS must contain at least two samples")
                if info["duration_s"] < take.excitation.duration_s:
                    raise ValueError("recording is shorter than the ESS")
            elif isinstance(take.excitation, KnownExcitation):
                excitation_info = _audio_info(take.excitation.path)
                if excitation_info["fs"] != self.fs or excitation_info["channels"] != 1:
                    raise ValueError("known excitation must be mono at the manifest sample rate")
                if excitation_info["duration_s"] > info["duration_s"]:
                    raise ValueError("recording is shorter than known excitation")
        return self


def load_manifest(path: str | Path) -> RecordingManifest:
    """Validate all headers without loading a recording collection into memory."""
    path = Path(path).resolve()
    with path.open() as stream:
        value = load_yaml_unique(stream)
    if isinstance(value, dict) and "fs_hz" in value:
        return _convert_user_manifest(value, path.parent)
    return RecordingManifest.model_validate(value, context={"base_dir": path.parent})


def channel_calibration(take: RecordingTake, channel: int) -> AudioCalibration | None:
    """Declare peak-FS calibration for one validated recording channel."""
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
        raise ValueError("channel must be a nonnegative integer")
    if take._channel_count is not None and channel >= take._channel_count:
        raise ValueError("channel outside recording")
    if take.calibration is None:
        return None
    scales = take.calibration.pa_per_fs
    scale = scales[channel] if isinstance(scales, list) else scales
    return AudioCalibration(float(scale), "recording", "manifest calibration.pa_per_fs")


def read_take(manifest: RecordingManifest, index: int) -> np.ndarray:
    """Read one take as (samples, channels) float32, FS or Pa if calibrated.

    Kit captures at 44.1/48 kHz are polyphase resampled to the analysis rate;
    there is no channel mixing or inference of a sample clock from another take.
    Nothing records or plays sound. A take must fit in memory.
    """
    take = manifest.takes[index]
    samples, fs = read_recording_audio(take.file)
    if fs != (take._file_fs or manifest.fs) or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("recording is empty/nonfinite or its sample rate changed")
    if take._channel_count is not None and samples.shape[1] != take._channel_count:
        raise ValueError("recording channel count changed since manifest validation")
    # Check digital rails BEFORE Pa/FS scaling. Do not reject arbitrary float WAV
    # peaks above one (they may be unnormalised synthesis/pressure); detect actual
    # saturation at the PCM rails, including converted PCM carried in float WAV.
    rails = (samples <= -1) & (samples >= -1.000001) | ((samples >= 32767/32768) & (samples <= 1.000001))
    # FLAC/ALAC are integer-PCM-only, but probe() labels every non-WAV "lossy"; without
    # this, lossless phone recordings bypassed the integer rail rule (cr-room-capture-01).
    info = probe(take.file)
    integer_pcm = (info.get("sample_format", "").startswith("int")
                   or info.get("codec") in ("flac", "alac"))
    saturated = (np.any(np.sum(rails, axis=0) >= 3) if integer_pcm else
                 np.any(rails[:-2] & rails[1:-1] & rails[2:] &
                        (samples[:-2] == samples[1:-1]) & (samples[1:-1] == samples[2:])))
    if saturated:
        raise ValueError(f"clipping at digital full scale in {take.file.name}; lower fixed gain/volume and re-record")
    if fs != manifest.fs:
        divisor = int(np.gcd(fs,manifest.fs))
        samples = resample_poly(samples,manifest.fs//divisor,fs//divisor,axis=0)
    if take.calibration:
        samples = samples * np.asarray(take.calibration.pa_per_fs)
    samples = samples.astype(np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError("recording/calibration exceeds float32 range")
    return samples


def read_recording_audio(path: Path) -> tuple[np.ndarray, int]:
    """Native WAV reuse; FLAC decode without seek (FFmpeg can drop short files at -ss 0)."""
    if path.suffix.lower() == ".wav":
        return read_audio(path)
    info = _audio_info(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path),
                          "-map", "0:a:0", "-threads", "2", "-f", "f32le", "-acodec",
                          "pcm_f32le", "-"], capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, info["channels"]), info["fs"]


class BoxObstacle(StrictModel):
    name: str
    min_xyz: tuple[Annotated[float, Field(ge=0)], Annotated[float, Field(ge=0)], Annotated[float, Field(ge=0)]]
    size_xyz: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    material: str


class SweepKit(StrictModel):
    """All parameters in room_recording_kit/sweep_kit.json, without WAV dependency."""
    fs_hz: Annotated[int, Field(strict=True, gt=0)]
    f1_hz: PositiveFloat
    f2_hz: PositiveFloat
    sweep_s: PositiveFloat
    fade_in_s: Annotated[float, Field(ge=0)]
    fade_out_s: Annotated[float, Field(ge=0)]
    peak_dbfs: Annotated[float, Field(le=0)]
    lead_silence_s: Annotated[float, Field(ge=0)]
    tail_silence_s: Annotated[float, Field(ge=0)]
    end_silence_s: Annotated[float, Field(ge=0)]
    repeats: Annotated[int, Field(strict=True, ge=1)]
    period_s: PositiveFloat
    first_sweep_start_s: Annotated[float, Field(ge=0)]
    formula: Literal["x(t)=sin(2*pi*f1*L*(exp(t/L)-1)), L=T/ln(f2/f1); half-Hann fades"]
    reference: str
    dither: str
    files: dict[str, dict] = Field(default_factory=dict)
    sweep_samples: Annotated[int, Field(strict=True, ge=2)] | None = Field(default=None, exclude_if=lambda v: v is None)
    resolved_sweep_s: PositiveFloat | None = Field(default=None, exclude_if=lambda v: v is None)
    duration_policy: Literal["round(duration_s*fs)/fs"] | None = Field(default=None, exclude_if=lambda v: v is None)
    kit_version: Literal['1','2'] | None = Field(default=None, exclude_if=lambda v: v is None)

    @property
    def excitation(self) -> ESSExcitation:
        return ESSExcitation(duration_s=self.sweep_s, f_start_hz=self.f1_hz,
                             f_end_hz=self.f2_hz, amplitude=10 ** (self.peak_dbfs / 20),
                             fade_in_s=self.fade_in_s, fade_out_s=self.fade_out_s)

    @model_validator(mode="after")
    def consistent(self):
        self.excitation  # validates fade and frequency ordering
        if self.kit_version is not None:
            tail=4. if self.kit_version=='1' else 8.
            if (self.sweep_s,self.tail_silence_s,self.repeats)!=(10.,tail,4):
                raise ValueError('named kit version requires four 10 s sweeps and its declared 4/8 s quiet tail')
        count = round(self.sweep_s * self.fs_hz)
        if ((self.sweep_samples is not None and self.sweep_samples != count) or
                (self.resolved_sweep_s is not None and self.resolved_sweep_s != count/self.fs_hz)):
            raise ValueError("kit rounded duration/sample count mismatch")
        if self.f2_hz >= self.fs_hz / 2 or round(self.sweep_s * self.fs_hz) < 2:
            raise ValueError("kit sweep outside sampling support")
        if not np.isclose(self.period_s, self.sweep_s + self.tail_silence_s, rtol=0, atol=1e-9):
            raise ValueError("kit period must equal sweep + tail")
        # generate_kit_playback lays each repeat out as round(sweep_s*fs)+round(tail_silence_s*fs)
        # samples, while the detector and timing QA read period_s. A period off that integer grid
        # invented clock drift and resampled a zero-drift capture (cr-room-dsp-03).
        playback_period = round(self.sweep_s * self.fs_hz) + round(self.tail_silence_s * self.fs_hz)
        if not np.isclose(playback_period / self.fs_hz, self.period_s, rtol=0, atol=1e-9):
            raise ValueError(f"kit period_s must be a whole number of samples: the generated playback repeats "
                             f"every {playback_period} samples, not period_s*fs_hz={self.period_s*self.fs_hz:.6g}")
        if self.first_sweep_start_s != self.lead_silence_s:
            raise ValueError("kit first start must equal lead silence")
        return self


def load_sweep_kit(path: str | Path) -> SweepKit:
    with Path(path).open() as stream:
        return SweepKit.model_validate(json.load(stream))


class QAAsset(StrictModel):
    file: FilePath
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    _resolve = field_validator("file", mode="before")(_relative_path)


class QALevelAssets(StrictModel):
    schema_version: Literal["1"] = "1"
    kit: QAAsset
    nominal: QAAsset
    quiet: QAAsset
    digital_difference_db: Literal[-6.0] = -6.0


def load_qa_assets(path: str | Path) -> QALevelAssets:
    """Check hashes AND PCM scaling; two self-consistent wrong hashes aren't evidence.

    Companion assets are separate from sweep_kit.json so its excitation and all
    legacy PCM hashes stay unchanged. Generated quiet PCM is nominal PCM scaled
    by 10**(-6/20), rounded once (maximum half an int16 count error).
    """
    from scipy.io import wavfile
    path = Path(path).resolve()
    assets = QALevelAssets.model_validate(json.loads(path.read_text()), context={"base_dir": path.parent})
    for asset in (assets.kit, assets.nominal, assets.quiet):
        if sha256_file(asset.file) != asset.sha256:
            raise ValueError(f"QA asset hash mismatch: {asset.file.name}")
    kit = load_sweep_kit(assets.kit.file)
    expected = kit.files.get(assets.nominal.file.name, {}).get("sha256")
    if expected != assets.nominal.sha256:
        raise ValueError("nominal QA asset hash must match the kit's playback entry")
    fs, nominal = wavfile.read(assets.nominal.file)
    fq, quiet = wavfile.read(assets.quiet.file)
    count = round((kit.lead_silence_s + kit.repeats*kit.period_s + kit.end_silence_s)*kit.fs_hz)
    if (fs != kit.fs_hz or fq != fs or nominal.dtype != np.int16 or quiet.dtype != np.int16 or
            nominal.shape != (count,) or quiet.shape != nominal.shape or not np.any(nominal)):
        raise ValueError("QA assets must be matching nonzero mono PCM16 kit playback")
    if not np.all(np.abs(quiet.astype(float)-nominal.astype(float)*10**(-6/20)) <= .5000001):
        raise ValueError("QA quiet asset does not match the declared -6 dB digital scale")
    return assets


class QACapture(StrictModel):
    id: Annotated[str, Field(min_length=1)]
    file: FilePath
    level: Literal["nominal", "quiet"]
    position: Annotated[str, Field(min_length=1)]
    source_xyz_m: XYZ
    receiver_xyz_m: XYZ
    noise_file: FilePath | None = None
    _resolve = field_validator("file", mode="before")(_relative_path)
    _resolve_noise = field_validator("noise_file", mode="before")(
        lambda value, info: _relative_path(value, info) if value is not None else None)


class CaptureQASession(StrictModel):
    schema_version: Literal["1"] = "1"
    assets: FilePath
    takes: Annotated[list[QACapture], Field(min_length=1)]
    notes: str | None = None
    _resolve = field_validator("assets", mode="before")(_relative_path)

    @model_validator(mode="after")
    def distinct_takes(self):
        if len({t.id for t in self.takes}) != len(self.takes) or len({t.file for t in self.takes}) != len(self.takes):
            raise ValueError("QA take ids and recording files must be unique")
        positions = {}
        for take in self.takes:
            coordinates = (take.source_xyz_m, take.receiver_xyz_m)
            if positions.setdefault(take.position, coordinates) != coordinates:
                raise ValueError("QA position must denote identical speaker/mic coordinates")
            if _audio_info(take.file)["channels"] != 1:
                raise ValueError("QA requires sequential mono recordings")
            if take.noise_file is not None and _audio_info(take.noise_file)["channels"] != 1:
                raise ValueError("QA noise recording must be mono")
        return self


def load_capture_qa(path: str | Path) -> CaptureQASession:
    """Separate QA takes never become production nominal sweep rows."""
    path = Path(path).resolve()
    session = CaptureQASession.model_validate(load_yaml_unique(path.read_text()),
                                              context={"base_dir": path.parent})
    load_qa_assets(session.assets)
    return session


class _UserRoom(StrictModel):
    dims_m: dict[Literal["x", "y", "z"], PositiveFloat]
    surfaces: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def dimensions(self):
        if set(self.dims_m) != {"x", "y", "z"}:
            raise ValueError("room dimensions require x, y, z")
        return self


class _UserPositions(StrictModel):
    speakers: dict[str, XYZ]
    mics: dict[str, XYZ]


class _UserExcitation(StrictModel):
    type: Literal["ess"]
    kit: FilePath
    _resolve = field_validator("kit", mode="before")(_relative_path)


class _UserTake(StrictModel):
    file: FilePath
    type: Literal["sweep", "real", "noise", "REF"]
    role: Literal["fit", "check"] = "fit"
    speaker: str | None = None
    mic: str | list[str]
    label: str | None = None
    notes: str | None = None
    calibration: Calibration | None = None
    event_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    event_window_s: EventWindow | None = None
    event_pre_roll_s: Annotated[float, Field(ge=0)] = .1
    event_post_roll_s: Annotated[float, Field(ge=0)] | None = None
    # Bursts closer than this form one event. Raise it (e.g. 0.5) for multi-part sounds such as
    # knocks or footsteps, which the default 0.15 s splits into separate events.
    event_merge_gap_s: Annotated[float, Field(ge=0, le=5)] = .15
    clock_drift_ppm: Annotated[float, Field(ge=-300, le=300)] | None = None
    _resolve = field_validator("file", mode="before")(_relative_path)


class _UserManifest(StrictModel):
    lineage: dict = Field(default_factory=dict, exclude_if=lambda v: not v)
    room_id: str
    fs_hz: Annotated[int, Field(strict=True, gt=0)]
    acquisition: Literal["legacy", "sequential_phone"] = "legacy"
    recorder: dict = Field(default_factory=dict)
    capture_identity: CaptureIdentity | None = None
    capture_qa: FilePath | None = None
    _resolve_qa = field_validator("capture_qa", mode="before")(
        lambda value, info: _relative_path(value, info) if value is not None else None)
    speaker: dict = Field(default_factory=dict)
    room: _UserRoom
    obstacles: list[BoxObstacle] = Field(default_factory=list)
    positions: _UserPositions
    takes: list[_UserTake]
    excitation: _UserExcitation
    notes: str | None = None


def _convert_user_manifest(value, base_dir: Path) -> RecordingManifest:
    raw = _UserManifest.model_validate(value, context={"base_dir": base_dir})
    kit = load_sweep_kit(raw.excitation.kit)
    dims = tuple(raw.room.dims_m[axis] for axis in ("x", "y", "z"))
    for xyz in [*raw.positions.speakers.values(), *raw.positions.mics.values()]:
        if np.any(np.asarray(xyz) < 0) or np.any(np.asarray(xyz) > dims):
            raise ValueError("named position outside room")
    takes = []
    for take in raw.takes:
        ids = [take.mic] if isinstance(take.mic, str) else take.mic
        if not ids or any(mic not in raw.positions.mics for mic in ids):
            raise ValueError("unknown or empty named mic position")
        if take.type != "noise" and take.speaker not in raw.positions.speakers:
            raise ValueError("unknown named speaker position")
        if take.speaker is not None and take.speaker not in raw.positions.speakers:
            raise ValueError("unknown named speaker position")
        sweep = take.type in ("sweep", "REF")
        takes.append(RecordingTake(
            file=take.file, type="sweep" if sweep else take.type,
            role=take.role,
            source_xyz_m=raw.positions.speakers.get(take.speaker),
            receiver_xyz_m=[raw.positions.mics[mic] for mic in ids], mic_ids=ids,
            speaker_id=take.speaker, is_reference=take.type == "REF" or all(mic.startswith("REF") for mic in ids),
            excitation=kit.excitation if sweep else UnknownExcitation(), label=take.label,
            calibration=take.calibration, notes=take.notes,
            event_index=take.event_index, event_window_s=take.event_window_s, event_pre_roll_s=take.event_pre_roll_s,
            event_post_roll_s=take.event_post_roll_s, event_merge_gap_s=take.event_merge_gap_s,
            clock_drift_ppm=take.clock_drift_ppm))
    if raw.fs_hz != kit.fs_hz and {raw.fs_hz,kit.fs_hz} != {44100,48000}:
        raise ValueError("kit and manifest sample rates must agree (44.1/48 kHz conversion is supported)")
    return RecordingManifest(lineage=raw.lineage, room_id=raw.room_id, fs=kit.fs_hz, acquisition=raw.acquisition, shoebox_dims_m=dims,
                             takes=takes, obstacles=raw.obstacles, surfaces=raw.room.surfaces,
                             recorder=raw.recorder, speaker=raw.speaker, sweep_kit=kit,
                             capture_identity=raw.capture_identity, capture_qa=raw.capture_qa,
                             named_positions=raw.positions.model_dump(), notes=raw.notes)


RecordingManifest.model_rebuild()
