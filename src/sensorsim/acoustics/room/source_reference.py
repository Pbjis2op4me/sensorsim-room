"""Dry-to-source boundary for the room renderer's 1/r, one-metre convention.

This removes geometric REF gain only. It does not remove mic coloration, infer
source directivity, or certify a near-field REF as a free-field measurement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

from sensorsim.core.audio_units import AudioCalibration


@dataclass(frozen=True)
class SourceReference:
    """Explicit amplitude/time reference, independent of waveform dtype or peak.

    Raw DryClip inputs declare relative source amplitude at 1 m by their API
    contract. Extracted Case B instead declares ref_microphone and must supply
    its measured distance before synthesis. Unknown distance is never 0.3 m.
    """
    target: Literal["source_1m", "ref_microphone", "played_digital", "effective_source"] = "source_1m"
    units: Literal["relative", "Pa", "peak_fs"] = "relative"
    reference_distance_m: float | None = None
    calibration: AudioCalibration | None = None
    latency_samples: int = 0
    time_origin: str = "input_recording; offline centered inverse adds no sample delay"
    version: str = "room-source-reference/1"

    def __post_init__(self):
        if self.target not in ("source_1m", "ref_microphone", "played_digital", "effective_source"):
            raise ValueError("unknown source target")
        if self.units not in ("relative", "Pa", "peak_fs") or self.version != "room-source-reference/1":
            raise ValueError("unknown source units/version")
        distance = self.reference_distance_m
        if distance is not None and (not np.isfinite(distance) or distance <= 0):
            raise ValueError("reference distance must be finite and positive")
        if self.target == "source_1m":
            if distance not in (None, 1.):
                raise ValueError("normalized source reference distance must be 1 m")
            object.__setattr__(self, "reference_distance_m", 1.)
        if (isinstance(self.latency_samples, bool) or not isinstance(self.latency_samples, int)
                or self.latency_samples < 0 or not self.time_origin):
            raise ValueError("nonnegative integer latency and explicit time origin required")
        if self.calibration is not None and not isinstance(self.calibration, AudioCalibration):
            raise TypeError("calibration must be AudioCalibration")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SourceReference:
        data = dict(value)
        if data.get("calibration") is not None:
            data["calibration"] = AudioCalibration(**data["calibration"])
        return cls(**data)


def source_at_one_m(samples, reference: SourceReference) -> tuple[np.ndarray, SourceReference, float]:
    """Apply Case-B REF distance exactly once; A is an effective relative source."""
    if not isinstance(reference, SourceReference):
        raise TypeError("typed SourceReference required")
    scale = 1.
    if reference.target == "ref_microphone":
        if reference.reference_distance_m is None:
            raise ValueError("Case B needs a declared reference distance before room synthesis")
        scale = reference.reference_distance_m
    output = replace(reference, target="source_1m", reference_distance_m=1.)
    return np.asarray(samples) * scale, output, scale
