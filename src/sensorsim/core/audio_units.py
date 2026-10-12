"""Explicit audio boundary references; internal DSP still uses ndarrays.

Pa is physical pressure. Peak full scale is a digital amplitude reference, not
RMS SPL: a full-scale sine has RMS 1/sqrt(2). Relative room/RIR amplitudes are
not calibrated pressure and must not enter these pressure adapters implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .clock import SampleTimeMap


@dataclass(frozen=True)
class AudioCalibration:
    pa_per_peak_fs: float
    purpose: str
    gain_provenance: str
    reference: str = "sample_peak_full_scale"

    def __post_init__(self):
        if not np.isfinite(self.pa_per_peak_fs) or self.pa_per_peak_fs <= 0:
            raise ValueError("pa_per_peak_fs must be finite and positive")
        if not self.purpose or not self.gain_provenance or self.reference != "sample_peak_full_scale":
            raise ValueError("explicit peak full-scale reference, purpose and gain provenance required")


@dataclass(frozen=True)
class AudioMetadata:
    fs_hz: int
    channel_labels: tuple[str, ...]
    start_sample: int
    time_map: SampleTimeMap
    calibration: AudioCalibration
    synthesis_profile: str
    latency_samples: int = 0
    units: str = "Pa"
    axes: tuple[str, str] = ("sample", "channel")
    version: str = "pressure-audio/1"

    def __post_init__(self):
        if (isinstance(self.fs_hz, bool) or not isinstance(self.fs_hz, Integral) or self.fs_hz <= 0
                or isinstance(self.start_sample, bool) or not isinstance(self.start_sample, Integral)
                or self.start_sample < 0 or not isinstance(self.latency_samples, Integral)
                or self.latency_samples < 0):
            raise ValueError("positive integer fs_hz and nonnegative sample/latency indices required")
        if (not self.channel_labels or any(not isinstance(c, str) or not c for c in self.channel_labels)
                or len(set(self.channel_labels)) != len(self.channel_labels)):
            raise ValueError("unique nonempty channel labels required")
        if (self.units != "Pa" or self.axes != ("sample", "channel") or self.version != "pressure-audio/1"
                or not self.synthesis_profile or not isinstance(self.time_map, SampleTimeMap)
                or not isinstance(self.calibration, AudioCalibration)):
            raise ValueError("explicit Pa/sample/channel, clock, calibration and synthesis profile required")

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["channel_labels"] = tuple(data["channel_labels"])
        data["axes"] = tuple(data["axes"])
        data["time_map"] = SampleTimeMap(**data["time_map"])
        data["calibration"] = AudioCalibration(**data["calibration"])
        return cls(**data)


