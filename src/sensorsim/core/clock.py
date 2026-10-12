"""Sample-to-time maps for audio units: sample indices to seconds on a declared time base."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np



@dataclass(frozen=True)
class SampleTimeMap:
    """Observable affine sample map, excluding unobserved ADC jitter.

    The map has a run-wide reference, rather than a new epoch at every block.
    ``master`` denotes synchronized observations, including their residual stamp
    error; it does not denote ground truth. ``node`` is ineligible for fusion.
    """
    clock_id: str
    reference_sample: int
    reference_time_s: float
    seconds_per_sample: float
    time_domain: str = "node"
    synchronization_id: str | None = None
    version: str = "sample-time-map/1"

    def __post_init__(self):
        if self.version != "sample-time-map/1":
            raise ValueError("unsupported sample time map version")
        if not self.clock_id or self.time_domain not in ("node", "master"):
            raise ValueError("sample time map requires a clock ID and supported time domain")
        if (not isinstance(self.reference_sample, Integral) or isinstance(self.reference_sample, bool)
                or not isfinite(self.reference_time_s) or not isfinite(self.seconds_per_sample)
                or self.seconds_per_sample <= 0):
            raise ValueError("sample time map requires a finite reference and positive slope")
        if self.time_domain == "master" and not self.synchronization_id:
            raise ValueError("master time map requires synchronization identity")
        if self.time_domain == "node" and self.synchronization_id is not None:
            raise ValueError("unsynchronized node map cannot claim synchronization")

    def timestamp(self, sample):
        self.__post_init__()  # ingest may follow deserialization rather than construction
        samples = np.asarray(sample)
        if samples.dtype.kind in "iu":
            # Subtract in integer space: casting an absolute counter first loses
            # adjacent samples above 2**53. Python integers also avoid int64 wrap.
            relative = np.asarray(samples.astype(object) - int(self.reference_sample), dtype=np.float64)
        else:
            relative = np.asarray(sample, dtype=np.float64) - self.reference_sample
        if not np.all(np.isfinite(relative)):
            raise ValueError("sample indices must be finite")
        return self.reference_time_s + relative * self.seconds_per_sample
