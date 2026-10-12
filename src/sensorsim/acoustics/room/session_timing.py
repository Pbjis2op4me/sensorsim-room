"""Opt-in, per-file acquisition timing; independent of any room truth engine.

Scenario parameters are engineering stress assumptions, not measured phone
statistics. No speech content is modelled. All randomness is local to this take.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal


class TakeTiming(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    lead_s: tuple[float, float] = (.1, 5.)
    trail_s: tuple[float, float] = (.1, 5.)
    clock_ppm: tuple[float, float] = (-300., 300.)
    start_jitter_s: float = Field(default=.05, ge=0)
    stop_jitter_s: float = Field(default=.05, ge=0)
    extraneous_events: int = Field(default=3, ge=0, le=100)
    event_peak: float = Field(default=.05, ge=0, le=10)

    @model_validator(mode="after")
    def bounds(self):
        for name in ("lead_s", "trail_s", "clock_ppm"):
            lo,hi = getattr(self,name)
            if lo > hi or (name != "clock_ppm" and lo < 0):
                raise ValueError("timing bounds must be ordered and silence nonnegative")
        if min(self.clock_ppm) < -300 or max(self.clock_ppm) > 300:
            raise ValueError("clock stress is bounded to +/-300 ppm")
        if max(self.lead_s+self.trail_s) > 600:
            raise ValueError("one take is bounded to ten minutes of lead/trail")
        return self


def apply_take_timing(samples, fs: int, timing: TakeTiming, *, seed: int,
                      inject_events=True) -> tuple[np.ndarray, dict]:
    """Return clocked/padded capture and truth-only timing/event metadata.

    Call before stationary recorder noise, processing and quantization. The
    input must already be on the declared nominal file rate with ZERO residual
    clock drift. Different per-file seeds give independent clocks and offsets;
    iteration order never changes them. Jitter cannot cut the source waveform.
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or fs <= 0:
        raise ValueError("timing needs finite mono samples and a positive rate")
    rng = np.random.default_rng(seed)
    ppm = float(rng.uniform(*timing.clock_ppm))
    lead_jitter = float(rng.uniform(-timing.start_jitter_s,timing.start_jitter_s))
    trail_jitter = float(rng.uniform(-timing.stop_jitter_s,timing.stop_jitter_s))
    lead = max(0.,float(rng.uniform(*timing.lead_s))+lead_jitter)
    trail = max(0.,float(rng.uniform(*timing.trail_s))+trail_jitter)
    n = max(2,round(len(x)*(1+ppm*1e-6)))
    actual_ppm = (n/len(x)-1)*1e6
    wet = np.r_[np.zeros(round(lead*fs)),signal.resample(x,n),np.zeros(round(trail*fs))]
    events = []
    for i in range(timing.extraneous_events if inject_events else 0):
        duration = float(rng.uniform(.01,.35))
        count = min(len(wet),max(2,round(duration*fs)))
        start = int(rng.integers(0,len(wet)-count+1))
        t = np.arange(count)/fs
        kind = ("handling", "burst", "click", "slam")[i%4]
        if kind == "handling":
            event = np.sin(2*np.pi*75*t)*np.hanning(count)
        elif kind == "click":
            event = np.zeros(count); event[0] = 1.
        else:
            event = rng.normal(size=count)*np.exp(-t/max(duration/4,.001))
            event /= max(np.max(abs(event)),1e-30)
        wet[start:start+count] += timing.event_peak*event
        events.append(dict(kind=kind,start_sample=start,stop_sample=start+count,peak=timing.event_peak))
    return wet, dict(lead_samples=round(lead*fs),trail_samples=round(trail*fs),
                     start_jitter_s=lead_jitter,stop_jitter_s=trail_jitter,
                     clock_drift_ppm=actual_ppm,requested_clock_ppm=ppm,events=events)
