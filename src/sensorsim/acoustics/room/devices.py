"""Opt-in, versioned laptop/phone engineering proxies, never device calibration.

Reuse SciPy Butterworth/SOS, project RBJ EQ and oversampled static compression.
The table is a named RoomModel obstacle; all reflected paths belong to synth.
See parent-project note ROOM_LAPTOP_CHAIN (not distributed) for measurement provenance and unsupported mechanisms.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal

from sensorsim.core.messages import ModelValidity
from .augment import EQBand, parametric_eq
from .model import RoomModel

VALIDITY = ModelValidity(
    name="Laptop speaker and phone acquisition chain", status="unvalidated",
    freq_range_hz=(125., 8000.),
    conditions="Deterministic engineering sensitivity only; minimum-phase response plus "
    "sample-quantized delay, mono equivalent source, optional static compression and "
    "legacy block AGC. Not measured laptop phase, stereo directivity, vendor DSP, "
    "phone calibration or absolute SPL. Table inherits room obstacle limitations.",
    citation="W3C Audio EQ Cookbook; SciPy butter/sosfilt/resample_poly; "
    "Manawyrm GPD Pocket 4 Speaker DSP (2025); Android raw-source documentation; "
    "parent-project note ROOM_LAPTOP_CHAIN (not distributed)",
)


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class LinearResponse(_Spec):
    """Causal pressure transfer; identity by default. No extrapolated Nyquist cutoffs.

    Delay rounds to the nearest integer sample (<=0.5 sample error). Filters
    retain their own phase/group delay. Output has input length; callers needing
    ring-out must supply trailing zeros, as the session adapter does.
    """
    highpass_hz: float | None = Field(default=None, gt=0.)
    highpass_order: int = Field(default=2, strict=True, ge=1, le=8)
    lowpass_hz: float | None = Field(default=None, gt=0.)
    resonances: tuple[EQBand, ...] = Field(default=(), max_length=8)
    delay_s: float = Field(default=0., ge=0., le=.05)

    @model_validator(mode="after")
    def _band(self):
        if self.highpass_hz and self.lowpass_hz and self.highpass_hz >= self.lowpass_hz:
            raise ValueError("highpass must be below lowpass")
        return self


class StaticCompressor(_Spec):
    """Instantaneous hard-knee pressure law, not an attack/release vendor model.

    Above threshold, output dB = threshold + (input dB - threshold)/ratio,
    bounded by ceiling. Units are peak digital FS before room spreading.
    The oversampled wrapper reduces aliasing; it is not alias-free for this
    non-bandlimited nonlinearity. Anti-alias FIR ring-out may overshoot ceiling.
    """
    threshold_dbfs: float = Field(default=-18., ge=-60., le=0.)
    ratio: float = Field(default=2., ge=1., le=20.)
    ceiling_dbfs: float = Field(default=-1., ge=-60., le=0.)

    @model_validator(mode="after")
    def _levels(self):
        if self.ceiling_dbfs < self.threshold_dbfs:
            raise ValueError("ceiling must be at or above threshold")
        return self


class TablePlacement(_Spec):
    """Source centre above the top of an existing finite box, in room metres."""
    obstacle: str = Field(min_length=1)
    source_height_m: float = Field(gt=0., le=1.)
    source_ids: tuple[str, ...] = Field(default=("S1",), min_length=1)


class DeviceChain(_Spec):
    version: Literal["laptop-phone/1"] = "laptop-phone/1"
    speaker: LinearResponse = Field(default_factory=lambda: LinearResponse(
        highpass_hz=250., highpass_order=4, lowpass_hz=10000., delay_s=.00025,
        resonances=(EQBand(frequency_hz=750., gain_db=3., q=1.2),
                    EQBand(frequency_hz=3500., gain_db=5., q=1.5))))
    phone: LinearResponse = Field(default_factory=lambda: LinearResponse(
        highpass_hz=80., lowpass_hz=14000.,
        resonances=(EQBand(frequency_hz=3000., gain_db=2., q=.8),)))
    compressor: StaticCompressor | None = None
    phone_agc: bool = False
    processing_declared: bool = False
    table: TablePlacement | None = None

    @model_validator(mode="after")
    def _declared(self):
        if (self.compressor is not None or self.phone_agc) and not self.processing_declared:
            raise ValueError("nonlinear processing must be explicitly declared")
        return self


def apply_linear(samples, fs: int, response: LinearResponse) -> np.ndarray:
    """Mono finite offline block, zero initial state, same sample count."""
    x = np.asarray(samples, dtype=float)
    if np.iscomplexobj(samples) or x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("device input must be finite real mono samples, length >=2")
    if not isinstance(fs, (int, np.integer)) or isinstance(fs, bool) or fs < 1000:
        raise ValueError("device fs must be an integer >=1000 Hz")
    if not isinstance(response, LinearResponse):
        raise TypeError("typed LinearResponse required")
    frequencies = [v for v in (response.highpass_hz, response.lowpass_hz) if v is not None]
    frequencies.extend(b.frequency_hz for b in response.resonances)
    if any(f >= fs/2 for f in frequencies):
        raise ValueError("device frequencies must lie below Nyquist")
    x = x.copy()
    for frequency, order, kind in ((response.highpass_hz, response.highpass_order, "highpass"),
                                   (response.lowpass_hz, 2, "lowpass")):
        if frequency is not None:
            x = signal.sosfilt(signal.butter(order, frequency, kind, fs=fs, output="sos"), x)
    if response.resonances:
        x = parametric_eq(x, fs, response.resonances)
    delay = round(response.delay_s*fs)
    if delay:
        x = np.pad(x, (delay, 0))[:len(x)]
    return x


def compress_static(samples, compressor: StaticCompressor) -> np.ndarray:
    """Odd symmetric static law; separate to support an analytic dB oracle."""
    x = np.asarray(samples, dtype=float)
    if np.iscomplexobj(samples) or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("compressor needs finite real mono samples")
    if not isinstance(compressor, StaticCompressor):
        raise TypeError("typed StaticCompressor required")
    threshold = 10**(compressor.threshold_dbfs/20)
    ceiling = 10**(compressor.ceiling_dbfs/20)
    level = np.abs(x)
    compressed = threshold*(level/threshold)**(1/compressor.ratio)
    return np.sign(x)*np.minimum(np.where(level > threshold, compressed, level), ceiling)


def speaker_signal(samples, fs: int, chain: DeviceChain) -> np.ndarray:
    """Apply speaker LTI response then optional fourfold oversampled compression."""
    if not isinstance(chain, DeviceChain):
        raise TypeError("typed DeviceChain required")
    y = apply_linear(samples, fs, chain.speaker)
    if chain.compressor is None:
        return y
    taps = signal.firwin(257, .25, window=("kaiser", 8.6))
    hi = signal.resample_poly(y, 4, 1, window=taps)
    return signal.resample_poly(compress_static(hi, chain.compressor), 1, 4, window=taps)[:len(y)]


def validate_table(model: RoomModel, speakers: dict, placement: TablePlacement) -> None:
    """Validate geometry, never synthesize an extra device echo or move the source."""
    matches = [b for b in model.obstacles if b.name == placement.obstacle]
    if len(matches) != 1:
        raise ValueError("table must name exactly one existing room obstacle")
    table = matches[0]
    for name in placement.source_ids:
        if name not in speakers:
            raise ValueError("table source ID is absent")
        xyz = np.asarray(speakers[name], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("table source needs finite XYZ")
        if not np.all((xyz[:2] >= np.array(table.min_xyz[:2])) & (xyz[:2] <= table.max_xyz[:2])):
            raise ValueError("table source must project onto the tabletop")
        if abs(xyz[2]-table.max_xyz[2]-placement.source_height_m) > 1e-9:
            raise ValueError("table source height disagrees with geometry")
