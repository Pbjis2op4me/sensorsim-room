"""PCM16 adapters between calibrated pressure (Pa) and signed 16-bit samples.

No propagation or normalization lives here.
"""
from __future__ import annotations

import numpy as np

from sensorsim.core.audio_units import AudioCalibration


def encode_pcm16(pressure_pa, calibration: AudioCalibration, *, clipping: str = "saturate") -> np.ndarray:
    """Round Pa/peak-FS * 32768; saturate asymmetrically to signed int16.

    ``raise`` rejects physical peaks at or above full scale before rounding;
    the final clip still handles the positive half-LSB endpoint correctly.
    """
    x = np.asarray(pressure_pa)
    if np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("PCM input pressure must be finite and real")
    if clipping not in ("saturate", "raise"):
        raise ValueError("clipping must be saturate or raise")
    scale = calibration.pa_per_peak_fs
    # Compare each sign directly: abs() overflows on a signed integer minimum.
    if clipping == "raise" and np.any((x >= scale) | (x <= -scale)):
        raise ValueError("pressure would clip at the declared peak full scale")
    # Clamp before multiplying as well, avoiding overflow on large finite Pa.
    q = np.rint(np.clip(x, -scale, scale)/scale*32768.)
    return np.clip(q, -32768, 32767).astype(np.int16)


def decode_pcm16(samples, calibration: AudioCalibration) -> np.ndarray:
    data = np.asarray(samples)
    if data.dtype.kind != "i" or data.dtype.itemsize != 2:
        raise ValueError("explicit signed int16 PCM required")
    return data.astype(np.float64)*(calibration.pa_per_peak_fs/32768.)
