"""Continuous A/C weighting, IEC 61672-1:2013 Annex E.

The +2 dB normalization is retained exactly; it is approximately unity at 1 kHz.
Digital meter filters remain separately characterized realizations of this curve.
"""
from __future__ import annotations

import numpy as np

A_WEIGHTING_METHOD = "a-continuous-precise/1"
A_POLES_HZ = (20.598997, 107.65265, 737.86223, 12194.217)
A_NORMALIZATION_DB = 2.0
C_WEIGHTING_METHOD = "c-continuous-precise/1"
C_NORMALIZATION_DB = 0.062  # -20 log10(Rc(1000 Hz)) = .061901858... dB, rounded to .001 dB.


def _response(f: np.ndarray) -> np.ndarray:
    f1, f2, f3, f4 = A_POLES_HZ
    squared = np.asarray(f, dtype=np.float64) ** 2
    return (f4 ** 2 * squared ** 2) / (
        (squared + f1 ** 2)
        * np.sqrt((squared + f2 ** 2) * (squared + f3 ** 2))
        * (squared + f4 ** 2))


def a_weighting_gain(f: np.ndarray) -> np.ndarray:
    """Continuous linear amplitude gain; DC is exactly zero."""
    return _response(f) * 10.0 ** (A_NORMALIZATION_DB / 20.0)


def a_weighting_exact_db(f: np.ndarray) -> np.ndarray:
    """Continuous dB gain; retain the meter's 1e-300 pre-normalization floor."""
    return 20.0 * np.log10(np.maximum(_response(f), 1e-300)) + A_NORMALIZATION_DB


def _c_response(f: np.ndarray) -> np.ndarray:
    f1, _, _, f4 = A_POLES_HZ
    squared = np.asarray(f, dtype=np.float64) ** 2
    return f4 ** 2 * squared / ((squared + f1 ** 2) * (squared + f4 ** 2))


def c_weighting_gain(f: np.ndarray) -> np.ndarray:
    """IEC Annex E C-curve linear pressure gain; DC is exactly zero."""
    return _c_response(f) * 10.0 ** (C_NORMALIZATION_DB / 20.0)


def c_weighting_exact_db(f: np.ndarray) -> np.ndarray:
    """Continuous C dB gain with the A API's 1e-300 pre-normalization floor."""
    return 20.0 * np.log10(np.maximum(_c_response(f), 1e-300)) + C_NORMALIZATION_DB


def a_weighted_mean_square(x: np.ndarray, fs: float) -> np.ndarray:
    """A-weighted mean square of x (..., n), in input units squared (rFFT Parseval)."""
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    w = a_weighting_gain(np.fft.rfftfreq(n, 1.0 / fs)) ** 2
    p = np.abs(spec) ** 2 * w
    p[..., 1:(n + 1) // 2] *= 2.0  # DC and even-length Nyquist counted once
    return p.sum(axis=-1) / n ** 2
