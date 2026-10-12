"""Small deterministic model and WAV helpers used only by evaluation tests."""
from __future__ import annotations

import json

import numpy as np
from scipy.io import wavfile

from sensorsim import edge


def save_edge_model(path, *, seed: int = 20261008, width: int = 48, blocks: int = 4):
    rng = np.random.default_rng(seed)
    arrays = {}

    def weights(name, shape, fan_in):
        arrays[name] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape).astype(np.float32)

    def bias(name, size):
        arrays[name] = np.zeros(size, np.float32)

    weights("conv0.w", (width, 1, 5, 5), 25)
    bias("conv0.b", width)
    layers = [{"type": "conv", "name": "conv0", "stride": [2, 2], "pad": [2, 2], "act": "relu"}]
    for i in range(blocks):
        name = f"b{i}"
        weights(f"{name}.dw.w", (width, 3, 3), 9)
        bias(f"{name}.dw.b", width)
        weights(f"{name}.pw.w", (width, width), width)
        bias(f"{name}.pw.b", width)
        layers.append({"type": "dw", "name": f"{name}.dw", "stride": [2, 1] if i == 1 else [1, 1],
                       "pad": [1, 1], "act": "relu"})
        layers.append({"type": "pw", "name": f"{name}.pw", "act": "relu"})
    layers.append({"type": "pool"})
    weights("fc.w", (3, 2*width), 2*width)
    bias("fc.b", 3)
    layers.append({"type": "fc", "name": "fc"})
    meta = {
        "format": edge.FORMAT,
        "classes": ["knock", "doorbell", "kettle"],
        "layers": layers,
        "features": {"fs": 16000, "window_s": 1.0, "n_fft": 512, "hop": 160,
                     "n_mels": 40, "fmin_hz": 60.0, "fmax_hz": 7600.0},
        "detector": {"hop_s": .25, "off_ratio": .6, "smooth_hops": 3, "min_hops": 2,
                     "on": {"knock": .5, "doorbell": .5, "kettle": .5}},
    }
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    np.savez(path, **arrays)
    return edge.Model(str(path))


def write_float_wav(path, audio: np.ndarray, fs: int = 16000):
    x = np.asarray(audio, float).reshape(-1)
    pcm = np.rint(np.clip(x, -1, 1) * 32767).astype(np.int16)
    wavfile.write(path, fs, pcm)


def tone(seconds: float = 2.0, fs: int = 16000, frequency: float = 900.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds*fs))/fs
    envelope = np.minimum(t/.01, 1.0) * np.exp(-t/1.5)
    return (.15*envelope*np.sin(2*np.pi*frequency*t) + .003*rng.normal(size=len(t))).astype(np.float32)
