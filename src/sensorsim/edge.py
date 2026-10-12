# SPDX-License-Identifier: Apache-2.0
"""Edge runtime: recognize trained sounds in live or recorded audio. numpy only.

Self-contained on purpose: copy this file and a trained ``model.npz`` to the device
(Raspberry Pi or similar). Needs Python 3.9+ and numpy 1.20+ (Raspberry Pi OS ships both
via apt); no scipy, no other package, no network, and the model loads without pickle.

    arecord -r 16000 -f S16_LE -c 1 -t raw | python3 edge.py listen model.npz      # live mic
    python3 edge.py listen model.npz recording.wav                                    # a file
    python3 edge.py calibrate model.npz quiet_day.wav --fa-per-hour 1 --out tuned.npz  # set thresholds
    python3 edge.py bench model.npz                                                   # speed on this device
    python3 edge.py info model.npz

Capture at the model's sample rate (``info`` shows it, normally 16 kHz) so the device never
resamples; WAV files at an integer multiple or fraction of it are resampled here.

``listen`` prints one JSON line when a sound starts (for alerts) and one when it ends.
Each window of audio is scored by a small depthwise-separable CNN with one sigmoid per
class; scores are smoothed over a few hops, and an event needs its score above the class's
"on" threshold and ends when it falls below the "off" threshold (hysteresis).

Part of sensorsim-room. Licensed under the Apache License, Version 2.0 (see LICENSE and
NOTICE, which must travel with this file).
"""
from __future__ import annotations

import os

if __name__ == "__main__":  # on the device: one stream per process, so extra BLAS threads only add overhead
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
import wave

import numpy as np

FORMAT = "sensorsim-room-edge/1"


# ----------------------------------------------------------------------------------- features
def mel_filterbank(fs: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular HTK-mel filters, shape (n_mels, n_fft // 2 + 1), float32."""
    mel = lambda f: 2595.0*np.log10(1.0 + f/700.0)
    hz = lambda m: 700.0*(10.0**(m/2595.0) - 1.0)
    edges = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0/fs)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0.0, np.minimum((freqs - lo)/(mid - lo), (hi - freqs)/(hi - mid))).astype(np.float32)


class Features:
    """Gain-normalized log-mel spectrogram of fixed-length windows (training and device share this)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n = int(round(cfg["window_s"]*cfg["fs"]))
        self.bank = mel_filterbank(cfg["fs"], cfg["n_fft"], cfg["n_mels"], cfg["fmin_hz"], cfg["fmax_hz"])
        self.window = np.hanning(cfg["n_fft"]).astype(np.float32)
        self.frames = (self.n - cfg["n_fft"])//cfg["hop"] + 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(n,) or (batch, n) float audio -> (batch, 1, n_mels, frames) float32."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        if x.shape[-1] != self.n:
            raise ValueError(f"expected windows of {self.n} samples, got {x.shape[-1]}")
        # Remove each window's DC: raw MEMS capsules on I2S often carry an offset that would
        # otherwise leak into the lowest mel bands of quiet rooms.
        x = x - x.mean(axis=-1, keepdims=True)
        frames = np.lib.stride_tricks.sliding_window_view(x, self.cfg["n_fft"], axis=-1)[:, ::self.cfg["hop"]]
        power = np.abs(np.fft.rfft(frames*self.window, axis=-1))**2
        log_mel = np.log(np.einsum("btk,mk->bmt", power, self.bank) + 1e-10)
        log_mel -= log_mel.mean(axis=(1, 2), keepdims=True)  # an overall gain change is removed exactly
        return log_mel[:, None].astype(np.float32)


# -------------------------------------------------------------------------------------- model
class Model:
    """Interpreter for the exported layer list (BatchNorm already folded into the weights)."""

    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as data:
            self.meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            self.weights = {k: data[k].astype(np.float32) for k in data.files if k != "meta"}
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} model")
        self.classes = list(self.meta["classes"])
        self.features = Features(self.meta["features"])

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """(batch, 1, n_mels, frames) -> (batch, classes)."""
        x = feats
        for layer in self.meta["layers"]:
            x = self._apply(layer, x)
        return x

    def _apply(self, layer: dict, x: np.ndarray) -> np.ndarray:
        kind = layer["type"]
        if kind in ("conv", "dw", "pw", "fc"):
            w, b = self.weights[layer["name"] + ".w"], self.weights[layer["name"] + ".b"]
        if kind in ("conv", "dw"):
            (sf, st), (pf, pt) = layer["stride"], layer["pad"]
            xp = np.pad(x, ((0, 0), (0, 0), (pf, pf), (pt, pt)))
            win = np.lib.stride_tricks.sliding_window_view(xp, w.shape[-2:], axis=(2, 3))[:, :, ::sf, ::st]
            spec = "bcftij,ocij->boft" if kind == "conv" else "bcftij,cij->bcft"
            x = np.einsum(spec, win, w, optimize=True) + b[None, :, None, None]
        elif kind == "pw":
            x = np.einsum("bcft,oc->boft", x, w, optimize=True) + b[None, :, None, None]
        elif kind == "pool":  # mean over frequency, then mean and max over time
            m = x.mean(axis=2)
            x = np.concatenate([m.mean(axis=-1), m.max(axis=-1)], axis=1)
        elif kind == "fc":
            x = x @ w.T + b
        else:
            raise ValueError(f"unknown layer type {kind!r}")
        return np.maximum(x, 0.0) if layer.get("act") == "relu" else x

    def scores(self, audio_windows: np.ndarray) -> np.ndarray:
        """Per-class probabilities (sigmoid) for (batch, n) or (n,) audio windows."""
        z = self.logits(self.features(audio_windows))
        return 1.0/(1.0 + np.exp(-z))

    def macs(self) -> int:
        """Multiply-accumulates per window, counted from the layer shapes."""
        x = np.zeros((1, 1, self.meta["features"]["n_mels"], self.features.frames), np.float32)
        total = 0
        for layer in self.meta["layers"]:
            x = self._apply(layer, x)
            if layer["type"] in ("conv", "dw", "pw", "fc"):
                w = self.weights[layer["name"] + ".w"]
                per_output = w.shape[1] if layer["type"] in ("pw", "fc") else int(np.prod(w.shape[1:]))
                total += x[0].size*per_output
        return int(total)


# ---------------------------------------------------------------------------------- detection
class Tracker:
    """Turns a sequence of per-window scores into events: smoothing, hysteresis, minimum length.

    One window is scored every hop; a class's score is averaged with its previous
    ``smooth_hops - 1``; the class starts when that average reaches its "on" threshold and
    ends when it drops below ``off_ratio`` times it. Events shorter than ``min_hops`` are
    reported as unconfirmed. ``on``, ``off_ratio``, ``smooth_hops`` and ``min_hops`` are each
    one value for every class or a {class: value} mapping: a short transient (a knock) is
    visible to the model for fewer hops than a long sound, so it needs less smoothing.
    """

    def __init__(self, classes: list, cfg: dict, window_s: float):
        self.classes, self.cfg, self.window_s = classes, cfg, window_s
        per_class = lambda key: np.array([cfg[key][c] if isinstance(cfg[key], dict) else cfg[key] for c in classes])
        self.on = per_class("on").astype(float)
        self.off = self.on*per_class("off_ratio")
        self.smooth_hops = per_class("smooth_hops").astype(int)
        self.min_hops = per_class("min_hops").astype(int)
        self.history: list = []
        self.active: dict = {}

    def step(self, score: np.ndarray, window_end: float) -> list:
        self.history = (self.history + [score])[-int(self.smooth_hops.max()):]
        recent = np.asarray(self.history)
        smooth = np.array([recent[-k:, i].mean() for i, k in enumerate(self.smooth_hops)])
        start = round(window_end - self.window_s, 3)
        out = []
        for i, label in enumerate(self.classes):
            event = self.active.get(i)
            if event is None and smooth[i] >= self.on[i]:
                self.active[i] = {"label": label, "start_s": start, "end_s": round(window_end, 3),
                                  "peak": float(smooth[i]), "hops": 1}
                out.append({"event": "start", "label": label, "time_s": start, "score": round(float(smooth[i]), 3)})
            elif event is not None:
                if smooth[i] >= self.off[i]:
                    event["peak"] = max(event["peak"], float(smooth[i]))
                    event["hops"] += 1
                    event["end_s"] = round(window_end, 3)
                else:
                    out.append(self._close(i))
        return out

    def _close(self, i: int) -> dict:
        e = self.active.pop(i)
        return {"event": "end", "label": e["label"], "start_s": e["start_s"], "end_s": e["end_s"],
                "peak": round(e["peak"], 3), "confirmed": bool(e["hops"] >= self.min_hops[i])}

    def flush(self) -> list:
        return [self._close(i) for i in list(self.active)]


class Detector:
    """Streaming detector: feed audio chunks (float in [-1, 1] or int16), receive event dicts."""

    def __init__(self, model: Model, *, hop_s: float | None = None, thresholds: dict | None = None):
        cfg = dict(model.meta["detector"])
        cfg.update(thresholds or {})
        self.model, self.fs, self.n = model, model.meta["features"]["fs"], model.features.n
        self.hop = int(round((hop_s or cfg["hop_s"])*self.fs))
        self.tracker = Tracker(model.classes, cfg, self.n/self.fs)
        self.buffer = np.zeros(0, np.float32)
        self.consumed = 0

    def push(self, chunk: np.ndarray) -> list:
        chunk = np.asarray(chunk)
        chunk = chunk.astype(np.float32)/32768.0 if chunk.dtype == np.int16 else chunk.astype(np.float32)
        self.buffer = np.concatenate([self.buffer, chunk])
        out = []
        while self.buffer.size >= self.n:
            out += self.tracker.step(self.model.scores(self.buffer[:self.n])[0], (self.consumed + self.n)/self.fs)
            self.buffer = self.buffer[self.hop:]
            self.consumed += self.hop
        return out

    def flush(self) -> list:
        return self.tracker.flush()


def window_scores(model: Model, audio: np.ndarray, hop_s: float, batch: int = 256) -> np.ndarray:
    """Per-window scores (windows, classes) for windows starting every hop; batched for speed."""
    n, hop = model.features.n, int(round(hop_s*model.meta["features"]["fs"]))
    starts = np.arange(0, audio.size - n + 1, hop)
    rows = [model.scores(np.stack([audio[s:s + n] for s in starts[i:i + batch]])) for i in range(0, starts.size, batch)]
    return np.concatenate(rows) if rows else np.zeros((0, len(model.classes)), np.float32)


def track(model: Model, scores: np.ndarray, hop_s: float, thresholds: dict | None = None) -> list:
    """Confirmed events from precomputed window scores (same logic as the streaming Detector)."""
    cfg = dict(model.meta["detector"])
    cfg.update(thresholds or {})
    window_s = model.features.n/model.meta["features"]["fs"]
    tracker, events = Tracker(model.classes, cfg, window_s), []
    for k, score in enumerate(scores):
        events += [e for e in tracker.step(score, round(k*hop_s + window_s, 6)) if e["event"] == "end"]
    return [e for e in events + tracker.flush() if e["confirmed"]]


def detect(model: Model, audio: np.ndarray, *, hop_s: float | None = None, thresholds: dict | None = None) -> list:
    """Offline detection over a whole recording (batched; identical to streaming)."""
    hop_s = hop_s or model.meta["detector"]["hop_s"]
    return track(model, window_scores(model, audio, hop_s), hop_s, thresholds)


# ---------------------------------------------------------------------------------------- audio
def resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Integer-ratio resampler, numpy only, with the filter of scipy.signal.resample_poly's defaults.

    A Kaiser-windowed sinc (beta 5, ten zero crossings per side of the slower rate, cutoff at the
    lower Nyquist, unit DC gain), applied without delay. The training data is resampled with
    resample_poly, so device input resampled here keeps the same response near Nyquist.
    """
    if up == down:
        return x
    g = np.gcd(up, down)
    up, down = up//g, down//g
    rate = max(up, down)
    half = 10*rate
    t = np.arange(-half, half + 1)
    h = np.sinc(t/rate)*np.kaiser(2*half + 1, 5.0)
    h = (h*up/h.sum()).astype(np.float32)
    stuffed = np.zeros(x.size*up, np.float32)
    stuffed[::up] = x
    y = np.convolve(stuffed, h)[half:][:x.size*up]
    return y[::down]


def _open_pcm_wav(path):
    try:
        return wave.open(str(path), "rb")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path}: expected an integer PCM WAV; float/compressed or malformed WAV "
                         "cannot be read. Keep the original and export a PCM16 (signed 16-bit PCM) "
                         "WAV copy without normalization or denoising.") from exc


def _wav_rate(path):
    with _open_pcm_wav(path) as wav:
        return wav.getframerate()


def read_wav(path: str, fs: int) -> np.ndarray:
    """Mono float32 audio at `fs` from an integer-PCM WAV (8/16/24/32-bit), channels averaged."""
    with _open_pcm_wav(path) as w:
        rate, ch, width, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8) >> 8
        x = x/8388608.0
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0)/128.0
    else:
        dtype = {2: np.int16, 4: np.int32}[width]
        x = np.frombuffer(raw, dtype).astype(np.float32)/float(-np.iinfo(dtype).min)
    x = x.reshape(-1, ch).mean(axis=1).astype(np.float32)
    if rate != fs:
        if max(rate, fs) % min(rate, fs):
            raise ValueError(f"{path}: {rate} Hz is not an integer ratio of the model's {fs} Hz; resample it first")
        x = resample(x, fs, rate)
    return x


# ------------------------------------------------------------------------------------ commands
def _audio_anchor_keys(audio):
    """Eight-sample exact-content lookup keys; SHA256 confirms every match.

    Uint64 arithmetic deliberately wraps. This is an index, not a cryptographic
    fingerprint or an amplitude-invariant acoustic signature. Numpy only, so the
    calibration record is also available in the standalone device folder.
    """
    words = np.ascontiguousarray(audio, dtype='<f4').view('<u4')
    keys = np.zeros(max(0, len(words)-7), dtype=np.uint64)
    for i in range(8):
        keys *= np.uint64(1000003)
        keys += words[i:i+len(keys)].astype(np.uint64)
    return keys


def calibration_audio_record(audio, fs, *, recording_id=None, sample_start=0, path=None):
    """Record the decoded calibration extent and exact one-second block hashes.

    Ranges are half-open samples at the model rate, in the original recording's
    timeline. Supply the same recording_id for every excerpt of that recording.
    Without an identity, the whole decoded hash is a content-addressed identity;
    it cannot establish the ancestry of transformed excerpts. Gain changes and
    resampling defeat exact hashes. A renamed crop containing a complete stored
    nonconstant block is detectable even when its starting offset is arbitrary.
    """
    import hashlib
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32)/32768
    audio = np.ascontiguousarray(audio, dtype='<f4')
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('calibration audio must be finite nonempty mono')
    if isinstance(fs, bool) or not np.isfinite(fs) or int(fs) != fs or fs < 1:
        raise ValueError('calibration sample_rate_hz must be a positive integer')
    if isinstance(sample_start, bool) or not isinstance(sample_start, (int, np.integer)) or sample_start < 0:
        raise ValueError('calibration sample_start must be a nonnegative integer')
    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    if recording_id is None:
        recording_id = 'decoded-sha256:' + digest
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError('calibration recording_id must be a nonempty string')
    size = max(8, int(fs))
    blocks, constant = [], 0
    for start in range(0, len(audio)-size+1, size):
        block = audio[start:start+size]
        if np.ptp(block) == 0:
            constant += 1
            continue
        blocks.append({'offset': start, 'anchor': format(int(_audio_anchor_keys(block[:8])[0]), '016x'),
                       'sha256': hashlib.sha256(block.tobytes()).hexdigest()})
    return {'schema': 'calibration-audio/1', 'path': str(path or '<calibration array>'),
            'recording_id': recording_id, 'sample_range': [int(sample_start), int(sample_start)+len(audio)],
            'sample_rate_hz': int(fs), 'decoded_sha256': digest,
            'fingerprint': {'algorithm': 'float32le-exact-blocks/1', 'block_samples': size,
                            'anchor_samples': 8, 'blocks': blocks, 'constant_blocks_omitted': constant,
                            'trailing_samples_omitted': len(audio) % size}}


def _calibration_grid_counts(scores, grid, *, off_ratio, smooth_hops, min_hops):
    """Count all grid candidates, including nonmonotone hysteresis splits.

    Vectorized over thresholds only. The final settings are independently replayed
    with Tracker below; this table is a search accelerator, not a second runtime.
    """
    active = np.zeros(len(grid), bool)
    hops = np.zeros(len(grid), int)
    starts = np.zeros(len(grid), int)
    confirmed = np.zeros(len(grid), int)
    for k in range(len(scores)):
        smooth = scores[max(0, k + 1 - smooth_hops):k + 1].mean()
        begin = ~active & (smooth >= grid)
        close = active & (smooth < grid*off_ratio)
        confirmed += close & (hops >= min_hops)
        hops = np.where(close, 0, np.where(begin, 1, hops + active))
        active = (active & ~close) | begin
        starts += begin
    confirmed += active & (hops >= min_hops)
    return {"start_alerts": starts, "confirmed_events": confirmed}


def calibrate(model: Model, audio: np.ndarray, *, fa_per_hour: float = 1.0,
              budget_mode: str = "combined", event_unit: str = "start_alerts",
              recording_id: str | None = None, sample_start: int = 0,
              calibration_path: str | None = None) -> dict:
    """Select thresholds on target-free development audio; never certify deployment.

    Combined counts sum alerts across labels (simultaneous labels count separately).
    Per-class mode gives each label the stated allowance. Defaults budget the start
    alerts emitted by listen, including unconfirmed starts. All candidate counts are
    evaluated: increasing a threshold can split an event. Best achievable means
    the minimum within the declared grid, not over every possible detector policy.
    Three zero-event hours are the minimum floor exposure for a 1/hour 95% bound;
    selected thresholds still require untouched audit audio and a recall check.
    """
    import hashlib
    if budget_mode not in ("combined", "per_class"):
        raise ValueError("budget_mode must be combined or per_class")
    if event_unit not in ("start_alerts", "confirmed_events"):
        raise ValueError("unknown event unit")
    if not np.isfinite(fa_per_hour) or fa_per_hour < 0:
        raise ValueError("fa_per_hour must be finite and nonnegative")
    audio = np.asarray(audio)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("calibration audio must be finite nonempty mono")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32)/32768
    cfg = {key: model.meta["detector"][key] for key in
           ("hop_s", "on", "off_ratio", "smooth_hops", "min_hops")}
    fs, n = model.meta["features"]["fs"], model.features.n
    audio_record = calibration_audio_record(audio, fs, recording_id=recording_id,
                                           sample_start=sample_start, path=calibration_path)
    # Recalibration starts from the previous thresholds: retain that data ancestry.
    previous = model.meta['detector'].get('calibration', {})
    audio_records = list(previous.get('calibration_audio_records', []))
    previous_hash = previous.get('calibration_audio_sha256')
    if previous_hash and not any(r.get('decoded_sha256') == previous_hash for r in audio_records):
        audio_records.append({'decoded_sha256': previous_hash, 'path': '<legacy calibration audio>'})
    audio_records.append(audio_record)
    hop_s = float(cfg["hop_s"])
    if not np.isfinite(fs) or fs <= 0 or n < 1 or not np.isfinite(hop_s) or round(hop_s*fs) < 1:
        raise ValueError("invalid sample rate, window or hop")
    # Match Detector sample rounding, rather than inventing fractional-sample hops.
    hop_s = round(hop_s*fs)/fs
    if len(audio) < n or not model.classes or len(set(model.classes)) != len(model.classes):
        raise ValueError("a complete window and unique nonempty classes are required")
    scores = window_scores(model, audio, hop_s)
    expected = 1 + (len(audio)-n)//round(hop_s*fs)
    if scores.shape != (expected, len(model.classes)) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("invalid or empty model scores; expected one probability per window/class")
    hours = len(audio)/fs/3600
    allowed = int(np.floor(fa_per_hour*hours))
    grids, tables = {}, {}
    value = lambda key, label: cfg[key][label] if isinstance(cfg[key], dict) else cfg[key]
    for i, label in enumerate(model.classes):
        default = float(value("on", label))
        off = float(value("off_ratio", label))
        smooth, minimum = value("smooth_hops", label), value("min_hops", label)
        if (not 0 < default <= 1 or not 0 < off <= 1 or
                int(smooth) != smooth or smooth < 1 or int(minimum) != minimum or minimum < 1):
            raise ValueError("invalid detector thresholds or persistence")
        # Preserve exact defaults, include the cap exactly, never lower a default.
        grid = np.unique(np.r_[default, np.arange(default, .995, .005), max(default, .995)])
        grids[label] = grid
        tables[label] = _calibration_grid_counts(scores[:, i], grid, off_ratio=off,
            smooth_hops=int(smooth), min_hops=int(minimum))
    counts = {c: tables[c][event_unit] for c in model.classes}
    best = {c: int(np.argmin(counts[c])) for c in model.classes}
    selected = dict.fromkeys(model.classes, 0)
    if budget_mode == "per_class":
        for c in model.classes:
            feasible = np.flatnonzero(counts[c] <= allowed)
            selected[c] = int(feasible[0]) if len(feasible) else best[c]
    else:
        # Start at the proven grid minimum, then relax one class at a time
        # within the shared allowance. This is deterministic, not recall-optimal.
        selected = dict(best)
        if sum(int(counts[c][selected[c]]) for c in model.classes) <= allowed:
            for c in model.classes:
                others = sum(int(counts[k][selected[k]]) for k in model.classes if k != c)
                choices = np.flatnonzero(counts[c] <= allowed-others)
                selected[c] = int(choices[0])

    on = {c: float(grids[c][selected[c]]) for c in model.classes}
    tracker = Tracker(model.classes, {**cfg, "on": on}, n/fs)
    events = []
    for k, score in enumerate(scores):
        events.extend(tracker.step(score, round(k*hop_s+n/fs, 6)))
    events.extend(tracker.flush())
    observed = {}
    for unit in ("start_alerts", "confirmed_events"):
        chosen = [e for e in events if (e["event"] == "start" if unit == "start_alerts"
                  else e["event"] == "end" and e["confirmed"])]
        by_class = {c: sum(e["label"] == c for e in chosen) for c in model.classes}
        if any(by_class[c] != int(tables[c][unit][selected[c]]) for c in model.classes):
            raise RuntimeError("calibration search and deployment tracker disagree")
        observed[unit] = {"by_class": by_class, "total": len(chosen),
                          "rate_per_hour": len(chosen)/hours}
    achieved = observed[event_unit]
    feasible = (achieved["total"] <= allowed if budget_mode == "combined" else
                all(k <= allowed for k in achieved["by_class"].values()))
    minimum_hours = float(np.ceil((-np.log(.05)/fa_per_hour)*10)/10) if fa_per_hour else None
    input_constant = bool(np.ptp(audio) == 0)
    sufficient = minimum_hours is not None and hours >= minimum_hours and not input_constant
    best_counts = {c: int(counts[c][best[c]]) for c in model.classes}
    return {"on": on, "status": "infeasible" if not feasible else "feasible" if sufficient else "insufficient_evidence",
            "empirical_feasible": feasible, "exposure_s": len(audio)/fs, "background_hours": hours,
            "minimum_exposure_hours": minimum_hours, "exposure_sufficient": sufficient,
            "input_constant": input_constant,
            "zero_event_poisson_upper_per_hour": -float(np.log(.05))/hours,
            "exposure_rule": "minimum for zero-event Poisson floor only; not a bound after threshold selection",
            "fa_per_hour_budget": fa_per_hour, "allowed_events": allowed,
            "budget_mode": budget_mode, "event_unit": event_unit, "observed": observed,
            "best_achievable_rate_per_hour": sum(best_counts.values())/hours,
            "best_achievable_by_class_per_hour": {c: k/hours for c, k in best_counts.items()},
            "best_achievable_on": {c: float(grids[c][best[c]]) for c in model.classes},
            "grid": {"step": .005, "cap": .995, "selected_indices": selected,
                     "policy": "exact defaults and cap; defaults above cap retained"},
            "calibration_audio_sha256": hashlib.sha256(np.asarray(audio, "<f4").tobytes()).hexdigest(),
            "calibration_audio_records": audio_records,
            "detector_settings": {**cfg, "on": on}, "deployment_validated": False,
            "selection_invalidated": any(on[c] != float(value("on", c)) for c in model.classes),
            "positive_feasibility": "not_checked",
            "requires_reselection": any(on[c] != float(value("on", c)) for c in model.classes),
            "claim": "development threshold selection only; changed policy invalidates prior selection; validate recall and FA on disjoint untouched audio"}


def save_detector(src: str, dst: str, **fields) -> None:
    """Copy a model, replacing detector settings (``on=...``, ``smooth_hops=...``, ...)."""
    with np.load(src, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(bytes(arrays["meta"]).decode("utf-8"))
    changed = any(key in fields and fields[key] != meta["detector"].get(key)
                  for key in ("on", "off_ratio", "smooth_hops", "min_hops", "hop_s"))
    if changed:
        previous = meta.get("training", {}).get("selection")
        if isinstance(previous, dict):
            selected = previous.get("selected") or {}
            frozen = selected.get("candidate", {}).get("policy", selected.get("detector"))
            effective = {**meta["detector"], **fields}
            if frozen is None or any(effective.get(k) != frozen.get(k)
                                     for k in ("on", "off_ratio", "smooth_hops", "min_hops", "hop_s")):
                meta["training"]["selection"] = {**previous, "integration": "stale",
                    "status": "requires_reselection", "deployment_calibrated": False,
                    "invalidation": "detector policy changed after selection"}
    meta["detector"].update(fields)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
    np.savez(dst, **arrays)


def bench(model: Model, seconds: float = 5.0) -> dict:
    rng = np.random.default_rng(0)
    x = (0.05*rng.standard_normal(model.features.n)).astype(np.float32)
    model.scores(x)
    times, start = [], time.perf_counter()
    while time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        model.scores(x)
        times.append(time.perf_counter() - t0)
    per = float(np.median(times))
    hop = model.meta["detector"]["hop_s"]
    params = int(sum(w.size for w in model.weights.values()))
    return {"ms_per_window": round(per*1000, 3), "hop_s": hop, "cpu_fraction_one_core": round(per/hop, 4),
            "macs_per_window": model.macs(), "parameters": params, "model_kib": round(params*4/1024, 1),
            "numpy": np.__version__, "threads": os.environ.get("OPENBLAS_NUM_THREADS")}


def _raw_stdin(fs: int, chunk_s: float = 0.1):
    n = int(fs*chunk_s)*2
    stream = sys.stdin.buffer
    while True:
        raw = stream.read(n)
        if not raw:
            return
        yield np.frombuffer(raw[:len(raw)//2*2], np.int16)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="sensorsim-room edge recognizer (numpy only)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("listen", help="detect sounds in a WAV file, or raw S16_LE mono PCM on stdin")
    p.add_argument("model")
    p.add_argument("audio", nargs="?", default="-", help="WAV file, or '-' for raw PCM on stdin (default)")
    p.add_argument("--hop", type=float, help="seconds between scores (default: the model's)")
    c = sub.add_parser("calibrate", help="set thresholds from a recording with NO target sounds, made on this mic")
    c.add_argument("model")
    c.add_argument("background")
    c.add_argument("--fa-per-hour", type=float, default=1.0)
    c.add_argument("--out", required=True)
    c.add_argument("--budget-mode", choices=("combined", "per_class"), default="combined")
    c.add_argument("--event-unit", choices=("start_alerts", "confirmed_events"), default="start_alerts")
    c.add_argument("--recording-id", help="stable original recording ID shared by all its excerpts")
    c.add_argument("--sample-start", type=int, default=0,
                   help="excerpt start in original recording, in samples at the model rate (default 0)")
    b = sub.add_parser("bench", help="time one window on this device")
    b.add_argument("model")
    b.add_argument("--seconds", type=float, default=5.0)
    i = sub.add_parser("info", help="classes, thresholds, size")
    i.add_argument("model")
    args = parser.parse_args(argv)
    model = Model(args.model)
    fs = model.meta["features"]["fs"]
    if args.command == "listen":
        detector = Detector(model, hop_s=args.hop)
        chunks = _raw_stdin(fs) if args.audio == "-" else iter([read_wav(args.audio, fs)])
        for chunk in chunks:
            for event in detector.push(chunk):
                if event["event"] == "start" or event.get("confirmed"):
                    print(json.dumps(event), flush=True)
        for event in detector.flush():
            if event.get("confirmed"):
                print(json.dumps(event), flush=True)
    elif args.command == "calibrate":
        if args.recording_id is None:
            print("warning: --recording-id omitted; declare the original recording ID and sample "
                  "ranges for calibration and evaluation to protect disjointness.", file=sys.stderr)
        rate = _wav_rate(args.background)
        if rate != fs:
            print(f"warning: background is {rate} Hz; model is {fs} Hz. Resampled fingerprints "
                  "cannot reliably match re-exported calibration audio; declare original recording "
                  "IDs and ranges. Prefer model-rate PCM before calibration; --sample-start is "
                  "in samples at the model rate.", file=sys.stderr)
        result = calibrate(model, read_wav(args.background, fs), fa_per_hour=args.fa_per_hour,
                           budget_mode=args.budget_mode, event_unit=args.event_unit,
                           recording_id=args.recording_id, sample_start=args.sample_start,
                           calibration_path=os.path.basename(args.background))
        # Never publish a silently unsuccessful calibration file.
        result["saved"] = result["status"] == "feasible"
        if result["saved"]:
            save_detector(args.model, args.out, on=result["on"], calibration=result)
        print(json.dumps(result, indent=1))
        if not result["saved"]:
            return 2
    elif args.command == "bench":
        print(json.dumps(bench(model, args.seconds), indent=1))
    else:
        print(json.dumps({"classes": model.classes, "features": model.meta["features"], "detector": model.meta["detector"],
                          "parameters": int(sum(w.size for w in model.weights.values())), "macs_per_window": model.macs(),
                          "training": model.meta.get("training", {})}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
