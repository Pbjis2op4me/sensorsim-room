"""Baseline noise recognizer over a room-dataset manifest.  REPLACE BEFORE PRODUCTION.

A deliberately simple, dependency-light reference: fixed-length windows, per-window
gain-normalized log-mel band statistics (mean, standard deviation and maximum of
each band over time), and a multinomial logistic regression. It exists to prove
the end-to-end loop (capture -> fit -> render -> train -> recognize) and to define
the contract a real model must meet; it is not a tuned or validated detector.

* Training needs the optional ``[recognizer]`` extra (scikit-learn).
* The trained model is plain JSON (feature settings, scaler, weights). Inference
  needs only numpy/scipy, never unpickles anything, and is easy to port.
* Validation scores come from synthetic renders in the same fitted room as the
  training data, so they are optimistic. Measure real performance on separate
  real recordings made with the deployment microphone.

How training windows are built, and why:

* Rendered target clips are clean and are stored at a *relative* level (the
  dataset preserves levels relative to a 1 m reference), while background
  recordings and deployment audio arrive at their real recorded level, typically
  tens of dB apart. So each target window is made the way deployment audio
  arises: a background window at its recorded level, with the target scaled
  INTO it at a random SNR measured over the event's audible span.
* Features remove the window's mean log-mel energy, so an overall gain change
  (microphone gain, distance) does not move the decision.
* Every window has the model's fixed length (``model["features"]["window_s"]``),
  with the event at a random offset inside it. ``predict`` uses the same window.

CLI::

    python -m sensorsim.recognizer add-background --manifest DATASET/manifest.jsonl --train a.wav b.wav --val c.wav
    python -m sensorsim.recognizer train   --manifest DATASET/manifest.jsonl --out model.json
    python -m sensorsim.recognizer predict --model model.json audio.wav [--min-prob .6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

FORMAT = "sensorsim-room-recognizer/2"
UNKNOWN = "unknown"


def feature_config(fs: int, window_s: float = 2.0) -> dict:
    """64 ms Hann frames, 16 ms hop, 40 mel bands from 50 Hz to fs/2, fixed window."""
    n_fft = 1 << int(np.ceil(np.log2(.064*fs)))
    return {"fs": int(fs), "window_s": float(window_s), "n_fft": n_fft, "hop": max(1, round(.016*fs)),
            "n_mels": 40, "fmin_hz": 50., "fmax_hz": fs/2, "stats": ["mean", "std", "max"],
            "normalization": "per-window mean log-mel energy removed"}


def mel_filterbank(cfg: dict) -> np.ndarray:
    """Triangular filters on the HTK mel scale, shape (n_mels, n_fft//2 + 1)."""
    mel = lambda f: 2595.*np.log10(1. + f/700.)
    hz = lambda m: 700.*(10.**(m/2595.) - 1.)
    edges = hz(np.linspace(mel(cfg["fmin_hz"]), mel(cfg["fmax_hz"]), cfg["n_mels"] + 2))
    freqs = np.fft.rfftfreq(cfg["n_fft"], 1/cfg["fs"])
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0., np.minimum((freqs-lo)/(mid-lo), (hi-freqs)/(hi-mid)))


def read_audio(path: str | Path, fs: int) -> np.ndarray:
    """Mono float audio at `fs` (channels averaged; integer PCM scaled to +-1)."""
    rate, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(-np.iinfo(x.dtype).min)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate != fs:
        g = int(np.gcd(rate, fs))
        x = resample_poly(x, fs//g, rate//g)
    return x


def features(x: np.ndarray, cfg: dict, bank: np.ndarray | None = None) -> np.ndarray:
    """Window vector: mean, std and max over time of each gain-normalized log-mel band."""
    bank = mel_filterbank(cfg) if bank is None else bank
    n_fft, hop = cfg["n_fft"], cfg["hop"]
    x = np.asarray(x, dtype=float)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    power = np.abs(np.fft.rfft(frames*np.hanning(n_fft), axis=1))**2
    log_mel = np.log(power @ bank.T + 1e-10)
    log_mel -= log_mel.mean()  # an overall gain g adds log(g**2) everywhere: removed here
    return np.concatenate([log_mel.mean(0), log_mel.std(0), log_mel.max(0)])


def _proba(model: dict, f: np.ndarray) -> np.ndarray:
    """Class probabilities from a feature vector, in the order of model["classes"]."""
    z = (f - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"])
    logits = np.asarray(model["coef"]) @ z + np.asarray(model["intercept"])
    if logits.size == 1:  # binary logistic regression stores one row
        p = 1/(1 + np.exp(-logits[0]))
        return np.array([1 - p, p])
    e = np.exp(logits - logits.max())
    return e / e.sum()


def predict_proba(model: dict, x: np.ndarray, bank: np.ndarray | None = None) -> np.ndarray:
    """Class probabilities for one window of audio (numpy only)."""
    return _proba(model, features(x, model["features"], bank))


def _load_manifest(manifest: Path) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{manifest} has no examples")
    from sensorsim.licensing import validate_training_lineage
    validate_training_lineage(manifest, rows)
    rates = {int(r["fs"]) for r in rows}
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates in manifest: {sorted(rates)}")
    return rows, rates.pop()


def add_background(manifest: str | Path, train: list[str | Path], val: list[str | Path] = (), *,
                   label: str = "background", clip_s: float = 10., storage: str = "pcm16") -> int:
    """Append raw room recordings to a dataset as a `label` class, WITHOUT re-rendering them.

    Recordings made in the room already contain its acoustics; rendering them again (as a
    dataset dry clip) would add a second reverb and turn ambient sound into a point source.
    Each recording is cut into consecutive pieces of up to `clip_s` at the dataset's sample
    rate; training draws its fixed-length windows from them. Splits are by recording: a file
    listed for both train and val is rejected. Returns the number of rows appended.
    """
    if storage not in {"pcm16", "float32"}:
        raise ValueError("unknown background storage")
    manifest = Path(manifest)
    rows, fs = _load_manifest(manifest)
    train, val = [Path(p) for p in train], [Path(p) for p in val]
    if not train:
        raise ValueError("give at least one training background recording")
    if {p.resolve() for p in train} & {p.resolve() for p in val}:
        raise ValueError("a background recording is listed for both train and val (leakage)")
    n = round(clip_s*fs)
    out_dir = manifest.parent/label
    out_dir.mkdir(exist_ok=True)
    added = []
    for split, paths in (("train", train), ("val", val)):
        for path in paths:
            x = read_audio(path, fs)
            if x.size < fs:
                raise ValueError(f"{path} is shorter than one second")
            for k, start in enumerate(range(0, max(1, x.size - n + 1), n) if x.size >= n else [0]):
                piece = x[start:start + n]
                target = out_dir/f"{path.stem}_{k:04d}.wav"
                if target.exists():
                    raise FileExistsError(f"refusing to overwrite {target}")
                payload = (piece.astype(np.float32) if storage == "float32" else
                           np.clip(np.round(piece*32767), -32768, 32767).astype(np.int16))
                wavfile.write(target, fs, payload)
                added.append({"file": f"{label}/{target.name}", "label": label, "split": split, "fs": fs,
                              "output_samples": int(piece.size), "source_group": f"{label}:{path.name}",
                              "raw_recording": str(path.name), "storage": storage,
                              "pcm_bits": 16 if storage == "pcm16" else None,
                              "quantization_error_bound_fs": (None if storage == "float32" else
                                  float((.5 + np.max(np.abs(piece)))/32768)),
                              "quantization_note": ("float32 cast; no PCM bound" if storage == "float32" else
                                  "legacy round(x*32767), decode /32768; bound excludes clipping"),
                              "input_clipped_samples": int(np.sum(np.abs(piece) > 1.))})
    with manifest.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in added)
    return len(added)


def _audible_span(x: np.ndarray, fs: int, rel_db: float = -25.) -> tuple[int, int]:
    """[first, last) sample where the 10 ms energy envelope is within `rel_db` of its peak."""
    w = max(1, round(.01*fs))
    env = np.convolve(x*x, np.ones(w)/w, mode="same")
    above = np.flatnonzero(env > env.max()*10**(rel_db/10)) if env.max() > 0 else np.array([0])
    return int(above[0]), int(above[-1]) + 1


def _event_window(x: np.ndarray, span: tuple[int, int], n: int, rng) -> tuple[np.ndarray, tuple[int, int]]:
    """An n-sample window holding the event (from its onset) at a random offset; and its span."""
    a, b = span
    length = min(b - a, n)
    offset = int(rng.integers(0, n - length + 1))
    seg = x[a:a + n - offset]
    out = np.zeros(n)
    out[offset:offset + seg.size] = seg
    return out, (offset, offset + length)


def _padded_window(x: np.ndarray, start: int, n: int) -> np.ndarray:
    """Crop in source coordinates, with silence outside the supplied recording."""
    out = np.zeros(n, dtype=float)
    lo, hi = max(0, start), min(len(x), start+n)
    if hi > lo:
        out[lo-start:hi-start] = x[lo:hi]
    return out


def _sliding_event_window(x, span, n, hop, rng):
    """Uniform phase, then uniform overlapping position on the deployed hop grid.

    Coordinates are samples; the event and window are half-open intervals. The
    same law applies to every class, without retrying ambiguous fragments. The
    support necessarily grows with event duration; identical *absolute* offsets
    across unequal durations would not match deployment. No class is an input.
    """
    a, b = span
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in (a, b, n, hop)) or not 0 <= a < b <= len(x) or not 1 <= hop <= n):
        raise ValueError('require integer samples, nonempty event span and 1 <= hop <= window')
    phase = int(rng.integers(0, hop))
    low, high = a-n+1, b-1
    first = low + (phase-low) % hop
    count = (high-first)//hop + 1
    start = first + hop*int(rng.integers(0, count))
    return _padded_window(x, start, n), (max(0, a-start), min(n, b-start)), start


def _crop(x: np.ndarray, n: int, rng) -> np.ndarray:
    """A random n-sample crop (tiled when shorter)."""
    if x.size <= n:
        return np.resize(x, n)
    start = int(rng.integers(0, x.size - n + 1))
    return x[start:start + n]


def _mix_into(background: np.ndarray, target: np.ndarray, span: tuple[int, int], snr_db: float) -> np.ndarray:
    """background (at its own level) + target scaled to `snr_db` over the event's audible span."""
    s = slice(*span)
    pt, pb = float(np.mean(target[s]**2)), float(np.mean(background[s]**2))
    if pt == 0:
        return background.copy()
    return background + (target if pb == 0 else target*np.sqrt(pb*10**(snr_db/10)/pt))


def _fit(X: np.ndarray, y: list[str], c: float):
    """Standardize + multinomial logistic regression. Returns (mean, scale, classifier)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.)
    clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit((X - scaler.mean_)/scale, y)
    return scaler.mean_, scale, clf


def train(manifest: str | Path, *, c: float = 1.0, background: str = "background", window_s: float = 2.0,
          snr_db: tuple[float, float] = (-5., 20.), copies: int = 3, background_windows: int = 4,
          seed: int = 0, mic_augment: float = 0.) -> dict:
    """Fit on split == "train", report on split == "val". Returns the JSON-ready model.

    A `background` class is required: trained on clean renders alone, a model reads every
    real event (always heard over room noise) as background. Each target clip yields
    `copies` windows, each mixed into a random background window FROM THE SAME SPLIT (no
    leakage) at a uniform random SNR in `snr_db` over the event's audible span. Each
    background clip yields `background_windows` random windows. Validation is built the
    same way, with its own seed.
    """
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as error:
        raise ImportError("training needs scikit-learn: pip install 'sensorsim-room[recognizer]'") from error
    manifest = Path(manifest)
    rows, fs = _load_manifest(manifest)
    cfg = feature_config(fs, window_s)
    bank, n = mel_filterbank(cfg), round(window_s*fs)
    data = {s: [r for r in rows if r["split"] == s] for s in ("train", "val")}
    if not any(r["label"] == background for r in data["train"]):
        raise ValueError(f"training needs a '{background}' class recorded in the room: "
                         "add it with `python -m sensorsim.recognizer add-background`")
    if len({r["label"] for r in data["train"]}) < 2:
        raise ValueError("training needs at least one target class besides background")
    warnings = ["validation windows are synthetic renders in the same fitted room as training; scores are optimistic"]
    X, y = {}, {}
    for k, split in enumerate(("train", "val")):
        rows_ = data[split]
        noise = [read_audio(manifest.parent/r["file"], fs) for r in rows_ if r["label"] == background]
        if not rows_ or not noise:
            if split == "val" and rows_:
                warnings.append("no validation background recording: validation unscored")
            continue
        rng = np.random.default_rng([seed, k])
        feats, labels = [], []

        def heard(window, ref):  # optionally through a random IoT microphone chain (sensorsim.iot_mic)
            if mic_augment and rng.random() < mic_augment:
                from sensorsim.iot_mic import random_chain
                return random_chain(rng).apply(window, fs, rng, ref_rms=ref)
            return window

        for x in noise:
            for _ in range(background_windows):
                crop = _crop(x, n, rng)
                feats.append(features(heard(crop, float(np.sqrt(np.mean(crop**2))) or 1e-4), cfg, bank))
                labels.append(background)
        for r in rows_:
            if r["label"] == background:
                continue
            x = read_audio(manifest.parent/r["file"], fs)
            span = _audible_span(x, fs)
            for _ in range(copies):
                w, wspan = _event_window(x, span, n, rng)
                bed = _crop(noise[int(rng.integers(len(noise)))], n, rng)
                mixed = _mix_into(bed, w, wspan, rng.uniform(*snr_db))
                feats.append(features(heard(mixed, float(np.sqrt(np.mean(bed**2))) or 1e-4), cfg, bank))
                labels.append(r["label"])
        X[split], y[split] = np.array(feats), labels
    mean, scale, clf = _fit(X["train"], y["train"], c)
    classes = [str(k) for k in clf.classes_]
    report = {"train_windows": {k: y["train"].count(k) for k in classes}, "warnings": warnings,
              "construction": {"window_s": window_s, "snr_db": list(snr_db), "snr_reference": "event audible span",
                               "mic_augment": mic_augment,
                               "copies": copies, "background_windows": background_windows, "background": background}}
    if "val" in X:
        missing = sorted(set(classes) - set(y["val"]))
        if missing:
            report["warnings"].append(f"classes absent from validation, so unscored: {missing}")
        predicted = clf.predict((X["val"] - mean)/scale)
        report["val_windows"] = {k: y["val"].count(k) for k in classes}
        report["val_accuracy"] = float(np.mean(predicted == np.array(y["val"])))
        report["val_per_class"] = {k: v for k, v in classification_report(
            y["val"], predicted, labels=classes, output_dict=True, zero_division=0).items() if k in classes}
        report["val_confusion"] = {"labels": classes,
                                   "matrix": confusion_matrix(y["val"], predicted, labels=classes).tolist()}
    return {"format": FORMAT, "features": cfg, "classes": classes,
            "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
            "training": {"manifest": manifest.name, "regularization_c": c, "seed": seed, **report}}


def predict(model: dict, path: str | Path, *, window_s: float | None = None, hop_s: float | None = None,
            min_prob: float = 0.) -> list[dict]:
    """Label each sliding window (default: the model's window, half-window hop).

    Below `min_prob` the label is "unknown". A file shorter than one window is padded.
    """
    if model.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} model (retrain with this version)")
    cfg = model["features"]
    bank = mel_filterbank(cfg)
    x = read_audio(path, cfg["fs"])
    window_s = window_s or cfg["window_s"]
    n, step = round(window_s*cfg["fs"]), round((hop_s or window_s/2)*cfg["fs"])
    if n <= 0 or step <= 0:
        raise ValueError("window and hop must be positive")
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    out = []
    for start in range(0, x.size - n + 1, step):
        p = predict_proba(model, x[start:start + n], bank)
        best = int(np.argmax(p))
        out.append({"file": str(path), "start_s": start/cfg["fs"], "end_s": (start + n)/cfg["fs"],
                    "label": model["classes"][best] if p[best] >= min_prob else UNKNOWN,
                    "probability": float(p[best]),
                    "probabilities": dict(zip(model["classes"], map(float, p)))})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("add-background", help="append raw room recordings as a background class")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--train", type=Path, nargs="+", required=True, help="recordings for the train split")
    b.add_argument("--val", type=Path, nargs="*", default=[], help="different recordings for validation")
    b.add_argument("--label", default="background")
    t = sub.add_parser("train", help="fit on a dataset manifest.jsonl")
    t.add_argument("--manifest", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--window", type=float, default=2.0, help="window length in seconds (stored in the model)")
    t.add_argument("--c", type=float, default=1.0, help="inverse L2 regularization strength")
    p = sub.add_parser("predict", help="label audio files in sliding windows")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--hop", type=float, help="hop in seconds (default: half the model's window)")
    p.add_argument("--min-prob", type=float, default=0., help="below this, report 'unknown'")
    args = parser.parse_args(argv)
    if args.command == "add-background":
        added = add_background(args.manifest, args.train, args.val, label=args.label)
        print(f"appended {added} '{args.label}' clips to {args.manifest}")
    elif args.command == "train":
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        model = train(args.manifest, c=args.c, window_s=args.window)
        args.out.write_text(json.dumps(model, indent=1) + "\n")
        print(json.dumps(model["training"], indent=1))
    else:
        model = json.loads(args.model.read_text())
        for path in args.files:
            for row in predict(model, path, hop_s=args.hop, min_prob=args.min_prob):
                print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
