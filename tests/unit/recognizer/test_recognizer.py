"""Baseline recognizer: exact JSON inference, gain invariance, and the deployment failure it must not repeat."""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim import recognizer

pytest.importorskip("sklearn")
FS = 8000


def _tone_event(f, k, seconds=.6):
    t = np.arange(int(seconds*FS))/FS
    return np.exp(-4*t)*np.sin(2*np.pi*f*(1 + .004*k)*t)


def _room_noise(seed, seconds):
    r = np.random.default_rng(seed); n = int(seconds*FS)
    w = np.fft.irfft(np.fft.rfft(r.normal(size=n)) / np.sqrt(np.maximum(np.fft.rfftfreq(n, 1/FS), 20)), n)
    return .05*w/np.std(w) + .01*np.sin(2*np.pi*50*np.arange(n)/FS)


def _write(path, x):
    wavfile.write(path, FS, np.clip(np.round(x*32767), -32768, 32767).astype(np.int16))


def _dataset(tmp_path, level=.003):
    """Clean target renders at a RELATIVE level far below the room noise (as the dataset stores them),
    plus two background recordings at their real level, one per split."""
    rows = []
    for label, f in (("low", 500), ("high", 2200)):
        for k in range(6):
            name = f"{label}_{k}.wav"
            _write(tmp_path/name, level*_tone_event(f, k))
            rows.append({"file": name, "label": label, "split": "val" if k >= 4 else "train", "fs": FS,
                         "dry_source": {"source_group": f"{label}-{k}"}})
    manifest = tmp_path/"manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name, seed in (("room_a", 1), ("room_b", 2)):
        _write(tmp_path/f"{name}.wav", _room_noise(seed, 8.))
    recognizer.add_background(manifest, [tmp_path/"room_a.wav"], [tmp_path/"room_b.wav"])
    return manifest


@pytest.mark.parametrize("n_classes", [2, 3])
def test_json_inference_matches_sklearn(n_classes):
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(n_classes), 60 // n_classes)
    X = rng.normal(size=(labels.size, 7)) + labels[:, None]
    mean, scale, clf = recognizer._fit(X, [f"c{i}" for i in labels], 1.0)
    model = json.loads(json.dumps({"classes": [str(c) for c in clf.classes_],
                                   "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
                                   "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}))
    for row in X[::7]:
        np.testing.assert_allclose(recognizer._proba(model, row), clf.predict_proba(((row - mean)/scale)[None])[0],
                                   rtol=0, atol=1e-12)


def test_features_ignore_overall_gain():
    cfg = recognizer.feature_config(FS)
    x = _room_noise(3, 2.) + _tone_event(900, 0, 2.)
    np.testing.assert_allclose(recognizer.features(x, cfg), recognizer.features(20*x, cfg), atol=1e-6)


def test_quiet_relative_renders_still_recognise_events_at_real_level(tmp_path):
    """Release-preparation failure: renders stored ~22 dB below real recordings taught 'quiet = target',
    so every real event, heard at real level over real room noise, read as background."""
    model = recognizer.train(_dataset(tmp_path), window_s=1.)
    assert model["features"]["window_s"] == 1.
    for label, f in (("low", 500), ("high", 2200)):
        heard = _room_noise(9, 1.)
        event = np.zeros(heard.size)
        e = _tone_event(f, 9)
        event[2000:2000 + e.size] = e[:heard.size - 2000]
        x = recognizer._mix_into(heard, event, recognizer._audible_span(event, FS), 6.)  # real level, 6 dB over room
        probs = recognizer.predict_proba(model, x)
        assert model["classes"][int(np.argmax(probs))] == label, dict(zip(model["classes"], probs))
    assert model["classes"][int(np.argmax(recognizer.predict_proba(model, _room_noise(10, 1.))))] == "background"


def test_predict_uses_the_model_window_and_finds_events_in_a_stream(tmp_path):
    model = recognizer.train(_dataset(tmp_path), window_s=1.)
    stream = _room_noise(11, 6.)
    for at, f in ((1.0, 500), (4.0, 2200)):
        e, s = _tone_event(f, 11), int(at*FS)
        stream[s:s + e.size] = recognizer._mix_into(stream[s:s + e.size], e, recognizer._audible_span(e, FS), 8.)
    _write(tmp_path/"stream.wav", stream)
    rows = recognizer.predict(model, tmp_path/"stream.wav", min_prob=.6)
    assert {round(r["end_s"] - r["start_s"], 6) for r in rows} == {1.}      # the model's window ...
    assert [r["start_s"] for r in rows[:3]] == [0., .5, 1.]                  # ... and a half-window hop
    near = lambda t: {r["label"] for r in rows if r["start_s"] <= t <= r["end_s"]}
    assert "low" in near(1.1) and "high" in near(4.1)
    assert all(r["label"] in ("background", recognizer.UNKNOWN) for r in rows if r["end_s"] < .9)


def test_training_requires_a_background_class(tmp_path):
    _write(tmp_path/"a.wav", .1*_tone_event(500, 0))
    _write(tmp_path/"b.wav", .1*_tone_event(2200, 0))
    manifest = tmp_path/"manifest.jsonl"
    manifest.write_text("".join(json.dumps({"file": n, "label": n[0], "split": "train", "fs": FS}) + "\n"
                                for n in ("a.wav", "b.wav")))
    with pytest.raises(ValueError, match="background"):
        recognizer.train(manifest)


def test_add_background_splits_by_recording(tmp_path):
    manifest = _dataset(tmp_path)
    bg = [r for r in map(json.loads, manifest.read_text().splitlines()) if r["label"] == "background"]
    assert {r["split"] for r in bg if r["source_group"] == "background:room_a.wav"} == {"train"}
    assert {r["split"] for r in bg if r["source_group"] == "background:room_b.wav"} == {"val"}
    with pytest.raises(ValueError, match="both train and val"):
        recognizer.add_background(manifest, [tmp_path/"room_a.wav"], [tmp_path/"room_a.wav"])
