"""Edge runtime (numpy only): streaming equals offline detection, per-class tracking, calibration that only
raises thresholds, gain-normalized features, WAV input, and a standalone file that runs on Python 3.9."""
import ast
import json
import sys
import wave

import numpy as np
import pytest

from sensorsim import edge

FS = 16000


def tiny_model(path, classes=("a", "b"), *, seed=0, width=4, on=.5):
    """A small random network in the export format (no PyTorch needed)."""
    rng = np.random.default_rng(seed)
    n = lambda *shape: rng.normal(scale=.3, size=shape)
    arrays = {"conv0.w": n(width, 1, 5, 5), "conv0.b": n(width), "b0.dw.w": n(width, 3, 3), "b0.dw.b": n(width),
              "b0.pw.w": n(width, width), "b0.pw.b": n(width), "fc.w": n(len(classes), 2*width), "fc.b": n(len(classes))}
    layers = [{"type": "conv", "name": "conv0", "stride": [2, 2], "pad": [2, 2], "act": "relu"},
              {"type": "dw", "name": "b0.dw", "stride": [2, 1], "pad": [1, 1], "act": "relu"},
              {"type": "pw", "name": "b0.pw", "act": "relu"}, {"type": "pool"}, {"type": "fc", "name": "fc"}]
    meta = {"format": edge.FORMAT, "classes": list(classes), "layers": layers,
            "features": {"fs": FS, "window_s": .5, "n_fft": 512, "hop": 160, "n_mels": 16, "fmin_hz": 60., "fmax_hz": 7600.},
            "detector": {"hop_s": .125, "off_ratio": .6, "smooth_hops": 3, "min_hops": 2, "on": {c: on for c in classes}}}
    arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
    np.savez(path, **arrays)
    return edge.Model(str(path))


def _bursty_audio(seconds, seed=0):
    r = np.random.default_rng(seed)
    t = np.arange(int(seconds*FS))/FS
    gate = (np.sin(2*np.pi*.7*t) > .3) | (np.sin(2*np.pi*.23*t + 1) > .8)
    return (.01*r.normal(size=t.size) + .2*gate*np.sin(2*np.pi*(600 + 400*np.sin(2*np.pi*.1*t))*t)).astype(np.float32)


def _run_tracker(classes, cfg, scores):
    tracker, events = edge.Tracker(classes, cfg, window_s=1.0), []
    for k, s in enumerate(scores):
        events += [e for e in tracker.step(np.asarray(s), .25*k + 1.0) if e["event"] == "end"]
    return [e for e in events + tracker.flush() if e["confirmed"]]


def test_a_one_window_transient_survives_only_with_its_own_short_smoothing():
    scores = np.zeros((16, 2))
    scores[3:9, 0] = .9                     # a long sound, visible for six hops
    scores[11, 1] = .95                     # a knock, visible in one window
    shared = {"on": {"long": .5, "short": .5}, "off_ratio": .6, "smooth_hops": 3, "min_hops": 2}
    per_class = {**shared, "smooth_hops": {"long": 3, "short": 1}, "min_hops": {"long": 2, "short": 1}}
    assert [e["label"] for e in _run_tracker(["long", "short"], shared, scores)] == ["long"]
    found = _run_tracker(["long", "short"], per_class, scores)
    assert sorted(e["label"] for e in found) == ["long", "short"]
    short = next(e for e in found if e["label"] == "short")
    assert (short["start_s"], short["end_s"]) == (2.75, 3.75)


def test_streaming_detection_equals_offline_detection(tmp_path):
    model = tiny_model(tmp_path/"m.npz")
    pcm = np.round(_bursty_audio(30.)*32767).astype(np.int16)          # what arecord delivers
    audio = pcm.astype(np.float32)/32768
    scores = edge.window_scores(model, audio, .125)
    on = {c: float(np.quantile(scores[:, i], .7)) for i, c in enumerate(model.classes)}   # make events happen
    offline = edge.detect(model, audio, thresholds={"on": on})
    assert offline
    detector, streamed, rng, i = edge.Detector(model, thresholds={"on": on}), [], np.random.default_rng(1), 0
    while i < pcm.size:                                   # irregular chunk sizes, int16 like arecord
        step = int(rng.integers(100, 5000))
        streamed += [e for e in detector.push(pcm[i:i + step]) if e["event"] == "end" and e["confirmed"]]
        i += step
    streamed += [e for e in detector.flush() if e["confirmed"]]
    assert [(e["label"], e["start_s"], e["end_s"]) for e in streamed] == [(e["label"], e["start_s"], e["end_s"]) for e in offline]


def test_calibration_only_raises_thresholds_and_meets_its_budget(tmp_path):
    background = _bursty_audio(240., seed=3)
    scores = edge.window_scores(tiny_model(tmp_path/"raw.npz"), background, .125)
    defaults = {c: round(float(np.quantile(scores[:, i], .75)), 3) for i, c in enumerate(("a", "b"))}
    edge.save_detector(str(tmp_path/"raw.npz"), str(tmp_path/"m.npz"), on=defaults, off_ratio=.95)   # fires often
    model = edge.Model(str(tmp_path/"m.npz"))
    before = edge.detect(model, background)
    result = edge.calibrate(model, background, fa_per_hour=16.)
    assert result["allowed_events"] == 1 and len(before) > 2
    assert all(result["on"][c] >= defaults[c] for c in model.classes) and result["on"] != defaults
    after = edge.detect(model, background, thresholds={"on": result["on"]})
    assert all(sum(e["label"] == c for e in after) <= 1 for c in model.classes)
    edge.save_detector(str(tmp_path/"m.npz"), str(tmp_path/"cal.npz"), on=result["on"])
    assert edge.Model(str(tmp_path/"cal.npz")).meta["detector"]["on"] == result["on"]


def test_features_ignore_a_dc_offset_from_a_raw_mems_capsule():
    features = edge.Features({"fs": FS, "window_s": .5, "n_fft": 512, "hop": 160, "n_mels": 16, "fmin_hz": 60., "fmax_hz": 7600.})
    quiet = (0.001*np.random.default_rng(2).standard_normal(FS//2)).astype(np.float32)        # a -60 dBFS room
    for offset in (0.01, 0.05, 0.2):
        np.testing.assert_allclose(features(quiet + offset), features(quiet), atol=2e-3)


def test_features_ignore_overall_gain():
    features = edge.Features({"fs": FS, "window_s": .5, "n_fft": 512, "hop": 160, "n_mels": 16, "fmin_hz": 60., "fmax_hz": 7600.})
    x = _bursty_audio(.5)
    np.testing.assert_allclose(features(x), features(20*x), atol=1e-4)


def _write_wav(path, x, rate, width=2, channels=1):
    full = 2**(8*width - 1) - 1
    pcm = np.round(np.repeat(x[:, None], channels, axis=1)*full).astype("<i4")
    raw = pcm.astype("<i2").tobytes() if width == 2 else b"".join(int(v).to_bytes(3, "little", signed=True) for v in pcm.ravel())
    with wave.open(str(path), "wb") as w:
        w.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        w.writeframes(raw)


def test_wav_input_is_resampled_and_downmixed(tmp_path):
    t = np.arange(48000)/48000
    tone = .5*np.sin(2*np.pi*1000*t)
    _write_wav(tmp_path/"stereo48k.wav", tone, 48000, channels=2)
    x = edge.read_wav(str(tmp_path/"stereo48k.wav"), FS)
    assert x.size == 16000
    spectrum = np.abs(np.fft.rfft(x[1000:-1000]*np.hanning(14000)))
    assert abs(np.fft.rfftfreq(14000, 1/FS)[np.argmax(spectrum)] - 1000) < 2
    assert abs(np.std(x[1000:-1000]) - .5/np.sqrt(2)) < .01
    _write_wav(tmp_path/"mono24.wav", tone[::3], 16000, width=3)
    np.testing.assert_allclose(edge.read_wav(str(tmp_path/"mono24.wav"), FS), tone[::3], atol=1e-6)
    _write_wav(tmp_path/"cd.wav", tone[:44100], 44100)
    with pytest.raises(ValueError, match="integer ratio"):
        edge.read_wav(str(tmp_path/"cd.wav"), FS)


def test_other_files_are_refused(tmp_path):
    np.savez(tmp_path/"x.npz", meta=np.frombuffer(json.dumps({"format": "something-else"}).encode(), np.uint8))
    with pytest.raises(ValueError, match="not a"):
        edge.Model(str(tmp_path/"x.npz"))


def test_the_runtime_is_one_file_of_numpy_and_python_3_9():
    source = open(edge.__file__, encoding="utf-8").read()
    tree = ast.parse(source, feature_version=(3, 9))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imported - set(sys.stdlib_module_names) == {"numpy"}


@pytest.mark.parametrize("rate", [48000, 32000, 8000])
def test_resampling_matches_the_training_pipeline(rate):
    """Training audio is resampled with scipy's resample_poly; the device must apply the same filter,
    which keeps a 7 kHz tone (inside the mel range) within 0.35 dB after 48 -> 16 kHz (the earlier
    design lost 1.5 dB there)."""
    from scipy.signal import resample_poly
    g = np.gcd(rate, FS)
    x = np.random.default_rng(4).standard_normal(rate//2).astype(np.float32)
    np.testing.assert_allclose(edge.resample(x, FS//g, rate//g), resample_poly(x.astype(float), FS//g, rate//g), atol=2e-6)
    if rate == 48000:
        tone = np.sin(2*np.pi*7000*np.arange(rate)/rate).astype(np.float32)
        y = edge.resample(tone, 1, 3)[2000:-2000]
        assert abs(20*np.log10(np.sqrt(2)*np.std(y))) < 0.35
