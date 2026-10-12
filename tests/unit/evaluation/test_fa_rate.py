from types import SimpleNamespace
import wave

import numpy as np
import pytest

from sensorsim import edge
from sensorsim.evaluation import fa_rate


class _ThresholdModel:
    classes = ["knock"]
    features = SimpleNamespace(n=4)
    meta = {
        "features": {"fs": 4},
        "detector": {
            "hop_s": 0.25,
            "on": {"knock": 0.5},
            "off_ratio": 0.6,
            "smooth_hops": 1,
            "min_hops": 2,
        },
    }

    def scores(self, windows):
        windows = np.asarray(windows)
        if windows.ndim == 1:
            windows = windows[None, :]
        return np.max(np.abs(windows), axis=1, keepdims=True)


def test_cowan_zero_one_two_event_upper_bounds():
    bounds = [fa_rate.cowan_poisson_upper(k, 3600, 0.95) for k in range(3)]
    assert bounds == pytest.approx([3.00, 4.74, 6.30], abs=0.01)


def test_clopper_pearson_edges_and_exposure_labels():
    assert fa_rate.clopper_pearson(0, 10)[0] == 0
    assert fa_rate.clopper_pearson(10, 10)[1] == 1
    assert fa_rate.exposure_label("model-based", 6 * 3600) == "model-based"
    assert fa_rate.exposure_label("measured", 1200) == "measured-smoke-test-not-FA/hour"
    assert fa_rate.exposure_label("measured", 1800) == "measured"


def test_audio_scoring_uses_edge_streaming_detector():
    model = _ThresholdModel()
    audio = np.asarray([0, 0, 0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    events = fa_rate.score_audio(model, audio, chunk_s=0.25)
    expected = edge.detect(model, audio)
    assert len(events) == len(expected) == 1
    assert events[0]["label"] == "knock"
    assert events[0]["confirmed"] is True


def test_challenge_scoring_attributes_confirmed_events_and_keeps_proxy_label(tmp_path):
    audio = np.zeros(16, dtype="<i2")
    audio[4:8] = 30000
    path = tmp_path / "impact.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(4)
        wav.writeframes(audio.tobytes())
    trials = fa_rate.score_challenge_records(
        _ThresholdModel(),
        [{
            "family": "impact",
            "path": str(path),
            "context": "isolated",
            "source_kind": "proxy",
            "events": [{"id": "source-1", "start_s": 1.0, "end_s": 1.25}],
        }],
    )
    assert len(trials) == 1
    assert trials[0]["confirmed_outputs"] == 1
    assert trials[0]["evidence_label"] == "model-based"
