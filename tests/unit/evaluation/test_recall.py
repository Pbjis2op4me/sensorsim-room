from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import edge
from sensorsim.evaluation.recall import clopper_pearson, evaluate_recall


def test_clopper_pearson_matches_registered_all_detected_bounds():
    lower, upper = clopper_pearson(5, 5)
    assert lower == pytest.approx(.478, abs=.002)
    assert upper == 1.0
    assert clopper_pearson(0, 0) == (None, None)


def test_recall_calls_exact_edge_detector_without_threshold_overrides(monkeypatch):
    calls = []
    def detect(model, audio):
        calls.append((model, len(audio)))
        return [{"label": "knock"}] if audio[0] > 0 else []
    monkeypatch.setattr(edge, "detect", detect)
    model = SimpleNamespace(classes=["knock"], meta={"features": {"fs": 16000}, "detector": {"on": {"knock": .7}}})
    audio = np.zeros(16000, np.float32)
    hit = audio.copy(); hit[0] = 1
    result = evaluate_recall(model, [
        {"label": "knock", "audio": hit, "file": "knock_1.wav", "group": "take-1"},
        {"label": "knock", "audio": audio, "file": "knock_2.wav", "group": "take-2"},
    ], data_kind="proxy")
    assert len(calls) == 2 and all(call[0] is model for call in calls)
    assert result["per_class"]["knock"]["hits"] == 1
    assert result["per_class"]["knock"]["takes"] == 2
    assert result["per_class"]["knock"]["data_kind"] == "proxy"
