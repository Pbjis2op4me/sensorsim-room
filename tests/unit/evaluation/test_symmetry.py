from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import edge
from sensorsim.evaluation.realism import provenance_shares, score_neutral_probes


def test_provenance_shares_keep_source_route_and_chain_separate():
    rows = [
        {"label": "knock", "dry_source": {"provenance": {"route": "room take"}},
         "receiver_augmentation": {"method": "fixed-room"}},
        {"label": "knock", "source_kind": "measured", "provenance": {"route": "held-out"}},
    ]
    result = provenance_shares(rows)["knock"]
    assert result["source"]["rendered"]["count"] == 1
    assert result["source"]["measured"]["count"] == 1
    assert result["route"]["room take"]["share"] == .5
    assert result["chain"]["fixed-room"]["count"] == 1


def test_neutral_probes_are_scored_through_both_paths(monkeypatch):
    calls = []
    def detect(model, audio):
        calls.append(len(audio))
        return []
    monkeypatch.setattr(edge, "detect", detect)
    model = SimpleNamespace(classes=["knock"], meta={"features": {"fs": 16000}, "detector": {"on": .5}})
    probes = {"silence": np.zeros(16000, np.float32)}
    result = score_neutral_probes(model, probes,
        positive_path=lambda audio: audio + .1,
        negative_path=lambda audio: audio - .1,
        data_kind="proxy")
    assert calls == [16000, 16000]
    assert set(result["probes"]["silence"]) == {"positive", "negative"}
    assert result["probes"]["silence"]["positive"]["peak_abs"] == pytest.approx(.1)
