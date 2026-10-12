# SPDX-License-Identifier: Apache-2.0
"""Cross-lane dispatch controls for the parent-level integration (synthetic proxy)."""
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import event_levels as el, nn, training_dynamics, training_negatives


@pytest.mark.parametrize("mode", [el.LEGACY, el.RELATIVE, el.CALIBRATED])
@pytest.mark.parametrize("cue_enabled", [False, True])
def test_parent_guard_precedes_matched_negative_dispatch(mode, cue_enabled, monkeypatch):
    """Legacy reaches the main dispatch; unsupported parents cannot bypass the guard."""
    cfg = dict(nn.DEFAULTS, event_level_mode=mode, cue_validity=cue_enabled)
    corpus = SimpleNamespace(cue_annotations={})
    rng = np.random.default_rng(11)
    audio, label = np.array([.1]), np.array([1.])
    reached = []
    monkeypatch.setattr(training_dynamics, "enabled", lambda c: False)
    monkeypatch.setattr(training_negatives, "enabled", lambda c: True)

    def draw(c, r, config, mixer, *, context):
        reached.append((c is corpus, r is rng, config is cfg, mixer is nn.mix_device_audio, context))
        return audio, label, {"route": "matched"}

    monkeypatch.setattr(training_dynamics, "draw_example", draw)
    if cue_enabled:
        from sensorsim import training_policy
        monkeypatch.setattr(training_policy, "validate_config", lambda c: reached.append("cue"))
    if mode == el.LEGACY:
        result = nn.make_example(corpus, rng, cfg)
        assert result[0] is audio and result[1] is label
        assert reached == (["cue"] if cue_enabled else []) + [(True, True, True, True, False)]
    else:
        with pytest.raises(ValueError, match="not integrated"):
            nn.make_example(corpus, rng, cfg)
        assert reached == []
