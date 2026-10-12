# SPDX-License-Identifier: Apache-2.0
"""Combined-lane regression: target-only confusers also need a cue identity hook."""
import numpy as np
import pytest

from sensorsim import nn, training_negatives as tn
from tests.validation.t3_label_policy.check_handoff import corpus_and_config


@pytest.mark.parametrize('policy,target_probability,blocked', [
    ('legacy', 1., False),
    ('matched/1', 0., False),
    ('matched/1', 1., True),
])
def test_cue_pairing_refuses_unbound_target_confusers(factory, monkeypatch,
                                                    policy, target_probability, blocked):
    fixture = factory(stage='device_recorded')
    corpus, cfg = corpus_and_config()
    corpus.negative_pool = fixture.build()
    corpus.pool_purpose = 'train'
    for row in corpus.rows:
        if row['label'] != 'background':
            row['room_draw_id'] = 'room-A'
    cfg.update(negative_context_policy=policy, p_confuser_given_target=target_probability,
               p_confuser_given_background=0., partial_windows=False, cue_masks=False,
               p_specaugment=0., batch=1)
    calls = []
    original = tn.draw_training_context

    def observe(*args, **kwargs):
        calls.append(kwargs['draw'].provenance['injected'])
        print('matched-context-injected:', calls[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(tn, 'draw_training_context', observe)
    batch = nn.Batches(corpus, cfg, 1, 37)
    if blocked:
        with pytest.raises(ValueError, match='confuser identity hook'):
            batch[0]
        assert calls == []
    else:
        x, y, _, metadata = batch[0]
        assert np.isfinite(x).all() and y.shape == (1, 2)
        assert len(metadata['cue_records']) == 1
        assert metadata['confuser_windows'] == 0
        assert calls == ([False, False] if policy == 'matched/1' else [])
