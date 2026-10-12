# SPDX-License-Identifier: Apache-2.0
"""Development controls, with hand-enumerated placement and loss oracles (proxy)."""
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import nn, recognizer


class Draw:
    def __init__(self, phase, index):
        self.values = iter((phase, index))
        self.calls = []

    def integers(self, low, high):
        value = next(self.values)
        assert low <= value < high
        self.calls.append((low, high))
        return value


def test_grid_enumeration_includes_before_onset_middle_and_after_offset():
    # n=8, hop=3, event [5,16). Independent exhaustive intersection calculation.
    x = np.arange(24, dtype=float)
    observed = set()
    for phase in range(3):
        starts = [s for s in range(-20, 30) if s % 3 == phase and s < 16 and s+8 > 5]
        for index, start in enumerate(starts):
            rng = Draw(phase, index)
            win, span, actual = recognizer._sliding_event_window(x, (5, 16), 8, 3, rng)
            assert rng.calls == [(0, 3), (0, len(starts))]
            assert actual == start
            expected = [x[s] if 0 <= s < len(x) else 0 for s in range(start, start+8)]
            np.testing.assert_array_equal(win, expected)
            assert span == (max(0, 5-start), min(8, 16-start))
            observed.add(start)
    assert observed == set(range(-2, 16))


def test_long_event_known_bad_has_fixed_onset_new_path_has_all_phases():
    x = np.ones(48)
    old = [recognizer._event_window(x, (0, 48), 16, np.random.default_rng(s))[1][0]
           for s in range(60)]
    assert set(old) == {0}  # designated bad control reaches the actual legacy function
    starts = [recognizer._sliding_event_window(x, (0, 48), 16, 4, np.random.default_rng(s))[2]
              for s in range(200)]
    assert min(starts) < 0 and max(starts) > 32 and any(0 < s < 32 for s in starts)
    assert {s % 4 for s in starts} == {0, 1, 2, 3}


@pytest.mark.parametrize('span,n,hop', [((3, 3), 8, 2), ((-1, 4), 8, 2),
                                      ((0, 50), 8, 2), ((0, 4), 0, 2), ((0, 4), 8, 9)])
def test_invalid_grid_fails_closed(span, n, hop):
    with pytest.raises(ValueError):
        recognizer._sliding_event_window(np.ones(24), span, n, hop, np.random.default_rng(1))


def test_semantic_cue_loss_is_unknown_even_when_energy_is_retained():
    from sensorsim.window_policy import supervision
    fs = 1000
    x = np.sin(np.arange(2000))
    # Equal-energy body; only the first 120 ms is the declared chime cue.
    good = supervision('doorbell', x, (0, 2000), 0, 1000, fs)
    bad = supervision('doorbell', x, (0, 2000), 700, 1000, fs)
    assert good['state'] == 'positive'
    assert bad['retained_fraction'] > .99 and bad['state'] == 'unknown'
    assert supervision('doorbell', x, (0, 2000), -999, 1000, fs)['state'] == 'unknown'


def test_kettle_requires_declared_boil_stage_and_never_labels_tail_absent():
    from sensorsim.window_policy import supervision
    x = np.sin(np.arange(4000))
    assert supervision('kettle', x, (0, 4000), 1500, 1000, 1000)['state'] == 'unknown'
    cue = {'units': 'samples', 'boil': [[1000, 3000]]}
    assert supervision('kettle', x, (0, 4000), 1500, 1000, 1000, cue)['state'] == 'positive'
    assert supervision('kettle', x, (0, 4000), 3500, 1000, 1000, cue)['state'] == 'unknown'
    with pytest.raises(ValueError, match='units'):
        supervision('kettle', x, (0, 4000), 1500, 1000, 1000, dict(cue, units='seconds'))


@pytest.mark.parametrize('x', [np.zeros(100), np.ones(100), np.array([]), np.full(100, np.nan)])
def test_degenerate_target_is_qa_error_not_negative(x):
    from sensorsim.window_policy import supervision
    with pytest.raises(ValueError):
        supervision('knock', x, (0, len(x)), 0, 50, 1000)


def test_masked_bce_has_independent_zero_gradient_and_known_label_oracle():
    from sensorsim.window_policy import masked_bce
    torch = nn._torch()
    z = torch.tensor([[0., 8., 0.]], requires_grad=True)
    y = torch.tensor([[1., -1., 0.]])
    loss = masked_bce(z, y)
    assert float(loss.detach()) == pytest.approx(np.log(2))
    loss.backward()
    np.testing.assert_allclose(z.grad.numpy(), [[-.25, 0., .25]])
    # An ordinary BCE silently treats -1 as a numeric target: known-bad mechanism.
    bad = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    assert float(bad.detach()) != pytest.approx(float(loss.detach()))
    unknown = torch.ones((2, 3), requires_grad=True)
    masked_bce(unknown, -torch.ones((2, 3))).backward()
    assert torch.count_nonzero(unknown.grad) == 0
    with pytest.raises(ValueError):
        masked_bce(z, torch.tensor([[1., float('nan'), 0.]]))


def test_new_policy_without_resampling_and_confusers_still_reached(monkeypatch):
    fs = 16000
    source = np.sin(2*np.pi*900*np.arange(3*fs)/fs)
    corpus = SimpleNamespace(classes=['doorbell'], targets={'doorbell': [source]},
        spans={'doorbell': [(0, len(source))]}, background=[np.ones(4*fs)*1e-4])
    cfg = dict(nn.DEFAULTS, window_policy='deployment/1', p_background_only=0., p_specaugment=0., p_freq_mixstyle=0.)
    seen = []
    original = nn.mix_device_audio
    def observe(*args, **kwargs):
        seen.append(args[1].copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(nn, 'mix_device_audio', observe)
    labels = []
    for seed in range(20):
        _, y, ctx = nn.make_example(corpus, np.random.default_rng(seed), cfg, return_context=True)
        labels.append(y[0])
        assert ctx['attempts'] == 1 and ctx['rejections'] == 0
        assert ctx['window_policy'] == 'deployment/1'
        assert ctx['loss_mask'] == [float(y[0] >= 0)]
    assert -1. in labels and 1. in labels
    assert any(np.any(event[:fs]) for event in seen), 'mid-event filter history was discarded'
    from sensorsim.negatives import integration
    calls = []
    def confuser(*args):
        calls.append(True)
        return np.ones(fs)*.01, (0, fs)
    monkeypatch.setattr(integration, 'draw_confuser', confuser)
    _, absent, ctx = nn.make_example(corpus, np.random.default_rng(4), dict(cfg, p_background_only=1.), return_context=True)
    assert calls and absent.tolist() == [0.] and ctx['loss_mask'] == [1.]


def test_unknowns_do_not_change_validation_ap_into_numeric_negative_targets():
    assert nn._average_precision(np.array([.9, .8, .1]), np.array([-1., 1., 0.])) == 1.


def test_new_policy_is_opt_in_and_rejects_unqualified_feature_augmentation():
    from sensorsim.window_policy import validate_config
    assert not validate_config(nn.DEFAULTS)
    with pytest.raises(ValueError, match='augmentation'):
        validate_config(dict(nn.DEFAULTS, window_policy='deployment/1'))
    assert validate_config(dict(nn.DEFAULTS, window_policy='deployment/1', p_specaugment=0., p_freq_mixstyle=0.))


def test_real_training_masks_unknowns_and_exports_state_counts(tmp_path):
    import json
    from scipy.io import wavfile
    from sensorsim import edge
    fs = 16000
    rows = []
    for split, seed in [('train', 72), ('val', 89)]:
        rng = np.random.default_rng(seed)
        for label in ('doorbell', 'kettle', 'knock', 'background'):
            x = rng.normal(size=3*fs)
            x *= .0001 if label == 'background' else .05
            name = f'{split}-{label}.wav'
            wavfile.write(tmp_path/name, fs, x.astype(np.float32))
            rows.append(dict(file=name, split=split, label=label, fs=fs, source_group=f'{split}-{label}',
                window_cues={'units': 'samples', 'boil': [[fs//2, 2*fs]]} if label == 'kettle' else None))
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cfg = dict(window_policy='deployment/1', p_specaugment=0., p_freq_mixstyle=0.,
               steps=3, batch=24, width=4, blocks=1, workers=0, threads=1, val_windows=64)
    report = nn.train(manifest, tmp_path/'model.npz', **cfg)
    states = report['window_supervision']['states']
    assert sum(r['unknown'] for r in states.values()) > 0
    assert all(sum(r.values()) == 72 for r in states.values())
    assert all(r['positive'] > 0 for r in states.values())
    assert report['negatives']['sampling']['confuser_windows'] > 0
    model = edge.Model(str(tmp_path/'model.npz'))
    assert model.meta['training']['window_supervision']['states'] == states
    assert model.meta['training']['config']['window_policy'] == 'deployment/1'


def test_audibility_is_required_at_the_cue_not_the_loud_tail(monkeypatch):
    from sensorsim import window_policy
    fs = 16000
    source = np.sin(2*np.pi*900*np.arange(3*fs)/fs)
    corpus = SimpleNamespace(classes=['doorbell'], targets={'doorbell': [source]},
        spans={'doorbell': [(0, len(source))]}, background=[np.ones(4*fs)*1e-4])
    cfg = dict(nn.DEFAULTS, window_policy='deployment/1', p_background_only=0., p_specaugment=0., p_freq_mixstyle=0.)
    monkeypatch.setattr(recognizer, '_sliding_event_window', lambda *a: (source[:fs], (0, fs), 0))
    calls = []
    def mix(bed, event, span, rng, cfg, erase=False):
        calls.append(erase)
        e = event.copy()
        if erase:
            e[fs:fs+round(.120*fs)] = 0
        return bed+e*.01
    _, good, a = window_policy.draw_example(corpus, np.random.default_rng(3), cfg, mix)
    _, bad, b = window_policy.draw_example(corpus, np.random.default_rng(3), cfg,
        lambda *args: mix(*args, erase=True))
    assert calls == [False, True] and good.tolist() == [1.] and bad.tolist() == [-1.]
    assert a['audible_frames'] >= 3 and b['audible_frames'] < 3


def test_full_take_gain_reaches_real_mixer_with_history_and_detects_stale_compensation(monkeypatch):
    from sensorsim import iot_mic, event_level
    fs = 16000
    source = np.sin(2*np.pi*1000*np.arange(3*fs)/fs)*np.linspace(.1, 1., 3*fs)
    corpus = SimpleNamespace(classes=['doorbell'], targets={'doorbell': [source]},
        spans={'doorbell': [(0, len(source))]}, background=[np.ones(4*fs)*1e-4])
    cfg = dict(nn.DEFAULTS, window_policy='deployment/1', p_background_only=0., p_specaugment=0., p_freq_mixstyle=0.)
    w = 160
    cumulative = np.r_[0., np.cumsum(source**2)]
    i = np.arange(len(source))
    env = (cumulative[np.minimum(i+80, len(source))]-cumulative[np.maximum(i-80, 0)])/w
    whole = np.mean(source[env > env.max()*10**(-2.5)]**2)
    captured = []
    def capture(self, x, fs, rng, **kw):
        captured.append(x.copy())
        return x
    monkeypatch.setattr(iot_mic.MicChain, 'apply', capture)
    observed = []
    original = nn.mix_device_audio
    def mix(bg, event, span, rng, settings):
        observed.append(event.copy())
        return original(bg, event, span, rng, settings)
    monkeypatch.setattr(nn, 'mix_device_audio', mix)
    def error():
        captured.clear();observed.clear()
        _, _, ctx = nn.make_example(corpus, np.random.default_rng(8), cfg, return_context=True)
        actual = np.dot(observed[0], captured[0])/np.dot(observed[0], observed[0])
        expected = 10**((ctx['requested_level_db']-94)/20)/np.sqrt(2*whole)
        return abs(20*np.log10(actual/expected))
    assert error() < 1e-5
    monkeypatch.setattr(event_level, 'level_power', lambda *a, **k: 1.)
    assert error() > .1


def test_missing_kettle_annotation_cannot_silently_train_an_all_unknown_class():
    from sensorsim.window_policy import validate_corpus
    x = np.sin(np.arange(48000))
    corpus = SimpleNamespace(classes=['kettle'], targets={'kettle': [x]},
        spans={'kettle': [(0, len(x))]}, window_cues={'kettle': [None]})
    with pytest.raises(ValueError, match='window_cues.boil'):
        validate_corpus(corpus, nn.DEFAULTS)


def test_evaluation_adapter_does_not_call_an_unknown_fragment_positive(monkeypatch):
    from sensorsim.evaluation.stream import training_example
    reached = []
    def make(*args, **kwargs):
        reached.append(True)
        return np.ones(16000), np.array([-1.], np.float32)
    monkeypatch.setattr(nn, 'make_example', make)
    with pytest.raises(ValueError, match='unknown'):
        training_example(None, np.random.default_rng(1), nn.DEFAULTS)
    assert reached
