# SPDX-License-Identifier: Apache-2.0
"""Development controls for the final WP7 repairs; synthetic proxy, not realism."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sensorsim import nn, iot_mic, project, training_device, training_dynamics as td
from tools import dynamics_sweep as sweep


def gapped_corpus():
    fs = 16000
    x = np.zeros(25600)
    for start, amplitude in [(800, .3), (8800, .15), (16800, .07)]:
        t = np.arange(800)/fs
        x[start:start+800] = amplitude*np.sin(2*np.pi*1000*t)*np.exp(-t/.01)
    return SimpleNamespace(classes=["knock"], targets={"knock": [x]},
                           spans={"knock": [(800, 17600)]}, background=[np.zeros(48000)])


def span_power(x, fs, span):
    a, b = span
    return float(np.sum(np.asarray(x[a:b], float)**2)/(b-a)) if b > a else 0.


def active_power(x, fs, span):
    # Independent simulated convention: ignore exactly silent inter-hit gaps.
    v = np.asarray(x[span[0]:span[1]], float)
    v = v[np.abs(v) > .001*np.max(np.abs(v), initial=0)]
    return float(np.dot(v, v)/len(v)) if len(v) else 0.


def gate_power(x, fs, span):
    # Independent re-implementation of the production convention documented in
    # training_device.active_power: centred 10 ms moving mean, -25 dB of maximum, within span.
    x = np.asarray(x, float); a, b = span; w = min(len(x), max(1, round(.01*fs)))
    c = np.concatenate([[0.], np.cumsum(x*x)]); i = np.arange(len(x))
    env = (c[np.clip(i+(w-1)//2+1, 0, len(x))] - c[np.clip(i-w//2, 0, len(x))])/w
    m = env > env.max()*10**(-2.5); m[:a] = False; m[b:] = False
    return float(np.mean(x[m]**2)) if m.any() else 0.


def gain_errors(measure):
    c = gapped_corpus()
    errors = []
    original = nn.mix_device_audio
    observed = []
    def apply(chain, x, fs, rng, **kwargs):
        observed.append(np.array(x))  # actual post-normalization production input
        return np.asarray(x)
    for seed in range(20):
        observed.clear()
        seen = []
        def mix(bg, event, span, rng, cfg):
            seen.append(event.copy())
            return original(bg, event, span, rng, cfg)
        cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), partial_windows=True, p_background_only=0)
        with patch.object(nn, "mix_device_audio", mix), patch.object(iot_mic.MicChain, "apply", apply):
            _, _, ctx = nn.make_example(c, np.random.default_rng(seed), cfg, return_context=True)
        assert seen and observed, "production path was not reached"
        event, rendered = seen[-1], observed[-1]
        gain = np.dot(event, rendered)/np.dot(event, event)
        whole = measure(c.targets["knock"][0], cfg["fs"], c.spans["knock"][0])
        expected = 10**((ctx["requested_level_db"]-94)/20)/np.sqrt(2*whole)
        errors.append(20*np.log10(gain/expected))
    return np.array(errors)


def assert_gain_contract(measure):
    np.testing.assert_allclose(gain_errors(measure), 0, atol=2e-6, rtol=0)


def test_gapped_partial_gain_matches_main_mixer():
    # Main's mixer normalizes on event-active samples (WP3); crops must inherit that whole-take gain.
    assert_gain_contract(gate_power)


def test_shared_measure_switch_reaches_mixer_and_compensation():
    # One change (the production measure) must move both the mixer and the crop compensation.
    with patch.object(training_device, "active_power", lambda x, fs, span=None: (span_power(x, fs, span), 0)):
        assert_gain_contract(span_power)


def test_divergence_detector_rejects_changed_mixer_only():
    # Same detector must reject a simulated active-sample mixer with stale compensation.
    original = nn.mix_device_audio
    calls = []
    def active_mixer(bg, event, span, rng, cfg):
        calls.append(span)
        # Mixer-only drift back to span-mean normalization; compensation keeps the shared measure.
        p, q = span_power(event, cfg["fs"], span), gate_power(event, cfg["fs"], span)
        shift = 10*np.log10(q/p) if p > 0 and q > 0 else 0
        changed = dict(cfg, event_spl_db=[v+shift for v in cfg["event_spl_db"]])
        return original(bg, event, span, rng, changed)
    with patch.object(nn, "mix_device_audio", active_mixer):
        with pytest.raises(AssertionError):
            assert_gain_contract(gate_power)
    assert len(calls) >= 20


def test_masks_count_and_skip_unmaskable_cues_then_continue():
    # All three frames are indispensable; frequency and time masks erase audibility.
    ctx = {"event": np.array([[10., 10., 10.], [0., 0., 0.]]),
           "floor": np.ones((2, 3)), "positive": True}
    x = np.ones((1, 1, 2, 3))
    out = td.event_aware_masks(x, np.random.default_rng(1), 1., [ctx])
    np.testing.assert_array_equal(out, 1.)
    assert ctx["mask_qa_skips"] == 1
    good = {"event": np.ones((40, 97))*10, "floor": np.ones((40, 97)), "positive": True}
    out = td.event_aware_masks(np.ones((1, 1, 40, 97)), np.random.default_rng(1), 1., [good])
    assert np.count_nonzero(out == 0) > 0 and good.get("mask_qa_skips", 0) == 0


@pytest.mark.parametrize("knob", ["p_specaugment", "p_freq_mixstyle", "event_aware_masks", "mixstyle_same_label"])
def test_training_augmentation_rng_pairing_zero_probability(knob):
    torch = nn._torch()
    x = np.random.default_rng(3).normal(size=(4, 1, 40, 97)).astype(np.float32)
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], np.float32)
    cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}))
    changed = dict(cfg, **{knob: 0. if knob.startswith("p_") else True})
    def run(config):
        r = np.random.default_rng(5)
        torch.manual_seed(42)
        outputs = [nn._augment_training_batch(x.copy(), y, r, config, torch).numpy() for _ in range(12)]
        return outputs, r.bit_generator.state, torch.get_rng_state().clone()
    a, b = run(cfg), run(changed)
    assert a[1] == b[1]
    assert torch.equal(a[2], b[2])
    assert any(not np.array_equal(u, v) for u, v in zip(a[0], b[0]))


def test_storage_metric_excludes_final_capture_rounding_and_observes_hook():
    c = gapped_corpus()
    x = c.targets["knock"][0]
    ideal = x + 1e-8*np.sin(np.arange(len(x)))
    cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), p_background_only=0)
    normal = sweep.post_chain_quantization(c, ideal, 7, cfg)
    # Capture-format defect targets quantization, not event level.
    def coarse(background, increment, rng, profile):
        return np.round((np.asarray(background)+np.asarray(increment))*1024)/1024
    with patch.object(training_device, "conditional_capture", coarse):
        coarse_result = sweep.post_chain_quantization(c, ideal, 7, cfg)
    assert normal["quantization_error_a_sine_dbfs"] == coarse_result["quantization_error_a_sine_dbfs"]
    assert normal["quantization_stage"] == "pre-final-capture"
    same = sweep.post_chain_quantization(c, x.copy(), 7, cfg)
    assert same["quantization_error_a_sine_dbfs"] <= -290
    damaged_c = copy.deepcopy(c)
    damaged_c.targets['knock'] = [np.round(x*127)/127]  # genuine coarse storage defect
    bad = sweep.post_chain_quantization(damaged_c, x, 7, cfg)
    assert bad["quantization_margin_db"] < 20
    with patch.object(nn, "make_example", return_value=(np.zeros(16000), np.ones(1))):
        with pytest.raises(RuntimeError, match="capture hook"):
            sweep.post_chain_quantization(c, ideal, 7, cfg)


def test_project_wires_allowed_knobs_and_forwards_them(tmp_path, monkeypatch):
    settings = dict(event_aware_masks=True, audibility_gate=True, mixstyle_same_label=True,
                    positive_fraction=.4, low_event_spl_tail=True)
    (tmp_path/"project.yaml").write_text("recognizer: " + json.dumps(settings))
    cfg = project.load_project(tmp_path)
    assert all(cfg["recognizer"][k] == v for k, v in settings.items())
    out = tmp_path/"out"
    (out/"dataset").mkdir(parents=True)
    (out/"dataset/manifest.jsonl").write_text(json.dumps({"label": "knock", "split": "val"})+"\n")
    calls = []
    def train(*args, **kwargs):
        calls.append(kwargs)
        return dict(classes=["knock"], parameters=1, macs_per_window=1, parity_max_abs_diff=0, best_val_mean_ap=1, detector={})
    monkeypatch.setattr(nn, "train", train)
    project._train(cfg, out)
    assert len(calls) == 1 and all(calls[0][k] == v for k, v in settings.items())


@pytest.mark.parametrize("settings, reason", [({"partial_windows": True}, "class-swap"),
                                            ({"validation_mode": "hard"}, "class-swap")])
def test_project_explains_held_settings(tmp_path, settings, reason):
    (tmp_path/"project.yaml").write_text("recognizer: " + json.dumps(settings))
    with pytest.raises(ValueError, match=reason):
        project.load_project(tmp_path)


def test_project_accepts_explicit_off_and_refuses_unknown(tmp_path):
    (tmp_path/"project.yaml").write_text("recognizer: {partial_windows: false, validation_mode: synthetic}")
    assert project.load_project(tmp_path)["recognizer"]["partial_windows"] is False
    (tmp_path/"project.yaml").write_text("recognizer: {invented_dynamics: true}")
    with pytest.raises(ValueError, match="unknown"):
        project.load_project(tmp_path)


@pytest.mark.parametrize('ap_values, chosen', [([.8123456789, .8123456789], 251),
                                             ([.9, .8], 251), ([.81234001, .81234999], 251)])
def test_diagnostic_final_uses_ordinary_tracker_and_ignores_ap_for_selection(tmp_path, monkeypatch, ap_values, chosen):
    torch = nn._torch()
    c = gapped_corpus()
    x = np.ones((4, 1, 40, 97), np.float32)
    y = np.array([[1], [0], [1], [0]], np.float32)
    evaluated_weights, evaluated_scores = [], []
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(40*97, 1)
        def forward(self, x):
            logits = self.fc(x.flatten(1))
            if not self.training:
                evaluated_weights.append(self.fc.weight.detach().clone())
                evaluated_scores.append(torch.sigmoid(logits).detach().numpy().copy())
            return logits
    monkeypatch.setattr(nn, "Corpus", lambda *args: c)
    monkeypatch.setattr(nn, "build_model", lambda *args: Tiny())
    monkeypatch.setattr(nn, "Batches", lambda *args, **kwargs: [(x, y)])
    monkeypatch.setattr(nn, "_training_batches", lambda *args: iter([(x, y, dict(windows=4,background_windows=2,confuser_windows=0))]*251))
    values = iter(ap_values)
    monkeypatch.setattr(nn, "_average_precision", lambda *args: next(values))
    captured = {}
    def export(*args, **kwargs):
        captured.update(kwargs["training"])
        assert torch.equal(args[0].fc.weight, evaluated_weights[0 if chosen == 250 else 1])
        return {}
    def tracker(model, corpus, cfg, out):
        assert corpus is c, "stress corpus reached deployed tracker"
        assert cfg["event_spl_db"] == (nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})["event_spl_db"]
        assert cfg["p_background_only"] == (nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})["p_background_only"]
        assert cfg["validation_mode"] == "synthetic"
        return {}
    monkeypatch.setattr(nn, "export", export)
    monkeypatch.setattr(nn.edge, "Model", lambda *args: None)
    monkeypatch.setattr(nn, "_fit_tracker", tracker)
    out = tmp_path/"model.npz"
    nn.train(tmp_path/"unused", out, steps=251, workers=0, threads=1, validation_mode="hard",
             p_specaugment=0, p_freq_mixstyle=0, p_confuser_given_background=0., selection_mode="diagnostic_final")
    assert captured["best_step"] == chosen
    assert [row["step"] for row in captured["history"]] == [250, 251]
    assert [row["val_mean_ap"] for row in captured["history"]] == ap_values
    scores = np.load(out.with_suffix(".validation_scores.npz"))
    assert scores["scores"].shape == (2, 4, 1)
    np.testing.assert_array_equal(scores["steps"], [250, 251])
    np.testing.assert_array_equal(scores["labels"], y)
    np.testing.assert_array_equal(scores['scores'], evaluated_scores)


@pytest.mark.parametrize("shape", [(1, 1, 0, 3), (1, 1, 2, 1), (1, 1, 2, 3)])
def test_mask_qa_empty_or_incompatible_context_is_counted(shape):
    x = np.ones(shape)
    ctx = {"event": np.zeros((2, 2)), "floor": np.zeros((2, 2)), "positive": True}
    td.event_aware_masks(x, np.random.default_rng(1), 1, [ctx])
    np.testing.assert_array_equal(x, 1)
    assert ctx["mask_qa_skips"] == 1


def test_mask_skip_counts_reach_batch_stats(monkeypatch):
    original = td.event_aware_masks
    def forced_skip(x, rng, p, contexts):
        for ctx in contexts:
            ctx["event"] = np.zeros((0, 0))
        return original(x, rng, 1., contexts)
    monkeypatch.setattr(td, "event_aware_masks", forced_skip)
    cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), batch=3, event_aware_masks=True, _dynamics_report=True)
    _, _, stats = nn.Batches(gapped_corpus(), cfg, 1, 9)[0]
    assert stats.shape == (3, 5)
    np.testing.assert_array_equal(stats[:, -1], 1)


@pytest.mark.parametrize("bad", [np.zeros(0), np.zeros(25600), np.full(25600, np.nan)])
def test_storage_metric_rejects_empty_silent_or_nonfinite_pairs(bad):
    with pytest.raises(ValueError, match="storage pair"):
        sweep.post_chain_quantization(gapped_corpus(), bad, 7, (nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}))


def test_storage_metric_refuses_to_score_incompatible_capture_path():
    c = gapped_corpus()
    result = sweep.post_chain_quantization(c, c.targets["knock"][0], 7,
                                          dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}), device_split="stress"))
    assert result["quantization_margin_db"] is None
    assert result["quantization_stage"] == "unsupported capture path"


def test_default_trainer_augmentation_equals_legacy_path():
    torch = nn._torch()
    x = np.random.default_rng(3).normal(size=(4, 1, 40, 97)).astype(np.float32)
    cfg = dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"}))
    rng = np.random.default_rng(5)
    torch.manual_seed(42)
    old = []
    for _ in range(12):
        a = torch.from_numpy(nn._spec_augment(x.copy(), rng, cfg["p_specaugment"]))
        if rng.random() < cfg["p_freq_mixstyle"]:
            a = nn._freq_mixstyle(a, torch, cfg["mixstyle_alpha"])
        old.append(a.numpy())
    old_state, torch_state = rng.bit_generator.state, torch.get_rng_state()
    rng = np.random.default_rng(5)
    torch.manual_seed(42)
    new = [nn._augment_training_batch(x.copy(), np.ones((4, 1)), rng, cfg, torch).numpy()
           for _ in range(12)]
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)
    assert old_state == rng.bit_generator.state
    assert torch.equal(torch_state, torch.get_rng_state())
