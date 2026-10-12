# SPDX-License-Identifier: Apache-2.0
"""Run explicitly on the proposed patched copy, never claim merged integration."""
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import nn, training_dynamics as td, training_policy as p


def corpus_and_config():
    cfg = dict(nn.DEFAULTS, **td.DEFAULTS)
    cfg.update(cue_validity=True, cue_masks=True, p_specaugment=1., p_freq_mixstyle=0.,
               p_confuser_given_background=0., p_background_only=0., batch=6, threads=2,
               partial_windows=True, _dynamics_report=True, _negative_report=True)
    fs = cfg['fs']
    grid = p.FeatureGrid.from_config(cfg)
    sources = {c: [np.sin(2*np.pi*f*np.arange(2*fs)/fs).astype(np.float32)]
               for c, f in [('a', 800), ('b', 1500)]}
    background = (np.round(np.random.default_rng(21).normal(0, .0001, 4*fs)*32768)/32768).astype(np.float32)
    rows = []
    for label, x in [('a', sources['a'][0]), ('b', sources['b'][0]), ('background', background)]:
        entries = {c: dict(state='verified_absent') for c in sources}
        if label != 'background':
            # Disclosed synthetic support; no assumed first attack or class-specific sampler.
            entries[label] = dict(state='present', alternatives=[
                [dict(start=fs//2, stop=fs//2+80, mel_bins=[4])],
                [dict(start=fs, stop=fs+80, mel_bins=[5])]])
        rows.append(dict(label=label, source_group=label, cue_support=dict(schema=p.POLICY,
            units='samples/mel_bins', reviewed=True, reviewer='development-control', annotation_id=label,
            source_id=label, source_sha256=p.audio_digest(x, fs), evidence_domain='synthetic proxy',
            feature_grid=vars(grid), classes=entries)))
    corpus = SimpleNamespace(classes=list(sources), rows=rows, targets=sources,
        spans={c: [(0, 2*fs)] for c in sources}, background=[background], confuser_draws=0,
        negative_pool=None)
    p.bind_corpus(corpus, cfg)
    return corpus, cfg


def test_handoff_reaches_mixer_policy_masks_and_loss(monkeypatch):
    corpus, cfg = corpus_and_config()
    reached = dict(mixer=0, policy=0, masks=0)
    original_mix, original_policy, original_masks = nn.mix_device_audio, p.annotate_context, p.constrain_masks
    def mix(*args):
        reached['mixer'] += 1
        return original_mix(*args)
    def context(*args):
        reached['policy'] += 1
        return original_policy(*args)
    def masks(*args, **kw):
        reached['masks'] += 1
        return original_masks(*args, **kw)
    monkeypatch.setattr(nn, 'mix_device_audio', mix)
    monkeypatch.setattr(p, 'annotate_context', context)
    monkeypatch.setattr(p, 'constrain_masks', masks)
    # Only observability is replaced here: support and proposals remain real.
    # It tests routing, not fidelity of the actual device audibility detector.
    monkeypatch.setattr(td, 'power', lambda x, cfg: np.full((40, 97), 1. if np.ptp(x) > .001 else 1e-12))
    # Force disclosed correct and bad crops; six random draws need not contain both.
    starts = iter((0, 8000, 21000, 0, 8000, 21000))
    monkeypatch.setattr(p, 'propose_crop_starts', lambda *args, **kwargs: (next(starts),))
    x, y, stats, sampling = nn.Batches(corpus, cfg, 1, 37)[0]
    assert reached == dict(mixer=12, policy=12, masks=6)  # comparator also rendered
    traces = sampling['cue_records']
    assert all(t['mask_eligible'] for t in traces)
    assert any(t['mask_accepted'] for t in traces)
    assert any(1 in t['states'] for t in traces)
    assert any(-1 in t['states'] for t in traces)
    assert all(not t['background_component_only'] for t in traces)
    import torch
    z = torch.zeros_like(torch.from_numpy(y), requires_grad=True)
    p.masked_bce(z, torch.from_numpy(y)).backward()
    assert torch.all(z.grad[torch.from_numpy(y) == -1] == 0)
    assert torch.any(z.grad[torch.from_numpy(y) == 0] != 0)


def test_validation_no_masks_and_missing_annotation_no_fallback(monkeypatch):
    corpus, cfg = corpus_and_config()
    def forbidden(*args, **kw):
        raise AssertionError('validation mask function reached')
    monkeypatch.setattr(p, 'constrain_masks', forbidden)
    validation = nn.Batches(corpus, cfg, 1, 3, training=False)
    assert not validation.cfg['cue_masks'] and validation.cfg['p_specaugment'] == 0
    assert len(validation[0]) == 3
    del corpus.cue_annotations
    with pytest.raises(ValueError, match='bound'):
        nn.make_example(corpus, np.random.default_rng(0), cfg)


def test_cue_erasure_after_actual_mix_changes_labels(monkeypatch):
    corpus, cfg = corpus_and_config()
    cfg = dict(cfg, partial_windows=False, cue_masks=False, p_specaugment=0.)
    calls = []
    original = nn.mix_device_audio
    def mix(*args):
        heard = original(*args)
        calls.append('mixed')
        return heard
    monkeypatch.setattr(nn, 'mix_device_audio', mix)
    # Independent hand-specified post-device observable cells: good then erased.
    monkeypatch.setattr(td, 'power', lambda x, cfg: np.full((40, 97), 1. if np.ptp(x) > .001 else 1e-12))
    _, good, good_ctx = nn.make_example(corpus, np.random.default_rng(3), cfg, return_context=True)
    monkeypatch.setattr(td, 'power', lambda x, cfg: np.full((40, 97), 1e-12))
    _, erased, erased_ctx = nn.make_example(corpus, np.random.default_rng(3), cfg, return_context=True)
    assert calls == ['mixed', 'mixed']
    index = good_ctx['class_index']
    assert good[index] == 1 and erased[index] == -1
    assert all(erased[j] == 0 for j in range(2) if j != index)


def test_background_and_confuser_route_keeps_verified_absence(monkeypatch):
    from sensorsim.negatives import integration
    corpus, cfg = corpus_and_config()
    seen = []
    def confuser(*args):
        seen.append(True)
        return np.sin(np.arange(cfg['fs']))*.01, (0, cfg['fs'])
    monkeypatch.setattr(integration, 'draw_confuser', confuser)
    _, y, ctx = nn.make_example(corpus, np.random.default_rng(3), dict(cfg, p_background_only=1.), return_context=True)
    assert seen and y.tolist() == [0, 0]
    assert ctx['cue_trace']['background_component_only']
    # The background annotation is never represented as provenance for the confuser.


def test_history_and_parent_level_survive_handoff(monkeypatch):
    from sensorsim import iot_mic, event_level
    corpus, cfg = corpus_and_config()
    cfg = dict(cfg, cue_masks=False, p_specaugment=0.)
    # Constant-amplitude non-DC sinusoids have independently calculable power 1/2.
    # The truncated, phase-shifted crop has a different finite-window power.
    starts = 17321
    monkeypatch.setattr(p, 'propose_crop_starts', lambda *args, **kw: (starts,))
    events, captured = [], []
    original = nn.mix_device_audio
    def mix(bed, event, span, rng, settings):
        events.append(event.copy())
        return original(bed, event, span, rng, settings)
    def capture(self, x, fs, rng, **kw):
        captured.append(x.copy())
        return x
    monkeypatch.setattr(nn, 'mix_device_audio', mix)
    monkeypatch.setattr(iot_mic.MicChain, 'apply', capture)
    _, _, ctx = nn.make_example(corpus, np.random.default_rng(3), cfg, return_context=True)
    assert np.any(events[0][:cfg['fs']])
    actual_gain = np.dot(events[0], captured[0])/np.dot(events[0], events[0])
    expected_gain = 10**((ctx['requested_level_db']-94)/20)  # sqrt(2 * 1/2) == 1
    assert abs(20*np.log10(actual_gain/expected_gain)) < 1e-5
    # Stale gain compensation is a mechanism-specific bad control.
    monkeypatch.setattr(event_level, 'level_power', lambda *args, **kw: 1.)
    events.clear(); captured.clear()
    nn.make_example(corpus, np.random.default_rng(3), cfg, return_context=True)
    bad_gain = np.dot(events[0], captured[0])/np.dot(events[0], events[0])
    assert abs(20*np.log10(bad_gain/expected_gain)) > 1e-5


def test_tiny_trainer_binds_masked_loss_and_trace_export(tmp_path, monkeypatch):
    """Software routing only: identical mocked corpus is not efficacy/evaluation."""
    import json
    corpus, cfg = corpus_and_config()
    monkeypatch.setattr(nn, 'Corpus', lambda *args: corpus)
    monkeypatch.setattr(td, 'power', lambda x, cfg: np.full((40, 97), 1. if np.ptp(x) > .001 else 1e-12))
    monkeypatch.setattr(p, 'propose_crop_starts', lambda *a, **k: (8000,))
    calls = []
    original = p.masked_bce
    def loss(logits, states):
        calls.append(states.detach().cpu().numpy().copy())
        return original(logits, states)
    monkeypatch.setattr(p, 'masked_bce', loss)
    report = nn.train(tmp_path/'unused.jsonl', tmp_path/'model.npz', cue_validity=True,
        cue_masks=True, partial_windows=True, p_specaugment=1., p_freq_mixstyle=0.,
        p_confuser_given_background=0., p_background_only=0., selection_mode='diagnostic_final',
        steps=3, batch=12, width=4, blocks=1, threads=2, workers=0, val_windows=64)
    trace = json.loads((tmp_path/'model.cue-policy.json').read_text())
    assert len(calls) == 4  # 3 SGD + diagnostic validation
    assert trace['summary']['mask_accepted'] > 0
    assert trace['summary']['examples'] == 36
    assert len(trace['records']) == 36
    assert report['cue_policy']['acceptance'] == 'not run'
    assert report['cue_policy']['default_promotion'] is False
    assert report['cue_policy']['paired_guardrails']['status'] == 'pass'
    from hashlib import sha256
    assert report['cue_policy']['trace_sha256'] == sha256((tmp_path/'model.cue-policy.json').read_bytes()).hexdigest()
    assert report['window_supervision']['policy'] == p.POLICY


def test_paired_comparator_is_actually_rendered(monkeypatch):
    corpus, cfg = corpus_and_config()
    monkeypatch.setattr(td, 'power', lambda x, cfg: np.full((40, 97), 1. if np.ptp(x) > .001 else 1e-12))
    monkeypatch.setattr(p, 'propose_crop_starts', lambda *a, **k: (8000,))
    calls = []
    original = nn.make_example
    def draw(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args[2]['partial_windows'], result))
        return result
    monkeypatch.setattr(nn, 'make_example', draw)
    _, y, _, sampling = nn.Batches(corpus, cfg, 1, 37)[0]
    assert len(calls) == 2*cfg['batch']  # separate baseline rendering, no inferred denominator
    for i, record in enumerate(sampling['cue_records']):
        baseline = record['paired_baseline']
        assert baseline['states'] == calls[2*i][1][1].tolist()
        assert calls[2*i][0] is False and calls[2*i+1][0] is True
        assert baseline['source_id'] == record['source_id']
        assert baseline['pair_context'] == record['pair_context']
        assert record['states'] == y[i].tolist()
    summary = p.training_diagnostics(sampling['cue_records'], corpus.classes, require_masks=True)
    assert summary['paired_guardrails']['status'] == 'pass'
    assert summary['paired_guardrails']['present_mass_fraction'] == {'a': 1., 'b': 1.}


def test_low_retention_training_cannot_export(tmp_path, monkeypatch):
    """RV-2: keep one positive in each twenty, yet never export a passing gate."""
    corpus, cfg = corpus_and_config()
    monkeypatch.setattr(nn, 'Corpus', lambda *args: corpus)
    monkeypatch.setattr(td, 'power', lambda x, cfg: np.full((40, 97), 1. if np.ptp(x) > .001 else 1e-12))
    monkeypatch.setattr(p, 'propose_crop_starts', lambda *a, **k: (8000,))
    original = nn.Batches.__getitem__
    seen = {}
    def defect(self, k):
        batch = original(self, k)
        if self.training:
            for i, record in enumerate(batch[-1]['cue_records']):
                for j, state in enumerate(record['states']):
                    if state == 1:
                        count = seen.get(j, 0); seen[j] = count + 1
                        if count % 20 != 0:
                            batch[1][i, j] = -1
                            record['states'][j] = -1
        return batch
    monkeypatch.setattr(nn.Batches, '__getitem__', defect)
    with pytest.raises(ValueError, match='paired.*0.80'):
        nn.train(tmp_path/'unused.jsonl', tmp_path/'model.npz', cue_validity=True,
            cue_masks=True, partial_windows=True, p_specaugment=1., p_freq_mixstyle=0.,
            p_confuser_given_background=0., p_background_only=0., selection_mode='diagnostic_final',
            steps=3, batch=20, width=4, blocks=1, threads=2, workers=0, val_windows=64)
    assert not (tmp_path/'model.npz').exists()


def test_original_random_partial_fixture_is_rejected(tmp_path, monkeypatch):
    """Preserve the former smoke fixture's disclosed 70%/56.25% retention failure."""
    import json
    corpus, cfg = corpus_and_config()
    monkeypatch.setattr(nn, 'Corpus', lambda *args: corpus)
    monkeypatch.setattr(td, 'power', lambda x, cfg: np.full((40, 97), 1. if np.ptp(x) > .001 else 1e-12))
    with pytest.raises(ValueError, match='paired.*0.80'):
        nn.train(tmp_path/'unused.jsonl', tmp_path/'model.npz', cue_validity=True,
            cue_masks=True, partial_windows=True, p_specaugment=1., p_freq_mixstyle=0.,
            p_confuser_given_background=0., p_background_only=0., selection_mode='diagnostic_final',
            steps=3, batch=12, width=4, blocks=1, threads=2, workers=0, val_windows=64)
    assert not (tmp_path/'model.npz').exists()
    trace = json.loads((tmp_path/'model.cue-policy.json').read_text())
    assert trace['summary']['paired_guardrails']['present_mass_fraction'] == {'a': .7, 'b': .5625}
    assert len(trace['records']) == 36


def test_missing_pairing_or_confuser_hook_fails_closed(monkeypatch):
    corpus, cfg = corpus_and_config()
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('must fail before rendering')
    for overrides, message in [({'cue_pairing': None}, 'protocol'),
                               ({'p_confuser_given_background': .5}, 'confuser identity hook')]:
        with pytest.raises(ValueError, match=message):
            p.paired_example(forbidden, corpus, np.random.default_rng(0), dict(cfg, **overrides), example_id='a')
    assert not calls


def test_pairing_does_not_change_candidate_stream_or_counts(monkeypatch):
    corpus, cfg = corpus_and_config()
    rng = np.random.default_rng(37)
    paired = p.paired_example(nn.make_example, corpus, rng, cfg, example_id='0')
    ordinary_rng = np.random.default_rng(37)
    ordinary = nn.make_example(corpus, ordinary_rng, cfg, return_context=True)
    np.testing.assert_array_equal(paired[0], ordinary[0])
    np.testing.assert_array_equal(paired[1], ordinary[1])
    assert rng.bit_generator.state == ordinary_rng.bit_generator.state
    assert corpus.confuser_draws == 0


def test_actual_manifest_training_export_with_paired_guard(tmp_path, monkeypatch):
    """Unmocked mixer/features/manifest path; analytic sine support; workers=0."""
    import json
    from scipy.io import wavfile
    corpus, cfg = corpus_and_config()
    centres = 700*(np.exp(np.linspace(np.log1p(cfg['fmin_hz']/700),
        np.log1p(cfg['fmax_hz']/700), cfg['n_mels']+2))-1)[1:-1]
    rows = []
    for split, offset in [('train', 0), ('val', 1)]:
        for row in corpus.rows:
            row = json.loads(json.dumps(row))
            label = row['label']
            x = corpus.background[0] if label == 'background' else corpus.targets[label][0]
            x = np.roll(x, offset*73).copy()
            row['source_group'] = f'{split}-{label}'
            support = row['cue_support']
            support['source_id'] = row['source_group']
            support['source_sha256'] = p.audio_digest(x, cfg['fs'])
            if label != 'background':
                band = int(np.argmin(abs(centres-{'a': 800, 'b': 1500}[label])))
                for alternative in support['classes'][label]['alternatives']:
                    alternative[0]['mel_bins'] = [band]
            row.update(file=f'{split}-{label}.wav', split=split, fs=cfg['fs'])
            wavfile.write(tmp_path/row['file'], cfg['fs'], x)
            rows.append(row)
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(p, 'propose_crop_starts', lambda *a, **k: (8000,))
    report = nn.train(manifest, tmp_path/'model.npz', cue_validity=True,
        cue_masks=True, partial_windows=True, p_specaugment=1., p_freq_mixstyle=0.,
        p_confuser_given_background=0., p_background_only=.25, selection_mode='diagnostic_final',
        steps=3, batch=12, width=4, blocks=1, threads=2, workers=0, val_windows=64)
    assert (tmp_path/'model.npz').is_file()
    assert report['cue_policy']['paired_guardrails']['present_mass_fraction'] == {'a': 1., 'b': 1.}
    trace = json.loads((tmp_path/'model.cue-policy.json').read_text())
    assert len(trace['records']) == 36
    assert all(r['paired_baseline']['protocol'] == p.PAIRING for r in trace['records'])
