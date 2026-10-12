# SPDX-License-Identifier: Apache-2.0
"""Disclosed deterministic synthetic development controls, not acceptance data."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from sensorsim import training_policy as p


@pytest.fixture
def fixture():
    grid = p.FeatureGrid(1000, 4, 2, 4, 0., 500.)
    source = np.sin(np.arange(32))
    classes = ('a', 'b', 'c')
    data = dict(schema=p.POLICY, units='samples/mel_bins', reviewed=True,
                source_id='parent-1', annotation_id='manual-1', reviewer='development-fixture-author',
                evidence_domain='synthetic proxy', source_sha256=p.audio_digest(source, 1000),
                feature_grid=vars(grid), classes={
                    'a': dict(state='present', alternatives=[
                        [dict(start=8, stop=10, mel_bins=[0])],
                        [dict(start=14, stop=16, mel_bins=[3])]]),
                    'b': dict(state='verified_absent'), 'c': dict(state='unknown')})
    annotation = p.Annotation.from_dict(data, source, grid, classes)
    return source, grid, classes, data, annotation


def test_three_states_middle_late_and_erased_energy(fixture):
    source, _, _, _, a = fixture
    good = p.constrain_crop(a, 6, 12, hop_samples=3)
    late = p.constrain_crop(a, 12, 12, hop_samples=3)
    bad = p.constrain_crop(a, 20, 12, hop_samples=3)
    assert good.states == late.states == (1., 0., -1.)
    assert good.preserves_present and late.preserves_present
    assert bad.states == (-1., 0., -1.) and not bad.preserves_present
    assert np.sum(source[20:32]**2) > np.sum(source[6:18]**2)*.8
    assert good.phase == 0 and p.constrain_crop(a, 7, 12, hop_samples=3).phase == 1


def test_projection_has_independent_exhaustive_rectangle_oracle(fixture):
    _, grid, _, _, a = fixture
    # Independent scalar enumeration, including crop padding and STFT remainder.
    for start in range(-12, 33):
        for n in range(4, 17):
            frames = [(start+2*j, start+2*j+4) for j in range(1+(n-4)//2)]
            sufficient = any(start <= lo and hi <= frames[-1][1] for lo, hi in [(8, 10), (14, 16)])
            assert p.class_validity(a, start, n)[0] == (1 if sufficient else -1)
    # The last sample is not represented by any STFT frame in this 13-sample crop.
    edge = replace(a, cues=(((p.Region(12, 13, (0,)),),), (), ()))
    assert p.class_validity(edge, 0, 13)[0] == -1


def test_combined_mask_kills_alternatives_individually_safe(fixture):
    _, _, _, _, a = fixture
    x = np.arange(1., 21.).reshape(4, 5)
    # Source crop [6,18), frames [6,10),[8,12),[10,14),[12,16),[14,18).
    time = p.MaskProposal((0, 0), (0, 2))  # removes first sufficient cue
    freq = p.MaskProposal((3, 4), (0, 0))  # removes second sufficient cue
    union = p.MaskProposal((3, 4), (0, 2))
    assert p.constrain_masks(x, a, 6, 12, [time], training=True).accepted
    assert p.constrain_masks(x, a, 6, 12, [freq], training=True).accepted
    bad = p.constrain_masks(x, a, 6, 12, [union], training=True)
    assert not bad.accepted and bad.attempts[0]['states'] == [-1., 0., -1.]
    assert bad.attempts[0]['reason'] == 'cue_erased'
    np.testing.assert_array_equal(bad.features, x)  # explicit keep-original on exhaustion
    good = p.MaskProposal((1, 2), (2, 3))
    result = p.constrain_masks(x, a, 6, 12, [union, good], training=True)
    assert result.accepted and len(result.attempts) == 2
    np.testing.assert_array_equal(result.features, [[1, 2, 0, 4, 5], [0, 0, 0, 0, 0],
                                                   [11, 12, 0, 14, 15], [16, 17, 0, 19, 20]])
    assert x[1, 0] == 6  # input not mutated


def test_every_required_rectangle_is_preserved(fixture):
    *_, a = fixture
    a = replace(a, cues=(((p.Region(8, 10, (0,)), p.Region(14, 16, (3,))),), (), ()))
    erased = p.MaskProposal((3, 4), (0, 0)).cells((4, 5))
    assert p.class_validity(a, 6, 12).tolist() == [1, 0, -1]
    assert p.class_validity(a, 6, 12, erased=erased).tolist() == [-1, 0, -1]


def test_post_device_availability_never_promotes_unknown_or_erased_cues(fixture):
    *_, a = fixture
    available = np.ones((4, 5), bool)
    available[0, :2] = False
    available[3, 3:] = False
    assert p.class_validity(a, 6, 12, available=available).tolist() == [-1, 0, -1]
    with pytest.raises(ValueError, match='boolean'):
        p.class_validity(a, 6, 12, available=available.astype(float))


@pytest.mark.parametrize('change', ['digest', 'units', 'unreviewed', 'grid', 'missing_class', 'missing_cue', 'duplicate_bins'])
def test_bad_annotation_fails_closed(fixture, change):
    source, grid, classes, data, _ = fixture
    data = copy.deepcopy(data)
    if change == 'digest': data['source_sha256'] = '0'*64
    elif change == 'units': data['units'] = 'seconds'
    elif change == 'unreviewed': data['reviewed'] = False
    elif change == 'grid': data['feature_grid']['hop'] = 1
    elif change == 'missing_class': del data['classes']['b']
    elif change == 'missing_cue': data['classes']['a']['alternatives'] = []
    elif change == 'duplicate_bins': data['classes']['a']['alternatives'][0][0]['mel_bins'] = [0, 0]
    with pytest.raises(ValueError):
        p.Annotation.from_dict(data, source, grid, classes)


@pytest.mark.parametrize('source', [[], [0, 0], [1, 1], [np.nan, 1], [[1, 2], [3, 4]]])
def test_degenerate_source_is_not_semantic_evidence(source):
    with pytest.raises(ValueError): p.audio_digest(source, 1000)


def test_bank_duplicate_and_empty_data_fail(fixture):
    *_, a = fixture
    for bank in ([], [a, replace(a, source_id='different-name')], [a]):
        with pytest.raises(ValueError): p.validate_bank(bank)
    # [a] lacks positive support for b and c: cannot train an all-unknown class.
    a = replace(a, classes=('a',), states=(1,), cues=a.cues[:1])
    p.validate_bank([a])


def test_proposals_are_repeatable_and_do_not_take_labels(fixture):
    import inspect
    _, grid, _, _, _ = fixture
    first = p.propose_masks(np.random.default_rng(14), grid.shape(12), 32)
    second = p.propose_masks(np.random.default_rng(14), grid.shape(12), 32)
    assert first == second and len(set(first)) > 1
    assert set(inspect.signature(p.propose_masks).parameters) == {'rng', 'shape', 'count', 'max_frequency', 'max_time'}
    starts = p.propose_crop_starts(np.random.default_rng(0), 32, 12, 3, 100)
    assert len(set(s % 3 for s in starts)) == 1 and min(starts) < 0 and max(starts) > 16
    with pytest.raises(ValueError): p.propose_masks(np.random.default_rng(0), (0, 5))
    with pytest.raises(ValueError): p.propose_crop_starts(np.random.default_rng(0), 32, 12, 13, 1)


def test_masks_validation_missing_hooks_identity_constant_and_invalid_ranges(fixture):
    *_, a = fixture
    x = np.arange(1., 21.).reshape(4, 5)
    mask = p.MaskProposal((1, 2), (2, 3))
    with pytest.raises(ValueError, match='validation'): p.constrain_masks(x, a, 6, 12, [mask], training=False)
    with pytest.raises(ValueError, match='hook'): p.constrain_masks(x, a, 6, 12, [], training=True)
    for bad in [np.zeros_like(x), np.ones_like(x), x*np.nan, x[:, :4]]:
        with pytest.raises(ValueError): p.constrain_masks(bad, a, 6, 12, [mask], training=True)
    result = p.constrain_masks(x, a, 6, 12, [p.MaskProposal((0, 0), (0, 0))], training=True)
    assert not result.accepted and result.attempts[0]['reason'] == 'identity'
    with pytest.raises(ValueError): p.MaskProposal((0, 5), (0, 0)).cells(x.shape)


def test_loss_independent_value_gradient_and_all_unknown_rejection():
    import torch
    logits = torch.tensor([[0., 40., 0.]], requires_grad=True)
    states = torch.tensor([[1., -1., 0.]])
    loss = p.masked_bce(logits, states)
    assert loss.detach().item() == pytest.approx(np.log(2))
    loss.backward()
    np.testing.assert_array_equal(logits.grad.numpy(), [[-.25, 0., .25]])
    # Known-bad ordinary BCE makes an unknown class contribute a gradient.
    logits.grad = None
    torch.nn.functional.binary_cross_entropy_with_logits(logits, states).backward()
    assert logits.grad[0, 1] != 0
    for labels in ([-1, -1], [], [np.nan], [2]):
        with pytest.raises(ValueError): p.masked_bce_weights(labels)
    with pytest.raises(ValueError): p.masked_bce(torch.tensor([[float('nan')]]), torch.tensor([[1.]]))


def ledger_record(ledger, i, *, final=(1, 0), augmented=True, eligible=True):
    ledger.record(example_id=str(i), source_id='parent', source_sha256='a'*64, phase=i % 4,
                  level_bin='60-65 dB SPL', baseline=[1, 0], final=final, eligible=eligible,
                  augmented=augmented, attempts=1, skipped=not augmented)


def test_guardrails_catch_loss_of_one_cell_and_all_skipped():
    # One independent parent, 20 derived examples: explicitly not 20 parents.
    # Boundary 80% positive mass and 25% nontrivial augmentation is reachable.
    def make(lost, augmented):
        ledger = p.PolicyLedger(['a', 'b'])
        for i in range(20):
            ledger_record(ledger, i, final=(-1 if i < lost else 1, 0), augmented=i < augmented)
        # Give b its own five unaugmented valid-present examples with absent a.
        for i in range(5):
            ledger.record(example_id='b'+str(i), source_id='other', source_sha256='b'*64, phase=i,
                level_bin='60-65 dB SPL', baseline=[0, 1], final=[0, 1], eligible=False,
                augmented=False, attempts=0, skipped=False)
        return ledger.summary()
    good = make(4, 5)
    assert good['status'] == 'pass' and good['present_mass_fraction'] == {'a': .8, 'b': 1.}
    assert good['nontrivial_fraction'] == .25 and good['unique_contents'] == good['parent_groups'] == 2
    assert make(5, 5)['status'] == 'fail'
    assert make(4, 4)['status'] == 'fail'
    assert make(0, 0)['status'] == 'fail'
    assert p.PolicyLedger(['a']).summary()['status'] == 'inconclusive'


def test_ledger_rejects_absence_loss_unknown_to_negative_and_duplicate():
    ledger = p.PolicyLedger(['a', 'b'])
    ledger_record(ledger, 0)
    with pytest.raises(ValueError, match='duplicate'): ledger_record(ledger, 0)
    with pytest.raises(ValueError, match='absence'): ledger_record(ledger, 1, final=(1, -1))
    with pytest.raises(ValueError, match='unknown'):
        ledger.record(example_id='u', source_id='s', source_sha256='c'*64, phase=0, level_bin='none',
            baseline=[-1, 0], final=[0, 0], eligible=True, augmented=True, attempts=1, skipped=False)


def test_configuration_cannot_enable_unguarded_masks_or_mixstyle():
    from sensorsim import nn
    base = dict(nn.DEFAULTS, cue_validity=True, cue_masks=True, p_freq_mixstyle=0.)
    assert p.validate_config(base)
    assert not p.validate_config(nn.DEFAULTS)
    for override in ({'cue_masks': False}, {'p_freq_mixstyle': .1}, {'event_aware_masks': True},
                     {'device_split': 'stress'}, {'window_policy': 'deployment/1'}, {'cue_masks': 1},
                     {'p_specaugment': float('nan')}, {'cue_validity': False}):
        with pytest.raises(ValueError): p.validate_config(dict(base, **override))


def test_missing_mask_context_and_degenerate_training_fail(fixture):
    *_, a = fixture
    x = np.arange(1., 21.).reshape(1, 1, 4, 5)
    with pytest.raises(ValueError, match='context'): p.mask_batch(x, np.random.default_rng(0), 1., [{}], training=True)
    with pytest.raises(ValueError, match='trace'): p.training_diagnostics([], ['a'], require_masks=False)
    r = dict(states=[-1], mask_eligible=True, mask_accepted=False, mask_attempts=[])
    with pytest.raises(ValueError, match='valid-present'): p.training_diagnostics([r], ['a'], require_masks=False)
    r['states'] = [1]
    with pytest.raises(ValueError, match='all-skipped'): p.training_diagnostics([r], ['a'], require_masks=True)


def test_unavailable_torch_is_an_explicit_dependency_failure(monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'torch': raise ImportError('torch unavailable development control')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(ImportError, match='unavailable'): p.masked_bce(None, None)


def test_retries_cannot_inflate_augmentation_success_fraction():
    ledger = p.PolicyLedger(['a'])
    ledger.record(example_id='0', source_id='parent', source_sha256='a'*64, phase=0,
        level_bin='none', baseline=[1], final=[1], eligible=True, augmented=True,
        attempts=5, skipped=False)
    result = ledger.summary()
    assert result['eligible_examples'] == 1 and result['eligible'] == 5
    assert result['nontrivial_fraction'] == .2 and result['status'] == 'fail'
    record = dict(states=[1], mask_eligible=True, mask_accepted=True, mask_attempts=[{}]*5)
    with pytest.raises(ValueError, match='0.25'):
        p.training_diagnostics([record], ['a'], require_masks=True)


def test_crop_trace_is_json_serializable(fixture):
    import json
    *_, a = fixture
    ctx = dict(event=np.ones((4, 5)), floor=np.full((4, 5), 1e-12), source_start=6,
               class_index=0, requested_level_db=65.)
    states = p.annotate_context(ctx, a, 12, dict(fs=1000, detector={'hop_s': .003}))
    decoded = json.loads(json.dumps(ctx['cue_trace']))
    assert decoded['structural_states'] == states.tolist() == [1, 0, -1]
