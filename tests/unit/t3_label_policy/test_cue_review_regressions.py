# SPDX-License-Identifier: Apache-2.0
"""Disclosed verifier counterexamples, reused as deterministic development data."""
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import training_policy as p


def make_ledger(erased_state):
    ledger = p.PolicyLedger(['a'])
    for i in range(20):
        ledger.record(example_id=str(i), source_id='synthetic-parent', source_sha256='a'*64,
            phase=0, level_bin='fixed', baseline=[1], final=[erased_state if i < 4 else 1],
            eligible=True, augmented=i < 5, attempts=1, skipped=i >= 5)
    return ledger


def test_positive_to_unknown_boundary_and_zero_gradient():
    import torch
    result = make_ledger(-1).summary()
    assert result['status'] == 'pass'
    assert result['present_mass_fraction'] == {'a': .8}
    assert result['nontrivial_fraction'] == .25
    logits = torch.zeros((20, 1), requires_grad=True)
    p.masked_bce(logits, torch.tensor([[-1.]]*4 + [[1.]]*16)).backward()
    np.testing.assert_array_equal(logits.grad[:4].numpy(), 0.)
    np.testing.assert_array_equal(logits.grad[4:].numpy(), -1/32)


def test_positive_to_absent_is_rejected_before_recording():
    with pytest.raises(ValueError, match='absen'):
        make_ledger(0)
    ledger = p.PolicyLedger(['a'])
    with pytest.raises(ValueError, match='absen'):
        ledger.record(example_id='0', source_id='p', source_sha256='a'*64, phase=0,
            level_bin='fixed', baseline=[1], final=[0], eligible=True, augmented=True,
            attempts=1, skipped=False)
    assert ledger.records == []
    # A rejected append must not reserve the ID.
    ledger.record(example_id='0', source_id='p', source_sha256='a'*64, phase=0,
        level_bin='fixed', baseline=[1], final=[-1], eligible=True, augmented=False,
        attempts=1, skipped=True)


def bound_corpus():
    grid = p.FeatureGrid(1000, 4, 2, 4, 0., 500.)
    target = np.sin(np.arange(32))
    background = np.cos(np.arange(64))*.0001
    rows = []
    for label, x in [('a', target), ('background', background)]:
        state = dict(state='verified_absent') if label == 'background' else dict(
            state='present', alternatives=[[dict(start=8, stop=10, mel_bins=[0])]])
        rows.append(dict(label=label, source_group=label, cue_support=dict(
            schema=p.POLICY, units='samples/mel_bins', reviewed=True, reviewer='disclosed-control',
            annotation_id=label, source_id=label, source_sha256=p.audio_digest(x, grid.fs),
            evidence_domain='synthetic proxy', feature_grid=vars(grid), classes={'a': state})))
    return SimpleNamespace(classes=['a'], rows=rows, targets={'a': [target]}, background=[background]), vars(grid)


def test_matching_target_and_background_parents_bind():
    corpus, cfg = bound_corpus()
    p.bind_corpus(corpus, cfg)
    assert corpus.cue_annotations['a'][0].source_id == 'a'
    assert corpus.cue_background_annotations[0].source_id == 'background'


@pytest.mark.parametrize('row_index', [0, 1], ids=['target', 'background'])
@pytest.mark.parametrize('parent', ['wrong-parent', None, ''], ids=['wrong', 'missing', 'empty'])
def test_parent_mismatch_or_absence_fails_closed(row_index, parent):
    corpus, cfg = bound_corpus()
    if parent is None:
        del corpus.rows[row_index]['source_group']
    else:
        corpus.rows[row_index]['source_group'] = parent
    with pytest.raises(ValueError, match='parent'):
        p.bind_corpus(corpus, cfg)
    assert not hasattr(corpus, 'cue_annotations')


def paired_records(lost=4, accepted=5):
    records = []
    for c in range(2):
        for i in range(20):
            baseline = [0., 0.]; baseline[c] = 1.
            final = baseline.copy(); final[c] = -1. if i < lost else 1.
            common = dict(source_id=f'parent-{c}', source_sha256=str(c)*64,
                annotation_sha256='a'*64, pair_context=dict(background_window_sha256='b'*64,
                mixer_rng_sha256='c'*64), level_db=65.)
            records.append(dict(common, example_id=f'{c}-{i}', phase=0, states=final,
                paired_baseline=dict(common, protocol=p.PAIRING, states=baseline,
                    audio_sha256='d'*64, source_start=0),
                crop_eligible=False, crop_accepted=False, crop_proposals=[],
                mask_eligible=True, mask_accepted=i < accepted,
                mask_attempts=[dict(changed=True, reason='accepted' if i < accepted else 'cue_erased')]))
    return records


def test_paired_diagnostics_boundary_and_smallest_count_defects():
    good = p.training_diagnostics(paired_records(), ['a', 'b'], require_masks=True)
    assert good['paired_guardrails']['present_mass_fraction'] == {'a': .8, 'b': .8}
    assert good['paired_guardrails']['nontrivial_fraction'] == .25
    assert good['paired_guardrails']['verified_absence_cells'] == 40
    assert good['paired_guardrails']['parent_groups'] == 2  # not 40 independent groups
    p.require_paired_guardrails(good)
    bad = p.training_diagnostics(paired_records(lost=5), ['a', 'b'], require_masks=True)
    assert bad['paired_guardrails']['present_mass_fraction'] == {'a': .75, 'b': .75}
    with pytest.raises(ValueError, match='paired.*0.80'):
        p.require_paired_guardrails(bad)
    with pytest.raises(ValueError, match='0.25'):
        p.training_diagnostics(paired_records(accepted=4), ['a', 'b'], require_masks=True)


@pytest.mark.parametrize('defect', ['missing', 'protocol', 'parent', 'bed', 'rng', 'level',
                                  'duplicate', 'absence', 'new_absence', 'missing_attempt'])
def test_paired_diagnostics_rejects_unbound_or_corrupted_rows(defect):
    rows = paired_records()
    if defect == 'missing': del rows[0]['paired_baseline']
    elif defect == 'protocol': rows[0]['paired_baseline']['protocol'] = 'unregistered'
    elif defect == 'parent': rows[0]['paired_baseline']['source_id'] = 'other'
    elif defect in ('bed', 'rng'):
        rows[0]['paired_baseline']['pair_context'] = dict(rows[0]['pair_context'])
        key = 'background_window_sha256' if defect == 'bed' else 'mixer_rng_sha256'
        rows[0]['paired_baseline']['pair_context'][key] = 'e'*64
    elif defect == 'level': rows[0]['paired_baseline']['level_db'] = 66.
    elif defect == 'duplicate': rows[1]['example_id'] = rows[0]['example_id']
    elif defect == 'absence': rows[0]['states'][1] = -1.
    elif defect == 'new_absence': rows[0]['states'][0] = 0.
    elif defect == 'missing_attempt': rows[0]['mask_attempts'] = []
    with pytest.raises(ValueError):
        p.training_diagnostics(rows, ['a', 'b'], require_masks=True)


def test_missing_paired_denominator_never_uses_final_counts():
    rows = [dict(states=[1 if i == 0 else -1, 0], mask_eligible=False,
                 mask_accepted=False, mask_attempts=[]) for i in range(20)]
    rows += [dict(states=[0, 1 if i == 0 else -1], mask_eligible=False,
                  mask_accepted=False, mask_attempts=[]) for i in range(20)]
    with pytest.raises(ValueError, match='paired comparator missing'):
        p.training_diagnostics(rows, ['a', 'b'], require_masks=False)


def test_combined_proposal_counts_every_retry_without_double_counting_crop():
    rows = paired_records(lost=0, accepted=20)
    for row in rows:
        row.update(crop_eligible=True, crop_accepted=True,
            crop_proposals=[dict(changed=True, accepted=True)])
        row['mask_attempts'] = [dict(changed=True, reason='cue_erased')]*3 + [dict(changed=True, reason='accepted')]
    good = p.training_diagnostics(rows, ['a', 'b'], require_masks=True)['paired_guardrails']
    assert good['eligible'] == 160 and good['augmented'] == 40
    assert good['retries'] == 120 and good['nontrivial_fraction'] == .25
    for row in rows:
        row['mask_attempts'].insert(0, dict(changed=True, reason='cue_erased'))
    with pytest.raises(ValueError, match='0.25'):
        p.training_diagnostics(rows, ['a', 'b'], require_masks=True)


def test_consumed_labels_match_trace_and_missing_rows_fail():
    p.validate_consumed_records([dict(states=[1, 0])], np.array([[1, 0]]))
    for records in ([dict(states=[-1, 0])], [], [{}]):
        with pytest.raises(ValueError, match='consumed'):
            p.validate_consumed_records(records, np.array([[1, 0]]))
