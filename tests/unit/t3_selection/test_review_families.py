# SPDX-License-Identifier: Apache-2.0
"""Disclosed development regressions derived from verifier F1 counterexamples."""
import copy
from hashlib import sha256
import json

import numpy as np
import pytest

from tests.unit.t3_selection.test_selection import make_bank, candidate, constraints, scores


def load(path, data, role):
    from sensorsim.evaluation.selection import build_validation_bank
    path.write_text(json.dumps(data))
    return build_validation_bank(path, expected_sha256=sha256(path.read_bytes()).hexdigest(), role=role)


def allocation(bank, *, optional):
    data = copy.deepcopy(bank.manifest)
    data['required_families'] = dict(train=['fan', 'dishes'], development=['fan'], audit=['dishes'])
    if optional:
        data['optional_families'] = dict(train=[], development=['dishes'], audit=['fan'])
    return data


@pytest.mark.parametrize('role', ['development', 'audit'])
def test_family_must_be_allocated_to_its_own_role(tmp_path, role):
    bank, path = make_bank(tmp_path, role=role)
    good = load(path, allocation(bank, optional=True), role)
    c = candidate()
    assert __import__('sensorsim.evaluation.selection', fromlist=['score_candidate']).score_candidate(
        c, good, constraints(), score_cache=scores(good, c))['status'] == 'feasible'
    with pytest.raises(ValueError, match='family.*role'):
        load(path, allocation(bank, optional=False), role)


@pytest.mark.parametrize('role', ['development', 'audit'])
def test_optional_nominal_outputs_and_parents_cannot_disappear(tmp_path, role):
    from sensorsim.evaluation.selection import cache_scores, score_candidate, paired_parent_improvement
    bank, path = make_bank(tmp_path, role=role)
    bank = load(path, allocation(bank, optional=True), role)
    c = candidate()
    seq = {k: v.copy() for k, v in scores(bank, c)['sequences'].items()}
    good = score_candidate(c, bank, constraints(), score_cache=cache_scores(c, bank, seq))
    hidden = 'dishes' if role == 'development' else 'fan'
    row = next(r for r in bank.rows if r['kind'] == 'nuisance' and r['family'] == hidden)
    assert good['status'] == 'feasible' and good['nuisance_starts'] == good['nuisance_confirmed'] == 0
    seq[row['id']][:] = .9
    bad = score_candidate(c, bank, constraints(), score_cache=cache_scores(c, bank, seq))
    observed = next(r for r in bad['outcomes'] if r['id'] == row['id'])
    assert observed['starts'] == observed['confirmed'] == 1  # actual Tracker output reached
    assert bad['nuisance_starts'] == bad['nuisance_confirmed'] == 1
    assert good['nuisance_parent_trigger'][row['parent_id']] is False
    assert bad['nuisance_parent_trigger'][row['parent_id']] is True
    assert bad['families'][hidden]['required'] is False
    assert bad['families'][hidden]['starts'] == bad['families'][hidden]['confirmed'] == 1
    assert bad['status'] == 'infeasible'
    assert paired_parent_improvement(bad, good)['worsened'] == 1


def test_outer_screen_rejects_distinct_candidate_extra_optional_alarm(tmp_path):
    from sensorsim.evaluation.selection import candidate_identity, cache_scores, score_candidate, outer_screen
    bank, path = make_bank(tmp_path, role='audit', fs=1, background_s=7200)
    bank = load(path, allocation(bank, optional=True), 'audit')
    def identity(step):
        return candidate_identity(model_sha256=sha256(str(step).encode()).hexdigest(),
            features=dict(fs=1, window_s=1), classes=['knock'],
            policy=dict(on=.5, off_ratio=.6, smooth_hops=1, min_hops=1, hop_s=1),
            step=step, seed=1103, code_revision='disclosed-verifier-F1')
    c, other = identity(250), identity(500)
    seq = {r['id']: np.zeros((r['sample_count'], 1)) for r in bank.rows}
    for r in bank.rows:
        if r['kind'] == 'target':
            seq[r['id']][1:3] = .9
    limits = constraints(min_background_hours=2)
    good = score_candidate(c, bank, limits, score_cache=cache_scores(c, bank, seq))
    assert outer_screen(good, good)['outcome'] == 'pass'
    seq['audit-nuisance-fan'][:] = .9
    bad = score_candidate(other, bank, limits, score_cache=cache_scores(other, bank, seq))
    assert bad['candidate']['candidate_sha256'] != good['candidate']['candidate_sha256']
    assert sum(r['starts'] for r in bad['outcomes'] if r['kind'] == 'nuisance') == 1
    result = outer_screen(bad, good)
    assert result['outcome'] == 'fail'
    assert {'additional nuisance starts', 'additional nuisance confirmed'} <= set(result['violations'])


@pytest.mark.parametrize('field, value', [
    ('required_families', dict(train=['fan', 'dishes'], development=['fan', 'dishes'])),
    ('required_families', dict(train=['fan', 'dishes'], development=['fan', 'dishes'], audit='fan')),
    ('optional_families', dict(train=[], development=['dishes', 'dishes'], audit=['fan'])),
    ('optional_families', dict(train=[], development=['fan'], audit=['fan'])),
])
def test_malformed_role_allocations_are_rejected(tmp_path, field, value):
    bank, path = make_bank(tmp_path)
    data = allocation(bank, optional=True)
    data[field] = value
    with pytest.raises(ValueError):
        load(path, data, 'development')
