"""Disclosed verifier counterexamples, now development regressions (not receipts)."""
from copy import deepcopy

import numpy as np
import pytest

from sensorsim.evaluation import power_audit as p
from tools.power_audit import procedural_factory


def matched():
    a, b = procedural_factory(4, 'null', 995019)
    for i, (left, right) in enumerate(zip(a['records'], b['records'])):
        for key in ('source_group', 'session_id', 'independence_group', 'condition_id'):
            right[key] = left[key]
        for row in (left, right):
            row.update(pair_id=f'pair-{i}', acquisition_id=f'acquisition-{i}',
                       source_position=[1., 2., 1.], microphone_position=[2., 2., 1.],
                       coordinate_convention='room_cartesian_metres',
                       framing_id=f'scheduled-frame-{i}', framing_basis='external_schedule')
    return a, b


def contract_qualification(raw):
    # Hypothetical metadata to exercise transfer only; never an acceptance receipt.
    return dict(family_sha256=raw['family_sha256'],
                implementation_sha256=raw['implementation_sha256'],
                feature_configuration_sha256=raw['feature_configuration_sha256'],
                qualification_scope_sha256=raw.get('qualification_scope_sha256'),
                evidence_domain='synthetic proxy', cells=[dict(groups=4, harness='qualified')])


def test_correct_pair_reaches_all_twelve_statistics(monkeypatch):
    calls = []
    original = p._paired_baseline
    def observe(*args, **kwargs):
        calls.append(args[4])
        return original(*args, **kwargs)
    monkeypatch.setattr(p, '_paired_baseline', observe)
    raw = p.run_family(*matched(), p.Family(mode='paired', permutations=19))
    assert raw['valid'], raw['errors']
    assert raw['observed_members'] == 12
    assert calls.count('mmd') == 6 and calls.count('c2st') == 3


@pytest.mark.parametrize('field,value', [
    ('acquisition_id', ''), ('framing_id', ''),
    ('source_position', 'unknown'), ('microphone_position', [1., 2.]),
    ('pair_id', '  '), ('pair_id', 123), ('pair_id', []),
    ('acquisition_id', ' \t'), ('framing_id', 17),
    ('source_position', [1., np.nan, 2.]), ('source_position', [1., np.inf, 2.]),
    ('source_position', [True, 2., 3.]), ('microphone_position', ['1', '2', '3']),
    ('coordinate_convention', 'centimetres'),
])
def test_invalid_metadata_never_invokes_statistic(field, value, monkeypatch):
    bundles = matched()
    for bundle in bundles:
        for row in bundle['records']:
            row[field] = value
    calls = []
    original = p._max_mean
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(p, '_max_mean', observe)
    raw = p.run_family(*bundles, p.Family(mode='paired', permutations=19))
    assert not raw['valid'], (field, value, raw['observed_members'])
    assert raw['difference'] == 'not_assessed'
    assert raw['observed_members'] == 0 and not calls
    assert any(field in error for error in raw['errors'])


def test_qualification_cannot_transfer_to_changed_acquisition():
    spec = p.Family(permutations=19)
    bundles = procedural_factory(4, 'null', 995022)
    raw = p.run_family(*bundles, spec)
    qualified = contract_qualification(raw)
    assert p.infer_difference(raw, qualification=qualified)['harness'] == 'qualified'
    different = deepcopy(bundles)
    for bundle in different:
        for row in bundle['records']:
            row['condition_id'] = 'different-unqualified-room-device-generator'
    other = p.run_family(*different, spec)
    assert other['valid'], other['errors']
    actual = p.infer_difference(other, qualification=qualified)
    assert actual['harness'] == 'unqualified'
    assert actual['low_power']
    assert actual['effect_specific_power'] == []


def test_missing_scope_cannot_qualify():
    raw = p.run_family(*procedural_factory(4, 'null', 995022), p.Family(permutations=19))
    q = contract_qualification(raw)
    q.pop('qualification_scope_sha256')
    assert p.infer_difference(raw, qualification=q)['harness'] == 'unqualified'


@pytest.mark.parametrize('change', ['generator_version', 'generator_code', 'generator_config',
                                    'null_population', 'effect_magnitude', 'missing', 'malformed'])
def test_design_scope_changes_do_not_inherit_power(change):
    spec = p.Family(permutations=19)
    bundles = procedural_factory(4, 'null', 995022)
    raw = p.run_family(*bundles, spec)
    q = contract_qualification(raw)
    assert raw['qualification_scope_sha256']
    assert p.infer_difference(raw, qualification=q)['harness'] == 'qualified'
    for bundle in bundles:
        scope = bundle['qualification_scope']
        if change == 'generator_version':
            scope['generator']['version'] = 'different/2'
        elif change == 'generator_code':
            scope['generator']['code_sha256'] = '0'*64
        elif change == 'generator_config':
            scope['generator']['configuration']['amplitude'] = .01
        elif change == 'null_population':
            scope['null_population']['condition_id'] = 'another-population'
        elif change == 'effect_magnitude':
            scope['effects']['scalar_plus6db']['gain_db'] = 5.
        elif change == 'missing':
            del bundle['qualification_scope']
        else:
            scope['generator']['code_sha256'] = 'invalid'
    other = p.run_family(*bundles, spec)
    assert other['valid']
    assert p.infer_difference(other, qualification=q)['harness'] == 'unqualified'


def test_scope_survives_new_independent_draw_and_is_bound_in_qualification():
    from tools.power_audit import procedural_scope_identity
    spec = p.Family(permutations=19)
    raw = p.run_family(*procedural_factory(4, 'null', 995022), spec)
    other = p.run_family(*procedural_factory(4, 'null', 995023), spec)
    assert raw['qualification_scope_sha256'] == other['qualification_scope_sha256'] == procedural_scope_identity()
    assert p.infer_difference(other, qualification=contract_qualification(raw))['harness'] == 'qualified'
    result = p.qualify_test(procedural_factory, spec, p.Budget(group_counts=(4,), null_datasets=1,
                                                            alternative_datasets=1))
    assert result['qualification_scope_sha256'] == raw['qualification_scope_sha256']
    assert all(row['result']['valid'] for row in result['attempts'])
    assert result['cells'][0]['harness'] == 'unqualified'  # no independent signoff or power evidence


def test_generator_scope_drift_invalidates_qualification():
    def drift(groups, effect, seed):
        bundles = procedural_factory(groups, effect, seed)
        if effect != 'null':
            for bundle in bundles:
                bundle['qualification_scope']['generator']['version'] = 'unregistered/2'
        return bundles
    result = p.qualify_test(drift, p.Family(permutations=19),
                           p.Budget(group_counts=(4,), null_datasets=1, alternative_datasets=1))
    assert result['attempts'][0]['result']['valid']
    assert all(not row['result']['valid'] for row in result['attempts'][1:])
    assert result['cells'][0]['harness'] == 'invalid'
