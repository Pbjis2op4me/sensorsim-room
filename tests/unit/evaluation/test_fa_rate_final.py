"""Final WP2 proxy regressions; generated signals, no measured deployment data."""
import hashlib
import json

import numpy as np
import pytest

from sensorsim.evaluation import fa_rate
from test_fa_rate_polish import ScoreModel, row, write_wav


@pytest.mark.parametrize('count', [0, 1, 8])
def test_n2_constant_bound_uses_one_component_error_budget(count):
    rows = [row(i, count) for i in range(100)]
    actual = fa_rate._count_interval(rows, np.full(100, count), .9875)
    q = 1 - .0125 ** (1 / 99)
    assert actual == pytest.approx([count * (1 - q), count + (8 - count) * q])
    result = fa_rate.summarize_challenges(rows, context_weights={'isolated': 1},
        bootstrap_replicates=100)['families']['impact']['target']
    assert result['status'] == 'scored'


def test_n2_never_firing_detector_has_tighter_but_still_large_bound():
    rows = [{**row(i, 0, family=f), 'max_confirmed_outputs': 31}
            for f in ('impact', 'flow', 'appliance') for i in range(100)]
    result = fa_rate.estimate_composite(fa_rate.summarize_floor([], 3.5 * 3600), rows,
        {'impact': 2, 'appliance': 5, 'flow': 10}, context_weights={'isolated': 1},
        bootstrap_replicates=100)['classes']['target']
    expected = -np.log(.0125 / 2) / 3.5 + 17 * 31 * (1 - .0125 ** (1 / 99))
    assert result['upper_bound_per_hour'] == pytest.approx(expected)
    assert result['status'] == 'scored'
    assert result['upper_bound_per_hour'] > 24  # no <=1/h claim


def test_n2_rate_free_headline_counts_triggering_whole_episodes_not_outputs():
    rows = [row(i, 3 if i % 2 else 0) for i in range(100)]
    result = fa_rate.estimate_composite(fa_rate.summarize_floor([], 3600), rows,
        {'impact': 2}, context_weights={'isolated': 1}, bootstrap_replicates=100)
    cls = result['classes']['target']
    family = cls['families']['impact']
    assert family['rate_free_headline']['triggers_per_100_nuisance_events'] == 50
    assert family['outputs_per_100_contexts'] == 150
    assert family['rate_free_headline']['denominator'] == 'whole nuisance episodes (parent contexts), not annotated sources'
    assert family['rate_free_headline']['interval_per_100'] == pytest.approx(
        np.asarray(fa_rate.clopper_pearson(50, 100)) * 100)
    assumed = cls['outputs_per_trigger_assumption']
    assert assumed['assumed_max_outputs_per_trigger'] == 1
    assert assumed['assumption_contradicted_by_bank'] is True
    assert assumed['upper_bound_per_hour'] is None
    assert cls['rate_per_hour'] == 3


def test_n2_dependent_rate_free_rows_are_not_independent_bernoulli_trials():
    rows = [row(i, i // 5 % 2, parent=str(i // 5)) for i in range(100)]
    result = fa_rate.summarize_challenges(rows, context_weights={'isolated': 1},
        bootstrap_replicates=100)['families']['impact']['target']
    assert result['rate_free_headline']['interval_method'] == 'parent-cluster bootstrap; full range for constant dependent counts'
    assert result['status'].startswith('inconclusive')


def test_n3_measured_smoke_floor_headline_has_no_rate_point():
    report = fa_rate.summarize_floor([1], 60, source_kind='measured')
    assert report['observed_count_rate_per_hour'] == 60
    assert report['headline']['point_per_hour'] is report['rate_per_hour'] is None
    assert report['headline']['evidence_label'] == 'measured-smoke-test-not-FA/hour'


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(fa_rate.edge, 'Model', lambda _: ScoreModel())
    model = tmp_path / 'model.npz'
    model.write_bytes(b'generated score-proxy model identity')
    write_wav(tmp_path / 'floor.wav', np.random.default_rng(18).uniform(-.002, .002, 48))
    write_wav(tmp_path / 'context.wav', np.r_[np.zeros(8), [.9, .9], np.zeros(38)])
    manifest = dict(source_kind='proxy', calibration_streams=[], floor_streams=[{'path': 'floor.wav'}],
        challenges=[dict(path='context.wav', family='impact')], context_weights={'isolated': 1},
        context_prevalence_per_hour={'impact': 2}, bootstrap_replicates=100)
    for entry in [*manifest['floor_streams'], *manifest['challenges']]:
        entry.update(recording_id=entry['path'], sample_range=[0, 48], sample_rate_hz=4)
    return model, tmp_path / 'manifest.json', manifest


def test_n3_cli_records_effective_detector_and_exact_model_hash(cli_inputs):
    model, path, manifest = cli_inputs
    manifest['thresholds'] = {'smooth_hops': 3}
    path.write_text(json.dumps(manifest))
    report = fa_rate._score_manifest(str(model), str(path))
    assert report['model_sha256'] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert report['detector_settings'] == {**ScoreModel.meta['detector'], 'smooth_hops': 3}
    assert report['model_detector_settings'] == ScoreModel.meta['detector']
    assert report['detector_overrides'] == {'smooth_hops': 3}
    assert report['bank_protocol']['bed'] == 'overlay nuisances on user target-free segments before scoring'


@pytest.mark.parametrize('key,value', [('detector_settings', {'smooth_hops': 9}), ('model_sha256', 'bad')])
def test_n3_cli_rejects_conflicting_challenge_provenance(cli_inputs, key, value):
    model, path, manifest = cli_inputs
    manifest['challenges'][0][key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=key):
        fa_rate._score_manifest(str(model), str(path))


def test_n3_summary_rejects_mixed_detector_settings():
    rows = [dict(row(i, i % 2), detector_settings={'smooth_hops': 1 if i else 3}) for i in range(100)]
    with pytest.raises(ValueError, match='detector_settings'):
        fa_rate.summarize_challenges(rows, context_weights={'isolated': 1}, bootstrap_replicates=100)


def test_default_sensitivity_varies_one_family_and_all_level_mixes():
    rows = [row(i, 0, context='low') for i in range(100)]
    rows += [row(i, 2, context='high') for i in range(100)]
    rows += [row(i, 1, family='flow', context='steady') for i in range(100)]
    result = fa_rate.estimate_composite(fa_rate.summarize_floor([], 3600), rows,
        {'impact': 2, 'flow': 4}, context_weights={'impact': {'low': .8, 'high': .2}, 'flow': {'steady': 1}},
        bootstrap_replicates=100)['classes']['target']
    assert result['rate_per_hour'] == pytest.approx(4.8)
    assert result['family_incidence_sensitivity']['impact']['3.0']['rate_per_hour'] == pytest.approx(6.4)
    assert result['level_mix_sensitivity']['rate_per_hour_range'] == [4, 8]
    assert result['level_mix_sensitivity']['scope'] == 'all mixtures of supplied strata; not unseen levels or families'


def test_cli_help_explains_applicable_bank_mitigations(capsys):
    with pytest.raises(SystemExit) as exc:
        fa_rate.main(['--help'])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for phrase in ('target-free', 'whole nuisance episode', 'mixed waveform', '0.3', '3', 'rough', 'level'):
        assert phrase in help_text
