"""Prospective final WP1 controls; all results are procedural proxies.

The failed historical registration is retained separately. This module executes
only the fresh, committed protocol, once per explicit evidence destination.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.signal import sosfilt
from scipy.stats import binom, t

from sensorsim import edge
from sensorsim.acoustics.room.session_sim import (
    Impairments, broadband_transient, canonical_kit, simulate_session,
)
from sensorsim.evaluation.c2st import refit_group_permutation
from sensorsim.evaluation.recall import clopper_pearson
from sensorsim.evaluation.reporting import compare_views
from sensorsim.evaluation.sensitivity import apply_power_gate
from sensorsim.evaluation.stream import feature_bucket, file_features
from sensorsim.iot_mic import peaking
from tests.unit.evaluation._fixture import save_edge_model
from tests.unit.evaluation.test_acceptance_truth import _room_rir


def _registration():
    path = Path(__file__).with_name('final_preregistration.yaml')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == path.with_suffix('.sha256').read_text().split()[0]
    return yaml.safe_load(path.read_text()), digest


def _seed(root, replicate, role):
    return int(np.random.SeedSequence([root, replicate, role]).generate_state(1)[0])


def _evidence(tmp_path, name, digest):
    destination = Path(os.environ.get('SENSORSIM_EVALUATION_EVIDENCE', tmp_path))
    destination.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an acceptance attempt, including an interrupted attempt.
    with (destination/f'{name}.started.json').open('x') as stream:
        json.dump({'preregistration_sha256': digest, 'result_label': 'proxy result'}, stream)
    return destination/f'{name}.json'


def _write(path, result):
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')


@pytest.mark.slow
def test_registered_e2_refit_permutation_calibration(tmp_path):
    registration, digest = _registration()
    spec = registration['E2_calibration_regression']
    path = _evidence(tmp_path, 'e2_calibration_final', digest)
    root = registration['seeds']['calibration_root']
    count, width = spec['groups_per_side'], spec['dimensions']
    labels = np.repeat([0, 1], count)
    groups = np.arange(2*count).astype(str)
    rows = []
    for replicate in range(spec['runs']):
        rng = np.random.default_rng(_seed(root, replicate, 0))
        x = np.sqrt(.5)*rng.normal(size=(2*count, width)) + np.sqrt(.5)*rng.normal(size=(2*count, 1))
        report = refit_group_permutation(x, labels, groups, folds=spec['folds'], l2=spec['l2'],
            fold_seed=_seed(root, replicate, 1), seed=_seed(root, replicate, 2),
            replicates=spec['permutation_replicates'])
        rows.append({'replicate': replicate, **report})
        if (replicate+1) % 10 == 0:
            print(f'proxy E2 calibration: {replicate+1}/{spec["runs"]}', flush=True)
    low, high = map(int, binom.interval(.95, spec['runs'], spec['alpha']))
    rejected = sum(r['p_value'] <= spec['alpha'] for r in rows)
    result = {'result_label': 'proxy result', 'measured_results': None,
              'preregistration_sha256': digest, 'runs': spec['runs'], 'rejections': rejected,
              'binomial_acceptance_count_interval': [low, high],
              'exact_95_rate_interval': clopper_pearson(rejected, spec['runs']),
              'replicates': rows, 'acceptance': 'pass' if low <= rejected <= high else 'fail'}
    _write(path, result)
    assert result['acceptance'] == 'pass', str(path)


@pytest.mark.slow
def test_registered_final_truth_framing_control(tmp_path):
    registration, digest = _registration()
    path = _evidence(tmp_path, 'acceptance_final', digest)
    fs = registration['truth']['sample_rate_hz']
    room, speakers, microphones, rirs, _ = _room_rir(fs)
    seeds = registration['seeds']
    model = save_edge_model(tmp_path/'fixture.npz', seed=seeds['model'], width=48, blocks=4)
    count, replicates = registration['split']['groups_per_side'], registration['split']['replicates']
    result = {'result_label': 'proxy result', 'measured_results': None,
              'preregistration_sha256': digest, 'replicates': [],
              'embedding_note': 'Frozen seeded untrained CNN fixture; no deployment recognizer validation.',
              'acceptance': 'inconclusive'}
    paired, clamp_count = [], 0
    for replicate in range(replicates):
        sources = {}
        for side, length, stream in [('ref', 6., 0), ('syn', 2., 1), ('pair', 2., 0)]:
            for group in range(count):
                seed = int(np.random.SeedSequence([seeds['source_root'], replicate, stream, group]).generate_state(1)[0])
                transient = broadband_transient(fs, seed)
                take = np.zeros(round(length*fs))
                start = round(.45*fs)
                take[start:start+len(transient)] = 2*transient
                sources[f'S1_M1_real_r{replicate:02d}_{side}_{group:02d}.wav'] = take
        work = tmp_path/f'replicate-{replicate:02d}'
        truth = simulate_session(work/'session', room, speakers, microphones,
            truth_dir=work/'truth', external_rirs=rirs, real_sources=sources,
            include_reference=False, kit=canonical_kit(), case='A', seed=seeds['session'],
            impairments=Impairments(clock_drift_ppm=registration['truth']['clock_drift_ppm']))
        del sources
        analysis = [_seed(seeds['analysis_root'], replicate, role) for role in range(6)]
        buckets = {}
        for side in ('ref', 'syn', 'pair'):
            items = []
            for group in range(count):
                name = f'S1_M1_real_r{replicate:02d}_{side}_{group:02d}.wav'
                items.append({'path': truth.manifest.parent/name, 'file': name,
                              'group': f'{side}-{group:02d}', 'label': 'event'})
            buckets[side] = file_features(items, model, kind='proxy', seed=analysis[5])['event']
            assert len(buckets[side]['groups']) == count
            clamp_count += buckets[side]['window_sampling']['clamped_windows']
        paired.extend(buckets['ref']['stats'] - buckets['pair']['stats'])
        row = {'replicate': replicate, 'analysis_seeds': analysis, 'conditions': {}}
        for name, gain in [('null', 0.), ('eq_plus2db', 2.), ('eq_plus6db', 6.)]:
            synthetic = buckets['syn']
            if gain:
                windows = [sosfilt(peaking(3000., gain, 1.5, 16000), w).astype(np.float32)
                           for w in synthetic['audio_windows']]
                synthetic = feature_bucket(windows, synthetic['groups'], synthetic['files'], model,
                                            kind='proxy', sampling=synthetic['window_sampling'])
            e1, e2 = compare_views(buckets['ref'], synthetic, seed=analysis[0], permutations=999,
                bootstrap_replicates=2000, bootstrap_seeds=tuple(analysis[2:5]),
                c2st_permutations=199, c2st_permutation_seed=analysis[1])
            row['conditions'][name] = {'e1': e1, 'e2': e2,
                                      'detected': 'fail' in (e1['status'], e2['status'])}
        power = {'per_test': {test: {'status': 'detected' if all(
            row['conditions'][name][test]['status'] == 'fail' for name in ('eq_plus2db','eq_plus6db'))
            else 'underpowered'} for test in ('e1','e2')}}
        row['control_power'] = power
        row['gated_null_status'] = {test: apply_power_gate(row['conditions']['null'][test]['status'], power, test=test)
                                    for test in ('e1','e2')}
        result['replicates'].append(row)
        _write(path, result)
        print('proxy framing:', replicate+1, '/', replicates,
              {name: (r['e1']['status'], r['e2']['status']) for name, r in row['conditions'].items()}, flush=True)
    paired = np.asarray(paired)
    np.save(path.with_name('paired_length_differences.npy'), paired)
    means, errors = paired.mean(0), paired.std(0, ddof=1)/np.sqrt(len(paired))
    statistics = np.divide(means, errors, out=np.zeros_like(means), where=errors != 0)
    degenerate_nonzero = (errors == 0) & (means != 0)
    cutoff = float(t.ppf(1-.05/(2*paired.shape[1]), len(paired)-1))
    exceedances = int(np.sum((np.abs(statistics) > cutoff) | degenerate_nonzero))
    counts = {name: {'any_fail': sum(row['conditions'][name]['detected'] for row in result['replicates']),
                    **{test+'_fail': sum(row['conditions'][name][test]['status'] == 'fail' for row in result['replicates'])
                       for test in ('e1','e2')}} for name in ('null','eq_plus2db','eq_plus6db')}
    descriptive = {test: {status: sum(row['conditions']['null'][test]['status'] == status for row in result['replicates'])
                         for status in ('pass','fail','inconclusive')} for test in ('e1','e2')}
    gated = {test: {status: sum(row['gated_null_status'][test] == status for row in result['replicates'])
                   for status in ('pass','fail','inconclusive')} for test in ('e1','e2')}
    criteria = registration['criteria']
    parts = {'specificity': counts['null']['any_fail'] <= criteria['null_any_fail_max'],
             'paired_length': exceedances <= criteria['paired_bonferroni_exceedances_max'],
             'plus2db': counts['eq_plus2db']['any_fail'] >= criteria['plus2db_detected_min'],
             'plus6db': counts['eq_plus6db']['any_fail'] >= criteria['plus6db_detected_min']}
    result.update(counts=counts, parts=parts, candidate_null_status_counts=descriptive,
                  power_gated_null_status_counts=gated, clamped_windows=clamp_count,
                  paired_length={'pairs': len(paired), 'features': paired.shape[1], 'cutoff': cutoff,
                                 'max_abs_t': float(np.max(np.abs(statistics))),
                                 'constant_nonzero_features': int(degenerate_nonzero.sum()),
                                 'bonferroni_exceedances': exceedances, 't_values': statistics.tolist()},
                  acceptance='pass' if all(parts.values()) else 'fail')
    _write(path, result)
    assert result['acceptance'] == 'pass', str(path)
