"""Conditional spike-in detection at the evaluation's actual retained group counts."""
import numpy as np
from scipy.signal import sosfilt

from sensorsim.iot_mic import peaking
from .stream import feature_bucket


def apply_power_gate(status, sensitivity, *, test=None):
    # Callers naming a test cannot inherit another test's or an aggregate detection.
    power = sensitivity if test is None else sensitivity.get('per_test', {}).get(test, {})
    return 'inconclusive' if status == 'pass' and power.get('status') != 'detected' else status


def spike_in_report(reference, synthetic, model, *, seeds=(20261008, 20261009, 20261010)):
    from .reporting import compare_views
    if not seeds:
        raise ValueError('spike-in power needs at least one seed')
    fs = int(model.meta['features']['fs'])
    transforms = {
        'eq_3khz_plus2db': lambda x: sosfilt(peaking(3000., 2., 1.5, fs), x).astype(np.float32),
        # S06 identifies quantization as a testable preprocessing fingerprint, not an established defect.
        'pcm8_fingerprint': lambda x: (np.clip(np.round(x*127.), -128, 127)/127.).astype(np.float32),
    }
    artifacts = {}
    for name, transform in transforms.items():
        windows = [transform(x) for x in synthetic['audio_windows']]
        spiked = feature_bucket(windows, synthetic['groups'], synthetic['files'], model,
                                 kind='proxy', sampling=synthetic['window_sampling'])
        outcomes = []
        for seed in seeds:
            e1, e2 = compare_views(reference, spiked, seed=seed)
            outcomes.append({'seed': seed, 'e1': e1['status'], 'e2': e2['status'],
                             'detected': 'fail' in (e1['status'], e2['status'])})
        detected = sum(r['detected'] for r in outcomes)
        artifacts[name] = {'conditional_detection_fraction': detected/len(outcomes),
                           'detected_runs': detected, 'runs': len(outcomes), 'seed_results': outcomes,
                           'status': 'detected' if detected == len(outcomes) else 'underpowered'}
    per_test = {}
    for test in ('e1', 'e2'):
        detections = {name: sum(r[test] == 'fail' for r in row['seed_results'])
                      for name, row in artifacts.items()}
        per_test[test] = {'status': 'detected' if all(n == len(seeds) for n in detections.values()) else 'underpowered',
                          'detected_runs_by_artifact': detections, 'runs_per_artifact': len(seeds)}
    return {'status': 'detected' if all(r['status'] == 'detected' for r in per_test.values()) else 'underpowered',
            'per_test': per_test,
            'result_label': 'proxy sensitivity check', 'artifacts': artifacts,
            'artifact_detection_definition': 'Any-test detection is descriptive; only per_test qualifies a test pass.',
            'group_counts': {'reference': len(np.unique(reference['groups'])),
                             'synthetic': len(np.unique(synthetic['groups']))},
            'power_definition': 'conditional detection fraction over fixed fold/permutation seeds; not independent replication power',
            'limits': 'Only these injected artefacts are tested; no minimum detectable effect or deployment power is inferred.'}
