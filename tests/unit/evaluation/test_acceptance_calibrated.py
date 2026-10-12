"""Preregistered repair acceptance. All results are generated proxies, never measured."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.stats import binom

from sensorsim import edge, project
from sensorsim.acoustics.room.session_sim import canonical_kit, simulate_session, broadband_transient
from sensorsim.evaluation.realism import realism_report
from sensorsim.evaluation.reporting import compare_views
from sensorsim.evaluation.sensitivity import spike_in_report, apply_power_gate
from sensorsim.iot_mic import peaking
from tests.unit.evaluation._fixture import save_edge_model, write_float_wav
from tests.unit.evaluation.test_acceptance_truth import _room_rir


def _draw(rng, shift):
    x = []
    for _ in range(15):
        x.append(shift + rng.normal(scale=np.sqrt(.2), size=120)
                 + rng.normal(scale=np.sqrt(.8), size=(6, 120)))
    return np.vstack(x)


def _write_evidence(tmp_path, result):
    evidence = tmp_path/'acceptance_calibrated.json'
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    if os.environ.get('SENSORSIM_EVALUATION_EVIDENCE'):
        destination = Path(os.environ['SENSORSIM_EVALUATION_EVIDENCE'])
        destination.mkdir(parents=True, exist_ok=True)
        (destination/evidence.name).write_bytes(evidence.read_bytes())
    return evidence


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="registered result was FAIL (F2 framing); its all-replicates criterion "
                   "passes about 2.4% of the time under a true null. Kept as the historical record; the "
                   "prospective control in test_acceptance_final.py replaced it.")
def test_registered_calibrated_proxy_acceptance(tmp_path):
    registration = Path(__file__).with_name('preregistration.yaml')
    digest = hashlib.sha256(registration.read_bytes()).hexdigest()
    assert digest == Path(__file__).with_name('preregistration.sha256').read_text().split()[0]
    result = {'result_label': 'proxy result', 'measured_results': None,
              'preregistration_sha256': digest, 'parts': {}}
    groups = np.repeat(np.arange(15).astype(str), 6)
    calibrations = []
    for shift, runs in [(0., 300), (.6, 100)]:
        rows = []
        delta = np.zeros(120); delta[:10] = shift
        for replicate in range(runs):
            rng = np.random.default_rng([15, 20, int(shift*1000), replicate])
            x, y = _draw(rng, np.zeros(120)), _draw(rng, delta)
            report = realism_report(x, y, reference_groups=groups, rendered_groups=groups,
                                     n_bands=40, permutations=199, seed=20261008)
            rows.append({'replicate': replicate, 'status': report['status'],
                         'joint_p': report['decision_null']['permutation_p'],
                         'mmd_p': report['mmd']['permutation_p']})
        calibrations.append({'shift_sd_in_10_dimensions': shift, 'runs': runs,
                             'rejections': sum(r['status'] == 'fail' for r in rows), 'replicates': rows})
    lower, upper = map(int, binom.interval(.95, 300, .05))
    result['calibration'] = calibrations
    result['null_acceptance_count_interval'] = [lower, upper]
    result['parts']['f1_calibration'] = lower <= calibrations[0]['rejections'] <= upper
    result['parts']['f1_power'] = calibrations[1]['rejections'] >= 80
    _write_evidence(tmp_path, result)

    fs = 48000
    room, speakers, microphones, rirs, _ = _room_rir(fs)
    sources = {}
    for replicate in range(3):
        for side, length in [('ref', 6.), ('ren', 2.)]:
            for i in range(15):
                seed = np.random.SeedSequence([9191, replicate, 0 if side == 'ref' else 1, i]).generate_state(1)[0]
                transient = broadband_transient(fs, int(seed))
                take = np.zeros(round(length*fs)); start = round(.45*fs)
                take[start:start+len(transient)] = transient*2.
                sources[f'S1_M1_real_r{replicate:02d}_{side}_{i:02d}.wav'] = take
    truth = simulate_session(tmp_path/'session', room, speakers, microphones,
        truth_dir=tmp_path/'truth', external_rirs=rirs, real_sources=sources,
        include_reference=False, kit=canonical_kit(), case='A', seed=9191)
    model = save_edge_model(tmp_path/'m.npz')
    framing = []
    sensitivity = None
    for replicate in range(3):
        features = {}
        for side in ['ref', 'ren']:
            items = []
            for i in range(15):
                name = f'S1_M1_real_r{replicate:02d}_{side}_{i:02d}.wav'
                items.append({'path': truth.manifest.parent/name, 'file': name,
                              'label': 'event', 'group': f'{side}-{i}'})
            features[side] = project._feature_rows(items, model, kind='proxy')['event']
        same1, same2 = compare_views(features['ref'], features['ren'])
        shifted_items = []
        for i in range(15):
            name = f'S1_M1_real_r{replicate:02d}_ren_{i:02d}.wav'
            audio = edge.read_wav(str(truth.manifest.parent/name), 16000)
            shifted = signal.sosfilt(peaking(3000., 6., 1.5, 16000), audio)
            path = tmp_path/f'shifted-{replicate}-{i}.wav'
            write_float_wav(path, shifted)
            shifted_items.append({'path': path, 'file': path.name, 'label': 'event', 'group': f'ren-{i}'})
        shifted_features = project._feature_rows(shifted_items, model, kind='proxy')['event']
        shift1, shift2 = compare_views(features['ref'], shifted_features)
        framing.append({'replicate': replicate, 'same': {'e1': same1, 'e2': same2},
                        'shifted': {'e1': shift1, 'e2': shift2}})
        if replicate == 0:
            sensitivity = spike_in_report(features['ref'], features['ren'], model)
    result['framing'] = framing
    result['spike_in'] = sensitivity
    result['parts']['f2_framing'] = all(r['same'][e]['status'] == 'pass' for r in framing for e in ['e1', 'e2'])
    result['parts']['f2_shift'] = all(r['shifted'][e]['status'] == 'fail' for r in framing for e in ['e1', 'e2'])
    result['parts']['f6_spike_contract'] = (sensitivity['status'] == 'detected' or
        (sensitivity['status'] == 'underpowered' and apply_power_gate('pass', sensitivity) == 'inconclusive'))
    result['acceptance'] = 'pass' if all(result['parts'].values()) else 'fail'
    evidence = _write_evidence(tmp_path, result)
    assert result['acceptance'] == 'pass', str(evidence)
