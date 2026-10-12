"""Historical WP2 proxy replay; current gates do not confer new acceptance."""
import json

import numpy as np
import pytest

from sensorsim import edge
from sensorsim.evaluation import fa_rate


HOP_S = 0.25
WINDOWS_PER_HOUR = int(3600 / HOP_S)
STREAM_HOURS = 6
THRESHOLDS = (0.35, 0.50, 0.65)
FAMILY_RATES = {"impact": 2.0, "appliance": 5.0, "flow": 10.0}
FAMILY_NAMES = tuple(FAMILY_RATES)


class _ProxyScoreModel:
    classes = ["target"]
    features = type("_Features", (), {"n": 4})()
    meta = {
        "features": {"fs": 4},
        "detector": {
            "hop_s": HOP_S,
            "on": {"target": 0.5},
            "off_ratio": 0.6,
            "smooth_hops": 1,
            "min_hops": 2,
        },
    }

    def scores(self, windows):
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 1:
            return np.asarray([[windows[0]]], dtype=np.float32)
        return windows[:, :1].copy()


def _stream_events(model, scores, threshold):
    # The proxy model reads the first sample of each four-sample window as its
    # score.  This makes the seeded score stream exercise edge.Detector's real
    # chunk buffering and Tracker hysteresis without allocating six hours of PCM.
    audio = np.concatenate((np.asarray(scores[:, 0], dtype=np.float32), np.zeros(3, dtype=np.float32)))
    detector = edge.Detector(model, thresholds={"on": {"target": threshold}})
    events = []
    chunk_size = 20  # five seconds at the proxy model's 4 Hz sample rate
    for start in range(0, audio.size, chunk_size):
        events.extend(event for event in detector.push(audio[start : start + chunk_size]) if event.get("confirmed"))
    events.extend(event for event in detector.flush() if event.get("confirmed"))
    return events


def _place_pulse(scores, onset_s, amplitude):
    index = int(round(onset_s / HOP_S))
    scores[index : index + 2, 0] = np.maximum(scores[index : index + 2, 0], amplitude)


def _floor_background(n_windows, rng, *, stress):
    levels_db = np.zeros(n_windows, dtype=np.float32)
    if stress:
        changes = sorted(rng.integers(1, n_windows, size=20).tolist())
        starts = [0] + changes
        ends = changes + [n_windows]
        for start, end in zip(starts, ends):
            level_db = rng.uniform(-10.0, 10.0)
            levels_db[start:end] = level_db
    # Synthetic background model scores stay below the preregistered on
    # thresholds; the steps vary by up to +/-10 dB-equivalent in stress.
    baseline = (0.04 * np.power(10.0, levels_db / 20.0)).astype(np.float32)
    return baseline.reshape(-1, 1)


def _make_six_hour_stream(world, seed):
    rng = np.random.default_rng(seed)
    n_windows = WINDOWS_PER_HOUR * STREAM_HOURS
    scores = _floor_background(n_windows, rng, stress=world == "stress")
    floor_count = int(rng.poisson(0.25 * STREAM_HOURS))
    floor_times = rng.uniform(0, STREAM_HOURS * 3600, floor_count)
    for onset in floor_times:
        _place_pulse(scores, onset, rng.uniform(0.82, 0.98))

    events = []
    if world == "independent":
        for family, rate in FAMILY_RATES.items():
            count = int(rng.poisson(rate * STREAM_HOURS))
            for onset in rng.uniform(0, STREAM_HOURS * 3600, count):
                events.append((family, float(onset), float(rng.uniform(0.20, 1.0))))
    else:
        # Six 20-minute activity periods are one third of the six-hour stream.
        activity_windows = [
            (period * 3600, period * 3600 + 1200)
            for period in range(6)
        ]
        bursts = []
        for family, rate in FAMILY_RATES.items():
            n_events = int(round(rate * STREAM_HOURS))
            assert n_events % 3 == 0
            for burst_index in range(n_events // 3):
                activity_start, activity_end = activity_windows[burst_index % len(activity_windows)]
                start = rng.uniform(activity_start, activity_end - 30)
                shared_strength = rng.uniform(0.20, 1.0)
                bursts.append((family, float(start), shared_strength))
        # Ten pairs overlap: 20 of 102 event instances, rounded to 19.6%.
        overlap_bursts = set(rng.choice(len(bursts), size=10, replace=False).tolist())
        for burst_index, (family, start, shared_strength) in enumerate(bursts):
            offsets = (0.0, 0.0, 16.0) if burst_index in overlap_bursts else (0.0, 8.0, 16.0)
            for offset in offsets:
                amplitude = float(np.clip(shared_strength + rng.normal(0, 0.025), 0.20, 1.0))
                events.append((family, start + offset, amplitude))

    for _, onset, amplitude in events:
        _place_pulse(scores, onset, amplitude)
    return scores


def _make_floor_sample(world, seed, threshold):
    rng = np.random.default_rng(seed)
    exposure_s = 1200.0
    n_windows = int(exposure_s / HOP_S)
    scores = _floor_background(n_windows, rng, stress=world == "stress")
    count = int(rng.poisson(0.25 * exposure_s / 3600))
    for onset in rng.uniform(0, exposure_s, count):
        _place_pulse(scores, onset, rng.uniform(0.82, 0.98))
    events = _stream_events(_ProxyScoreModel(), scores, threshold)
    return fa_rate.summarize_floor((event["start_s"] for event in events), exposure_s, block_s=10)


def _context_bank(world, seed):
    """Independent held-out whole contexts from the declared world distribution.

    Independent: 200 single-cause contexts/family. Stress: 66 complete triplets
    (198 instances/family), under the original 200-instance cap. The overlap
    stratum is sampled separately and weighted by the world's 10/34 burst share.
    A single score bank is replayed unchanged at every threshold.
    """
    rng = np.random.default_rng(seed)
    bank = []
    for family in FAMILY_NAMES:
        strata = [('isolated', 200)] if world == 'independent' else [('clustered', 46), ('overlapping', 20)]
        for context, count in strata:
            for index in range(count):
                scores = _floor_background(40 if world == 'independent' else 96, rng,
                                           stress=world == 'stress')
                strength = rng.uniform(.20, 1.0)
                if world == 'independent':
                    offsets, amplitudes = [2.0], [strength]
                else:
                    offsets = [4.0, 4.0 if context == 'overlapping' else 12.0, 20.0]
                    amplitudes = [float(np.clip(strength + rng.normal(0, .025), .20, 1.0)) for _ in offsets]
                instances = []
                for onset, amplitude in zip(offsets, amplitudes):
                    _place_pulse(scores, onset, amplitude)
                    instances.append({'start_s': onset, 'end_s': onset + .5})
                bank.append((family, context, f'{world}-{family}-{context}-{index}', scores, instances))
    return bank


def _score_context_bank(bank, threshold):
    rows = []
    for family, context, context_id, scores, instances in bank:
        events = _stream_events(_ProxyScoreModel(), scores, threshold)
        rows.extend(fa_rate.context_outcomes(
            events, ['target'], family=family, context=context, context_id=context_id,
            instances=instances, max_confirmed_outputs={'target': (len(scores) + 1) // 3},
        ))
    return rows


def _evaluate_known_worlds(seeds):
    report = {'label': 'model-based score-proxy result; no measured audio', 'worlds': {}}
    for world in ('independent', 'stress'):
        world_seeds = seeds[world]
        brute_scores = _make_six_hour_stream(world, world_seeds['truth'])
        bank = _context_bank(world, world_seeds['challenge'])
        weights = {'isolated': 1.0} if world == 'independent' else {'clustered': 24 / 34, 'overlapping': 10 / 34}
        incidence = {f: r / (1 if world == 'independent' else 3) for f, r in FAMILY_RATES.items()}
        results = {}
        for threshold in THRESHOLDS:
            outputs = _stream_events(_ProxyScoreModel(), brute_scores, threshold)
            brute_rate = len(outputs) / STREAM_HOURS
            floor = _make_floor_sample(world, world_seeds['floor'], threshold)
            trials = _score_context_bank(bank, threshold)
            estimate = fa_rate.estimate_composite(
                floor, trials, incidence, context_weights=weights, bootstrap_replicates=4000,
                bootstrap_seed=world_seeds['bootstrap'], dispersion_corrected=True,
            )
            row = estimate['classes']['target']
            interval = row['interval_95_per_hour']
            # Diagnostic comparator: incorrectly treat each output in a burst
            # as an independent Poisson count, with the same error allocation.
            component_confidence = 1 - .05 / (1 + len(incidence) * len(weights))
            naive = np.asarray(fa_rate.poisson_rate_interval(floor['events'], floor['exposure_s'],
                                                            component_confidence))
            for family, rate in incidence.items():
                for context, weight in weights.items():
                    group = [r for r in trials if r['family'] == family and r['context_group'] == context]
                    # 3600 seconds is the numerical unit conversion: here the
                    # exposure is a number of contexts, not hours of audio.
                    bounds = fa_rate.poisson_rate_interval(sum(r['confirmed_outputs'] for r in group),
                                                          len(group) * 3600, component_confidence)
                    naive += rate * weight * np.asarray(bounds)
            results[str(threshold)] = {
                'brute_force_events_6h': len(outputs), 'brute_force_rate_per_hour': brute_rate,
                'composite_rate_per_hour': row['rate_per_hour'], 'interval_95_per_hour': interval,
                'estimate_to_brute_ratio': row['rate_per_hour'] / brute_rate if brute_rate else None,
                'interval_covers_brute_force': interval[0] <= brute_rate <= interval[1],
                'interval_width_per_hour': interval[1] - interval[0],
                'absolute_bias_vs_finite_stream_per_hour': row['rate_per_hour'] - brute_rate,
                'interval_method': estimate['interval_method'], 'dispersion_corrected': estimate['dispersion_corrected'],
                'status': row['status'], 'floor': floor,
                'context_prevalence_per_hour': incidence, 'context_weights': weights,
                'family_context_counts': row['families'],
                'uncorrected_output_Poisson_interval_diagnostic_only': naive.tolist(),
                'uncorrected_interval_covers_brute_force': bool(naive[0] <= brute_rate <= naive[1]),
            }
        report['worlds'][world] = results
    report['cowan_upper_95_per_hour_at_one_hour'] = [fa_rate.cowan_poisson_upper(k, 3600) for k in range(3)]
    rows = [row for world in report['worlds'].values() for row in world.values()]
    sufficient = all(row['brute_force_events_6h'] >= 10 and row['status'] == 'scored' for row in rows)
    cowan = np.allclose(report['cowan_upper_95_per_hour_at_one_hour'], [3, 4.74, 6.30], atol=.01, rtol=0)
    passed = (cowan and all(1 / 1.5 <= row['estimate_to_brute_ratio'] <= 1.5
                           and row['interval_covers_brute_force'] for row in rows)
              and all(r['dispersion_corrected'] for r in report['worlds']['stress'].values())) if sufficient else False
    report['decision'] = 'inconclusive' if not sufficient else 'pass' if passed else 'fail'
    return report


# Frozen before the one-shot fresh acceptance; previous seeds are development only.
FRESH_SEEDS = {
    'independent': {'truth': 876220101, 'floor': 876220201, 'challenge': 876220301, 'bootstrap': 876220401},
    'stress': {'truth': 876220102, 'floor': 876220202, 'challenge': 876220302, 'bootstrap': 876220402},
}
PREREG_SHA256 = '1f5542ae622e81082d85bc920b0c421bc9ab48e5d3518a7f70115e044613e4c2'


@pytest.mark.slow
def test_wp2_fresh_six_hour_proxy_acceptance():
    """Replay the recorded PASS and explicitly report the current inconclusive gate."""
    # Explicit opt-in keeps this historical replay out of broad slow-suite runs.
    import hashlib
    import os
    from pathlib import Path
    path = os.environ.get('FA_RATE_PREREGISTRATION')
    output = os.environ.get('FA_RATE_ACCEPTANCE_OUTPUT')
    if not path or not output:
        pytest.skip('historical replay requires original preregistration and a new evidence path')
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == PREREG_SHA256
    # Reserve the artifact before generating any data; never overwrite a run.
    with Path(output).open('x') as handle:
        report = _evaluate_known_worlds(FRESH_SEEDS)
        report['preregistration_sha256'] = PREREG_SHA256
        report['scope'] = 'historical replay only; no new acceptance claim'
        # 944e005 counted >=100 contexts OR annotated sources per family,
        # >=2 independent parents per stratum, and nonconstant counts. Keep
        # that rule here only; production continues to require 100/20 parents.
        historical_sufficient = all(
            r['brute_force_events_6h'] >= 10 and all(
                max(f['contexts'], f['source_instances']) >= 100
                and all(s['independent_parents'] >= 2 and s['sample_variance'] > 0
                        for s in f['by_context'].values())
                for f in r['family_context_counts'].values())
            for world in report['worlds'].values() for r in world.values())
        historical_pass = historical_sufficient and all(
            1 / 1.5 <= r['estimate_to_brute_ratio'] <= 1.5 and r['interval_covers_brute_force']
            for world in report['worlds'].values() for r in world.values())
        historical_pass &= all(r['dispersion_corrected'] for r in report['worlds']['stress'].values())
        historical_pass &= np.allclose(report['cowan_upper_95_per_hour_at_one_hour'],
                                       [3, 4.74, 6.30], atol=.01, rtol=0)
        report['historical_gate'] = '944e005: >=100 contexts or sources per family; >=2 parents/stratum; nonconstant'
        report['historical_decision'] = ('inconclusive' if not historical_sufficient
                                         else 'pass' if historical_pass else 'fail')
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    assert report['historical_decision'] == 'pass'
    assert report['decision'] == 'inconclusive'
    assert fa_rate.MIN_PARENT_CONTEXTS == 100 and fa_rate.MIN_PARENTS_PER_STRATUM == 20
    assert all(r['status'] == 'inconclusive-insufficient-context-evidence'
               for r in report['worlds']['stress'].values())
