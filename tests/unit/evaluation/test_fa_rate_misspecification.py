"""Pre-registered diagnostic score proxies, never an acceptance rerun.

Run with --preregistration, --addendum and a new --output directory. The CNN
is a numerical score proxy; all event assembly uses the real edge.Detector.
No third-party waveform, model weights, or measured data are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import edge
from sensorsim.evaluation import fa_rate

PREREG_SHA256 = '0a1fe529fc61f6bd9850b1009bfb8462c9049d81c6bf6a9efaecd18cd340b6d4'
ADDENDUM_SHA256 = '74cd2f9319a753fea5ad9390f5a26fabdb0b632d8926c6cf646e5ae1777e6c76'
SCENARIOS = ('mix', 'superposition', 'drift', 'splitting', 'smoothing')
UNITS = ('confirmed_events', 'start_alerts')


class ScoreModel:
    classes = ['target']
    features = SimpleNamespace(n=4)

    def __init__(self, smooth=1):
        self.meta = {'features': {'fs': 4}, 'detector': {
            'hop_s': .25, 'on': {'target': .5}, 'off_ratio': .6,
            'smooth_hops': smooth, 'min_hops': 2}}

    def scores(self, audio):
        return np.atleast_2d(audio)[:, :1]


def events(audio, smooth):
    return fa_rate.score_audio(ScoreModel(smooth), audio, include_starts=True)


def counts(outputs):
    return {'confirmed_events': sum(bool(e.get('confirmed')) for e in outputs),
            'start_alerts': sum(e.get('event') == 'start' for e in outputs)}


def signal(scenario, *, level=.85, length=4, runs=3, bridge=.15):
    size = 64 if scenario in ('drift', 'splitting') else 48
    x = np.zeros(size, np.float32)
    if scenario == 'drift':
        x[8:8 + length] = level
        x[8 + length:16 + length] = bridge
        x[16 + length:16 + 2 * length] = level
    elif scenario == 'splitting':
        for i in range(runs):
            x[8 + i * (length + 8):8 + i * (length + 8) + length] = level
    else:
        x[8:8 + length] = level
    return x


def truth_marks(scenario):
    """Finite mark distribution; exact mean evaluated through Detector, not inferred."""
    if scenario == 'mix':
        for family_p, high_p, low, high in ((.75, .8, .15, .8), (.25, .3, .2, .75)):
            for probability, level in ((1 - high_p, low), (high_p, high)):
                for length in (3, 4, 5):
                    yield family_p * probability / 3, signal(scenario, level=level, length=length)
    elif scenario == 'superposition':
        for length in (3, 4, 5):
            # Two separate subthreshold score contributions, added, never max.
            yield 1 / 3, signal(scenario, level=.28, length=length) + signal(scenario, level=.34, length=length)
    elif scenario == 'drift':
        for probability, bridge in ((.75, .35), (.25, .15)):
            yield probability, signal(scenario, bridge=bridge)
    elif scenario == 'splitting':
        for probability, runs in ((.6, 3), (.4, 2)):
            yield probability, signal(scenario, runs=runs)
    else:
        for length in (1, 2):
            yield .7 / 2, signal(scenario, level=.6, length=length)
        for length in (4, 5, 6):
            yield .3 / 3, signal(scenario, level=.85, length=length)


def truth(scenario, rng):
    marks = list(truth_marks(scenario))
    p = np.asarray([p for p, _ in marks])
    samples = np.zeros(6 * 3600 * 4, np.float32)
    for slot in range(360):
        if rng.random() < .5:
            _, x = marks[rng.choice(len(marks), p=p / p.sum())]
            samples[slot * 240:slot * 240 + len(x)] += x
    return samples


def bank(scenario, mitigation, rng):
    """Proposal waveforms differ from truth marks even after oracle mitigation."""
    rows = []
    families = ('impact', 'flow') if scenario == 'mix' else ('scene',)
    smooth = 3 if scenario == 'smoothing' and mitigation else 1
    for family in families:
        strata = ('low', 'high') if scenario == 'mix' and mitigation else ('ordinary',)
        n = 100 if len(strata) == 2 else 200
        for stratum in strata:
            for i in range(n):
                length = int(rng.integers(3, 7)) if mitigation else 4
                if scenario == 'mix':
                    if mitigation:
                        level = rng.uniform(.1, .25) if stratum == 'low' else rng.uniform(.65, .95)
                    else:
                        level = rng.uniform(.15, .75) if family == 'impact' else rng.uniform(.25, .85)
                    x = signal(scenario, level=level, length=length)
                elif scenario == 'superposition':
                    x = signal(scenario, level=rng.uniform(.27, .36), length=length)
                    if mitigation:
                        x += signal(scenario, level=rng.uniform(.27, .36), length=length)
                elif scenario == 'drift':
                    bridge = (rng.uniform(.32, .4) if rng.random() < .75 else rng.uniform(.1, .25)) if mitigation else 0
                    x = signal(scenario, level=rng.uniform(.75, .95),
                               length=int(rng.integers(3, 6)) if mitigation else 4, bridge=bridge)
                elif scenario == 'splitting':
                    if mitigation:
                        x = signal(scenario, level=rng.uniform(.75, .95),
                                   length=int(rng.integers(3, 6)), runs=3 if rng.random() < .6 else 2)
                    else:
                        x = np.zeros(64, np.float32)
                        x[8:36] = rng.uniform(.75, .95)
                else:
                    short = rng.random() < .7
                    x = signal(scenario, level=rng.uniform(.55, .7) if short else rng.uniform(.8, .95),
                               length=2 if short else 4)
                outputs = events(x, smooth)
                n_windows = len(x) - 3
                rows.extend(fa_rate.context_outcomes(
                    outputs, ['target'], family=family, context=stratum,
                    context_id=f'{family}-{stratum}-{i}', max_confirmed_outputs={'target': (n_windows + 1) // 3},
                    max_start_outputs={'target': (n_windows + 1) // 2},
                    instances=[{'start_s': 2, 'end_s': (len(x) - 4) / 4}], source_kind='proxy'))
    if scenario == 'mix':
        incidence = {'impact': 22.5, 'flow': 7.5} if mitigation else {'impact': 7.5, 'flow': 22.5}
        weights = {'impact': {'low': .2, 'high': .8}, 'flow': {'low': .7, 'high': .3}} if mitigation else {'ordinary': 1}
    else:
        incidence, weights = {'scene': 30}, {'ordinary': 1}
    return rows, incidence, weights


def run(preregistration, output, addendum):
    assert hashlib.sha256(preregistration.read_bytes()).hexdigest() == PREREG_SHA256
    assert hashlib.sha256(addendum.read_bytes()).hexdigest() == ADDENDUM_SHA256
    run_hash = ADDENDUM_SHA256
    output.mkdir()  # Exclusive: never overwrite, resume, select seeds, or retune.
    (output / 'prereg_sha256.txt').write_text(run_hash + '\n')
    floor_audio = np.zeros(1200 * 4, np.float32)
    all_rows = []
    with (output / 'runs.jsonl').open('x') as handle:
        for scenario_index, scenario in enumerate(SCENARIOS):
            smooth = 3 if scenario == 'smoothing' else 1
            expected = dict.fromkeys(UNITS, 0.)
            for probability, x in truth_marks(scenario):
                for unit, k in counts(events(x, smooth)).items():
                    expected[unit] += 30 * probability * k
            floor_events = events(floor_audio, smooth)
            floor = fa_rate.summarize_floor((e['start_s'] for e in floor_events if e.get('confirmed')), 1200)
            floor['start_alerts'] = fa_rate.summarize_floor((e['time_s'] for e in floor_events if e.get('event') == 'start'), 1200)
            for replicate in range(30):
                seed = 108230000 + scenario_index * 100000 + replicate * 100
                brute = counts(events(truth(scenario, np.random.default_rng(seed + 1)), smooth))
                for mitigation in (False, True):
                    rows, incidence, weights = bank(scenario, mitigation, np.random.default_rng(seed + (3 if mitigation else 2)))
                    for unit in UNITS:
                        estimate = fa_rate.estimate_composite(
                            floor, rows, incidence, context_weights=weights, event_unit=unit,
                            floor_time_fraction=0, bootstrap_replicates=1000, bootstrap_seed=seed + 4)
                        c = estimate['classes']['target']
                        lo, hi = c['interval_95_per_hour']
                        m = 1 + sum(len(w) for w in estimate['context_weights'].values())
                        floor_hi = fa_rate.poisson_rate_interval(0, 1200, 1 - .05 / m)[1]
                        rec = dict(label='proxy', scenario=scenario, replicate=replicate, seed_base=seed,
                                   variant='mitigated' if mitigation else 'baseline', event_unit=unit,
                                   known_mean_per_hour=expected[unit], brute_count_6h=brute[unit], brute_rate=brute[unit] / 6,
                                   point=c['rate_per_hour'], interval=[lo, hi], status=c['status'],
                                   covers_mean=bool(lo <= expected[unit] <= hi),
                                   includes_finite_stream=bool(lo <= brute[unit] / 6 <= hi),
                                   interval_with_20min_floor=[lo, hi + floor_hi],
                                   covers_mean_with_20min_floor=bool(lo <= expected[unit] <= hi + floor_hi))
                        handle.write(json.dumps(rec) + '\n')
                        handle.flush()
                        all_rows.append(rec)
                print(f'proxy {scenario} replicate {replicate + 1}/30', flush=True)
    summaries = []
    for scenario, variant, unit in itertools.product(SCENARIOS, ('baseline', 'mitigated'), UNITS):
        group = [r for r in all_rows if (r['scenario'], r['variant'], r['event_unit']) == (scenario, variant, unit)]
        reference = group[0]['known_mean_per_hour']
        points = np.asarray([r['point'] for r in group])
        brute_rates = np.asarray([r['brute_rate'] for r in group])
        cover = sum(r['covers_mean'] for r in group)
        bias = float(points.mean() - reference)
        summaries.append(dict(label='proxy', scenario=scenario, variant=variant, event_unit=unit, n=len(group),
            known_mean_per_hour=reference, mean_point=float(points.mean()), mean_brute=float(brute_rates.mean()),
            bias_per_hour=bias, relative_bias=bias / reference if reference else None,
            bias_vs_brute_per_hour=float(np.mean(points - brute_rates)),
            bias_mean_standard_error=float(points.std(ddof=1) / np.sqrt(len(group))),
            rmse_vs_brute=float(np.sqrt(np.mean((points - brute_rates)**2))),
            mean_coverage=cover / len(group), coverage_cp95=fa_rate.clopper_pearson(cover, len(group)),
            finite_stream_inclusion=sum(r['includes_finite_stream'] for r in group) / len(group),
            mean_width=float(np.mean([r['interval'][1] - r['interval'][0] for r in group])),
            mean_coverage_with_20min_floor=sum(r['covers_mean_with_20min_floor'] for r in group) / len(group),
            statuses=dict(Counter(r['status'] for r in group)),
            diagnostic_gap_closed=abs(bias / reference) <= .10 and cover / len(group) >= .90 if reference else False))
    report = dict(label='proxy result; no measured audio; not calibration', preregistration_sha256=run_hash,
                  original_preregistration_sha256=PREREG_SHA256,
                  status_meaning=fa_rate.STATUS_MEANING,
                  mitigation_scope='oracle controls copy truth mixture probabilities; not evidence users can close gaps',
                  historical_acceptance='unchanged: w2-fa FAIL; w2-fa-fix PASS', summaries=summaries)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


@pytest.mark.slow
def test_misspecification_mechanisms_use_real_detector():
    assert counts(events(signal('superposition', level=.28) + signal('superposition', level=.34), 1))['confirmed_events'] == 1
    assert counts(events(signal('superposition', level=.34), 1))['confirmed_events'] == 0
    assert counts(events(signal('drift', bridge=.35), 1))['confirmed_events'] == 1
    assert counts(events(signal('drift', bridge=.15), 1))['confirmed_events'] == 2
    assert counts(events(signal('splitting', runs=3), 1))['confirmed_events'] == 3
    assert counts(events(signal('smoothing', level=.6, length=2), 1))['confirmed_events'] == 1
    assert counts(events(signal('smoothing', level=.6, length=2), 3))['confirmed_events'] == 0
    assert ScoreModel(3).meta['detector']['smooth_hops'] == 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--addendum', type=Path, required=True, help='registered N2 diagnostic addendum')
    args = parser.parse_args()
    run(args.preregistration, args.output, args.addendum)
