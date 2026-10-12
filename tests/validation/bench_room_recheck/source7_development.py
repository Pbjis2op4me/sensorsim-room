"""Source-7 descriptive development replay; never a two-source acceptance run.

Reads CC BY v1 in place and saves derived metrics outside the repository. No
C2ST, LF decision, replacement accuracy table or acceptance receipt is emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from tests.validation.bench_room_recheck import run
from tests.validation.bench_room_recheck.corrected import pair_seed, write_json
from tests.validation.bench_room_closedloop.measured import MeasuredEngine, run_measured_scenario
from tests.validation.dechorate_v1 import STATES
from sensorsim.acoustics.room.metrics import OCTAVE_BANDS
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from sensorsim.evaluation.room_recheck import octave_room_metrics


def metric_summary(rows):
    """Keep missing-value denominators; no threshold or decision is applied."""
    result = {}
    for name in ('edt_s', 't30_s', 'c50_db', 'drr_db'):
        summaries = []
        for b, band in enumerate(OCTAVE_BANDS):
            measured = np.array([r['measured'][name][b] for r in rows], float)
            simulated = np.array([r['model_based'][name][b] for r in rows], float)
            valid = np.isfinite(measured) & np.isfinite(simulated)
            if name.endswith('_s'):
                valid &= measured > 0
                error = np.abs(simulated[valid]-measured[valid])/measured[valid]*100
                unit = '%'
            else:
                error = np.abs(simulated[valid]-measured[valid])
                unit = 'dB'
            summaries.append(dict(band_hz=band, total=len(rows), finite_pairs=int(valid.sum()),
                measured_finite=int(np.isfinite(measured).sum()),
                simulated_finite=int(np.isfinite(simulated).sum()), unit=unit,
                median_absolute_error=float(np.median(error)) if len(error) else None,
                p90_absolute_error=float(np.quantile(error, .9)) if len(error) else None))
        result[name] = summaries
    return result


def wp8(data_root, output):
    pairs, audit = run._load_v1_pairs(data_root, 48000, sources=(7,))
    run.TEMP_C, run.RH_PCT = audit['temperature_c'], audit['humidity_pct']
    fitted, details = run._fit_absorption(pairs)
    write_json(output/'wp8_fit_details.json', details)
    rows = []
    for pair in pairs:
        if pair.split != 'validation_only':
            continue
        seed = pair_seed(20261008, pair.pair_id)
        sim = synthesize(run._model(pair.room_dims_m, fitted[pair.state]),
            pair.source_xyz_m, pair.receiver_xyz_m, 48000, seed=seed, config=SynthConfig())
        on = round(sim.metadata['direct_delay_s']*48000+sim.metadata['latency_samples'])
        rows.append(dict(pair_id=pair.pair_id, seed=seed,
            measured_onset=pair.measured_onset, model_onset=on,
            measured=pair.measured_metrics,
            model_based=octave_room_metrics(sim.samples, 48000, onset_sample=on, noise_power=0.0)))
    write_json(output/'wp8_pairs.json', rows)
    result = dict(label='proxy result; source 7 descriptive development only',
        evidence_domain='measured-RIR proxy', dataset=audit,
        fit_pairs=sum(p.split == 'fit_absorption' for p in pairs),
        evaluated_pairs=len(rows), distinct_validation_positions=5,
        room_count=1, acceptance='not run; original two-source requirement remains inconclusive',
        metrics=metric_summary(rows),
        fit_fallbacks=[dict(state=s, **b) for s, d in details.items() for b in d['bands']
                       if b['fit_status'] == 'fallback_default'])
    write_json(output/'wp8_source7_summary.json', result)
    return result


def t2(data_root, output, scratch):
    engine = MeasuredEngine(data=data_root, scratch=scratch)
    rows, runs = [], []
    for state in STATES:
        scenario = engine.dechorate_v1(state, 'full', sources=(7,))
        with tempfile.TemporaryDirectory(dir=scratch, prefix='source7-') as temp:
            scored, provenance, _ = run_measured_scenario(scenario, engine, Path(temp)/'work',
                output/('t2_'+state), dry_mode='legacy')
        rows.extend(r for r in scored if r['eval_kind'] == 'heldout')
        runs.append(dict(state=state, pipeline_error=provenance['pipeline_error'],
            fitted_model_available=provenance['fitted_model_available']))
        print(json.dumps({'development_state_completed': state}), flush=True)
    metrics = {}
    for name in ('t30', 'edt', 'c50_db', 'drr_db'):
        values = [r for r in rows if r['metric'] == name]
        errors = [abs(r['error']) for r in values if r['error'] is not None]
        metrics[name] = dict(total=len(values), finite_pairs=len(errors),
            median_absolute_error=float(np.median(errors)) if errors else None,
            p90_absolute_error=float(np.quantile(errors, .9)) if errors else None,
            unit=values[0]['unit'] if values else None)
    result = dict(label='proxy result; source 7 descriptive development only',
        evidence_domain='measured-RIR proxy', room_count=1, distinct_evaluation_positions=4,
        evaluation_pairs=44, runs=runs, metrics=metrics,
        read_log=engine._v1_archive.read_log,
        acceptance='not run; original two-source requirement remains inconclusive')
    write_json(output/'t2_source7_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--bench', choices=('wp8', 't2'), required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.scratch.mkdir(parents=True, exist_ok=True)
    if a.bench == 'wp8':
        wp8(a.data_root, a.output)
    else:
        t2(a.data_root, a.output, a.scratch)


if __name__ == '__main__':
    main()
