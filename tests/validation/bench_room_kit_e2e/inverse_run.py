"""I2 frozen dry-stage sidecar; public acquisition/selection runs once per session.

python -m tests.validation.bench_room_kit_e2e.inverse_run --out JOB/evidence/inverse
No truth enters the capture wrappers or candidate parameter selection. The
unchanged public pipeline receives/returns the legacy result. Candidate audio
is scored later with the frozen observer, in the same source-reference domain.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import time
from unittest.mock import patch

import numpy as np
import scipy
from scipy.io import wavfile

from sensorsim.acoustics.room import pipeline
from sensorsim.acoustics.room.dry import DryConfig, extract_dry
from sensorsim.acoustics.room.dereverb import artifact_metrics
from sensorsim.acoustics.room.source_reference import source_at_one_m
from tests.validation.bench_room_closedloop.compare import compare_dry, align_signals, context
from tests.validation.bench_room_closedloop.engines import T0Engine, assemble_session
from tests.validation.bench_room_closedloop.measured import MeasuredEngine, assemble_measured_session, run_measured_scenario
from tests.validation.bench_room_closedloop.runner import run_scenario, file_hash
from tests.validation.bench_room_closedloop.scenario import load_scenario
from tests.validation.bench_room_closedloop.scorecard import write_json, write_scorecard


CONFIGS = {
    'legacy': DryConfig(),
    'tikhonov': DryConfig(inverse_mode='tikhonov'),
}
METHODS = tuple(CONFIGS)


def array_hash(x):
    return hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).hexdigest()


class Capture:
    """Records measured inputs only; never gets an engine, truth or evaluator."""
    def __init__(self):
        self.takes = {}

    def dry(self, recordings, responses, noise, **kwargs):
        # This exact result continues through the unmodified public pipeline.
        baseline = extract_dry(recordings, responses, noise, **kwargs)
        options = {**kwargs}; options.pop('config', None)
        results = {'legacy': (baseline.samples.copy(), baseline.report, None)}
        try:
            result = extract_dry(recordings, responses, noise, **options, config=CONFIGS['tikhonov'])
            results['tikhonov'] = (result.samples, result.report, None)
        except ValueError as exc:
            results['tikhonov'] = (None, {}, str(exc))
        key = array_hash(baseline.samples)
        if key in self.takes:
            raise ValueError('ambiguous dry output hash; cannot pair takes by order')
        self.takes[key] = results
        return baseline


def score_session(scenario, engine, assembled, work, out):
    capture = Capture()
    runner = run_measured_scenario if engine.tier == 'T2' else run_scenario
    with patch.object(pipeline, 'extract_dry', capture.dry):
        outcome = runner(scenario, engine, work, out/'public', assembled=assembled)
    provenance = outcome[1]
    fs = assembled.kit.fs_hz
    rows, details = [], []
    dry_dir = work/'pipeline/dry'
    # Completed outputs only. Every planned real take remains in the denominator.
    for i, take in enumerate(assembled.manifest['takes']):
        if take['type'] != 'real':
            continue
        stem = dry_dir/f'take_{i:04d}_ch_00'
        production = stem.with_suffix('.wav')
        info = json.loads(stem.with_suffix('.json').read_text()) if stem.with_suffix('.json').exists() else None
        results = None
        if production.exists() and info is not None:
            rate, audio = wavfile.read(production)
            assert rate == fs
            results = capture.takes[array_hash(audio)]
        truth = assembled.truth.dry_sources[take['file']]
        ctx = context(scenario, engine, 'real_sound', take['speaker'], take['mic'])
        has_ref = any(t['type']=='sweep' and t.get('speaker')==take['speaker'] and
                      t['mic']==f'REF_{take["speaker"]}' for t in assembled.manifest['takes'])
        for method in METHODS:
            audio, report, error = (results[method] if results is not None else
                                     (None, {}, 'public event unresolved'))
            if audio is not None:
                # Same once-only REF conversion used by public dataset-source output.
                from sensorsim.acoustics.room.source_reference import SourceReference
                reference = SourceReference.from_dict(info['source_reference'])
                audio, _, _ = source_at_one_m(audio, reference)
            note = f'take={take["file"]}; method={method}; dry-stage sidecar; event fixed to public legacy'
            values = compare_dry(truth, audio, None, fs, ctx, notes=note)
            # No new oracle claim; oracle rows are not part of this sidecar.
            values = [r for r in values if not r['metric'].startswith('oracle')]
            for r in values:
                r.update(method=method, take=take['file'])
            rows.extend(values)
            artifacts = None
            if audio is not None:
                _, _, lag, _ = align_signals(truth, audio, fs)
                artifacts = artifact_metrics(audio, truth, fs, lag_samples=lag)
            details.append(dict(scenario=scenario.id, tier=engine.tier, take=take['file'], method=method,
                                error=error, artifacts=artifacts, design=report.get('inverse_design'),
                                residual=report.get('dereverberation'),
                                source_reference=None if info is None else info['source_reference'],
                                event_region=None if info is None else info.get('event_region')))
            if has_ref and take.get('calibration') is None:
                from tests.validation.bench_room_resim.chain import recorder_source
                acquired_h = assembled.truth.rirs[f'{take["speaker"]}:{take["mic"]}']
                support = max(len(truth)+len(acquired_h)-1, 0 if audio is None else len(audio))
                target = recorder_source(np.pad(truth,(0,support-len(truth))),fs,assembled.impairments)
                same = compare_dry(target, audio, None, fs, ctx, notes=note+'; domain=same_recorder')
                same = [r for r in same if not r['metric'].startswith('oracle')]
                for r in same:
                    r.update(method=method, take=take['file'])
                details[-1]['same_recorder_scores'] = same
        write_json(out/'details.json', details)
    for method in METHODS:
        write_scorecard(out/(method+'.jsonl.gz'),
                        [{k:v for k,v in r.items() if k not in ('method','take')}
                         for r in rows if r['method']==method])
        same = [r for d in details if d['method']==method for r in d.get('same_recorder_scores', [])]
        if same:
            write_scorecard(out/('same_recorder_'+method+'.jsonl.gz'),
                            [{k:v for k,v in r.items() if k not in ('method','take')} for r in same])
    return rows, details, dict(scenario=scenario.id, tier=engine.tier,
        event_selection=provenance['event_selection'], pipeline_error=provenance['pipeline_error'],
        session_sha256=provenance.get('session_sha256'), peak_rss_mib=provenance['peak_rss_mib'])


def summarize(rows, details):
    output = {}
    for tier in sorted({r['tier'] for r in rows}):
        output[tier] = {}
        for method in METHODS:
            group = [r for r in rows if r['tier']==tier and r['method']==method]
            baseline = {(r['scenario'],r['take'],r['metric'],r['band_hz']):r for r in rows
                        if r['tier']==tier and r['method']=='legacy'}
            metrics = {}
            for metric in sorted({r['metric'] for r in group}):
                selected = [r for r in group if r['metric']==metric]
                vals = [r['reconstructed'] for r in selected if isinstance(r['reconstructed'],(int,float)) and np.isfinite(r['reconstructed'])]
                paired, regressions = [], 0
                for r in selected:
                    b = baseline[r['scenario'], r['take'], metric, r['band_hz']]
                    regressions += bool(b['pass'] and not r['pass'])
                    x, y = b['reconstructed'], r['reconstructed']
                    if isinstance(x,(int,float)) and isinstance(y,(int,float)) and np.isfinite([x,y]).all():
                        paired.append(y-x)
                metrics[metric] = dict(pass_count=sum(r['pass'] is True for r in selected), planned=len(selected),
                    unavailable=sum(r['pass'] is None for r in selected),
                    missing_estimate=sum(r['pass'] is False and r['reconstructed'] is None for r in selected),
                    gate_regressions=regressions,
                    median=float(np.median(vals)) if vals else None,
                    paired_delta_median=float(np.median(paired)) if paired else None, paired_count=len(paired))
            artifacts = {}
            for key in ('log_spectral_distance_db','pre_echo_energy_ratio_db','late_to_early_db','spectral_isolation_fraction'):
                vals = [(d['artifacts'] or {}).get(key) for d in details if d['tier']==tier and d['method']==method]
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                artifacts[key] = dict(median=float(np.median(vals)) if vals else None, defined=len(vals))
            output[tier][method] = dict(metrics=metrics, artifacts=artifacts)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tiers', nargs='+', choices=['T0','T2'], default=['T0','T2'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    root = Path(__file__).parents[1]/'bench_room_closedloop/scenarios'
    runs, rows, details = [], [], []
    for tier in args.tiers:
        with tempfile.TemporaryDirectory(prefix='i2-engine-', dir=os.environ['TMPDIR']) as reader:
            engine = T0Engine() if tier == 'T0' else MeasuredEngine(scratch=reader)
            names = ['poor', 'bedroom_ref'] if tier == 'T0' else ['dechorate_020002']
            for name in names:
                scenario = load_scenario(root/(name+'.yaml')) if tier=='T0' else engine.scenario(name,'full')
                with tempfile.TemporaryDirectory(prefix='i2-session-', dir=os.environ['TMPDIR']) as tmp:
                    folder = Path(tmp)
                    assembled = (assemble_session if tier=='T0' else assemble_measured_session)(scenario,engine,folder/'assembled')
                    r, d, p = score_session(scenario, engine, assembled, folder/'run', args.out/name)
                    rows.extend(r); details.extend(d); runs.append(p)
                print(json.dumps(p), flush=True)
                write_json(args.out/'progress.json', runs)
    sources = [Path(__file__), Path(__file__).with_name('INVERSE_PROTOCOL.md'),
               *(Path('src/sensorsim/acoustics/room')/n for n in ('rir.py','dry.py','dereverb.py'))]
    summary = dict(version='dry-inverse-comparison/2', rir_policy='same public averaged RIR for both modes',
        git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        source_hashes={str(p):file_hash(p) if p.exists() else None for p in sources}, configs={k:asdict(v) for k,v in CONFIGS.items()},
        runs=runs, results=summarize(rows,details),
        same_recorder_results=summarize([r for d in details for r in d.get('same_recorder_scores', [])], []),
        seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        default_changed=False, selection='paired frozen subset; insufficient for promotion', evidence=str(args.out.resolve()))
    write_json(args.out/'summary.json',summary)


if __name__ == '__main__':
    main()
