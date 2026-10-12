"""Once-only, hash-gated final WP3 proxy acceptance; no external audio/weights.

python -m tests.validation.bench_negatives.final_review --job JOB
The prior FAIL/PASS evidence remains untouched. In particular the old L4 PASS
was degenerate: it compared the same posterior-dither estimator with itself.
The known-analog single-capture comparator here is independently constructed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
import numpy as np
from sensorsim import iot_mic
from sensorsim.negatives.procedural import TONES, generate
from sensorsim.negatives.tonal import target_partials
from sensorsim.training_device import _check_recording
from tests.validation.bench_negatives import final_l4, final_path
from tests.validation.bench_negatives.final_common import FS, PROFILE, truth_performance, bed_analog


def save(path, value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item())+'\n')


def run(job):
    expected=(job/'evidence/prereg_sha256.txt').read_text().split()[0]
    actual=hashlib.sha256((job/'preregistration.yaml').read_bytes()).hexdigest()
    frozen=Path('tests/unit/negatives/final_preregistration.sha256').read_text().strip()
    if actual != expected or expected != frozen: raise ValueError('pre-registration hash mismatch')
    out=job/'evidence/acceptance';out.mkdir(exist_ok=False)
    start=time.monotonic();save(out/'started.json',{'prereg_sha256':actual,'evidence_label':'proxy'})
    paths=list(Path('src/sensorsim/negatives').glob('*.py'))+[Path('src/sensorsim/training_device.py')]
    paths+=list(Path('tests/validation/bench_negatives').glob('final_*.py'))
    save(out/'implementation-sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    final_l4.main(out/'l4.json')
    l4=json.loads((out/'l4.json').read_text())
    for row in l4['positives']:
        stat=row['c2st_conditional_vs_single']; lo,hi=stat['auc_ci95']
        err=row['error_vs_analog_truth']; d=[abs(err['conditional'][k]-err['single'][k]) for k in ('mean_lsb','rms_lsb')]
        row['mean_rms_difference_lsb']=d
        row['old_exact_null_rule']='pass' if .45<=stat['auc']<=.55 and lo<=.5<=hi else 'fail'
        row['registered_chance_equivalence']='pass' if .45<=lo<=hi<=.55 and max(d)<=.005 else (
            'fail' if hi<.45 or lo>.55 or max(d)>.005 else 'inconclusive')
    save(out/'l4.json',l4)
    final_path.main(out/'path.json')
    partials=target_partials([truth_performance('doorbell',k) for k in range(6)],fs=FS)
    tonal={'training_partial_hz':partials,'rows':[]}
    for seed in (1701,1702,1703):
        for family in TONES:
            admitted=rejected=intersections=0
            for k in range(200):
                try:
                    src=generate(family,fs=FS,duration_s=1,seed=int(np.random.SeedSequence([seed,k]).generate_state(1)[0]),target_partials_hz=partials)
                except ValueError as exc:
                    if 'exclusion' not in str(exc): raise
                    rejected+=1;continue
                admitted+=1
                for lo,hi in src.provenance['parameters']['tonal_support_hz']:
                    intersections+=sum(hi>=f*.97 and lo<=f*1.03 for f in partials)
            tonal['rows'].append(dict(seed=seed,family=family,admitted=admitted,rejected=rejected,intersections=intersections))
    tonal['decision']='pass' if not any(r['intersections'] for r in tonal['rows']) else 'fail'
    save(out/'tonal.json',tonal)
    floor=[]
    for seed in (1701,1702,1703):
        alarms=0
        for k in range(100):
            rng=np.random.default_rng([seed,k,1]);profile=PROFILE.with_variation(rng)
            analog=np.zeros(2*FS) if k%2==0 else bed_analog(seed*1000+k,spl_db=20+15*k/100,seconds=2)
            recording=profile.apply(analog,FS,rng,input_reference_spl_db=94.)
            check=PROFILE.with_variation(np.random.default_rng([seed,k,2]))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');_check_recording(recording,FS,check,True)
            alarms+=any('below the profile' in str(w.message) for w in caught)
        with warnings.catch_warnings(record=True) as long:
            warnings.simplefilter('always');_check_recording(np.full(2*FS,1/32768),FS,PROFILE,True)
        with warnings.catch_warnings(record=True) as short:
            warnings.simplefilter('always');_check_recording(np.full(FS//2,1/32768),FS,PROFILE,True)
        floor.append(dict(seed=seed,false_alarms=alarms,recordings=100,false_alarm_fraction=alarms/100,
            suppressed_floor_detected=any('below the profile' in str(w.message) for w in long),
            short_dip_warned=any('below the profile' in str(w.message) for w in short)))
    save(out/'floor.json',floor)
    suites={}
    for name,selection in [('realism',['tests/unit/negatives/test_final_fixes.py','-k','pool or impact']),
                           ('default',[]),('slow',['-m','slow'])]:
        before=time.monotonic()
        with (out/f'{name}-suite.txt').open('w') as log:
            proc=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                '--basetemp='+str(job/'tmp'/('pytest-acceptance-'+name)),*selection],stdout=log,stderr=subprocess.STDOUT)
        suites[name]={'exit_code':proc.returncode,'wall_seconds':time.monotonic()-before}
        print(name,suites[name],flush=True)
    paths=json.loads((out/'path.json').read_text())
    statuses=[r['registered_chance_equivalence'] for r in l4['positives']]
    statuses += [r['decision'] for r in paths if not r['negative_shift_db']]
    statuses += [tonal['decision'],'pass' if all(r['false_alarm_fraction']<=.05 and r['suppressed_floor_detected'] and not r['short_dip_warned'] for r in floor) else 'fail']
    statuses += ['pass' if s['exit_code']==0 else 'fail' for s in suites.values()]
    result={'evidence_label':'proxy','prereg_sha256':actual,'decision':
        'fail' if 'fail' in statuses else 'pass' if all(s=='pass' for s in statuses) else 'inconclusive',
        'suites':suites,'wall_seconds':time.monotonic()-start,'gate_decisions':statuses,
        'limits':'No measured results, training gain, target recall or deployment FA/hour qualification.'}
    save(out/'results.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=Path)
    run(p.parse_args().job)
