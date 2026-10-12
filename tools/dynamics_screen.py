# SPDX-License-Identifier: Apache-2.0
"""One-seed proxy screen with paired held-out take-group McNemar comparisons.

Accepts standard training manifests and a JSON bank of untouched context WAVs.
All thresholds and budgets come from a hashed preregistration; this runner does
not tune thresholds on the audit bank. Data, model weights and evidence stay out
of the repository. A screen is not an adoption test or measured deployment rate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, binomtest
from sensorsim import edge, nn, recognizer, training_device, training_dynamics, iot_mic
from tools.dynamics_sweep import verify_preregistration

ARMS = {'control':{}, 'event_aware_masks':{'event_aware_masks':True},
        'partial_windows':{'partial_windows':True}, 'low_event_spl_tail':{'low_event_spl_tail':True},
        'mixstyle_same_label':{'mixstyle_same_label':True}}
SETTINGS = dict(steps=1000,batch=128,seed=1701,workers=0,threads=2,val_windows=512,
                detector=dict(default_on=.5,off_ratio=.6,smooth_hops=3,min_hops=2,hop_s=.25))


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def mcnemar(control,candidate):
    c,k=np.asarray(control,bool),np.asarray(candidate,bool)
    lost=int(np.sum(c&~k));gained=int(np.sum(~c&k));n=lost+gained
    return dict(control_only=lost,candidate_only=gained,discordant=n,
                p=float(binomtest(lost,n,.5).pvalue) if n else 1.)


def verify_bank(manifest,bank_path):
    rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    train_groups={r['source_group'] for r in rows}
    train_hashes={hashlib.sha256((Path(manifest).parent/r['file']).read_bytes()).hexdigest() for r in rows}
    bank=json.loads(Path(bank_path).read_text())
    groups=[]
    for row in bank:
        if row['source_group'] in train_groups:raise ValueError('heldout group overlaps training/development')
        digest=hashlib.sha256((Path(bank_path).parent/row['audio']).read_bytes()).hexdigest()
        if digest in train_hashes:raise ValueError('heldout content overlaps training/development')
        if digest!=row['sha256']:raise ValueError('heldout content changed')
        groups.append(row['source_group'])
    if len(groups)!=len(set(groups)):raise ValueError('one context per independent heldout group required')
    return bank


def score(model_path,bank_path):
    model=edge.Model(str(model_path));rows=json.loads(Path(bank_path).read_text());results=[]
    # Freeze operating policy across arms; trainer-fitted persistence is not used.
    tracker_cfg=dict(on=.5,off_ratio=.6,smooth_hops=3,min_hops=2)
    for row in rows:
        audio=recognizer.read_audio(Path(bank_path).parent/row['audio'],16000)
        scores=edge.window_scores(model,audio,.25,batch=32)
        tracker=edge.Tracker(model.classes,tracker_cfg,1.);events=[]
        for k,s in enumerate(scores):events.extend(tracker.step(s,1.+k*.25))
        events.extend(tracker.flush())
        starts=[e for e in events if e['event']=='start']
        confirmed=[e for e in events if e['event']=='end' and e['confirmed']]
        detected=any(e['label']==row.get('label') and e['end_s']>row.get('start_s',0) and
                     e['start_s']<row.get('end_s',len(audio)/16000) for e in confirmed)
        results.append(dict(row,detected=detected,trigger=bool(confirmed),start_alerts=len(starts),confirmed=len(confirmed),
                            start_by_class={c:sum(e['label']==c for e in starts) for c in model.classes},
                            confirmed_by_class={c:sum(e['label']==c for e in confirmed) for c in model.classes}))
    return results


def summarize(arms):
    output={'evidence_kind':'proxy screen','adoption':'inconclusive','arms':{},'comparisons':{}}
    # The WP2 module is absent on the frozen main base; do not call this a WP2 estimator.
    output['rate_method']='simple stand-in: exact edge.Tracker start and confirmed events per 100 fixed nuisance contexts; no prevalence or FA/h inference'
    for name,rows in arms.items():
        target=[r for r in rows if r['kind']=='target'];nuis=[r for r in rows if r['kind']=='nuisance']
        classes=sorted({r['label'] for r in target});recall={}
        for c in classes:
            group=[r for r in target if r['label']==c];k=sum(r['detected'] for r in group)
            recall[c]=dict(detected=k,n=len(group),rate=k/len(group),ci95=interval(k,len(group)))
        triggers=sum(r['trigger'] for r in nuis)
        output['arms'][name]=dict(recall=recall,nuisance_trigger_groups=triggers,nuisance_groups=len(nuis),
            nuisance_trigger_ci95=interval(triggers,len(nuis)),start_alerts=sum(r['start_alerts'] for r in nuis),
            confirmed=sum(r['confirmed'] for r in nuis),start_per_100=100*sum(r['start_alerts'] for r in nuis)/len(nuis),
            confirmed_per_100=100*sum(r['confirmed'] for r in nuis)/len(nuis),
            nuisance_by_family={f:dict(n=sum(r.get('family')==f for r in nuis),
                triggers=sum(r['trigger'] for r in nuis if r.get('family')==f)) for f in sorted({r['family'] for r in nuis})})
    tests=[]
    for name in arms:
        if name=='control':continue
        comparisons={}
        for kind,field in [('target','detected'),('nuisance','trigger')]:
            control=[r[field] for r in arms['control'] if r['kind']==kind]
            candidate=[r[field] for r in arms[name] if r['kind']==kind]
            result=mcnemar(control,candidate);comparisons[kind]=result;tests.append((name,kind,result))
        output['comparisons'][name]=comparisons
    previous=0.
    for rank,(name,kind,test) in enumerate(sorted(tests,key=lambda x:x[2]['p'])):
        test['holm_p']=max(previous,min(1.,test['p']*(len(tests)-rank)));previous=test['holm_p']
        better=test['candidate_only']>test['control_only'] if kind=='target' else test['candidate_only']<test['control_only']
        test['outcome']=('screen signal' if better else 'adverse signal') if test['holm_p']<.05 else 'inconclusive'
    return output


def run(args):
    prereg=verify_preregistration(args.prereg,args.prereg_sha)
    bank=verify_bank(args.manifest,args.bank)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.estimate:
        if (out/'estimate.json').exists():raise FileExistsError('estimate already exists')
        timings={}
        for arm,knobs in ARMS.items():
            start=time.perf_counter()
            nn.train(args.manifest,out/(arm+'_timing.npz'),**dict(SETTINGS,steps=30,val_windows=64),**knobs)
            timings[arm]=time.perf_counter()-start
        estimate=sum(timings.values())/30*1000
        result=dict(prereg_sha256=prereg,seconds_30_steps=timings,projected_five_arm_seconds=estimate,
                    budget_seconds=10800,launch_allowed=estimate<10800)
        (out/'estimate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return
    estimate=json.loads((out/'estimate.json').read_text())
    if not estimate['launch_allowed']:raise RuntimeError('prelaunch timing exceeds frozen budget')
    identity = dict(prereg_sha256=prereg, settings=SETTINGS, arms=ARMS,
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
        code_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                     for m in (nn,edge,recognizer,training_device,training_dynamics,iot_mic)},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    inputs=[json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    identity['training_input_sha256']={row['file']:hashlib.sha256((Path(args.manifest).parent/row['file']).read_bytes()).hexdigest() for row in inputs}
    identity_path=out/'run_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('screen inputs, code, settings or registration changed; use a fresh output directory')
    identity_path.write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
    arms={}
    for name,knobs in ARMS.items():
        model=out/(name+'.npz');report_path=out/(name+'_training.json');scores_path=out/(name+'_scores.json')
        if not report_path.exists():
            start=time.perf_counter();report=nn.train(args.manifest,model,**SETTINGS,**knobs)
            report.update(wall_seconds=time.perf_counter()-start,prereg_sha256=prereg,arm=name,evidence_kind='proxy screen')
            report_path.write_text(json.dumps(report,indent=2)+'\n');print(name,'trained',report['wall_seconds'],flush=True)
        if not scores_path.exists():
            rows=score(model,args.bank);scores_path.write_text(json.dumps(rows,indent=2)+'\n')
        arms[name]=json.loads(scores_path.read_text());print(name,'scored',flush=True)
    result=summarize(arms);result['prereg_sha256']=prereg
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['bank_sha256']=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','bank','prereg','prereg-sha','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--estimate',action='store_true');run(p.parse_args())
