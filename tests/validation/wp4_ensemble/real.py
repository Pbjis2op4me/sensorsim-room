"""WP4(b) decision runner for frozen WP1/WP2 outputs, pending real captures.

Input JSON: {label: measured, calibration_groups: [...], training_groups: [...], seeds: [row,...]}.
Each row has seed, steps_a/b, training_rows_a/b, takes_a/b (lists of
{source_group, label, hit}), auc_a/b, c2st_b: {point, lower, upper}, and
wp2_audit_ratio_upper. A/B must score the same held-out groups. The upstream
WP1/WP2 harness owns training, exact streaming detection and confidence bounds;
this runner cannot turn synthetic inputs into measured evidence.

python tests/validation/wp4_ensemble/real.py --input COMPARISON.json --out RESULT.json
A missing input writes pending, not a fabricated success. The JSON interface is
unchanged when real captures and the upstream evaluation outputs become available.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def recall_interval(hits, count):
    return [float(beta.ppf(.025,hits,count-hits+1)) if hits else 0.,
            float(beta.ppf(.975,hits+1,count-hits)) if hits<count else 1.]


def compare(payload):
    if payload.get('label') != 'measured':
        raise ValueError('WP4(b) requires measured held-out real takes')
    rows=payload['seeds']
    if sorted(r['seed'] for r in rows)!=[4108,4109,4110]:
        raise ValueError('need the three preregistered seeds')
    checks=[]
    for r in rows:
        if r['steps_a']!=r['steps_b'] or r['training_rows_a']!=r['training_rows_b']:
            raise ValueError('A/B must use equal steps and training data volume')
        a,b=({(t['source_group'],t['label']):t['hit'] for t in r['takes_'+side]} for side in ('a','b'))
        if not a or set(a)!=set(b) or len(a)!=len(r['takes_a']) or len(b)!=len(r['takes_b']):
            raise ValueError('A/B must score identical unique held-out takes')
        if any(type(hit) is not bool for hit in [*a.values(),*b.values()]):
            raise ValueError('take hits must be booleans')
        if {g for g,_ in a} & (set(payload['calibration_groups']) | set(payload['training_groups'])):
            raise ValueError('held-out source group leaked into calibration or training')
        c=r['c2st_b']
        if not 0<=c['lower']<=c['point']<=c['upper']<=1 or not 0<=r['auc_a']<=1 or r['auc_b']!=c['point']:
            raise ValueError('invalid AUC bounds or inconsistent B point AUC')
        outcome='pass' if c['upper']<=.70 and c['point']<=.60 else 'fail' if c['lower']>=.60 else 'inconclusive'
        per_class={}
        for label in sorted({label for _,label in a}):
            keys=[k for k in a if k[1]==label]
            ha,hb=sum(a[k] for k in keys),sum(b[k] for k in keys)
            per_class[label]=dict(a_hits=ha,b_hits=hb,takes=len(keys),difference=hb-ha,
                                 a_interval95=recall_interval(ha,len(keys)),b_interval95=recall_interval(hb,len(keys)))
        checks.append(dict(seed=r['seed'],classes=per_class,auc_a=r['auc_a'],auc_b=r['auc_b'],
            c2st_b=outcome,audit_ratio_upper=r['wp2_audit_ratio_upper']))
    labels=set(checks[0]['classes'])
    if labels != {'knock','doorbell','kettle'}:
        raise ValueError('need all three deployment target classes')
    if any(set(c['classes'])!=labels for c in checks):raise ValueError('class sets changed across seeds')
    recall_ok=all(np.median([c['classes'][label]['difference'] for c in checks])>=-1 for label in labels)
    auc_ok=np.median([c['auc_b']-c['auc_a'] for c in checks])<=0
    ratio_ok=np.median([c['audit_ratio_upper'] for c in checks])<=1.25
    domain_ok=sum(c['c2st_b']!='fail' for c in checks)>=2
    return dict(label='measured',acceptance='pass' if recall_ok and auc_ok and ratio_ok and domain_ok else 'fail',
        checks=checks,rule='median over three equal-budget seeds; C2ST must not fail on majority',
        scope='WP4(b) comparison only; no deployment FA/hour claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=compare(json.loads(args.input.read_text())) if args.input.exists() else dict(
        acceptance='pending',label='no measured result',reason='real held-out captures and WP1/WP2 comparison outputs required')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
