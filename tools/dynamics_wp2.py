# SPDX-License-Identifier: Apache-2.0
"""Optional WP2 scoring of the same frozen dynamics-screen nuisance contexts.

Uses float audio directly, with no extra PCM encoding. Short banks and open
trailing guards remain explicit failures of WP2 qualification; descriptive
counts do not become measured deployment rates. No prevalence is invented.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from sensorsim import edge, recognizer


def evaluate(models, bank_path, output, *, scorer=None):
    if scorer is None:
        from sensorsim.evaluation import fa_rate as scorer
    output=Path(output)
    if output.exists():raise FileExistsError('use a fresh WP2 output directory')
    output.mkdir(parents=True)
    bank_path=Path(bank_path)
    bank=[row for row in json.loads(bank_path.read_text()) if row['kind']=='nuisance']
    thresholds=dict(on=.5,off_ratio=.6,smooth_hops=3,min_hops=2,hop_s=.25)
    summary={'evidence_kind':'proxy result','detector_thresholds':thresholds,
             'fa_module_sha256':hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
             'bank_sha256':hashlib.sha256(bank_path.read_bytes()).hexdigest(),'arms':{},
             'rate_unit':'outputs per 100 whole nuisance contexts; no assumed prevalence or deployment FA/h'}
    for row in bank:
        if hashlib.sha256((bank_path.parent/row['audio']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('heldout audio changed before WP2 scoring')
    for name,model_path in models.items():
        model=edge.Model(str(model_path));trials=[];open_guards=[]
        for row in bank:
            audio=recognizer.read_audio(bank_path.parent/row['audio'],model.meta['features']['fs'])
            try:
                events=scorer.score_audio(model,audio,thresholds=thresholds,include_starts=True,require_closed_context=True)
            except ValueError as error:
                if 'active tracker' not in str(error):raise
                open_guards.append(row['source_group'])
                # Preserve the entire bank's descriptive counts, but mark the
                # qualification failure instead of silently dropping hard cases.
                events=scorer.score_audio(model,audio,thresholds=thresholds,include_starts=True)
            fs=model.meta['features']['fs'];n_windows=1+(len(audio)-model.features.n)//round(.25*fs)
            trials.extend(scorer.context_outcomes(events,model.classes,family=row['family'],context='isolated',
                context_id=row['source_group'],parent_id=row['source_group'],source_kind='model-based',
                max_confirmed_outputs={c:(n_windows+1)//3 for c in model.classes},
                max_start_outputs={c:(n_windows+1)//2 for c in model.classes}))
        options=dict(context_weights={'isolated':1.},bootstrap_replicates=4000,bootstrap_seed=1701)
        confirmed=scorer.summarize_challenges(trials,**options)
        starts=scorer.summarize_challenges([{**r,'confirmed_outputs':r['start_outputs'],
            'max_confirmed_outputs':r['max_start_outputs']} for r in trials],**options)
        counts={c:dict(start_alerts=sum(r['start_outputs'] for r in trials if r['class']==c),
                       confirmed=sum(r['confirmed_outputs'] for r in trials if r['class']==c)) for c in model.classes}
        summary['arms'][name]=dict(contexts=len(bank),by_class=counts,
            start_per_100=100*sum(r['start_outputs'] for r in trials)/len(bank),
            confirmed_per_100=100*sum(r['confirmed_outputs'] for r in trials)/len(bank),
            open_guard_contexts=open_guards,wp2_qualification='inconclusive: 16 parents/family, below WP2 100; '+
                ('open trailing guards also fail' if open_guards else 'guards pass'),
            confirmed_events=confirmed,start_alerts=starts,
            model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest())
        (output/(name+'_trials.json')).write_text(json.dumps(trials,indent=2)+'\n')
        print(name,'WP2 scored',flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',required=True,type=Path,help='JSON mapping arm name to model path, relative to this JSON')
    parser.add_argument('--bank',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();models=json.loads(args.models.read_text())
    evaluate({k:args.models.parent/v for k,v in models.items()},args.bank,args.out)
