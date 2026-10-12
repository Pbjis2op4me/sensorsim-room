# SPDX-License-Identifier: Apache-2.0
"""Proxy screen endpoints and paired statistics; no acceptance receipt authority."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, beta
from sensorsim import edge
from sensorsim.negatives.integration import verified_file
from sensorsim.negatives.composer import audio_hash


def mcnemar(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype != bool or right.dtype != bool or left.ndim != 1 or left.shape != right.shape or not left.size:
        raise ValueError("nonempty paired Boolean parent outcomes required")
    lost=int(np.sum(left & ~right)); gained=int(np.sum(~left & right)); n=lost+gained
    return dict(left_only=lost,right_only=gained,discordant=n,groups=len(left),
                p=float(binomtest(lost,n,.5).pvalue) if n else 1.)


def holm(tests):
    previous=0.
    for rank,row in enumerate(sorted(tests,key=lambda r:r["p"])):
        previous=max(previous,min(1.,row["p"]*(len(tests)-rank)))
        row["holm_p"]=previous
    return tests


def score_audit(model_path, bank_path, expected_sha256):
    from sensorsim.recognizer import read_audio
    from sensorsim.evaluation.fa_rate import score_audio, context_outcomes
    model=edge.Model(str(model_path)); path=Path(bank_path)
    data=json.loads(verified_file(path,expected_sha256)); results=[];wp2=[]
    groups=set();content=set()
    for row in data["rows"]:
        verified_file(path.parent/row["audio"],row["file_sha256"])
        x=read_audio(path.parent/row["audio"],model.meta["features"]["fs"])
        h=audio_hash(x)
        if h != row["audio_sha256"] or h in content or row["source_group"] in groups:
            raise ValueError("audit content changed or duplicated")
        if x.size < model.features.n or not np.isfinite(x).all() or np.ptp(x)==0:
            raise ValueError("empty/constant audit context")
        content.add(h); groups.add(row["source_group"])
        events=score_audio(model,x,include_starts=True)
        starts=[e for e in events if e["event"]=="start"]
        ends=[e for e in events if e["event"]=="end" and e["confirmed"]]
        detected=any(e["label"]==row.get("label") and e["end_s"]>row.get("start_s",0) and e["start_s"]<row.get("end_s",len(x)/model.meta["features"]["fs"]) for e in ends)
        results.append(dict(source_group=row["source_group"],kind=row["kind"],label=row.get("label"),family=row.get("family"),
            detected=detected,trigger=bool(ends),starts=len(starts),confirmed=len(ends),
            confirmed_by_class={c:sum(e["label"]==c for e in ends) for c in model.classes}))
        if row["kind"]=="nuisance":
            nwin=1+(len(x)-model.features.n)//round(model.meta["detector"]["hop_s"]*model.meta["features"]["fs"])
            wp2.extend(context_outcomes(events,model.classes,family=row["family"],context="isolated",
                context_id=row["source_group"],parent_id=row["source_group"],
                max_confirmed_outputs={c:nwin for c in model.classes},max_start_outputs={c:nwin for c in model.classes},
                source_kind="model-based"))
    return dict(outcomes=results,wp2=wp2,summary=summarize(results,model.classes),detector=model.meta["detector"])


def summarize(rows,classes):
    target=[r for r in rows if r["kind"]=="target"]; nuisance=[r for r in rows if r["kind"]=="nuisance"]
    if not target or not nuisance: raise ValueError("target and nuisance denominators required")
    recall={}
    for c in classes:
        group=[r for r in target if r["label"]==c]; n=len(group);k=sum(r["detected"] for r in group)
        if not n: raise ValueError("missing audit class")
        recall[c]=dict(detected=k,groups=n,rate=k/n,ci95=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.])
    matrix={}
    for r in rows:
        key=r["label"] if r["kind"]=="target" else r.get("family") or "background"
        counts=matrix.setdefault(key,dict.fromkeys(classes,0))
        for c in classes: counts[c] += int(r["confirmed_by_class"][c]>0)
    return dict(evidence_domain="synthetic proxy",recall=recall,
        nuisance_groups=len(nuisance),nuisance_trigger_groups=sum(r["trigger"] for r in nuisance),
        wp2_start_per_100_contexts=100*sum(r["starts"] for r in nuisance)/len(nuisance),
        wp2_confirmed_per_100_contexts=100*sum(r["confirmed"] for r in nuisance)/len(nuisance),
        class_swap_matrix=matrix,deployment_false_alarms_per_hour=None)


def compare(left,right):
    a={r["source_group"]:r for r in left};b={r["source_group"]:r for r in right}
    if set(a)!=set(b) or len(a)!=len(left) or len(b)!=len(right):
        raise ValueError("paired screen parents differ or repeat")
    tests=[]
    for kind,field in (("target","detected"),("nuisance","trigger")):
        groups=sorted(k for k in a if a[k]["kind"]==kind)
        if any(b[k]["kind"]!=kind for k in groups): raise ValueError("paired kind mismatch")
        tests.append(dict(endpoint=kind,**mcnemar(np.array([a[k][field] for k in groups],bool),np.array([b[k][field] for k in groups],bool))))
    return tests
