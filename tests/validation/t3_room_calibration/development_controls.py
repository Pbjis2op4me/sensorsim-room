"""Declared development controls; this executable cannot issue a P4 receipt.

Normal log-T30 conjugacy supplies an independent reference. The production late
update uses its own Cholesky/weight path. A C50 mutation must change the early
adequacy output while preserving late weights. P4 still needs verifier-owned
200-case fixtures, reference convergence and real-render/measurement controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

from sensorsim.acoustics.room.calibration_discrepancy import decompose_uncertainty, late_block_update, weighted_quantiles


def reachability(independent_cases=200, family_size=4, defect_coverage=.80):
    """Exact null rejection and power of simultaneous binomial count limits.

    Registered design proposal: familywise false rejection <= .05, detection
    >= .80 at a 0.10 absolute coverage defect. This is a calculation, not a
    demonstration that room proxy groups have binomial independence.
    """
    alpha = .05/family_size
    low, high = int(binom.ppf(alpha/2,independent_cases,.9)), int(binom.ppf(1-alpha/2,independent_cases,.9))
    rejection = lambda p: float(binom.cdf(low-1,independent_cases,p)+binom.sf(high,independent_cases,p))
    null, power = rejection(.9), rejection(defect_coverage)
    return dict(independent_cases=independent_cases,family_size=family_size,count_limits=[low,high],
        nominal_coverage=.9,defect_coverage=defect_coverage,null_false_rejection=null,
        familywise_union_bound=family_size*null,defect_detection=power,
        design_status="reachable" if family_size*null <= .05 and power >= .8 else "inconclusive")


def run_development(cases=32, seed=31001):
    if cases != 32 or seed != 31001:
        raise ValueError("this development fixture fixes 32 cases and seed 31001; amendments require a new fixture")
    rng = np.random.default_rng(seed)
    key, early = "1000.t30_s", "1000.c50_db"
    mean, prior_sd, noise_sd = np.log(.5), .3, .1
    # Deterministic normal midpoint quadrature; analytic reference uses no production code.
    theta = norm.ppf((np.arange(2048)+.5)/2048,loc=mean,scale=prior_sd)
    predictions = [{key:float(np.exp(t)),early:0.} for t in theta]
    mutated = [{key:p[key],early:3.} for p in predictions]
    records = []
    for i in range(cases):
        true_log_t = rng.normal(mean,prior_sd)
        y = rng.normal(true_log_t,noise_sd)
        budget = decompose_uncertainty([{key:float(np.exp(y)),early:0.}],keys=[key,early],
            units={key:"s",early:"dB"},repeat_ids=[f"dev-{i}"],
            content_sha256=[hashlib.sha256(f"dev-{seed}-{i}".encode()).hexdigest()],
            measurement_covariance=[[noise_sd**2,0.],[0.,1.]],playback_covariance=np.zeros((2,2)),
            simulator_covariance=np.zeros((2,2)),discrepancy_covariance=np.zeros((2,2)),policy_id="dev-conjugate/1")
        out = late_block_update(predictions,np.full(2048,1/2048),budget,early_tolerances={early:1.})
        bad = late_block_update(mutated,np.full(2048,1/2048),budget,early_tolerances={early:1.})
        variance = 1/(1/prior_sd**2+1/noise_sd**2)
        posterior_mean = variance*(mean/prior_sd**2+y/noise_sd**2)
        reference = np.exp(norm.ppf([.05,.5,.95],loc=posterior_mean,scale=np.sqrt(variance)))
        empirical = np.array(weighted_quantiles(np.exp(theta),out["weights"]))
        error = float(np.max(abs(empirical-reference))/np.exp(y))
        records.append(dict(case_id=f"dev-{i}",true_t30_s=float(np.exp(true_log_t)),observed_t30_s=float(np.exp(y)),
            reference_quantiles_s=reference.tolist(),candidate_quantiles_s=empirical.tolist(),
            maximum_relative_quantile_error=error,covered=bool(empirical[0]<=np.exp(true_log_t)<=empirical[2]),
            good_early=out["early_status"],bad_early=bad["early_status"],
            weights_unchanged_by_bad_early=out["weights"]==bad["weights"],
            effective_members=out["effective_members"],inference_status=out["status"]))
    return dict(label="proxy result",scope="development only; no acceptance launched",cases=records,
        oracle_quantile_tolerance_over_observed=.01,
        oracle_controls_pass=all(r["maximum_relative_quantile_error"]<=.01 for r in records),
        mechanism_controls_pass=all(r["good_early"]=="compatible" and r["bad_early"]=="mismatch" and r["weights_unchanged_by_bad_early"] for r in records),
        covered=sum(r["covered"] for r in records),total=cases,
        planned_statistical_reachability=reachability(),harness="unqualified",registered_outcome="inconclusive")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run_development()
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
    return 0 if result["oracle_controls_pass"] and result["mechanism_controls_pass"] else 1


if __name__=="__main__":
    raise SystemExit(main())
