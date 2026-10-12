#!/usr/bin/env python3
"""Audit supplied room products on frozen, explicit truth support.

This is an implementer diagnostic, never an independent acceptance receipt.
Input JSON contains cases, required_keys, score_scales, seed and evidence_domain.
Each case has case_id, independent_group_id, truth, units, candidate and baseline
(the products from predictive_products), plus a partition declaration. JSON null
means unavailable truth. No RIR loader is implicit and no truth is recomputed
with a different noise policy to fill gaps. User recording summaries use the
same interface as proxies. Historical registrations may be attached verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sensorsim.acoustics.room.calibration_discrepancy import digest, metric_unit, BANDS

PRODUCTS = ("parameter_only", "actual_render", "discrepancy_inclusive")
REQUIRED_KEYS = tuple(f"{b}.{m}" for b in BANDS for m in ("t30_s", "edt_s", "c50_db"))


def validate_partition(partition):
    """Check declared internal development partitions before any data loading."""
    if partition.get("role") not in ("development", "permitted-heldout"):
        raise ValueError("explicit development or permitted-heldout partition required")
    if not partition.get("split_sha256") or not partition.get("independent_group_id"):
        raise ValueError("partition digest and independent group ID required")
    dataset = partition.get("dataset")
    if dataset not in ("synthetic", "MeshRIR", "dEchorate", "deployment"):
        raise ValueError("unsupported or undeclared data source")
    if dataset in ("MeshRIR", "dEchorate"):
        if partition.get("licence") != "CC-BY-4.0" or partition.get("use") != "internal-validation":
            raise ValueError("measured RIRs require recorded CC BY 4.0 and internal-only use")
    if dataset == "dEchorate":
        if partition.get("format") not in ("official-hdf5-v1", "official-hdf5-v2"):
            raise ValueError("only the permitted official HDF5 artifacts, not SOFA conversions")
        source, array = partition.get("source_id"), partition.get("array_id")
        if type(source) is not int or type(array) is not int or not 1 <= source <= 8 or not 1 <= array <= 5:
            raise ValueError("source/array outside the supported internal development subset")
    if dataset == "deployment" and partition.get("recording_kind") not in ("known-playback", "room-response"):
        raise ValueError("deployment room truth must distinguish playback from room response")


def _truth(value, key):
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not np.isfinite(value) or (metric_unit(key) == "s" and value <= 0):
        raise ValueError(f"invalid truth {key}; use null for unavailable cells")
    return float(value)


def _interval(product, key):
    if product is None or key not in product.get("intervals", {}):
        return None
    if product.get("nominal_interval") != .9 or product.get("units", {}).get(key) != metric_unit(key):
        raise ValueError("incompatible interval level or units")
    values = np.asarray(product["intervals"][key], float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(np.diff(values) < 0):
        raise ValueError("ordered finite low/median/high required")
    if metric_unit(key) == "s" and np.any(values <= 0):
        raise ValueError("decay intervals must be positive seconds")
    return values


def interval_score(truth, interval, scale):
    """Central-90% Winkler score in native units / a predeclared fixed scale."""
    lo, _, hi = interval
    return float((hi-lo + 20*max(lo-truth, 0) + 20*max(truth-hi, 0))/scale)


def _summarize(rows, seed):
    requested = len(rows)
    truth = [r for r in rows if r["truth_valid"]]
    predicted = [r for r in rows if r["predicted"]]
    scored = [r for r in rows if r["scoreable"]]
    covered = sum(r["covered"] for r in scored)
    groups = sorted({r["group"] for r in rows})
    totals = {g: dict(requested=sum(r["group"] == g for r in rows),
        scoreable=sum(r["group"] == g for r in scored),
        covered=sum(r["group"] == g and r["covered"] for r in scored)) for g in groups}
    # Resample whole independent groups, never position x band x metric cells.
    bounds = None
    if len(groups) >= 2 and scored:
        rng = np.random.default_rng(seed)
        counts = np.array([[totals[g]["covered"], totals[g]["scoreable"]] for g in groups])
        draws = counts[rng.integers(len(groups), size=(1000, len(groups)))].sum(axis=1)
        valid = draws[:, 1] > 0
        if valid.all():
            bounds = np.quantile(draws[:, 0]/draws[:, 1], [.025, .975]).tolist()
    return dict(requested=requested, truth_valid=len(truth), predicted=len(predicted),
        scoreable=len(scored), covered=covered,
        truth_available_fraction=len(truth)/requested if requested else None,
        coverage_all_requested=covered/requested if requested else None,
        coverage_scored=covered/len(scored) if scored else None,
        missing_prediction_on_truth_valid=len(truth)-len(scored),
        independent_group_count=len(groups), groups=totals,
        group_bootstrap_95=bounds, group_uncertainty="unavailable" if bounds is None else "approximate; declared groups, not cells; 1000 bootstrap draws",
        median_relative_halfwidth=_median([r["relative_halfwidth"] for r in scored if "relative_halfwidth" in r]),
        median_absolute_error=_median([r["absolute_error"] for r in scored]),
        median_relative_error=_median([r["relative_error"] for r in scored if "relative_error" in r]))


def _median(values):
    return float(np.median(values)) if values else None


def audit_products(document):
    """Proposed P4 screen numbers; all formal claims remain verifier-owned."""
    keys = document["required_keys"]
    if len(set(keys)) != len(keys) or set(keys) != set(REQUIRED_KEYS):
        raise ValueError("P4 requires exactly T30/EDT/C50 at 500/1000/2000/4000 Hz")
    scales = document["score_scales"]
    if set(scales) != set(keys) or any(not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("one positive preregistered native-unit score scale per metric required")
    domain = document["evidence_domain"]
    if domain not in ("synthetic proxy", "measured-RIR proxy", "deployment recording"):
        raise ValueError("explicit evidence domain required")
    cases = document["cases"]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("nonempty unique cases required")
    if type(document["seed"]) is not int or document["seed"] < 0:
        raise ValueError("explicit nonnegative bootstrap seed required")
    # Repeated content must not multiply the apparent independent group count.
    content = [c["truth_content_sha256"] for c in cases]
    if len(set(content)) != len(content) or any(len(h) != 64 for h in content):
        raise ValueError("duplicate or missing truth content digests")
    tables = {p: [] for p in PRODUCTS}
    paired = {p: [] for p in PRODUCTS}
    for case in cases:
        validate_partition(case["partition"])
        dataset = case["partition"]["dataset"]
        expected_domain = ("synthetic proxy" if dataset == "synthetic" else
            "deployment recording" if dataset == "deployment" else "measured-RIR proxy")
        if domain != expected_domain:
            raise ValueError("dataset and evidence-domain label disagree")
        if case["independent_group_id"] != case["partition"]["independent_group_id"]:
            raise ValueError("independent group ID disagrees with partition")
        if case["units"] != {k: metric_unit(k) for k in keys} or set(case["truth"]) != set(keys):
            raise ValueError("truth keys and native units must equal frozen required support")
        for product in PRODUCTS:
            candidate = case["candidate"].get(product)
            baseline = case["baseline"].get(product)
            for key in keys:
                y = _truth(case["truth"][key], key)
                ci, bi = _interval(candidate, key), _interval(baseline, key)
                row = dict(case_id=case["case_id"], group=case["independent_group_id"], key=key,
                    truth_valid=y is not None, predicted=ci is not None,
                    scoreable=y is not None and ci is not None, covered=False)
                if row["scoreable"]:
                    lo, mid, hi = ci
                    row.update(covered=bool(lo <= y <= hi), absolute_error=float(abs(mid-y)),
                        interval_score=interval_score(y, ci, scales[key]))
                    if metric_unit(key) == "s":
                        row.update(relative_halfwidth=float((hi-lo)/(2*y)), relative_error=float(abs(mid-y)/y))
                tables[product].append(row)
                if y is not None:
                    paired[product].append(None if ci is None or bi is None else
                        [interval_score(y, ci, scales[key]), interval_score(y, bi, scales[key])])
    output = dict(schema="room-calibration-audit/1", input_sha256=digest(document),
        evidence_domain=domain, result_label="measured" if domain == "deployment recording" else "proxy result",
        execution="completed", harness="unqualified", registered_outcome="inconclusive",
        review_decision="defer", integration="not checked", training_promotion=False,
        claim_type="acoustic-summary diagnostic; not distributional realism",
        historical_records=document.get("historical_records", []), products={})
    for product, rows in tables.items():
        summary = _summarize(rows, document["seed"])
        by_metric = {m: _summarize([r for r in rows if r["key"].endswith("."+m)], document["seed"])
                     for m in ("t30_s", "edt_s", "c50_db")}
        summary["by_metric"] = by_metric
        summary["cells"] = rows
        summary["constant_truth_keys"] = [key for key in keys
            if len({case["truth"][key] for case in cases if case["truth"][key] is not None}) == 1]
        pairs = paired[product]
        improvement = None
        if pairs and all(p is not None for p in pairs):
            means = np.mean(pairs, axis=0)
            if means[1] > 0:
                improvement = float(1-means[0]/means[1])
        summary["standardized_interval_score_improvement"] = improvement
        summary["score_support"] = "all truth-valid cells; no candidate-dependent filtering"
        missing = (any(m["truth_available_fraction"] < .9 or m["predicted"] != m["requested"]
                       for m in by_metric.values()) or not summary["scoreable"])
        failures = []
        t30 = by_metric["t30_s"]
        if t30["coverage_scored"] is not None and t30["coverage_scored"] < .75:
            failures.append("T30 coverage below 0.75")
        if t30["median_relative_halfwidth"] is not None and t30["median_relative_halfwidth"] > .4:
            failures.append("T30 half-width above 0.40")
        if product == "discrepancy_inclusive":
            if improvement is None:
                missing = True
            elif improvement < .2:
                failures.append("interval-score improvement below 0.20")
        if product == "actual_render":
            for m in ("edt_s", "c50_db"):
                if by_metric[m]["coverage_scored"] is not None and by_metric[m]["coverage_scored"] < .75:
                    failures.append(f"{m} coverage below 0.75")
            if by_metric["edt_s"]["median_relative_error"] is not None and by_metric["edt_s"]["median_relative_error"] > .2:
                failures.append("EDT median relative error above 0.20")
            if by_metric["c50_db"]["median_absolute_error"] is not None and by_metric["c50_db"]["median_absolute_error"] > 1:
                failures.append("C50 median absolute error above 1 dB")
        summary["screen_outcome"] = "fail" if failures else "inconclusive" if missing else "pass"
        summary["failures"] = failures
        summary["availability_gate"] = "inconclusive" if missing else "complete"
        output["products"][product] = summary
    output["promotion_blockers"] = ["independent verifier receipt required", "P4 synthetic controls and convergence qualification required",
        "frozen permitted measured-RIR partitions required", "P0 terminal operational audit required",
        "merged dataset path not audited by this summary tool"]
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    document = json.loads(raw)
    result = audit_products(document)
    result["input_file_sha256"] = hashlib.sha256(raw).hexdigest()
    result["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Refuse overwriting any earlier result, including a failure.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
