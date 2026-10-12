"""Strict portable score rows and planned-denominator reconstruction summaries."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

FIELDS = ("scenario", "tier", "engine", "impairment", "eval_kind", "src", "mic", "band_hz",
          "metric", "truth", "reconstructed", "error", "unit", "gate", "pass", "notes")
KINDS = ("heldout", "novel", "real_sound", "parameter", "timing")


def finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def row(context, *, metric, truth, reconstructed, gate, unit, error=None,
        relative=False, lower=False, upper=False, notes="", truth_defined=None):
    """Missing prediction fails; missing truth is unavailable. Infinity is explicit."""
    if truth_defined is None:
        truth_defined = truth is not None and not np.isnan(truth)
    predicted = reconstructed is not None and not np.isnan(reconstructed)
    if error is None and truth_defined and predicted:
        error = (100*(reconstructed/truth-1) if relative and truth != 0 else
                 reconstructed-truth if not relative else None)
    passed = None
    if truth_defined:
        passed = bool(predicted and (reconstructed >= gate if lower else reconstructed <= gate if upper else
                                    error is not None and abs(error) <= gate))
    extra = []
    for label, value in (("truth", truth), ("reconstructed", reconstructed), ("error", error)):
        if value is not None and not np.isfinite(value):
            extra.append(f"{label}={value}")
    data = {**context, "metric": metric, "truth": finite(truth), "reconstructed": finite(reconstructed),
            "error": finite(error), "unit": unit,
            "gate": {"op": ">=" if lower else "<=" if upper else "abs<=", "value": gate}, "pass": passed,
            "notes": "; ".join(x for x in (notes, *extra) if x)}
    if set(data) != set(FIELDS) or data["eval_kind"] not in KINDS:
        raise ValueError("scorecard context/schema mismatch")
    return data


def _counts(rows):
    count = Counter("unavailable" if r["pass"] is None else "pass" if r["pass"] else "fail" for r in rows)
    p, f, u = (count[k] for k in ("pass", "fail", "unavailable"))
    return dict(pass_count=p, fail_count=f, unavailable_count=u, planned=p+f+u,
                pass_rate=p/(p+f+u) if p+f+u else None,
                evaluable_pass_rate=p/(p+f) if p+f else None,
                all_pass=bool(p and not (f+u)))


def summarize(rows, *, profile):
    groups, splits, scenarios = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        groups[(r["tier"], r["metric"], r["unit"])].append(r)
        splits[(r["tier"], r["eval_kind"])].append(r)
        scenarios[(r["tier"], r["scenario"])].append(r)
    accuracy = []
    for (tier, metric, unit), values in sorted(groups.items()):
        errors = [abs(r["error"]) for r in values if r["error"] is not None]
        record = dict(tier=tier, metric=metric, unit=unit, **_counts(values),
                      finite_errors=len(errors), median_absolute_error=float(np.median(errors)) if errors else None,
                      p90_absolute_error=float(np.quantile(errors, .9)) if errors else None)
        if "si_sdr" in metric:
            scores = [r["reconstructed"] for r in values if r["reconstructed"] is not None]
            record.update(median_score=float(np.median(scores)) if scores else None,
                          p10_score=float(np.quantile(scores, .1)) if scores else None)
        accuracy.append(record)
    return dict(version="room-closedloop/2", profile=profile,
                caveat="T0 is a same-model inverse crime, not measured reconstruction accuracy",
                overall=_counts(rows), accuracy=accuracy,
                splits=[dict(tier=t, eval_kind=k, **_counts(v)) for (t, k), v in sorted(splits.items())],
                scenarios=[dict(tier=t, scenario=s, **_counts(v)) for (t, s), v in sorted(scenarios.items())])


def write_scorecard(path, rows):
    rows = sorted(rows, key=lambda r: (tuple(str(r[k]) for k in (
        "scenario", "tier", "eval_kind", "src", "mic", "metric", "band_hz", "notes")),
        json.dumps(r, sort_keys=True, allow_nan=False)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for value in rows:
        if set(value) != set(FIELDS):
            raise ValueError("unexpected scorecard fields")
        lines.append(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
    data = "".join(lines).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def read_scorecard(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def check_baseline(actual, baseline):
    """Exact deterministic non-regression guard; timings/provenance are excluded."""
    actual, baseline = Path(actual), Path(baseline)
    a, b = (json.loads((p/"index.json").read_text()) for p in (actual, baseline))
    if (a["tier"], a["profile"]) != (b["tier"], b["profile"]):
        raise ValueError("baseline tier/profile differs")
    if set(a["scorecards"]) != set(b["scorecards"]):
        raise ValueError("baseline scenario selection differs")
    if set(a.get('reference_scorecards',[])) != set(b.get('reference_scorecards',[])):
        raise ValueError('baseline reference selection differs')
    names = [*a['scorecards'], *a.get('reference_scorecards',[])]
    changed = [name for name in names if read_scorecard(actual/name) != read_scorecard(baseline/name)]
    return dict(pass_check=not changed, changed_scorecards=changed,
                policy="exact deterministic score rows; review all changes, including improvements")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
