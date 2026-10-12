"""Exposure-aware false-alarm estimates using whole nuisance contexts.

A version-3 manifest contains ``floor_streams`` (target-free PCM WAVs),
``challenges`` (whole guarded nuisance episodes), and an explicit
``context_prevalence_per_hour`` table. Each challenge has ``family``, ``path``,
``context`` (a mutually exclusive stratum), and optionally annotated ``events``.
``context_weights`` declares the prevalence mixture of those strata, e.g.
{"impact": {"clustered": 0.7, "overlapping": 0.3}}. Flat legacy weights
are explicitly broadcast to each family; nested weights allow different strata. Mixed-family scenes need their own family
and incidence; do not count their constituent sources again in other families.

Schema /3 reports both start_alerts (what edge listen emits immediately) and
confirmed_events (confirmed end lines), each with its own rate and interval.
The confirmed-only scoring API stays the default; include_starts opts into
actual tracker start records. Missing starts cannot be inferred from ends.

Count every confirmed edge.Detector output once per context and class, including
multiple outputs from a single cause. Source annotations are only an audit of
matched instances; they never supply the rate estimate. All contexts need quiet
lead-in and trailing guards sufficient for the model window, smoothing and
hysteresis to reset. An active tracker at the end of a context is rejected.

The additive model assumes disjoint parent contexts and a separately estimated
floor contribution. ``floor_time_fraction`` (default 1) may declare the fraction
of time outside the scored contexts to avoid counting their background twice.
Context incidence and this fraction are assumptions unless explicitly measured.
Unmodelled interactions between contexts invalidate this decomposition.

Intervals resample independent parent contexts within strata, preserving bursts
and count over-dispersion. Floor blocks are resampled with an exact Poisson
sparse-count envelope; the latter assumes an independent floor process. These
are approximate confidence intervals for a mean rate, not prediction intervals
for an arbitrary finite household recording. No measured deployment rate is
claimed from synthetic exposure or assumed prevalence. Sparse floor points must
be read beside their upper bounds. floor_resolved means the floor's allocated
upper bound meets a declared floor budget (default 1/h), not that the composite
meets it. With zero events, 95% one-sided resolution at 1/h needs 2.996 hours
(round up to 3.0); the composite floor needs 5.075 or 5.635 hours for 4 or 7
Bonferroni components. These horizons assume no further events and Poisson floor.

The frozen proxy check's historical pass is not calibration. The independent
review's diagnostic pass rates were 0.64 at x0.8 and 0.69 at x1.0 of the challenge
component: the rule has low power and favours under-reporting. Neither it nor
this mean-rate interval predicts an arbitrary finite stream. Context sufficiency
requires 100 independent parents per family and 20 per stratum; constant counts
retain a physical-cap deviation bound and can be scored. "Scored" certifies only
this engineering sample-size gate, not representativeness, interval precision,
coverage calibration, a passed FA budget, or measured deployment validity.

Rate-free headlines count triggering whole nuisance episodes per 100 contexts,
not matched source annotations or output counts. A separate composite assumes
at most ``outputs_per_trigger_assumption`` outputs per triggering context
(default 1); it is labelled as an assumption and suppressed if contradicted by
the bank. The physical-cap composite remains the primary output-count bound.

For deployment banks, overlay whole nuisance episodes on segments of the user's
target-free recordings, keeping lead/trailing guards. Render overlapping sources
as a single mixed waveform before CNN scoring, never combine source scores.
An overlay_path manifest invokes the negatives composer and shared mixer on
declared disjoint recordings; supplied waveforms can also be scored. Overlays
use the deployment settings without overrides and remain proxy evidence. Estimate
family incidence and level weights roughly from annotated deployment audio;
incidence x0.3/x3 and the full supplied-stratum mixture range are reported by
default. These sensitivities cannot cover missing families or unseen levels.

Before calibration, split each recording into reserved, nonoverlapping extents.
Write each extent to its own PCM WAV. Every calibration_streams, floor_streams,
plain challenge, overlay background and recorded dry-source entry requires
``recording_id`` (the original recording, shared across its excerpts),
``sample_rate_hz`` (the model rate) and ``sample_range: [start, end]`` (half-open
original-recording sample indices, whose difference equals the WAV length).
For example, reserve [0, C] for calibration, [C, C+F] for floor and [C+F, N]
for contexts in an N-sample recording. Calibrate only on the first, with
``edge calibrate ... --recording-id session-1 --sample-start 0``; reserve the
others for scoring. Size each split before tuning: the default 1/hour calibration
requires at least three hours; held-out floor exposure has its own bound.
Keep parent_id stable for dependent contexts;
disjoint time spans alone do not make independent statistical groups.

All manifest paths check the model's saved calibration digests, declared ranges
and exact decoded block fingerprints before rendering/scoring. Include original
calibration_streams for legacy models; [] never disables saved-model checks.
Legacy models without records warn and report the gap, rather than crash.
Fingerprint blocks span one second (at least eight samples). An undeclared crop
must contain a full nonconstant stored block to be caught; two block lengths
suffice for alignment, but constant blocks are omitted. Gain changes, resampling
and shorter overlaps can escape exact hashes. No robust fingerprint is claimed.
Ranges and file lengths use one mono sample timeline; manifest inputs must
already be at the model rate. Direct score_audio/score_wav are low-level detector
utilities, not recording-partition audits; use the manifest entry point for FA
evaluation. None of these checks establishes measured deployment performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2

from .intervals import clopper_pearson, constant_count_interval
from .recordings import audit_manifest

from sensorsim import edge


DEFAULT_CONTEXT_WEIGHTS = {"isolated": 0.5, "contextual": 0.5}
DEFAULT_BOOTSTRAP_REPLICATES = 4000
MIN_MEASURED_EXPOSURE_S = 30 * 60
MIN_PARENT_CONTEXTS = 100
MIN_PARENTS_PER_STRATUM = 20
DEFAULT_FLOOR_BUDGET_PER_HOUR = 1.0
STATUS_MEANING = ("scored means the independent-context sample-size gate is met; "
                  "not representativeness, precision, calibration, an FA-budget pass or deployment validity")
BANK_PROTOCOL = {
    "bed": "overlay nuisances on user target-free segments before scoring",
    "episode": "each context holds the whole nuisance episode plus reset guards",
    "superposition": "render one mixed waveform before CNN scoring; use its own family and incidence",
    "prevalence": "rough episode counts / observed hours; assumptions unless measured",
    "levels": "rough deployment level weights; inspect full supplied-stratum mixture sensitivity",
    "renderer": "optional overlay_path uses negatives.composer and the shared device mixer",
}


def _validate_count_exposure(count: int, exposure_s: float) -> tuple[int, float]:
    count = int(count)
    exposure_s = float(exposure_s)
    if count < 0:
        raise ValueError("event count must be nonnegative")
    if not math.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError("exposure_s must be finite and positive")
    return count, exposure_s


def cowan_poisson_upper(count: int, exposure_s: float, confidence: float = 0.95) -> float:
    """Exact one-sided Poisson upper rate bound per hour.

    For zero, one, and two events at 95%, the count-scale bounds are 3.00,
    4.74, and 6.30.  `exposure_s` is the actual continuous or aggregate
    exposure represented by the count; callers retain the exposure label.
    """
    count, exposure_s = _validate_count_exposure(count, exposure_s)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    upper_count = 0.5 * float(chi2.ppf(confidence, 2 * (count + 1)))
    return upper_count * 3600.0 / exposure_s


def poisson_rate_interval(count: int, exposure_s: float, confidence: float = 0.95) -> tuple[float, float]:
    """Central exact Garwood interval for a Poisson event rate, per hour."""
    count, exposure_s = _validate_count_exposure(count, exposure_s)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    alpha = 1.0 - confidence
    lower = 0.0 if count == 0 else 0.5 * float(chi2.ppf(alpha / 2, 2 * count))
    upper = 0.5 * float(chi2.ppf(1 - alpha / 2, 2 * (count + 1)))
    scale = 3600.0 / exposure_s
    return lower * scale, upper * scale



def exposure_label(source_kind: str, exposure_s: float, *, longest_continuous_s: float | None = None) -> str:
    """Classify evidence without treating synthetic duration as measured exposure."""
    source_kind = str(source_kind).strip().lower()
    if source_kind in {"proxy", "synthetic", "model", "model-based"}:
        return "model-based"
    if source_kind not in {"measured", "real"}:
        raise ValueError("source_kind must be 'measured' or 'model-based'")
    continuous = float(longest_continuous_s if longest_continuous_s is not None else exposure_s)
    if not math.isfinite(continuous) or continuous < 0 or continuous > float(exposure_s):
        raise ValueError("longest_continuous_s must be between zero and exposure_s")
    if continuous >= MIN_MEASURED_EXPOSURE_S:
        return "measured"
    return "measured-smoke-test-not-FA/hour"


def _source_label(source_kind: str) -> str:
    source_kind = str(source_kind).strip().lower()
    if source_kind in {"measured", "real"}:
        return "measured"
    if source_kind in {"proxy", "synthetic", "model", "model-based"}:
        return "model-based"
    raise ValueError("source_kind must be 'measured' or 'model-based'")


def summarize_floor(
    event_times_s: Iterable[float],
    exposure_s: float,
    *,
    source_kind: str = "model-based",
    longest_continuous_s: float | None = None,
    confidence: float = 0.95,
    block_s: float = 10.0,
    floor_budget_per_hour: float = DEFAULT_FLOOR_BUDGET_PER_HOUR,
) -> dict[str, Any]:
    """Return a floor/bed rate, exact exposure bound, and block counts.

    For a measured input shorter than 30 continuous minutes, the numerical
    count rate remains a smoke-test diagnostic and the output label explicitly
    prevents it being read as a measured FA/hour result.
    """
    times = sorted(float(t) for t in event_times_s)
    count, exposure_s = _validate_count_exposure(len(times), exposure_s)
    if any(not math.isfinite(t) or t < 0 or t > exposure_s for t in times):
        raise ValueError("event times must lie inside the declared exposure")
    if block_s <= 0 or not math.isfinite(block_s):
        raise ValueError("block_s must be finite and positive")
    if not math.isfinite(floor_budget_per_hour) or floor_budget_per_hour <= 0:
        raise ValueError("floor_budget_per_hour must be finite and positive")
    upper_one_sided = cowan_poisson_upper(count, exposure_s, confidence)
    rate = count * 3600.0 / exposure_s
    lower, upper = poisson_rate_interval(count, exposure_s, confidence)
    continuous_s = float(longest_continuous_s if longest_continuous_s is not None else exposure_s)
    if not math.isfinite(continuous_s) or continuous_s < 0 or continuous_s > exposure_s:
        raise ValueError("longest_continuous_s must be between zero and exposure_s")
    label = exposure_label(source_kind, exposure_s, longest_continuous_s=continuous_s)
    n_blocks = max(1, int(math.ceil(exposure_s / block_s)))
    block_counts = np.zeros(n_blocks, dtype=int)
    for time_s in times:
        block_counts[min(int(time_s / block_s), n_blocks - 1)] += 1
    return {
        "events": count,
        "exposure_s": exposure_s,
        "exposure_hours": exposure_s / 3600.0,
        "longest_continuous_s": continuous_s,
        "observed_count_rate_per_hour": rate,
        "rate_per_hour": rate if label != "measured-smoke-test-not-FA/hour" else None,
        "upper_bound_per_hour_95_one_sided": upper_one_sided,
        "floor_resolved": upper_one_sided <= floor_budget_per_hour,
        "floor_resolution_rule": "Poisson upper bound <= declared floor budget; not a deployment guarantee",
        "floor_budget_per_hour": floor_budget_per_hour,
        "sparse_count": count <= 2,
        "zero_event_hours_for_budget_one_sided": -math.log(1 - confidence) / floor_budget_per_hour,
        "headline": {"point_per_hour": rate if label != "measured-smoke-test-not-FA/hour" else None,
                     "evidence_label": label, "upper_bound_per_hour": upper_one_sided,
                     "confidence": confidence, "bound": "one-sided Poisson"},
        "interval_95_two_sided_per_hour": [lower, upper],
        "evidence_label": label,
        "interval_method": "exact-Poisson-Garwood",
        "block_s": float(block_s),
        "block_counts": block_counts.tolist(),
        "event_times_s": times,
    }


def confirmed_events_from_scores(
    model: Any,
    scores: np.ndarray,
    *,
    hop_s: float | None = None,
    thresholds: dict | None = None,
) -> list[dict]:
    """Run precomputed scores through edge.track and the shipped Tracker."""
    hop_s = float(hop_s or model.meta["detector"]["hop_s"])
    return edge.track(model, np.asarray(scores, dtype=np.float32), hop_s, thresholds)


def _confirmed_only(events: Iterable[dict]) -> list[dict]:
    return [dict(event) for event in events if event.get("confirmed", False)]


def _reported_events(events: Iterable[dict], include_starts: bool) -> list[dict]:
    return [dict(e) for e in events if e.get("confirmed", False)
            or (include_starts and e.get("event") == "start")]


def score_audio(
    model: edge.Model,
    audio: np.ndarray,
    *,
    thresholds: dict | None = None,
    chunk_s: float = 5.0,
    require_closed_context: bool = False,
    include_starts: bool = False,
) -> list[dict]:
    """Score with edge.Detector; optionally retain starts as well as confirmed ends."""
    fs = int(model.meta["features"]["fs"])
    if chunk_s <= 0 or not math.isfinite(chunk_s):
        raise ValueError("chunk_s must be finite and positive")
    samples = np.asarray(audio)
    detector = edge.Detector(model, thresholds=thresholds)
    step = max(1, int(round(chunk_s * fs)))
    events: list[dict] = []
    for start in range(0, samples.size, step):
        events.extend(_reported_events(detector.push(samples[start : start + step]), include_starts))
    if require_closed_context and detector.tracker.active:
        raise ValueError("context ends with an active tracker; extend its trailing guard")
    events.extend(_reported_events(detector.flush(), include_starts))
    return events


def _decode_pcm(raw: bytes, width: int, channels: int) -> np.ndarray:
    if width == 1:
        samples = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        values = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        values = (values << 8) >> 8
        samples = values.astype(np.float32) / 8388608.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported integer PCM width: {width} bytes")
    return samples.reshape(-1, channels).mean(axis=1).astype(np.float32)


def score_wav(
    model: edge.Model,
    path: str | Path,
    *,
    thresholds: dict | None = None,
    chunk_s: float = 5.0,
    require_closed_context: bool = False,
    include_starts: bool = False,
) -> tuple[list[dict], float]:
    """Stream a PCM WAV through edge.Detector; resample with edge.read_wav if needed."""
    path = Path(path)
    fs = int(model.meta["features"]["fs"])
    if chunk_s <= 0 or not math.isfinite(chunk_s):
        raise ValueError("chunk_s must be finite and positive")
    with wave.open(str(path), "rb") as wav:
        source_rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        frames = wav.getnframes()
        exposure_s = frames / float(source_rate)
        if source_rate != fs:
            # edge.read_wav is the export's canonical integer-ratio resampler.
            audio = edge.read_wav(str(path), fs)
            return score_audio(model, audio, thresholds=thresholds, chunk_s=chunk_s,
                               require_closed_context=require_closed_context, include_starts=include_starts), exposure_s
        detector = edge.Detector(model, thresholds=thresholds)
        frames_per_chunk = max(1, int(round(chunk_s * source_rate)))
        events: list[dict] = []
        while True:
            raw = wav.readframes(frames_per_chunk)
            if not raw:
                break
            chunk = _decode_pcm(raw, width, channels)
            events.extend(_reported_events(detector.push(chunk), include_starts))
        if require_closed_context and detector.tracker.active:
            raise ValueError("context ends with an active tracker; extend its trailing guard")
        events.extend(_reported_events(detector.flush(), include_starts))
    return events, exposure_s


def context_outcomes(
    events: Iterable[dict],
    classes: Iterable[str],
    *,
    family: str,
    context: str,
    context_id: str,
    max_confirmed_outputs: dict[str, int],
    max_start_outputs: dict[str, int] | None = None,
    parent_id: str | None = None,
    instances: Iterable[dict] = (),
    source_kind: str = "model-based",
) -> list[dict]:
    """Package exact tracker counts; matching source intervals is diagnostic only.

    ``max_confirmed_outputs`` is the physical per-record bound given its window
    count and tracker minimum duration, not an observed maximum or tuning knob.
    A parent ID links dependent records for cluster resampling. A whole activity
    burst should normally be one record, with any mixed/overlapping sources in it.
    """
    events = list(events)
    confirmed = _confirmed_only(events)
    instances = list(instances)
    rows = []
    for label in classes:
        outputs = [event for event in confirmed if event["label"] == label]
        hits = sum(any(event["start_s"] <= instance["end_s"]
                       and event["end_s"] >= instance["start_s"] for event in outputs)
                   for instance in instances)
        rows.append({
            "family": str(family), "class": str(label), "context_group": str(context),
            "context_id": str(context_id), "parent_id": str(parent_id or context_id),
            "confirmed_outputs": len(outputs),
            "max_confirmed_outputs": max_confirmed_outputs[label],
            "source_instances": len(instances), "matched_instances": hits,
            "evidence_label": _source_label(source_kind),
        })
        if max_start_outputs is not None:
            rows[-1].update(start_outputs=sum(e.get("event") == "start" and e["label"] == label for e in events),
                            max_start_outputs=max_start_outputs[label])
    return rows


def score_challenge_records(
    model: edge.Model,
    records: Iterable[dict],
    *,
    thresholds: dict | None = None,
    chunk_s: float = 5.0,
) -> list[dict]:
    """One output count per guarded context/class through the exact edge.Detector."""
    rows = []
    seen_paths = set()
    cfg = {**model.meta["detector"], **(thresholds or {})}
    fs, n = model.meta["features"]["fs"], model.features.n
    hop = int(round(cfg["hop_s"] * fs))
    for index, record in enumerate(records):
        if "_audio" in record:
            audio = record["_audio"]
            events = score_audio(model, audio, thresholds=thresholds, chunk_s=chunk_s,
                                 require_closed_context=True, include_starts=True)
            exposure = len(audio)/fs
        else:
            path = Path(record["path"])
            if path.resolve() in seen_paths:
                raise ValueError("duplicate context path; use one mixed-context family instead")
            seen_paths.add(path.resolve())
            events, exposure = score_wav(model, path, thresholds=thresholds, chunk_s=chunk_s,
                                         require_closed_context=True, include_starts=True)
        instances = record.get("events", record.get("instances", []))
        for instance in instances:
            if not 0 <= float(instance["start_s"]) <= float(instance["end_s"]) <= exposure:
                raise ValueError("source annotation lies outside its context")
        n_windows = max(0, 1 + (round(exposure * fs) - n) // hop)
        bounds = {}
        for label in model.classes:
            min_hops = cfg["min_hops"]
            minimum = int(min_hops[label] if isinstance(min_hops, dict) else min_hops)
            bounds[label] = (n_windows + 1) // (minimum + 1)
        rows.extend(context_outcomes(
            events, model.classes, family=record["family"],
            context=str(record.get("context", "isolated")),
            context_id=str(record.get("context_id", f"context-{index}")),
            parent_id=record.get("parent_id", record.get("cluster_id")),
            max_confirmed_outputs=bounds,
            max_start_outputs={label: (n_windows + 1) // 2 for label in model.classes}, instances=instances,
            source_kind=record.get("source_kind", "model-based"),
        ))
    for row in rows:
        row["detector_settings"] = cfg
    return rows


def _seed_for(seed: int, *parts: str) -> int:
    # Stable seed splitting: avoid additive character-sum collisions across strata.
    key = json.dumps([int(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _context_data(trials: Iterable[dict], context_weights: dict[str, Any] | None):
    rows = list(trials)
    if not rows:
        raise ValueError("at least one context outcome is required")
    for field in ("detector_settings", "model_sha256"):
        if any(field in r for r in rows):
            identities = {json.dumps(r.get(field), sort_keys=True) for r in rows}
            if len(identities) != 1 or any(field not in r for r in rows):
                raise ValueError(f"context rows have missing or different {field}; rescore with one model/config")
    if any("confirmed_outputs" not in row for row in rows):
        raise ValueError("per-instance success rows cannot estimate rates; rescore whole contexts")
    families = {r["family"] for r in rows}
    supplied = dict(DEFAULT_CONTEXT_WEIGHTS if context_weights is None else context_weights)
    nested = any(isinstance(w, dict) for w in supplied.values())
    weights = supplied if nested else {family: supplied for family in families}
    if set(weights) != families:
        raise ValueError("context weights must match the scored families exactly")
    normalized = {}
    for family, mixture in weights.items():
        if not isinstance(mixture, dict) or not mixture or any(
            not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in mixture.values()
        ):
            raise ValueError("context weights must be finite and positive per family")
        normalized[family] = {key: value / sum(mixture.values()) for key, value in mixture.items()}
    weights = normalized
    seen = set()
    parent_strata = {}
    for row in rows:
        if "confirmed_outputs" not in row:
            raise ValueError("per-instance success rows cannot estimate rates; rescore whole contexts")
        for field in ("confirmed_outputs", "max_confirmed_outputs"):
            value = row[field]
            if not math.isfinite(value) or value < 0 or int(value) != value:
                raise ValueError(f"{field} must be a nonnegative integer")
        if row["confirmed_outputs"] > row["max_confirmed_outputs"]:
            raise ValueError("confirmed outputs exceed the physical context bound")
        if row["context_group"] not in weights[row["family"]]:
            raise ValueError("context outcome has no declared stratum weight")
        identity = (row["context_id"], row["class"])
        if identity in seen:
            raise ValueError("duplicate context/class outcome")
        seen.add(identity)
        parent = str(row["parent_id"])
        stratum = (row["family"], row["context_group"])
        if parent in parent_strata and parent_strata[parent] != stratum:
            raise ValueError("dependent parents cross strata; combine into a mixed context family")
        parent_strata[parent] = stratum
    return rows, weights


def _count_bootstrap(rows: list[dict], replicates: int, seed: int) -> tuple[np.ndarray, int]:
    """Resample parent clusters, keeping all their counts and denominators together."""
    parents = defaultdict(list)
    for row in rows:
        parents[str(row["parent_id"])].append(int(row["confirmed_outputs"]))
    counts = np.asarray([sum(values) for values in parents.values()], dtype=float)
    sizes = np.asarray([len(values) for values in parents.values()], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(counts), size=(replicates, len(counts)))
    return counts[picks].sum(axis=1) / sizes[picks].sum(axis=1), len(counts)


def _count_interval(rows: list[dict], draws: np.ndarray, confidence: float) -> list[float]:
    alpha = 1 - confidence
    interval = [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))]
    # A constant observed count c is not population certainty. Bound the
    # probability of an unseen deviation from c at the component alpha.
    # The shared helper proves validity across all data-selected values of c.
    # Use the physical cap, never the largest count observed in the sample.
    counts = [row["confirmed_outputs"] for row in rows]
    if len(set(counts)) == 1:
        parents = {row["parent_id"] for row in rows}
        cap = max(row["max_confirmed_outputs"] for row in rows)
        c = counts[0]
        if len(parents) == len(rows):
            interval = list(constant_count_interval(c, len(parents), cap, confidence))
        else:
            # Unequal/dependent parent records need a parent-level bound.
            interval = [0.0, float(cap)]
    return interval


def _trigger_summary(group: list[dict], confidence: float, replicates: int, seed: int) -> dict:
    """P(whole context emits any output), preserving dependence between records."""
    binary = [{**r, "confirmed_outputs": int(r["confirmed_outputs"] > 0),
               "max_confirmed_outputs": 1} for r in group]
    independent = len({r["parent_id"] for r in group}) == len(group)
    successes = sum(r["confirmed_outputs"] for r in binary)
    if independent:
        interval = clopper_pearson(successes, len(group), confidence)
        method = "Clopper-Pearson; independent whole contexts"
    else:
        draws, _ = _count_bootstrap(binary, replicates, seed)
        interval = _count_interval(binary, draws, confidence)
        method = "parent-cluster bootstrap; full range for constant dependent counts"
    return {"probability": successes / len(group), "interval": list(interval), "interval_method": method}


def summarize_challenges(
    trials: Iterable[dict], *, context_weights: dict[str, Any] | None = None,
    confidence: float = 0.95, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = 20261012, dispersion_corrected: bool | None = None,
) -> dict[str, Any]:
    """Expected output counts per context, stratified parent-cluster intervals.

    Binary per-instance trigger probabilities are deliberately not accepted:
    those data have lost merging and splitting information. The correction is
    always applied, even if a legacy caller passes dispersion_corrected=False.
    A rate-free Bernoulli summary is derived from whole-context counts separately.
    Scored describes the sample-size gate only; it does not certify a matched bank.
    """
    if not 0 < confidence < 1 or bootstrap_replicates < 100:
        raise ValueError("require 0 < confidence < 1 and at least 100 bootstrap replicates")
    rows, weights = _context_data(trials, context_weights)
    families, classes = sorted({r["family"] for r in rows}), sorted({r["class"] for r in rows})
    summaries = {}
    for family in families:
        summaries[family] = {}
        for label in classes:
            subset = [r for r in rows if r["family"] == family and r["class"] == label]
            by_context = {}
            point = lower = upper = 0.0
            trigger_point = 0.0
            trigger_bounds = np.zeros(2)
            trigger_methods = set()
            sufficient = True
            for context, weight in weights[family].items():
                group = [r for r in subset if r["context_group"] == context]
                if not group:
                    raise ValueError(f"missing contexts for {family}/{label}/{context}")
                counts = np.asarray([r["confirmed_outputs"] for r in group], dtype=float)
                draws, parents = _count_bootstrap(group, bootstrap_replicates,
                                                 _seed_for(bootstrap_seed, family, label, context))
                interval = _count_interval(group, draws, 1 - (1 - confidence) / len(weights[family]))
                mean = float(counts.mean())
                point += weight * mean
                lower += weight * interval[0]
                upper += weight * interval[1]
                sufficient &= parents >= MIN_PARENTS_PER_STRATUM
                trigger = _trigger_summary(group, 1 - (1 - confidence) / len(weights[family]),
                                           bootstrap_replicates, _seed_for(bootstrap_seed, family, label, context, "trigger"))
                trigger_point += weight * trigger["probability"]
                trigger_bounds += weight * np.asarray(trigger["interval"])
                trigger_methods.add(trigger["interval_method"])
                by_context[context] = {
                    "contexts": len(group), "independent_parents": parents,
                    "confirmed_outputs": int(counts.sum()), "mean_outputs_per_context": mean,
                    "outputs_per_100_contexts": 100 * mean,
                    "sample_variance": float(counts.var(ddof=1)) if len(counts) > 1 else None,
                    "variance_to_mean": float(counts.var(ddof=1) / mean) if len(counts) > 1 and mean else None,
                    "interval_bonferroni": interval,
                    "trigger_probability": trigger,
                }
            source_count = sum(r.get("source_instances", 0) for r in subset)
            parent_count = len({r["parent_id"] for r in subset})
            sufficient &= len(subset) >= MIN_PARENT_CONTEXTS and parent_count >= MIN_PARENT_CONTEXTS
            sources = {_source_label(r.get("evidence_label", "model-based")) for r in subset}
            summaries[family][label] = {
                "contexts": len(subset), "source_instances": source_count,
                "independent_parents": parent_count,
                "sufficiency_rule": "100 independent parent contexts per family; 20 per stratum",
                "status_meaning": STATUS_MEANING,
                "matched_instances_audit_only": sum(r.get("matched_instances", 0) for r in subset),
                "mean_outputs_per_context": point, "outputs_per_100_contexts": 100 * point,
                "interval_95": [lower, upper], "by_context": by_context,
                "rate_free_headline": {
                    "triggers_per_100_nuisance_events": 100 * trigger_point,
                    "interval_per_100": (100 * trigger_bounds).tolist(), "confidence": confidence,
                    "denominator": "whole nuisance episodes (parent contexts), not annotated sources",
                    "dependent_records_note": "linked records retain per-record denominator and parent-cluster uncertainty",
                    "interval_method": "; ".join(sorted(trigger_methods)),
                },
                "dispersion_corrected": True, "interval_method": "stratified-parent-context-count-bootstrap",
                "status": "scored" if sufficient else "inconclusive-insufficient-context-evidence",
                "evidence_label": next(iter(sources)) if len(sources) == 1 else "mixed",
            }
    return {"context_weights": weights, "confidence": confidence, "families": summaries}


def _floor_interval(floor: dict, confidence: float, replicates: int, seed: int) -> list[float]:
    interval = list(poisson_rate_interval(int(floor["events"]), float(floor["exposure_s"]), confidence))
    if "pooled_files" in floor:
        # Resample complete equal-duration blocks within each file. Never join
        # partial blocks across recording boundaries or invent continuity.
        total = float(floor["exposure_s"])
        draws = np.zeros(replicates)
        rng = np.random.default_rng(seed)
        for item in floor["pooled_files"]:
            counts = np.asarray(item["block_counts"][:int(item["exposure_s"] // item["block_s"])])
            if counts.size >= 2:
                sampled = counts[rng.integers(0, len(counts), size=(replicates, len(counts)))].mean(axis=1)
                draws += sampled * 3600 / item["block_s"] * item["exposure_s"] / total
            else:
                draws += item["events"] * 3600 / total
        alpha = 1 - confidence
        return [min(interval[0], float(np.quantile(draws, alpha / 2))),
                max(interval[1], float(np.quantile(draws, 1 - alpha / 2)))]
    counts = np.asarray(floor.get("block_counts", []), dtype=float)
    if counts.size >= 2:
        # Equal-duration complete blocks only: a partial final block is excluded
        # from this diagnostic; the exact bound retains the full exposure.
        full_blocks = int(float(floor["exposure_s"]) // float(floor["block_s"]))
        counts = counts[:full_blocks]
        if counts.size >= 2:
            rng = np.random.default_rng(seed)
            draws = counts[rng.integers(0, len(counts), size=(replicates, len(counts)))].mean(axis=1)
            draws *= 3600 / float(floor["block_s"])
            alpha = 1 - confidence
            interval[0] = min(interval[0], float(np.quantile(draws, alpha / 2)))
            interval[1] = max(interval[1], float(np.quantile(draws, 1 - alpha / 2)))
    return interval


def _name_start_counts(value):
    """Keep start summaries' count-field names consistent with their event unit."""
    if isinstance(value, dict):
        return {("start_outputs" if k == "confirmed_outputs" else k): _name_start_counts(v)
                for k, v in value.items()}
    return value


def estimate_composite(
    floor: dict, trials: Iterable[dict], context_prevalence_per_hour: dict[str, float], *,
    context_weights: dict[str, Any] | None = None, confidence: float = 0.95,
    sensitivity_multipliers: Iterable[float] = (0.3, 1.0, 3.0),
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES, bootstrap_seed: int = 20261012,
    dispersion_corrected: bool | None = None, floor_time_fraction: float = 1.0,
    prediction_exposure_hours: float | None = None,
    event_unit: str = "confirmed_events",
    floor_budget_per_hour: float = DEFAULT_FLOOR_BUDGET_PER_HOUR,
    outputs_per_trigger_assumption: float = 1.0,
) -> dict[str, Any]:
    """Compute floor + sum(contexts/hour * outputs/context) for the chosen event unit.

    Incidences must refer to disjoint whole contexts, never source instances.
    Bonferroni error allocation covers the floor and all family/stratum means.
    Bootstrap coverage is asymptotic, not an exact small-sample guarantee.
    Finite-stream prediction is intentionally unsupported: it additionally needs
    an arrival/count model for the parent contexts and their activity states.
    """
    if prediction_exposure_hours is not None:
        raise ValueError("prediction needs a parent-context arrival model; this is a mean-rate interval")
    if not math.isfinite(outputs_per_trigger_assumption) or outputs_per_trigger_assumption < 1:
        raise ValueError("outputs_per_trigger_assumption must be finite and >= 1")
    if not math.isfinite(floor_time_fraction) or not 0 <= floor_time_fraction <= 1:
        raise ValueError("floor_time_fraction must be between zero and one")
    if not math.isfinite(floor_budget_per_hour) or floor_budget_per_hour <= 0:
        raise ValueError("floor_budget_per_hour must be finite and positive")
    trials = list(trials)
    if event_unit == "start_alerts":
        if "start_alerts" not in floor or any("start_outputs" not in r or "max_start_outputs" not in r for r in trials):
            raise ValueError("start alerts unavailable; rescore floor and contexts with starts included")
        floor = floor["start_alerts"]
        trials = [{**r, "confirmed_outputs": r["start_outputs"],
                   "max_confirmed_outputs": r["max_start_outputs"]} for r in trials]
    elif event_unit != "confirmed_events":
        raise ValueError("event_unit must be confirmed_events or start_alerts")
    rows, weights = _context_data(trials, context_weights)
    summary = summarize_challenges(rows, context_weights=weights, confidence=confidence,
                                  bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed)
    prevalence = {str(k): float(v) for k, v in context_prevalence_per_hour.items()}
    families = summary["families"]
    if set(prevalence) != set(families):
        raise ValueError("context prevalence must match the scored families exactly")
    if any(not math.isfinite(v) or v < 0 for v in prevalence.values()):
        raise ValueError("context prevalence must be finite and nonnegative")
    multipliers = list(map(float, sensitivity_multipliers))
    if any(not math.isfinite(v) or v <= 0 for v in multipliers):
        raise ValueError("sensitivity multipliers must be finite and positive")
    component_confidence = 1 - (1 - confidence) / (1 + sum(len(w) for w in weights.values()))
    floor_bounds = np.asarray(_floor_interval(floor, component_confidence, bootstrap_replicates,
                                             _seed_for(bootstrap_seed, "floor"))) * floor_time_fraction
    floor_rate = float(floor["events"]) * 3600 / float(floor["exposure_s"]) * floor_time_fraction
    zero_hours = (-math.log((1 - component_confidence) / 2) * floor_time_fraction / floor_budget_per_hour)
    floor_resolved = bool(floor_bounds[1] <= floor_budget_per_hour)
    result = {
        "event_unit": event_unit,
        "deployment_alert_unit": "start_alerts",
        "deployment_semantics": "edge listen emits every start immediately; confirmed_events counts only confirmed end lines",
        "floor_resolved": floor_resolved,
        "floor_budget_per_hour": floor_budget_per_hour,
        "floor_resolution_rule": "allocated floor upper bound <= floor budget; does not bound the whole composite",
        "zero_event_floor_hours_for_budget_composite": zero_hours,
        "equation": "floor_time_fraction * r_floor + sum(contexts_per_hour * mean_outputs_per_context)",
        "evidence_label": "model-based", "prevalence_source": "declared adjustable assumptions",
        "context_prevalence_per_hour": prevalence, "prevalence_unit": "whole-contexts/hour",
        "context_weights": weights, "confidence": confidence, "dispersion_corrected": True,
        "status_meaning": STATUS_MEANING,
        "interval_method": "dispersion-corrected parent-context count bootstrap; Bonferroni components; floor block/Poisson envelope",
        "interval_target": "mean-rate (not finite-stream prediction)",
        "floor_rate_per_hour": floor_rate, "floor_time_fraction": floor_time_fraction,
        "floor_evidence_label": floor.get("evidence_label", "model-based"),
        "floor_interval_per_hour": floor_bounds.tolist(), "classes": {},
        "sensitivity_multipliers": multipliers,
        "tail_extrapolation": {"enabled": False, "reason": "requires separate preregistered validation"},
    }
    for label in sorted({r["class"] for r in rows}):
        point_challenge = 0.0
        challenge_bounds = np.zeros(2)
        contributions = {}
        trigger_upper = 0.0
        level_points = np.zeros(2)
        level_bounds = np.zeros(2)
        for family in families:
            family_point = prevalence[family] * families[family][label]["mean_outputs_per_context"]
            point_challenge += family_point
            family_bounds = np.zeros(2)
            stratum_bounds = []
            stratum_means = []
            for context, weight in weights[family].items():
                group = [r for r in rows if r["family"] == family and r["class"] == label
                         and r["context_group"] == context]
                draws, _ = _count_bootstrap(group, bootstrap_replicates,
                                           _seed_for(bootstrap_seed, family, label, context))
                bounds = _count_interval(group, draws, component_confidence)
                family_bounds += prevalence[family] * weight * np.asarray(bounds)
                challenge_bounds += prevalence[family] * weight * np.asarray(bounds)
                stratum_bounds.append(bounds)
                stratum_means.append(families[family][label]["by_context"][context]["mean_outputs_per_context"])
                trigger = _trigger_summary(group, component_confidence, bootstrap_replicates,
                                           _seed_for(bootstrap_seed, family, label, context, "trigger"))
                trigger_upper += prevalence[family] * weight * trigger["interval"][1]
            contributions[family] = (family_point, family_bounds)
            level_points += prevalence[family] * np.asarray([min(stratum_means), max(stratum_means)])
            level_bounds += prevalence[family] * np.asarray([min(b[0] for b in stratum_bounds),
                                                            max(b[1] for b in stratum_bounds)])
        contradicted = any(r["confirmed_outputs"] > outputs_per_trigger_assumption
                           for r in rows if r["class"] == label)
        result["classes"][label] = {
            "rate_per_hour": floor_rate + point_challenge,
            "upper_bound_per_hour": float((floor_bounds + challenge_bounds)[1]),
            "floor_resolved": floor_resolved,
            "headline": {"point_per_hour": floor_rate + point_challenge,
                         "upper_bound_per_hour": float((floor_bounds + challenge_bounds)[1]),
                         "confidence": confidence, "event_unit": event_unit},
            "interval_95_per_hour": (floor_bounds + challenge_bounds).tolist(),
            "challenge_component_per_hour": point_challenge,
            "challenge_interval_per_hour": challenge_bounds.tolist(),
            "families": {f: families[f][label] for f in families},
            "status": "scored" if all(families[f][label]["status"] == "scored" for f in families)
                      else "inconclusive-insufficient-context-evidence",
            "status_meaning": STATUS_MEANING,
            "outputs_per_trigger_assumption": {
                "label": "model-based assumption; not the physical-cap bound",
                "assumed_max_outputs_per_trigger": outputs_per_trigger_assumption,
                "assumption": "every triggering whole context emits at most the declared number of outputs",
                "assumption_contradicted_by_bank": contradicted,
                "upper_bound_per_hour": None if contradicted else float(
                    floor_bounds[1] + outputs_per_trigger_assumption * trigger_upper),
                "confidence": confidence,
            },
            "family_incidence_sensitivity": {
                f: {str(m): {"rate_per_hour": floor_rate + point_challenge + (m - 1) * p,
                             "interval_per_hour": (floor_bounds + challenge_bounds + (m - 1) * b).tolist()}
                    for m in multipliers} for f, (p, b) in contributions.items()},
            "level_mix_sensitivity": {
                "scope": "all mixtures of supplied strata; not unseen levels or families",
                "rate_per_hour_range": (floor_rate + level_points).tolist(),
                "interval_envelope_per_hour": (floor_bounds + level_bounds).tolist(),
                "assumption": "incidence fixed; each supplied stratum may carry all family mass",
            },
            "sensitivity": {str(m): {"rate_per_hour": floor_rate + m * point_challenge,
                                     "interval_95_per_hour": (floor_bounds + m * challenge_bounds).tolist()}
                            for m in multipliers},
        }
    return _name_start_counts(result) if event_unit == "start_alerts" else result


def pool_floor_summaries(floors: Iterable[dict], *, confidence: float = 0.95,
                         floor_budget_per_hour: float = DEFAULT_FLOOR_BUDGET_PER_HOUR) -> dict:
    """Pool independent files' counts/exposure, retaining boundaries and provenance.

    Files must be target-free samples of the same deployment floor process.
    Short files never become a continuous measured recording when concatenated.
    The per-file reports remain available to diagnose nonstationary floors.
    """
    floors = list(floors)
    if not floors:
        raise ValueError("at least one floor file is required")
    offset = 0.0
    times = []
    for floor in floors:
        times.extend(offset + float(t) for t in floor["event_times_s"])
        offset += float(floor["exposure_s"])
    all_measured = all(str(f["evidence_label"]).startswith("measured") for f in floors)
    pooled = summarize_floor(times, offset, source_kind="measured" if all_measured else "model-based",
                             longest_continuous_s=max(f["longest_continuous_s"] for f in floors),
                             confidence=confidence, floor_budget_per_hour=floor_budget_per_hour)
    pooled["pooled_files"] = floors
    pooled["pooling_assumption"] = "independent files from the same target-free deployment floor process"
    # Synthetic concatenation blocks are not evidence. _floor_interval uses the
    # original file blocks and their separate boundaries instead.
    pooled.pop("block_counts")
    pooled.pop("block_s")
    return pooled


def bed_occupancy(model, audio, *, thresholds=None):
    """Smoothed off-threshold occupancy; diagnostic only, never a FA probability.

    Warm-up averages use the available history, exactly as Tracker does. File
    boundaries reset history. Fractions count scored hops, not padded windows.
    """
    cfg = {**model.meta["detector"], **(thresholds or {})}
    fs = model.meta["features"]["fs"]
    hop_s = round(cfg["hop_s"]*fs)/fs
    scores = edge.window_scores(model, audio, hop_s)
    if (scores.shape != (len(scores), len(model.classes)) or not len(scores)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("invalid or empty score windows for bed occupancy")
    tracker = edge.Tracker(model.classes, cfg, model.features.n/fs)
    count = np.zeros(len(model.classes), int)
    for k in range(len(scores)):
        for i, length in enumerate(tracker.smooth_hops):
            count[i] += scores[max(0, k+1-length):k+1, i].mean() >= tracker.off[i]
    return {"scored_hops": len(scores), "hop_s": hop_s,
            "warmup": "available history; reset at each file boundary",
            "by_class": {c: {"fraction_at_or_above_off": int(count[i])/len(scores),
                             "off_threshold": float(tracker.off[i])}
                         for i, c in enumerate(model.classes)}}


def overlay_context(model, background, events, *, path, seed, capture):
    """Whole WP3 episodes on an unchanged deployed-device bed (always proxy).

    The composer invokes the common room/device mixer once for the entire scene.
    Recorded ambience retains its noise, quantization and state progression.
    Explicit source lengths plus natural trailing audio must fit; no truncation,
    artificial silence, bed recapture or independently summed model scores.
    Current composer limit is 60 seconds; longer episodes are refused, not split.
    """
    from sensorsim.negatives import composer
    fs = int(model.meta["features"]["fs"])
    bed = np.asarray(background, float)
    if (bed.ndim != 1 or not len(bed) or not np.isfinite(bed).all()
            or np.max(abs(bed)) > 1 or np.ptp(bed) == 0):
        raise ValueError("bed must be finite nonconstant mono PCM; silence is not a deployment bed")
    if (capture.get("deployed_device") is not True or capture.get("target_free") is not True
            or capture.get("profile_id") != path.device_id or not capture.get("capture_device")):
        raise ValueError("target_free deployed-device capture with matching profile required")
    if not capture.get("parent_id") or not capture.get("recording_sha256"):
        raise ValueError("capture requires recording digest and independent parent_id")
    segment = capture.get("segment_samples", [])
    if len(segment) != 2 or segment[0] < 0 or segment[1]-segment[0] != len(bed):
        raise ValueError("bed segment_samples must identify the complete segment")
    events = tuple(events)
    if not events:
        raise ValueError("overlay needs at least one whole nuisance episode")
    cfg = model.meta["detector"]
    smooth = cfg["smooth_hops"]
    longest = max(smooth.values()) if isinstance(smooth, dict) else smooth
    guard_s = model.features.n/fs + longest*round(cfg["hop_s"]*fs)/fs
    for event in events:
        if (not np.isfinite(event.source.audio).all() or not len(event.source.audio)
                or np.ptp(event.source.audio) == 0 or event.label != "background"):
            raise ValueError("finite nonconstant confuser source required")
        if event.onset_s < guard_s or event.onset_s+len(event.source.audio)/fs+guard_s > len(bed)/fs:
            raise ValueError("whole episode plus deployment guards must fit; never split or crop")
    # _context is the composer engine behind captured_context; its public 10 s
    # wrapper cannot represent long whole episodes. Keep this dependency explicit.
    context = composer._context(events, path=path, fs=fs, seed=seed, background=bed,
        domain="device_recorded", cfg={"background_native_s16": np.array_equal(bed*32768, np.rint(bed*32768))},
        duration_s=len(bed)/fs)
    if any(r["tail_truncated_samples"] != 0 or r["rendered_end_s"]+guard_s > len(bed)/fs
           for r in context.events):
        raise ValueError("rendered tail plus deployment guard truncated; extend natural bed segment")
    context.provenance.update(whole_episode=True, capture=dict(capture),
        parent_id=capture["parent_id"], bed_recording_sha256=capture["recording_sha256"],
        bed_segment_samples=list(segment), rendered_events=list(context.events), evidence_label="proxy",
        policy="empirical-capture-overlay/1", deployment_fa_per_hour="not measured",
        bed_occupancy=bed_occupancy(model, bed))
    return context


def _prepare_overlays(model, manifest, root, *, audit=None):
    """Audit every input before rendering; retain the exact decoded inputs scored."""
    from sensorsim.negatives import composer
    from sensorsim.negatives.procedural import Source, generate
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import SynthConfig
    from sensorsim import iot_mic
    if audit is None:
        audit, entries = audit_manifest(model, manifest, root)
    else:
        entries = manifest['challenges']
    fs = int(model.meta["features"]["fs"])
    def read(record, role):
        audio = audit.cache[(root / record['path']).resolve()]
        return audio, composer.audio_hash(audio)
    loaded = []
    for record in entries:
        descriptor = record['overlay']
        audio, digest = read(descriptor['background'], 'bed')
        loaded.append((record, descriptor, audio, digest))
    settings = manifest["overlay_path"]
    room = RoomModel.model_validate_json((root / settings["room"]).read_text())
    device = settings["device_id"]
    path = composer.RenderPath(room, tuple(settings["receiver_xyz_m"]), iot_mic.PROFILES[device], device,
                              SynthConfig(**settings.get("config", {})))
    resolved, content, identities = [], set(), []
    for index, (record, desc, audio, digest) in enumerate(loaded):
        if desc.get("whole_episode") is not True:
            raise ValueError("whole_episode: true must attest that no source episode was cropped")
        duration = float(desc["duration_s"])
        size = round(duration*fs)
        if not np.isfinite(duration) or size < 1 or size > len(audio):
            raise ValueError("bed recording too short for the whole episode")
        seed = int(desc["seed"])
        start_s = desc.get("start_s")
        start = (int(np.random.default_rng(seed).integers(len(audio)-size+1)) if start_s is None
                 else round(float(start_s)*fs))
        if start < 0 or start+size > len(audio):
            raise ValueError("bed segment outside recording")
        bed = audio[start:start+size]
        origin = desc['background']['sample_range'][0]
        capture = dict(desc["background"], recording_sha256=digest,
                       parent_id=desc['background'].get('parent_id', desc['background']['recording_id']),
                       segment_samples=[origin+start, origin+start+size])
        event_list = []
        links = {"bed-parent:" + capture["parent_id"], "bed-content:" + digest}
        for source in desc["events"]:
            if source.get("whole_episode") is not True:
                raise ValueError("each source must declare whole_episode: true")
            if "procedural" in source:
                src = generate(**source["procedural"], fs=fs)
            else:
                # No external corpus licence is inferred from a filename.
                if source.get("licence") != "user-owned" or source.get("acoustic_stage") != "dry":
                    raise ValueError("source WAV requires user-owned licence and dry acoustic_stage")
                x, xhash = read(source, "context")
                src = Source(x, source["family"], source.get("parent_id", source["recording_id"]),
                             {"fs": fs, "acoustic_stage": "dry", "content_sha256": xhash,
                              "licence": "user-owned"})
            links.update(("source-id:" + src.source_id, "source-content:" + composer.audio_hash(src.audio)))
            event_list.append(composer.Event(src, float(source["onset_s"]), tuple(source["source_xyz_m"]),
                spl_db=float(source["spl_db"]), reviewed_label_boundary=source.get("reviewed_label_boundary") is True))
        parent_lock = None
        if "negative_parent_lock" in manifest:
            from sensorsim.training_negatives import validate_overlay_sources
            from sensorsim.negatives.splits import Item
            from sensorsim.negatives.integration import verified_file
            settings_lock = manifest["negative_parent_lock"]
            parents = [Item(**source["locked_parent"]) for source in desc["events"]]
            reviews = [json.loads(verified_file(root/source["semantic_review"]["file"],
                source["semantic_review"]["sha256"])) for source in desc["events"]]
            parent_lock = validate_overlay_sources(event_list,
                lock=root/settings_lock["file"], lock_sha256=settings_lock["sha256"],
                purpose="audit", parent_items=parents, scope=settings_lock["target_scope"], reviews=reviews,
                source_records=desc["events"], root=root)
            # The audit lock owns acquisition identity even when two descendants
            # have different recipes, source IDs and decoded content.
            links.update("source-group:" + parent["source_group"] for parent in parent_lock["parents"])
        context = overlay_context(model, bed, event_list, path=path, seed=seed, capture=capture)
        if parent_lock is not None:
            context.provenance["negative_parent_lock"] = parent_lock
        output_hash = composer.audio_hash(context.audio)
        if output_hash in content:
            raise ValueError("duplicate mixed context content cannot add independent evidence")
        content.add(output_hash)
        identities.append(links)
        # Repeated random draws from one recording remain one parent cluster.
        resolved.append({**record, "_audio": context.audio, "source_kind": "proxy",
            "parent_id": capture["parent_id"], "context_id": record.get("context_id", f"overlay-{index}"),
            "events": [{"start_s": e["onset_s"], "end_s": e["source_end_s"]} for e in context.events],
            "overlay_provenance": context.provenance})
    # Connected components preserve dependence from either reused beds OR dry
    # episodes. Rerendering one source never creates independent take evidence.
    components = []
    for i, links in enumerate(identities):
        joined = {i}
        for members, nodes in components[:]:
            if links & nodes:
                joined |= members
                links |= nodes
                components.remove((members, nodes))
        components.append((joined, links))
    for members, nodes in components:
        parent = "overlay-parent-" + composer.digest(sorted(nodes))
        for i in members:
            resolved[i]["parent_id"] = parent
            resolved[i]["overlay_provenance"]["independent_parent_id"] = parent
            resolved[i]["overlay_provenance"]["dependency_ids"] = sorted(nodes)
    return resolved, audit.report()


def _score_manifest(model_path: str, manifest_path: str, *, bootstrap_replicates: int | None = None) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = manifest_file.parent

    def absolute_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    digest = hashlib.sha256()
    with Path(model_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    model_sha256 = digest.hexdigest()
    model = edge.Model(str(model_path))
    thresholds = manifest.get("thresholds")
    detector_settings = {**model.meta["detector"], **(thresholds or {})}

    def check_provenance(record):
        for field, expected in (("model_sha256", model_sha256), ("detector_settings", detector_settings)):
            if field in record and record[field] != expected:
                raise ValueError(f"{field} differs from this scoring run; use the deployment model/config")

    check_provenance(manifest)
    source_default = str(manifest.get("source_kind", "model-based"))
    confidence = float(manifest.get("confidence", 0.95))
    budget = float(manifest.get("floor_budget_per_hour", DEFAULT_FLOOR_BUDGET_PER_HOUR))
    # This check also runs for plain manifests and uses model-saved calibration
    # records even when the caller supplies calibration_streams: [].
    audit, challenge_entries = audit_manifest(model, manifest, root)
    overlay_records, partition_audit = None, audit.report()
    if "overlay_path" in manifest:
        if thresholds:
            raise ValueError("overlay banks must use deployment smoothing/settings without overrides")
        overlay_records, partition_audit = _prepare_overlays(model, manifest, root, audit=audit)
    floors = []
    seen_floor_paths = set()
    for entry in manifest.get("floor_streams", []):
        check_provenance(entry)
        path = absolute_path(entry["path"])
        if path.resolve() in seen_floor_paths:
            raise ValueError("duplicate floor path; exposure must not be counted twice")
        seen_floor_paths.add(path.resolve())
        audio = audit.cache[path.resolve()]
        events = score_audio(model, audio, thresholds=thresholds, include_starts=True)
        exposure_s = len(audio)/model.meta['features']['fs']
        for label in model.classes:
            common = dict(source_kind=entry.get("source_kind", source_default),
                          longest_continuous_s=float(entry.get("longest_continuous_s", exposure_s)),
                          confidence=confidence, block_s=float(entry.get("block_s", 10.0)),
                          floor_budget_per_hour=budget)
            item = summarize_floor((e["start_s"] for e in _confirmed_only(events) if e["label"] == label),
                                   exposure_s, **common)
            item["event_unit"] = "confirmed_events"
            item["start_alerts"] = summarize_floor(
                (e["time_s"] for e in events if e.get("event") == "start" and e["label"] == label),
                exposure_s, **common)
            item["start_alerts"]["event_unit"] = "start_alerts"
            item.update({"name": str(entry.get("name", Path(entry["path"]).stem)),
                         "kind": str(entry.get("kind", "floor")), "class": label})
            floors.append(item)
    resolved = (overlay_records if overlay_records is not None else
        [{**r, "path": str(absolute_path(r["path"])),
          "_audio": audit.cache[absolute_path(r['path']).resolve()],
          "parent_id": r.get('parent_id', r['recording_id']),
          "source_kind": r.get("source_kind", source_default)} for r in challenge_entries])
    for record in resolved:
        check_provenance(record)
    if not floors:
        raise ValueError("manifest must contain at least one floor_streams record")
    if not resolved:
        raise ValueError("manifest must contain at least one challenges record")
    if "context_prevalence_per_hour" not in manifest:
        raise ValueError("manifest needs context_prevalence_per_hour in whole contexts/hour; "
                         "legacy per-instance prevalence cannot be converted without burst structure")
    if "prevalence_per_hour" in manifest:
        raise ValueError("remove legacy prevalence_per_hour; declare context_prevalence_per_hour")
    trials = score_challenge_records(model, resolved, thresholds=thresholds)
    for trial in trials:
        trial.update(model_sha256=model_sha256, detector_settings=detector_settings)
    prevalence = manifest["context_prevalence_per_hour"]
    weights = manifest.get("context_weights", DEFAULT_CONTEXT_WEIGHTS)
    reps = int(bootstrap_replicates or manifest.get("bootstrap_replicates", DEFAULT_BOOTSTRAP_REPLICATES))
    seed = int(manifest.get("bootstrap_seed", 20261012))
    prevalence_source = manifest.get("prevalence_source", "declared adjustable assumptions")
    all_challenge_sources = {_source_label(r["source_kind"]) for r in resolved}
    pooled_floors, composite_results = [], []
    for label in model.classes:
        files = [f for f in floors if f["class"] == label]
        floor = pool_floor_summaries(files, confidence=confidence, floor_budget_per_hour=budget)
        floor.update(name="pooled", **{"class": label, "event_unit": "confirmed_events"})
        floor["start_alerts"] = pool_floor_summaries([f["start_alerts"] for f in files],
                                                    confidence=confidence, floor_budget_per_hour=budget)
        floor["start_alerts"]["event_unit"] = "start_alerts"
        pooled_floors.append(floor)
        for unit in ("confirmed_events", "start_alerts"):
            composite = estimate_composite(
                floor, [r for r in trials if r["class"] == label], prevalence,
                context_weights=weights, confidence=confidence,
                sensitivity_multipliers=manifest.get("sensitivity_multipliers", (0.3, 1.0, 3.0)),
                bootstrap_replicates=reps, bootstrap_seed=seed,
                prediction_exposure_hours=manifest.get("prediction_exposure_hours"),
                floor_time_fraction=float(manifest.get("floor_time_fraction", 1.0)),
                floor_budget_per_hour=budget, event_unit=unit,
                outputs_per_trigger_assumption=float(manifest.get("outputs_per_trigger_assumption", 1)),
            )
            composite.update(floor_stream="pooled", **{"class": label})
            all_measured = (floor["evidence_label"] == "measured"
                and all_challenge_sources == {"measured"}
                and str(prevalence_source).lower() == "measured"
                and str(manifest.get("context_weights_source", "assumed")).lower() == "measured"
                and str(manifest.get("floor_time_fraction_source", "assumed")).lower() == "measured")
            composite["evidence_label"] = "measured" if all_measured else "model-based"
            composite["prevalence_source"] = prevalence_source
            composite_results.append(composite)
    summary_options = dict(context_weights=weights, confidence=confidence,
                           bootstrap_replicates=reps, bootstrap_seed=seed)
    challenge_summary = summarize_challenges(trials, **summary_options)
    start_trials = [{**r, "confirmed_outputs": r["start_outputs"],
                     "max_confirmed_outputs": r["max_start_outputs"]} for r in trials]
    start_summary = summarize_challenges(start_trials, **summary_options)
    return {
        "schema": "sensorsim-room-fa-rate/3",
        "recording_partition_audit": partition_audit,
        "overlays": [r["overlay_provenance"] for r in resolved if "overlay_provenance" in r],
        "detector": "sensorsim.edge.Detector and sensorsim.edge.Tracker",
        "model_sha256": model_sha256,
        "model_detector_settings": model.meta["detector"],
        "detector_settings": detector_settings,
        "detector_overrides": thresholds or {},
        "feature_settings": model.meta["features"],
        "bank_protocol": BANK_PROTOCOL,
        "status_meaning": STATUS_MEANING,
        "deployment_alert_unit": "start_alerts",
        "deployment_semantics": "edge listen emits every start immediately; confirmed_events counts only confirmed end lines",
        "source_kind_default": _source_label(source_default),
        "floor_streams": floors, "pooled_floor_streams": pooled_floors,
        "challenge_results": challenge_summary,
        "start_alert_challenge_results": _name_start_counts(start_summary),
        "composite_estimates": composite_results,
        "context_prevalence_per_hour": prevalence, "prevalence_unit": "whole-contexts/hour",
        "tail_extrapolation": {"enabled": False, "reason": "POT tail not enabled without a separately preregistered validation pass"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate start alerts/hour (edge listen deployment) and confirmed events/hour, with upper bounds and pooled floors.",
        epilog=("Use the deployment model: its detector settings and SHA256 are recorded. "
                "Build each context from the whole nuisance episode plus guards, overlaid on user target-free "
                "recording segments. Render superposition as one mixed waveform, never combine scores; "
                "give mixed scenes their own family and incidence. overlay_path invokes the negatives composer and shared mixer; declare calibration_streams (explicit [] if unused). Each WAV record requires recording_id, sample_rate_hz and sample_range [start, end) in the original recording at model rate. Split calibration, floor, contexts and beds into nonoverlapping ranges before tuning; exported excerpt length must match its range. The model retains calibration digests and exact block fingerprints, which do not detect gain changes or resampling. Legacy models warn when ancestry is unavailable. "
                "Declare rough context_prevalence_per_hour and per-family context_weights (including level strata). "
                "Default sensitivities vary incidence by x0.3/x3 globally and per family and span all supplied level "
                "mixtures. outputs_per_trigger_assumption defaults to 1 and labels a separate assumed upper bound; "
                "the physical-cap bound remains available. Rate-free headlines count triggering whole episodes "
                "per 100 nuisance events. Scored means enough independent contexts, not a representative bank "
                "or a passed FA budget. Floor files containing those nuisances already measure their contribution: "
                "omit duplicate bank families or declare floor_time_fraction to avoid double counting.")
    )
    parser.add_argument("--model", required=True, help="edge model NPZ")
    parser.add_argument("--manifest", required=True, help="JSON input manifest; relative audio paths resolve beside it")
    parser.add_argument("--out", help="write JSON result to this path instead of stdout")
    parser.add_argument("--bootstrap-replicates", type=int, help=f"cluster bootstrap draws (default {DEFAULT_BOOTSTRAP_REPLICATES})")
    args = parser.parse_args(argv)
    try:
        result = _score_manifest(args.model, args.manifest, bootstrap_replicates=args.bootstrap_replicates)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
