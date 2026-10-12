"""Prospective, group-level difference tests and complete-family power audits.

No equivalence margin is supplied or estimated. Qualification is conditional on
the declared generator, complete family and retained group counts; it is never a
deployment or independent-verifier receipt. See tools/power_audit.py for recording
every attempted run. All physical levels use digital peak full scale = 1, NOT
sine-referenced dBFS(A), SPL or sound power.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing
import os
from hashlib import sha256
from itertools import product, groupby
import importlib
import json
from numbers import Real
from math import comb
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.stats import beta

from .c2st import refit_group_permutation as _baseline_refit_group_permutation
from .realism import band_statistics, mmd_permutation_test, pooled_embedding


# Versioned before qualification. Bands are fixed Hz, never fitted to audit data.
PHYSICAL_BANK = (
    ("centered_rms_dbfs", "dB re RMS 1"),
    ("peak_dbfs", "dB re peak 1"),
    ("dc", "digital amplitude"),
    ("clipped_fraction", "fraction abs(sample)>=1"),
    ("energy_centroid_s", "s relative to window start"),
    ("late_early_db", "dB energy last/first half"),
    ("contrast_3khz_db", "dB power 2500:3500 / 1000:2000 Hz"),
)
BANK_VERSION = "physical-features/1"
DOMAINS = ("synthetic proxy", "measured-RIR proxy", "deployment recording")
TESTS = (("physical", "max_mean"), ("physical", "mmd"),
         ("deployed", "mmd"), ("deployed", "c2st"))
DEFECTS = ("eq_3khz_plus2db", "scalar_plus6db", "frame_reversal")


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def audio_digest(audio) -> str:
    """Decoded float64 content, including shape; no filename-based identity."""
    x = np.ascontiguousarray(audio, dtype="<f8")
    return sha256(str(x.shape).encode() + x.tobytes()).hexdigest()


def implementation_fingerprint(*, execution="full") -> str:
    """Bind reports to the actual imported modules; no receipt survives a change."""
    names = (__name__, "sensorsim.evaluation.realism", "sensorsim.evaluation.c2st", "sensorsim.edge")
    if execution == "decision":
        names += ("sensorsim.evaluation.permutation_decision",)
    elif execution != "full":
        raise ValueError("unknown family execution mode")
    return digest({name: sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() for name in names})


def _audio(audio):
    x = np.asarray(audio, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or not len(x) or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("audio must be nonempty finite mono windows; no filtering or padding")
    return x


def build_physical_features(audio, fs: int, *, starts_s=None) -> dict:
    """Raw diagnostics beside deployed features, with explicit degeneracy flags.

    Silence/DC is a valid boundary input but cannot qualify distributional realism.
    Log floors are fixed at -240 dB and flagged, never treated as measured floors.
    """
    x = _audio(audio)
    if type(fs) is not int or fs <= 7000:
        raise ValueError("fs must be integer Hz >7000 for the frozen 3 kHz band")
    starts = np.zeros(len(x)) if starts_s is None else np.asarray(starts_s, float)
    if starts.shape != (len(x),) or not np.isfinite(starts).all() or np.any(starts < 0):
        raise ValueError("one finite nonnegative start time in seconds per window required")
    centered = x - x.mean(axis=1, keepdims=True)
    energy = centered**2
    total = energy.sum(axis=1)
    half = x.shape[1] // 2
    floor = 1e-24
    spectrum = abs(np.fft.rfft(centered * np.hanning(x.shape[1]), axis=1))**2
    freq = np.fft.rfftfreq(x.shape[1], 1/fs)
    bands = [(freq >= a) & (freq < b) for a, b in ((2500, 3500), (1000, 2000))]
    if not all(np.any(band) for band in bands):
        raise ValueError("window too short to resolve frozen frequency bands")
    values = np.column_stack((
        10*np.log10(np.maximum(energy.mean(axis=1), floor)),
        20*np.log10(np.maximum(abs(x).max(axis=1), 1e-12)),
        x.mean(axis=1), (abs(x) >= 1).mean(axis=1),
        (energy @ (np.arange(x.shape[1])/fs))/np.maximum(total, floor),
        10*np.log10(np.maximum(energy[:, half:].sum(axis=1), floor)
                    / np.maximum(energy[:, :half].sum(axis=1), floor)),
        10*np.log10(np.maximum(spectrum[:, bands[0]].mean(axis=1), floor)
                    / np.maximum(spectrum[:, bands[1]].mean(axis=1), floor)),
    ))
    return dict(version=BANK_VERSION, columns=[c for c, _ in PHYSICAL_BANK],
                units=[u for _, u in PHYSICAL_BANK], values=values, starts_s=starts.tolist(),
                constant_audio=(np.ptp(x, axis=1) == 0).tolist(),
                level_floor_used=(energy.mean(axis=1) <= floor).tolist(),
                audio_sha256=[audio_digest(row) for row in x])


def feature_bundle(audio, features, records: list[dict], *, starts_s=None, model=None,
                   qualification_scope: dict | None = None) -> dict:
    """Extract unchanged edge.Features; missing embedding hooks raise, never fall back.

    Records use source_group/background_groups/parent_audio_sha256 from the
    operational manifest, plus explicit session, route and framing provenance.
    """
    x = _audio(audio)
    if len(records) != len(x):
        raise ValueError("one provenance record per audio window required")
    physical = build_physical_features(x, features.cfg["fs"], starts_s=starts_s)
    mel = features(x)
    views = {"physical": physical["values"], "deployed": band_statistics(mel)}
    if model is not None:
        views["embedding"] = pooled_embedding(model, mel)
    rows = []
    for row, content in zip(records, physical["audio_sha256"]):
        row = dict(row)
        if "audio_sha256" in row and row["audio_sha256"] != content:
            raise ValueError("declared decoded content hash differs from actual audio")
        row["audio_sha256"] = content
        rows.append(row)
    return dict(views=views, records=rows, physical=physical,
                feature_config=dict(features.cfg), bank_version=BANK_VERSION,
                qualification_scope=qualification_scope,
                units={"physical": physical["units"],
                       "deployed": ["normalized log-mel statistic"]*views["deployed"].shape[1],
                       **({"embedding": ["frozen CNN activation"]*views["embedding"].shape[1]}
                          if "embedding" in views else {})})


def validate_pairing(reference: list[dict], candidate: list[dict], *, mode="independent",
                     null_justification="") -> dict:
    """Collapse aliases transitively before counting; abstain on invalid dependence.

    A declaration cannot prove exchangeability. The verifier must independently
    check the acquisition design and its justification. Pairing on class/peak
    alone is forbidden. No implicit switch from paired to independent analysis.
    """
    rows = reference + candidate
    errors = []
    if mode not in ("independent", "paired") or not null_justification.strip():
        errors.append("declared mode and nonempty null justification required")
    if not reference or not candidate:
        errors.append("empty or filtered comparison")
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tokens = {}
    required = ("source_group", "independence_group", "session_id", "recording_id",
                "label", "route", "condition_id", "role", "audio_sha256")
    for i, row in enumerate(rows):
        for key in required:
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f"row {i}: missing {key}")
        content = row.get("audio_sha256", "")
        if not isinstance(content, str) or len(content) != 64 or any(c not in "0123456789abcdef" for c in content):
            errors.append(f"row {i}: invalid decoded content SHA256")
        lists = {}
        for key in ("background_groups", "parent_audio_sha256"):
            if not isinstance(row.get(key), list) or any(not isinstance(v, str) or not v for v in row[key]):
                errors.append(f"row {i}: explicit {key} list required (empty means none)")
                lists[key] = []
            else:
                lists[key] = row[key]
        identities = [(key, row.get(key)) for key in
                      ("source_group", "independence_group", "session_id", "recording_id")]
        identities += [("content", row.get("audio_sha256"))]
        identities += [("source_group", v) for v in lists["background_groups"]]
        identities += [("content", v) for v in lists["parent_audio_sha256"]]
        for token in identities:
            if not isinstance(token[1], str) or not token[1]:
                continue
            if token in tokens:
                parent[root(i)] = root(tokens[token])
            tokens[token] = i
    groups = [f"g{root(i)}" for i in range(len(rows))]
    for group in set(groups):
        members = [i for i, g in enumerate(groups) if g == group]
        if len({rows[i].get("role") for i in members}) > 1:
            errors.append("alias crosses evaluation roles")
    left, right = groups[:len(reference)], groups[len(reference):]
    shared = set(left) & set(right)
    if mode == "independent" and shared:
        errors.append("cross-domain ancestry: independent-label null invalid")
    if {r.get("audio_sha256") for r in reference} & {r.get("audio_sha256") for r in candidate}:
        errors.append("identical cross-domain content is route identity, not distributional evidence")
    if mode == "paired":
        pairs = {}
        for i, row in enumerate(rows):
            invalid = False
            for key in ("pair_id", "acquisition_id", "framing_id"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    errors.append(f"row {i}: nonblank string {key} required")
                    invalid = True
            for key in ("source_position", "microphone_position"):
                value = row.get(key)
                if (not isinstance(value, (list, tuple)) or len(value) != 3
                        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                               or not np.isfinite(v) for v in value)):
                    errors.append(f"row {i}: {key} must be three finite numeric coordinates")
                    invalid = True
            # This API's fixed coordinate convention is room Cartesian metres.
            # An explicit incompatible declaration is never converted silently.
            if row.get("coordinate_convention", "room_cartesian_metres") != "room_cartesian_metres":
                errors.append(f"row {i}: incompatible coordinate_convention")
                invalid = True
            if invalid:
                continue
            pairs.setdefault(row["pair_id"], [[], []])[int(i >= len(reference))].append(i)
        for pair_id, sides in pairs.items():
            if any(len(side) != 1 for side in sides):
                errors.append(f"pair {pair_id}: exactly one preaggregated window per side required")
                continue
            a, b = (rows[side[0]] for side in sides)
            keys = ("source_group", "session_id", "condition_id", "label", "route",
                    "acquisition_id", "source_position", "microphone_position", "framing_id")
            if any(a.get(k) is None or a.get(k) != b.get(k) for k in keys):
                errors.append(f"pair {pair_id}: source/acquisition/geometry/framing mismatch")
            if a.get("framing_basis") not in ("external_schedule", "annotation") or a.get("framing_basis") != b.get("framing_basis"):
                errors.append(f"pair {pair_id}: class/peak matching does not justify pairing")
        if set(left) != set(right):
            errors.append("paired sides must retain identical independent clusters")
    return dict(valid=not errors, mode=mode, errors=sorted(set(errors)), groups_left=left,
                groups_right=right, retained_groups=[len(set(left)), len(set(right))],
                null_justification=null_justification, paired_assumption_verified=False)


@dataclass(frozen=True)
class Family:
    """Every class/route/view/test is required; absence invalidates the family."""
    strata: tuple[tuple[str, str], ...] = (("knock", "generated"), ("doorbell", "generated"), ("kettle", "generated"))
    tests: tuple[tuple[str, str], ...] = TESTS
    mode: str = "independent"
    null_justification: str = "independent sources and sessions, identical procedural pipeline under null"
    permutations: int = 1999
    alpha: float = 0.05
    folds: int = 2
    seed: int = 20261009

    def __post_init__(self):
        if (not self.strata or len(set(self.strata)) != len(self.strata) or
                not self.tests or len(set(self.tests)) != len(self.tests)):
            raise ValueError("nonempty unique strata and tests required")
        if not 0 < self.alpha < 1 or type(self.permutations) is not int or self.permutations < 1:
            raise ValueError("invalid alpha/permutation budget")
        if self.mode not in ("independent", "paired") or self.folds < 2:
            raise ValueError("invalid null mode/folds")
        if not any(test == "mmd" for _, test in self.tests):
            raise ValueError("complete family must retain an omnibus MMD comparison")
        if any(test not in ("max_mean", "mmd", "c2st") for _, test in self.tests):
            raise ValueError("unknown test; no fallback")

    @property
    def sha256(self):
        return digest({"family": asdict(self), "bank": PHYSICAL_BANK, "version": BANK_VERSION})


def _holm(pvalues):
    names = sorted(pvalues, key=lambda k: (pvalues[k], k))
    adjusted, running = {}, 0.
    for i, name in enumerate(names):
        running = max(running, min(1., (len(names)-i)*pvalues[name]))
        adjusted[name] = running
    return adjusted


def _max_mean(x, y, gx, gy, *, paired, permutations, seed):
    # Equal-cluster mean diagnostic. MMD separately retains within-cluster variation.
    names_x, names_y = sorted(set(gx)), sorted(set(gy))
    a = np.array([x[np.asarray(gx) == g].mean(axis=0) for g in names_x])
    b = np.array([y[np.asarray(gy) == g].mean(axis=0) for g in names_y])
    scale = np.std(np.r_[a, b], axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    observed = float(np.max(abs((a.mean(axis=0)-b.mean(axis=0))/scale)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        if paired:
            diff = ((a-b)*rng.choice([-1, 1], size=(len(a), 1))).mean(axis=0)
        else:
            z = np.r_[a, b][rng.permutation(len(a)+len(b))]
            diff = z[:len(a)].mean(axis=0)-z[len(a):].mean(axis=0)
        exceed += np.max(abs(diff/scale)) >= observed - 1e-12
    return dict(p_value=float((1+exceed)/(1+permutations)), statistic=observed,
                raw_mean_difference=(b.mean(axis=0)-a.mean(axis=0)).tolist(),
                constant_columns=np.flatnonzero(np.std(np.r_[a, b], axis=0) <= 1e-12).tolist(),
                implementation="equal-cluster max standardized mean", unit="independent cluster")


def _paired_baseline(x, y, gx, gy, test, spec, seed):
    # Existing refit_group_permutation only permits one domain per group; do NOT
    # feed it paired labels. Within-cluster swaps use the existing statistics.
    from .c2st import _cv_scores, auc_score
    from .realism import _kernel_matrix, _mmd_from_kernel
    groups = np.asarray(gx + gy)
    names = np.unique(groups)
    labels = np.r_[np.zeros(len(x), int), np.ones(len(y), int)]
    z = np.r_[x, y]
    kernel = _kernel_matrix(x, y, seed)[0] if test == "mmd" else None

    def statistic(assignment):
        if test == "mmd":
            return _mmd_from_kernel(kernel, assignment == 0, assignment == 1)
        scores = _cv_scores(z, assignment, groups, folds=spec.folds, l2=1., fold_seed=spec.seed)
        return auc_score(assignment, scores)

    observed = statistic(labels)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(spec.permutations):
        flip = names[rng.integers(0, 2, len(names)).astype(bool)]
        assignment = np.where(np.isin(groups, flip), 1-labels, labels)
        exceed += statistic(assignment) >= observed - 1e-12
    return dict(p_value=float((1+exceed)/(1+spec.permutations)), statistic=float(observed),
                implementation=f"existing {test} statistic, whole-cluster paired swaps",
                refit_each_permutation=test == "c2st", weighting="windows")


def _cached_refit_group_permutation(x, labels, groups, *, folds, fold_seed,
                                    replicates, seed, _decision=None):
    """Execute the original C2ST algorithm with bounded, call-local memoization.

    Every RNG draw still contributes, including repeated assignments. A cached
    score is the full refit for that *identical* assignment, never the observed
    fit reused under different labels. Fold predictions may also be reused when
    both the held-out row indices and every training label are identical. Data,
    groups, fold seed and regularization are fixed for the lifetime of the cache.

    Pure-group fold construction below is the index form of _group_folds: sort
    groups, shuffle each class with the fixed fold seed, then distribute round
    robin. All floating point fitting, scoring and AUC operations are unchanged.
    Paired/mixed groups continue through their existing separate implementation.
    """
    from .c2st import _fit_logistic, _sigmoid, auc_score

    labels, groups = np.asarray(labels), np.asarray(groups)
    names, inverse = np.unique(groups, return_inverse=True)
    group_labels = []
    for name in names:
        values = np.unique(labels[groups == name])
        if len(values) != 1:
            raise ValueError("group permutation requires one domain label per group")
        group_labels.append(values[0])
    group_labels = np.asarray(group_labels)
    if folds < 2 or len(names) < folds:
        raise ValueError("grouped CV needs at least two folds and at least one group per fold")
    fold_rng = np.random.default_rng(fold_seed)
    orders = []
    for cls in (0, 1):
        count = np.count_nonzero(group_labels == cls)
        if count < folds:
            raise ValueError("each class needs at least one independent group per fold")
        orders.append(fold_rng.permutation(count))
    assignments, predictions = {}, {}
    cache_limit = 4096

    def statistic(assignment):
        key = assignment.tobytes()
        if key in assignments:
            return assignments[key]
        row_labels = assignment[inverse]
        fold_for_group = np.empty(len(names), dtype=int)
        for cls in (0, 1):
            ordered = np.flatnonzero(assignment == cls)[orders[cls]]
            fold_for_group[ordered] = np.arange(len(ordered)) % folds
        row_folds = fold_for_group[inverse]
        scores = np.full(len(labels), np.nan, float)
        for fold in range(folds):
            test = np.flatnonzero(row_folds == fold)
            train_mask = row_folds != fold
            fit_key = (test.tobytes(), row_labels[train_mask].tobytes())
            if fit_key in predictions:
                scores[test] = predictions[fit_key]
                continue
            weights, mean, scale = _fit_logistic(x[train_mask], row_labels[train_mask], l2=1.)
            z = (x[test] - mean) / scale
            predicted = _sigmoid(np.column_stack((np.ones(len(test)), z)) @ weights)
            scores[test] = predicted
            if len(predictions) < cache_limit:
                predictions[fit_key] = predicted
        value = auc_score(row_labels, scores)
        if len(assignments) < cache_limit:
            assignments[key] = value
        return value

    observed = statistic(group_labels)
    if _decision is not None:
        if _decision.permutations != replicates or not np.isfinite(observed):
            raise ValueError("decision budget mismatch or nonfinite observed C2ST")
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(replicates):
        assignment = rng.permutation(group_labels)
        if _decision is None:
            exceedances += int(statistic(assignment) >= observed)
        elif not _decision.resolved:
            value = statistic(assignment)
            if not np.isfinite(value):
                raise ValueError("nonfinite permutation C2ST")
            _decision.observe(bool(value >= observed))
    if _decision is not None:
        return dict(_decision.report(), observed_auc=float(observed),
                    replicates=int(replicates), seed=int(seed), alternative="greater",
                    refit_each_evaluated_permutation=True, unit="whole independent group",
                    fold_policy="rebuild class-stratified grouped folds with fixed fold seed")
    return dict(p_value=(1+exceedances)/(1+replicates), exceedances=exceedances,
                replicates=int(replicates), seed=int(seed), alternative="greater",
                refit_each_permutation=True, unit="whole independent group",
                fold_policy="rebuild class-stratified grouped folds with fixed fold seed")


def refit_group_permutation(x, labels, groups, *, folds=5, l2=1.0,
                            fold_seed=20261008, replicates=199, seed=20261009,
                            observed_auc=None):
    """Preserve the instrumentable C2ST hook and its original calling contract.

    P5 uses the exact cached execution above. Nondefault regularization or a
    supplied observed statistic retain the original implementation, as do inputs
    requiring its validation. These compatibility routes are not used by P5.
    """
    if (l2 != 1.0 or observed_auc is not None or isinstance(replicates, bool)
            or not isinstance(replicates, (int, np.integer)) or replicates < 1
            or not np.isin(np.asarray(labels), (0, 1)).all()):
        return _baseline_refit_group_permutation(
            x, labels, groups, folds=folds, l2=l2, fold_seed=fold_seed,
            replicates=replicates, seed=seed, observed_auc=observed_auc)
    return _cached_refit_group_permutation(
        x, labels, groups, folds=folds, fold_seed=fold_seed,
        replicates=replicates, seed=seed)


def qualification_scope(reference: dict, candidate: dict) -> dict:
    """Bind declared population/effect design AND observed acquisition metadata.

    Scope metadata is an assertion to be reviewed, not proof of provenance. A
    missing or malformed scope permits raw difference testing but never transfers
    qualification. Unique acquisition IDs and seeds intentionally stay out of the
    population identity; the actual condition, framing and geometry stay in it.
    """
    declarations = [b.get("qualification_scope") for b in (reference, candidate)]
    errors = []
    for declaration in declarations:
        if not isinstance(declaration, dict):
            errors.append("missing qualification scope")
            continue
        generator = declaration.get("generator", {})
        if (declaration.get("schema") != "qualification-scope/1"
                or not isinstance(generator, dict)
                or not isinstance(generator.get("version"), str) or not generator["version"].strip()
                or not isinstance(generator.get("code_sha256"), str)
                or len(generator["code_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in generator["code_sha256"])
                or not isinstance(generator.get("configuration"), dict) or not generator["configuration"]
                or not isinstance(declaration.get("null_population"), dict) or not declaration["null_population"]
                or not isinstance(declaration.get("effects"), dict) or not declaration["effects"]):
            errors.append("incomplete generator/population/effect scope")
    if errors:
        return dict(sha256=None, errors=errors, value=None)
    try:
        if digest(declarations[0]) != digest(declarations[1]):
            raise ValueError("reference/candidate scope declarations differ")
        fields = ("condition_id", "framing_basis", "framing_id", "coordinate_convention",
                  "source_position", "microphone_position")
        acquisition = []
        for bundle in (reference, candidate):
            rows = [dict(label=r.get("label"), route=r.get("route"),
                         **{k: r.get(k) for k in fields}) for r in bundle["records"]]
            unique = {digest(r): r for r in rows}
            acquisition.append([unique[k] for k in sorted(unique)])
        value = dict(declaration=declarations[0], observed_acquisition=acquisition)
        return dict(sha256=digest(value), value=value, errors=[])
    except (ValueError, TypeError) as exc:
        return dict(sha256=None, errors=[str(exc)], value=None)


def run_family(reference: dict, candidate: dict, spec: Family, *, seed=None,
               execution="full") -> dict:
    """Raw complete-family difference test; never emits an acceptance receipt.

    Opt-in ``decision`` execution bounds the original fixed-budget C2ST p-values
    and preserves only the family rejection event. It reports intervals instead
    of complete per-member p-values and Holm diagnostics. All structural checks
    and observed fits still run, including after another member rejects.
    """
    if execution not in ("full", "decision"):
        raise ValueError("unknown family execution mode")
    if execution == "decision" and spec.mode != "independent":
        raise ValueError("decision execution supports independent groups only")
    if execution == "decision":
        from .permutation_decision import FixedBudgetDecision, first_holm_rejects
    expected = len(spec.strata)*len(spec.tests)
    family_decided = False
    seed = spec.seed if seed is None else int(seed)
    pairing = validate_pairing(reference["records"], candidate["records"],
                               mode=spec.mode, null_justification=spec.null_justification)
    errors = list(pairing["errors"])
    for bundle in (reference, candidate):
        if bundle.get("bank_version") != BANK_VERSION:
            errors.append("feature bank version mismatch")
        if any(bundle["physical"]["constant_audio"]) or any(bundle["physical"]["level_floor_used"]):
            errors.append("constant or floor-censored audio: boundary control only")
        if len(bundle["records"]) != len(bundle["physical"]["audio_sha256"]):
            errors.append("audio/provenance row mismatch")
        elif any(r["audio_sha256"] != h for r, h in zip(bundle["records"], bundle["physical"]["audio_sha256"])):
            errors.append("stale audio provenance")
        if bundle.get("units", {}).get("physical") != [u for _, u in PHYSICAL_BANK]:
            errors.append("incompatible physical units for frozen bank")
        if bundle["physical"].get("columns") != [c for c, _ in PHYSICAL_BANK]:
            errors.append("physical bank columns changed")
        if any((r.get("label"), r.get("route")) not in spec.strata for r in bundle["records"]):
            errors.append("undeclared class/route rows would be silently filtered")
    if reference.get("feature_config") != candidate.get("feature_config"):
        errors.append("incompatible deployed feature configuration or sample rate")
    results, counts, cluster_sizes = {}, {}, {}
    for label, route in spec.strata:
        key = f"{label}/{route}"
        indexes = [[i for i, r in enumerate(b["records"]) if (r["label"], r["route"]) == (label, route)]
                   for b in (reference, candidate)]
        # Duplicate content stays in its alias cluster; retain one copy per side.
        for side, bundle in enumerate((reference, candidate)):
            seen = set()
            indexes[side] = [i for i in indexes[side] if not (bundle["records"][i]["audio_sha256"] in seen
                             or seen.add(bundle["records"][i]["audio_sha256"]))]
        ix, iy = indexes
        conditions = {b["records"][i].get("condition_id") for b, indexes_side in
                      ((reference, ix), (candidate, iy)) for i in indexes_side}
        if len(conditions) > 1:
            errors.append(f"{key}: multiple acquisition conditions require prospectively separate strata")
        gx = [pairing["groups_left"][i] for i in ix]
        gy = [pairing["groups_right"][i] for i in iy]
        counts[key] = [len(set(gx)), len(set(gy))]
        cluster_sizes[key] = [sorted(groups.count(g) for g in set(groups)) for groups in (gx, gy)]
        if min(counts[key]) < 2:
            errors.append(f"{key}: fewer than two retained independent groups")
        if spec.mode == "paired" and set(gx) != set(gy):
            errors.append(f"{key}: paired clusters differ after retention")
        for view, test in spec.tests:
            name = f"{key}/{view}/{test}"
            try:
                a, b = (np.asarray(bundle["views"][view], float) for bundle in (reference, candidate))
                if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != len(reference["records"])
                        or b.shape[0] != len(candidate["records"]) or a.shape[1] != b.shape[1]
                        or not a.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all()):
                    raise ValueError("empty, incompatible or nonfinite view")
                if reference["units"][view] != candidate["units"][view]:
                    raise ValueError("incompatible units")
                if len(reference["units"][view]) != a.shape[1]:
                    raise ValueError("one declared unit per feature column required")
                x, y = a[ix], b[iy]
                if len(x) and len(y) and np.all(np.ptp(np.r_[x, y], axis=0) <= 1e-12):
                    raise ValueError("constant feature view: no distributional evidence")
                if errors:
                    continue
                if test == "max_mean":
                    result = _max_mean(x, y, gx, gy, paired=spec.mode == "paired",
                                       permutations=spec.permutations, seed=seed)
                elif spec.mode == "paired":
                    result = _paired_baseline(x, y, gx, gy, test, spec, seed)
                elif test == "mmd":
                    result = mmd_permutation_test(x, y, groups_x=gx, groups_y=gy,
                                                  permutations=spec.permutations, seed=seed)
                    if result["window_subsampled"]:
                        raise ValueError("undeclared baseline subsampling fallback")
                    result.update(p_value=result["permutation_p"], implementation="realism.mmd_permutation_test", weighting="windows")
                elif execution == "decision":
                    bound = FixedBudgetDecision(spec.permutations, expected, spec.alpha,
                                                family_already_rejected=family_decided)
                    result = _cached_refit_group_permutation(
                        np.r_[x, y], np.r_[np.zeros(len(x), int), np.ones(len(y), int)],
                        np.asarray(gx+gy), folds=spec.folds, fold_seed=spec.seed,
                        replicates=spec.permutations, seed=seed, _decision=bound)
                    result["implementation"] = "fixed-budget bounded C2ST decision"
                else:
                    result = refit_group_permutation(np.r_[x, y], np.r_[np.zeros(len(x), int), np.ones(len(y), int)],
                                                     np.asarray(gx+gy), folds=spec.folds, fold_seed=spec.seed,
                                                     replicates=spec.permutations, seed=seed)
                    result["implementation"] = "c2st.refit_group_permutation"
                if execution == "decision" and test == "c2st":
                    lower, upper = result["p_value_interval"]
                    if not (np.isfinite([lower, upper]).all() and 0 <= lower <= upper <= 1):
                        raise ValueError("invalid fixed-budget p-value bounds")
                else:
                    if not np.isfinite(result["p_value"]) or not 0 <= result["p_value"] <= 1:
                        raise ValueError("baseline returned invalid p-value")
                    if execution == "decision":
                        value = result.pop("p_value")
                        result["p_value_interval"] = [value, value]
                if execution == "decision":
                    family_decided |= first_holm_rejects(result["p_value_interval"][1], expected, spec.alpha)
                results[name] = result
            except (KeyError, ValueError, ImportError, FloatingPointError) as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if len(results) != expected:
        errors.append(f"incomplete required family: {len(results)}/{expected}")
    if execution == "decision" and not errors and not family_decided:
        if any(first_holm_rejects(r["p_value_interval"][0], expected, spec.alpha)
               for r in results.values()):
            errors.append("unresolved fixed-budget family decision")
    adjusted = _holm({k: v["p_value"] for k, v in results.items()}) if not errors and execution == "full" else {}
    for k, p in adjusted.items():
        results[k]["holm_p"] = p
    scope = qualification_scope(reference, candidate)
    report = dict(schema="power-family/2", family_sha256=spec.sha256, valid=not errors,
                qualification_scope_sha256=scope["sha256"], qualification_scope=scope["value"],
                qualification_scope_errors=scope["errors"],
                implementation_sha256=implementation_fingerprint(),
                feature_configuration_sha256=digest({"features": reference.get("feature_config"),
                                                    "units": reference.get("units"),
                                                    "cluster_size_proportions": {
                                                        key: [[(size, side.count(size)/len(side)) for size in sorted(set(side))]
                                                              for side in sides]
                                                        for key, sides in cluster_sizes.items()}}),
                errors=errors, results=results, pairing=pairing, retained_groups=counts,
                retained_windows_per_cluster=cluster_sizes,
                family_rejected=any(p <= spec.alpha for p in adjusted.values()) if not errors else None,
                difference=("detected" if any(p <= spec.alpha for p in adjusted.values()) else "not_detected") if not errors else "not_assessed",
                equivalence="not_assessed", multiplicity="Holm across all declared class/route/view/test cells",
                required_members=expected, observed_members=len(results), seed=seed)
    if execution == "decision":
        from . import permutation_decision
        # Bind the additional dependency without changing full-mode fingerprints.
        report.update(schema="power-family-decision/1", execution_mode="decision",
                      implementation_sha256=implementation_fingerprint(execution="decision"),
                      decision_implementation_sha256=sha256(Path(permutation_decision.__file__).read_bytes()).hexdigest(),
                      family_rejected=bool(family_decided) if not errors else None,
                      difference=("detected" if family_decided else "not_detected") if not errors else "not_assessed",
                      diagnostics="fixed-budget p-value intervals; individual Holm decisions not computed")
    return report


def binomial_bounds(successes: int, trials: int, confidence=.95) -> dict:
    """Exact one-sided Clopper-Pearson bounds, not a two-sided 95% interval."""
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("invalid binomial counts")
    return dict(count=successes, trials=trials, fraction=successes/trials,
                lower=float(beta.ppf(1-confidence, successes, trials-successes+1)) if successes else 0.,
                upper=float(beta.ppf(confidence, successes+1, trials-successes)) if successes < trials else 1.,
                confidence=confidence, interval="pointwise one-sided Clopper-Pearson")


@dataclass(frozen=True)
class Budget:
    group_counts: tuple[int, ...] = (4, 6, 10, 15, 18, 20)
    null_datasets: int = 1000
    alternative_datasets: int = 300
    defects: tuple[str, ...] = DEFECTS
    seed: int = 20261010
    null_upper_max: float = .075
    power_lower_min: float = .80
    max_seconds: float = 41400.
    workers: int = 1

    def __post_init__(self):
        if (not self.group_counts or len(set(self.group_counts)) != len(self.group_counts)
                or any(type(n) is not int or n < 2 for n in self.group_counts)
                or self.null_datasets < 1 or self.alternative_datasets < 1
                or not self.defects or len(set(self.defects)) != len(self.defects)):
            raise ValueError("invalid fixed replicate/group/defect budget")
        if type(self.workers) is not int or self.workers not in (1, 2, 4, 8):
            raise ValueError("workers must be one of 1, 2, 4, 8")
        if not 0 < self.null_upper_max < 1 or not 0 < self.power_lower_min < 1:
            raise ValueError("invalid qualification bounds")
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("positive finite wall-clock budget required")


def planned_cases(budget: Budget):
    """Stable case/seed order independent of worker count; no statistical retries."""
    for groups, effect in product(budget.group_counts, ("null",)+budget.defects):
        repetitions = budget.null_datasets if effect == "null" else budget.alternative_datasets
        for replicate in range(repetitions):
            seed = int(np.random.SeedSequence([budget.seed, groups,
                       (("null",)+budget.defects).index(effect), replicate]).generate_state(1)[0])
            yield dict(groups=groups, effect=effect, replicate=replicate, seed=seed)


def validate_case_plan(budget: Budget, cases) -> list[dict]:
    """Validate custodian cases without deriving, replacing or sorting any seed."""
    plan = list(cases)
    expected = [(g, effect, replicate)
                for g, effect in product(budget.group_counts, ("null",)+budget.defects)
                for replicate in range(budget.null_datasets if effect == "null" else budget.alternative_datasets)]
    if len(plan) != len(expected):
        raise ValueError("case plan count differs from fixed budget")
    seen = set()
    for row, identity in zip(plan, expected):
        if (not isinstance(row, dict) or set(row) != {"groups", "effect", "replicate", "seed"}
                or type(row["groups"]) is not int or type(row["replicate"]) is not int
                or (row["groups"], row["effect"], row["replicate"]) != identity
                or type(row["seed"]) is not int or not 0 <= row["seed"] < 2**64
                or row["seed"] in seen):
            raise ValueError("invalid, unordered or duplicate case plan")
        seen.add(row["seed"])
    return [dict(row) for row in plan]


def scientific_payload(report: dict) -> bytes:
    """Deterministic results; execution provenance remains in the raw report.

    Only worker PID/times/paths and scheduling resource settings are excluded.
    No statistic, interval, decision, seed, group or content digest is removed.
    """
    value = dict(report)
    value["budget"] = {k: v for k, v in report["budget"].items()
                       if k not in ("workers", "max_seconds")}
    value["attempts"] = [{k: v for k, v in row.items() if k != "worker"}
                         for row in report["attempts"]]
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _dataset_attempt(factory, spec, item, *, execution="full"):
    item = dict(item)
    item["worker"] = dict(pid=os.getpid(), start=datetime.now(timezone.utc).isoformat(),
                          module_path=str(Path(__file__).resolve()),
                          implementation_sha256=implementation_fingerprint(execution=execution),
                          threads={k: os.environ.get(k) for k in THREAD_VARIABLES})
    try:
        left, right = factory(item["groups"], item["effect"], item["seed"])
        contents = {r["audio_sha256"] for b in (left, right) for r in b["records"]}
        result = run_family(left, right, spec, seed=item["seed"],
                            **({"execution": execution} if execution != "full" else {}))
        item.update(execution="completed", result=result,
                    fixture_sha256=digest(sorted(contents)),
                    split_sha256=digest([b["records"] for b in (left, right)]),
                    input_records=[b["records"] for b in (left, right)],
                    feature_sha256=[{view: audio_digest(array) for view, array in b["views"].items()}
                                    for b in (left, right)],
                    independent_group_ids={key: result["pairing"][key] for key in ("groups_left", "groups_right")})
    except Exception as exc:
        item.update(execution="errored", error=f"{type(exc).__name__}: {exc}")
    item["worker"]["end"] = datetime.now(timezone.utc).isoformat()
    return item


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _scheduled_attempts(factory, spec, budget, started, *, execution="full",
                        cases=None, phase_seconds=None, phase_record=None):
    options = {"execution": execution} if execution != "full" else {}
    plan = list(planned_cases(budget)) if cases is None else validate_case_plan(budget, cases)
    if phase_seconds is not None and (set(phase_seconds) != set(budget.group_counts)
            or any(not np.isfinite(v) or v <= 0 for v in phase_seconds.values())):
        raise ValueError("positive finite deadline required for every group-count phase")
    def stopped(item, reason):
        return dict(item, execution="not run", error=reason+"; no retries")
    total_end = started + budget.max_seconds
    if budget.workers == 1 and phase_seconds is None:
        reason = None
        for item in plan:
            if time.monotonic() >= total_end:
                reason = "registered wall-clock budget exhausted"
            row = stopped(item, reason) if reason else _dataset_attempt(factory, spec, item, **options)
            if row["execution"] == "errored":
                reason = "prior worker error"
            yield row
        return
    threads = {os.environ.get(k) for k in THREAD_VARIABLES}
    if threads not in ({"1"}, {"2"}):
        raise ValueError("parallel workers require all numerical thread variables set to 1 or 2 before launch")
    # A deadline-enabled serial run also uses a spawned process, so a hung
    # numerical call can be stopped. There is never an IPC-to-serial fallback.
    pool = ProcessPoolExecutor(max_workers=budget.workers, mp_context=multiprocessing.get_context("spawn"))
    reason = None
    aborted = False
    try:
        for groups, phase in groupby(plan, key=lambda row: row["groups"]):
            phase_end = min(total_end, time.monotonic()+phase_seconds[groups]) if phase_seconds else total_end
            if phase_record:
                phase_record(dict(groups=groups, event="start", utc=datetime.now(timezone.utc).isoformat(),
                                  elapsed_seconds=time.monotonic()-started,
                                  remaining_seconds=max(0., phase_end-time.monotonic())))
            pending = deque()
            def consume(entry):
                nonlocal reason, aborted
                item, future, immediate = entry
                if future is None:
                    return immediate
                if aborted:
                    future.cancel()
                    return stopped(item, reason)
                try:
                    return future.result(timeout=max(0., phase_end-time.monotonic()))
                except Exception as exc:
                    reason = f"worker error or phase deadline: {type(exc).__name__}: {exc}"
                    aborted = True
                    return dict(item, execution="errored", error=reason)
            for item in phase:
                if len(pending) == budget.workers:
                    row = consume(pending.popleft())
                    if row["execution"] == "errored":
                        reason = row["error"]
                    yield row
                if time.monotonic() >= phase_end:
                    reason = "registered phase or total wall-clock budget exhausted"
                if reason:
                    pending.append((item, None, stopped(item, reason)))
                else:
                    try:
                        pending.append((item, pool.submit(_dataset_attempt, factory, spec, item, **options), None))
                        if phase_record:
                            phase_record(dict(item, event="submitted", utc=datetime.now(timezone.utc).isoformat(),
                                              elapsed_seconds=time.monotonic()-started))
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        aborted = True
                        pending.append((item, None, dict(item, execution="errored", error=reason)))
            while pending:
                row = consume(pending.popleft())
                if row["execution"] == "errored":
                    reason = row["error"]
                yield row
            if phase_record:
                phase_record(dict(groups=groups, event="end", utc=datetime.now(timezone.utc).isoformat(),
                                  elapsed_seconds=time.monotonic()-started, stop_reason=reason))
    finally:
        # Python 3.12 has no public terminate_workers API. Stop only processes
        # created by this executor; the launcher's separate group watchdog is
        # the final bound if shutdown itself fails. Never kill by process name.
        if aborted:
            processes = list((getattr(pool, "_processes", None) or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=1.)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.)
        pool.shutdown(wait=True, cancel_futures=True)


def qualify_test(factory: Callable, spec: Family, budget: Budget, *, record: Callable | None = None,
                 controls: dict | None = None, execution="full", cases=None, phase_seconds=None,
                 phase_record=None) -> dict:
    """Execute the fixed seed plan; retain all failures across worker boundaries.

    Development calls cannot qualify P5. Parallel execution changes scheduling
    only, with at most two numerical threads per fresh process and no retries.
    The verifier supplies signoff and enforces the outer hard wall-clock limit.
    """
    if execution not in ("full", "decision"):
        raise ValueError("unknown family execution mode")
    if execution == "decision" and spec.mode != "independent":
        raise ValueError("decision execution supports independent groups only")
    rows, seen = [], set()
    started = time.monotonic()
    feature_configuration = None
    scope_sha = None
    scope_value = None
    implementation = implementation_fingerprint(execution=execution)
    options = {"execution": execution} if execution != "full" else {}
    if cases is not None:
        options["cases"] = validate_case_plan(budget, cases)
    if phase_seconds is not None:
        options["phase_seconds"] = phase_seconds
    if phase_record is not None:
        options["phase_record"] = phase_record
    for item in _scheduled_attempts(factory, spec, budget, started, **options):
        if item.get("execution") == "completed":
            result = item["result"]
            contents = {r["audio_sha256"] for side in item["input_records"] for r in side}
            if feature_configuration is None:
                feature_configuration = result["feature_configuration_sha256"]
            if scope_sha is None:
                scope_sha = result["qualification_scope_sha256"]
                scope_value = result["qualification_scope"]
            problems = []
            if contents & seen:
                problems.append("content reused across independent datasets")
            if result["implementation_sha256"] != implementation:
                problems.append("implementation changed during fixed run")
            if result["feature_configuration_sha256"] != feature_configuration:
                problems.append("feature configuration changed during fixed run")
            if not scope_sha or result["qualification_scope_sha256"] != scope_sha:
                problems.append("missing or changed qualification scope during fixed run")
            if scope_value and set(scope_value["declaration"]["effects"]) != set(budget.defects):
                problems.append("registered effects differ from qualification scope")
            seen.update(contents)
            if any(count != [item["groups"]]*2 for count in result["retained_groups"].values()):
                problems.append("actual retained group count differs from registered count")
            if problems:
                result["errors"].extend(problems)
                result.update(valid=False, family_rejected=None, difference="not_assessed")
        rows.append(item)
        if record is not None:
            record(item)
    cells = []
    p5_budget = (budget.null_datasets >= 1000 and budget.alternative_datasets >= 300
                 and spec.permutations >= 1999 and spec.alpha == .05
                 and budget.null_upper_max <= .075 and budget.power_lower_min >= .80)
    signed = bool(controls and controls.get("decision") == "qualified"
                  and controls.get("family_sha256") == spec.sha256
                  and scope_sha and controls.get("qualification_scope_sha256") == scope_sha
                  and controls.get("implementation_sha256") == implementation
                  and controls.get("verifier") and controls.get("record_sha256"))
    for groups in budget.group_counts:
        outcomes = {}
        for effect in ("null",)+budget.defects:
            attempted = [r for r in rows if r["groups"] == groups and r["effect"] == effect]
            valid = [r for r in attempted if r.get("result", {}).get("valid")]
            bound = binomial_bounds(sum(r["result"]["family_rejected"] for r in valid), len(valid)) if valid else None
            # A missing/errored replicate never disappears from the denominator gate.
            complete = len(valid) == len(attempted)
            met = complete and bound is not None and (bound["upper"] <= budget.null_upper_max if effect == "null" else bound["lower"] >= budget.power_lower_min)
            outcomes[effect] = dict(attempted=len(attempted), valid=len(valid), bounds=bound,
                                    criterion_met=bool(met), missing_or_invalid=len(attempted)-len(valid))
        valid_all = all(v["missing_or_invalid"] == 0 for v in outcomes.values())
        qualified = valid_all and all(v["criterion_met"] for v in outcomes.values()) and p5_budget and signed
        cells.append(dict(groups=groups, effects=outcomes,
                          harness="qualified" if qualified else "unqualified" if valid_all else "invalid",
                          registered_outcome="pass" if qualified else "fail" if p5_budget and signed and valid_all else "inconclusive",
                          low_power=not all(outcomes[d]["criterion_met"] for d in budget.defects),
                          qualification_failed=not all(v["criterion_met"] for v in outcomes.values()),
                          minimum_mc_p=1/(spec.permutations+1),
                          first_holm_threshold=spec.alpha/(len(spec.strata)*len(spec.tests)),
                          two_sided_exact_minimum_p=2/(2**groups if spec.mode == "paired" else comb(2*groups, groups)),
                          two_sided_first_step_reachable=2/(2**groups if spec.mode == "paired" else comb(2*groups, groups)) <= spec.alpha/(len(spec.strata)*len(spec.tests)),
                          label_assignments=2**groups if spec.mode == "paired" else comb(2*groups, groups)))
    report = dict(schema="power-qualification/2", family_sha256=spec.sha256, budget=asdict(budget),
                implementation_sha256=implementation,
                feature_configuration_sha256=feature_configuration,
                qualification_scope_sha256=scope_sha, qualification_scope=scope_value,
                evidence_domain="synthetic proxy", cells=cells, attempts=rows,
                p5_budget=p5_budget, control_signoff_present=signed, independent_receipt=None,
                equivalence="not_assessed", limits="Power is specific to the named generator/effects/counts; no acoustic harmlessness or deployment claim.")
    if execution == "decision":
        report.update(schema="power-qualification-decision/1", execution_mode="decision")
    return report


def infer_difference(raw: dict, *, qualification: dict | None = None,
                     evidence_domain="synthetic proxy") -> dict:
    """Honest axes; a non-rejection NEVER establishes equivalence or realism."""
    if evidence_domain not in DOMAINS:
        raise ValueError("unknown evidence domain")
    supported = []
    if (qualification and qualification.get("family_sha256") == raw["family_sha256"]
            and qualification.get("implementation_sha256") == raw["implementation_sha256"]
            and qualification.get("feature_configuration_sha256") == raw["feature_configuration_sha256"]
            and raw.get("qualification_scope_sha256")
            and qualification.get("qualification_scope_sha256") == raw["qualification_scope_sha256"]
            and qualification.get("evidence_domain") == evidence_domain):
        counts = list(raw["retained_groups"].values())
        for cell in qualification["cells"]:
            if counts and all(c == [cell["groups"]]*2 for c in counts):
                supported.append(cell)
    harness = "invalid" if not raw["valid"] else "qualified" if supported and all(c["harness"] == "qualified" for c in supported) else "unqualified"
    return dict(execution="completed", harness=harness,
                difference=raw["difference"], equivalence="not_assessed",
                registered_outcome="fail" if harness == "qualified" and raw["difference"] == "detected" else "inconclusive",
                evidence_domain=evidence_domain, integration="not checked", review_decision="defer",
                low_power=harness != "qualified", effect_specific_power=supported,
                family_sha256=raw["family_sha256"], raw=raw,
                reason="Difference testing is separate from bounded equivalence; no margins are supplied.")


def legacy_status_adapter(e1: dict, e2: dict) -> dict:
    """Additive integration bridge: preserve historical outcomes without promotion."""
    return dict(historical_status={"e1": e1.get("status"), "e2": e2.get("status")},
                difference="not_assessed", equivalence="not_assessed", harness="unqualified",
                registered_outcome="inconclusive", review_decision="defer",
                reason="Historical E1/E2 gates were not qualified as this complete class/view/test family.")
