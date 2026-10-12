"""Numpy-only grouped cross-validated L2-logistic classifier two-sample tests."""
from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _loss(x: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float) -> float:
    z = x @ weights
    return float(np.mean(np.logaddexp(0.0, z) - y*z) + .5*l2*np.dot(weights[1:], weights[1:]))


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0,
                  max_iter: int = 80, tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.r_[0.0, np.full(design.shape[1] - 1, l2)]
    for _ in range(max_iter):
        probability = _sigmoid(design @ weights)
        grad = design.T @ (probability - y) / len(y) + penalty * weights
        variance = probability * (1.0 - probability)
        hessian = (design.T * variance) @ design / len(y)
        hessian.flat[::hessian.shape[0] + 1] += penalty + 1e-9
        try:
            delta = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        if np.max(np.abs(grad)) < tolerance:
            break
        before = _loss(design, y, weights, l2)
        step = 1.0
        while step > 1e-6:
            candidate = weights - step*delta
            if _loss(design, y, candidate, l2) <= before + 1e-12:
                weights = candidate
                break
            step *= .5
        else:
            break
        if np.max(np.abs(step*delta)) < tolerance:
            break
    return weights, mean, scale


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    positive, negative = int(np.count_nonzero(y == 1)), int(np.count_nonzero(y == 0))
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_scores = s[order]
    # Keep the stable ordering and average ranks, including singleton NaNs.
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(s)]
    ranks[order] = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
    return float((ranks[y == 1].sum() - positive*(positive + 1)/2) / (positive*negative))


def _group_folds(groups: np.ndarray, labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    names = np.unique(groups)
    if folds < 2 or len(names) < folds:
        raise ValueError("grouped CV needs at least two folds and at least one group per fold")
    member = {name: np.flatnonzero(groups == name) for name in names}
    group_labels = {name: np.unique(labels[index]) for name, index in member.items()}
    rng = np.random.default_rng(seed)
    if all(len(value) == 1 for value in group_labels.values()):
        assignments: list[list[str]] = [[] for _ in range(folds)]
        for cls in (0, 1):
            class_groups = np.asarray([name for name in names if group_labels[name][0] == cls], dtype=object)
            if len(class_groups) < folds:
                raise ValueError("each class needs at least one independent group per fold")
            class_groups = class_groups[rng.permutation(len(class_groups))]
            for i, name in enumerate(class_groups):
                assignments[i % folds].append(str(name))
    else:
        totals = np.asarray([(labels[member[name]] == 0).sum() for name in names], float)
        positives = np.asarray([(labels[member[name]] == 1).sum() for name in names], float)
        order = np.argsort(-(totals + positives) + rng.random(len(names))*1e-6)
        counts = np.zeros((folds, 2), float)
        assignments = [[] for _ in range(folds)]
        target = np.asarray([np.sum(totals), np.sum(positives)]) / folds
        for index in order:
            name = str(names[index])
            add = np.asarray([totals[index], positives[index]])
            objectives = [np.sum(((counts[k] + add - target) / np.maximum(target, 1.0))**2) for k in range(folds)]
            chosen = int(np.argmin(objectives))
            assignments[chosen].append(name)
            counts[chosen] += add
        if any(len(np.unique(labels[np.isin(groups, names_in_fold)])) != 2 for names_in_fold in assignments):
            raise ValueError("could not form grouped folds")
    return [np.flatnonzero(np.isin(groups, fold_groups)) for fold_groups in assignments]


def _cluster_bootstrap(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, *,
                       replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    group_class = {group: np.unique(labels[index]) for group, index in members.items()}
    pure = all(len(value) == 1 for value in group_class.values())
    candidates_by_class = [
        np.asarray([group for group in unique if group_class[group][0] == cls], dtype=object)
        for cls in (0, 1)
    ] if pure else []
    draws = np.empty(replicates, float)
    for i in range(replicates):
        selected = []
        if pure:
            for candidates in candidates_by_class:
                sampled = rng.choice(candidates, len(candidates), replace=True)
                selected.extend(members[group] for group in sampled)
        else:
            sampled = rng.choice(unique, len(unique), replace=True)
            selected.extend(members[group] for group in sampled)
        index = np.concatenate(selected)
        draws[i] = auc_score(labels[index], scores[index])
    return draws


def decision(point_auc: float, lower: float, upper: float) -> str:
    """Shared classifier two-sample decision rule."""
    if upper < .5:
        return "inconclusive"
    if upper <= .70 and point_auc <= .60:
        return "pass"
    if lower >= .60:
        return "fail"
    return "inconclusive"


def _cv_scores(x, labels, groups, *, folds, l2, fold_seed):
    """Refit all training-fold preprocessing and probes for this label assignment."""
    test_folds = _group_folds(groups, labels, folds, fold_seed)
    scores = np.full(len(labels), np.nan, float)
    for test in test_folds:
        train_mask = np.ones(len(labels), bool)
        train_mask[test] = False
        if len(np.unique(labels[train_mask])) < 2 or len(np.unique(labels[test])) < 2:
            raise ValueError("every grouped CV split needs both classes in train and test")
        weights, mean, scale = _fit_logistic(x[train_mask], labels[train_mask], l2=l2)
        z = (x[test] - mean) / scale
        scores[test] = _sigmoid(np.column_stack((np.ones(len(test)), z)) @ weights)
    return scores


def refit_group_permutation(x, labels, groups, *, folds=5, l2=1.0,
                            fold_seed=20261008, replicates=199, seed=20261009,
                            observed_auc=None):
    """One-sided whole-group label randomization of the complete CV algorithm.

    Group counts by domain are fixed, and class-stratified folds are rebuilt for
    each assignment with the same fold seed. Every normalization and probe is
    refitted. No held-out scores are reused as permutation statistics.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("permutation replicates must be a positive integer")
    labels, groups = np.asarray(labels), np.asarray(groups)
    names, inverse = np.unique(groups, return_inverse=True)
    group_labels = []
    for name in names:
        values = np.unique(labels[groups == name])
        if len(values) != 1:
            raise ValueError("group permutation requires one domain label per group")
        group_labels.append(values[0])
    group_labels = np.asarray(group_labels)
    if observed_auc is None:
        observed_auc = auc_score(labels, _cv_scores(x, labels, groups, folds=folds,
                                                   l2=l2, fold_seed=fold_seed))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(replicates):
        permuted = rng.permutation(group_labels)[inverse]
        scores = _cv_scores(x, permuted, groups, folds=folds, l2=l2, fold_seed=fold_seed)
        exceedances += int(auc_score(permuted, scores) >= observed_auc)
    return {"p_value": (1 + exceedances)/(1 + replicates), "exceedances": exceedances,
            "replicates": int(replicates), "seed": int(seed), "alternative": "greater",
            "refit_each_permutation": True, "unit": "whole independent group",
            "fold_policy": "rebuild class-stratified grouped folds with fixed fold seed"}


PASS_INTERPRETATION = (
    "An E2 pass is weak evidence at realistic take counts, even with spike-in-qualified power. "
    "The independent proxy review found single-view passes in 10–40% of runs at true AUC "
    "0.70–0.80, and 52/100 at 60 groups per side with true AUC 0.75. "
    "Those are review evidence, not power measured for this run.")
BOOTSTRAP_LIMIT = (
    "The cluster bootstrap resamples fixed out-of-fold scores without refitting. "
    "In the independent proxy review at 15 single-window groups per side, its 95% "
    "interval excluded 0.5 in 12–14% of null runs. It under-covers CV variability; "
    "the refit group-permutation p-value tests greater-than-null AUC, not equivalence.")


def _one_probe(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, folds: int,
               l2: float, bootstrap_replicates: int, bootstrap_seeds: tuple[int, ...],
               fold_seed: int, permutation_replicates: int = 199,
               permutation_seed: int = 20261009) -> dict:
    scores = _cv_scores(x, labels, groups, folds=folds, l2=l2, fold_seed=fold_seed)
    point = auc_score(labels, scores)
    permutation = refit_group_permutation(x, labels, groups, folds=folds, l2=l2,
        fold_seed=fold_seed, replicates=permutation_replicates, seed=permutation_seed,
        observed_auc=point)
    seed_results = []
    for seed in bootstrap_seeds:
        draws = _cluster_bootstrap(labels, scores, groups, replicates=bootstrap_replicates, seed=seed)
        lower, upper = np.quantile(draws, [.025, .975])
        seed_results.append({"seed": int(seed), "lower_95": float(lower), "upper_95": float(upper)})
    lowers = np.asarray([r["lower_95"] for r in seed_results])
    uppers = np.asarray([r["upper_95"] for r in seed_results])
    lower, upper = float(np.median(lowers)), float(np.median(uppers))
    return {
        "point_auc": float(point), "lower_95": lower, "upper_95": upper,
        "outcome": decision(float(point), lower, upper), "folds": int(folds), "l2": float(l2),
        "cluster_bootstrap": seed_results,
        "group_permutation": permutation,
        "below_chance_cv_instability": bool(upper < .5),
        "reason": "inconclusive: CV instability" if upper < .5 else None,
        "bootstrap_limit": BOOTSTRAP_LIMIT, "pass_interpretation": PASS_INTERPRETATION,
        "bootstrap_seed_spread": {"lower_95": [float(lowers.min()), float(lowers.max())],
                                   "upper_95": [float(uppers.min()), float(uppers.max())]},
        "n_groups": {"class_0": int(len(np.unique(groups[labels == 0]))),
                     "class_1": int(len(np.unique(groups[labels == 1])))},
        "n_windows": {"class_0": int(np.count_nonzero(labels == 0)),
                      "class_1": int(np.count_nonzero(labels == 1))},
    }


def compare_samples(x0: np.ndarray, x1: np.ndarray, groups0, groups1, *, folds: int = 5,
                    l2: float = 1.0, bootstrap_replicates: int = 2000,
                    bootstrap_seeds: tuple[int, ...] = (20261008, 20261009, 20261010),
                    fold_seed: int = 20261008, permutation_replicates: int = 199,
                    permutation_seed: int = 20261009) -> dict:
    """Run grouped C2ST probes for 120-D band statistics and pooled embeddings."""
    a, b = np.asarray(x0, float), np.asarray(x1, float)
    g0, g1 = np.asarray(groups0).astype(str), np.asarray(groups1).astype(str)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        raise ValueError("C2ST inputs must be nonempty matrices with equal feature widths")
    if g0.shape != (len(a),) or g1.shape != (len(b),):
        raise ValueError("one group ID is required per feature row")
    x = np.vstack((a, b))
    labels = np.r_[np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)]
    groups = np.r_[np.char.add("reference:", g0), np.char.add("synthetic:", g1)]
    n_folds = min(int(folds), len(np.unique(g0)), len(np.unique(g1)))
    if n_folds < 2:
        return {"status": "inconclusive", "reason": "fewer than two independent groups in a class"}
    try:
        result = _one_probe(x, labels, groups, folds=n_folds, l2=l2,
            bootstrap_replicates=bootstrap_replicates, bootstrap_seeds=tuple(bootstrap_seeds),
            fold_seed=fold_seed, permutation_replicates=permutation_replicates,
            permutation_seed=permutation_seed)
    except ValueError as error:
        return {"status": "inconclusive", "reason": str(error)}
    return {"status": "ok", "result": result}


def compare_representations(stats0: np.ndarray, stats1: np.ndarray, embed0: np.ndarray,
                            embed1: np.ndarray, groups0, groups1, **kwargs) -> dict:
    """Compare both required feature views with identical folds and bootstrap budgets."""
    stats = compare_samples(stats0, stats1, groups0, groups1, **kwargs)
    embeddings = compare_samples(embed0, embed1, groups0, groups1, **kwargs)
    return {"band_statistics": stats, "pooled_embedding": embeddings}
