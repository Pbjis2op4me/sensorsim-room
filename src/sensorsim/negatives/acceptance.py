"""WP3 reusable evaluation helpers; no implicit training or threshold selection."""
from __future__ import annotations

import numpy as np


def symmetry(positive_scores, negative_scores, positive_embeddings, negative_embeddings,
             groups, *, seed: int, bootstrap_replicates: int = 1000,
             integrated_frozen_b: bool = False) -> dict:
    """Paired source-group C2ST, grouped CV and paired-cluster bootstrap.

    Inputs come from a frozen checkpoint, identical neutral source content and
    matched room/device draws. The scorer's provenance belongs in the caller's
    evidence. This is a path check, not an event false-alarm or realism metric.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p, n = np.asarray(positive_scores, float), np.asarray(negative_scores, float)
    a, b = np.asarray(positive_embeddings, float), np.asarray(negative_embeddings, float)
    g = np.asarray(groups)
    if (p.ndim != 2 or p.shape != n.shape or not p.shape[1] or
            a.ndim != 2 or a.shape != b.shape or not a.shape[1] or
            len(p) != len(a) or len(g) != len(a) or len(np.unique(g)) < 4 or
            not all(np.isfinite(x).all() for x in (p, n, a, b))):
        raise ValueError('paired finite matrices and at least four independent groups are required')
    if np.any((p < 0) | (p > 1)) or np.any((n < 0) | (n > 1)) or bootstrap_replicates < 100:
        raise ValueError('scores must be probabilities; use at least 100 bootstrap replicates')
    x = np.concatenate([a, b])
    labels = np.repeat([0, 1], len(a))
    paired_groups = np.tile(g, 2)
    predictions = np.zeros(len(x))
    coefficients = []
    for train, test in GroupKFold(4).split(x, labels, paired_groups):
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500, random_state=seed))
        probe.fit(x[train], labels[train])
        predictions[test] = probe.predict_proba(x[test])[:, 1]
        coefficients.append(probe[-1].coef_[0])
    auc = float(roc_auc_score(labels, predictions))
    unique = np.unique(g)
    clusters = [np.flatnonzero(paired_groups == key) for key in unique]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_replicates):
        ids = np.concatenate([clusters[k] for k in rng.integers(len(clusters), size=len(clusters))])
        draws.append(roc_auc_score(labels[ids], predictions[ids]))
    lo, hi = map(float, np.quantile(draws, [.025, .975]))
    differences = np.abs(p.mean(axis=0)-n.mean(axis=0))
    if np.any(differences > .02) or lo >= .60:
        decision = 'fail'
    elif auc <= .60 and hi <= .70:
        decision = 'pass'
    else:
        decision = 'inconclusive'
    coeff = np.mean(np.abs(coefficients), axis=0)
    return {'evidence_label': 'proxy', 'structural_result': decision,
            'wp3_b_result': decision if integrated_frozen_b else 'inconclusive',
            'limitation': None if integrated_frozen_b else 'trained frozen B / integrated positive path unavailable',
            'mean_score_differences': differences.tolist(), 'auc': auc, 'auc_ci95': [lo, hi],
            'groups': len(unique), 'pairs': len(a), 'bootstrap_replicates': bootstrap_replicates,
            'top_embedding_dimensions': np.argsort(-coeff, kind='stable')[:5].tolist(),
            'coefficient_magnitudes': np.sort(coeff)[-5:][::-1].tolist()}


def training_comparison_plan(*, split_sha256: str, seeds=(1701, 1702, 1703)) -> dict:
    """WP3(a) harness contract for the trainer/WP2 integration lane.

    This deliberately cannot produce a pass from surrogate window probabilities:
    the event tracker, frozen thresholds, exposures and target recall are needed.
    """
    if len(seeds) != 3 or len(set(seeds)) != 3 or len(split_sha256) != 64:
        raise ValueError('three distinct seeds and the locked split SHA-256 are required')
    return {'version': 'wp3-training-comparison/1', 'status': 'pending',
            'reason': 'requires integrated training stream, WP2 event counts and held-out TRUTH targets',
            'evidence_label': 'proxy result', 'seeds': list(seeds), 'split_sha256': split_sha256,
            'arms': {'A': 'floor + white/pink noise', 'B': 'reviewed negative library'},
            'same_budget': ['training_steps', 'data_volume', 'source_split', 'target_thresholds'],
            'audit': {'minimum_families': 8, 'minimum_instances_per_family': 100,
                      'held_out_source_items': True, 'never_mine_or_tune': True},
            'metrics_required': ['event triggers and exposures per family/class/seed',
                                 'TRUTH held-out target successes and totals per class/seed',
                                 'WP2 confidence intervals; frozen detector settings'],
            'decision': {'median_trigger_reduction_min': .5, 'apply_to': ['overall', 'each class'],
                         'recall_absolute_drop_max': .05, 'wide_intervals': 'inconclusive'},
            'deployment_fa_per_hour': 'not measured'}


def run_training_comparison(*, split, expected_split_sha256: str, train_arm, score_arm,
                            steps: int, examples: int, seeds=(1701, 1702, 1703)) -> dict:
    """Execute the WP3(a) six-run schedule once integrations are supplied.

    train_arm(arm, seed, steps, examples, training_items, development_items)
    returns a checkpoint handle. score_arm(checkpoint, audit_items, seed) returns
    WP2 event-count/exposure and TRUTH recall evidence; it must not fit thresholds.
    Audit objects are passed ONLY to score_arm. No automatic acceptance decision
    is made from unchecked callback output: the returned plan names the required
    metrics and pre-registered thresholds for the integration lane's analysis.
    """
    from collections import Counter
    split.verify_lock(expected_split_sha256)
    if not isinstance(steps, int) or not isinstance(examples, int) or min(steps, examples) < 1:
        raise ValueError('positive matched step and example budgets are required')
    plan = training_comparison_plan(split_sha256=expected_split_sha256, seeds=seeds)
    training = tuple(item for item, part in split.rows if part == 'train')
    development = tuple(item for item, part in split.rows if part == 'development')
    audit = tuple(item for item, part in split.rows if part == 'audit')
    counts = Counter(item.family for item in audit)
    independent = {family: min(len({item.source_group for item in audit if item.family == family}),
                               len({item.sha256 for item in audit if item.family == family}))
                   for family in counts}
    if not training or len(counts) < 8 or min(independent.values()) < 100:
        raise ValueError('require training data and locked audit of >=8 families x >=100 independent source items')
    runs = []
    for seed in seeds:
        for arm in ('A', 'B'):
            checkpoint = train_arm(arm=arm, seed=seed, steps=steps, examples=examples,
                                   training_items=training, development_items=development)
            metrics = score_arm(checkpoint=checkpoint, audit_items=audit, seed=seed)
            runs.append({'arm': arm, 'seed': seed, 'steps': steps, 'examples': examples, 'metrics': metrics})
    return {**plan, 'status': 'inconclusive', 'reason': 'runs collected; WP2 intervals and TRUTH recall require analysis',
            'runs': runs, 'audit_family_counts': dict(counts), 'audit_independent_counts': independent}
