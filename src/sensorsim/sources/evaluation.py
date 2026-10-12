"""Prepare frozen WP6(b/c) jobs without fabricating recognition results."""
from __future__ import annotations

import numpy as np

from .catalog import audit_splits, _keys


def prepare_acceptance(assigned, *, training_steps, examples_per_run, seeds=(610, 611, 612)):
    """Serializable handoff to WP1/WP2/WP4; no fitting/evaluation is performed.

    The caller pre-registers this plan before training. RH identities are never
    chosen as train or calibration inputs. Insufficient groups are explicit.
    A model-based plan cannot establish measured recall or false-alarm rates.
    """
    assigned = list(assigned)
    if any(a.take.details.get('mode') == 'replay' or a.take.details.get('c2st_eligible') is False
           for a in assigned):
        raise ValueError('gain replay is excluded from WP6 acceptance and C2ST diversity evidence')
    audit = audit_splits(assigned)
    if (type(training_steps) is not int or training_steps <= 0 or type(examples_per_run) is not int
            or examples_per_run <= 0 or len(seeds) != 3 or len(set(seeds)) != 3):
        raise ValueError('positive identical training/data budgets and three distinct seeds required')
    from sensorsim.acoustics.room.dataset import _connected_groups
    classes = sorted({a.take.label for a in assigned})
    training = [a.take for a in assigned if a.split == 'train']
    components = _connected_groups([_keys(t) for t in training])
    if any(len({training[i].label for i in indices}) != 1 for indices in components.values()):
        raise ValueError('learning-curve group counts require single-class connected performances')
    units = {key: sorted({training[i].source_group for i in indices})
             for key, indices in components.items()}
    jobs, shortages = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        by_class = {}
        for label in classes:
            groups = [key for key, indices in components.items() if training[indices[0]].label == label]
            by_class[label] = rng.permutation(groups).tolist()
        for count in (4, 8, 'all'):
            chosen = {label: groups if count == 'all' else groups[:count] for label, groups in by_class.items()}
            missing = [label for label, groups in chosen.items() if count != 'all' and len(groups) < count]
            independent_counts = {label: len(groups) for label, groups in chosen.items()}
            expanded = {label: [g for key in groups for g in units[key]] for label, groups in chosen.items()}
            if missing:
                shortages.append(dict(seed=int(seed), training_groups=count, classes=missing))
            for mode in ('real_only', 'real_plus_rendered'):
                jobs.append(dict(seed=int(seed), training_groups=count, mode=mode,
                    training_steps=training_steps, examples_per_run=examples_per_run,
                    source_groups=expanded, independent_counts=independent_counts, status='insufficient_groups' if missing else 'pending'))
    return dict(schema_version='wp6-acceptance-plan/1', status='pending',
        result_label='measured' if assigned and all(a.take.evidence_label == 'measured' for a in assigned) else 'proxy',
        leakage=audit, jobs=jobs, shortages=shortages,
        RH=[a.take.metadata() for a in assigned if a.split == 'RH'],
        validation=[a.take.metadata() for a in assigned if a.split == 'val'],
        b=dict(status='pending', rule='Median of three matched-budget seeds: recall >= real-only on >=2 classes; '
               'loss <=1 RH take in every class; WP2 audit trigger ratio <=1.25. '
               'Report per-class Clopper-Pearson 95% intervals and seed spread.'),
        c=dict(status='pending', rule='Per-class C2ST must not fail; RH recall improves by >=1 take; '
               'no FA increase. Pass C2ST only for point AUC <=0.60 and upper95 <=0.70; '
               'fail if lower95 >=0.60; otherwise inconclusive.'),
        missing=['paired in_situ RH and training anchors', 'WP1 measured recall/C2ST',
                 'WP2 frozen negative audit', 'WP4 anchored render ensemble'],
        warning='A plan is not evidence of adoption. Fewer than eight independent train groups cannot run the 8-group arm.')
