"""Matched-group permutation calibration for the joint E1 decision."""
import numpy as np

from .realism import _band_scores, _group_rbf, _mmd_from_group_blocks


def joint_permutation(x, y, gx, gy, standardized, gamma, n_bands, permutations, seed):
    if permutations < 19:
        raise ValueError('at least 19 permutations are needed for alpha=0.05')
    values = np.vstack((x, y))
    groups = np.r_[np.char.add('x:', gx), np.char.add('y:', gy)]
    names, blocks, counts = _group_rbf(standardized, groups, gamma)
    # Sufficient statistics avoid repeatedly gathering all windows on each permutation.
    sums = np.asarray([values[groups == name].sum(0) for name in names])
    squares = np.asarray([(values[groups == name]**2).sum(0) for name in names])
    nl = len(np.unique(gx))
    left = np.flatnonzero(np.char.startswith(names, 'x:'))
    rng = np.random.default_rng(seed)
    right_mask = np.ones(len(names), dtype=bool)
    mmds, bands, smds = [], [], []
    for iteration in range(permutations+1):
        if iteration:
            left = rng.choice(len(names), nl, replace=False)
        right_mask[:] = True
        right_mask[left] = False
        right = np.flatnonzero(right_mask)
        mmds.append(_mmd_from_group_blocks(blocks, counts, left, right))
        moments = []
        for codes in (left, right):
            count = counts[codes].sum()
            total = sums[codes].sum(0)
            mean = total/count
            var = np.maximum(0, (squares[codes].sum(0)-total*mean)/max(count-1, 1))
            moments.append((mean, var))
        smd = (moments[0][0]-moments[1][0])/np.maximum(np.sqrt((moments[0][1]+moments[1][1])/2), 1e-12)
        smds.append(smd)
        bands.append(_band_scores(smd, n_bands) if n_bands is not None else np.abs(smd))
    mmds, bands = np.asarray(mmds), np.asarray(bands)
    # Include observed and permutations symmetrically when setting marginal scales.
    mmd_scale = max(float(np.quantile(mmds, .95)), 1e-12)
    band_scale = np.maximum(np.quantile(bands, .95, axis=0), 1e-12)
    # The ceil(.9*B)-th smallest ratio <= threshold means >=90% are inside.
    coverage_rank = int(np.ceil(.9*bands.shape[1]))-1
    coverage_score = np.sort(bands/band_scale, axis=1)[:, coverage_rank]
    joint = np.maximum(mmds/mmd_scale, coverage_score)
    p = float((1+np.count_nonzero(joint[1:] >= joint[0]))/(permutations+1))
    critical = float(np.quantile(joint, .95, method='higher'))
    group_counts = {'left': len(np.unique(gx)), 'right': len(np.unique(gy))}
    decision = {'method': 'joint-group-label-permutation', 'alpha': .05,
                'permutation_p': p, 'permutations': permutations, 'n_groups': group_counts,
                'statistic': float(joint[0]), 'critical_q95': critical,
                'mmd2_limit': critical*mmd_scale, 'band_smd_envelope': (critical*band_scale).tolist(),
                'required_band_coverage': .9,
                'decision_rule': 'fail if joint permutation p <= 0.05; pass requires both constraints',
                'assumption': 'independent exchangeable take groups under the null'}
    mmd = {'mmd2': float(mmds[0]), 'permutation_p': float((1+(mmds[1:] >= mmds[0]).sum())/(permutations+1)),
           'permutations': permutations, 'kernel': 'rbf-median-distance', 'gamma': gamma,
           'n_groups': group_counts, 'n_windows': {'left': len(x), 'right': len(y)}}
    return decision, mmd, smds[0], bands[0]
