"""Recognizer-space realism metrics built on the device runtime's windows."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Callable, Iterable

import numpy as np


def window_starts(sample_count: int, window_size: int, hop_size: int) -> np.ndarray:
    """Sample offsets used by ``edge.window_scores`` for a continuous recording."""
    if min(sample_count, window_size, hop_size) < 1:
        raise ValueError("sample_count, window_size and hop_size must be positive")
    return np.arange(0, max(0, sample_count - window_size + 1), hop_size, dtype=np.int64)


def band_statistics(log_mel: np.ndarray) -> np.ndarray:
    """Return 120 values per window: per-band mean, standard deviation and maximum."""
    x = np.asarray(log_mel, dtype=np.float32)
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-1] == 0:
        raise ValueError("log_mel must have shape (windows, 1, bands, frames)")
    x = x[:, 0]
    return np.concatenate((x.mean(axis=-1), x.std(axis=-1), x.max(axis=-1)), axis=1).astype(np.float32)


def mel_centres_hz(features) -> np.ndarray:
    """Approximate each mel band's centre by the peak of edge.Features' exact filter bank."""
    cfg = features.cfg
    freqs = np.fft.rfftfreq(cfg["n_fft"], 1.0 / cfg["fs"])
    return freqs[np.argmax(features.bank, axis=1)].astype(float)


def band_containing_hz(features, frequency_hz: float) -> int:
    """Index of the strongest mel filter at the FFT bin nearest ``frequency_hz``."""
    cfg = features.cfg
    freqs = np.fft.rfftfreq(cfg["n_fft"], 1.0 / cfg["fs"])
    bin_index = int(np.argmin(np.abs(freqs - float(frequency_hz))))
    return int(np.argmax(features.bank[:, bin_index]))


def pooled_embedding(model, feature_windows: np.ndarray) -> np.ndarray:
    """Read the 96-value (default model) activation immediately after the CNN pool layer.

    The layer interpreter and weights are the frozen exported ``edge.Model``. This walks the
    existing layer list and stops at its pooling layer, before the trainable output layer.
    """
    x = np.asarray(feature_windows, dtype=np.float32)
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("feature_windows must have shape (windows, 1, bands, frames)")
    for layer in model.meta["layers"]:
        x = model._apply(layer, x)
        if layer["type"] == "pool":
            if x.ndim != 2 or not np.isfinite(x).all():
                raise ValueError("CNN pooled embedding must be a finite matrix")
            return x.astype(np.float32, copy=False)
    raise ValueError("edge model has no pooled embedding layer")


def extract_audio_features(audio: np.ndarray, model, *, hop_s: float | None = None) -> dict:
    """Extract device-identical log-mel windows, 120-D statistics and pooled embeddings."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    fs = int(model.meta["features"]["fs"])
    hop_s = float(model.meta["detector"]["hop_s"] if hop_s is None else hop_s)
    hop = int(round(hop_s * fs))
    starts = window_starts(x.size, model.features.n, hop)
    if starts.size == 0:
        n_bands = int(model.meta["features"]["n_mels"])
        return {
            "starts_s": [], "band_statistics": np.zeros((0, 3 * n_bands), np.float32),
            "embedding": np.zeros((0, 0), np.float32),
            "band_centres_hz": mel_centres_hz(model.features),
            "band_3khz": band_containing_hz(model.features, 3000.0),
        }
    # Bound CNN activations independently of take duration.
    stats, embeddings = [], []
    for offset in range(0, len(starts), 16):
        windows = np.stack([x[int(start):int(start) + model.features.n]
                            for start in starts[offset:offset + 16]])
        log_mel = model.features(windows)
        stats.append(band_statistics(log_mel))
        embeddings.append(pooled_embedding(model, log_mel))
    return {
        "starts_s": (starts / fs).astype(float).tolist(),
        "band_statistics": np.concatenate(stats),
        "embedding": np.concatenate(embeddings),
        "band_centres_hz": mel_centres_hz(model.features),
        "band_3khz": band_containing_hz(model.features, 3000.0),
    }


def _smd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not x.shape[0] or not y.shape[0]:
        raise ValueError("SMD inputs must be nonempty matrices with the same feature width")
    vx = x.var(axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
    vy = y.var(axis=0, ddof=1) if len(y) > 1 else np.zeros(y.shape[1])
    scale = np.sqrt((vx + vy) / 2.0)
    scale = np.maximum(scale, 1e-12)
    return (x.mean(axis=0) - y.mean(axis=0)) / scale


def _band_scores(smd: np.ndarray, n_bands: int) -> np.ndarray:
    if smd.size != 3 * n_bands:
        raise ValueError(f"expected {3*n_bands} band-statistic SMD values, got {smd.size}")
    return np.max(np.abs(smd.reshape(3, n_bands).T), axis=1)


def _sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = (x*x).sum(axis=1)[:, None] + (y*y).sum(axis=1)[None, :] - 2.0 * x @ y.T
    return np.maximum(d, 0.0)


def _kernel_parameters(x, y, seed, center=None, scale=None, gamma=None):
    both = np.vstack((x, y)).astype(float, copy=False)
    center = both.mean(axis=0) if center is None else np.asarray(center, float)
    scale = both.std(axis=0) if scale is None else np.asarray(scale, float).copy()
    scale[scale < 1e-12] = 1.0
    both = (both - center) / scale
    if gamma is None:
        rng = np.random.default_rng(seed)
        sample = both if len(both) <= 512 else both[np.sort(rng.choice(len(both), 512, replace=False))]
        distances = _sqdist(sample, sample)
        positive = distances[np.triu_indices(len(sample), 1)]
        positive = positive[positive > 0]
        gamma = 1.0 / max(float(np.median(positive)) if positive.size else 1.0, 1e-12)
    return both, float(gamma), center, scale


def _kernel_matrix(x, y, seed, *, center=None, scale=None, gamma=None):
    both, gamma, _, _ = _kernel_parameters(x, y, seed, center, scale, gamma)
    return np.exp(-gamma * _sqdist(both, both)), gamma


def _group_rbf(values, groups, gamma):
    """Accumulate group kernel sums in 256-window tiles, never a full window kernel."""
    names, codes = np.unique(groups, return_inverse=True)
    size = len(names)
    counts = np.bincount(codes, minlength=size)
    blocks = np.zeros((size, size))
    for i in range(0, len(values), 256):
        left = values[i:i+256]
        for j in range(0, len(values), 256):
            kernel = np.exp(-gamma * _sqdist(left, values[j:j+256]))
            indexes = codes[i:i+256, None] * size + codes[None, j:j+256]
            # add.at avoids a group_count squared temporary for every tile.
            np.add.at(blocks.ravel(), indexes.ravel(), kernel.ravel())
    return names, blocks, counts


def _mmd_from_kernel(kernel: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    if left.sum() == 0 or right.sum() == 0:
        return float("nan")
    a, b = np.flatnonzero(left), np.flatnonzero(right)
    return float(kernel[np.ix_(a, a)].mean() + kernel[np.ix_(b, b)].mean()
                 - 2.0 * kernel[np.ix_(a, b)].mean())


def _group_kernel_blocks(kernel: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    members = [np.flatnonzero(codes == i) for i in range(len(names))]
    counts = np.asarray([len(index) for index in members], dtype=np.int64)
    blocks = np.zeros((len(names), len(names)), dtype=float)
    for i, left in enumerate(members):
        for j in range(i, len(members)):
            value = float(kernel[np.ix_(left, members[j])].sum())
            blocks[i, j] = value
            blocks[j, i] = value
    return names, blocks, counts


def _bounded_group_sample(values: np.ndarray, groups: np.ndarray, cap: int,
                          seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Deterministically subsample windows while retaining take groups where possible."""
    x = np.asarray(values)
    g = np.asarray(groups).astype(str)
    if len(x) <= cap:
        return x, g, False
    rng = np.random.default_rng(seed)
    names = np.unique(g)
    if len(names) > cap:
        names = np.sort(rng.choice(names, cap, replace=False))
    per_group = max(1, cap // len(names))
    selected = []
    for name in names:
        members = np.flatnonzero(g == name)
        selected.extend(rng.choice(members, min(per_group, len(members)), replace=False).tolist())
    if len(selected) > cap:
        selected = sorted(rng.choice(selected, cap, replace=False).tolist())
    else:
        selected.sort()
    index = np.asarray(selected, dtype=int)
    return x[index], g[index], True


def _mmd_integer_blocks(blocks: np.ndarray) -> tuple[np.ndarray, int]:
    """Represent finite binary64 block entries exactly in common dyadic units."""
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 2 or blocks.shape[0] != blocks.shape[1] or not np.isfinite(blocks).all():
        raise ValueError("MMD blocks must be a finite square matrix")
    ratios = [float(value).as_integer_ratio() for value in blocks.flat]
    # Every denominator is a power of two, so the largest is a common multiple.
    divisor = max((denominator for _, denominator in ratios), default=1)
    integers = np.array([numerator * (divisor // denominator)
                         for numerator, denominator in ratios], dtype=object)
    return integers.reshape(blocks.shape), divisor


def _mmd_from_group_blocks(blocks: np.ndarray, counts: np.ndarray,
                           left: Iterable[int], right: Iterable[int], *,
                           exact: bool = False,
                           _scaled_blocks: tuple[np.ndarray, int] | None = None) -> float | Fraction:
    """Biased MMD of the symmetrized stored kernel, accumulated exactly.

    Binary64 block entries become integers with a common power-of-two divisor;
    all sums, count weights and subtraction precede the only rounding (to the
    returned float). ``exact=True`` retains the rational for permutation ranks.
    ``_scaled_blocks`` is a call-local precomputation for repeated partitions.

    Conditional on the same stored blocks/counts, the result is invariant to
    index order, simultaneous block/count reindexing, transpose and swapping the
    two sides. Repeated indices retain bootstrap multiplicity. Using both cross
    orientations also handles roundoff asymmetry in tiled kernel construction.
    This does not assert row-order invariance of that floating-point construction.
    """
    left = np.asarray(left if isinstance(left, np.ndarray) else list(left), int)
    right = np.asarray(right if isinstance(right, np.ndarray) else list(right), int)
    counts = np.asarray(counts)
    integers, divisor = _mmd_integer_blocks(blocks) if _scaled_blocks is None else _scaled_blocks
    if (counts.shape != (len(integers),) or counts.dtype.kind not in "iu"
            or np.any(counts < 0)):
        raise ValueError("MMD counts must be nonnegative integer group sizes")
    n_left, n_right = sum(map(int, counts[left])), sum(map(int, counts[right]))
    if not n_left or not n_right:
        if exact:
            raise ValueError("MMD needs nonempty sides")
        return float("nan")
    xx = int(integers[np.ix_(left, left)].sum())
    yy = int(integers[np.ix_(right, right)].sum())
    xy = int(integers[np.ix_(left, right)].sum())
    yx = int(integers[np.ix_(right, left)].sum())
    result = Fraction(xx*n_right**2 + yy*n_left**2 - (xy+yx)*n_left*n_right,
                      divisor*n_left**2*n_right**2)
    return result if exact else float(result)


def mmd_permutation_test(x: np.ndarray, y: np.ndarray, *, groups_x: Iterable | None = None,
                         groups_y: Iterable | None = None, permutations: int = 999,
                         seed: int = 0, center: np.ndarray | None = None,
                         scale: np.ndarray | None = None, gamma: float | None = None) -> dict:
    """Biased RBF-MMD with whole-group randomization and inclusive exact ties.

    Compare unrounded rational statistics, with p=(1+#(null >= observed))/(B+1).
    No epsilon, clipping, randomized ties or rounding-dependent rank is used.
    Canonical group IDs bind the RNG index space; sorting selected indices is
    unnecessary because the reduction is exact. Renaming IDs can change the
    finite Monte Carlo partitions; the invariant concerns the same partitions.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y)
            or not x.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("MMD inputs must be nonempty matrices with the same feature width")
    if isinstance(permutations, (bool, np.bool_)) or not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    for name, value in (("center", center), ("scale", scale)):
        if value is not None:
            value = np.asarray(value, float)
            if value.shape != (x.shape[1],) or not np.isfinite(value).all() or (name == "scale" and np.any(value <= 0)):
                raise ValueError(f"MMD {name} must be finite, feature-sized and scale positive")
    if gamma is not None and (not np.isfinite(gamma) or gamma <= 0):
        raise ValueError("MMD gamma must be finite and positive")
    original_windows = (len(x), len(y))
    gx = np.arange(len(x)).astype(str) if groups_x is None else np.asarray(list(groups_x)).astype(str)
    gy = np.arange(len(y)).astype(str) if groups_y is None else np.asarray(list(groups_y)).astype(str)
    if gx.shape != (len(x),) or gy.shape != (len(y),):
        raise ValueError("one group ID is required per feature row")
    x, gx, sampled_x = _bounded_group_sample(x, gx, 1024, seed)
    y, gy, sampled_y = _bounded_group_sample(y, gy, 1024, seed + 1)
    standardized, gamma, _, _ = _kernel_parameters(x, y, seed, center, scale, gamma)
    if not np.isfinite(standardized).all() or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("MMD kernel parameters are nonfinite or degenerate")
    all_groups = np.r_[np.char.add("x:", gx), np.char.add("y:", gy)]
    unique_groups, blocks, counts = _group_rbf(standardized, all_groups, gamma)
    order = np.argsort(unique_groups)
    unique_groups, blocks, counts = unique_groups[order], blocks[np.ix_(order, order)], counts[order]
    scaled = _mmd_integer_blocks(blocks)
    name_to_code = {name: i for i, name in enumerate(unique_groups)}
    observed_left = [name_to_code[f"x:{name}"] for name in np.unique(gx)]
    observed_right = [name_to_code[f"y:{name}"] for name in np.unique(gy)]
    obs = _mmd_from_group_blocks(blocks, counts, observed_left, observed_right,
                                 exact=True, _scaled_blocks=scaled)
    left_group_count = len(observed_left)
    rng = np.random.default_rng(seed)
    exceedances = 0
    right_mask = np.ones(len(unique_groups), dtype=bool)
    for _ in range(permutations):
        selected = rng.choice(len(unique_groups), left_group_count, replace=False)
        right_mask[:] = True
        right_mask[selected] = False
        right = np.flatnonzero(right_mask)
        value = _mmd_from_group_blocks(blocks, counts, selected, right,
                                       exact=True, _scaled_blocks=scaled)
        exceedances += value >= obs
    p = (1 + exceedances) / (permutations + 1)
    return {"mmd2": float(obs), "permutation_p": float(p), "permutations": int(permutations),
            "statistic_policy": "exact-dyadic-symmetrized-blocks/1",
            "pvalue_policy": "inclusive-exact-rational-plus-one/1",
            "kernel": "rbf-median-distance", "gamma": float(gamma),
            "n_windows": {"left": int(len(x)), "right": int(len(y))},
            "n_windows_input": {"left": int(original_windows[0]), "right": int(original_windows[1])},
            "window_subsampled": bool(sampled_x or sampled_y),
            "n_groups": {"left": int(len(np.unique(gx))), "right": int(len(np.unique(gy)))}}


def _partition_list(groups: list[str], cap: int, seed: int) -> tuple[list[tuple[str, ...]], int]:
    n = len(groups)
    side = n // 2
    total = comb(n, side)
    if n % 2 == 0:
        total //= 2
    if total <= cap:
        result = []
        for part in combinations(groups, side):
            if n % 2 == 0 and groups[0] not in part:
                continue  # do not include both a split and its complement
            result.append(tuple(part))
        return result, total
    rng = np.random.default_rng(seed)
    chosen: set[tuple[str, ...]] = set()
    while len(chosen) < cap:
        part = tuple(sorted(rng.choice(groups, side, replace=False).tolist()))
        if n % 2 == 0 and groups[0] not in part:
            other = tuple(group for group in groups if group not in part)
            part = tuple(sorted(other))
        chosen.add(part)
    return sorted(chosen), total


def real_vs_real_null(values: np.ndarray, groups: Iterable, *, cap: int = 200,
                      bootstrap_replicates: int = 500, seed: int = 0,
                      n_bands: int | None = None, center: np.ndarray | None = None,
                      scale: np.ndarray | None = None, kernel_gamma: float | None = None) -> dict:
    """Real-vs-real MMD/SMD envelope from distinct take-group resplits.

    With fewer than ten distinct group splits, resample whole groups and mark the result
    ``low-power`` as required by the programme.
    """
    x = np.asarray(values, float)
    if x.ndim != 2 or not len(x):
        raise ValueError("values must be a nonempty feature matrix")
    group_values = np.asarray(list(groups) if not isinstance(groups, np.ndarray) else groups).astype(str)
    if group_values.shape != (len(x),):
        raise ValueError("one group ID is required per feature row")
    names = sorted(set(group_values.tolist()))
    if len(names) < 2:
        return {"label": "low-power", "method": "unavailable", "distinct_splits": 0,
                "group_count": int(len(names)), "resplits_used": 0,
                "mmd2_q95": None, "band_smd_q95": [],
                "null_mmd2": [], "null_band_smd": []}
    x, group_values, window_subsampled = _bounded_group_sample(x, group_values, 1024, seed)
    names = sorted(set(group_values.tolist()))
    parts, total = _partition_list(names, cap, seed)
    rng = np.random.default_rng(seed)
    low_power = total < 10
    if low_power:
        bootstrap_parts = []
        for _ in range(max(bootstrap_replicates, 1)):
            left_n = max(1, len(names)//2)
            left = list(rng.choice(names, left_n, replace=True).tolist())
            right_n = max(1, len(names)-left_n)
            right = list(rng.choice(names, right_n, replace=True).tolist())
            bootstrap_parts.append((left, right))
    else:
        bootstrap_parts = [(list(part), [name for name in names if name not in set(part)]) for part in parts]
    mmd_values, smd_values = [], []
    by_group = {name: x[group_values == name] for name in names}
    standardized, gamma, _, _ = _kernel_parameters(x, x[:0], seed, center, scale, kernel_gamma)
    group_names, blocks, counts = _group_rbf(standardized, group_values, gamma)
    name_to_code = {name: i for i, name in enumerate(group_names)}
    for left_names, right_names in bootstrap_parts:
        left = np.concatenate([by_group[name] for name in left_names])
        right = np.concatenate([by_group[name] for name in right_names])
        left_codes = [name_to_code[name] for name in left_names]
        right_codes = [name_to_code[name] for name in right_names]
        mmd_values.append(_mmd_from_group_blocks(blocks, counts, left_codes, right_codes))
        smd = np.abs(_smd(left, right))
        if n_bands is not None:
            smd_values.append(_band_scores(smd, n_bands))
        else:
            smd_values.append(smd)
    return {
        "label": "low-power" if low_power else "adequate-split-count",
        "method": "group-bootstrap" if low_power else "take-group-resplit",
        "distinct_splits": int(total), "resplits_used": int(len(bootstrap_parts)),
        "group_count": int(len(names)), "window_subsampled": bool(window_subsampled),
        "mmd2_q95": float(np.quantile(mmd_values, .95)),
        "band_smd_q95": np.quantile(np.asarray(smd_values), .95, axis=0).astype(float).tolist(),
        "null_mmd2": np.asarray(mmd_values, float).tolist(),
    }


def realism_report(reference: np.ndarray, rendered: np.ndarray, *, reference_groups: Iterable,
                   rendered_groups: Iterable, n_bands: int | None = None, permutations: int = 999,
                   seed: int = 0, resplit_cap: int = 200) -> dict:
    """Group-count-matched joint MMD/SMD test; real-only splits are descriptive.

    The joint permutation calibration controls the size of the *combined* MMD and
    90%-band-coverage rule, instead of comparing full samples to half-sample nulls.
    With fewer than ten reference halvings the decision remains inconclusive.
    """
    from .calibration import joint_permutation

    x, y = np.asarray(reference, float), np.asarray(rendered, float)
    gx, gy = np.asarray(list(reference_groups)).astype(str), np.asarray(list(rendered_groups)).astype(str)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y):
        raise ValueError("features must be nonempty matrices with equal widths")
    if gx.shape != (len(x),) or gy.shape != (len(y),):
        raise ValueError("one group ID is required per feature row")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features must be finite")
    original = (len(x), len(y))
    x, gx, sx = _bounded_group_sample(x, gx, 1024, seed)
    y, gy, sy = _bounded_group_sample(y, gy, 1024, seed+1)
    standardized, gamma, center, scale = _kernel_parameters(x, y, seed)
    decision, mmd, smd, scores = joint_permutation(x, y, gx, gy, standardized, gamma,
                                                 n_bands, permutations, seed)
    null = real_vs_real_null(x, gx, cap=resplit_cap, seed=seed, n_bands=n_bands,
                             center=center, scale=scale, kernel_gamma=gamma)
    null['role'] = 'diagnostic-only; half-split group counts differ from the decision'
    mmd.update(n_windows_input={'left': original[0], 'right': original[1]}, window_subsampled=bool(sx or sy))
    envelope = np.asarray(decision['band_smd_envelope'])
    coverage = float(np.mean(scores <= envelope))
    low_power = null['label'] == 'low-power' or len(np.unique(gy)) < 5
    status = 'inconclusive' if low_power else ('fail' if decision['permutation_p'] <= .05 else 'pass')
    return {
        'status': status, 'mmd': mmd, 'decision_null': decision,
        'real_vs_real': {k: v for k, v in null.items() if k != 'null_mmd2'},
        'band_smd': {'feature_smd': smd.tolist(), 'absolute_band_smd': scores.tolist(),
                     'within_decision_envelope_fraction': coverage,
                     'within_real_vs_real_envelope_fraction': (float(np.mean(scores <= np.asarray(null['band_smd_q95'])))
                                                               if null['band_smd_q95'] else None),
                     'bands_within_envelope': int(np.count_nonzero(scores <= envelope)),
                     'band_count': len(scores)},
    }


def provenance_shares(rows: Iterable[dict]) -> dict:
    """Count and normalize per-class source, route, and capture-chain provenance."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        label = str(row.get("label", "unknown"))
        dry = row.get("dry_source") or {}
        provenance = dry.get("provenance") or row.get("provenance") or {}
        route = str(provenance.get("route", "unspecified"))
        source = "rendered" if dry else str(row.get("source_kind", "real/background"))
        receiver = row.get("receiver_augmentation") or {}
        chain = str(row.get("training_chain") or receiver.get("profile") or receiver.get("method")
                    or receiver.get("type") or ("unspecified-render-receiver" if dry else "already-recorded; chain unspecified"))
        for dimension, value in (("source", source), ("route", route), ("chain", chain)):
            bucket = counts.setdefault(label, {}).setdefault(dimension, {})
            bucket[value] = bucket.get(value, 0) + 1
    result = {}
    for label, dimensions in counts.items():
        result[label] = {}
        for dimension, bucket in dimensions.items():
            total = sum(bucket.values())
            result[label][dimension] = {
                key: {"count": count, "share": float(count / total)}
                for key, count in sorted(bucket.items())
            }
    return result


def score_neutral_probes(model, probes: dict[str, np.ndarray], *,
                         positive_path: Callable[[np.ndarray], np.ndarray],
                         negative_path: Callable[[np.ndarray], np.ndarray],
                         data_kind: str = "proxy") -> dict:
    """Run fixed neutral audio through supplied positive and negative render paths.

    This is a diagnostic, not a significance test: only three fixed probes are used. Each path
    is scored by the exact ``edge.detect`` runtime and its frozen model thresholds.
    """
    from sensorsim import edge

    fs = int(model.meta["features"]["fs"])
    result = {"status": "diagnostic-only", "data_kind": data_kind,
              "detector_thresholds": model.meta["detector"], "probes": {}}
    for name, audio in sorted(probes.items()):
        row = {}
        for path_name, path in (("positive", positive_path), ("negative", negative_path)):
            processed = np.asarray(path(np.asarray(audio, np.float32)), dtype=np.float32)
            events = edge.detect(model, processed)
            row[path_name] = {
                "samples": int(processed.size), "duration_s": float(processed.size / fs),
                "peak_abs": float(np.max(np.abs(processed))) if processed.size else 0.0,
                "events_by_class": {label: int(sum(event["label"] == label for event in events))
                                     for label in model.classes},
            }
        result["probes"][name] = row
    return result
