"""Disclosed mathematical controls, not population or deployment acceptance."""

from fractions import Fraction
from itertools import combinations, permutations
import math

import numpy as np
import pytest

from sensorsim.evaluation import realism as r


def rational_mmd(blocks, counts, left, right):
    """Independent signed-measure quadratic form, including repeated groups."""
    nl = sum(int(counts[i]) for i in left)
    nr = sum(int(counts[i]) for i in right)
    weights = [Fraction(left.count(i), nl) - Fraction(right.count(i), nr)
               for i in range(len(counts))]
    return sum(Fraction(float(blocks[i, j])) * weights[i] * weights[j]
               for i in range(len(counts)) for j in range(len(counts)))


def fixture(kind):
    counts = np.array([1, 3, 2, 4, 1, 2])
    points = np.array([0., .25, .25, 1., 2., 4.])
    if kind == "constant":
        kernel = np.ones((6, 6))
    elif kind == "near-cancellation":
        kernel = np.array([[math.exp(-1e-16 * (x-y)**2) for y in points]
                           for x in points])
    else:
        kernel = np.array([[math.exp(-.5 * (x-y)**2) for y in points]
                           for x in points])
    return kernel * counts[:, None] * counts[None, :], counts


@pytest.mark.parametrize("kind", ["unequal-duplicates", "constant", "near-cancellation"])
def test_all_small_partitions_match_signed_rational_oracle(kind):
    blocks, counts = fixture(kind)
    for size in (1, 2, 3):
        for part in combinations(range(6), size):
            left, right = list(part), [i for i in range(6) if i not in part]
            expected = float(rational_mmd(blocks, counts, left, right))
            assert r._mmd_from_group_blocks(blocks, counts, left, right) == expected
            assert r._mmd_from_group_blocks(blocks, counts, right, left) == expected
            assert r._mmd_from_group_blocks(blocks.T, counts, left, right) == expected


def test_order_reindex_transpose_complement_and_bootstrap_multiplicity():
    blocks, counts = fixture("unequal-duplicates")
    # Tile accumulation can make the two orientations differ by one ulp.
    blocks[0, 4] = np.nextafter(blocks[0, 4], 0.)
    left, right = [0, 1, 1, 2], [3, 4, 5]
    expected = float(rational_mmd(blocks, counts, left, right))
    for a in permutations(left):
        for b in permutations(right):
            assert r._mmd_from_group_blocks(blocks, counts, a, b) == expected
            assert r._mmd_from_group_blocks(blocks.T, counts, b, a) == expected
    order = np.array([4, 2, 0, 5, 3, 1])
    inverse = np.argsort(order)
    assert r._mmd_from_group_blocks(blocks[np.ix_(order, order)], counts[order],
                                    inverse[left], inverse[right]) == expected


def test_unrounded_rational_rank_distinguishes_sub_ulp_statistics():
    # Both display values round to 1; an epsilon or float-rank rule loses the order.
    tiny = 2.**-56
    blocks = np.array([[1., tiny, 0.], [tiny, 1., 0.], [0., 0., 0.]])
    counts = np.ones(3, int)
    a = r._mmd_from_group_blocks(blocks, counts, [0], [2], exact=True)
    b = r._mmd_from_group_blocks(blocks, counts, [0], [1, 2], exact=True)
    assert a == rational_mmd(blocks, counts, [0], [2])
    assert b == rational_mmd(blocks, counts, [0], [1, 2])
    # Independent near-tie pair with a single sub-ulp cross perturbation.
    changed = blocks.copy(); changed[0, 2] = tiny
    c = r._mmd_from_group_blocks(changed, counts, [0], [2], exact=True)
    assert float(a) == float(c) and c < a


def old_reduction(blocks, counts, left, right, **kwargs):
    """Designated order-dependent mutant, copied from the defect being repaired."""
    left, right = np.array(list(left)), np.array(list(right))
    nl, nr = int(counts[left].sum()), int(counts[right].sum())
    value = (float(blocks[np.ix_(left, left)].sum()) / nl**2
             + float(blocks[np.ix_(right, right)].sum()) / nr**2
             - 2 * float(blocks[np.ix_(left, right)].sum()) / (nl*nr))
    return Fraction(value) if kwargs.get("exact") else value


def test_order_dependent_mutant_is_reached_and_rejected(monkeypatch):
    blocks, counts = fixture("near-cancellation")
    expected = rational_mmd(blocks, counts, [0, 1, 2], [3, 4, 5])
    assert r._mmd_from_group_blocks(blocks, counts, [0, 1, 2], [3, 4, 5]) == float(expected)
    reached = []

    def mutant(*args, **kwargs):
        result = old_reduction(*args, **kwargs)
        reached.append(result)
        return result

    monkeypatch.setattr(r, "_mmd_from_group_blocks", mutant)
    values = [r._mmd_from_group_blocks(blocks, counts, a, [3, 4, 5])
              for a in permutations([0, 1, 2])]
    assert len(reached) == 6
    assert any(v != float(expected) for v in values)
    assert len(set(values)) > 1


def test_scalar_rbf_and_exhaustive_permutation_p_oracle(monkeypatch):
    x, y = np.array([[0.], [1.], [1.]]), np.array([[2.], [3.]])
    gx, gy = ["a", "b", "b"], ["c", "d"]
    # Direct scalar squared distances: no production kernel/reducer/oracle imports.
    rows = [0., 1., 1., 2., 3.]
    kernel = [[Fraction(math.exp(-.5 * (a-b)**2)) for b in rows] for a in rows]
    memberships = [[0], [1, 2], [3], [4]]

    def oracle(selected):
        a = [i for g in selected for i in memberships[g]]
        b = [i for g in range(4) if g not in selected for i in memberships[g]]
        return (sum(kernel[i][j] for i in a for j in a) / len(a)**2
                + sum(kernel[i][j] for i in b for j in b) / len(b)**2
                - 2 * sum(kernel[i][j] for i in a for j in b) / (len(a)*len(b)))

    parts = list(combinations(range(4), 2))
    visited = []

    class Draws:
        def choice(self, n, size, replace):
            assert (n, size, replace) == (4, 2, False)
            value = parts[len(visited)]; visited.append(value)
            return np.array(value[::-1])

    monkeypatch.setattr(np.random, "default_rng", lambda seed: Draws())
    result = r.mmd_permutation_test(x, y, groups_x=gx, groups_y=gy,
                                    center=[0.], scale=[1.], gamma=.5, permutations=6)
    obs = oracle((0, 1))
    assert visited == parts
    assert result["mmd2"] == pytest.approx(float(obs), abs=2e-15)
    assert result["permutation_p"] == (1 + sum(oracle(s) >= obs for s in parts)) / 7


def test_group_builder_order_and_transpose_do_not_change_draws_or_result(monkeypatch):
    rng = np.random.default_rng(713)
    x, y = rng.normal(size=(5, 3)), rng.normal(size=(6, 3))
    options = dict(groups_x=["a", "b", "b", "c", "c"],
                   groups_y=["d", "d", "e", "f", "f", "f"], permutations=79, seed=101)
    correct = r.mmd_permutation_test(x, y, **options)
    builder = r._group_rbf
    reached = []

    def shuffled(*args):
        names, blocks, counts = builder(*args)
        ix = np.arange(len(names))[::-1]
        reached.append(True)
        return names[ix], blocks.T[np.ix_(ix, ix)], counts[ix]

    monkeypatch.setattr(r, "_group_rbf", shuffled)
    assert r.mmd_permutation_test(x, y, **options) == correct
    assert reached == [True]


def test_constants_are_ties_and_strict_tail_mutant_fails(monkeypatch):
    import inspect
    options = dict(groups_x=["a", "a", "b"], groups_y=["c", "d"], permutations=39)
    x, y = np.ones((3, 2)), np.ones((2, 2))
    good = r.mmd_permutation_test(x, y, **options)
    assert good["mmd2"] == 0 and good["permutation_p"] == 1
    source = inspect.getsource(r.mmd_permutation_test)
    assert "value >= obs" in source
    namespace = dict(r.__dict__)
    exec(source.replace("value >= obs", "value > obs"), namespace)
    bad = namespace["mmd_permutation_test"](x, y, **options)
    assert bad["permutation_p"] == 1/40 and bad != good


@pytest.mark.parametrize("bad", ["empty", "width", "nan", "inf", "groups", "gamma", "scale", "permutations"])
def test_invalid_input_is_explicit(bad):
    x, y = np.ones((3, 2)), np.zeros((3, 2))
    opts = dict(permutations=3)
    if bad == "empty": x = x[:0]
    if bad == "width": x, y = x[:, :0], y[:, :0]
    if bad == "nan": x[0, 0] = np.nan
    if bad == "inf": y[0, 0] = np.inf
    if bad == "groups": opts["groups_x"] = ["only-one"]
    if bad == "gamma": opts["gamma"] = -1.
    if bad == "scale": opts["scale"] = [1., 0.]
    if bad == "permutations": opts["permutations"] = True
    with pytest.raises(ValueError):
        r.mmd_permutation_test(x, y, **opts)


@pytest.mark.parametrize("groups", [4, 6, 10, 15, 18, 20])
def test_p5_group_counts_reach_null_and_resolve_tiny_kernel_defect(groups):
    """Binary-atom numerical controls, not sensitivity to an acoustic defect."""
    delta = 2.**-26  # A cross-kernel value just two binary64 steps below one.
    x, y = np.zeros((groups, 1)), np.full((groups, 1), delta)
    opts = dict(permutations=1999, seed=713, center=[0.], scale=[1.], gamma=1.)
    bad = r.mmd_permutation_test(x, y, **opts)
    c = Fraction(math.exp(-delta**2))
    assert bad["mmd2"] == float(2*(1-c)) > 0
    # Central partitions of the same pooled atoms have the smallest discrepancy.
    left = np.r_[x[:groups//2], y[:groups-groups//2]]
    right = np.r_[x[groups//2:], y[groups-groups//2:]]
    good = r.mmd_permutation_test(left, right, **opts)
    assert good["permutation_p"] == 1
    # Count the exact randomization tail by hypergeometric atom memberships.
    rng = np.random.default_rng(713)
    extremes = 0
    for _ in range(1999):
        a = sum(int(i < groups) for i in rng.choice(2*groups, groups, replace=False))
        extremes += a in (0, groups)
    assert bad["permutation_p"] == (1+extremes)/2000
    if groups == 4:
        assert 12*bad["permutation_p"] > .05  # Honest resolution limitation.
    else:
        assert 12*bad["permutation_p"] <= .05


def test_subsampling_is_reported_and_empty_group_statistic_is_not_evidence():
    x = np.arange(1025, dtype=float)[:, None]
    result = r.mmd_permutation_test(x, np.array([[0.], [1.]]),
                                    groups_x=["one"]*len(x), permutations=1)
    assert result["window_subsampled"] is True
    assert result["n_windows_input"]["left"] == 1025
    assert result["n_windows"]["left"] == 1024
    assert math.isnan(r._mmd_from_group_blocks(np.eye(2), np.ones(2, int), [], [1]))
    with pytest.raises(ValueError, match="nonempty"):
        r._mmd_from_group_blocks(np.eye(2), np.ones(2, int), [], [1], exact=True)


def test_disclosed_tie_loss_reproduction_and_reached_production_mutant(monkeypatch):
    from tools.power_audit import procedural_factory

    seed = 2272642362
    bundles = procedural_factory(4, "frame_reversal", seed)
    x, y = [b["views"]["physical"][[i for i, row in enumerate(b["records"])
                                   if row["label"] == "kettle"]] for b in bundles]
    opts = dict(groups_x=np.arange(4), groups_y=np.arange(4), permutations=1999, seed=seed)
    correct = r.mmd_permutation_test(x, y, **opts)
    reached = []

    def mutant(*args, **kwargs):
        value = old_reduction(*args, **kwargs)
        reached.append(value)
        return value

    monkeypatch.setattr(r, "_mmd_from_group_blocks", mutant)
    bad = r.mmd_permutation_test(x, y, **opts)
    assert len(reached) == 2000
    assert correct["permutation_p"] == .028
    assert bad["permutation_p"] == .003
    assert min(1., 12*correct["permutation_p"]) > .05
    assert min(1., 12*bad["permutation_p"]) <= .05
