import numpy as np

from sensorsim.evaluation.c2st import auc_score, compare_samples, decision


def _grouped_draw(seed, shift):
    rng = np.random.default_rng(seed)
    x, groups = [], []
    for i in range(15):
        x.extend(rng.normal(loc=shift, size=(4, 8)))
        groups.extend([f"take-{i:02d}"]*4)
    return np.asarray(x), np.asarray(groups)


def test_numpy_grouped_logistic_detects_a_large_shift():
    x0, g0 = _grouped_draw(3, -1.0)
    x1, g1 = _grouped_draw(4, 1.0)
    g0 = np.char.add("reference-", g0)
    g1 = np.char.add("rendered-", g1)
    result = compare_samples(x0, x1, g0, g1, bootstrap_replicates=250,
                             bootstrap_seeds=(10, 11, 12), fold_seed=13)["result"]
    assert result["point_auc"] > .9
    assert result["lower_95"] > .7
    assert result["outcome"] == "fail"  # visibly distinguishable is a C2ST failure


def test_auc_handles_ties_and_decision_rule_boundaries():
    assert auc_score(np.asarray([0, 1]), np.asarray([.5, .5])) == .5
    assert decision(.60, .50, .70) == "pass"
    assert decision(.75, .60, .85) == "fail"
    assert decision(.62, .55, .72) == "inconclusive"


def test_grouped_cv_keeps_parent_groups_out_of_their_training_fold():
    x0, g0 = _grouped_draw(33, 0.0)
    x1, g1 = _grouped_draw(34, 0.0)
    g0 = np.char.add("reference-", g0)
    g1 = np.char.add("rendered-", g1)
    result = compare_samples(x0, x1, g0, g1, bootstrap_replicates=100,
                             bootstrap_seeds=(2, 3, 4), fold_seed=5)["result"]
    assert result["folds"] == 5
    assert result["n_groups"] == {"class_0": 15, "class_1": 15}
