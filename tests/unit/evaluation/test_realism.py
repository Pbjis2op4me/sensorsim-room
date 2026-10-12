from math import comb

import numpy as np

from sensorsim.evaluation.realism import (
    _partition_list,
    band_containing_hz,
    band_statistics,
    extract_audio_features,
    mmd_permutation_test,
    real_vs_real_null,
    realism_report,
)
from tests.unit.evaluation._fixture import save_edge_model, tone


def test_device_window_features_are_120d_and_pool_is_96d(tmp_path):
    model = save_edge_model(tmp_path/"model.npz")
    features = extract_audio_features(tone(), model)
    assert features["band_statistics"].shape == (5, 120)  # 1 s windows, 0.25 s hop
    assert features["embedding"].shape == (5, 96)
    assert features["starts_s"] == [0.0, .25, .5, .75, 1.0]
    assert band_containing_hz(model.features, 3000) in range(40)


def test_band_statistics_are_mean_std_max_per_mel_band():
    x = np.asarray([[[[0., 1., 2.], [3., 3., 3.]]]], dtype=np.float32)
    out = band_statistics(x)
    np.testing.assert_allclose(out, [[1., 3., np.sqrt(2/3), 0., 2., 3.]])


def test_mmd_uses_group_permutation_and_real_take_resplit_null():
    rng = np.random.default_rng(14)
    groups = np.repeat([f"t{i}" for i in range(5)], 4)
    reference = rng.normal(size=(20, 6))
    rendered = rng.normal(size=(20, 6))
    result = mmd_permutation_test(reference, rendered, groups_x=groups, groups_y=[f"s{g}" for g in groups],
                                  permutations=39, seed=11)
    assert result["permutations"] == 39
    assert 0 < result["permutation_p"] <= 1
    null = real_vs_real_null(reference, groups, cap=200, seed=5, n_bands=2)
    assert null["distinct_splits"] == 10
    assert null["label"] == "adequate-split-count"
    assert len(null["band_smd_q95"]) == 2


def test_real_null_is_low_power_below_ten_distinct_take_splits():
    values = np.arange(32, dtype=float).reshape(16, 2)
    groups = np.repeat(["a", "b", "c", "d"], 4)
    null = real_vs_real_null(values, groups, seed=1)
    assert null["distinct_splits"] == 3
    assert null["label"] == "low-power"
    assert null["method"] == "group-bootstrap"


def test_large_take_group_split_count_is_exact_and_capped():
    groups = [f"take-{i:03d}" for i in range(70)]
    splits, total = _partition_list(groups, cap=200, seed=21)
    assert total == comb(70, 35)//2
    assert len(splits) == 200
    assert len(set(splits)) == 200


def test_realism_report_keeps_mmd_kernel_fixed_across_observed_and_null():
    rng = np.random.default_rng(3)
    groups = np.repeat([f"g{i}" for i in range(5)], 2)
    x = rng.normal(size=(10, 6))
    y = x + rng.normal(scale=.01, size=x.shape)
    # Two bands times the three required statistics.
    report = realism_report(x, y, reference_groups=groups, rendered_groups=[f"r{g}" for g in groups],
                            n_bands=2, permutations=19, seed=4)
    assert report["status"] == "pass"
    assert report["decision_null"]["permutation_p"] > .05
    assert report["band_smd"]["within_decision_envelope_fraction"] >= .9
    assert report["mmd"]["gamma"] > 0


def test_realism_report_is_inconclusive_with_only_one_reference_take_group():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(4, 6))
    rendered = rng.normal(size=(5, 6))
    report = realism_report(reference, rendered, reference_groups=["single-take"]*4,
                            rendered_groups=[f"render-{i}" for i in range(5)],
                            n_bands=2, permutations=19, seed=9)
    assert report["status"] == "inconclusive"
    assert report["real_vs_real"]["label"] == "low-power"
    assert report["band_smd"]["within_real_vs_real_envelope_fraction"] is None
