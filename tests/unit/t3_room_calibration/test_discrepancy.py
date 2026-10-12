"""Development fixtures only; no frozen acceptance data or receipts."""
import copy
import hashlib

import numpy as np
import pytest

from sensorsim.acoustics.room.calibration_discrepancy import (
    METRIC_POLICY, collect_render_predictions, convergence_check,
    decompose_uncertainty, late_block_update, predictive_products,
    refine_particles, weighted_quantiles,
)

KEYS = ("1000.t30_s", "1000.edt_s", "1000.c50_db")


def budget(repeats=None, measurement=.01, shared=.01):
    repeats = repeats or [dict(zip(KEYS, (.5, .5, 0.)))]
    return decompose_uncertainty(repeats, keys=KEYS,
        units=dict(zip(KEYS, ("s", "s", "dB"))),
        repeat_ids=[f"repeat-{i}" for i in range(len(repeats))],
        content_sha256=[hashlib.sha256(f"content-{i}".encode()).hexdigest() for i in range(len(repeats))],
        measurement_covariance=np.eye(3)*measurement, playback_covariance=np.eye(3)*.02,
        simulator_covariance=np.eye(3)*.03, discrepancy_covariance=np.eye(3)*shared,
        policy_id="development-gaussian/1", simulator_repeats=3)


def predictions():
    return [dict(zip(KEYS, (x, .5, 0.))) for x in (.25, .4, .5, .65, 1.)]


def render_rows(rows):
    return [dict(member_id=str(i), model_hash=f"model-{i}", metrics=p,
        rir_sha256=hashlib.sha256(f"waveform-{i}".encode()).hexdigest(),
        imports={"declared-development-control": {"sha256": "a"*64}}, renderer="development-control",
        seed=i, metric_policy=METRIC_POLICY, branch="physical-member-render") for i, p in enumerate(rows)]


def test_repeat_noise_shrinks_but_shared_components_do_not():
    one = budget()
    four = budget([dict(zip(KEYS, (.5, .5, 0.)))]*4)
    assert four["constant_repeat_coordinates"] == list(KEYS)
    np.testing.assert_allclose(four["covariance"]["measurement_mean"], np.eye(3)*.0025)
    for name in ("playback_shared", "discrepancy_shared", "simulator_mean"):
        assert four["covariance"][name] == one["covariance"][name]
    np.testing.assert_allclose(four["covariance"]["likelihood"], np.eye(3)*.0425)


def test_empirical_measurement_covariance_uses_log_seconds():
    out = decompose_uncertainty([{KEYS[0]: 1.}, {KEYS[0]: np.e**2}], keys=[KEYS[0]],
        units={KEYS[0]: "s"}, repeat_ids=["a", "b"], content_sha256=["a"*64, "b"*64],
        playback_covariance=[[0.]], simulator_covariance=[[0.]], discrepancy_covariance=[[0.]], policy_id="dev")
    assert out["covariance"]["measurement_single"][0][0] == pytest.approx(2.)
    assert out["covariance"]["measurement_mean"][0][0] == pytest.approx(1.)
    assert out["values"][KEYS[0]] == pytest.approx(np.e)


@pytest.mark.parametrize("change", ["duplicate_ids", "duplicate_content", "wrong_units", "one_repeat", "negative_covariance", "empty"])
def test_uncertainty_refuses_ambiguous_evidence(change):
    args = dict(repeats=[{KEYS[0]: .5}, {KEYS[0]: .6}], keys=[KEYS[0]], units={KEYS[0]: "s"},
        repeat_ids=["a", "b"], content_sha256=["a"*64, "b"*64], playback_covariance=[[0]],
        simulator_covariance=[[0]], discrepancy_covariance=[[.01]], policy_id="dev")
    if change == "duplicate_ids": args["repeat_ids"] = ["a", "a"]
    if change == "duplicate_content": args["content_sha256"] = ["a"*64, "a"*64]
    if change == "wrong_units": args["units"] = {KEYS[0]: "ms"}
    if change == "one_repeat":
        for k in ("repeats", "repeat_ids", "content_sha256"): args[k] = args[k][:1]
    if change == "negative_covariance": args["discrepancy_covariance"] = [[-.01]]
    if change == "empty": args["repeats"] = []
    with pytest.raises(ValueError): decompose_uncertainty(**args)


def test_late_weights_match_independent_scalar_calculation_and_cut_early_feedback():
    b = budget(); rows = predictions()
    out = late_block_update(rows, [.2]*5, b, early_tolerances={KEYS[1]: .2, KEYS[2]: 1.})
    # Independent scalar closed form: log-T residual; variance .01+.02+.03/3+.01.
    expected = np.exp(-np.log(np.array([.25, .4, .5, .65, 1.])/.5)**2/(2*.05))
    expected /= expected.sum()
    np.testing.assert_allclose(out["weights"], expected, rtol=1e-13)
    assert out["early_status"] == "compatible"
    mutated = copy.deepcopy(rows)
    for row in mutated: row[KEYS[2]] += 3.
    bad = late_block_update(mutated, [.2]*5, b, early_tolerances={KEYS[1]: .2, KEYS[2]: 1.})
    assert bad["early_status"] == "mismatch"
    assert bad["early_residuals"][KEYS[2]]["median"] == 3.
    assert bad["weights"] == out["weights"]
    assert not bad["training_promotion"]


def test_correlated_late_block_matches_hand_inverted_two_by_two():
    keys = ["500.t30_s", "1000.t30_s"]
    b = decompose_uncertainty([dict(zip(keys, (1., 1.)))], keys=keys,
        units=dict.fromkeys(keys, "s"), repeat_ids=["a"], content_sha256=["a"*64],
        measurement_covariance=[[1., .5], [.5, 1.]], playback_covariance=np.zeros((2,2)),
        simulator_covariance=np.zeros((2,2)), discrepancy_covariance=np.zeros((2,2)), policy_id="dev")
    rows = [dict(zip(keys, (np.e, np.e))), dict(zip(keys, (np.e, 1/np.e)))]
    result = late_block_update(rows, [.5, .5], b, early_tolerances={})
    # inverse covariance = [[4/3,-2/3],[-2/3,4/3]]; quadratic 4/3 vs 4.
    expected = np.exp(np.array([-2/3, -2.])); expected /= expected.sum()
    np.testing.assert_allclose(result["weights"], expected)
    assert result["early_status"] == "unqualified"


def test_missing_late_and_singular_covariance_abstain_without_filtering():
    rows = predictions(); del rows[0][KEYS[0]]
    out = late_block_update(rows, [.2]*5, budget(), early_tolerances={})
    assert out["status"] == "unqualified" and out["weights"] == [.2]*5
    b = decompose_uncertainty([{KEYS[0]: .5}]*2, keys=[KEYS[0]], units={KEYS[0]: "s"},
        repeat_ids=["a", "b"], content_sha256=["a"*64, "b"*64], playback_covariance=[[0]],
        simulator_covariance=[[0]], discrepancy_covariance=[[0]], policy_id="dev")
    assert "singular" in late_block_update(predictions(), [.2]*5, b, early_tolerances={})["reasons"][0]


def test_changed_budget_digest_and_nonfinite_weights_rejected():
    b = budget(); b["values"][KEYS[0]] = .01
    with pytest.raises(ValueError, match="changed"): late_block_update(predictions(), [.2]*5, b, early_tolerances={})
    with pytest.raises(ValueError): weighted_quantiles([1,2], [np.nan, np.nan])
    assert weighted_quantiles([2,2], [.3,.7]) == [2.,2.,2.]


def test_importance_correction_recovers_discrete_target_not_proposal():
    def proposal(rng, n, previous):
        assert n == 4
        return dict(particles=[[0.],[0.],[0.],[1.]], log_proposal=np.log([.75,.75,.75,.25]),
                    ancestor_ids=[None]*4, proposal_id="dev-discrete")
    out = refine_particles(propose=proposal, log_prior=lambda x: np.log(np.full(len(x),.5)),
        log_likelihood=lambda x: np.zeros(len(x)), seed=4, initial_count=4, maximum_count=4)
    np.testing.assert_allclose(out["weights"], [1/6,1/6,1/6,.5])
    assert sum(w*x[0] for w,x in zip(out["weights"],out["particles"])) == pytest.approx(.5)
    assert out["unique_particles"] == 2
    assert out["convergence"] == "not checked"
    # Designated bad oracle (forgetting q) returns .25 rather than the .5 target.
    assert np.mean([0.,0.,0.,1.]) != pytest.approx(.5)


def test_refinement_retains_budget_exhaustion_and_ancestry():
    def proposal(rng, n, previous):
        return dict(particles=np.arange(n)[:,None], log_proposal=np.full(n,-np.log(n)),
                    ancestor_ids=[None]*n, proposal_id=f"dev-{n}")
    out = refine_particles(propose=proposal, log_prior=lambda x: np.zeros(len(x)),
        log_likelihood=lambda x: -1000*x[:,0], seed=3, initial_count=8, maximum_count=16)
    assert [r["count"] for r in out["attempts"]] == [8,16]
    assert out["status"] == "budget-exhausted"
    assert out["effective_members"] == 1.
    assert len(out["log_proposal"]) == 16


@pytest.mark.parametrize("bad", ["missing", "nan_q", "no_support", "ancestor"])
def test_bad_refinement_hooks_fail_explicitly(bad):
    def proposal(rng, n, prev):
        return dict(particles=np.zeros((n,1)), log_proposal=[np.nan if bad == "nan_q" else 0.]*n,
                    ancestor_ids=[1 if bad == "ancestor" else None]*n, proposal_id="dev")
    with pytest.raises(ValueError):
        refine_particles(propose=None if bad == "missing" else proposal,
            log_prior=lambda x: np.full(len(x), -np.inf if bad == "no_support" else 0.),
            log_likelihood=lambda x: np.zeros(len(x)), seed=1, initial_count=4, maximum_count=4)


def test_convergence_requires_independent_runs_and_reference():
    q = {KEYS[0]: [.4,.5,.6]}
    runs = [dict(seed=s,status="ess-sufficient",quantiles=q) for s in (1,2)]
    assert convergence_check(runs,q,{KEYS[0]:.5},reference_converged=True)["status"] == "pass"
    assert convergence_check(runs,q,{KEYS[0]:.5},reference_converged=False)["status"] == "inconclusive"
    runs[1] = dict(runs[1],quantiles={KEYS[0]:[.5,.6,.7]})
    assert convergence_check(runs,q,{KEYS[0]:.5},reference_converged=True)["status"] == "fail"
    runs[1]["seed"] = 1
    assert convergence_check(runs,q,{KEYS[0]:.5},reference_converged=True)["status"] == "inconclusive"


def test_three_products_do_not_turn_error_bars_into_renders():
    params = predictions(); actual = [dict(row, **{KEYS[2]: 3.}) for row in params]
    product = predictive_products(params,render_rows(actual),[.2]*5,
        member_ids=list(map(str,range(5))),uncertainty=budget(shared=9.))
    assert product["parameter_only"]["intervals"][KEYS[2]] == [0.,0.,0.]
    assert product["actual_render"]["intervals"][KEYS[2]] == [3.,3.,3.]
    assert product["discrepancy_inclusive"]["intervals"][KEYS[2]][0] < 0.
    assert not product["training_promotion"]
    assert product["predictive_additive_covariance"][0][0] == 9.


def test_prediction_missing_mass_duplicate_content_and_hook_visible():
    rows = predictions(); actual = render_rows(rows)
    actual[0]["metrics"] = {KEYS[1]:.5,KEYS[2]:0.}
    actual[1]["rir_sha256"] = actual[0]["rir_sha256"]
    out = predictive_products(rows,actual,[.2]*5,member_ids=list(map(str,range(5))),uncertainty=budget())
    assert KEYS[0] not in out["actual_render"]["intervals"]
    assert out["actual_render"]["availability"][KEYS[0]]["predicted_mass"] == pytest.approx(.8)
    assert out["duplicate_render_content"] == 1
    out = predictive_products(rows,None,[.2]*5,member_ids=list(map(str,range(5))),uncertainty=budget())
    assert out["actual_render"] is None and out["actual_render_reasons"]
    actual[0]["metric_policy"] = "milliseconds/1"
    with pytest.raises(ValueError):
        predictive_products(rows,actual,[.2]*5,member_ids=list(map(str,range(5))),uncertainty=budget())


def test_actual_renderer_called_and_output_bound_to_member():
    from sensorsim.acoustics.room.prior import RoomSurvey, build_prior
    survey = RoomSurvey.model_validate(dict(room=dict(length_x=4.,width_y=3.,height_z=2.5),
        microphone=dict(x=1.,y=1.,z=1.), loudspeaker=dict(x=2.,y=2.,z=1.), prior=dict(count=2,seed=300)))
    ensemble = build_prior(survey)
    records = collect_render_predictions(ensemble,[2.,2.,1.],[1.,1.,1.],16000,seed=200,duration_s=.8)
    assert len(records) == 2
    assert records[0]["model_hash"] == ensemble.members[0].model_hash
    assert records[0]["rir_sha256"] != records[1]["rir_sha256"]
    assert records[0]["metrics"][KEYS[2]] != records[1]["metrics"][KEYS[2]]
    assert records[0]["branch"] == "physical-member-render"
    assert all(len(v["sha256"]) == 64 for v in records[0]["imports"].values())


def test_scene_state_good_bad_and_missing_hooks():
    from sensorsim.acoustics.room.calibration_discrepancy import audit_scene_states
    row = dict(scene_id="scene-1",ensemble_hash="e",ensemble_member_hash="m",installation_id="i",receiver_xyz_m=[1.,1.,1.])
    assert audit_scene_states([row,row.copy()])["status"] == "pass"
    bad = dict(row,receiver_xyz_m=[1.,1.,1.1])
    assert audit_scene_states([row,bad])["status"] == "fail"
    assert audit_scene_states([{}])["status"] == "unqualified"
    assert audit_scene_states([])["status"] == "unqualified"


def test_adapter_preserves_lineage_blocks_promotion_and_reused_geometry():
    from sensorsim.acoustics.room.calibration_discrepancy import update_room_ensemble, require_discrepancy_review
    from sensorsim.acoustics.room.prior import RoomSurvey, build_prior, CalibrationObservation
    survey = RoomSurvey.model_validate(dict(room=dict(length_x=4.,width_y=3.,height_z=2.5),
        microphone=dict(x=1.,y=1.,z=1.),loudspeaker=dict(x=2.,y=2.,z=1.),prior=dict(count=5,seed=300)))
    ensemble = build_prior(survey)
    obs = CalibrationObservation(kind="known-source",source_group="dev",input_sha256="a"*64,
        source_xyz_m=(2.,2.,1.),receiver_xyz_m=(1.,1.,1.),values=budget()["values"],
        uncertainty=dict.fromkeys(KEYS,.1),qualified=True,metadata={"lineage":{"use":"internal-validation"}})
    result = update_room_ensemble(ensemble,obs,predictions(),uncertainty=budget(),early_tolerances={KEYS[1]:.2,KEYS[2]:1.})
    assert result.evidence[-1]["accepted"]
    assert result.evidence[-1]["observation"]["metadata"]["lineage"] == obs.metadata["lineage"]
    again = update_room_ensemble(result,obs,predictions(),uncertainty=budget(),early_tolerances={KEYS[1]:.2})
    assert not again.evidence[-1]["accepted"]
    assert again.weights == result.weights
    with pytest.raises(ValueError,match="P4"): require_discrepancy_review(result)


def test_adaptive_normal_proposal_uses_density_and_fresh_batches():
    from scipy.stats import norm
    def proposal(rng,n,previous):
        mean,sd=(0.,2.) if previous is None else (.6,.12)
        x=rng.normal(mean,sd,size=(n,1))
        return dict(particles=x,log_proposal=norm.logpdf(x[:,0],loc=mean,scale=sd),
                    ancestor_ids=[None]*n,proposal_id=f"normal-{mean}-{sd}")
    out=refine_particles(propose=proposal,log_prior=lambda x:norm.logpdf(x[:,0],loc=0.,scale=2.),
        log_likelihood=lambda x:norm.logpdf(.6,loc=x[:,0],scale=.12),seed=4001)
    assert [r["count"] for r in out["attempts"]]==[256,512]
    assert out["status"]=="ess-sufficient"
    mean=sum(x[0]*w for x,w in zip(out["particles"],out["weights"]))
    assert mean==pytest.approx(.6*4/(4+.12**2),abs=.02)
    assert out["attempts"][0]["proposal_id"]!=out["attempts"][1]["proposal_id"]
    assert len(out["particles"])==512  # not a naively pooled 768-particle mixture
