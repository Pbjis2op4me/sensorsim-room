"""Modular room calibration and separate predictive products (model-based).

This module does not change a renderer or identify wall materials. Covariances
are declared in log(seconds) for decay times and dB for ratios. Independent
repeat noise may shrink; shared discrepancy and playback nuisance may not.
The late block is a cut update: early summaries diagnose inadequacy but cannot
reweight T30. No function here issues an acceptance receipt or training approval.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from pydantic import model_validator

from sensorsim.acoustics.room.prior import Contract, RoomEnsemble, RoomSurvey
from sensorsim.acoustics.room.model import RoomModel

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr

SCHEMA = "room-calibration-discrepancy/1"
METRIC_POLICY = "room-measured-metrics/2; drr/2"
BANDS = (500, 1000, 2000, 4000)


def digest(value) -> str:
    """Content identity; non-finite values are errors, never JSON NaN."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def metric_unit(key: str) -> str:
    band, sep, metric = key.partition(".")
    if not sep or not band.isdigit() or int(band) not in (125, 250, *BANDS, 8000):
        raise ValueError(f"unsupported acoustic metric: {key}")
    if metric in ("t30_s", "edt_s"):
        return "s"
    if metric in ("c50_db", "drr_db"):
        return "dB"
    raise ValueError(f"unsupported acoustic metric: {key}")


def _value(key, value):
    if value is None or not np.isfinite(value) or (metric_unit(key) == "s" and value <= 0):
        raise ValueError(f"invalid {key} value; positive seconds or finite dB required")
    return float(value)


def _transform(key, value):
    value = _value(key, value)
    return float(np.log(value)) if metric_unit(key) == "s" else value


def _inverse(key, value):
    result = float(np.exp(value)) if metric_unit(key) == "s" else float(value)
    if not np.isfinite(result) or (metric_unit(key) == "s" and result <= 0):
        raise ValueError("predictive value outside representable metric range")
    return result


def _covariance(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size, size) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size} by {size} covariance")
    if not np.allclose(array, array.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(array).min() < -1e-12*max(1., float(np.linalg.norm(array, ord=2))):
        raise ValueError(f"{name} must be positive semidefinite")
    return array


def _weights(weights, count):
    w = np.asarray(weights, dtype=float)
    if w.shape != (count,) or not count or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("one finite nonnegative weight per member required")
    if not np.isclose(w.sum(), 1., rtol=0, atol=1e-10):
        raise ValueError("weights must sum to one")
    return w / w.sum()


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def decompose_uncertainty(
    repeats: Sequence[Mapping[str, float]], *, keys: Sequence[str],
    units: Mapping[str, str], repeat_ids: Sequence[str], content_sha256: Sequence[str],
    playback_covariance, simulator_covariance, discrepancy_covariance,
    policy_id: str, measurement_covariance=None, discrepancy_bias=None,
    simulator_repeats: int = 1,
) -> dict:
    """Pool one geometry/state/chain in transformed metric space.

    The caller must group independent repeats of the *same* transfer. IDs and
    content hashes cannot be reused. With one repeat, a separately justified
    single-repeat measurement covariance is mandatory. Otherwise sample
    covariance is used unless an explicit covariance was supplied. Constant
    repeats are recorded, not silently regularized. All covariance arguments
    use the coordinate order in ``keys``; seconds-squared is incompatible.
    Playback, simulator and discrepancy matrices are explicit even when zero.
    ``simulator_covariance`` is for one seed; only its mean-estimation part is
    divided by the declared count of independent simulator seeds.
    """
    keys = tuple(keys)
    n, d = len(repeats), len(keys)
    if not n or not d or len(set(keys)) != d or not policy_id:
        raise ValueError("nonempty unique keys, repeats and policy_id required")
    if dict(units) != {k: metric_unit(k) for k in keys}:
        raise ValueError("units must exactly match the native metric keys (s or dB)")
    for identifiers, name in ((repeat_ids, "repeat IDs"), (content_sha256, "content hashes")):
        if len(identifiers) != n or any(not s for s in identifiers) or len(set(identifiers)) != n:
            raise ValueError(f"independent {name} required; duplicate content is not a repeat")
    if any(len(s) != 64 or any(c not in "0123456789abcdef" for c in s) for s in content_sha256):
        raise ValueError("content hashes must be lowercase SHA-256")
    samples = np.array([[_transform(k, row[k]) for k in keys] for row in repeats])
    if measurement_covariance is None:
        if n < 2:
            raise ValueError("one repeat requires declared measurement covariance")
        measurement = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
        measurement_source = "sample covariance of declared independent repeats"
    else:
        measurement = measurement_covariance
        measurement_source = "externally declared single-repeat covariance"
    matrices = {"measurement_single": _covariance(measurement, d, "measurement"),
                "playback_shared": _covariance(playback_covariance, d, "playback"),
                "simulator_single": _covariance(simulator_covariance, d, "simulator"),
                "discrepancy_shared": _covariance(discrepancy_covariance, d, "discrepancy")}
    matrices["measurement_mean"] = matrices["measurement_single"] / n
    matrices["simulator_mean"] = matrices["simulator_single"] / _positive_integer(simulator_repeats, "simulator_repeats")
    matrices["likelihood"] = sum(matrices[k] for k in (
        "measurement_mean", "playback_shared", "simulator_mean", "discrepancy_shared"))
    bias = np.zeros(d) if discrepancy_bias is None else np.asarray(discrepancy_bias, float)
    if bias.shape != (d,) or not np.isfinite(bias).all():
        raise ValueError("one finite transformed discrepancy bias per metric required")
    mean = samples.mean(axis=0)
    result = dict(schema=SCHEMA, keys=list(keys), units=dict(units),
        covariance_units=["log(s)" if units[k] == "s" else "dB" for k in keys],
        transformed_mean=mean.tolist(), values={k: _inverse(k, v) for k, v in zip(keys, mean)},
        covariance={k: v.tolist() for k, v in matrices.items()}, discrepancy_bias=bias.tolist(),
        repeat_ids=list(repeat_ids), content_sha256=list(content_sha256), independent_repeats=n,
        independent_simulator_seeds=simulator_repeats, measurement_source=measurement_source,
        constant_repeat_coordinates=[k for k, v in zip(keys, np.ptp(samples, axis=0)) if v == 0],
        policy_id=policy_id, result_label="model-based", covariance_assumption="declared components independent; cross-component covariance not identified")
    result["sha256"] = digest(result)
    return result


def _check_budget(budget):
    payload = {k: v for k, v in budget.items() if k != "sha256"}
    if budget.get("schema") != SCHEMA or digest(payload) != budget.get("sha256"):
        raise ValueError("changed or incompatible uncertainty record")
    keys = budget["keys"]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("nonempty unique uncertainty coordinates required")
    if budget["units"] != {k: metric_unit(k) for k in keys}:
        raise ValueError("incompatible uncertainty units")
    for name, value in budget["covariance"].items():
        _covariance(value, len(keys), name)
    return keys


def weighted_quantiles(values, weights, probabilities=(.05, .5, .95)) -> list[float]:
    """Inverse empirical CDF; a constant array has a point interval."""
    x = np.asarray(values, float)
    w = _weights(weights, len(x))
    q = np.asarray(probabilities, float)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("finite scalar values and probabilities in [0, 1] required")
    order = np.argsort(x, kind="stable")
    order = order[w[order] > 0]
    cumulative = np.cumsum(w[order]); cumulative[-1] = 1.
    return x[order[np.minimum(np.searchsorted(cumulative, q, side="left"), len(order)-1)]].tolist()


def late_block_update(predictions: Sequence[Mapping[str, float]], weights,
                      uncertainty: dict, *, early_tolerances: Mapping[str, float]) -> dict:
    """Sum the correlated Gaussian late likelihood; early data has no feedback.

    Early tolerances are relative errors for EDT and absolute dB for C50/DRR.
    They are declared diagnostic limits, not acceptance. A missing early hook,
    singular late covariance or missing positive-mass prediction abstains.
    Low ESS requests refinement; it never silently returns a point estimate.
    """
    keys = _check_budget(uncertainty)
    w = _weights(weights, len(predictions))
    late = [i for i, k in enumerate(keys) if k.endswith(".t30_s")]
    result = dict(schema=SCHEMA, weights=w.tolist(), status="unqualified", reasons=[],
        uncertainty_sha256=uncertainty["sha256"], policy_id=uncertainty["policy_id"],
        used_keys=[keys[i] for i in late], early_status="unqualified", early_residuals={},
        training_promotion=False, result_label="model-based")
    if not late:
        result["reasons"].append("no late T30 coordinates")
        return result
    cov = np.asarray(uncertainty["covariance"]["likelihood"])[np.ix_(late, late)]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        result["reasons"].append("singular late covariance; no jitter fallback")
        return result
    residuals = np.zeros((len(w), len(late)))
    observed = np.asarray(uncertainty["transformed_mean"])[late]
    bias = np.asarray(uncertainty["discrepancy_bias"])[late]
    try:
        for i in np.flatnonzero(w):
            residuals[i] = [_transform(keys[j], predictions[i][keys[j]]) for j in late]
            residuals[i] += bias - observed
    except (KeyError, ValueError):
        result["reasons"].append("missing or invalid positive-mass late prediction")
        return result
    whitened = np.linalg.solve(chol, residuals.T).T
    ll = -.5 * np.sum(whitened**2, axis=1)
    if not np.isfinite(ll[w > 0]).all():
        result["reasons"].append("nonfinite late likelihood")
        return result
    logw = np.full(len(w), -np.inf)
    logw[w > 0] = np.log(w[w > 0]) + ll[w > 0]
    posterior = np.exp(logw-logsumexp(logw))
    ess = float(1 / (posterior @ posterior))
    result.update(weights=posterior.tolist(), log_likelihood=ll.tolist(), effective_members=ess,
                  status="needs-refinement" if ess < .25*len(w) else "updated")
    if not early_tolerances:
        result["reasons"].append("early adequacy hook absent")
        return result
    missing = False
    mismatch = False
    for key, tolerance in early_tolerances.items():
        if key.endswith(".t30_s") or metric_unit(key) not in ("s", "dB") or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("positive early-metric tolerances required")
        try:
            actual = _value(key, uncertainty["values"][key])
            values = [_value(key, p[key]) for i, p in enumerate(predictions) if posterior[i] > 0]
            ww = posterior[posterior > 0]; ww /= ww.sum()
            lo, median, hi = weighted_quantiles(values, ww)
        except (KeyError, ValueError):
            missing = True
            result["early_residuals"][key] = {"status": "unavailable"}
            continue
        error = abs(median-actual) / actual if metric_unit(key) == "s" else abs(median-actual)
        bad = error > tolerance or not lo <= actual <= hi
        mismatch |= bad
        result["early_residuals"][key] = dict(observed=actual, median=median,
            actual_member_interval=[lo, hi], error=error, tolerance=float(tolerance),
            error_unit="relative" if metric_unit(key) == "s" else "dB", status="mismatch" if bad else "compatible")
    result["early_status"] = "mismatch" if mismatch else "unqualified" if missing else "compatible"
    return result


def refine_particles(*, propose: Callable, log_prior: Callable, log_likelihood: Callable,
                     seed: int, initial_count: int = 256, maximum_count: int = 2048,
                     ess_fraction: float = .25) -> dict:
    """Corrected importance batches, replacing (never naively pooling) proposals.

    ``propose(rng, count, previous)`` returns particles [N,D], log_proposal [N],
    proposal_id, ancestor_ids [N]. Densities must share one declared base measure;
    this API cannot verify a caller's density formula. Adaptation may use the
    previous batch, but each new batch is scored with its own normalized q.
    Persist every attempt, including failures to achieve the ESS threshold.
    ESS alone does not establish convergence: use independent runs/reference.
    """
    if not all(callable(f) for f in (propose, log_prior, log_likelihood)):
        raise ValueError("proposal, prior and likelihood hooks are required")
    count = _positive_integer(initial_count, "initial_count")
    cap = _positive_integer(maximum_count, "maximum_count")
    if cap < count or not np.isfinite(ess_fraction) or not 0 < ess_fraction <= 1:
        raise ValueError("invalid refinement budget")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng, previous, history = np.random.default_rng(seed), None, []
    while True:
        proposed = propose(rng, count, previous)
        x = np.asarray(proposed["particles"], float)
        if x.ndim != 2 or x.shape[0] != count or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError("proposal must return finite [count, dimension] particles")
        q = np.asarray(proposed["log_proposal"], float)
        lp, ll = np.asarray(log_prior(x), float), np.asarray(log_likelihood(x), float)
        if any(v.shape != (count,) for v in (q, lp, ll)) or not np.isfinite(q).all():
            raise ValueError("one finite proposal log density per particle required")
        if any(np.isnan(v).any() or np.isposinf(v).any() for v in (lp, ll)):
            raise ValueError("prior/likelihood must be finite or minus infinity")
        ancestors = proposed["ancestor_ids"]
        if len(ancestors) != count or not proposed.get("proposal_id"):
            raise ValueError("proposal identity and ancestry are required")
        if any(a is not None and (type(a) is not int or previous is None or not 0 <= a < len(previous["particles"])) for a in ancestors):
            raise ValueError("ancestor must index previous batch or be null")
        logw = lp + ll - q
        if not np.isfinite(logw).any():
            raise ValueError("proposal has no target support")
        w = np.exp(logw-logsumexp(logw)); ess = float(1/(w@w))
        previous = dict(particles=x.tolist(), weights=w.tolist(), log_prior=lp.tolist(),
            log_likelihood=ll.tolist(), log_proposal=q.tolist(), ancestor_ids=list(ancestors),
            proposal_id=proposed["proposal_id"], effective_members=ess,
            unique_particles=len(np.unique(x, axis=0)))
        # JSON has no infinities. Preserve zero-target log values as explicit null.
        for name in ("log_prior", "log_likelihood"):
            previous[name] = [v if np.isfinite(v) else None for v in previous[name]]
        history.append({**previous, "count": count, "batch_sha256": digest(previous)})
        if ess >= ess_fraction*count or count == cap:
            break
        count = min(2*count, cap)
    return dict(schema=SCHEMA, **previous, seed=seed, attempts=history,
        status="ess-sufficient" if ess >= ess_fraction*count else "budget-exhausted",
        convergence="not checked", training_promotion=False,
        weight_rule="prior times likelihood divided by this batch proposal; batches replaced")


def convergence_check(runs: Sequence[dict], reference: Mapping[str, Sequence[float]],
                      observed: Mapping[str, float], *, reference_converged: bool,
                      relative_tolerance: float = .05) -> dict:
    """Compare independent-run and reference 5/50/95% T30 quantiles in seconds."""
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("positive relative tolerance required")
    reasons, errors = [], {}
    if len(runs) != 2 or len({r.get("seed") for r in runs}) != 2:
        reasons.append("two independently seeded runs required")
    if not reference_converged:
        reasons.append("reference not independently converged")
    if not observed or any(not k.endswith(".t30_s") for k in observed):
        raise ValueError("observed T30 coordinates required")
    for key, value in observed.items():
        scale = _value(key, value)
        try:
            q = np.asarray([r["quantiles"][key] for r in runs] + [reference[key]], float)
            if q.shape != (3, 3) or not np.isfinite(q).all() or np.any(q <= 0) or np.any(np.diff(q, axis=1) < 0):
                raise ValueError
            errors[key] = float(np.max(np.ptp(q, axis=0)) / scale)
        except (KeyError, ValueError):
            reasons.append(f"missing or incompatible quantiles: {key}")
    if any(r.get("status") != "ess-sufficient" for r in runs):
        reasons.append("inference unresolved")
    return dict(status="inconclusive" if reasons else "fail" if any(v > relative_tolerance for v in errors.values()) else "pass",
                reasons=reasons, maximum_quantile_difference_over_observed=errors,
                relative_tolerance=relative_tolerance)


def collect_render_predictions(ensemble, source, receiver, fs: int, *, seed: int,
                               duration_s: float | None = None) -> list[dict]:
    """Invoke the real RIR renderer, retaining imports, content and branch evidence.

    These are physical member draws, not discrepancy-adjusted waveforms. This is
    a deployed-path probe, not an independent metric oracle or a dataset receipt.
    Licensing checks in synthesize remain active. Nothing is written to audio.
    """
    from . import prior, synth, metrics, late
    imports = {m.__name__: {"path": str(Path(m.__file__).resolve()),
        "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()} for m in (prior, synth, metrics, late)}
    rows = []
    for i, member in enumerate(ensemble.members):
        rir = synth.synthesize(member, source, receiver, fs, seed=seed+i,
                               config=synth.SynthConfig(duration_s=duration_s))
        onset = round(rir.metadata["direct_delay_s"]*fs)+rir.metadata["latency_samples"]
        values = prior.metric_vector(rir.samples, fs, onset_sample=onset, noise_power=0.)
        values.pop("direct_delay_s", None)
        samples = np.asarray(rir.samples)
        if not len(samples) or not np.isfinite(samples).all() or np.ptp(samples) == 0:
            raise ValueError("empty, constant or invalid rendered RIR")
        rows.append(dict(member_id=str(i), model_hash=member.model_hash, metrics=values,
            rir_sha256=hashlib.sha256(str(samples.dtype).encode()+samples.tobytes()).hexdigest(),
            renderer="sensorsim.acoustics.room.synth.synthesize", imports=imports,
            seed=seed+i, fs=fs, sample_count=len(samples), source_xyz_m=list(source),
            receiver_xyz_m=list(receiver), metric_policy=METRIC_POLICY, noise_power=0.,
            provenance=rir.metadata, branch="physical-member-render", result_label="model-based"))
    return rows


def _intervals(predictions, weights, keys, *, bias=None, covariance=None):
    w = _weights(weights, len(predictions))
    intervals, availability = {}, {}
    for j, key in enumerate(keys):
        valid, values = [], []
        for i, row in enumerate(predictions):
            try:
                values.append(_transform(key, row[key])); valid.append(i)
            except (KeyError, ValueError):
                pass
        mass = float(w[valid].sum())
        availability[key] = dict(predicted_mass=mass, valid_members=len(valid),
            constant=bool(values and np.ptp(values) == 0))
        # No per-key renormalization of missing positive posterior mass.
        if not valid or mass < 1-1e-12:
            continue
        x, ww = np.asarray(values), w[valid] / mass
        mu = x + (0. if bias is None else bias[j])
        sd = 0. if covariance is None else float(np.sqrt(covariance[j, j]))
        if sd == 0:
            q = weighted_quantiles(mu, ww)
        else:
            lo, hi = float(mu.min()-12*sd), float(mu.max()+12*sd)
            q = [float(brentq(lambda z: float(ww @ ndtr((z-mu)/sd))-p, lo, hi)) for p in (.05, .5, .95)]
        intervals[key] = [_inverse(key, v) for v in q]
    return dict(intervals=intervals, availability=availability, nominal_interval=.90,
                units={k: metric_unit(k) for k in keys})


def predictive_products(parameter_predictions, actual_renders, weights, *,
                        member_ids: Sequence[str], uncertainty: dict,
                        predictive_target: str = "room-response") -> dict:
    """Parameter-only, actual-render and discrepancy-inclusive marginal products.

    Parameter predictions are caller-supplied deterministic forward summaries.
    Actual renders require one content-bound record for each member in order.
    Discrepancy-inclusive room-response adds shared discrepancy to render metrics;
    playback-measurement additionally adds playback and one future measurement's
    noise. Simulator randomness already occurs in actual renders, never twice.
    Covariance is retained, but marginal intervals do not assert joint coverage.
    """
    keys = _check_budget(uncertainty)
    w = _weights(weights, len(member_ids))
    if len(set(member_ids)) != len(member_ids) or any(not i for i in member_ids):
        raise ValueError("unique member identities required")
    if len(parameter_predictions) != len(w):
        raise ValueError("one parameter prediction per member required")
    if predictive_target not in ("room-response", "playback-measurement"):
        raise ValueError("declare room-response or playback-measurement target")
    result = dict(schema=SCHEMA, result_label="model-based", member_ids=list(member_ids),
        weights=w.tolist(), uncertainty_sha256=uncertainty["sha256"],
        parameter_only=_intervals(parameter_predictions, w, keys), actual_render=None,
        discrepancy_inclusive=None, actual_render_reasons=[], training_promotion=False,
        predictive_target=predictive_target, joint_coverage="not claimed")
    if actual_renders is None:
        result["actual_render_reasons"].append("actual-render hook missing")
        return result
    if len(actual_renders) != len(w) or [r.get("member_id") for r in actual_renders] != list(member_ids):
        raise ValueError("actual renders must preserve member order and identity")
    for row in actual_renders:
        required = ("rir_sha256", "model_hash", "imports", "renderer", "seed", "metric_policy", "branch")
        if any(k not in row for k in required) or row["branch"] != "physical-member-render" or row["metric_policy"] != METRIC_POLICY:
            raise ValueError("missing actual-render provenance or incompatible metric policy")
        if len(row["rir_sha256"]) != 64 or not row["imports"]:
            raise ValueError("render content digest and import records required")
    hashes = [r["rir_sha256"] for r in actual_renders]
    result["duplicate_render_content"] = len(hashes)-len(set(hashes))
    result["render_records_sha256"] = digest(actual_renders)
    rows = [r["metrics"] for r in actual_renders]
    result["actual_render"] = _intervals(rows, w, keys)
    cov = np.asarray(uncertainty["covariance"]["discrepancy_shared"])
    if predictive_target == "playback-measurement":
        cov = cov + np.asarray(uncertainty["covariance"]["playback_shared"]) + np.asarray(uncertainty["covariance"]["measurement_single"])
    result["discrepancy_inclusive"] = _intervals(rows, w, keys,
        bias=np.asarray(uncertainty["discrepancy_bias"]), covariance=cov)
    result["predictive_additive_covariance"] = cov.tolist()
    return result


class ResearchRoomEnsemble(Contract):
    """Diagnostic state, deliberately not a training-compatible RoomEnsemble.

    The separate version survives JSON serialization: legacy ensemble loaders
    refuse it. ``status`` describes the numerical update only. No method here
    converts this state to a trainable ensemble or grants P4/P0 authority.
    """
    version: Literal["room-research-ensemble/1"] = "room-research-ensemble/1"
    purpose: Literal["calibration"] = "calibration"
    survey: RoomSurvey
    members: tuple[RoomModel, ...]
    weights: tuple[float, ...]
    status: Literal["assumed", "known-source", "weak-decay", "prior-conflict"] = "assumed"
    evidence: tuple[dict, ...] = ()
    result_label: Literal["model-based"] = "model-based"

    @model_validator(mode="after")
    def _validate_support(self):
        # Reuse the input contract, without returning a training-capable object.
        RoomEnsemble.model_validate({**self.model_dump(), "version": "room-ensemble/1"})
        return self

    @property
    def ensemble_hash(self):
        return digest(self.model_dump(mode="json"))

    @property
    def provenance(self):
        # Legacy generate_dataset probes this single-RoomModel attribute for
        # non-RoomEnsemble inputs. Refuse before it can write any artifacts.
        raise ValueError("research room state requires independently verified P4/P0 "
                         "receipts on the merged training path; promotion unavailable")


def update_room_ensemble(ensemble, observation, predictions, *, uncertainty,
                         early_tolerances):
    """Opt-in adapter preserving observation lineage and sticky prior conflicts.

    Repeat decomposition must precede this call. Reusing one geometry or any
    accepted constituent content is refused. Existing uncertainty scales are
    never silently reinterpreted as measurement noise or log-space covariance.
    A modular ensemble is research-only until the integration owner binds P4/P0
    verifier receipts. The distinct result type and JSON version also refuse
    legacy dataset consumers that do not call require_discrepancy_review.
    """
    if ensemble.purpose != "calibration":
        raise ValueError("modular update requires calibration support")
    if not isinstance(ensemble, (RoomEnsemble, ResearchRoomEnsemble)):
        raise TypeError("modular update requires a room ensemble or research state")
    ensemble = ResearchRoomEnsemble.model_validate({**ensemble.model_dump(),
        "version": "room-research-ensemble/1"})
    _check_budget(uncertainty)
    record = dict(method=SCHEMA, observation=observation.model_dump(mode="json"),
                  accepted=False, uncertainty=uncertainty)
    reasons = []
    if not observation.qualified or not observation.values:
        reasons.append("unqualified or empty observation")
    if ensemble.status == "prior-conflict":
        reasons.append("sticky prior-conflict; explicitly revise upstream evidence")
    # The pooled observation and decomposition must describe the same values.
    for key in uncertainty["keys"]:
        if key not in observation.values or not np.isclose(observation.values[key], uncertainty["values"][key], rtol=1e-10, atol=1e-12):
            raise ValueError("pooled observation and uncertainty values disagree")
    for old in ensemble.evidence:
        if not old.get("accepted"):
            continue
        previous = old.get("observation", {})
        if (previous.get("source_xyz_m") == list(observation.source_xyz_m) and
                previous.get("receiver_xyz_m") == list(observation.receiver_xyz_m)):
            reasons.append("shared geometry already updated; pool before updating")
        old_hashes = set(old.get("uncertainty", {}).get("content_sha256", []))
        if old_hashes.intersection(uncertainty["content_sha256"]):
            reasons.append("accepted repeat content reused")
    if reasons:
        record["reasons"] = reasons
        return ensemble.model_copy(update={"evidence": (*ensemble.evidence, record)})
    update = late_block_update(predictions, ensemble.weights, uncertainty, early_tolerances=early_tolerances)
    record["late_block"] = update
    changes = {}
    if update["status"] == "updated":
        record["accepted"] = True  # update executed; not scientific acceptance
        changes = dict(weights=tuple(update["weights"]),
            status="known-source" if observation.kind == "known-source" else "weak-decay")
    record["acceptance_scope"] = "parameter update only; no training or scientific acceptance"
    record["training_promotion"] = False
    return ensemble.model_copy(update={**changes, "evidence": (*ensemble.evidence, record)})


def require_discrepancy_review(ensemble):
    """Fail closed for modular candidates until merged receipt binding exists.

    This lane does not own the verifier's receipt format or the training gate.
    It deliberately provides no boolean or caller-supplied bypass token.
    """
    if isinstance(ensemble, ResearchRoomEnsemble) or any(e.get("method") == SCHEMA for e in ensemble.evidence):
        raise ValueError("modular room calibration requires independently verified P4 actual-render, "
                         "convergence and P0 receipts bound to the merged training path; promotion unavailable")


def audit_scene_states(rows: Sequence[Mapping]) -> dict:
    """Verify descendants share a declared room/state/installation and receiver.

    Missing identity hooks are unqualified. Door-state labels remain descriptive:
    the shoebox renderer does not model a physical opening.
    """
    if not rows:
        return dict(status="unqualified", reasons=["empty scene rows"], scenes=0)
    seen, reasons = {}, []
    required = ("scene_id", "ensemble_hash", "ensemble_member_hash", "installation_id", "receiver_xyz_m")
    for row in rows:
        if any(k not in row or row[k] is None for k in required):
            return dict(status="unqualified", reasons=["missing scene/state/installation hooks"], scenes=len(seen))
        receiver = np.asarray(row["receiver_xyz_m"], float)
        if receiver.shape != (3,) or not np.isfinite(receiver).all():
            raise ValueError("finite 3D receiver required")
        identity = digest({k:row[k] for k in required if k != "scene_id"})
        scene = row["scene_id"]
        if scene in seen and seen[scene] != identity:
            reasons.append(f"incoherent room/state/installation for scene {scene}")
        seen[scene] = identity
    return dict(status="fail" if reasons else "pass", reasons=sorted(set(reasons)), scenes=len(seen))
