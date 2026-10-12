"""Assumed, surveyed room ensembles and conservative no-sweep calibration.

No material tables, recordings or trained estimators are bundled. A posterior is
still a model-based approximation of the speaker-room-mic chain; a good acoustic
fit does not establish event recognition accuracy or a measured false-alarm rate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq
from scipy.special import ndtr

from sensorsim.core.serialization import canonical_json, load_yaml_unique
from .model import BoxObstacle, OCTAVE_HZ, RoomModel, SURFACES


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Site(Contract):
    """Surveyed Cartesian centre and axis-wise uniform uncertainty, in metres."""
    x: float
    y: float
    z: float
    spread_m: tuple[float, float, float] = (0., 0., 0.)
    facing: str = "unspecified; omnidirectional assumption"

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])

    @model_validator(mode="after")
    def _spread(self):
        if min(self.spread_m) < 0:
            raise ValueError("site spread must be nonnegative")
        return self


class Microphone(Site):
    tolerance_m: float = Field(default=.05, ge=0., le=1.)

    @model_validator(mode="after")
    def _mic_spread(self):
        if any(self.spread_m):
            raise ValueError("microphone uses tolerance_m, not source spread_m")
        return self


class Dimensions(Contract):
    length_x: float = Field(gt=0.)
    width_y: float = Field(gt=0.)
    height_z: float = Field(gt=0.)

    @property
    def xyz(self):
        return (self.length_x, self.width_y, self.height_z)


class RoomPriorConfig(Contract):
    """Assumed bounds, not population estimates; octave order is OCTAVE_HZ.

    Correlated log-uniform marginals via a smooth Gaussian copula. A sampled
    T30-equivalent decay is mapped to uniform effective wall absorption under
    this model's Eyring/air/obstacle budget. Unattainable draws are rejected.
    This cannot identify per-wall materials or scattering.
    """
    count: int = Field(default=32, ge=2, le=2048, strict=True)
    seed: int = Field(default=4108, ge=0, strict=True)
    # Engineering coverage assumptions in octave order, NOT household population
    # bounds. The synthesis's .25-.9 s assumption covered only 500-1000 Hz and
    # excludes dry rooms. A .05 s design floor admits strongly damped transfers;
    # falling HF ceilings allow air/furnishing losses without imposing a slope.
    t30_s: tuple[tuple[float, float], ...] = ((.05,1.5), (.05,1.3), (.05,1.1),
                                           (.05,.9), (.05,.8), (.05,.7), (.05,.6))
    absorption: tuple[tuple[float, float], ...] = ((.005, .98),)*7
    provenance: str = "per-band engineering coverage assumptions, not measured population bounds; replace using independent deployment evidence"
    status: Literal["assumed"] = "assumed"

    @model_validator(mode="after")
    def _bounds(self):
        if len(self.t30_s) != 7 or len(self.absorption) != 7:
            raise ValueError("prior requires seven octave ranges, 125..8000 Hz")
        if any(not 0 < lo <= hi for lo, hi in self.t30_s):
            raise ValueError("T30 ranges must be positive and ordered")
        if any(not 0 < lo <= hi < 1 for lo, hi in self.absorption):
            raise ValueError("absorption ranges must lie inside (0,1)")
        return self


class RoomSurvey(Contract):
    """room.yaml; source_sites maps each class to one or more surveyed patches.

    Example site: knock: [{x: 0.02, y: 1.0, z: 1.25,
                           spread_m: [0.0, 0.2, 0.35]}].
    Missing class sites are errors when rendering, never inferred from the speaker.
    Door states/tags document assumptions; they do not create openings in a shoebox.
    """
    units: Literal["m"] = "m"
    origin: str = "room floor corner; z up"
    room: Dimensions
    microphone: Microphone
    loudspeaker: Site
    source_sites: dict[str, tuple[Site, ...]] = Field(default_factory=dict)
    obstacles: tuple[BoxObstacle, ...] = ()
    furnishing_tags: tuple[str, ...] = ()
    door_states: tuple[str, ...] = ("unspecified",)
    notes: str = ""
    temperature_c: float = 20.
    humidity_pct: float = 50.
    prior: RoomPriorConfig = Field(default_factory=RoomPriorConfig)

    @model_validator(mode="after")
    def _geometry(self):
        base = RoomModel(shoebox_dims_m=self.room.xyz, obstacles=self.obstacles,
                         temperature_c=self.temperature_c, humidity_pct=self.humidity_pct)
        if any(not label.strip() or not sites for label, sites in self.source_sites.items()):
            raise ValueError("every source class needs a nonempty label and sites")
        for site in (self.microphone, self.loudspeaker, *(s for sites in self.source_sites.values() for s in sites)):
            if np.any(site.xyz <= 0) or np.any(site.xyz >= base.shoebox_dims_m):
                raise ValueError("survey site outside room; sites must be strictly inside")
            if any(b.contains(site.xyz) for b in self.obstacles):
                raise ValueError("survey site inside obstacle")
            if not isinstance(site, Microphone) and (np.any(site.xyz-np.array(site.spread_m) <= 0)
                    or np.any(site.xyz+np.array(site.spread_m) >= base.shoebox_dims_m)):
                raise ValueError("source site spread extends outside room")
        if np.linalg.norm(self.microphone.xyz-self.loudspeaker.xyz) < .05:
            raise ValueError("microphone and loudspeaker need at least 5 cm separation")
        return self

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate(load_yaml_unique(Path(path).read_text()))

    @property
    def survey_hash(self):
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


class RoomEnsemble(Contract):
    version: Literal["room-ensemble/1"] = "room-ensemble/1"
    purpose: Literal["calibration", "domestic-render"] = "calibration"
    survey: RoomSurvey
    members: tuple[RoomModel, ...]
    weights: tuple[float, ...]
    status: Literal["assumed", "known-source", "weak-decay", "prior-conflict"] = "assumed"
    evidence: tuple[dict, ...] = ()
    result_label: Literal["model-based"] = "model-based"

    @model_validator(mode="after")
    def _weights(self):
        if not self.members or len(self.members) != len(self.weights):
            raise ValueError("one weight per ensemble member is required")
        if min(self.weights) < 0 or not np.isclose(sum(self.weights), 1., atol=1e-10):
            raise ValueError("weights must be nonnegative and sum to one")
        if any(m.shoebox_dims_m != self.survey.room.xyz for m in self.members):
            raise ValueError("ensemble members must share surveyed geometry")
        return self

    @property
    def ensemble_hash(self):
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()

    def member_schedule(self, count: int, seed: int):
        """Stratified posterior draws; one member per whole rendered scene."""
        rng = np.random.default_rng(seed)
        points = (np.arange(count)+rng.random())/count
        indices = np.searchsorted(np.cumsum(self.weights), points, side="right")
        rng.shuffle(indices)
        return indices.tolist()


def build_prior(survey: RoomSurvey, *, purpose: Literal["calibration", "domestic-render"] = "calibration") -> RoomEnsemble:
    """Sample calibration support, or a separate labelled rendering assumption."""
    cfg = survey.prior
    if purpose not in ("calibration", "domestic-render"):
        raise ValueError("unknown ensemble purpose")
    assumptions = {}
    if purpose == "domestic-render":
        from .render_prior import DOMESTIC_T30_S, DOMESTIC_RATIONALES, DOMESTIC_POLICY
        # Calibration count/seed control compute and reproducibility, not the
        # rendering support. Even a custom calibration prior stays separate.
        cfg = cfg.model_copy(update={"t30_s": DOMESTIC_T30_S,
            "absorption": RoomPriorConfig().absorption,
            "provenance": DOMESTIC_POLICY + "; every band is an engineering assumption, not measured"})
        assumptions = {"rendering_policy": DOMESTIC_POLICY,
            "band_rationales": dict(zip(map(str, OCTAVE_HZ), DOMESTIC_RATIONALES))}
    rng = np.random.default_rng(cfg.seed)
    base = RoomModel(shoebox_dims_m=survey.room.xyz, obstacles=survey.obstacles,
                     temperature_c=survey.temperature_c, humidity_pct=survey.humidity_pct)
    # Solve a single shared effective wall coefficient for each octave, leaving
    # declared obstacle properties untouched. Eyring includes air loss exactly as
    # the renderer does; no copied material tables or alternative decay formula.
    lower, upper = np.array(cfg.absorption).T
    tlo = base.t60_estimates(room_absorption=np.tile(lower, (6, 1)))
    thi = base.t60_estimates(room_absorption=np.tile(upper, (6, 1)))
    support = np.array(cfg.t30_s)
    if np.any(np.maximum(support[:, 0], thi) > np.minimum(support[:, 1], tlo)):
        raise ValueError("T30 and absorption prior ranges have no physically attainable intersection")
    members = []
    tilt = np.linspace(-1, 1, 7)
    for attempt in range(cfg.count*1000):
        g, slope = rng.normal(size=2)
        z = (.85*g+.35*tilt*slope+.15*rng.normal(size=7))/np.sqrt(.85**2+(.35*tilt)**2+.15**2)
        t = np.exp(np.log(support[:, 0])+ndtr(z)*np.log(support[:, 1]/support[:, 0]))
        if np.any(t < thi) or np.any(t > tlo):
            continue
        alpha = []
        for j in range(7):
            def residual(a):
                table = np.full((6, 7), .2); table[:, j] = a
                return base.t60_estimates(room_absorption=table)[j]-t[j]
            alpha.append(float(brentq(residual, lower[j], upper[j])) if lower[j] != upper[j] else float(lower[j]))
        members.append(RoomModel(shoebox_dims_m=survey.room.xyz, obstacles=survey.obstacles,
            temperature_c=survey.temperature_c, humidity_pct=survey.humidity_pct,
            absorption={s: tuple(alpha) for s in SURFACES}, late_t60_s=tuple(t),
            provenance={"status": "assumed", "method": "surveyed-room-prior/1", "survey_hash": survey.survey_hash,
                        "draw": len(members), "seed": cfg.seed, "prior": cfg.model_dump(mode="json"),
                        "purpose": purpose, **assumptions,
                        "mapping": "shared effective absorption via Eyring, including air and obstacles",
                        "furnishing_tags": list(survey.furnishing_tags), "door_states": list(survey.door_states),
                        "door_state_effect": "descriptive only; no open-wall propagation model"},
            uncertainty={"status": "assumed", "octave_hz": list(OCTAVE_HZ),
                         "t30_ranges_s": [list(v) for v in cfg.t30_s],
                         "absorption_ranges": [list(v) for v in cfg.absorption],
                         "materials": "unidentified", "scattering": "unfitted; zero on walls"}))
        if len(members) == cfg.count:
            return RoomEnsemble(survey=survey, purpose=purpose, members=tuple(members), weights=(1/cfg.count,)*cfg.count)
    raise ValueError("prior rejection budget exhausted; declare compatible T30 and absorption ranges")


def metric_vector(samples, fs: int, *, onset_sample: int | None = None,
                  noise_power: float | None = None) -> dict[str, float]:
    """Version-2 metrics, invalid/unsupported values omitted (never replaced by 0)."""
    from .metrics import room_metrics, exact_center
    from .rir import detect_onset
    onset = detect_onset(samples) if onset_sample is None else onset_sample
    result = {"direct_delay_s": onset/fs}
    for band in (500, 1000, 2000, 4000):
        if exact_center(band)*10**.15 >= fs/2:
            continue
        m = room_metrics(samples, fs, band=band, onset_sample=onset, noise_power=noise_power)
        for key, fit in (("t30_s", m.t30), ("edt_s", m.edt)):
            if fit.valid and fit.r_squared is not None and fit.r_squared >= .9:
                result[f"{band}.{key}"] = fit.rt60_s
        for key, value in (("c50_db", m.c50_db), ("drr_db", m.drr_db)):
            if value is not None and np.isfinite(value):
                result[f"{band}.{key}"] = value
    return result


def predict_metrics(ensemble: RoomEnsemble, source, receiver, fs: int, *, seed: int = 4108,
                    duration_s: float | None = None) -> list[dict]:
    from .synth import SynthConfig, synthesize
    rows = []
    for i, member in enumerate(ensemble.members):
        rir = synthesize(member, source, receiver, fs, seed=seed+i,
                         config=SynthConfig(duration_s=duration_s))
        onset = round(rir.metadata["direct_delay_s"]*fs)+rir.metadata["latency_samples"]
        row = metric_vector(rir.samples, fs, onset_sample=onset, noise_power=0.)
        row["direct_delay_s"] = rir.metadata["direct_delay_s"]
        rows.append(row)
    return rows


class CalibrationObservation(Contract):
    kind: Literal["known-source", "impulsive-decay"]
    source_group: str
    input_sha256: str
    values: dict[str, float]
    uncertainty: dict[str, float]
    source_xyz_m: tuple[float, float, float]
    receiver_xyz_m: tuple[float, float, float]
    qualified: bool = False
    reasons: tuple[str, ...] = ()
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _uncertainty(self):
        if not self.source_group or not self.input_sha256:
            raise ValueError("observation needs source group and input hash")
        if set(self.values) != set(self.uncertainty) or any(x <= 0 for x in self.uncertainty.values()):
            raise ValueError("each observation metric needs positive uncertainty")
        return self


def update_ensemble(ensemble: RoomEnsemble, observation: CalibrationObservation,
                    predictions: list[dict], *, modular_uncertainty: dict | None = None,
                    early_tolerances: dict | None = None) -> RoomEnsemble:
    """Approximate Gaussian likelihood with explicit discrepancy, not identified truth.

    Rejected/censored observations leave weights unchanged. Very concentrated or
    unsupported likelihoods also abstain; no silent point estimate is returned.
    Correlated metrics are averaged within a take to avoid pseudo-replication.
    """
    if modular_uncertainty is not None:
        from .calibration_discrepancy import update_room_ensemble
        return update_room_ensemble(ensemble, observation, predictions,
            uncertainty=modular_uncertainty, early_tolerances=early_tolerances or {})
    if ensemble.purpose != "calibration":
        raise ValueError("calibration requires calibration_ensemble.json, not the domestic rendering assumption")
    if len(predictions) != len(ensemble.members):
        raise ValueError("one prediction per member required")
    evidence = {"observation": observation.model_dump(mode="json"), "accepted": False}
    if not observation.qualified or not observation.values:
        evidence["reason"] = "unqualified or empty observation"
        return ensemble.model_copy(update={"evidence": (*ensemble.evidence, evidence)})
    allowed = {k: v for k, v in observation.values.items()
               if observation.kind == "known-source" or k.endswith(".t30_s")}
    if "direct_delay_s" in allowed and not _hardware_anchored_delay(observation):
        allowed.pop("direct_delay_s")
        evidence["direct_delay"] = "omitted: hardware-anchored recording offset not declared"
    if ensemble.status == "prior-conflict":
        from .render_prior import conflict_warning
        evidence["reason"] = conflict_warning(ensemble)
        return ensemble.model_copy(update={"evidence": (*ensemble.evidence, evidence)})
    # One declared discrepancy scale on EACH metric, not an average over keys.
    # This is a conservative support gate, not a significance test.
    conflicts, keys = {}, []
    for key, value in allowed.items():
        valid = [p[key] for p in predictions if key in p and np.isfinite(p[key])]
        if len(valid) == len(predictions):
            keys.append(key)
        if valid:
            lo, hi = min(valid)-observation.uncertainty[key], max(valid)+observation.uncertainty[key]
            if not lo <= value <= hi:
                conflicts[key] = {"observed": value, "support_plus_discrepancy": [lo, hi]}
    if conflicts:
        evidence.update(reason="prior-conflict: observation outside every member plus declared discrepancy",
                        conflicts=conflicts, support_discrepancy_multiplier=1.)
        return ensemble.model_copy(update={"status": "prior-conflict", "evidence": (*ensemble.evidence, evidence)})
    if len(keys) < 2:
        evidence["reason"] = "fewer than two jointly supported metrics"
        return ensemble.model_copy(update={"evidence": (*ensemble.evidence, evidence)})
    # Repeated playback at one geometry shares model discrepancy. The pipeline
    # pools it, and the API also refuses a second update at that same geometry.
    if any(e.get("accepted") and e["observation"]["source_xyz_m"] == list(observation.source_xyz_m)
           and e["observation"]["receiver_xyz_m"] == list(observation.receiver_xyz_m)
           for e in ensemble.evidence):
        evidence["reason"] = "shared geometry already updated; pool repeated evidence before updating"
        return ensemble.model_copy(update={"evidence": (*ensemble.evidence, evidence)})
    # Keep scalar arithmetic and metric order exactly as before. One contiguous
    # row reduction avoids a NumPy dispatch per ensemble member without
    # reassociating the sum or changing the per-take likelihood.
    loss = np.mean([[((p[k]-allowed[k])/observation.uncertainty[k])**2
                     for k in keys] for p in predictions], axis=1)
    weights = np.array(ensemble.weights)*np.exp(-.5*(loss-loss.min()))
    weights /= weights.sum()
    ess = float(1/np.sum(weights**2))
    if ess < min(3, len(weights)):
        evidence.update(reason="insufficient effective members", effective_members=ess)
        return ensemble.model_copy(update={"evidence": (*ensemble.evidence, evidence)})
    evidence.update(accepted=True, effective_members=ess, metric_keys=keys,
                    metric_policy="room-measured-metrics/2; drr/2")
    status = "known-source" if observation.kind == "known-source" else "weak-decay"
    return ensemble.model_copy(update={"weights": tuple(map(float, weights)), "status": status,
                                       "evidence": (*ensemble.evidence, evidence)})


def _hardware_anchored_delay(observation: CalibrationObservation) -> bool:
    """Pooling keeps authority only for delay-contributing constituent observations."""
    rows = observation.metadata.get("observations")
    if rows is not None:
        contributors = [r for r in rows if "direct_delay_s" in r["values"]]
        return bool(contributors) and all(
            r["metadata"].get("recording_offset_hardware_anchored") is True for r in contributors)
    return observation.metadata.get("recording_offset_hardware_anchored") is True


def known_source_observation(recording, excitation, fs, *, source, receiver,
                             source_group, input_sha256, recording_offset_s: float | None = None,
                             recording_offset_hardware_anchored: bool = False,
                             rir_duration_s=1.2) -> CalibrationObservation:
    """Known playback chain; an unanchored recording cannot measure propagation time.

    An offset normally removes recording pre-roll only. Set
    recording_offset_hardware_anchored=True ONLY when the offset includes all
    acquisition/playback-chain latency and leaves acoustic propagation time.
    Drift must be identifiable before deconvolution. Under-excited octave bins are
    excluded. Playback/mic coloration remains; discrepancy scales are assumptions.
    """
    from .rir import deconvolve_known, correct_known_clock
    metadata = {"scope": "speaker-room-mic transfer in excited bands",
                "discrepancy": "assumed, not a calibrated confidence interval"}
    base = dict(kind="known-source", source_group=source_group, input_sha256=input_sha256,
                source_xyz_m=tuple(source), receiver_xyz_m=tuple(receiver))
    try:
        y, x = np.asarray(recording, float), np.asarray(excitation, float)
        if y.ndim != 1 or x.ndim != 1 or not len(y) or not len(x):
            raise ValueError("known playback calibration requires mono files")
        if full_scale_run(y) or full_scale_run(x):
            raise ValueError("possible clipping: consecutive full-scale samples in playback or recording")
        if max(np.max(abs(y)), np.max(abs(x))) >= 1-1/32768:
            metadata["level_warning"] = "isolated full-scale peak; no clipping run"
        if recording_offset_s is not None and (not np.isfinite(recording_offset_s) or recording_offset_s < 0):
            raise ValueError("recording offset must be finite and nonnegative")
        if not isinstance(recording_offset_hardware_anchored, bool):
            raise ValueError("recording_offset_hardware_anchored must be a boolean")
        if recording_offset_hardware_anchored and recording_offset_s is None:
            raise ValueError("hardware-anchored timing requires recording_offset_s")
        metadata["recording_offset_hardware_anchored"] = recording_offset_hardware_anchored
        corrected, diagnostics = correct_known_clock(y, x, fs)
        metadata.update(diagnostics)
        # Unknown acquisition offset: preserve a 50 ms margin before the fitted
        # chain onset. The delay is deliberately not a propagation observation.
        extra = (round(recording_offset_s*fs) if recording_offset_s is not None else
                 max(0, int(diagnostics["total_delay_samples"])-round(.05*fs)))
        transfer = deconvolve_known(corrected, x, rir_samples=round(rir_duration_s*fs)+extra)
        transfer = transfer[extra:]
        metadata["extraction_start_sample"] = extra
        values = metric_vector(transfer, fs)
        if not recording_offset_hardware_anchored:
            metadata["chain_delay_after_offset_s"] = values.pop("direct_delay_s", None)
            metadata["direct_delay"] = "omitted: offset removes pre-roll, not unknown hardware latency"
        freq = np.fft.rfftfreq(len(x), 1/fs)
        power = abs(np.fft.rfft(x))**2
        band_power = {b: power[(freq >= b/np.sqrt(2)) & (freq <= b*np.sqrt(2))] for b in (500,1000,2000,4000)}
        # At least 80% of octave bins must lie within 40 dB of the peak spectrum.
        excited = [b for b, p in band_power.items() if len(p) and np.mean(p > power.max()*1e-4) >= .8]
        values = {k:v for k,v in values.items() if k == "direct_delay_s" or int(k.split('.')[0]) in excited}
        scales = {k: discrepancy_scale(k, v) for k,v in values.items()}
        metadata.update(excited_bands_hz=excited, recording_offset_s=recording_offset_s)
        return CalibrationObservation(**base, values=values, uncertainty=scales,
                                      qualified=len(values) >= 3, metadata=metadata)
    except ValueError as exc:
        return CalibrationObservation(**base, values={}, uncertainty={}, qualified=False,
                                      reasons=(str(exc),), metadata=metadata)


def impulsive_decay_observation(samples, fs, *, source, receiver, source_group, input_sha256,
                                excitation_end_s, floor_end_s, source_tail_upper_s=None,
                                censored=False) -> CalibrationObservation:
    """Weak knock evidence only after a declared excitation end and quiet pre-roll.

    An unknown source tail, censoring or clipping yields no update. Only late T30
    contributes, with a broad uncertainty including the source-tail bound. Bell
    and kettle envelopes should not be sent here without an independently justified
    excitation end and source-tail bound.
    """
    from .metrics import room_metrics
    x = np.asarray(samples, float)
    base = dict(kind="impulsive-decay", source_group=source_group, input_sha256=input_sha256,
                source_xyz_m=tuple(source), receiver_xyz_m=tuple(receiver))
    reasons = []
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("expected finite mono impulse evidence")
    if censored or source_tail_upper_s is None:
        reasons.append("censored decay or unknown source-tail bound")
    if not 0.02 <= floor_end_s < excitation_end_s < len(x)/fs-.1:
        reasons.append("insufficient pre-event floor or post-excitation support")
    if full_scale_run(x):
        reasons.append("possible clipping")
    if source_tail_upper_s is not None and (not np.isfinite(source_tail_upper_s) or source_tail_upper_s < 0):
        raise ValueError("source-tail bound must be nonnegative and finite")
    values, scales = {}, {}
    floor = float(np.mean(x[:max(1, round(floor_end_s*fs))]**2))
    if not reasons:
        from .metrics import reverse_filter
        for band in (500, 1000, 2000, 4000):
            if band*10**.15 >= fs/2:
                continue
            filtered = reverse_filter(x, fs, band)
            floor_band = float(np.mean(filtered[:round(floor_end_s*fs)]**2))
            m = room_metrics(x, fs, band=band, onset_sample=round(excitation_end_s*fs), noise_power=floor_band)
            if m.t30.valid and m.t30.r_squared >= .9:
                key = f"{band}.t30_s"
                values[key] = m.t30.rt60_s
                scales[key] = max(.1, .5*m.t30.rt60_s, source_tail_upper_s)
    return CalibrationObservation(**base, values=values, uncertainty=scales, qualified=bool(values), reasons=tuple(reasons),
        metadata={"excitation_end_s": excitation_end_s, "floor_end_s": floor_end_s,
                  "pre_event_noise_power": floor, "source_tail_upper_s": source_tail_upper_s,
                  "noise_method": "declared quiet pre-event mean-square", "censored": censored,
                  "discrepancy": "weak bound; source-tail uncertainty retained"})


def full_scale_run(samples, minimum_run=2):
    """Two consecutive rail samples flag clipping; an isolated normalized peak does not."""
    x = np.asarray(samples)
    rail = np.abs(x) >= 1-1/32768
    return bool(np.any(np.convolve(rail.astype(int), np.ones(minimum_run, int), mode="valid") >= minimum_run)) if len(x) >= minimum_run else False


def discrepancy_scale(key, value):
    """Declared model discrepancy (engineering assumptions, not measured errors)."""
    return (max(.025, abs(value)*.25) if key.endswith('t30_s') else
            max(.035, abs(value)*.35) if key.endswith('edt_s') else
            4. if key.endswith('c50_db') else 6. if key.endswith('drr_db') else .002)


def predictive_intervals(predictions: list[dict], weights, *, diagnostics: dict | None = None) -> dict:
    """Central 90% Gaussian-mixture predictions including declared discrepancy.

    Valid members are renormalized per key; missing weight is reported separately.
    At least half the posterior mass must support a metric. These model-based
    intervals are not measured confidence bounds, and do not cure a prior conflict.
    """
    result = {}
    w = np.asarray(weights, float)
    if len(w) != len(predictions) or not len(w) or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError("predictions need matching normalized nonnegative weights")
    for key in sorted(set().union(*(set(p) for p in predictions))):
        idx = [i for i,p in enumerate(predictions) if key in p and np.isfinite(p[key])]
        mass = float(w[idx].sum())
        if diagnostics is not None:
            diagnostics[key] = {"missing_weight": 1-mass, "valid_members": len(idx),
                                "discrepancy": "assumed Gaussian; shared by repeats"}
        if mass < .5 or not idx:
            continue
        values = np.array([predictions[i][key] for i in idx]); ww = w[idx]/mass
        scales = np.array([discrepancy_scale(key,v) for v in values])
        def cdf(value):
            return float(ww @ ndtr((value-values)/scales))
        low, high = float(min(values-9*scales)), float(max(values+9*scales))
        result[key] = [float(brentq(lambda v: cdf(v)-q, low, high)) for q in (.05,.5,.95)]
        if key.endswith('_s'):
            result[key] = [max(0., v) for v in result[key]]
    return result
