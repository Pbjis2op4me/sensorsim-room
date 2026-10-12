# SPDX-License-Identifier: Apache-2.0
"""Opt-in parent-level contracts; the reviewed active-sample mixer stays default.

An empirical draw selects a *whole waveform and its metadata together*. It does
not draw a marginal level and attach it to an unrelated hit/cycle. Relative
anchors are explicit session-level assumptions, not SPL calibration. No fitted
tail, universal SPL distribution, device response or room response is inferred.

Apply the returned scalar once, before propagation/device filtering and cropping.
The integration owner must bypass receiver/crop normalization for these modes.
This module alone does not activate a new trainer or qualify deployment claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from numbers import Integral
from typing import Mapping

import numpy as np

from sensorsim.acoustics.room.source_reference import SourceReference

VERSION = "parent-event-levels/1"
LEGACY = "assumed_active_sample/1"
RELATIVE = "relative_parent_anchor/1"
CALIBRATED = "calibrated_preserve/1"
MODES = (LEGACY, RELATIVE, CALIBRATED)


def validate_mode(cfg):
    mode = cfg.get("event_level_mode", LEGACY)
    if mode not in MODES:
        raise ValueError("unknown/versionless event_level_mode")
    if mode != LEGACY and cfg.get("device_split", "in_distribution") != "in_distribution":
        raise ValueError("parent levels cannot be mixed with the legacy stress SNR contract")
    return mode


def require_legacy_training(cfg):
    """Fail closed until full-parent trainer/context integration is independently verified.

    The handoff wires the mixer and metadata independently. The integration owner
    replaces this guard only after target/confuser symmetry, filter history,
    deployment windows and persistence all consume the same parent draw.
    """
    if validate_mode(cfg) != LEGACY:
        raise ValueError("parent-level training requires the reviewed full-parent context adapter; not integrated")


def role_bindings_from_bank(manifest, *, expected_sha256):
    """Consume lane B's operational-bank/1 all-role provenance without opening audio.

    The digest uses B's canonical JSON (sorted compact keys). Original parent
    hashes must include this module's rate-tagged float64 content hash; B's
    float32 rendered-audio hash has different semantics and is never substituted.
    source_group belongs in lineage_ids. This checks supplied lineage, not acoustic
    near-duplicate identity or independence of the original recording sessions.
    """
    import json
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if actual != expected_sha256 or manifest.get("schema") != "operational-bank/1" or manifest.get("allocation_frozen") is not True:
        raise ValueError("frozen operational-bank/1 digest required")
    rows = manifest.get("rows", [])
    if not rows or {r.get("role") for r in rows} != {"train", "development", "audit"}:
        raise ValueError("complete train/development/audit allocation required")
    result, identities, contents, seen = {}, {}, {}, set()
    for row in rows:
        ident = _text(row.get("id"), "bank row")
        if ident in seen:
            raise ValueError("duplicate bank row")
        seen.add(ident)
        role = row["role"]
        if not row.get("lineage_ids") or not row.get("parent_audio_sha256"):
            raise ValueError("complete lineage and original parent hashes required")
        for name in (row.get("parent_id"), row.get("session_id"), *row["lineage_ids"]):
            _text(name, "bank identity")
            if identities.setdefault(name, role) != role:
                raise ValueError("parent/session/lineage crosses roles")
        result["parent:"+row["parent_id"]] = role
        result["session:"+row["session_id"]] = role
        for name in row["lineage_ids"]:
            result["group:"+name] = role
        for digest in (row.get("audio_sha256"), row.get("file_sha256"), *row["parent_audio_sha256"]):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("SHA256 bank content identities required")
            if contents.setdefault(digest, role) != role:
                raise ValueError("duplicate bank content crosses roles")
            result["content:"+digest] = role
    return result


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"explicit {name} required")
    return value


def _audio(samples):
    raw = np.asarray(samples)
    if raw.dtype.kind not in "fiu" or raw.ndim != 1 or not raw.size:
        raise ValueError("nonempty real mono audio required")
    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("audio must be finite")
    return x


def _rate(fs):
    if isinstance(fs, bool) or not isinstance(fs, Integral) or fs <= 0:
        raise ValueError("positive integer fs required")
    return int(fs)


def _interval(interval, n):
    if (len(interval) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in interval)
            or not 0 <= interval[0] < interval[1] <= n):
        raise ValueError("nonempty half-open sample interval inside parent required")
    return tuple(map(int, interval))


@dataclass(frozen=True)
class LevelReference:
    """A meter definition, not a source of gain or a certification of calibration.

    ``rms`` and ``exposure`` use 10 log10 of energy, with exposure referenced
    to one second. ``peak`` is an unweighted sample peak (not Fast/Slow max).
    A is the repository's FFT continuous-curve power meter; Z means recorded
    bandwidth, not a certified full-band instrument. Interval is never inferred.
    """
    quantity: str
    units: str
    reference_value: float
    weighting: str
    interval_samples: tuple[int, int]
    fs: int
    position_id: str
    version: str = "parent-level-reference/1"

    def __post_init__(self):
        if self.version != "parent-level-reference/1" or self.quantity not in ("rms", "exposure", "peak"):
            raise ValueError("unsupported level quantity/version; sound power and LUFS are not pressure")
        if self.units not in ("Pa", "relative", "peak_fs"):
            raise ValueError("explicit Pa, relative or peak_fs level units required")
        expected = 20e-6 if self.units == "Pa" else 1.
        if _finite(self.reference_value, "reference_value", positive=True) != expected:
            raise ValueError("reference must be 20 uPa for pressure or 1 for relative/peak_fs")
        if self.weighting not in ("Z", "A") or (self.quantity == "peak" and self.weighting != "Z"):
            raise ValueError("supported weighting is Z or A power; peak is Z only")
        _rate(self.fs)
        _text(self.position_id, "measurement position")
        if len(self.interval_samples) != 2:
            raise ValueError("explicit interval required")
        object.__setattr__(self, "interval_samples", _interval(self.interval_samples, self.interval_samples[1]))

    def as_dict(self):
        """JSON wire form; the validated in-memory interval remains a tuple."""
        return asdict(self) | {"interval_samples": list(self.interval_samples)}


def validate_level_reference(value, source_reference: SourceReference, *, sample_count=None):
    """Validate quantity/reference/weighting/interval without upgrading unknowns."""
    if not isinstance(source_reference, SourceReference):
        raise TypeError("typed SourceReference required")
    ref = value if isinstance(value, LevelReference) else LevelReference(**value)
    if ref.units != source_reference.units:
        raise ValueError("level and SourceReference units differ; convert explicitly before measuring")
    if sample_count is not None:
        _interval(ref.interval_samples, sample_count)
    return ref


def measure_level(samples, reference: LevelReference):
    """Finite dB or None for zero power; no artificial floor or silence rejection."""
    x = _audio(samples)
    a, b = _interval(reference.interval_samples, len(x))
    x = x[a:b]
    if reference.quantity == "peak":
        value = float(np.max(np.abs(x))**2)
    else:
        value = _mean_square(x, reference.weighting, reference.fs)
    if reference.quantity == "exposure":
        value *= (b-a)/reference.fs
    if not np.isfinite(value):
        raise ValueError("level power overflow")
    return None if value == 0 else float(10*np.log10(value/reference.reference_value**2))


def _mean_square(samples, weighting, fs):
    if weighting == "A":
        from sensorsim.acoustics.weighting import a_weighted_mean_square
        # A rejects DC exactly; FFT roundoff must not invent energy for constants.
        value = 0. if np.all(samples == samples[0]) else float(a_weighted_mean_square(samples, fs))
    elif weighting == "Z":
        value = float(np.mean(samples*samples))
    else:
        raise ValueError("unsupported power weighting")
    return _finite(value, "mean square power")


def content_sha256(samples, fs):
    """Decoded float samples plus rate; normalize signed zero, never sample peaks."""
    x = _audio(samples).copy()
    x[x == 0] = 0.
    h = hashlib.sha256(_rate(fs).to_bytes(8, "little"))
    h.update(x.astype("<f8").tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class ParentEvent:
    """One complete event/cycle, after declared acquisition grouping, before crops.

    Provenance role bindings must include parent, source group, capture session
    and decoded content. Censored silence is retained rather than filtered out.
    Commercial eligibility is owned by the upstream licensing/intake gate.
    """
    parent_id: str
    source_group: str
    session_id: str
    label: str
    source_id: str
    state: str
    role: str
    capture_route: str
    response_domain: str
    samples: np.ndarray
    source_reference: SourceReference
    level_reference: LevelReference
    censored: bool = False

    def __post_init__(self):
        for name in ("parent_id", "source_group", "session_id", "label", "source_id", "state"):
            _text(getattr(self, name), name)
        if self.role not in ("train", "development", "audit"):
            raise ValueError("unknown parent role")
        if self.capture_route not in ("synthetic_proxy", "chamber_target", "actual_source", "known_playback"):
            raise ValueError("explicit source/playback capture route required")
        if self.response_domain not in ("source_only", "receiver_pre_device", "device_recording", "effective_playback"):
            raise ValueError("explicit response domain required")
        if not isinstance(self.censored, bool):
            raise ValueError("censored must be boolean")
        x = _audio(self.samples)
        reference = validate_level_reference(self.level_reference, self.source_reference, sample_count=len(x))
        object.__setattr__(self, "level_reference", reference)
        if self.level_reference.interval_samples != (0, len(x)):
            raise ValueError("parent reference must cover the complete event/cycle")
        if np.all(x == x[0]) and (x[0] != 0 or not self.censored):
            raise ValueError("constant parent is invalid; silence requires explicit censoring")
        # Immutable bytes prevent callers mutating a fitted prior through array aliases.
        object.__setattr__(self, "samples", np.frombuffer(x.astype("<f8").tobytes(), dtype="<f8"))

    @property
    def content_id(self):
        return content_sha256(self.samples, self.level_reference.fs)

    @property
    def condition(self):
        return self.label, self.source_id, self.state

    def metadata(self):
        return {k: getattr(self, k) for k in ("parent_id", "source_group", "session_id", "label",
                "source_id", "state", "role", "capture_route", "response_domain", "censored")} | {
            "content_sha256": self.content_id, "source_reference": self.source_reference.as_dict(),
            "level_reference": self.level_reference.as_dict(),
            "observed_level_db": measure_level(self.samples, self.level_reference)}


@dataclass(frozen=True)
class RelativeAnchor:
    """Session-wide multiplier into the renderer's peak-sine reference units.

    It is an assumed relative mapping even if a numeric SPL reference is named.
    Fit/authorize this anchor outside the sampler and reuse for every descendant.
    """
    gain: float
    anchor_id: str

    def __post_init__(self):
        _finite(self.gain, "anchor gain", positive=True)
        _text(self.anchor_id, "anchor identity")


@dataclass(frozen=True)
class LevelPrior:
    mode: str
    parents: tuple[ParentEvent, ...]
    gains: tuple[float, ...]
    anchors: tuple[str, ...]
    render_reference_spl_db: float
    digest: str
    version: str = VERSION

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Verify the fitted content binding, including after deserialization.

        A digest detects inconsistent replacement, not dishonest acquisition or
        a malicious caller deliberately constructing a new internally consistent
        prior. Protocol approval and file custody remain external authorities.
        """
        if self.version != VERSION or self.mode not in (RELATIVE, CALIBRATED):
            raise ValueError("unsupported fitted prior version/mode")
        if (not all(isinstance(v, tuple) for v in (self.parents, self.gains, self.anchors))
                or not self.parents or len(self.parents) != len(self.gains) or len(self.parents) != len(self.anchors)):
            raise ValueError("complete immutable fitted prior rows required")
        _finite(self.render_reference_spl_db, "render_reference_spl_db")
        for p, gain, anchor in zip(self.parents, self.gains, self.anchors):
            _validate_parent_mode(p, self.mode)
            _finite(gain, "parent gain", positive=True)
            _text(anchor, "anchor identity")
        payload = _prior_payload(self.mode, self.parents, self.gains, self.anchors, self.render_reference_spl_db)
        if self.digest != _prior_digest(payload):
            raise ValueError("fitted prior content/digest mismatch")

    def metadata(self):
        self.validate()
        return dict(version=self.version, mode=self.mode, digest=self.digest,
                    render_reference_spl_db=self.render_reference_spl_db,
                    sampling="uniform source group, then uniform complete parent within condition",
                    extrapolation=False, independent_groups=len({p.source_group for p in self.parents}),
                    parents=[p.metadata() | {"gain": g, "anchor_id": a}
                             for p, g, a in zip(self.parents, self.gains, self.anchors)])


def _validate_parent_mode(parent, mode):
    if not isinstance(parent, ParentEvent):
        raise ValueError("typed fitted parent required")
    sr = parent.source_reference
    validate_level_reference(parent.level_reference, sr, sample_count=len(parent.samples))
    if (parent.role != "train" or sr.target != "source_1m" or parent.capture_route == "known_playback"
            or parent.response_domain != "source_only"):
        raise ValueError("fitted parent must retain train source-only authority")
    if mode == RELATIVE:
        if sr.units != "relative" or sr.calibration is not None:
            raise ValueError("relative mode needs uncalibrated relative units")
        return "unknown"
    if mode != CALIBRATED or not (sr.units == "Pa" or (sr.units == "peak_fs" and sr.calibration is not None)):
        raise ValueError("absolute calibration is unknown or incompatible with fitted mode")
    return "known"


def _prior_payload(mode, parents, gains, anchors, reference):
    return dict(version=VERSION, mode=mode, render_reference_spl_db=reference,
                parents=[p.metadata() for p in parents], gains=list(gains), anchors=list(anchors),
                role_bindings={key: "train" for p in parents for key in
                    (f"parent:{p.parent_id}", f"group:{p.source_group}",
                     f"session:{p.session_id}", f"content:{p.content_id}")})


def _prior_digest(payload):
    import json
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit_level_prior(parents, *, mode, role_bindings: Mapping[str, str], relative_anchors=None,
                    session_offsets_db=None, render_reference_spl_db=94.):
    """Fit an empirical joint distribution using only explicitly bound train parents.

    Bindings use ``parent:<id>``, ``group:<id>``, ``session:<id>``, ``content:<hash>``.
    They come from the common evaluation protocol, never inferred from a filename.
    All input rows must be eligible; nothing is silently filtered. Session offsets
    are fixed external calibration realizations, never independent per-event draws.
    """
    import json
    if mode not in (RELATIVE, CALIBRATED):
        raise ValueError("empirical fitting requires a versioned parent mode; legacy stays in the mixer")
    rows = tuple(parents)
    if not rows:
        raise ValueError("empty parent distribution")
    if any(not isinstance(p, ParentEvent) for p in rows):
        raise TypeError("typed complete ParentEvent records required")
    reference = _finite(render_reference_spl_db, "render_reference_spl_db")
    pressure_unit = np.sqrt(2)*20e-6*10.**(reference/20.)
    _finite(pressure_unit, "pressure unit", positive=True)
    anchors, offsets = dict(relative_anchors or {}), dict(session_offsets_db or {})
    sessions = {p.session_id for p in rows}
    if set(offsets)-sessions or set(anchors)-sessions:
        raise ValueError("unused session anchor/offset is not permitted")
    if mode == CALIBRATED and anchors:
        raise ValueError("relative anchors cannot override pressure calibration")
    seen_ids, seen_content, session_refs, group_conditions = set(), set(), {}, {}
    gains, names = [], []
    for p in rows:
        if not isinstance(p, ParentEvent):
            raise TypeError("typed complete ParentEvent records required")
        keys = (f"parent:{p.parent_id}", f"group:{p.source_group}",
                f"session:{p.session_id}", f"content:{p.content_id}")
        if p.role != "train" or any(role_bindings.get(key) != "train" for key in keys):
            raise ValueError("parent/group/session/content must be explicitly bound to train")
        if p.parent_id in seen_ids or p.content_id in seen_content:
            raise ValueError("duplicate parent or decoded content; aliases cannot add empirical mass")
        seen_ids.add(p.parent_id)
        seen_content.add(p.content_id)
        if group_conditions.setdefault(p.source_group, p.condition) != p.condition:
            raise ValueError("one acquisition group cannot change source/class/state")
        sr = p.source_reference
        if sr.target != "source_1m" or p.capture_route == "known_playback" or p.response_domain != "source_only":
            raise ValueError("source_1m authority required; playback/effective/receiver paths need a separate adapter")
        identity = (sr.units, json.dumps(sr.as_dict(), sort_keys=True))
        if session_refs.setdefault(p.session_id, identity) != identity:
            raise ValueError("session calibration/reference changed within one session")
        offset = _finite(offsets.get(p.session_id, 0.), "fixed session offset")
        if mode == RELATIVE:
            if sr.units != "relative" or sr.calibration is not None:
                raise ValueError("relative mode needs uncalibrated relative units")
            anchor = anchors.get(p.session_id)
            if not isinstance(anchor, RelativeAnchor):
                raise ValueError("explicit relative anchor required for each session")
            gain, name = anchor.gain, anchor.anchor_id
        else:
            if sr.units == "Pa":
                gain = 1./pressure_unit
            elif sr.units == "peak_fs" and sr.calibration is not None:
                gain = sr.calibration.pa_per_peak_fs/pressure_unit
            else:
                raise ValueError("absolute calibration is unknown; cannot preserve calibrated pressure")
            name = sr.calibration.gain_provenance if sr.calibration else "samples explicitly in Pa"
        gain *= 10.**(offset/20.)
        gains.append(_finite(gain, "parent gain", positive=True))
        names.append(name)
    digest = _prior_digest(_prior_payload(mode, rows, gains, names, reference))
    return LevelPrior(mode, rows, tuple(gains), tuple(names), reference, digest)


@dataclass(frozen=True)
class ParentGain:
    """One selected row bound to the exact fitted prior; never a free gain override."""
    parent: ParentEvent
    gain: float
    mode: str
    prior_digest: str
    anchor_id: str
    render_reference_spl_db: float
    prior: LevelPrior = field(repr=False, compare=False)

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.prior, LevelPrior):
            raise ValueError("fitted LevelPrior required for parent gain")
        self.prior.validate()
        status = _validate_parent_mode(self.parent, self.mode)
        if (self.mode != self.prior.mode or self.prior_digest != self.prior.digest
                or self.render_reference_spl_db != self.prior.render_reference_spl_db):
            raise ValueError("parent gain mode/reference/digest differs from fitted prior")
        indices = [i for i, p in enumerate(self.prior.parents) if p is self.parent]
        if len(indices) != 1:
            raise ValueError("parent gain must select one fitted parent")
        i = indices[0]
        if (_finite(self.gain, "parent gain", positive=True) != self.prior.gains[i]
                or self.anchor_id != self.prior.anchors[i]):
            raise ValueError("parent gain/anchor differs from fitted prior")
        return status

    def metadata(self):
        status = self.validate()
        return self.parent.metadata() | dict(version=VERSION, mode=self.mode, parent_gain=self.gain,
            prior_digest=self.prior_digest, anchor_id=self.anchor_id,
            render_reference_spl_db=self.render_reference_spl_db,
            absolute_calibration=status,
            calibration_scope="capture session; fixed before parent sampling")


def sample_parent_gain(prior: LevelPrior, rng: np.random.Generator, *, condition):
    """Select one complete joint observation; no redraw for quiet/censored parents."""
    if not isinstance(prior, LevelPrior) or not isinstance(rng, np.random.Generator):
        raise TypeError("typed LevelPrior and numpy Generator required")
    prior.validate()
    if len(condition) != 3:
        raise ValueError("condition is (label, source_id, state)")
    indices = [i for i, p in enumerate(prior.parents) if p.condition == tuple(condition)]
    if not indices:
        raise ValueError("no training parents for condition; no cross-state fallback")
    groups = sorted({prior.parents[i].source_group for i in indices})
    group = groups[int(rng.integers(len(groups)))]
    indices = [i for i in indices if prior.parents[i].source_group == group]
    i = indices[int(rng.integers(len(indices)))]
    return ParentGain(prior.parents[i], prior.gains[i], prior.mode, prior.digest,
                      prior.anchors[i], prior.render_reference_spl_db, prior)


def apply_parent_gain(samples, draw: ParentGain):
    """One scalar for the whole parent (or a descendant); no metering/normalization."""
    if not isinstance(draw, ParentGain) or draw.mode not in (RELATIVE, CALIBRATED):
        raise ValueError("a parent-level draw is required")
    draw.validate()
    out = _audio(samples)*_finite(draw.gain, "parent gain", positive=True)
    if not np.isfinite(out).all():
        raise ValueError("parent gain overflow")
    return out


def receiver_gain(draw: ParentGain, cfg):
    """Mixer handoff: use the source's scalar after *linear* propagation exactly once.

    Caller provides the original, unnormalized receiver render. This does not
    authorize re-filtering a device recording or renormalizing a crop.
    """
    mode = validate_mode(cfg)
    if not isinstance(draw, ParentGain) or mode != draw.mode:
        raise ValueError("mixer requires a matching typed parent gain")
    draw.validate()
    if float(cfg.get("render_reference_spl_db", 94.)) != draw.render_reference_spl_db:
        raise ValueError("parent and mixer pressure references differ")
    return _finite(draw.gain, "parent gain", positive=True)


def recover_render(samples, row, *, relative_samples=None):
    """Undo dataset export scaling, preferably using its lossless relative render.

    The file-reading adapter must verify file hashes before calling this helper.
    Input ``samples`` is decoded peak-FS PCM, not integer PCM. No lost precision is
    claimed recovered. This never applies SourceReference distance a second time.
    """
    sr = SourceReference.from_dict(row["dry_source"]["source_reference"])
    if row["dry_source"].get("level_reference", {}).get("effective_source"):
        raise ValueError("effective playback transfer needs an explicit separate adapter")
    level, boundary = row["training_level"], row["audio_boundary"]
    if level["relative_units"] != sr.units or boundary["source_units"] != sr.units:
        raise ValueError("render/source units disagree")
    if relative_samples is not None:
        if not level.get("relative_file") or not level.get("relative_sha256"):
            raise ValueError("lossless relative render needs declared file/hash")
        return _audio(relative_samples).copy(), dict(recovery="lossless_relative", quantization_error_bound=0.)
    if level.get("relative_file"):
        raise ValueError("declared relative render unavailable; no silent PCM fallback")
    raw = np.asarray(samples)
    if raw.dtype.kind != "f":
        raise ValueError("decode PCM to peak-FS float before recovery")
    scale = _finite(level["total_export_scale"], "total_export_scale", positive=True)
    factor = 1.
    if sr.units == "Pa":
        factor = _finite(boundary["pa_per_peak_fs"], "pa_per_peak_fs", positive=True)
        if sr.calibration is None or factor != sr.calibration.pa_per_peak_fs:
            raise ValueError("PCM/source calibration mismatch")
    return _audio(samples)*factor/scale, dict(recovery="quantized_pcm", quantization_error_bound=.5/32768*factor/scale)


def level_diagnostics(samples, reference: LevelReference, *, background=None, source_noise=None):
    """Report level/energy and matched *waveform* noise rise, including extremes.

    Noise powers use the declared weighting, interval and units. The explicit
    noise_reference always measures mean square (even for peak/exposure target
    references); rms/energy/peak diagnostic fields remain raw Z-band quantities.
    Noise is not inferred from a target+noise capture and stationary independence
    is not assumed. Caller supplies matched source-noise and background arrays in
    the same units/reference and at the same processing point. Zero background
    leaves the rise undefined, rather than producing a passing zero statistic.
    """
    x = _audio(samples)
    a, b = _interval(reference.interval_samples, len(x))
    part = x[a:b]
    row = dict(reference=reference.as_dict(), level_db=measure_level(x, reference),
               energy=float(np.sum(part*part)/reference.fs), energy_units=reference.units+"^2 s",
               rms=float(np.sqrt(np.mean(part*part))), peak=float(np.max(np.abs(part))),
               a_weighted_mean_square=_mean_square(part, "A", reference.fs),
               noise_diagnostic_version="matched-source-noise/2",
               noise_reference=replace(reference, quantity="rms").as_dict(),
               noise_power_units=reference.units+"^2",
               constant=bool(np.all(part == part[0])), silent=bool(not np.any(part)),
               source_noise_status="unavailable", background_rise_db=None)
    if (background is None) != (source_noise is None):
        raise ValueError("matched background and isolated source noise are both required")
    if background is not None:
        bg, noise = _audio(background), _audio(source_noise)
        if bg.shape != x.shape or noise.shape != x.shape:
            raise ValueError("matched noise diagnostics require identical sample alignment")
        bp = _mean_square(bg[a:b], reference.weighting, reference.fs)
        cp = _mean_square(bg[a:b]+noise[a:b], reference.weighting, reference.fs)
        if not np.isfinite([bp, cp]).all():
            raise ValueError("noise diagnostic power overflow")
        row.update(source_noise_status="measured_components" if bp > 0 and cp > 0 else "undefined_zero_power",
                   source_noise_power=_mean_square(noise[a:b], reference.weighting, reference.fs), background_power=bp,
                   combined_power=cp,
                   background_rise_db=float(10*np.log10(cp/bp)) if bp > 0 and cp > 0 else None)
    return row


def source_noise_gate(diagnostics):
    """P3 descriptive matched-background gate; retain all rows and extremes."""
    rows = list(diagnostics)
    rises = [r.get("background_rise_db") for r in rows]
    result = dict(claim_type="matched_source_noise_diagnostic", requested=len(rows),
                  rows=rows, rises_db=rises, outcome="inconclusive", median_rise_db=None, maximum_rise_db=None)
    if not rows or any(v is None or not np.isfinite(v) for v in rises):
        return result | {"reason": "missing/undefined matched components; no filtered-data pass"}
    meter = None
    for row in rows:
        if row.get("noise_diagnostic_version") != "matched-source-noise/2" or not row.get("noise_reference"):
            return result | {"reason": "missing versioned noise weighting/interval reference"}
        ref = LevelReference(**row["reference"])
        # Validate both sides before comparing meter semantics. JSON arrays and
        # historical in-memory tuples describe the same half-open interval.
        if "version" not in row["noise_reference"]:
            raise ValueError("explicit noise reference version required")
        noise_ref = LevelReference(**row["noise_reference"])
        if noise_ref != replace(ref, quantity="rms"):
            raise ValueError("noise weighting/interval/reference mismatch")
        identity = (ref.units, ref.reference_value, ref.weighting, ref.fs, ref.position_id)
        if meter is not None and identity != meter:
            raise ValueError("incompatible noise meters cannot be pooled")
        meter = identity
    median = float(np.median(rises))
    return result | dict(median_rise_db=median, maximum_rise_db=float(max(rises)),
                         outcome="pass" if median <= 1. else "fail")


def playback_level_gate(pairs):
    """P3 *playback-only* finite-panel screen, not physical-prior acceptance.

    Each row names a matched held-out playback pair and group, two LevelReference
    dictionaries and finite measured/predicted values. No arbitrary class pairing,
    window pseudoreplication, censor deletion or null-as-zero fallback is allowed.
    """
    rows = list(pairs)
    result = dict(claim_type="playback_level_prediction_only", requested=len(rows), scored=0,
                  outcome="inconclusive", physical_prior_outcome="inconclusive", errors_db=[],
                  median_absolute_error_db=None, fraction_within_six_db=None)
    if not rows:
        return result | {"reason": "no matched held-out playback pairs"}
    seen, content = set(), {"measured": set(), "predicted": set()}
    for row in rows:
        if (row.get("route") != "known_playback" or row.get("role") != "audit"
                or row.get("matched_acquisition") is not True or row.get("censored", False)):
            return result | {"reason": "unmatched, censored or non-held-out playback row"}
        key = _text(row.get("group_id"), "independent playback group")
        _text(row.get("pair_id"), "playback pair")
        if key in seen:
            raise ValueError("one playback pair per independent group required")
        seen.add(key)
        for side in content:
            digest = _text(row.get(side+"_content_sha256"), side+" content digest")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("SHA256 decoded-content identities required")
            if digest in content[side]:
                raise ValueError("duplicate playback content cannot add independent groups")
            content[side].add(digest)
        left, right = LevelReference(**row["measured_reference"]), LevelReference(**row["predicted_reference"])
        if left != right:
            raise ValueError("playback quantity/reference/weighting/interval/position mismatch")
        if row.get("measured_db") is None or row.get("predicted_db") is None:
            return result | {"reason": "missing or zero-power level; retain requested denominator"}
        result["errors_db"].append(_finite(row["predicted_db"], "predicted level")-_finite(row["measured_db"], "measured level"))
    errors = np.abs(result["errors_db"])
    median, fraction = float(np.median(errors)), float(np.mean(errors <= 6.))
    return result | dict(scored=len(rows), median_absolute_error_db=median, fraction_within_six_db=fraction,
                         outcome="pass" if median <= 3. and fraction >= .8 else "fail",
                         reason="finite paired playback panel only; no population-tail or natural-source claim")
