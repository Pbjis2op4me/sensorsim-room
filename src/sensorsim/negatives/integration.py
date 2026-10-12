# SPDX-License-Identifier: Apache-2.0
"""Locked receiver-domain confusers on the ordinary all-zero-label training path.

Non-tonal/non-impact families use the explicit engineering admission policy.
This policy is not a recorded-family realism result. Tonal and impact receipts
bind real evidence files; a tonal population additionally needs held-out recall.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .composer import audio_hash, digest
from .splits import Item, SplitPlan


def verified_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("expected file SHA-256 is required")
    try:
        raw = Path(path).read_bytes()
    except (TypeError, OSError) as exc:
        raise ValueError("receipt file is missing") from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("receipt file hash mismatch")
    return raw


def authorize_audio(audio, item, *, plan, expected_sha256, purpose):
    plan.verify_lock(expected_sha256)
    if not isinstance(item, Item):
        raise ValueError("an explicit locked source Item is required")
    if audio_hash(audio) != item.sha256:
        raise ValueError("decoded audio does not match locked source content")
    plan.require(item, purpose=purpose)


def require_doorbell_recall(row):
    """Recompute the fixed admission rule from a reviewed paired development check.

    This is a prerequisite study, not permission to tune on a sealed audit. The
    receipt must bind this confuser and the tested training population. Counts
    are by distinct held-out source parents, never repeated windows.
    """
    receipt = row.provenance.get("doorbell_recall", {})
    if receipt.get("review_decision") != "accept" or not receipt.get("reviewer"):
        raise ValueError("held-out doorbell recall review required before tonal admission")
    result = json.loads(verified_file(receipt.get("file"), receipt.get("sha256")))
    for key in ("baseline_model", "candidate_model"):
        verified_file(Path(receipt["file"]).parent/result.get(key+"_file", ""), result.get(key+"_sha256"))
    groups = result.get("source_groups", [])
    reference = np.asarray(result.get("baseline_detected", []))
    candidate = np.asarray(result.get("candidate_detected", []))
    if (len(groups) < 10 or len(set(groups)) != len(groups)
            or reference.shape != (len(groups),) or candidate.shape != reference.shape
            or reference.dtype != bool or candidate.dtype != bool
            or set(groups) & set(result.get("training_source_groups", []))
            or not result.get("training_source_groups")
            or row.item.sha256 not in result.get("confuser_audio_sha256", [])
            or result.get("claim") != "heldout-doorbell-recall/1"
            or result.get("units") != "independent_source_parent"
            or not result.get("baseline_model_sha256") or not result.get("candidate_model_sha256")
            or reference.mean() < .75 or candidate.mean()-reference.mean() < -.05):
        raise ValueError("held-out doorbell recall gate failed or is incomplete")


def proxy_screen_receipt(item):
    """A scoped engineering permission, NEVER a realism or listening review."""
    return dict(kind="proxy_screen_only", family=item.family, audio_sha256=item.sha256,
                human_listening_review=False,
                note="Synthetic proxy screen only; not a human listening review; production and export forbidden.")


def require_family_provenance(row):
    from .procedural import FAMILY_MODELS
    p = row.provenance
    family = row.item.family
    expected = FAMILY_MODELS[family]
    if p.get("provenance") == "synthetic" or "generator" in p or "model" in p:
        actual = p.get("model")
        declared = p.get("family")
        if (p.get("generator") != "negative-procedural/2" or actual != expected["generator"]
                or (declared is not None and declared != family)):
            raise ValueError(f"declared family {family!r} mismatches provenance family {declared!r} "
                             f"model {actual!r} generator {p.get('generator')!r}")
        # Older sources share model names (rotor_noise/contact_body). Their
        # parameter region still binds a specific family; absent hooks fail.
        region = p.get("parameters", {}).get("assumption_region")
        if digest(region) != digest(expected["region"]):
            raise ValueError(f"declared family {family!r} mismatches provenance model {actual!r} family parameters")
    elif p.get("family") != family:
        raise ValueError(f"declared family {family!r} mismatches recording provenance family {p.get('family')!r}")


def require_production_metadata(training):
    """Refuse research-only weights at the device packaging boundary."""
    negatives = training.get("negatives", {})
    if (training.get("config", {}).get("proxy_screen") or negatives.get("proxy_screen_only")
            or any(r.get("receipt_kind") == "proxy_screen_only" for r in negatives.get("items", []))):
        raise ValueError("proxy_screen_only weights cannot be exported for production")


def require_realism(row, *, allow_proxy_screen=False):
    from .procedural import TONES, IMPACTS, EVENT_FAMILIES, BED_FAMILIES
    x = np.asarray(row.audio, float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.ptp(x) == 0:
        raise ValueError("confuser audio is empty, nonfinite, silent or constant")
    p = getattr(row, "provenance", {})
    if p.get("training_eligible") is not True:
        raise ValueError("confuser is not training eligible; explicit admission required")
    if row.item.family not in EVENT_FAMILIES + BED_FAMILIES:
        raise ValueError("unknown confuser family; no admission policy")
    require_family_provenance(row)
    receipt = p.get("realism_gate")
    if receipt and receipt.get("kind") == "proxy_screen_only":
        if not allow_proxy_screen:
            raise ValueError("proxy_screen_only receipt requires explicit proxy_screen; production training refused")
        if (receipt.get("audio_sha256") != row.item.sha256 or receipt.get("family") != row.item.family
                or receipt.get("human_listening_review") is not False or not receipt.get("note")
                or p.get("provenance") != "synthetic" or row.item.family not in TONES + IMPACTS):
            raise ValueError("bound synthetic proxy_screen_only receipt required; not a listening review")
        return
    if row.item.family not in TONES + IMPACTS and receipt is None:
        return  # Lead-approved engineering policy; metadata preserves proxy status.
    receipt = receipt or {}
    if (receipt.get("decision") != "pass" or receipt.get("audio_sha256") != row.item.sha256
            or receipt.get("family") != row.item.family
            or receipt.get("reference_kind") != "recorded_family" or not receipt.get("reviewer")):
        raise ValueError("bound recorded-family realism gate receipt required")
    if row.item.family in TONES + IMPACTS and receipt.get("reviewed_label_boundary") is not True:
        raise ValueError("confuser requires a reviewed target/negative boundary")
    for key in ("evidence", "reference"):
        verified_file(receipt.get(key+"_file"), receipt.get(key+"_sha256"))
    if row.item.family in TONES:
        if p.get("provenance") == "synthetic":
            from .tonal import tonal_support, EXCLUSION_FRACTION
            params = p.get("parameters", {})
            partials = np.asarray(params.get("target_partials_hz", ()), float)
            support = tonal_support(row.item.family, float(params.get("base_hz", float("nan"))), row.fs)
            if (not partials.size or not np.isfinite(partials).all() or np.any(partials <= 0)
                    or any(not np.isfinite([lo, hi]).all() or
                           np.any((hi >= partials*(1-EXCLUSION_FRACTION)) &
                                  (lo <= partials*(1+EXCLUSION_FRACTION))) for lo, hi in support)):
                raise ValueError("synthetic tonal confuser requires target partial exclusion")
        require_doorbell_recall(row)


class LockedEventPool:
    """Weighted confusers; unallocated procedural slots remain ordinary background.

    max_procedural_share caps procedural probability among *background-route
    opportunities*, including unchanged recorded backgrounds, not among the
    subset of emitted confusers. Effective probability also multiplies the
    configured p_confuser_given_background. Weights are sampling weights, not
    loss weights or deployment prevalence. Mining caps are relative to base mass.
    """
    def __init__(self, rows, *, plan, expected_sha256, purpose, max_procedural_share=1., allow_proxy_screen=False):
        if purpose not in ("train", "tune"):
            raise ValueError("event pool purpose must be train or tune")
        if not np.isfinite(max_procedural_share) or not 0 <= max_procedural_share <= 1:
            raise ValueError("max_procedural_share must lie in [0,1]")
        self.plan, self.expected_sha256, self.purpose = plan, expected_sha256, purpose
        self.allow_proxy_screen = bool(allow_proxy_screen)
        self.rows = tuple(rows)
        if not self.rows:
            raise ValueError("nonempty authorized event pool required")
        if len({r.item.sha256 for r in self.rows}) != len(self.rows):
            raise ValueError("duplicate confuser content")
        if len({r.item.item_id for r in self.rows}) != len(self.rows):
            raise ValueError("duplicate confuser item")
        for row in self.rows:
            self._authorize(row)
        if len({row.fs for row in self.rows}) != 1:
            raise ValueError("event pool rate mismatch")
        self.fs = self.rows[0].fs
        self.weights = np.array([r.provenance.get("weight", 1.) for r in self.rows], float)
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0):
            raise ValueError("finite positive sampling weights required")
        mined = np.array([bool(r.provenance.get("mined")) for r in self.rows])
        if mined.any():
            base = self.weights[~mined].sum()
            if base <= 0 or self.weights[mined].sum() > .2*base + 1e-12:
                raise ValueError("mined weight exceeds base training budget")
            for field, cap in (("source_group", .05), ("family", .1)):
                for identity in {getattr(r.item, field) for r in self.rows}:
                    mass = sum(w for r, w, m in zip(self.rows, self.weights, mined)
                               if m and getattr(r.item, field) == identity)
                    if mass > cap*base + 1e-12:
                        raise ValueError("mined source/family weight exceeds cap")
        self.probabilities = self.weights/self.weights.sum()
        synthetic = np.array([r.provenance.get("provenance") == "synthetic" for r in self.rows])
        mass = self.probabilities[synthetic].sum()
        if mass > max_procedural_share:
            self.probabilities[synthetic] *= max_procedural_share/mass
            if (~synthetic).any():
                self.probabilities[~synthetic] *= (1-max_procedural_share)/(1-mass)
        self.max_procedural_share = float(max_procedural_share)

    def _authorize(self, row):
        require_realism(row, allow_proxy_screen=self.allow_proxy_screen)
        purpose = self.purpose
        if row.provenance.get("mined"):
            if purpose != "train" or row.provenance.get("role") != "mining":
                raise ValueError("calibration/audit audio cannot be mined")
            review = row.provenance.get("mining_review", {})
            if review.get("decision") != "confirmed_negative" or not review.get("reviewer") or not review.get("notes"):
                raise ValueError("mined confuser requires listening review")
            candidate = row.provenance.get("mining_candidate", {})
            from dataclasses import asdict
            identity = {"item": asdict(row.item), "start": candidate.get("start_sample"),
                        "end": candidate.get("end_sample"), "model": candidate.get("model_sha256"),
                        "split": self.plan.sha256, "audio": candidate.get("audio_sha256")}
            a, b = candidate.get("start_sample", -1), candidate.get("end_sample", -1)
            if (candidate.get("review_id") != digest(identity) or candidate.get("item") != asdict(row.item)
                    or candidate.get("split_sha256") != self.plan.sha256 or candidate.get("fs") != row.fs
                    or not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(row.audio)
                    or audio_hash(row.audio[a:b]) != candidate.get("audio_sha256")
                    or candidate.get("weight") != row.provenance.get("weight")):
                raise ValueError("mined window identity, units or weight changed")
            purpose = "mine"
        authorize_audio(row.audio, row.item, plan=self.plan, expected_sha256=self.expected_sha256, purpose=purpose)

    def metadata(self):
        return dict(policy=("proxy screen only; not a human listening review; production/export forbidden"
                            if self.allow_proxy_screen else "non-gated engineering admission; tonal/impact reviewed"),
            proxy_screen_only=self.allow_proxy_screen,
            procedural_cap_basis="background-route opportunities, including plain background",
            max_procedural_share=self.max_procedural_share,
            plain_background_probability=max(0., float(1-self.probabilities.sum())),
            split_sha256=self.expected_sha256,
            families=[dict(family=f, count=sum(r.item.family == f for r in self.rows),
                weight=float(sum(w for r,w in zip(self.rows,self.weights) if r.item.family == f)),
                probability=float(sum(p for r,p in zip(self.rows,self.probabilities) if r.item.family == f)))
                for f in sorted({r.item.family for r in self.rows})],
            items=[dict(item_id=r.item.item_id, source_group=r.item.source_group, family=r.item.family,
                        audio_sha256=r.item.sha256, weight=float(w), probability=float(p),
                        room_route=r.provenance.get("room_route"),
                        receipt_kind=r.provenance.get("realism_gate", {}).get("kind"),
                        provenance=r.provenance.get("provenance"), mined=bool(r.provenance.get("mined")))
                   for r,w,p in zip(self.rows,self.weights,self.probabilities)])

    def window(self, rng, n, *, fs, purpose):
        from sensorsim.recognizer import _audible_span, _event_window
        if self.fs != fs or self.purpose != purpose:
            raise ValueError("event pool rate or purpose mismatch")
        k = int(np.searchsorted(np.cumsum(self.probabilities), rng.random(), side="right"))
        if k == len(self.rows):
            return None
        row = self.rows[k]
        self._authorize(row)
        audio = np.asarray(row.audio, float)
        if row.provenance.get("mined"):
            candidate = row.provenance["mining_candidate"]
            audio = audio[candidate["start_sample"]:candidate["end_sample"]]
        return _event_window(audio, _audible_span(audio, fs), n, rng)


def build_pool(path, expected_sha256, *, fs, max_procedural_share, allowed_room_draws, allow_proxy_screen=False):
    """Load a frozen manifest of receiver-domain sources; never render unknown rooms.

    JSON schema negative-training/1: split_lock, split_file_sha256, rows with
    file/file_sha256/item/provenance/room_draw_id. Paths are relative to manifest.
    Separate source families can be reserved for mining and sealed audit in the
    split lock. No audit audio is opened here.
    """
    from .mining import PoolItem
    from sensorsim.recognizer import read_audio
    if path is None:
        raise ValueError("negatives are on: supply negative_manifest and its SHA-256, or explicitly set p_confuser_given_background: 0 for a diagnostic")
    path = Path(path)
    data = json.loads(verified_file(path, expected_sha256))
    if data.get("schema") != "negative-training/1":
        raise ValueError("unsupported negative training manifest")
    plan = SplitPlan.from_audit_lock(path.parent/data["split_lock"], expected_file_sha256=data["split_file_sha256"])
    rows = []
    for entry in data["rows"]:
        p = dict(entry["provenance"])
        item = Item(**entry["item"])
        if p.get("mined"):
            bank = json.loads(verified_file(path.parent/entry["mining_bank_file"], entry["mining_bank_sha256"]))
            candidates = [c for c in bank if c["review_id"] == entry["review_id"]]
            if len(candidates) != 1 or candidates[0].get("label") != "background":
                raise ValueError("unique reviewed mining addition required")
            candidate = candidates[0]
            p.update(mining_candidate=candidate, weight=candidate["weight"], mining_review=candidate["review"])

        plan.require(item, purpose="mine" if p.get("mined") else "train")
        if (entry.get("room_draw_id") not in allowed_room_draws or p.get("acoustic_stage") != "receiver"):
            raise ValueError("confuser must use an identified target room draw in receiver domain")
        if p.get("license_id") not in ("Apache-2.0", "user-owned", "CC0-1.0", "CC-BY-4.0"):
            raise ValueError("uncleared confuser licence")
        if p.get("license_id") != "Apache-2.0":
            verified_file(path.parent/p["licence_file"], p["licence_sha256"])
        verified_file(path.parent/entry["file"], entry["file_sha256"])
        for field in ("realism_gate", "doorbell_recall"):
            if field in p:
                p[field] = dict(p[field])
                for key in ("evidence_file", "reference_file", "file"):
                    if key in p[field]: p[field][key] = str(path.parent/p[field][key])
        from scipy.io import wavfile
        rate, raw_audio = wavfile.read(path.parent/entry["file"])
        if rate != fs or raw_audio.ndim != 1:
            raise ValueError("confuser sample rate/channel units mismatch")
        x = read_audio(path.parent/entry["file"], fs)
        rows.append(PoolItem(item, x, fs, p))
    return LockedEventPool(rows, plan=plan, expected_sha256=plan.sha256,
                           purpose="train", max_procedural_share=max_procedural_share, allow_proxy_screen=allow_proxy_screen)


def draw_confuser(corpus, rng, cfg, n):
    """Draw only after choosing background; zero preserves the former RNG stream."""
    probability = float(cfg.get("p_confuser_given_background", 0.))
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("p_confuser_given_background must lie in [0,1]")
    if probability == 0:
        return None
    pool = getattr(corpus, "negative_pool", None)
    if pool is None:
        raise ValueError("nonzero confuser probability requires a reviewed negative pool")
    if getattr(pool, "allow_proxy_screen", False) and cfg.get("proxy_screen") is not True:
        raise ValueError("proxy_screen_only pool is forbidden in production training")
    # Isolate extra negative draws so paired arms keep the same target/background
    # selection and mixer draws. The in-distribution mixer reserves its level draw
    # even for silence; stress mixing does not offer this pairing guarantee.
    from sensorsim.training_dynamics import auxiliary_rng
    confuser_rng = auxiliary_rng(rng, "confuser")
    if confuser_rng.random() >= probability:
        return None
    result = pool.window(confuser_rng, n, fs=cfg["fs"], purpose=corpus.pool_purpose)
    if result is not None and hasattr(corpus, "confuser_draws"):
        corpus.confuser_draws += 1
    return result
