# SPDX-License-Identifier: Apache-2.0
"""Development-only joint checkpoint/threshold/tracker selection.

A frozen manifest identifies target parents, confuser parents and background
sessions. The sealed audit is an identity inventory only: its audio is never
loaded by training. Repeated views do not manufacture independent groups.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from sensorsim import edge
from sensorsim.negatives.integration import verified_file
from sensorsim.negatives.composer import audio_hash

DEFAULT_GRID = [dict(on=on, smooth_hops=smooth, min_hops=minimum)
                for on in (.5, .7, .85, .95, .995) for smooth, minimum in ((1, 1), (3, 2))]


def source_group(row):
    value = row.get("source_group") or row.get("dry_source", {}).get("source_group")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source parent identity is required; filenames are not parents")
    return value


def corpus_identities(manifest, fs, *, split=None):
    from sensorsim.recognizer import _load_manifest, read_audio
    rows, rate = _load_manifest(Path(manifest))
    if rate != fs:
        raise ValueError("identity manifest rate mismatch")
    identities = []
    for row in rows:
        if split is None or row["split"] == split:
            identities.append(dict(source_group=source_group(row),
                audio_sha256=audio_hash(read_audio(Path(manifest).parent/row["file"], fs)),
                parent_audio_sha256=[row["dry_source"][key] for key in
                    ("content_sha256", "normalized_content_sha256") if row.get("dry_source", {}).get(key)]))
    return identities


def _identity_sets(rows):
    groups, hashes = set(), set()
    for row in rows:
        groups.add(source_group(row))
        groups.update(row.get("background_groups", []))
        for value in [row.get("audio_sha256"), *row.get("parent_audio_sha256", [])]:
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("decoded content and parent SHA-256 identities required")
            hashes.add(value)
    return groups, hashes


def load_panel(path, expected_sha256, *, classes, fs, excluded, min_groups=10):
    """Validate the complete panel before any checkpoint is trained or scored.

    Each development row has one source_group, background_groups, file/container
    and decoded hashes, parent_audio_sha256, kind, and evidence_domain. Audit
    inventory has identities/hashes but no accessible audio is required here.
    Parent metadata still requires independent protocol review; no signal-only
    check proves that two differently encoded captures have different parents.
    """
    from sensorsim.recognizer import read_audio
    if path is None:
        raise ValueError("selection needs a frozen selection_panel and SHA-256; diagnostic_final is an explicit unqualified alternative")
    path = Path(path)
    data = json.loads(verified_file(path, expected_sha256))
    if data.get("schema") != "training-panel/1" or data.get("role") != "development":
        raise ValueError("selection must use a development panel")
    rows, audit = data.get("rows", []), data.get("sealed_audit", [])
    if not rows or not audit:
        raise ValueError("nonempty development and sealed audit inventories required")
    if min_groups < 1:
        raise ValueError("positive independent group count required")
    seen_groups, seen_hashes = set(), set()
    audit_groups, audit_hashes = _identity_sets(audit)
    excluded_groups, excluded_hashes = _identity_sets(excluded)
    if audit_groups & excluded_groups or audit_hashes & excluded_hashes:
        raise ValueError("sealed audit overlaps training/mining")
    for row in rows:
        if row.get("role", "development") != "development" or row.get("kind") not in ("target", "nuisance", "background"):
            raise ValueError("unknown panel role/kind")
        if row.get("evidence_domain") not in ("synthetic proxy", "measured-RIR proxy", "deployment recording"):
            raise ValueError("explicit panel evidence domain required")
        if not row.get("background_groups") or not row.get("parent_audio_sha256"):
            raise ValueError("background and source parent identities required")
        group = source_group(row)
        if group in seen_groups or row.get("audio_sha256") in seen_hashes:
            raise ValueError("duplicate panel content/independent group")
        seen_groups.add(group); seen_hashes.add(row.get("audio_sha256"))
        verified_file(path.parent/row["audio"], row["file_sha256"])
        from scipy.io import wavfile
        rate, raw_audio = wavfile.read(path.parent/row["audio"])
        if rate != fs or raw_audio.ndim != 1:
            raise ValueError("panel sample rate/channel units mismatch")
        x = read_audio(path.parent/row["audio"], fs)
        if x.size < fs or not np.isfinite(x).all() or np.ptp(x) == 0:
            raise ValueError("empty/constant/nonfinite panel audio")
        if audio_hash(x) != row["audio_sha256"]:
            raise ValueError("decoded panel content changed")
        if row["kind"] == "target":
            if row.get("label") not in classes or not 0 <= row["start_s"] < row["end_s"] <= len(x)/fs:
                raise ValueError("invalid target class/interval")
        elif row["kind"] == "nuisance" and not row.get("family"):
            raise ValueError("nuisance family required")
    groups, hashes = _identity_sets(rows)
    if groups & (excluded_groups | audit_groups) or hashes & (excluded_hashes | audit_hashes):
        raise ValueError("development overlaps training/mining or sealed audit parents/content")
    for c in classes:
        if sum(r["kind"] == "target" and r["label"] == c for r in rows) < min_groups:
            raise ValueError(f"insufficient independent development parents for {c}")
    for kind in ("nuisance", "background"):
        if not any(r["kind"] == kind for r in rows):
            raise ValueError(f"missing required {kind} development data")
    return dict(data, root=str(path.parent), sha256=expected_sha256)


def validate_policy(detector, classes):
    for key in ("on", "off_ratio", "smooth_hops", "min_hops"):
        value = detector.get(key)
        if isinstance(value, dict):
            if set(value) != set(classes): raise ValueError("policy class mapping is incomplete")
            values = list(value.values())
        else:
            values = [value]
        try:
            values = np.asarray(values, float)
        except (ValueError, TypeError) as exc:
            raise ValueError("numeric detector policy required") from exc
        if not np.isfinite(values).all(): raise ValueError("finite detector policy required")
        if key in ("on", "off_ratio"):
            if np.any((values <= 0) | (values > 1)): raise ValueError("detector thresholds/ratios must be in (0,1]")
        elif np.any(values < 1) or np.any(values != np.floor(values)):
            raise ValueError("tracker lengths must be positive integer hops")


def score_policy(sequences, rows, classes, detector, *, window_s, trigger_budget, min_recall):
    """Replay the exact deployed tracker; count starts AND confirmed events.

    The budget is a total confuser-parent trigger fraction, across all classes,
    plus zero background starts. Recall has an explicit per-class lower floor.
    This is development feasibility, not an interval/FA-per-hour guarantee.
    """
    if len(sequences) != len(rows) or not rows or not classes:
        raise ValueError("empty or incomplete score panel")
    if not 0 <= trigger_budget <= 1 or not 0 <= min_recall <= 1:
        raise ValueError("selection budget and recall are fractions in [0,1]")
    validate_policy(detector, classes)
    hop = float(detector["hop_s"])
    if not np.isfinite(hop) or hop <= 0:
        raise ValueError("hop_s must be finite positive seconds")
    outcomes=[]
    for scores, row in zip(sequences, rows):
        scores=np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != len(classes) or not len(scores) or not np.isfinite(scores).all() or np.any((scores < 0)|(scores > 1)):
            raise ValueError("finite nonempty probability sequences required")
        tracker=edge.Tracker(classes, detector, window_s)
        events=[]
        for i,s in enumerate(scores):
            events.extend(tracker.step(s,window_s+i*hop))
        events.extend(tracker.flush())
        starts=[e for e in events if e["event"]=="start"]
        ends=[e for e in events if e["event"]=="end" and e["confirmed"]]
        detected=any(e["label"]==row.get("label") and e["end_s"]>row.get("start_s",0) and
                     e["start_s"]<row.get("end_s",float("inf")) for e in ends)
        outcomes.append(dict(source_group=row["source_group"],kind=row["kind"],label=row.get("label"),
            family=row.get("family"),detected=detected,trigger=bool(ends),starts=len(starts),confirmed=len(ends),
            start_by_class={c:sum(e["label"]==c for e in starts) for c in classes},
            confirmed_by_class={c:sum(e["label"]==c for e in ends) for c in classes}))
    recall={}; counts={}
    for c in classes:
        target=[r for r in outcomes if r["kind"]=="target" and r["label"]==c]
        if not target: raise ValueError("missing target class denominator")
        hits=sum(r["detected"] for r in target)
        recall[c]=hits/len(target); counts[c]=dict(detected=hits,groups=len(target))
    nuis=[r for r in outcomes if r["kind"]=="nuisance"]
    bg=[r for r in outcomes if r["kind"]=="background"]
    if not nuis or not bg: raise ValueError("missing nuisance/background denominator")
    triggers=sum(r["trigger"] for r in nuis); starts=sum(r["starts"] for r in nuis)
    confirms=sum(r["confirmed"] for r in nuis); background_starts=sum(r["starts"] for r in bg)
    matrix={}
    for r in outcomes:
        key=r["label"] if r["kind"]=="target" else r["family"] or "background"
        cell=matrix.setdefault(key,dict.fromkeys(classes,0))
        for c in classes: cell[c] += int(r["confirmed_by_class"][c] > 0)
    return dict(feasible=all(v>=min_recall for v in recall.values()) and triggers/len(nuis)<=trigger_budget and background_starts==0,
        recall=recall,recall_counts=counts,nuisance_trigger_groups=triggers,nuisance_groups=len(nuis),
        nuisance_starts=starts,nuisance_confirmed=confirms,background_starts=background_starts,
        start_per_100_contexts=100*starts/len(nuis),confirmed_per_100_contexts=100*confirms/len(nuis),
        class_swap_matrix=matrix,outcomes=outcomes)


def candidate_key(row):
    recalls=list(row["recall"].values())
    return (min(recalls),float(np.mean(recalls)),-row["nuisance_trigger_groups"],
            -row["nuisance_starts"],row["step"],-row["policy_index"])


def choose_candidate(rows):
    return max((r for r in rows if r["feasible"]),key=candidate_key,default=None)


def score_checkpoint(model_path, panel, cfg, step):
    from sensorsim.recognizer import read_audio
    runtime=edge.Model(str(model_path))
    hop=cfg["detector"]["hop_s"]
    sequences=[edge.window_scores(runtime,read_audio(Path(panel["root"])/r["audio"],cfg["fs"]),hop,batch=32) for r in panel["rows"]]
    cache=Path(model_path).with_suffix(".panel_scores.npz")
    np.savez_compressed(cache,**{f"context_{i}":s for i,s in enumerate(sequences)})
    grid=cfg["selection_grid"]
    if not grid or len(grid)>256: raise ValueError("selection grid must have 1..256 policies")
    results=[]
    for i,policy in enumerate(grid):
        if set(policy)-{"on","smooth_hops","min_hops","off_ratio"}:
            raise ValueError("unsupported selection policy field")
        detector={**cfg["detector"],**policy}
        detector.setdefault("on",detector["default_on"])
        metrics=score_policy(sequences,panel["rows"],runtime.classes,detector,window_s=cfg["window_s"],
            trigger_budget=cfg["selection_trigger_budget"],min_recall=cfg["selection_min_recall"])
        results.append(dict(step=step,policy_index=i,detector=detector,**metrics))
    return dict(step=step,policies=results,scores_file=cache.name,
                scores_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                model_file=Path(model_path).name,model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest())
