"""Held-out take recall through edge.detect with exact binomial intervals."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, chi2


def clopper_pearson(successes: int, trials: int, confidence: float = .95) -> tuple[float | None, float | None]:
    """Two-sided exact Clopper-Pearson interval for a binomial proportion."""
    if isinstance(successes, bool) or isinstance(trials, bool) or not isinstance(successes, (int, np.integer)) \
            or not isinstance(trials, (int, np.integer)):
        raise TypeError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if trials == 0:
        return None, None
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials-successes+1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0-tail, successes+1, trials-successes))
    return lower, upper


def _is_negative(label: str) -> bool:
    value = label.casefold()
    return value in {"background", "negative"} or value.startswith("negative_")


def evaluate_recall(model, takes, *, data_kind: str = "measured", confidence: float = .95) -> dict:
    """Score complete takes by class with the model's stored thresholds and exact edge.detect.

    Aggregate WAV outcomes by the declared performance group. The headline is all-hit
    recall; any-hit recall is also shown. Repeated files do not create independent trials.
    """
    from sensorsim import edge

    if data_kind not in {"measured", "proxy"}:
        raise ValueError("data_kind must be 'measured' or 'proxy'")
    rows = []
    class_groups = defaultdict(lambda: defaultdict(list))
    detected_groups = defaultdict(lambda: defaultdict(set))
    negative_durations = []
    exposure_s = 0.0
    false_events = 0
    for index, take in enumerate(takes):
        group = str(take.get("group", take.get("group_id", take.get("file", take.get("path", index)))))
        label = str(take["label"])
        audio = take.get("audio")
        if audio is None:
            path = Path(take["path"])
            audio = edge.read_wav(str(path), int(model.meta["features"]["fs"]))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        fs = int(model.meta["features"]["fs"])
        events = edge.detect(model, audio)  # stored edge thresholds, smoothing and hysteresis
        detected = sorted({str(event["label"]) for event in events})
        event_counts = Counter(str(event['label']) for event in events)
        wrong = sorted(set(detected) - {label})
        negative = _is_negative(label)
        if negative:
            event_count = len(events)
            false_events += event_count
            exposure_s += len(audio) / fs
            negative_durations.append(len(audio) / fs)
            hit = None
        else:
            event_count = len(events)
            hit = label in detected
            class_groups[label][group].append(hit)
            detected_groups[label][group].update(detected)
        rows.append({
            "file": str(take.get("file", Path(take.get("path", "take")).name)),
            "label": label, "group": group,
            "data_kind": data_kind, "negative": negative, "hit": hit,
            "detected_labels": detected, "confirmed_event_count": int(event_count),
            "confirmed_events_by_class": {c: event_counts[c] for c in sorted(set(model.classes) | set(detected))},
            "wrong_class_labels": wrong,
            "wrong_class_event_count": sum(event_counts[c] for c in wrong),
            "exposure_seconds": float(len(audio) / fs),
        })
    per_class = {}
    for label, groups in sorted(class_groups.items()):
        hits, trials = sum(all(v) for v in groups.values()), len(groups)
        any_hits = sum(any(v) for v in groups.values())
        lower, upper = clopper_pearson(hits, trials, confidence)
        per_class[label] = {"hits": int(hits), "takes": int(trials), "recall": float(hits/trials),
                            "confidence": float(confidence), "interval": [lower, upper],
                            "data_kind": data_kind, "group_rule": "all-hit",
                            "recordings": sum(len(v) for v in groups.values()),
                            "any_hit": {"hits": any_hits, "recall": any_hits/trials,
                                        "interval": list(clopper_pearson(any_hits, trials, confidence))}}
        recordings = [r for r in rows if r['label'] == label and not r['negative']]
        observed = detected_groups[label]
        wrong_groups = sum(bool(classes - {label}) for classes in observed.values())
        per_class[label].update(
            wrong_class_takes=wrong_groups,
            wrong_class_take_fraction=wrong_groups/trials,
            wrong_class_interval=list(clopper_pearson(wrong_groups, trials, confidence)),
            wrong_class_group_rule='any wrong-class detection in any recording in the group',
            wrong_class_recordings=sum(bool(r['wrong_class_labels']) for r in recordings),
            wrong_class_events=sum(r['wrong_class_event_count'] for r in recordings),
            both_correct_and_wrong_takes=sum(label in classes and bool(classes-{label})
                                             for classes in observed.values()))
    columns = sorted(set(model.classes) | {c for r in rows for c in r['detected_labels']})
    confusion = {
        'unit': 'independent take groups with any detection of each class',
        'group_rule': 'union of detected labels across recordings; each group counts once per cell',
        'multilabel': True,
        'note': 'A group can occupy several columns; row sums can exceed the number of groups. '
                'Diagonal is any-hit; headline recall remains all-hit.',
        'columns': columns,
        'counts': {label: {c: sum(c in classes for classes in groups.values()) for c in columns}
                   for label, groups in sorted(detected_groups.items())},
        'no_detection': {label: sum(not classes for classes in groups.values())
                         for label, groups in sorted(detected_groups.items())},
    }
    false_alarm = poisson_exposure(false_events, exposure_s, data_kind=data_kind, confidence=confidence,
                                   longest_continuous_s=max(negative_durations, default=0.))
    return {
        "status": "ok" if per_class else "inconclusive",
        "reason": None if per_class else "no labelled target takes were supplied",
        "data_kind": data_kind,
        "interval_method": "Clopper-Pearson exact two-sided",
        "confidence": float(confidence),
        "detector_thresholds": model.meta["detector"],
        "per_class": per_class,
        "confusion_matrix": confusion,
        "negative_exposure_seconds": float(exposure_s),
        "negative_confirmed_events": int(false_events),
        "false_events_per_hour": false_alarm["events_per_hour"],
        "false_alarm": false_alarm,
        "takes": rows,
    }


def poisson_exposure(count, exposure_s, *, data_kind, confidence=.95, longest_continuous_s=None):
    """Exact Poisson limits, with WP2's one-sided chi-square upper-bound convention.

    The programme requires 30 minutes of continuous target-free audio before a rate
    is shown. Short clips remain a count/exposure smoke test, never a measured rate.
    This is a minimum evidence gate, not a claim of deployment representativeness.
    """
    minimum = 1800.
    longest = exposure_s if longest_continuous_s is None else longest_continuous_s
    enough = exposure_s >= minimum and longest >= minimum
    hours = exposure_s/3600.
    alpha = 1-confidence
    interval = [None, None]
    upper = None
    if enough:
        interval = [float(.5*chi2.ppf(alpha/2, 2*count)/hours) if count else 0.,
                    float(.5*chi2.ppf(1-alpha/2, 2*(count+1))/hours)]
        upper = float(.5*chi2.ppf(confidence, 2*(count+1))/hours)
    return {'status': 'ok' if enough else 'inconclusive', 'data_kind': data_kind,
            'label': ('measured' if data_kind == 'measured' else 'proxy result') if enough else 'smoke test, not FA/hour',
            'events': count, 'exposure_seconds': exposure_s, 'longest_continuous_seconds': longest,
            'minimum_continuous_seconds': minimum, 'confidence': confidence,
            'events_per_hour': count/hours if enough else None,
            'interval_95_per_hour': interval, 'upper_95_per_hour': upper,
            'interval_method': 'exact Poisson chi-square; two-sided interval and one-sided upper bound'}
