"""Chamber and exact-digital intake; reports precede any admission to rendering."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .catalog import SourceTake, mono, content_hash, as_dry_clip
from .kinds import source_kind


@dataclass(frozen=True)
class IntakeResult:
    take: SourceTake | None
    report: dict


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_session_string(session):
    from datetime import date
    if isinstance(session, date):
        raise ValueError("quote the session string in YAML, for example session: '2026-10-09'")


def _identity(metadata):
    allowed = {'label', 'source_group', 'object_id', 'session', 'capture_route', 'evidence_label'}
    if set(metadata) != allowed:
        raise ValueError(f'mandatory source provenance fields: {sorted(allowed)}')
    require_session_string(metadata['session'])
    # Validate even when a capture is later rejected or has no detectable event.
    SourceTake(samples=np.array([.1, 0.]), fs=8000, **metadata)
    return dict(metadata)


def _approval(report, approval):
    # Bind decisions to immutable content, bounds, identity and settings, not
    # floating diagnostics whose last bits can change with BLAS implementations.
    keys = ('schema_version', 'label', 'source_group', 'object_id', 'session', 'capture_route',
            'evidence_label', 'fs', 'input_content_sha256', 'crop_content_sha256', 'crop_samples',
            'event_samples', 'quiet_window_s', 'detection', 'qa_settings', 'session_reference_hash',
            'rejected', 'review', 'kind')
    report['approval_key'] = _digest({k: report.get(k) for k in keys})
    if approval is not None:
        if (set(approval) != {'reviewer', 'decision', 'approval_key'} or
                not isinstance(approval['reviewer'], str) or not approval['reviewer'].strip() or
                approval['decision'] != 'approve' or approval['approval_key'] != report['approval_key']):
            raise ValueError('listening approval must name a reviewer and match the current approval_key')
        report['listening_approval'] = dict(approval)
        if not report['rejected']:
            report['status'] = 'pass'
    return report['status'] == 'pass'


def intake_known_digital(samples, fs, *, metadata, loudspeaker_played=False, loudspeaker=None,
                         loudspeaker_position=None, digital_origin=None, approval=None, kind=None):
    from .speaker import playback_contract
    from .digital_qa import assess_digital
    identity = _identity(metadata)
    if identity['capture_route'] != 'known_digital':
        raise ValueError('known digital intake requires capture_route=known_digital')
    position = playback_contract(identity['capture_route'], loudspeaker_played,
                                 loudspeaker_position, response=loudspeaker)
    x = mono(samples, fs)
    parent = content_hash(x, fs)
    report = dict(schema_version='source-crop-qa/2', **identity, input_content_sha256=parent,
        fs=int(fs), crop_samples=[0, len(x)], crop_content_sha256=parent,
        result_label=identity['evidence_label'], **assess_digital(x, fs, identity['label'], kind=kind),
        qa_settings=dict(digital_origin=digital_origin, loudspeaker_played=loudspeaker_played,
            loudspeaker_position=position, loudspeaker=loudspeaker.metadata() if loudspeaker else None))
    report['late_decay']['evidence_label'] = identity['evidence_label']
    if not _approval(report, approval):
        return IntakeResult(None, report)
    speaker = {'applied': False, 'status': 'unknown'}
    if loudspeaker is not None:
        x = loudspeaker.apply(x, fs)
        speaker = loudspeaker.metadata()
        report['input_crop_content_sha256'] = report['crop_content_sha256']
        report['crop_content_sha256'] = content_hash(x, fs)
    take = SourceTake(samples=x, fs=fs, parent_content_sha256=parent, **identity,
        details=dict(qa=report, digital_origin=digital_origin, loudspeaker_played=loudspeaker_played,
                     loudspeaker_position=position, loudspeaker_response=speaker))
    return IntakeResult(take, report)


def intake_chamber(samples, fs, *, metadata, quiet_window_s=(0., .2), event_index=None,
                   event_window_s=None, merge_gap_s=None, event_merge_gap_s=None, post_roll_s=.5,
                   dryness_window_s=None, sir_db=None, contaminated=False, approval=None,
                   loudspeaker_played=False, loudspeaker=None, session_reference=None, kind=None,
                   heterogeneous_hits=False):
    """Detect with the existing event detector; never normalize or repair bad crops.

    quiet_window_s declares same-gain target-free pre/post take audio. It is not
    a separate background recording. Listening approval contains reviewer,
    decision='approve', and the report's approval_key. Hard rejects cannot be
    overridden. No approval means a pending crop, not an assumed clean source.
    """
    from sensorsim.acoustics.room.sweep_detect import detect_events, select_event_region, DetectionError
    from sensorsim.acoustics.room.capture_qa import assess_source_crop
    identity = _identity(metadata)
    kind = source_kind(identity['label'], kind)
    if identity['capture_route'] != 'near_dry':
        raise ValueError('chamber intake requires near_dry; in_situ cannot enter the renderer')
    from .speaker import playback_contract
    from .decay import qualify_session
    playback_contract('near_dry', loudspeaker_played, None, response=loudspeaker)
    if merge_gap_s is not None and event_merge_gap_s is not None:
        raise ValueError('use merge_gap_s or its alias event_merge_gap_s, not both')
    if merge_gap_s is None:
        merge_gap_s = event_merge_gap_s
    if merge_gap_s is None:
        merge_gap_s = .7 if kind == 'knock' else .15
    reference = qualify_session(session_reference, session=identity['session']) if session_reference else None
    x = mono(samples, fs)
    q = np.asarray(quiet_window_s, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all() or not 0 <= q[0] < q[1] <= len(x)/fs:
        raise ValueError('quiet_window_s must be inside the recording')
    begin, end = np.round(q*fs).astype(int)
    noise = mono(x[begin:end], fs)
    base = dict(**identity, kind=kind, fs=int(fs), input_content_sha256=content_hash(x, fs),
        quiet_window_s=list(quiet_window_s), result_label=identity['evidence_label'],
        detection=dict(merge_gap_s=merge_gap_s, post_roll_s=post_roll_s,
                       event_index=event_index, event_window_s=list(event_window_s) if event_window_s is not None else None),
        session_reference_hash=reference['content_sha256'] if reference else None,
        qa_settings=dict(dryness_window_s=list(dryness_window_s) if dryness_window_s is not None else None, sir_db=sir_db, contaminated=contaminated,
                         **({'heterogeneous_hits': True} if heterogeneous_hits is True else {}),
                         session_reference={k: v for k, v in reference.items() if k != 'decay'} if reference else None))
    if np.max(abs(x)) < 1/32768:
        return IntakeResult(None, dict(base, schema_version='source-crop-qa/2', status='reject',
                                       rejected=['silent_or_sub_lsb'], event_count=0))
    regions = detect_events(x, fs, noise, merge_gap_s=merge_gap_s, post_roll_s=post_roll_s)
    try:
        region = select_event_region(regions, fs, event_index=event_index, event_window_s=list(event_window_s) if event_window_s is not None else None)
    except DetectionError as exc:
        return IntakeResult(None, dict(base, schema_version='source-crop-qa/2', status='review',
                                       reason=str(exc)+'; for a knock sequence raise merge_gap_s (event_merge_gap_s alias) to span the gaps before selecting an event', event_count=len(regions)))
    sequence = {}
    if kind == 'knock':
        from .sequence import screen_knock_sequence
        original = region
        region, sequence = screen_knock_sequence(x, fs, noise, region, post_roll_s=post_roll_s)
        if heterogeneous_hits is True:
            region = original  # retain the alleged final frame hit for listening
        regions = [region if r == original else r for r in regions]
    qa = assess_source_crop(x, fs, region, regions, noise,
                            contaminated=contaminated, heterogeneous_hits=heterogeneous_hits,
                            incompatible_rises_samples=sequence.get('incompatible_rises_samples', ()),
                            sir_db=sir_db, dryness_window_s=dryness_window_s,
                            label=identity['label'], kind=kind, session_reference=reference)
    if np.any(x >= 32767/32768) or np.any(x <= -1):
        if 'clipped' not in qa['rejected']:
            qa['rejected'].append('recording_clipped')
        qa['status'] = 'reject'
    if any(r.onset_sample < end and r.offset_sample > begin for r in regions):
        qa['rejected'].append('quiet_window_contains_event')
        qa['status'] = 'reject'
    crop = x[region.segment_start:region.segment_stop]
    report = {**base, **qa, **sequence, 'crop_content_sha256': content_hash(crop, fs), 'event_count': len(regions)}
    report['schema_version'] = 'source-crop-qa/2'
    report['late_decay']['evidence_label'] = identity['evidence_label']
    if not _approval(report, approval):
        return IntakeResult(None, report)
    return IntakeResult(SourceTake(samples=crop, fs=fs, **identity,
        parent_content_sha256=base['input_content_sha256'],
        details=dict(qa=report, loudspeaker_played=False,
                     loudspeaker_response={'applied': False, 'status': 'not applicable'})), report)


def load_project_sources(rows, targets, report_path, excluded_roots=()):
    """project.dataset.sources rows; every referenced WAV stays under targets/.

    Configuration carries provenance and response arrays, so the project's
    existing dataset cache key observes edits to metadata, QA or speaker data.
    All input WAVs live in the hashed targets tree. The dataset owns train/val
    assignment before rendering. Use split_sources for an explicit RH plan.
    """
    from sensorsim.acoustics.room.ingest import read_recording_audio
    from sensorsim.core.serialization import atomic_json
    from .speaker import LoudspeakerResponse
    if not isinstance(rows, list) or not rows:
        raise ValueError('dataset.sources must be a nonempty list of provenance rows')
    targets = Path(targets).resolve()
    reports, takes, failures, paths, references = [], [], [], set(), set()
    for row in rows:
        row = dict(row)
        metadata = {k: row.pop(k) for k in ('label', 'source_group', 'object_id', 'session',
                    'capture_route', 'evidence_label') if k in row}
        _identity(metadata)
        if metadata['capture_route'] == 'in_situ':
            raise ValueError('in_situ takes are real evaluation/training anchors, never room renderer inputs')
        path = (targets / row.pop('file')).resolve()
        if not path.is_relative_to(targets) or path.suffix.lower() != '.wav':
            raise ValueError('source files must be WAVs inside the hashed targets directory')
        if 'room_recordings' in path.relative_to(targets).parts or path.stem.lower().endswith('__room'):
            raise ValueError('room_recordings/ and *__room.wav are in_situ, never dry sources')
        if 'background' in path.relative_to(targets).parts:
            raise ValueError('background/ contains negatives, not target sources')
        if (metadata['capture_route'] == 'known_digital' or row.get('loudspeaker_played')
                or row.get('loudspeaker') is not None or 'played' in path.relative_to(targets).parts):
            raise ValueError('playback recordings are for room calibration and comparison '
                             '(the WP4 known-source path); targets are rendered at their true sites. '
                             'known_digital/playback rows cannot be project training targets')
        if not is_target_path(path, targets, excluded_roots):
            raise ValueError('reserved project-generated/evaluation folders cannot supply target sources')
        reference = row.get('session_reference')
        if reference is not None:
            reference = dict(reference)
            require_session_string(reference.get('session'))
            rp = (targets / reference.pop('file')).resolve()
            if (not rp.is_relative_to(targets) or rp.suffix.lower() != '.wav'
                    or not is_target_path(rp, targets, excluded_roots) or rp.stem.lower().endswith('__room')):
                raise ValueError('session_reference must be a chamber WAV inside hashed targets')
            rx, rfs = read_recording_audio(rp)
            if rx.shape[1] != 1:
                raise ValueError('session_reference requires mono')
            row['session_reference'] = dict(reference, samples=rx[:, 0], fs=rfs)
            references.add(rp)
        paths.add(path)
        audio, fs = read_recording_audio(path)
        if audio.shape[1] != 1:
            raise ValueError('chamber/digital intake requires mono; no implicit mixdown')
        speaker = row.pop('loudspeaker', None)
        if speaker is not None:
            speaker = dict(speaker)
            if 'frequency_hz' in speaker:
                row['loudspeaker'] = LoudspeakerResponse.from_magnitude(fs=fs, **speaker)
            else:
                row['loudspeaker'] = LoudspeakerResponse(fs=fs, **speaker)
        fn = intake_chamber if metadata['capture_route'] == 'near_dry' else intake_known_digital
        result = fn(audio[:, 0], fs, metadata=metadata, **row)
        reports.append(dict(file=path.relative_to(targets).as_posix(), **result.report))
        if result.take is None:
            failures.append(f"{path.name}: {result.report['status']}")
        else:
            takes.append(result.take)
    unlisted = {p.resolve() for p in targets.rglob('*') if p.suffix.lower() in ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aif', '.aiff')
                and is_target_path(p, targets, excluded_roots)} - paths - references
    if unlisted:
        failures.append('audio files lack mandatory provenance in dataset.sources: ' + ', '.join(sorted(p.name for p in unlisted)))
    # Different excerpts of identical parent recordings must use the same take group.
    parent_groups = {}
    for take in takes:
        old = parent_groups.setdefault(take.parent_content_sha256, take.source_group)
        if old != take.source_group:
            failures.append('identical parent recordings must declare the same source_group')
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, dict(schema_version='source-intake/1', clips=reports,
                                  status='reject' if failures else 'pass', failures=failures))
    if failures:
        raise ValueError('source intake stopped; inspect source_qa.json: ' + '; '.join(failures))
    return [as_dry_clip(t) for t in takes]


def is_target_path(path, targets, excluded_roots=()):
    """Exclude reserved roles, generated output and configured project folders.

    Resolve symlinks too: a dry-looking alias does not change an audio file's role.
    Explicit source rows use the same boundary as undeclared-file discovery.
    """
    path, targets = Path(path).resolve(), Path(targets).resolve()
    if not path.is_relative_to(targets):
        return False
    reserved = {'room_recordings', 'background', 'played', 'out', 'evaluate', 'session'}
    return (not (reserved & set(path.relative_to(targets).parts[:-1]))
            and not any(path.is_relative_to(Path(root).resolve()) for root in excluded_roots))


if __name__ == '__main__':
    import sys
    for clip in json.loads(Path(sys.argv[1]).read_text())['clips']:
        bounds = [round(s/clip['fs'], 3) for s in clip.get('crop_samples', [])]
        print(clip['file'], 'seconds=', bounds, 'status=', clip['status'],
              'approval_key=', clip.get('approval_key', 'unavailable'))
