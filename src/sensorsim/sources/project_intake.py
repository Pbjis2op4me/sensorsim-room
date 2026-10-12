"""Project intake boundaries, checked before crop review or rendering."""
from pathlib import Path


def excluded_roots(cfg):
    root = Path(cfg['root'])
    return tuple((root/cfg[key]).resolve() for key in ('out', 'evaluate', 'background', 'session'))


def target_fingerprint(cfg):
    from sensorsim.core.fingerprint import sha256_file
    from .intake import is_target_path
    targets = Path(cfg['root'])/cfg['targets']
    excluded = excluded_roots(cfg)
    return {p.relative_to(targets).as_posix(): sha256_file(p) for p in sorted(targets.rglob('*'))
            if p.is_file() and p.suffix.lower() in ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aif', '.aiff')
            and is_target_path(p, targets, excluded)}


def require_target_sites(cfg, rows):
    """Report every missing declared class before asking for listening approvals."""
    if not cfg.get('room_yaml'):
        return
    from sensorsim.acoustics.room.prior import RoomSurvey
    survey = RoomSurvey.load(cfg['room_yaml'])
    labels = {row['label'] for row in rows if isinstance(row.get('label'), str)}
    missing = sorted(label for label in labels if not survey.source_sites.get(label))
    if missing:
        raise ValueError('room.yaml source_sites required for target classes: ' + ', '.join(missing)
                         + '; add their actual positions before reviewing source crops')
