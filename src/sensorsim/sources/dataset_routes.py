"""WP4 dataset integration hooks; the owning lane applies the small call-site patch.

Legacy sweep-extracted DryClip objects are outside the raw-capture route schema.
These hooks preserve that API while refusing explicit in-situ metadata and
honouring preassigned splits for the new intake. They are not installed through
monkeypatching or side effects.
"""
from .catalog import _identifier, content_hash


def route_metadata(clip):
    p = clip.provenance or {}
    if 'capture_route' not in p:
        return {}  # Existing extracted-source API; never silently relabel it near_dry.
    if p['capture_route'] == 'in_situ':
        raise ValueError('in_situ takes cannot enter the room renderer')
    if p['capture_route'] not in ('near_dry', 'known_digital'):
        raise ValueError('invalid capture_route')
    for name in ('source_group', 'object_id', 'session'):
        _identifier(p.get(name), name)
    if p['source_group'] != clip.source_group:
        raise ValueError('source_group disagrees with source provenance')
    if p.get('details', {}).get('synthetic') and p.get('split') != 'train':
        raise ValueError('fitted procedural children are training-only; preassign split=train')
    if p.get('split') not in (None, 'train', 'val'):
        raise ValueError('RH or invalid split cannot enter the training dataset')
    qa = p.get('details', {}).get('qa', {})
    if qa.get('status') != 'pass':
        raise ValueError('source crop QA must pass before rendering')
    return {k: p[k] for k in ('capture_route', 'object_id', 'session', 'parent_content_sha256', 'split') if k in p}


def validate_route_audio(clip, samples, fs):
    """Check the decoded, unscaled source against the approval's crop identity."""
    p = clip.provenance or {}
    if 'capture_route' in p:
        expected = p.get('details', {}).get('qa', {}).get('crop_content_sha256')
        if expected != content_hash(samples, fs):
            raise ValueError('source samples changed after crop QA')


def route_keys(item):
    """Add to existing content and source_group keys before connected splitting."""
    p = item.get('provenance', {})
    keys = []
    if p.get('parent_content_sha256'):
        keys.append(('content', p['parent_content_sha256']))
    details = p.get('details', {})
    keys.extend(('recording', g) for g in details.get('fit_groups', []))
    keys.extend(('content', h) for h in details.get('fit_hashes', []))
    return keys


def declared_source_pools(catalog):
    """None means use the existing automatic splitter; mixed plans are errors."""
    from sensorsim.acoustics.room.dataset import _connected_groups
    splits = [r.get('provenance', {}).get('split') for r in catalog]
    if not any(s is not None for s in splits):
        return None
    if any(s not in ('train', 'val') for s in splits):
        raise ValueError('all sources need train/val assignments when any split is preassigned')
    keys = [[('recording', r['source_group']), ('content', r['content_sha256']),
             ('content', r['normalized_content_sha256']), *route_keys(r)] for r in catalog]
    for indices in _connected_groups(keys).values():
        if len({splits[i] for i in indices}) != 1:
            raise ValueError('split leakage in preassigned source groups or hashes')
    for r, s in zip(catalog, splits):
        if r.get('provenance', {}).get('details', {}).get('synthetic') and s != 'train':
            raise ValueError('fitted procedural children are training-only')
    return {s: [i for i, v in enumerate(splits) if v == s] for s in ('train', 'val')}
