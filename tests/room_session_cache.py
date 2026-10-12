"""Session-local snapshots of generated test inputs, never pipeline results.

Only the explicit ``fake_session`` helper opts in. Files are copied as bytes and
returned objects are deep-copied; callers can modify either without poisoning
the next test. Outside the pytest session context the original builder runs.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy


_active = None


@contextmanager
def session_cache():
    """Start a fresh, bounded cache and discard it at the end of the session."""
    global _active
    previous, _active = _active, OrderedDict()
    try:
        yield
    finally:
        _active = previous


def generated_session(folder, *, full, build, dependencies):
    """Reuse only an identical fixture variant with the same generation hooks.

    The builder returns ``(folder, room, truth_arrays, latency)`` and writes flat
    files. It remains callable directly when a test needs fresh rendering.
    Failed builds are never cached. At most two variants are retained.
    """
    if _active is None:
        return build(folder, full=full)
    key = (full, build, dependencies)
    if key not in _active:
        result = build(folder, full=full)
        files = tuple((p.name, p.read_bytes()) for p in sorted(folder.iterdir()))
        _active[key] = files, deepcopy(result[1:])
        if len(_active) > 2:
            _active.popitem(last=False)
        return result
    files, payload = _active[key]
    # Match the builder's destination contract: no merging into an existing
    # directory and no implicit creation of missing parents.
    folder.mkdir()
    for name, contents in files:
        (folder/name).write_bytes(contents)
    _active.move_to_end(key)
    return (folder, *deepcopy(payload))
