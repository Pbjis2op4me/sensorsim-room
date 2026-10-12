# SPDX-License-Identifier: Apache-2.0
"""Bounded process-local memoization of deterministic matched-parent renders.

Pool authorization remains the caller's responsibility on every access. Keys
cover physical inputs, parent bytes and invoked hooks. No RNG is retained or
advanced: RIR seeds use the original per-room convention, and window/level draws
remain in the caller. No audio is written to disk.
"""
from collections import OrderedDict
import hashlib
from threading import Lock

import numpy as np

from sensorsim.negatives.composer import audio_hash, digest

_MAX_ENTRIES = 128
_MAX_BYTES = 64 * 1024 * 1024
_values = OrderedDict()
_bytes = 0
_lock = Lock()


def _get(key):
    with _lock:
        entry = _values.get(key)
        if entry is not None:
            _values.move_to_end(key)
            return entry[0]
    return None


def _put(key, value, size):
    global _bytes
    with _lock:
        previous = _values.pop(key, None)
        if previous is not None:
            _bytes -= previous[1]
        if size <= _MAX_BYTES:
            _values[key] = (value, size)
            _bytes += size
        while _values and (len(_values) > _MAX_ENTRIES or _bytes > _MAX_BYTES):
            _, (_, removed) = _values.popitem(last=False)
            _bytes -= removed


def _immutable(x):
    # A writable flag alone is reversible for owning arrays. Bytes-backed arrays
    # cannot be modified by callers and poison later cache hits.
    x = np.asarray(x)
    return np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape)


def render_parent(audio, *, draw, position, fs):
    """Return whole wet parent, audible span, RIR hash and original RIR seed."""
    from scipy.signal import oaconvolve
    from sensorsim.acoustics.room.synth import synthesize
    from sensorsim.recognizer import _audible_span

    seed = int(np.random.SeedSequence([draw.seed, 0, 11]).generate_state(1)[0])
    physical = digest(dict(room=draw.room.model_dump(mode='json'),
        config=draw.config.model_dump(mode='json') if draw.config is not None else None,
        source=list(position), receiver=list(draw.receiver_xyz_m),
        fs=fs.item() if isinstance(fs, np.generic) else fs, seed=seed))
    rir_key = ('rir', synthesize, physical)
    source = np.asarray(audio)
    key = ('wet', rir_key, oaconvolve, _audible_span, source.dtype.str, source.shape,
           hashlib.sha256(source.tobytes()).digest())
    value = _get(key)
    if value is not None:
        return value
    rir = _get(rir_key)
    if rir is None:
        result = synthesize(draw.room, position, draw.receiver_xyz_m, fs, seed=seed, config=draw.config)
        samples = _immutable(result.samples)
        rir = (samples, audio_hash(samples))
        _put(rir_key, rir, samples.nbytes)
    wet = oaconvolve(audio, rir[0])
    span = _audible_span(wet, fs)
    wet = _immutable(wet)
    value = (wet, span, rir[1], seed)
    _put(key, value, wet.nbytes)
    return value


def clear():
    """Drop worker-local computed values; authorization state is unaffected."""
    global _bytes
    with _lock:
        _values.clear()
        _bytes = 0


def info():
    """Current retained audio storage, for development resource checks."""
    with _lock:
        return dict(entries=len(_values), bytes=_bytes)
