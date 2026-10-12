# SPDX-License-Identifier: Apache-2.0
"""Bounded, process-local memoization of whole-take level measurements.

Only scalar results and content digests are retained, never recordings. Current
content is hashed on every lookup, including writable arrays; object identity
alone cannot safely key mutable corpus inputs. This is a computation cache, not
an authorization or provenance cache.
"""
from collections import OrderedDict
import hashlib
from threading import Lock

import numpy as np

_LIMIT = 128
_values = OrderedDict()
_lock = Lock()


def take_level_power(event, fs, span, *, split='in_distribution'):
    """Use the same level convention as the mixer, without remeasuring a take."""
    from sensorsim import event_level, training_device
    x = np.asarray(event, float)
    # Invalid/unusual inputs keep the underlying function's validation and errors.
    if x.ndim != 1 or not x.size or not isinstance(span, tuple):
        return event_level.level_power(x, fs, span, split=split)
    key = (event_level.level_power, training_device.active_power, fs, span, split,
           x.shape, hashlib.sha256(x.tobytes()).digest())
    with _lock:
        if key in _values:
            _values.move_to_end(key)
            return _values[key]
    value = event_level.level_power(x, fs, span, split=split)
    with _lock:
        _values[key] = value
        _values.move_to_end(key)
        if len(_values) > _LIMIT:
            _values.popitem(last=False)
    return value


def clear():
    """Drop development/worker-local cached values; never changes an RNG stream."""
    with _lock:
        _values.clear()
