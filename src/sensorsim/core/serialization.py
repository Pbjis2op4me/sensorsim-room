"""Strict, dependency-light serialization shared by configuration and provenance.

Canonical JSON preserves the historical finite JSON bytes (sorted keys, compact
separators, ASCII escapes). Supported values are native JSON scalars, lists,
tuples and string-keyed dicts. Convert paths, arrays and typed records explicitly
at their schema boundary. Reports may opt into ``report_data``; keys may not.
YAML aliases are allowed; merge keys are rejected (the library loader's policy).
"""
from __future__ import annotations

import json
import math
import os
import secrets
import stat
from pathlib import Path

import yaml


def _validate(value, ancestors: set[int] | None = None) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite value is not valid canonical JSON")
        return
    if type(value) not in (dict, list, tuple):
        raise TypeError(f"unsupported JSON type: {type(value).__name__}")
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        raise ValueError("circular JSON container")
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise TypeError("JSON mapping keys must be strings")
            children = value.values()
        else:
            children = value
        for child in children:
            _validate(child, ancestors)
    finally:
        ancestors.remove(id(value))


def to_native(value):
    """Explicit schema-boundary adapter: NumPy scalars/arrays to native Python, recursively.

    Writers whose payloads may carry NumPy values call this before ``atomic_json`` /
    ``canonical_json``. It never alters finite values or hides nonfinite ones (the strict
    validator still rejects those). Tuples become lists, matching canonical JSON.
    """
    ancestors = set()

    def convert(item):
        numpy = type(item).__module__ == "numpy" and hasattr(item, "tolist")
        if not isinstance(item, (dict, list, tuple)) and not numpy:
            return item
        if id(item) in ancestors:
            raise ValueError("circular JSON container")
        ancestors.add(id(item))
        try:
            if isinstance(item, dict):
                return {key: convert(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [convert(child) for child in item]
            native = item.tolist()
            # longdouble/clongdouble have no lossless Python scalar equivalent:
            # tolist returns another NumPy scalar, so recursing never terminates.
            if type(native).__module__ == "numpy":
                raise TypeError(f"unsupported JSON type: {type(item).__name__}")
            return convert(native)
        finally:
            ancestors.remove(id(item))

    return convert(value)


def canonical_json(value) -> str:
    """Canonical finite JSON; no ``default=str`` or implicit schema conversion."""
    _validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def report_data(value: dict) -> dict:
    """Replace unavailable numeric report values with null and JSON-pointer reasons.

    Finite reports keep their shape. ``nonfinite_status`` is reserved. This is an
    explicit report adapter, never a config, manifest identity or cache-key policy.
    """
    if type(value) is not dict or "nonfinite_status" in value:
        raise ValueError("report must be a dict without reserved nonfinite_status")
    reasons = {}
    ancestors = set()

    def clean(item, path):
        if type(item) is float and not math.isfinite(item):
            reasons[path] = "nan" if math.isnan(item) else ("+infinity" if item > 0 else "-infinity")
            return None
        if type(item) not in (dict, list, tuple):
            return item
        if id(item) in ancestors:
            raise ValueError("circular JSON container")
        ancestors.add(id(item))
        try:
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise TypeError("JSON mapping keys must be strings")
                return {key: clean(child, path + "/" + key.replace("~", "~0").replace("/", "~1"))
                        for key, child in item.items()}
            return [clean(child, f"{path}/{i}") for i, child in enumerate(item)]
        finally:
            ancestors.remove(id(item))

    result = clean(value, "")
    if reasons:
        result["nonfinite_status"] = reasons
    _validate(result)
    return result


_STAGED_FLAGS = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))


def _staged_sibling(path: Path, mode: int) -> tuple[int, Path]:
    """Exclusively create the hidden sibling `.<name>.<random>.tmp`. Like open(), the kernel filters
    `mode` through the umask, or through the folder's default ACL when it has one."""
    for _ in range(100):
        candidate = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            return os.open(candidate, _STAGED_FLAGS, mode), candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"no free staging name beside {path}")


def atomic_json(path: str | Path, value, *, indent: int | None = 2, sort_keys: bool = False) -> None:
    """Replace a JSON file atomically after successful serialization and flush.

    An encode/write/replace failure preserves the previous destination. Temporary
    files use the destination filesystem and are removed on failure. This provides
    atomic visibility; it does not promise directory durability after power loss.
    File modes: a new file gets the mode open() would give it (0o666 less the umask,
    or the folder's default ACL); a replaced file keeps its mode (through a symbolic
    link, its target's), and the staged copy is never more permissive than that.
    """
    _validate(value)
    encoded = json.dumps(value, sort_keys=sort_keys, indent=indent, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        keep = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:   # nothing (or nothing reachable) to keep: a new file
        keep = None
    temporary = None
    try:
        fd, temporary = _staged_sibling(path, 0o666 if keep is None else keep & 0o666)
        try:
            f = os.fdopen(fd, "w", encoding="utf8")
        except BaseException:
            os.close(fd)
            raise
        with f:
            f.write(encoded)
            f.flush()
            if keep is not None and stat.S_IMODE(os.fstat(f.fileno()).st_mode) != keep:
                os.fchmod(f.fileno(), keep)
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _UniqueLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    """Shared duplicate-rejecting loader; schema and path resolution stay with callers."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("YAML merge keys are unsupported; use explicit configuration overrides")
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as e:
            raise ValueError("YAML mapping keys must be hashable scalars") from e
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_yaml_unique(content):
    """Parse YAML text/bytes/a stream safely, rejecting nested duplicates and merges.

    No filesystem access, caching or schema validation is performed here.
    """
    return yaml.load(content, Loader=_UniqueLoader)
