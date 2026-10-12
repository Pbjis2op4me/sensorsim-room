"""atomic_json mode regression tests."""
import json
import os
import stat
from contextlib import contextmanager

import pytest

from sensorsim.core.serialization import atomic_json

DATA = {"z": [1., True, None], "a": "µ"}
BYTES = json.dumps(DATA, indent=2).encode("utf8")


def mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@contextmanager
def umask(value: int):
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


def test_new_file_gets_the_mode_open_gives(tmp_path):
    with umask(0o027):   # 0o640 without a default ACL; mkstemp's 0o600 differs
        atomic_json(tmp_path / "metrics.json", DATA)
        with open(tmp_path / "control.json", "w"):
            pass
    assert mode(tmp_path / "metrics.json") == mode(tmp_path / "control.json")
    assert (tmp_path / "metrics.json").read_bytes() == BYTES


@pytest.mark.parametrize("existing", [0o664, 0o444, 0o600], ids=oct)
def test_replacement_keeps_the_existing_mode(tmp_path, existing):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    path.chmod(existing)
    with umask(0o027):   # a new file would get 0o640: neither that nor 0o600 passes for 0o664/0o444
        atomic_json(path, DATA)
    assert mode(path) == existing
    assert path.read_bytes() == BYTES
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]   # no staging file left


def test_symlinked_destination_takes_its_target_mode(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    target.chmod(0o664)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with umask(0o027):
        atomic_json(link, DATA)
    # os.replace swaps the link itself for the new file, as before; its mode is the target's,
    # not the link's own 0o777.
    assert not link.is_symlink() and mode(link) == 0o664
    assert link.read_bytes() == BYTES and target.read_text() == "{}"
