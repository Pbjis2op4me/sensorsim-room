"""Run fingerprint: resolved config + git hash + environment, stored with every run."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path


_PKGS = ("numpy", "scipy", "torch", "pydantic", "pyroomacoustics", "PyYAML", "threadpoolctl")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """SHA256 of exact file bytes, with bounded memory and unchanged digest inputs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint(root: str | Path | None = None) -> str:
    """Content identity of installed project code/data, including uncommitted edits.

    Complete-run scope, deliberately broader than the flight-only cache. No git
    timestamp or memoization: modifying a file in this process invalidates reuse.
    External measured inputs remain caller-owned identity inputs.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(b"sensorsim-implementation/1\0")
    for path in sorted(p for p in root.rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")):
        name = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _run(cmd: list[str]) -> tuple[bool, str]:
    """Return success separately: empty stdout can be a valid result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False, ""


def git_state(repo: Path) -> dict:
    head_ok, head = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if not head_ok or not head:
        return {"commit": None, "dirty": None}
    status_ok, status = _run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])
    return {"commit": head, "dirty": bool(status) if status_ok else None}


def source_revision() -> dict:
    """Identity of THIS package's source, never of the caller's repository.

    Reports the installed distribution version, and a git commit only when the
    code is imported from its own checkout (git top level == the directory that
    holds src/). An installed copy living inside some other repository reports
    no commit, rather than recording that repository's HEAD as the simulator
    revision. Running git in the caller's cwd, or walking up from site-packages,
    did exactly that (cr-room-capture-04).
    """
    try:
        version = metadata.version("sensorsim-room")
    except metadata.PackageNotFoundError:
        version = None
    checkout = Path(__file__).resolve().parents[3]  # <checkout>/src/sensorsim/core/fingerprint.py
    ok, top = _run(["git", "-C", str(checkout), "rev-parse", "--show-toplevel"])
    own = ok and bool(top) and Path(top).resolve() == checkout
    return {"package_version": version, **(git_state(checkout) if own else {"commit": None, "dirty": None})}


def environment() -> dict:
    pkgs = {}
    for p in _PKGS:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    rocm = Path("/opt/rocm/.info/version")
    return {
        "python": platform.python_version(),
        "kernel": platform.release(),
        "rocm": rocm.read_text().strip() if rocm.exists() else None,
        "packages": pkgs,
    }


