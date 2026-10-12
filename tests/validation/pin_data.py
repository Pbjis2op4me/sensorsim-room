"""Write MANIFEST.json for measured-room data you downloaded, so the validation benches accept it.

The benches (bench_room_fit, bench_room_closedloop) refuse any input whose sha256 or size differs
from MANIFEST.json, to catch files that change between runs. This pins what is on disk now. It
records your download; it does not prove the files match the ones the parent project used.

    python tests/validation/pin_data.py /path/to/room_acoustics            # refuses to overwrite
    python tests/validation/pin_data.py /path/to/room_acoustics --force    # re-pin after a new download

Expected layout and dataset licences: THIRD_PARTY.md, "Plug-in parts".
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pin(root: str | Path, *, force: bool = False) -> list[dict]:
    root = Path(root)
    manifest = root/"MANIFEST.json"
    if manifest.exists() and not force:
        raise FileExistsError(f"{manifest} exists; pass force=True (--force) to re-pin")
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != manifest):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        rows.append({"path": str(path.relative_to(root)), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size})
    manifest.write_text(json.dumps({"files": rows, "note": "pinned locally by tests/validation/pin_data.py"}, indent=1) + "\n")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("root", type=Path, help="the folder holding dechorate/, meshrir/, bras/")
    parser.add_argument("--force", action="store_true", help="replace an existing MANIFEST.json")
    args = parser.parse_args(argv)
    rows = pin(args.root, force=args.force)
    print(f"pinned {len(rows)} files in {args.root/'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
