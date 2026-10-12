"""Metadata-only feasibility check; does not render, fit or emit acceptance.

Run as a module with --data-root ROOT --output NEW_JSON. Hashing input bytes is
allowed; no waveform hyperslabs (including disclosed array 6) are accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from tests.validation import dechorate_v1


def inspect(data_root):
    archive = dechorate_v1.V1Archive(data_root)
    record_path = Path(data_root)/'metadata/zenodo-5562386.json'
    record = json.loads(record_path.read_text())
    if record['id'] != 5562386 or record['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('cached publisher record has the wrong ID or licence')
    entries = {r['key']: r for r in record['files']}
    checksums = {}
    for relative in (dechorate_v1.RIR_FILE, dechorate_v1.GEOMETRY_FILE):
        path = Path(data_root)/relative
        with path.open('rb') as stream:
            md5 = hashlib.file_digest(stream, 'md5').hexdigest()
        expected = entries[path.name]['checksum']
        if expected != 'md5:'+md5:
            raise ValueError('cached publisher MD5 mismatch')
        checksums[relative] = {'md5': md5, 'published_checksum': expected, 'matches': True}
    failures = {}
    for label, sources, arrays in (('requested_t2', (7, 8), (1, 3, 5)),
                                    ('requested_wp8', (7, 8), (1, 2, 3, 4, 5))):
        try:
            dechorate_v1.validate_selection(sources, arrays)
        except ValueError as exc:
            failures[label] = str(exc)
    if not failures:
        raise ValueError('unexpected feasibility change; registration amendment required')
    return dict(
        schema='dechorate-recompute-preflight/1', claim_type='input feasibility, not acoustic acceptance',
        metadata=archive.metadata(), publisher_md5_checks=checksums,
        cached_publisher_record_sha256=dechorate_v1.sha256(record_path),
        source_verification='offline cached publisher record; no network request in this run',
        waveforms_read=archive.read_log, requested_selection_failures=failures,
        requested_t2={'sources': [7, 8], 'train_mics': [1, 11, 21], 'evaluation_mics': [22, 23, 24, 25],
                      'required_evaluation_pairs': 88, 'scored_evaluation_pairs': 0},
        requested_wp8={'sources': [7, 8], 'fit_arrays': [1, 2, 3, 4], 'evaluation_arrays': [5],
                      'required_fit_pairs': 440, 'required_evaluation_pairs': 110,
                      'scored_evaluation_pairs': 0},
        registered_outcome='inconclusive', harness='unqualified', acceptance_run_started=False,
        reason='source 8 is absent; no subset substitution; this metadata preflight does not assess signoff or custody',
        environment={'python': platform.python_version(),
                     **{p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'netCDF4')}},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        imports={name: {'path': module.__file__, 'sha256': dechorate_v1.sha256(module.__file__)}
                 for name, module in sorted(sys.modules.items())
                 if (name.startswith(('sensorsim', 'tests.validation')) and
                     getattr(module, '__file__', None) and Path(module.__file__).is_file())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect(args.data_root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
