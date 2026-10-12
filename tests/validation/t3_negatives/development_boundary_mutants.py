# SPDX-License-Identifier: Apache-2.0
"""Development-only controls for F1/F2; never an acceptance runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MUTANTS = [
    ('mixed_checkpoints', 'src/sensorsim/training_negatives.py',
     'if binding is not None and current != binding:', 'if False:',
     'tests/unit/t3_negatives/test_bank_identity.py::test_one_checkpoint_per_mining_bank'),
    ('unchecked_additions', 'src/sensorsim/training_negatives.py',
     '    _validate_bank_addition(addition, lock_sha256)\n', '    pass  # mutant bypasses bank validation\n',
     'tests/unit/t3_negatives/test_bank_identity.py::test_training_admission_revalidates_persisted_bank'),
    ('unlinked_parents', 'src/sensorsim/evaluation/fa_rate.py',
     '            links.update("source-group:" + parent["source_group"] for parent in parent_lock["parents"])',
     '            pass  # mutant loses locked source groups',
     'tests/validation/t3_negatives/check_parent_groups.py'),
]

PLUGIN = '''
import hashlib, json, os, sys
from pathlib import Path
def pytest_sessionfinish(session, exitstatus):
    imports = {name: dict(path=str(Path(m.__file__).resolve()),
        sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
        for name, m in list(sys.modules.items())
        if name.startswith("sensorsim") and getattr(m, "__file__", None) and Path(m.__file__).is_file()}
    Path(os.environ["MUTATION_IMPORTS"]).write_text(json.dumps(dict(imports=imports,
        collected=[i.nodeid for i in session.items], exitstatus=int(exitstatus)), indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name, module_name, old, new, node in MUTANTS:
        root = args.output/name
        shutil.copytree(args.candidate/'src', root/'src')
        shutil.copytree(args.candidate/'tests', root/'tests')
        shutil.copy2(args.candidate/'pyproject.toml', root/'pyproject.toml')
        (root/'mutation_imports.py').write_text(PLUGIN)
        module = root/module_name
        source = module.read_text()
        if source.count(old) != 1:
            raise ValueError('missing unique mutation hook: '+name)
        command = [sys.executable, '-m', 'pytest', '-q', '-s', '-p', 'no:cacheprovider',
                   '-p', 'mutation_imports', '--basetemp='+str(root/'fixtures'), node]
        pair = {}
        for kind in ('correct', 'bad'):
            if kind == 'bad':
                module.write_text(source.replace(old, new))
            env = dict(os.environ, PYTHONPATH=str(root/'src')+os.pathsep+str(root),
                       MUTATION_IMPORTS=str(root/(kind+'.imports.json')))
            record = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(),
                          source_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
            started = time.monotonic()
            with (root/(kind+'.stdout')).open('w') as out, (root/(kind+'.stderr')).open('w') as err:
                run = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, timeout=120)
            record.update(exit_code=run.returncode, ended_utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.monotonic()-started)
            for suffix in ('stdout', 'stderr'):
                record[suffix+'_sha256'] = hashlib.sha256((root/(kind+'.'+suffix)).read_bytes()).hexdigest()
            pair[kind] = record
        results.append(dict(mutant=name, **pair,
                            killed=pair['correct']['exit_code'] == 0 and pair['bad']['exit_code'] == 1))
        (root/'run.json').write_text(json.dumps(results[-1], indent=2)+'\n')
        # Keep logs and imports, remove only copied source/fixtures to bound storage.
        for directory in ('src', 'tests', 'fixtures'):
            shutil.rmtree(root/directory, ignore_errors=True)
    summary = dict(domain='synthetic proxy', kind='development controls; not acceptance',
                   results=results, survivors=[r['mutant'] for r in results if not r['killed']])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
    return int(bool(summary['survivors']))


if __name__ == '__main__':
    raise SystemExit(main())
