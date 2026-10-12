"""Required host controls. Do not replace IPC failures with skips or serial runs.

The all-count control must be run by the verifier on an idle host. Its tiny
procedural fixtures are disclosed development data, never P5 fresh cases.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sensorsim.evaluation import power_audit as p
from tools.power_audit import procedural_factory


def identity(value):
    return value


def test_tiny_spawn_pool_probe():
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(identity, 7).result(timeout=20) == 7


def worker_mutant(groups,effect,seed,*,workers):
    return procedural_factory(groups,effect,seed+(workers if workers != 1 else 0))


def run_fixture(workers,mutant,output):
    budget=p.Budget(group_counts=(2,),null_datasets=2,alternative_datasets=2,workers=workers,max_seconds=60)
    cases=[dict(row,seed=78200+i) for i,row in enumerate(p.planned_cases(budget))]
    factory=partial(worker_mutant,workers=workers) if mutant else procedural_factory
    report=p.qualify_test(factory,p.Family(permutations=3),budget,execution='decision',cases=cases,phase_seconds={2:60.})
    assert len(report['attempts']) == 8
    assert all(r['execution']=='completed' and r['result']['valid'] for r in report['attempts'])
    assert all(set(r['worker']['threads'].values()) == {'2'} for r in report['attempts'])
    output=Path(output);output.mkdir()
    (output/'scientific.json').write_bytes(p.scientific_payload(report))
    (output/'raw.json').write_text(json.dumps(report,sort_keys=True))


@pytest.mark.slow
def test_spawn_1_2_4_8_and_reached_worker_mutant(tmp_path, python_subprocess_env):
    def launch(workers,mutant=False):
        output=tmp_path/f'{workers}-{mutant}'
        run=subprocess.run([sys.executable,__file__,str(workers),str(int(mutant)),str(output)],
            env=python_subprocess_env,capture_output=True,text=True,timeout=90)
        assert run.returncode == 0, (run.stdout,run.stderr)
        return (output/'scientific.json').read_bytes(),json.loads((output/'raw.json').read_text())
    reference,first=launch(1)
    for workers in (2,4,8):
        value,raw=launch(workers)
        assert value == reference
        assert all(r['worker']['pid'] != os.getpid() for r in raw['attempts'])
    bad,raw=launch(8,True)
    assert bad != reference
    assert raw['attempts'][0]['feature_sha256'] != first['attempts'][0]['feature_sha256']
    assert len(raw['attempts'][0]['result']['results']) == 12


if __name__ == '__main__':
    run_fixture(int(sys.argv[1]),bool(int(sys.argv[2])),sys.argv[3])
