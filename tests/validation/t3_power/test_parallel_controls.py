"""Development execution controls; parallelism must not change the experiment."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sensorsim.evaluation import power_audit as p


def launch(tmp_path, workers, env, *, max_seconds=60):
    config = tmp_path/f'config-{workers}.json'
    config.write_text(json.dumps(dict(family=dict(permutations=19), budget=dict(
        group_counts=[4], null_datasets=1, alternative_datasets=1,
        workers=workers, max_seconds=max_seconds))))
    output = tmp_path/f'output-{workers}'
    env = dict(env, **{key: '1' for key in p.THREAD_VARIABLES})
    script = Path(__file__).resolve().parents[3]/'tools/power_audit.py'
    run = subprocess.run([sys.executable, str(script), '--config', str(config),
                          '--output', str(output)], env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, (run.stderr, (output/'stderr.txt').read_text())
    return (json.loads((output/'result.json').read_text()),
            [json.loads(line) for line in (output/'replicates.jsonl').read_text().splitlines()])


@pytest.mark.slow
def test_fresh_parallel_processes_match_serial_raw_results(tmp_path, python_subprocess_env):
    serial, rows = launch(tmp_path, 1, python_subprocess_env)
    parallel, other = launch(tmp_path, 2, python_subprocess_env)
    assert len(rows) == len(other) == 4
    assert serial['cells'] == parallel['cells']
    assert serial['qualification_scope_sha256'] == parallel['qualification_scope_sha256']
    assert len({r['worker']['pid'] for r in other}) == 2
    for a, b in zip(rows, other):
        wa, wb = a.pop('worker'), b.pop('worker')
        assert a == b
        assert wa['implementation_sha256'] == wb['implementation_sha256']
        assert all(value == '1' for value in wb['threads'].values())
        assert Path(wb['module_path']).resolve() == Path(p.__file__).resolve()
        assert wb['start'] <= wb['end']


def test_parallel_deadline_retains_every_unrun_case(tmp_path, python_subprocess_env):
    result, rows = launch(tmp_path, 2, python_subprocess_env, max_seconds=1e-12)
    assert len(rows) == 4
    assert all(r['execution'] == 'not run' for r in rows)
    assert result['cells'][0]['harness'] == 'invalid'
    assert result['cells'][0]['registered_outcome'] == 'inconclusive'


def test_parallel_rejects_nested_numerical_threads(monkeypatch):
    from tools.power_audit import procedural_factory
    monkeypatch.setenv('OMP_NUM_THREADS', '3')
    with pytest.raises(ValueError, match='thread variables'):
        p.qualify_test(procedural_factory, p.Family(permutations=19),
                       p.Budget(group_counts=(4,), null_datasets=1, alternative_datasets=1, workers=2))


@pytest.mark.parametrize('name', [
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
])
@pytest.mark.parametrize('value', [None, '0', '3', 'invalid'])
def test_parallel_rejects_each_invalid_thread_setting(monkeypatch, name, value):
    from tools.power_audit import procedural_factory
    for key in p.THREAD_VARIABLES:
        monkeypatch.setenv(key, '2')
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    reached = []
    def forbidden_pool(**kwargs):
        reached.append(kwargs)
        raise AssertionError('invalid thread setting reached a process pool')
    monkeypatch.setattr(p, 'ProcessPoolExecutor', forbidden_pool)
    with pytest.raises(ValueError, match='thread variables'):
        p.qualify_test(procedural_factory, p.Family(permutations=19),
                       p.Budget(group_counts=(4,), null_datasets=1, alternative_datasets=1, workers=2))
    assert reached == []


def test_worker_count_cannot_change_fixed_seeds():
    budget = p.Budget(null_datasets=1, alternative_datasets=1)
    assert list(p.planned_cases(budget)) == list(p.planned_cases(replace(budget, workers=4)))
    with pytest.raises(ValueError, match='workers'):
        replace(budget, workers=5)


def repeated_and_errored_factory(groups, effect, seed):
    from tools.power_audit import procedural_factory
    if effect == 'scalar_plus6db':
        raise ImportError('designated development unavailable dependency')
    return procedural_factory(groups, 'null', 995024)


def test_parallel_duplicate_and_dependency_failures_are_not_dropped(monkeypatch):
    for key in p.THREAD_VARIABLES:
        monkeypatch.setenv(key, '1')
    budget = p.Budget(group_counts=(4,), null_datasets=1, alternative_datasets=1, workers=2)
    rows = []
    result = p.qualify_test(repeated_and_errored_factory, p.Family(permutations=19), budget, record=rows.append)
    assert len(rows) == 4
    assert rows[0]['result']['valid']
    assert rows[2]['execution'] == 'errored'
    assert 'unavailable dependency' in rows[2]['error']
    for index in (1, 3):
        assert not rows[index]['result']['valid']
        assert 'content reused' in str(rows[index]['result']['errors'])
    assert result['cells'][0]['harness'] == 'invalid'
    assert result['cells'][0]['registered_outcome'] == 'inconclusive'
