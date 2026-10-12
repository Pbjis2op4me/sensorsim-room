"""Disclosed development fixtures for scheduling, not P5 statistical acceptance."""
from concurrent.futures import Future
from dataclasses import replace
import json
import time

import pytest

from sensorsim.evaluation import power_audit as p
from tools.power_audit import procedural_factory


def budget(workers=1):
    return p.Budget(group_counts=(2,), null_datasets=1, alternative_datasets=1,
                    workers=workers, max_seconds=60)


def cases(b):
    return [dict(row, seed=78000+i) for i, row in enumerate(p.planned_cases(b))]


class LocalPool:
    """Exercises scheduling without IPC; host spawn test is separately required."""
    reached = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._processes = {}
    def submit(self, fn, *args, **kwargs):
        self.reached.append((self.kwargs['max_workers'], args[2]['seed']))
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future
    def shutdown(self, **kwargs):
        pass


def test_eight_workers_and_explicit_cases_reach_identical_family(monkeypatch):
    monkeypatch.setattr(p, 'ProcessPoolExecutor', LocalPool)
    LocalPool.reached = []
    reports = []
    for workers in (1, 2, 4, 8):
        b = budget(workers)
        reports.append(p.qualify_test(procedural_factory, p.Family(permutations=3), b,
                       execution='decision', cases=cases(b)))
    reference = p.scientific_payload(reports[0])
    for report in reports:
        assert all(r['execution'] == 'completed' and r['result']['valid'] for r in report['attempts'])
        assert p.scientific_payload(report) == reference
    assert {w for w, _ in LocalPool.reached} == {2, 4, 8}
    assert [r['seed'] for r in reports[0]['attempts']] == [78000, 78001, 78002, 78003]


def test_worker_dependent_mutant_is_detected_after_reaching_family(monkeypatch):
    monkeypatch.setattr(p, 'ProcessPoolExecutor', LocalPool)
    original = p._dataset_attempt
    reached = []
    def mutant(factory, spec, item, **kwargs):
        reached.append(item['seed'])
        return original(factory, spec, dict(item, seed=item['seed'] + 8), **kwargs)
    good = p.qualify_test(procedural_factory, p.Family(permutations=3), budget(),
                          execution='decision', cases=cases(budget()))
    monkeypatch.setattr(p, '_dataset_attempt', mutant)
    bad = p.qualify_test(procedural_factory, p.Family(permutations=3), budget(8),
                         execution='decision', cases=cases(budget(8)))
    assert reached == [78000, 78001, 78002, 78003]
    assert all(r['result']['valid'] for r in bad['attempts'])
    assert good['attempts'][0]['feature_sha256'] != bad['attempts'][0]['feature_sha256']
    assert p.scientific_payload(good) != p.scientific_payload(bad)


@pytest.mark.parametrize('bad', ['empty','missing','duplicate','order','seed_bool','seed_negative','seed_repeat','extra'])
def test_invalid_case_plan_fails_before_factory(bad):
    plan = cases(budget())
    if bad == 'empty': plan = []
    if bad == 'missing': plan.pop()
    if bad == 'duplicate': plan[1] = plan[0].copy()
    if bad == 'order': plan.reverse()
    if bad == 'seed_bool': plan[0]['seed'] = True
    if bad == 'seed_negative': plan[0]['seed'] = -1
    if bad == 'seed_repeat': plan[1]['seed'] = plan[0]['seed']
    if bad == 'extra': plan[0]['other'] = 1
    def never(*args):
        raise AssertionError('invalid plan reached factory')
    with pytest.raises(ValueError, match='case plan'):
        p.qualify_test(never, p.Family(permutations=3), budget(), cases=plan)


def test_visible_worker_error_stops_submissions_without_losing_cases(monkeypatch):
    monkeypatch.setattr(p, 'ProcessPoolExecutor', LocalPool)
    class BrokenPool(LocalPool):
        def submit(self, *a, **kw):
            raise RuntimeError('designated worker failure')
    monkeypatch.setattr(p, 'ProcessPoolExecutor', BrokenPool)
    report = p.qualify_test(procedural_factory, p.Family(permutations=3), budget(2), cases=cases(budget()))
    assert len(report['attempts']) == 4
    assert report['attempts'][0]['execution'] == 'errored'
    assert 'designated worker failure' in report['attempts'][0]['error']
    assert report['cells'][0]['registered_outcome'] == 'inconclusive'


def test_phase_deadline_retains_full_plan(monkeypatch):
    monkeypatch.setattr(p, 'ProcessPoolExecutor', LocalPool)
    rows = list(p._scheduled_attempts(procedural_factory, p.Family(permutations=3),
             budget(2), time.monotonic()-10, cases=cases(budget()), phase_seconds={2: 1e-12}))
    assert len(rows) == 4
    assert all(row['execution'] == 'not run' for row in rows)


def test_valid_worker_counts_only():
    for workers in (0, 3, 5, True, 16):
        with pytest.raises(ValueError, match='workers'):
            budget(workers)


def test_future_timeout_is_visible_and_stops_refill(monkeypatch):
    class NeverFinishes(LocalPool):
        shutdown_called = False
        submitted = []
        def submit(self, fn, *args, **kwargs):
            self.submitted.append(args[2]['seed'])
            f = Future()
            f.set_exception(TimeoutError('designated hung worker'))
            return f
        def shutdown(self, **kwargs):
            type(self).shutdown_called = True
    monkeypatch.setattr(p, 'ProcessPoolExecutor', NeverFinishes)
    phases=[]
    report=p.qualify_test(procedural_factory,p.Family(permutations=3),budget(2),cases=cases(budget()),
                          phase_seconds={2:60},phase_record=phases.append)
    assert len(report['attempts']) == 4
    assert report['attempts'][0]['execution'] == 'errored'
    assert all(r['execution']=='not run' for r in report['attempts'][1:])
    assert NeverFinishes.submitted == [78000,78001] and NeverFinishes.shutdown_called
    assert [row['event'] for row in phases] == ['start','submitted','submitted','end']
    assert [row['seed'] for row in phases if row['event']=='submitted'] == [78000,78001]
    assert report['cells'][0]['registered_outcome'] == 'inconclusive'


def test_reverse_completion_is_collected_in_frozen_order(monkeypatch):
    completed=[]
    class ReversePool(LocalPool):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.waiting=[]
        def submit(self,fn,*args,**kwargs):
            pool=self
            class Deferred(Future):
                def result(self,timeout=None):
                    waiting,pool.waiting=pool.waiting,[]
                    for future,item in reversed(waiting):
                        completed.append(item['seed'])
                        future.set_result(dict(item,execution='completed',value=item['seed']*2))
                    return super().result(timeout=timeout)
            future=Deferred();self.waiting.append((future,args[2]));return future
    monkeypatch.setattr(p,'ProcessPoolExecutor',ReversePool)
    rows=list(p._scheduled_attempts(procedural_factory,p.Family(permutations=3),budget(2),
              time.monotonic(),cases=cases(budget())))
    assert completed == [78001,78000,78002,78003]
    assert [row['seed'] for row in rows] == [78000,78001,78002,78003]
    assert [row['value'] for row in rows] == [156000,156002,156004,156006]
