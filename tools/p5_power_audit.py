#!/usr/bin/env python3
"""Frozen P5 launcher. No seed generation, automatic retries or acceptance receipts.

The lead supplies a detached custody digest, reviewer control record and fresh
case plan after freezing the registration. The launcher validates all bindings
before opening that plan. Numerical results and scheduling provenance are saved
separately. The outer process supervises only its own spawned process group.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

from sensorsim.evaluation import power_audit as audit
from tools import power_audit as original


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def environment():
    env = original.environment()
    for name in ('sensorsim.evaluation.permutation_decision', 'tools.power_audit'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        env['modules'][name] = dict(path=str(path), sha256=file_sha(path), version=None)
    # Bind all imported project code, including transitive feature/device helpers.
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('sensorsim.') and path and Path(path).is_file():
            env['modules'][name] = dict(path=str(Path(path).resolve()), sha256=file_sha(path), version=None)
    env['tool_sha256'] = file_sha(__file__)
    env['thread_environment'] = {k: os.environ.get(k) for k in audit.THREAD_VARIABLES}
    env['nice'] = os.getpriority(os.PRIO_PROCESS, 0)
    return env


def configuration(registration):
    family = dict(registration['family'])
    for name in ('tests', 'strata'):
        family[name] = tuple(tuple(x) for x in family[name])
    budget = dict(registration['budget'])
    for name in ('group_counts', 'defects'):
        budget[name] = tuple(budget[name])
    return audit.Family(**family), audit.Budget(**budget)


def validate_bindings(args, registration, env, argv):
    """Detached reviewer signoff binds this already-frozen registration."""
    root = Path(audit.__file__).resolve().parents[3]
    spec, budget = configuration(registration)
    reg_sha = file_sha(args.registration)
    if registration['schema'] != 'p5-launch-registration/1':
        raise ValueError('unsupported registration schema')
    if args.workers != budget.workers or list(argv) != registration['runner_argv']:
        raise ValueError('frozen runner argv or workers changed')
    if registration['candidate_revision'] != env['git_revision']:
        raise ValueError('candidate revision changed; merged trees require new review')
    if registration['environment_sha256'] != audit.digest(original.environment_identity(env)):
        raise ValueError('frozen imported code or environment changed')
    if registration['family_sha256'] != spec.sha256 or registration['budget_sha256'] != audit.digest(asdict(budget)):
        raise ValueError('frozen family or budget changed')
    if registration['qualification_scope_sha256'] != original.procedural_scope_identity():
        raise ValueError('frozen fixture, group formula or split changed')
    if registration['implementation_sha256'] != audit.implementation_fingerprint(execution='decision'):
        raise ValueError('decision implementation changed')
    if env['nice'] != 19 or set(env['thread_environment'].values()) != {'2'}:
        raise ValueError('P5 requires nice 19 and two numerical threads per process')
    for relative, expected in registration['file_sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_sha(path) != expected:
            raise ValueError(f'frozen file changed: {relative}')
    for external, expected in registration['external_file_sha256'].items():
        if file_sha(external) != expected:
            raise ValueError(f'frozen external harness changed: {external}')
    for name in ('custody', 'control_record', 'fresh_seeds'):
        path = Path(getattr(args, name)).resolve()
        if path.is_relative_to(root) or path.is_relative_to(Path(args.registration).resolve().parent):
            raise ValueError('custody, controls and fresh cases must be independently held')
    if Path(args.custody).read_text().split()[0] != reg_sha:
        raise ValueError('independent custody digest mismatch')
    controls = json.loads(Path(args.control_record).read_text())
    required = registration['required_test_node_ids']
    if not required or len(required) != len(set(required)) or registration['required_test_count'] != len(required):
        raise ValueError('empty or duplicate required collection')
    if (controls.get('decision') != 'qualified' or not controls.get('verifier')
            or controls.get('registration_sha256') != reg_sha
            or controls.get('candidate_revision') != env['git_revision']
            or controls.get('environment_sha256') != registration['environment_sha256']
            or controls.get('implementation_sha256') != registration['implementation_sha256']
            or controls.get('family_sha256') != spec.sha256
            or controls.get('qualification_scope_sha256') != registration['qualification_scope_sha256']
            or controls.get('required_test_node_ids') != required
            or sorted(controls.get('collected_test_node_ids', [])) != sorted(required)
            or sorted(controls.get('passed_test_node_ids', [])) != sorted(required)
            or controls.get('unexpected_skips_xfails', 1) != 0
            or controls.get('known_correct_accepted') is not True
            or controls.get('known_bad_rejected') is not True
            or controls.get('host_spawn_workers') != [1, 2, 4, 8]
            or controls.get('four_group_resolution_acknowledged') is not True):
        raise ValueError('reviewer has not qualified the complete frozen controls')
    if registration.get('execution') != 'decision' or registration.get('retry_policy') != 'none':
        raise ValueError('execution or retry policy changed')
    controls['record_sha256'] = file_sha(args.control_record)
    return spec, budget, controls, reg_sha


def load_fresh_cases(path, budget, registration_sha256):
    """Only the acceptance runner opens this file, after every other guard."""
    raw = Path(path).read_bytes()  # missing file is an error; no fallback
    value = json.loads(raw)
    if (set(value) != {'schema', 'registration_sha256', 'cases'}
            or value['schema'] != 'p5-fresh-seeds/1'
            or value['registration_sha256'] != registration_sha256):
        raise ValueError('fresh seeds must bind this registration')
    return audit.validate_case_plan(budget, value['cases']), sha256(raw).hexdigest()


def execute(args, argv):
    output = Path(args.output)
    metadata = dict(command=[sys.executable, str(Path(__file__).resolve()), *argv], start=utc(),
                    execution='errored', retry_policy='none', evidence_domain='synthetic proxy')
    save(output/'attempt.json', metadata)
    code = 1
    try:
        registration = json.loads(Path(args.registration).read_text())
        env = environment()
        metadata['environment'] = env
        save(output/'attempt.json', metadata)
        spec, budget, controls, reg_sha = validate_bindings(args, registration, env, argv)
        metadata.update(preregistration_sha256=reg_sha, control_record_sha256=controls['record_sha256'])
        save(output/'attempt.json', metadata)
        plan, seed_sha = load_fresh_cases(args.fresh_seeds, budget, reg_sha)
        metadata.update(fresh_seeds_sha256=seed_sha, case_plan_sha256=audit.digest(plan), case_count=len(plan))
        save(output/'attempt.json', metadata)
        with (output/'replicates.jsonl').open('x') as raw, (output/'scientific-replicates.jsonl').open('x') as scientific, (output/'phases.jsonl').open('x') as phases:
            def record(row):
                raw.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                raw.flush()
                scientific.write(json.dumps({k: v for k, v in row.items() if k != 'worker'},
                                            sort_keys=True, allow_nan=False)+'\n')
                scientific.flush()
            def phase_record(row):
                phases.write(json.dumps(row, sort_keys=True)+'\n')
                phases.flush()
            report = audit.qualify_test(original.procedural_factory, spec, budget, record=record,
                    controls=controls, execution='decision', cases=plan,
                    phase_seconds={int(k): v for k, v in registration['phase_seconds'].items()},
                    phase_record=phase_record)
        (output/'scientific.json').write_bytes(audit.scientific_payload(report))
        report.pop('attempts')  # preserved verbatim in replicates.jsonl
        report.update(fresh_seeds_sha256=seed_sha, preregistration_sha256=reg_sha)
        save(output/'result.json', report)
        complete = all(c['harness'] != 'invalid' for c in report['cells'])
        # A statistical FAIL is a completed experiment. No new seed or retry.
        metadata['execution'] = 'completed' if complete else 'errored'
        code = 0 if complete else 2
    except Exception:
        traceback.print_exc()
    finally:
        metadata.update(end=utc(), exit_code=code)
        save(output/'attempt.json', metadata)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    child = '--execute' in argv
    if child:
        argv.remove('--execute')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'custody', 'control-record', 'fresh-seeds', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4, 8), required=True)
    args = parser.parse_args(argv)
    if child:
        if os.environ.get('SENSORSIM_P5_SUPERVISED') != '1':
            raise ValueError('execution requires the deadline supervisor')
        return execute(args, argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(command=[sys.executable, str(Path(__file__).resolve()), *argv], start=utc(),
                retry_policy='none', execution='errored', child_pid=None)
    save(output/'driver.json', meta)
    code = 1
    process = None
    def interrupted(signum, frame):
        raise InterruptedError(f'launch interrupted by signal {signum}')
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        registration = json.loads(Path(args.registration).read_text())
        limit = registration['hard_wall_seconds']
        if not isinstance(limit, (int, float)) or not 0 < limit <= 86400:
            raise ValueError('invalid hard wall deadline')
        with (output/'stdout.txt').open('x') as stdout, (output/'stderr.txt').open('x') as stderr:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--execute', *argv],
                stdout=stdout, stderr=stderr, start_new_session=True,
                env=dict(os.environ, SENSORSIM_P5_SUPERVISED='1'))
            meta['child_pid'] = process.pid
            save(output/'driver.json', meta)
            try:
                code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                meta['error'] = 'hard wall deadline exhausted; no automatic retry'
                code = 124
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    # The leader can exit before its descendants. Always address
                    # the owned group, not just a still-live leader.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        meta['execution'] = 'completed' if code == 0 else 'errored'
    except Exception as exc:
        meta['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        meta.update(end=utc(), exit_code=code)
        save(output/'driver.json', meta)
        signal.signal(signal.SIGTERM, previous_handler)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
