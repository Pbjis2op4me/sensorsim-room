"""Unsigned development controls; fixture signatures are not reviewer receipts."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sensorsim.evaluation import power_audit as p
from tools import p5_power_audit as runner
from tools import power_audit as original


def fixture(tmp_path, *, nice=19, threads=None):
    lane=tmp_path/'lane'; lane.mkdir()
    custody=tmp_path/'independent'; custody.mkdir()
    args=SimpleNamespace(registration=lane/'registration.json', custody=custody/'digest.txt',
                         control_record=custody/'controls.json', fresh_seeds=custody/'development-seeds.json',
                         output=lane/'output', workers=8)
    spec=p.Family(permutations=3)
    budget=p.Budget(group_counts=(2,), null_datasets=1, alternative_datasets=1, workers=8)
    env=runner.environment()
    # Inject the resource probe, not the guard: the caller need not be nice 19.
    env.update(nice=nice, thread_environment=(
        dict.fromkeys(p.THREAD_VARIABLES, '2') if threads is None else threads))
    nodes=['unsigned-development-parser-fixture']
    reg=dict(schema='p5-launch-registration/1', family=asdict(spec), budget=asdict(budget),
        candidate_revision=env['git_revision'], runner_argv=['development-only'],
        environment_sha256=p.digest(original.environment_identity(env)), family_sha256=spec.sha256,
        budget_sha256=p.digest(asdict(budget)), qualification_scope_sha256=original.procedural_scope_identity(),
        implementation_sha256=p.implementation_fingerprint(execution='decision'),
        file_sha256={'tools/p5_power_audit.py':runner.file_sha(runner.__file__)},external_file_sha256={},
        required_test_node_ids=nodes,required_test_count=1,execution='decision',retry_policy='none')
    args.registration.write_text(json.dumps(reg))
    digest=runner.file_sha(args.registration)
    args.custody.write_text(digest+'\n')
    controls=dict(decision='qualified', verifier='UNSIGNED DEVELOPMENT FIXTURE', registration_sha256=digest,
        candidate_revision=reg['candidate_revision'],environment_sha256=reg['environment_sha256'],
        implementation_sha256=reg['implementation_sha256'],family_sha256=spec.sha256,
        qualification_scope_sha256=reg['qualification_scope_sha256'],required_test_node_ids=nodes,
        collected_test_node_ids=nodes,passed_test_node_ids=nodes,unexpected_skips_xfails=0,
        known_correct_accepted=True,known_bad_rejected=True,host_spawn_workers=[1,2,4,8],
        four_group_resolution_acknowledged=True)
    args.control_record.write_text(json.dumps(controls))
    return args,reg,env,controls


def test_correct_detached_control_reaches_seed_gate_without_reading_seeds(tmp_path):
    args,reg,env,_=fixture(tmp_path)
    spec,budget,controls,digest=runner.validate_bindings(args,reg,env,['development-only'])
    assert budget.workers == 8 and controls['record_sha256']
    with pytest.raises(FileNotFoundError):
        runner.load_fresh_cases(args.fresh_seeds,budget,digest)


@pytest.mark.parametrize('mutation', ['workers','argv','revision','environment','family','budget','scope',
    'implementation','file','custody','control_digest','missing_test','skip','xfail','unknown_worker',
    'bad_mutant','correct_control','resolution','mode','retry','empty','duplicate'])
def test_binding_mutants_are_refused_before_fresh_file(tmp_path, mutation):
    args,reg,env,control=fixture(tmp_path)
    if mutation == 'workers': args.workers=4
    elif mutation == 'argv': reg['runner_argv']=[]
    elif mutation == 'revision': reg['candidate_revision']='bad'
    elif mutation == 'environment': reg['environment_sha256']='bad'
    elif mutation == 'family': reg['family_sha256']='bad'
    elif mutation == 'budget': reg['budget_sha256']='bad'
    elif mutation == 'scope': reg['qualification_scope_sha256']='bad'
    elif mutation == 'implementation': reg['implementation_sha256']='bad'
    elif mutation == 'file': reg['file_sha256']['tools/p5_power_audit.py']='bad'
    elif mutation == 'custody': args.custody.write_text('bad')
    elif mutation == 'control_digest': control['registration_sha256']='bad'
    elif mutation == 'missing_test': control['passed_test_node_ids']=[]
    elif mutation in ('skip','xfail'): control['unexpected_skips_xfails']=1
    elif mutation == 'unknown_worker': control['host_spawn_workers']=[1,2]
    elif mutation == 'bad_mutant': control['known_bad_rejected']=False
    elif mutation == 'correct_control': control['known_correct_accepted']=False
    elif mutation == 'resolution': control['four_group_resolution_acknowledged']=False
    elif mutation == 'mode': reg['execution']='full'
    elif mutation == 'retry': reg['retry_policy']='automatic'
    elif mutation == 'empty': reg['required_test_node_ids']=[]
    elif mutation == 'duplicate': reg['required_test_node_ids']*=2
    args.control_record.write_text(json.dumps(control))
    expected = {
        'workers': 'frozen runner argv or workers changed',
        'argv': 'frozen runner argv or workers changed',
        'revision': 'candidate revision changed',
        'environment': 'frozen imported code or environment changed',
        'family': 'frozen family or budget changed',
        'budget': 'frozen family or budget changed',
        'scope': 'frozen fixture, group formula or split changed',
        'implementation': 'decision implementation changed',
        'file': 'frozen file changed',
        'custody': 'independent custody digest mismatch',
        'empty': 'empty or duplicate required collection',
        'duplicate': 'empty or duplicate required collection',
        'mode': 'execution or retry policy changed',
        'retry': 'execution or retry policy changed',
    }.get(mutation, 'reviewer has not qualified the complete frozen controls')
    with pytest.raises(ValueError, match=expected):
        runner.validate_bindings(args,reg,env,['development-only'])
    assert not args.fresh_seeds.exists()


@pytest.mark.parametrize('nice', [0, 10, 18])
def test_launcher_rejects_wrong_niceness(tmp_path, nice):
    args,reg,env,_=fixture(tmp_path, nice=nice)
    with pytest.raises(ValueError, match='P5 requires nice 19 and two numerical threads'):
        runner.validate_bindings(args,reg,env,['development-only'])
    assert not args.fresh_seeds.exists()


@pytest.mark.parametrize('name', [
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
])
@pytest.mark.parametrize('value', [None, '1', '3'])
def test_launcher_rejects_wrong_thread_probe(tmp_path, name, value):
    threads=dict.fromkeys(p.THREAD_VARIABLES, '2')
    threads[name]=value
    args,reg,env,_=fixture(tmp_path, threads=threads)
    with pytest.raises(ValueError, match='P5 requires nice 19 and two numerical threads'):
        runner.validate_bindings(args,reg,env,['development-only'])
    assert not args.fresh_seeds.exists()


def test_explicit_development_seed_file_binds_and_preserves_bytes(tmp_path):
    args,reg,env,_=fixture(tmp_path)
    _,budget,_,digest=runner.validate_bindings(args,reg,env,['development-only'])
    plan=[dict(row,seed=78100+i) for i,row in enumerate(p.planned_cases(budget))]
    value=dict(schema='p5-fresh-seeds/1',registration_sha256=digest,cases=plan)
    args.fresh_seeds.write_text(json.dumps(value))
    rows,actual=runner.load_fresh_cases(args.fresh_seeds,budget,digest)
    assert rows == plan and actual == sha256(args.fresh_seeds.read_bytes()).hexdigest()
    value['registration_sha256']='wrong'
    args.fresh_seeds.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='bind'):
        runner.load_fresh_cases(args.fresh_seeds,budget,digest)


def test_execute_missing_seed_has_raw_error_and_no_factory(tmp_path,monkeypatch,capsys):
    args,reg,env,_=fixture(tmp_path)
    monkeypatch.setattr(runner,'environment',lambda: env)
    args.output.mkdir()
    def never(*a,**kw):
        raise AssertionError('seed failure must not reach generator')
    monkeypatch.setattr(p,'qualify_test',never)
    assert runner.execute(args,['development-only']) == 1
    metadata=json.loads((args.output/'attempt.json').read_text())
    assert metadata['execution'] == 'errored' and metadata['exit_code'] == 1
    assert metadata['preregistration_sha256'] == runner.file_sha(args.registration)
    assert not (args.output/'replicates.jsonl').exists()
    assert 'FileNotFoundError' in capsys.readouterr().err


def test_driver_deadline_kills_only_owned_process_group(tmp_path,monkeypatch):
    # No real process is killed by this development control.
    reg=tmp_path/'registration.json';reg.write_text(json.dumps({'hard_wall_seconds':.001}))
    out=tmp_path/'output';signals=[]
    class HungChild:
        pid=912345
        def __init__(self,*a,**kw):
            self.count=0
            assert kw['start_new_session'] is True
        def wait(self,timeout=None):
            self.count+=1
            if self.count==1:raise runner.subprocess.TimeoutExpired('fixture',timeout)
            return -15
        def poll(self):return None
    monkeypatch.setattr(runner.subprocess,'Popen',HungChild)
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    code=runner.main(['--registration',str(reg),'--custody',str(tmp_path/'none'),
        '--control-record',str(tmp_path/'none2'),'--fresh-seeds',str(tmp_path/'none3'),
        '--output',str(out),'--workers','8'])
    assert code==124
    assert signals==[(912345,runner.signal.SIGTERM),(912345,runner.signal.SIGKILL)]
    assert json.loads((out/'driver.json').read_text())['exit_code']==124
    assert (out/'stdout.txt').exists() and (out/'stderr.txt').exists()
    with pytest.raises(FileExistsError):
        runner.main(['--registration',str(reg),'--custody','unused','--control-record','unused',
                     '--fresh-seeds','unused','--output',str(out),'--workers','8'])
