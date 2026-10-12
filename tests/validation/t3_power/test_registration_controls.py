"""Input-contract fixtures only; these unsigned examples authorize no acceptance."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sensorsim.evaluation import power_audit as p
from tools import power_audit as runner


def contract(tmp_path):
    spec, budget, env = p.Family(), p.Budget(), runner.environment()
    tests = {'tests/validation/t3_power/test_registration_controls.py':
             sha256(Path(__file__).read_bytes()).hexdigest()}
    required = ['development-contract-placeholder']
    controls = dict(decision='qualified', verifier='UNSIGNED development fixture ONLY',
                    family_sha256=spec.sha256,
                    qualification_scope_sha256=runner.procedural_scope_identity(),
                    implementation_sha256=p.implementation_fingerprint(),
                    environment_sha256=p.digest(runner.environment_identity(env)),
                    test_file_sha256=tests, required_test_node_ids=required,
                    collected_test_node_ids=required, unexpected_skips_xfails=0,
                    known_correct_accepted=True, known_bad_rejected=True)
    control_path = tmp_path/'controls.json'
    control_path.write_text(json.dumps(controls))
    registration = dict(family_sha256=spec.sha256, budget_sha256=p.digest(asdict(budget)),
                        environment_sha256=controls['environment_sha256'],
                        qualification_scope_sha256=controls['qualification_scope_sha256'],
                        code_sha256={name: m['sha256'] for name, m in env['modules'].items()},
                        test_file_sha256=tests, tool_sha256=env['tool_sha256'],
                        control_record_sha256=sha256(control_path.read_bytes()).hexdigest(),
                        required_test_node_ids=required)
    custody = tmp_path/'separate-contract-fixture'/'digest.txt'
    custody.parent.mkdir()
    args = SimpleNamespace(registration=str(tmp_path/'registration.json'), custody=str(custody),
                           control_record=str(control_path))
    return args, spec, budget, env, registration, controls


def write_contract(args, registration, controls):
    # Not independent custody: only tests the parser. Never call CLI acceptance.
    Path(args.control_record).write_text(json.dumps(controls))
    registration['control_record_sha256'] = sha256(Path(args.control_record).read_bytes()).hexdigest()
    Path(args.registration).write_text(json.dumps(registration))
    Path(args.custody).write_text(sha256(Path(args.registration).read_bytes()).hexdigest())


def test_control_binding_accepts_matching_contract_and_relocated_snapshot(tmp_path):
    args, spec, budget, env, registration, controls = contract(tmp_path)
    write_contract(args, registration, controls)
    checked, _ = runner.verifier_inputs(args, spec, budget, env)
    assert checked['qualification_scope_sha256'] == runner.procedural_scope_identity()
    relocated = deepcopy(env)
    for name, module in relocated['modules'].items():
        module['path'] = 'another-snapshot/'+name
    relocated['git_revision'] = ''  # archive provenance recorded separately
    assert runner.environment_identity(relocated) == runner.environment_identity(env)
    runner.verifier_inputs(args, spec, budget, relocated)


@pytest.mark.parametrize('mutation', ['scope', 'implementation', 'environment', 'test_binding',
                                     'missing_node', 'bad_control', 'changed_import', 'changed_test',
                                     'changed_generator'])
def test_stale_controls_or_candidate_cannot_authorize_launch(tmp_path, mutation):
    args, spec, budget, env, registration, controls = contract(tmp_path)
    if mutation == 'scope':
        controls['qualification_scope_sha256'] = '0'*64
    elif mutation == 'implementation':
        controls['implementation_sha256'] = '0'*64
    elif mutation == 'environment':
        controls['environment_sha256'] = '0'*64
    elif mutation == 'test_binding':
        controls['test_file_sha256'] = {}
    elif mutation == 'missing_node':
        controls['collected_test_node_ids'] = []
    elif mutation == 'bad_control':
        controls['known_bad_rejected'] = False
    elif mutation == 'changed_import':
        env['modules']['sensorsim.edge']['sha256'] = '0'*64
    elif mutation == 'changed_test':
        registration['test_file_sha256'] = {name: '0'*64 for name in registration['test_file_sha256']}
    elif mutation == 'changed_generator':
        registration['qualification_scope_sha256'] = '0'*64
    write_contract(args, registration, controls)
    with pytest.raises(ValueError):
        runner.verifier_inputs(args, spec, budget, env)
