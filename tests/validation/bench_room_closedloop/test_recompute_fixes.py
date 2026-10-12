"""Development regression controls for the independent dEchorate review.

Procedural fixtures only. Expected ancestry follows declared metadata, never
an inference from the samples or an arbitrary project directory.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from sensorsim import licensing, project
from tests.unit.project.test_project import _layout, quiet


@pytest.mark.parametrize('ancestor', ['ordinary', 'dechorate-recompute', 'dechorate',
    'MeshRIR-notes', 'meshrir', 'fsd50k', 'DECHORATE_RECOMPUTE', 'dechorate.recompute'])
@pytest.mark.parametrize('field', ['path', 'recording_file', 'source_path', 'source'])
def test_incidental_ancestor_is_not_lineage(ancestor, field):
    value = {field: f'/ordinary/{ancestor}/user-project/generated.wav',
             'provenance': {'source': 'user-owned procedural signal', 'license': 'Apache-2.0'}}
    assert licensing.lineage_metadata(value) == {}
    licensing.require_commercial_lineage(value, context='user generated audio')
    value[field] = value[field].replace('/', '\\')
    assert licensing.lineage_metadata(value) == {}


@pytest.mark.parametrize('word', ['dechorate', 'meshrir', 'fsd50k'])
@pytest.mark.parametrize('field', ['label', 'uploader', 'notes'])
def test_descriptive_names_do_not_assert_dataset(word, field):
    assert licensing.lineage_metadata({'source': {field: word}}) == {}


RESTRICTED = [
    {'dataset': 'dEchorate'}, {'source': {'registry_id': 'dechorate-hdf5-v2'}},
    {'license': 'CC-BY-NC-4.0'}, {'internal_validation_only': True},
    {'source_url': 'https://zenodo.org/records/6576203'},
    {'doi': '10.14279/depositonce-6726.3'},
    {'restricted_assets': ['dechorate-hdf5-v2']},
]


@pytest.mark.parametrize('metadata', RESTRICTED)
def test_declared_restrictions_reach_training_and_export(metadata, tmp_path):
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps({'lineage': metadata})+'\n')
    assert licensing.lineage_metadata(metadata)
    with pytest.raises(licensing.LicenseGateError):
        licensing.validate_training_lineage(manifest)
    with pytest.raises(licensing.LicenseGateError):
        licensing.validate_project_lineage(tmp_path, metadata)


@pytest.mark.parametrize('name', ['dechorate_rir_processed.hdf5',
    'dEchorate_room000000_src7_arr5_mics21-25.sofa',
    'MeshRIR_S32-M441_npy.zip', 'FSD50K.dev_audio.zip',
    'BRAS/CR2/LS1_MP1.wav', 'bras/1_scene_descriptions-CR2.zip',
    'bras/2_source_and_receiver_descriptions-Genelec_8020c.zip'])
def test_declared_data_file_components_remain_restricted(name):
    for value in (name, '/user/archives/'+name, ('C:/user/archives/'+name).replace('/', '\\')):
        with pytest.raises(licensing.LicenseGateError):
            licensing.require_commercial_lineage({'recording_file': value}, context='data file')


def test_same_content_relocation_trains_and_exports(tmp_path, monkeypatch):
    original = _layout(tmp_path/'neutral'/'project')
    from scipy.io import wavfile
    from tests.unit.project.test_project import _chamber_tone
    from tests.unit.sources.target_standins import approve_project_rows
    for label, hz in (('test tone 0', 700.), ('test tone 1', 1900.)):
        (original/'targets'/label).mkdir(parents=True)
        for k in range(2):
            wavfile.write(original/'targets'/label/f'clean_{k}.wav', 24000, _chamber_tone(hz, k))
    cfg = yaml.safe_load((original/'project.yaml').read_text())
    cfg['dataset'].update(fs=16000, count=3, sources=[dict(
        file=f'{label}/clean_{k}.wav', label=label, source_group=f'{label}-take-{k}',
        object_id=label+'-oscillator', session='proxy-session', capture_route='near_dry', evidence_label='proxy')
        for label in ('test tone 0', 'test tone 1') for k in range(2)])
    cfg['recognizer'] = dict(type='cnn', steps=1, width=8, blocks=2, batch=4,
                             workers=0, threads=1, val_windows=8)
    (original/'project.yaml').write_text(yaml.safe_dump(cfg))
    approved = approve_project_rows(project.load_project(original), original/'qa-preview')
    cfg['dataset']['sources'] = approved['dataset']['sources']
    (original/'project.yaml').write_text(yaml.safe_dump(cfg))
    roots = [original]
    for ancestor in ('dechorate', 'meshrir', 'fsd50k', 'DECHORATE_RECOMPUTE'):
        dest = tmp_path/ancestor/'project'
        shutil.copytree(original, dest)
        roots.append(dest)
    hashes = [{p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in root.rglob('*.wav')} for root in roots]
    assert hashes[0] and all(h == hashes[0] for h in hashes)
    gate = licensing.require_commercial_lineage
    calls = []
    def observed(*values, context):
        calls.append(context)
        return gate(*values, context=context)
    monkeypatch.setattr(licensing, 'require_commercial_lineage', observed)
    trace = []
    for root in roots:
        calls.clear()
        report = project.run(root, echo=quiet)
        assert report['train']['status'] == 'ran'
        assert 'dataset generation' in calls and 'training manifest' in calls
        project.export(root, root/'device')
        assert 'model export' in calls
        assert (root/'device'/'model.npz').is_file()
        from sensorsim.edge import Model
        model = Model(str(root/'out'/'model.npz'))
        assert licensing.lineage_metadata(model.meta) == {}
        trace.append(dict(root=str(root), gate_contexts=list(calls), trained=True, exported=True))
    if os.environ.get('LANE_PYTEST_EVIDENCE'):
        evidence = Path(os.environ['LANE_PYTEST_EVIDENCE']).with_name('same-content-trace.json')
        evidence.write_text(json.dumps(dict(input_wav_sha256=hashes, routes=trace), indent=2)+'\n')
    # Moving an explicitly restricted source never clears the saved declaration.
    for root in roots:
        with pytest.raises(licensing.LicenseGateError):
            licensing.validate_project_lineage(root/'out', {'dataset': 'dEchorate'})


def test_legacy_entrypoint_refuses_before_any_io(tmp_path, monkeypatch):
    from tests.validation.bench_room_recheck import run
    reached = []
    def forbidden(*args, **kwargs):
        reached.append('forbidden IO or SOFA loader')
        raise AssertionError('legacy entrypoint performed IO')
    with monkeypatch.context() as patch:
        patch.setattr(run, '_load_pairs', forbidden)
        patch.setattr(Path, 'open', forbidden)
        patch.setattr(Path, 'mkdir', forbidden)
        with pytest.raises(RuntimeError, match='retired.*preflight_v1'):
            run.run(tmp_path/'missing', tmp_path/'out', tmp_path/'missing.yaml')
    assert reached == []


@pytest.mark.parametrize('splits', [None, ('fit_absorption', 'validation_only'), ('locked_audit',)])
def test_direct_legacy_loader_refuses_before_glob(tmp_path, monkeypatch, splits):
    from tests.validation.bench_room_recheck import run
    reached = []
    def forbidden(*args, **kwargs):
        reached.append('glob')
        raise AssertionError('legacy loader listed inputs')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'glob', forbidden)
        with pytest.raises(RuntimeError, match='retired.*HDF5'):
            run._load_pairs(tmp_path, 48000, allowed_splits=splits)
    assert reached == []


def test_old_cli_refuses_with_loader_sentinel(tmp_path, monkeypatch):
    from tests.validation.bench_room_recheck import run
    reg = tmp_path/'legacy.yaml'
    reg.write_text(yaml.safe_dump({'comparisons': [
        {'id': 'wp8_a_matched_measured_rir', 'data': {'source': 'fixture 48 kHz'}, 'seeds': [1]},
        {'id': 'wp8_b_join_continuity'}, {'id': 'wp8_lf_preparation'}]}))
    def forbidden(*args, **kwargs):
        pytest.fail('CLI reached legacy SOFA loader')
    monkeypatch.setattr(run, '_load_pairs', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run', '--data-root', str(tmp_path/'inputs'),
        '--preregistration', str(reg), '--output', str(tmp_path/'out')])
    with pytest.raises(RuntimeError, match='retired.*preflight_v1'):
        run.main()
    assert not (tmp_path/'out').exists()


def test_old_command_in_fresh_process_refuses(tmp_path, python_subprocess_env):
    result = subprocess.run([sys.executable, '-m', 'tests.validation.bench_room_recheck.run',
        '--data-root', str(tmp_path/'absent'), '--preregistration', str(tmp_path/'absent.yaml'),
        '--output', str(tmp_path/'out')], env=python_subprocess_env, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert 'retired' in result.stderr and 'preflight_v1' in result.stderr
    assert not (tmp_path/'out').exists()


def test_descriptive_metric_units_missing_denominators_and_sign():
    import numpy as np
    from tests.validation.bench_room_recheck.source7_development import metric_summary
    def metrics(edt, clarity):
        return {'edt_s': [edt]*8, 't30_s': [edt]*8,
                'c50_db': [clarity]*8, 'drr_db': [clarity]*8}
    same = {'measured': metrics(2, -4), 'model_based': metrics(2, -4)}
    bad = {'measured': metrics(2, -4), 'model_based': metrics(1, -7)}
    missing = {'measured': metrics(np.nan, np.nan), 'model_based': metrics(1, -7)}
    assert metric_summary([same])['edt_s'][0]['median_absolute_error'] == 0
    result = metric_summary([bad, missing])
    assert result['edt_s'][0]['median_absolute_error'] == 50
    assert result['c50_db'][0]['median_absolute_error'] == 3
    assert result['edt_s'][0]['total'] == 2 and result['edt_s'][0]['finite_pairs'] == 1
    assert metric_summary([])['edt_s'][0]['median_absolute_error'] is None


@pytest.mark.parametrize('description', ['HUB5/SRE retransmissions in ReverbDB',
                                        'BRAS measured room acoustics/directivity'])
@pytest.mark.parametrize('field', ['source', 'dataset', 'citation'])
def test_declared_source_prose_with_slash_is_not_a_location(description, field):
    with pytest.raises(licensing.LicenseGateError):
        licensing.require_commercial_lineage({field: description}, context='declared source')
