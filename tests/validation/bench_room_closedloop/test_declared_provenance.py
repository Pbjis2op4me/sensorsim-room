"""Procedural regression controls from the fix2 verifier's versioned-ancestry bypass.

Literal field contracts are the oracle. These deterministic route controls say
nothing about acoustic realism or whether unknown audio has clean provenance.
"""
import json
import os
from pathlib import Path

import pytest

from sensorsim import licensing
from tests.unit.licences.test_matcher_fields import _train_with_metadata


def _trace(row):
    output = os.environ.get('VERIFY_OUTPUT')
    if output:
        with (Path(output) / 'declared-provenance.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')


@pytest.mark.parametrize('declaration,asset', [
    ('dEchorate/v1', 'dechorate-hdf5-v2'),
    ('MeshRIR/v1.2', 'meshrir-v1-2'),
    ('dEchorate\\v1', 'dechorate-hdf5-v2'),
    ('%64Echorate%2Fv1', 'dechorate-hdf5-v2'),
    ('ｄＥｃｈｏｒａｔｅ／v1', 'dechorate-hdf5-v2'),
])
@pytest.mark.parametrize('field', [
    'dataset', 'dataset_id', 'datasets', 'database', 'DatabaseName', 'corpus',
    'collection', 'registry_id', 'source', 'source_name', 'citation',
])
def test_versioned_declarations_reach_both_gates(declaration, asset, field, tmp_path):
    """A slash alone cannot turn declared ancestry into an incidental location."""
    metadata = {field: declaration}
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'lineage': metadata}) + '\n')
    lineage = licensing.lineage_metadata(metadata)
    observations = []
    for function, call in [
        ('validate_training_lineage', lambda: licensing.validate_training_lineage(manifest)),
        ('validate_project_lineage', lambda: licensing.validate_project_lineage(tmp_path, metadata)),
    ]:
        try:
            call()
        except licensing.LicenseGateError as exc:
            observations.append(dict(function=function, blocked=True, reason=str(exc)))
        else:
            observations.append(dict(function=function, blocked=False))
    _trace(dict(metadata=metadata, lineage=lineage, calls=observations))
    assert lineage.get('restricted_assets') == [asset]
    assert all(row['blocked'] for row in observations), observations
    assert all(field in row['reason'] for row in observations)


@pytest.mark.parametrize('metadata', [
    {'dataset': '/catalogue/dEchorate/v1'},
    {'registry_id': './dEchorate/v1'},
    {'datasets': [{'name': 'dEchorate/v1'}]},
    {'source': {'dataset': {'name': 'dEchorate/v1'}}},
    {'dataset': {'identifier': 'C:\\catalogue\\dEchorate\\v1'}},
    {'dataset': ['dEchorate/v1', 'dEchorate/v1']},
])
def test_explicit_dataset_identity_survives_path_spelling_and_nesting(metadata):
    assert licensing.lineage_metadata(metadata)['restricted_assets'] == ['dechorate-hdf5-v2']
    with pytest.raises(licensing.LicenseGateError):
        licensing.require_commercial_lineage(metadata, context='explicit dataset identity')


@pytest.mark.parametrize('metadata', [
    {'source': '/ordinary/dechorate-recompute/my-project/generated.wav'},
    {'source': 'projects/dechorate-recompute/generated.wav'},
    {'source_path': 'dEchorate/v1'},
    {'dataset_path': 'dEchorate/v1'},
    {'dataset': {'path': '/ordinary/dechorate/project.wav', 'notes': 'dEchorate/v1'}},
    {'source': {'uploader': 'MeshRIR/v1.2'}},
    {'dataset': 'user-owned/v1'},
    {'dataset': []},
])
def test_incidental_paths_and_descriptions_still_clear(metadata):
    assert licensing.lineage_metadata(metadata) == {}
    licensing.require_commercial_lineage(metadata, context='benign counterpart')


@pytest.mark.parametrize('metadata,blocked', [
    ({'source': 'user-owned procedural signal', 'license': 'Apache-2.0'}, False),
    ({'dataset': 'dEchorate'}, True),
    ({'dataset': 'dEchorate/v1'}, True),
    ({'dataset': 'MeshRIR/v1.2'}, True),
])
def test_declared_ancestry_refuses_actual_training(metadata, blocked, tmp_path, monkeypatch):
    """Observe the real gate with intact metadata and require no model on refusal."""
    calls = []
    lineage_calls = []
    original_lineage = licensing.lineage_metadata
    original = licensing.require_commercial_lineage

    def observed_lineage(*values):
        result = original_lineage(*values)
        lineage_calls.append(dict(values=repr(values), result=result))
        return result

    def observed(*values, context):
        call = dict(context=context, values=repr(values), blocked=False)
        calls.append(call)
        try:
            return original(*values, context=context)
        except licensing.LicenseGateError:
            call['blocked'] = True
            raise

    monkeypatch.setattr(licensing, 'lineage_metadata', observed_lineage)
    monkeypatch.setattr(licensing, 'require_commercial_lineage', observed)
    model = tmp_path / 'ordinary/proj/out/model.json'
    was_blocked = False
    try:
        _train_with_metadata(tmp_path, monkeypatch, metadata)
    except licensing.LicenseGateError:
        was_blocked = True
    _trace(dict(metadata=metadata, calls=calls, lineage_calls=lineage_calls,
                blocked=was_blocked, model_created=model.exists()))
    assert calls and any(call['context'] == 'dataset generation' for call in calls)
    assert all(any(str(value) in call['values'] for call in calls + lineage_calls)
               for value in metadata.values())
    if was_blocked:
        asset = 'meshrir-v1-2' if metadata.get('dataset', '').startswith('MeshRIR') else 'dechorate-hdf5-v2'
        assert any(asset in call['result'].get('restricted_assets', []) for call in lineage_calls)
        assert any(asset in call['values'] and call['blocked'] for call in calls)
    assert was_blocked == blocked
    assert model.exists() == (not blocked)
    assert any(call['blocked'] for call in calls) == blocked


@pytest.mark.parametrize('suffix', ['.flac', '.ogg', '.mp3', '.aiff', '.npz', '.csv', '.json'])
def test_relative_source_files_are_locations_but_dataset_fields_stay_strict(suffix):
    value = 'dechorate-recompute/procedural' + suffix
    assert licensing.lineage_metadata({'source': value}) == {}
    with pytest.raises(licensing.LicenseGateError):
        licensing.require_commercial_lineage({'dataset': value}, context='explicit dataset')
