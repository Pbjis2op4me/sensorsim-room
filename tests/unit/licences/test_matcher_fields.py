"""Deterministic matcher controls; all audio is generated synthetic proxy data.

The case oracle is the field contract, not another invocation of the matcher.
Training completion proves these are usable projects, not only empty matches.
"""
from datetime import date
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from sensorsim import licensing, project
from sensorsim.scene.raycore import RayCore
from tests.unit.project.test_project import _layout, quiet

BENIGN = {
    'on-demand-path': {'recording_file': '/recordings/on-demand/knock_01.wav'},
    'demand-label': {'label': 'demand response alarm'},
    'arni-path': {'path': '/recordings/arni/knock.wav'},
    'sins-path': {'recording_file': '/recordings/sins/take3.wav'},
    'bras-label': {'label': 'bras robot'},
    'door-chime': {'label': 'door chime', 'file': 'targets/door chime/clean_0.wav'},
    'cc-by': {'license': 'CC-BY-4.0', 'source_url': 'https://freesound.org/s/1/'},
    'cc0': {'licence': 'CC0-1.0'},
    'negated-note': {'notes': 'recorded by me; no noncommercial restriction'},
    'air-comment': {'comment': 'air v1 4 settings'},
    'pytest-path': {'recording_file': 'scratch/pytest-of-u/pytest-3/test_x0/take.wav'},
    'musanga': {'label': 'musanga leaves'},
}


def _train_with_metadata(tmp_path, monkeypatch, metadata, *, parent='ordinary', label=None):
    root = _layout(tmp_path/parent/'proj')
    manifest = root/'session'/'manifest.yaml'
    raw = yaml.safe_load(manifest.read_text())
    raw['lineage'] = metadata
    if label:
        raw['takes'][next(i for i, t in enumerate(raw['takes']) if t['type'] == 'real')]['label'] = label
    manifest.write_text(yaml.safe_dump(raw))
    calls = []
    original = licensing.require_commercial_lineage
    def observed(*values, context):
        calls.append((context, repr(values)))
        return original(*values, context=context)
    monkeypatch.setattr(licensing, 'require_commercial_lineage', observed)
    report = project.run(root, echo=quiet)
    assert report['train']['status'] == 'ran'
    assert (root/'out'/'model.json').is_file()
    assert any(context == 'dataset generation' for context, _ in calls)
    # The manifest carrier must reach the actual gate, including all test values.
    for value in metadata.values():
        if isinstance(value, str):
            assert any(value in text for _, text in calls), (value, calls)
    return root, calls


@pytest.mark.parametrize('case', BENIGN)
def test_review_benign_metadata_trains(case, tmp_path, monkeypatch):
    meta = BENIGN[case]
    _train_with_metadata(tmp_path, monkeypatch, meta, label=meta.get('label'))
    assert licensing.lineage_metadata(meta) == {}


@pytest.mark.parametrize('parent', ['on-demand', 'arni', 'sins'])
def test_common_word_project_directory_trains(parent, tmp_path, monkeypatch):
    _train_with_metadata(tmp_path, monkeypatch, {}, parent=parent)


@pytest.mark.parametrize('uploader', ['arni', 'demand', 'sins', 'bras'])
def test_cc_by_uploader_record_loads_and_trains(uploader, tmp_path, monkeypatch):
    from sensorsim.negatives.clips import load_clip
    from tests.unit.negatives.test_negatives import clip_fixture
    root, item, _, review = clip_fixture(tmp_path, 'CC-BY-4.0')
    item.update(registry_id='freesound-cc-by-4', freesound_id='123', uploader=uploader,
                source_url='https://freesound.org/sounds/123/',
                license_url='https://creativecommons.org/licenses/by/4.0/legalcode.en', modifications='none')
    loaded = load_clip(root, item, register_path=licensing._default_registry_path(),
                       review=review, as_of=date(2026, 10, 8))
    assert np.any(loaded.audio)
    assert loaded.provenance['licence_record']['uploader'] == uploader
    _train_with_metadata(tmp_path, monkeypatch, loaded.provenance)
    assert not licensing.lineage_metadata(loaded.provenance)


def test_taxonomy_citations_train(tmp_path, monkeypatch):
    import sensorsim.negatives
    taxonomy = yaml.safe_load(Path(sensorsim.negatives.__file__).with_name('taxonomy.yaml').read_text())
    assert 'INMP441' in repr(taxonomy)
    _train_with_metadata(tmp_path, monkeypatch, {'citations': taxonomy})
    assert not licensing.lineage_metadata(taxonomy)


@pytest.mark.parametrize('word,asset', [('arni','arni-rir'), ('demand','demand'), ('sins','sins'), ('bras','bras-v3')])
@pytest.mark.parametrize('field', ['dataset', 'source', 'rights', 'citation', 'source_name', 'dataset_id', 'DatabaseName'])
def test_common_word_in_declared_source_is_restricted(word, asset, field):
    result = licensing.lineage_metadata({field: word})
    assert asset in result['restricted_assets']
    with pytest.raises(licensing.LicenseGateError) as caught:
        licensing.require_commercial_lineage({'parent': [{field: word}]}, context='control')
    assert word in str(caught.value) and f'parent[0].{field}' in str(caught.value)


@pytest.mark.parametrize('value', ['fsd50k', 'meshrir', 'dechorate'])
@pytest.mark.parametrize('field', ['path', 'label', 'uploader'])
def test_bare_names_in_descriptive_fields_are_not_lineage(value, field):
    assert licensing.lineage_metadata({field: value}) == {}
    licensing.require_commercial_lineage({field: value}, context='correct control')
    with pytest.raises(licensing.LicenseGateError):
        licensing.require_commercial_lineage({'dataset': value}, context='declared bad control')


@pytest.mark.parametrize('value', ['bras_cr2', 'BRAS Genelec 8020c'])
@pytest.mark.parametrize('field', ['path', 'label', 'uploader'])
def test_qualified_names_are_not_arbitrary_path_ancestry(value, field):
    if field == 'path':
        assert licensing.lineage_metadata({field: value}) == {}
    else:
        with pytest.raises(licensing.LicenseGateError):
            licensing.require_commercial_lineage({field: value}, context='qualified product')


@pytest.mark.parametrize('value', ['BRAS/CR2/LS1.wav', '10.14279/depositonce-6726.3',
                                  'https://zenodo.org/records/1227121'])
@pytest.mark.parametrize('field', ['path', 'label', 'uploader'])
def test_distinctive_identifiers_still_restricted(value, field):
    with pytest.raises(licensing.LicenseGateError):
        licensing.require_commercial_lineage({field: value}, context='control')


def test_declared_bad_control_reaches_dataset_and_prevents_training(tmp_path, monkeypatch):
    with pytest.raises(licensing.LicenseGateError, match='dataset generation.*demand'):
        _train_with_metadata(tmp_path, monkeypatch, {'dataset': 'demand'})
    assert not (tmp_path/'ordinary'/'proj'/'out'/'model.json').exists()


def test_non_audio_rows_are_citations_not_ancestry():
    kinds = {'technical-document', 'documentation', 'numeric-standard-table',
             'historical-mathematical-reference', 'python-package', 'validation-tool'}
    for row in licensing.load_register()['assets']:
        if row.get('kind') in kinds:
            for key in ('id', 'name', 'url'):
                assert not licensing.lineage_metadata({'citation': row[key]}), row['id']


def test_empty_duplicate_constant_and_nested_metadata():
    assert licensing.lineage_metadata({}, [], '', None, np.zeros(8), np.ones(8)) == {}
    assert not licensing.lineage_metadata({'source': {'uploader': 'arni', 'path': 'on-demand/take.wav'},
                                          'source_path': 'arni/take.wav', 'source_group': 'demand'})
    assert licensing.lineage_metadata({'source': {'name': 'demand'}})['restricted_assets'] == ['demand']
    nested = {'source': {'dataset': 'demand'}, 'copy': [{'dataset': 'demand'}]}
    lineage = licensing.lineage_metadata(nested, nested)
    assert lineage['restricted_assets'] == ['demand']
    assert licensing.lineage_metadata({'rights': 'CC-BY-SA', 'child': {'bras_derived': False}})


def test_missing_register_is_not_an_allow_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(licensing, '_default_registry_path', lambda: tmp_path/'missing.yaml')
    with pytest.raises(licensing.LicenseGateError, match='cannot read lineage register'):
        licensing.lineage_metadata({'dataset': 'demand'})


def test_cpu_contract_has_no_removed_gpu_resources():
    assert 'vram_bytes' not in inspect.signature(RayCore).parameters
    with RayCore() as core:
        assert not hasattr(core, '_engine')
        assert not hasattr(core, 'vram_bytes')
        inp = core.backend_input()
        assert not hasattr(inp, 'vram_bytes')
        np.testing.assert_array_equal(core.any_hit([0,0,0], [1,1,1]), [False])
    with pytest.raises(RuntimeError, match='closed'):
        core.cpu_reference()


@pytest.mark.parametrize('text', ['', '{}', 'assets: []', 'assets: null'])
def test_empty_or_malformed_register_is_not_clearance(tmp_path, monkeypatch, text):
    register = tmp_path/'register.yaml'
    register.write_text(text)
    monkeypatch.setattr(licensing, '_default_registry_path', lambda: register)
    with pytest.raises(licensing.LicenseGateError, match='lineage register'):
        licensing.lineage_metadata({'source': 'demand'})
