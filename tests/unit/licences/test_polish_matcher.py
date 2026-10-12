"""Field-contract oracle; names are not real corpus audio."""
import pytest
from sensorsim import licensing
from tests.unit.licences.test_matcher_fields import _train_with_metadata


@pytest.mark.parametrize('field',['attribution','corpus','collection','title','origin','references'])
@pytest.mark.parametrize('name,asset',[('DEMAND','demand'),('SINS','sins'),('BRAS','bras-v3')])
def test_provenance_fields_reject_restricted_names(field,name,asset):
    with pytest.raises(licensing.LicenseGateError,match=asset):
        licensing.require_commercial_lineage({field:name},context='development control')


@pytest.mark.parametrize('identifier',['10.14279/depositonce-6726','10.14279/depositonce-6726.2',
    'bras/1_scene_descriptions-CR2.zip','bras/2_source_and_receiver_descriptions-Genelec_8020c.zip'])
def test_distinctive_bras_versions_and_archives(identifier):
    with pytest.raises(licensing.LicenseGateError,match='bras-v3'):
        licensing.require_commercial_lineage({'path':identifier},context='development control')


@pytest.mark.parametrize('name',['chime home','esc 10','desed'])
def test_ambiguous_names_only_in_declared_fields(name,tmp_path,monkeypatch):
    _train_with_metadata(tmp_path,monkeypatch,{'label':name,'file':name+'/take.wav'},parent=name)
    with pytest.raises(licensing.LicenseGateError):
        licensing.require_commercial_lineage({'dataset':name},context='development control')
