"""Release review regressions; procedural proxies, no external audio."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sensorsim import licensing, recognizer
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.session_sim import simulate_session, Impairments
from sensorsim.acoustics.room.synth import SynthConfig
from sensorsim.scene.raycore import RayCore


@pytest.mark.parametrize('field', ['licence', 'license', 'rights', 'url', 'doi', 'citation',
                                   'source', 'recording_file', 'path', 'filename', 'custom_source'])
@pytest.mark.parametrize('value', [
    'Creative Commons Attribution-ShareAlike 4.0 International',
    'Attribution-NonCommercial 3.0', 'Creative Commons Attribution NoDerivatives 2.0',
    'CC-BY-NC-SA-4.0', 'CC BY ND 1.0', 'https://creativecommons.org/licenses/by-nc/4.0/',
    'https://doi.org/10.14279/depositonce-6726.3', 'BRAS/CR2/LS1_MP1.wav',
])
def test_declared_restrictions_across_provenance_fields(field, value):
    with pytest.raises(licensing.LicenseGateError):
        licensing.require_commercial_lineage({field: value}, context='proxy')


def test_bras_geographic_name_is_not_dataset_provenance():
    assert not licensing.lineage_metadata({'room_id': "Bras d'Or cabin"})
    assert not licensing.lineage_metadata({'name': 'zebras enclosure'})


def test_every_restricted_register_identifier_is_recognised():
    for row in licensing.load_register()['assets']:
        if row['disposition'] not in {'never', 'internal'} or row.get('kind') in {
                'technical-document', 'documentation', 'numeric-standard-table',
                'historical-mathematical-reference', 'python-package', 'validation-tool'}:
            continue
        for key in ('id', 'name', 'url'):
            with pytest.raises(licensing.LicenseGateError, match='internal-validation'):
                licensing.require_commercial_lineage({'source': row[key]}, context=row['id'])


@pytest.mark.parametrize('backend', ['argument', 'environment'])
def test_vulkan_is_removed_with_clear_error(monkeypatch, backend):
    if backend == 'environment':
        monkeypatch.setenv('SENSORSIM_RAY_BACKEND', 'vulkan')
    with pytest.raises(ValueError, match='Vulkan.*removed'):
        RayCore(backend='vulkan' if backend == 'argument' else None)


def test_auto_uses_only_cpu():
    with RayCore(backend='auto') as core:
        assert not core.any_hit([0, 0, 0], [1, 1, 1]).any()
        assert core.backend == 'cpu'
    import sensorsim.scene
    assert not Path(sensorsim.scene.__file__).with_name('raycore_vulkan.py').exists()


def test_internal_session_preserves_restriction_at_every_boundary(tmp_path):
    from tests.validation.bench_room_kit_e2e.test_session import tiny_kit
    model = RoomModel(room_id='bras_cr2', shoebox_dims_m=(6, 5, 4))
    truth = simulate_session(tmp_path/'session', model, {'S1': (1, 2, 2)}, {'M1': (2, 2, 2)},
        truth_dir=tmp_path/'truth', kit=tiny_kit(), impairments=Impairments.clean(8000), seed=7, noise_seconds=.1,
        synth_config=SynthConfig(max_order=1, late=False, duration_s=.1),
        purpose='internal-validation')
    manifest = yaml.safe_load(truth.manifest.read_text())
    assert manifest['lineage']['bras_derived']
    assert json.loads(truth.evidence.read_text())['lineage']['bras_derived']
    assert all(m['lineage']['bras_derived'] for m in truth.rir_metadata.values())
    from sensorsim.acoustics.room.ingest import load_manifest
    loaded = load_manifest(truth.manifest)
    assert licensing.lineage_metadata(loaded)['bras_derived']
    with pytest.raises(licensing.LicenseGateError):
        licensing.validate_project_lineage(tmp_path, manifest)


def test_external_rir_lineage_refused_before_output(tmp_path):
    with pytest.raises(licensing.LicenseGateError):
        simulate_session(tmp_path/'session', RoomModel(shoebox_dims_m=(6, 5, 4)),
            {'S1': (1, 2, 2)}, {'M1': (2, 2, 2)}, truth_dir=tmp_path/'truth',
            external_rirs={('S1', 'M1'): np.ones(1)},
            external_rir_provenance={'rights': 'CC BY-NC 3.0'})
    assert not (tmp_path/'session').exists()


def test_restricted_filename_rejected_before_opening_dry_audio():
    from sensorsim.acoustics.room.dataset import DryClip, read_dry
    with pytest.raises(licensing.LicenseGateError):
        read_dry(DryClip('tone', Path('BRAS')/'CR2'/'take.wav'), 8000)


def test_synth_config_refuses_removed_backend_even_without_obstacles():
    with pytest.raises(ValueError, match='Vulkan.*removed'):
        SynthConfig(obstacle_backend='vulkan')


def test_test_tool_directory_is_not_derivative_audio_provenance():
    assert not licensing.lineage_metadata({'recording_file': 'scratch/pytest/take.wav'})
    assert not licensing.lineage_metadata({'source': 'pytest'})  # tool citation, not audio ancestry
