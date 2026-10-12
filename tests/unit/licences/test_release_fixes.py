"""WP9 release controls; all audio/BRAS markers here are procedural proxies."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import iot_mic, licensing, project, recognizer
from sensorsim.acoustics import bands
from sensorsim.core import geometry
from sensorsim.core import schema
from sensorsim.acoustics.room import dataset
from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.directivity import Directivity, Orientation
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from tests.unit.acoustics_room_synth.test_directivity_scatter import profile


def test_unreferenced_coordinate_table_removed():
    assert not hasattr(geometry, "APPENDIX_A_MICS_M")
    assert not hasattr(geometry, "RAISED_MIC_INDEX")
    assert not hasattr(schema, "AcousticNodeSpec")            # the scenario node schema is not shipped


def test_renard_labels_identical_to_base_literals_without_copying_table():
    # SHA-256 of json.dumps(list(tuple)) at export base 3fb1667. No source table in this fixture.
    expected = {
        "THIRD_NOMINAL_HZ": "67876596e0ff0dfafec0b73a4804e74dae82053453a51495187c260f7c096389",
        "OCTAVE_NOMINAL_HZ": "b5134252f9b441f1a3dd347338b0fcfcee171df5e61ead96a7211fb97648b893",
        "OCTAVE_NOMINAL_EXTENDED_HZ": "e2e50e2cc1fadfe45bcaae85d939079c90dc9b3fce2f892a7a8651a6d4f8ce43",
        "THIRD_NOMINAL_100_3150_HZ": "0163ff33871fea3e2a2421a9a6f469c2e35367ea0cd6cd77cc5e6051037a4b05",
        "OCTAVE_NOMINAL_125_8000_HZ": "61f70433f7c1abda7f95227740e46dfd38fddda241f074bc055fca7b1de167c6",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(json.dumps(list(getattr(bands, name))).encode()).hexdigest() == digest
    assert bands.renard_nominal_hz(-17) == bands.THIRD_NOMINAL_HZ[0]
    assert bands.renard_nominal_hz(13) == bands.THIRD_NOMINAL_HZ[-1]


def test_project_quantizer_generated_bounded_and_not_ima():
    assert not hasattr(iot_mic, "_IMA_STEPS")
    assert not hasattr(iot_mic, "_IMA_INDEX")
    assert np.all(np.diff(iot_mic._ADPCM_STEPS) > 0)
    assert iot_mic._ADPCM_STEPS[-1] == 32767
    for x in (np.zeros(64), np.ones(64), -np.ones(64), np.linspace(-2, 2, 100)):
        y = iot_mic.adpcm_like(x)
        assert y.shape == x.shape and np.isfinite(y).all() and np.max(abs(y)) <= 1
        assert np.array_equal(y, iot_mic.ima_adpcm(x))  # old API name is a documented compatibility alias


def marked_directivity():
    p = profile().model_copy(update={"provenance": {"source": "https://doi.org/10.14279/depositonce-6726.3"}})
    return Directivity(pattern="measured", profile=p, orientation=Orientation())


@pytest.mark.parametrize("location", ["source", "receiver", "model", "config"])
def test_bras_lineage_survives_rir_serialization(location):
    kwargs = {location + "_directivity": marked_directivity()} if location in ("source", "receiver") else {}
    m = RoomModel(shoebox_dims_m=(6, 5, 4), **kwargs)
    cfg = SynthConfig(max_order=1, late=False, duration_s=.1)
    if location == "model":
        m = m.model_copy(update={"provenance": {"dataset": "BRAS-RS1-rigid"}})
    if location == "config":
        cfg = cfg.model_copy(update={"source_directivity": marked_directivity()})
    m = RoomModel.from_json(m.to_json())
    rir = synthesize(m, (1, 2, 2), (2, 2, 2), 8000, config=cfg)
    assert json.loads(json.dumps(rir.metadata))["lineage"]["bras_derived"] is True


def test_dataset_refuses_planted_bras_render_before_wav(tmp_path, monkeypatch):
    m = RoomModel(shoebox_dims_m=(6, 5, 4))
    rir = synthesize(m, (1, 2, 2), (2, 2, 2), 8000, config=SynthConfig(max_order=1, late=False, duration_s=.1))
    rir.metadata["lineage"] = {"bras_derived": True}
    monkeypatch.setattr(dataset, "synthesize", lambda *a, **k: rir)
    with pytest.raises(licensing.LicenseGateError, match="BRAS"):
        dataset.generate_dataset(m, dataset.SamplingSpec(), [dataset.DryClip("tone", np.ones(64)*.01, 8000)],
            AugmentationConfig(), 1, 0, tmp_path/"data", fs=8000, val_fraction=0, peak_policy="scale")
    assert not list(tmp_path.rglob("*.wav"))


@pytest.mark.parametrize("marker", [{"lineage": {"bras_derived": True}}, {"dataset": "BRAS-CR2"},
                                     {"license": "CC-BY-SA-4.0"}])
def test_training_manifest_refuses_marked_render(tmp_path, marker):
    manifest = tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps({"file": "planted.wav", "fs": 8000, "label": "knock", "split": "train", "rir": marker}) + "\n")
    with pytest.raises(licensing.LicenseGateError, match="internal-validation"):
        recognizer._load_manifest(manifest)  # shared by baseline and CNN Corpus


@pytest.mark.parametrize("where", ["model", "manifest", "run", "fit"])
def test_export_refuses_bras_lineage_before_creating_folder(tmp_path, monkeypatch, where):
    root = tmp_path/"project"
    out = root/"out"
    out.mkdir(parents=True)
    (out/"model.npz").write_bytes(b"proxy")
    from sensorsim import edge
    monkeypatch.setattr(edge, "Model", lambda p: SimpleNamespace(meta={"lineage": {"bras_derived": True}} if where == "model" else {}, classes=["knock"], weights={}, macs=lambda: 0))
    monkeypatch.setattr(project, "_deploy_notes", lambda model: "proxy")
    marker = {"lineage": {"bras_derived": True}}
    if where in ("manifest", "run"):
        (out/"dataset").mkdir()
        name = "manifest.jsonl" if where == "manifest" else "run.json"
        (out/"dataset"/name).write_text(json.dumps(marker) + "\n")
    if where == "fit":
        (out/"fit").mkdir()
        (out/"fit"/"room_model.json").write_text(json.dumps(marker))
    with pytest.raises(licensing.LicenseGateError, match="BRAS"):
        project.export(root, tmp_path/"device")
    assert not (tmp_path/"device").exists()


def test_fitted_model_retains_measured_rir_lineage():
    from sensorsim.acoustics.room.fit import FitConfig, fit_room, observe
    from tests.unit.acoustics_room.test_fit import _single_reflection
    config = FitConfig(bands=(1000,), late=False, bootstrap=0, max_nfev=5)
    observations = []
    for i in range(3):
        rir = _single_reflection(i)
        rir.metadata['dataset'] = 'BRAS proxy control'
        observations.append(observe(rir, id=str(i), config=config))
    result = fit_room(observations, RoomModel(shoebox_dims_m=(8, 8, 4)), config=config)
    assert result.model.provenance['lineage']['bras_derived']
    assert RoomModel.from_json(result.model.to_json()).provenance['lineage']['bras_derived']


def test_dry_extraction_and_result_adapter_retain_lineage():
    from sensorsim.acoustics.room.dry import extract_dry, DryConfig
    result = extract_dry({'REF': np.ones(2000)*.01}, {'REF': np.ones(1)}, np.zeros(500),
        fs=8000, case='A', config=DryConfig(noise_suppression=False),
        metadata={'REF': {'dataset': 'BRAS proxy control'}})
    assert result.report['lineage']['bras_derived']
    clip = dataset.DryClip.from_result('knock', result)
    with pytest.raises(licensing.LicenseGateError, match='BRAS'):
        dataset.read_dry(clip, 8000)


def test_bras_flag_cannot_be_cleared_by_false_child_or_clean_override():
    m = RoomModel(shoebox_dims_m=(6, 5, 4), source_directivity=marked_directivity())
    cfg = SynthConfig(max_order=1, late=False, duration_s=.1, source_directivity=Directivity())
    rir = synthesize(m, (1, 2, 2), (2, 2, 2), 8000, config=cfg)
    assert rir.metadata['lineage']['bras_derived']
    with pytest.raises(licensing.LicenseGateError, match='BRAS'):
        licensing.require_commercial_lineage({'lineage': {'bras_derived': False}, 'parent': rir.metadata}, context='proxy')


def test_generic_csv_provenance_is_retained(tmp_path):
    from sensorsim.acoustics.room.directivity import load_balloon_csv
    import csv
    path = tmp_path/'neutral.csv'
    with path.open('w') as f:
        w = csv.writer(f)
        w.writerow(['azimuth_deg', 'elevation_deg', 'frequency_hz', 'pressure_gain'])
        for az, el in ((0, 0), (180, 0), (90, 0), (270, 0), (0, 90), (0, -90)):
            for frequency in (80, *bands.OCTAVE_NOMINAL_125_8000_HZ, 12000):
                w.writerow([az, el, frequency, 1])
    p = load_balloon_csv(path, name='proxy', provenance={'dataset': 'BRAS'})
    assert p.provenance['lineage']['bras_derived']


def test_clean_synthetic_rir_has_no_restricted_ancestry():
    m = RoomModel(shoebox_dims_m=(6, 5, 4))
    rir = synthesize(m, (1, 2, 2), (2, 2, 2), 8000, config=SynthConfig(max_order=1, late=False, duration_s=.1))
    assert not licensing.lineage_metadata(rir.metadata)
    assert 'lineage' not in rir.metadata


def test_session_render_rejects_bras_carrier_before_writing(tmp_path):
    from sensorsim.acoustics.room.session_sim import simulate_session
    model = RoomModel(room_id='bras_cr2', shoebox_dims_m=(6, 5, 4))
    with pytest.raises(licensing.LicenseGateError, match='BRAS'):
        simulate_session(tmp_path/'session', model, {'S1': (1, 2, 2)}, {'M1': (2, 2, 2)}, truth_dir=tmp_path/'truth')
    assert not list(tmp_path.rglob('*.wav'))


def test_reference_correction_retains_reference_lineage():
    from sensorsim.acoustics.room.fit import correct_reference
    from tests.unit.acoustics_room.test_fit import _single_reflection
    rir, ref = _single_reflection(0), _single_reflection(1)
    ref.metadata['lineage'] = {'bras_derived': True}
    corrected = correct_reference(rir, ref)
    assert corrected.metadata['lineage']['bras_derived']
