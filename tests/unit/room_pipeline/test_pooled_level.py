from types import SimpleNamespace
import json
import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim.acoustics.room import pipeline, metrics
from sensorsim.acoustics.room.dry import DryConfig


def test_scalar_pool_changes_only_post_inverse_gain(tmp_path, monkeypatch):
    fs = 24000
    takes = [SimpleNamespace(speaker_id='S1', source_xyz_m=(2.,2.,1.5),
        calibration=None, _channel_count=1, label='tone', file=tmp_path/f'take{i}.wav',
        excitation=SimpleNamespace(type='ess', f_start_hz=40, f_end_hz=10000)) for i in range(4)]
    for t in takes:
        t.file.write_bytes(b'fixture')
    manifest = SimpleNamespace(fs=fs, takes=takes, acquisition='synchronized',
        shoebox_dims_m=(7.,6.,4.), obstacles=())
    rng = np.random.default_rng(55)
    sound = rng.normal(size=1100)*.001
    monkeypatch.setattr(pipeline.ingest, 'read_take', lambda _, i:
        (sound if i == 2 else np.zeros_like(sound))[:,None])
    monkeypatch.setattr(pipeline, '_mic', lambda *_:'M1')
    outputs = []
    for mode in ('effective_source', 'pooled_scalar'):
        out = tmp_path/mode
        out.mkdir()
        records = {}
        for i, distance in enumerate((1.,2.)):
            h = np.zeros(800)
            h[round(distance/343*fs)] = .2/distance
            h[650] = .03
            np.save(out/f'rir{i}.npy',h)
            records[i,0] = dict(take=i,channel=0,file=f'rir{i}.npy',is_reference=False,
                source_xyz_m=(2.,2.,1.5),receiver_xyz_m=(2.+distance,2.,1.5),
                metadata=dict(direct_samples=[round(distance/343*fs)],sound_speed_m_s=343.),
                bands={str(b):dict(noise_power=0.,excitation_supported=True) for b in (1000,2000,4000)})
        config = pipeline.PipelineConfig(count=1,val_fraction=0.,no_ref_target=mode,
            dry=DryConfig(noise_suppression=False, max_gain_db=1.))
        row, = pipeline._dry(manifest,records,[dict(take=2,channel=0,sweep=(0,0),
            reference=None,noise=(3,0),noise_proxy=False)],out,config)
        dry_report = json.loads((out/row['report']).read_text())
        assert any('gain_cap' in band['reasons'] for band in dry_report['bands'])
        _, y = wavfile.read(out/row['dataset_source'])
        outputs.append(y)
        if mode == 'pooled_scalar':
            assert row['level_reference']['effective_source']
            assert row['scalar_level_correction']['amplitude_scale'] == pytest.approx(.2,abs=1e-10)
            assert row['scalar_level_correction']['stage'] == 'post_inverse'
            assert json.loads((out/'level_references.json').read_text())
    np.testing.assert_allclose(outputs[1],outputs[0]*.2,rtol=2e-7,atol=1e-10)
    assert abs(metrics.si_sdr(sound,outputs[0])-metrics.si_sdr(sound,outputs[1])) < 1e-6


def test_normalized_dataset_retains_relative_product(tmp_path):
    from sensorsim.acoustics.room.augment import AugmentationConfig
    from sensorsim.acoustics.room.dataset import DryClip, SamplingSpec, generate_dataset
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import SynthConfig
    path = generate_dataset(RoomModel(shoebox_dims_m=(6,5,3)),SamplingSpec(),
        [DryClip('tone',np.sin(np.arange(80))*.01,4000)],
        AugmentationConfig(training_level=dict(mode='rms',target_rms_dbfs=-24)),
        1,3,tmp_path/'dataset',fs=4000,val_fraction=0,
        synth_config=SynthConfig(late=False,max_order=0))
    row = json.loads(path.read_text())
    meta = row['training_level']
    _, relative = wavfile.read(path.parent/meta['relative_file'])
    _, pcm = wavfile.read(path.parent/row['file'])
    recovered = pcm.astype(float)/32768/meta['total_export_scale']
    np.testing.assert_allclose(recovered,relative,rtol=0,atol=meta['quantization_error_bound_fs']+1e-15)
    assert relative.dtype == np.float64
