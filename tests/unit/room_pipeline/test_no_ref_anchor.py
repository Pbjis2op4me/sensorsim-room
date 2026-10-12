"""Fail-first no-REF same-recorder contract; no measured data dependencies."""
from types import SimpleNamespace

import numpy as np
from scipy import signal
from scipy.io import wavfile
import pytest

from sensorsim.acoustics.room import pipeline
from sensorsim.acoustics.room.dry import DryConfig
from tests.validation.bench_room_resim.chain import score


def test_no_ref_anchor_preserves_recorder_gain_and_cancels_sweep_speaker(tmp_path, monkeypatch):
    fs = 24000
    x = np.random.default_rng(32).normal(size=1400)*.005
    mic, speaker = np.array([.8, .2]), np.array([1., -.3])
    # Geometry: direct distance 2 m, common anchor distance 1 m. Reflections
    # arrive after the geometry-bounded window; neither position is a close REF.
    responses = []
    for distance, echo in ((2., .10), (1., .05)):
        h = np.zeros(1200)
        h[round(distance/343*fs)] = 1/distance
        h[850] = echo
        responses.append(h)
    takes = [SimpleNamespace(speaker_id='S1', source_xyz_m=(2., 2., 1.5),
             calibration=None, _channel_count=1, label='tone', file=tmp_path/f'take{i}.wav') for i in range(4)]
    for take in takes:
        take.file.write_bytes(b'identity fixture')
    manifest = SimpleNamespace(fs=fs, takes=takes, acquisition='synchronized',
                              shoebox_dims_m=(7., 6., 4.), obstacles=())
    cfg = pipeline.PipelineConfig(count=1, val_fraction=0., case='B',
        dry=DryConfig(ridge=1e-10, noise_suppression=False), no_ref_target='direct_anchor')
    for gain in (.02, .2):
        out = tmp_path/str(gain)
        out.mkdir()
        records = {}
        for i, (h, distance) in enumerate(zip(responses, (2., 1.))):
            values = gain*signal.convolve(signal.convolve(h, mic), speaker)
            filename = f'rir{i}.npy'
            np.save(out/filename, values)
            records[i, 0] = dict(take=i, channel=0, file=filename,
                source_xyz_m=(2., 2., 1.5), receiver_xyz_m=(2.+distance, 2., 1.5),
                metadata=dict(direct_samples=[round(distance/343*fs)], sound_speed_m_s=343.),
                is_reference=False)
        recorded = gain*signal.convolve(signal.convolve(x, mic), responses[0])
        monkeypatch.setattr(pipeline.ingest, 'read_take', lambda _, i:
            (recorded if i == 2 else np.zeros_like(recorded))[:, None])
        monkeypatch.setattr(pipeline, '_mic', lambda *_: 'M1')
        jobs = [dict(take=2, channel=0, sweep=(0, 0), reference=None,
                     noise=(3, 0), noise_proxy=False)]
        row, = pipeline._dry(manifest, records, jobs, out, cfg)
        _, recovered = wavfile.read(out/row['dataset_source'])
        measured = score(gain*signal.convolve(x, mic), recovered, fs)
        assert measured['lsd'] < 1e-4, measured
        assert row['source_reference']['target'] == 'source_1m'
        assert row['dataset_source_scale'] == 1.
        assert row['reference_kind'] == 'training_direct_anchor'
        assert row['reference_rir'] == 'rir1.npy'
        assert jobs[0]['reference'] is None  # no mutation of the public plan


def test_direct_anchor_rejects_an_obstructed_target_and_leaves_actual_ref_alone(tmp_path):
    take = SimpleNamespace(speaker_id='S1', source_xyz_m=(1., 1., 1.), calibration=None)
    manifest = SimpleNamespace(takes=[take], fs=24000)
    records = {(0, 0): dict(take=0, channel=0, is_reference=False,
                           metadata={'direct_path_obstructed': [True], 'direct_samples': [70]})}
    job = dict(take=0, sweep=(0, 0), reference=None)
    with pytest.raises(ValueError, match='no unobstructed training sweep'):
        pipeline._direct_anchor_jobs(manifest, records, [job], tmp_path)
    existing = {**job, 'reference': (0, 0)}
    updated, jobs = pipeline._direct_anchor_jobs(manifest, records, [existing], tmp_path)
    assert updated == records and jobs == [existing]
    assert 'no_ref_target' not in pipeline.PipelineConfig().model_dump(mode='json')
    # A nonzero late reflection does not establish a nonzero direct target.
    manifest.shoebox_dims_m, manifest.obstacles = (6., 5., 4.), ()
    h = np.zeros(1000)
    h[700] = 1.
    np.save(tmp_path/'late_only.npy', h)
    records[0, 0].update(file='late_only.npy', source_xyz_m=(1., 1., 1.),
        receiver_xyz_m=(2., 1., 1.), metadata={'direct_samples':[70], 'sound_speed_m_s':343.})
    with pytest.raises(ValueError, match='no unobstructed training sweep'):
        pipeline._direct_anchor_jobs(manifest, records, [job], tmp_path)
