"""ML dataset contracts and known split-policy xfail.

Small direct-only RIRs isolate dataset contracts from room-physics accuracy.
No external audio or GPU is required.
"""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.dataset import (
    DryClip, PositionPair, SamplingSpec, generate_dataset, read_dry, split_positions,
)
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig


@pytest.fixture
def tiny_dataset(tmp_path):
    def generate(clips=None, **overrides):
        fs = 8000
        t = np.arange(160) / fs
        if clips is None:
            clips = [
                DryClip('drone', .01 * np.sin(2*np.pi*500*t), fs,
                        provenance={'source_session_id': 'session-a'}),
                DryClip('ambient', .01 * np.sin(2*np.pi*1000*t), fs,
                        provenance={'source_session_id': 'session-b'}),
            ]
        args = dict(model=RoomModel(shoebox_dims_m=(6, 5, 3)),
                    sampling=SamplingSpec(), clips=clips,
                    augmentation=AugmentationConfig(), count=10, seed=0,
                    output_dir=tmp_path/'dataset', fs=fs,
                    synth_config=SynthConfig(max_order=0, late=False, air_absorption=False))
        args.update(overrides)
        path = generate_dataset(**args)
        return path, [json.loads(line) for line in path.read_text().splitlines()]
    return generate


def test_transitive_cross_role_positions_stay_in_one_split():
    # A receiver becomes a source, a later pair connects the chain in reverse,
    # and sub-micrometre serialization noise rounds to the original coordinate.
    points = [(float(i), 1., 1.) for i in range(1, 9)]
    a, b, c, d, e, f, g, h = points
    pairs = [PositionPair(source_xyz_m=s, receiver_xyz_m=r) for s, r in
             [(a, b), (c, d), (b, c), ((a[0]+.1e-6, 1., 1.), d), (e, f), (g, h)]]
    labels = split_positions(pairs, .34, seed=19)
    assert len(set(labels[:4])) == 1
    groups = [{tuple(np.round(p, 6)) for pair, split in zip(pairs, labels)
               if split == name for p in (pair.source_xyz_m, pair.receiver_xyz_m)}
              for name in ('train', 'val')]
    assert all(groups) and groups[0].isdisjoint(groups[1])


def test_evaluation_does_not_reuse_training_recordings(tiny_dataset):
    t = np.arange(160) / 8000
    clips = [DryClip('drone', .01*np.sin(2*np.pi*f*t), 8000,
                     provenance={'source_session_id': f'session-{f}'})
             for f in (500, 1000)]
    _, rows = tiny_dataset(clips=clips)
    sets = [{r['dry_source']['provenance']['source_session_id'] for r in rows if r['split'] == name}
            for name in ('train', 'val')]
    assert all(sets) and sets[0].isdisjoint(sets[1])


@pytest.mark.xfail(strict=True, reason='ML02: uniform clip draws leave an available class out of validation')
def test_each_split_covers_available_classes_when_capacity_allows(tiny_dataset):
    _, rows = tiny_dataset()
    assert all({r['label'] for r in rows if r['split'] == name} == {'drone', 'ambient'}
               for name in ('train', 'val'))


def test_conflicting_labels_for_identical_audio_fail_preflight(tiny_dataset):
    x = .01 * np.sin(2*np.pi*500*np.arange(160)/8000)
    with pytest.raises(ValueError, match=r"drone.*ambient.*clips"):
        tiny_dataset(clips=[DryClip('drone', x, 8000), DryClip('ambient', x.copy(), 8000)])


@pytest.mark.parametrize('amplitude', [0., 1e-9], ids=['silent', 'quantized-to-silence'])
def test_zero_pcm_positive_examples_are_rejected_or_flagged(tiny_dataset, amplitude):
    x = amplitude * np.sin(2*np.pi*500*np.arange(160)/8000)
    try:
        path, rows = tiny_dataset(clips=[DryClip('drone', x, 8000)], count=1, val_fraction=0)
    except ValueError as error:
        assert 'clip' in str(error).lower() and 'pcm16' in str(error).lower()
        return
    _, pcm = wavfile.read(path.parent/rows[0]['file'])
    # A future explicit, machine-readable rejection flag can preserve diagnostics.
    assert np.any(pcm) or rows[0].get('quality', {}).get('usable_for_training') is False


def test_array_identity_distinguishes_different_decoded_audio():
    x = np.array([0, 1, 0, -1], dtype=np.int16)
    quiet, integer_info = read_dry(DryClip('tone', x, 8000), 8000)
    loud, float_info = read_dry(DryClip('tone', x.astype(np.float64), 8000), 8000)
    assert not np.array_equal(quiet, loud)
    assert integer_info['sha256'] != float_info['sha256']


def test_failed_explicit_run_preserves_the_full_requested_geometry(tiny_dataset, tmp_path):
    pairs = [PositionPair(source_xyz_m=(1., 1., 1.), receiver_xyz_m=(1.5, 1., 1.))]
    with pytest.raises(ValueError, match='would clip'):
        tiny_dataset(clips=[DryClip('tone', np.full(160, 2.), 8000)],
                     count=1, target_positions=pairs, val_fraction=0)
    run = json.loads((tmp_path/'dataset'/'run.json').read_text())
    assert run['state'] == 'incomplete'
    assert run.get('target_positions') == [p.model_dump(mode='json') for p in pairs]


def test_stereo_input_cannot_silently_change_channel_or_label():
    with pytest.raises(ValueError, match='mono'):
        read_dry(DryClip('drone', np.zeros((16, 2)), 8000), 8000)


def test_default_synthesis_accepts_legal_corner_positions(tiny_dataset):
    pair = PositionPair(source_xyz_m=(.25, .25, .5), receiver_xyz_m=(3.75, 2.75, 1.8))
    _, rows = tiny_dataset(model=RoomModel(shoebox_dims_m=(4, 3, 2.6)),
                          target_positions=[pair], count=1, val_fraction=0,
                          synth_config=SynthConfig())
    assert len(rows) == 1
