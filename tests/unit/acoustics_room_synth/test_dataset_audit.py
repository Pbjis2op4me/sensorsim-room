"""Source-content leakage and restartable manifest regressions."""
import json
import numpy as np
import pytest
from scipy.io import wavfile
from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.dataset import (DryClip, SamplingSpec, generate_dataset,
    sample_positions, read_dry, _source_pools)
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig


def test_training_validation_cannot_share_dry_content(tmp_path):
    fs = 4000
    x = (.01 * np.sin(2 * np.pi * 300 * np.arange(80) / fs)).astype(np.float32)
    wavfile.write(tmp_path / 'copy.wav', fs, x)
    # Identical decoded audio with different path, container and label.
    clips = [DryClip('engine', x, fs), DryClip('renamed', tmp_path / 'copy.wav')]
    room = RoomModel(shoebox_dims_m=(6, 5, 3))
    with pytest.raises(ValueError, match='source|clip'):
        generate_dataset(room, SamplingSpec(), clips, AugmentationConfig(), 3, 4,
            tmp_path / 'dataset', fs=fs, synth_config=SynthConfig(late=False, max_order=0))
    assert not (tmp_path / 'dataset' / 'manifest.jsonl').exists()


def test_run_config_retains_all_explicit_positions(tmp_path):
    room = RoomModel(shoebox_dims_m=(6, 5, 3))
    spec = SamplingSpec()
    pairs = sample_positions(room, spec, 2, 7)
    generate_dataset(room, spec, [DryClip('tone', np.ones(8) * .01, 4000)],
        AugmentationConfig(), 2, 7, tmp_path / 'dataset', fs=4000,
        target_positions=pairs, val_fraction=0, synth_config=SynthConfig(late=False, max_order=0))
    config = json.loads((tmp_path / 'dataset' / 'run.json').read_text())
    assert config['target_positions'] == [p.model_dump(mode='json') for p in pairs]


def test_source_group_connections_are_transitive_even_for_renamed_copies():
    # One shared label (sensorsim-room): splitting is stratified by label, and a label with a
    # single recording group always stays in training. This test is about transitive grouping.
    clips = [DryClip('x', np.ones(8)*.01, 4000, source_group='event-a'),
             DryClip('x', np.ones(8)*.02, 4000, source_group='event-a'),
             DryClip('x', np.ones(8)*.02, 4000, source_group='event-b'),
             DryClip('x', np.ones(8)*.03, 4000, source_group='event-b'),
             DryClip('x', np.ones(8)*.04, 4000)]
    catalog = [read_dry(c, 4000)[1] for c in clips]
    pools = _source_pools(catalog, .5, 7)
    assert {frozenset(p) for p in pools.values()} == {frozenset(range(4)), frozenset([4])}


@pytest.mark.parametrize("fraction,held_out", [(.01, 1), (.4, 2), (.99, 4)])
def test_all_sources_cover_only_their_split_positions(tmp_path, fraction, held_out):
    # Six channels from five events: all channels of event 0 must stay together.
    groups = [0, 0, 1, 2, 3, 4]
    clips = [DryClip('tone', np.full(8, .001*(i+1)), 4000, source_group=str(group))
             for i, group in enumerate(groups)]
    path = generate_dataset(
        RoomModel(shoebox_dims_m=(6, 5, 3)), SamplingSpec(), clips,
        AugmentationConfig(), 5, 7, tmp_path / 'dataset', fs=4000,
        val_fraction=fraction, all_sources=True,
        synth_config=SynthConfig(late=False, max_order=0, air_absorption=False))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    run = json.loads(path.with_name('run.json').read_text())
    pools = run['source_split_pools']
    assert len({groups[i] for i in pools['val']}) == held_out
    assert {groups[i] for i in pools['train']}.isdisjoint(groups[i] for i in pools['val'])
    positions_by_split = {}
    for subset in ('train', 'val'):
        selected = [r for r in rows if r['split'] == subset]
        positions = {(tuple(r['source_xyz_m']), tuple(r['receiver_xyz_m'])) for r in selected}
        positions_by_split[subset] = {p for pair in positions for p in pair}
        actual = [(r['source_catalog_index'], tuple(r['source_xyz_m']), tuple(r['receiver_xyz_m']))
                  for r in selected]
        expected = {(index, *pair) for index in pools[subset] for pair in positions}
        assert len(actual) == len(expected) and set(actual) == expected
        assert run['split_counts'][subset] == len(actual)
    assert all(positions_by_split.values())
    assert positions_by_split['train'].isdisjoint(positions_by_split['val'])


def test_dataset_help_names_required_clip_source_reference(capsys):
    from sensorsim.acoustics.room.dataset import main
    with pytest.raises(SystemExit):
        main(['--help'])
    help_text = capsys.readouterr().out
    assert 'source_reference' in help_text and 'dry_report' in help_text
