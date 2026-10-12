"""Parent ce62c28's end-to-end cr-tests-harness-05 test, adapted to the export (tiny_scenario inlined)."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from tests.validation.bench_room_closedloop import measured, runner
from tests.validation.bench_room_closedloop.engines import T0Engine, assemble_session
from tests.validation.bench_room_closedloop.scenario import Scenario
from tests.validation.bench_room_closedloop.scorecard import read_scorecard


def tiny_scenario():
    return Scenario.model_validate(dict(id="tiny", description="temporal-decimation control",
        variant="idealised", room=dict(dims_m=[2.6, 2.2, 2.2], surfaces={s: {"absorption": [.5]*7}
            for s in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}),
        positions={"S1": [.4, .6, 1.], "REF_S1": [.7, .6, 1.], "M1": [.9, .6, 1.],
                   "M2": [.5, 1.4, 1.], "M3": [1.8, 1.5, 1.2]},
        eval_positions=dict(heldout_mic="M3", novel_grid=dict(source=[.8, 1.7, 1.1],
            x_m=[1.6], y_m=[.8], z_m=[1.4])), real_sounds=["transient"],
        acquisition=dict(lead_s=[0, 0], trail_s=[0, 0], clock_ppm=[0, 0], jitter_s=[0, 0], extraneous=[]),
        render=dict(profile="smoke", rir_duration_s=.22), seed=81))


SCENARIO = tiny_scenario()


def _fails_at_dataset(assembled, incomplete):
    def pipeline(session, output, config):
        fs = assembled.kit.fs_hz
        (output/'dry').mkdir(parents=True)
        inputs = {str((session/t['file']).resolve()): runner.file_hash(session/t['file'])
                  for t in assembled.manifest['takes']}
        (output/'resolved_config.json').write_text(json.dumps(dict(input_sha256=inputs)))
        (output/'room_model.json').write_text(assembled_model.to_json())
        selection = []
        for i, take in enumerate(assembled.manifest['takes']):
            if take['type'] != 'real':
                continue
            selection.append(dict(take=i, channel=0, file=str(session/take['file']), event_index=0))
            stem = output/f'dry/take_{i:04d}_ch_00'
            source = assembled.truth.dry_sources[take['file']]
            estimate = (.5*source+1e-4*np.random.default_rng(i).normal(size=len(source))).astype(np.float32)
            wavfile.write(stem.with_suffix('.wav'), fs, estimate)
            stem.with_suffix('.json').write_text(json.dumps(dict(schema_version='room-dry/3', fs=fs)))
            if take['file'] != incomplete:
                wavfile.write(stem.parent/(stem.name+'_dataset_source.wav'), fs, estimate)
        (output/'dry/event_selection.json').write_text(json.dumps(selection))
        (output/'run.json').write_text(json.dumps(dict(stage='dataset', state='incomplete')))
        raise RuntimeError('injected dataset-stage failure')
    assembled_model = SCENARIO.room_model()
    return pipeline


class _Measured(T0Engine):
    tier, input_hashes, origins = 'T2', {}, {}


class _Coordinates:
    positions = SCENARIO.all_positions
    metadata = {}

    def __getattr__(self, name):
        return getattr(SCENARIO, name)


@pytest.mark.parametrize('driver', ['shared', 'measured'])
def test_completed_dry_takes_are_scored_after_a_later_stage_fails(tmp_path, monkeypatch, driver):
    # These benchmark drivers record the repository HEAD from their working
    # directory; declare it here instead of inheriting pytest's caller cwd.
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    assembled = assemble_session(SCENARIO, T0Engine(), tmp_path/'input')
    real = [t['file'] for t in assembled.manifest['takes'] if t['type'] == 'real']
    incomplete = real[-1]
    pipeline = _fails_at_dataset(assembled, incomplete)
    if driver == 'shared':
        monkeypatch.setattr(runner.pipeline, 'run_pipeline', pipeline)
        rows, provenance = runner.run_scenario(SCENARIO, T0Engine(), tmp_path/'work', tmp_path/'out',
                                               assembled=assembled)
    else:
        monkeypatch.setattr(measured.pipeline, 'run_pipeline', pipeline)
        _, provenance, cards = measured.run_measured_scenario(_Coordinates(), _Measured(), tmp_path/'work',
                                                              tmp_path/'out', assembled=assembled)
        rows = [r for card in cards for r in read_scorecard(card)]
    assert 'injected dataset-stage failure' in provenance['pipeline_error']
    assert provenance['pipeline_complete'] is False and provenance['pipeline_stage'] == 'dataset'
    assert provenance['completed_dry_takes'] == len(real)-1 >= 2
    scores = {r['notes'].split('take=')[1].split(';')[0]: r for r in rows if r['metric'] == 'dry_si_sdr'}
    assert sorted(scores) == sorted(real)
    for name, score in scores.items():
        if name == incomplete:
            assert score['reconstructed'] is None and score['pass'] is False
        else:
            assert score['reconstructed'] > 20 and score['pass'] is True
