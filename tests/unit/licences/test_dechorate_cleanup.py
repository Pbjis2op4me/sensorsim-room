"""Development controls for removing legacy dataset entrypoints; no archive reads."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile


def test_fit_bench_rejects_removed_dataset_before_creating_output(tmp_path, python_subprocess_env):
    out = tmp_path / 'output'
    proc = subprocess.run([sys.executable, '-m', 'tests.validation.bench_room_fit.run',
        '--case', 'dechorate', '--data', str(tmp_path/'absent'), '--out', str(out)],
        env=python_subprocess_env, capture_output=True, text=True)
    assert proc.returncode == 2 and 'invalid choice' in proc.stderr
    assert not out.exists()


def test_timing_bench_has_no_legacy_reader_or_cli_option(python_subprocess_env):
    bench = importlib.import_module('tests.validation.bench_room_join.fit_timing')
    assert not hasattr(bench, 'read_sofa')
    proc = subprocess.run([sys.executable, '-m', bench.__name__, '--help'],
                          env=python_subprocess_env, capture_output=True, text=True, check=True)
    assert '--rir-manifest' in proc.stdout
    assert '--sofa-dir' not in proc.stdout


@pytest.mark.parametrize('entrypoint', ['read_pair', 'main'])
def test_wp4_retired_entrypoints_refuse_before_io(monkeypatch, entrypoint):
    bench = importlib.import_module('tests.validation.wp4_ensemble.run')
    reached = []
    def forbidden(*args, **kwargs):
        reached.append(args)
        raise AssertionError('retired entrypoint attempted input or output')
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run', '--data', 'absent', '--out', 'absent',
        '--preregistration', 'absent', '--prereg-sha256', 'absent'])
    args = (Path('absent'), 'unused', 1) if entrypoint == 'read_pair' else ()
    with pytest.raises(RuntimeError, match='retired.*HDF5'):
        getattr(bench, entrypoint)(*args)
    assert reached == []


def test_recheck_removes_legacy_filename_pattern():
    bench = importlib.import_module('tests.validation.bench_room_recheck.run')
    assert not hasattr(bench, 'SOFA_PATTERN')


def test_timing_manifest_preserves_procedural_rir_and_provenance(tmp_path, monkeypatch):
    from tests.validation.bench_room_join import fit_timing
    samples = np.zeros(1024, dtype=np.float32)
    samples[37], samples[83] = .5, -.125
    wavfile.write(tmp_path/'impulse.wav', 16000, samples)
    manifest = tmp_path/'inputs.json'
    manifest.write_text(json.dumps({'model':{'shoebox_dims_m':[4., 5., 3.]},
        'observations':[{'file':'impulse.wav', 'source_xyz_m':[1., 1., 1.],
            'receiver_xyz_m':[2., 2., 1.], 'metadata':{'synthetic':True}}]}))
    reached = []
    def observe(rir, **kwargs):
        reached.append(rir)
        return rir
    monkeypatch.setattr(fit_timing.fit, 'observe', observe)
    model, observations, inputs = fit_timing.read_manifest(manifest, object())
    assert observations == reached and len(reached) == 1
    np.testing.assert_array_equal(reached[0].samples, samples)
    assert reached[0].metadata['synthetic'] is True
    assert model.shoebox_dims_m == (4., 5., 3.)
    assert inputs[0]['file'] == 'impulse.wav' and len(inputs[0]['sha256']) == 64
