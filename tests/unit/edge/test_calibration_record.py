"""Synthetic calibration provenance controls; no measured deployment claim."""
import hashlib
import json
import subprocess
import sys

import numpy as np

from sensorsim import edge
from test_calibration_budget import fixture


def test_calibrate_records_digest_identity_and_range_and_preserves_history():
    model, audio, _ = fixture(hours=3, bursts=0)
    result = edge.calibrate(model, audio, recording_id='session', sample_start=137, calibration_path='cal.wav')
    saved = result['calibration_audio_records'][0]
    assert saved['sample_range'] == [137, 137+len(audio)]
    assert saved['sample_rate_hz'] == 1 and saved['recording_id'] == 'session'
    assert saved['decoded_sha256'] == hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()
    assert saved['decoded_sha256'] == result['calibration_audio_sha256']
    model.meta['detector']['calibration'] = result
    updated = edge.calibrate(model, audio, recording_id='session-2')
    assert len(updated['calibration_audio_records']) == 2
    assert updated['calibration_audio_records'][0] == saved


def test_cli_persists_calibration_record_in_npz(tmp_path, monkeypatch, capsys):
    model, audio, _ = fixture(hours=3, bursts=0)
    monkeypatch.setattr(edge, 'Model', lambda _: model)
    monkeypatch.setattr(edge, '_wav_rate', lambda _: model.meta['features']['fs'])
    monkeypatch.setattr(edge, 'read_wav', lambda *a: audio)
    source, target = tmp_path / 'model.npz', tmp_path / 'calibrated.npz'
    np.savez(source, meta=np.frombuffer(json.dumps(model.meta).encode(), np.uint8))
    assert edge.main(['calibrate', str(source), 'cal.wav', '--recording-id', 'session',
                      '--sample-start', '9000', '--out', str(target)]) == 0
    with np.load(target, allow_pickle=False) as data:
        calibration = json.loads(bytes(data['meta']))['detector']['calibration']
    assert calibration['calibration_audio_records'][0]['sample_range'] == [9000, 19800]
    assert calibration['calibration_audio_records'][0]['path'] == 'cal.wav'
    assert json.loads(capsys.readouterr().out)['saved']


def test_standalone_device_copy_records_without_package_imports(tmp_path):
    import ast
    from pathlib import Path
    source = Path(edge.__file__).read_text()
    imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(getattr(n, 'module', '').startswith('sensorsim') for n in imports if isinstance(n, ast.ImportFrom))
    path = tmp_path / 'device_edge.py'
    path.write_text(source)
    x = np.arange(40, dtype=np.int16)
    script = """import json, runpy, sys
namespace = runpy.run_path(sys.argv[1], run_name='device_copy')
np = namespace['np']
assert not any(name == 'sensorsim' or name.startswith('sensorsim.') for name in sys.modules)
assert 'scipy' not in sys.modules and 'torch' not in sys.modules
print(json.dumps(namespace['calibration_audio_record'](np.arange(40, dtype=np.int16), 32)))
"""
    result = subprocess.run([sys.executable, '-I', '-c', script, str(path)],
                            capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout) == edge.calibration_audio_record(x, 32)
