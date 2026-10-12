"""Annotated WP6 sources through the existing fitted-room/CNN/device export path."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile
import yaml

from sensorsim import project
from tests.unit.project.test_project import _layout, _tone, _chamber_tone, quiet


@pytest.mark.slow
def test_annotated_chamber_sources_train_evaluate_and_export(tmp_path):
    pytest.importorskip('torch')
    root = _layout(tmp_path/'project')
    sources = []
    for label, frequency in [('test tone 0', 700.), ('test tone 1', 1900.)]:
        (root/'targets'/label).mkdir(parents=True)
        for k in range(2):
            name = f'{label}/take_{k}.wav'
            wavfile.write(root/'targets'/name, 24000, _chamber_tone(frequency, k))
            sources.append(dict(file=name, label=label, source_group=f'{label}-take-{k}',
                object_id=label+'-oscillator', session='proxy-session',
                capture_route='near_dry', evidence_label='proxy'))
    (root/'evaluate'/'test tone 0').mkdir(parents=True)
    wavfile.write(root/'evaluate'/'test tone 0'/'real.wav', 16000,
                  np.r_[np.zeros(8000, np.int16), _tone(700., 3, 16000)])
    (root/'evaluate'/'background').mkdir()
    wavfile.write(root/'evaluate'/'background'/'quiet.wav', 16000,
                  np.random.default_rng(610).normal(scale=600, size=48000).astype(np.int16))
    (root/'project.yaml').write_text(yaml.safe_dump(dict(
        dataset=dict(count=3, fs=16000, seed=1, sources=sources),
        recognizer=dict(type='cnn', steps=6, width=8, blocks=2, batch=16, workers=0,
                        threads=1, val_windows=64))))
    from tests.unit.sources.target_standins import approve_project_rows
    approved = approve_project_rows(project.load_project(root), root/'qa-preview')
    raw = yaml.safe_load((root/'project.yaml').read_text())
    raw['dataset']['sources'] = approved['dataset']['sources']
    (root/'project.yaml').write_text(yaml.safe_dump(raw))
    result = project.run(root, echo=quiet)
    assert result['train']['type'] == 'cnn'
    assert 'false_alarm_events_per_background_hour' in result['evaluate']  # proxy, not a measured deployment rate
    assert project.status(root) == {stage: 'up to date' for stage in project.STAGES}
    qa = json.loads((root/'out'/'source_qa.json').read_text())
    assert qa['status'] == 'pass' and len(qa['clips']) == 4
    manifest = [json.loads(line) for line in (root/'out'/'dataset'/'manifest.jsonl').read_text().splitlines()]
    seen = {}
    for row in manifest:
        if 'dry_source' not in row:
            continue
        p = row['dry_source']['provenance']
        assert p['object_id'] and p['session'] and p['capture_route'] == 'near_dry'
        for key in (p['source_group'], p['parent_content_sha256'], row['dry_source']['content_sha256']):
            assert seen.setdefault(key, row['split']) == row['split']
    folder = tmp_path/'device'
    exported = project.export(root, folder)
    assert sorted(p.name for p in folder.iterdir()) == ['DEPLOY.md', 'LICENSE', 'NOTICE', 'SHA256SUMS', 'edge.py', 'model.npz']
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ')
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    info = subprocess.run([sys.executable, '-I', 'edge.py', 'info', 'model.npz'], cwd=folder,
                          capture_output=True, text=True)
    assert info.returncode == 0, info.stderr
    assert json.loads(info.stdout)['classes'] == exported['classes']
