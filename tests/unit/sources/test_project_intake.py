"""WP6 intake contracts. All recordings here are project-generated proxy fixtures.

Acceptance preregistration SHA256:
bf17ed12d1e23db533d98ba12b57ec3f7b9e3a6adfaa22d7865ea4a3fc772b2c
"""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim import project


def layout(tmp_path):
    out = tmp_path / 'out'
    (out / 'fit').mkdir(parents=True)
    (out / 'fit' / 'report.json').write_text(json.dumps({'dry': []}))
    target = tmp_path / 'targets' / 'knock'
    target.mkdir(parents=True)
    wavfile.write(target / 'one.wav', 8000, np.sin(np.arange(8000)).astype('float32') * .1)
    return project.load_project(tmp_path), out


def test_unannotated_folder_audio_is_not_assumed_dry(tmp_path):
    cfg, out = layout(tmp_path)
    with pytest.raises(ValueError, match='provenance|dataset.sources'):
        project._clips(cfg, out)


def test_in_situ_cannot_enter_project_renderer(tmp_path):
    cfg, out = layout(tmp_path)
    cfg['dataset']['sources'] = [dict(file='knock/one.wav', label='knock',
        source_group='performance-1', object_id='door-1', session='capture-1',
        capture_route='in_situ', evidence_label='proxy')]
    with pytest.raises(ValueError, match='in_situ'):
        project._clips(cfg, out)
