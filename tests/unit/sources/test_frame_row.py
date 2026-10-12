"""Companion patch development controls; reviewer must verify merged candidate."""
import numpy as np
import pytest
from sensorsim.sources.intake import intake_chamber, load_project_sources
from tests.unit.sources import final_standins as F


def frame_take():
    x = F.door(gap=.24)
    t = np.arange(round(.07*F.FS))/F.FS
    a = round((.55+.48+.6)*F.FS)
    x[a:a+len(t)] += .07*np.sin(2*np.pi*1381*t)*np.exp(-t/.012)
    return x, a


def test_row_listening_approval_and_restored_final_hit(tmp_path):
    from scipy.io import wavfile
    x, hit = frame_take()
    targets = tmp_path/'targets'; targets.mkdir()
    # Use the decoded PCM for both key generation and the real row loader.
    pcm=np.round(x*32768).astype(np.int16)
    wavfile.write(targets/'frame.wav', F.FS, pcm)
    x=pcm.astype(float)/32768
    original = intake_chamber(x, F.FS, metadata=F.meta(), event_index=0)
    assert 'contaminated' in original.report['rejected']
    assert original.report['event_samples'][1] < hit
    kwargs=dict(metadata=F.meta(), event_index=0, heterogeneous_hits=True)
    proposed = intake_chamber(x, F.FS, **kwargs)
    assert proposed.report['status'] == 'review' and proposed.take is None
    assert proposed.report['event_samples'][1] > hit
    assert proposed.report['crop_samples'][1] > hit
    assert proposed.report['approval_key'] != original.report['approval_key']
    approval=dict(reviewer='development proxy listener', decision='approve', approval_key=proposed.report['approval_key'])
    assert intake_chamber(x, F.FS, approval=approval, **kwargs).take is not None
    with pytest.raises(ValueError, match='approval_key'):
        intake_chamber(x, F.FS, metadata=F.meta(), event_index=0, approval=approval)
    with pytest.raises(ValueError, match='approval_key'):
        intake_chamber(x*.95, F.FS, approval=approval, **kwargs)
    row=dict(file='frame.wav', **F.meta(), event_index=0, heterogeneous_hits=True, approval=approval)
    assert len(load_project_sources([row], targets, tmp_path/'qa.json')) == 1


def test_frame_flag_never_overrides_known_contamination_or_clipping():
    x, _ = frame_take()
    result = intake_chamber(x, F.FS, metadata=F.meta(), event_index=0,
                            heterogeneous_hits=True, contaminated=True)
    assert 'contaminated' in result.report['rejected']
    approval=dict(reviewer='development proxy listener', decision='approve', approval_key=result.report['approval_key'])
    assert intake_chamber(x, F.FS, metadata=F.meta(), event_index=0,
        heterogeneous_hits=True, contaminated=True, approval=approval).take is None
    x[8000]=1.
    assert intake_chamber(x, F.FS, metadata=F.meta(), event_index=0,
        heterogeneous_hits=True).take is None
