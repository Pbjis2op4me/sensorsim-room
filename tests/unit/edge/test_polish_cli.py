"""WAV and calibration instructions: deterministic proxy controls."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.io import wavfile
from sensorsim import edge,project


@pytest.mark.parametrize('rate',[16000,48000])
@pytest.mark.parametrize('recording_id',[None,'original-session'])
def test_calibrate_warns_about_identity_and_rate(tmp_path,monkeypatch,capsys,rate,recording_id):
    path=tmp_path/'bed.wav'; wavfile.write(path,rate,np.arange(rate,dtype=np.int16))
    model=SimpleNamespace(meta={'features':{'fs':16000}});calls=[]
    monkeypatch.setattr(edge,'Model',lambda _:model)
    def observe(m,x,**kwargs):
        calls.append((m,len(x),kwargs));return {'status':'insufficient_evidence'}
    monkeypatch.setattr(edge,'calibrate',observe)
    args=['calibrate','model.npz',str(path),'--out',str(tmp_path/'tuned.npz')]
    if recording_id:args+=['--recording-id',recording_id]
    assert edge.main(args)==2
    output=capsys.readouterr(); assert json.loads(output.out)['saved'] is False
    assert ('--recording-id' in output.err)==(recording_id is None)
    assert ('fingerprints' in output.err)==(rate!=16000)
    if rate!=16000:assert '48000' in output.err and '16000' in output.err and 'ranges' in output.err
    assert calls[0][0] is model and calls[0][1]==16000
    assert calls[0][2]['recording_id']==recording_id
    assert not (tmp_path/'tuned.npz').exists()


def test_float_wav_project_error_explains_pcm_export(tmp_path,monkeypatch,capsys):
    path=tmp_path/'float.wav';wavfile.write(path,16000,np.sin(np.arange(16000)).astype(np.float32)*.1)
    def evaluate(*a,**k):return edge.read_wav(str(path),16000)
    monkeypatch.setattr(project,'_evaluate',evaluate)
    assert project.main(['evaluate',str(tmp_path)])==2
    err=capsys.readouterr().err
    assert 'integer PCM' in err and 'PCM16' in err and 'normaliz' in err and 'Traceback' not in err


@pytest.mark.parametrize('width',[1,2,3,4])
def test_pcm_widths_and_48khz_remain_supported(tmp_path,width):
    import wave
    path=tmp_path/'pcm.wav'
    with wave.open(str(path),'wb') as w:
        w.setparams((1,width,48000,48000,'NONE','not compressed'))
        w.writeframes((b'\x80' if width==1 else b'\0'*width)*48000)
    x=edge.read_wav(str(path),16000)
    assert x.shape==(16000,) and np.isfinite(x).all() and not x.any()


def test_deploy_recipe_has_sufficient_exposure_and_refusal():
    notes=project.DEPLOY_TEMPLATE
    assert '-d 10800 background.wav' in notes
    assert '--recording-id' in notes
    assert 'insufficient_evidence' in notes and 'exits 2' in notes
    assert '30 minutes' in notes and '6/h' in notes
