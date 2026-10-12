"""Small API/analytic checks; ensemble and full recording-kit comparisons are scripts."""
import json
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import fftconvolve, windows
import yaml

from sensorsim.acoustics.room import (
    ESSExcitation, RIR, SweepKit, deconvolve_blind, deconvolve_ess, deconvolve_known,
    detect_onset, early_reflections, estimate_reference_eq,
    extract_take, generate_ess, generate_kit_playback, load_manifest, read_take,
    room_metrics, schroeder_decay, trim_rir,
)
from sensorsim.acoustics.room.metrics import octave_sos
from sensorsim.realdata.audio_io import have_ffmpeg


def kit(fs=8000, duration=.25):
    return SweepKit(fs_hz=fs, f1_hz=40, f2_hz=fs*.45, sweep_s=duration,
                    fade_in_s=.01, fade_out_s=.005, peak_dbfs=-6,
                    lead_silence_s=.05, tail_silence_s=.15, end_silence_s=.05,
                    repeats=4, period_s=duration+.15, first_sweep_start_s=.05,
                    formula="x(t)=sin(2*pi*f1*L*(exp(t/L)-1)), L=T/ln(f2/f1); half-Hann fades",
                    reference="Farina", dither="none")


def generic_manifest(tmp_path):
    wavfile.write(tmp_path / 'take.wav', 8000, np.zeros((400, 2), dtype=np.int16))
    raw = dict(room_id='room', fs=8000, shoebox_dims_m=[4,3,2.5], takes=[dict(
        file='take.wav', source_xyz_m=[1,1,1], receiver_xyz_m=[[2,1,1],[2.1,1,1]],
        excitation='unknown', mic_ids=['L','R'], calibration={'pa_per_fs':[2,3]})])
    path = tmp_path / 'manifest.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path, raw


def test_manifest_channels_calibration_and_validation(tmp_path):
    path, raw = generic_manifest(tmp_path)
    wavfile.write(tmp_path / 'take.wav', 8000, np.full((400,2), 8192, dtype=np.int16))
    manifest = load_manifest(path)
    np.testing.assert_array_equal(read_take(manifest,0)[0], [.5,.75])
    for key, value in [('source_xyz_m',[5,1,1]), ('receiver_xyz_m',[[1,1,1]]),
                       ('source_xyz_m',[float('nan'),1,1]), ('file','missing.wav'),
                       ('calibration',{'pa_per_fs':-1}), ('excitation',{'type':'bogus'})]:
        invalid = json.loads(json.dumps(raw))
        invalid['takes'][0][key] = value
        path.write_text(yaml.safe_dump(invalid))
        with pytest.raises(ValueError):
            load_manifest(path)
    raw['fs'] = 16000
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='sample rate mismatch'):
        load_manifest(path)


def test_flac_preserves_multichannel(tmp_path):
    if not have_ffmpeg():
        pytest.skip('existing FFmpeg needed for FLAC')
    path, raw = generic_manifest(tmp_path)
    wave = (np.arange(800).reshape(400,2) - 400).astype(np.int16)
    wavfile.write(tmp_path/'take.wav',8000,wave)
    subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(tmp_path/'take.wav'),
                    str(tmp_path/'take.flac')],check=True)
    raw['takes'][0]['file']='take.flac'
    path.write_text(yaml.safe_dump(raw))
    np.testing.assert_allclose(read_take(load_manifest(path),0), wave/32768*[2,3], atol=1e-7)


def test_exact_kit_formula():
    k = kit(48000,10).model_copy(update={'f2_hz':20000,'fade_in_s':.1,'fade_out_s':.01})
    t=np.arange(480000)/48000
    length=10/np.log(20000/40)
    expected=np.sin(2*np.pi*40*length*(np.exp(t/length)-1))
    expected[:4800]*=windows.hann(9600)[:4800]
    expected[-480:]*=windows.hann(960)[480:]
    expected*=10**(-6/20)
    np.testing.assert_allclose(generate_ess(48000,k.excitation),expected,rtol=0,atol=1e-10)


def test_tiny_user_session(tmp_path):
    k=kit()
    (tmp_path/'sweep_kit.json').write_text(k.model_dump_json())
    h=np.zeros(600); h[8]=1; h[248]=.3
    y=np.r_[np.zeros(296),fftconvolve(generate_kit_playback(k),h)]
    wavfile.write(tmp_path/'sweep.wav',8000,y.astype(np.float32))
    wavfile.write(tmp_path/'noise.wav',8000,np.zeros(400,dtype=np.float32))
    raw=dict(room_id='tiny',fs_hz=8000,room={'dims_m':{'x':4,'y':3,'z':2.5}},
             obstacles=[dict(name='desk',min_xyz=[2,2,0],size_xyz=[1,.5,.7],material='wood')],
             positions={'speakers':{'S1':[1,1,1]},'mics':{'M1':[1.343,1,1],'REF_S1':[1.3,1,1]}},
             excitation={'type':'ess','kit':'sweep_kit.json'},
             takes=[{'file':'sweep.wav','type':'sweep','speaker':'S1','mic':'M1'},
                    {'file':'noise.wav','type':'noise','mic':'M1'},
                    {'file':'noise.wav','type':'real','speaker':'S1','mic':'REF_S1'}])
    path=tmp_path/'manifest.yaml'; path.write_text(yaml.safe_dump(raw))
    manifest=load_manifest(path)
    assert manifest.takes[2].is_reference and manifest.obstacles[0].name=='desk'
    rir=extract_take(manifest,0,rir_duration_s=.075)
    assert rir.metadata['repeats']==4 and np.argmax(np.abs(rir.samples[:,0]))==8
    assert abs(np.argmax(np.abs(rir.samples[220:270,0]))+220-248)<=1
    for index in (1,2):
        with pytest.raises(ValueError,match='sweep take'):
            extract_take(manifest,index,rir_duration_s=.05)
    raw['positions']['mics']['M1']=[5,1,1]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='outside room'):
        load_manifest(path)


def test_known_excitation_multichannel_and_unsupported():
    x=np.random.default_rng(52).normal(size=512)
    h=np.zeros((32,2)); h[3]=[1,.5]; h[12]=[.2,-.2]
    y=np.stack([fftconvolve(x,h[:,i]) for i in range(2)],axis=1)
    np.testing.assert_allclose(deconvolve_known(y,x),h,atol=1e-5)
    with pytest.raises(ValueError,match='zero energy'):
        deconvolve_known(y,np.zeros(512))
    with pytest.raises(NotImplementedError,match='not supported yet'):
        deconvolve_blind(y)


def test_ess_harmonic_packets_and_overlap():
    fs=8000; spec=ESSExcitation(duration_s=1,f_start_hz=40,f_end_hz=3600)
    x=generate_ess(fs,spec)
    result=deconvolve_ess(np.r_[x+.1*x*x,np.zeros(400)],fs,spec,rir_duration_s=.025,orders=(2,))
    assert abs(np.argmax(np.abs(result.harmonics[2]))-round(.0025*fs))<=16
    with pytest.raises(ValueError,match='overlap'):
        deconvolve_ess(np.r_[x,np.zeros(4000)],fs,spec,rir_duration_s=.5,orders=(2,3))


def test_analytic_energy_metrics():
    fs=16000; t=np.arange(32000)/fs; h=np.exp(-3*np.log(10)*t/.8)
    m=room_metrics(h,fs,onset_sample=0,noise_power=0)
    a=6*np.log(10)/.8
    assert abs(m.t30.rt60_s/.8-1)<.0001
    assert abs(m.c50_db-10*np.log10(np.expm1(a*.05)))<.02
    assert abs(m.c80_db-10*np.log10(np.expm1(a*.08)))<.02
    assert abs(m.d50-(1-np.exp(-a*.05)))<.0002
    assert abs(m.ts_s-1/a)<1/fs


def test_noisy_decay_and_missing_dynamic_range():
    fs=16000; rng=np.random.default_rng(1000); t=np.arange(38400)/fs
    h=rng.normal(size=len(t))*np.exp(-3*np.log(10)*t/.8)+rng.normal(size=len(t))*10**(-55/20)
    m=room_metrics(h,fs,onset_sample=0)
    assert m.t20.valid and m.t30.valid
    assert abs(m.t30.rt60_s/.8-1)<.02
    noise=room_metrics(rng.normal(size=4000),fs,onset_sample=0)
    assert not noise.t20.valid and noise.c50_db is None
    short=schroeder_decay(np.ones(3),fs)
    assert not short.valid
    with pytest.raises(ValueError,match='Nyquist'):
        octave_sos(8000,8000)


def test_sparse_onset_drr_reflection_and_trim():
    h=np.zeros(2000); h[200]=1; h[350]=.5; h[550]=.2
    assert detect_onset(h)==200
    m=room_metrics(h,10000,onset_sample=200,noise_power=0)
    assert abs(m.drr_db-10*np.log10(1/.29))<.01
    assert [p.sample for p in early_reflections(h,10000)]==[200,350,550]
    trimmed=trim_rir(RIR(10000,h),pre_s=.0025)
    assert trimmed.metadata['onset_sample']==25
    assert trimmed.metadata['time_origin_s']==.0175


def test_ref_eq_excludes_late_reflection_and_bounds_gain():
    h=np.zeros(4096); h[480]=1/.3; h[481]=-.4/.3; h[2000]=2
    eq=estimate_reference_eq(RIR(48000,h),direct_sample=480)
    known=1-.4*np.exp(-2j*np.pi*eq.frequency_hz/48000)
    error=20*np.log10(np.abs((known*eq.correction)[eq.supported]))
    assert np.max(np.abs(error))<.5
    assert np.all(eq.correction[~eq.supported]==0)
    assert np.max(np.abs(eq.correction))<=10**(12/20)
