"""Small regression controls; full-session numerical comparisons live in run.py."""

import numpy as np
import pytest
from scipy.io import wavfile
import yaml

from sensorsim.acoustics.room.ingest import SweepKit, load_manifest, read_take
from sensorsim.acoustics.room.rir import generate_kit_playback, deconvolve_repetitions


def tiny_kit():
    return SweepKit(fs_hz=8000, f1_hz=80, f2_hz=3500, sweep_s=.2,
                    fade_in_s=.005, fade_out_s=.002, peak_dbfs=-12,
                    lead_silence_s=.05, tail_silence_s=.1, end_silence_s=.1,
                    repeats=4, period_s=.3, first_sweep_start_s=.05,
                    formula='x(t)=sin(2*pi*f1*L*(exp(t/L)-1)), L=T/ln(f2/f1); half-Hann fades',
                    reference='Farina; short regression', dither='none')


def write_manifest(tmp_path, *, recorder=None, clipped=False):
    k = tiny_kit()
    (tmp_path/'sweep_kit.json').write_text(k.model_dump_json())
    x = generate_kit_playback(k)
    if clipped:
        x = np.clip(x * 20, -1, 32767/32768)
    wavfile.write(tmp_path/'S1_M1_sweep.wav', k.fs_hz, (x*32768).astype(np.int16))
    raw = dict(room_id='tiny', fs_hz=k.fs_hz, recorder=recorder or {},
               room={'dims_m': dict(x=4, y=3, z=2.5)},
               positions={'speakers': {'S1': [1,1,1]}, 'mics': {'M1': [2,1,1]}},
               excitation={'type':'ess', 'kit':'sweep_kit.json'},
               takes=[dict(file='S1_M1_sweep.wav', type='sweep', speaker='S1', mic='M1',
                           calibration={'pa_per_fs': .001})])
    path = tmp_path/'manifest.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path


def test_clipping_rejected_before_pa_calibration(tmp_path):
    manifest = load_manifest(write_manifest(tmp_path, clipped=True))
    with pytest.raises(ValueError, match='clipp'):
        read_take(manifest, 0)


@pytest.mark.parametrize('recorder', [dict(agc=True), dict(noise_suppression=True),
                                    dict(processing='AGC on'), dict(processing='noise suppression ON')])
def test_declared_processing_rejected(tmp_path, recorder):
    with pytest.raises(ValueError, match='AGC|noise suppression'):
        load_manifest(write_manifest(tmp_path, recorder=recorder))


def test_large_start_latency_warned():
    k = tiny_kit()
    y = np.r_[np.zeros(12000), generate_kit_playback(k), np.zeros(800)]
    with pytest.warns(UserWarning, match='latency'):
        r = deconvolve_repetitions(y, k, rir_duration_s=.08)
    assert abs(r.metadata['latency_s'][0] - 1.5) < 1/k.fs_hz


def test_repeat_gain_change_warned():
    k = tiny_kit()
    y = generate_kit_playback(k)
    for i in range(k.repeats):
        a = round((k.first_sweep_start_s + i*k.period_s)*k.fs_hz)
        y[a:a+round(k.sweep_s*k.fs_hz)] *= 1 - .15*i
    with pytest.warns(UserWarning, match='gain|AGC'):
        r = deconvolve_repetitions(y, k, rir_duration_s=.08)
    assert r.metadata['quality']['repeat_gain_range_db'][0] > 1


def test_equal_level_sweeps_do_not_certify_agc_off():
    k = tiny_kit()
    with pytest.warns(UserWarning, match='AGC.*cannot|cannot.*AGC'):
        deconvolve_repetitions(generate_kit_playback(k), k, rir_duration_s=.08)


def test_simulator_order_independence_and_template_roundtrip(tmp_path):
    from sensorsim.acoustics.room.model import RoomModel, BoxObstacle
    from sensorsim.acoustics.room.synth import SynthConfig
    from sensorsim.acoustics.room.session_sim import Impairments, simulate_session
    room = RoomModel(shoebox_dims_m=(4,3,2.5), obstacles=(
        BoxObstacle(name='box', min_xyz=(2,2,0), size_xyz=(.5,.4,.6)),))
    kwargs = dict(model=room, speakers={'S1': (1,1,1)}, mics={'M1': (2,1,1)},
                  kit=tiny_kit(), impairments=Impairments.clean(8000),
                  synth_config=SynthConfig(max_order=1, late=False, duration_s=.08),
                  noise_seconds=.05)
    one = simulate_session(tmp_path/'one', truth_dir=tmp_path/'truth1', **kwargs)
    two = simulate_session(tmp_path/'two', truth_dir=tmp_path/'truth2', reverse_order=True, **kwargs)
    assert {k: v['sha256'] for k,v in one.takes.items()} == {k: v['sha256'] for k,v in two.takes.items()}
    assert len(load_manifest(one.manifest).takes) == 5
    assert not (one.manifest.parent/'truth.json').exists()
    assert one.rir_metadata['S1:M1']['obstacles']['boxes'] == 1
    assert one.takes['S1_REF_sweep.wav']['peak_dbfs'] < -3


def test_clock_and_nominal_device_conversion_are_separate():
    from sensorsim.acoustics.room.session_sim import Impairments, _clock
    fs = 48000
    x = np.sin(2*np.pi*1000*np.arange(fs)/fs)
    y, header = _clock(x, fs, Impairments.clean(device_fs=44100, clock_drift_ppm=200))
    assert header == fs and abs(len(y) - len(x)*1.0002) < 2
    z, header = _clock(x, fs, Impairments.clean(device_fs=44100, rate_mode='wrong_header'))
    assert header == 44100 and len(z) == 44100


def test_float_audio_above_one_is_not_necessarily_clipped(tmp_path):
    path = write_manifest(tmp_path)
    x = np.linspace(-2, 2, 10000).astype(np.float32)
    wavfile.write(tmp_path/'S1_M1_sweep.wav', 8000, x)
    assert np.max(read_take(load_manifest(path), 0)) > .001


def test_obstructed_direct_path_requires_timing_review(tmp_path):
    from sensorsim.acoustics.room.rir import extract_take
    path = write_manifest(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw['obstacles'] = [dict(name='box', min_xyz=[1.4,.8,0], size_xyz=[.2,.4,1.4], material='wood')]
    path.write_text(yaml.safe_dump(raw))
    with pytest.warns(UserWarning, match='obstacle.*direct|direct.*obstacle'):
        rir = extract_take(load_manifest(path), 0, rir_duration_s=.08)
    assert rir.metadata['direct_path_obstructed'] == [True]


def test_speaker_harmonics_do_not_alias_back_into_measurement_band():
    from sensorsim.acoustics.room.session_sim import Impairments, _speaker
    fs = 48000
    t = np.arange(9600)/fs
    x = .5*np.sin(2*np.pi*9000*t)
    y = _speaker(x, fs, Impairments.clean(harmonic2=.08, harmonic3=.04))
    y, t = y[960:-960], t[960:-960]
    amplitude = lambda hz: 2*abs(np.mean(y*np.exp(-2j*np.pi*hz*t)))
    # cos(18 kHz) amplitude = a2*A^2/2. The 27 kHz third harmonic
    # must be removed before ADC sampling, not folded back to 21 kHz.
    assert abs(amplitude(18000) - .08*.5**2/2) < .0001
    assert amplitude(21000) < .00002


def test_competing_arrivals_do_not_turn_supported_drift_into_missing_repeats():
    from scipy.signal import fftconvolve, resample
    k = tiny_kit()
    n = np.arange(600)
    h = (np.sinc(n-24.2)*np.exp(-((n-24.2)/8)**2) +
         .9*np.sinc(n-104.3)*np.exp(-((n-104.3)/8)**2))
    raw = fftconvolve(generate_kit_playback(k), h)
    recording = resample(raw, round(len(raw)*1.0002))
    actual_ppm = (len(recording)/len(raw)-1)*1e6
    with pytest.warns(UserWarning):
        rir = deconvolve_repetitions(recording, k, rir_duration_s=.06)
    # The 0.9 s baseline resolves one sample as 139 ppm. Require better than
    # one sample using parabolic interpolation; do not claim long-take precision.
    assert abs(rir.metadata['clock_drift_ppm']-actual_ppm) < 100
    assert rir.metadata['clock_resampled']
