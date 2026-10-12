"""Small numerical contracts; quality campaign lives in scripts, not pytest."""
import json

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile
import yaml

from sensorsim.acoustics.room.dry import DryConfig, extract_dry, run_manifest


def test_a_and_b_have_different_identifiable_targets():
    rng = np.random.default_rng(2)
    x = rng.normal(size=1500)
    speaker = np.array([1., -.65])
    mic = np.array([.9, .2])
    room = np.zeros(128); room[12] = 1; room[65] = .2
    h = signal.convolve(signal.convolve(speaker, mic), room)
    ref_room = np.zeros(128); ref_room[3] = 1
    ref = signal.convolve(signal.convolve(speaker, mic), ref_room)
    metadata = {'REF': {'direct_start_sample': 3, 'direct_end_sample': 6},
                'M': {'direct_start_sample': 12, 'direct_end_sample': 15}}
    cfg = DryConfig(ridge=1e-10, max_gain_db=80, noise_suppression=False)
    def recover(recording, case):
        return extract_dry({'M': recording}, {'M': h, 'REF': ref}, np.zeros(500),
                           fs=8000, case=case, metadata=metadata, config=cfg)
    a = recover(signal.convolve(x, h), 'A')
    b = recover(signal.convolve(signal.convolve(x, mic), room), 'B')
    assert np.linalg.norm(a.samples[:len(x)]-x)/np.linalg.norm(x) < 1e-3
    truth_b = signal.convolve(x, mic)
    assert np.linalg.norm(b.samples[:len(truth_b)]-truth_b)/np.linalg.norm(truth_b) < 1e-3
    assert not np.allclose(b.samples[:len(x)], a.samples[:len(x)])
    assert 'mic response' in b.report['target']


def test_noise_notch_reliability_and_determinism():
    rng = np.random.default_rng(4)
    noise = rng.normal(size=1000)*.1
    h = np.zeros(100); h[0]=1; h[8]=-1
    kwargs = dict(fs=8000, case='A', config=DryConfig(max_gain_db=10))
    a = extract_dry({'M': noise}, {'M': h}, noise, **kwargs)
    b = extract_dry({'M': noise}, {'M': h}, noise, **kwargs)
    assert np.array_equal(a.samples, b.samples)
    assert np.isfinite(a.samples).all()
    assert all('low_snr' in r['reasons'] for r in a.report['bands'])
    assert any('spectral_notch' in r['reasons'] for r in a.report['bands'])
    assert all('no_resolved_signal_power' in r['reasons'] for r in a.report['bands'])
    assert all(r['recovery_status'] == 'unresolved' for r in a.report['bands'])
    assert a.report['unresolved_bands_hz'] == [r['center_hz'] for r in a.report['bands']]
    assert a.report['max_inverse_noise_gain_db'] <= 10 + 1e-10
    json.dumps(a.report, allow_nan=False)


def test_joint_filter_uses_independent_mics():
    rng = np.random.default_rng(20)
    x = rng.normal(size=2500)
    n1, n2 = rng.normal(size=(2, 2500))*.2
    hs = {'REF': np.array([1.]), 'M': np.array([1.])}
    cfg = DryConfig(noise_suppression=False, ref_weight=1)
    one = extract_dry({'REF': x+n1}, hs, n1, fs=8000, case='A', config=cfg)
    joint = extract_dry({'REF': x+n1, 'M': x+n2}, hs, {'REF': n1, 'M': n2},
                        fs=8000, case='A', synchronized=True, config=cfg)
    assert np.mean((joint.samples-x)**2) < .7*np.mean((one.samples-x)**2)
    with pytest.raises(ValueError, match='synchronized'):
        extract_dry({'REF': x, 'M': x}, hs, n1, fs=8000, case='A')


def test_linear_convolution_does_not_wrap_tail_to_start():
    x = np.zeros(4000); x[-1] = 1
    h = np.array([1.])
    result = extract_dry({'REF': x}, {'REF': h}, np.zeros(500), fs=8000, case='A',
                         config=DryConfig(noise_suppression=False, block_seconds=.13))
    assert np.max(np.abs(result.samples[:-1])) < 1e-12
    assert abs(result.samples[-1] - 1) < 1e-12


def test_recoverable_weak_band_keeps_its_power():
    # A coloured training source through a matched identity path. The weaker
    # octave has 10 dB SNR: its own power, not the strong low tone, sets the gate.
    rng = np.random.default_rng(42)
    fs = 8000
    t = np.arange(fs) / fs
    sos = signal.butter(4, [1500, 2500], btype='bandpass', fs=fs, output='sos')
    weak = signal.sosfilt(sos, rng.normal(size=fs))
    weak *= .03 / np.std(weak)
    noise = signal.sosfilt(sos, rng.normal(size=fs))
    noise *= .03 / np.sqrt(10) / np.std(noise)
    floor = signal.sosfilt(sos, rng.normal(size=fs))
    floor *= .03 / np.sqrt(10) / np.std(floor)
    x = np.sin(2*np.pi*125*t) + weak
    result = extract_dry({'REF': x+noise}, {'REF': np.array([1.])}, floor,
                         fs=fs, case='A')
    f, truth = signal.welch(x, fs, nperseg=512)
    _, recovered = signal.welch(result.samples, fs, nperseg=512)
    band = (f >= 1500) & (f <= 2500)
    error = 10*np.log10(recovered[band].sum()/truth[band].sum())
    # Power preservation within 0.5 dB; phase recovery is not asserted.
    assert abs(error) < .5, error


def test_zero_noise_power_subtraction_is_identity_for_coloured_joint_source():
    rng = np.random.default_rng(57)
    x = signal.lfilter([1], [1, -.98], rng.normal(size=2000))
    hs = {'REF': np.array([1., .1]), 'M': np.array([.3, -.28])}
    y = {k: signal.convolve(x, h) for k, h in hs.items()}
    kwargs = dict(fs=8000, case='A', synchronized=True)
    clean = extract_dry(y, hs, np.zeros(500), **kwargs,
                        config=DryConfig(noise_suppression=False))
    subtracted = extract_dry(y, hs, np.zeros(500), **kwargs)
    np.testing.assert_allclose(subtracted.samples, clean.samples, atol=1e-12, rtol=0)


def test_weak_sweep_band_with_information_is_not_a_broadband_notch():
    rng = np.random.default_rng(58)
    fs = 8000
    x = rng.normal(size=3000)
    h = .97*signal.firwin(129, 500, fs=fs)
    h[64] += .03  # A quiet, invertible upper shelf, 30.5 dB below the LF band.
    y = signal.convolve(x, h)
    result = extract_dry({'REF': y}, {'REF': h}, np.zeros(500), fs=fs, case='A',
                         config=DryConfig(max_gain_db=40))
    f, truth = signal.welch(x, fs, nperseg=512)
    _, recovered = signal.welch(result.samples[:len(x)], fs, nperseg=512)
    band = (f >= 1500) & (f <= 2500)
    assert abs(10*np.log10(recovered[band].sum()/truth[band].sum())) < .1


def test_retained_transfer_reports_gain_cap_loss():
    x = np.random.default_rng(59).normal(size=1000)
    result = extract_dry({'REF': .01*x}, {'REF': np.array([.01])}, np.zeros(500),
                         fs=8000, case='A', config=DryConfig(max_gain_db=20))
    # Inverse is capped at 10, so the actual clean transfer is .01*10 = .1.
    np.testing.assert_allclose(result.samples, .1*x, atol=1e-12)
    assert all(abs(b['retained_transfer']-.1) < 1e-12 for b in result.report['bands'])
    assert all('gain_cap' in b['reasons'] for b in result.report['bands'])


@pytest.mark.parametrize('fault', ['nan', 'missing_ref', 'zero', 'source'])
def test_invalid_inputs_rejected(fault):
    x = np.ones(100)
    hs = {'REF': np.ones(3)}
    meta = {}
    if fault == 'nan': x[0] = np.nan
    if fault == 'missing_ref': hs = {'M': np.ones(3)}
    if fault == 'zero': hs = {'REF': np.zeros(3)}
    if fault == 'source':
        hs['M'] = np.ones(3)
        meta = {'REF': {'source_id': 'S1'}, 'M': {'source_id': 'S2'}}
    with pytest.raises(ValueError):
        extract_dry({'M' if fault in ('source', 'missing_ref') else 'REF': x}, hs,
                    np.zeros(100), fs=8000, case='B', metadata=meta)


def test_kit_manifest_keeps_separate_physical_performances(tmp_path):
    fs = 8000
    for name, n in [('ref.wav', 1000), ('far.wav', 850), ('noise_floor.wav', 200)]:
        wavfile.write(tmp_path/name, fs, np.zeros(n, dtype=np.int16))
    np.save(tmp_path/'h.npy', np.array([1.]))
    manifest = {'fs_hz': fs, 'takes': [
        {'file': 'ref.wav', 'type': 'real', 'speaker': 'S1', 'mic': 'REF_S1', 'label': 'machine'},
        {'file': 'far.wav', 'type': 'real', 'speaker': 'S1', 'mic': 'M1', 'label': 'machine'},
        {'file': 'noise_floor.wav', 'type': 'noise', 'mic': 'M1'}]}
    # Use the same validated recording-kit schema as ingestion, not a partial
    # dict accepted only by the old calibration-bypassing CLI.
    from tests.unit.acoustics_room.test_room import kit
    (tmp_path/'kit.json').write_text(kit().model_dump_json())
    manifest.update(room_id='dry-fixture', room={'dims_m': {'x': 4, 'y': 3, 'z': 2.5}},
                    excitation={'type': 'ess', 'kit': 'kit.json'},
                    positions={'speakers': {'S1': [1, 1, 1]},
                               'mics': {'REF_S1': [1.3, 1, 1], 'M1': [2, 1, 1]}})
    index = {'fs_hz': fs, 'source': 'S1', 'reference': 'REF_S1',
             'rirs': {k: {'file': 'h.npy', 'direct_start_sample': 0, 'direct_end_sample': 1}
                      for k in ['REF_S1', 'M1']}}
    (tmp_path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
    (tmp_path/'rirs.yaml').write_text(yaml.safe_dump(index))
    summary = run_manifest(tmp_path, tmp_path/'rirs.yaml', label='machine', case='B', output=tmp_path/'out')
    assert summary['recommended'] == 'REF_S1'
    assert set(summary['outputs']) == {'REF_S1', 'M1'}
    assert len(wavfile.read(summary['outputs']['M1'])[1]) == 850
    with pytest.raises(FileExistsError):
        run_manifest(tmp_path, tmp_path/'rirs.yaml', label='machine', case='B', output=tmp_path/'out')


def test_memmap_blocks_match_full_processing(tmp_path):
    rng = np.random.default_rng(21)
    x = rng.normal(size=6000)
    noise = rng.normal(size=800)*.1
    h = np.zeros(300); h[10]=1; h[100]=.2
    kwargs = dict(fs=8000, case='A')
    whole = extract_dry({'REF': x}, {'REF': h}, noise, **kwargs)
    output = np.lib.format.open_memmap(tmp_path/'dry.npy', mode='w+', dtype='float32', shape=x.shape)
    blocks = extract_dry({'REF': x}, {'REF': h}, noise, output_buffer=output,
                         config=DryConfig(block_seconds=.113), **kwargs)
    assert blocks.samples is output
    np.testing.assert_allclose(whole.samples, blocks.samples, atol=1e-6)
