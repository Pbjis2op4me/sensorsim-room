"""Frozen I6 controls: software consistency, never physical phone accuracy."""
import json

import numpy as np
import pytest
import yaml
from scipy.io import wavfile
from scipy.signal import fftconvolve

from sensorsim.acoustics.room import ingest
from sensorsim.acoustics.room.rir import generate_kit_playback, generate_ess, ess_inverse
from sensorsim.acoustics.room.sweep_detect import RepeatDetection, SweepDetection
from tests.validation.bench_room_kit_e2e.test_session import tiny_kit, write_manifest


def test_identity_unknown_is_not_a_verified_raw_path(tmp_path):
    path = write_manifest(tmp_path)
    assert ingest.load_manifest(path).capture_identity is None
    raw = yaml.safe_load(path.read_text())
    raw['capture_identity'] = dict(device='example phone', os_version='example OS',
        app='example recorder', app_version='1', unprocessed_source=dict(requested='UNPROCESSED'))
    path.write_text(yaml.safe_dump(raw))
    identity = ingest.load_manifest(path).capture_identity
    assert identity.unprocessed_source.support_query == 'unknown'
    raw['capture_identity']['unprocessed_source']['support_query'] = 'supported'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='evidence'):
        ingest.load_manifest(path)
    raw['capture_identity']['unprocessed_source']['evidence'] = 'saved API query log'
    raw['capture_identity']['agc'] = 'on'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='OFF'):
        ingest.load_manifest(path)


def test_companion_assets_are_hashed_and_scaled(tmp_path, monkeypatch):
    from scripts.room_sweep_kit import main
    from sensorsim.core.fingerprint import sha256_file
    monkeypatch.setattr('sys.argv', ['room_sweep_kit', '--out', str(tmp_path)])
    main()
    assets_path = tmp_path/'capture_qa_assets.json'
    assets = ingest.load_qa_assets(assets_path)
    assert assets.digital_difference_db == -6
    fs, a = wavfile.read(assets.nominal.file)
    fq, b = wavfile.read(assets.quiet.file)
    assert fs == fq and a.shape == b.shape
    np.testing.assert_allclose(b, a*10**(-6/20), atol=.5)
    assert sha256_file(assets.nominal.file) == '344442e14883b01462e3e13f375b402b706266b1bedbf3132f0adf915de3d370'
    value = json.loads(assets_path.read_text())
    value['digital_difference_db'] = -3
    assets_path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        ingest.load_qa_assets(assets_path)
    value['digital_difference_db'] = -6
    assets_path.write_text(json.dumps(value))
    wavfile.write(assets.quiet.file, fs, a)  # stale hash rejects
    with pytest.raises(ValueError, match='hash'):
        ingest.load_qa_assets(assets_path)
    value['quiet']['sha256'] = sha256_file(assets.quiet.file)
    assets_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='scale'):
        ingest.load_qa_assets(assets_path)


def _rirs():
    fs = 24000
    rng = np.random.default_rng(816)
    t = np.arange(round(.6*fs))/fs
    h = rng.normal(size=len(t))*np.exp(-3*np.log(10)*t/.16)
    h += rng.normal(0, 1e-5, len(t))
    return fs, h


def test_band_level_linear_eq_compression_and_low_support():
    from sensorsim.acoustics.room.capture_qa import assess_repeat_rirs, compare_levels
    fs, h = _rirs()
    h = np.convolve(h, [.6, .2, .1], mode='same')  # stationary coloration
    nominal = assess_repeat_rirs([h]*4, fs)
    quiet = assess_repeat_rirs([h*10**(-6/20)]*4, fs)
    result = compare_levels(nominal, quiet, after=nominal)
    assert all(v['status'] == 'pass' for v in result['bands'].values())
    assert result['primary_status'] == 'pass'
    compressed = assess_repeat_rirs([h*.8]*4, fs)
    assert all(v['status'] == 'review' for v in compare_levels(nominal, compressed)['bands'].values())
    noise = np.random.default_rng(7).normal(0, .2, len(h))
    weak = assess_repeat_rirs([h*1e-3+noise]*4, fs)
    assert all(v['status'] == 'unavailable' for v in compare_levels(nominal, weak)['bands'].values())


def test_repeat_metric_consistency_and_per_metric_decay_eligibility():
    from sensorsim.acoustics.room.capture_qa import assess_repeat_rirs, compare_levels
    fs, h = _rirs()
    report = assess_repeat_rirs([h]*4, fs)
    band = report['bands']['1000']
    assert band['consistency']['t20']['status'] == 'pass'
    assert band['repeats'][0]['metrics']['t30']['endpoint_margin_db'] >= 10
    changed = h.copy()
    changed[int(.02*fs):] = 0  # noise gate removes weak tail
    bad = assess_repeat_rirs([h, h, h, changed], fs)['bands']['1000']
    assert bad['consistency']['t20']['status'] in ('review', 'unavailable')
    assert bad['consistency']['c50']['status'] in ('review', 'unavailable')
    assert bad['planned_repeats'] == 4
    unsupported = assess_repeat_rirs([h[:300]]*2, fs)
    assert unsupported['bands']['1000']['consistency']['t30']['status'] == 'unavailable'
    assert unsupported['bands']['1000']['planned_repeats'] == 2
    gated = assess_repeat_rirs([changed]*4,fs)
    gated_low = assess_repeat_rirs([changed*10**(-6/20)]*4,fs)
    assert compare_levels(gated,gated_low)['bands']['1000']['status'] != 'pass'


def _detection(k, starts=None):
    fs = k.fs_hz
    starts = np.array(starts if starts is not None else
                      (k.first_sweep_start_s+np.arange(4)*k.period_s)*fs)
    reps = [RepeatDetection(i, float(s), float(s+k.sweep_s*fs), 1, 60, 1,
        max(0, int(s-.01*fs)), int(s+(k.sweep_s+k.tail_silence_s)*fs)) for i,s in enumerate(starts)]
    return SweepDetection(fs, reps, [], 0., 0., 0., [1.]*4, False)


@pytest.mark.parametrize('change', ['linear', 'insertion', 'deletion', 'ramp', 'latency'])
def test_raw_clock_stationarity_before_correction(change):
    from sensorsim.acoustics.room.capture_qa import assess_timing
    k = tiny_kit()
    x = fftconvolve(generate_kit_playback(k), [.5, .3, -.1])
    d = _detection(k)
    if change == 'latency':
        for rep in d.repeats[2:]:
            rep.start_sample += .005*k.fs_hz
        report = assess_timing(x, k, d)
        assert report['status'] == 'review'
        return
    if change != 'linear':
        rep = d.repeats[2]
        start, stop = round(rep.start_sample), round(rep.end_sample)
        n = np.arange(stop-start)
        offset = (5 if change == 'insertion' else -5)*(n >= len(n)//2)
        if change == 'ramp':
            offset = 14*(n/len(n))**2
        x[start:stop] = np.interp(n-offset, n, x[start:stop])
    before = x.copy()
    report = assess_timing(x, k, d)
    assert report['status'] == ('pass' if change == 'linear' else 'review')
    np.testing.assert_array_equal(x, before)
    assert report['correction_applied'] is False


def test_harmonic_packet_timing_and_overlap_are_not_thd():
    from sensorsim.acoustics.room.capture_qa import assess_harmonics
    k = tiny_kit().model_copy(update=dict(sweep_s=1., period_s=1.1))
    spec, fs = k.excitation, k.fs_hz
    sweep = generate_ess(fs, spec)
    wet = np.r_[sweep+.3*sweep**2+.4*sweep**3, np.zeros(fs)]
    full = fftconvolve(wet, ess_inverse(fs, spec))
    start = len(sweep)-1
    short = assess_harmonics(full, fs, spec, linear_start_sample=start,
                            decay_support_s={1000: .005})
    for order in (2,3):
        row = short['bands']['1000']['orders'][str(order)]
        expected = start-fs*spec.duration_s*np.log(order)/np.log(spec.f_end_hz/spec.f_start_hz)
        assert abs(row['peak_sample']-expected) <= 2
        assert row['packet_ratio_db'] is not None
    long = assess_harmonics(full, fs, spec, linear_start_sample=start,
                           decay_support_s={1000: 1.})
    assert long['bands']['1000']['status'] == 'review'
    assert long['primary_status'] == 'review'
    assert long['bands']['1000']['orders']['2']['packet_ratio_db'] is None
    assert 'THD' in long['warning']


def test_report_hook_preserves_pipeline_and_missing_evidence():
    from sensorsim.acoustics.room.capture_qa import attach_capture_qa
    report = dict(state='complete', measurements=[])
    qa = dict(schema_version='capture-qa/1', status='unavailable', reason='no pilot')
    updated = attach_capture_qa(report, qa)
    assert updated['capture_qa'] == qa and 'capture_qa' not in report
    assert updated['state'] == 'complete'
    json.dumps(updated, allow_nan=False)


def test_optional_report_has_unknown_identity_without_pilot(tmp_path):
    from sensorsim.acoustics.room.capture_qa import build_capture_qa_report
    report = build_capture_qa_report(ingest.load_manifest(write_manifest(tmp_path)))
    assert report['status'] == 'unavailable'
    assert report['capture_identity']['unprocessed_source']['support_query'] == 'unknown'
    assert report['processing_verified'] is False
    json.dumps(report, allow_nan=False)


def test_capture_uses_quiet_evidence_and_stops_before_correction(monkeypatch):
    from sensorsim.acoustics.room import capture_qa as qa
    k = tiny_kit()
    x = generate_kit_playback(k)
    # No actual RIR correction is allowed when detector evidence shows a clock step.
    d = _detection(k)
    d.repeats[-1].start_sample += 80
    monkeypatch.setattr(qa, 'detect_sweeps', lambda *a, **kw: d)
    monkeypatch.setattr(qa, 'compress_repeat', lambda *a, **kw: pytest.fail('corrected before timing review'))
    result = qa.assess_capture(x,k,rir_duration_s=.08)
    assert result['status'] == 'review' and 'repeat_metrics' not in result
    monkeypatch.undo()
    result = qa.assess_capture(x,k,rir_duration_s=.08,
        noise_samples=np.random.default_rng(6).normal(0,.2,800))
    assert result['band_capture_snr']['1000']['status'] == 'unavailable'
    assert result['repeat_metrics']['bands']['1000']['repeats'][0]['snr_db'] < 20
    json.dumps(result,allow_nan=False)


def test_physical_pilot_keeps_return_and_missing_metric_denominators():
    from sensorsim.acoustics.room.capture_qa import summarize_pilot, assess_repeat_rirs
    fs,h = _rirs()
    good = assess_repeat_rirs([h]*4,fs)
    rows = [dict(id=name,level='nominal',position='P1',qa={'repeat_metrics':good}) for name in ('A','B','C','D')]
    report = summarize_pilot(rows)
    assert report['nominal_repeatability']['bands']['1000']['t20']['status'] == 'pass'
    assert report['return_to_P1']['bands']['1000']['t20']['status'] == 'pass'
    no_return = summarize_pilot(rows[:3])
    assert no_return['nominal_repeatability']['bands']['1000']['t20']['status'] == 'pass'
    assert no_return['return_to_P1']['bands']['1000']['t20']['status'] == 'unavailable'
    bad = assess_repeat_rirs([h,h,h,h*.6],fs)
    rows[2]['qa']['repeat_metrics'] = bad
    report = summarize_pilot(rows)
    assert report['nominal_repeatability']['bands']['1000']['level_db']['status'] == 'review'
    rows[2]['qa'] = dict(status='reject',reason='bad take')
    report = summarize_pilot(rows)
    assert report['nominal_repeatability']['bands']['1000']['t20']['planned_count'] == 3
    assert report['nominal_repeatability']['bands']['1000']['t20']['status'] == 'unavailable'


def test_sidecar_report_end_to_end_and_geometry_contract(tmp_path):
    from sensorsim.acoustics.room.capture_qa import build_capture_qa_report
    from sensorsim.core.fingerprint import sha256_file
    path = write_manifest(tmp_path)
    k = tiny_kit()
    nominal = np.rint(generate_kit_playback(k)*32767).astype(np.int16)
    quiet = np.rint(nominal.astype(float)*10**(-6/20)).astype(np.int16)
    wavfile.write(tmp_path/'nominal.wav', k.fs_hz, nominal)
    wavfile.write(tmp_path/'quiet.wav', k.fs_hz, quiet)
    def asset(name):
        return dict(file=name, sha256=sha256_file(tmp_path/name))
    kit = k.model_dump()
    kit['files'] = {'nominal.wav': {'sha256':sha256_file(tmp_path/'nominal.wav')}}
    (tmp_path/'sweep_kit.json').write_text(json.dumps(kit))
    (tmp_path/'assets.json').write_text(json.dumps(dict(schema_version='1',kit=asset('sweep_kit.json'),
        nominal=asset('nominal.wav'),quiet=asset('quiet.wav'),digital_difference_db=-6)))
    rows = []
    for label in ('A','B','L','C','P2','P3','D'):
        filename = label+'.wav'
        wavfile.write(tmp_path/filename,k.fs_hz,quiet if label == 'L' else nominal)
        rows.append(dict(id=label,file=filename,level='quiet' if label=='L' else 'nominal',
                         position=label if label in ('P2','P3') else 'P1',
                         source_xyz_m=[1,1,1],receiver_xyz_m=[2,1,1]))
    sidecar = dict(schema_version='1',assets='assets.json',takes=rows)
    (tmp_path/'qa.yaml').write_text(yaml.safe_dump(sidecar))
    raw = yaml.safe_load(path.read_text())
    raw['capture_qa'] = 'qa.yaml'
    path.write_text(yaml.safe_dump(raw))
    manifest = ingest.load_manifest(path)
    assert len(manifest.takes) == 1  # pilot/quiet captures never enter the fitting rows
    report = build_capture_qa_report(manifest,rir_duration_s=.08)
    assert len(report['takes']) == 7 and report['missing_pilot_takes'] == []
    assert set(report['comparisons']) == {'two_level','nominal_repeatability','return_to_P1'}
    assert all('repeat_metrics' in row['qa'] for row in report['takes'])
    assert report['physical_accuracy'] == 'unverified'
    json.dumps(report,allow_nan=False)
    rows[2]['receiver_xyz_m'] = [3,1,1]
    (tmp_path/'qa.yaml').write_text(yaml.safe_dump(sidecar))
    with pytest.raises(ValueError,match='identical'):
        ingest.load_capture_qa(tmp_path/'qa.yaml')
