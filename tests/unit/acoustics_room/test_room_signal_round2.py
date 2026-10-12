"""Export regressions for capture admission, RIR extraction and dry-signal integrity."""
import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to encode FLAC")
def test_cr_room_capture_01_flac_and_wav_get_the_same_clipping_decision(tmp_path):
    """Lossless FLAC must not bypass the integer-PCM rail rule that rejects the same WAV samples."""
    from sensorsim.acoustics.room.ingest import RecordingManifest, read_recording_audio, read_take

    pcm = np.zeros(1000, np.int16)
    pcm[[100, 200, 300]] = 32767  # three isolated positive ADC saturations
    wave, flac = tmp_path/"clipped.wav", tmp_path/"clipped.flac"
    wavfile.write(wave, 8000, pcm)
    subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(wave), "-threads", "2", str(flac)],
                   check=True)
    np.testing.assert_array_equal(read_recording_audio(wave)[0], read_recording_audio(flac)[0])
    for path in (wave, flac):
        manifest = RecordingManifest(room_id="clipping-probe", fs=8000, takes=[dict(
            file=path, type="noise", receiver_xyz_m=(1, 1, 1), excitation={"type": "unknown"})])
        with pytest.raises(ValueError, match="clipping at digital full scale"):
            read_take(manifest, 0)
    # Control: the same FLAC path still admits an unclipped recording.
    pcm[[100, 200, 300]] = 32000
    wavfile.write(wave, 8000, pcm)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", str(wave), "-threads", "2", str(flac)],
                   check=True)
    clean = RecordingManifest(room_id="clipping-probe", fs=8000, takes=[dict(
        file=flac, type="noise", receiver_xyz_m=(1, 1, 1), excitation={"type": "unknown"})])
    assert read_take(clean, 0).shape == (1000, 1)


def test_cr_room_dsp_02_output_buffer_aliasing_inputs_is_rejected():
    """An output buffer sharing the noise floor used to zero the default result silently."""
    from sensorsim.acoustics.room.dry import extract_dry

    x = np.random.default_rng(3).normal(size=800)
    noise = np.zeros_like(x)
    kwargs = dict(recordings={"M": x}, rirs={"M": np.ones(1)}, noise_floor=noise, fs=8000, case="A")
    independent = extract_dry(**kwargs).samples
    assert np.linalg.norm(independent) > 1  # a real signal, not the corrupted ~5e-7 norm
    with pytest.raises(ValueError, match="must not alias noise floors"):
        extract_dry(**kwargs, output_buffer=noise)
    backing = np.zeros(1200)  # a partial overlap: output starts halfway through the noise floor
    with pytest.raises(ValueError, match="must not alias noise floors"):
        extract_dry(**{**kwargs, "noise_floor": {"M": backing[:800]}}, output_buffer=backing[400:])
    # RIRs are reread after output writes (measured-residual/Tikhonov designs): reject that alias too.
    shared = np.zeros(800)
    rir = shared[:1]
    rir[0] = 1.
    with pytest.raises(ValueError, match="must not alias RIRs"):
        extract_dry(**{**kwargs, "rirs": {"M": rir}}, output_buffer=shared)
    # Control: a distinct buffer holding the same values is accepted and matches the default output.
    buffer = np.empty(800)
    np.testing.assert_array_equal(extract_dry(**kwargs, output_buffer=buffer).samples, independent)


def _kit(sweep_s=.25, tail_s=.15):
    """The reviewers' 8 kHz four-repeat kit (cr-room-dsp-01/03/04 reproductions)."""
    from sensorsim.acoustics.room.ingest import SweepKit
    return SweepKit(fs_hz=8000, f1_hz=40., f2_hz=3600., sweep_s=sweep_s, fade_in_s=.01, fade_out_s=.005,
                    peak_dbfs=-6., lead_silence_s=.05, tail_silence_s=tail_s, end_silence_s=.05, repeats=4,
                    period_s=sweep_s+tail_s, first_sweep_start_s=.05,
                    formula="x(t)=sin(2*pi*f1*L*(exp(t/L)-1)), L=T/ln(f2/f1); half-Hann fades",
                    reference="Farina", dither="none")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cr_room_dsp_04_every_channel_passes_the_clipping_admission_rule():
    """Saturation confined to a quieter non-reference channel must reject the stereo take."""
    from sensorsim.acoustics.room.rir import deconvolve_repetitions, generate_kit_playback

    k = _kit()
    playback = generate_kit_playback(k)
    clean = np.column_stack([playback, .1*playback])
    assert deconvolve_repetitions(clean, k, rir_duration_s=.06).samples.shape[1] == 2  # control
    for rail in (1., -1.):
        y = clean.copy()
        y[500:503, 1] = rail  # three exact digital full-scale samples in the quiet channel only
        with pytest.raises(ValueError, match=r"clipping detected in sweep take channel\(s\) \[1\]"):
            deconvolve_repetitions(y, k, rir_duration_s=.06)
        with pytest.raises(ValueError, match=r"clipping detected in sweep take channel\(s\) \[0\]"):
            deconvolve_repetitions(y[:, ::-1], k, rir_duration_s=.06)
        with pytest.raises(ValueError, match="clipping detected in sweep take"):  # mono control
            deconvolve_repetitions(y[:, 1], k, rir_duration_s=.06)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cr_room_dsp_03_kit_period_must_match_the_generated_playback_grid():
    """A declared period off the playback's integer sample grid used to invent clock drift."""
    from sensorsim.acoustics.room.rir import deconvolve_repetitions, generate_kit_playback

    # 0.25006 s rounds to the same 2000 sweep samples, so its playback repeats every 3200
    # samples, not the declared 0.40006 s (3200.48 samples): -150 ppm of invented drift.
    with pytest.raises(ValueError, match="period_s must be a whole number of samples"):
        _kit(sweep_s=.25006)
    # Rounding the sweep alone is the documented duration policy; an on-grid period is accepted
    # and its playback is bit-identical to the nominal kit's, with no clock correction.
    nominal, rounded = _kit(), _kit(sweep_s=.25006, tail_s=.14994)
    playback = generate_kit_playback(nominal)
    np.testing.assert_array_equal(generate_kit_playback(rounded), playback)
    for kit in (nominal, rounded):
        result = deconvolve_repetitions(playback, kit, rir_duration_s=.06)
        assert result.metadata["clock_drift_ppm"] == 0.
        assert result.metadata["clock_resampled"] is False


def test_cr_room_capture_03_harmonic_and_fundamental_packets_use_matching_gates():
    """Identical packets at 0.1 pressure must give -20 dB; the denominator gate dropped pre-peak energy."""
    from sensorsim.acoustics.room.capture_qa import assess_harmonics
    from sensorsim.acoustics.room.ingest import ESSExcitation

    fs, center = 24000, 20000
    spec = ESSExcitation(duration_s=1, f_start_hz=250, f_end_hz=4000, amplitude=.25)
    advance = 6000  # fs*T*ln(2)/ln(16): an exact integer H2 advance, no interpolation ambiguity
    offset = np.arange(-48, 49)
    packet = np.cos(2*np.pi*1000*offset/fs)*np.exp(-.5*(offset/8)**2)  # half its energy precedes the peak
    compressed = np.zeros(40000)
    compressed[center+offset] = packet
    compressed[center-advance+offset] = .1*packet
    report = assess_harmonics(compressed, fs, spec, linear_start_sample=center, decay_support_s={1000: .005})
    row = report['bands']['1000']['orders']['2']
    assert row['status'] == 'pass'
    assert row['packet_ratio_db'] == pytest.approx(-20., abs=1e-6)  # base reported -15.519 dB
    gate, fundamental_gate = row['window_samples'], row['fundamental_window_samples']
    np.testing.assert_allclose(np.subtract(fundamental_gate, gate), advance, atol=1e-6)
    # A fundamental gate running past the recording is unavailable, never a shorter denominator.
    short = assess_harmonics(compressed[:center+100], fs, spec, linear_start_sample=center,
                             decay_support_s={1000: .005})['bands']['1000']['orders']['2']
    assert (short['status'], short['packet_ratio_db']) == ('unavailable', None)
    assert short['reason'] == 'compressed packet outside recorded support'


def test_cr_room_capture_02_two_level_comparison_keeps_within_level_failures():
    """A passing 6 dB median must not erase a 12 dB within-take level spread or a missing repeat."""
    from sensorsim.acoustics.room.capture_qa import assess_repeat_rirs, compare_levels

    fs = 24000
    rng = np.random.default_rng(816)
    t = np.arange(round(.6*fs))/fs
    h = rng.normal(size=len(t))*np.exp(-3*np.log(10)*t/.16) + rng.normal(0, 1e-5, len(t))
    nominal = assess_repeat_rirs([h]*4, fs)
    consistent = assess_repeat_rirs([h*10**(-6/20)]*4, fs)
    control = compare_levels(nominal, consistent, after=nominal)['bands']['1000']
    assert (control['status'], control['linearity_status']) == ('pass', 'pass')
    quiet = assess_repeat_rirs([h*10**(-db/20) for db in (6, 6, 6, 18)], fs)
    assert quiet['bands']['1000']['consistency']['level_db']['status'] == 'review'  # 12 dB vs 1 dB limit
    row = compare_levels(nominal, quiet)['bands']['1000']
    assert row['nominal_minus_quiet_db'] == pytest.approx(6.)  # the median statistic is preserved
    assert (row['status'], row['linearity_status']) == ('review', 'review')  # base: pass, pass
    assert row['input_status'] == dict(nominal='pass', quiet='review')
    # The optional return capture is an input too: equal median, inconsistent repeats.
    returned = assess_repeat_rirs([h, h, h, h*10**(-12/20)], fs)
    row = compare_levels(nominal, consistent, after=returned)['bands']['1000']
    assert (row['nominal_return']['status'], row['status']) == ('review', 'review')
    # Planned-versus-retained support: three of four planned repeats cannot pass the comparison.
    partial = assess_repeat_rirs([h*10**(-6/20)]*3, fs, planned_repeats=4)
    row = compare_levels(nominal, partial)['bands']['1000']
    assert row['status'] != 'pass' and row['linearity_status'] != 'pass'


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cr_room_dsp_01_each_channel_is_searched_near_its_own_arrival():
    """A distant microphone (80 ms TDOA) was searched only +/-30 ms around the reference arrival."""
    from scipy.signal import fftconvolve
    from sensorsim.acoustics.room.rir import deconvolve_repetitions, generate_kit_playback

    k = _kit()
    playback = generate_kit_playback(k)

    def take(far):  # clean two-channel FIRs with direct impulses at samples 8 and `far`
        h = np.zeros((960, 2))
        h[8, 0], h[far, 1] = 1., .7
        return np.column_stack([fftconvolve(playback, h[:, channel]) for channel in range(2)])

    origin, near = (0., 0., 0.), (8*343/8000, 0., 0.)
    # Close control (5 ms TDOA): relative metadata matches the waveform, as before.
    close = deconvolve_repetitions(take(48), k, rir_duration_s=.12)
    assert close.metadata["direct_samples"] == np.argmax(abs(close.samples), axis=0).tolist() == [0, 40]
    y, far = take(648), (648*343/8000, 0., 0.)
    joint = deconvolve_repetitions(y, k, rir_duration_s=.12, source_xyz_m=origin, receiver_xyz_m=[near, far])
    alone = deconvolve_repetitions(y[:, 1], k, rir_duration_s=.12, source_xyz_m=origin, receiver_xyz_m=far)
    assert np.argmax(abs(joint.samples), axis=0).tolist() == [8, 648]  # base: [8, 949], far amplitude 9e-4
    np.testing.assert_allclose(joint.samples[:, 1], alone.samples, rtol=0, atol=1e-6)  # measured: identical
    # Never align from sidelobes: without geometry, or with geometry contradicting the arrival, reject.
    with pytest.raises(ValueError, match="arrival lies outside the 30 ms search"):
        deconvolve_repetitions(y, k, rir_duration_s=.12)  # base: direct_samples [0, 50], waveform peak 640
    with pytest.raises(ValueError, match="no supported linear packet within 30 ms of its geometric arrival"):
        deconvolve_repetitions(y, k, rir_duration_s=.12, source_xyz_m=origin, receiver_xyz_m=[near, near])


def _stereo_take(playback, far):
    """Clean two-channel FIRs with direct impulses at samples 8 and `far` (the reviewers' fixture)."""
    from scipy.signal import fftconvolve
    h = np.zeros((960, 2))
    h[8, 0], h[far, 1] = 1., .7
    return np.column_stack([fftconvolve(playback, h[:, channel]) for channel in range(2)])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cr_room_dsp_01_packet_medoid_rejects_arrivals_outside_its_early_window():
    """packet-medoid/1 returned the 80 ms far channel's peak at 928 against direct metadata 648."""
    from sensorsim.acoustics.room.rir import deconvolve_repetitions, generate_kit_playback

    k = _kit()
    playback = generate_kit_playback(k)
    origin, near = (0., 0., 0.), (8*343/8000, 0., 0.)
    medoid = dict(rir_duration_s=.12, repeat_alignment="packet-medoid/1")
    # Control at 47.5 ms, inside the +50 ms early window: honoured as before. Peaks keep one
    # common offset from the direct metadata (the method's threshold-crossing origin).
    close = deconvolve_repetitions(_stereo_take(playback, 380), k, source_xyz_m=origin,
                                   receiver_xyz_m=[near, (380*343/8000, 0., 0.)], **medoid)
    assert close.metadata["direct_samples"] == [8, 380]
    assert np.ptp(np.argmax(abs(close.samples), axis=0)-[8, 380]) == 0
    for far in (420, 648):  # 52.5 and 80 ms; base peaks [12, 550] and [12, 928]
        y = _stereo_take(playback, far)
        with pytest.raises(ValueError, match="packet-medoid/1 cannot align channel 1"):
            deconvolve_repetitions(y, k, source_xyz_m=origin, receiver_xyz_m=[near, (far*343/8000, 0., 0.)],
                                   **medoid)
        with pytest.raises(ValueError, match="packet-medoid/1 cannot align channel 1"):
            deconvolve_repetitions(y, k, **medoid)  # base: direct [0, 286] / [0, 74], peaks 416 / 644


def test_capture_qa_reports_are_labelled_v2_and_v1_reports_still_attach():
    """capture-02/03 changed two-level statuses and packet ratios under the same keys: a new label."""
    from sensorsim.acoustics.room.capture_qa import assess_repeat_rirs, attach_capture_qa, compare_levels

    fs = 24000
    t = np.arange(round(.3*fs))/fs
    h = np.random.default_rng(816).normal(size=len(t))*np.exp(-3*np.log(10)*t/.16)
    nominal = assess_repeat_rirs([h]*4, fs)
    assert nominal["schema_version"] == compare_levels(nominal, nominal)["schema_version"] == "capture-qa/2"
    for version in ("capture-qa/1", "capture-qa/2"):  # a /1 report still attaches, under its own label
        qa = dict(schema_version=version, status="unavailable", reason="no pilot")
        assert attach_capture_qa({"state": "complete"}, qa)["capture_qa"] == qa
    with pytest.raises(ValueError, match="unsupported capture QA report version"):
        attach_capture_qa({}, dict(schema_version="capture-qa/3"))
