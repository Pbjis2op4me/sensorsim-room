"""Near-equal early wavefront regression; qualification lives in ROOM_RIR_ALIGN."""
import numpy as np
import pytest
from scipy import signal
from types import SimpleNamespace

from sensorsim.acoustics.room.rir import deconvolve_repetitions, generate_kit_playback
from tests.unit.acoustics_room.test_room import kit


def two_wavefront_capture(ppm):
    k = kit(fs=48000)
    h = np.zeros(4800)
    # 0.208 ms reflection; deliberately close to equal peak height.
    h[120], h[130] = 1., 1.0008
    wet = .1 * signal.fftconvolve(generate_kit_playback(k), h)
    n = np.arange(len(wet), dtype=float)
    a, b = .5 * ppm * 1e-6 / (len(wet) - 1), 1 - .5 * ppm * 1e-6
    positions = 2 * n / (b + np.sqrt(b*b + 4*a*n))
    up = signal.resample_poly(wet, 4, 1)
    y = wet if ppm == 0 else np.interp(4*positions, np.arange(len(up)), up)
    t = (k.first_sweep_start_s + np.arange(4)*k.period_s)*k.fs_hz
    emissions = t + .5*ppm*1e-6*(t*t/(len(wet)-1) - t)
    return k, y, emissions


def test_near_equal_early_reflection_does_not_switch_between_repeats():
    k, y, emissions = two_wavefront_capture(-1.)
    with pytest.warns(UserWarning):
        result = deconvolve_repetitions(y, k, rir_duration_s=.08,
            source_xyz_m=(0., 0., 0.), receiver_xyz_m=(3.43, 0., 0.),
            repeat_alignment="packet-medoid/1")
    arrivals = np.asarray(result.metadata['repeat_peak_samples'])[:, 0]
    assert result.metadata['repeats'] == 4
    assert np.ptp(arrivals - emissions)*1000/k.fs_hz <= .02


def test_fractional_packet_match_scores_the_registered_waveform():
    """Half-sample grid mismatch must not masquerade as a bad repeat."""
    from sensorsim.acoustics.room.rir import _packet_repeat_alignment
    n = np.arange(9000)
    h = np.sinc(n-2000) + np.sinc(n-2009.5)*.999
    records = [(h[:, None], 0, 1., SimpleNamespace(slot=i, segment_stop=len(h)),
                2000+(i % 2)*9.5) for i in range(4)]
    detection = SimpleNamespace(dropped=[])
    windows, arrivals, kept, _, _ = _packet_repeat_alignment(
        records, 48000, 2400, 0, np.array([480.]), True, 3, detection)
    assert len(kept) == 4 and not detection.dropped
    assert np.ptp(np.asarray(arrivals)[:, 0]) < .02*48
    assert np.linalg.norm(windows[0]-windows[1])/np.linalg.norm(windows[0]) < 1e-3


@pytest.mark.parametrize('bad_repeat', [0, 2])
def test_late_transient_is_removed_even_outside_alignment_packet(bad_repeat):
    from sensorsim.acoustics.room.rir import _packet_repeat_alignment
    h = np.zeros((9000, 2))
    h[2000, 0], h[2032, 1] = 1., .7
    records = []
    for i in range(4):
        full = h.copy()
        if i == bad_repeat:
            full[5000, 1] = 4.  # Only the non-reference channel is corrupted.
        records.append((full, 0, 1., SimpleNamespace(slot=i, segment_stop=len(h)), 2000.))
    detection = SimpleNamespace(dropped=[])
    windows, _, kept, _, _ = _packet_repeat_alignment(
        records, 48000, 4800, 0, np.zeros(2), False, 3, detection)
    assert [r.slot for r in kept] == [i for i in range(4) if i != bad_repeat]
    assert detection.dropped[0]['slot'] == bad_repeat
    assert len(windows) == 3


def test_packet_alignment_retains_existing_multichannel_tdoa_fixture():
    # Same 8/40-sample fixture as test_measurement_audit, with the new option.
    k = kit()
    h = np.zeros((480, 2))
    h[8, 0], h[40, 1] = 1., .7
    playback = generate_kit_playback(k)
    recording = np.stack([signal.fftconvolve(playback, h[:, i]) for i in range(2)], axis=1)
    with pytest.warns(UserWarning):
        result = deconvolve_repetitions(recording, k, rir_duration_s=.06,
                                       repeat_alignment='packet-medoid/1')
    peaks = np.argmax(abs(result.samples), axis=0)
    assert peaks[1]-peaks[0] == pytest.approx(32, abs=1)


def test_mean_origin_does_not_switch_wavefront_across_captures():
    origins = []
    for ppm in (0., -1., 1.):
        k, y, emissions = two_wavefront_capture(ppm)
        with pytest.warns(UserWarning):
            result = deconvolve_repetitions(y, k, rir_duration_s=.08,
                source_xyz_m=(0., 0., 0.), receiver_xyz_m=(3.43, 0., 0.),
                repeat_alignment='packet-medoid/1')
        origins.append(np.median(np.asarray(result.metadata['repeat_peak_samples'])[:, 0]-emissions))
    assert np.ptp(origins)*1000/k.fs_hz <= .02


def test_unknown_repeat_alignment_is_rejected():
    with pytest.raises(ValueError, match='unsupported repeat alignment'):
        deconvolve_repetitions(np.zeros(10), kit(), repeat_alignment='typo')


def test_packet_crop_cannot_end_before_its_geometry_origin():
    k = kit()
    with pytest.raises(ValueError, match='direct path'):
        deconvolve_repetitions(generate_kit_playback(k), k, rir_duration_s=.06,
            source_xyz_m=(0., 0., 0.), receiver_xyz_m=(343*.08, 0., 0.),
            repeat_alignment='packet-medoid/1')
