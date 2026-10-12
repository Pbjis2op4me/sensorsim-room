"""Independent failure cases, not a replay of implementation formula assertions."""
from dataclasses import fields

import numpy as np
import pytest

from sensorsim.acoustics.room.augment import doppler
from sensorsim.acoustics.room.model import BoxObstacle, RoomModel, SURFACES
from sensorsim.acoustics.room.obstacles import apply_obstacles, obstacle_scene, _clear_leg
from sensorsim.acoustics.room.synth import SynthConfig, image_paths, synthesize, _render_paths
from sensorsim.scene.raycore import RayCore


@pytest.mark.parametrize('low_absorption', [1., .9999])
def test_zero_dc_reflection_retains_high_frequency_energy(low_absorption):
    room = RoomModel(shoebox_dims_m=(6, 5, 3),
        absorption={s: (low_absorption, low_absorption, .36, .36, .36, .36, .36) for s in SURFACES})
    paths = image_paths(room, (1, 1, 1), (3, 2, 1), max_order=1, air_absorption=False)
    one = type(paths)(**{f.name: getattr(paths, f.name)[1:2] for f in fields(paths)})
    fs, frequency = 16000, 4000
    config = SynthConfig(late=False, air_absorption=False)
    h = _render_paths(room, one, fs, 2000, config)
    response = np.exp(-2j * np.pi * frequency * np.arange(len(h)) / fs) @ h
    # One pressure reflection sqrt(1-.36)=.8, spherical 1/r spreading.
    expected = .8 / one.distance_m[0]
    assert abs(20 * np.log10(abs(response) / expected)) < .1


def test_diffraction_cannot_escape_around_wall_contact_partition():
    room = RoomModel(shoebox_dims_m=(6, 5, 3), obstacles=(BoxObstacle(
        name='partition', min_xyz=(2.9, 0, 0), size_xyz=(.2, 5, 3)),))
    source, receiver = (1, 2, 1), (5, 2, 1)
    config = SynthConfig(late=False, max_order=0)
    paths = image_paths(room, source, receiver, max_order=0)
    adjusted, report = apply_obstacles(room, paths, source, receiver, config)
    assert np.all(adjusted.gain_octave == 0)
    assert report['dropped_paths'] == 1


def test_short_wall_contact_box_uses_free_roof_edge():
    room = RoomModel(shoebox_dims_m=(6, 5, 3), obstacles=(BoxObstacle(
        name='cabinet', min_xyz=(2.9, 0, 0), size_xyz=(.2, 4.5, 2)),))
    source, receiver = (1, .1, 1), (5, .1, 1)
    config = SynthConfig(late=False, max_order=0)
    paths = image_paths(room, source, receiver, max_order=0)
    adjusted, _ = apply_obstacles(room, paths, source, receiver, config)
    assert adjusted.distance_m[0] == pytest.approx(2 * np.hypot(1.9, 1) + .2, abs=1e-10)


def test_face_endpoint_visibility_distinguishes_departure_from_penetration():
    room = RoomModel(shoebox_dims_m=(6, 5, 3), obstacles=(BoxObstacle(
        name='box', min_xyz=(2, 2, 0), size_xyz=(1, 1, 1)),))
    starts = np.array([[2, 2.5, .5], [2, 2.5, .5], [1, 2.5, .5]])
    ends = np.array([[1, 2.5, .5], [4, 2.5, .5], [2, 2.5, .5]])
    with RayCore(obstacle_scene(room), backend='cpu') as rays:
        assert _clear_leg(rays, starts, ends).tolist() == [True, False, True]


def test_roof_grazing_uses_zero_excess_barrier_limit():
    room = RoomModel(shoebox_dims_m=(6, 5, 3), obstacles=(BoxObstacle(
        name='box', min_xyz=(2, 2, 0), size_xyz=(1, 1, 1.4)),))
    source, receiver = (1, 2.5, 1.4), (4, 2.5, 1.4)
    paths = image_paths(room, source, receiver, max_order=0, air_absorption=False)
    adjusted, _ = apply_obstacles(room, paths, source, receiver,
        SynthConfig(max_order=0, late=False, air_absorption=False))
    assert adjusted.distance_m[0] == pytest.approx(3., abs=1e-10)
    np.testing.assert_allclose(-20*np.log10(adjusted.gain_octave[0]*3), 5., atol=1e-10, rtol=0)


def test_unrepresentable_late_crossfade_rejected_before_nan_audio():
    room = RoomModel(shoebox_dims_m=(6, 5, 3), late_t60_s=(1e-6,) * 7)
    with pytest.raises(ValueError, match='crossfade|late.*decay'):
        synthesize(room, (1, 1, 1), (3, 2, 1), 8000)


def test_approaching_doppler_does_not_fold_ultrasonic_output_into_audio():
    fs = 8000
    t = np.arange(1601) / fs
    x = np.sin(2 * np.pi * 3000 * t)
    # tau'=-.5: 3000 Hz becomes 4500 Hz, above 4 kHz Nyquist.
    y = doppler(x, fs, .12, .02)
    interior = y[800:1440]
    assert np.sqrt(np.mean(interior ** 2)) < .01 / np.sqrt(2)


def test_integer_doppler_delay_preserves_last_nonzero_sample():
    fs, shift = 48000, 7
    x = np.linspace(1, 2, 64)
    y = doppler(x, fs, shift/fs, shift/fs)
    np.testing.assert_allclose(y, np.r_[np.zeros(shift), x], rtol=0, atol=1e-12)


@pytest.mark.parametrize('delays,expected', [((.01, .03), 360), ((.03, .01), 440)])
def test_doppler_approach_and_recession_sign(delays, expected):
    fs = 8000
    x = np.sin(2 * np.pi * 400 * np.arange(1601) / fs)
    y = doppler(x, fs, *delays)[400:1200]
    # Interpolate zero crossings; avoids an FFT-bin-sized frequency tolerance.
    i = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    crossing = i - y[i] / (y[i+1] - y[i])
    assert fs / np.mean(np.diff(crossing)) == pytest.approx(expected, rel=.01)
