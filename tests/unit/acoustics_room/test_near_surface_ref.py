"""Halfspace oracle: separation bounds must stay safe and avoid false refusal."""
import numpy as np
import pytest

from sensorsim.acoustics.room.model import BoxObstacle
from sensorsim.acoustics.room.rir import first_reflection_delays, reference_direct_window


def test_table_bound_keeps_horizontal_path_and_preserves_isolation_guard():
    table = BoxObstacle(name='table', min_xyz=(1, 1, 1), size_xyz=(2, 2, .1))
    for height in (.05, .15):
        source = np.array([2., 2., 1.1+height])
        receiver = source + [.3, 0, 0]
        expected = (np.hypot(.3, 2*height)-.3)/343.
        delay, = first_reflection_delays(source, receiver, (8, 8, 6), (table,), 343.)
        assert delay == pytest.approx(expected, abs=1e-12)
        reverse, = first_reflection_delays(receiver, source, (8, 8, 6), (table,), 343.)
        assert reverse == pytest.approx(delay, abs=1e-12)
        h = np.zeros(2000)
        h[100] = 1
        if height == .05:
            with pytest.raises(ValueError, match='no resolved'):
                reference_direct_window(h, 48000, 100, first_reflection_delay_s=delay)
        else:
            _, _, stop, low = reference_direct_window(h, 48000, 100,
                                                     first_reflection_delay_s=delay)
            assert stop > 101 and low > 8000  # admission is NOT full-band support


def test_box_bound_is_conservative_for_every_sampled_surface_path():
    box = BoxObstacle(name='box', min_xyz=(2, 2, 1), size_xyz=(1, 1, 1))
    # Off-footprint, opposite sides, grazing and same-side oblique endpoints.
    for source, receiver in [([1, 1, 2.2], [1.3, 1, 2.2]),
                             ([1, 2.5, 1.5], [4, 2.5, 1.5]),
                             ([1, 1, 2], [4, 1, 2]),
                             ([1, 1, 1.5], [1.3, 1.2, 1.6])]:
        s, r = np.array(source), np.array(receiver)
        delay, = first_reflection_delays(s, r, (8, 8, 6), (box,), 343.)
        points = np.array(np.meshgrid(np.linspace(2, 3, 7), np.linspace(2, 3, 7),
                                     np.linspace(1, 2, 7))).reshape(3, -1).T
        lengths = np.linalg.norm(points-s, axis=1)+np.linalg.norm(points-r, axis=1)
        assert 0 <= delay <= (lengths.min()-np.linalg.norm(r-s))/343.+1e-12
