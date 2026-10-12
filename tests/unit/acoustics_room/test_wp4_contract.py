"""WP4 contracts; fixtures are project-generated proxy audio."""
import numpy as np
import pytest
from sensorsim.acoustics.room.dataset import PositionPair, split_positions
from sensorsim.acoustics.room.rir import deconvolve_known


def test_fixed_mic_source_split():
    pairs = [PositionPair(source_xyz_m=(1+i*.4, 1, 1), receiver_xyz_m=(3, 3, 1)) for i in range(6)]
    labels = split_positions(pairs, seed=4108, policy="source-region", region_size_m=.25)
    assert set(labels) == {"train", "val"}


@pytest.mark.parametrize("ppm", [-200, 150])
def test_known_clock_drift(ppm):
    from scipy.signal import fftconvolve
    from scipy.ndimage import map_coordinates
    fs = 8000
    x = np.random.default_rng(4108).normal(0, .1, fs*3)
    h = np.zeros(1600); h[31] = .8; h[83] = .2
    y = fftconvolve(x, h)
    ratio = 1 + ppm*1e-6
    warped = map_coordinates(y, [np.arange(int(len(y)*ratio))/ratio], order=3, mode="constant")
    timing = {}
    recovered = deconvolve_known(warped, x, fs=fs, correct_clock_drift=True,
                                 rir_samples=1200, timing=timing)
    assert abs(timing["drift_ppm"]-ppm) <= 20
    assert abs(int(np.argmax(abs(recovered)))-31) <= 2


def test_no_sweep_api_exists():
    from sensorsim.acoustics.room.prior import RoomSurvey, build_prior
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    assert callable(build_prior) and callable(run_no_sweep) and RoomSurvey
