"""Independent witnesses for the proxy join diagnostic, not renderer tuning."""
import itertools
import numpy as np
from scipy import signal

from sensorsim.acoustics.room import synth
from sensorsim.acoustics.room.model import RoomModel
from tests.validation.bench_room_join.fair import independent_path_power, windows


def test_independent_path_reference_equals_enumerated_signs():
    model = RoomModel(shoebox_dims_m=(6, 5, 3))
    fs, length = 8000, 700
    config = synth.SynthConfig(late=False, max_order=1)
    paths = synth.image_paths(model, (1,1,1), (4,3,2), max_order=1)
    analysis = signal.butter(2, [.1,.3], btype='bandpass', output='sos')
    columns = []
    for _, index, kernels in synth.path_kernels(model, paths, fs, config):
        for ix, kernel in zip(index, kernels):
            x = np.zeros(length); ok = (ix >= 0) & (ix < length)
            np.add.at(x, ix[ok], kernel[ok]); columns.append(signal.sosfilt(analysis, x))
    columns = np.array(columns)
    signs = np.array(list(itertools.product((-1,1), repeat=len(columns))))
    expected = np.mean((signs@columns)**2, axis=0)
    actual = independent_path_power(model, paths, fs, length, config, [analysis])[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)


def test_before_window_does_not_move_back_into_direct_packet():
    fs = 16000
    analyses = [synth.octave_sos(fs, b) for b in (125, 4000)]
    masks, ends = windows(fs, 2000, 64, .025, .012, analyses)
    assert ends[0] > .025
    assert not masks[0][0].any()  # LF analysis packet occupies the entire before window.
    assert masks[1][0].any()
    for sos, mm, end in zip(analyses, masks, ends):
        impulse = np.zeros(round(.2*fs)); impulse[0] = 1
        h2 = signal.sosfilt(sos, impulse)**2
        delay = np.dot(np.arange(len(h2))/fs, h2)/h2.sum()
        local = (np.arange(2000)-64)/fs-delay
        for index in (0,2):
            assert np.all(local[mm[index]] >= end)
