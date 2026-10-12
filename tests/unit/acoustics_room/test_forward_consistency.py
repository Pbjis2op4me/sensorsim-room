"""Frozen numerical tolerances: parent-project note ROOM_FIT_RERUN (not distributed).

These compare two forward paths without fitting either to the other.
"""
import numpy as np
import pytest

from sensorsim.acoustics.room.fit import (
    FitConfig, FitResult, _cache, _expected, observe, predict,
)
from sensorsim.acoustics.room.model import OCTAVE_HZ, RoomModel, SURFACES, uniform_bands


@pytest.mark.parametrize("case", ["flat", "coloured", "coloured_walls", "scattering"])
def test_cached_energy_and_waveform_metrics_agree(case):
    config = FitConfig(bootstrap=0)
    decay = (.9, .8, .6, .45, .3, .15, .75) if case == "coloured" else (.4,) * 7
    absorption = ({s: (.7, .55, .45, .3, .25, .15, .1) for s in SURFACES}
                  if case == "coloured_walls" else uniform_bands(.25))
    model = RoomModel(shoebox_dims_m=(6, 5, 3), absorption=absorption,
                      scattering=uniform_bands(.7 if case == "scattering" else 0.),
                      late_t60_s=decay, mixing_time_s=.04)
    result = FitResult(model, config, {})
    waveforms = [observe(predict(result, (1.1, 1.3, 1.2), (4.2, 3.7, 1.7), 48000,
                                id=str(seed)), id=str(seed), config=config) for seed in range(8)]
    obs = waveforms[0]
    cache = _cache(model, obs, config)
    for band in config.bands:
        b = OCTAVE_HZ.index(band)
        alpha = np.array([model.absorption[s][b] for s in SURFACES])
        fast = _expected(model, obs, cache, band, alpha, decay[b], config)[0]
        for metric in ("edt", "t30", "c50_db"):
            measured = np.median([o.bands[band].values[metric] for o in waveforms])
            error = abs(fast[metric] - measured)
            tolerance = 1. if metric == "c50_db" else .1 * measured
            assert error <= tolerance, (case, band, metric, fast[metric], measured)


def test_isolated_reflections_use_native_decay_and_direct_windows():
    config = FitConfig(bands=(1000,), late=False, bootstrap=0)
    alpha = uniform_bands(1.); alpha['z_min'] = (.4,)*7
    model = RoomModel(shoebox_dims_m=(8, 8, 4), absorption=alpha)
    result = FitResult(model, config, {})
    obs = observe(predict(result, (2, 2, 1), (3, 2, 1.2), 44100),
                  id='pulse', config=config)
    fast = _expected(model, obs, _cache(model, obs, config), 1000,
                     np.array([alpha[s][3] for s in SURFACES]), .4, config)[0]
    for metric in ('edt', 't30'):
        assert fast[metric] == pytest.approx(obs.bands[1000].values[metric], rel=.01)
    # Exact +/-2.5 ms sample window; 3% energy error would be ~0.13 dB.
    assert fast['drr_db'] == pytest.approx(obs.bands[1000].values['drr_db'], abs=.13)
