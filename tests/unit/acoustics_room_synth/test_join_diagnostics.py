"""Checks for the independent trend/precursor diagnostics, not room qualification."""
import numpy as np
import pytest

from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, image_paths, path_kernels
from tests.validation.bench_room_join.run import energy_step, precursor, registered_geometry, verify_registration


def test_trend_reference_removes_decay_but_retains_known_step():
    fs, t60, mixing = 44100, .3, .02731
    time = np.arange(4000)/fs
    x = np.exp(-3*np.log(10)*time/t60)
    raw, corrected = energy_step(x, time, mixing, t60)
    assert raw < -1.9
    assert corrected == pytest.approx(0., abs=1e-12)
    x[time >= mixing] *= 10**(6/20)
    assert energy_step(x, time, mixing, t60)[1] == pytest.approx(6., abs=1e-12)


def test_nonconstant_window_power_is_not_erased_by_trend_reference():
    time = np.arange(4000)/48000
    x = np.exp(-3*np.log(10)*time/.5)
    x[time >= .03] *= .25
    assert energy_step(x, time, .03, .5)[1] == pytest.approx(20*np.log10(.25), abs=1e-12)


def test_composite_interpolation_basis_cancels_its_precursors():
    model = RoomModel(shoebox_dims_m=(6,5,3))
    paths = image_paths(model, (1,1,1), (3,2,1), max_order=0, air_absorption=False)
    cfg = SynthConfig(late=False, max_order=0, air_absorption=False)
    _, _, full = next(path_kernels(model, paths, 48000, cfg))
    components = []
    for b in range(7):
        pressure = np.zeros_like(paths.wall_pressure); pressure[:, b] = 1.
        components.append(next(path_kernels(model, paths, 48000, cfg, wall_pressure=pressure))[2])
    np.testing.assert_allclose(np.sum(components, axis=0), full, rtol=1e-12, atol=1e-14)
    peak = np.argmax(np.abs(full[0]))
    individual = sum(np.sum(h[0, :peak]**2) for h in components)
    composite = np.sum(full[0, :peak]**2)
    assert individual > composite


def test_precursor_excludes_main_lobe_and_detects_remote_pulse():
    x = np.zeros(512); x[300] = 1; x[100] = .01
    got, mask = precursor(x, 48000, 300.25)
    assert not mask[300] and mask[100]
    assert got['energy_fraction'] == pytest.approx(.0001/1.0001)
    assert got['peak_db'] == pytest.approx(-40.)
    x[100] = 0
    got, _ = precursor(x, 48000, 300.25)
    assert got['energy_fraction'] == 0 and got['peak_db'] is None


@pytest.mark.parametrize('bad', [np.zeros(100), np.full(100, np.nan)])
def test_undefined_signal_is_rejected(bad):
    with pytest.raises(ValueError):
        energy_step(bad, np.arange(100)/1000, .05, .5)
    with pytest.raises(ValueError):
        precursor(bad, 1000, 50)


def test_registration_hash_checked_before_run(tmp_path):
    path, digest = tmp_path/'prereg.yaml', tmp_path/'sha.txt'
    path.write_text('threshold: 1\n'); digest.write_text('0'*64)
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_registration(path, digest)


def test_geometry_comes_from_registration_and_cannot_silently_fall_back():
    geometry = dict(dimensions_m=[4,5,3], source_m=[1,2,1], receiver_m=[2,3,1])
    assert registered_geometry({'geometry': geometry}) == geometry
    assert registered_geometry({'splits': {'acceptance':
        'procedural room dimensions [4,5,3], source [1,2,1], receiver [2,3,1]'}}) == geometry
    with pytest.raises(ValueError, match='declare'):
        registered_geometry({})
