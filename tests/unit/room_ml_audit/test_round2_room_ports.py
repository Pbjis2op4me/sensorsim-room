"""Round-two regression tests for adapted parent room fixes."""
import numpy as np
import pytest

from sensorsim.acoustics.room.directivity import Directivity, DirectivityProfile, Orientation
from sensorsim.acoustics.room.fit import FitConfig, Observation, _cache, _expected_all, estimate_geometry
from sensorsim.acoustics.room.model import BoxObstacle, EarlyScattering, RoomModel, SURFACES, uniform_bands
from sensorsim.acoustics.room.pipeline import PipelineConfig
from sensorsim.acoustics.room.synth import (SynthConfig, _render_paths, crossfade_fraction,
    early_paths, resolve_config, synthesize)

FS = 8000
SOURCE, RECEIVER = (1.1, 1.3, 1.2), (4.2, 3.7, 1.7)


def _balloon():
    return DirectivityProfile(name="band-varying fixture",
        directions_xyz=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        pressure_gain=((1.,)*7, (.2, .3, .4, .5, .6, .7, .8), (.4,)*7, (.5,)*7, (.6,)*7, (.7,)*7),
        solid_angle_sr=(4*np.pi/6,)*6, reference_gain=(1.,)*7, provenance={"source": "analytic fixture"})


def _late_components(model):
    config = FitConfig(bands=(1000,), neighbor_bands=False, bootstrap=0)
    obs = Observation("probe", SOURCE, RECEIVER, FS, {}, np.array([]))
    cache = _cache(model, obs, config)
    alpha = np.array([model.absorption[s] for s in SURFACES])
    _expected_all(model, obs, cache, alpha, model.late_t60(), config)
    return cache


@pytest.mark.parametrize("pattern", ["cardioid", "measured"])
def test_fit_late_tail_carries_both_devices_diffuse_power(pattern):
    omni = RoomModel(shoebox_dims_m=(6, 5, 3), mixing_time_s=.04, late_t60_s=(.2,)*7)
    if pattern == "cardioid":
        source = receiver = Directivity(pattern="cardioid")
    else:
        source = Directivity(pattern="measured", profile=_balloon(), orientation=Orientation())
        receiver = Directivity(pattern="measured", profile=_balloon(), orientation=Orientation(yaw_deg=90))
    directed = RoomModel.model_validate({**omni.model_dump(), "source_directivity": source,
                                         "receiver_directivity": receiver})
    expected = np.broadcast_to(np.asarray(source.diffuse_power)*receiver.diffuse_power, (7,))
    base, actual = _late_components(omni), _late_components(directed)
    assert actual.device_power == pytest.approx(expected)
    assert base.device_power == pytest.approx(np.ones(7))
    assert actual.late_components.keys() == base.late_components.keys()
    for (band, octave), (_, reference_component) in base.late_components.items():
        _, directed_component = actual.late_components[band, octave]
        assert directed_component == pytest.approx(expected[octave]*reference_component, rel=1e-12)
    if pattern == "cardioid":
        start = round(.08*FS)+64
        tails = [np.sum(synthesize(m, SOURCE, RECEIVER, FS, seed=3).samples[start:].astype(float)**2)
                 for m in (omni, directed)]
        assert tails[1]/tails[0] == pytest.approx(1/9, rel=1e-6)


def test_config_json_and_pipeline_roundtrip_preserve_inherit_override_disable():
    cardioid = RoomModel(shoebox_dims_m=(6, 5, 3), source_directivity=Directivity(pattern="cardioid"))
    scatter = RoomModel(shoebox_dims_m=(6, 5, 3), early_scattering=EarlyScattering(rays=64))
    replayers = (lambda c: SynthConfig.model_validate_json(c.model_dump_json()),
                 lambda c: PipelineConfig.model_validate(PipelineConfig(synthesis=c).model_dump()).synthesis)
    for replay in replayers:
        inherited = SynthConfig(late=False, max_order=0, air_absorption=False)
        explicit_omni = SynthConfig(late=False, max_order=0, air_absorption=False,
                                    source_directivity=Directivity())
        assert resolve_config(cardioid, replay(inherited)).source_directivity.pattern == "cardioid"
        assert resolve_config(cardioid, replay(explicit_omni)).source_directivity.pattern == "omni"
        assert resolve_config(scatter, replay(SynthConfig(early_scattering=None))).early_scattering is None
        assert resolve_config(scatter, replay(SynthConfig())).early_scattering == EarlyScattering(rays=64)


def test_config_roundtrip_preserves_rendered_samples():
    model = RoomModel(shoebox_dims_m=(6, 5, 3), source_directivity=Directivity(pattern="cardioid"))
    config = SynthConfig(late=False, max_order=0, air_absorption=False)
    replay = SynthConfig.model_validate_json(config.model_dump_json())
    a = synthesize(model, (1, 2, 1), (1, 3, 1), FS, config=config)
    b = synthesize(model, (1, 2, 1), (1, 3, 1), FS, config=replay)
    assert np.array_equal(a.samples, b.samples)


def test_whole_number_coordinates_are_not_truncated_in_geometry_fit():
    def labelled(source, receivers, dims):
        observations = []
        for i, receiver in enumerate(receivers):
            s, r = np.asarray(source, float), np.asarray(receiver, float)
            echoes = {}
            for axis, name in enumerate(("x_max", "y_max", "z_max")):
                image = s.copy()
                image[axis] = 2*dims[axis]-s[axis]
                echoes[name] = float((np.linalg.norm(image-r)-np.linalg.norm(s-r))/343.)
            observations.append(Observation(str(i), tuple(s), tuple(r), 48000, {},
                                           np.array(list(echoes.values())), {"wall_echoes_s": echoes}))
        return observations
    dims = np.array([5.4, 6.2, 3.7])
    receivers = ((2,2,1), (3,1,2), (3,4,2))
    integer = estimate_geometry(labelled((1,1,1), receivers, dims))
    floating = estimate_geometry(labelled((1.,1.,1.), receivers, dims))
    np.testing.assert_allclose(integer["shoebox_dims_m"], dims, atol=1e-6)
    np.testing.assert_allclose(integer["shoebox_dims_m"], floating["shoebox_dims_m"], atol=1e-8)


CASES = {
    "near-floor-screen": ((2.99, 1., 0.), (.02, 3., .5), (2., 2., .25), (4., 2., .25)),
    "elevated-thin-screen": ((2.99, 1., .4), (.02, 3., .5), (2., 2., .6), (4., 2., .6)),
    "shadow-body": ((2.64, 1., 0.), (.72, 3., 1.4), (1., 2.5, .4), (5., 2.5, .4)),
    "lit-low-body": ((2.64, 1., 0.), (.72, 3., .72), (1., 2.5, .8), (5., 2.5, .8)),
}


@pytest.mark.parametrize("case", CASES)
def test_cached_obstacle_floor_groups_follow_fitted_absorption(case):
    minimum, size, source, receiver = CASES[case]
    box = BoxObstacle(name="box", min_xyz=minimum, size_xyz=size)
    config = FitConfig(bands=(1000,), late=False, neighbor_bands=False, bootstrap=0)
    def model(floor):
        alpha = uniform_bands(1.)
        alpha["z_min"] = tuple(floor) if isinstance(floor, tuple) else (floor,)*7
        return RoomModel(shoebox_dims_m=(6,5,3), absorption=alpha, obstacles=(box,))
    prior = model(.8)
    obs = Observation(case, source, receiver, FS, {}, np.array([]))
    cache = _cache(prior, obs, config)
    for floor in (.05, .37, .8, .99, (.05, .6, .37, .9, .2, .99, .5)):
        candidate = model(floor)
        alpha = np.array([candidate.absorption[s] for s in SURFACES])
        _expected_all(candidate, obs, cache, alpha, candidate.late_t60(), config)
        fitted = sum(cache.early_components[j][1] for j in range(7))
        rendered = synthesize(candidate, source, receiver, FS,
                              config=SynthConfig(late=False, max_order=1)).samples.astype(float)
        assert len(fitted) == len(rendered)
        assert np.linalg.norm(fitted-rendered)/np.linalg.norm(rendered) < 1e-6


@pytest.mark.parametrize("fs", [16000, 48000])
def test_late_cached_floor_groups_match_renderer(fs):
    minimum, size, source, receiver = CASES["near-floor-screen"]
    box = BoxObstacle(name="box", min_xyz=minimum, size_xyz=size)
    def model(floor):
        alpha = uniform_bands(.3)
        alpha["z_min"] = tuple(floor) if isinstance(floor, tuple) else (floor,)*7
        return RoomModel(shoebox_dims_m=(6,5,3), absorption=alpha, obstacles=(box,))
    config = FitConfig(bands=(1000,), neighbor_bands=False, bootstrap=0)
    obs = Observation("late floor", source, receiver, fs, {}, np.array([]))
    cache = _cache(model(.8), obs, config)
    for floor in (.05, .37, .8, .99, (.05, .6, .37, .9, .2, .99, .5)):
        candidate = model(floor)
        alpha = np.array([candidate.absorption[s] for s in SURFACES])
        _expected_all(candidate, obs, cache, alpha, candidate.late_t60(), config)
        fitted = sum(cache.early_components[j][1] for j in range(7))
        cfg = SynthConfig()
        paths = early_paths(candidate, np.array(source), np.array(receiver), cfg)[0]
        rendered = _render_paths(candidate, paths, fs, len(fitted), cfg)
        t = (np.arange(len(fitted))-cfg.filter_half_length)/fs
        rendered *= np.cos(np.pi/2*crossfade_fraction(t, candidate.mixing_time(), cfg.crossfade_s))
        assert np.linalg.norm(fitted-rendered)/np.linalg.norm(rendered) < 1e-6
