"""Speed of sound: Cramer (1993) for humid air by default, the dry approximation by name, one policy
shared by the room model and the pipeline's arrival anchor."""
import numpy as np
import pytest

from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.pipeline import PipelineConfig, _model_speed, _timing_speed
from sensorsim.core.geometry import sound_speed, speed_of_sound, speed_of_sound_cramer

# Reference values from pyfar 0.8.1's independent implementation of Cramer's equation (CO2 425.19 ppm).
CRAMER = [((20., 50., 101.325), 343.9844829107664), ((0., 0., 101.325), 331.44552662632987),
          ((30., 100., 90.), 351.74975303157265)]


@pytest.mark.parametrize("state, expected", CRAMER)
def test_cramer_matches_an_independent_implementation(state, expected):
    assert speed_of_sound_cramer(*state, co2_ppm=425.19) == pytest.approx(expected, abs=1e-9)


def test_humid_air_is_faster_than_dry_air_and_rises_with_humidity():
    speeds = [speed_of_sound_cramer(20., rh) for rh in (0., 25., 50., 75., 100.)]
    assert np.all(np.diff(speeds) > 0)
    assert 0.6 < speed_of_sound_cramer(20., 50.) - speed_of_sound(20.) < 0.9          # about 0.77 m/s


@pytest.mark.parametrize("bad", [dict(temperature_c=-5.), dict(temperature_c=31.), dict(pressure_kpa=60.),
                                 dict(relative_humidity_pct=120.)])
def test_outside_its_range_it_raises_instead_of_extrapolating(bad):
    with pytest.raises(ValueError, match="humid-air-cramer/1"):
        speed_of_sound_cramer(**{"temperature_c": 20., "relative_humidity_pct": 50., **bad})


def test_the_room_model_and_the_arrival_anchor_share_one_policy():
    humid = RoomModel(shoebox_dims_m=(5, 4, 3), temperature_c=22., humidity_pct=60.)
    assert humid.sound_speed_policy == "humid-air-cramer/1"
    assert humid.sound_speed_m_s == pytest.approx(sound_speed("humid-air-cramer/1", 22., 60.))
    config = PipelineConfig(temperature_c=22., humidity_pct=60.)
    assert _timing_speed(config) == humid.sound_speed_m_s
    assert RoomModel(shoebox_dims_m=(5, 4, 3), temperature_c=22., humidity_pct=60.,
                     **_model_speed(config)).sound_speed_m_s == _timing_speed(config)
    dry = PipelineConfig(timing_policy="dry-air-core/1", temperature_c=22., humidity_pct=60.)
    assert _timing_speed(dry) == pytest.approx(speed_of_sound(22.))
    assert RoomModel(shoebox_dims_m=(5, 4, 3), temperature_c=22., **_model_speed(dry)).sound_speed_policy == "dry-air-core/1"
    fixed = PipelineConfig(timing_policy="fixed-reference/343")
    assert _timing_speed(fixed) == 343. and RoomModel(shoebox_dims_m=(5, 4, 3), **_model_speed(fixed)).sound_speed_m_s == 343.


def test_a_room_outside_the_humid_range_must_choose_a_policy():
    with pytest.raises(ValueError, match="dry-air-core/1"):
        RoomModel(shoebox_dims_m=(5, 4, 3), temperature_c=35.)
    assert RoomModel(shoebox_dims_m=(5, 4, 3), temperature_c=35., sound_speed_model="dry-air-core/1").sound_speed_m_s \
        == pytest.approx(speed_of_sound(35.))
