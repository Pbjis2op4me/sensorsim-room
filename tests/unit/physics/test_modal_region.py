"""The Schroeder frequency 2000 sqrt(T/V) bounds the diffuse-field model from below: it is computed
from the room's mid-band late T60 and volume, falls as rooms grow, and travels with every synthesized
response so low bands are flagged as the modal region."""
import numpy as np

from sensorsim.acoustics.room.model import RoomModel, uniform_bands
from sensorsim.acoustics.room.pipeline import _modal_region_lines
from sensorsim.acoustics.room.synth import synthesize


def test_schroeder_frequency_follows_its_definition():
    model = RoomModel(shoebox_dims_m=(5.4, 4.2, 2.7), absorption=uniform_bands(.2))
    t = np.mean(model.late_t60()[2:4])
    assert np.isclose(model.schroeder_frequency_hz(), 2000*np.sqrt(t/model.volume_m3))
    assert 120 < model.schroeder_frequency_hz() < 250                      # a living room: about 170 Hz
    assert model.modal_region()["bands_below_hz"] == [125]


def test_larger_rooms_have_lower_schroeder_frequencies():
    small = RoomModel(shoebox_dims_m=(3., 2.5, 2.4), absorption=uniform_bands(.2))
    hall = RoomModel(shoebox_dims_m=(30., 20., 12.), absorption=uniform_bands(.2))
    assert hall.schroeder_frequency_hz() < 125 < small.schroeder_frequency_hz()
    assert hall.modal_region()["bands_below_hz"] == []
    assert small.modal_region()["bands_below_hz"][:2] == [125, 250]


def test_synthesized_responses_and_reports_carry_the_modal_region():
    model = RoomModel(shoebox_dims_m=(5.4, 4.2, 2.7), absorption=uniform_bands(.2))
    rir = synthesize(model, (1.2, 1.1, 1.3), (4.1, 3.0, 1.4), 16000)
    assert rir.metadata["modal_region"] == model.modal_region()
    line, = _modal_region_lines(model.modal_region())
    assert "125 Hz octave band lies below it" in line
    assert _modal_region_lines(None) == []
