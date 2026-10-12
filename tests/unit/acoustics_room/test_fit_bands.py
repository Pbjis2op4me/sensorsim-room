"""Band support controls; measured gates stay in the frozen campaign."""
from dataclasses import replace

import numpy as np

from sensorsim.acoustics.room.fit import FitConfig, FitResult, compare, fit_room, observe, predict
from sensorsim.acoustics.room.model import RoomModel


def test_neighbor_decay_is_constrained_without_expanding_score_bands():
    cfg = FitConfig(bootstrap=0, max_nfev=1)
    room = RoomModel(shoebox_dims_m=(6, 5, 3), late_t60_s=(.12,)*7,
                     mixing_time_s=.04)
    rir = predict(FitResult(room, cfg, {}), (1, 1, 1), (4, 3, 1), 24000)
    train = observe(rir, id="train", config=cfg)
    assert set(train.bands) == {250, 500, 1000, 2000, 4000, 8000}
    # Supported measurements must replace a wrong neighboring decay prior,
    # not merely add bands to the report. No measured-room data enters here.
    prior = room.model_copy(update={"late_t60_s": (.12, .8, .12, .12, .12, .12, .8)})
    fitted = fit_room([train], prior, config=cfg)
    assert fitted.report["support_bands_hz"] == [250, 8000]
    assert fitted.model.late_t60_s[1] < .2
    assert fitted.model.late_t60_s[-1] < .2
    assert fitted.config.bands == (500, 1000, 2000, 4000)
    rows = compare(fitted, train, train)
    assert len(rows) == 32
    assert {r["band_hz"] for r in rows} == set(cfg.bands)
    assert fitted.model.provenance["training_ids"] == ["train"]


def test_neighbor_constraints_require_supported_valid_decay():
    cfg = FitConfig(bootstrap=0, max_nfev=1)
    room = RoomModel(shoebox_dims_m=(6, 5, 3), late_t60_s=(.12,)*7,
                     mixing_time_s=.04)
    rir = predict(FitResult(room, cfg, {}), (1, 1, 1), (4, 3, 1), 16000)
    # Complete 500..4000 Hz, but neither 250 nor 8000 Hz is observable.
    rir.metadata['excitation'] = dict(type='ess', f_start_hz=350, f_end_hz=6500)
    train = observe(rir, id='limited', config=cfg)
    assert 8000 not in train.bands  # Nyquist is not passed to the band filter.
    assert train.bands[250].values['t30'] is None
    fitted = fit_room([train], room, config=cfg)
    assert fitted.report['support_bands_hz'] == []
    assert fitted.model.late_t60_s[1] == .12
    # Missing decay cannot become eligible just because early echoes exist.
    train.bands[250] = replace(train.bands[500], values={k: None for k in train.bands[500].values})
    again = fit_room([train], room, config=cfg)
    assert again.report['support_bands_hz'] == []
    np.testing.assert_array_equal(again.model.late_t60(), fitted.model.late_t60())
