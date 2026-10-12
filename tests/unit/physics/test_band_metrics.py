"""room-measured-metrics/2: band analysis that keeps the octave filter's own ringing and delay out of
the metrics. Time-reversed filtering for decays, zero-phase filtering for energy ratios."""
import numpy as np
import pytest
from scipy.signal import sosfilt

from sensorsim.acoustics.room.metrics import (LEGACY_METRIC_POLICY, METRIC_POLICY, filter_span, metric_policy,
                                               octave_sos, reverse_filter, room_metrics, zero_phase_filter)

FS = 48000


def test_time_reversed_filtering_puts_the_ringing_before_the_event():
    x = np.zeros(FS); x[30000] = 1.
    y = reverse_filter(x, FS, 125)
    h = sosfilt(octave_sos(FS, 125), np.r_[1., np.zeros(FS - 1)])
    assert np.sum(y[30001:]**2) < 1e-12*np.sum(y**2)                     # nothing after the event
    assert np.sum(y**2) == pytest.approx(np.sum(h**2), rel=1e-6)          # same band energy


def test_zero_phase_filtering_is_symmetric_and_keeps_pre_ringing_before_the_first_sample():
    x = np.zeros(FS); x[0] = 1.                                            # an event at the very start
    y, offset = zero_phase_filter(x, FS, 250)
    assert offset == filter_span(FS, 250)
    left, right = y[:offset + 1][::-1], y[offset:offset + offset + 1]
    np.testing.assert_allclose(left, right, atol=1e-9)                     # symmetric about the event


def _decay(t60, seed, seconds=1.6, lead=.05):
    t = np.arange(int(seconds*FS))/FS
    noise = np.random.default_rng(seed).standard_normal(t.size)
    return np.r_[np.zeros(int(lead*FS)), noise*np.exp(-3*np.log(10)*t/t60)]


def _ensemble(band, t60, metric, policy, draws=48):
    out = []
    for seed in range(draws):
        m = room_metrics(_decay(t60, seed), FS, onset_sample=int(.05*FS), noise_power=0., band=band, policy=policy)
        out.append(getattr(m, metric).rt60_s if metric in ("edt", "t30") else getattr(m, metric))
    return np.array(out)


def test_low_band_edt_no_longer_includes_the_filter_ringing():
    """125 Hz, T60 0.3 s: the causal filter's ringing lengthens EDT by about 23 % in expectation."""
    legacy, current = _ensemble(125, .3, "edt", LEGACY_METRIC_POLICY), _ensemble(125, .3, "edt", METRIC_POLICY)
    assert np.mean(legacy - current) > .04                                 # paired: same draws
    assert abs(np.median(current)/.3 - 1) < .12


def test_low_band_clarity_matches_the_closed_form():
    """250 Hz, T60 0.5 s: C50 = 10 log10(e^{13.8*0.05/T60} - 1) = 4.74 dB."""
    truth = 10*np.log10(np.exp(6*np.log(10)/.5*.05) - 1)
    legacy, current = _ensemble(250, .5, "c50_db", LEGACY_METRIC_POLICY), _ensemble(250, .5, "c50_db", METRIC_POLICY)
    assert abs(np.mean(current) - truth) < .8
    assert np.mean(current - legacy) > 1.0


def test_the_policy_is_recorded():
    assert metric_policy(125)["method"] == METRIC_POLICY == "room-measured-metrics/2"
    with pytest.raises(ValueError, match="unknown metric policy"):
        room_metrics(_decay(.5, 0), FS, onset_sample=2400, band=1000, policy="room-measured-metrics/9")
