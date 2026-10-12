"""Independent room controls frozen in parent-project note verify-room (not distributed)."""
import numpy as np
import pytest

from sensorsim.acoustics.room.ingest import ESSExcitation
from sensorsim.acoustics.room.metrics import room_metrics
from sensorsim.acoustics.room.rir import (
    RIR, adaptive_decay_window, deconvolve_ess, first_reflection_delays,
    generate_ess, trim_rir,
)


def test_reflection_bound_keeps_fractional_wall_with_integer_source():
    # Nearest wall x=1.3: reflected source x=1.6, path=.5 vs direct=.1 m.
    expected = .4 / 343.
    for source in ([1, 2, 2], [1., 2., 2.]):
        actual = first_reflection_delays(source, [1.1, 2., 2.], (1.3, 6., 6.), (), 343.)
        assert actual == pytest.approx([expected], abs=1e-12)


def test_adaptive_window_only_evaluates_acquired_complete_octaves():
    stop, report = adaptive_decay_window(np.zeros(800), 8000, minimum_s=.1,
                                         f_start_hz=40., f_end_hz=3600.)
    assert stop == 800 and report['tail_limited']
    step, = report['steps']
    assert step['supported_bands'] == 5
    assert {r['band_hz'] for r in step['unresolved']} == {125, 250, 500, 1000, 2000}


def test_trim_preserves_absolute_fractional_arrivals_through_two_crops():
    h = np.zeros((300, 2))
    h[100, 0], h[120, 1] = 1., .7
    rir = RIR(10000, h, metadata={'direct_samples': [100, 120],
                                'direct_samples_fractional': [100.25, 120.75]})
    first = trim_rir(rir, onset_sample=100, pre_s=.0025)
    second = trim_rir(first, onset_sample=25, pre_s=.001)
    for cropped in (first, second):
        absolute = (np.array(cropped.metadata['direct_samples_fractional'])/10000
                    + cropped.metadata['time_origin_s'])
        np.testing.assert_allclose(absolute, [.010025, .012075], atol=1e-12, rtol=0)
    assert rir.metadata['direct_samples_fractional'] == [100.25, 120.75]


def test_pressure_metrics_reject_complex_samples_before_casting():
    with pytest.raises(ValueError, match='real'):
        room_metrics(np.ones(80, dtype=complex)*(1+1j), 8000,
                     onset_sample=0, noise_power=0)


def test_lundeby_missing_tail_matches_infinite_geometric_energy_and_moment():
    fs, t60, noise = 8000, .4, 1e-5
    q = np.exp(-6*np.log(10)/(fs*t60))
    # A deterministic power oracle avoids both stochastic uncertainty and a
    # noise cross term. The supplied stationary power floor is exact.
    h = np.sqrt(q**np.arange(8000)+noise)
    measured = room_metrics(h, fs, onset_sample=0, noise_power=noise)
    assert measured.decay.valid and measured.decay.tail_energy > 0
    assert measured.decay.truncation_sample < len(h)
    assert measured.decay.energy[0] == pytest.approx(1/(1-q), rel=1e-6)
    assert measured.ts_s == pytest.approx(q/(fs*(1-q)), abs=1e-6)
    for fit in (measured.edt, measured.t20, measured.t30):
        assert fit.valid and fit.rt60_s == pytest.approx(t60, rel=1e-4)


def test_ess_quadratic_packet_has_analytic_in_band_transfer_amplitude():
    fs = 8000
    spec = ESSExcitation(duration_s=1., f_start_hz=80., f_end_hz=3600.,
                         amplitude=.5, fade_s=.005)
    x = generate_ess(fs, spec)
    result = deconvolve_ess(np.r_[x+.2*x*x, np.zeros(400)], fs, spec,
                            rir_duration_s=.025, packet_pre_s=.005, orders=(2,))
    packet = result.harmonics[2]
    frequency = np.arange(700., 2501., 100.)
    response = np.exp(-2j*np.pi*frequency[:, None]*np.arange(len(packet))/fs) @ packet
    # sin(theta)^2 = (1-cos(2 theta))/2. Stationary-phase magnitudes
    # scale as A/sqrt(f), so quadratic/linear transfer is .2*A/2 = .05.
    np.testing.assert_allclose(abs(response), .05, rtol=.02, atol=0)
    expected_advance = np.log(2)/np.log(45)
    assert result.harmonic_advance_s[2] == pytest.approx(expected_advance, abs=1/fs)
