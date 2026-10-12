"""Output-envelope regression and independent expected-energy witnesses."""
import numpy as np
import pytest
from scipy import signal
from sensorsim.acoustics.room import synth
from sensorsim.acoustics.room.model import RoomModel
from tests.validation.bench_room_join.expected import expected_step, summarize


def test_expected_statistic_averages_power_before_log():
    e = np.array([[[[1., 4.]]], [[[4., 1.]]]])
    result = summarize(e, np.ones((2,1,2)), seed=4, repetitions=40)
    assert result["point_db"] == pytest.approx([0.])
    assert expected_step([[1., 10**.15]], [[1., 1.]]) == pytest.approx([1.5])


def test_stationary_output_envelope_has_no_filter_startup_hole():
    # Zero early pressure leaves the public renderer's real late algorithm.
    from unittest.mock import patch
    fs = 8000
    model = RoomModel(shoebox_dims_m=(6,5,3), late_t60_s=(.4,)*7, mixing_time_s=.04)
    band = synth.late_bands(fs)[1]
    impulse = np.zeros(fs); impulse[0] = 1
    variance = np.sum(signal.sosfilt(band.sos, impulse)**2)
    levels = np.zeros(len(synth.late_bands(fs))); levels[1] = 1
    powers = []
    with patch.object(synth, "_render_paths", side_effect=lambda m,p,f,n,c: np.zeros(n)), \
         patch.object(synth, "late_band_levels", return_value=levels*fs):
        for seed in range(64):
            rir = synth.synthesize(model, (1,1,1), (4,3,2), fs, seed=seed)
            time = (np.arange(len(rir.samples))-rir.metadata["latency_samples"])/fs
            mask = abs(time-.04) < .001
            env = synth.late_envelope(time, .04, .012, .4)
            powers.append(np.mean(rir.samples[mask]**2)/np.mean(env[mask]**2)/variance)
    assert np.mean(powers) == pytest.approx(1., abs=.25)


def test_postfilter_expected_power_matches_independent_impulse_columns():
    from sensorsim.acoustics.room.late import expected_filtered_power
    # Summing every white-innovation column is an independent reference.
    a = signal.butter(2, .25, output="sos")
    b = signal.butter(2, [.2,.5], btype="bandpass", output="sos")
    warm, n = 200, 160
    t = np.arange(n)
    envelope = np.maximum(np.sin(np.pi*np.minimum(t/60,1)/2),0)*np.exp(-t/100)
    innovations = np.eye(warm+n)
    z = signal.sosfilt(a, innovations, axis=1)[:,warm:]
    reference = np.sum(signal.sosfilt(b,z*envelope,axis=1)**2,axis=0)
    actual = expected_filtered_power(a,b,envelope)
    np.testing.assert_allclose(actual,reference,rtol=1e-9,atol=1e-13)


def test_expected_power_settles_to_cascaded_stationary_filter_energy():
    from sensorsim.acoustics.room.late import expected_filtered_power
    fs = 8000
    impulse = np.zeros(fs); impulse[0] = 1
    for k, centre in ((0,125),(1,500),(2,500),(3,1000)):
        synthesis = synth.late_bands(fs)[k].sos
        analysis = synth.octave_sos(fs,centre)
        expected = np.sum(signal.sosfilt(analysis,signal.sosfilt(synthesis,impulse))**2)
        power = expected_filtered_power(synthesis,analysis,np.ones(fs//2))
        assert power[-1] == pytest.approx(expected,rel=1e-7,abs=1e-13)


@pytest.mark.parametrize("decay", [.05,.4,3.])
def test_analytic_tail_bins_equal_the_native_sample_covariance(decay):
    from sensorsim.acoustics.room.late import expected_filtered_power, exponential_tail_energy
    fs = 8000
    time = np.arange(2400)/fs
    envelope = synth.late_envelope(time,.04,.012,decay)
    stop = int(np.ceil(.046*fs))
    edges = np.r_[np.arange(37,2300,13),2399]
    for k, centre in ((0,125),(1,250),(2,500),(3,500)):
        synthesis = synth.late_bands(fs)[k].sos
        analysis = synth.octave_sos(fs,centre)
        reference = expected_filtered_power(synthesis,analysis,envelope)
        reference = np.diff(np.r_[0.,np.cumsum(reference)][edges])
        actual = exponential_tail_energy(synthesis,analysis,envelope[:stop],
                envelope[stop],np.exp(-3*np.log(10)/fs/decay),edges)
        np.testing.assert_allclose(actual,reference,rtol=2e-7,atol=1e-10)
