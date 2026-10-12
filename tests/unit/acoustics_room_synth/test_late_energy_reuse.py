"""Exact tail state must be shared across observation bin grids."""
import numpy as np
from scipy import signal

from sensorsim.acoustics.room import late


def test_one_fade_propagation_serves_multiple_observations(monkeypatch):
    synthesis = signal.butter(2, .273, output='sos')
    analysis = signal.butter(2, [.183,.417], btype='bandpass', output='sos')
    prefix = np.sin(np.linspace(0, np.pi/2, 93))*np.exp(-np.arange(93)/317)
    amplitude, ratio = np.exp(-93/317), np.exp(-1/317)
    original = late.expected_filtered_power
    calls = []
    def counted(*args, **kwargs):
        calls.append(len(args[2]))
        return original(*args, **kwargs)
    monkeypatch.setattr(late, 'expected_filtered_power', counted)
    for offset in (3, 19, 44):
        edges = np.r_[np.arange(offset, 440, 13), 443]
        actual = late.exponential_tail_energy(synthesis, analysis, prefix, amplitude, ratio, edges)
        envelope = np.r_[prefix, amplitude*ratio**np.arange(443-len(prefix))]
        expected = np.diff(np.r_[0.,np.cumsum(original(synthesis, analysis, envelope))][edges])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)
    assert len(calls) == 1, 'the same fade covariance was propagated once per observation'


def test_tail_handles_empty_repeated_and_short_prefix_bins():
    synthesis = signal.butter(2, .31, output='sos')
    analysis = signal.butter(2, .24, output='sos')
    for prefix in (np.array([]), np.zeros(12), np.linspace(.1,1.,12)):
        ratio, amplitude = .994, .8
        edges = np.array([0,0,1,5,12,12,13,66,170])
        envelope = np.r_[prefix, amplitude*ratio**np.arange(edges[-1]-len(prefix))]
        reference = np.diff(np.r_[0.,np.cumsum(late.expected_filtered_power(synthesis,analysis,envelope))][edges])
        actual = late.exponential_tail_energy(synthesis,analysis,prefix,amplitude,ratio,edges)
        np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-13)


def test_separable_tail_equals_ungrouped_integer_rate_sum():
    from sensorsim.acoustics.room import synth
    fs, stop = 8000, 368
    time = np.arange(stop)/fs
    edges = np.r_[np.arange(19, fs, 8), fs]
    for k, band, decay in ((0, 250, .1), (2, 500, .4), (3, 2000, 1.3)):
        envelope = synth.late_envelope(time, .04, .012, decay)
        synthesis, analysis = synth.late_bands(fs)[k].sos, synth.octave_sos(fs, band)
        amplitude = float(synth.late_envelope(stop/fs, .04, .012, decay))
        ratio = float(np.exp(-3*np.log(10)/fs/decay))
        state = late.prepare_tail(synthesis, analysis, envelope, amplitude, ratio)
        actual, trace = state.integrate_edges(edges, diagnostics=True)
        assert trace['branch'] == 'separable'
        p, _, _ = late._filter_modes(synthesis)
        a, _, _ = late._filter_modes(analysis)
        poles = np.r_[ratio*p, a]
        rates = np.r_[(poles[:, None]*poles[None, :]).ravel(), ratio*ratio]
        coefficients = np.r_[state.coefficients, state.last_coefficient]
        lower = np.maximum(edges[:-1]-stop, 0)
        count = np.maximum(edges[1:]-stop, 0)-lower
        # Independently sum each ungrouped geometric sequence. No cached block,
        # coefficient consolidation or decay separability is used here.
        terms = coefficients[:, None]*rates[:, None]**lower
        terms *= -np.expm1(np.log(rates[:, None])*count)/(1-rates[:, None])
        full = np.asarray(np.sum(terms.real, axis=0, dtype=np.longdouble), float)
        full += np.diff(state.cumulative[np.minimum(edges, stop)])
        np.testing.assert_allclose(actual, full, rtol=1e-9, atol=1e-12*full.max())
