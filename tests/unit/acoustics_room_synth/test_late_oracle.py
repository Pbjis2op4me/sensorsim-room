"""Independent real-SOS oracle for the late tail's expected filtered power (ported from w8e-renderer-speed2).

The oracle propagates scaled state-space covariance through actual SOS sections; it shares no formulas with
late.py's modal recurrence, so it can check the production operator rather than restate it.
"""
import numpy as np
from scipy import signal
from sensorsim.acoustics.room import late
from late_real_oracle import real_expected_filtered_power, direct_sos_factor_power


def fixture():
    synthesis = signal.butter(2, .273, output='sos')
    analysis = signal.butter(2, [.183, .417], btype='bandpass', output='sos')
    prefix = np.r_[np.zeros(13), np.sin(np.linspace(0, np.pi/2, 93))*np.exp(-np.arange(93)/317)]
    amplitude, ratio = float(np.exp(-93/317)), float(np.exp(-1/317))
    return synthesis, analysis, np.r_[prefix, amplitude*ratio**np.arange(100)]


def test_oracle_qualification_and_wrong_modulation_control():
    b, a, e = fixture()
    oracle, diagnostics = real_expected_filtered_power(b, a, e)
    columns = direct_sos_factor_power(b, a, e)
    np.testing.assert_allclose(oracle, columns, rtol=1e-9, atol=1e-12*oracle.max())
    # Known-bad operator: the envelope moved after analysis. Same filters, but the output must change.
    stationary, _ = real_expected_filtered_power(b, a, np.ones(len(e)))
    assert np.max(abs(stationary*e**2 - oracle))/oracle.max() > 1e-3
    assert diagnostics['stationary_direct_vs_bilinear_norm_relative'] < 1e-10


def test_production_operator_matches_oracle_on_a_well_conditioned_band():
    b, a, e = fixture()
    oracle, _ = real_expected_filtered_power(b, a, e)
    production = np.asarray(late.expected_filtered_power(b, a, e), dtype=float)
    assert production.shape == oracle.shape
    # WP8E-NUM remains open; this procedural fixture is well conditioned.
    assert np.max(abs(production - oracle))/oracle.max() < 1e-7
