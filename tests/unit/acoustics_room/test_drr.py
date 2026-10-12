"""Frozen drr/2 constructs; independent spectral oracle, no room-fit tuning."""
import numpy as np
import pytest

from sensorsim.acoustics.room.metrics import OCTAVE_BANDS, analyze_octaves, metric_policy, room_metrics

FS = 48000
ONSET = 480


def response():
    h = np.zeros(12000)
    h[ONSET] = 1.
    return h


def measure(h, onset=ONSET):
    return np.array([room_metrics(h, FS, band=b,
                    onset_sample=onset, noise_power=0).drr_db for b in OCTAVE_BANDS])


def spectral_energy(h, band):
    """Parseval oracle using the analytic bilinear Butterworth magnitude.

    Deliberately does not use octave_sos, sosfilt, or production DRR helpers.
    65536 bins resolve the narrowest octave and its exponential ringing.
    """
    nfft = 65536
    centers = 1000 * 10.**(.3*np.arange(-3, 4))
    center = centers[OCTAVE_BANDS.index(band)]
    lo, hi = np.tan(np.pi*center*np.array([10**-.15, 10**.15])/FS)
    w = np.tan(np.pi*np.arange(1, nfft//2)/nfft)
    magnitude_squared = 1/(1+((w*w-lo*hi)/(w*(hi-lo)))**12)
    spectrum = np.fft.rfft(h, nfft)
    return float(2/nfft*np.dot(abs(spectrum[1:-1])**2, magnitude_squared))


def test_direct_impulse_has_no_reverberation_in_any_octave():
    assert np.all(measure(response()) > 120)


def test_isolated_reflection_has_analytic_band_drr():
    h = response()
    h[ONSET+round(.020*FS)] = .5
    np.testing.assert_allclose(measure(h), -20*np.log10(.5), atol=.1, rtol=0)
    # A reflection at the final recorded sample must retain the same ring-out.
    np.testing.assert_allclose(measure(h[:ONSET+round(.020*FS)+1]),
                               -20*np.log10(.5), atol=.1, rtol=0)


def test_exponential_diffuse_tail_matches_independent_energy_oracle():
    direct = response()
    tail = np.zeros_like(direct)
    start = ONSET+round(.020*FS)
    n = np.arange(len(tail)-start)
    tail[start:] = np.random.default_rng(20261003).normal(size=len(n))*np.exp(-n/(.015*FS))
    tail *= np.sqrt(.25/np.dot(tail, tail))
    expected = [10*np.log10(spectral_energy(direct, b)/spectral_energy(tail, b))
                for b in OCTAVE_BANDS]
    np.testing.assert_allclose(measure(direct+tail), expected, atol=.1, rtol=0)


def test_subwindow_onset_perturbation_does_not_move_band_energy():
    h = response()
    h[ONSET+round(.020*FS)] = .5
    nominal = measure(h)
    for offset in (-3, -2, -1, 1, 2, 3):
        np.testing.assert_allclose(measure(h, ONSET+offset), nominal, atol=.05, rtol=0)


def test_default_and_explicit_legacy_are_versioned_without_changing_decay():
    from dataclasses import replace

    h = response()
    h[ONSET+960] = .5
    current = analyze_octaves(h, FS, onset_sample=ONSET, noise_power=0)
    legacy = analyze_octaves(h, FS, onset_sample=ONSET, noise_power=0, legacy_drr=True)
    assert legacy[250].drr_db == pytest.approx(-30.124903554158998, abs=1e-10)
    for band in ("broadband", *OCTAVE_BANDS):
        assert current[band].drr_definition == "drr/2"
        assert legacy[band].drr_definition == "drr/1"
        assert metric_policy(band)["drr"]["definition"] == "drr/2"
        assert metric_policy(band, legacy_drr=True)["drr"]["definition"] == "drr/1"
        for field in ("edt", "t20", "t30", "c50_db", "c80_db", "d50", "ts_s"):
            assert getattr(current[band], field) == getattr(legacy[band], field)
        np.testing.assert_array_equal(current[band].decay.energy, legacy[band].decay.energy)
    with pytest.raises(ValueError, match="unsupported DRR"):
        replace(current[250], drr_definition="drr/unknown")


def test_unavailable_noise_or_direct_support_is_not_an_infinite_drr():
    assert room_metrics(np.zeros(512), FS, onset_sample=0, band=125, noise_power=0).drr_db is None
    h = response()
    assert room_metrics(h[:ONSET+1], FS, onset_sample=ONSET, band=125, noise_power=0).drr_db is None
    noisy = room_metrics(np.ones(512), FS, onset_sample=0, band=125, noise_power=1)
    assert not noisy.decay.valid and noisy.drr_db is None


def test_fit_rescore_labels_both_definitions_and_retains_missing_predictions():
    import json
    from sensorsim.acoustics.room.rir import RIR
    from tests.validation.bench_room_fit.run import paired_drr_rows

    h = response()
    h[ONSET+960] = .5
    measured = RIR(FS, h, metadata={'synthetic': True})
    rows = paired_drr_rows(measured, None, id='held-out', split='unseen_source',
                           bands=(125, 250), onset=ONSET)
    assert {(r['drr_definition'],r['band_hz']) for r in rows} == {
        ('drr/1',125),('drr/1',250),('drr/2',125),('drr/2',250)}
    assert all(r['status']=='FAIL' and r['predicted'] is None and r['tolerance']==3 for r in rows)
    assert all(r['id']=='held-out' and r['split']=='unseen_source' for r in rows)
    json.dumps(rows, allow_nan=False)
