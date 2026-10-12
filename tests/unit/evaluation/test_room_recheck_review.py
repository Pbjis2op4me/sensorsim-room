"""Procedural regressions for the review fixes; no measured dataset is opened."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal

from sensorsim.evaluation import room_recheck as rr
from sensorsim.acoustics.room.metrics import OCTAVE_BANDS, room_metrics


def load_bench():
    path = Path(__file__).resolve().parents[2] / "validation/bench_room_recheck/run.py"
    spec = importlib.util.spec_from_file_location("wp8_review_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_b5_noise_free_truncated_decay_retains_metrics():
    fs = 24000
    t = np.arange(round(.5 * fs)) / fs
    x = np.random.default_rng(7).normal(size=len(t)) * np.exp(-np.log(1000) * t / .6)
    got = rr.octave_room_metrics(x, fs, onset_sample=0, noise_power=0.0)
    assert got["t30_valid"].all()
    assert np.isfinite(got["c50_db"]).all()
    for i, band in enumerate(OCTAVE_BANDS):
        ref = room_metrics(x, fs, onset_sample=0, band=band, noise_power=0.0)
        assert got["edt_s"][i] == pytest.approx(ref.edt.rt60_s)


def fake_pair():
    return SimpleNamespace(state="fixture", sample_rate_hz=24000,
        source_xyz_m=(1., 1., 1.), receiver_xyz_m=(2., 2., 1.), room_dims_m=(4., 4., 3.),
        measured_onset=0, rir=np.ones(24000), split="locked_audit", pair_id="procedural",
        measured_metrics={k: np.ones(7) for k in ("edt_s", "t30_s", "c50_db", "drr_db", "edt_valid", "t30_valid")})


def fake_synth(*args, **kwargs):
    return SimpleNamespace(samples=np.ones(24000), metadata={"direct_delay_s": .004,
        "latency_samples": 64, "mixing_time_s": .03, "crossfade_s": .012})


def test_b5_measured_comparison_explicitly_disables_sim_noise(monkeypatch, tmp_path):
    bench = load_bench()
    monkeypatch.setattr(bench, "synthesize", fake_synth)
    monkeypatch.setattr(bench, "compare_edge_feature_pair", lambda *a, **k: (np.ones((3, 120)),)*2)
    class Reached(Exception): pass
    def metric(*args, **kwargs):
        assert kwargs.get("noise_power") == 0.0
        raise Reached
    monkeypatch.setattr(bench, "octave_room_metrics", metric)
    with pytest.raises(Reached):
        bench._render_measured_comparison([fake_pair()], {"fixture": np.ones(7)*.2}, tmp_path, 1)


def test_b5_all_early_variants_explicitly_disable_sim_noise(monkeypatch, tmp_path):
    bench = load_bench()
    monkeypatch.setattr(bench, "synthesize", fake_synth)
    calls = []
    def metric(*args, **kwargs):
        calls.append(kwargs.get("noise_power"))
        return {"edt_s": np.ones(7)}
    monkeypatch.setattr(bench, "octave_room_metrics", metric)
    bench._early_variants([fake_pair()], {"fixture": np.ones(7)*.2},
        {"measured_metrics": {"edt_s": np.ones((1, 7))}}, tmp_path, 1)
    assert calls == [0.0]*5


def test_join_null_removes_decay_and_samples_ensemble_energy():
    # Every realization has a large legitimate decay step. Comparing the
    # ensemble energy to its no-join continuation must yield exactly zero.
    baseline = np.random.default_rng(12).lognormal(size=(80, 2, 4))
    baseline[:, :, 1] *= .2
    got = rr.join_ensemble_summary(baseline, baseline, bootstrap_draws=200, seed=9, threshold_db=1.)
    np.testing.assert_allclose(got["step_excess_db"], 0., atol=1e-12)
    np.testing.assert_allclose(got["step_excess_ci95_db"], 0., atol=1e-12)
    assert got["acceptance"] == "pass"
    injected = baseline.copy()
    injected[:, :, 1] *= 10**(.3)
    bad = rr.join_ensemble_summary(injected, baseline, bootstrap_draws=200, seed=9, threshold_db=1.)
    np.testing.assert_allclose(bad["step_excess_db"], 3.)
    assert bad["acceptance"] == "fail"
    assert np.any(np.asarray(got["ism_independent_null_ci95_db"]) != 0)


def test_join_zero_phase_windows_remove_filter_delay():
    fs = 24000
    x = np.zeros(fs)
    middle = fs//2
    x[middle] = 1.
    # Symmetric response about the impulse: causal filtering would put nearly
    # all energy after the centre, especially in the low band.
    energy = rr.join_window_energies(x, fs, middle, bands=(125,), window_s=(.020,))
    assert energy[0, 0] == pytest.approx(energy[0, 1], rel=.02)
    with pytest.raises(ValueError, match="window"):
        rr.join_window_energies(x, fs, 2, bands=(125,), window_s=(.020,))


def test_tail_direct_normalization_cancels_common_channel_gain():
    fs = 24000
    pulse = signal.firwin(63, .4)
    x = np.zeros(fs)
    onset = 100
    x[onset:onset+len(pulse)] = pulse
    x[onset+4000:onset+4000+len(pulse)] = pulse*.5
    flat = rr.direct_normalized_tail(x, fs, onset+31, duration_s=.6, noise_power=0.)
    gain = rr.direct_normalized_tail(7*x, fs, onset+31, duration_s=.6, noise_power=0.)
    np.testing.assert_allclose(flat["tail_to_direct_db"], gain["tail_to_direct_db"], atol=1e-9)
    # All energy of the isolated, identical direct/tail pulses must cancel per
    # octave. Sum-to-one tail normalization would produce a spectral fraction.
    np.testing.assert_allclose(flat["tail_to_direct_db"], 20*np.log10(.5), atol=.02)


def test_early_peaks_exclude_tail_and_compare_density_matched_null():
    fs = 24000
    x = np.zeros(fs)
    x[100] = 1.
    x[[220, 340, 460]] = [.5, .4, .3]
    kwargs = dict(fs=fs, measured_onset=100, simulated_onset=100, cutoff_s=.020,
                  tolerance_s=.002, null_draws=100, seed=11)
    original = rr.early_peak_diagnostic(x, x, **kwargs)
    changed = x.copy()
    changed[2000:4000:30] = 100.  # Must not influence peak floor or counts.
    result = rr.early_peak_diagnostic(changed, x, **kwargs)
    assert result == original
    assert result["measured_count"] == result["simulated_count"] == 3
    assert result["matched_count"] == 3
    assert result["matching_loss_s"] == 0.
    assert len(result["null_matching_loss_s"]) == 100
    assert np.median(result["null_matching_loss_s"]) > 0.
    assert result["null_peak_count"] == result["simulated_count"]


def test_split_filter_skips_audit_before_any_dataset_open(monkeypatch, tmp_path):
    bench = load_bench()
    opened = []
    def forbidden(*args, **kwargs):
        opened.append(args)
        raise AssertionError("locked audit opened")
    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=forbidden))
    with pytest.raises(RuntimeError, match="retired.*HDF5"):
        bench._load_pairs(tmp_path, 48000, allowed_splits=("fit_absorption", "validation_only"))
    assert opened == []


def load_corrected():
    path = Path(__file__).resolve().parents[2] / "validation/bench_room_recheck/corrected.py"
    spec = importlib.util.spec_from_file_location("wp8_corrected_bench", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("run")
    sys.modules["run"] = load_bench()
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["run"]
        else:
            sys.modules["run"] = previous
    return module


def test_registration_tamper_fails_before_parsing_or_data_reads(tmp_path):
    bench = load_corrected()
    registration = tmp_path/"preregistration.yaml"
    digest = tmp_path/"prereg.sha256"
    registration.write_text("this is not the registered content")
    digest.write_text("0"*64)
    with pytest.raises(ValueError, match="hash mismatch"):
        bench.read_registration(registration, digest)


def test_registration_cannot_enable_locked_audit(tmp_path):
    import hashlib
    bench = load_corrected()
    registration = tmp_path/"preregistration.yaml"
    digest = tmp_path/"prereg.sha256"
    registration.write_text("protocol: sensorsim-room-wp8-review/1\n"
                            "data:\n  allowed_splits: [locked_audit]\n")
    digest.write_text(hashlib.sha256(registration.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="only fit and validation"):
        bench.read_registration(registration, digest)


def test_early_peak_cutoff_sample_does_not_set_peak_threshold():
    fs, onset = 24000, 100
    x = np.zeros(fs)
    x[onset] = 1.
    x[onset+240] = .5
    y = x.copy()
    y[onset+480] = 1000.  # Precisely at the exclusive cutoff.
    kwargs = dict(fs=fs, measured_onset=onset, simulated_onset=onset, cutoff_s=.02,
                  tolerance_s=.002, null_draws=10, seed=11)
    assert rr.early_peak_diagnostic(x, x, **kwargs) == rr.early_peak_diagnostic(y, x, **kwargs)
