#!/usr/bin/env python3
"""Check Pi MEMS capture format, level and quiet-floor evidence.

Run ``python tools/capture_check.py --self-test`` for deterministic proxy checks. For a real
capture, supply a mono device WAV, quiet durations at its beginning and end, and (optionally) a
separate 1 kHz calibration-tone WAV with its measured SPL. This tool does not configure I2S or
claim that hardware clocking has been verified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import wave

import numpy as np

from sensorsim import edge, iot_mic

PI_PROFILE_IDS = ("pi5_inmp441/1", "pi5_ics43434/1", "pi5_sph0645/1")


def s32_to_s16(samples: np.ndarray) -> np.ndarray:
    """Truncate signed left-justified S32_LE / 24-in-32 words to S16_LE."""
    samples = np.asarray(samples)
    if samples.dtype != np.int32:
        raise TypeError("S32_LE samples must use numpy.int32")
    return (samples.astype(np.int64) >> 16).astype(np.int16)


def _read_wave(path: str | Path) -> tuple[int, int, int, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        fs = source.getframerate()
        raw = source.readframes(source.getnframes())
    if width == 1:
        pcm = np.frombuffer(raw, np.uint8).astype(np.int16) - 128
    elif width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").copy()
    elif width == 3:
        packed = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        pcm24 = (packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8)
                 | (packed[:, 2].astype(np.int32) << 16))
        pcm = (pcm24 << 8) >> 8
    elif width == 4:
        pcm = np.frombuffer(raw, dtype="<i4").copy()
    else:
        raise ValueError(f"unsupported PCM width: {width} bytes")
    return fs, channels, width, pcm.reshape(-1, channels)


def _float_audio(path: str | Path, fs: int) -> np.ndarray:
    return edge.read_wav(str(path), fs)


def fit_sine_peak_dbfs(samples: np.ndarray, fs: int, frequency_hz: float = 1000.0) -> float:
    """Least-squares sinusoid peak level, relative to a full-scale sine peak."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size < int(fs/frequency_hz):
        raise ValueError("tone must be mono and at least one period long")
    x = x - x.mean()
    t = np.arange(x.size, dtype=np.float64)/fs
    basis = np.column_stack((np.sin(2*np.pi*frequency_hz*t), np.cos(2*np.pi*frequency_hz*t)))
    coefficients, *_ = np.linalg.lstsq(basis, x, rcond=None)
    peak = float(np.hypot(*coefficients))
    return float(20*np.log10(max(peak, 1e-15)))


def _dbfs(value: float) -> float | None:
    return None if value <= 0 else float(20*np.log10(value*np.sqrt(2)))


def _feature_parity(path: Path, audio: np.ndarray, fs: int) -> dict:
    cfg = {"fs": fs, "window_s": 1.0, "n_fft": 512, "hop": 160, "n_mels": 40,
           "fmin_hz": 60.0, "fmax_hz": min(7600.0, fs*0.475)}
    features = edge.Features(cfg)
    n = features.n
    if audio.size < n:
        return {"pass": False, "status": "fail: capture needs at least one feature window"}
    device_samples = np.asarray(audio[:n], dtype=np.float32)
    # Independent WAV decoders: stdlib/edge on device, scipy/recognizer in Corpus.
    # This checks decode/feature parity, not the physical I2S bit conversion.
    from sensorsim.recognizer import read_audio
    training_samples = np.asarray(read_audio(path, fs)[:n], dtype=np.float32)
    difference = float(np.max(np.abs(features(device_samples) - features(training_samples))))
    return {"max_abs_feature_difference": difference, "threshold": 1e-5,
            "pass": difference <= 1e-5, "sample_rate_hz": fs}


def inspect_capture(path: str | Path, *, fs: int = 16000, pre_quiet_s: float | None = None,
                    post_quiet_s: float | None = None, tone_wav: str | Path | None = None,
                    tone_spl_db: float | None = None, tone_frequency_hz: float = 1000.0,
                    sensitivity_dbfs_at_94db_spl: float | None = None,
                    profile_id: str = "pi5_inmp441/1",
                    evidence_label: str = "measured") -> dict:
    if evidence_label not in ("measured", "proxy result"):
        raise ValueError("evidence_label must be 'measured' or 'proxy result'")
    if profile_id not in PI_PROFILE_IDS:
        raise ValueError(f"profile_id must be one of {PI_PROFILE_IDS}")
    profile = iot_mic.PROFILES[profile_id]
    sensitivity_dbfs_at_94db_spl = (profile.sensitivity_dbfs_at_94db_spl
                                    if sensitivity_dbfs_at_94db_spl is None
                                    else sensitivity_dbfs_at_94db_spl)
    path = Path(path)
    native_fs, channels, width, pcm = _read_wave(path)
    if fs != 16000 or native_fs != fs or channels != 1 or width != 2:
        return {"evidence_label": evidence_label, "profile_id": profile_id, "file": path.name,
                "native_sample_rate_hz": native_fs, "analysis_sample_rate_hz": fs,
                "channels": channels, "sample_width_bytes": width,
                "s16_capture_format": "fail: expected native 16000 Hz mono S16 PCM",
                "capture_quality_status": "fail", "feature_parity": {"status": "not_checked"},
                "pcm_conversion": {"status": "not_checked", "reason": "invalid application format"}}
    audio = _float_audio(path, fs)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    result = {"evidence_label": evidence_label, "profile_id": profile_id, "file": path.name,
              "native_sample_rate_hz": native_fs,
              "analysis_sample_rate_hz": fs, "channels": channels, "sample_width_bytes": width,
              "samples": int(pcm.shape[0]), "peak_fs": peak,
              "below_0_99_fs": bool(peak < 0.99),
              "feature_parity": _feature_parity(path, audio, fs)}
    result["s16_capture_format"] = "pass"
    result["pcm_conversion"] = {
        "status": "not_checked",
        "reason": "A single S16 WAV cannot verify physical 24/32-to-16 conversion. "
                  "Self-test checks software arithmetic only; compare native and application "
                  "captures of a digital test pattern on the Pi to verify that path."}

    if pre_quiet_s is not None or post_quiet_s is not None:
        if pre_quiet_s is None or post_quiet_s is None or pre_quiet_s <= 0 or post_quiet_s <= 0:
            raise ValueError("supply both positive --pre-quiet-s and --post-quiet-s durations")
        pre_n, post_n = round(pre_quiet_s*fs), round(post_quiet_s*fs)
        if pre_n + post_n > audio.size:
            raise ValueError("pre/post quiet intervals exceed the recording duration")
        pre, post = audio[:pre_n], audio[-post_n:]
        pre_level = _dbfs(iot_mic.a_weighted_rms(pre, fs))
        post_level = _dbfs(iot_mic.a_weighted_rms(post, fs))
        combined_level = _dbfs(iot_mic.a_weighted_rms(np.concatenate((pre, post)), fs))
        expected_floor = profile.self_noise_dbfs_a
        levels = {"pre_take_dbfs_a": pre_level, "post_take_dbfs_a": post_level,
                  "combined_dbfs_a": combined_level, "expected_profile_floor_dbfs_a": expected_floor,
                  "pre_delta_db": None if pre_level is None else pre_level-expected_floor,
                  "post_delta_db": None if post_level is None else post_level-expected_floor,
                  "combined_delta_db": None if combined_level is None else combined_level-expected_floor,
                  "within_3db_of_profile_floor": (combined_level is not None
                                                    and abs(combined_level-expected_floor) <= 3.0),
                  "pre_seconds": pre_quiet_s, "post_seconds": post_quiet_s}
        result["quiet_floor"] = {**levels, "status": evidence_label + " floor; not a false-alarm/hour estimate",
                                          "reference": "AES17 full-scale sine; IEC A weighting"}

    if tone_wav is not None:
        if tone_spl_db is None:
            raise ValueError("--tone-spl-db is required with --tone-wav")
        tone_rate, tone_channels, tone_width, _ = _read_wave(tone_wav)
        if (tone_rate, tone_channels, tone_width) != (fs, 1, 2):
            raise ValueError("calibration tone must be native 16000 Hz mono S16 PCM")
        tone = _float_audio(tone_wav, fs)
        measured = fit_sine_peak_dbfs(tone, fs, tone_frequency_hz)
        expected = sensitivity_dbfs_at_94db_spl + tone_spl_db - 94.0
        tolerance = 1.0 if profile_id == "pi5_ics43434/1" else 3.0
        error = measured-expected
        shift = round(error/(20*np.log10(2)))
        result["calibration_tone"] = {"evidence_label": evidence_label, "frequency_hz": tone_frequency_hz,
                                      "spl_db": tone_spl_db, "measured_peak_dbfs": measured,
                                      "expected_peak_dbfs": expected, "error_db": measured-expected,
                                      "sensitivity_tolerance_db": tolerance,
                                      "tolerance_source": ("TDK ICS-43434 specification" if profile_id == "pi5_ics43434/1"
                                                           else "InvenSense DS-INMP441-00 Rev 1.1" if profile_id == "pi5_inmp441/1"
                                                           else "assumed; SPH sensitivity requires calibration"),
                                      "pass_within_sensitivity_tolerance": abs(error) <= tolerance,
                                      "possible_bit_shift_or_gain_error": bool(shift != 0 and abs(error-shift*20*np.log10(2)) < 1.)}
    else:
        result["calibration_tone"] = "pending: provide a known-level 1 kHz recording to check sensitivity and bit alignment"
    required = [result["s16_capture_format"] == "pass",
                result["below_0_99_fs"], result["feature_parity"]["pass"]]
    pending = ["physical 24/32-to-16 capture conversion"]
    if "quiet_floor" in result:
        required.append(result["quiet_floor"]["within_3db_of_profile_floor"])
    else:
        pending.append("pre/post-take quiet floor")
    if isinstance(result["calibration_tone"], dict):
        required.append(result["calibration_tone"]["pass_within_sensitivity_tolerance"])
    else:
        pending.append("known-level 1 kHz calibration tone")
    result["pending_checks"] = pending
    result["capture_quality_status"] = ("fail" if not all(required) else "inconclusive" if pending else "pass")
    result["wp5b_per_band_smd_and_c2st"] = "pending: requires real and simulated floor windows"
    result["wp5c_heldout_recall_and_audit_ratio"] = "pending: requires held-out real takes and paired models"
    return result


def self_test(seed: int = 20261008, fs: int = 16000) -> dict:
    """Run generated proxy conversion and feature-path checks without Pi hardware."""
    test_words = np.array([np.iinfo(np.int32).min, -32768 << 16, -1, 0, 1,
                           32767 << 16, np.iinfo(np.int32).max], dtype=np.int32)
    expected_words = np.array([-32768, -32768, -1, 0, 0, 32767, 32767], dtype=np.int16)
    converted_extremes = s32_to_s16(test_words)
    if not np.array_equal(converted_extremes, expected_words):
        raise AssertionError(f"S32 extreme conversion mismatch: {converted_extremes.tolist()}")

    duration = 2.0
    t = np.arange(round(duration*fs))/fs
    source = 0.25*np.sin(2*np.pi*1000*t)
    source_s32 = np.rint(source*(2**31-1)).astype(np.int32)
    tone_s16 = s32_to_s16(source_s32)
    tone_dbfs = fit_sine_peak_dbfs(tone_s16.astype(np.float64)/32768.0, fs)
    expected_dbfs = float(20*np.log10(0.25))
    tone_error_db = tone_dbfs-expected_dbfs
    if abs(tone_error_db) > 0.01:
        raise AssertionError(f"S32-to-S16 tone conversion error {tone_error_db:.6f} dB exceeds 0.01 dB")

    profile = iot_mic.PROFILES["pi5_inmp441/1"]
    rng = np.random.default_rng(seed)
    training_audio = profile.apply(0.1*np.sin(2*np.pi*1000*t), fs, rng)
    pcm16 = np.clip(np.rint(training_audio*32768), -32768, 32767).astype("<i2")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp_dir:
        wav_path = Path(temp_dir)/"device_format.wav"
        with wave.open(str(wav_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(fs)
            output.writeframes(pcm16.tobytes())
        device_audio = _float_audio(wav_path, fs)
        if device_audio.shape != training_audio.shape:
            raise AssertionError("device WAV and training path lengths differ")
        features = edge.Features({"fs": fs, "window_s": 1.0, "n_fft": 512, "hop": 160,
                                  "n_mels": 40, "fmin_hz": 60.0, "fmax_hz": min(7600., fs*.475)})
        feature_error = float(np.max(np.abs(features(device_audio[:features.n])
                                             - features(training_audio[:features.n]))))
    if feature_error > 1e-5:
        raise AssertionError(f"device WAV feature difference {feature_error:g} exceeds 1e-5")
    return {"evidence_label": "proxy result", "seed": seed, "sample_rate_hz": fs,
            "s32_extreme_samples": converted_extremes.tolist(), "s32_extremes_pass": True,
            "tone_expected_dbfs_peak": expected_dbfs, "tone_measured_dbfs_peak": tone_dbfs,
            "tone_conversion_error_db": tone_error_db, "tone_conversion_pass": True,
            "feature_max_abs_difference": feature_error, "feature_parity_pass": True,
            "hardware_i2s_clocking": "pending; no Pi hardware used",
            "wp5b_per_band_smd_and_c2st": "pending: requires real takes",
            "wp5c_heldout_recall_and_audit_ratio": "pending: requires held-out real takes and paired models"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", help="device-format WAV to inspect")
    parser.add_argument("--self-test", action="store_true", help="run deterministic generated proxy checks")
    parser.add_argument("--fs", type=int, default=16000, help="training/device analysis rate (default: 16000)")
    parser.add_argument("--pre-quiet-s", type=float, help="quiet seconds at start of the recording")
    parser.add_argument("--post-quiet-s", type=float, help="quiet seconds at end of the recording")
    parser.add_argument("--tone-wav", help="separate known-level 1 kHz calibration tone recording")
    parser.add_argument("--tone-spl-db", type=float, help="measured acoustic SPL of the calibration tone")
    parser.add_argument("--tone-frequency-hz", type=float, default=1000.0)
    parser.add_argument("--sensitivity-dbfs-at-94db-spl", type=float, default=None,
                        help="profile sensitivity for the tone comparison")
    parser.add_argument("--profile", choices=PI_PROFILE_IDS, default="pi5_inmp441/1",
                        help="versioned Pi MEMS profile to compare")
    parser.add_argument("--proxy", action="store_true", help="label generated or simulated WAV input as proxy")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.wav:
            parser.error("--self-test does not take a WAV path")
        report = self_test(fs=args.fs)
    elif args.wav:
        report = inspect_capture(args.wav, fs=args.fs, pre_quiet_s=args.pre_quiet_s,
                                 post_quiet_s=args.post_quiet_s, tone_wav=args.tone_wav,
                                 tone_spl_db=args.tone_spl_db,
                                 tone_frequency_hz=args.tone_frequency_hz,
                                 sensitivity_dbfs_at_94db_spl=args.sensitivity_dbfs_at_94db_spl,
                                 profile_id=args.profile,
                                 evidence_label="proxy result" if args.proxy else "measured")
    else:
        parser.error("supply a WAV path or --self-test")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
