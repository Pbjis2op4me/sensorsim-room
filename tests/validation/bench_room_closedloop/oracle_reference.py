"""Privileged evaluator-only reference; never a reconstruction timing adapter."""
import numpy as np
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft
from scipy.io import wavfile
from sensorsim.acoustics.room.session_sim import device_sos


def conditional_wiener_reference(assembled, take):
    """Conditional LTI Wiener oracle: truth power only, never source phase in inverse.

    Uses oracle clock/offset and the REAL captured take; extraneous events and
    quantization remain. Device transfer includes recording gain and mic filter.
    This evaluator's privileged inputs never enter run_pipeline.
    """
    name, fs = take["file"], assembled.kit.fs_hz
    true_source = assembled.truth.dry_sources[name]
    h = assembled.truth.rirs[f"{take['speaker']}:{take['mic']}"]
    h = assembled.impairments.recording_gain*signal.sosfilt(device_sos(fs, assembled.impairments, speaker=False), h)
    _, capture = wavfile.read(assembled.capture/name)
    captured = capture.astype(float)/32768
    timing = assembled.timing[name]
    length = len(true_source)+len(h)-1
    count = round(length*timing["clock_ratio"])
    captured = captured[timing["lead_samples"]:timing["lead_samples"]+count]
    captured = signal.resample(captured, length)
    n = next_fast_len(length+len(h)-1)
    H, Y = rfft(h, n), rfft(captured, n)
    power = np.abs(rfft(true_source, n))**2  # only the power, not phase
    noise_psd = n*(assembled.truth.noise_rms**2+1/(6*32768**2))
    regularizer = noise_psd/np.maximum(power, 1e-20)
    inverse = np.conj(H)/np.maximum(np.abs(H)**2+regularizer, 1e-20)
    return irfft(Y*inverse, n)[:length]


