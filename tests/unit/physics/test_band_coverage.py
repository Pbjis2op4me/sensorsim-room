"""The late reverberant tail covers DC to Nyquist at every rate: the late bands tile the spectrum
with flat summed power, rendered tails are flat across it, and the band above a complete 8 kHz
octave decays faster (stronger air absorption)."""
import numpy as np
import pytest
from scipy.signal import sosfreqz, welch

from sensorsim.acoustics.room.model import RoomModel, uniform_bands
from sensorsim.acoustics.room.synth import late_bands, synthesize


@pytest.mark.parametrize("fs", [8000, 16000, 22050, 32000, 48000])
def test_late_bands_tile_dc_to_nyquist_with_flat_power(fs):
    f = np.geomspace(20, 0.95*fs/2, 400)
    power = sum(np.abs(sosfreqz(b.sos, worN=f, fs=fs)[1])**2 for b in late_bands(fs))
    assert np.all(np.abs(10*np.log10(power)) < 1.5)                 # skirts cross at -3 dB; worst -1.03 dB


def test_band_kinds_follow_the_sample_rate():
    kinds = {fs: [b.kind for b in late_bands(fs)] for fs in (16000, 48000)}
    assert kinds[16000][0] == "lowpass" and kinds[16000][-1] == "highpass"         # 8 kHz octave cut by Nyquist
    assert kinds[48000][0] == "lowpass" and kinds[48000][-2:] == ["bandpass", "extension"]


@pytest.mark.parametrize("fs, top", [(16000, 7800.), (48000, 11000.)])
def test_rendered_tails_are_flat_from_low_to_high_frequencies(fs, top):
    model = RoomModel(shoebox_dims_m=(5.4, 4.2, 2.7), absorption=uniform_bands(.2), late_t60_s=(.5,)*7)
    psd = 0.
    for seed in range(4):
        rir = synthesize(model, (1.2, 1.1, 1.3), (4.1, 3.0, 1.4), fs, seed=seed)
        start = rir.metadata["latency_samples"] + int((rir.metadata["mixing_time_s"] + rir.metadata["crossfade_s"])*fs)
        f, p = welch(rir.samples[start:start + int(.15*fs)].astype(float), fs,
                     nperseg=int(2**np.ceil(np.log2(fs/30))))
        psd = psd + p
    ref = np.median(psd[(f > 300) & (f < 3000)])
    for lo, hi in ((30, 90), (90, 180), (5700, top)):
        level = 10*np.log10(np.mean(psd[(f >= lo) & (f < hi)])/ref)
        assert abs(level) < 2.5, (fs, lo, hi, level)


def test_the_band_above_a_complete_8k_octave_decays_faster():
    rir = synthesize(RoomModel(shoebox_dims_m=(5.4, 4.2, 2.7), absorption=uniform_bands(.2)), (1.2, 1.1, 1.3),
                     (4.1, 3.0, 1.4), 48000)
    bands = rir.metadata["late_bands"]
    assert bands[-1]["kind"] == "extension" and bands[-1]["air_hz"] == 16000.
    assert bands[-1]["t60_s"] < bands[-2]["t60_s"]
