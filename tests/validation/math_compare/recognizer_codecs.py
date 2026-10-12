"""Recognizer front end and simulated-microphone codecs against open-source equivalents.

Not part of the test suite: it needs librosa and torchaudio (and `audioop`, which is in the Python
standard library up to 3.12). Run one section or all:

    pip install librosa torchaudio      # plus this package; Python 3.12 for audioop
    python tests/validation/math_compare/recognizer_codecs.py [mel|logmel|resample|adpcm|mulaw|batchnorm|eq]

Results and verdicts are summarized in MATH.md. Every comparison runs the other library's code locally.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
FS = 16000


def _table(rows, header):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def mel():
    """Mel filter matrices: ours (HTK scale, no area normalization) vs librosa and torchaudio in the same mode."""
    import librosa
    from sensorsim.edge import mel_filterbank
    rows = []
    for n_fft, n_mels, fmin, fmax in ((512, 40, 60., 7600.), (1024, 64, 0., 8000.), (400, 80, 20., 7800.)):
        ours = mel_filterbank(FS, n_fft, n_mels, fmin, fmax).astype(np.float64)
        lib = librosa.filters.mel(sr=FS, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True, norm=None)
        row = [f"n_fft {n_fft}, {n_mels} bands, {fmin:g}-{fmax:g} Hz", f"{np.max(np.abs(ours - lib)):.1e}"]
        try:
            import torch, torchaudio
            ta = torchaudio.functional.melscale_fbanks(n_fft//2 + 1, fmin, fmax, n_mels, FS, norm=None, mel_scale="htk")
            row.append(f"{np.max(np.abs(ours - ta.numpy().T)):.1e}")
        except Exception as error:                       # torchaudio and torch versions must match
            row.append(f"unavailable ({type(error).__name__})")
        rows.append(row)
    print(f"  librosa {librosa.__version__}")
    _table(rows, ["filterbank", "max |ours - librosa|", "max |ours - torchaudio|"])


def logmel():
    """Spectrogram conventions: our framing and window vs librosa's (centred frames, periodic Hann)."""
    import librosa
    from sensorsim.edge import Features
    cfg = {"fs": FS, "window_s": 1.0, "n_fft": 512, "hop": 160, "n_mels": 40, "fmin_hz": 60., "fmax_hz": 7600.}
    rng = np.random.default_rng(0)
    t = np.arange(FS)/FS
    x = (0.3*np.sin(2*np.pi*440*t) + 0.1*np.sin(2*np.pi*2500*t) + 0.05*rng.standard_normal(FS)).astype(np.float32)
    ours = Features(cfg)(x)[0, 0]
    rows = []
    for name, kwargs in (("same conventions (center=False, symmetric Hann)", dict(center=False, window=np.hanning(512))),
                         ("librosa defaults (center=True, periodic Hann)", dict(center=True, window="hann"))):
        m = librosa.feature.melspectrogram(y=x, sr=FS, n_fft=512, hop_length=160, power=2.0, htk=True, norm=None,
                                           fmin=60., fmax=7600., n_mels=40, **kwargs)
        lm = np.log(m + 1e-10)
        lm -= lm.mean()
        frames = min(lm.shape[1], ours.shape[1])
        offset = 0 if not kwargs["center"] else 0
        d = np.abs(lm[:, offset:offset + frames] - ours[:, :frames])
        rows.append([name, lm.shape[1], ours.shape[1], f"{d.max():.2e}", f"{d.mean():.2e}"])
    _table(rows, ["librosa setting", "frames librosa", "frames ours", "max |diff| (ln units)", "mean |diff|"])


def resample():
    """48 -> 16 kHz: our Kaiser windowed-sinc vs scipy.signal.resample_poly; tones in band and above Nyquist."""
    from scipy.signal import resample_poly
    from sensorsim.edge import resample as ours_resample
    fs_in, n = 48000, 48000
    t = np.arange(n)/fs_in
    rows = []
    for f in (100., 1000., 4000., 7000., 7600., 8500., 10000., 16000., 23000.):
        x = np.sin(2*np.pi*f*t)
        a, b = ours_resample(x.astype(np.float32), 1, 3).astype(float), resample_poly(x, 1, 3)
        core = slice(2000, len(a) - 2000)
        rms = lambda y: np.sqrt(np.mean(y[core]**2))/np.sqrt(0.5)
        if f < 8000:
            expect = np.sin(2*np.pi*f*np.arange(len(a))/16000)
            err = lambda y: np.max(np.abs(y[core] - expect[core]))
            rows.append([f"{f:g} Hz", f"{20*np.log10(rms(a)):+.3f} dB", f"{20*np.log10(rms(b)):+.3f} dB",
                         f"{err(a):.1e}", f"{err(b):.1e}"])
        else:
            rows.append([f"{f:g} Hz (aliases)", f"{20*np.log10(rms(a)):+.1f} dB", f"{20*np.log10(rms(b)):+.1f} dB", "", ""])
    _table(rows, ["input tone", "ours level", "scipy level", "ours max error vs exact", "scipy max error vs exact"])


def adpcm():
    """Project ADPCM-like stress quantizer: proxy SNR/spectral error, not bit-exact IMA."""
    from sensorsim.iot_mic import adpcm_like
    t = np.arange(16000)/8000
    signals = {"proxy speech-like": .3*np.sin(2*np.pi*300*t)*np.sin(2*np.pi*3*t) + .1*np.sin(2*np.pi*1100*t),
               "proxy tonal": .5*np.sin(2*np.pi*313*t),
               "proxy knock-like": .8*np.exp(-50*t)*np.sin(2*np.pi*900*t)}
    rows = []
    for name, x in signals.items():
        y = adpcm_like(x)
        snr = 10*np.log10(np.sum(x*x)/np.sum((y-x)**2))
        magnitude = np.abs(np.fft.rfft(x))
        spectral = np.linalg.norm(np.abs(np.fft.rfft(y))-magnitude)/np.linalg.norm(magnitude)
        rows.append([name, len(x), f"{snr:.3f}", f"{spectral:.6f}"])
    _table(rows, ["signal (proxy)", "samples", "SNR (dB)", "relative magnitude spectral error"])


def mulaw():
    """mu-law: our continuous mu=255 curve on 255 levels vs G.711's segmented law (audioop) and torchaudio."""
    import audioop
    from sensorsim.iot_mic import mulaw as ours_mulaw
    t = np.arange(8000)/8000
    rows = []
    for level_db in (0., -10., -20., -30., -40., -50.):
        x = 10**(level_db/20)*0.99*np.sin(2*np.pi*313*t)
        pcm = np.round(x*32767).astype(np.int16)
        g711 = np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(pcm.tobytes(), 2), 2), np.int16)/32767
        ours = ours_mulaw(pcm/32767)
        snr = lambda y: 10*np.log10(np.sum((pcm/32767)**2)/np.sum((y - pcm/32767)**2))
        row = [f"{level_db:g} dBFS", f"{snr(ours):.1f}", f"{snr(g711):.1f}", f"{np.max(np.abs(ours - g711))*32767:.0f}"]
        try:
            import torch, torchaudio
            enc = torchaudio.functional.mu_law_encoding(torch.from_numpy(pcm/32767), 256)
            ta = torchaudio.functional.mu_law_decoding(enc, 256).numpy()
            row.append(f"{snr(ta):.1f}")
        except Exception as error:
            row.append(f"unavailable ({type(error).__name__})")
        rows.append(row)
    _table(rows, ["sine level", "SNR ours (dB)", "SNR G.711 (dB)", "max |ours - G.711| (LSB)", "SNR torchaudio mu-law (dB)"])


def batchnorm():
    """BatchNorm folding in the export vs torch.nn.utils.fusion.fuse_conv_bn_eval."""
    import tempfile
    from pathlib import Path
    import torch
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    from sensorsim import nn
    torch.manual_seed(0)
    model = nn.build_model(3, 8, 2)
    for m in model.modules():                                  # give BatchNorm non-trivial statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5); m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5); m.bias.data.uniform_(-0.2, 0.2)
    model.eval()
    cfg = {**nn.DEFAULTS, "width": 8, "blocks": 2}
    features = np.random.default_rng(0).standard_normal((4, 1, 40, 98)).astype(np.float32)
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)/"m.npz"
        report = nn.export(model, ["a", "b", "c"], cfg, out, features, {"best_val_mean_ap": 0., "history": [{}]})
        arrays = np.load(out)
        fused = fuse_conv_bn_eval(model.conv0, model.conv0_bn)
        dw = fuse_conv_bn_eval(model.blocks[0].dw, model.blocks[0].dw_bn)
        print(f"  conv0 weights max |ours - torch| {np.max(np.abs(arrays['conv0.w'] - fused.weight.detach().numpy())):.1e}, "
              f"bias {np.max(np.abs(arrays['conv0.b'] - fused.bias.detach().numpy())):.1e}")
        print(f"  block 0 depthwise weights {np.max(np.abs(arrays['b0.dw.w'] - dw.weight.detach().numpy()[:, 0])):.1e}, "
              f"bias {np.max(np.abs(arrays['b0.dw.b'] - dw.bias.detach().numpy())):.1e}")
        print(f"  end-to-end numpy runtime vs PyTorch: {report['parity_max_abs_diff']:.1e}")


def eq():
    """RBJ cookbook identities: peaking gain at f0 and unity at DC/Nyquist; shelf asymptotes and half gain at f0."""
    from scipy.signal import sosfreqz
    from sensorsim.iot_mic import high_shelf, peaking
    rows = []
    for f0, g, q in ((1000., 9., 2.), (3400., -6., 0.7), (5200., 12., 4.)):
        w, h = sosfreqz(np.atleast_2d(peaking(f0, g, q, FS)), worN=[0., f0, FS/2], fs=FS)
        rows.append([f"peaking {f0:g} Hz {g:+g} dB Q {q:g}", f"{20*np.log10(abs(h[1])):+.6f} (expect {g:+g})",
                     f"{20*np.log10(abs(h[0])):+.1e}", f"{20*np.log10(abs(h[2])):+.1e}"])
    for f0, g in ((3000., -6.), (4000., -24.), (2000., 3.)):
        w, h = sosfreqz(np.atleast_2d(high_shelf(f0, g, FS)), worN=[0., f0, FS/2], fs=FS)
        rows.append([f"high shelf {f0:g} Hz {g:+g} dB", f"{20*np.log10(abs(h[1])):+.6f} (expect {g/2:+g})",
                     f"{20*np.log10(abs(h[0])):+.1e}", f"{20*np.log10(abs(h[2])):+.4f} (expect {g:+g})"])
    _table(rows, ["filter", "gain at f0 (dB)", "gain at DC (dB)", "gain at Nyquist (dB)"])


SECTIONS = {"mel": mel, "logmel": logmel, "resample": resample, "adpcm": adpcm, "mulaw": mulaw,
            "batchnorm": batchnorm, "eq": eq}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        print(f"== {name}: {(SECTIONS[name].__doc__ or '').strip().splitlines()[0]}")
        SECTIONS[name]()
