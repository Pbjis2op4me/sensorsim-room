"""Independent reviewer proxy controls, retained for the final WP3 acceptance."""
import hashlib
import numpy as np
from scipy import signal
from sensorsim import edge, iot_mic, nn
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import synthesize

FS = 16000
ROOM = RoomModel(shoebox_dims_m=(6, 5, 3), late_t60_s=(.3,)*7)
RX = (4., 3., 1.5)
POS = ((1., 1., 1.), (2., 2., 1.), (3., 1., 1.5))
PROFILE = iot_mic.PROFILES['pi5_inmp441/1']


def h(x):
    return hashlib.sha256(np.asarray(x, '<f8').tobytes()).hexdigest()


def stats120(windows):
    f = edge.Features(nn._feature_cfg(nn.DEFAULTS))(np.asarray(windows))[:, 0]
    return np.concatenate([f.mean(2), f.std(2), f.max(2)], axis=1)


def neutral_audio(seed, g):
    rng = np.random.default_rng([seed, g]); t = np.arange(FS)/FS
    if g % 3 == 0:
        return rng.standard_normal(FS)
    if g % 3 == 1:
        return np.sin(2*np.pi*(400+g*31)*t)
    return rng.standard_normal(FS)*np.exp(-t/.1)


def bed_analog(seed, spl_db=35., seconds=10):
    """Low-passed white room tone at a declared RMS SPL (peak-1 sine = 94 dB convention)."""
    rng = np.random.default_rng([seed, 4242])
    x = signal.sosfilt(signal.butter(2, 500, fs=FS, output='sos'), rng.standard_normal(seconds*FS))
    x = x/np.sqrt(np.mean(x*x))
    return x*10**((spl_db-94)/20)/np.sqrt(2)  # RMS -> peak-sine reference


def rir(position, seed):
    return synthesize(ROOM, position, RX, FS, seed=seed).samples


def truth_performance(kind, k, fs=FS):
    """Copied formula: benchmarks/2026-10-07/room_walkthrough/truth.py:45-52 (project's own TRUTH stand-in)."""
    r = np.random.default_rng(100 + k); dur = r.uniform(.9, 1.4); t = np.arange(int(dur*fs))/fs
    env = np.minimum(1, t/r.uniform(.01, .05)) * np.exp(-t*r.uniform(.5, 2.5))
    if kind == "doorbell": x = np.sin(2*np.pi*880*(1+r.uniform(-.01,.01))*t) + .5*np.sin(2*np.pi*1320*(1+r.uniform(-.01,.01))*t)
    if kind == "kettle":   x = np.sin(2*np.pi*(2400+r.uniform(-30,30))*t + .3*np.sin(2*np.pi*5*t))
    if kind == "knock":    x = r.normal(size=t.size) * (np.sin(2*np.pi*3*t) > .95)
    return .3*r.uniform(.5, 1)*env*x/np.max(np.abs(x))


def c2st(a, b, seed):
    from sensorsim.negatives.acceptance import symmetry
    z = np.zeros((len(a), 3))
    r = symmetry(z, z, a, b, np.arange(len(a)), seed=seed, bootstrap_replicates=1000)
    return {'auc': r['auc'], 'auc_ci95': r['auc_ci95'], 'decision': r['structural_result']}
