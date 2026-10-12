"""Independent, analytic door panels for proxy regression, not measured ranges."""
import numpy as np
from scipy import signal

FS = 16000
# Frequencies, relative amplitudes, amplitude e-folding seconds: test assumptions.
PANELS = (
    ((173, .5, .028), (437, 1., .042), (923, .3, .017), (2347, .1, .006)),
    ((227, .4, .021), (613, 1., .046), (1187, .25, .023), (2879, .15, .005)),
    ((139, .6, .049), (557, 1., .035), (1571, .2, .011), (3253, .08, .008)),
)


def door(panel=0, seed=801, gap=.29):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1e-5, FS*4)
    t = np.arange(round(.6*FS))/FS
    for onset in (.55, .55+gap, .55+2*gap):
        hit = sum(a*np.cos(2*np.pi*f*t+rng.uniform(-.08, .08))*np.exp(-t/tau)
                  for f, a, tau in PANELS[panel])
        hit *= 1-np.exp(-t/.0003)
        hit += .15*rng.normal(size=len(t))*np.exp(-t/.001)
        hit *= .22/max(abs(hit))
        start = round(onset*FS)
        x[start:start+len(hit)] += hit
    return x


def clap(seed=804):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1e-5, FS*3)
    n = round(.002*FS)
    hit = rng.normal(size=n)*np.hanning(n)
    x[FS//2:FS//2+n] += .3*hit/max(abs(hit))
    return x


def wet(x, seed=805):
    rng = np.random.default_rng(seed)
    t = np.arange(FS)/FS
    h = rng.normal(size=FS)*np.exp(-np.log(1000)*t/.55)
    h *= 1/np.linalg.norm(h)
    h[0] += 1
    y = signal.fftconvolve(x, h)[:len(x)]
    return y*.25/max(abs(y))


def meta(label='knock', group='panel'):
    return dict(label=label, source_group=group, object_id='independent-panel',
                session='final-session', capture_route='near_dry', evidence_label='proxy')


def reference(x=None):
    return dict(samples=clap() if x is None else x, fs=FS, session='final-session', evidence_label='proxy')
