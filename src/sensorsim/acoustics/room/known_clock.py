"""Known broadband playback timing; no learned or external signal assets."""
from __future__ import annotations
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import minimize_scalar
from scipy.signal import correlate


def fit_clock(y, x, fs, max_ppm):
    """Whole-record resampling search, retaining all early paths in one FIR.

    The objective is FIR concentration, not a succession of independently chosen
    echoes. A narrow isolated maximum is required. A sweep with a drift/delay
    ridge, periodic playback, changing rooms or insufficient support must abstain.
    The returned delay is a chain offset, never an absolute propagation estimate.
    """
    ref = y if y.ndim == 1 else y[:, 0]
    if len(x) < fs or len(ref) <= len(x):
        raise ValueError('known signal needs >=1 second and recorded decay')
    ac = correlate(x-x.mean(), x-x.mean(), mode='full', method='fft')[len(x)-1:]
    if ac[0] <= 0 or np.max(abs(ac[max(2, round(.002*fs)):round(.2*fs)]))/ac[0] > .85:
        raise ValueError('known signal is silent or periodic: clock drift unidentifiable')
    # Require overlapping broadband evidence at separated times. A monotonic ESS
    # commonly trades clock error against an apparent chain delay and fails this
    # check; such a sweep needs independent timing anchors (the ESS capture path).
    spectra = []
    for part in np.array_split(x, 4):
        n = min(len(part), 4096)
        blocks = part[:len(part)//n*n].reshape(-1,n)
        spectra.append(np.mean(abs(rfft(blocks*np.hanning(n), axis=1))**2, axis=0))
    active = np.array([s > max(s)*1e-3 for s in spectra])
    if np.mean(np.all(active, axis=0)) < .08:
        raise ValueError('no separated overlapping broadband support: sweep clock/delay ambiguous')
    nfft = next_fast_len(len(ref)+len(x)-1)
    X = rfft(x, nfft); power = abs(X)**2
    inverse = X.conj()/(power+1e-6*power.max())
    coeff = spline_filter(ref, order=3)
    indices = np.arange(len(ref), dtype=float)
    def response(ppm):
        ratio=1+ppm*1e-6
        positions=indices[:int(np.floor((len(ref)-1)/ratio))+1]*ratio
        corrected=map_coordinates(coeff,[positions],order=3,mode='constant',prefilter=False)
        h=irfft(rfft(corrected,nfft)*inverse,nfft)
        return h[:max(1,len(corrected)-len(x)+1)]
    def score(ppm):
        h=response(ppm)
        # Three samples capture fractional-delay energy without rewarding just
        # one phase of one reflection. All resolvable paths contribute.
        energy=np.convolve(h*h,np.ones(3),mode='same')
        return float(np.sum(energy**2))
    step=min(10.,max_ppm/6)
    grid=np.linspace(-max_ppm,max_ppm,int(np.ceil(2*max_ppm/step))+1)
    scores=np.array([score(p) for p in grid]); j=int(np.argmax(scores))
    if j in (0,len(grid)-1):
        raise ValueError('clock maximum at declared ppm boundary; unidentifiable or outside bound')
    # Resampling creates small phase lobes. Do not assume the entire coarse
    # bracket is unimodal (especially after 30 seconds of playback).
    fine_step = min(1., 5e5/len(x))
    fine = np.linspace(grid[j-1], grid[j+1], int(np.ceil((grid[j+1]-grid[j-1])/fine_step))+1)
    fine_scores = np.array([score(p) for p in fine]); k = int(np.argmax(fine_scores))
    left, right = fine[max(0,k-1)], fine[min(len(fine)-1,k+1)]
    opt=minimize_scalar(lambda p:-score(p),bounds=(left,right),
                        method='bounded',options={'xatol':.02})
    ppm=float(opt.x); peak=-float(opt.fun)
    if fine_scores[k] > peak:
        ppm, peak = float(fine[k]), float(fine_scores[k])
    delta=max(10., 2e6/len(x))
    flank=max(score(ppm-delta),score(ppm+delta))
    if peak <= 0 or flank/peak > .98:
        raise ValueError('broad clock/delay ridge: drift unidentifiable')
    # A second separated maximum indicates a periodic source or changing transfer.
    far = abs(grid-ppm) > max(2*delta,2*step)
    if np.any(far) and max(scores[far]) > .95*peak:
        raise ValueError('multiple clock maxima: drift unidentifiable')
    ratio=1+ppm*1e-6
    positions=np.arange(int(np.floor((len(y)-1)/ratio))+1)*ratio
    corrected=(map_coordinates(y,[positions],order=3,mode='constant') if y.ndim==1 else
        np.column_stack([map_coordinates(y[:,i],[positions],order=3,mode='constant') for i in range(y.shape[1])]))
    h=response(ppm)
    # Onset threshold is used only to place a generously padded extraction window.
    onset=int(np.flatnonzero(abs(h) >= .1*max(abs(h)))[0])
    return corrected,dict(method='known-whole-response-clock/2',drift_ppm=ppm,
        capture_to_playback_ratio=ratio,total_delay_samples=float(onset),
        delay_status='acquisition plus propagation; no absolute emission anchor inferred',
        concentration=peak,flank_ratio=flank/peak,search_bound_ppm=max_ppm,
        correction='cubic resample; intercept retained')
