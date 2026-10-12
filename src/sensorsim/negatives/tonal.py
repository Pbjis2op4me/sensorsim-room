"""Training-target spectral partials and conservative tonal exclusion, proxy rules.

Use training takes only; held-out takes must never shape the negative generator.
Peak height (-30 dB), at most 16 peaks per take, 1 Hz FFT resolution, and a 15%
frequency margin are engineering assumptions, not population/recall evidence.
The separate listening/label-boundary gate remains mandatory.
"""
import numpy as np
from scipy.signal import find_peaks

EXCLUSION_FRACTION = .15  # conservative proxy margin, approximately one mel filter near 880 Hz


def target_partials(training_targets, *, fs):
    """Estimate partials from every supplied training-class take, preserving gaps.

    Accept either an iterable of mono takes or a class -> takes mapping. A caller
    reruns this unchanged on training portions of the user's recorded targets.
    Broadband targets can conservatively exclude much of the draw support.
    """
    takes = (x for xs in training_targets.values() for x in xs) if isinstance(training_targets, dict) else training_targets
    result = []
    for take in takes:
        x = np.asarray(take, float)
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or not np.any(x):
            raise ValueError('target partial estimation requires finite non-silent training takes')
        # Bound FFT memory and aggregate power over consecutive <=1 s frames.
        n = int(fs); power = np.zeros(n//2+1)
        for start in range(0, len(x), n):
            frame = x[start:start+n]
            power += abs(np.fft.rfft((frame-frame.mean())*np.hanning(len(frame)), n=n))**2
        peaks, _ = find_peaks(power, height=power.max()*1e-3, distance=3)
        peaks = peaks[np.argsort(power[peaks])[-16:]]
        result.extend(float(f) for f in peaks if 60 <= f <= .475*fs)
    if not result:
        raise ValueError('no resolved training-target partials; tonal exclusion cannot be established')
    return tuple(sorted(set(result)))


def tonal_support(family, base, fs):
    """Closed intervals of carrier partials, or the complete swept carrier band.

    Modulation sidebands/room filtering are not a label decision; review and real
    target recall testing still apply. Nyquist-folded/changing sweeps are avoided.
    """
    ratios = {'other_doorbell': [1.,2.71,5.4], 'phone_notification':[1.,1.5],
              'appliance_beeps':[1.], 'timer':[1.,3.],
              'tv_jingle':[1.,1.25,1.5,1.125,2.,2.5,3.,2.25,4.]}
    if family == 'alarm_chirp':
        return [(base, min(base+3600., .4*fs))]
    return [(min(base*r,.4*fs) if family == 'other_doorbell' else base*r,)*2
            for r in ratios[family]]


def draw_base(rng, family, fs, partials):
    partials = np.asarray(partials, float)
    if partials.ndim != 1 or not np.isfinite(partials).all() or np.any((partials <= 0)|(partials >= fs/2)):
        raise ValueError('target partials must be finite frequencies below Nyquist')
    for _ in range(1024):
        base = float(rng.uniform(650.,1000.))
        support = tonal_support(family,base,fs)
        if all(not np.any((hi >= partials*(1-EXCLUSION_FRACTION))&(lo <= partials*(1+EXCLUSION_FRACTION))) for lo,hi in support):
            return base, support
    raise ValueError(f'{family}: target partial exclusion leaves no sampled safe draw; omit this family')
