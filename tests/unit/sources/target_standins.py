"""Project-generated proxy stand-ins based on the independent w6 review.

All modal frequencies, decay constants, levels and timings below are REVIEWER ASSUMPTIONS chosen
to be plausible household-like stand-ins; they are not literature values and not measurements.
Run with PYTHONPATH pointing at the w6-targets worktree src.
"""
import json

import numpy as np
from scipy import signal

from sensorsim.sources import intake_chamber, intake_known_digital


FS = 48000


def floor_noise(n, rms, seed):
    return np.random.default_rng(seed).normal(0, rms, n)


def modal_ir(fs, modes, dur):
    t = np.arange(round(dur*fs))/fs
    return sum(a*np.sin(2*np.pi*f*t)*np.exp(-t/tau) for f, a, tau in modes)


DOOR = [(95, 1.0, .045), (180, .8, .035), (310, .6, .025), (520, .5, .02),
        (890, .35, .012), (1500, .2, .008), (2600, .1, .005)]


def knock_hit(fs, contact_ms=3., seed=0):
    """Half-sine contact force (broadband) through door modes plus a short contact-noise burst."""
    n = max(2, round(contact_ms*1e-3*fs))
    force = np.sin(np.pi*np.arange(n)/n)
    y = np.convolve(force, modal_ir(fs, DOOR, .4))
    burst = np.random.default_rng(seed).normal(0, 1, n)*np.hanning(n)
    y[:n] += .4*np.max(abs(y))*burst/np.max(abs(burst))
    return y


def place(x, ev, t0, fs):
    a = round(t0*fs)
    x[a:a+len(ev)] += ev[:len(x)-a]
    return x


def knock_take(onsets, fs=FS, lead=.5, dur=2.5, peak=10**(-12/20), floor_rms=1e-4, seed=1):
    x = np.zeros(round(dur*fs))
    ev = np.zeros_like(x)
    for k, t in enumerate(onsets):
        place(ev, knock_hit(fs, 2.+k % 3, seed+k)*(1-.12*k), lead+t, fs)
    ev *= peak/np.max(abs(ev))
    return ev+floor_noise(len(x), floor_rms, seed+100), ev


def room_rir(fs, t60, drr_db, seed=7):
    n = round(1.2*t60*fs)
    t = np.arange(n)/fs
    tail = np.random.default_rng(seed).normal(0, 1, n)*np.exp(-6.9*t/t60)
    tail[:round(.002*fs)] = 0
    h = np.zeros(n)
    h[0] = 1.
    scale = np.sqrt(10**(-drr_db/10)/np.sum(tail**2))
    return h+scale*tail


def wet(ev, fs, t60, drr_db, peak, floor_rms, seed):
    y = signal.fftconvolve(ev, room_rir(fs, t60, drr_db, seed))[:len(ev)]
    y *= peak/np.max(abs(y))
    return y+floor_noise(len(y), floor_rms, seed+5)


CHIME_DING = [(659, 1., .9), (1531, .4, .45), (2690, .2, .25), (3950, .1, .15)]
CHIME_DONG = [(523, 1., 1.), (1215, .4, .5), (2135, .2, .3), (3140, .1, .18)]


def chime_take(fs=FS, lead=.5, gap=.65, dur=12., peak=10**(-12/20), floor_rms=1e-4, seed=3):
    ev = np.zeros(round(dur*fs))
    strike = np.hanning(round(.002*fs))
    place(ev, np.convolve(strike, modal_ir(fs, CHIME_DING, 10.)), lead, fs)
    place(ev, np.convolve(strike, modal_ir(fs, CHIME_DONG, 10.)), lead+gap, fs)
    ev *= peak/np.max(abs(ev))
    return ev+floor_noise(len(ev), floor_rms, seed), ev


def kettle_take(fs=16000, floor_rms=1e-4, seed=5):
    """0-1 s quiet; switch-on click at 1.0 s; rumble rising 3->50 s; boil 50->65 s; shutoff click 65 s; boil decays."""
    r = np.random.default_rng(seed)
    dur = 75.
    n = round(dur*fs)
    t = np.arange(n)/fs
    ev = np.zeros(n)
    click = np.hanning(round(.004*fs))*r.normal(0, 1, round(.004*fs))
    place(ev, .1*click/np.max(abs(click)), 1.0, fs)
    rumble = signal.lfilter(*signal.butter(2, [80, 800], 'bandpass', fs=fs), r.normal(0, 1, n))
    for f0 in (180, 420):
        b, a = signal.iirpeak(f0, 8, fs)
        rumble += 3*signal.lfilter(b, a, r.normal(0, 1, n))
    rumble /= np.std(rumble)
    level_db = np.interp(t, [3, 50, 65, 69], [-85, -32, -30, -95])
    level_db[t < 3] = -200
    level_db[t > 69] = -200
    ev += rumble*10**(level_db/20)
    boil = signal.lfilter(*signal.butter(2, [200, 6000], 'bandpass', fs=fs), r.normal(0, 1, n))
    boil /= np.std(boil)
    boil_db = np.interp(t, [45, 50, 65, 66.5, 70], [-200, -24, -22, -40, -200])
    boil_db[t < 45] = -200
    ev += boil*10**(boil_db/20)
    place(ev, .3*click/np.max(abs(click)), 65.0, fs)
    return ev+floor_noise(n, floor_rms, seed+1), ev, fs


def meta(label, group, route='near_dry'):
    return dict(label=label, source_group=group, object_id=label+'-obj', session='s1',
                capture_route=route, evidence_label='proxy')


def true_span(ev, fs, floor_rms, frame=.01):
    size = round(frame*fs)
    p = np.add.reduceat(ev**2, np.arange(0, len(ev), size))/size
    act = np.flatnonzero(p > 2*floor_rms**2)
    return (act[0]*size/fs, (act[-1]+1)*size/fs) if len(act) else (None, None)


def summarize(res, fs):
    rep = res.report
    out = {k: rep.get(k) for k in ('status', 'rejected', 'review', 'event_count', 'estimated_drr_db',
                                   'tail_energy_fraction', 'snr_proxy_db', 'reason')}
    if 'crop_samples' in rep:
        out['crop_s'] = [round(v/fs, 3) for v in rep['crop_samples']]
        out['event_s'] = [round(v/fs, 3) for v in rep['event_samples']]
    return out


def approve(x, fs, m, **kw):
    first = intake_chamber(x, fs, metadata=m, **kw)
    if 'approval_key' not in first.report:
        return first, None
    second = intake_chamber(x, fs, metadata=m, approval=dict(reviewer='reviewer-sim', decision='approve',
                            approval_key=first.report['approval_key']), **kw)
    return first, second



def approved_digital(samples, fs, **options):
    """Simulated listener for generated TEST data only, not production approval."""
    first = intake_known_digital(samples, fs, **options)
    if first.report['status'] != 'review':
        return first
    return intake_known_digital(samples, fs, **options, approval=dict(reviewer='proxy fixture listener',
        decision='approve', approval_key=first.report['approval_key']))


def approve_project_rows(cfg, out):
    """Bind fixture approvals to the exact decoded file (test-generated audio)."""
    from sensorsim import project
    try:
        project._clips(cfg, out)
    except ValueError as exc:
        report = json.loads((out/'source_qa.json').read_text())
        assert all(c['status'] != 'reject' for c in report['clips']), (exc, report)
        for row, clip in zip(cfg['dataset']['sources'], report['clips']):
            row['approval'] = dict(reviewer='proxy fixture listener', decision='approve',
                                   approval_key=clip['approval_key'])
    return cfg
