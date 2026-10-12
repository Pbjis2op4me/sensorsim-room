"""Knock sequence diagnostics. Spectral similarity is a proxy, not recognition.

The long merge gap joins compatible hits only. A spectrally different rise in
that gap is retained as contamination, including after the last target hit.
These development heuristics still require real-take validation and listening.
"""
from dataclasses import asdict, replace
import numpy as np
from .decay import impact_onsets


def screen_knock_sequence(x, fs, noise, region, *, post_roll_s):
    from sensorsim.acoustics.room.sweep_detect import detect_events
    raw = detect_events(x, fs, noise, merge_gap_s=0., post_roll_s=post_roll_s)
    onsets = [s for s in impact_onsets(x, fs)
              if region.onset_sample <= s < region.offset_sample]
    size = max(2, round(.04*fs))
    freq = np.fft.rfftfreq(size, 1/fs)
    edges = np.array([0., 177., 354., 707., 1414., 2828., fs/2+1])
    def spectrum(start):
        y = np.zeros(size)
        part = x[start:start+size]
        y[:len(part)] = part
        power = abs(np.fft.rfft(y*np.hanning(size)))**2
        bands = np.array([power[(freq >= a) & (freq < b)].sum() for a, b in zip(edges, edges[1:])])
        return bands/max(np.linalg.norm(bands), 1e-30)
    similarities, stray = [], []
    if onsets:
        anchor = spectrum(onsets[0])
        for onset in onsets:
            similarity = float(np.dot(anchor, spectrum(onset)))
            similarities.append(dict(onset_sample=onset, cosine=similarity))
            if similarity < .5:
                stray.append(onset)
    if stray:
        # Restore the target's own offset and post-roll. A following stranger
        # must not redefine the last hit or its decay start by being merged.
        before = [r for r in raw if r.onset_sample < stray[0] and r.offset_sample <= stray[0]]
        if before:
            offset = before[-1].offset_sample
            stop = offset+round(post_roll_s*fs)
            region = replace(region, offset_sample=offset, segment_stop=min(len(x), stop),
                             tail_truncated=stop > len(x))
    return region, dict(premerge_regions=[asdict(r) for r in raw],
        impact_onsets_samples=onsets, impact_spectral_similarity=similarities,
        incompatible_rises_samples=stray, similarity_limit=.5,
        limitation='Proxy spectral check, not semantic classification; inspect every crop by listening.')
