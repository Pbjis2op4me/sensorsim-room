"""Acoustic screening of exact digital files, without cropping their sequence."""
import numpy as np

from .decay import late_decay, LATE_LIMIT_S, KNOCK_DECAY_REVIEW


def assess_digital(x, fs, label, *, kind=None):
    from .kinds import source_kind
    kind = source_kind(label, kind)
    review_messages = {}
    rejected, review = [], []
    peak = float(np.max(abs(x)))
    rails = (x >= 32767/32768) | (x <= -1)
    edges = np.diff(np.r_[False, rails, False].astype(int))
    runs = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
    if np.any(abs(x) > 1) or (len(runs) and max(runs) >= 2):
        rejected.append('clipped')
    elif np.any(rails):
        review.append('isolated_full_scale_peak')
    if peak < 1/32768:
        rejected.append('silent_or_sub_lsb')
    # Quietest 100-ms block is a candidate floor ONLY if well below activity.
    frame = max(2, round(.1*fs))
    n = len(x)//frame
    noise = None
    if n:
        blocks = x[:n*frame].reshape(n, frame)
        powers = np.mean(blocks**2, axis=1)
        k = int(np.argmin(powers))
        if powers[k] < max(float(powers.max())*.05, 1e-30):
            noise = blocks[k]
    floor = float(np.mean(noise**2)) if noise is not None else None
    db = float(10*np.log10(floor)) if floor is not None and floor > 0 else None
    if noise is None:
        review.append('noise_floor_unavailable')
    elif db is not None and db > -90:
        review.append('microphone_like_noise_floor')
    decay = late_decay(x, fs, np.zeros(frame) if noise is None else noise)
    mid = decay['bands']['500']['t20_s']
    if mid is not None and mid > LATE_LIMIT_S:
        review.append('late_decay_exceeds_limit')
        if kind == 'knock':
            review_messages['late_decay_exceeds_limit'] = KNOCK_DECAY_REVIEW
    # Unknown excitation cannot prove acoustic dryness even when created digitally.
    # A file without a usable decay remains explicit in the report; a listener
    # must clear that uncertainty before admission.
    if mid is None:
        review.append('late_decay_unavailable')
    return dict(rejected=rejected, review=review, review_messages=review_messages, kind=kind, peak=peak,
                clipped='clipped' in rejected, noise_floor_dbfs=db,
                noise_rms=float(np.sqrt(floor)) if floor is not None else None,
                late_decay=decay, status='reject' if rejected else 'review' if review else 'pass')
