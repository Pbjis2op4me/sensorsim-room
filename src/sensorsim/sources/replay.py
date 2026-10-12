"""Opt-in gain replay: output = gain * an approved training take.

No acoustic diversity, new performance or procedural model is produced.
Replays are excluded from WP6(c) and C2ST source-diversity evidence.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .decay import impact_onsets
from .catalog import SourceTake, AssignedSource, audit_splits, content_hash, require_renderable


@dataclass(frozen=True)
class ReplayFit:
    family: str
    templates: tuple
    fit_groups: tuple
    fit_hashes: tuple


def fit_replay(assigned, family, *, enabled=False, kettle_stages=None):
    """Record approved train templates; full lineage accompanies every replay.

    kettle_stages maps source_group to three contiguous sample intervals in the
    approved crop: rumble, boil, click. Their times cannot be inferred reliably
    from an unknown recording. Other families use observed energy onsets.
    """
    if not enabled:
        raise ValueError('replay sources are opt-in; set enabled=True')
    rows = list(assigned)
    if family not in ('knock', 'chime', 'kettle') or not rows:
        raise ValueError('choose knock, chime or kettle and supply training sources')
    audit_splits(rows)
    if any(r.split != 'train' for r in rows):
        raise ValueError('replay fitting uses training groups only')
    if len({r.take.label for r in rows}) != 1:
        raise ValueError('fit one target class at a time')
    templates = []
    for row in rows:
        t = row.take
        require_renderable(t)
        x, fs = t.samples, t.fs
        power = signal.convolve(x*x, np.ones(max(1, round(.01*fs)))/max(1, round(.01*fs)), mode='same')
        active = np.flatnonzero(power > np.max(power)*.01)
        if not len(active):
            raise ValueError('silent training source')
        template = dict(identity=t.metadata(), fs=fs, n=len(x), peak=float(np.max(abs(x))))
        if family == 'kettle':
            stages = (kettle_stages or {}).get(t.source_group)
            if (stages is None or len(stages) != 3 or any(len(s) != 2 for s in stages)
                    or any(isinstance(v, bool) or not isinstance(v, int) for s in stages for v in s)
                    or any(not 0 <= s[0] < s[1] <= len(x) for s in stages)
                    or any(stages[i][1] != stages[i+1][0] for i in (0, 1))):
                raise ValueError('kettle needs three contiguous annotated rumble/boil/click sample intervals')
            template['stages'] = [dict(start=s, stop=e) for s, e in stages]
            # Retain the nonstationary envelope AND waveform, including the
            # deterministic click. Independent stage phases destroy both.
            template['waveform'] = x.tolist()
        else:
            onsets = impact_onsets(x, fs)
            if not onsets:
                raise ValueError('training take has no resolved impacts; annotate a clearer take')
            template['onsets'] = onsets
            template['waveform'] = x.tolist()
        # Conservative, observed within-take gain support. This is resampling
        # recorded patterns, NOT an identified population of physical objects.
        size = max(1, round(.01*fs))
        envelope = np.sqrt(np.maximum(signal.convolve(x*x, np.ones(size)/size, mode='same'), 0))
        levels = envelope[envelope > max(float(envelope.max())*.1, 1e-15)]
        low, high = np.quantile(levels, [.25, .75])
        template['gain_bounds'] = [float(low/high), 1.] if high > low > 0 else [1., 1.]
        templates.append(template)
    return ReplayFit(family, tuple(templates), tuple(sorted({r.take.source_group for r in rows})),
        tuple(sorted({h for r in rows for h in (r.take.parent_content_sha256, content_hash(r.take.samples, r.take.fs))})))


def generate_replay(fit, *, seed, enabled=False):
    """Resample fitted templates, never held-out parameters or invented ranges."""
    if not enabled:
        raise ValueError('replay sources are opt-in; set enabled=True')
    rng = np.random.default_rng(seed)
    p = fit.templates[int(rng.integers(len(fit.templates)))]
    x = np.asarray(p['waveform'], dtype=float).copy()
    gain = float(rng.uniform(*p['gain_bounds']))
    x *= gain
    if not np.any(x) or not np.isfinite(x).all():
        raise ValueError('replay fit produced no finite nonzero audio')
    identity = {k: p['identity'][k] for k in ('label', 'source_group', 'object_id', 'session')}
    qa = dict(status='pass', crop_content_sha256=content_hash(x, p['fs']),
              limitation='gain replay; no acoustic diversity or independent source realization')
    take = SourceTake(samples=x, fs=p['fs'], **identity, capture_route=p['identity']['capture_route'],
        evidence_label='proxy', parent_content_sha256=p['identity']['parent_content_sha256'],
        details=dict(synthetic=True, family=fit.family,
                     loudspeaker_played=p['identity']['details'].get('loudspeaker_played', False),
                     loudspeaker_position=p['identity']['details'].get('loudspeaker_position'),
                     loudspeaker_response=p['identity']['details'].get('loudspeaker_response', {'applied': False}),
                     generator='gain-replay/1', mode='replay', amplitude_scale=gain,
                     c2st_eligible=False, independent_realization=False,
                     diversity='none; gain-only replay of a training take',
                     physical_parameter_bounds=None, qa=qa, fit_groups=list(fit.fit_groups),
                     fit_hashes=list(fit.fit_hashes), seed=seed, pitch_semitones=0., stretch_factor=1.,
                     adoption='excluded from WP6(c) and C2ST; gain replay is not procedural diversity'))
    return AssignedSource(take, 'train')
