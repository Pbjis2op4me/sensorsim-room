"""User-supplied loudspeaker FIR or magnitude response; no bundled response data."""
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .catalog import mono, content_hash


@dataclass(frozen=True)
class LoudspeakerResponse:
    impulse: np.ndarray
    fs: int
    kind: str
    reference: str
    phase: str = 'measured'

    def __post_init__(self):
        x = mono(self.impulse, self.fs)
        if not np.any(x) or self.kind not in ('measured', 'datasheet') or not self.reference.strip():
            raise ValueError('nonzero response, measured/datasheet kind and provenance reference required')
        if self.phase not in ('measured', 'linear-phase approximation'):
            raise ValueError('speaker phase provenance is required')
        x.setflags(write=False)
        object.__setattr__(self, 'impulse', x)

    @classmethod
    def from_magnitude(cls, frequency_hz, gain_db, fs, *, reference, taps=257, reference_db=None):
        f, db = np.asarray(frequency_hz, float), np.asarray(gain_db, float)
        if (f.ndim != 1 or len(f) < 2 or f.shape != db.shape or not np.isfinite(f).all()
                or not np.isfinite(db).all() or f[0] != 0 or f[-1] != fs/2
                or np.any(np.diff(f) <= 0) or isinstance(taps, bool)
                or not isinstance(taps, int) or taps < 3 or taps % 2 == 0):
            raise ValueError('ordered finite magnitude points must cover DC to Nyquist; odd taps >=3')
        if reference_db is not None:
            if not np.isfinite(reference_db):
                raise ValueError('reference_db must be finite')
            db = db-float(reference_db)
        if np.any(abs(db) > 40):
            raise ValueError('gain_db must be relative dB within +/-40; SPL curves need reference_db')
        with np.errstate(over='raise'):
            gain = 10**(db/20)
        h = signal.firwin2(taps, f, gain, fs=fs)
        return cls(h, fs, 'datasheet', reference, 'linear-phase approximation')

    def apply(self, samples, fs):
        if fs != self.fs:
            raise ValueError('loudspeaker response sample rate mismatch')
        y = signal.fftconvolve(mono(samples, fs), self.impulse)
        if not np.isfinite(y).all():
            raise ValueError('nonfinite loudspeaker output')
        return y

    def metadata(self):
        return dict(applied=True, kind=self.kind, reference=self.reference, phase=self.phase,
                    fs=self.fs, response_sha256=content_hash(self.impulse, self.fs),
                    level='relative; no automatic normalization or SPL calibration')


def playback_contract(route, played, position, *, response=None):
    """Only exact digital playback at a declared speaker site can use its FIR."""
    if type(played) is not bool:
        raise ValueError('loudspeaker_played must be boolean')
    if route != 'known_digital' and (played or response is not None):
        raise ValueError('loudspeaker response requires known_digital playback, never a chamber capture')
    if response is not None and not played:
        raise ValueError('loudspeaker response requires explicit loudspeaker_played=True')
    if played:
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('loudspeaker playback requires its finite xyz loudspeaker_position')
        return p.tolist()
    if position is not None:
        raise ValueError('loudspeaker_position requires loudspeaker_played=True')
    return None


def validate_source_position(details, source_position):
    """Do not re-use a speaker-filtered source at a door/chime or sampled site."""
    if details.get('loudspeaker_played') or details.get('loudspeaker_response', {}).get('applied'):
        p = np.asarray(source_position, dtype=float)
        expected = np.asarray(details.get('loudspeaker_position'), dtype=float)
        if (p.shape != (3,) or expected.shape != (3,) or not np.isfinite(p).all()
                or not np.allclose(p, expected, rtol=0, atol=1e-9)):
            raise ValueError('render source position must match the declared loudspeaker_position')
