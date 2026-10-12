"""Shared receiver-level event mixing for target and confuser labels.

The 60–80 dB SPL default is an assumption, not measured class evidence. Levels
refer to active-sample RMS at the microphone, before the device response: a
10 ms energy envelope, gated at -25 dB relative to its peak, within the event
span. Silent inter-hit gaps and trailing padding do not dilute this power.
Partial crops must inherit full-take gain using this same active_power metric.
A room state and device realization are explicit inputs; neither depends on a label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import numpy as np
from scipy import signal
from sensorsim import iot_mic


@dataclass(frozen=True)
class RoomDraw:
    """A caller-selected room realization, shared with the positive dataset path."""
    room: object
    receiver_xyz_m: tuple[float, float, float]
    config: object = None
    seed: int = 0


@dataclass(frozen=True)
class ReceiverEvent:
    audio: np.ndarray
    span: tuple[int, int] | None = None
    spl_db: float | None = None
    metadata: dict = field(default_factory=dict)
    parent_gain: object | None = None


@dataclass(frozen=True)
class MixResult:
    audio: np.ndarray
    events: tuple[dict, ...]
    device: object


def render_room_event(audio, *, fs, draw: RoomDraw, position, onset_s, n, index=0,
                      spl_db=None, metadata=None) -> ReceiverEvent:
    """Production RIR and convolution; no source-level normalization or capture."""
    from sensorsim.acoustics.room.synth import synthesize
    import hashlib
    x = np.asarray(audio, float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('source must be finite nonempty mono')
    if not np.isfinite(onset_s) or not 0 <= onset_s < n/fs:
        raise ValueError('event onset must lie inside the context')
    rir_seed = int(np.random.SeedSequence([draw.seed, index, 11]).generate_state(1)[0])
    rir = synthesize(draw.room, position, draw.receiver_xyz_m, fs, seed=rir_seed, config=draw.config)
    wet = signal.oaconvolve(x, rir.samples)
    start = round(onset_s*fs)
    take = min(len(wet), n-start)
    out = np.zeros(n)
    out[start:start+take] = wet[:take]
    latency = int(rir.metadata['latency_samples'])
    distance = float(np.linalg.norm(np.asarray(position)-draw.receiver_xyz_m))
    row = dict(metadata or {}) | {
        'onset_s': start/fs, 'source_end_s': min(n, start+len(x))/fs,
        'first_arrival_s': start/fs+distance/draw.room.sound_speed_m_s+latency/fs,
        'rendered_end_s': (start+take)/fs, 'tail_truncated_samples': len(wet)-take,
        'source_xyz_m': list(position), 'rir_seed': rir_seed,
        'rir_sha256': hashlib.sha256(np.asarray(rir.samples, '<f8').tobytes()).hexdigest(),
        'room_hash': draw.room.model_hash, 'latency_samples': latency,
    }
    return ReceiverEvent(out, (start, start+take), spl_db, row)


def active_power(event, fs, span=None):
    """10 ms energy gate at -25 dB within the supplied span, excluding silent gaps.

    The gate is an exposed modelling convention shared by both labels. Trailing
    zero padding cannot change its RMS. Filter tails remain in the output.
    """
    x = np.asarray(event, float)
    a, b = span if span is not None else (0, len(x))
    if not 0 <= a <= b <= len(x):
        raise ValueError('event span is outside audio')
    # Empty arrays still reach numpy's existing error; a nonempty silent event
    # has no active samples regardless of the envelope kernel or span.
    if x.size and not np.any(x):
        return 0., 0
    w = min(len(x), max(1, round(.01*fs)))
    env = np.convolve(x*x, np.ones(w)/w, mode='same')
    mask = env > env.max()*10**(-25/10)
    mask[:a] = False
    mask[b:] = False
    return (float(np.mean(x[mask]**2)) if mask.any() else 0.), int(mask.sum())


def conditional_capture(background, increment, rng, profile):
    """Add a rendered float increment to a capture, retaining overload clipping.

    The recording already owns its noise and quantization error. Dither and a
    second quantization would add label-correlated noise. Training features accept
    float windows; only a later PCM file/device boundary quantizes. ``rng`` stays
    in this compatibility signature but is not consumed.
    """
    level = (10**((profile.sensitivity_dbfs_at_94db_spl+profile.max_spl_peak_db-94)/20)
             if profile.max_spl_peak_db is not None else 1.)
    return np.clip(np.asarray(background, float)+np.asarray(increment, float),
                   -min(level, 1.), min(level, 1.))


def _check_recording(x, fs, profile, native):
    if not native or np.max(x) > 32767/32768 or not np.array_equal(x*32768, np.rint(x*32768)):
        warnings.warn('device-recorded background is not native S16; review capture provenance', UserWarning, stacklevel=3)
    if not np.any(x):
        warnings.warn('device-recorded background is silent', UserWarning, stacklevel=3)
    else:
        # Compare with the nominal profile, not a fresh per-example variation.
        # A 6 dB deficit for >=1 s is a conservative QA assumption, allowing the
        # profile's +/-2 dB device uncertainty and short quiet/filter transients.
        nominal = iot_mic.PROFILES.get(profile.profile_id, profile)
        frame = max(1, round(.25*fs))
        # The meter supports leading batch axes. Keep the same per-frame FFT,
        # reduction and scalar threshold arithmetic while constructing its
        # frequency weighting once for this recording instead of once per frame.
        from sensorsim.acoustics.weighting import a_weighted_mean_square
        frames = np.asarray(x[:len(x)//frame*frame], float).reshape(-1, frame)
        rms_values = np.sqrt(a_weighted_mean_square(frames, fs)) if len(frames) else ()
        run = 0
        for value in rms_values:
            rms = float(value)*np.sqrt(2)
            below = 20*np.log10(max(rms, 1e-30)) < nominal.self_noise_dbfs_a-6.
            run = run+frame if below else 0
            if run >= fs:
                warnings.warn('device-recorded background is below the profile noise floor '
                              'by more than 6 dB for at least 1 s', UserWarning, stacklevel=3)
                break


def mix_events(background, events, rng, cfg, *, device_draw=None) -> MixResult:
    """One receiver level policy and capture for either label, with any event count.

    Events are receiver-domain audio, optionally made by render_room_event with
    the same RoomDraw as target data. A supplied device_draw is already varied;
    otherwise exactly one variation is drawn, including for empty event lists.
    A synthetic bed is in peak-sine pressure-reference units and is captured once.
    """
    from sensorsim import event_levels
    mode = event_levels.validate_mode(cfg)
    fs = cfg['fs']
    raw = np.asarray(background)
    bg = raw.astype(float)/(32768 if raw.dtype == np.int16 else 1)
    if bg.ndim != 1 or not bg.size or not np.isfinite(bg).all():
        raise ValueError('background must be finite nonempty mono')
    domain = cfg.get('background_domain', 'device_recorded')
    if domain not in ('device_recorded', 'spl_referenced'):
        raise ValueError("background_domain must be 'device_recorded' or 'spl_referenced'")
    if cfg.get('device_split', 'in_distribution') != 'in_distribution':
        raise ValueError('mix_events requires in_distribution; stress mixing is an explicit separate API')
    profile = device_draw or iot_mic.PROFILES.get(cfg.get('device_profile', 'pi5_inmp441/1'))
    if profile is None or profile.profile_id is None:
        raise ValueError('device_profile must name a versioned absolute-level Pi profile')
    if device_draw is None:
        profile = profile.with_variation(rng)
    reference = float(cfg.get('render_reference_spl_db', 94.))
    levels = np.asarray(cfg.get('event_spl_db', [60., 80.]), float)
    if levels.shape != (2,) or not np.isfinite(levels).all() or levels[0] > levels[1]:
        raise ValueError('event_spl_db must be a finite ordered pair of SPL levels')
    if not np.isfinite(reference):
        raise ValueError('render_reference_spl_db must be finite')
    combined, rows = np.zeros_like(bg), []
    for ev in events:
        x = np.asarray(ev.audio, float)
        if x.shape != bg.shape or not np.isfinite(x).all():
            raise ValueError('event must match background length and be finite mono')
        power, count = active_power(x, fs, ev.span)
        parent_metadata = {}
        if mode == event_levels.LEGACY:
            if ev.parent_gain is not None:
                raise ValueError('parent gain cannot be silently discarded in legacy mode')
            # Preserve reviewed RNG consumption, level and gain exactly.
            level = float(rng.uniform(*levels)) if ev.spl_db is None else float(ev.spl_db)
            if not np.isfinite(level):
                raise ValueError('event SPL must be finite')
            gain = 10**((level-reference)/20)/np.sqrt(2*power) if power else 0.
        else:
            if ev.spl_db is not None:
                raise ValueError('receiver SPL normalization conflicts with parent authority')
            level = None
            if ev.parent_gain is None and not np.any(x):
                gain = 0.  # Explicit event-free context, no new level draw.
            else:
                gain = event_levels.receiver_gain(ev.parent_gain, cfg)
                parent_metadata = {'parent_level': ev.parent_gain.metadata()}
        combined += x*gain
        rows.append(ev.metadata | parent_metadata | {'event_spl_db': level, 'receiver_gain': gain,
                                    'active_samples': count, 'active_power_before_gain': power})
    if domain == 'spl_referenced':
        y = profile.apply(bg+combined, fs, rng, input_reference_spl_db=reference)
    else:
        if np.max(np.abs(bg)) > 1:
            raise ValueError('device recording must be PCM in [-1,1]')
        _check_recording(bg, fs, profile, cfg.get('background_native_s16', True))
        if not np.any(combined):
            y = bg.copy()
        else:
            increment = profile.apply(combined, fs, rng, input_reference_spl_db=reference,
                                      include_self_noise=False, quantize=False)
            y = conditional_capture(bg, increment, rng, profile)
    return MixResult(y, tuple(rows), profile)


def mix_device_audio(background: np.ndarray, event: np.ndarray, span: tuple[int, int],
                     rng: np.random.Generator, cfg: dict, *, device_draw=None, parent_gain=None) -> np.ndarray:
    """Existing make_example entry point delegates to the common event mixer."""
    from sensorsim import event_levels
    event_levels.validate_mode(cfg)
    if parent_gain is not None and cfg.get('device_split', 'in_distribution') == 'stress':
        raise ValueError('parent gain cannot enter stress normalization')
    if cfg.get('device_split', 'in_distribution') == 'stress':
        background, event = np.asarray(background, float), np.asarray(event, float)
        a, b = span
        pt = float(np.mean(event[a:b]**2)) if b > a else 0.
        pb = float(np.mean(background[a:b]**2)) if b > a else 0.
        audio = background.copy()
        if pt > 0:
            audio += event*(np.sqrt(pb*10**(rng.uniform(*cfg['snr_db'])/10)/pt) if pb > 0 else 1.)
        if rng.random() < cfg.get('p_mic_chain', .85):
            ref = float(np.sqrt(np.mean(background**2)) or 1e-4)
            return iot_mic.random_chain(rng).apply(audio, cfg['fs'], rng, ref_rms=ref)
        return np.clip(np.round(audio*32767)/32767, -1, 1)
    return mix_events(background, [ReceiverEvent(event, span, parent_gain=parent_gain)], rng, cfg, device_draw=device_draw).audio
