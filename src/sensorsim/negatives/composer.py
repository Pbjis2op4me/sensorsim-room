"""Context wrappers over training_device's common room, level and capture helpers."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
from sensorsim.training_device import RoomDraw, active_power, mix_events, render_room_event
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig
from .procedural import BED_FAMILIES, EVENT_FAMILIES, TONES, IMPACTS, WATER, MOTORS, Source, generate


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def audio_hash(x):
    return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


@dataclass(frozen=True)
class Event:
    source: Source
    onset_s: float
    source_xyz_m: tuple[float,float,float]
    spl_db: float | None = None
    label: str = 'background'
    reviewed_label_boundary: bool = False


@dataclass(frozen=True)
class Context:
    audio: np.ndarray
    fs: int
    events: tuple[dict,...]
    provenance: dict


@dataclass(frozen=True)
class RenderPath:
    room: RoomModel
    receiver_xyz_m: tuple[float,float,float]
    device: object
    device_id: str
    config: SynthConfig | None = None
    input_reference_spl_db: float = 94.

    def __post_init__(self):
        if (getattr(self.device,'profile_id',None) != self.device_id or
                getattr(self.device,'sensitivity_dbfs_at_94db_spl',None) is None):
            raise ValueError('a versioned absolute device profile is required')
        if not np.isfinite(self.input_reference_spl_db):
            raise ValueError('input reference must be finite')

    def room_draw(self,seed):
        return RoomDraw(self.room,self.receiver_xyz_m,self.config,seed)

    def describe(self):
        return {'room_hash':self.room.model_hash,'room':self.room.model_dump(mode='json'),
                'receiver_xyz_m':self.receiver_xyz_m,'device_id':self.device_id,'device':asdict(self.device),
                'config':(self.config or SynthConfig()).model_dump(mode='json'),
                'input_reference_spl_db':self.input_reference_spl_db,'path_version':'shared-room-device/2'}


def _context(events, *, path, fs, seed, background, domain, bed=None, cfg=None,
             device_draw=None, room_draw=None, duration_s=10.):
    if isinstance(fs,bool) or not isinstance(fs,(int,np.integer)) or not 8000<=fs<=96000:
        raise ValueError('fs must be an integer in [8000,96000]')
    if not np.isfinite(duration_s) or not 0<duration_s<=60:
        raise ValueError('context duration must lie in (0,60]')
    events=tuple(events);n=round(fs*duration_s)
    if len(events)>256: raise ValueError('context exceeds 256-event budget')
    draw=room_draw or path.room_draw(seed)
    if (draw.room.model_hash != path.room.model_hash or tuple(draw.receiver_xyz_m) != tuple(path.receiver_xyz_m)
            or draw.config != path.config):
        raise ValueError('room draw must describe the supplied path')
    def render(ev,index,role):
        src=ev.source
        if src.family in TONES+IMPACTS and ev.label == 'background' and not ev.reviewed_label_boundary:
            raise ValueError(f'{src.family} requires an explicit reviewed target/negative boundary')
        if src.provenance.get('acoustic_stage')!='dry':
            raise ValueError('only dry sources may be room-rendered')
        if src.provenance.get('fs')!=fs: raise ValueError('source rate mismatch')
        return render_room_event(src.audio,fs=fs,draw=draw,position=ev.source_xyz_m,
            onset_s=ev.onset_s,n=n,index=index,spl_db=ev.spl_db,
            metadata={'source_id':src.source_id,'family':src.family,'label':ev.label,'role':role,'source':src.provenance})
    bg=np.zeros(n) if background is None else np.asarray(background)
    if bg.shape != (n,): raise ValueError('background must match context duration')
    settings=dict(cfg or {}) | {'fs':fs,'background_domain':domain,'device_profile':path.device_id,
                                'render_reference_spl_db':path.input_reference_spl_db}
    rows=[]
    if bed is not None:
        if domain!='spl_referenced': raise ValueError('synthetic bed cannot recapture a device recording')
        if bed.spl_db is None: raise ValueError('bed requires explicit receiver SPL assumption')
        wet=render(bed,256,'bed')
        power,count=active_power(wet.audio,fs,wet.span)
        gain=10**((bed.spl_db-path.input_reference_spl_db)/20)/np.sqrt(2*power) if power else 0.
        bg=bg+wet.audio*gain
        rows.append(wet.metadata|{'event_spl_db':bed.spl_db,'receiver_gain':gain,'active_samples':count})
    receiver_events=[render(ev,k,'event') for k,ev in enumerate(events)]
    rng=np.random.default_rng(seed)
    device=device_draw if device_draw is not None else path.device.with_variation(rng)
    result=mix_events(bg,receiver_events,rng,settings,device_draw=device)
    y=result.audio;y.setflags(write=False)
    info=path.describe()|{'room_seed':int(draw.seed),'device_draw':asdict(result.device)}
    from sensorsim.core.fingerprint import source_revision
    return Context(y,fs,tuple(rows)+result.events,{
        'version':'negative-context/2','provenance':'synthetic','evidence_label':'proxy',
        'duration_s':duration_s,'seed':int(seed),'path':info,'path_sha256':digest(info),
        'audio_sha256':audio_hash(y),'source_revision':source_revision(),
        'background_domain':domain,'device_floor':'recording owns floor' if domain=='device_recorded' else 'profile floor added once',
        'level_basis':'microphone active-sample RMS SPL; shared 10 ms / -25 dB gate',
        'output_fullscale_samples':int(np.count_nonzero(np.abs(y)>=1)),
        'deployment_fa_per_hour':'not measured',
    })


def render_context(events, *, path: RenderPath, fs=16000, seed, bed=None,
                   background=None, cfg=None, device_draw=None, room_draw=None, duration_s=10.) -> Context:
    """Synthetic pressure-reference background and dry events, captured once.

    Event.spl_db defaults to the SAME 60–80 dB receiver policy as make_example.
    Supply a shared RoomDraw and already-varied device_draw to pair paths.
    """
    return _context(events,path=path,fs=fs,seed=seed,background=background,domain='spl_referenced',
                    bed=bed,cfg=cfg,device_draw=device_draw,room_draw=room_draw,duration_s=duration_s)


def captured_context(source: Source, *, fs: int, path: RenderPath | None=None, seed=0,
                     events=(), cfg=None, device_draw=None, room_draw=None) -> Context:
    """Keep deployed-device ambience; send only new dry events through the chain.

    An external recording needs capture provenance AND a caller-supplied SPL
    reference. Its existing room is preserved and the deployment device is
    applied; unknown original response remains an explicit stress domain.
    """
    p=source.provenance
    if p.get('acoustic_stage')!='captured' or p.get('fs')!=fs:
        raise ValueError('a captured source at the declared sample rate is required')
    if not p.get('capture_device') or not p.get('profile_id') or not isinstance(p.get('deployed_device'),bool):
        raise ValueError('capture device, profile_id and deployed_device provenance are required')
    if path is None: raise ValueError('captured contexts require a shared RenderPath')
    deployed=p['deployed_device'] and p['profile_id']==path.device_id
    if p['deployed_device'] and not deployed: raise ValueError('capture profile differs from deployed path')
    x=np.asarray(source.audio,float)
    if x.ndim!=1 or len(x)!=fs*10 or not np.isfinite(x).all() or np.max(abs(x))>1:
        raise ValueError('captured context must be finite 10-second mono PCM in [-1,1]')
    settings=dict(cfg or {})
    if deployed:
        settings['background_native_s16']=p.get('native_s16',False)
        background=x;domain='device_recorded'
    else:
        ref=settings.get('captured_reference_spl_db')
        if ref is None or not np.isfinite(ref):
            raise ValueError('external capture requires captured_reference_spl_db; no implicit reference')
        background=x*10**((ref-path.input_reference_spl_db)/20);domain='spl_referenced'
    ctx=_context(events,path=path,fs=fs,seed=seed,background=background,domain=domain,cfg=settings,
                 device_draw=device_draw,room_draw=room_draw)
    ctx.provenance.update(source=p,room_applied=bool(events),device_applied=bool(events) or not deployed,
                          domain='deployment_capture' if deployed else 'external_capture_stress')
    return ctx


def compose(*, path: RenderPath, family_rates_per_hour: dict[str,float], positions: dict,
            seed: int, fs=16000, bed_family='room_tone', bed_spl_db=35.,
            event_spl_db=(60.,80.), mains_hz=60., reviewed_label_boundaries=(),
            background: Source | None=None, device_draw=None, room_draw=None,
            duration_ranges_s: dict | None=None, training_targets=None) -> Context:
    """10 s Poisson proxy; rates, durations and bed level are caller assumptions.

    Kettle-side sources default to 10–30 s continuous forms (including tails
    extending beyond the context), not short faded bursts. All tonal and impact families
    require a reviewed boundary. Procedural realism is not established.
    """
    for family,rate in family_rates_per_hour.items():
        if family not in EVENT_FAMILIES or not np.isfinite(rate) or not 0<=rate<=3600:
            raise ValueError('unknown family or rate outside [0,3600] events/hour')
        if family in TONES+IMPACTS and family not in reviewed_label_boundaries:
            raise ValueError(f'{family} requires an explicit reviewed target/negative boundary')
    partials=()
    if any(f in TONES for f in family_rates_per_hour):
        if training_targets is None:
            raise ValueError('tonal confusers require training targets for partial exclusion')
        from .tonal import target_partials
        partials=target_partials(training_targets,fs=fs)
    if bed_family not in BED_FAMILIES or bed_family=='device_floor':
        raise ValueError('bed must be acoustic; device_floor is supplied by the device')
    required=tuple(family_rates_per_hour)+( (bed_family,) if background is None else () )
    for family in required:
        if not positions.get(family): raise ValueError(f'provide plausible source positions for {family}')
    rng=np.random.default_rng(seed);events=[]
    for family,rate in sorted(family_rates_per_hour.items()):
        count=int(rng.poisson(rate*10/3600))
        if len(events)+count>256: raise ValueError('Poisson draw exceeds 256-event context budget')
        bounds=(duration_ranges_s or {}).get(family,(10.,30.) if family in WATER+MOTORS else (.25,1.5))
        if len(bounds)!=2 or not .1<=bounds[0]<=bounds[1]<=60: raise ValueError('invalid duration bounds')
        for onset in rng.uniform(0,10,count):
            src=generate(family,fs=fs,duration_s=float(rng.uniform(*bounds)),seed=int(rng.integers(2**32)),mains_hz=mains_hz,
                         target_partials_hz=partials)
            xyz=positions[family][int(rng.integers(len(positions[family])))]
            events.append(Event(src,float(onset),xyz,reviewed_label_boundary=family in reviewed_label_boundaries))
    kwargs=dict(path=path,fs=fs,seed=seed,cfg={'event_spl_db':event_spl_db},device_draw=device_draw,room_draw=room_draw)
    events=sorted(events,key=lambda e:e.onset_s)
    if background is None:
        bed=Event(generate(bed_family,fs=fs,duration_s=10,seed=int(rng.integers(2**32)),mains_hz=mains_hz),0.,positions[bed_family][0],bed_spl_db)
        ctx=render_context(events,bed=bed,**kwargs)
    else:
        ctx=captured_context(background,events=events,**kwargs)
    ctx.provenance.update(composer='negative-poisson/2',rates_per_hour=dict(family_rates_per_hour),
        prevalence_basis='caller assumptions',event_spl_db=list(event_spl_db),bed_spl_db=bed_spl_db,
        duration_basis='explicit proxy assumptions; long continuous kettle-side forms',
        reviewed_label_boundaries=list(reviewed_label_boundaries),training_eligible=False)
    return ctx
