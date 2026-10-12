"""Project-owned acoustic proxies; every numerical region is an assumption.

S05's errata supply no transferable contact, mode, spacing, chime or kettle-stage
ranges. The declared regions below are deliberately distinct engineering proxies,
not fitted household populations. They must pass a recorded-family realism gate
before training use. No observed appliance prevalence is implied.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy import signal

IMPACTS = ('door_closure', 'cabinet_door', 'footsteps_wood', 'hammering',
           'dropped_objects', 'dishes_cutlery', 'furniture_taps')
TONES = ('phone_notification', 'appliance_beeps', 'timer', 'tv_jingle', 'other_doorbell', 'alarm_chirp')
WATER = ('running_tap', 'boiling_pan', 'frying', 'washer_fill', 'coffee_machine')
MOTORS = ('extractor_hood', 'hair_dryer', 'vacuum')
EVENT_FAMILIES = IMPACTS + TONES + WATER + MOTORS
BED_FAMILIES = ('device_floor', 'room_tone', 'hvac_fan', 'fridge_hum', 'distant_traffic')
# Explicit proxy assumptions, not citations to physical population measurements.
# Shared contact-body form has distinct count/decay/modal regions by family.
IMPACT_REGIONS = {
    'door_closure': dict(hits=1, body_s=(.10,.15), modal_hz=(90.,180.), contact_s=(.001,.003)),
    'cabinet_door': dict(hits=3, body_s=(.025,.040), modal_hz=(220.,400.), contact_s=(.0005,.001)),
    'footsteps_wood': dict(hits=4, body_s=(.060,.075), modal_hz=(55.,85.), contact_s=(.004,.008)),
    'hammering': dict(hits=5, body_s=(.012,.020), modal_hz=(450.,850.), contact_s=(.0001,.0004)),
    'dropped_objects': dict(hits=3, body_s=(.040,.055), modal_hz=(950.,1400.), contact_s=(.001,.002)),
    'dishes_cutlery': dict(hits=7, body_s=(.003,.008), modal_hz=(2200.,3100.), contact_s=(.0001,.0003)),
    'furniture_taps': dict(hits=1, body_s=(.008,.012), modal_hz=(1500.,2000.), contact_s=(.0003,.0007)),
}
MOTOR_REGIONS = {
    'hvac_fan': dict(motor_hz=(40.,65.), lowpass_hz=900., harmonics=1, tone_weight=.05),
    'extractor_hood': dict(motor_hz=(70.,90.), lowpass_hz=1700., harmonics=2, tone_weight=.12),
    'hair_dryer': dict(motor_hz=(250.,350.), lowpass_hz=5000., harmonics=3, tone_weight=.08),
    'vacuum': dict(motor_hz=(600.,800.), lowpass_hz=3000., harmonics=5, tone_weight=.30),
}
FAMILY_MODELS = {f: {'generator':'broadband_contact_body', 'region':r} for f,r in IMPACT_REGIONS.items()}
FAMILY_MODELS.update({f:{'generator':'rotor_noise','region':r} for f,r in MOTOR_REGIONS.items()})
FAMILY_MODELS.update({f:{'generator':f, 'region':{}} for f in EVENT_FAMILIES+BED_FAMILIES if f not in FAMILY_MODELS})


@dataclass(frozen=True)
class Source:
    audio: np.ndarray
    family: str
    source_id: str
    provenance: dict


def _unit(x):
    rms = float(np.sqrt(np.mean(x*x)))
    if not np.isfinite(rms) or rms < 1e-15:
        raise ValueError('generator produced an empty or nonfinite signal')
    x = np.asarray(x/rms, np.float64)
    x.setflags(write=False)
    return x


def generate(family: str, *, fs: int, duration_s: float, seed: int, mains_hz: float = 60.,
             target_partials_hz=()) -> Source:
    """Repeatable 0.1–60 s proxy. Continuous appliances have no artificial fades.

    Unsupported numerical physical ranges are explicitly assumptions in both
    FAMILY_MODELS and returned provenance. Mains frequency is caller-selected.
    The composer sets receiver SPL; this function's unit RMS is not a level prior.
    """
    if family not in EVENT_FAMILIES+BED_FAMILIES:
        raise ValueError(f'unknown family {family!r}')
    if isinstance(fs,bool) or not isinstance(fs,(int,np.integer)) or not 8000 <= fs <= 96000:
        raise ValueError('fs must be an integer in [8000,96000]')
    if not np.isfinite(duration_s) or not .1 <= duration_s <= 60:
        raise ValueError('duration_s must be in [0.1,60]')
    if mains_hz not in (50.,60.):
        raise ValueError('mains_hz must be 50 or 60')
    rng=np.random.default_rng(seed)
    n=round(fs*duration_s); t=np.arange(n)/fs
    p={'duration_s':float(duration_s),'mains_hz':float(mains_hz), 'assumption_region':FAMILY_MODELS[family]['region']}
    def low(x,hz):
        return signal.sosfilt(signal.butter(2,min(hz,.4*fs),fs=fs,output='sos'),x)
    def band(x,lo,hi):
        return signal.sosfilt(signal.butter(2,[min(lo,.3*fs),min(hi,.4*fs)],btype='bandpass',fs=fs,output='sos'),x)
    if family in IMPACTS:
        r=IMPACT_REGIONS[family]; count=r['hits']
        intervals=rng.uniform(.65,1.35,max(0,count-1))
        if family=='dropped_objects': intervals*=.64**np.arange(count-1)
        onsets=np.r_[0,np.cumsum(intervals)]
        if count>1: onsets=onsets/onsets[-1]*duration_s*.68
        p['onsets_s']=onsets.tolist();p['hits']=[]
        x=np.zeros(n)
        for k,onset in enumerate(onsets):
            a=round(onset*fs); u=np.arange(n-a)/fs
            body=float(rng.uniform(*r['body_s']));contact=float(rng.uniform(*r['contact_s']))
            shape=float(rng.uniform(.7,1.6));amp=float(rng.uniform(.65,1.3))
            if family=='dropped_objects': amp*=.7**k
            modes=rng.uniform(*r['modal_hz'],8);decay=body*rng.uniform(1.,1.8)
            env=(1-np.exp(-(u+1/fs)/contact))*np.exp(-(u/body)**shape)
            # Broadband body persists with the ring, preventing a few-partial chime.
            noise=rng.standard_normal(len(u))
            modal=sum(np.sin(2*np.pi*min(f,.4*fs)*u+rng.uniform(0,2*np.pi)) for f in modes)/np.sqrt(4)
            x[a:]+=amp*(env*noise+.08*modal*np.exp(-u/decay))
            p['hits'].append(dict(body_s=body,contact_s=contact,shape=shape,amplitude=amp,
                                  modes_hz=modes.tolist(),modal_decay_s=decay,broadband_weight=1.,modal_weight=.08))
    elif family in TONES:
        x=np.zeros(n)
        # Different forms; all remain gated against the deployment's doorbell.
        from .tonal import draw_base, EXCLUSION_FRACTION
        base,support=draw_base(rng,family,fs,target_partials_hz)
        p.update(base_hz=base,tonal_support_hz=support,
                 target_partials_hz=list(target_partials_hz),exclusion_fraction=EXCLUSION_FRACTION)
        if family=='other_doorbell':
            p['model']='struck_chime';p['partial_ratios']=[1.,2.71,5.4]
            p['decay_s']=float(rng.uniform(.35,.65))
            for ratio,amp in zip(p['partial_ratios'],[1.,.4,.18]):
                x+=amp*np.sin(2*np.pi*min(base*ratio,.4*fs)*t)*np.exp(-t/p['decay_s'])
        elif family=='phone_notification':
            width=duration_s*.32;p['note_ratios']=[1.,1.5]
            for onset,ratio in zip([0,duration_s*.45],p['note_ratios']):
                u=t-onset;v=np.maximum(u,0)
                x+=(u>=0)*np.sin(2*np.pi*base*ratio*v)*np.exp(-v/width)
        elif family=='appliance_beeps':
            duty=(t/duration_s*3)%1 < .42
            x=np.sin(2*np.pi*base*t)*duty
            p.update(cycles=3,duty=.42)
        elif family=='timer':
            duty=(t/duration_s*8)%1 < .20
            x=(np.sin(2*np.pi*base*t)+.25*np.sin(2*np.pi*base*3*t))*duty
            p.update(cycles=8,duty=.20)
        elif family=='tv_jingle':
            p['note_ratios']=[1.,1.25,1.5,1.125,2.]
            for k,ratio in enumerate(p['note_ratios']):
                a=round(k*n/5);b=round((k+1)*n/5);u=np.arange(b-a)/fs
                x[a:b]=(np.sin(2*np.pi*base*ratio*u)+.35*np.sin(2*np.pi*2*base*ratio*u))*np.sin(np.linspace(0,np.pi,b-a))
        else:
            sweep=support[0][1]-base
            x=np.sin(2*np.pi*(base*t+sweep/2*t*t/duration_s))*(.5+.5*np.sin(2*np.pi*5*t/duration_s))
            p['sweep_hz']=sweep
    else:
        white=rng.standard_normal(n)
        if family=='device_floor': x=white
        elif family=='room_tone': x=low(white,500);p['lowpass_hz']=500.
        elif family=='fridge_hum':
            amplitudes=np.array([1.,.5,1/3,.25])
            tonal=sum(amp*np.sin(2*np.pi*mains_hz*k*t+rng.uniform(0,2*np.pi)) for k,amp in enumerate(amplitudes,1))
            x=tonal+.15*low(white,700)
            p.update(harmonic_amplitudes=amplitudes.tolist(),broadband_weight=.15,lowpass_hz=700.,
                     energy_split_basis='unsourced engineering assumption; fixed-speed compressor proxy, not inverter coverage')
        elif family=='distant_traffic':
            hz=float(rng.uniform(180,350));x=low(white,hz)*(.4+.6*np.sin(np.pi*t/duration_s)**2);p['lowpass_hz']=hz
        elif family in MOTOR_REGIONS:
            r=MOTOR_REGIONS[family];f=float(rng.uniform(*r['motor_hz']))
            x=low(white,r['lowpass_hz'])+r['tone_weight']*sum(np.sin(2*np.pi*f*k*t)/k for k in range(1,r['harmonics']+1))
            p.update(motor_hz=f,motor_hz_basis='unsourced engineering assumption',**{k:v for k,v in r.items() if k!='motor_hz'})
        elif family=='running_tap':
            x=band(white,300,5000)*(1+.12*np.sin(2*np.pi*1.2*t));p.update(band_hz=[300,5000],am_hz=1.2)
        elif family=='boiling_pan':
            bubbles=signal.lfilter([1.],[1.,-.92],rng.random(n)<.005)
            x=low(white,1600)*(1+3*bubbles);p.update(lowpass_hz=1600.,bubble_probability=.005)
        elif family=='frying':
            crackles=(rng.random(n)<.02)*rng.normal(size=n)
            x=band(white,1500,6000)*.25+signal.lfilter([1.,-.6],[1.],crackles);p.update(band_hz=[1500,6000],crackle_probability=.02)
        elif family=='washer_fill':
            x=band(white,70,650)*(1+.6*np.sin(2*np.pi*.35*t));p.update(band_hz=[70,650],am_hz=.35)
        else:  # coffee pump pulses with water, distinct from running water alone
            pump=(.2+.8*(np.sin(2*np.pi*5*t)>0))
            x=band(white,700,2500)*pump+.3*np.sin(2*np.pi*110*t);p.update(band_hz=[700,2500],pump_hz=5.,motor_hz=110.,motor_hz_basis='assumption')
    audio=_unit(x);digest=hashlib.sha256(audio.astype('<f8').tobytes()).hexdigest()
    from .assumptions import parameter_assumptions
    return Source(audio,family,f'procedural:{family}:{digest}',{
        'generator':'negative-procedural/2','family':family,'model':FAMILY_MODELS[family]['generator'],
        'license_id':'Apache-2.0','provenance':'synthetic','evidence_label':'proxy','acoustic_stage':'dry',
        'source_sha256':digest,'fs':int(fs),'seed':int(seed),'parameters':p,
        'range_basis':'explicit unsourced engineering assumptions; not S05 population ranges',
        'parameter_assumptions':parameter_assumptions(family,FAMILY_MODELS[family]['region'],
            impact=family in IMPACTS,tonal=family in TONES),
        'population_validated':False,'training_eligible':False,
    })
