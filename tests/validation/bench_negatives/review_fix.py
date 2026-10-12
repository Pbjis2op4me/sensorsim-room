"""Historical WP3 review-fix run; retain its results, not its L4 interpretation.

The former L4 PASS was degenerate: the reference used the same posterior-dither
estimator with another random draw. The shared-RNG path probe was a route-identity
check. final_review.py supersedes these with independent path draws and a known
analog single capture; do not rerun this historical runner as current acceptance.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import warnings
import numpy as np
from scipy import signal
from scipy.io import wavfile
import yaml
from sensorsim import edge, nn, iot_mic
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import synthesize
from sensorsim.negatives.acceptance import symmetry
from sensorsim.negatives.composer import Event, RenderPath, audio_hash, captured_context, render_context, compose
from sensorsim.negatives.procedural import generate, Source, IMPACTS, TONES, EVENT_FAMILIES, BED_FAMILIES, FAMILY_MODELS
from sensorsim.negatives.clips import load_clip, user_register_row
from sensorsim.training_device import active_power, mix_device_audio

FS=16000
PROFILE=iot_mic.PROFILES['pi5_inmp441/1']
ROOM=RoomModel(shoebox_dims_m=(6,5,3),late_t60_s=(.3,)*7)
RX=(4.,3.,1.5)
POS=((1.,1.,1.),(2.,2.,1.),(3.,1.,1.5))
PATH=RenderPath(ROOM,RX,PROFILE,PROFILE.profile_id)
FEATURES=edge.Features(nn._feature_cfg(nn.DEFAULTS))


def stats(windows):
    f=FEATURES(np.asarray(windows))[:,0]
    return np.concatenate([f.mean(2),f.std(2),f.max(2)],axis=1)


def c2st(a,b,seed):
    z=np.zeros((len(a),3))
    r=symmetry(z,z,a,b,np.arange(len(a)),seed=seed,bootstrap_replicates=1000)
    return {k:r[k] for k in ('auc','auc_ci95','structural_result','top_embedding_dimensions')}


def neutral(seed,g,n=FS):
    rng=np.random.default_rng([seed,g]);t=np.arange(n)/FS
    x=rng.standard_normal(n)
    if g%3==1: x=np.sin(2*np.pi*(400+g*31)*t)
    elif g%3==2:x*=np.exp(-t/.1)
    return Source(x,'neutral',f'neutral:{seed}:{g}',{'fs':FS,'acoustic_stage':'dry','license_id':'Apache-2.0','evidence_label':'proxy'})


def path_probe(out,seed,domain):
    positive=[];negative=[];shifted=[];pairs=[]
    for g in range(144):
        src=neutral(seed,g)
        draw_seed=int(np.random.SeedSequence([seed,g,51]).generate_state(1)[0])
        rng=np.random.default_rng([seed,g,5])
        bed=generate('room_tone',fs=FS,duration_s=10,seed=int(rng.integers(2**32))).audio*1e-5
        if domain=='device_recorded':
            background=PROFILE.apply(bed,FS,rng,input_reference_spl_db=94)
        else: background=bed
        # Independent positive production-dataset convolution, actual make_example mixer.
        rir_seed=int(np.random.SeedSequence([draw_seed,0,11]).generate_state(1)[0])
        rir=synthesize(ROOM,POS[g%3],RX,FS,seed=rir_seed).samples
        wet=signal.oaconvolve(src.audio,rir)
        event=np.zeros(10*FS);take=min(len(wet),7*FS);event[3*FS:3*FS+take]=wet[:take]
        cfg=dict(nn.DEFAULTS,background_domain=domain)
        pos=mix_device_audio(background,event,(3*FS,3*FS+take),np.random.default_rng(draw_seed),cfg)
        ev=Event(src,3.,POS[g%3])
        if domain=='device_recorded':
            capture=Source(background,'room_tone',f'bed:{g}',{
                'fs':FS,'acoustic_stage':'captured','capture_device':'proxy-pi','profile_id':PROFILE.profile_id,
                'deployed_device':True,'native_s16':True,'evidence_label':'proxy'})
            neg=captured_context(capture,fs=FS,path=PATH,seed=draw_seed,events=[ev])
        else:
            neg=render_context([ev],path=PATH,seed=draw_seed,background=background)
        assert neg.events[0]['rir_sha256']==audio_hash(rir)
        positive.append(pos[3*FS:4*FS]);negative.append(neg.audio[3*FS:4*FS])
        # Positive control is a real alternative receiver-level path, not feature offsets.
        control=mix_device_audio(background,event,(3*FS,3*FS+take),np.random.default_rng(draw_seed),
                                 cfg|{'event_spl_db':[90,110]})
        shifted.append(control[3*FS:4*FS])
        pairs.append({'group':g,'rir_sha256':audio_hash(rir),'room_hash':ROOM.model_hash,
                      'positive_audio_sha256':audio_hash(pos),'negative_audio_sha256':audio_hash(neg.audio),
                      'device_draw':neg.provenance['path']['device_draw'],
                      'event_spl_db':neg.events[0]['event_spl_db'],
                      'max_difference_lsb':float(np.max(abs(pos-neg.audio))*32768)})
    a,b,c=stats(positive),stats(negative),stats(shifted)
    np.savez(out/f'path-{domain}-{seed}.npz',positive=a,negative=b,shifted=c)
    (out/f'path-pairs-{domain}-{seed}.json').write_text(json.dumps(pairs,indent=2)+'\n')
    return {'seed':seed,'domain':domain,**c2st(a,b,seed),'control':c2st(a,c,seed),
            'max_difference_lsb':max(p['max_difference_lsb'] for p in pairs),
            'note':'separate positive mixer and negative composer entry points; shared RNG is intentional; no trained-B claim'}


def quantization_probe(out,seed):
    a=[];b=[];errors=[];own_errors=[];ref_errors=[];expected=[];identity=True
    for g in range(144):
        rng=np.random.default_rng([seed,g,75])
        bg=rng.integers(-8,9,FS)/32768
        x=neutral(seed,g).audio
        power,_=active_power(x,FS)
        scaled=x*10**((70-94)/20)/np.sqrt(2*power)
        increment=PROFILE.apply(scaled,FS,rng,input_reference_spl_db=94,include_self_noise=False,quantize=False)
        mix_seed=int(rng.integers(2**32));mix_rng=np.random.default_rng(mix_seed)
        y=mix_device_audio(bg,x,(0,FS),mix_rng,{'fs':FS,'event_spl_db':[70,70]},device_draw=PROFILE)
        # Independent capture of the analog bed conditional on the same observed b.
        analog=bg+rng.random(FS)/32768+increment
        z=np.floor(analog*32768)/32768
        replay=np.random.default_rng(mix_seed);replay.uniform(70,70)
        own_analog=bg+replay.random(FS)/32768+increment
        own_errors.extend(((y-own_analog)*32768).tolist())
        ref_errors.extend(((z-analog)*32768).tolist())
        errors.extend(((y-z)*32768).tolist())
        frac=(increment*32768)%1;expected.extend((2*frac*(1-frac)).tolist())
        a.append(y);b.append(z)
        silent=mix_device_audio(bg,x*0,(0,FS),np.random.default_rng(mix_seed),{'fs':FS},device_draw=PROFILE)
        identity &= np.array_equal(silent,bg)
    fa,fb=stats(a),stats(b)
    np.savez(out/f'l4-{seed}.npz',positive=fa,single_capture=fb)
    def summary(e):return {'mean_lsb':float(np.mean(e)),'rms_lsb':float(np.sqrt(np.mean(np.square(e))))}
    return {'seed':seed,**c2st(fa,fb,seed),'zero_event_identical':bool(identity),
            'paired_error':summary(errors),'expected_paired_rms_lsb':float(np.sqrt(np.mean(expected))),
            'own_analog_error':summary(own_errors),'reference_analog_error':summary(ref_errors)}


def texture_metrics(x):
    p=abs(np.fft.rfft(x*np.hanning(len(x)),32768))**2
    f=np.fft.rfftfreq(32768,1/FS);p=p[(f>=60)&(f<=7600)]+1e-20
    return [float(np.exp(np.log(p).mean())/p.mean()),float(20*np.log10(max(abs(x))/np.sqrt(np.mean(x*x))))]


def texture_probe():
    values={f:[] for f in (*IMPACTS,'struck_chime','truth_knock_standin','decaying_noise_burst')};jitter={f:[] for f in IMPACTS}
    for k in range(200):
        rng=np.random.default_rng(k);duration=float(rng.uniform(.25,1.5))
        for family in IMPACTS:
            src=generate(family,fs=FS,duration_s=duration,seed=k)
            values[family].append(texture_metrics(src.audio))
            ioi=np.diff(src.provenance['parameters']['onsets_s'])
            if len(ioi)>1:jitter[family].append(float(np.std(ioi)/np.mean(ioi)))
        values['struck_chime'].append(texture_metrics(generate('other_doorbell',fs=FS,duration_s=duration,seed=k).audio))
        r=np.random.default_rng(100+k);t=np.arange(int(r.uniform(.9,1.4)*FS))/FS
        env=np.minimum(1,t/r.uniform(.01,.05))*np.exp(-t*r.uniform(.5,2.5))
        x=r.normal(size=t.size)*(np.sin(2*np.pi*3*t)>.95)
        x=.3*r.uniform(.5,1)*env*x/max(abs(x))
        values['truth_knock_standin'].append(texture_metrics(x))
        r=np.random.default_rng(k);t=np.arange(FS)/FS
        values['decaying_noise_burst'].append(texture_metrics(r.standard_normal(FS)*np.exp(-t/r.uniform(.01,.08))))
    result={f:{'flatness_p5_p50_p95':np.percentile(v,[5,50,95],axis=0)[:,0].tolist(),
               'crest_db_p5_p50_p95':np.percentile(v,[5,50,95],axis=0)[:,1].tolist(),
               'ioi_cv_p5_p50_p95':np.percentile(jitter[f],[5,50,95]).tolist() if jitter.get(f) else None}
            for f,v in values.items()}
    ref=result['decaying_noise_burst']['flatness_p5_p50_p95'][1]
    passed=all(result[f]['flatness_p5_p50_p95'][1]>=max(.1,.2*ref) and
        3<=result[f]['crest_db_p5_p50_p95'][1]<=40 and
        (not jitter[f] or result[f]['ioi_cv_p5_p50_p95'][1]>=.05) for f in IMPACTS)
    return {'result':'pass' if passed else 'fail','families':result}


def identity_probe():
    collisions=[];n=0
    for seed in (0,1,2,3,5,77,1701,1702,1703):
        for duration in (.1,.4,1.,1.5,10.):
            seen={}
            for f in EVENT_FAMILIES+BED_FAMILIES:
                h=audio_hash(generate(f,fs=FS,duration_s=duration,seed=seed).audio);n+=1
                if h in seen:collisions.append([seed,duration,seen[h],f])
                seen[h]=f
    gates={}
    for family in TONES:
        try:compose(path=PATH,family_rates_per_hour={family:3600},positions={},seed=1)
        except ValueError as e:gates[family]='reviewed' in str(e)
        else:gates[family]=False
    return {'result':'pass' if not collisions and all(gates.values()) else 'fail','draws':n,
            'collisions':collisions,'review_gates':gates,'models':FAMILY_MODELS}


def holdout_probe(out,seed,audit):
    train=[f for f in EVENT_FAMILIES if f not in audit]
    features={}
    for fi,family in enumerate(EVENT_FAMILIES):
        features[family]=stats([generate(family,fs=FS,duration_s=1,seed=int(np.random.SeedSequence([seed,fi,k,92]).generate_state(1)[0])).audio for k in range(100)])
    np.savez(out/f'holdout-features-{seed}.npz',**features)
    comparisons={a:{t:c2st(features[t],features[a],seed) for t in train} for a in audit}
    nearest={a:{'training_family':min(c,key=lambda t:c[t]['auc']),
                 **min(c.values(),key=lambda r:r['auc'])} for a,c in comparisons.items()}
    return {'seed':seed,'nearest':nearest,'all_comparisons':comparisons}


def license_probe(out,module,register):
    spec=importlib.util.spec_from_file_location('wp9_read_only',module)
    w9=importlib.util.module_from_spec(spec);spec.loader.exec_module(w9)
    root=out/'licence-fixtures';root.mkdir()
    file=root/'fixture.wav';wavfile.write(file,FS,(np.sin(np.arange(FS)*.1)*1000).astype(np.int16))
    h=hashlib.sha256(file.read_bytes()).hexdigest()
    base=dict(path=file.name,sha256=h,registry_id='user-test',license_id='user-owned',verified_on='2026-10-08',
              owner='fixture',ownership_statement='Project-generated fixture only.')
    reg=yaml.safe_load(register.read_text());reg['assets'].append(user_register_row(base,name='Generated control',evidence_url='https://example.test/ownership'))
    combined=root/'combined.yaml';combined.write_text(yaml.safe_dump(reg))
    w9.validate_register(combined)
    review=dict(family='timer',source_group='fixture',acoustic_stage='captured',capture_device='fixture-recorder',
                profile_id='external/1',deployed_device=False,listened=True,target_free=True,rights_cleared=True,
                reviewer='fixture',notes='Planted metadata, not an external licence claim.')
    cases=[]
    for licence,rid in [('CC0-1.0','freesound-cc0'),('CC-BY-4.0','freesound-cc-by-4')]:
        item=base|dict(license_id=licence,registry_id=rid,freesound_id='123',uploader='fixture',
            source_url='https://freesound.org/sounds/123/',modifications='test fixture',
            license_url='https://creativecommons.org/publicdomain/zero/1.0/legalcode.en' if licence=='CC0-1.0' else 'https://creativecommons.org/licenses/by/4.0/legalcode.en')
        cases.append((licence,item,review,combined,date(2026,10,8),True))
    cases += [('user-owned',base,review,combined,date(2026,10,8),True),
              ('user-owned-365-days',base,review,combined,date(2027,10,8),True)]
    for lic in ('CC-BY-NC-4.0','CC-BY-SA-4.0','unknown'):
        cases.append((lic,base|{'license_id':lic},review,combined,date(2026,10,8),False))
    for name,item,rev in [('unreviewed',base,review|{'listened':False}),('hash-tamper',base|{'sha256':'0'*64},review),
            ('missing-register',base|{'registry_id':'absent'},review),('missing-capture-provenance',base,review|{'capture_device':None})]:
        cases.append((name,item,rev,combined,date(2026,10,8),False))
    reg['assets'][-1]['disposition']='internal';internal=root/'internal.yaml';internal.write_text(yaml.safe_dump(reg))
    cases.append(('internal',base,review,internal,date(2026,10,8),False))
    controls=[]
    for name,item,rev,registry,when,want in cases:
        try:
            src=load_clip(root,item,register_path=registry,review=rev,as_of=when,register_gate=w9.validate_commercial_manifest)
            accepted=True;error=None
            assert src.provenance['source_sha256']==h and src.provenance['register_sha256']
        except ValueError as exc:accepted=False;error=str(exc)
        controls.append({'name':name,'accepted':accepted,'correct':accepted==want,'error':error})
    # Exercise WP9 itself on a prohibited manifest as well as the outer allow-list.
    cc=cases[0][1];keys=('path','sha256','registry_id','license_id','verified_on','freesound_id','uploader','source_url','modifications','license_url')
    try:w9.validate_commercial_manifest({'schema_version':1,'purpose':'commercial_training','items':[{k:cc[k] for k in keys}|{'license_id':'CC-BY-NC-4.0'}]},registry_path=combined,audio_root=root)
    except ValueError:controls.append({'name':'direct-wp9-NC','accepted':False,'correct':True})
    else:controls.append({'name':'direct-wp9-NC','accepted':True,'correct':False})
    return {'result':'pass' if all(c['correct'] for c in controls) else 'fail','controls':controls,
            'user_register_valid':True,'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
            'register_sha256':hashlib.sha256(register.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preregistration',type=Path,required=True)
    p.add_argument('--prereg-sha256',required=True);p.add_argument('--w9-module',type=Path,required=True);p.add_argument('--register',type=Path,required=True)
    a=p.parse_args();raw=a.preregistration.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.prereg_sha256:raise ValueError('preregistration changed')
    prereg=yaml.safe_load(raw);a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result={'evidence_label':'proxy','preregistration_sha256':a.prereg_sha256,'path':[],'l4':[],'holdout':[]}
    def save():
        result['elapsed_s']=time.monotonic()-start
        (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    warnings.simplefilter('once',UserWarning)
    for seed in prereg['seeds']:
        for domain in ('device_recorded','spl_referenced'):
            result['path'].append(path_probe(a.out,seed,domain));save();print('path',seed,domain,flush=True)
        result['l4'].append(quantization_probe(a.out,seed));save();print('l4',seed,flush=True)
    result['h2']=texture_probe();result['h3_identity']=identity_probe();save();print('texture/identity',flush=True)
    for seed in prereg['seeds']:
        result['holdout'].append(holdout_probe(a.out,seed,prereg['h3']['audit_families']));save();print('holdout',seed,flush=True)
    result['licence']=license_probe(a.out,a.w9_module,a.register)
    decisions={}
    for domain in ('device_recorded','spl_referenced'):
        rows=[r for r in result['path'] if r['domain']==domain]
        auc=float(np.median([r['auc'] for r in rows]));lo,hi=np.median([r['auc_ci95'] for r in rows],axis=0)
        decisions['path_'+domain]='pass' if auc<=.60 and hi<=.70 else 'fail' if lo>=.60 else 'inconclusive'
        decisions['control_'+domain]='pass' if np.median([r['control']['auc_ci95'][0] for r in rows])>=.60 else 'fail'
    rows=result['l4'];point=float(np.median([r['auc'] for r in rows]));lo,hi=np.median([r['auc_ci95'] for r in rows],axis=0)
    l4metrics=all(r['zero_event_identical'] and abs(r['paired_error']['mean_lsb'])<=.02 and
        abs(r['paired_error']['rms_lsb']-r['expected_paired_rms_lsb'])<=.03 and
        all(abs(r[k]['mean_lsb']+.5)<=.01 and abs(r[k]['rms_lsb']-np.sqrt(1/3))<=.01 for k in ('own_analog_error','reference_analog_error')) for r in rows)
    decisions['l4']='pass' if .45<=point<=.55 and lo<=.5<=hi<=.60 and l4metrics else 'fail'
    for family in prereg['h3']['audit_families']:
        lower=[r['nearest'][family]['auc_ci95'][0] for r in result['holdout']]
        decisions['holdout_'+family]='pass' if np.median(lower)>=.60 else 'fail'
    decisions.update(h2=result['h2']['result'],h3_identity=result['h3_identity']['result'],licence=result['licence']['result'])
    result['decisions']=decisions;result['result']='pass' if all(v=='pass' for v in decisions.values()) else 'fail' if 'fail' in decisions.values() else 'inconclusive'
    result['wp3_a']='pending integration';result['trained_b_wp3_b']='pending';result['measured_results']=[]
    import sensorsim.training_device,sensorsim.negatives
    paths=list(Path(sensorsim.negatives.__file__).parent.glob('*.py'))+[Path(sensorsim.training_device.__file__),Path(__file__)]
    result['source_sha256']={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save();print(json.dumps({'result':result['result'],'decisions':decisions,'elapsed_s':result['elapsed_s']},indent=2))


if __name__=='__main__':main()
