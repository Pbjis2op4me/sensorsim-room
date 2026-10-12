"""New WP4 review acceptance (proxy only), driven by a frozen external registration.

No archive waveform or coordinates are bundled. The optional v1 HDF5 reader uses
netCDF4 already present in the validation environment. It refuses arrays 6 and
sources 8/9, and records the legacy metadata limitation instead of clearing it.
Usage: python -m tests.validation.wp4_ensemble.review_acceptance --registration FILE --hash FILE --data DIR --out DIR
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates
import yaml
from sensorsim.acoustics.room.prior import (RoomSurvey,build_prior,predict_metrics,metric_vector,
    known_source_observation,update_ensemble,predictive_intervals)
from sensorsim.acoustics.room.rir import detect_onset,correct_known_clock
from tests.validation.wp4_ensemble.run import score_cells


def geometry(data):
    # Parse numerical assignments only; never execute an external Python file.
    tree=ast.parse((data/'dechorate/v1/load_positions_and_plot.py').read_text())
    out={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('room_size','mics','srcs_dir','srcs_omn'):
                value=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
                out[name]=np.array(ast.literal_eval(value),float)
    return out


def read_pair(data,state,source,mic=1):
    import netCDF4
    if source not in range(1,8) or mic not in range(1,26):
        raise ValueError('only source columns 1-7, microphone arrays 1-5 allowed')
    g=geometry(data)
    # v1 columns assigned in listed source order, six directional then one omni.
    # Source/rate metadata are incomplete: this is an explicit hypothesis, not an
    # independently verified cross-position calibration acceptance.
    with netCDF4.Dataset(data/'dechorate/v1/dechorate_rir_processed.hdf5') as f:
        raw=np.asarray(f[state][:,mic-1,source-1],float)
    src=g['srcs_dir'][:,source-1] if source<7 else g['srcs_omn'][:,0]
    return raw,g['room_size'],src,g['mics'][:,mic-1]


def excitation(kind,fs,seed,seconds):
    rng=np.random.default_rng(seed);n=round(seconds*fs)
    if kind=='sweep':
        t=np.arange(n)/fs; x=signal.chirp(t,50,seconds,fs*.49,method='logarithmic')
    elif kind=='sweep_packets':
        x=np.zeros(n);t=np.arange(round(.45*fs))/fs
        chirp=signal.chirp(t,50,.45,fs*.49,method='logarithmic')*np.hanning(len(t))
        for start in np.arange(.1,seconds-.5,.7):
            i=round((start+rng.uniform(0,.03))*fs);x[i:i+len(t)]+=chirp*rng.choice([-1,1])
    else:
        x=rng.normal(size=n)
        if kind=='pink':
            x=np.fft.irfft(np.fft.rfft(x)/np.sqrt(np.maximum(np.fft.rfftfreq(n,1/fs),20)),n)
    return x/max(abs(x))*10**(-3/20)


def capture(x,h,fs,seed,ppm,snr,pre=0.):
    y=signal.fftconvolve(x,h);gain=10**(-45/20)/np.sqrt(np.mean(y[:len(x)]**2))
    y*=gain
    # Keep the digital playback <= -3 dBFS; speaker-chain gain lives in h.
    y=np.r_[np.zeros(round(pre*fs)),y,np.zeros(2*fs)]
    ratio=1+ppm*1e-6
    y=map_coordinates(y,[np.arange(int(len(y)*ratio))/ratio],order=3,mode='constant')
    y+=np.random.default_rng(seed+1).normal(0,10**((-45-snr)/20),len(y))
    return y


def summary(cells):
    valid=[c for c in cells if 'prior_winkler' in c]
    widths=[c['relative_half_width'] for c in cells if 'relative_half_width' in c]
    improvement=1-np.mean([c['posterior_winkler'] for c in valid])/np.mean([c['prior_winkler'] for c in valid]) if valid else None
    coverage=sum(c['covered'] for c in cells)/len(cells)
    half=float(np.median(widths)) if widths else None
    return dict(cells=len(cells),valid_cells=len(valid),coverage=coverage,t30_half_width=half,
        winkler_improvement=improvement,
        available_cell_result='pass' if coverage>=.75 and half is not None and half<=.4 and improvement>=.2 else 'fail')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('registration','hash','data','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();sha=hashlib.sha256(a.registration.read_bytes()).hexdigest()
    if sha!=a.hash.read_text().split()[0]:raise ValueError('registration hash mismatch')
    cfg=yaml.safe_load(a.registration.read_text());a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();fs=cfg['fs'];clock=[];cases=[];cells=[];pre=[]
    for state in cfg['states']:
        raw,dims,src,mic=read_pair(a.data,state,7)
        h=signal.resample_poly(raw,1,3);h=np.r_[np.zeros(round(np.linalg.norm(src-mic)/343*fs)),h[detect_onset(h):]]
        # Clock qualification: no metric/model tuning from these diagnostics.
        for seed in cfg['seeds']:
            for kind in cfg['clock']['excitations']:
                seconds=10 if kind.startswith('sweep') else 30
                x=excitation(kind,fs,seed,seconds)
                for ppm in cfg['clock']['drift_ppm']:
                    y=capture(x,h,fs,seed,ppm,45)
                    try:
                        _,d=correct_known_clock(y,x,fs)
                        row=dict(qualified=True,error_ppm=d['drift_ppm']-ppm,diagnostics=d)
                    except ValueError as e:row=dict(qualified=False,reason=str(e))
                    clock.append(dict(state=state,seed=seed,kind=kind,ppm=ppm,**row))
            survey=RoomSurvey.model_validate(dict(room=dict(zip(('length_x','width_y','height_z'),dims)),
                microphone=dict(zip('xyz',mic)),loudspeaker=dict(zip('xyz',src)),
                prior=dict(seed=seed,count=cfg['members'])))
            prior=build_prior(survey);pred=predict_metrics(prior,src,mic,fs,seed=seed)
            posts={}
            for kind in ('white','pink'):
                x=excitation(kind,fs,seed,30)
                for snr in cfg['snr_db']:
                    y=capture(x,h,fs,seed,150,snr,pre=2.)
                    obs=known_source_observation(y,x,fs,source=src,receiver=mic,source_group=kind,input_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest())
                    post=update_ensemble(prior,obs,pred);posts[(kind,snr)]=post
                    case=dict(state=state,seed=seed,kind=kind,snr_db=snr,status=post.status,observation=obs.model_dump(mode='json'),evidence=post.evidence,weights=post.weights)
                    cases.append(case)
            # Persist the posteriors before reading ANY held-out transfer for this case.
            (a.out/f'posteriors_{state}_{seed}.json').write_text(json.dumps(cases[-4:],indent=2)+'\n')
            for source in cfg['heldout_sources']:
                raw,_,hs,hm=read_pair(a.data,state,source)
                truth=metric_vector(signal.resample_poly(raw,1,3),fs)
                predictions=predict_metrics(prior,hs,hm,fs,seed=seed)
                diagnostics={};prior_pi=predictive_intervals(predictions,prior.weights,diagnostics=diagnostics)
                for (kind,snr),post in posts.items():
                    scored=score_cells(truth,prior_pi,predictive_intervals(predictions,post.weights),state,source,seed)
                    cells += [dict(c,kind=kind,snr_db=snr,missing_prediction_weight=diagnostics.get(f"{c['band_hz']}.{c['metric']}",{}).get('missing_weight')) for c in scored]
            print(f'proxy {state} {seed}: '+str([p.status for p in posts.values()]),flush=True)
        x=excitation('white',fs,cfg['seeds'][0],30)
        for offset in (0.,.5,2.):
            o=known_source_observation(capture(x,h,fs,cfg['seeds'][0],150,45,offset),x,fs,source=src,receiver=mic,source_group='pre',input_sha256='proxy')
            pre.append(dict(state=state,pre_s=offset,qualified=o.qualified,t30_500=o.values.get('500.t30_s'),reason=o.reasons))
    summaries=[]
    for kind in ('white','pink'):
        for snr in cfg['snr_db']:
            for seed in cfg['seeds']:
                summaries.append(dict(kind=kind,snr_db=snr,seed=seed,**summary([c for c in cells if c['kind']==kind and c['snr_db']==snr and c['seed']==seed])))
    clock_summary={}
    for kind in cfg['clock']['excitations']:
        rr=[r for r in clock if r['kind']==kind];ok=[r for r in rr if r['qualified']]
        clock_summary[kind]=dict(qualified=len(ok),total=len(rr),max_error_ppm=max([abs(r['error_ppm']) for r in ok],default=None))
    report=dict(label='proxy result',preregistration_sha256=sha,acceptance='inconclusive',
        reason='v1 source-axis/rate metadata incomplete; numeric scores conditional on listed-order/48 kHz hypothesis. No user data.',
        clock_summary=clock_summary,clock=clock,pre_roll=pre,summaries=summaries,cells=cells,
        calibration_cases=cases,elapsed_s=time.monotonic()-start)
    (a.out/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('clock','cells','calibration_cases')},indent=2))

if __name__=='__main__':main()
