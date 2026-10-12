# SPDX-License-Identifier: Apache-2.0
"""Project-generated TRUTH inputs for a small dynamics proxy screen, never real evidence.

Writes a standard training manifest and a held-out context manifest outside the
repository. The screen runner accepts the same schemas for user-owned takes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from sensorsim import iot_mic, nn, recognizer
from sensorsim.acoustics.room.model import RoomModel, uniform_bands
from sensorsim.acoustics.room.session_sim import canonical_kit, simulate_session, Impairments
from sensorsim.acoustics.weighting import a_weighted_mean_square
from tools.prepare_dynamics_proxy import caller_levels
from tools.prepare_dynamics_proxy import require_unclipped, truth_rir

FS = 16000


def source(label, group):
    rng = np.random.default_rng([20261008, group, ['knock','doorbell','kettle'].index(label)])
    seconds = {'knock':.8,'doorbell':4.,'kettle':20. if group == 0 else 8.}[label]
    t = np.arange(round(seconds*FS))/FS
    if label == 'knock':
        x = np.zeros(len(t))
        for start in np.cumsum(rng.uniform(.1,.22,int(rng.integers(1,4)))):
            u = np.maximum(t-start,0)
            pulse = rng.normal(size=len(t))*np.exp(-u/rng.uniform(.002,.009))
            modes = sum(rng.uniform(.2,1)*np.sin(2*np.pi*rng.uniform(150,2000)*u)*np.exp(-u/rng.uniform(.012,.07)) for _ in range(4))
            x += (pulse+modes)*(t>=start)
    elif label == 'doorbell':
        f=rng.uniform(550,1000)
        x=sum(rng.uniform(.2,1)*np.sin(2*np.pi*f*ratio*t)*np.exp(-t/rng.uniform(.4,1.8)) for ratio in (1.,1.33,2.01))
        x*=1-np.exp(-t*150)
    else:
        noise=signal.sosfilt(signal.butter(2,rng.uniform(150,600),fs=FS,btype='highpass',output='sos'),rng.normal(size=len(t)))
        x=noise*(.25+.75*t/seconds)*(1+.2*np.sin(2*np.pi*rng.uniform(.7,3)*t))
        x*=np.minimum(1,t/.08)*np.minimum(1,(seconds-t)/.1)
    return x/(np.sqrt(np.mean(x*x))*np.sqrt(2))


def bed(rng, seconds, dba):
    x=signal.lfilter([1],[1,-.95],rng.normal(size=round(seconds*FS)))
    x*=10**((dba-120-10*np.log10(2))/20)/np.sqrt(a_weighted_mean_square(x,FS))
    return iot_mic.PROFILES['pi5_inmp441/1'].apply(x,FS,rng,input_reference_spl_db=120.)


def nuisance(family, item):
    rng=np.random.default_rng([45691,family,item]);t=np.arange(4*FS)/FS
    if family < 4:
        x=np.zeros_like(t)
        for onset in rng.uniform(.1,1.8,1+family):
            u=np.maximum(t-onset,0)
            x+=(t>=onset)*(rng.normal(size=len(t))*np.exp(-u/rng.uniform(.003,.045))+
                np.sin(2*np.pi*rng.uniform(80,1800)*u)*np.exp(-u/rng.uniform(.02,.18)))
    elif family < 6:
        freq=rng.uniform(400,2000); x=np.sin(2*np.pi*freq*t)+.5*np.sin(2*np.pi*freq*1.49*t)
        x*=((t%.6)<.22)*np.minimum(1,t/.01)*np.exp(-t/(1.5 if family==4 else 8.))
    else:
        x=signal.sosfilt(signal.butter(2,350 if family==6 else 1500,fs=FS,btype='lowpass',output='sos'),rng.normal(size=len(t)))
        x*=np.minimum(1,t/.05)*(1+.3*np.sin(2*np.pi*7*t))
    return x/max(1e-12,np.sqrt(np.mean(x*x)))*.01


def prepare(out, bed_dba=None):
    """bed_dba = (quiet, typical, loud) background levels in dBA, supplied by the caller."""
    import pyroomacoustics as pra
    quiet,typical,loud=caller_levels(bed_dba,'bed_dba',3)
    out=Path(out)
    if out.exists():raise FileExistsError('use a fresh screen input directory')
    out.mkdir(parents=True)
    kit=canonical_kit().model_copy(update=dict(fs_hz=FS,f2_hz=7000.,sweep_s=.3,fade_in_s=.01,
        fade_out_s=.01,lead_silence_s=.1,tail_silence_s=.2,end_silence_s=.1,repeats=1,period_s=.5,first_sweep_start_s=.1))
    dims=(5.,4.,2.7);mic=(1.,2.,1.2);src=(2.5,2.,1.2)
    room=RoomModel(shoebox_dims_m=dims,absorption=uniform_bands(.3))
    truth=pra.ShoeBox(dims,fs=FS,materials=pra.Material(.3),max_order=30)
    truth.add_source(src);truth.add_microphone_array(np.array(mic)[:,None]);truth.compute_rir();h=truth_rir(truth)
    rows=[];bank=[]
    for group in range(30):
        dry={f'S1_M1_real_{label}.wav':source(label,group)*.01 for label in ('knock','doorbell','kettle')}
        for key,x in dry.items():require_unclipped(signal.fftconvolve(x,h),key)
        session=out/f'group_{group:02d}'
        simulate_session(session,room,{'S1':src},{'M1':mic},truth_dir=out/f'truth_{group:02d}',kit=kit,
            impairments=Impairments.clean(FS,recording_gain=1.),external_rirs={('S1','M1'):h},
            real_sources=dry,include_reference=False,noise_seconds=2.,seed=20261008+group)
        split='train' if group<18 else 'val' if group<20 else 'heldout'
        for class_index,label in enumerate(('knock','doorbell','kettle')):
            path=session/f'S1_M1_real_{label}.wav'
            wet=recognizer.read_audio(path,FS);require_unclipped(wet,str(path))
            if split!='heldout':
                rows.append(dict(file=str(path.relative_to(out)),fs=FS,label=label,split=split,source_group=f'{label}:{group}',evidence_kind='model-based'))
                continue
            rng=np.random.default_rng([7822,group,class_index])
            b=bed(rng,8,quiet if group%2 else loud)
            event=np.zeros(8*FS);a,z=recognizer._audible_span(wet,FS)
            mid=label!='knock' and group%2==0
            clip=wet[a:z];clip=clip[len(clip)//2:] if mid else clip
            begin=0 if mid else 2*FS;clip=clip[:len(event)-begin]
            event[begin:begin+len(clip)]=clip
            level=(45.,55.,65.)[(group-20)%3]
            audio=nn.mix_device_audio(b,event,(begin,begin+len(clip)),rng,dict(nn.DEFAULTS,event_spl_db=[level,level]))
            dst=out/f'held_{label}_{group}.wav';wavfile.write(dst,FS,audio.astype(np.float32))
            bank.append(dict(audio=dst.name,source_group=f'{label}:{group}',kind='target',label=label,
                             start_s=begin/FS,end_s=(begin+len(clip))/FS,event_spl_db=level,mid_event=mid,evidence_kind='model-based'))
        print('source group',group,'prepared',flush=True)
    for split,code in (('train',81),('val',82)):
        for i,dba in enumerate((quiet,typical)):
            audio=bed(np.random.default_rng([code,i]),16,dba)
            dst=out/f'bed_{split}_{i}.wav';wavfile.write(dst,FS,audio.astype(np.float32))
            rows.append(dict(file=dst.name,label='background',fs=FS,split=split,source_group=f'bed:{split}:{i}',evidence_kind='model-based'))
    families=['closure','dropped_object','footsteps','dishes','notification','alarm','water','fan']
    for family,name in enumerate(families):
        for item in range(16):
            rng=np.random.default_rng([8921,family,item]);b=bed(rng,8,typical)
            wet=signal.fftconvolve(nuisance(family,item),h)
            event=np.zeros(8*FS);event[2*FS:2*FS+len(wet)]=wet
            level=(45.,55.,65.)[item%3]
            audio=nn.mix_device_audio(b,event,(2*FS,2*FS+len(wet)),rng,dict(nn.DEFAULTS,event_spl_db=[level,level]))
            dst=out/f'nuisance_{family}_{item}.wav';wavfile.write(dst,FS,audio.astype(np.float32))
            bank.append(dict(audio=dst.name,kind='nuisance',family=name,source_group=f'{name}:{item}',evidence_kind='model-based'))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for row in bank:row['sha256']=hashlib.sha256((out/row['audio']).read_bytes()).hexdigest()
    (out/'heldout.json').write_text(json.dumps(bank,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(evidence_kind='proxy result',pyroomacoustics=pra.__version__,
        source_seed=20261008,train_groups=list(range(18)),dev_groups=[18,19],heldout_groups=list(range(20,30)),
        third_party_audio=False,room=dict(dimensions=dims,mic=mic,source=src,absorption=.3),
        truth='PRA ISM order 30 + session_sim clean PCM captures; assumed source population'),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--bed-dba',type=float,nargs=3,required=True,metavar=('QUIET','TYPICAL','LOUD'),
                        help='background levels in dBA, from a source you may use (none ship)')
    args=parser.parse_args()
    prepare(args.out,args.bed_dba)
