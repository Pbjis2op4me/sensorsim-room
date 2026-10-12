# SPDX-License-Identifier: Apache-2.0
"""Prepare procedural TRUTH/native and optional CC BY MeshRIR audit inputs.

All outputs are model-based proxy results and belong outside the repository.
No external audio is downloaded. The optional archive is read in place, requires
its local CC BY 4.0 licence record, and is for internal validation only.
"""
from pathlib import Path
import argparse
import re
import hashlib, io, itertools, json, zipfile
import numpy as np
from scipy import signal
from scipy.io import wavfile
from sensorsim import iot_mic
from sensorsim.acoustics.room.model import RoomModel, uniform_bands
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from sensorsim.acoustics.room.session_sim import canonical_kit, simulate_session, Impairments, broadband_transient
from sensorsim.acoustics.weighting import a_weighted_mean_square


def caller_levels(values, name, count=None):
    """Background levels (dBA) the caller supplies from a source they may use; no measured levels ship."""
    levels = [float(v) for v in (values or ())]
    if not levels or (count is not None and len(levels) != count) or not all(np.isfinite(levels)):
        need = f"{count} " if count else ""
        raise ValueError(f"{name}: supply {need}A-weighted background levels in dBA from a source you are "
                         "allowed to use; this project ships no measured household levels")
    return levels


def truth_rir(room):
    """PRA's ISM amplitude is already 1/d (no 4*pi compensation)."""
    return np.asarray(room.rir[0][0], dtype=float)


def require_unclipped(x, name):
    if not np.isfinite(x).all() or np.max(np.abs(x)) >= 32767/32768:
        raise ValueError(f'proxy input QA: clipped/nonfinite {name}')


def prepare(output, archive=None, licence=None, floor_dba=None):
    import pyroomacoustics as pra
    output=Path(output)
    if output.exists():
        raise FileExistsError('use a fresh proxy output directory')
    if archive is not None:
        archive, licence=Path(archive),Path(licence)
        record=licence.read_text().lower()
        allowed = re.search(r'(?:^attribution 4\.0 international$|creativecommons\.org/licenses/by/4\.0/?|\bcc[- ]by[- ]4\.0\b)', record, re.MULTILINE)
        denied = any(term in record for term in ('noncommercial', 'non-commercial', 'sharealike', 'share-alike', 'by-nc', 'by-sa'))
        if not allowed or denied:
            raise ValueError('expected the MeshRIR CC BY 4.0 local licence record')
    floor_dba=caller_levels(floor_dba,'floor_dba')
    output.mkdir(parents=True)
    out=output/'inputs';out.mkdir()
    fs=16000; dims=(5.,4.,2.7); mic=(1.,2.,1.2)
    kit=canonical_kit().model_copy(update=dict(fs_hz=fs,f2_hz=7000.,sweep_s=.3,fade_in_s=.01,
     fade_out_s=.01,lead_silence_s=.1,tail_silence_s=.2,end_silence_s=.1,repeats=1,period_s=.5,first_sweep_start_s=.1))
    sources={}
    rng=np.random.default_rng(20261008)
    t=np.arange(round(2.4*fs))/fs
    knock=broadband_transient(fs,6001)
    # A single short session-simulator impact, two-tone chime, evolving filtered noise.
    # Shapes are diagnostic surrogates, not measured target populations.
    sources['knock']=knock
    sources['doorbell']=(np.sin(2*np.pi*660*t)*np.exp(-t*2)+.6*np.sin(2*np.pi*880*t)*np.exp(-t*3))*(1-np.exp(-t*200))
    sources['kettle']=signal.sosfilt(signal.butter(2,200,fs=fs,btype='highpass',output='sos'),rng.normal(size=t.size))*(.2+.8*t/t[-1])*np.minimum(1,t/.02)*np.minimum(1,(t[-1]-t)/.03)
    for k,x in sources.items():
     sources[k]=x/(np.sqrt(np.mean(x*x))*np.sqrt(2))

    floor_files={}
    for dba in floor_dba:
     f=signal.lfilter([1],[1,-.95],rng.normal(size=6*fs))
     # Nominal SPL -> digital A-weighted RMS, sine reference, -26 dBFS at 94 SPL.
     f*=10**((dba-120-10*np.log10(2))/20)/np.sqrt(a_weighted_mean_square(f,fs))
     floor_files[dba]=out/f'floor_{dba}.wav'
     f=iot_mic.PROFILES['pi5_inmp441/1'].apply(f,fs,rng,input_reference_spl_db=120.)
     wavfile.write(floor_files[dba],fs,f.astype(np.float32))

    cases=[]; rirs=[]
    for alpha,distance in itertools.product((.15,.55),(1.,3.)):
     src=(1.+distance,2.,1.2)
     room=RoomModel(shoebox_dims_m=dims,absorption=uniform_bands(alpha))
     truth=pra.ShoeBox(dims,fs=fs,materials=pra.Material(alpha),max_order=30)
     truth.add_source(src); truth.add_microphone_array(np.array(mic)[:,None]); truth.compute_rir()
     native=synthesize(room,src,mic,fs,seed=8801,config=SynthConfig(duration_s=1.2))
     # Installed PRA ISM already applies 1/d. No extra 4*pi factor.
     htruth=truth_rir(truth)
     for engine,h in [('sensorsim',native.samples),('TRUTH',htruth)]:
      for spl in (50,80):
       dry={f'S1_M1_real_{k}.wav':x*10**((spl-120)/20) for k,x in sources.items()}
       session=None
       if engine=='TRUTH':
        tag=f'truth_a{alpha}_d{distance}_s{spl}'
        session=simulate_session(out/tag,room,{'S1':src},{'M1':mic},truth_dir=out/(tag+'_truth'),
          kit=kit,impairments=Impairments.clean(fs,recording_gain=1.),external_rirs={('S1','M1'):h},
          real_sources=dry,include_reference=False,noise_seconds=2.,seed=8801)
       for label in sources:
        ident=f'{engine}_{label}_a{alpha}_d{distance}_s{spl}'
        wet=signal.fftconvolve(dry[f'S1_M1_real_{label}.wav'],h)
        require_unclipped(wet,ident)
        ideal=out/(ident+'_ideal.wav'); wavfile.write(ideal,fs,wet.astype(np.float32))
        if session:
         target=out/tag/f'S1_M1_real_{label}.wav'
        else:
         target=out/(ident+'.wav'); wavfile.write(target,fs,np.rint(wet*32768).astype(np.int16))
        for dba,floor in floor_files.items():
         cases.append(dict(id=ident+f'_f{dba}',source_group=f'procedural-{label}-a{alpha}-d{distance}',
          label=label,evidence_kind='model-based',proxy=engine,source_spl_db=spl,distance_m=distance,
          room=f'alpha={alpha}',floor_dba=dba,target=str(target.relative_to(output)),
          ideal_target=str(ideal.relative_to(output)),floor=str(floor.relative_to(output))))

    if archive is not None:
        with zipfile.ZipFile(archive) as z:
         meta=json.loads(z.read('S1-M3969_npy/data.json'))
         for index in (0,3968):
          member=f'S1-M3969_npy/ir_{index}.npy'; raw=z.read(member)
          h=signal.resample_poly(np.load(io.BytesIO(raw),allow_pickle=False).reshape(-1),1,3)
          pos=np.array([meta[f'ir_{index}']['pos'][a] for a in 'xyz'])
          distance=float(np.linalg.norm(pos-np.array([2.,1.5,0.])))
          # Capture gain is unknown: explicit assumed direct-peak normalization, not SPL calibration.
          h=h/(np.max(np.abs(h))*distance)
          rirs.append(dict(member=member,sha256=hashlib.sha256(raw).hexdigest(),distance_m=distance))
          for spl,label in itertools.product((50,80),sources):
           wet=signal.fftconvolve(sources[label]*10**((spl-120)/20),h)
           ident=f'MEAS_{index}_{label}_s{spl}'
           require_unclipped(wet,ident)
           ideal=out/(ident+'_ideal.wav'); target=out/(ident+'.wav')
           wavfile.write(ideal,fs,wet.astype(np.float32)); wavfile.write(target,fs,np.rint(wet*32768).astype(np.int16))
           for dba,floor in floor_files.items():
            cases.append(dict(id=ident+f'_f{dba}',source_group=f'meshrir-{index}-{label}',label=label,
             evidence_kind='model-based',proxy='MEAS',source_spl_db=spl,distance_m=distance,
             room='MeshRIR S1-M3969; absorption unknown',floor_dba=dba,
             target=str(target.relative_to(output)),ideal_target=str(ideal.relative_to(output)),
             floor=str(floor.relative_to(output))))
    config=dict(cases=cases,seeds=[1701,1702,1703],snr_db=[-5,3,20],event_spl_db=[60,70,80],
     profiles=['default_pi','stress'],augmentations=['off','spec','mixstyle','spec_mixstyle'])
    (output/'sweep_config.json').write_text(json.dumps(config,indent=2)+'\n')
    qa=[]
    for case in cases:
     for field in ('target','floor'):
      rate,a=wavfile.read(output/case[field])
      if np.issubdtype(a.dtype,np.integer):a=a.astype(float)/(-np.iinfo(a.dtype).min)
      require_unclipped(a,case[field])
      qa.append(dict(file=case[field],peak=float(np.max(np.abs(a))),clipped_samples=int(np.sum(np.abs(a)>=32767/32768))))
    (output/'input_provenance.json').write_text(json.dumps(dict(pyroomacoustics=pra.__version__,
     measured_rirs=rirs,licence_sha256=hashlib.sha256(licence.read_bytes()).hexdigest() if licence else None,
     measured_capture_count=0,all_results='proxy result',source_models='diagnostic procedural; no population validation',
     truth='PRA order 30, native 1/d pressure; session_sim clean capture and PCM dither',
     floor='assumed room spectra plus Pi response and self-noise once, saved float32',
     mesh='internal validation only; RIR normalized by peak*distance; unknown capture gain',
     case_count=len(cases),input_qa=qa),indent=2)+'\n')
    print('prepared',len(cases),'proxy cases')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--mesh-archive',type=Path)
    parser.add_argument('--mesh-licence',type=Path)
    parser.add_argument('--floor-dba',type=float,nargs='+',required=True,
                        help='device-floor levels in dBA, from a source you may use (none ship)')
    args=parser.parse_args()
    prepare(args.out,args.mesh_archive,args.mesh_licence,args.floor_dba)
