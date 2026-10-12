"""WP4 F1/F2 engineering acceptance, synthetic proxies only; no external assets.

Freeze a registration before invoking acoustic(). The legacy command runs under
both code revisions; its output is compared independently by the job runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
import yaml



def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def survey(seed, count=32):
    from sensorsim.acoustics.room.prior import RoomSurvey
    return RoomSurvey.model_validate(dict(
        room=dict(length_x=5, width_y=4, height_z=3),
        microphone=dict(x=3, y=2, z=1.4, tolerance_m=0),
        loudspeaker=dict(x=1, y=1, z=1),
        source_sites={'knock': [dict(x=.02, y=1, z=1.25, spread_m=[0,.2,.35])],
                      'doorbell': [dict(x=2, y=2, z=2.5)]},
        prior=dict(seed=seed, count=count)))


def write_playback(root, s, seed, latency_ms=0., anchored=False, compensate=False, seconds=30):
    """A .45 s room; strip known renderer FIR latency before adding test latency."""
    from sensorsim.acoustics.room.prior import build_prior
    from sensorsim.acoustics.room.synth import synthesize
    fs = 16000
    fixed = s.model_copy(update={'prior': s.prior.model_copy(update={'t30_s': ((.45,.45),)*7})})
    truth = build_prior(fixed).members[0]
    rir = synthesize(truth, s.loudspeaker.xyz, s.microphone.xyz, fs, seed=seed)
    h = rir.samples[rir.metadata['latency_samples']:]
    rng = np.random.default_rng(seed)
    x = rng.normal(size=fs*seconds); x *= .1/max(abs(x))
    y = np.r_[np.zeros(round((2+latency_ms/1000)*fs)), signal.fftconvolve(x, h), np.zeros(fs)]
    y += rng.normal(0, 1e-8, len(y))
    for folder in ('played', 'room_recordings'):
        (root/folder).mkdir()
    wavfile.write(root/'played'/'noise.wav', fs, x.astype('float32'))
    wavfile.write(root/'room_recordings'/'noise__room.wav', fs, y.astype('float32'))
    settings = {'recording_offsets_s': {'noise.wav': 2+(latency_ms/1000 if compensate else 0)},
                'recording_offsets_hardware_anchored': {'noise.wav': anchored},
                'holdout_groups': [], 'evidence_label': 'proxy'}
    (root/'calibration.yaml').write_text(yaml.safe_dump(settings))
    return {'direct_delay_s': rir.metadata['direct_delay_s'], 'added_latency_ms': latency_ms,
            'removed_renderer_latency_samples': rir.metadata['latency_samples']}


def write_project(root, seed, calibrated):
    root.mkdir(parents=True)
    s = survey(seed)
    (root/'room.yaml').write_text(yaml.safe_dump(s.model_dump(mode='json')))
    (root/'calibration.yaml').write_text('evidence_label: proxy\n')
    if calibrated:
        write_playback(root, s, seed, latency_ms=4, anchored=False)
    fs = 16000
    for label, f in [('knock', 700), ('doorbell', 2000)]:
        folder = root/'targets'/label; folder.mkdir(parents=True)
        for i in range(3):
            t = np.arange(fs//4)/fs
            audio = .02*np.sin(2*np.pi*(f+i*17)*t)*np.exp(-t*10)
            wavfile.write(folder/f'{i}.wav', fs, audio.astype('float32'))
    folder = root/'background'; folder.mkdir()
    for i in range(3):
        wavfile.write(folder/f'{i}.wav', fs,
            np.random.default_rng(seed+i).normal(0,.0003,fs*3).astype('float32'))
    (root/'project.yaml').write_text(yaml.safe_dump(dict(
        dataset=dict(count=8, fs=fs, seed=seed),
        recognizer=dict(type='cnn', steps=3, width=8, blocks=2, batch=8,
                        workers=0, threads=2, val_windows=16))))


def acoustic(out):
    from sensorsim.acoustics.room.prior import RoomEnsemble, build_prior
    from sensorsim.acoustics.room.pipeline import run_no_sweep
    from sensorsim import project
    from sensorsim.acoustics.room.render_prior import DOMESTIC_T30_S, select_render_ensemble
    from sensorsim.acoustics.room.dataset import DryClip, SamplingSpec, generate_dataset
    from sensorsim.acoustics.room.augment import AugmentationConfig
    out.mkdir(parents=True)
    result = {'label': 'proxy', 'offset_matrix': [], 'compensated_controls': [],
              'floor': [], 'projects': []}
    for seed in (4108, 4109, 4110):
        s = survey(seed)
        for anchored in (False, True):
            for latency in (0, 2, 4, 8):
                root = out/f'offset-{seed}-{anchored}-{latency}'; root.mkdir()
                (root/'room.yaml').write_text(yaml.safe_dump(s.model_dump(mode='json')))
                truth = write_playback(root, s, seed, latency, anchored)
                r = run_no_sweep(root/'room.yaml', root/'fit', fs=16000, seed=seed)
                expected = 'prior-conflict' if anchored and latency > 2 else 'known-source'
                conflicts = {k for e in r['evidence'] for k in e.get('conflicts', {})}
                ok = r['calibration_status'] == expected
                if expected == 'prior-conflict':
                    ok &= conflicts == {'direct_delay_s'} and any('hardware anchor' in w for w in r['warnings'])
                result['offset_matrix'].append(dict(seed=seed, anchored=anchored, latency_ms=latency,
                    status=r['calibration_status'], expected=expected, passed=bool(ok), truth=truth,
                    conflicts=sorted(conflicts), evidence=r['evidence']))
        for latency in (4, 8):
            root = out/f'compensated-{seed}-{latency}'; root.mkdir()
            (root/'room.yaml').write_text(yaml.safe_dump(s.model_dump(mode='json')))
            write_playback(root, s, seed, latency, True, compensate=True)
            r = run_no_sweep(root/'room.yaml', root/'fit', seed=seed)
            result['compensated_controls'].append(dict(seed=seed, latency_ms=latency,
                status=r['calibration_status'], passed=r['calibration_status']=='known-source'))
        broad = build_prior(survey(seed, count=512))
        domestic = select_render_ensemble(broad)
        values = np.array([m.late_t60_s for m in domestic.members])
        clips = [DryClip('knock', np.random.default_rng(seed+i).normal(0,.01,4000), 16000,
                         source_group=f'proxy-{i}') for i in range(3)]
        manifest = generate_dataset(broad, SamplingSpec(), clips, AugmentationConfig(), 64, seed,
                                    out/f'floor-{seed}', fs=16000)
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        lookup = {m.model_hash:m for m in domestic.members}
        drawn = [lookup[r['ensemble_member_hash']].late_t60_s[3] for r in rows]
        ok = bool(np.all(values >= np.array(DOMESTIC_T30_S)[:,0]) and
                  np.all(values <= np.array(DOMESTIC_T30_S)[:,1]) and min(drawn) >= .25)
        result['floor'].append(dict(seed=seed, passed=ok, members=len(values), rendered_rows=len(rows),
            assumption_min_by_band_s=values.min(axis=0).tolist(), assumption_max_by_band_s=values.max(axis=0).tolist(),
            rendered_draw_1khz_min_s=min(drawn), below_floor=sum(v < .25 for v in drawn),
            calibration_support_below_015_fraction=float(np.mean([m.late_t60_s[3]<.15 for m in broad.members]))))
        for calibrated in (False, True):
            root = out/f'project-{seed}-{calibrated}'
            write_project(root, seed, calibrated)
            report = project.run(root)
            fit = json.loads((root/'out/fit/report.json').read_text())
            ens = RoomEnsemble.model_validate_json((root/'out/fit/room_ensemble.json').read_text())
            calibration = RoomEnsemble.model_validate_json((root/'out/fit/calibration_ensemble.json').read_text())
            rows = [json.loads(line) for line in (root/'out/dataset/manifest.jsonl').read_text().splitlines()]
            target_rows = [r for r in rows if 'ensemble_member_hash' in r]
            expected = 'known-source' if calibrated else 'assumed'
            ok = (fit['calibration_status']==expected and fit['observation_label']=='proxy' and
                  report['train']['status']=='ran' and len(target_rows)>0 and
                  all(r['ensemble_hash']==ens.ensemble_hash for r in target_rows))
            if calibrated:
                ok &= ens.model_dump()==calibration.model_dump()
            else:
                ok &= min(m.late_t60_s[3] for m in ens.members)>=.25
                ok &= 'ROOM IS ASSUMED, NOT MEASURED' in (root/'out/fit/report.md').read_text()
                ok &= any('ROOM IS ASSUMED, NOT MEASURED' in w for w in report['fit']['warnings'])
            result['projects'].append(dict(seed=seed, calibrated=calibrated, passed=bool(ok),
                status=fit['calibration_status'], stages={k:v['status'] for k,v in report.items()},
                target_rows=len(target_rows), background_rows=len(rows)-len(target_rows),
                rendering_distribution=fit['rendering_distribution'],
                posterior_unchanged=ens.model_dump()==calibration.model_dump(),
                observation_label=fit['observation_label']))
        dump(out/'acceptance.json', result)
    result['acceptance'] = 'pass' if all(r['passed'] for key in (
        'offset_matrix', 'compensated_controls', 'floor', 'projects') for r in result[key]) else 'fail'
    dump(out/'acceptance.json', result)
    print(json.dumps({k:v if not isinstance(v,list) else {'cases':len(v),'passed':sum(r['passed'] for r in v)}
                      for k,v in result.items()}, indent=2))


def legacy(out):
    """Exercise the old sweep pipeline's real dataset stage, including the late tail."""
    from sensorsim.acoustics.room import pipeline
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.source_reference import SourceReference
    from sensorsim.acoustics.room.synth import SynthConfig
    for seed, count, late in ((1,20,True), (2,20,True), (3,200,False)):
        root = out/str(seed); (root/'dry').mkdir(parents=True)
        rows=[]
        rng=np.random.default_rng(732)
        for i,label in enumerate(['knock']*3+['doorbell']*3):
            file=f'dry/{i}.wav'
            wavfile.write(root/file,8000,rng.normal(0,.02,256).astype('float32'))
            rows.append(dict(label=label,take=i,channel=0,dataset_source=file,
                source_reference=SourceReference().as_dict()))
        cfg=pipeline.PipelineConfig(count=count,seed=seed,
            synthesis=SynthConfig() if late else SynthConfig(late=False,max_order=0))
        summary=pipeline._dataset(RoomModel(shoebox_dims_m=(6,5,3)), rows, root, cfg, 8000)
        dump(root/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['acoustic','legacy']);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    (acoustic if args.mode=='acoustic' else legacy)(args.out)
