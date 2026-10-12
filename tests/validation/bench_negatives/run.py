"""Historical label-independence check; NOT a WP3(b) path acceptance result.

Run as a script, with output and all licence-lane paths supplied explicitly.
No downloads, third-party audio, trained checkpoints or user data are required.
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

import numpy as np
from scipy.io import wavfile
import yaml

from sensorsim import edge, nn
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.iot_mic import PROFILES
from sensorsim.negatives.acceptance import symmetry, training_comparison_plan
from sensorsim.negatives.clips import load_clip
from sensorsim.negatives.composer import Event, RenderPath, render_context
from sensorsim.negatives.procedural import Source


def licence_controls(out: Path, module: Path, register: Path):
    spec = importlib.util.spec_from_file_location('wp9_licensing', module)
    licensing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(licensing)
    original_register = yaml.safe_load(register.read_text())
    results = []
    root = out/'licence-fixtures'
    root.mkdir()
    review = dict(family='timer', source_group='generated-control', acoustic_stage='captured',
                  listened=True, target_free=True, rights_cleared=True,
                  capture_device='fixture-recorder',profile_id='external/1',deployed_device=False,
                  reviewer='fixture only', notes='Project-generated tone; no real asset licence claim.')
    for k, licence in enumerate(('user-owned', 'CC0-1.0', 'CC-BY-4.0', 'CC-BY-NC-4.0')):
        name = f'fixture-{k}.wav'
        wavfile.write(root/name, 16000, (.02*np.sin(np.arange(16000)*.3)).astype(np.float32))
        h = hashlib.sha256((root/name).read_bytes()).hexdigest()
        registry_id = 'user-test' if licence=='user-owned' else 'freesound-cc0' if licence=='CC0-1.0' else 'freesound-cc-by-4'
        item = dict(path=name, sha256=h, registry_id=registry_id, license_id=licence,
                    verified_on='2026-10-08', owner='fixture owner', ownership_statement='Generated here.',
                    freesound_id='123', uploader='planted-test-fixture',
                    source_url='https://freesound.org/sounds/123/', modifications='test signal only',
                    license_url='https://creativecommons.org/publicdomain/zero/1.0/legalcode.en' if licence=='CC0-1.0'
                        else 'https://creativecommons.org/licenses/by/4.0/legalcode.en')
        registry = json.loads(json.dumps(original_register))
        registry['assets'].append(dict(id='user-test',name='Generated fixture',url='https://example.test/ownership', license_id='user-owned', version_or_hash=h,
            checked_on='2026-10-08', evidence_quote='Generated test fixture, not external audio.', disposition='plug-in'))
        # WP9 validates the full register and its external schema excludes user rows.
        # Only the user fixture needs its item-level ownership register extension.
        use_register = register
        if licence=='user-owned':
            use_register = root/'user-register.yaml'
            use_register.write_text(yaml.safe_dump(registry))
        variants = [('ordinary', item, review, use_register)]
        if licence=='user-owned':
            variants += [('missing-register', item | {'registry_id':'missing'}, review, use_register),
                         ('hash-mismatch', item | {'sha256':'0'*64}, review, use_register),
                         ('unreviewed', item, review | {'listened':False}, use_register)]
            internal = root/'internal-register.yaml'
            registry['assets'][-1]['disposition'] = 'internal'
            internal.write_text(yaml.safe_dump(registry))
            variants.append(('internal-only',item,review,internal))
        for variant, test_item, test_review, test_register in variants:
            wanted = variant=='ordinary' and licence!='CC-BY-NC-4.0'
            try:
                loaded = load_clip(root, test_item, register_path=test_register, review=test_review,
                                   as_of=date(2026,10,8), register_gate=licensing.validate_commercial_manifest)
                accepted, error = True, None
                assert loaded.provenance['register_sha256']
            except ValueError as exc:
                accepted, error = False, str(exc)
            results.append(dict(license_id=licence, control=variant, expected_accept=wanted,
                                accepted=accepted, correct=accepted==wanted, error=error))
    return {'result':'pass' if all(row['correct'] for row in results) else 'fail',
            'evidence_label':'proxy; generated fixtures, planted metadata',
            'controls':results, 'accepted_registered':sum(r['accepted'] for r in results),
            'accepted_total':sum(r['accepted'] for r in results),
            'gate_source_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
            'register_sha256':hashlib.sha256(register.read_bytes()).hexdigest()}


def probes(out, seed):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = nn.build_model(3).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.cpu().numpy().tobytes())
    model_hash = h.hexdigest()
    features = edge.Features(nn.DEFAULTS)
    captured = []
    hook = model.fc.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
    path = RenderPath(RoomModel(shoebox_dims_m=(6,5,3), late_t60_s=(.3,)*7), (4.,3.,1.5),
                      PROFILES['pi5_inmp441/1'], 'pi5_inmp441/1')
    positive, negative, record = [], [], []
    positions = ((1.,1.,1.), (2.,2.,1.), (3.,1.,1.5))
    for group in range(24):
        rng=np.random.default_rng([seed,group])
        x=rng.standard_normal(16000)
        if group%3==1: x=np.sin(2*np.pi*(400+group*31)*np.arange(16000)/16000)
        if group%3==2: x*=np.exp(-np.arange(16000)/1600)
        source=Source(x,'neutral',f'neutral:{seed}:{group}',{'acoustic_stage':'dry','fs':16000,
                      'license_id':'Apache-2.0','evidence_label':'proxy'})
        event=Event(source,1.,positions[group%3],70.)
        render_seed=int(np.random.SeedSequence([seed,group,1]).generate_state(1)[0])
        p=render_context([replace(event,label='neutral-positive')],path=path,seed=render_seed)
        n=render_context([replace(event,label='background')],path=path,seed=render_seed)
        positive.append(p.audio[16000:32000]); negative.append(n.audio[16000:32000])
        record.append({'group':group, 'seed':render_seed, 'positive_sha256':p.provenance['audio_sha256'],
                       'negative_sha256':n.provenance['audio_sha256'], 'path_sha256':p.provenance['path_sha256'],
                       'max_absolute_sample_difference':float(np.max(abs(p.audio-n.audio)))})
    with torch.inference_mode():
        pscore=torch.sigmoid(model(torch.from_numpy(features(np.asarray(positive))))).numpy()
        nscore=torch.sigmoid(model(torch.from_numpy(features(np.asarray(negative))))).numpy()
    hook.remove()
    result=symmetry(pscore,nscore,captured[0],captured[1],np.arange(24),seed=seed)
    result.update(seed=seed, model_sha256=model_hash, model_status='frozen untrained project CNN; not B',
                  waveform_pairs=record, embedding_dimension=captured[0].shape[1])
    np.savez(out/f'probe-{seed}.npz',positive_scores=pscore,negative_scores=nscore,
             positive_embeddings=captured[0],negative_embeddings=captured[1],groups=np.arange(24))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preregistration',type=Path,required=True)
    parser.add_argument('--prereg-sha256',required=True)
    parser.add_argument('--w9-module',type=Path,required=True)
    parser.add_argument('--register',type=Path,required=True)
    args=parser.parse_args()
    raw=args.preregistration.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.prereg_sha256: raise ValueError('preregistration hash mismatch')
    prereg=yaml.safe_load(raw)
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    result={'preregistration_sha256':args.prereg_sha256,'evidence_label':'proxy; no measured deployment audio',
            'label_independence_only':[probes(args.out,seed) for seed in prereg['seeds']],
            'wp3_c':licence_controls(args.out,args.w9_module,args.register),
            'wp3_a':training_comparison_plan(split_sha256='0'*64)}
    result['wp3_a']['split_sha256']=None
    result['wp3_a']['reason']+='; lock actual integrated audit before running'
    import sensorsim.negatives
    source_root=Path(sensorsim.negatives.__file__).parent
    result['implementation_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(source_root.glob('*.py'))}
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['elapsed_s']=time.monotonic()-start
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='label_independence_only'},indent=2))
    print(json.dumps([{'seed':r['seed'],'structural_result':r['structural_result'],
                       'wp3_b_result':r['wp3_b_result'],'auc':r['auc'],'auc_ci95':r['auc_ci95'],
                       'mean_score_differences':r['mean_score_differences']} for r in result['label_independence_only']],indent=2))


if __name__=='__main__': main()
