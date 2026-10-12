# SPDX-License-Identifier: Apache-2.0
"""Automatic engineering proxies and held-out development panels.

These are usability defaults, not measured household populations. A manifest
without room geometry uses an explicitly labelled direct-field proxy. A panel
without enough disjoint source parents cannot qualify model selection.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
from .composer import audio_hash, digest
from .procedural import generate
from .splits import Item, SplitPlan
from .mining import PoolItem
from .integration import LockedEventPool

UNGATED = ('hvac_fan', 'fridge_hum', 'running_tap', 'extractor_hood')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _renderers(manifest, rows, fs):
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import SynthConfig, synthesize
    result = []
    seen = set()
    for row in rows:
        if row['label'] == 'background':
            continue
        resolved = row.get('resolved_config', {})
        model = resolved.get('model')
        if row.get('ensemble_member_index') is not None:
            ensemble = json.loads((Path(manifest).parent/'ensemble.json').read_text())
            model = ensemble['members'][row['ensemble_member_index']]
        if not model:
            continue
        identity = digest([model, row['source_xyz_m'], row['receiver_xyz_m'], resolved.get('synthesis')])
        if identity in seen:
            continue
        seen.add(identity)
        room = RoomModel.model_validate(model)
        if room.model_hash != row['model_hash']:
            raise ValueError('automatic confuser room model hash mismatch')
        h = synthesize(room, row['source_xyz_m'], row['receiver_xyz_m'], fs,
                       seed=73017, config=SynthConfig.model_validate(resolved.get('synthesis', {}))).samples
        result.append((h, dict(room_hash=room.model_hash, source_xyz_m=row['source_xyz_m'],
                              receiver_xyz_m=row['receiver_xyz_m'], route='dataset room renderer')))
        if len(result) == 4:
            break
    return result or [(np.ones(1), dict(route='direct-field proxy; manifest has no room geometry', room_hash=None))]


def automatic_pool(manifest, corpus, cfg, folder):
    """Create a small original ungated pool and a real unused reserve, once."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    renderers = _renderers(manifest, corpus.rows, cfg['fs'])
    rows = []
    for fi, family in enumerate(UNGATED):
        for i in range(4):
            source = generate(family, fs=cfg['fs'], duration_s=4., seed=73000 + 100*fi + i)
            h, route = renderers[i % len(renderers)]
            x = signal.oaconvolve(source.audio*.005, h).astype(np.float32)
            name = f'{family}-{i}'
            wavfile.write(folder/(name+'.wav'), cfg['fs'], x)
            item = Item(name, family, 'automatic-negative-'+name, audio_hash(x))
            rows.append(PoolItem(item, x, cfg['fs'], dict(source.provenance,
                training_eligible=True, acoustic_stage='receiver', weight=1., room_route=route)))
    # Actual generated content, retained but never offered for training or tuning.
    reserve = generate('distant_traffic', fs=cfg['fs'], duration_s=4., seed=73999)
    wavfile.write(folder/'unused-reserve.wav', cfg['fs'], reserve.audio.astype(np.float32))
    item = Item('unused-reserve', 'distant_traffic', 'automatic-unused-reserve',
                audio_hash(reserve.audio.astype(np.float32)))
    plan = SplitPlan.create([r.item for r in rows]+[item], audit_families=['distant_traffic'],
                            development_families=[], seed=73000)
    plan.save(folder/'split.json')
    plan = SplitPlan.load(folder/'split.json', expected_sha256=plan.sha256)
    pool = LockedEventPool(rows, plan=plan, expected_sha256=plan.sha256, purpose='train',
                           max_procedural_share=cfg['max_procedural_share'])
    (folder/'provenance.json').write_text(json.dumps(dict(evidence_domain='synthetic proxy',
        automatic=True, routes=[r for _, r in renderers],
        reserve='Unused procedural reserve only; not a held-out deployment evaluation',
        metadata=pool.metadata()), indent=2)+'\n')
    return pool


def automatic_panel(manifest, train_c, val_c, cfg, folder):
    """Use disjoint validation parents once; never synthesize extra target groups.

    This development panel has no sealed evaluation allocation. It cannot emit
    an acceptance claim; allocation of real locked evaluation stays external.
    """
    from sensorsim import nn
    from sensorsim.recognizer import read_audio, _audible_span
    fs = cfg['fs']
    def identity(row):
        x = read_audio(Path(manifest).parent/row['file'], fs)
        h = audio_hash(x)
        parents = set(row.get('dry_source', {}).get(k) for k in
                      ('content_sha256', 'normalized_content_sha256')) - {None}
        group = row.get('source_group') or row.get('dry_source', {}).get('source_group')
        return x, h, parents, group
    train_ids = [identity(r) for r in train_c.rows]
    blocked_hashes = {h for _,h,_,_ in train_ids} | set().union(*(p for _,_,p,_ in train_ids))
    blocked_groups = {g for _,_,_,g in train_ids if g}
    if train_c.negative_pool is not None:
        blocked_groups.update(r.item.source_group for r in train_c.negative_pool.rows)
        blocked_hashes.update(r.item.sha256 for r in train_c.negative_pool.rows)
    selected = []; seen_groups = set(); seen_hashes = set()
    for row in val_c.rows:
        x,h,parents,group = identity(row)
        if row['label'] == 'background' and not group:
            group = 'background-content-'+h
        if (not group or group in blocked_groups or group in seen_groups
                or ({h} | parents) & (blocked_hashes | seen_hashes)
                or len(x) < fs or not np.isfinite(x).all() or np.ptp(x) == 0):
            continue
        selected.append((row,x,h,parents,group));seen_groups.add(group);seen_hashes.update({h}|parents)
    counts = {c:sum(r['label']==c for r,*_ in selected) for c in train_c.classes}
    backgrounds = [(x,g) for r,x,h,p,g in selected if r['label']=='background']
    if not backgrounds or any(n < cfg['selection_min_groups'] for n in counts.values()):
        return None, ('fixed detector: too few disjoint held-out groups for automatic selection '
                      f'(targets={counts}, backgrounds={len(backgrounds)}, required per class={cfg["selection_min_groups"]})')
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=False)
    rows = []
    def write(x, **row):
        x=np.asarray(x,np.float32); path=folder/f'context-{len(rows)}.wav';wavfile.write(path,fs,x)
        rows.append(dict(row,audio=path.name,file_sha256=sha(path),audio_sha256=audio_hash(x),
                         evidence_domain='synthetic proxy', role='development'))
    for i,(r,x,h,parents,g) in enumerate(selected):
        if r['label']=='background':
            write(x,kind='background',source_group=g,background_groups=[g],parent_audio_sha256=[h])
            continue
        bg,bg_group=backgrounds[i % len(backgrounds)]
        n=len(x)+2*fs;bed=np.resize(bg,n);event=np.zeros(n);event[fs:fs+len(x)]=x
        a,b=_audible_span(x,fs)
        heard=nn.mix_device_audio(bed,event,(fs+a,fs+b),np.random.default_rng(74000+i),cfg)
        write(heard,kind='target',label=r['label'],start_s=1+a/fs,end_s=1+b/fs,
              source_group=g,background_groups=[bg_group],parent_audio_sha256=sorted({h}|parents))
    # New procedural parents, held apart from the automatic training pool.
    renderers=_renderers(manifest, train_c.rows, fs)
    for i,family in enumerate(('door_closure','dropped_objects','footsteps_wood','dishes_cutlery',
                              'phone_notification','alarm_chirp','boiling_pan','vacuum')):
        source=generate(family,fs=fs,duration_s=4.,seed=75000+i)
        h,_=renderers[i % len(renderers)]
        x=signal.oaconvolve(source.audio*.005,h)
        bg,bg_group=backgrounds[i % len(backgrounds)];n=len(x)+2*fs;event=np.zeros(n);event[fs:fs+len(x)]=x
        heard=nn.mix_device_audio(np.resize(bg,n),event,(fs,fs+len(x)),np.random.default_rng(76000+i),cfg)
        write(heard,kind='nuisance',family=family,source_group='automatic-panel-'+family,
              background_groups=[bg_group],parent_audio_sha256=[audio_hash(source.audio)])
    payload=dict(schema='training-panel-auto/1',role='development',rows=rows,
                 note='Automatic development selection only; no sealed deployment evaluation allocated')
    path=folder/'panel.json';path.write_text(json.dumps(payload,indent=2)+'\n')
    return dict(payload,root=str(folder),sha256=sha(path)), payload['note']
