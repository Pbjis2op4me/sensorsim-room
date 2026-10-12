# SPDX-License-Identifier: Apache-2.0
"""Prepare exact WAV auditions and bind explicit listening decisions to content.

Run ``python -m sensorsim.negatives.audition --help``. Only decisions.json is
edited after preparation. A receipt is a listener's scoped declaration, not an
independent scientific acceptance. Tonal recall and split/licence admission
remain separate gates. Proxy references cannot authorize recorded-family claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.io import wavfile

from .composer import audio_hash
from .procedural import IMPACTS, TONES


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _audio(path):
    from sensorsim.recognizer import read_audio
    path = Path(path).resolve()
    fs, raw = wavfile.read(path)
    if raw.ndim != 1 or not 8000 <= fs <= 96000:
        raise ValueError('audition requires mono WAV at 8000..96000 Hz; no implicit conversion')
    x = read_audio(path, fs)
    if len(x) < 2 or not np.isfinite(x).all() or np.ptp(x) == 0 or np.max(abs(x)) > 1:
        raise ValueError('audition audio is empty, constant, nonfinite or outside full scale')
    return dict(file=str(path), file_sha256=_sha(path), audio_sha256=audio_hash(x),
                sample_rate_hz=int(fs), samples=len(x))


def prepare(manifest, folder):
    """Copy complete WAVs unchanged; reject ambiguous identity and mismatched units."""
    manifest, folder = Path(manifest).resolve(), Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(f'audition output already exists: {folder}')
    data = json.loads(manifest.read_text())
    if data.get('schema') != 'negative-audition-input/1' or not data.get('rows'):
        raise ValueError('nonempty negative-audition-input/1 manifest required')
    rows, ids, hashes = [], set(), set()
    for entry in data['rows']:
        item = entry.get('item_id')
        family = entry.get('family')
        if not isinstance(item, str) or not item.strip() or item in ids or family not in IMPACTS+TONES:
            raise ValueError('unique item_id and a gated impact/tonal family required')
        candidate = _audio(manifest.parent/entry['file'])
        if candidate['audio_sha256'] in hashes:
            raise ValueError('duplicate candidate content')
        ids.add(item); hashes.add(candidate['audio_sha256'])
        reference = None
        if entry.get('reference_file'):
            if (entry.get('reference_kind') != 'recorded_family'
                    or entry.get('reference_evidence_domain') != 'deployment recording'
                    or entry.get('reference_license_id') not in ('user-owned', 'CC0-1.0', 'CC-BY-4.0')):
                raise ValueError('reference must be a declared licence-cleared recorded family, not a proxy')
            reference = _audio(manifest.parent/entry['reference_file'])
            if candidate['sample_rate_hz'] != reference['sample_rate_hz']:
                raise ValueError('candidate/reference sample-rate units mismatch')
            if candidate['audio_sha256'] == reference['audio_sha256']:
                raise ValueError('identical candidate and reference cannot establish family realism')
            reference.update(kind='recorded_family', evidence_domain='deployment recording',
                             license_id=entry['reference_license_id'])
        rows.append(dict(item_id=item, family=family, candidate=candidate, reference=reference))
    folder.mkdir(parents=True)
    (folder/'clips').mkdir()
    for i, row in enumerate(rows):
        for key in ('candidate', 'reference'):
            if row[key] is not None:
                name=f'clips/{i:04d}-{key}.wav'
                shutil.copyfile(row[key]['file'], folder/name)
                row[key]['audition_file']=name
    plan=dict(schema='negative-audition-plan/1', input_sha256=_sha(manifest), rows=rows,
              scope='Listener declaration only; independent acceptance, licences, split and tonal recall remain separate.')
    _write(folder/'plan.json', plan)
    _write(folder/'decisions.json', dict(schema='negative-audition-decisions/1',
        plan_sha256=_sha(folder/'plan.json'), reviewer='', rows=[dict(item_id=r['item_id'],
        decision='pending', reviewed_label_boundary=False, notes='') for r in rows]))
    (folder/'README.txt').write_text(
        'Listen to every complete candidate WAV beside its reference. Files are copied without gain changes.\n'
        'Edit decisions.json only: reviewer, approve/reject, reviewed_label_boundary and notes.\n'
        'Approval asserts this family is plausible and contains no target event. Missing recorded-family\n'
        'references cannot be approved. Add them to the input manifest and prepare a fresh folder.\n'
        'Run: python -m sensorsim.negatives.audition finalize THIS_FOLDER\n'
        'Tonal receipts do not replace held-out doorbell recall or partial-exclusion checks.\n')
    return dict(folder=str(folder), items=len(rows), pending=len(rows))


def finalize(folder):
    """Recheck original and audition bytes before writing any decisions/receipts."""
    folder=Path(folder).resolve()
    plan_path=folder/'plan.json'
    decisions=json.loads((folder/'decisions.json').read_text())
    if (decisions.get('schema') != 'negative-audition-decisions/1'
            or decisions.get('plan_sha256') != _sha(plan_path)):
        raise ValueError('audition plan hash changed; prepare again')
    plan=json.loads(plan_path.read_text())
    if plan.get('schema') != 'negative-audition-plan/1' or not plan.get('rows'):
        raise ValueError('missing audition plan rows')
    reviewer=decisions.get('reviewer')
    if not isinstance(reviewer,str) or not reviewer.strip():
        raise ValueError('explicit listening reviewer required; pending review cannot authorize audio')
    choices=decisions.get('rows',[])
    if (len(choices) != len(plan['rows']) or len({r['item_id'] for r in choices}) != len(choices)
            or {r['item_id'] for r in choices} != {r['item_id'] for r in plan['rows']}):
        raise ValueError('decisions must cover every prepared item exactly once')
    by_id={r['item_id']:r for r in choices}
    for row in plan['rows']:
        decision=by_id[row['item_id']]
        if decision.get('decision') not in ('approve','reject'):
            raise ValueError('pending or unknown listening decision')
        if decision['decision']=='approve' and (not row['reference'] or decision.get('reviewed_label_boundary') is not True):
            raise ValueError('approval requires recorded-family reference and reviewed target/negative boundary')
        for key in ('candidate','reference'):
            entry=row[key]
            if entry is None:
                continue
            audition=(folder/entry['audition_file']).resolve()
            if not audition.is_relative_to(folder):
                raise ValueError('audition path must stay inside the prepared folder')
            for path in (Path(entry['file']),audition):
                if _sha(path) != entry['file_sha256']:
                    raise ValueError(f'audition/original {key} file changed: {path}')
            current=_audio(entry['file'])
            if any(current[k] != entry[k] for k in ('audio_sha256','samples','sample_rate_hz')):
                raise ValueError('decoded audition identity changed')
    targets=[folder/'receipts.json', *(folder/f'decision-{i:04d}.json' for i in range(len(plan['rows'])))]
    if any(p.exists() for p in targets):
        raise FileExistsError('decisions already finalized; prepare a new folder for another review')
    output=[]
    for i,row in enumerate(plan['rows']):
        decision=by_id[row['item_id']]
        evidence=dict(schema='negative-listening-decision/1',plan_sha256=_sha(plan_path),
                      reviewer=reviewer, prepared=row, decision=decision)
        path=folder/f'decision-{i:04d}.json';_write(path,evidence)
        result=dict(item_id=row['item_id'],family=row['family'],decision=decision['decision'])
        if decision['decision']=='approve':
            result['realism_gate']=dict(kind='human_listening', decision='pass',
                audio_sha256=row['candidate']['audio_sha256'],family=row['family'],
                reference_kind='recorded_family',reviewer=reviewer,reviewed_label_boundary=True,
                evidence_file=path.relative_to(folder).as_posix(),evidence_sha256=_sha(path),
                reference_file=row['reference']['audition_file'],reference_sha256=row['reference']['file_sha256'])
        output.append(result)
    payload=dict(schema='negative-listening-receipts/1',rows=output,
                 scope='Paths are relative to this review folder. Place the training manifest here or rebase '
                       'the receipt paths relative to it. Attach realism_gate to the matching item provenance; '
                       'training still requires matching audio/splits/licences/room and tonal recall.')
    _write(folder/'receipts.json',payload)
    return payload


def generate(folder, *, seed=8100, target_partials_hz=()):
    """Write original proxy candidates for EVERY gated family; approvals stay pending."""
    from .procedural import generate as source
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i,family in enumerate(IMPACTS+TONES):
        candidate=source(family,fs=16000,duration_s=4.,seed=seed+i,target_partials_hz=target_partials_hz)
        x=np.asarray(candidate.audio);x=x*(.5/np.max(abs(x)))
        path=folder/f'{family}.wav';wavfile.write(path,16000,x.astype(np.float32))
        rows.append(dict(item_id=family,family=family,file=path.name,reference_file=None,
                         candidate_provenance=candidate.provenance))
    _write(folder/'input.json',dict(schema='negative-audition-input/1',rows=rows,
        evidence_domain='synthetic proxy',note='Unrendered proxy previews only; prepare actual receiver-domain '
        'training files before approving them. Add a recorded-family reference and its domain/licence to each row.'))
    return prepare(folder/'input.json',folder/'audition')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('generate',help='write previews for all gated families; no automatic approvals')
    p.add_argument('folder');p.add_argument('--seed',type=int,default=8100)
    p.add_argument('--target-partials',type=float,nargs='*',default=[])
    p=sub.add_parser('prepare',help='copy manifest candidate/reference WAVs and write editable decisions')
    p.add_argument('manifest');p.add_argument('folder')
    p=sub.add_parser('finalize',help='validate explicit decisions and unchanged audio; write receipts')
    p.add_argument('folder')
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':result=prepare(args.manifest,args.folder)
        elif args.command=='finalize':result=finalize(args.folder)
        else:result=generate(args.folder,seed=args.seed,target_partials_hz=args.target_partials)
        print(json.dumps(result,indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'audition stopped: {exc}',file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
