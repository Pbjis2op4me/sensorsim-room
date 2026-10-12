"""Time a registered fit on supplied RIRs.

Inputs stay external; legacy dataset timings are withdrawn.
The JSON RIR manifest route accepts a model object and observations with WAV
file, source_xyz_m and receiver_xyz_m fields; it never changes the fitter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile

from sensorsim.acoustics.room import fit
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.rir import RIR
from .run import verify_registration


def read_manifest(path, config):
    manifest = json.loads(path.read_text())
    model = RoomModel.model_validate(manifest['model'])
    observations, inputs = [], []
    for row in manifest['observations']:
        filename = path.parent/row['file']
        fs, samples = wavfile.read(filename)
        if samples.ndim != 1:
            raise ValueError('supply one mono RIR per manifest observation')
        if np.issubdtype(samples.dtype, np.signedinteger):
            samples = samples.astype(float)/(-np.iinfo(samples.dtype).min)
        elif samples.dtype == np.uint8:
            samples = (samples.astype(float)-128)/128
        rir = RIR(fs, samples.astype(float), tuple(row['source_xyz_m']),
                  tuple(row['receiver_xyz_m']), row.get('metadata', {}))
        observations.append(fit.observe(rir, id=row['file'], config=config))
        with filename.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        inputs.append(dict(file=row['file'], sha256=digest))
    return model, observations, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rir-manifest', type=Path, required=True)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--hash-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('preserve previous evidence')
    digest, _ = verify_registration(args.preregistration, args.hash_file)
    config = fit.FitConfig(bootstrap=args.bootstrap, seed=args.seed)
    model, observations, inputs = read_manifest(args.rir_manifest, config)
    wall, cpu = time.perf_counter(), time.process_time()
    result = fit.fit_room(observations, model, config=config)
    out = dict(label='supplied RIR benchmark; see registered input provenance',
               wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu,
               config=config.model_dump(mode='json'), inputs=inputs,
               model=result.model.model_dump(mode='json'), report=result.report,
               preregistration_sha256=digest)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:out[key] for key in ('label','wall_s','cpu_s')}))


if __name__ == '__main__': main()
