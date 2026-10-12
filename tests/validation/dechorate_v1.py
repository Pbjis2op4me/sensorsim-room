"""Read-only adapter for dEchorate processed v1 (Zenodo 5562386, CC BY 4.0).

Internal validation tooling: no data, geometry table or recording ships here.
The attribute-free archive has 20,000 samples, 30 microphones and SEVEN sources.
The w4 independent review supports 48 kHz and column 7 = omni source 7 using
arrival geometry and a historical sample comparison. Those are explicit external
metadata assumptions, not attributes supplied by this HDF5 file. Source 8 is
absent. Never interpret directional columns as a substitute omni source.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


STATES = ('000000', '000001', '000010', '000100', '001000', '010000',
          '011000', '011100', '011110', '011111', '020002')
RIR_FILE = 'dechorate/v1/dechorate_rir_processed.hdf5'
GEOMETRY_FILE = 'dechorate/v1/load_positions_and_plot.py'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_selection(sources, arrays):
    """Reject unavailable/disclosed groups before opening any waveform dataset."""
    for values, label in ((sources, 'sources'), (arrays, 'arrays')):
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{label} must be nonempty and unique')
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
            raise ValueError(f'{label} must use integer one-based IDs')
    if 8 in sources:
        raise ValueError('source 8 is absent from processed v1: seven columns, only source 7 is omni; '
                         'the requested sources 7/8 computation is inconclusive')
    if tuple(sources) != (7,):
        raise ValueError('this omni recheck permits only available source 7; source 9 is excluded')
    if any(x not in range(1, 6) for x in arrays):
        raise ValueError('only arrays 1-5 are permitted; disclosed array 6 is excluded')


def _geometry(path):
    # Extract literal assignments; never execute downloaded Python.
    required = {'room_size', 'mics', 'srcs_omn', 'room_temperature'}
    values = {}
    for node in ast.parse(path.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in required):
            continue
        name, value = node.targets[0].id, node.value
        if name in values:
            raise ValueError(f'duplicate geometry assignment: {name}')
        if isinstance(value, ast.Call):
            if (not isinstance(value.func, ast.Attribute) or not isinstance(value.func.value, ast.Name)
                    or value.func.value.id != 'np' or value.func.attr != 'array'
                    or len(value.args) != 1 or value.keywords):
                raise ValueError('geometry must contain literal arrays')
            value = value.args[0]
        values[name] = np.asarray(ast.literal_eval(value), dtype=float)
    shapes = {'room_size': (3,), 'mics': (3, 30), 'srcs_omn': (3, 3), 'room_temperature': ()}
    if set(values) != required or any(values[k].shape != shape or not np.isfinite(values[k]).all()
                                      for k, shape in shapes.items()):
        raise ValueError('missing/nonfinite geometry, temperature or incompatible geometry shape')
    dims = values['room_size']
    if np.any(dims <= 0) or not -30 <= float(values['room_temperature']) <= 60:
        raise ValueError('invalid dimensions in metres or temperature in Celsius')
    for k in ('mics', 'srcs_omn'):
        if np.any(values[k] <= 0) or np.any(values[k] >= dims[:, None]):
            raise ValueError('coordinates must be in the room frame, in metres')
    return values


class V1Archive:
    """Verified files, explicit native rate, and one-channel bounded reads.

    A local MANIFEST is an integrity check, not independent registration custody.
    The verifier must also freeze these input digests outside the lane tree.
    """
    fs = 48000
    count = 20000

    def __init__(self, data_root, *, expected_fs=48000):
        if expected_fs != self.fs:
            raise ValueError('processed v1 requires the reviewed rate of 48000 Hz')
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError('netCDF4 is required for the internal HDF5 validation reader') from exc
        self._dataset = Dataset
        self.root = Path(data_root)
        self.path = self.root/RIR_FILE
        manifest = json.loads((self.root/'MANIFEST.json').read_text())['files']
        self.input_hashes = {}
        for name in (RIR_FILE, GEOMETRY_FILE):
            entries = [r for r in manifest if r['path'] == name]
            if len(entries) != 1 or entries[0].get('licence') != 'CC-BY-4.0':
                raise ValueError(f'missing, duplicate or unapproved manifest entry: {name}')
            path, pin = self.root/name, entries[0]
            digest = sha256(path)
            if digest != pin['sha256'] or path.stat().st_size != pin['size_bytes']:
                raise ValueError(f'input integrity mismatch: {name}')
            self.input_hashes[name] = digest
        self.geometry = _geometry(self.root/GEOMETRY_FILE)
        self.temperature_c = float(self.geometry['room_temperature'])
        self.read_log = []
        self._content_pairs = {}
        with Dataset(self.path, 'r') as archive:
            if set(archive.variables) != set(STATES):
                raise ValueError('processed v1 state coverage changed')
            for state in STATES:
                variable = archive.variables[state]
                if variable.shape != (self.count, 30, 7) or np.dtype(variable.dtype) != np.dtype('float64'):
                    raise ValueError('processed v1 must have shape (20000, 30, 7), float64')

    def metadata(self):
        return dict(edition='dEchorate processed v1.0', record_id=5562386, licence='CC-BY-4.0',
            evidence_domain='measured-RIR proxy', input_sha256=dict(self.input_hashes),
            native_shape=[self.count, 30, 7], sample_rate_hz=self.fs,
            native_duration_s=self.count/self.fs, temperature_c=self.temperature_c,
            humidity_pct=50., humidity_basis='assumed; no logged humidity',
            rate_and_axis_basis='w4 independent reviewer: 48 kHz; column 7 equals omni source 7; '
                                'no publisher axis/rate attributes',
            geometry_basis='literal v1 publisher geometry script; metres; room frame',
            missing_sources=[8, 9], excluded_arrays=[6],
            limits=['native record is 20000 samples, not the legacy 48000 samples',
                    'short records can lack a noise-only tail and valid T30 support',
                    'no absolute acquisition time, independent range truth or Pa/SPL calibration'])

    def read_pair(self, state, source, microphone):
        if isinstance(microphone, bool) or not isinstance(microphone, (int, np.integer)) or microphone < 1:
            raise ValueError('microphone must be a positive integer one-based ID')
        validate_selection((source,), ((microphone-1)//5+1,))
        if state not in STATES:
            raise ValueError('unknown room state')
        with self._dataset(self.path, 'r') as archive:
            # Never read a room tensor or microphones from array 6.
            raw = archive.variables[state][:, microphone-1, source-1]
        self.read_log.append(dict(state=state, source=source, microphone=microphone,
                                 slice=[[0, self.count], microphone-1, source-1]))
        if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
            raise ValueError('missing measured RIR samples')
        values = np.asarray(raw, dtype=np.float64)
        if (values.shape != (self.count,) or not np.isfinite(values).all()
                or np.ptp(values) == 0):
            raise ValueError('empty, constant, silent or nonfinite measured RIR')
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        pair = (state, source, microphone)
        previous = self._content_pairs.setdefault(digest, pair)
        if previous != pair:
            raise ValueError(f'duplicate waveform content across pair IDs: {previous}, {pair}')
        return values.copy()
