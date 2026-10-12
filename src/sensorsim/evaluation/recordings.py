"""Recording ancestry checks for FA manifests, before rendering or scoring.

No acoustic similarity claim: exact blocks detect unmodified excerpts only.
Identity/range declarations remain necessary for short or transformed overlaps.
"""
from __future__ import annotations

import hashlib
import warnings
import wave
from pathlib import Path

import numpy as np

from sensorsim import edge


def _extent(record, *, fs, length=None):
    identity = record.get('recording_id')
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(f"{record.get('path')}: recording_id must name the original recording")
    rate = record.get('sample_rate_hz')
    if isinstance(rate, bool) or not isinstance(rate, int) or rate != fs:
        raise ValueError(f"{record.get('path')}: sample_rate_hz must equal model rate {fs}; no implicit resampling")
    span = record.get('sample_range')
    if (not isinstance(span, (list, tuple)) or len(span) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in span)
            or not 0 <= span[0] < span[1]):
        raise ValueError(f"{record.get('path')}: sample_range must be integer [start, end), 0 <= start < end")
    if length is not None and span[1]-span[0] != length:
        raise ValueError(f"{record.get('path')}: sample_range length differs from decoded samples ({length})")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate_fingerprint(record):
    fp = record['fingerprint']
    if not isinstance(fp, dict) or fp.get('algorithm') != 'float32le-exact-blocks/1':
        raise ValueError('unsupported calibration fingerprint algorithm; regenerate the calibration record')
    size = fp.get('block_samples')
    if isinstance(size, bool) or not isinstance(size, int) or size != max(8, record['sample_rate_hz']):
        raise ValueError('invalid calibration fingerprint block_samples')
    if fp.get('anchor_samples') != 8 or not isinstance(fp.get('blocks'), list):
        raise ValueError('invalid calibration fingerprint anchors/blocks')
    length = record['sample_range'][1]-record['sample_range'][0]
    offsets = set()
    for block in fp['blocks']:
        if not isinstance(block, dict):
            raise ValueError('invalid calibration fingerprint block')
        offset, anchor = block.get('offset'), block.get('anchor')
        if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset % size
                or offset+size > length or offset in offsets or not _sha(block.get('sha256'))
                or not isinstance(anchor, str) or len(anchor) != 16
                or any(c not in '0123456789abcdef' for c in anchor)):
            raise ValueError('invalid calibration fingerprint offset, hash or anchor')
        offsets.add(offset)
    for key, expected in [('trailing_samples_omitted', length % size),
                          ('constant_blocks_omitted', length // size-len(offsets))]:
        if type(fp.get(key)) is not int or fp[key] != expected:
            raise ValueError('incomplete calibration fingerprint coverage record')


def _block_match(audio, reference):
    """Find a reference block at ANY sample offset; bound temporary array memory.

    The cheap eight-sample index only locates candidates. SHA256 of the entire
    block is the decision. There is no downsampling, gain normalisation or fuzzy
    threshold. A nonconstant matching block is a conservative content conflict.
    """
    fp = reference.get('fingerprint')
    if not fp or not fp['blocks'] or len(audio) < fp['block_samples']:
        return None
    size = fp['block_samples']
    index = {}
    for block in fp['blocks']:
        index.setdefault(int(block['anchor'], 16), {}).setdefault(block['sha256'], block['offset'])
    wanted = np.array(sorted(index), dtype=np.uint64)
    # Keys overlap by seven samples at chunk edges; every eligible start is read.
    for start in range(0, len(audio)-size+1, 262144):
        end = min(start+262144, len(audio)-size+1)
        keys = edge._audio_anchor_keys(audio[start:end+7])
        positions = np.searchsorted(wanted, keys)
        candidates = np.flatnonzero(wanted[np.minimum(positions, len(wanted)-1)] == keys)
        for local in candidates:
            at = start+int(local)
            digest = hashlib.sha256(np.asarray(audio[at:at+size], '<f4').tobytes()).hexdigest()
            reference_offset = index[int(keys[local])].get(digest)
            if reference_offset is not None:
                return reference_offset, at, size
    return None


def _describe(row, span=None):
    span = span or row.get('sample_range', ['unknown', 'unknown'])
    return (f"{row['role']} file {row.get('path', '<saved calibration audio>')!r}, "
            f"recording {row.get('recording_id', '<unknown>')!r}, "
            f"samples [{span[0]}, {span[1]}) at {row.get('sample_rate_hz', 'unknown')} Hz")


def _conflict(reason, one, two, spans=(None, None)):
    raise ValueError(f"recording conflict ({reason}): {_describe(one, spans[0])} conflicts with "
                     f"{_describe(two, spans[1])}; split calibration, floor, context and bed audio into disjoint ranges")


class RecordingAudit:
    """Audit immutable decoded inputs and model-saved calibration ancestry."""

    def __init__(self, model, root):
        self.fs = int(model.meta['features']['fs'])
        self.minimum = model.features.n
        self.root = Path(root)
        self.cache, self.rows, self.messages = {}, [], []
        calibration = model.meta['detector'].get('calibration', {})
        if not isinstance(calibration, dict):
            raise ValueError('invalid saved calibration record')
        records = calibration.get('calibration_audio_records', [])
        if not isinstance(records, list):
            raise ValueError('invalid saved calibration_audio_records')
        for record in records:
            if not isinstance(record, dict) or not _sha(record.get('decoded_sha256')):
                raise ValueError('invalid saved calibration audio digest')
            row = dict(record, role='calibration', origin='model')
            if record.get('schema') is not None:
                if record['schema'] != 'calibration-audio/1':
                    raise ValueError('unsupported saved calibration record schema')
                _extent(record, fs=self.fs)
                if 'fingerprint' not in record:
                    raise ValueError('missing calibration fingerprint in versioned record')
                _validate_fingerprint(record)
            else:
                self._warn('Legacy model calibration has no ranges or block fingerprints; excerpt disjointness cannot be established from the model alone.')
            self.rows.append(row)
        digest = calibration.get('calibration_audio_sha256')
        if digest is not None:
            if not _sha(digest):
                raise ValueError('invalid saved calibration_audio_sha256')
            if not any(r['decoded_sha256'] == digest for r in self.rows):
                self.rows.append(dict(role='calibration', origin='model', decoded_sha256=digest,
                                      path='<saved calibration audio>'))
                self._warn('Legacy model calibration has only a whole-file digest; ranges and excerpt fingerprints are unavailable. Supply original calibration_streams.')
        if not self.rows:
            self._warn('Model has no calibration audio record; unknown calibration ancestry cannot be checked. Supply original calibration_streams if calibration was performed.')

    def _warn(self, message):
        if message not in self.messages:
            self.messages.append(message)
            warnings.warn(message, UserWarning, stacklevel=3)

    def read(self, record, role, *, allow_reuse=False):
        _extent(record, fs=self.fs)
        file = (self.root / record['path']).resolve()
        if file not in self.cache:
            with wave.open(str(file), 'rb') as wav:
                if wav.getnchannels() != 1 or wav.getframerate() != self.fs:
                    raise ValueError(f'{file}: inputs require mono at model sample rate; no implicit resampling or channel mixing')
            audio = edge.read_wav(str(file), self.fs)
            if len(audio) < self.minimum or not np.isfinite(audio).all() or np.ptp(audio) == 0:
                raise ValueError(f'{file}: empty, short or constant recording is not usable exposure')
            self.cache[file] = audio
        audio = self.cache[file]
        _extent(record, fs=self.fs, length=len(audio))
        row = dict(edge.calibration_audio_record(audio, self.fs, recording_id=record['recording_id'],
                   sample_start=record['sample_range'][0], path=str(file)), role=role, origin='manifest',
                   parent_id=record.get('parent_id', record['recording_id']))
        if not isinstance(row['parent_id'], str) or not row['parent_id'].strip():
            raise ValueError(f'{file}: parent_id must be a nonempty independent group ID')
        for other in self.rows:
            same_role = other['role'] == role
            # An explicitly listed calibration input can duplicate its model record.
            if same_role and other['origin'] == 'model':
                continue
            shared_id = row['recording_id'] == other.get('recording_id')
            same_parent = row['parent_id'] == other.get('parent_id')
            repeat_context = same_role and allow_reuse and shared_id and same_parent
            a, b = row['sample_range'], other.get('sample_range')
            if shared_id and b and max(a[0], b[0]) < min(a[1], b[1]) and not repeat_context:
                _conflict('overlapping sample ranges', other, row)
            if row['decoded_sha256'] == other['decoded_sha256'] and not repeat_context:
                _conflict('identical decoded content / duplicate exposure', other, row)
            if not repeat_context:
                match = _block_match(audio, other)
                if match:
                    old_at, new_at, size = match
                    old_start = other['sample_range'][0]+old_at
                    new_start = a[0]+new_at
                    _conflict('exact block content overlap', other, row,
                              ([old_start, old_start+size], [new_start, new_start+size]))
        self.rows.append(row)
        return audio, row['decoded_sha256']

    def report(self):
        report = {role: [] for role in ('calibration', 'floor', 'context', 'bed')}
        for row in self.rows:
            fp = row.get('fingerprint', {})
            report[row['role']].append({k: v for k, v in row.items() if k != 'fingerprint'} | {
                'fingerprinted_blocks': len(fp.get('blocks', [])),
                'block_samples': fp.get('block_samples'),
                'constant_blocks_omitted': fp.get('constant_blocks_omitted'),
                'trailing_samples_omitted': fp.get('trailing_samples_omitted')})
        report['warnings'] = self.messages
        report['limits'] = ('Exact mono float32 blocks only; gain changes and resampling defeat fingerprints. '
            'Undeclared overlaps shorter than two blocks, or containing only constant stored blocks, may escape. '
            'Recording identity and sample ranges remain mandatory; disjoint ranges do not prove statistical independence.')
        return report


def audit_manifest(model, manifest, root):
    """Load every calibration/floor/context/bed input before any detector call."""
    if not isinstance(manifest.get('calibration_streams'), list):
        raise ValueError('manifest must declare calibration_streams (explicit [] if unused)')
    audit = RecordingAudit(model, root)
    for record in manifest['calibration_streams']:
        audit.read(record, 'calibration')
    for record in manifest.get('floor_streams', []):
        audit.read(record, 'floor')
    entries = manifest.get('challenges', [])
    if isinstance(entries, dict):
        entries = [{**r, 'family': r.get('family', family)}
                   for family, records in entries.items() for r in records]
    if not isinstance(entries, list) or not entries:
        raise ValueError('challenges must be a nonempty list')
    for record in entries:
        if 'overlay_path' in manifest:
            if 'overlay' not in record:
                raise ValueError('overlay banks require an overlay descriptor for every challenge')
            descriptor = record['overlay']
            audit.read(descriptor['background'], 'bed', allow_reuse=True)
            for source in descriptor['events']:
                if 'procedural' not in source:
                    audit.read(source, 'context', allow_reuse=True)
        else:
            audit.read(record, 'context')
    return audit, entries
