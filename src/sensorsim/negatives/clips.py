"""User-owned and licence-registered WAV loading; no data discovery/downloads.

CC0/CC BY clips additionally pass WP9's gate (lazy import, fail closed if absent).
The register path and optional gate callable are parameters for deployment and
independent verification. Gate injection is trusted application configuration,
not a field accepted from untrusted audio metadata.

An item-level listen review is always required. Licences authorize use; they do
not establish that an event is a negative, dry, or clear of unconsented voices.
"""
from __future__ import annotations

from datetime import date
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
import re

import numpy as np
from scipy import signal
from scipy.io import wavfile
import yaml

from .procedural import EVENT_FAMILIES, BED_FAMILIES, Source

ALLOWED = frozenset(('user-owned', 'CC0-1.0', 'CC-BY-4.0'))
EXTERNAL_FIELDS = ('freesound_id', 'uploader', 'source_url', 'registry_id', 'license_id',
                   'license_url', 'modifications', 'path', 'sha256', 'verified_on')


def _nonempty(row, keys):
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in keys):
        raise ValueError(f'missing text evidence fields: {keys}')


def _recent(value, as_of, *, owned=False):
    try:
        age = (as_of - date.fromisoformat(str(value))).days
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid verification date') from exc
    if age < 0 or (not owned and age > 30):
        raise ValueError('licence checks must be at most 30 days old and not future-dated')


def load_clip(root: str | Path, item: dict, *, register_path: str | Path,
              review: dict, fs: int = 16000, as_of: date | None = None,
              register_gate=None) -> Source:
    """Read one reviewed mono WAV; preserve digital level, resample explicitly.

    User rows: registry_id, path, sha256, license_id=user-owned, verified_on,
    owner, ownership_statement. Matching register asset: id, license_id,
    version_or_hash (the clip hash), checked_on, evidence_quote, disposition.
    External rows: exactly WP9's per-item manifest fields. Review metadata lives
    outside that schema: family, source_group, acoustic_stage (dry/captured),
    listened=true, target_free=true, rights_cleared=true, reviewer, notes.
    Neither location under user-data/background nor a filename implies ownership.
    """
    today = as_of or date.today()
    if item.get('license_id') not in ALLOWED:
        raise ValueError('audio licence is prohibited or unverified')
    if isinstance(fs, bool) or not isinstance(fs, int) or not 8000 <= fs <= 96000:
        raise ValueError('fs must be an integer in [8000,96000]')
    for key in ('listened', 'target_free', 'rights_cleared'):
        if review.get(key) is not True:
            raise ValueError('listening, negative-label and voice/privacy rights review are required')
    _nonempty(review, ('family', 'source_group', 'reviewer', 'notes'))
    if review['family'] not in EVENT_FAMILIES + BED_FAMILIES:
        raise ValueError('unknown reviewed family')
    if review.get('acoustic_stage') not in ('dry', 'captured'):
        raise ValueError('review must distinguish dry source and captured room audio')
    if review['acoustic_stage'] == 'captured':
        _nonempty(review, ('capture_device', 'profile_id'))
        if not isinstance(review.get('deployed_device'), bool):
            raise ValueError('captured audio needs deployed_device provenance')
    _nonempty(item, ('path', 'registry_id', 'sha256', 'verified_on'))
    owned = item['license_id'] == 'user-owned'
    _recent(item['verified_on'], today, owned=owned)
    if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
        raise ValueError('invalid audio SHA-256')
    relative = PurePosixPath(item['path'])
    if relative.is_absolute() or not relative.parts or '..' in relative.parts or '\\' in item['path'] or ':' in item['path']:
        raise ValueError('audio path must remain relative to the declared root')
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if base not in path.parents or not path.is_file() or path.suffix.lower() != '.wav':
        raise ValueError('missing WAV or path escapes declared audio root')
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError('clip exceeds 256 MiB decode budget')
    # Gate and decode the same byte snapshot, even if the file is later replaced.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('audio hash mismatch')
    register_raw = Path(register_path).read_bytes()
    registry = yaml.safe_load(register_raw)
    if not isinstance(registry, dict) or not isinstance(registry.get('assets'), list):
        raise ValueError('register requires assets rows')
    matches = [row for row in registry['assets'] if row.get('id') == item['registry_id']]
    if len(matches) != 1:
        raise ValueError('missing or duplicate register asset')
    registered = matches[0]
    if registered.get('license_id') != item['license_id'] or registered.get('disposition') not in ('ship', 'plug-in'):
        raise ValueError('registered licence/disposition does not authorize commercial training')
    _nonempty(registered, ('evidence_quote', 'version_or_hash'))
    _recent(registered.get('checked_on'), today, owned=owned)
    if owned:
        _nonempty(registered, ('name', 'url'))
        from urllib.parse import urlparse
        parsed = urlparse(registered['url'])
        if parsed.scheme not in ('https', 'http') or not parsed.netloc:
            raise ValueError('ownership evidence URL must satisfy the WP9 register schema')
    if item['license_id'] == 'user-owned':
        _nonempty(item, ('owner', 'ownership_statement'))
        if registered['version_or_hash'] != item['sha256']:
            raise ValueError('user ownership register must bind this exact clip hash')
    else:
        if register_gate is None:
            try:
                from sensorsim.licensing import validate_commercial_manifest
            except ImportError as exc:
                raise ValueError('WP9 commercial licence gate is required for external clips') from exc
            register_gate = validate_commercial_manifest
        missing = set(EXTERNAL_FIELDS) - item.keys()
        if missing:
            raise ValueError(f'external manifest missing {sorted(missing)}')
        manifest = {'schema_version': 1, 'purpose': 'commercial_training',
                    'items': [{key: item[key] for key in EXTERNAL_FIELDS}]}
        count = register_gate(manifest, registry_path=register_path, audio_root=base)
        if count != 1:
            raise ValueError('external licence gate did not verify exactly one item')
        if Path(register_path).read_bytes() != register_raw:
            raise ValueError('register changed during validation')
    source_fs, pcm = wavfile.read(BytesIO(raw))
    if pcm.ndim != 1 or len(pcm) == 0 or source_fs < 1000 or len(pcm) / source_fs > 600:
        raise ValueError('clip must be nonempty mono and at most 600 seconds')
    if pcm.dtype.kind == 'i':
        audio = pcm.astype(float) / 2**(pcm.dtype.itemsize*8-1)
    elif pcm.dtype == np.uint8:
        audio = (pcm.astype(float)-128)/128
    elif pcm.dtype.kind == 'f':
        audio = pcm.astype(float)
    else:
        raise ValueError('unsupported WAV sample format')
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1 or not np.any(audio):
        raise ValueError('audio must be finite, non-silent PCM in [-1,1]')
    if source_fs != fs:
        gcd = int(np.gcd(source_fs, fs))
        audio = signal.resample_poly(audio, fs//gcd, source_fs//gcd)
        if np.max(np.abs(audio)) > 1:
            raise ValueError('resampling overshoots full scale; explicit gain review is required')
    audio.setflags(write=False)
    return Source(audio, review['family'], f"clip:{item['sha256']}", {
        'provenance': 'recorded', 'evidence_label': 'measured', 'acoustic_stage': review['acoustic_stage'],
        'fs': fs, 'original_fs': int(source_fs), 'original_dtype': str(pcm.dtype),
        'resampler': 'scipy.signal.resample_poly' if source_fs != fs else 'none',
        'source_sha256': item['sha256'], 'register_sha256': hashlib.sha256(register_raw).hexdigest(),
        'license_id': item['license_id'], 'licence_record': dict(item), 'review': dict(review),
        'capture_device': review.get('capture_device'), 'profile_id': review.get('profile_id'),
        'deployed_device': review.get('deployed_device'),
        'native_s16': pcm.dtype == np.int16 and source_fs == fs,
        'source_group': review['source_group'], 'audio_sha256': hashlib.sha256(audio.astype('<f8').tobytes()).hexdigest(),
    })


def user_register_row(item, *, name: str, evidence_url: str, checked_on: date | None = None) -> dict:
    """WP9-compatible ownership row; caller supplies the real evidence URL.

    No URL or ownership claim is invented from a local filename. Ownership has
    no 30-day expiry. checked_on records this row-validation operation, not the
    original ownership attestation; its date is retained separately.
    """
    _nonempty(item, ('registry_id','sha256','verified_on','owner','ownership_statement'))
    if item.get('license_id') != 'user-owned':
        raise ValueError('user_register_row requires user-owned audio')
    from urllib.parse import urlparse
    u = urlparse(evidence_url)
    if not name.strip() or u.scheme not in ('http','https') or not u.netloc:
        raise ValueError('name and ownership evidence URL are required')
    today = checked_on or date.today()
    _recent(item['verified_on'], today, owned=True)
    if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
        raise ValueError('invalid audio SHA-256')
    return {'id':item['registry_id'], 'name':name, 'url':evidence_url,
            'license_id':'user-owned', 'version_or_hash':item['sha256'],
            'checked_on':today.isoformat(), 'ownership_attested_on':item['verified_on'], 'evidence_quote':item['ownership_statement'],
            'owner':item['owner'], 'disposition':'plug-in'}
