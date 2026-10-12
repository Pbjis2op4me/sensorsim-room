"""Source-item AND whole-family partitions with a content-locked audit manifest."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

PARTITIONS = ('train', 'development', 'calibration', 'audit')


@dataclass(frozen=True)
class Item:
    item_id: str
    family: str
    source_group: str
    sha256: str


@dataclass(frozen=True)
class SplitPlan:
    # Immutable tuples avoid a caller accidentally relabelling an audit row.
    rows: tuple[tuple[Item, str], ...]
    seed: int
    _lock_path: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'rows', tuple((item, split) for item, split in self.rows))
        identities, families, groups, hashes = {}, {}, {}, {}
        for item, split in self.rows:
            if split not in PARTITIONS or not all((item.item_id, item.family, item.source_group)):
                raise ValueError('invalid split or empty source identity')
            if len(item.sha256) != 64 or any(c not in '0123456789abcdef' for c in item.sha256):
                raise ValueError('item must have a lowercase content SHA-256')
            if item.item_id in identities:
                raise ValueError('duplicate source item')
            identities[item.item_id] = split
            for key, table in ((item.family, families), (item.source_group, groups), (item.sha256, hashes)):
                if key in table and table[key] != split:
                    raise ValueError('family, source group or duplicate content crosses partitions')
                table[key] = split
        if not self.rows or 'audit' not in identities.values():
            raise ValueError('a nonempty locked audit partition is required')

    @classmethod
    def create(cls, items, *, audit_families, development_families, seed: int, calibration_families=()):
        """Declare families before looking at scores; remaining families train.

        seed records the upstream source selection seed. It deliberately does not
        reassign a family when new source items are appended to the catalogue.
        """
        audit, development = set(audit_families), set(development_families)
        calibration = set(calibration_families)
        if not audit or audit & development or calibration & (audit | development):
            raise ValueError('audit and development families must be nonempty/disjoint')
        items = tuple(items)
        known = {item.family for item in items}
        if not (audit | development | calibration) <= known:
            raise ValueError('reserved family is absent from the catalogue')
        return cls(tuple(sorted(((item, 'audit' if item.family in audit else
                                   'development' if item.family in development else
                                   'calibration' if item.family in calibration else 'train')
                                  for item in items), key=lambda row: row[0].item_id)), seed)

    def payload(self):
        return {'version': 'negative-splits/1', 'seed': self.seed,
                'rows': [{'item': asdict(item), 'split': split} for item, split in self.rows]}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def require(self, item: Item, *, purpose: str):
        wanted = {'train': 'train', 'mine': 'development', 'tune': 'development', 'calibrate': 'calibration', 'audit': 'audit'}.get(purpose)
        if wanted is None:
            raise ValueError('unknown partition purpose')
        if not any(original == item and split == wanted for original, split in self.rows):
            raise ValueError(f'source identity is not authorized for {purpose}; audit is never mined/tuned')

    def save(self, path: str | Path):
        """Exclusive creation: never overwrite a locked plan. Keep hash in preregistration."""
        with Path(path).open('x', encoding='utf-8') as out:
            json.dump({'sha256': self.sha256, **self.payload()}, out, indent=2)
            out.write('\n')
        with Path(str(path)+'.sha256').open('x', encoding='utf-8') as receipt:
            receipt.write(self.sha256+'\n')

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != 'negative-splits/1':
            raise ValueError('unsupported split manifest')
        plan = cls(tuple((Item(**row['item']), row['split']) for row in data['rows']), data['seed'])
        receipt = Path(str(path)+'.sha256').read_text().strip()
        if plan.sha256 != expected_sha256 or data['sha256'] != expected_sha256 or receipt != expected_sha256:
            raise ValueError('locked partition changed')
        object.__setattr__(plan, '_lock_path', str(Path(path).resolve()))
        return plan

    @classmethod
    def from_audit_lock(cls, path, *, expected_file_sha256):
        """Load mining authority from the file hash pinned by the preregistration.

        A rebuilt plan plus its self-written receipt is not the original lock.
        Callers must retain the original path/hash, not recompute the expected
        hash from a candidate plan. This is an integrity check, not access control
        against an operator who rewrites the study's preregistration itself.
        """
        if not isinstance(path, (str, Path)):
            raise ValueError('mining requires the preregistered audit lock path and file hash')
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
            raise ValueError('preregistered audit lock file hash mismatch')
        data = json.loads(raw)
        plan = cls.load(path, expected_sha256=data['sha256'])
        if Path(path).read_bytes() != raw:
            raise ValueError('audit lock changed during validation')
        return plan

    def verify_lock(self, expected_sha256):
        """Require the persisted plan AND independent creation receipt at every use.

        The study must retain this path and receipt in its preregistration. A new
        in-memory plan or a relabelled loaded object has no mining authority.
        Deleting all study files is outside this filesystem integrity boundary.
        """
        if self._lock_path is None:
            raise ValueError('persisted audit lock required; save and load the preregistered plan')
        loaded = type(self).load(self._lock_path, expected_sha256=expected_sha256)
        if loaded.sha256 != self.sha256:
            raise ValueError('persisted audit lock differs from in-memory plan')
        return self
