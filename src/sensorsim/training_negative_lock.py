# SPDX-License-Identifier: Apache-2.0
"""One parsed training split, with persisted content checked at every access.

This is an integrity optimization, not independent custody. Never use file stat
metadata as a content oracle or renew a digest after its first admission.
"""
from pathlib import Path

from sensorsim.negatives.integration import verified_file
from sensorsim.negatives.splits import SplitPlan


class PinnedSplitPlan(SplitPlan):
    """Retain SplitPlan's authorization semantics without reparsing its JSON."""

    @classmethod
    def open(cls, path, expected_file_sha256):
        plan = cls.from_audit_lock(path, expected_file_sha256=expected_file_sha256)
        object.__setattr__(plan, '_pinned_path', plan._lock_path)
        object.__setattr__(plan, '_file_sha256', expected_file_sha256)
        object.__setattr__(plan, '_plan_sha256', plan.sha256)
        return plan

    def verify_lock(self, expected_sha256):
        if (self._lock_path != self._pinned_path or expected_sha256 != self._plan_sha256
                or self.sha256 != self._plan_sha256):
            raise ValueError('frozen split plan changed')
        verified_file(self._pinned_path, self._file_sha256)
        # Preserve SplitPlan.load's whitespace convention for the receipt.
        try:
            receipt = Path(self._pinned_path+'.sha256').read_text().strip()
        except OSError as exc:
            raise ValueError('locked partition receipt is missing') from exc
        if receipt != self._plan_sha256:
            raise ValueError('locked partition changed')
        return self
