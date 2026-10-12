"""Offline evaluation tools: realism and recall against take-grouped references, false-alarm estimation (`fa_rate`) and room-simulation checks (`room_recheck`)."""

from .c2st import compare_samples, decision
from .recall import clopper_pearson, evaluate_recall
from .realism import (
    band_statistics,
    extract_audio_features,
    mmd_permutation_test,
    provenance_shares,
    realism_report,
    real_vs_real_null,
)

__all__ = [
    "band_statistics",
    "clopper_pearson",
    "compare_samples",
    "decision",
    "evaluate_recall",
    "extract_audio_features",
    "mmd_permutation_test",
    "provenance_shares",
    "realism_report",
    "real_vs_real_null",
]
