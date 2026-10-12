"""Versioned training RNG schedules and portable DataLoader helpers.

``stream/1`` retains the historical batch RNG and loader behavior. ``stream/2``
is opt-in: each example uses PCG64 seeded by ``[run_seed, example_index, 2]``.
It changes samples relative to stream/1, without changing the sampling policy.
Worker scheduling never enters that seed. Batch-level masks still use the
existing batch schedule; changing batch size is not a training equivalence claim.
"""
from __future__ import annotations

import os

import numpy as np


def _available_cpu_count():
    # Affinity is more useful than the machine total in constrained environments.
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count()


def default_workers() -> int:
    """Reserve two available CPUs for training; unknown capacity stays serial."""
    return max(0, min((_available_cpu_count() or 1) - 2, 8))


def version(cfg: dict) -> str:
    value = cfg.get("stream_version", "stream/1")
    if value not in ("stream/1", "stream/2"):
        raise ValueError("stream_version must be 'stream/1' or 'stream/2'")
    return value


def configure(cfg: dict) -> None:
    """Validate the opt-in and retain historical unversioned default metadata."""
    if version(cfg) == "stream/1":
        cfg.pop("stream_version", None)


def example_rng(seed: int, example_index: int) -> np.random.Generator:
    """A versioned, random-access example stream; no worker or process IDs."""
    return np.random.default_rng([seed, example_index, 2])


def identity_collate(batch):
    """Preserve NumPy arrays and nested provenance instead of tensorizing them."""
    return batch


def loader_options(cfg: dict, torch) -> dict:
    """Keep the legacy path unchanged; isolate model RNG for stream/2 workers.

    DataLoader draws its worker base seed in the parent. Giving it a private
    generator keeps dropout/model RNG identical to the serial stream/2 route.
    NumPy results use ordinary pickling rather than Torch tensor sharing.
    Errors propagate: there is no automatic serial fallback.
    """
    if version(cfg) == "stream/1":
        return {"prefetch_factor": 4}
    seed = int(np.random.SeedSequence([cfg["seed"], 2, 0xDADA]).generate_state(
        1, dtype=np.uint64)[0])
    return {"prefetch_factor": 2, "collate_fn": identity_collate,
            "generator": torch.Generator().manual_seed(seed)}
