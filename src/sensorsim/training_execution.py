# SPDX-License-Identifier: Apache-2.0
"""Execution helpers for the trainer, without changing selection or example draws."""
from __future__ import annotations

import json

import numpy as np


class CheckpointExport:
    """An in-memory copy of one checkpoint that already passed export parity.

    Keep the selected checkpoint's arrays, not its mutable file or the terminal
    model. Finalizing changes only training metadata. No feature extraction,
    inference, BatchNorm folding or RNG draws are needed a second time.
    """

    def __init__(self, path, report):
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files if key != 'meta'}
            self.meta = json.loads(bytes(data['meta']).decode('utf-8'))
        for array in self.arrays.values():
            array.setflags(write=False)
        self.report = dict(report)

    def write(self, out, cfg, training, *, screen_artifact=False):
        # Final training metadata still crosses the normal production gate.
        from sensorsim.negatives.integration import require_production_metadata
        if not screen_artifact:
            require_production_metadata(training)
            require_production_metadata({'config': cfg})
        elif cfg.get('proxy_screen') is not True:
            raise ValueError('screen artifacts require explicit proxy_screen')
        meta = dict(self.meta, training=training)
        arrays = dict(self.arrays, meta=np.frombuffer(json.dumps(meta).encode('utf-8'), np.uint8))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        return dict(self.report, model=str(out), best_val_mean_ap=training['best_val_mean_ap'],
                    history_tail=training['history'][-2:])


def export_selected(cached, exporter, model, classes, cfg, out, check_features, training, *, screen_artifact=False):
    """Finalize a selected snapshot, or run normal export if none was produced."""
    if cached is not None:
        return cached.write(out, cfg, training, screen_artifact=screen_artifact)
    return exporter(cpu_export_model(model), classes, cfg, out, check_features,
                    training=training, screen_artifact=screen_artifact)


def resolve_device(torch, requested):
    """CPU is the stable default; explicit CUDA never silently falls back.

    PyTorch exposes both NVIDIA CUDA and AMD ROCm through ``torch.cuda``.
    Accelerator training is float32, but is not bit-identical to CPU training.
    """
    if not isinstance(requested, str) or requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if requested == 'cpu':
        return torch.device('cpu'), None
    available = torch.cuda.is_available()
    if requested == 'cuda' and not available:
        raise RuntimeError('CUDA/ROCm training requested but unavailable in this PyTorch environment')
    device = torch.device('cuda' if available else 'cpu')
    return device, dict(requested_device=requested, device=device.type,
        backend=('rocm' if torch.version.hip else 'cuda') if available else 'cpu',
        device_name=torch.cuda.get_device_name(device) if available else 'cpu',
        numerical_contract='non-bit-identical to CPU' if available else 'CPU reference',
        fallback=None if available else 'CUDA/ROCm unavailable')


def cpu_export_model(model):
    """Snapshot GPU weights for the existing CPU/NumPy export parity gate.

    Moving the live model would detach the optimizer from its device state.
    The CPU path returns the original object and consumes no RNG draws.
    """
    if next(model.parameters()).device.type == 'cpu':
        return model
    import copy
    return copy.deepcopy(model).cpu()
