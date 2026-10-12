"""Train the edge CNN on a project dataset; export it for the numpy-only runtime (``sensorsim.edge``).

Training needs the optional ``[nn]`` extra (PyTorch). Install the CPU build, which keeps
NVIDIA's libraries out of the environment::

    pip install torch --index-url https://download.pytorch.org/whl/cpu

Design (choices and their sources are summarized in USAGE.md):

* Model: a DS-CNN-style network (Zhang et al., "Hello Edge", 2017): one strided convolution
  then depthwise-separable blocks, mean+max pooling over time, one sigmoid per class (no
  class active = background). Tens of thousands of parameters, a few million MACs.
* Inputs: gain-normalized log-mel windows computed by ``sensorsim.edge.Features``, the same
  code the device runs, so training and deployment features cannot drift apart.
* Device-recorded background stays at its digital level. Rendered targets use an explicit
  event-span SPL prior and the versioned Pi microphone response. SPL-referenced synthetic
  beds pass through that profile once with its absolute floor. Training, validation and
  detector fitting share this contract; random codecs/AGC/NS/loss require the stress split.
* Feature-level augmentation: SpecAugment (Park et al., 2019) and Freq-MixStyle (device
  robustness in DCASE acoustic scene classification; Kim et al. / Schmid et al., 2022).
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from sensorsim import edge, iot_mic, training_dynamics, training_negatives
from sensorsim.training_device import mix_device_audio
from sensorsim import training_selection
from sensorsim import training_stream
from sensorsim.recognizer import _audible_span, _crop, _event_window, _load_manifest, read_audio

DEFAULTS = {
    **training_negatives.DEFAULTS,
    "window_s": 1.0, "hop_s": 0.25, "fs": 16000, "n_fft": 512, "hop": 160, "n_mels": 40,
    "fmin_hz": 60.0, "fmax_hz": 7600.0, "width": 48, "blocks": 4,
    "steps": 3000, "batch": 128, "lr": 3e-3, "weight_decay": 1e-4, "seed": 0,
    "p_confuser_given_background": 0.5, "max_procedural_share": 0.2,
    "negative_manifest": None, "negative_manifest_sha256": None, "proxy_screen": False,
    "selection_mode": "panel", "selection_panel": None, "selection_panel_sha256": None,
    "selection_grid": training_selection.DEFAULT_GRID, "selection_trigger_budget": 0.1,
    "selection_min_recall": 0.5, "selection_min_groups": 10,
    "p_background_only": 0.35, "p_mic_chain": 0.85, "snr_db": [-5.0, 20.0],
    "device_split": "in_distribution", "device_profile": "pi5_inmp441/1",
    "background_domain": "device_recorded", "render_reference_spl_db": 94.0,
    "event_level_mode": "assumed_active_sample/1",
    "event_spl_db": [60.0, 80.0],  # assumed event-active RMS SPL at mic; replace with deployment evidence
    "p_specaugment": 0.5, "p_freq_mixstyle": 0.4, "mixstyle_alpha": 0.4,
    "detector": {"hop_s": 0.25, "off_ratio": 0.6, "smooth_hops": 3, "min_hops": 2, "default_on": 0.5},
    "val_windows": 1536, "workers": training_stream.default_workers(), "threads": 4,
    "stream_version": "stream/1",
}


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("CNN training needs PyTorch: pip install torch --index-url "
                          "https://download.pytorch.org/whl/cpu   (or: pip install 'sensorsim-room[nn]')") from error
    return torch


def build_model(n_classes: int, width: int = 48, blocks: int = 4):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, stride):
            super().__init__()
            self.dw = nn.Conv2d(width, width, 3, stride=stride, padding=1, groups=width, bias=False)
            self.dw_bn = nn.BatchNorm2d(width)
            self.pw = nn.Conv2d(width, width, 1, bias=False)
            self.pw_bn = nn.BatchNorm2d(width)

        def forward(self, x):
            x = torch.relu(self.dw_bn(self.dw(x)))
            return torch.relu(self.pw_bn(self.pw(x)))

    class EdgeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv0 = nn.Conv2d(1, width, 5, stride=2, padding=2, bias=False)
            self.conv0_bn = nn.BatchNorm2d(width)
            self.blocks = nn.ModuleList([Block((2, 1) if i == 1 else (1, 1)) for i in range(blocks)])
            self.drop = nn.Dropout(0.2)
            self.fc = nn.Linear(2*width, n_classes)

        def forward(self, x):
            x = torch.relu(self.conv0_bn(self.conv0(x)))
            for block in self.blocks:
                x = block(x)
            m = x.mean(dim=2)
            return self.fc(self.drop(torch.cat([m.mean(dim=-1), m.amax(dim=-1)], dim=1)))

    return EdgeNet()


# ------------------------------------------------------------------------------------- data
class Corpus:
    """Train or val audio from a dataset manifest: rendered targets and real background."""

    def __init__(self, manifest: Path, split: str, classes: list[str] | None, fs: int):
        rows, rate = _load_manifest(manifest)
        if rate != fs:
            raise ValueError(f"dataset is {rate} Hz; the edge model is configured for {fs} Hz "
                             f"(set dataset fs to {fs} in project.yaml)")
        rows = [r for r in rows if r["split"] == split]
        self.rows = rows
        # Keep export/pressure/parent provenance aligned with each target array.
        self.target_rows = {c: [r for r in rows if r["label"] == c]
                            for c in (classes or sorted({r["label"] for r in rows if r["label"] != "background"}))}
        self.negative_pool = None
        self.confuser_draws = 0
        self.pool_purpose = "train" if split == "train" else "tune"
        # float32 is exact for 16-bit audio and halves the copy each spawned data worker receives.
        load = lambda r: read_audio(manifest.parent/r["file"], fs).astype(np.float32)
        self.background = [load(r) for r in rows if r["label"] == "background"]
        self.classes = classes or sorted({r["label"] for r in rows if r["label"] != "background"})
        self.targets = {c: [load(r) for r in rows if r["label"] == c] for c in self.classes}
        self.spans = {c: [_audible_span(x, fs) for x in xs] for c, xs in self.targets.items()}
        self.window_cues = {c: [r.get('window_cues') for r in rows if r['label'] == c] for c in self.classes}
        if not self.background:
            raise ValueError(f"no '{split}' background recordings in {manifest}")
        if not all(self.targets.values()):
            empty = [c for c, v in self.targets.items() if not v]
            raise ValueError(f"no '{split}' recordings of {empty}")


def make_example(corpus: Corpus, rng: np.random.Generator, cfg: dict, *, return_context=False) -> tuple:
    """One audio window (n,) and its multi-hot label, built as deployment audio arises."""
    from sensorsim.event_levels import require_legacy_training
    require_legacy_training(cfg)
    if cfg.get('cue_validity'):
        from sensorsim.training_policy import validate_config as validate_cues
        validate_cues(cfg)
        if not hasattr(corpus, 'cue_annotations'):
            raise ValueError('cue corpus must be bound before make_example')
    if return_context or training_dynamics.enabled(cfg) or training_negatives.enabled(cfg):
        audio, label, context = training_dynamics.draw_example(corpus, rng, cfg, mix_device_audio, context=return_context)
        return (audio, label, context) if return_context else (audio, label)
    fs, n = cfg["fs"], int(round(cfg["window_s"]*cfg["fs"]))
    label = np.zeros(len(corpus.classes), np.float32)
    background = corpus.background[int(rng.integers(len(corpus.background)))]
    audio = _crop(background, 2*n, rng).astype(float)
    event, active = np.zeros(2*n), (n, 2*n)
    if rng.random() >= cfg["p_background_only"]:
        c = int(rng.integers(len(corpus.classes)))
        k = int(rng.integers(len(corpus.targets[corpus.classes[c]])))
        x, span = corpus.targets[corpus.classes[c]][k], corpus.spans[corpus.classes[c]][k]
        window, (a, b) = _event_window(x, span, n, rng)
        event[n:] = window
        active = (n + a, n + b)
        if b > a and np.mean(window[a:b]**2) > 0:
            label[c] = 1.0
    if not label.any():
        from sensorsim.negatives.integration import draw_confuser
        confuser = draw_confuser(corpus, rng, cfg, n)
        if confuser is not None:
            window, (a, b) = confuser
            event[n:] = window
            active = (n+a, n+b)
    audio = mix_device_audio(audio, event, active, rng, cfg)
    return audio[n:].astype(np.float32), label


class Batches:
    """Random-access batches of features (batch, 1, mels, frames) and labels.

    stream/1 keeps seed [seed, k]; opt-in stream/2 seeds each example by its global
    index. Both preserve batch order across worker counts. The module-level class
    needs no PyTorch, so spawned DataLoader workers can receive it by pickling.
    """

    def __init__(self, corpus: Corpus, cfg: dict, count: int, seed: int, *, training=True):
        self.stream_version = training_stream.version(cfg)
        cfg = dict(cfg) if training else training_dynamics.validation_config(cfg)
        if training and cfg.get('positive_fraction') is not None:
            fraction = float(cfg['positive_fraction'])
            if not 0 < fraction < 1:
                raise ValueError('positive_fraction must be between zero and one')
            cfg['p_background_only'] = 1 - fraction
        if not training:
            corpus = training_dynamics.validation_corpus(corpus, cfg)
        self.training = training
        self.corpus, self.cfg, self.count, self.seed = corpus, cfg, count, seed
        self.features = edge.Features(_feature_cfg(cfg))

    def _example_rngs(self, k: int):
        if self.stream_version == "stream/1":
            rng = np.random.default_rng([self.seed, k])
            for _ in range(self.cfg["batch"]):
                yield rng
        else:
            for i in range(self.cfg["batch"]):
                yield training_stream.example_rng(self.seed, k*self.cfg["batch"] + i)

    def _example_id(self, k: int, i: int) -> str:
        if self.stream_version == "stream/1":
            return f'{self.seed}:{k}:{i}'
        return f'{self.seed}:stream/2:{k*self.cfg["batch"] + i}'

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= k < self.count:
            raise IndexError(k)
        if self.training and (self.cfg.get('cue_validity') or self.cfg.get('event_aware_masks') or self.cfg.get('_dynamics_report')):
            before = getattr(self.corpus, "confuser_draws", 0)
            if self.cfg.get('cue_validity'):
                from sensorsim.training_policy import paired_example
                examples = [paired_example(make_example, self.corpus, rng, self.cfg,
                    example_id=self._example_id(k, i)) for i, rng in enumerate(self._example_rngs(k))]
            else:
                examples = [make_example(self.corpus, rng, self.cfg, return_context=True)
                            for rng in self._example_rngs(k)]
            x = self.features(np.stack([e[0] for e in examples]))
            if self.cfg.get('cue_masks'):
                x = _spec_augment(x, np.random.default_rng([self.seed, k, 842]),
                    self.cfg['p_specaugment'], contexts=[e[2] for e in examples],
                    cue_validity=True, training=self.training)
            if self.cfg.get('event_aware_masks'):
                x = _spec_augment(x, np.random.default_rng([self.seed, k, 842]),
                                  self.cfg['p_specaugment'], contexts=[e[2] for e in examples])
            y = np.stack([e[1] for e in examples])
            if self.cfg.get('_dynamics_report'):
                stats = np.array([[e[2]['class_index'], e[2]['attempts'], e[2]['rejections'],
                                   e[2]['requested_level_db'] if e[2]['requested_level_db'] is not None else np.nan,
                                   e[2].get('mask_qa_skips', 0)]
                                  for e in examples])
                result = (x, y, stats)
                if self.cfg.get('_negative_report'):
                    result += ((training_negatives.context_counts(examples) if training_negatives.enabled(self.cfg) else
                        dict(windows=len(examples),background_windows=int(np.sum(~y.any(axis=1))),
                             confuser_windows=self.corpus.confuser_draws-before)),)
                    if self.cfg.get('cue_validity'):
                        result[-1]['cue_records'] = [e[2]['cue_trace'] for e in examples]
                return result
            return x, y
        before = getattr(self.corpus, "confuser_draws", 0)
        examples = [make_example(self.corpus, rng, self.cfg) for rng in self._example_rngs(k)]
        result = self.features(np.stack([e[0] for e in examples])), np.stack([e[1] for e in examples])
        if self.cfg.get('_negative_report'):
            return (*result, dict(windows=len(examples), background_windows=int(np.sum(~result[1].any(axis=1))),
                                  confuser_windows=self.corpus.confuser_draws-before))
        return result


def _feature_cfg(cfg: dict) -> dict:
    return {k: cfg[k] for k in ("fs", "window_s", "n_fft", "hop", "n_mels", "fmin_hz", "fmax_hz")}


def _spec_augment(x, rng, p: float, *, contexts=None, cue_validity=False, training=True):
    """In-place time and frequency masks on a (batch, 1, mels, frames) numpy array."""
    if cue_validity:
        from sensorsim.training_policy import mask_batch
        return mask_batch(x, rng, p, contexts, training=training)
    if p <= 0:
        return x
    if contexts is not None:
        return training_dynamics.event_aware_masks(x, rng, p, contexts)
    for i in range(x.shape[0]):
        if rng.random() < p:
            f, t = int(rng.integers(1, 6)), int(rng.integers(1, 11))
            f0, t0 = int(rng.integers(0, x.shape[2] - f)), int(rng.integers(0, x.shape[3] - t))
            x[i, :, f0:f0 + f] = 0.0
            x[i, :, :, t0:t0 + t] = 0.0
    return x


def _freq_mixstyle(x, torch, alpha: float, *, labels=None):
    """Mix per-frequency statistics between examples (device style lives in frequency-wise patterns)."""
    mu, sig = x.mean(dim=3, keepdim=True), (x.var(dim=3, keepdim=True) + 1e-5).sqrt()
    lam = torch.distributions.Beta(alpha, alpha).sample((x.shape[0], 1, 1, 1))
    perm = torch.randperm(x.shape[0])
    if labels is not None:
        y = torch.as_tensor(labels)
        for i in range(x.shape[0]):
            peers = torch.nonzero(torch.all(y == y[i], dim=1)).flatten()
            perm[i] = peers[torch.randint(len(peers), (1,)).item()]
    return (x - mu)/sig*(lam*sig + (1 - lam)*sig[perm]) + lam*mu + (1 - lam)*mu[perm]


def _augment_training_batch(x, y, rng, cfg, torch):
    """Pair ablations to the legacy default NumPy and torch draw schedule.

    Changed mask probabilities use a replay of the control's pre-mask state.
    Changed MixStyle arms replay torch state, then reserve the control's draws,
    keeping later dropout and optimizer inputs paired even when p is zero.
    The all-default path keeps the legacy draws and output exactly.
    """
    mask_rng = training_dynamics.clone_rng(rng)
    if not cfg.get('event_aware_masks') and not cfg.get('cue_validity'):
        x = _spec_augment(x, mask_rng, cfg['p_specaugment'])
    _spec_augment(np.empty_like(x), rng, DEFAULTS['p_specaugment'])
    gate = rng.random()
    x = torch.from_numpy(x)
    p = cfg['p_freq_mixstyle']
    if p == DEFAULTS['p_freq_mixstyle'] and not cfg.get('mixstyle_same_label'):
        return _freq_mixstyle(x, torch, cfg['mixstyle_alpha']) if gate < p else x
    with torch.random.fork_rng(devices=[]):
        result = (_freq_mixstyle(x, torch, cfg['mixstyle_alpha'],
                                labels=y if cfg.get('mixstyle_same_label') else None)
                  if gate < p else x)
    if gate < DEFAULTS['p_freq_mixstyle']:
        _freq_mixstyle(x, torch, cfg['mixstyle_alpha'])
    return result


# --------------------------------------------------------------------------------- training
def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    from sensorsim.evaluation.operating_point import average_precision
    return average_precision(scores, labels)


def train(manifest: str | Path, out: str | Path, *, checkpoint_selector=None, device: str = "cpu", **overrides) -> dict:
    """Select checkpoint and detector jointly on a frozen development panel.

    Explicit diagnostic_final mode exports the terminal checkpoint without an
    acceptance claim. Synthetic AP remains a diagnostic, never a selector.
    ``device`` is cpu (default), cuda (CUDA or ROCm), or auto (GPU if available).
    GPU training is opt-in and non-bit-identical to CPU; export stays CPU/NumPy.
    """
    from sensorsim.training_execution import CheckpointExport, export_selected, resolve_device, cpu_export_model
    unknown = set(overrides) - (set(DEFAULTS) | set(training_dynamics.DEFAULTS))
    if unknown:
        raise ValueError(f"unknown CNN settings {sorted(unknown)}; "
                         f"known: {sorted(set(DEFAULTS) | set(training_dynamics.DEFAULTS))}")
    manifest, out = Path(manifest), Path(out)
    if out.suffix != ".npz":
        raise ValueError(f"the model file must end in .npz: {out}")
    torch = _torch()
    compute_device, execution = resolve_device(torch, device)
    cfg = {**DEFAULTS, **overrides, "detector": {**DEFAULTS["detector"], **overrides.get("detector", {})}}
    training_stream.configure(cfg)
    if execution is not None:
        cfg["execution"] = execution
    for key, value in training_dynamics.DEFAULTS.items():
        cfg.setdefault(key, value)
    from sensorsim.window_policy import validate_config, masked_bce
    from sensorsim import training_policy as cue
    cue_enabled = cue.validate_config(cfg)
    deployment_windows = validate_config(cfg)
    matched_negatives = training_negatives.validate_config(cfg)
    if (cfg["selection_mode"] == "operational") != callable(checkpoint_selector):
        raise ValueError("operational mode requires a frozen checkpoint_selector callback, and only that mode accepts it")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(max(1, min(4, cfg["threads"])))   # the model is tiny; data workers need the cores
    train_c = Corpus(manifest, "train", None, cfg["fs"])
    val_c = Corpus(manifest, "val", train_c.classes, cfg["fs"])
    if cue_enabled:
        cue.bind_corpus(train_c, cfg)
        cue.bind_corpus(val_c, cfg)
    if deployment_windows:
        from sensorsim.window_policy import validate_corpus
        validate_corpus(train_c, cfg)
        validate_corpus(val_c, cfg)
    if not np.isfinite(cfg["p_confuser_given_background"]) or not 0 <= cfg["p_confuser_given_background"] <= 1:
        raise ValueError("p_confuser_given_background must lie in [0,1]")
    if not isinstance(cfg["proxy_screen"], bool):
        raise ValueError("proxy_screen must be an explicit Boolean")
    if cfg["selection_mode"] in ("panel", "operational") and (out.exists() or out.with_suffix(".selection.json").exists()):
        raise FileExistsError("use a fresh output path for a selection run")
    negative_metadata = {"enabled": False, "families": [], "items": [], "proxy_screen_only": cfg["proxy_screen"]}
    if matched_negatives:
        if cfg["negative_manifest"] is None or cfg["negative_manifest_sha256"] is None:
            raise ValueError("matched/1 requires a frozen negative-training/2 manifest; no automatic fallback")
        train_c.negative_pool = training_negatives.build_training_pool(
            cfg["negative_manifest"], cfg["negative_manifest_sha256"], fs=cfg["fs"],
            room_draws=training_negatives.corpus_room_draws(manifest, train_c),
            max_procedural_share=cfg["max_procedural_share"], source_cap=cfg["negative_source_cap"],
            family_cap=cfg["negative_family_cap"])
        negative_metadata = dict(train_c.negative_pool.metadata(), enabled=True, automatic=False,
            p_confuser_given_background=cfg["p_confuser_given_background"],
            p_confuser_given_target=cfg["p_confuser_given_target"])
    elif cfg["p_confuser_given_background"] > 0:
        from sensorsim.negatives.integration import build_pool
        automatic = cfg["negative_manifest"] is None
        if automatic:
            if cfg["negative_manifest_sha256"] is not None:
                raise ValueError("negative manifest digest supplied without its file")
            from sensorsim.negatives.defaults import automatic_pool
            train_c.negative_pool = automatic_pool(manifest, train_c, cfg, out.with_suffix(".negative-inputs"))
        else:
            train_c.negative_pool = build_pool(cfg["negative_manifest"], cfg["negative_manifest_sha256"],
                fs=cfg["fs"], max_procedural_share=cfg["max_procedural_share"], allow_proxy_screen=cfg["proxy_screen"],
                allowed_room_draws={r.get("room_draw_id", r.get("model_hash")) for r in train_c.rows
                                    if r.get("room_draw_id", r.get("model_hash")) is not None})
        negative_metadata = dict(train_c.negative_pool.metadata(), enabled=True, automatic=automatic,
            p_confuser_given_background=cfg["p_confuser_given_background"])
    panel, selection_note = None, None
    if cfg["selection_mode"] == "panel":
        if cfg["selection_panel"] is None:
            if cfg["selection_panel_sha256"] is not None:
                raise ValueError("selection panel digest supplied without its file")
            from sensorsim.negatives.defaults import automatic_panel
            panel, selection_note = automatic_panel(manifest, train_c, val_c, cfg, out.with_suffix(".panel-inputs"))
        else:
            excluded = training_selection.corpus_identities(manifest, cfg["fs"], split="train")
            excluded += [dict(source_group=r["source_group"], audio_sha256=r["audio_sha256"])
                         for r in negative_metadata["items"]]
            panel = training_selection.load_panel(cfg["selection_panel"], cfg["selection_panel_sha256"],
                classes=train_c.classes, fs=cfg["fs"], excluded=excluded, min_groups=cfg["selection_min_groups"])
    elif cfg["selection_mode"] not in ("diagnostic_final", "operational"):
        raise ValueError("selection_mode must be panel, operational or diagnostic_final")
    if panel is None and cfg["selection_mode"] == "panel":
        import warnings
        warnings.warn(selection_note, UserWarning, stacklevel=2)
    model = build_model(len(train_c.classes), cfg["width"], cfg["blocks"]).to(compute_device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=.1)
    loss_fn = cue.masked_bce if cue_enabled else masked_bce if deployment_windows else torch.nn.BCEWithLogitsLoss()
    # A fixed validation set: unseen recordings, unseen background, freshly drawn microphone chains.
    val_set = Batches(val_c, {**cfg, "batch": 64, "p_confuser_given_background": 0.}, max(1, -(-cfg["val_windows"]//64)), cfg["seed"] + 999, training=False)
    val = [val_set[k] for k in range(len(val_set))]
    vx, vy = np.concatenate([v[0] for v in val]), np.concatenate([v[1] for v in val])
    # Preserve the single validation forward call and its reduction order.
    # GPU copies are made once; the CPU tensors share the existing fixed arrays.
    validation_x = torch.from_numpy(vx).to(compute_device)
    validation_y = torch.from_numpy(vy).to(compute_device)
    report_dynamics = training_dynamics.enabled(cfg) or cfg.get('event_aware_masks') or matched_negatives
    loader = _training_batches(train_c, dict(cfg, _dynamics_report=report_dynamics, _negative_report=True), torch)
    dynamics_counts = {}
    cue_records = []
    sampling_counts = dict(windows=0, background_windows=0, confuser_windows=0)
    if matched_negatives:
        sampling_counts.update(target_windows=0, target_confuser_windows=0, background_confuser_windows=0)
    label_states = {c: dict(positive=0, negative=0, unknown=0) for c in train_c.classes}
    rng = np.random.default_rng(cfg["seed"] + 1)
    best, history, start = (-1.0, None), [], time.time()
    best_step, checkpoint_scores = None, []
    panel_history, selected = [], None
    selected_export, terminal_export = None, None
    for step in range(1, cfg["steps"] + 1):
        batch = next(loader)
        x, y = batch[:2]
        for i, name in enumerate(train_c.classes):
            for state, value in (('positive', 1), ('negative', 0), ('unknown', -1)):
                label_states[name][state] += int(np.sum(y[:, i] == value))
        if len(batch) > 2 and isinstance(batch[-1], dict):
            for key, value in batch[-1].items():
                if key == 'cue_records':
                    cue.validate_consumed_records(value, y)
                    cue_records.extend(value)
                else:
                    sampling_counts[key] += int(value)
        else:
            raise ValueError("training sampling telemetry is missing")
        if report_dynamics:
            for cls, attempts, rejects, level, mask_skips in batch[2]:
                name = train_c.classes[int(cls)] if cls >= 0 else 'background'
                key = name + ':' + (str(5*int(level//5)) if np.isfinite(level) else 'none')
                totals = dynamics_counts.setdefault(key, dict(examples=0, attempts=0, rejections=0, mask_qa_skips=0))
                totals['examples'] += 1
                totals['attempts'] += int(attempts)
                totals['rejections'] += int(rejects)
                totals['mask_qa_skips'] += int(mask_skips)
        x = _augment_training_batch(x, y, rng, cfg, torch).to(compute_device)
        model.train()
        loss = loss_fn(model(x), torch.from_numpy(y).to(compute_device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == cfg["steps"]:
            model.eval()
            with torch.no_grad():
                logits = model(validation_x)
                vs = torch.sigmoid(logits).cpu().numpy()
                val_loss = float(loss_fn(logits, validation_y))
            checkpoint_scores.append(vs.copy())
            ap = {c: _average_precision(vs[:, i], vy[:, i]) for i, c in enumerate(train_c.classes)}
            mean_ap = float(np.nanmean(list(ap.values())))
            history.append({"step": step, "loss": round(float(loss.detach()), 4), "val_ap": ap, "val_mean_ap": mean_ap,
                            "val_loss": val_loss,
                            "minutes": round((time.time() - start)/60, 2)})
            take_checkpoint = panel is None and checkpoint_selector is None
            if panel is not None or checkpoint_selector is not None:
                checkpoint = out.with_name(out.stem + f".checkpoint-{step}.npz")
                checkpoint_report = export(cpu_export_model(model), train_c.classes, cfg, checkpoint, vx[:256], screen_artifact=cfg["proxy_screen"],
                       training={"best_val_mean_ap": mean_ap, "history": history, "selection": "pending",
                                 "negatives": negative_metadata, "panel_sha256": panel["sha256"] if panel else None,
                                 "config": {k: (Path(v).name if k in ("negative_manifest", "selection_panel") and v else v)
                                            for k, v in cfg.items() if k != "workers"}})
                current_export = CheckpointExport(checkpoint, checkpoint_report)
                if checkpoint_selector is not None:
                    from sensorsim.evaluation.selection import candidate_key as operational_key, evaluate_checkpoint_selector
                    panel_score = evaluate_checkpoint_selector(checkpoint_selector, checkpoint, step=step, seed=cfg["seed"])
                    candidate = panel_score["selected"]
                    if candidate is not None:
                        candidate = dict(candidate, step=step, detector=candidate["candidate"]["policy"])
                    take_checkpoint = candidate is not None and (
                        selected is None or operational_key(candidate) < operational_key(selected))
                else:
                    panel_score = training_selection.score_checkpoint(checkpoint, panel, cfg, step)
                    candidate = training_selection.choose_candidate(panel_score["policies"])
                    take_checkpoint = candidate is not None and (selected is None or
                        training_selection.candidate_key(candidate) > training_selection.candidate_key(selected))
                panel_history.append(panel_score)
                if take_checkpoint:
                    selected = candidate
                    selected_export = current_export
                if step == cfg["steps"] and selected_export is None:
                    terminal_export = current_export
                out.with_suffix(".selection.json").write_text(json.dumps(dict(
                    panel_sha256=panel["sha256"] if panel else None,
                    checkpoints=panel_history, selected=selected,
                    status="feasible" if selected else panel_score.get("status", "infeasible")), indent=2)+"\n")
            if take_checkpoint:
                best = (mean_ap, {k: v.detach().clone() for k, v in model.state_dict().items()})
                best_step = step
    if best[1] is None:
        import warnings
        selection_note = "fixed detector: no feasible checkpoint/threshold/tracker on development panel; terminal checkpoint retained"
        warnings.warn(selection_note, UserWarning, stacklevel=2)
        best = (mean_ap, {k: v.detach().clone() for k, v in model.state_dict().items()})
        best_step = cfg["steps"]
        selected_export = terminal_export
    selection_status = ("feasible" if selected else panel_history[-1]["status"]) if checkpoint_selector else ("feasible" if selected else "infeasible") if panel else (
        "fixed_fallback" if cfg["selection_mode"] == "panel" else "diagnostic_final")
    cue_summary = (cue.training_diagnostics(cue_records, train_c.classes, require_masks=cfg['cue_masks'])
                   if cue_enabled else None)
    if cue_enabled:
        cue_path = out.with_suffix('.cue-policy.json')
        cue_path.parent.mkdir(parents=True, exist_ok=True)
        cue_path.write_text(json.dumps(dict(summary=cue_summary, records=cue_records), indent=2)+'\n')
    if cue_enabled:
        if cfg.get('partial_windows') or cfg.get('cue_masks'):
            cue.require_paired_guardrails(cue_summary)
        from hashlib import sha256
        cue_summary = dict(cue_summary, trace_file=cue_path.name,
                           trace_sha256=sha256(cue_path.read_bytes()).hexdigest())
    negative_metadata['sampling'] = dict(sampling_counts,
        effective_share_of_background=(sampling_counts.get('background_confuser_windows', sampling_counts['confuser_windows'])/sampling_counts['background_windows']
                                       if sampling_counts['background_windows'] else None),
        effective_share_of_all_windows=sampling_counts['confuser_windows']/sampling_counts['windows'])
    if matched_negatives:
        negative_metadata['sampling']['effective_share_of_targets'] = (
            sampling_counts['target_confuser_windows']/sampling_counts['target_windows']
            if sampling_counts['target_windows'] else None)
    model.load_state_dict(best[1])
    model.eval()
    scores_path = out.with_suffix('.validation_scores.npz')
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_path, steps=np.array([row['step'] for row in history]),
                        scores=np.stack(checkpoint_scores), labels=vy)
    import hashlib
    report = export_selected(selected_export, export, model, train_c.classes, cfg, out, vx[:256], screen_artifact=cfg["proxy_screen"],
                    training={"steps": cfg["steps"], "best_val_mean_ap": best[0], "history": history,
                              "best_step": best_step, "selection_rule": (
                                  "P0: worst-class recall, macro recall, nuisance starts/confirmations, earlier checkpoint, threshold tuple"
                                  if checkpoint_selector else "legacy panel: worst-class recall, macro recall, triggers/starts, later checkpoint, grid order"
                                  if panel else "diagnostic terminal checkpoint; no model-selection qualification"),
                              "selection": {"panel_sha256": panel["sha256"] if panel else None,
                                            "selected": selected, "checkpoints": panel_history, "status": selection_status, "note": selection_note,
                                            "deployment_calibrated": False},
                              "negatives": negative_metadata,
                              "validation_scores": {"file": scores_path.name,
                                                    "sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                                                    "features_sha256": hashlib.sha256(vx.tobytes()).hexdigest()},
                              "torch": torch.__version__, "classes": train_c.classes,
                              "val_note": "synthetic window AP is diagnostic only; panel selection uses the deployed tracker",
                              "dynamics_rejections": dynamics_counts,
                              "cue_policy": cue_summary,
                              "window_supervision": {"policy": cue.POLICY if cue_enabled else cfg['window_policy'], "states": label_states,
                                  "unknown_loss": "masked class cells", "default_promotion": "not qualified"},
                              "prior_note": {"training_positive_fraction": cfg.get('positive_fraction') or 1-cfg['p_background_only'],
                                             "default_on": cfg['detector']['default_on'],
                                             "deployment_calibrated": False},
                              "config": {k: (Path(v).name if k in ("negative_manifest", "selection_panel") and v else v)
                                         for k, v in cfg.items() if k != "workers"}})
    report["dynamics_rejections"] = dynamics_counts
    report['cue_policy'] = cue_summary
    report['window_supervision'] = {"policy": cue.POLICY if cue_enabled else cfg['window_policy'], "states": label_states}
    tracker_cfg = training_dynamics.validation_config(dict(cfg, validation_mode='synthetic'))
    if selected is not None:
        d = selected["detector"]
        on = d["on"] if isinstance(d["on"], dict) else {c: d["on"] for c in train_c.classes}
        fields = dict(on=on, smooth_hops=d["smooth_hops"], min_hops=d["min_hops"],
                      off_ratio=d["off_ratio"], hop_s=d["hop_s"])
        edge.save_detector(str(out), str(out), **fields)
        report["detector"] = fields
    elif cfg["selection_mode"] in ("panel", "operational"):
        report["detector"] = edge.Model(str(out)).meta["detector"]
    else:
        report["detector"] = _fit_tracker(edge.Model(str(out)), val_c, tracker_cfg, out)
    if checkpoint_selector is not None:
        report["artifact_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        report["selected_checkpoint_sha256"] = selected["candidate"]["model_sha256"] if selected else None
        report["tracker_order"] = "policy frozen before scoring; no post-selection fitting"
    report["negatives"] = negative_metadata
    report["selection"] = {"mode": cfg["selection_mode"], "selected": selected, "status": selection_status, "note": selection_note,
                           "panel_sha256": panel["sha256"] if panel else None}
    return report


def _persistence(model: edge.Model, corpus: Corpus, cfg: dict, seed: int, per_class: int = 24) -> dict:
    """Per class: median number of consecutive hops a validation event scores >= 0.5.

    Each event is mixed into unseen background with the same level and device contract
    as training, and scored by the deployed runtime. A long sound (a kettle) stays visible for many
    hops; a short transient (a knock) may fill a single window, and smoothing it over several
    hops would average it away. Events never scored >= 0.5 are left out of the median.
    """
    from sensorsim.event_levels import require_legacy_training
    require_legacy_training(cfg)
    rng = np.random.default_rng(seed)
    fs, n = cfg["fs"], model.features.n
    hop = int(round(cfg["detector"]["hop_s"]*fs))
    result = {}
    for c, name in enumerate(corpus.classes):
        runs = []
        for _ in range(per_class):
            k = int(rng.integers(len(corpus.targets[name])))
            x, (a, b) = corpus.targets[name][k], corpus.spans[name][k]
            background = corpus.background[int(rng.integers(len(corpus.background)))]
            audio = _crop(background, 3*n + b - a, rng).astype(float)      # lead-in, a window, the event, a window
            event = np.zeros_like(audio)
            event[2*n:2*n + b - a] = x[a:b]
            audio = mix_device_audio(audio, event, (2*n, 2*n + b - a), rng, cfg)
            scores = edge.window_scores(model, audio[n:].astype(np.float32), hop/fs)[:, c]
            starts = np.arange(scores.size)*hop
            run = best = 0
            for high in scores[(starts < n + b - a) & (starts + n > n)] >= 0.5:
                run = run + 1 if high else 0
                best = max(best, run)
            if best:
                runs.append(best)
        result[name] = int(np.median(runs)) if runs else None
    return result


def _fit_tracker(model: edge.Model, corpus: Corpus, cfg: dict, out: Path) -> dict:
    """Cap each class's smoothing and minimum length at its measured persistence; save them.

    Values given per class in the config ({"knock": 1}) are the caller's choice and win; the
    other classes are fitted.
    """
    d = cfg["detector"]
    hops = _persistence(model, corpus, cfg, cfg["seed"] + 7)
    fields = {"persistence_hops": hops}
    for key in ("smooth_hops", "min_hops"):
        given = d[key] if isinstance(d[key], dict) else {}
        unknown = set(given) - set(hops)
        if unknown:
            raise ValueError(f"detector {key} names unknown classes {sorted(unknown)}; classes: {sorted(hops)}")
        cap = DEFAULTS["detector"][key] if isinstance(d[key], dict) else d[key]
        fields[key] = {c: int(given[c]) if c in given else int(min(cap, h or cap)) for c, h in hops.items()}
    edge.save_detector(str(out), str(out), **fields)
    return fields


def _training_batches(corpus: Corpus, cfg: dict, torch):
    """The training stream, built in worker processes (numpy data generation is the bottleneck, not the model)."""
    data = Batches(corpus, cfg, cfg["steps"], cfg["seed"])
    if cfg["workers"] <= 0:
        return (data[k] for k in range(len(data)))
    loader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=False, num_workers=cfg["workers"],
                                         **training_stream.loader_options(cfg, torch))
    if training_stream.version(cfg) == "stream/2":
        return iter(loader)
    return (tuple(item if isinstance(item, dict) else item.numpy() for item in batch) for batch in loader)


# ------------------------------------------------------------------------------------- export
def export(model, classes: list[str], cfg: dict, out: Path, check_features: np.ndarray, training: dict, *, screen_artifact=False) -> dict:
    """Fold BatchNorm, write the edge .npz, and verify the numpy runtime against PyTorch."""
    from sensorsim.negatives.integration import require_production_metadata
    if not screen_artifact:
        require_production_metadata(training)
        require_production_metadata({"config": cfg})
    elif cfg.get("proxy_screen") is not True:
        raise ValueError("screen artifacts require explicit proxy_screen")
    torch = _torch()
    arrays, layers = {}, []

    def fold(conv, bn):
        scale = (bn.weight/torch.sqrt(bn.running_var + bn.eps)).detach()
        w = (conv.weight.detach()*scale[:, None, None, None]).numpy().astype(np.float32)
        b = (bn.bias.detach() - bn.running_mean*scale).numpy().astype(np.float32)
        return w, b

    arrays["conv0.w"], arrays["conv0.b"] = fold(model.conv0, model.conv0_bn)
    layers.append({"type": "conv", "name": "conv0", "stride": [2, 2], "pad": [2, 2], "act": "relu"})
    for i, block in enumerate(model.blocks):
        w, b = fold(block.dw, block.dw_bn)
        arrays[f"b{i}.dw.w"], arrays[f"b{i}.dw.b"] = w[:, 0], b
        layers.append({"type": "dw", "name": f"b{i}.dw", "stride": list(block.dw.stride), "pad": [1, 1], "act": "relu"})
        w, b = fold(block.pw, block.pw_bn)
        arrays[f"b{i}.pw.w"], arrays[f"b{i}.pw.b"] = w[:, :, 0, 0], b
        layers.append({"type": "pw", "name": f"b{i}.pw", "act": "relu"})
    layers.append({"type": "pool"})
    arrays["fc.w"] = model.fc.weight.detach().numpy().astype(np.float32)
    arrays["fc.b"] = model.fc.bias.detach().numpy().astype(np.float32)
    layers.append({"type": "fc", "name": "fc"})
    d = cfg["detector"]
    meta = {"format": edge.FORMAT, "classes": list(classes), "features": _feature_cfg(cfg), "layers": layers,
            "detector": {"hop_s": d["hop_s"], "off_ratio": d["off_ratio"], "smooth_hops": d["smooth_hops"],
                         "min_hops": d["min_hops"], "on": {c: d["default_on"] for c in classes}},
            "training": training}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    # Parity on REAL validation features, not random input.
    runtime = edge.Model(str(out))
    with torch.no_grad():
        reference = torch.sigmoid(model(torch.from_numpy(check_features))).numpy()
    ours = 1/(1 + np.exp(-runtime.logits(check_features)))
    parity = float(np.max(np.abs(reference - ours)))
    if not parity < 1e-4:
        raise RuntimeError(f"numpy runtime disagrees with PyTorch by {parity:.2e}; export refused")
    return {"model": str(out), "classes": list(classes), "parity_max_abs_diff": parity,
            "parameters": int(sum(a.size for k, a in arrays.items() if k != "meta")), "macs_per_window": runtime.macs(),
            "best_val_mean_ap": training["best_val_mean_ap"], "history_tail": training["history"][-2:]}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="train the edge CNN on a dataset manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--width", type=int, default=DEFAULTS["width"])
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    for name in ("negative_manifest", "selection_panel"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path)
        parser.add_argument("--"+name.replace("_", "-")+"-sha256")
    parser.add_argument("--p-confuser-given-background", type=float, default=DEFAULTS["p_confuser_given_background"])
    parser.add_argument("--max-procedural-share", type=float, default=DEFAULTS["max_procedural_share"])
    parser.add_argument("--selection-mode", choices=("panel", "diagnostic_final"), default=DEFAULTS["selection_mode"])
    args = parser.parse_args(argv)
    settings = vars(args).copy()
    settings.pop("manifest"); settings.pop("out")
    report = train(args.manifest, args.out, **settings)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
