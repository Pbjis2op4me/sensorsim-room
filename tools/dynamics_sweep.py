# SPDX-License-Identifier: Apache-2.0
"""Audit the actual CNN training stream; does not train or change the pipeline.

Usage (from a checkout)::

    python tools/dynamics_sweep.py --config sweep.json --prereg preregistration.yaml \
        --prereg-sha prereg_sha256.txt --out evidence/sweep

Config contains ``cases`` (target/floor WAV paths relative to config, label,
source_group, evidence_kind, proxy, source_spl_db, distance_m, room), ``seeds``,
``snr_db``, ``event_spl_db``, ``profiles`` (default_pi/stress), and ``augmentations``.
``evidence_kind`` is 'model-based' for all proxy-convolved audio, even measured
RIRs; 'measured' describes an actual input recording, never the generated mixture.
Optional reference_takes contain audio, label, source_group, split='heldout',
and evidence_kind='measured'; these are scored directly at the deployed hop,
without augmentation, in references.csv. A case may supply an
``ideal_target`` WAV for intermediate quantization error, ``boundaries`` in
samples and ``floor_period_samples``. No third-party audio is included here.
New user takes use this same schema without re-convolution; source level and
room may be null if unknown. Never infer SPL from an uncalibrated WAV.

The current make_example and recognizer helpers are called, not reimplemented.
A scoped observer captures the pre-chain mixture, then replays the exact chain
and noise RNG on the background alone. Event = mixture minus this counterfactual
floor. For nonlinear AGC/NS/codecs this is an EFFECTIVE contribution, not an
independent physical source estimate. Device arms select the current production contract: default Pi or opt-in stress.
Old pinned-chain substitutions are rejected because they double-process floors.
Observers temporarily patch module globals: run this audit single-threaded.

Levels: RMS dBFS uses peak full scale; *_sine_dbfs adds 3.0103 dB. All zero
levels use a -300 dB reporting sentinel. No acoustic statistic establishes
recognition accuracy, false alarms, or WP7 model-ablation acceptance.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import signal

from sensorsim import edge, iot_mic, nn, recognizer, training_device, training_dynamics
from sensorsim.acoustics.weighting import a_weighted_mean_square

SINE_DB = 10*np.log10(2.)


def db(power):
    return float(10*np.log10(max(float(power), 1e-30)))


def levels(x, fs):
    raw = np.asarray(x, dtype=float)
    x = raw - raw.mean()
    p = float(np.mean(x*x))
    peak = float(np.max(np.abs(x)))
    a = db(a_weighted_mean_square(x, fs))
    return {'rms_dbfs': db(p), 'peak_dbfs': db(peak*peak),
            'a_dbfs': a, 'a_sine_dbfs': a + SINE_DB,
            'crest_db': db(peak*peak)-db(p), 'dc_offset_fs': float(raw.mean()),
            'raw_rms_dbfs': db(np.mean(raw*raw)),
            'raw_peak_dbfs': db(np.max(np.abs(raw))**2),
            'clipped_fraction': float(np.mean(np.abs(raw) >= 32767/32768)),
            'zero_fraction': float(np.mean(raw == 0)), 'digital_silence': bool(not np.any(raw))}


def input_qa(x, fs):
    x = np.asarray(x, float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('input audio must be finite nonempty mono')
    return dict(samples=len(x), peak_fs=float(np.max(np.abs(x))),
                clipped_samples=int(np.sum(np.abs(x) >= 32767/32768)), **levels(x, fs))


def mel_power(x, features):
    # Same window, hop and bank as edge.Features, without gain normalization/log.
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    frames = np.lib.stride_tricks.sliding_window_view(x, features.cfg['n_fft'])[::features.cfg['hop']]
    return (np.abs(np.fft.rfft(frames*features.window, axis=-1))**2 @ features.bank.T).T


def audibility(event, floor, features):
    event = np.asarray(event, float) - np.mean(event)
    floor = np.asarray(floor, float) - np.mean(floor)
    e, f = mel_power(event, features), mel_power(floor, features)
    active = e.sum(axis=0) > max(e.sum(axis=0).max()*10**(-2.5), 1e-30)
    audible = np.any(e >= 10**(.3)*np.maximum(f, 1e-30), axis=0) & active
    return {'audible_frames': int(audible.sum()), 'event_frames': int(active.sum()),
            'below_audibility_rule': bool(audible.sum() < 3),
            'event_to_floor_db': db(np.mean(event**2))-db(np.mean(floor**2)),
            'event_to_floor_a_db': db(a_weighted_mean_square(event, features.cfg['fs']))-
                                   db(a_weighted_mean_square(floor, features.cfg['fs']))}


def mask_metrics(mask, event_power):
    frame_energy = event_power.sum(axis=0)
    active = frame_energy > max(frame_energy.max()*10**(-2.5), 1e-30)
    masked = np.all(mask, axis=0)
    return {'masked_event_frame_fraction': float(np.sum(masked & active)/max(1, active.sum())),
            'masked_event_tf_energy_fraction': float(np.sum(event_power*mask)/max(1e-30, event_power.sum())),
            'time_mask_frames': int(masked.sum())}


def fingerprints(x, fs, boundaries=(), floor_period_samples=None, baseline_difference_rms=None):
    x = np.asarray(x, dtype=float)
    freq, power = signal.welch(x, fs, nperseg=min(1024, len(x)))
    energy = lambda a, b: np.sum(power[(freq >= a) & (freq < b)])
    block = max(1, round(.02*fs))
    blocks = x[:len(x)//block*block].reshape(-1, block)
    hashes = [hashlib.sha256(b.tobytes()).digest() for b in blocks if np.any(b)]
    repeats = (len(hashes)-len(set(hashes)))/max(1, len(blocks))
    delta = np.diff(x)
    baseline = (float(np.sqrt(np.mean(delta*delta))) if baseline_difference_rms is None
                else float(baseline_difference_rms))
    indices = [int(i) for i in boundaries if 0 < i < len(x)]
    jumps = np.abs(x[indices]-x[np.asarray(indices)-1]) if indices else np.array([])
    lag = int(floor_period_samples or 0)
    corr = None
    if 0 < lag < len(x) and np.std(x[:-lag]) > 0 and np.std(x[lag:]) > 0:
        corr = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    return {'high_low_band_db': db(energy(4000, 7600))-db(energy(500, 3500)),
            'edge_4k_db': db(energy(4000, 4400))-db(energy(3600, 4000)),
            'repeated_block_fraction': float(repeats), 'floor_period_correlation': corr,
            'boundary_count': len(indices),
            'boundary_click_candidates': int(np.sum(jumps > 8*baseline)) if baseline > 0 else None,
            'max_boundary_step': float(jumps.max()) if jumps.size else None,
            'global_step_candidates': int(np.sum(np.abs(delta) > 8*baseline)) if baseline > 0 else None}


def _clone_rng(rng):
    clone = np.random.Generator(type(rng.bit_generator)())
    clone.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return clone


def trace_example(corpus, rng, cfg, comparison_chain=None):
    """Observe the production mixer and device calls; unknown streams fail closed.

    Counterfactual floor uses the observed chain and cloned noise RNG. For recorded
    Pi backgrounds it is the unchanged recorded bed (float addition), with no second
    filtering or self-noise. Nonlinear stress differences are effective contributions.
    """
    if comparison_chain is not None:
        raise ValueError('legacy pinned-chain comparison is unsupported; select a device contract')
    trace = {}
    original_event, original_mixer = nn._event_window, nn.mix_device_audio
    original_apply, original_quantize = iot_mic.MicChain.apply, iot_mic.quantize_profile
    n = round(cfg['window_s']*cfg['fs'])
    inside_apply = False

    def event(*args):
        trace['event_rng_state'] = copy.deepcopy(args[-1].bit_generator.state)
        y, span = original_event(*args)
        trace['window'], trace['span'] = y.copy(), span
        return y, span

    def apply(chain, x, fs, chain_rng, **kwargs):
        nonlocal inside_apply
        trace['chain'] = iot_mic.describe(chain)
        trace['stage_input'] = np.asarray(x).copy()
        bed = trace['input_floor']
        is_recorded = cfg.get('device_split', 'in_distribution') == 'in_distribution' and cfg.get('background_domain', 'device_recorded') == 'device_recorded'
        inside_apply = True
        try:
            if is_recorded:
                trace['floor_full'] = np.asarray(bed, float).copy()  # capture keeps the recorded bed
                basis = trace['input_event']
                residual = x
            else:
                trace['floor_full'] = original_apply(chain, bed, fs, _clone_rng(chain_rng), **kwargs)
                basis = trace['input_event']
                residual = np.asarray(x)-bed
            trace['mix_gain'] = float(np.dot(residual, basis)/max(np.dot(basis, basis), 1e-30))
            return original_apply(chain, x, fs, chain_rng, **kwargs)
        finally:
            inside_apply = False

    def quantize(chain, x):
        if not inside_apply:
            trace['chain'] = iot_mic.describe(chain)
            trace['floor_full'] = original_quantize(chain, trace['input_floor'])
        return original_quantize(chain, x)

    def mixer(background, event_audio, span, mix_rng, mix_cfg):
        trace.update(hook='mix_device_audio', input_floor=background.copy(), input_event=event_audio.copy())
        y = original_mixer(background, event_audio, span, mix_rng, mix_cfg)
        trace['mixer_output'] = y.copy()
        if 'floor_full' not in trace:
            if mix_cfg.get('device_split', 'in_distribution') == 'stress':
                # An observed mixer call with no apply is the explicit no-chain branch.
                trace['floor_full'] = np.clip(np.round(background*32767)/32767, -1, 1)
                trace['chain'] = {'name':'stress_no_chain_pcm16', 'codec':'pcm16'}
            elif not np.any(event_audio):
                trace['floor_full'] = y.copy()
                trace['chain'] = {'name':'recorded_background_passthrough', 'profile_id':mix_cfg.get('device_profile')}
            else:
                raise RuntimeError('device apply/quantize hook did not fire')
        return y

    with patch.object(nn, '_event_window', event), patch.object(nn, 'mix_device_audio', mixer), \
         patch.object(iot_mic.MicChain, 'apply', apply), patch.object(iot_mic, 'quantize_profile', quantize):
        audio, label = nn.make_example(corpus, rng, cfg)
    if trace.get('hook') != 'mix_device_audio':
        raise RuntimeError('production mixer hook did not fire')
    trace['floor'] = trace['floor_full'][n:].astype(np.float32)
    trace['event'] = audio.astype(float)-trace['floor'].astype(float)
    trace.setdefault('mix_gain', None)
    trace['mixer_return_error'] = float(np.max(np.abs(audio-trace['mixer_output'][n:].astype(np.float32))))
    return audio, label, trace


def post_chain_quantization(corpus, ideal, seed, cfg, *, observed_audio=None):
    """Paired storage error after device response, BEFORE final capture rounding.

    Observe (do not replace) the production capture input (recorded bed plus rendered
    increment). Any capture-stage difference is reported separately and never compared
    to the storage margin.
    Support/placement and random draws are fixed across the stored/ideal pair.
    This metric is qualified only for a single positive, device-recorded Pi case.
    """
    if (cfg.get('device_split', 'in_distribution') != 'in_distribution' or
            cfg.get('background_domain', 'device_recorded') != 'device_recorded'):
        return {'quantization_error_a_sine_dbfs': None, 'quantization_margin_db': None,
                'quantization_stage': 'unsupported capture path'}
    if len(corpus.classes) != 1 or len(corpus.targets[corpus.classes[0]]) != 1:
        raise ValueError('storage metric requires exactly one target source')
    stored = corpus.targets[corpus.classes[0]][0]
    ideal = np.asarray(ideal, float)
    if (ideal.shape != stored.shape or ideal.ndim != 1 or not ideal.size or
            not np.isfinite(ideal).all() or not np.isfinite(stored).all()):
        raise ValueError('storage pair must be finite nonempty mono with identical shape')
    if not np.any(ideal) or not np.any(stored):
        raise ValueError('storage pair must contain a nonzero target')
    original_capture = training_device.conditional_capture
    captured = []
    def observe(background, increment, capture_rng, profile):
        captured.append(np.asarray(background, float) + np.asarray(increment, float))
        return original_capture(background, increment, capture_rng, profile)
    ideal_c = copy.copy(corpus)
    ideal_c.targets = {corpus.classes[0]: [ideal]}
    results = []
    for current in (corpus, ideal_c):
        captured.clear()
        with patch.object(training_device, 'conditional_capture', observe):
            audio, label = nn.make_example(current, np.random.default_rng(seed), cfg)
        if len(captured) != 1 or not label.any():
            raise RuntimeError('storage metric final capture hook must fire exactly once on a positive')
        n = round(cfg['window_s']*cfg['fs'])
        results.append((captured[0][n:], audio.astype(float)))
    if observed_audio is not None and not np.array_equal(observed_audio, results[0][1]):
        raise RuntimeError('storage replay differs from observed production audio')
    error = levels(results[0][0]-results[1][0], cfg['fs'])['a_sine_dbfs']
    captured_error = levels(results[0][1]-results[1][1], cfg['fs'])['a_sine_dbfs']
    rounding = levels((results[0][1]-results[0][0])-(results[1][1]-results[1][0]),
                      cfg['fs'])['a_sine_dbfs']
    return {'quantization_error_a_sine_dbfs': error, 'quantization_margin_db': -87-error,
            'post_capture_source_difference_a_sine_dbfs': captured_error,
            'final_capture_rounding_difference_a_sine_dbfs': rounding,
            'quantization_stage': 'pre-final-capture'}


def verify_preregistration(path, hash_path):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != Path(hash_path).read_text().split()[0]:
        raise ValueError('preregistration hash mismatch')
    return actual


def _write_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_case(case, config_dir, fs):
    for key in ('id', 'source_group', 'label', 'evidence_kind', 'target', 'floor'):
        if key not in case:
            raise ValueError(f'case missing {key}')
    if case['evidence_kind'] not in ('model-based', 'measured'):
        raise ValueError('evidence_kind must be model-based or measured')
    if case.get('proxy') and case['evidence_kind'] != 'model-based':
        raise ValueError('proxy audio must be model-based, including measured-RIR renders')
    x, floor = (recognizer.read_audio(config_dir/case[k], fs) for k in ('target', 'floor'))
    for y in (x, floor):
        if not y.size or not np.isfinite(y).all():
            raise ValueError('empty or nonfinite source')
    return x, floor


def reference_windows(references, root, training_groups, features):
    """Direct held-out capture statistics; never pass reference recordings through augmentation."""
    rows = []
    for ref in references:
        if ref.get('evidence_kind') != 'measured' or ref.get('split') != 'heldout':
            raise ValueError('references require measured heldout takes')
        if ref['source_group'] in training_groups:
            raise ValueError('reference take group overlaps training inputs')
        x = recognizer.read_audio(root/ref['audio'], features.cfg['fs'])
        if len(x) < features.n or not np.isfinite(x).all():
            raise ValueError('reference audio must be finite and at least one window long')
        for start in range(0,len(x)-features.n+1,round(.25*features.cfg['fs'])):
            w=x[start:start+features.n]
            rows.append(dict(source_group=ref['source_group'],label=ref['label'],
                             evidence_kind='measured',start_sample=start,
                             feature_rms=float(np.sqrt(np.mean(features(w)**2))),
                             **levels(w,features.cfg['fs'])))
    return rows


def run(config_path, prereg, prereg_sha, output):
    prereg_hash = verify_preregistration(prereg, prereg_sha)
    path, output = Path(config_path), Path(output)
    config = json.loads(path.read_text())
    if output.exists():
        raise FileExistsError('use a fresh output directory for each audit')
    profiles = config['profiles']
    if set(profiles) - {'random_chain', 'stress', 'default_pi', 'pi5_inmp441/1'}:
        raise ValueError('unknown profile')
    if set(config['augmentations']) - {'off', 'spec', 'mixstyle', 'spec_mixstyle'}:
        raise ValueError('unknown augmentation')
    comparison_sha = None
    if config.get('wp5_revision'):
        raise ValueError('use current device contracts; pinned pre-WP5 comparison is obsolete')
    torch = nn._torch() if any('mixstyle' in a for a in config['augmentations']) else None
    if torch is not None:
        torch.set_num_threads(2)
    feature_fn = edge.Features(nn._feature_cfg(nn.DEFAULTS))
    fs, n = nn.DEFAULTS['fs'], feature_fn.n
    rows, input_hashes, qa_rows = [], {}, []
    for case_number, case in enumerate(config['cases']):
        x, floor = _read_case(case, path.parent, fs)
        qa_rows.extend(dict(id=case['id'], role=role, **input_qa(y, fs)) for role,y in [('target',x),('floor',floor)])
        ideal = recognizer.read_audio(path.parent/case['ideal_target'], fs) if case.get('ideal_target') else None
        if ideal is not None and len(ideal) != len(x):
            raise ValueError('ideal_target must have identical sample count')
        for key in ('target', 'floor', 'ideal_target'):
            if case.get(key):
                input_hashes[case[key]] = hashlib.sha256((path.parent/case[key]).read_bytes()).hexdigest()
        c = SimpleNamespace(classes=[case['label']], targets={case['label']: [x]},
                            spans={case['label']: [recognizer._audible_span(x, fs)]}, background=[floor])
        group_seed = int.from_bytes(hashlib.sha256(case['source_group'].encode()).digest()[:4], 'little')
        for snr, profile, seed in itertools.product(config['snr_db'], profiles, config['seeds']):
            stress = profile in ('random_chain', 'stress')
            level_index = config['snr_db'].index(snr)
            level_grid = config.get('event_spl_db', np.linspace(60, 80, len(config['snr_db'])))
            if len(level_grid) != len(config['snr_db']):
                raise ValueError('event_spl_db and snr_db grids must have the same number of cells')
            event_spl = float(level_grid[level_index])
            cfg = dict(nn.DEFAULTS, snr_db=[snr, snr], event_spl_db=[event_spl,event_spl],
                       p_background_only=0, device_split='stress' if stress else 'in_distribution')
            # Case id/order never changes random draws: matched conditions share seed.
            audio, label, trace = trace_example(c, np.random.default_rng([seed,group_seed]), cfg,
                                                None)
            features = feature_fn(audio)
            floor_features = feature_fn(trace['floor'])
            event_power = mel_power(trace['event'], feature_fn)
            a, b = trace.get('span', (0, n))
            boundaries = [a, b, *case.get('boundaries', [])]
            period = case.get('floor_period_samples')
            if period and period < 2*n:
                boundaries += [i-n for i in range(period, 2*n, period) if i > n]
            base = {k: case.get(k) for k in ('id','source_group','label','evidence_kind','proxy',
                    'source_spl_db','distance_m','room','floor_dba')}
            base['source_evidence_kind'] = case['evidence_kind']
            base['evidence_kind'] = 'model-based'
            base.update(seed=seed, snr_db=snr if stress else None, event_spl_db=None if stress else event_spl,
                        level_cell=snr if stress else event_spl, profile=profile, positive=bool(label.any()),
                        codec=trace['chain'].get('codec'), agc=bool(trace['chain'].get('agc')),
                        noise_suppression=bool(trace['chain'].get('noise_suppression')),
                        packet_loss=trace['chain'].get('packet_loss', 0), mix_gain=trace['mix_gain'],
                        mixer_return_error=trace['mixer_return_error'], hook=trace['hook'],
                        chain_name=trace['chain'].get('profile_id') or trace['chain'].get('name'),
                        known_floor_tiled=bool(len(floor) < 2*n),
                        recognizer_feature_l2=float(np.linalg.norm(recognizer.features(audio, recognizer.feature_config(fs,1.)))),
                        feature_rms=float(np.sqrt(np.mean(features**2))),
                        waveform_sha256=hashlib.sha256(audio.tobytes()).hexdigest())
            base.update(levels(audio, fs))
            base.update({'floor_'+k: v for k,v in levels(trace['floor'],fs).items()})
            base['floor_too_clean'] = base['floor_a_sine_dbfs'] < -90
            base.update(audibility(trace['event'],trace['floor'],feature_fn))
            base.update(fingerprints(audio,fs,boundaries,period,
                                     np.sqrt(np.mean(np.diff(trace['floor'])**2))))
            # For an ideal/PCM paired source, preserve placement and production gain.
            base['quantization_error_a_sine_dbfs'] = None
            base['quantization_margin_db'] = None
            if ideal is not None:
                base.update(post_chain_quantization(c, ideal, [seed,group_seed], cfg, observed_audio=audio))
            # A symmetric resampling round-trip stress diagnostic, not measured capture.
            rt = signal.resample_poly(signal.resample_poly(audio,3,1),1,3)[:n]
            base['resample_roundtrip_feature_rms_delta'] = float(np.sqrt(np.mean((feature_fn(rt)-features)**2)))
            for augmentation in config['augmentations']:
                batch = np.concatenate([features, floor_features])
                mask = np.zeros_like(batch, dtype=bool)
                if 'spec' in augmentation:
                    # A ones sentinel with the same RNG reveals masks without conflating true zeros.
                    sentinel = nn._spec_augment(np.ones_like(batch), np.random.default_rng([seed,group_seed,42]), .5)
                    mask = sentinel == 0
                    batch = nn._spec_augment(batch.copy(), np.random.default_rng([seed,group_seed,42]), .5)
                before_style = batch.copy()
                style_applied = False
                if 'mixstyle' in augmentation and np.random.default_rng([seed,group_seed,43]).random() < .4:
                    torch.manual_seed(seed+group_seed)
                    batch = nn._freq_mixstyle(torch.from_numpy(batch), torch, .4).numpy()
                    style_applied = True
                row = dict(base, augmentation=augmentation, mixstyle_applied=style_applied,
                           independent_draw_id=f'{seed}:{group_seed}',
                           mixstyle_feature_rms_delta=float(np.sqrt(np.mean((batch-before_style)**2))))
                row.update(mask_metrics(mask[0,0],event_power))
                rows.append(row)
        print(f'case {case_number+1}/{len(config["cases"])} complete', flush=True)
    references = reference_windows(config.get('reference_takes', []), path.parent,
                                   {c['source_group'] for c in config['cases']}, feature_fn)
    for ref in config.get('reference_takes', []):
        input_hashes[ref['audio']] = hashlib.sha256((path.parent/ref['audio']).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    if references:
        _write_csv(output/'references.csv', references)
    _write_csv(output/'windows.csv',rows)
    _write_csv(output/'input_qa.csv',qa_rows)
    # Per-cell quantiles across seeds (descriptive, no inferential interval from n=3).
    groups = {}
    for row in rows:
        key = (row['id'],row['level_cell'],row['profile'],row['augmentation'])
        groups.setdefault(key, []).append(row)
    cells=[]
    for key, group in groups.items():
        cell = dict(zip(('id','level_cell','profile','augmentation'),key), n=len(group))
        for k,v in group[0].items():
            if isinstance(v,(int,float,bool)) and k not in ('seed','snr_db'):
                vals = [float(r[k]) for r in group if r[k] is not None]
                for name,q in [('p05',.05),('p50',.5),('p95',.95)]:
                    cell[k+'_'+name]=float(np.quantile(vals,q))
                cell[k+'_mean']=float(np.mean(vals))
        cells.append(cell)
    _write_csv(output/'cells.csv', cells)
    audit = {'schema':'dynamics-sweep/3','prereg_sha256':prereg_hash,
             'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'pipeline_sha256':{m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                for m in (nn,recognizer,edge,iot_mic,training_device,training_dynamics)},
             'wp5_revision':config.get('wp5_revision'),'wp5_module_sha256':comparison_sha,
             'input_sha256':input_hashes,'rows':len(rows),'cells':len(cells),
             'proxy_rows':sum(r['evidence_kind']=='model-based' for r in rows),
             'measured_rows':len(references),
             'conditional_positive_sampling':True,
             'independent_augmentation_draws':len({r['independent_draw_id'] for r in rows}),
             'waveform_conditions':sum(r['augmentation']=='off' for r in rows),
             'input_clipped_samples':sum(r['clipped_samples'] for r in qa_rows),
             'quantization_stage':'pre-final-capture paired source; rounding difference separate; Pi recorded beds only',
             'mixstyle_probe':'two-example batch: target mixture and its paired floor; not the default batch of 128',
             'below_audibility':sum(r['positive'] and r['below_audibility_rule'] for r in rows),
             'over_half_masked':sum(r['positive'] and r['masked_event_frame_fraction']>.5 for r in rows),
             'clipping_match':'inconclusive: no held-out real clipping reference supplied',
             'wp7_adoption':'inconclusive: descriptive sweep, no paired trained-model ablations',
             'versions':{'numpy':np.__version__, 'python':sys.version.split()[0]}}
    if references:
        ref_clip = float(np.mean([r['clipped_fraction'] for r in references]))
        deltas = {profile: float(np.mean([r['clipped_fraction'] for r in rows
                  if r['profile']==profile and r['augmentation']=='off']))-ref_clip for profile in profiles}
        audit['clipping_match'] = {'reference_fraction':ref_clip, 'deltas':deltas,
                                  'outcomes':{p:'pass' if abs(d)<=.01 else 'fail' for p,d in deltas.items()}}
    audit['unit_gate_outcome']='fail' if audit['below_audibility'] or audit['over_half_masked'] else 'inconclusive'
    (output/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    return audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--prereg',required=True,type=Path)
    parser.add_argument('--prereg-sha',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    audit=run(args.config,args.prereg,args.prereg_sha,args.out)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('input_sha256','pipeline_sha256')},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
