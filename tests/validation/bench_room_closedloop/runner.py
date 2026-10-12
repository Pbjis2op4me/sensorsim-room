"""Engine-neutral closed loop. The system under test receives only the session."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import inspect
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import warnings

import numpy as np
from scipy import signal
from scipy.io import wavfile

from sensorsim.acoustics.room import pipeline
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import synthesize
from .compare import compare_dry, compare_parameters, compare_rirs, compare_sound, compare_timing, context
from .engines import assemble_session, stable_seed, synthesis_config
from .oracle_reference import conditional_wiener_reference
from .scorecard import row, summarize, write_json, write_scorecard

# Existing measured-tier drivers import this evaluator name. The implementation
# remains separate from acquisition and is never passed to the public pipeline.
oracle_dry = conditional_wiener_reference


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_hash(path):
    """Hash of a provenance document, or None: the parent project's protocol documents are not
    distributed with this package."""
    return file_hash(path) if Path(path).exists() else None


def versions(*names):
    """Installed versions; None for an optional package that is absent (torch)."""
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _legacy_timing(metadata, fs, kit):
    """Read shared detector ADC metadata; missing detection remains missing.

    Retain the adapter name/signature for other tier drivers.
    """
    if "sweep_detection" in metadata:
        detection = metadata["sweep_detection"]
        repeats = detection.get("repeats", [])
        if not repeats:
            return None
        return dict(start_s=repeats[0]["start_sample"]/fs,
                    end_s=repeats[-1]["end_sample"]/fs,
                    clock_ppm=detection.get("clock_drift_ppm"))
    return None


def load_public_model(output):
    path = Path(output)/"room_model.json"
    return RoomModel.from_json(path.read_text()) if path.exists() else None


def public_outcome(report, output, completed_dry_takes):
    """Observe public coverage separately from execution success or stage name."""
    output = Path(output)
    run = json.loads((output/'run.json').read_text()) if (output/'run.json').exists() else {}
    state = (report or {}).get('state', 'incomplete')
    events = (report or {}).get('event_selection')
    if events is None and (output/'dry/event_summary.json').exists():
        events = json.loads((output/'dry/event_summary.json').read_text())
    stage = run.get('stage', 'preflight')
    return dict(pipeline_complete=report is not None and state == 'complete', pipeline_state=state,
                pipeline_stage=stage, event_selection=events, completed_dry_takes=completed_dry_takes,
                session_status=state if state in ('complete','partial') else 'stopped-at-'+stage)


@dataclass(frozen=True)
class PipelineAttempt:
    """One pipeline call whose output directory was fresh when it began."""
    session: Path
    output: Path


def begin_attempt(session, output):
    """Bind recovery to a new output directory so stale files cannot be scored."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'a pipeline attempt needs a fresh output folder; {output} exists')
    return PipelineAttempt(Path(session).resolve(), output)


def recover_dry_takes(attempt, takes, fs):
    """Recover fully written dry products only when their attempt and source are proven."""
    if not isinstance(attempt, PipelineAttempt):
        raise TypeError('dry recovery needs the PipelineAttempt that began with a fresh output folder')
    output = attempt.output
    selection, receipt = output/'dry/event_selection.json', output/'resolved_config.json'
    if not (selection.exists() and receipt.exists()):
        return {}
    selected = {(r['take'], r['channel']): r for r in json.loads(selection.read_text())}
    digests = json.loads(receipt.read_text()).get('input_sha256', {})
    recovered = {}
    for i, take in enumerate(takes):
        stem = f'dry/take_{i:04d}_ch_00'
        files = [stem+'.wav', stem+'.json', stem+'_dataset_source.wav']
        if take['type'] != 'real' or not all((output/f).exists() for f in files):
            continue
        source = (attempt.session/take['file']).resolve()
        entry = selected.get((i, 0))
        if entry is None or entry['event_index'] is None or Path(entry['file']).resolve() != source:
            raise ValueError(f"{stem}: dry output does not match this session's event selection")
        if not source.is_file() or digests.get(str(source)) != file_hash(source):
            raise ValueError(f"{stem}: dry output was not made from the current input bytes of {take['file']}")
        rate = json.loads((output/files[1]).read_text()).get('fs')
        if rate != fs:
            raise ValueError(f'{stem}: dry report sample rate {rate} differs from the session {fs} Hz')
        recovered[i] = dict(take=i, dataset_source=files[2], report=files[1])
    return recovered


def window_options(policy):
    if policy not in ('auto', 'fixed_1p5'):
        raise ValueError('unknown public RIR window policy')
    return dict(adaptive_rir_window=None if policy == 'auto' else False)


def apply_event_hints(assembled, hints):
    """Supply explicit real-take annotations only through the public manifest."""
    if not hints:
        return
    import yaml
    path = assembled.folder/'manifest.yaml'
    manifest = yaml.safe_load(path.read_text())
    takes = {t['file']: t for t in manifest['takes'] if t['type']=='real'}
    if not set(hints) <= takes.keys():
        raise ValueError('event hints must name delivered real takes')
    for name, hint in hints.items():
        if not hint or not set(hint) <= {'event_index', 'event_window_s'}:
            raise ValueError('only public event_index/event_window_s hints are allowed')
        takes[name].update(hint)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))


@contextmanager
def repeat_alignment_scope(method):
    """Serial benchmark selector for the existing RIR API, restored on failure.

    PipelineConfig does not expose this experimental keyword. No DSP is copied;
    the default arm does not bind or replace anything. Not for concurrent runs.
    """
    if method not in ('legacy', 'packet-medoid/1'):
        raise ValueError('unknown benchmark repeat alignment')
    if method == 'legacy':
        yield
        return
    from sensorsim.acoustics.room import rir
    original = rir.deconvolve_repetitions
    try:
        rir.deconvolve_repetitions = partial(original, repeat_alignment=method)
        yield
    finally:
        rir.deconvolve_repetitions = original


def render_prediction(model, scenario, pair, fs):
    """Shared deterministic evaluation renderer, also used for fixed-model replay."""
    if model is None:
        return None
    s, m = pair
    rendered = synthesize(model, scenario.all_positions[s], scenario.all_positions[m], fs,
        seed=stable_seed(scenario.seed, f"reconstruction:{s}:{m}"), config=synthesis_config(scenario))
    h = np.asarray(rendered.samples[rendered.metadata["latency_samples"]:], dtype=np.float64)
    n = round(scenario.render.rir_duration_s*fs)
    return np.pad(h[:n], (0, max(0, n-len(h))))


def run_scenario(scenario, engine, work: Path, result_dir: Path, *, dry_mode="legacy",
                 window_policy="auto", event_hints=None, assembled=None,
                 inverse_mode="legacy", no_ref_target="effective_source",
                 relative_level_sidecar=False, repeat_alignment="legacy"):
    """Return every planned row even when the public pipeline rejects acquisition."""
    from sensorsim.acoustics.room.dry import DryConfig
    from sensorsim.acoustics.room.obstacles import obstacle_scene
    from sensorsim.scene.raycore import RayCore
    started = time.perf_counter()
    work, result_dir = Path(work), Path(result_dir)
    work.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    assembled = assembled or assemble_session(scenario, engine, work)
    apply_event_hints(assembled, event_hints)
    fs = assembled.kit.fs_hz
    config = pipeline.PipelineConfig(count=1, val_fraction=0., seed=scenario.seed, case="B",
        rir_duration_s=scenario.render.rir_duration_s, synthesis=synthesis_config(scenario),
        peak_policy="scale", preview_seconds=.1,
        dry=DryConfig(dereverb_mode=dry_mode, inverse_mode=inverse_mode),
        no_ref_target=no_ref_target,
        **window_options(window_policy))
    output = work/"pipeline"
    attempt = begin_attempt(assembled.folder, output)
    report, model, error = None, None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # THE ONLY reconstruction entry point. The optional selector changes
            # the existing RIR option only; no truth reaches inference.
            with repeat_alignment_scope(repeat_alignment):
                report = pipeline.run_pipeline(assembled.folder, output, config)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}".replace(str(work), "WORK")
    model = load_public_model(output)
    rows, predictions, actual, diagnostics = [], {}, {}, []
    same_recorder, recorder_diagnostics, closure_rows = [], [], []
    # Measured adapters supply surveyed positions but no verified obstacle model.
    visibility = (RayCore(obstacle_scene(scenario.room_model()), backend="cpu")
                  if hasattr(scenario, 'room_model') else None)
    relative_levels = []
    reference_path = output/'level_references.json'
    references = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    behind_pairs = []
    note = "same-model inverse crime" if engine.tier == "T0" else engine.name
    if error:
        note += "; pipeline="+error
    for kind, pairs in (("heldout", scenario.heldout_pairs), ("novel", scenario.novel_pairs)):
        for pair in pairs:
            actual[pair] = engine.render_rirs(scenario, [pair], fs)[pair]
            predicted = render_prediction(model, scenario, pair, fs)
            predictions[pair] = predicted
            s, m = pair
            if visibility is not None and not visibility.visibility(np.array([scenario.all_positions[s]]), np.array([scenario.all_positions[m]]))[0]:
                behind_pairs.append(pair)
            split = "novel grid" if kind == "novel" else (
                "measured holdout" if not hasattr(scenario, 'eval_positions') else
                "joint holdout" if s == scenario.eval_positions.heldout_speaker and m == scenario.eval_positions.heldout_mic else
                "speaker holdout" if s == scenario.eval_positions.heldout_speaker else "mic holdout")
            rows.extend(compare_rirs(actual[pair], predicted, fs, context(scenario, engine, kind, s, m), notes=note+"; "+split))
            if relative_level_sidecar:
                from .relative_levels import compare_relative_levels
                candidates = [r for r in references.values() if r['source_id'] == s]
                reference = candidates[0] if len(candidates) == 1 else None
                relative_levels.extend(compare_relative_levels(actual[pair], predicted, fs,
                    context(scenario, engine, kind, s, m), reference=reference,
                    impairments=assembled.impairments))
    rows.extend(compare_parameters(engine.truth_parameters(scenario), model,
        context(scenario, engine, "parameter", "room", "room"), obstacles=[o.name for o in scenario.obstacles], notes=note))
    measurements = [] if report is None else report["measurements"]
    if not measurements and (output/"measurements.json").exists():
        measurements = json.loads((output/"measurements.json").read_text())
    measurement_by_take = {r["take"]: r for r in measurements}
    dry_by_take = (recover_dry_takes(attempt, assembled.manifest["takes"], fs) if report is None
                   else {r["take"]: r for r in report["dry"]})
    for i, take in enumerate(assembled.manifest["takes"]):
        s, m = take.get("speaker", ""), take["mic"]
        ctx = context(scenario, engine, "timing", s, m)
        if take["type"] == "sweep":
            record = measurement_by_take.get(i)
            detected = None if record is None else _legacy_timing(record["metadata"], fs, assembled.kit)
            detector_note = "detector=room.sweep_detect; ADC start/end samples; "
            rows.extend(compare_timing(assembled.timing[take["file"]], detected, ctx, notes=detector_note+(error or "")))
        if take["type"] != "real":
            continue
        true_source = assembled.truth.dry_sources[take["file"]]
        recovered = None
        if i in dry_by_take:
            recovered_fs, recovered = wavfile.read(output/dry_by_take[i]["dataset_source"])
            if recovered_fs != fs:
                raise ValueError("pipeline dry sample rate changed")
        oracle = conditional_wiener_reference(assembled, take)
        real_note = note+"; take="+take["file"]
        rows.extend(compare_dry(true_source, recovered, oracle, fs,
            context(scenario, engine, "real_sound", s, m), notes=real_note))
        # Case B with a real matching REF retains the recorder chain g M.
        # Reuse the approved adapter AFTER public inference, never as its input.
        has_ref = any(t['type']=='sweep' and t.get('speaker')==s and t['mic']==f'REF_{s}'
                      for t in assembled.manifest['takes'])
        if has_ref and take.get('calibration') is None:
            from tests.validation.bench_room_resim.chain import recorder_source, recorder_render
            acquired_h = assembled.truth.rirs[f'{s}:{m}']
            support = max(len(true_source)+len(acquired_h)-1, len(oracle),
                          0 if recovered is None else len(recovered))
            target = recorder_source(np.pad(true_source,(0,support-len(true_source))), fs, assembled.impairments)
            oracle_target = recorder_source(np.pad(oracle,(0,support-len(oracle))), fs, assembled.impairments)
            detail = real_note+'; reference=same_recorder; case=B; units=relative-same-recorder'
            ctx = context(scenario, engine, 'real_sound', s, m)
            same_recorder.extend(compare_dry(target, recovered, oracle_target, fs, ctx, notes=detail))
            recorder_diagnostics.extend(compare_sound(
                recorder_render(true_source, acquired_h, fs, assembled.impairments),
                None if recovered is None else signal.fftconvolve(recovered, acquired_h), fs, ctx,
                prefix='exact_rir_', notes=detail+'; claim=exact-RIR-rerender; acquired-position'))
        if m.startswith("REF"):
            continue
        for pair, h in actual.items():
            wet_truth = signal.fftconvolve(true_source, h)
            wet_prediction = None if recovered is None or predictions[pair] is None else signal.fftconvolve(recovered, predictions[pair])
            rows.extend(compare_sound(wet_truth, wet_prediction, fs,
                context(scenario, engine, "real_sound", *pair), notes=real_note))
            if has_ref and take.get('calibration') is None:
                ctx = context(scenario, engine, 'real_sound', *pair)
                rec_truth = recorder_render(true_source, h, fs, assembled.impairments)
                same_recorder.extend(compare_sound(rec_truth, wet_prediction, fs, ctx,
                    notes=detail+'; claim=sample-realisation-diagnostic; unmeasured-position'))
                model_only = None if predictions[pair] is None else recorder_render(
                    true_source, predictions[pair], fs, assembled.impairments)
                recorder_diagnostics.extend(compare_sound(rec_truth, model_only, fs, ctx,
                    prefix='model_only_', notes=detail+'; claim=sample-realisation-diagnostic'))
                recorder_diagnostics.extend(compare_sound(rec_truth,
                    recorder_render(oracle, h, fs, assembled.impairments), fs, ctx,
                    prefix='oracle_rerender_', notes=detail+'; timing=oracle; true-room'))
                for dtype, gate in ((np.float64,1e-10),(np.float32,1e-6)):
                    convolved = signal.oaconvolve(true_source.astype(dtype), h.astype(dtype))
                    closed = recorder_source(convolved, fs, assembled.impairments)
                    relative = float(np.linalg.norm(closed-rec_truth)/np.linalg.norm(rec_truth))
                    closure_rows.append(row(ctx, metric='closure_relative_l2_'+np.dtype(dtype).name,
                        truth=0., reconstructed=relative, gate=gate, unit='1',
                        notes=detail+'; no alignment or gain fitting; convolution component arithmetic'))
            if scenario.render.profile == "full" and engine.tier != 'T2':
                diagnostics.extend(compare_sound(wet_truth, signal.fftconvolve(oracle, h), fs,
                    context(scenario, engine, "real_sound", *pair), prefix="oracle_rerender_",
                    notes=real_note+"; variant=conditional-oracle; timing=oracle; true-room"))
                model_only = None if predictions[pair] is None else signal.fftconvolve(true_source, predictions[pair])
                diagnostics.extend(compare_sound(wet_truth, model_only, fs,
                    context(scenario, engine, "real_sound", *pair), prefix="model_only_",
                    notes=real_note+"; variant=true-source-fitted-room; timing=source-origin"))
    for r in rows:
        timing = "oracle" if r["metric"] == "oracle_si_sdr" else "detected"
        r["notes"] += f"; dry_mode={dry_mode}; timing={timing}"
        if r["metric"] in ("oracle_si_sdr", "oracle_gap_db"):
            r["notes"] += "; reference=conditional-oracle-timing+true-RIR+source-power"
        if r['eval_kind']=='real_sound' and r['metric'] in ('si_sdr','lsd','band_level_db'):
            r['notes'] += '; claim=sample-realisation-diagnostic; reference=room_only'
    for r in diagnostics:
        r["notes"] += f"; dry_mode={dry_mode}"
    if diagnostics:
        write_scorecard(result_dir/"diagnostics.jsonl.gz", diagnostics)
    reference_scorecards = []
    for name, records in (('same_recorder',same_recorder), ('same_recorder_diagnostics',recorder_diagnostics),
                          ('numerical_closure',closure_rows), ('relative_levels',relative_levels)):
        if records:
            for r in records:
                r['notes'] += f'; dry_mode={dry_mode}; window_policy={window_policy}'
            filename = name+'.jsonl.gz'
            write_scorecard(result_dir/filename, records)
            reference_scorecards.append(filename)
    summary = summarize(rows, profile=scenario.render.profile)
    write_scorecard(result_dir/"scorecard.jsonl.gz", rows)
    write_json(result_dir/"summary.json", summary)
    write_json(result_dir/"scenario.json", scenario.model_dump(mode="json"))
    outcome = public_outcome(report, output, len(dry_by_take))
    provenance = dict(version="room-closedloop/2", scenario=scenario.id, tier=engine.tier,
        engine=engine.name, engine_class=type(engine).__qualname__,
        engine_source_sha256=file_hash(inspect.getfile(type(engine))),
        protocol_sha256=optional_hash(Path(__file__).with_name("PROTOCOL.md")),
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment=dict(python=platform.python_version(), **versions("numpy", "scipy", "pydantic", "torch")),
        acquisition_layer=assembled.timing_layer, acquisition_protocol=assembled.manifest["acquisition"],
        dry_mode=dry_mode, timing_variant="detected", room_model_available=model is not None,
        window_policy=window_policy, repeat_alignment=repeat_alignment,
        event_variant='manifest_hints' if event_hints else 'auto',
        event_hints=event_hints or {}, reference_scorecards=reference_scorecards,
        behind_obstacle_pairs=behind_pairs,
        detector="room.sweep_detect",
        detector_completed_takes=sum("sweep_detection" in r["metadata"] for r in measurements),
        pipeline_config=config.model_dump(mode="json"), pipeline_error=error, **outcome,
        recovered_dry_takes=sorted(dry_by_take) if report is None else [],
        pipeline_source_sha256=file_hash(pipeline.__file__),
        engine_truth_parameters=engine.truth_parameters(scenario),
        fitted_model_sha256=file_hash(output/"room_model.json") if model is not None else None,
        train_pairs=scenario.train_pairs, heldout_pairs=scenario.heldout_pairs, novel_pairs=scenario.novel_pairs,
        material_evidence=scenario.material_evidence() if hasattr(scenario, 'material_evidence') else None,
        timing_truth=assembled.timing,
        session_sha256={p.name:file_hash(p) for p in sorted(assembled.folder.iterdir())},
        warnings=sorted({str(w.message).replace(str(work), "WORK") for w in caught}),
        elapsed_s=time.perf_counter()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if engine.tier == 'T2':
        provenance.update(input_sha256=dict(engine.input_hashes),
            origin_transforms={k:v for k,v in engine.origins.items() if k.startswith(scenario.id+':')},
            capture_snr_and_clipping=assembled.truth.takes)
    write_json(result_dir/"provenance.json", provenance)
    # Preserve fitted model and small diagnostic summaries as derived products.
    if model is not None:
        (result_dir/"room_model.json").write_text(model.to_json()+"\n")
        write_json(result_dir/"pipeline_diagnostics.json", dict(
            fit=report["fit"] if report is not None else model.provenance,
            dry=[{k:r[k] for k in ("take", "mic_id", "label", "untrustworthy_bands_hz", "warnings")}
                 for r in report["dry"]] if report is not None else [],
            pipeline_error=error, **outcome))
    return rows, provenance
