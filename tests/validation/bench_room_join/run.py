"""Preregistered model-based join checks; no recordings or measured-room claims.

The short-window statistic includes stochastic fluctuation and analysis-filter
startup. Always retain the no-join control; its rejection does not alter the
registered threshold or turn a renderer failure into a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import signal
import yaml

from sensorsim.acoustics.absorption import amplitude_factor
from sensorsim.acoustics.room.metrics import octave_sos
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from sensorsim.edge import Features


def energy_step(samples, time, mixing, t60, window_s=.01):
    """Return raw and trend-corrected dB steps from adjacent equal windows.

    Integrate the reference exponential on the identical samples: do not just
    subtract a nominal window-centre slope on a rounded sample grid.
    """
    samples, time = np.asarray(samples, float), np.asarray(time, float)
    if (samples.ndim != 1 or samples.shape != time.shape or not len(samples)
            or not np.isfinite(samples).all() or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0) or not np.isfinite(mixing)
            or not np.isfinite(t60) or t60 <= 0 or not np.isfinite(window_s) or window_s <= 0):
        raise ValueError("finite aligned samples, increasing times and positive durations required")
    before = (time >= mixing-window_s) & (time < mixing)
    after = (time >= mixing) & (time < mixing+window_s)
    if not before.any() or not after.any():
        raise ValueError("both energy windows must contain samples")
    power_before, power_after = np.mean(samples[before]**2), np.mean(samples[after]**2)
    if min(power_before, power_after) <= 0:
        raise ValueError("step is undefined for a zero-energy window")
    trend = np.exp(-6*np.log(10)*(time[before | after]-mixing)/t60)
    raw = 10*np.log10(power_after/power_before)
    correction = 10*np.log10(trend[time[before | after] >= mixing].mean()
                             / trend[time[before | after] < mixing].mean())
    return float(raw), float(raw-correction)


def precursor(samples, fs, arrival_sample, guard_s=.002):
    """Composite early energy outside a declared main-lobe guard, not basis energy."""
    x = np.asarray(samples, float)
    if (x.ndim != 1 or not len(x) or not np.isfinite(x).all()
            or not np.isfinite(fs) or fs <= 0 or not np.isfinite(arrival_sample)
            or not np.isfinite(guard_s) or guard_s < 0 or not np.any(x)):
        raise ValueError("finite nonzero mono signal and a valid sample grid required")
    mask = np.arange(len(x)) < arrival_sample-guard_s*fs
    peak = np.max(np.abs(x[mask]), initial=0.) / np.max(np.abs(x))
    return {"energy_fraction": float(np.sum(x[mask]**2)/np.sum(x*x)),
            "peak_db": float(20*np.log10(peak)) if peak else None,
            "guard_s": guard_s}, mask


def join_trials(model, source, receiver, fs, seeds, bands, render=synthesize):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    rows, raw, zero_phase = [], [], []
    for seed in seeds:
        rir = render(model, source, receiver, fs, seed=int(seed))
        time = (np.arange(len(rir.samples))-rir.metadata['latency_samples'])/fs
        one, uncorrected, zp = [], [], []
        for b in bands:
            index = (125, 250, 500, 1000, 2000, 4000, 8000).index(b)
            args = time, rir.metadata['mixing_time_s'], rir.metadata['t60_target_s'][index]
            sos = octave_sos(fs, b)
            r, c = energy_step(signal.sosfilt(sos, rir.samples), *args)
            uncorrected.append(r); one.append(c)
            zp.append(energy_step(signal.sosfiltfilt(sos, rir.samples), *args)[1])
        raw.append(uncorrected); rows.append(one); zero_phase.append(zp)
    return {"label": "proxy result; model-based", "seeds": list(seeds), "bands_hz": bands,
            "raw_db": raw, "corrected_db": rows, "zero_phase_corrected_db": zero_phase,
            "median_signed_db": np.median(rows, axis=0).tolist(),
            "median_absolute_db": np.median(np.abs(rows), axis=0).tolist()}


def no_join_control(fs, seeds, bands, t60=.5):
    """Continuous stationary octave noise times an exponential, with no handoff."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    time = np.arange(round(.6*fs))/fs-.55
    # Bound the exponential outside the witness interval; only the final 100 ms
    # is scored, after 0.5 s of causal filter warmup.
    envelope = np.exp(-3*np.log(10)*np.maximum(time, -.05)/t60)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows.append([energy_step(signal.sosfilt(octave_sos(fs, b), rng.standard_normal(len(time)))
                                 * envelope, time, 0., t60)[1] for b in bands])
    return {"label": "model-based no-join control", "seeds": list(seeds), "bands_hz": bands,
            "corrected_db": rows, "median_signed_db": np.median(rows, axis=0).tolist(),
            "median_absolute_db": np.median(np.abs(rows), axis=0).tolist(),
            "positive_control_median_signed_db": (np.median(rows, axis=0)+6).tolist()}


def precursor_trials(spec, render=synthesize):
    model = RoomModel(shoebox_dims_m=spec['room_dimensions_m'])
    source, receiver = np.array(spec['source_m']), np.array(spec['receiver_m'])
    direction = (receiver-source)/np.linalg.norm(receiver-source)
    feature = Features(dict(fs=16000, window_s=1., n_fft=1024, hop=256,
                            n_mels=40, fmin_hz=50., fmax_hz=8000.))
    rows = []
    for fs in spec['fs']:
        integer = np.floor(np.linalg.norm(receiver-source)/model.sound_speed_m_s*fs)
        for fraction in spec['fractional_samples']:
            moved = source+direction*(integer+fraction)/fs*model.sound_speed_m_s
            for air in (False, True):
                rir = render(model, source, moved, fs, config=SynthConfig(late=False, max_order=0,
                                                                         air_absorption=air))
                arrival = rir.metadata['direct_delay_s']*fs+rir.metadata['latency_samples']
                metrics, mask = precursor(rir.samples, fs, arrival)
                trimmed = rir.samples.copy(); trimmed[mask] = 0
                difference = []
                for kind in ('impulse', 'chime'):
                    dry = np.zeros(fs)
                    if kind == 'impulse':
                        dry[round(.1*fs)] = 1
                    else:
                        t = np.arange(round(.4*fs))/fs
                        dry[round(.1*fs):round(.1*fs)+len(t)] = sum(np.sin(2*np.pi*f*t) for f in (880,1175,1568))*np.exp(-t/.4)
                    features = []
                    for h in (rir.samples, trimmed):
                        y = signal.fftconvolve(dry, h)[:fs]
                        y = signal.resample_poly(y, 16000, fs)
                        features.append(feature(y))
                    d = features[0]-features[1]
                    difference.append({"source": kind, "rms_log_mel_delta": float(np.sqrt(np.mean(d*d))),
                                       "max_log_mel_delta": float(np.max(np.abs(d)))})
                frequencies = np.array([b for b in (125,250,500,1000,2000,4000,8000) if b < fs/2])
                distance = np.linalg.norm(moved-source)
                magnitude = (amplitude_factor(frequencies, distance, model.temperature_c, model.humidity_pct)
                             if air else np.ones(len(frequencies)))/distance
                response = np.exp(-2j*np.pi*frequencies[:,None]*np.arange(len(rir.samples))/fs) @ rir.samples
                ratio = response/(magnitude*np.exp(-2j*np.pi*frequencies*arrival/fs))
                metrics.update(max_magnitude_error_db=float(np.max(abs(20*np.log10(abs(ratio))))),
                               max_phase_delay_error_samples=float(np.max(abs(np.angle(ratio)/(2*np.pi*frequencies/fs)))))
                metrics.update(fs=fs, fractional_sample=fraction, air=air, features=difference,
                               peak_arrival_error_samples=float(np.argmax(abs(rir.samples))-arrival))
                rows.append(metrics)
    return {"label": "model-based composite full-band direct render", "cases": rows}


def verify_registration(path, hash_path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != hash_path.read_text().split()[0]:
        raise ValueError("preregistration hash mismatch; refusing acceptance run")
    return digest, yaml.safe_load(path.read_text())


def registered_geometry(spec):
    """Read geometry from the registration, including the original prose schema."""
    if 'geometry' in spec:
        return spec['geometry']
    text = spec.get('splits', {}).get('acceptance', '')
    match = re.fullmatch(r"procedural room dimensions (\[[^]]+\]), source (\[[^]]+\]), receiver (\[[^]]+\])", text)
    if match is None:
        raise ValueError("preregistration must declare dimensions, source and receiver geometry")
    return dict(zip(('dimensions_m', 'source_m', 'receiver_m'),
                    (yaml.safe_load(x) for x in match.groups())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--hash-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--controls-preregistration', type=Path)
    parser.add_argument('--controls-hash-file', type=Path)
    args = parser.parse_args()
    digest, spec = verify_registration(args.preregistration, args.hash_file)
    controls = None
    if bool(args.controls_preregistration) != bool(args.controls_hash_file):
        parser.error('controls preregistration and hash file must be supplied together')
    if args.controls_preregistration:
        controls_digest, controls = verify_registration(args.controls_preregistration, args.controls_hash_file)
        if controls['parent_sha256'] != digest:
            raise ValueError('controls do not reference this main preregistration')
    if args.out.exists():
        raise FileExistsError("preserve existing acceptance evidence")
    join = spec['join']
    # Geometry is parsed from a structured preregistration or supplied through the
    # callable API; never read measured RIRs or a locked audit implicitly.
    geometry = registered_geometry(spec)
    model = RoomModel(shoebox_dims_m=geometry['dimensions_m'])
    seeds = range(join['seeds']['start'], join['seeds']['start']+join['seeds']['count'])
    result = join_trials(model, geometry['source_m'], geometry['receiver_m'], join['fs'], seeds, join['bands_hz'])
    result['acceptance'] = 'pass' if max(result['median_absolute_db']) <= join['threshold_db'] else 'fail'
    pre = precursor_trials(spec['precursor'])
    pre['energy_peak_acceptance'] = 'pass' if all(
        c['energy_fraction'] <= spec['precursor']['threshold_energy_fraction'] and
        (c['peak_db'] is None or c['peak_db'] <= spec['precursor']['threshold_peak_db']) for c in pre['cases']) else 'fail'
    pre['full_band_acceptance'] = 'pass' if all(c['max_magnitude_error_db'] <= .1 and
        c['max_phase_delay_error_samples'] <= .02 for c in pre['cases']) else 'fail'
    output = dict(preregistration_sha256=digest, join=result, precursor=pre)
    if controls:
        seeds = range(controls['seeds']['start'], controls['seeds']['start']+controls['seeds']['count'])
        control = no_join_control(controls['fs'], seeds, controls['bands_hz'])
        control['criterion_outcome'] = 'pass' if max(control['median_absolute_db']) <= join['threshold_db'] else 'fail'
        output.update(controls_preregistration_sha256=controls_digest, no_join_control=control)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(join=result['acceptance'], median_abs_step_db=result['median_absolute_db'],
                          precursor_energy_peak=pre['energy_peak_acceptance'])))


if __name__ == '__main__':
    main()
