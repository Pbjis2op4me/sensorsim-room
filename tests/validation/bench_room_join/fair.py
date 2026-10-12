"""Proxy join diagnostic with coherent and exact independent-path references.

The historical paired-ISM gates are retained in expected.py and run.py. This
new diagnostic excludes the direct packet and never changes the renderer's
diffuse prior. It can use registered user dimensions/coordinates unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import signal

from sensorsim.acoustics.room import synth
from sensorsim.acoustics.room.late import expected_filtered_power
from sensorsim.acoustics.room.metrics import octave_sos
from sensorsim.acoustics.room.model import RoomModel
from .expected import decision
from .run import verify_registration


def independent_path_power(model, paths, fs, length, config, analyses):
    """Sum analysed squared path kernels, exactly E[(sum sign_i path_i)^2].

    Each independent sign has zero mean and unit variance. Cross terms vanish;
    filter each finite kernel then shift its squared response to its arrival.
    No sign-draw error, whitened-path assumption, or fixed ringing cutoff.
    """
    power = np.zeros((len(analyses), length))
    for _, indices, kernels in synth.path_kernels(model, paths, fs, config):
        starts = indices[:, 0]
        # path_kernels uses causal stored kernels (latency already included).
        if np.any(starts < 0):
            raise ValueError("positive propagation delays required")
        span = max(0, length-int(starts.min()))
        if not span:
            continue
        bank = np.pad(kernels[:, :span], ((0, 0), (0, max(0, span-kernels.shape[1]))))
        for b, sos in enumerate(analyses):
            filtered = signal.sosfilt(sos, bank, axis=1)
            for start, row in zip(starts, filtered):
                count = length-int(start)
                if count > 0:
                    power[b, start:] += row[:count]**2
    return power


def windows(fs, length, latency, mixing, direct, analyses):
    """Direct-packet-safe windows; an empty window stays unavailable."""
    t = (np.arange(length)-latency)/fs
    masks, packet_ends = [], []
    for sos in analyses:
        impulse = np.zeros(round(.2*fs)); impulse[0] = 1
        h2 = signal.sosfilt(sos, impulse)**2
        centroid = np.dot(np.arange(len(h2))/fs, h2)/h2.sum()
        end = np.searchsorted(np.cumsum(h2), .999*h2.sum())/fs
        packet_end = direct+latency/fs+end-centroid
        local = t-centroid
        bounds = [(max(mixing-.020, packet_end), mixing),
                  (mixing, mixing+.020),
                  (max(mixing-.006, packet_end), mixing+.006),
                  (mixing+.020, mixing+.060)]
        bins = [(mixing+j*.001, mixing+(j+2)*.001) for j in range(-6, 12, 2)]
        one = [(local >= lo) & (local < hi) for lo, hi in bounds]
        one += [(local >= lo) & (local < hi) & (lo >= packet_end) for lo, hi in bins]
        masks.append(one); packet_ends.append(packet_end)
    return masks, packet_ends


def geometry(model, source, receiver, spec):
    fs, bands = spec['fs'], spec['bands_hz']
    config = synth.SynthConfig()
    source, receiver = np.asarray(source), np.asarray(receiver)
    mixing, _ = synth.effective_mixing_time(model, source, receiver, config.crossfade_s)
    direct = np.linalg.norm(source-receiver)/model.sound_speed_m_s
    analyses = [octave_sos(fs, b) for b in bands]
    length = int(np.ceil((mixing+.15)*fs))+config.filter_half_length
    t = (np.arange(length)-config.filter_half_length)/fs
    paths, anchor, _, _ = synth.early_paths(model, source, receiver, config)
    early = synth._render_paths(model, paths, fs, length, config)
    early *= np.cos(np.pi/2*synth.crossfade_fraction(t, mixing, config.crossfade_s))
    late_bands = synth.late_bands(fs)
    levels = synth.late_band_levels(anchor, mixing, config.crossfade_s, late_bands,
                                    model, config.air_absorption)/fs
    decays = synth.late_band_t60(late_bands, model.decay_budget().t60_s, model, config.air_absorption)
    rendered = np.array([signal.sosfilt(a, early)**2 for a in analyses])
    for band, level, decay in zip(late_bands, levels, decays):
        if level:
            envelope = synth.late_envelope(t, mixing, config.crossfade_s, decay)
            for j, a in enumerate(analyses):
                rendered[j] += level*expected_filtered_power(band.sos, a, envelope)
    ref_config = synth.SynthConfig(late=False, early_time_s=mixing+.080)
    ref_paths = synth.early_paths(model, source, receiver, ref_config)[0]
    ref = synth._render_paths(model, ref_paths, fs, length, ref_config)
    coherent = np.array([signal.sosfilt(a, ref)**2 for a in analyses])
    incoherent = independent_path_power(model, ref_paths, fs, length, ref_config, analyses)
    masks, packet_ends = windows(fs, length, config.filter_half_length, mixing, direct, analyses)
    energies = {}
    for name, power in [('renderer', rendered), ('coherent', coherent), ('incoherent', incoherent)]:
        energies[name] = [[float(p[m].sum()) if m.any() else None for m in mm]
                          for p, mm in zip(power, masks)]
    return dict(source_m=source.tolist(), receiver_m=receiver.tolist(), mixing_s=mixing,
                direct_s=float(direct), direct_packet_end_s=packet_ends,
                window_samples=[[int(m.sum()) for m in mm] for mm in masks], energy=energies)


def _ratio_summary(numerator, denominator, bootstrap):
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    # Same geometry rows in numerator and reference, with missingness explicit.
    n, d = np.where(valid, numerator, 0), np.where(valid, denominator, 0)
    def ratio(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            return 10*np.log10(a/b)
    point = ratio(n.sum(0), d.sum(0))
    boot = ratio(n[bootstrap].sum(1), d[bootstrap].sum(1))
    ci = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return dict(point_db=point.tolist(), ci95_db=ci.tolist(), coverage=valid.sum(0).tolist())


def _step_summary(energy, reference, bootstrap):
    valid = np.isfinite(energy).all(-1) & np.isfinite(reference).all(-1)
    e, r = np.where(valid[...,None], energy, 0), np.where(valid[...,None], reference, 0)
    def step(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            return 10*np.log10(a[...,1]/a[...,0])-10*np.log10(b[...,1]/b[...,0])
    point = step(e.sum(0), r.sum(0))
    ci = np.nanpercentile(step(e[bootstrap].sum(1), r[bootstrap].sum(1)), [2.5,97.5], axis=0)
    return dict(point_db=point.tolist(), ci95_db=ci.tolist(), coverage=valid.sum(0).tolist())


def summarize(rows, model, spec):
    energy = {name: np.asarray([r['energy'][name] for r in rows], float)
              for name in ('renderer', 'coherent', 'incoherent')}
    rng = np.random.default_rng(81089901)
    bootstrap = rng.integers(len(rows), size=(2000, len(rows)))
    result = {}
    for reference in ('coherent', 'incoherent'):
        e, r = energy['renderer'], energy[reference]
        metrics = {}
        for name, index in [('centred', 2), ('late', 3), ('before', 0), ('after', 1), ('bins_2ms', slice(4,None))]:
            metrics[name] = _ratio_summary(e[..., index], r[..., index], bootstrap)
        # Before and after windows can have different durations; paired ratios
        # cancel those durations exactly. Keep the joint bootstrap pairing.
        metrics['step_excess'] = _step_summary(e[...,:2], r[...,:2], bootstrap)
        select = [spec['bands_hz'].index(b) for b in spec['gating_bands_hz']]
        centred = metrics['centred']
        metrics['decision'] = decision(np.array(centred['point_db'])[select],
                                       np.array(centred['ci95_db'])[:, select])
        # Taking the minimum after pooling avoids selecting a noisy bin per geometry.
        points = np.asarray(metrics['bins_2ms']['point_db'])
        metrics['deepest_2ms_db'] = [float(np.nanmin(p)) if np.isfinite(p).any() else None for p in points]
        result[reference] = metrics
    wh = _ratio_summary(energy['coherent'][...,3], energy['incoherent'][...,3], bootstrap)
    dims = np.asarray(model.shoebox_dims_m)
    volume, surface = np.prod(dims), 2*np.sum(dims*np.roll(dims, 1))
    target = -10*np.log10(1+surface*model.sound_speed_m_s/(8*volume*np.array(spec['bands_hz'])))
    wh['candidate_target_db'] = target.tolist()
    wh['residual_db'] = (np.array(wh['point_db'])-target).tolist()
    wh['residual_ci95_db'] = (np.array(wh['ci95_db'])-target).tolist()
    result['waterhouse_not_adopted'] = wh
    result['schroeder_hz'] = (2000*np.sqrt(model.decay_budget().t60_s/volume)).tolist()
    return result


def run(spec):
    families = []; rng = np.random.default_rng(spec['geometry_seed'])
    for family in spec['families']:
        model = RoomModel(shoebox_dims_m=family['dimensions_m'])
        pairs = [(family['source_m'], family['receiver_m'])]
        while len(pairs) <= spec['random_pairs_per_family']:
            s, r = rng.uniform(.5, np.array(family['dimensions_m'])-.5, (2,3))
            if np.linalg.norm(s-r) >= 1:
                pairs.append((s,r))
        rows = []
        for i, (source, receiver) in enumerate(pairs):
            rows.append(geometry(model, source, receiver, spec))
            if i % 10 == 0:
                print(family['id'], i, flush=True)
        families.append(dict(id=family['id'], summary=summarize(rows, model, spec), geometries=rows))
    return dict(label='proxy result; model-based exact expected energy', families=families,
                history='The w8b and w8c registered paired-ISM FAILs remain unchanged.')


def _finite_json(value):
    if isinstance(value, dict): return {k:_finite_json(v) for k,v in value.items()}
    if isinstance(value, list): return [_finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--hash-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('preserve previous evidence')
    digest, spec = verify_registration(args.preregistration, args.hash_file)
    start = time.perf_counter(); result = run(spec['join'])
    result.update(preregistration_sha256=digest, elapsed_s=time.perf_counter()-start)
    args.out.write_text(json.dumps(_finite_json(result), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
