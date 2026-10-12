"""Proxy regressions for WP2 review F2-F10; no measured audio."""
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from sensorsim import edge
from sensorsim.evaluation import fa_rate


class ScoreModel:
    classes = ['target']
    features = SimpleNamespace(n=4)
    meta = {'features': {'fs': 4}, 'detector': {
        'hop_s': .25, 'on': {'target': .5}, 'off_ratio': .6,
        'smooth_hops': 1, 'min_hops': 2}}

    def scores(self, windows):
        return np.atleast_2d(windows)[:, :1]


def row(i, count, *, family='impact', context='isolated', parent=None):
    return dict(family=family, context_group=context, **{'class': 'target'},
                context_id=f'{family}-{context}-{i}', parent_id=parent or f'{family}-{context}-{i}',
                confirmed_outputs=count, max_confirmed_outputs=8, source_instances=3)


def write_wav(path, samples, fs=4):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(fs)
        wav.writeframes((np.asarray(samples) * 30000).astype('<i2').tobytes())


@pytest.mark.parametrize('smooth,pattern', [(1, [.875, 0, 0, 0]), (3, [.875, 0, .75, 0])])
def test_f2_start_alerts_survive_unconfirmed_ends(tmp_path, smooth, pattern):
    model = ScoreModel()
    model.meta = {**model.meta, 'detector': {**model.meta['detector'], 'smooth_hops': smooth}}
    samples = np.zeros(48, np.float32)
    samples[8:12] = pattern
    # Exactly representable PCM scores, without clipping at signed int16 max.
    samples *= 32768 / 30000
    path = tmp_path / 'blip.wav'
    write_wav(path, samples)
    rows = fa_rate.score_challenge_records(model, [dict(path=path, family='impact')])
    assert rows[0]['confirmed_outputs'] == 0
    assert rows[0]['start_outputs'] == 1
    assert rows[0]['max_start_outputs'] >= 1
    old = fa_rate.score_wav(model, path)[0]
    assert old == edge.detect(model, edge.read_wav(str(path), 4)) == []


def test_f3_sparse_floor_reports_bound_and_resolution_exposure():
    floor = fa_rate.summarize_floor([], 1200)
    assert floor['floor_resolved'] is False
    assert floor['zero_event_hours_for_budget_one_sided'] == pytest.approx(2.995732273554)
    rows = [row(i, i % 2) for i in range(100)]
    report = fa_rate.estimate_composite(floor, rows, {'impact': 2},
                                      context_weights={'isolated': 1}, bootstrap_replicates=100)
    c = report['classes']['target']
    assert not c['floor_resolved']
    assert c['upper_bound_per_hour'] == c['interval_95_per_hour'][1] > c['rate_per_hour']
    assert report['zero_event_floor_hours_for_budget_composite'] == pytest.approx(-np.log(.025 / 2))
    assert fa_rate.summarize_floor([], 3 * 3600)['floor_resolved']


@pytest.mark.parametrize('count', [2, 8])
def test_f5_constant_nonzero_context_counts_have_uncertainty(count):
    result = fa_rate.summarize_challenges([row(i, count) for i in range(100)],
        context_weights={'isolated': 1}, bootstrap_replicates=100)['families']['impact']['target']
    lo, hi = result['interval_95']
    assert lo < hi
    assert lo <= count <= hi
    assert result['status'] == 'scored'


@pytest.mark.parametrize('kind', ['annotations', 'two_parents', 'dependent'])
def test_f6_sufficiency_counts_independent_contexts(kind):
    if kind == 'annotations':
        rows = [row(i, i % 3) for i in range(34)]
        weights = {'isolated': 1}
    elif kind == 'two_parents':
        rows = [row(i, i % 2) for i in range(98)] + [row(i, i, context='overlap') for i in range(2)]
        weights = {'isolated': .7, 'overlap': .3}
    else:
        rows = [row(i, i % 2, parent=str(i // 50)) for i in range(100)]
        weights = {'isolated': 1}
    result = fa_rate.summarize_challenges(rows, context_weights=weights,
        bootstrap_replicates=100)['families']['impact']['target']
    assert result['status'].startswith('inconclusive')


def test_f7_each_family_has_its_own_strata_and_weights():
    rows = [row(i, i % 2, context='quiet') for i in range(100)]
    rows += [row(i, 2 + i % 2, context='loud') for i in range(100)]
    rows += [row(i, i % 2, family='flow', context='steady') for i in range(100)]
    weights = {'impact': {'quiet': .2, 'loud': .8}, 'flow': {'steady': 1}}
    report = fa_rate.estimate_composite(fa_rate.summarize_floor([], 3600), rows,
        {'impact': 2, 'flow': 4}, context_weights=weights, bootstrap_replicates=100)
    assert report['classes']['target']['rate_per_hour'] == pytest.approx(6.2)
    assert report['context_weights'] == weights
    assert report['zero_event_floor_hours_for_budget_composite'] == pytest.approx(-np.log(.025 / 4))


def test_f8_f9_cli_scores_wavs_pools_floor_and_reports_both_units(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fa_rate.edge, 'Model', lambda _: ScoreModel())
    # Only the CNN is a score proxy: CLI, PCM decode, Detector, guards, caps,
    # floor pooling, estimation and JSON serialization all run without mocks.
    for name, n, pulses in [('f1.wav', 48, [(8, 1)]), ('f2.wav', 96, [(8, 2)]),
                             ('challenge.wav', 48, [(8, 1), (20, 2)])]:
        samples = np.zeros(n)
        for i, length in pulses:
            samples[i:i + length] = .9
        samples += np.random.default_rng(n+len(pulses)).uniform(-.002, .002, n)
        write_wav(tmp_path / name, samples)
    manifest = dict(source_kind='proxy', calibration_streams=[], floor_streams=[{'path': 'f1.wav'}, {'path': 'f2.wav'}],
                    challenges=[dict(path='challenge.wav', family='impact', context='isolated')],
                    context_weights={'impact': {'isolated': 1}}, context_prevalence_per_hour={'impact': 2},
                    bootstrap_replicates=100)
    for entry in [*manifest['floor_streams'], *manifest['challenges']]:
        with wave.open(str(tmp_path / entry['path']), 'rb') as wav:
            entry.update(recording_id=entry['path'], sample_range=[0, wav.getnframes()], sample_rate_hz=4)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    model_path = tmp_path / 'proxy.npz'
    model_path.write_bytes(b'generated score proxy')
    assert fa_rate.main(['--model', str(model_path), '--manifest', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['schema'] == 'sensorsim-room-fa-rate/3'
    assert result['deployment_alert_unit'] == 'start_alerts'
    assert [f['events'] for f in result['floor_streams']] == [0, 1]
    pooled = result['pooled_floor_streams'][0]
    assert pooled['events'] == 1 and pooled['start_alerts']['events'] == 2
    assert pooled['exposure_s'] == 36 and pooled['longest_continuous_s'] == 24
    confirmed, starts = result['composite_estimates']
    assert confirmed['event_unit'] == 'confirmed_events' and starts['event_unit'] == 'start_alerts'
    assert confirmed['classes']['target']['rate_per_hour'] == pytest.approx(102)
    assert starts['classes']['target']['rate_per_hour'] == pytest.approx(204)
    assert starts['classes']['target']['upper_bound_per_hour'] > 204
    assert all(r['evidence_label'] == 'model-based' for r in result['composite_estimates'])
    # A truncated context still fails through the CLI.
    write_wav(tmp_path / 'challenge.wav', np.ones(48) * .9)
    with pytest.raises(SystemExit):
        fa_rate.main(['--model', str(model_path), '--manifest', str(path)])


def test_f10_cp_is_shared_and_rejects_fractional_counts():
    from sensorsim.evaluation.intervals import clopper_pearson
    assert fa_rate.clopper_pearson is clopper_pearson
    assert clopper_pearson(0, 10)[0] == 0
    assert clopper_pearson(10, 10)[1] == 1
    with pytest.raises(ValueError):
        clopper_pearson(.5, 10)


def test_f2_api_names_start_counts_and_never_infers_missing_starts():
    floor = fa_rate.summarize_floor([], 1200)
    rows = [row(i, i % 2) for i in range(100)]
    kwargs = dict(context_weights={'isolated': 1}, bootstrap_replicates=100, event_unit='start_alerts')
    with pytest.raises(ValueError, match='rescore'):
        fa_rate.estimate_composite(floor, rows, {'impact': 2}, **kwargs)
    floor['start_alerts'] = fa_rate.summarize_floor([1], 1200)
    for r in rows:
        r.update(start_outputs=1 + r['confirmed_outputs'], max_start_outputs=12)
    result = fa_rate.estimate_composite(floor, rows, {'impact': 2}, **kwargs)
    assert result['classes']['target']['rate_per_hour'] == 6
    group = result['classes']['target']['families']['impact']['by_context']['isolated']
    assert group['start_outputs'] == 150 and 'confirmed_outputs' not in group


def test_f8_pool_keeps_partial_file_blocks_and_measured_continuity():
    files = [fa_rate.summarize_floor([3], 15, source_kind='measured'),
             fa_rate.summarize_floor([], 25, source_kind='measured')]
    pooled = fa_rate.pool_floor_summaries(files)
    assert pooled['events'] == 1 and pooled['exposure_s'] == 40
    assert pooled['longest_continuous_s'] == 25
    assert pooled['evidence_label'] == 'measured-smoke-test-not-FA/hour'
    assert 'block_counts' not in pooled
    assert [len(f['block_counts']) for f in pooled['pooled_files']] == [2, 3]
    bounds = fa_rate._floor_interval(pooled, .95, 100, 4)
    assert bounds[1] >= fa_rate.poisson_rate_interval(1, 40)[1]


def test_f9_cli_loads_generated_npz_and_scores_pcm_with_real_features(tmp_path):
    # The complete deployed Model/Features path, using generated constant-low
    # weights. The other CLI test exercises nonzero and unconfirmed outputs.
    meta = dict(format=edge.FORMAT, classes=['target'],
                layers=[{'type': 'pool'}, {'type': 'fc', 'name': 'fc'}],
                features=dict(fs=64, window_s=.5, n_fft=16, hop=4,
                              n_mels=4, fmin_hz=2, fmax_hz=30),
                detector=dict(hop_s=.25, on={'target': .5}, off_ratio=.6, smooth_hops=3, min_hops=2))
    model_path = tmp_path / 'model.npz'
    np.savez(model_path, **{'meta': np.frombuffer(json.dumps(meta).encode(), np.uint8),
                           'fc.w': np.zeros((1, 2), np.float32), 'fc.b': np.array([-5], np.float32)})
    # Manifest provenance uses samples at the model rate, without implicit resampling.
    samples = np.sin(np.arange(768)) * .1
    write_wav(tmp_path / 'floor.wav', samples, fs=64)
    write_wav(tmp_path / 'context.wav', np.cos(np.arange(768))*.1, fs=64)
    manifest = dict(source_kind='proxy', calibration_streams=[], floor_streams=[{'path': 'floor.wav'}],
                    challenges=[dict(path='context.wav', family='impact')],
                    context_prevalence_per_hour={'impact': 2}, context_weights={'isolated': 1},
                    bootstrap_replicates=100)
    for entry in [*manifest['floor_streams'], *manifest['challenges']]:
        entry.update(recording_id=entry['path'], sample_range=[0, 768], sample_rate_hz=64)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / 'result.json'
    assert fa_rate.main(['--model', str(model_path), '--manifest', str(manifest_path), '--out', str(out)]) == 0
    report = json.loads(out.read_text())
    assert report['schema'] == 'sensorsim-room-fa-rate/3'
    assert len(report['composite_estimates']) == 2
    for estimate in report['composite_estimates']:
        c = estimate['classes']['target']
        assert c['rate_per_hour'] == 0 and c['upper_bound_per_hour'] > 0
        assert c['status'].startswith('inconclusive')
