"""Deterministic development score controls; no deployment-rate claim."""
from types import SimpleNamespace
import numpy as np
import pytest
from sensorsim import edge


def fixture(hours=1, bursts=1, peak=.8):
    classes = ['knock', 'doorbell', 'kettle']
    model = SimpleNamespace(classes=classes, features=SimpleNamespace(n=1),
        meta={'features': {'fs': 1}, 'detector': dict(hop_s=1., on=dict.fromkeys(classes, .5),
              off_ratio=.6, smooth_hops=1, min_hops=2)})
    scores = np.zeros((round(hours*3600), 3))
    for b in range(bursts):
        for i in range(3):
            scores[100+100*b+10*i:104+100*b+10*i, i] = peak
    model.scores = lambda windows: scores[windows[:, 0].astype(int)]
    return model, np.arange(len(scores), dtype=np.float32), scores


def test_combined_budget_and_per_class_are_distinct():
    model, audio, scores = fixture()
    combined = edge.calibrate(model, audio)
    per_class = edge.calibrate(model, audio, budget_mode='per_class')
    assert len(edge.track(model, scores, 1, {'on': combined['on']})) <= 1
    assert len(edge.track(model, scores, 1, {'on': per_class['on']})) == 3
    assert combined['budget_mode'] == 'combined'
    assert combined['status'] == 'insufficient_evidence'
    assert combined['exposure_s'] == 3600
    assert combined['minimum_exposure_hours'] == 3


@pytest.mark.parametrize('bursts,expected', [(1, 3), (2, 6)])
def test_saturated_infeasibility_is_explicit(bursts, expected):
    model, audio, scores = fixture(bursts=bursts, peak=1.)
    result = edge.calibrate(model, audio)
    assert result['status'] == 'infeasible'
    assert result['best_achievable_rate_per_hour'] == expected
    assert result['observed']['start_alerts']['total'] == expected
    assert result['observed']['confirmed_events']['total'] == expected
    assert len(edge.track(model, scores, 1, {'on': result['on']})) == expected


def test_feasible_zero_control_and_exposure_boundary():
    model, audio, _ = fixture(hours=3, bursts=0)
    result = edge.calibrate(model, audio)
    assert result['status'] == 'feasible'
    assert result['on'] == model.meta['detector']['on']
    assert result['zero_event_poisson_upper_per_hour'] == pytest.approx(-np.log(.05)/3)
    assert result['deployment_validated'] is False


def test_start_alerts_are_budgeted_even_when_unconfirmed():
    model, audio, scores = fixture(peak=1.)
    model.meta['detector']['min_hops'] = 5
    result = edge.calibrate(model, audio)
    assert result['status'] == 'infeasible'
    assert result['observed']['confirmed_events']['total'] == 0
    assert result['observed']['start_alerts']['total'] == 3
    assert edge.calibrate(model, audio, event_unit='confirmed_events')['empirical_feasible']


@pytest.mark.parametrize('audio', [np.array([]), np.zeros((2, 2)), np.array([np.nan])])
def test_invalid_audio_refused(audio):
    model, _, _ = fixture()
    with pytest.raises(ValueError):
        edge.calibrate(model, audio)


@pytest.mark.parametrize('budget', [-1, np.nan, np.inf])
def test_invalid_budget_refused(budget):
    model, audio, _ = fixture()
    with pytest.raises(ValueError):
        edge.calibrate(model, audio, fa_per_hour=budget)


def test_grid_counts_match_scalar_tracker_including_splitting():
    model, _, _ = fixture()
    # Frozen development draws, not a stochastic statistical acceptance gate.
    rng = np.random.default_rng(230)
    scores = rng.choice([0., .29, .31, .4, .7, 1.], 300)
    grid = np.array([.5, .7, .8, .995])
    for smooth, minimum in [(1, 1), (1, 3), (3, 2)]:
        actual = edge._calibration_grid_counts(scores, grid, off_ratio=.6,
            smooth_hops=smooth, min_hops=minimum)
        for j, t in enumerate(grid):
            tracker = edge.Tracker(['knock'], dict(on=t, off_ratio=.6,
                smooth_hops=smooth, min_hops=minimum), 1.)
            records = []
            for k, value in enumerate(scores):
                records += tracker.step(np.array([value]), k+1.)
            records += tracker.flush()
            assert actual['start_alerts'][j] == sum(e['event'] == 'start' for e in records)
            assert actual['confirmed_events'][j] == sum(e.get('confirmed', False) for e in records)
    sequence = np.r_[0., .9, .9, .4, .4, .9, .9, 0.]
    count = edge._calibration_grid_counts(sequence, grid, off_ratio=.6, smooth_hops=1, min_hops=1)
    assert list(count['start_alerts']) == [1, 2, 2, 0]


def test_unreachable_zero_budget_and_constant_input_are_flagged():
    model, audio, _ = fixture(hours=3, bursts=0)
    assert edge.calibrate(model, audio, fa_per_hour=0)['status'] == 'insufficient_evidence'
    assert edge.calibrate(model, np.zeros_like(audio))['input_constant'] is True
    assert edge.calibrate(model, np.zeros_like(audio))['status'] == 'insufficient_evidence'


def test_cli_does_not_save_unsuccessful_calibration(tmp_path, monkeypatch, capsys):
    import json
    model, audio, _ = fixture(peak=1.)
    monkeypatch.setattr(edge, 'Model', lambda _: model)
    monkeypatch.setattr(edge, '_wav_rate', lambda _: model.meta['features']['fs'])
    monkeypatch.setattr(edge, 'read_wav', lambda *args: audio)
    saved = []
    monkeypatch.setattr(edge, 'save_detector', lambda *a, **kw: saved.append(kw))
    assert edge.main(['calibrate', 'model.npz', 'bed.wav', '--out', str(tmp_path/'tuned.npz')]) == 2
    assert not saved
    assert json.loads(capsys.readouterr().out)['status'] == 'infeasible'
