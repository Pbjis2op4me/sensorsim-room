"""Deterministic development controls for the separable discrete tail operator."""
import numpy as np
import pytest
from scipy import signal
from sensorsim.acoustics.room import late, fit
from late_real_oracle import real_expected_filtered_power, direct_sos_factor_power


def fixture():
    synthesis = signal.butter(2, .273, output='sos')
    analysis = signal.butter(2, [.183, .417], btype='bandpass', output='sos')
    prefix = np.r_[np.zeros(13), np.sin(np.linspace(0, np.pi/2, 93))*np.exp(-np.arange(93)/317)]
    return synthesis, analysis, prefix, float(np.exp(-93/317)), float(np.exp(-1/317))


def test_prepared_reuses_physical_state_and_decay_independent_grid(monkeypatch):
    args = fixture()
    late.clear_energy_caches()
    calls = []
    original = late.expected_filtered_power
    def counted(*a, **kw):
        calls.append(len(a[2]))
        return original(*a, **kw)
    monkeypatch.setattr(late, 'expected_filtered_power', counted)
    state = late.prepare_tail(*args)
    assert late.prepare_tail(*args) is state
    for offset in (0, 3, 19):
        edges = np.r_[np.arange(offset, 400, 13), 443]
        actual = state.integrate_edges(edges)
        envelope = np.r_[args[2], args[3]*args[4]**np.arange(443-len(args[2]))]
        expected = np.diff(np.r_[0, np.cumsum(original(*args[:2], envelope))][edges])
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12*expected.max())
    assert calls == [92]  # exact zero prefix, including sin(0), is skipped
    before = late.energy_cache_info()['geometry']
    changed = late.prepare_tail(*args[:-1], np.nextafter(args[-1], 0.))
    assert changed is not state
    changed.integrate_edges(edges)
    after = late.energy_cache_info()['geometry']
    assert after.hits == before.hits+1
    assert after.misses == before.misses


@pytest.mark.parametrize('prefix', [np.array([]), np.zeros(12), np.ones(12), np.linspace(.1, 1., 12)])
def test_edges_against_independent_oracle(prefix):
    synthesis, analysis, _, amplitude, ratio = fixture()
    edges = np.array([0, 0, 1, 5, 12, 12, 13, 66, 170])
    envelope = np.r_[prefix, amplitude*ratio**np.arange(edges[-1]-len(prefix))]
    expected, diagnostics = real_expected_filtered_power(synthesis, analysis, envelope)
    expected = np.diff(np.r_[0, np.cumsum(expected)][edges])
    state = late.prepare_tail(synthesis, analysis, prefix, amplitude, ratio)
    actual, trace = state.integrate_edges(edges, diagnostics=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12*expected.max())
    assert actual[0] == actual[4] == 0
    if not np.any(prefix):
        assert trace['branch'] == 'native-empty-prefix'
        assert np.all(actual[edges[1:] <= len(prefix)] == 0)
    assert diagnostics['stationary_normalized_residual'] < 1e-12
    assert not diagnostics['stationary_warnings']
    assert np.all(state.integrate_edges([0, 0, 0]) == 0)


def test_float32_and_prefix_only_are_explicit():
    b, a, e, amp, q = fixture()
    b, a = b.astype('float32'), a.astype('float32')
    state = late.prepare_tail(b, a, e, amp, q)
    assert state is late.prepare_tail(b.astype(float), a.astype(float), e, amp, q)
    edges = np.array([0, 1, 12, 19, 19, 50])
    expected = late.expected_filtered_power(b, a, e)
    np.testing.assert_array_equal(state.integrate_edges(edges), np.diff(np.r_[0, np.cumsum(expected)][edges]))
    changed = b.copy(); changed[0, 0] *= 1.001
    mutant = late.prepare_tail(changed, a, e, amp, q)
    assert mutant is not state
    assert not np.allclose(state.integrate_edges(edges), mutant.integrate_edges(edges), rtol=1e-5, atol=0)


@pytest.mark.parametrize('edges', [[], [1], [-1, 2], [3, 2], [[0, 2]], [0, .5], [0, np.nan], [0, np.inf]])
def test_invalid_edges_rejected(edges):
    with pytest.raises(ValueError):
        late.prepare_tail(*fixture()).integrate_edges(edges)


@pytest.mark.parametrize('which,value', [(2, [[1]]), (2, [np.nan]), (3, np.inf), (3, -1), (4, 1), (4, 0)])
def test_invalid_physical_state_rejected(which, value):
    args = list(fixture()); args[which] = value
    with pytest.raises(ValueError):
        late.prepare_tail(*args)


def test_byte_cache_eviction_accounts_for_keys_views_and_oversize():
    calls = []
    def produce(key):
        calls.append(key)
        return np.ones(400)[::2]  # count retained base, not just the view
    cache = late.ByteCache(produce, 6000)
    cache(b'a'); cache(b'a')
    assert len(calls) == 1
    cache(b'b')
    info = cache.cache_info()
    assert info.evictions == 1 and info.bytes <= info.limit_bytes
    cache(b'a')
    assert len(calls) == 3
    cache(b'x'*7000); cache(b'x'*7000)
    assert len(calls) == 5 and cache.cache_info().oversize == 2
    assert cache.cache_info().peak_bytes <= 6000
    cache.cache_clear()
    assert cache.cache_info().bytes == cache.cache_info().currsize == 0


def test_fit_preparation_identity_and_wall_reuse(monkeypatch):
    late.clear_energy_caches(); fit._tail_power.cache_clear(); fit._prepared_tail.cache_clear()
    args = (8000, 0, 250, .001, 200, 8, 32, .04, .012, .4)
    fit._tail_power(*args)
    prepared = fit._prepared_tail.cache_info()
    fit._tail_power(*args[:4], 210, 9, *args[6:])
    assert fit._prepared_tail.cache_info().hits == prepared.hits+1
    for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        changed = list(args)
        changed[index] = {0:16000, 1:1, 2:500, 3:.002, 4:201, 5:10, 6:33, 7:.041, 8:.013, 9:np.nextafter(.4, 1.)}[index]
        before = fit._tail_power.cache_info().misses
        fit._tail_power(*changed)
        assert fit._tail_power.cache_info().misses == before+1
    assert sum(i.limit_bytes for i in late.energy_cache_info().values()) + sum(
        f.cache_info().limit_bytes for f in (fit._tail_power, fit._tail_edges, fit._tail_envelope, fit._prepared_tail)) <= 1120*1024**2


def test_oracle_qualification_and_wrong_modulation_control():
    b, a, e, amp, q = fixture()
    e = np.r_[e, amp*q**np.arange(100)]
    oracle, diagnostics = real_expected_filtered_power(b, a, e)
    columns = direct_sos_factor_power(b, a, e)
    np.testing.assert_allclose(oracle, columns, rtol=1e-9, atol=1e-12*oracle.max())
    # Known-bad operator moves the envelope after analysis. Same filters and
    # independent state propagation, but it demonstrably changes the output.
    stationary, _ = real_expected_filtered_power(b, a, np.ones(len(e)))
    wrong = stationary*e**2
    assert np.max(abs(wrong-oracle))/oracle.max() > 1e-3
    assert diagnostics['stationary_direct_vs_bilinear_norm_relative'] < 1e-10


def test_explicit_empty_covariance_and_near_pole_fallback(monkeypatch):
    b = signal.butter(2, .23, output='sos')
    out, cross, state = late.expected_filtered_power(b, b, [], return_state=True)
    assert out.size == 0 and np.all(cross == 0) and np.all(state == 0)
    late.clear_energy_caches()
    calls = []
    original = late.expected_filtered_power
    def counted(*a, **kw):
        calls.append(len(a[2])); return original(*a, **kw)
    monkeypatch.setattr(late, 'expected_filtered_power', counted)
    prefix = np.linspace(0, 1., 40)
    prepared = late.prepare_tail(b, b, prefix, 1., 1-1e-12)
    actual, trace = prepared.integrate_edges([0, 40, 80], diagnostics=True)
    assert trace['branch'] == 'native-near-pole'
    assert calls == [39, 80]
    expected, _ = real_expected_filtered_power(b, b, np.r_[prefix, (1-1e-12)**np.arange(40)])
    np.testing.assert_allclose(actual, [expected[:40].sum(), expected[40:].sum()], rtol=1e-9)


def test_forced_eviction_and_conditioning_fallback_are_observed(monkeypatch):
    args = fixture(); edges = np.arange(0, 800, 7)
    expected = late.prepare_tail(*args).integrate_edges(edges)
    cache = late.ByteCache(late._edge_geometry.__wrapped__, 1000)
    monkeypatch.setattr(late, '_edge_geometry', cache)
    actual = late.prepare_tail(*args).integrate_edges(edges)
    np.testing.assert_array_equal(actual, expected)
    assert cache.cache_info().oversize >= 1 and cache.cache_info().bytes == 0
    # Plausible broken consolidation: negate modal coefficients. The guard
    # must observe the resulting invalid sum and actually invoke recurrence.
    from dataclasses import replace
    state = late.prepare_tail(*args)
    mutant = replace(state, coefficients=-abs(state.coefficients)*1e6)
    corrected, trace = mutant.integrate_edges(edges, diagnostics=True)
    assert trace['branch'] == 'native-conditioning'
    envelope = np.r_[args[2], args[3]*args[4]**np.arange(edges[-1]-len(args[2]))]
    native = late.expected_filtered_power(*args[:2], envelope)
    np.testing.assert_array_equal(corrected, np.diff(np.r_[0, np.cumsum(native)][edges]))


@pytest.mark.parametrize('fs,dt', [(44100, .0007), (48000, .001), (8000, .0001)])
def test_native_edges_cover_the_actual_sample_assignment(fs, dt):
    n = 97
    edges = fit._tail_edges(fs, dt, n)
    assignment = np.floor(np.arange(int(np.ceil(n*dt*fs))+1)/fs/dt).astype(int)
    # Independent enumeration, including empty bins at widths below one sample.
    expected = np.array([np.count_nonzero(assignment < k) for k in range(n+1)])
    np.testing.assert_array_equal(edges, expected)


def test_wrong_edge_units_and_bad_rate_mutants_are_detected():
    state = late.prepare_tail(*fixture())
    edges = np.arange(0, 401, 20)
    correct = state.integrate_edges(edges)
    wrong_grid = state.integrate_edges(edges*2)
    b, a, e, amp, q = fixture()
    wrong_decay = late.prepare_tail(b, a, e, amp, q**2).integrate_edges(edges)
    for mutant in (wrong_grid, wrong_decay):
        assert np.max(abs(mutant-correct))/correct.max() > 1e-3


def test_relative_blocks_reuse_onsets_without_losing_absolute_sample_origin():
    late.clear_energy_caches()
    b, a, e, amp, q = fixture()
    state = late.prepare_tail(b, a, e, amp, q)
    first = np.arange(200, 200+257*8, 8)
    second = first+19
    one = state.integrate_edges(first)
    before = late.energy_cache_info()['geometry']
    two = state.integrate_edges(second)
    after = late.energy_cache_info()['geometry']
    assert after.hits > before.hits and after.misses == before.misses
    assert not np.allclose(one, two, rtol=1e-5, atol=0)
    envelope = np.r_[e, amp*q**np.arange(second[-1]-len(e))]
    power = late.expected_filtered_power(b, a, envelope)
    expected = np.diff(np.r_[0, np.cumsum(power)][second])
    np.testing.assert_allclose(two, expected, rtol=1e-9, atol=1e-12*expected.max())


def test_physical_accuracy_gate_reaches_its_smallest_declared_gain_defect():
    b, a, prefix, amplitude, ratio = fixture()
    edges = np.arange(0, 481, 16)
    envelope = np.r_[prefix, amplitude*ratio**np.arange(edges[-1]-len(prefix))]
    power, _ = real_expected_filtered_power(b, a, envelope)
    expected = np.diff(np.r_[0, np.cumsum(power)][edges])
    def discrepancy(relative_energy_change):
        scale = np.sqrt(1+relative_energy_change)
        actual = late.prepare_tail(b, a, prefix*scale, amplitude*scale, ratio).integrate_edges(edges)
        return np.max(abs(actual-expected))/expected.max()
    assert discrepancy(0.) <= 1e-7
    assert discrepancy(5e-8) <= 1e-7
    assert discrepancy(1.01e-7) > 1e-7


def test_desktop_geometry_budget_retains_realistic_working_set():
    assert late._edge_geometry.cache_info().limit_bytes == 1024*1024**2
