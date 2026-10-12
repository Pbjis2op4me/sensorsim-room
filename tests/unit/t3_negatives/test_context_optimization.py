# SPDX-License-Identifier: Apache-2.0
"""Deterministic cache and integrity controls; synthetic proxy route equivalence."""
from dataclasses import replace
import os
import pickle

import numpy as np
import pytest

from sensorsim import training_negatives as tn
from sensorsim.acoustics.room import synth
from sensorsim.negatives.composer import audio_hash
from sensorsim.negatives.splits import SplitPlan


def test_pool_access_does_not_reparse_lock(factory, monkeypatch):
    f = factory(); pool = f.build()
    # Admission already succeeded. Ordinary access must still hash the lock but
    # must not repeat the semantic loader (the original path does).
    def repeated(*args, **kwargs):
        raise AssertionError('split reconstructed after admission')
    monkeypatch.setattr(SplitPlan, 'load', repeated)
    for _ in range(3):
        pool.choose(np.random.default_rng(1))


@pytest.mark.parametrize('kind', ['split', 'receipt', 'audio', 'review', 'weights', 'position'])
def test_midrun_tampering_is_refused_even_with_same_stat(factory, kind):
    f = factory(); pool = f.build(); pool.choose(np.random.default_rng(1))
    if kind in ('split', 'receipt', 'audio', 'review'):
        path = {'split':f.root/'split.json', 'receipt':f.root/'split.json.sha256',
                'audio':f.root/'parent-3.wav', 'review':f.root/'review-3.json'}[kind]
        stat = path.stat(); raw = bytearray(path.read_bytes())
        raw[len(raw)//2] ^= 1
        path.write_bytes(raw); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    elif kind == 'weights':
        pool.weights[0] += 1
    else:
        pool.rows[0].provenance['positions']['room-A'][0] += .25
    with pytest.raises(ValueError):
        pool.choose(np.random.default_rng(1))


def test_pool_pickle_rechecks_pinned_bytes(factory):
    f = factory(); pool = f.build()
    clone = pickle.loads(pickle.dumps(pool))
    assert clone.choose(np.random.default_rng(1)).item == pool.choose(np.random.default_rng(1)).item
    (f.root/'split.json.sha256').unlink()
    with pytest.raises((ValueError, OSError)):
        clone.choose(np.random.default_rng(1))


def test_receipt_whitespace_keeps_original_semantics(factory):
    f = factory(); pool = f.build(); path=f.root/'split.json.sha256'
    path.write_text(' \n'+path.read_text()+'\n ')
    assert pool.choose(np.random.default_rng(1)).item in f.parents


def test_content_checks_catch_designated_no_verification_mutant(factory, monkeypatch):
    f=factory(); pool=f.build(); pool.choose(np.random.default_rng(1))
    # A broken optimization trusts previous admission even after an unselected
    # external parent changes. The integrity oracle must reject that route.
    path=f.root/'parent-3.wav'; raw=bytearray(path.read_bytes()); raw[-1]^=1;path.write_bytes(raw)
    def requires_rejection():
        with pytest.raises(ValueError):pool.choose(np.random.default_rng(1))
    requires_rejection()
    reached=[]
    def bad():reached.append(True); return pool.admission.plan
    monkeypatch.setattr(pool,'verify',bad)
    with pytest.raises(pytest.fail.Exception):requires_rejection()
    assert reached


def test_prepare_checks_integrity_once_on_injected_and_off_routes(factory, monkeypatch):
    f = factory(); pool = f.build(); f.corpus.negative_pool = pool
    calls = []; original = pool.verify
    def counted():
        calls.append(1); return original()
    monkeypatch.setattr(pool, 'verify', counted)
    for probability in (1., 0.):
        calls.clear()
        tn.prepare_context(f.corpus, np.random.default_rng(1),
            dict(f.cfg, p_confuser_given_target=probability), f.fs, class_name='knock', target_index=0)
        assert len(calls) == 1


def uncached(audio, draw, position, fs):
    """Original arithmetic, including its whole-parent convolution before crop."""
    from scipy.signal import oaconvolve
    from sensorsim.recognizer import _audible_span
    seed = int(np.random.SeedSequence([draw.seed, 0, 11]).generate_state(1)[0])
    rir = synth.synthesize(draw.room, position, draw.receiver_xyz_m, fs, seed=seed, config=draw.config)
    wet = oaconvolve(audio, rir.samples)
    return wet, _audible_span(wet, fs), audio_hash(rir.samples), seed


def exact(a, b):
    assert a[0].dtype == b[0].dtype and a[0].shape == b[0].shape
    assert a[0].tobytes() == b[0].tobytes()
    assert a[1:] == b[1:]


@pytest.mark.parametrize('late', [False, True])
def test_cached_render_matches_original_arithmetic_and_rng(factory, monkeypatch, late):
    from sensorsim import training_context_cache as cache
    cache.clear(); f = factory(); pool = f.build(); row = pool.rows[0]
    draw = replace(f.draws['room-A'], config=synth.SynthConfig() if late else f.draws['room-A'].config)
    position = [1.,1.,1.]; expected = uncached(row.audio, draw, position, f.fs)
    calls = []; original = synth.synthesize
    def counted(*args, **kwargs):
        calls.append(kwargs['seed']); return original(*args, **kwargs)
    monkeypatch.setattr(synth, 'synthesize', counted)
    rng_before = np.random.get_state()
    for _ in range(2):
        actual = cache.render_parent(row.audio, draw=draw, position=position, fs=f.fs)
        exact(actual, expected)
    assert calls == [expected[3]]
    np.testing.assert_array_equal(np.random.get_state()[1], rng_before[1])
    # Same geometry, different parent: RIR reused, convolution distinct.
    second = cache.render_parent(pool.rows[1].audio, draw=draw, position=position, fs=f.fs)
    assert len(calls) == 1 and second[0].tobytes() != actual[0].tobytes()
    with pytest.raises(ValueError):
        actual[0].setflags(write=True)


@pytest.mark.parametrize('change', ['position', 'receiver', 'seed', 'config', 'room', 'rate', 'content', 'dtype'])
def test_key_changes_reach_renderer_and_match_original(factory, change):
    from sensorsim import training_context_cache as cache
    cache.clear(); f=factory(); audio=f.build().rows[0].audio.copy(); draw=f.draws['room-A']
    position=[1.,1.,1.]; fs=f.fs
    old=cache.render_parent(audio, draw=draw, position=position, fs=fs)
    if change=='position': position[0] += .5
    if change=='receiver': draw=replace(draw,receiver_xyz_m=(2.5,2.,1.))
    if change=='seed': draw=replace(draw,seed=19,config=synth.SynthConfig())
    if change=='config': draw=replace(draw,config=synth.SynthConfig(max_order=1,late=False))
    if change=='room': draw=replace(draw,room=draw.room.model_copy(update={'temperature_c':24.}))
    if change=='rate': fs=8000
    if change=='content': audio[100] += .001
    if change=='dtype': audio=audio.astype(np.float32)
    expected=uncached(audio,draw,position,fs)
    actual=cache.render_parent(audio,draw=draw,position=position,fs=fs)
    exact(actual,expected)
    # Designated incorrect optimization: reuse the first geometry/content key.
    if change not in ('dtype',):
        with pytest.raises(AssertionError): exact(old,expected)


def test_changed_hook_and_unavailable_renderer_do_not_hit_old_cache(factory, monkeypatch):
    from sensorsim import training_context_cache as cache
    cache.clear(); f=factory(); audio=f.build().rows[0].audio; draw=f.draws['room-A']
    kwargs=dict(draw=draw,position=[1.,1.,1.],fs=f.fs)
    expected=cache.render_parent(audio,**kwargs)
    original=synth.synthesize; reached=[]
    def bad(*args,**kw):
        reached.append(True); rir=original(*args,**kw)
        return replace(rir,samples=np.zeros_like(rir.samples))
    monkeypatch.setattr(synth,'synthesize',bad)
    wrong=cache.render_parent(audio,**kwargs)
    assert reached and not np.any(wrong[0])
    with pytest.raises(AssertionError): exact(wrong,expected)
    def missing(*args,**kw):raise ImportError('renderer unavailable')
    monkeypatch.setattr(synth,'synthesize',missing)
    with pytest.raises(ImportError,match='unavailable'):cache.render_parent(audio,**kwargs)


def test_one_ulp_mutant_is_rejected(factory):
    from sensorsim import training_context_cache as cache
    f=factory(); expected=cache.render_parent(f.build().rows[0].audio,
        draw=f.draws['room-A'],position=[1.,1.,1.],fs=f.fs)
    audio=expected[0].copy(); audio[100]=np.nextafter(audio[100],np.inf)
    with pytest.raises(AssertionError):exact((audio,*expected[1:]),expected)


def test_late_seed_alone_changes_rir(factory):
    from sensorsim import training_context_cache as cache
    cache.clear(); f=factory(); audio=f.build().rows[0].audio
    draw=replace(f.draws['room-A'],config=synth.SynthConfig())
    a=cache.render_parent(audio,draw=draw,position=[1.,1.,1.],fs=f.fs)
    other=replace(draw,seed=draw.seed+1)
    b=cache.render_parent(audio,draw=other,position=[1.,1.,1.],fs=f.fs)
    exact(b,uncached(audio,other,[1.,1.,1.],f.fs))
    assert a[2]!=b[2] and a[0].tobytes()!=b[0].tobytes()


@pytest.mark.parametrize('level',[0.,.1])
def test_helper_constant_inputs_keep_original_arithmetic(factory, level):
    from sensorsim import training_context_cache as cache
    cache.clear();f=factory();draw=f.draws['room-A'];audio=np.full(f.fs,level)
    actual=cache.render_parent(audio,draw=draw,position=[1.,1.,1.],fs=f.fs)
    exact(actual,uncached(audio,draw,[1.,1.,1.],f.fs))


def test_cache_budget_eviction_and_duplicate_content(factory, monkeypatch):
    from sensorsim import training_context_cache as cache
    cache.clear(); f=factory(); audio=f.build().rows[0].audio
    kwargs=dict(draw=f.draws['room-A'],position=[1.,1.,1.],fs=f.fs)
    expected=cache.render_parent(audio,**kwargs)
    exact(cache.render_parent(audio.copy(),**kwargs),expected)
    cache.clear(); monkeypatch.setattr(cache,'_MAX_BYTES',1)
    for _ in range(2): exact(cache.render_parent(audio,**kwargs),expected)
    assert cache.info()['entries']==0 and cache.info()['bytes']==0


def test_numpy_integer_sample_rate_keeps_original_route(factory):
    from sensorsim import training_context_cache as cache
    cache.clear(); f=factory(); audio=f.build().rows[0].audio
    draw=f.draws['room-A']; fs=np.int64(f.fs)
    # The original synthesizer accepts NumPy integer rates; the JSON cache key
    # must not narrow that contract to Python integers.
    expected=uncached(audio,draw,[1.,1.,1.],fs)
    exact(cache.render_parent(audio,draw=draw,position=[1.,1.,1.],fs=fs),expected)
