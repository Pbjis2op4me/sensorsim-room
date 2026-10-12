"""Generated full-take cache controls; byte equivalence, not distribution realism."""
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from sensorsim import nn, event_level, training_device


def test_repeated_partial_take_is_measured_once(monkeypatch):
    source=np.sin(np.arange(8000)*.2)
    corpus=SimpleNamespace(classes=['knock'],targets={'knock':[source]},spans={'knock':[(0,8000)]},
        background=[np.ones(32000)/32768],negative_pool=None)
    original=event_level.level_power
    calls=[]
    def meter(x,fs,span,**kwargs):
        if len(x)==len(source): calls.append(True)
        return original(x,fs,span,**kwargs)
    monkeypatch.setattr(event_level,'level_power',meter)
    cfg=dict(nn.DEFAULTS,partial_windows=True,p_background_only=0.,p_confuser_given_background=0.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for seed in (71,72): nn.make_example(corpus,np.random.default_rng(seed),cfg)
    assert len(calls)==1


@pytest.mark.parametrize('kind',['noise','zeros','constant','empty','nonfinite'])
def test_cache_matches_uncached_scalar(kind):
    from sensorsim import training_take_cache as cache
    cache.clear()
    x=np.random.default_rng(113).normal(size=16000)
    if kind=='zeros': x[:]=0
    if kind=='constant': x[:]=.03
    if kind=='empty': x=x[:0]
    if kind=='nonfinite': x[51]=np.nan
    span=(0,len(x))
    if kind=='empty':
        for f in (event_level.level_power,cache.take_level_power):
            with pytest.raises(ValueError):f(x,16000,span)
    else:
        expected=event_level.level_power(x,16000,span)
        for take in (x,x.copy(),x):
            assert cache.take_level_power(take,16000,span).hex()==expected.hex()


def test_content_span_rate_split_and_hook_invalidate(monkeypatch):
    from sensorsim import training_take_cache as cache
    cache.clear()
    original=training_device.active_power
    calls=[]
    def meter(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(training_device,'active_power',meter)
    x=np.sin(np.arange(16000)*.5);span=(0,len(x))
    one=cache.take_level_power(x,16000,span)
    assert cache.take_level_power(x.copy(),16000,span)==one and len(calls)==1
    x*=2
    assert cache.take_level_power(x,16000,span)==4*one and len(calls)==2
    cache.take_level_power(x,8000,span)
    cache.take_level_power(x,8000,(20,100))
    assert len(calls)==4
    assert cache.take_level_power(x,8000,span,split='stress')==event_level.level_power(x,8000,span,split='stress')
    monkeypatch.setattr(training_device,'active_power',lambda *args:(.125,1))
    assert cache.take_level_power(x,16000,span)==.125


def test_cache_is_bounded_and_retains_no_audio():
    from sensorsim import training_take_cache as cache
    cache.clear()
    for k in range(cache._LIMIT+1):
        cache.take_level_power(np.full(200,float(k)),16000,(0,200))
    assert len(cache._values)==cache._LIMIT
    assert all(isinstance(v,float) for v in cache._values.values())
    assert all(not isinstance(part,np.ndarray) for key in cache._values for part in key)


def test_oracle_rejects_known_bad_content_blind_cache(monkeypatch):
    from sensorsim import training_take_cache as cache
    cache.clear()
    first=np.ones(4000);changed=np.full(4000,2.)
    assert cache.take_level_power(first,16000,(0,4000))==1.
    assert cache.take_level_power(changed,16000,(0,4000))==4.
    cache.clear()
    reached=[]
    def wrong_content_key(raw):
        reached.append(len(raw))
        return SimpleNamespace(digest=lambda:b'constant-bad-key')
    # Deliberately ignores content, as an unsafe identity-only optimization would.
    monkeypatch.setattr(cache.hashlib,'sha256',wrong_content_key)
    cache.take_level_power(first,16000,(0,4000))
    bad=cache.take_level_power(changed,16000,(0,4000))
    assert reached==[32000,32000]
    assert bad != event_level.level_power(changed,16000,(0,4000))


def test_byte_oracle_detects_one_float32_ulp_in_invoked_mixer(monkeypatch):
    corpus=SimpleNamespace(classes=['knock'],targets={'knock':[np.sin(np.arange(8000)*.2)]},
        spans={'knock':[(0,8000)]},background=[np.full(32000,.001)],negative_pool=None)
    cfg=dict(nn.DEFAULTS,p_background_only=0.,p_confuser_given_background=0.)
    def draw():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return nn.make_example(corpus,np.random.default_rng(71),cfg)
    expected,label=draw()
    correct,_=draw()
    assert expected.tobytes()==correct.tobytes()
    original=nn.mix_device_audio
    reached=[]
    def one_ulp(*args,**kwargs):
        audio=original(*args,**kwargs)
        index=len(audio)//2
        before=np.float32(audio[index])
        audio[index]=np.nextafter(before,np.float32(np.inf))
        reached.append((before,np.float32(audio[index])))
        return audio
    monkeypatch.setattr(nn,'mix_device_audio',one_ulp)
    wrong,wrong_label=draw()
    assert len(reached)==1 and reached[0][0]!=reached[0][1]
    assert label.tobytes()==wrong_label.tobytes()
    assert np.count_nonzero(expected!=wrong)==1
    assert expected.tobytes()!=wrong.tobytes()
