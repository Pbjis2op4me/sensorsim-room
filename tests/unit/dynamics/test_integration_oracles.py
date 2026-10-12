"""WP7 reviewer decomposition and stress-gain controls, project-generated audio."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from sensorsim import nn, iot_mic, training_device, event_level
from tools import dynamics_sweep as sweep


def fixture():
    rng=np.random.default_rng(7)
    t=np.arange(48000)/16000
    x=.02*np.sin(2*np.pi*700*t)*np.exp(-(t%1)*18)*(t%1<.2)
    return SimpleNamespace(classes=["knock"],targets={"knock":[x.astype(np.float32)]},
        spans={"knock":[(0,len(x))]},background=[rng.normal(0,1e-5,64000)])


def decomposition_error(seed=42):
    calls=[]
    original=iot_mic.MicChain.apply
    def spy(chain,x,fs,rng,**kw):
        calls.append((chain,np.asarray(x,float).copy(),fs,dict(kw)))
        return original(chain,x,fs,rng,**kw)
    with patch.object(iot_mic.MicChain,"apply",spy):
        audio,labels,trace=sweep.trace_example(fixture(),np.random.default_rng(seed),
            dict(nn.DEFAULTS,p_background_only=0,p_confuser_given_background=0))
    chain,x,fs,kw=calls[-1]
    assert kw["include_self_noise"] is False and kw["quantize"] is False
    increment=original(chain,x,fs,np.random.default_rng(0),**kw)
    n=len(audio)
    assert np.any(increment[n:]) and labels[0]==1
    # Neither tracer event subtraction nor production conditional_capture is the oracle.
    expected=trace["input_floor"][n:]+increment[n:]
    assert np.max(abs(expected)) < .5  # no overload on this control
    return max(float(np.max(abs(trace["floor_full"]-trace["input_floor"]))),
               float(np.max(abs(trace["mixer_output"][n:]-expected))))*32768


def test_tracer_decomposition_and_capture_mutants():
    assert max(decomposition_error(s) for s in (1,2,3,42)) < 1e-3
    def rounded(b,e,rng,p): return iot_mic.quantize_profile(p,b+e)
    def dither(b,e,rng,p): return np.where(e==0,b,np.floor(b*32768+rng.random(b.shape)+e*32768)/32768)
    def drop(b,e,rng,p): return b.copy()
    for mutant in (rounded,dither,drop):
        with patch.object(training_device,"conditional_capture",mutant):
            assert decomposition_error() > 1e-3


def stress_errors():
    c=fixture(); src=c.targets["knock"][0].astype(float)
    full=float(np.mean(src**2)); errors=[]
    original=nn.mix_device_audio
    seen=[]
    def spy(b,e,span,rng,cfg):
        state=copy.deepcopy(rng); y=original(b,e,span,rng,cfg)
        seen.append((b,e,span,state,cfg,y)); return y
    cfg=dict(nn.DEFAULTS,p_background_only=0,p_confuser_given_background=0,
        partial_windows=True,device_split="stress",p_mic_chain=0.)
    with patch.object(nn,"mix_device_audio",spy):
        for seed in range(20):
            nn.make_example(c,np.random.default_rng(seed),cfg)
            b,e,(a,z),rng,used,y=seen[-1]
            requested=copy.deepcopy(rng).uniform(*cfg["snr_db"])
            applied=rng.uniform(*used["snr_db"])
            pb=np.mean(b[a:z]**2); pt=np.mean(e[a:z]**2)
            gain=np.sqrt(pb*10**(applied/10)/pt)
            expected=np.clip(np.round((b+e*gain)*32767)/32767,-1,1)
            np.testing.assert_array_equal(expected,y)
            inherited=np.sqrt(pb*10**(requested/10)/full)
            errors.append(20*np.log10(gain/inherited))
    return np.asarray(errors)


def test_stress_split_partial_windows_inherit_span_mean_snr():
    assert np.max(abs(stress_errors())) < 1e-9


def test_stress_oracle_rejects_active_measure_mutant():
    def wrong(x,fs,span,*,split): return training_device.active_power(x,fs,span)[0]
    with patch.object(event_level,"level_power",wrong):
        assert np.max(abs(stress_errors())) > .5
