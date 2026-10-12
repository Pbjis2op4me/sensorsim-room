"""Development controls for label provenance and explicit proxy-only authority."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from sensorsim.negatives import integration, procedural
from sensorsim.negatives.composer import audio_hash


def source_row(family='hvac_fan'):
    from sensorsim.negatives.splits import Item
    source=procedural.generate(family,fs=16000,duration_s=2,seed=913,target_partials_hz=[440.])
    return SimpleNamespace(audio=source.audio,fs=16000,
        item=Item('control',family,'control',audio_hash(source.audio)),
        provenance=dict(source.provenance,training_eligible=True,acoustic_stage='receiver'))


def test_declared_family_must_match_generator_model():
    good=source_row();integration.require_realism(good)
    bad=source_row('phone_notification');bad.item=replace(bad.item,family='hvac_fan')
    with pytest.raises(ValueError,match='hvac_fan.*phone_notification'):
        integration.require_realism(bad)


def test_shared_generator_still_binds_the_family_parameters():
    bad=source_row('extractor_hood');bad.item=replace(bad.item,family='hvac_fan')
    with pytest.raises(ValueError,match='hvac_fan.*extractor_hood|family.*parameters'):
        integration.require_realism(bad)


@pytest.mark.parametrize('family',['door_closure','phone_notification'])
def test_proxy_receipt_is_explicit_not_a_listening_review(family):
    r=source_row(family)
    with pytest.raises(ValueError,match='receipt'):
        integration.require_realism(r)
    r.provenance['realism_gate']=integration.proxy_screen_receipt(r.item)
    with pytest.raises(ValueError,match='screen'):
        integration.require_realism(r)
    integration.require_realism(r,allow_proxy_screen=True)
    assert r.provenance['realism_gate']['human_listening_review'] is False
    r.provenance['realism_gate']['audio_sha256']='0'*64
    with pytest.raises(ValueError,match='bound'):
        integration.require_realism(r,allow_proxy_screen=True)


def test_screen_pool_cannot_be_sampled_by_production_route(tmp_path):
    from tests.unit.negatives.test_training_integration import plan_for
    from sensorsim import nn
    r=source_row('door_closure');r.provenance['realism_gate']=integration.proxy_screen_receipt(r.item)
    plan=plan_for(tmp_path,[r])
    pool=integration.LockedEventPool([r],plan=plan,expected_sha256=plan.sha256,purpose='train',allow_proxy_screen=True)
    corpus=SimpleNamespace(negative_pool=pool,pool_purpose='train')
    cfg=dict(nn.DEFAULTS,p_confuser_given_background=1.)
    with pytest.raises(ValueError,match='screen'):
        integration.draw_confuser(corpus,np.random.default_rng(1),cfg,16000)
    assert integration.draw_confuser(corpus,np.random.default_rng(1),dict(cfg,proxy_screen=True),16000) is not None
    assert pool.metadata()['proxy_screen_only'] is True


def test_missing_procedural_identity_is_refused():
    r=source_row();r.provenance.pop('model')
    with pytest.raises(ValueError,match='provenance|model'):
        integration.require_realism(r)
