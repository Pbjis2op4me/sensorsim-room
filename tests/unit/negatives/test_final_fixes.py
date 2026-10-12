"""Final-review regressions using project-generated proxy controls only."""
from dataclasses import replace
from datetime import date
import hashlib
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from sensorsim import iot_mic, nn
from sensorsim.negatives import procedural, mining
from sensorsim.negatives.composer import Event, render_context, audio_hash
from sensorsim.negatives.integration import LockedEventPool
from sensorsim.negatives.splits import Item, SplitPlan
from sensorsim.training_device import conditional_capture, active_power, _check_recording
from test_review_fixes import path


def test_l4_float_increment_and_overload():
    profile = iot_mic.PROFILES['pi5_inmp441/1']
    b = np.full(16000, 2/32768.)
    e = np.full(16000, .3/32768.)
    rng = np.random.default_rng(8)
    state = repr(rng.bit_generator.state)
    np.testing.assert_array_equal(conditional_capture(b, e, rng, profile), b+e)
    assert repr(rng.bit_generator.state) == state
    limited = replace(profile, sensitivity_dbfs_at_94db_spl=-26., max_spl_peak_db=114.)
    level = 10**(-6/20)
    np.testing.assert_allclose(conditional_capture(np.array([0.,0.]), np.array([-2.,2.]), rng, limited), [-level,level])


def pool_fixture(tmp_path, provenance):
    src = procedural.generate('hammering', fs=16000, duration_s=1, seed=5)
    item = Item('impact','hammering','source',audio_hash(src.audio))
    audit = Item('audit','vacuum','audit','a'*64)
    plan = SplitPlan.create([item,audit],audit_families=['vacuum'],development_families=[],seed=1)
    plan.save(tmp_path/'lock.json')
    plan = SplitPlan.load(tmp_path/'lock.json',expected_sha256=plan.sha256)
    row = SimpleNamespace(audio=src.audio,item=item,fs=16000,provenance=dict(src.provenance, **provenance))
    return row,plan


def test_pool_refuses_explicit_ineligible(tmp_path):
    row,p = pool_fixture(tmp_path, {'training_eligible':False})
    with pytest.raises(ValueError, match='realism|eligible'):
        LockedEventPool([row],plan=p,expected_sha256=p.sha256,purpose='train')


def test_pool_refuses_missing_realism_receipt(tmp_path):
    row,p = pool_fixture(tmp_path, {'training_eligible':True})
    with pytest.raises(ValueError, match='realism'):
        LockedEventPool([row],plan=p,expected_sha256=p.sha256,purpose='train')


@pytest.mark.parametrize('family', procedural.IMPACTS)
def test_impact_boundary_cannot_bypass_composer(family):
    src = procedural.generate(family,fs=16000,duration_s=1,seed=5)
    with pytest.raises(ValueError, match='reviewed'):
        render_context([Event(src,0,(1.,1.,1.))],path=path(),seed=1)


def test_generator_regions_have_specific_assumption_rationales():
    for family in procedural.EVENT_FAMILIES+procedural.BED_FAMILIES:
        p = procedural.generate(family,fs=16000,duration_s=1,seed=2).provenance
        assert p.get('parameter_assumptions'), family
        assert all(row.get('rationale') and row.get('values') is not None
                   for row in p['parameter_assumptions'].values())


def test_saved_rebuilt_plan_cannot_replace_preregistered_file(tmp_path):
    row,p = pool_fixture(tmp_path, {})
    audit = p.rows[0][0]
    assert audit.family == 'vacuum'
    orig = tmp_path/'lock.json'
    registered_file_hash = hashlib.sha256(orig.read_bytes()).hexdigest()
    rebuilt = SplitPlan.create([i for i,_ in p.rows],audit_families=['hammering'],development_families=['vacuum'],seed=1)
    rebuilt.save(tmp_path/'rebuilt.json')
    # Rebuilt file has a perfectly self-consistent payload and receipt.
    # Mining must compare against the original FILE hash registered by the study.
    model_file = tmp_path/'model'; model_file.write_bytes(b'fixture')
    model = mining.FrozenModel(model_file,lambda x:[.9],classes=['knock'],fs=16000)
    with pytest.raises(ValueError, match='hash|lock'):
        mining.mine([],model=model,split=tmp_path/'rebuilt.json',expected_split_sha256=registered_file_hash)


def test_user_row_separates_original_attestation_from_current_check():
    from sensorsim.negatives.clips import user_register_row
    item = dict(registry_id='owned',sha256='a'*64,license_id='user-owned',verified_on='2026-08-01',
                owner='fixture',ownership_statement='generated test control')
    row = user_register_row(item,name='Fixture',evidence_url='https://example.test/fixture')
    assert row['checked_on'] == date.today().isoformat()
    assert row['ownership_attested_on'] == item['verified_on']


def test_tonal_draws_exclude_training_target_partials():
    from sensorsim.negatives.tonal import target_partials
    fs=16000; t=np.arange(fs)/fs
    target=np.sin(2*np.pi*880*t)+.5*np.sin(2*np.pi*1320*t)
    partials=target_partials([target],fs=fs)
    assert min(abs(np.asarray(partials)-880)) < 2
    assert min(abs(np.asarray(partials)-1320)) < 2
    for family in procedural.TONES:
        for seed in range(20):
            try:
                src=procedural.generate(family,fs=fs,duration_s=1,seed=seed,target_partials_hz=partials)
            except ValueError as exc:
                assert 'exclusion' in str(exc)  # Entire sweep support can be excluded.
                continue
            ranges=src.provenance['parameters']['tonal_support_hz']
            for lo,hi in ranges:
                assert all(hi < f*.97 or lo > f*1.03 for f in partials)


def test_floor_warning_requires_margin_and_sustained_duration():
    profile=iot_mic.PROFILES['pi5_inmp441/1']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for seed in range(40):
            rng=np.random.default_rng(seed)
            recorded=profile.with_variation(rng).apply(np.zeros(32000),16000,rng,input_reference_spl_db=94.)
            _check_recording(recorded,16000,profile.with_variation(np.random.default_rng(seed+100)),True)
        _check_recording(np.full(8000,1/32768.),16000,profile,True)
    assert not [w for w in caught if 'below the profile' in str(w.message)]
    with pytest.warns(UserWarning,match='below the profile'):
        _check_recording(np.full(32000,1/32768.),16000,profile,True)


def test_active_level_and_partial_crop_gain_inheritance():
    t=np.arange(32000)/16000
    source=np.sin(2*np.pi*700*t)*np.exp(-t*2)*(np.sin(2*np.pi*3*t)>.7)
    full,_=active_power(source,16000)
    for start in (1000,7000,15000):
        crop=source[start:start+16000]
        padded=np.r_[np.zeros(16000),crop]
        power,_=active_power(padded,16000,(16000,32000))
        shift=10*np.log10(power/full)
        assert 10**(shift/20)/np.sqrt(power) == pytest.approx(1/np.sqrt(full))


def test_evaluation_adapter_accepts_off_grid_training_windows():
    from sensorsim.evaluation.stream import training_example
    c=SimpleNamespace(classes=['knock'],background=[np.full(48000,10/32768.,np.float32)],
        targets={'knock':[np.sin(np.arange(16000)*.3).astype(np.float32)]},spans={'knock':[(0,16000)]})
    out=training_example(c,np.random.default_rng(6),dict((nn.DEFAULTS | {"p_confuser_given_background": 0., "selection_mode": "diagnostic_final"})))
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert np.any(out*32768 != np.rint(out*32768))


def test_pool_revalidates_realism_receipt_at_draw(tmp_path):
    row,p = pool_fixture(tmp_path, {'training_eligible':True})
    receipt = dict(decision='pass', family=row.item.family, audio_sha256=row.item.sha256,
        reference_kind='recorded_family',reviewer='fixture control only',evidence_sha256='a'*64,
        reference_sha256='b'*64,reviewed_label_boundary=True)
    for key in ("evidence", "reference"):
        path = tmp_path/key
        path.write_bytes((key+" test fixture").encode())
        receipt[key+"_file"] = str(path)
        receipt[key+"_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row.provenance['realism_gate'] = receipt
    pool = LockedEventPool([row],plan=p,expected_sha256=p.sha256,purpose='train')
    window,span = pool.window(np.random.default_rng(3),16000,fs=16000,purpose='train')
    assert np.any(window)
    receipt['audio_sha256'] = 'c'*64
    with pytest.raises(ValueError,match='realism'):
        pool.window(np.random.default_rng(3),16000,fs=16000,purpose='train')


def test_tonal_composer_requires_training_anchors_even_after_review():
    from sensorsim.negatives.composer import compose
    with pytest.raises(ValueError,match='training targets'):
        compose(path=path(),family_rates_per_hour={'phone_notification':1},positions={},seed=1,
                reviewed_label_boundaries=['phone_notification'])


@pytest.mark.parametrize('probability', [-.1, 1.1, float('nan')])
def test_confuser_probability_is_explicit_and_bounded(probability):
    from sensorsim.negatives.integration import draw_confuser
    with pytest.raises(ValueError,match='p_confuser'):
        draw_confuser(SimpleNamespace(),np.random.default_rng(1),{'fs':16000,
            'p_confuser_given_background':probability},16000)


def test_confuser_policy_cannot_silently_ignore_missing_pool():
    from sensorsim.negatives.integration import draw_confuser
    rng=np.random.default_rng(1); state=repr(rng.bit_generator.state)
    assert draw_confuser(SimpleNamespace(),rng,{'fs':16000},16000) is None
    assert repr(rng.bit_generator.state) == state
    with pytest.raises(ValueError,match='pool'):
        draw_confuser(SimpleNamespace(),rng,{'fs':16000,'p_confuser_given_background':1.},16000)


def test_training_pool_rejects_impact_without_boundary_review(tmp_path):
    row,p=pool_fixture(tmp_path, {'training_eligible':True})
    row.provenance['realism_gate']=dict(decision='pass',family=row.item.family,
        audio_sha256=row.item.sha256,reference_kind='recorded_family',reviewer='fixture only',
        evidence_sha256='a'*64,reference_sha256='b'*64,reviewed_label_boundary=False)
    with pytest.raises(ValueError,match='boundary'):
        LockedEventPool([row],plan=p,expected_sha256=p.sha256,purpose='train')
