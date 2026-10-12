"""WP3 contracts: project-generated fixtures only; no downloaded audio."""
import numpy as np
import pytest

from sensorsim.negatives.procedural import generate, EVENT_FAMILIES, BED_FAMILIES


def test_seeded_generators_are_finite_and_bounded():
    for family in (*EVENT_FAMILIES, *BED_FAMILIES):
        a = generate(family, fs=16000, duration_s=1, seed=5)
        b = generate(family, fs=16000, duration_s=1, seed=5)
        assert len(a.audio) == 16000
        assert np.array_equal(a.audio, b.audio)
        assert np.isfinite(a.audio).all()
        assert np.sqrt(np.mean(a.audio**2)) == pytest.approx(1)
        assert a.provenance['license_id'] == 'Apache-2.0'
        assert a.provenance['evidence_label'] == 'proxy'

from pathlib import Path
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path

from scipy.io import wavfile
import yaml

from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, synthesize
from sensorsim.iot_mic import MicChain, PROFILES
from sensorsim.negatives.composer import (Event, RenderPath, audio_hash, captured_context,
                                         compose, render_context)
from sensorsim.negatives.clips import load_clip
from sensorsim.negatives.splits import Item, SplitPlan
from sensorsim.negatives.mining import (FrozenModel, PoolItem, export_review, mine, reviewed_bank)
from sensorsim.negatives.acceptance import symmetry, training_comparison_plan


@pytest.fixture
def path():
    return RenderPath(RoomModel(shoebox_dims_m=(6., 5., 3.)), (4., 3., 1.5),
                      PROFILES['pi5_inmp441/1'], 'pi5_inmp441/1',
                      SynthConfig(late=False, max_order=1))


@pytest.mark.parametrize('kwargs', [dict(fs=7999), dict(fs=16000.5), dict(duration_s=0),
                                   dict(duration_s=float('nan')), dict(mains_hz=55)])
def test_generator_rejects_invalid_support(kwargs):
    with pytest.raises(ValueError):
        generate('fridge_hum', **(dict(fs=16000, duration_s=1, seed=1) | kwargs))


def test_mains_frequency_moves_harmonics():
    for hz in (50., 60.):
        source = generate('fridge_hum', fs=16000, duration_s=2, seed=3, mains_hz=hz)
        frequency = np.fft.rfftfreq(len(source.audio), 1/16000)
        assert frequency[np.argmax(abs(np.fft.rfft(source.audio)))] == hz


def test_taxonomy_covers_all_families_and_marks_unknowns():
    tax = yaml.safe_load((Path(__file__).parents[3]/'src/sensorsim/negatives/taxonomy.yaml').read_text())
    assert {r['family'] for r in tax['families']} == set(EVENT_FAMILIES + BED_FAMILIES)
    for row in tax['families']:
        level = row['typical_level']
        assert level['citation'] or (level['value'] is None and level['status'].startswith('unknown'))
        assert row['prevalence']['events_per_hour'] is None
        assert 'assumption' in row['proxy_level_status']


def test_metadata_only_label_change_is_bit_identical(path):
    source = generate('cabinet_door', fs=16000, duration_s=1, seed=6)
    e = Event(source, .4, (1., 2., 1.), 70, reviewed_label_boundary=True)
    a = render_context([replace(e, label='knock')], path=path, seed=8)
    b = render_context([e], path=path, seed=8)
    assert np.array_equal(a.audio, b.audio)
    assert a.provenance['path_sha256'] == b.provenance['path_sha256']
    assert a.events[0]['first_arrival_s'] > .4
    assert a.events[0]['rendered_end_s'] > a.events[0]['source_end_s']
    assert a.provenance['device_floor'].endswith('added once')
    assert a.provenance['deployment_fa_per_hour'] == 'not measured'


def test_exact_room_convolution_before_single_device_call(path, monkeypatch):
    from scipy import signal
    calls = []
    def capture(self, x, fs, rng, **kwargs):
        calls.append(x.copy())
        return np.clip(x, -1, 1)
    monkeypatch.setattr(MicChain, 'apply', capture)
    src = generate('timer', fs=16000, duration_s=.3, seed=9)
    ev = Event(src, 1., (1., 2., 1.), 70, reviewed_label_boundary=True)
    ctx = render_context([ev], path=path, seed=7)
    row = ctx.events[0]
    rir = synthesize(path.room, ev.source_xyz_m, path.receiver_xyz_m, 16000,
                     seed=row['rir_seed'], config=path.config)
    expected = np.zeros(160000)
    wet = signal.oaconvolve(src.audio, rir.samples)*row['receiver_gain']
    expected[16000:16000+len(wet)] = wet
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], expected, atol=1e-16, rtol=1e-12)


def test_absolute_device_profile_is_parameterized(path):
    with pytest.raises(ValueError, match='absolute'):
        replace(path, device=MicChain())
    ctx = render_context([], path=path, seed=5)
    assert ctx.provenance['path']['input_reference_spl_db'] == 94
    assert ctx.provenance['path']['device_draw']['profile_id'] == path.device_id


def test_poisson_context_timestamps_and_repeatability(path):
    families = {'hammering': 3600., 'running_tap': 1200.}
    positions = {f: ((1., 2., 1.),) for f in (*families, 'room_tone')}
    a = compose(path=path, family_rates_per_hour=families, positions=positions, seed=5, reviewed_label_boundaries=['hammering'])
    b = compose(path=path, family_rates_per_hour=families, positions=positions, seed=5, reviewed_label_boundaries=['hammering'])
    assert len(a.audio) == 160000
    assert np.array_equal(a.audio, b.audio)
    assert len(a.events) > 1
    assert all(0 <= e['onset_s'] < e['rendered_end_s'] <= 10 for e in a.events)
    assert a.events[0]['role'] == 'bed'
    assert a.provenance['prevalence_basis'] == 'caller assumptions'


def test_other_doorbell_needs_review_and_no_double_floor(path):
    with pytest.raises(ValueError, match='reviewed'):
        compose(path=path, family_rates_per_hour={'other_doorbell': 1}, positions={}, seed=1)
    with pytest.raises(ValueError, match='device_floor'):
        compose(path=path, family_rates_per_hour={}, positions={}, seed=1, bed_family='device_floor')


def test_captured_audio_is_not_convolved_or_recaptured(path):
    dry = generate('room_tone', fs=16000, duration_s=10, seed=3)
    wet = replace(dry, audio=np.floor(dry.audio*.01*32768)/32768, provenance=dry.provenance | {
        'acoustic_stage':'captured','capture_device':'fixture-pi','profile_id':path.device_id,
        'deployed_device':True,'native_s16':True})
    with pytest.raises(ValueError, match='only dry'):
        render_context([Event(wet, 0, (1., 2., 1.), 70)], path=path, seed=1)
    ctx = captured_context(wet, fs=16000, path=path)
    assert np.array_equal(wet.audio, ctx.audio)
    assert not ctx.provenance['room_applied'] and not ctx.provenance['device_applied']


def test_tail_crop_is_reported(path):
    src = generate('door_closure', fs=16000, duration_s=1, seed=2)
    ctx = render_context([Event(src, 9.9, (1., 2., 1.), 70, reviewed_label_boundary=True)], path=path, seed=5)
    assert ctx.events[0]['tail_truncated_samples'] > 0
    assert ctx.events[0]['rendered_end_s'] == 10


def clip_fixture(tmp_path, license_id='user-owned'):
    root = tmp_path/'background'
    root.mkdir()
    x = (.05*np.sin(np.arange(16000)*.4)).astype(np.float32)
    wavfile.write(root/'one.wav', 16000, x)
    digest = hashlib.sha256((root/'one.wav').read_bytes()).hexdigest()
    item = dict(path='one.wav', sha256=digest, registry_id='item-1', license_id=license_id,
                verified_on='2026-10-08', owner='test owner', ownership_statement='I made this test signal.')
    register = tmp_path/'register.yaml'
    register.write_text(yaml.safe_dump({'assets':[{'id':'item-1', 'license_id':license_id,
        'version_or_hash':digest, 'checked_on':'2026-10-08', 'evidence_quote':'Project-generated test fixture.',
        'disposition':'plug-in','name':'Fixture recording','url':'https://example.test/ownership'}]}))
    review = dict(family='timer', source_group='one-take', acoustic_stage='captured', listened=True,
                  target_free=True, rights_cleared=True, reviewer='fixture', notes='generated control',
                  capture_device='fixture-recorder',profile_id='external/1',deployed_device=False)
    return root, item, register, review


def test_user_loader_preserves_level_and_provenance(tmp_path):
    root, item, register, review = clip_fixture(tmp_path)
    loaded = load_clip(root, item, register_path=register, review=review, as_of=date(2026,10,8))
    assert max(abs(loaded.audio)) == pytest.approx(.05, abs=1e-6)
    assert loaded.provenance['source_sha256'] == item['sha256']
    assert loaded.provenance['acoustic_stage'] == 'captured'
    assert loaded.provenance['original_fs'] == 16000


@pytest.mark.parametrize('license_id', ['CC-BY-NC-4.0', 'CC-BY-SA-4.0', 'Sampling+', 'unknown'])
def test_loader_rejects_prohibited_licences(tmp_path, license_id):
    root, item, register, review = clip_fixture(tmp_path, license_id)
    with pytest.raises(ValueError, match='prohibited'):
        load_clip(root, item, register_path=register, review=review, as_of=date(2026,10,8))


@pytest.mark.parametrize('change', ['hash', 'register', 'review', 'stale', 'future', 'escape', 'internal'])
def test_loader_fails_closed(tmp_path, change):
    root, item, register, review = clip_fixture(tmp_path)
    if change=='hash': item['sha256'] = '0'*64
    if change=='register': item['registry_id'] = 'absent'
    if change=='review': review['listened'] = False
    if change=='stale':
        item['license_id']='CC0-1.0'
        item['verified_on']='2026-08-01'
    if change=='future': item['verified_on'] = '2026-12-01'
    if change=='escape': item['path'] = '../outside.wav'
    if change=='internal': register.write_text(register.read_text().replace('plug-in', 'internal'))
    with pytest.raises(ValueError):
        load_clip(root, item, register_path=register, review=review, as_of=date(2026,10,8))


def test_external_loader_calls_gate_and_refuses_false_success(tmp_path):
    root, item, register, review = clip_fixture(tmp_path, 'CC0-1.0')
    item.update(freesound_id='123', uploader='fixture', source_url='https://freesound.org/sounds/123/',
                license_url='https://creativecommons.org/publicdomain/zero/1.0/legalcode.en', modifications='none')
    calls=[]
    def gate(manifest, **kwargs):
        calls.append((manifest, kwargs))
        return 1
    load_clip(root, item, register_path=register, review=review, register_gate=gate, as_of=date(2026,10,8))
    assert len(calls)==1 and calls[0][1]['registry_path'] == register
    with pytest.raises(ValueError, match='exactly one'):
        load_clip(root, item, register_path=register, review=review, register_gate=lambda *a,**k:0, as_of=date(2026,10,8))


def pool_fixture(tmp_path=None):
    x = np.sin(np.arange(16000*3)*.1)*.1
    items = [Item('train','door_closure','take-a','a'*64),
             Item('dev','timer','take-b',audio_hash(x)),
             Item('audit','running_tap','take-c','c'*64)]
    split = SplitPlan.create(items, audit_families=['running_tap'], development_families=['timer'], seed=1701)
    if tmp_path is not None:
        split.save(tmp_path/'locked.json')
        split=SplitPlan.load(tmp_path/'locked.json',expected_sha256=split.sha256)
    return split, PoolItem(items[1], x, 16000), items


def test_split_rejects_family_source_and_hash_leaks():
    split, pool, items = pool_fixture()
    for field, value in [('family', items[0].family), ('source_group', items[0].source_group), ('sha256', items[0].sha256)]:
        with pytest.raises(ValueError, match='crosses'):
            SplitPlan(((items[0], 'train'), (replace(items[2], **{field:value}), 'audit')), 1)
    with pytest.raises(ValueError, match='audit'):
        split.require(items[2], purpose='mine')
    with pytest.raises(ValueError, match='audit'):
        split.require(items[2], purpose='tune')


def test_locked_manifest_cannot_be_overwritten_or_edited(tmp_path):
    split, _, _ = pool_fixture()
    file = tmp_path/'split.json'
    split.save(file)
    assert SplitPlan.load(file, expected_sha256=split.sha256) == split
    with pytest.raises(FileExistsError): split.save(file)
    file.write_text(file.read_text().replace('take-c', 'take-changed'))
    with pytest.raises(ValueError, match='changed'): SplitPlan.load(file, expected_sha256=split.sha256)


def model_fixture(tmp_path, scorer=None):
    file=tmp_path/'frozen.json'
    file.write_text('{"model":"test-only"}')
    return FrozenModel(file, scorer or (lambda x: [.9, .1, .2]), classes=['knock','doorbell','kettle'], fs=16000)


def test_mining_rejects_entire_mixed_audit_pool_before_scoring(tmp_path):
    split, pool, items = pool_fixture(tmp_path)
    def never(x): raise AssertionError('must reject pool before any scoring')
    model=model_fixture(tmp_path, never)
    with pytest.raises(ValueError, match='audit'):
        mine([pool, PoolItem(items[2],pool.audio,16000)], model=model, **mining_lock(split))


def test_mining_hash_lock_and_review_caps(tmp_path):
    split,pool,_=pool_fixture(tmp_path)
    model=model_fixture(tmp_path)
    candidates=mine([pool],model=model,**mining_lock(split), max_per_item=2)
    assert len(candidates)==2
    assert abs(candidates[0].start_sample-candidates[1].start_sample)>=16000
    assert reviewed_bank(candidates,{},**mining_lock(split),base_training_weight=10)==()
    review={c.review_id:dict(decision='confirmed_negative',reviewer='human',notes='listened, no target') for c in candidates}
    bank=reviewed_bank(candidates,review,**mining_lock(split),base_training_weight=.1)
    assert sum(r['weight'] for r in bank)<=.025+1e-12
    assert sum(r['weight'] for r in bank)<=.25
    assert all(r['origin_split']=='development' for r in bank)
    folder=tmp_path/'listen'
    export_review(candidates,[pool],folder)
    assert len(list(folder.glob('*.wav')))==2
    assert all(row['decision']=='pending' for row in json.loads((folder/'reviews.json').read_text()))
    model.checkpoint.write_text('changed model')
    with pytest.raises(ValueError,match='changed'):
        mine([pool],model=model,**mining_lock(split))


def test_content_change_and_model_mutation_during_mining_rejected(tmp_path):
    split,pool,_=pool_fixture(tmp_path)
    model=model_fixture(tmp_path)
    with pytest.raises(ValueError,match='content'):
        mine([replace(pool,audio=pool.audio*.5)],model=model,**mining_lock(split))
    def mutate(x):
        model.checkpoint.write_text('mutated')
        return [.9,.2,.1]
    model.scorer=mutate
    with pytest.raises(ValueError,match='changed'):
        mine([pool],model=model,**mining_lock(split))


def test_symmetry_grouped_probe_and_positive_control():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(16,8))
    scores=np.full((16,3),.2)
    report=symmetry(scores,scores,x,x,np.arange(16),seed=1701,bootstrap_replicates=100)
    assert report['structural_result']=='pass'
    assert report['auc']==.5 and report['auc_ci95']==[.5,.5]
    assert report['wp3_b_result']=='inconclusive'
    shifted=symmetry(scores,scores,x,x+15,np.arange(16),seed=1701,bootstrap_replicates=100)
    assert shifted['structural_result']=='fail' and shifted['auc']>.9


def test_training_comparison_harness_remains_pending():
    split,_,_=pool_fixture()
    plan=training_comparison_plan(split_sha256=split.sha256)
    assert plan['status']=='pending'
    assert plan['audit']['minimum_families']==8 and plan['audit']['minimum_instances_per_family']==100
    assert plan['decision']['recall_absolute_drop_max']==.05


def test_training_schedule_keeps_audit_out_of_training(tmp_path):
    from sensorsim.negatives.acceptance import run_training_comparison
    items=[Item('train','train','train','a'*64)]
    for family in range(8):
        for source in range(100):
            key=f'audit-{family}-{source}'
            items.append(Item(key,f'family-{family}',key,hashlib.sha256(key.encode()).hexdigest()))
    split=SplitPlan.create(items,audit_families=[f'family-{i}' for i in range(8)],development_families=[],seed=1701)
    split.save(tmp_path/'study.json')
    split=SplitPlan.load(tmp_path/'study.json',expected_sha256=split.sha256)
    seen=[]
    def train(**kwargs):
        assert all(item.family=='train' for item in kwargs['training_items'])
        assert kwargs['development_items']==()
        seen.append(kwargs)
        return (kwargs['arm'],kwargs['seed'])
    def score(**kwargs):
        assert len(kwargs['audit_items'])==800
        return {'fixture_only':True}
    result=run_training_comparison(split=split,expected_split_sha256=split.sha256,train_arm=train,
                                   score_arm=score,steps=5,examples=100)
    assert len(seen)==6 and len(result['runs'])==6
    assert {(r['steps'],r['examples']) for r in seen}=={(5,100)}
    assert result['status']=='inconclusive'


def test_changed_candidate_is_not_admitted(tmp_path):
    split,pool,_=pool_fixture(tmp_path)
    c=mine([pool],model=model_fixture(tmp_path),**mining_lock(split))[0]
    with pytest.raises(ValueError,match='identity'):
        reviewed_bank([replace(c,start_sample=c.start_sample+1)],{},**mining_lock(split),base_training_weight=1)


def test_repeated_copies_do_not_satisfy_independent_audit_budget(tmp_path):
    from sensorsim.negatives.acceptance import run_training_comparison
    items=[Item('train','train','train','a'*64)]
    for family in range(8):
        for source in range(100):
            key=f'audit-{family}-{source}'
            items.append(Item(key,f'family-{family}',key,hashlib.sha256(f'family-{family}'.encode()).hexdigest()))
    split=SplitPlan.create(items,audit_families=[f'family-{i}' for i in range(8)],development_families=[],seed=1701)
    split.save(tmp_path/'study.json')
    split=SplitPlan.load(tmp_path/'study.json',expected_sha256=split.sha256)
    def never(**kwargs): raise AssertionError('must reject before training')
    with pytest.raises(ValueError,match='independent'):
        run_training_comparison(split=split,expected_split_sha256=split.sha256,train_arm=never,
                                score_arm=never,steps=5,examples=100)


def mining_lock(plan):
    # The study pins these arguments at registration, before mining any scores.
    path = Path(plan._lock_path)
    return dict(split=path, expected_split_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
