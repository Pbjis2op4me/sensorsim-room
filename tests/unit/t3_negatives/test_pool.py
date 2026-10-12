"""Deterministic development controls for pool identity, eligibility and weights."""
from dataclasses import replace
import json
import numpy as np
import pytest
from sensorsim import training_negatives as tn
from sensorsim.negatives.splits import Item, SplitPlan
from .conftest import sha, document


def save(f):
    document(f.root,'pool.json',f.data)


def test_recorded_control_and_conditional_procedural_refusal(factory):
    f=factory();pool=f.build()
    np.testing.assert_allclose(pool.probabilities,[.25]*4)
    assert pool.metadata()['procedural_cap_basis'].startswith('conditional emitted')
    bad=factory(procedural=True)
    with pytest.raises(ValueError,match='conditional procedural cap'):
        bad.build()


@pytest.mark.parametrize('mutation', ['ambiguous','target','unheard','wrong_parent','wrong_scope','rights'])
def test_semantic_controls(factory,mutation):
    f=factory();assert len(f.build().rows)==4
    ref=f.data['rows'][0]['semantic_review'];review=json.loads((f.root/ref['file']).read_text())
    if mutation=='ambiguous':review['ambiguous']=True
    if mutation=='target':review['target_free']=False
    if mutation=='unheard':review['listened']=False
    if mutation=='wrong_parent':review['parent']['source_group']='another-parent'
    if mutation=='wrong_scope':review['target_scope']='different definition'
    if mutation=='rights':review['rights_cleared']=False
    f.data['rows'][0]['semantic_review']=document(f.root,ref['file'],review);save(f)
    with pytest.raises(ValueError,match='review'):
        f.build()


@pytest.mark.parametrize('licence',['CC-BY-NC-4.0','CC-BY-SA-4.0','unknown','Apache-2.0'])
def test_uncleared_assets_refused(factory,licence):
    f=factory();f.data['rows'][0]['source']['asset']['license_id']=licence;save(f)
    with pytest.raises(ValueError,match='licence|prohibited'):
        f.build()


def test_split_raw_digest_rechecked_and_not_self_renewed(factory):
    f=factory();pool=f.build();pool.choose(np.random.default_rng(0))
    lock=f.root/'split.json';lock.write_bytes(lock.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash'):
        pool.choose(np.random.default_rng(0))
    # Semantically identical reformatting still invalidates raw-file authority.
    with pytest.raises(ValueError,match='hash'):
        f.build()


def test_audit_relabel_and_duplicate_content_fail(factory):
    f=factory();f.data['rows'][0]['item']['source_group']='sealed-group';save(f)
    with pytest.raises(ValueError,match='authorized'):
        f.build()
    f=factory();f.data['rows'].append(f.data['rows'][0]);save(f)
    with pytest.raises(ValueError,match='duplicate'):
        f.build()


def test_empty_missing_hook_units_and_fallbacks(factory):
    f=factory();f.data['rows']=[];save(f)
    with pytest.raises(ValueError,match='nonempty'):
        f.build()
    f=factory();f.data['rows'][0]['positions']={};save(f)
    with pytest.raises(ValueError,match='position'):
        f.build()
    f=factory();f.data['schema']='negative-training/1';save(f)
    with pytest.raises(ValueError,match='negative-training/2'):
        f.build()
    f=factory();f.corpus.negative_pool=None
    with pytest.raises(ValueError,match='strict training pool'):
        tn.prepare_context(f.corpus,np.random.default_rng(1),f.cfg,f.fs)


def test_source_weight_is_consumed_and_cap_is_aggregate(factory):
    f=factory(weight=[1,2,3,4]);pool=f.build(source_cap=.4,family_cap=.4)
    np.testing.assert_allclose(pool.probabilities,[.1,.2,.3,.4])
    # Deterministic inverse-CDF oracle; no binomial acceptance criterion.
    class Draw:
        def __init__(self,x):self.x=x
        def random(self):return self.x
    assert [pool.choose(Draw(x)).item.item_id for x in (.05,.15,.35,.8)] == [f'parent-{i}' for i in range(4)]
    with pytest.raises(ValueError,match='source_group.*cap'):
        f.build(source_cap=.39,family_cap=1.)
    pool.probabilities[0]=.9
    with pytest.raises(ValueError,match='state changed'):
        pool.choose(Draw(.01))


def test_realism_evidence_and_licence_bytes_rechecked(factory):
    f=factory();pool=f.build()
    source=f.data['rows'][0]['source'];p=f.root/source['register']['file']
    p.write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        pool.choose(np.random.default_rng(1))


def test_procedural_share_reweight_preserves_conditional_mass(factory):
    f=factory();g=factory(procedural=True,count=1)
    # Pure numerical independent oracle for cap semantics. No admission bypass.
    rows=list(f.build().rows)
    generated=g.build(max_procedural_share=1.,source_cap=1.,family_cap=1.).rows[0]
    weights,probs=tn._probabilities(rows+[generated],procedural_cap=.1,source_cap=.5,family_cap=.6)
    np.testing.assert_allclose(probs,[.225,.225,.225,.225,.1],atol=1e-14)


def test_invalid_probability_and_domain_are_not_silent(factory):
    f=factory()
    for cfg in (dict(f.cfg,p_confuser_given_target=float('nan')),dict(f.cfg,device_split='stress'),
                dict(f.cfg,proxy_screen=True),dict(f.cfg,negative_context_policy='oops')):
        with pytest.raises(ValueError):tn.validate_config(cfg)


def test_missing_rights_dependency_has_no_fallback(factory,monkeypatch):
    from sensorsim.negatives import clips
    f=factory()
    def unavailable(*args,**kwargs):raise ImportError('development control: licence gate unavailable')
    monkeypatch.setattr(clips,'load_clip',unavailable)
    with pytest.raises(ImportError,match='licence gate unavailable'):
        f.build()


@pytest.mark.parametrize('content',[np.zeros(48000),np.ones(48000)*.01,np.array([])])
def test_empty_and_constant_parent_files_refused(factory,content):
    from scipy.io import wavfile
    from sensorsim.negatives.composer import audio_hash
    from dataclasses import asdict
    f=factory(count=1);entry=f.data['rows'][0];asset=entry['source']['asset']
    wavfile.write(f.root/asset['path'],f.fs,content.astype(np.float64))
    asset['sha256']=sha(f.root/asset['path']);entry['item']['sha256']=audio_hash(content)
    register=entry['source']['register'];data=json.loads((f.root/register['file']).read_text())
    data['assets'][0]['version_or_hash']=asset['sha256']
    entry['source']['register']=document(f.root,register['file'],data)
    review=entry['semantic_review'];data=json.loads((f.root/review['file']).read_text());data['parent']=entry['item']
    entry['semantic_review']=document(f.root,review['file'],data)
    plan=SplitPlan.create([Item(**entry['item']),Item('audit','sealed-family','sealed','e'*64)],
        audit_families=['sealed-family'],development_families=[],seed=31)
    plan.save(f.root/'constant-lock.json');f.data.update(split_lock='constant-lock.json',split_file_sha256=sha(f.root/'constant-lock.json'))
    save(f)
    with pytest.raises(ValueError,match='silent|constant|nonempty'):
        f.build(source_cap=1.,family_cap=1.)


def test_finite_individual_weights_cannot_overflow_sampling_mass(factory):
    f=factory(weight=[1e308]*4)
    with np.errstate(over='ignore'),pytest.raises(ValueError,match='weight sum'):
        f.build()
