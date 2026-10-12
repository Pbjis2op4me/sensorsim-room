"""Deterministic development controls; fixture declarations are not human reviews."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim.negatives.composer import audio_hash
from sensorsim.negatives.integration import require_realism
from sensorsim.negatives.splits import Item


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(tmp_path, family='hammering', domain='deployment recording'):
    rng = np.random.default_rng(8)
    for name in ('candidate', 'reference'):
        wavfile.write(tmp_path/f'{name}.wav', 16000, (rng.normal(size=16000)*100).astype(np.int16))
    spec = dict(schema='negative-audition-input/1', rows=[dict(
        item_id='one', family=family, file='candidate.wav', reference_file='reference.wav',
        reference_kind='recorded_family', reference_evidence_domain=domain,
        reference_license_id='user-owned')])
    write_json(tmp_path/'input.json', spec)
    return tmp_path/'input.json'


def decide(folder, decision='approve'):
    path=folder/'decisions.json'; data=json.loads(path.read_text())
    data['reviewer']='automated test fixture, not a human review'
    data['rows'][0].update(decision=decision, reviewed_label_boundary=True, notes='development control')
    write_json(path, data)


def test_approved_receipt_reaches_production_gate(tmp_path):
    from sensorsim.negatives.audition import prepare, finalize
    source=fixture(tmp_path); folder=tmp_path/'audition'; prepare(source, folder); decide(folder)
    result=finalize(folder)
    gate=result['rows'][0]['realism_gate']
    gate={**gate, **{k:str(folder/gate[k]) for k in ('evidence_file','reference_file')}}
    from sensorsim.recognizer import read_audio
    x=read_audio(tmp_path/'candidate.wav', 16000)
    row=SimpleNamespace(audio=x,fs=16000,item=Item('one','hammering','group',audio_hash(x)),
        provenance=dict(training_eligible=True,family='hammering',provenance='user-owned',realism_gate=gate))
    require_realism(row)
    assert gate['evidence_sha256']==hashlib.sha256(open(gate['evidence_file'],'rb').read()).hexdigest()
    assert gate['audio_sha256']==audio_hash(x)
    assert result['rows'][0]['decision']=='approve'


@pytest.mark.parametrize('changed',['candidate.wav','reference.wav','clips/0000-candidate.wav','clips/0000-reference.wav'])
def test_changed_audio_refuses_receipt(tmp_path, changed):
    from sensorsim.negatives.audition import prepare, finalize
    folder=tmp_path/'audition'; prepare(fixture(tmp_path), folder); decide(folder)
    path=(folder if changed.startswith('clips/') else tmp_path)/changed
    rate,x=wavfile.read(path); x[0]+=1; wavfile.write(path,rate,x)
    with pytest.raises(ValueError,match='changed|hash'):
        finalize(folder)
    assert not (folder/'receipts.json').exists()


def test_pending_and_reject_do_not_authorize(tmp_path):
    from sensorsim.negatives.audition import prepare, finalize
    folder=tmp_path/'audition'; prepare(fixture(tmp_path),folder)
    with pytest.raises(ValueError,match='pending|reviewer'):
        finalize(folder)
    decide(folder,'reject'); result=finalize(folder)
    assert 'realism_gate' not in result['rows'][0]


@pytest.mark.parametrize('defect',['empty','constant','duplicate','rate','reference_domain','missing_reference','unknown_family'])
def test_invalid_inputs_fail_closed(tmp_path,defect):
    from sensorsim.negatives.audition import prepare
    path=fixture(tmp_path); data=json.loads(path.read_text())
    if defect in ('empty','constant'):
        wavfile.write(tmp_path/'candidate.wav',16000,np.zeros(0 if defect=='empty' else 16000,np.int16))
    elif defect=='duplicate': data['rows'].append(dict(data['rows'][0],item_id='two'))
    elif defect=='rate':
        rate,x=wavfile.read(tmp_path/'reference.wav');wavfile.write(tmp_path/'reference.wav',8000,x)
    elif defect=='reference_domain': data['rows'][0]['reference_evidence_domain']='synthetic proxy'
    elif defect=='missing_reference': data['rows'][0]['reference_file']='absent.wav'
    else: data['rows'][0]['family']='not-a-family'
    write_json(path,data)
    with pytest.raises((ValueError,OSError)):
        prepare(path,tmp_path/'audition')


def test_tonal_listening_does_not_replace_recall_gate(tmp_path):
    from sensorsim.negatives.audition import prepare, finalize
    from sensorsim.recognizer import read_audio
    folder=tmp_path/'audition';prepare(fixture(tmp_path,'appliance_beeps'),folder);decide(folder)
    result=finalize(folder); x=read_audio(tmp_path/'candidate.wav',16000)
    row=SimpleNamespace(audio=x,fs=16000,item=Item('one','appliance_beeps','group',audio_hash(x)),
        provenance=dict(training_eligible=True,family='appliance_beeps',provenance='user-owned',
            realism_gate=result['rows'][0]['realism_gate']))
    gate=row.provenance['realism_gate']
    row.provenance['realism_gate']={**gate, **{k:str(folder/gate[k]) for k in ('evidence_file','reference_file')}}
    with pytest.raises(ValueError,match='doorbell recall'):
        require_realism(row)


def test_manifest_binding_and_no_overwrite(tmp_path):
    from sensorsim.negatives.audition import prepare, finalize
    path=fixture(tmp_path); folder=tmp_path/'audition';prepare(path,folder);decide(folder)
    with pytest.raises(FileExistsError): prepare(path,folder)
    plan=folder/'plan.json';data=json.loads(plan.read_text());data['rows'][0]['family']='door_closure';write_json(plan,data)
    with pytest.raises(ValueError,match='plan.*hash|plan.*changed'):
        finalize(folder)


def test_every_gated_family_has_a_pending_preview(tmp_path):
    from sensorsim.negatives.audition import generate, finalize
    from sensorsim.negatives.procedural import IMPACTS, TONES
    folder=tmp_path/'previews';generate(folder)
    plan=json.loads((folder/'audition/plan.json').read_text())
    assert {r['family'] for r in plan['rows']}==set(IMPACTS+TONES)
    assert all((folder/'audition'/r['candidate']['audition_file']).is_file() for r in plan['rows'])
    choices=json.loads((folder/'audition/decisions.json').read_text())
    choices['reviewer']='automated test fixture'
    for r in choices['rows']: r.update(decision='approve',reviewed_label_boundary=True)
    write_json(folder/'audition/decisions.json',choices)
    with pytest.raises(ValueError,match='recorded-family reference'):
        finalize(folder/'audition')


def test_cli_bad_decision_exits_nonzero_without_receipts(tmp_path):
    from sensorsim.negatives.audition import main
    source=fixture(tmp_path);folder=tmp_path/'audition'
    assert main(['prepare',str(source),str(folder)])==0
    assert main(['finalize',str(folder)])==2
    assert not (folder/'receipts.json').exists()
