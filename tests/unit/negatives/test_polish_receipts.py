"""Relocation and one-byte tamper controls for listening receipts (proxy only)."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import pytest
from sensorsim.negatives.audition import prepare,finalize
from sensorsim.negatives.integration import require_realism
from sensorsim.negatives.splits import Item
from sensorsim.negatives.composer import audio_hash
from sensorsim.recognizer import read_audio
from tests.unit.negatives.test_audition import fixture,decide


def test_finalized_review_moves_and_still_checks_hashes(tmp_path):
    original=tmp_path/'original';original.mkdir()
    folder=original/'review';prepare(fixture(original),folder);decide(folder)
    gate=finalize(folder)['rows'][0]['realism_gate']
    assert not Path(gate['evidence_file']).is_absolute()
    assert not Path(gate['reference_file']).is_absolute()
    moved=tmp_path/'moved';shutil.move(folder,moved);shutil.rmtree(original)
    x=read_audio(moved/'clips/0000-candidate.wav',16000)
    resolved={**gate,**{k:str(moved/gate[k]) for k in ('evidence_file','reference_file')}}
    row=SimpleNamespace(audio=x,fs=16000,item=Item('one','hammering','group',audio_hash(x)),
        provenance=dict(training_eligible=True,family='hammering',provenance='user-owned',realism_gate=resolved))
    require_realism(row)
    assert hashlib.sha256((moved/gate['evidence_file']).read_bytes()).hexdigest()==gate['evidence_sha256']
    path=moved/gate['reference_file'];path.write_bytes(path.read_bytes()+b'X')
    with pytest.raises(ValueError,match='hash|changed|SHA'):require_realism(row)
