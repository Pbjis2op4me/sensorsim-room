# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from tools.dynamics_screen import interval, mcnemar, verify_bank


def test_paired_exact_test_uses_discordances_not_total_windows():
    control=[True]*10+[False]*100
    candidate=[False]*110
    result=mcnemar(control,candidate)
    assert result['discordant']==10
    assert result['p']==pytest.approx(2/1024)
    assert mcnemar([True,False],[True,False])['p']==1.


def test_exact_recall_interval_does_not_claim_high_recall_from_ten_takes():
    lo,hi=interval(10,10)
    assert lo==pytest.approx(.6915028922) and hi==1.
    assert interval(0,10)[0]==0.


def test_screen_refuses_heldout_group_leakage(tmp_path):
    p=tmp_path/'manifest.jsonl';(tmp_path/'source.wav').write_bytes(b'audio')
    p.write_text(json.dumps(dict(source_group='shared',file='source.wav'))+'\n')
    bank=tmp_path/'bank.json';bank.write_text(json.dumps([dict(source_group='shared')]))
    with pytest.raises(ValueError,match='group overlaps'):
        verify_bank(p,bank)
