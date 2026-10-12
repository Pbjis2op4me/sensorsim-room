from fractions import Fraction
from math import comb
import numpy as np
import pytest
from sensorsim.training_screen import mcnemar,holm,compare


def test_exact_mcnemar_against_independent_enumeration_and_holm():
    for n in range(1,15):
        for k in range(n+1):
            left=np.array([True]*k+[False]*(n-k)); right=~left
            expected=min(Fraction(1),2*sum(Fraction(comb(n,j),2**n) for j in range(min(k,n-k)+1)))
            assert mcnemar(left,right)["p"]==pytest.approx(float(expected))
    tests=holm([dict(p=.04),dict(p=.001),dict(p=.02)])
    assert [r["holm_p"] for r in tests]==pytest.approx([.04,.003,.04])
    assert mcnemar(np.zeros(12,bool),np.zeros(12,bool))["p"]==1
    assert mcnemar(np.ones(16,bool),np.zeros(16,bool))["p"] < .0001
    with pytest.raises(ValueError):mcnemar([],[])
    with pytest.raises(ValueError):mcnemar(np.array([.5]),np.array([1.]))


def test_pairing_uses_parent_identity_and_rejects_repeated_groups():
    left=[dict(source_group="t",kind="target",detected=True),dict(source_group="n",kind="nuisance",trigger=False)]
    assert all(x["p"]==1 for x in compare(left,left[::-1]))
    with pytest.raises(ValueError):compare(left,left+[left[0]])
