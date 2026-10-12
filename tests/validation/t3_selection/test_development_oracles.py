"""Independent arithmetic/finite-state DEVELOPMENT controls, not acceptance.

No same-code oracle: the reference state machine below uses scalar queues and
counters; exact null probabilities are enumerated with integer combinations.
"""
from collections import deque
import itertools
import math

import numpy as np
import pytest


def reference_counts(values, on, off, smooth, minimum):
    history=deque(maxlen=smooth)
    active=0; starts=0; confirmed=0
    for value in values:
        history.append(float(value)); mean=sum(history)/len(history)
        if active:
            if mean < off:
                confirmed += active >= minimum
                active=0
            else:
                active+=1
        elif mean >= on:
            starts+=1;active=1
    confirmed+=active >= minimum
    return starts,confirmed


def test_tracker_against_independent_scalar_oracle():
    from sensorsim.evaluation.operating_point import replay_policy
    checked=0
    for on,smooth,minimum in itertools.product((.5,.8),(1,3),(1,2,4)):
        p=dict(on=on,off_ratio=.6,smooth_hops=smooth,min_hops=minimum,hop_s=.25)
        for seq in itertools.product((0.,.4,.9),repeat=5):
            scores=np.asarray(seq)[:,None]
            actual=replay_policy([scores[:2],scores[2:]],['knock'],p,window_s=1.,sample_rate_hz=100,sample_count=200)
            assert (actual['starts'],actual['confirmed'])==reference_counts(seq,on,.6*on,smooth,minimum)
            checked+=1
    assert checked==2916


def test_exact_mcnemar_valid_null_and_defect_resolution():
    from sensorsim.evaluation.selection import paired_parent_improvement
    rows=[]
    for n in (4,5,20):
        rejected=[]
        for improved in range(n+1):
            candidate=dict(bank_sha256='development',candidate={'seed':1103},
                nuisance_parent_trigger={i:i >= improved for i in range(n)})
            comparator=dict(candidate,nuisance_parent_trigger={i:i < improved for i in range(n)})
            p=paired_parent_improvement(candidate,comparator)['one_sided_exact_p']
            independent=sum(math.comb(n,k) for k in range(improved,n+1))/2**n
            assert p==pytest.approx(independent)
            if p < .05:rejected.append(improved)
        null=sum(math.comb(n,k) for k in rejected)/2**n
        power=sum(math.comb(n,k)*.8**k*.2**(n-k) for k in rejected)
        assert null <= .05
        rows.append((n,null,power))
    assert rows[0]==(4,0,0)
    assert rows[1][1]==.03125
    assert rows[2][2] > .80
    # This power statement is conditional on 20 independent discordances and
    # P(improvement | discordance)=0.8, not 20 total household source parents.
