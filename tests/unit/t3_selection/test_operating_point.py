"""Development controls. These are not verifier acceptance fixtures."""
import itertools
import numpy as np
import pytest


def policy(**kw):
    return dict(on=.5, off_ratio=.6, smooth_hops=1, min_hops=2, hop_s=.25, **kw)


def replay(x, **kw):
    from sensorsim.evaluation.operating_point import replay_policy
    return replay_policy(x, ['knock'], policy(), window_s=1., sample_rate_hz=100,
                         sample_count=100+(len(x)-1)*25, **kw)


def test_chunk_boundary_flush_and_decision_times():
    from sensorsim.evaluation.operating_point import replay_policy
    x=np.array([[0.], [.9], [.9], [0.], [.9], [.9]])
    whole=replay(x)
    chunked=replay_policy([x[:2], x[2:5], x[5:]], ['knock'], policy(), window_s=1.,
                          sample_rate_hz=100, sample_count=225)
    assert whole == chunked
    assert whole['starts_by_class'] == {'knock': 2}
    assert whole['confirmed_by_class'] == {'knock': 2}
    ends=[e for e in whole['events'] if e['event']=='end']
    assert [e['activation_s'] for e in ends] == [1.25, 2.]
    assert [e['qualification_s'] for e in ends] == [1.5, 2.25]
    assert ends[-1]['flushed'] is True


def test_pre_event_alarm_and_one_output_cannot_match_two_targets():
    from sensorsim.evaluation.operating_point import match_events
    events=replay(np.array([[.9], [.9], [.9], [0.]]))['events']
    late=[dict(label='knock', start_s=1.2, end_s=1.4)]
    assert match_events(events, late, latency_s={'knock': 1.25})['matched']==0
    targets=[dict(label='knock', start_s=.9, end_s=1.1)]*2
    result=match_events(events, targets, latency_s={'knock': 1.25})
    assert result['matched']==1 and not result['all_matched']


def test_one_hop_start_is_counted_even_if_unconfirmed():
    r=replay(np.array([[0.], [.9], [0.]]))
    assert r['starts']==1 and r['confirmed']==0


@pytest.mark.parametrize('x', [np.empty((0,1)), np.array([[np.nan]]), np.array([[1.1]]), np.zeros((2,2))])
def test_bad_score_units_and_empty_arrays_fail(x):
    with pytest.raises(ValueError): replay(x)


def test_constant_scores_are_valid_controls_not_missing_data():
    assert replay(np.zeros((4,1)))['starts']==0
    assert replay(np.ones((4,1)))['confirmed']==1


def test_tied_ap_all_permutations_and_raw_improvement():
    from sensorsim.evaluation.operating_point import average_precision
    for y in set(itertools.permutations([0,0,1,1])):
        assert average_precision(np.ones(4), y)==.5
    assert average_precision([.9,.8,.2,.1], [1,1,0,0])==1.
    assert average_precision([.9,.8,.2], [1,-1,0])==1.
    # A single adjacent inversion near the bottom: strict improvement survives.
    y=np.r_[np.ones(200),0,1]
    v=average_precision(np.arange(len(y))[::-1],y)
    assert .9999 < v < 1.
    assert np.isnan(average_precision([.1,.2],[-1,-1]))


def test_nonmonotone_splitting_is_observed():
    from sensorsim.evaluation.operating_point import replay_policy
    x=np.array([[.95],[.95],[.35],[.95],[.95],[0.]])
    counts=[]
    for on in (.5,.8,.995):
        p=policy(); p['on']=on
        counts.append(replay_policy(x,['knock'],p,window_s=1.,sample_rate_hz=100,
                                   sample_count=225)['confirmed'])
    assert counts==[1,2,0]


def test_missing_tracker_observation_hook_fails_closed(monkeypatch):
    from sensorsim import edge
    class Broken:
        def __init__(self,*a): pass
    monkeypatch.setattr(edge,'Tracker',Broken)
    with pytest.raises(RuntimeError,match='observation hooks'):replay(np.zeros((2,1)))


def test_incomplete_score_extent_and_policy_units_rejected():
    from sensorsim.evaluation.operating_point import replay_policy,threshold_policies
    with pytest.raises(ValueError,match='incomplete'):
        replay_policy(np.zeros((2,1)),['knock'],policy(),window_s=1.,sample_rate_hz=100,sample_count=200)
    p=policy();p['min_hops']=1.5
    with pytest.raises(ValueError,match='integer hops'):
        replay_policy(np.zeros((2,1)),['knock'],p,window_s=1.,sample_rate_hz=100,sample_count=125)
    assert len(threshold_policies(['a','b','c'],policy(),[.5,.6,.7,.8,.9,.95,.99,.995]))==512
    with pytest.raises(ValueError,match='budget'):
        threshold_policies(['a','b','c'],policy(),[.5,.8],max_candidates=7)
