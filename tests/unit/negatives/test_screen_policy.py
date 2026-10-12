import pytest
from sensorsim.negatives.screen import scoring_plan, FIXED_DETECTOR


def test_any_infeasible_cell_scores_all_at_terminal_fixed_policy():
    good={'a':dict(selection=dict(status='feasible')),'b':dict(selection=dict(status='feasible'))}
    assert scoring_plan(good,['a','b'],terminal_step=1000)['mode']=='selected'
    good['b']['selection']['status']='infeasible'
    plan=scoring_plan(good,['a','b'],terminal_step=1000)
    assert plan['mode']=='terminal_fixed' and plan['selection_outcomes']==dict(a='feasible',b='infeasible')
    assert plan['detector']==dict(on=.5,off_ratio=.6,smooth_hops=3,min_hops=2,hop_s=.25)
    with pytest.raises(ValueError,match='all registered'):
        scoring_plan(good,['a','b','c'],terminal_step=1000)
    good['b']['selection']['status']='errored'
    with pytest.raises(ValueError,match='unexpected'):
        scoring_plan(good,['a','b'],terminal_step=1000)
