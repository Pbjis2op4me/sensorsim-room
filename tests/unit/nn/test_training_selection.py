"""Independent score-sequence controls for the complete deployed policy selector."""
import numpy as np
import pytest


def row(kind,label=None,family=None):
    return dict(kind=kind,label=label,family=family,start_s=.5,end_s=2.5,source_group=kind)


def test_joint_selector_prefers_recall_subject_to_total_confuser_budget():
    from sensorsim.training_selection import score_policy, choose_candidate
    classes=["doorbell","kettle","knock"]
    rows=[row("target",c) for c in classes]+[row("nuisance",family="fan"),row("background")]
    sequences=[]
    for c in range(3):
        s=np.zeros((12,3));s[2:6,c]=.9;sequences.append(s)
    bad=np.zeros((12,3));bad[2:6,1]=.7
    sequences += [bad,np.zeros((12,3))]
    outcomes=[]
    for i,on in enumerate((.5,.8,.99)):
        d=dict(on=on,smooth_hops=1,min_hops=1,off_ratio=.6,hop_s=.25)
        m=score_policy(sequences,rows,classes,d,window_s=1.,trigger_budget=0.,min_recall=.5)
        outcomes.append(dict(step=250,policy_index=i,detector=d,**m))
    assert [r["feasible"] for r in outcomes]==[False,True,False]
    assert choose_candidate(outcomes)["detector"]["on"]==.8
    assert outcomes[0]["nuisance_trigger_groups"]==1
    assert outcomes[0]["class_swap_matrix"]["fan"]["kettle"]==1
    assert outcomes[1]["recall"]==dict.fromkeys(classes,1.)
    assert choose_candidate([outcomes[0],outcomes[2]]) is None


def test_constant_high_scores_do_not_pass_by_suppressing_all_recall():
    from sensorsim.training_selection import score_policy
    rows=[row("target","doorbell"),row("nuisance",family="fan"),row("background")]
    m=score_policy([np.ones((10,1))]*3,rows,["doorbell"],
        dict(on=.99,smooth_hops=1,min_hops=1,off_ratio=.6,hop_s=.25),
        window_s=1.,trigger_budget=0.,min_recall=.5)
    assert not m["feasible"] and m["nuisance_trigger_groups"]==1 and m["background_starts"]==1
    for bad in ([],[np.array([])]*3,[np.full((10,1),np.nan)]*3):
        with pytest.raises(ValueError):
            score_policy(bad,rows,["doorbell"],dict(on=.5,hop_s=.25),window_s=1.,trigger_budget=0.,min_recall=.5)


def test_ties_choose_later_checkpoint_without_using_synthetic_ap():
    from sensorsim.training_selection import choose_candidate
    a=dict(feasible=True,recall={"a":.8,"b":.9},nuisance_trigger_groups=0,nuisance_starts=0,step=250,policy_index=0)
    b=dict(a,step=500,val_mean_ap=.1)
    assert choose_candidate([dict(a,val_mean_ap=1.),b])["step"]==500


def test_defaults_require_negatives_and_joint_selection():
    from sensorsim import nn
    assert nn.DEFAULTS["p_confuser_given_background"] > 0
    assert 0 < nn.DEFAULTS["max_procedural_share"] < 1
    assert nn.DEFAULTS["selection_mode"] == "panel"
