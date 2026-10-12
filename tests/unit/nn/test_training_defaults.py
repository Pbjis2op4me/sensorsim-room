"""Small real execution with library admission/selection defaults; proxy only."""
import pytest
from sensorsim import nn, edge
from tests.unit.nn.test_train_end_to_end import inputs, sha

BUDGET=dict(steps=3,width=4,blocks=1,batch=8,workers=0,threads=1,val_windows=64)


def test_nn_defaults_make_pool_and_fixed_detector_with_small_data(tmp_path,monkeypatch):
    manifest,_,_=inputs(tmp_path/'inputs')
    def no_fit(*a,**kw): raise AssertionError('fallback must use the fixed detector')
    monkeypatch.setattr(nn,'_fit_tracker',no_fit)
    out=tmp_path/'model.npz'
    with pytest.warns(UserWarning,match='fixed detector'):
        report=nn.train(manifest,out,**BUDGET)
    m=edge.Model(str(out))
    assert report['negatives']['enabled']
    assert report['selection']['status']=='fixed_fallback'
    assert m.meta['detector']==dict(on={'doorbell':.5},off_ratio=.6,hop_s=.25,smooth_hops=3,min_hops=2)
    assert m.meta['training']['negatives']['automatic']
    assert report['selection']['note']


def test_explicit_infeasible_panel_retains_fallback_and_descriptive_selection(tmp_path):
    manifest,negative,panel=inputs(tmp_path/'inputs')
    out=tmp_path/'model.npz'
    with pytest.warns(UserWarning,match='fixed detector'):
        report=nn.train(manifest,out,**BUDGET,p_confuser_given_background=0,
            selection_panel=str(panel),selection_panel_sha256=sha(panel),selection_min_groups=1,
            selection_grid=[dict(on=.999999,smooth_hops=1,min_hops=1)])
    assert report['selection']['status']=='infeasible'
    assert report['selection']['selected'] is None
    assert report['detector']['on']=={'doorbell':.5}
    assert edge.Model(str(out)).meta['training']['best_step']==3


def test_project_rejects_screen_flag_at_settings_boundary():
    from sensorsim import project
    with pytest.raises(ValueError,match='screen'):
        project._recognizer_settings(dict(recognizer=dict(type='cnn',proxy_screen=True),dataset=dict(fs=16000)))


def test_automatic_panel_uses_heldout_groups_without_inventing_denominators(tmp_path):
    import json
    import numpy as np
    from scipy.io import wavfile
    from sensorsim.negatives.defaults import automatic_panel
    manifest,_,_=inputs(tmp_path/'inputs')
    # Two distinct actual held-out parents; source identity and content both matter.
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    extra=dict(next(r for r in rows if r['label']=='doorbell' and r['split']=='val'))
    extra.update(file='val-extra.wav',source_group='val-extra')
    wavfile.write(manifest.parent/extra['file'],16000,np.random.default_rng(103).normal(0,.001,48000).astype(np.float32))
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows+[extra]))
    def build(folder):
        return automatic_panel(manifest,nn.Corpus(manifest,'train',None,16000),
            nn.Corpus(manifest,'val',None,16000),dict(nn.DEFAULTS,selection_min_groups=2),folder)
    panel,note=build(tmp_path/'panel')
    assert panel is not None
    assert len([r for r in panel['rows'] if r['kind']=='target'])==2
    assert {r['source_group'] for r in panel['rows'] if r['kind']=='target'}=={'valdoorbell','val-extra'}
    extra['source_group']='traindoorbell'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows+[extra]))
    assert build(tmp_path/'leak')[0] is None


def test_screen_training_cannot_reach_device_export(tmp_path,monkeypatch):
    import json
    from sensorsim import project
    from sensorsim.negatives.integration import proxy_screen_receipt
    from sensorsim.negatives.splits import Item
    manifest,negative,panel=inputs(tmp_path/'inputs')
    # The flag marks all artifacts, even this ungated control model.
    out=tmp_path/'model.npz'
    report=nn.train(manifest,out,**BUDGET,negative_manifest=str(negative),negative_manifest_sha256=sha(negative),
                    proxy_screen=True,selection_panel=str(panel),selection_panel_sha256=sha(panel),selection_min_groups=1)
    model=edge.Model(str(out))
    assert model.meta['training']['config']['proxy_screen'] is True
    monkeypatch.setattr(project,'load_project',lambda *a:dict(recognizer=dict(type='cnn')))
    monkeypatch.setattr(project,'_model_file',lambda *a:out)
    with pytest.raises(ValueError,match='proxy_screen_only'):
        project.export(tmp_path,tmp_path/'device')
    assert not (tmp_path/'device').exists()
    with pytest.raises(ValueError,match='proxy_screen_only'):
        nn.export(None,['doorbell'],dict(nn.DEFAULTS),tmp_path/'bad.npz',None,model.meta['training'])


def test_sampling_counts_follow_actual_changed_negative_intermediate(tmp_path,monkeypatch):
    import numpy as np
    from sensorsim.negatives.integration import build_pool
    manifest,negative,_=inputs(tmp_path/'inputs');c=nn.Corpus(manifest,'train',None,16000)
    c.negative_pool=build_pool(negative,sha(negative),fs=16000,max_procedural_share=1.,allowed_room_draws={'room1'})
    seen=[]
    def mix(bg,event,span,rng,cfg):seen.append(bool(np.any(event)));return bg+event
    monkeypatch.setattr(nn,'mix_device_audio',mix)
    cfg=dict(nn.DEFAULTS,batch=12,p_background_only=1.,p_confuser_given_background=1.,_negative_report=True)
    batch=nn.Batches(c,cfg,1,890)[0]
    assert batch[-1]==dict(windows=12,background_windows=12,confuser_windows=12)
    assert all(seen) and not batch[1].any()
    seen.clear();cfg['p_confuser_given_background']=0.
    batch=nn.Batches(c,cfg,1,890)[0]
    assert batch[-1]==dict(windows=12,background_windows=12,confuser_windows=0)
    assert not any(seen)


def test_worker_conversion_preserves_sampling_metadata_without_spawning(tmp_path,monkeypatch):
    import numpy as np
    torch=nn._torch()
    # Exercise the DataLoader conversion branch without sandbox worker sockets.
    class Loader:
        def __init__(self,*a,**kw):assert kw['num_workers']==1
        def __iter__(self):
            yield [torch.zeros((2,1,40,97)),torch.zeros((2,1)),dict(windows=2,background_windows=2,confuser_windows=1)]
    monkeypatch.setattr(torch.utils.data,'DataLoader',Loader)
    monkeypatch.setattr(nn,'Batches',lambda *a:None)
    x,y,stats=next(nn._training_batches(None,dict(nn.DEFAULTS,workers=1),torch))
    assert isinstance(x,np.ndarray) and isinstance(y,np.ndarray)
    assert stats==dict(windows=2,background_windows=2,confuser_windows=1)
