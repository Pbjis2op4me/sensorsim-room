"""Drop-folder contract controls, using synthetic development fixtures only."""
import json
import numpy as np
import pytest
import yaml
from scipy.io import wavfile
from sensorsim import project


def survey():
    return dict(units='m',origin='floor corner nearest the door',
        room=dict(length_x=5.,width_y=4.,height_z=2.7),
        microphone=dict(x=3.,y=2.,z=1.2,facing='up'),
        loudspeaker=dict(x=1.,y=1.,z=1.,facing='toward microphone'),notes='proxy')


def test_minimal_geometry_loads_but_target_sites_remain_required(tmp_path):
    from sensorsim.acoustics.room.prior import RoomSurvey
    from sensorsim.acoustics.room.dataset import _anchored_position, SamplingSpec
    from sensorsim.acoustics.room.model import RoomModel
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(survey()))
    cfg=project.load_project(tmp_path)
    room=RoomSurvey.load(cfg['room_yaml'])
    assert not room.source_sites
    with pytest.raises(ValueError, match='source_sites'):
        _anchored_position(room, RoomModel(shoebox_dims_m=room.room.xyz), 'knock', 'train', .2,
                           np.random.default_rng(1), SamplingSpec())
    # Coordinate requirements must never be filled from the loudspeaker silently.
    assert room.loudspeaker.xyz.tolist()==[1.,1.,1.]


def test_session_reference_filename_discovery_reaches_intake(tmp_path,monkeypatch):
    from sensorsim.sources import intake
    root=tmp_path/'dry';root.mkdir()
    wavfile.write(root/'session_reference_2026-10-09.wav',16000,np.ones(16000,dtype=np.int16))
    captured=[]
    monkeypatch.setattr(intake,'load_project_sources',lambda rows,*args: captured.extend(rows) or [])
    data=survey();data['source_sites']={'knock':[dict(x=.08,y=1.2,z=1.2)]}
    (tmp_path/'room.yaml').write_text(yaml.safe_dump(data))
    (tmp_path/'project.yaml').write_text(yaml.safe_dump({'dataset':{'sources':[
        dict(file='knock_01.wav',label='knock',source_group='g1',object_id='door',
             session='2026-10-09',capture_route='near_dry',evidence_label='proxy')]}}))
    cfg=project.load_project(tmp_path);project._clips(cfg,tmp_path/'out')
    assert cfg['targets']=='dry'
    assert captured[0]['session_reference']==dict(file='session_reference_2026-10-09.wav',
        session='2026-10-09',evidence_label='proxy')


def test_discovery_never_substitutes_another_session(tmp_path):
    from sensorsim.acoustics.room.capture_qa import discover_session_reference
    wavfile.write(tmp_path/'session_reference_2026-10-09.wav',16000,np.ones(2000,np.int16))
    assert discover_session_reference(tmp_path,session='2026-10-10',evidence_label='proxy') is None
    with pytest.raises(ValueError,match='session'):
        discover_session_reference(tmp_path,session='../2026-10-09',evidence_label='proxy')


def test_surveyed_class_position_reaches_renderer():
    from sensorsim.acoustics.room.prior import RoomSurvey
    from sensorsim.acoustics.room.dataset import _anchored_position, SamplingSpec
    from sensorsim.acoustics.room.model import RoomModel
    data=survey();data['source_sites']={'knock':[dict(x=.08,y=1.2,z=1.2)]}
    room=RoomSurvey.model_validate(data)
    pair,region=_anchored_position(room,RoomModel(shoebox_dims_m=room.room.xyz),'knock','train',.2,
                                  np.random.default_rng(1),SamplingSpec())
    assert pair.source_xyz_m==(.08,1.2,1.2)
    assert pair.receiver_xyz_m!=pair.source_xyz_m


def test_discovered_reference_is_qualified_and_hashed(tmp_path):
    from sensorsim.acoustics.room.capture_qa import discover_session_reference
    from sensorsim.acoustics.room.ingest import read_recording_audio
    from sensorsim.sources.decay import qualify_session
    from sensorsim.sources.catalog import content_hash
    rng=np.random.default_rng(84);fs=16000;x=rng.normal(0,1e-5,fs*3)
    burst=rng.normal(size=32)*np.hanning(32);x[fs//2:fs//2+32]+=.3*burst/max(abs(burst))
    wavfile.write(tmp_path/'session_reference_2026-10-09.wav',fs,x.astype(np.float32))
    found=discover_session_reference(tmp_path,session='2026-10-09',evidence_label='proxy')
    audio,rate=read_recording_audio(tmp_path/found.pop('file'))
    report=qualify_session(dict(found,samples=audio[:,0],fs=rate),session='2026-10-09')
    assert report['status']=='qualified'
    assert report['content_sha256']==content_hash(audio[:,0],rate)
    bad=qualify_session(dict(found,samples=np.ones(fs*3)*1e-5,fs=rate),session='2026-10-09')
    assert bad['status']=='unavailable'


def test_project_cli_reports_actionable_input_error(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(project,'run',lambda *a,**k: (_ for _ in ()).throw(ValueError('source_sites required')))
    assert project.main(['run',str(tmp_path)])==2
    assert capsys.readouterr().err=='stopped: source_sites required\n'
