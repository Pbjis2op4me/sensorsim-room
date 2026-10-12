"""Procedural loader controls, not acoustic or distributional acceptance.

All geometry and signals below are invented fixtures. No archive data are used.
"""
import hashlib
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings('ignore:Setting the shape on a NumPy array:DeprecationWarning')

def loader():
    return importlib.import_module('tests.validation.dechorate_v1')


@pytest.fixture
def archive(tmp_path):
    nc = pytest.importorskip('netCDF4')
    root = tmp_path/'dechorate/v1'
    root.mkdir(parents=True)
    geometry = root/'load_positions_and_plot.py'
    geometry.write_text('room_size = [6., 7., 3.]\nroom_temperature = 24\n'
        'mics = np.array('+repr([[1.+i*.05 for i in range(30)], [2.]*30, [1.]*30])+')\n'
        'srcs_omn = np.array([[3., 4., 5.], [4., 4., 4.], [1., 1., 1.]])\n'
        'raise RuntimeError("external geometry must never execute")\n')
    path = root/'dechorate_rir_processed.hdf5'
    x = np.zeros(20000)
    x[503], x[509], x[19001] = 1., -.3, .01
    with nc.Dataset(path, 'w') as f:
        for name, size in zip(('sample','mic','source'), (20000,30,7)):
            f.createDimension(name,size)
        for state in loader().STATES:
            v=f.createVariable(state,'f8',('sample','mic','source'), zlib=True)
            for mic in range(1,26):
                v[:,mic-1,6]=x*(1+mic/100.)
    repin(tmp_path)
    return tmp_path, x


def repin(root):
    files=[]
    for p in (root/'dechorate/v1').iterdir():
        b=p.read_bytes()
        files.append({'path':p.relative_to(root).as_posix(), 'licence':'CC-BY-4.0',
            'sha256':hashlib.sha256(b).hexdigest(), 'size_bytes':len(b)})
    (root/'MANIFEST.json').write_text(json.dumps({'files':files}))


def test_required_hdf5_route_exists_on_both_benches():
    from tests.validation.bench_room_closedloop import measured
    from tests.validation.bench_room_recheck import run
    assert callable(getattr(measured.MeasuredEngine, 'dechorate_v1', None))
    assert callable(getattr(run, '_load_v1_pairs', None))


def test_selection_refuses_missing_source_and_disclosed_array_before_open(monkeypatch):
    m=loader()
    opened=[]
    def forbidden(*a,**kw):
        opened.append(a)
        raise AssertionError('dataset opened')
    monkeypatch.setitem(sys.modules,'netCDF4',SimpleNamespace(Dataset=forbidden))
    for sources, arrays in (((7,8),(5,)), ((9,),(6,)), ((7,),(6,)), ((),(5,)), ((7,7),(5,))):
        with pytest.raises(ValueError):
            m.validate_selection(sources,arrays)
    assert opened==[]


def test_hyperslab_index_and_native_timing_are_exact(archive):
    root,x=archive
    a=loader().V1Archive(root)
    got=a.read_pair('000000',7,22)
    # Independent known samples detect wrong source, mic, transpose or onset shift.
    np.testing.assert_array_equal(got,x*1.22)
    assert len(got)==20000 and np.argmax(got)==503
    assert a.fs==48000 and a.temperature_c==24
    assert a.read_log==[{'state':'000000','source':7,'microphone':22,
                        'slice':[[0,20000],21,6]}]
    assert a.metadata()['native_duration_s']==pytest.approx(5/12)


def test_integrity_change_is_not_a_fallback(archive):
    root,_=archive
    with (root/'dechorate/v1/load_positions_and_plot.py').open('a') as f:
        f.write('# changed\n')
    with pytest.raises(ValueError,match='integrity'):
        loader().V1Archive(root)


@pytest.mark.parametrize('kind', ['zero','constant','nan','duplicate','wrong_shape'])
def test_invalid_waveforms_and_schema_refused(archive,kind):
    import netCDF4
    root,x=archive
    with netCDF4.Dataset(root/'dechorate/v1/dechorate_rir_processed.hdf5','a') as f:
        if kind=='wrong_shape':
            f.renameVariable('000000','unexpected')
        else:
            value={'zero':np.zeros(20000),'constant':np.ones(20000),
                   'nan':np.full(20000,np.nan),'duplicate':x*1.22}[kind]
            f['000000'][:,23,6]=value
    repin(root)
    with pytest.raises(ValueError):
        a=loader().V1Archive(root)
        a.read_pair('000000',7,22)
        a.read_pair('000000',7,24)


def test_missing_dependency_is_explicit(archive, monkeypatch):
    root,_=archive
    monkeypatch.setitem(sys.modules,'netCDF4',None)
    with pytest.raises(RuntimeError,match='netCDF4'):
        loader().V1Archive(root)


def test_units_and_unavailable_source_are_not_silently_relabelled(archive):
    root,_=archive
    with pytest.raises(ValueError,match='48000'):
        loader().V1Archive(root,expected_fs=16000)
    a=loader().V1Archive(root)
    for src,mic in ((8,21),(9,26),(7,26),(7,0)):
        with pytest.raises(ValueError):
            a.read_pair('000000',src,mic)
    assert a.read_log==[]


def test_t2_mapping_split_and_geometry_origin(archive):
    from tests.validation.bench_room_closedloop import measured
    from sensorsim.core.geometry import HUMID_SOUND_SPEED_MODEL, sound_speed
    root,x=archive
    e=measured.MeasuredEngine(data=root)
    s=e.dechorate_v1('000000','smoke',sources=(7,))
    assert s.train_pairs==[('S1','M1'),('S1','M2'),('S1','M3')]
    assert s.heldout_pairs==[('S1',f'H{i}') for i in range(22,26)]
    assert s.novel_pairs==[]
    assert s.metadata['native_mapping']['S1:M3']['mic']==21
    h=e.render_rirs(s,[('S1','H22')],48000)[('S1','H22')]
    distance=np.linalg.norm(np.subtract(s.positions['S1'],s.positions['H22']))
    expected=round(distance/sound_speed(HUMID_SOUND_SPEED_MODEL,24,50)*48000)
    assert np.argmax(h)==expected
    assert e.origins['dechorate_000000:S1:H22']['raw_onset_at_output_fs']==503
    with pytest.raises(ValueError,match='source 8'):
        e.dechorate('000000','smoke')


def test_wp8_source8_is_required_not_filtered(archive):
    from tests.validation.bench_room_recheck import run
    root,_=archive
    with pytest.raises(ValueError,match='source 8'):
        run._load_v1_pairs(root,48000,allowed_splits=('fit_absorption','validation_only'))


def test_wp8_hook_preserves_native_samples_and_its_own_split(archive, monkeypatch):
    from tests.validation.bench_room_recheck import run
    root,x=archive
    reached=[]
    def metric(rir,fs,*,onset_sample):
        reached.append((rir.copy(),fs,onset_sample))
        return {'fixture_only':True}
    monkeypatch.setattr(run,'octave_room_metrics',metric)
    pairs,audit=run._load_v1_pairs(root,48000,sources=(7,),states=('000000',))
    assert len(pairs)==25 and len(reached)==25
    assert audit['split_counts']=={'fit_absorption':20,'validation_only':5}
    assert [p.microphone_id for p in pairs if p.split=='validation_only']==[21,22,23,24,25]
    np.testing.assert_array_equal(reached[21][0],x*1.22)
    assert reached[21][1:]==(48000,503)
    assert audit['temperature_c']==24 and audit['humidity_pct']==50
    assert all(row['source']==7 and row['microphone']<=25 for row in audit['read_log'])
