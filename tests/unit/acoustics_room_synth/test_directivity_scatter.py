"""I4 opt-in contracts; numerical tolerances frozen before implementation."""
import csv
from zipfile import ZipFile

import numpy as np
import pytest

from sensorsim.acoustics.room.directivity import (
    Directivity, DirectivityProfile, Orientation, load_balloon_csv,
    load_bras_genelec, load_sofa,
)
from sensorsim.acoustics.room.model import BoxObstacle, EarlyScattering, RoomModel, uniform_bands
from sensorsim.acoustics.room.synth import (
    SynthConfig, image_paths, scatter_energy_split, scattering_energy, synthesize,
)


def profile():
    return DirectivityProfile(name="synthetic six-axis balloon",
        directions_xyz=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)),
        pressure_gain=((1.,)*7,(.2,.3,.4,.5,.6,.7,.8),(.4,)*7,(.5,)*7,(.6,)*7,(.7,)*7),
        solid_angle_sr=(4*np.pi/6,)*6, reference_gain=(1.,)*7,
        provenance={"source":"analytic fixture"})


def test_measured_orientation_and_reflected_first_last_legs():
    p = profile()
    d = Directivity(pattern="measured", profile=p, orientation=Orientation())
    m = RoomModel(shoebox_dims_m=(6,5,4), source_directivity=d, receiver_directivity=d)
    paths = image_paths(m, (1,2,2), (2,2,2), max_order=1, air_absorption=False)
    direct = np.flatnonzero(paths.wall_hits.sum(axis=1)==0)[0]
    wall = np.flatnonzero(paths.wall_hits[:,0]==1)[0]
    rear = np.array(p.pressure_gain[1])
    assert paths.directivity_gain[direct] == pytest.approx(rear, abs=1e-12)
    assert paths.directivity_gain[wall] == pytest.approx(rear**2, abs=1e-12)
    turned = d.model_copy(update={"orientation":Orientation(yaw_deg=90)})
    assert turned.gain(np.array([[0.,1.,0.]]))[0] == pytest.approx(np.ones(7), abs=1e-12)
    assert Orientation(pitch_deg=90).to_local([[0.,0.,1.]])[0] == pytest.approx([1,0,0], abs=1e-12)
    assert RoomModel.from_json(m.to_json()).model_hash == m.model_hash
    cfg=SynthConfig(max_order=1,late=False,air_absorption=False)
    h=synthesize(m,(1,2,2),(2,2,2),8000,config=cfg)
    explicit=synthesize(m,(1,2,2),(2,2,2),8000,config=cfg.model_copy(update={"source_directivity":d,"receiver_directivity":d}))
    assert np.array_equal(h.samples,explicit.samples)
    assert "source_directivity" not in RoomModel(shoebox_dims_m=(6,5,4)).model_dump()
    with pytest.raises(ValueError): Directivity(pattern="measured",profile=p)
    with pytest.raises(ValueError): Directivity(pattern="omni",profile=p)


def test_balloon_power_integration_and_missing_support(tmp_path):
    path=tmp_path/'balloon.csv'
    with path.open('w') as f:
        w=csv.writer(f);w.writerow(['azimuth_deg','elevation_deg','frequency_hz','pressure_gain'])
        for az,el,g in [(0,0,2),(180,0,1),(90,0,1),(270,0,1),(0,90,1),(0,-90,1)]:
            for freq in (80,125,250,500,1000,2000,4000,8000,12000): w.writerow([az,el,freq,g])
    p=load_balloon_csv(path,name='fixture')
    assert np.array(p.reference_gain)==pytest.approx(np.full(7,2.))
    assert p.gain(np.array([[-1.,0,0]]))[0]==pytest.approx(np.full(7,.5))
    assert p.diffuse_power==pytest.approx(np.full(7,.375))
    colour=p.model_dump(mode='json');colour['pressure_gain'][0]=[2.]*7
    with pytest.raises(ValueError,match='on-axis'): DirectivityProfile.model_validate(colour)
    broken=p.model_dump(mode='json');broken['pressure_gain'][1]=[None]*7
    bad=DirectivityProfile.model_validate(broken)
    assert np.isnan(bad.gain(np.array([[-1.,0,0]]))).all()
    with pytest.raises(ValueError,match='unsupported'):
        image_paths(RoomModel(shoebox_dims_m=(4,4,4),
            source_directivity=Directivity(pattern='measured',profile=bad,orientation=Orientation())),
            (2,2,2),(1,2,2),max_order=0)


def test_bras_front_pole_streaming_and_sofa_tf(tmp_path):
    archive=tmp_path/'bras.zip'
    freqs=(80,125,250,500,1000,2000,4000,8000,12000)
    text='f in Hz,'+','.join(map(str,freqs))+',\n'
    # Front pole, back pole, +z,+y,-z,-y. Complex magnitudes include colour.
    for key,g in [('P000T000',2),('P000T180',1),('P000T090',1),('P090T090',1),('P180T090',1),('P270T090',1)]:
        text+=key+','+','.join(f'{g} + 0i' for _ in freqs)+',\n'
    # Published MPS repeats its table with last-decimal export-rounding changes.
    text+=text.replace('2 + 0i','2.0000001 + 0i')
    with ZipFile(archive,'w') as z: z.writestr('Genelec_MPS_front_pole.csv',text)
    p=load_bras_genelec(archive,angular_step_deg=90)
    assert p.gain(np.array([[0,1.,0]]))[0]==pytest.approx(np.full(7,.5))
    assert p.provenance['licence']=='CC-BY-SA-4.0'
    assert p.provenance['lineage']['bras_derived'] is True
    assert p.provenance['duplicate_rows']==6
    bad=tmp_path/'conflict.zip'
    with ZipFile(bad,'w') as z:
        z.writestr('Genelec_MPS_front_pole.csv',text+'P000T000,'+','.join('9 + 0i' for _ in freqs)+',\n')
    with pytest.raises(ValueError,match='conflicting'): load_bras_genelec(bad,angular_step_deg=90)
    from netCDF4 import Dataset
    file=tmp_path/'directivity.sofa'
    with Dataset(file,'w') as d:
        d.SOFAConventions='FreeFieldDirectivityTF';d.SOFAConventionsVersion='1.1';d.DataType='TF'
        d.DatabaseName='BRAS proxy fixture';d.GLOBAL_License='CC-BY-SA-4.0'
        for key,size in [('M',1),('R',6),('N',9),('C',3),('I',1)]: d.createDimension(key,size)
        def var(name,dims,data,**attrs):
            v=d.createVariable(name,'f8',dims);v[:]=data
            for k,x in attrs.items(): setattr(v,k,x)
        for prefix in ['Listener','Source']:
            var(prefix+'Position',('I','C'),[[0,0,0]],Type='cartesian',Units='metre')
            var(prefix+'View',('I','C'),[[1,0,0]],Type='cartesian',Units='metre')
            var(prefix+'Up',('I','C'),[[0,0,1]],Type='cartesian',Units='metre')
        var('EmitterPosition',('I','C'),[[0,0,0]],Type='cartesian',Units='metre')
        var('ReceiverPosition',('R','C'),np.array(p.directions_xyz),Type='cartesian',Units='metre')
        var('N',('N',),freqs,Units='hertz')
        var('Data.Real',('M','R','N'),np.array([[2]*9]+[[1]*9]*5)[None])
        var('Data.Imag',('M','R','N'),np.zeros((1,6,9)))
    q=load_sofa(file)
    assert q.provenance['lineage']['bras_derived'] is True
    np.testing.assert_allclose(q.pressure_gain,p.pressure_gain,rtol=1e-12)
    with Dataset(file,'a') as d: d.SOFAConventions='SimpleFreeFieldHRIR'
    with pytest.raises(ValueError,match='convention'): load_sofa(file)


def test_scattering_boundary_energy_and_time_partition():
    incoming=np.arange(1.,8.)
    a=np.linspace(0,1,7);s=np.linspace(1,0,7)
    absorbed,specular,scattered=scatter_energy_split(incoming,a,s)
    np.testing.assert_allclose(absorbed+specular+scattered,incoming,rtol=1e-12)
    assert np.array_equal(scattered[[0,-1]],[incoming[0],0])
    m=RoomModel(shoebox_dims_m=(6,5,4),scattering=uniform_bands(.6))
    cfg=SynthConfig(early_scattering=EarlyScattering(rays=256,max_bounces=6))
    energy,info=scattering_energy(m,(1,2,1),(4,3,1.5),8000,config=cfg,seed=5)
    assert energy.shape[1]==7 and energy.sum()>0 and np.isfinite(energy).all()
    assert info['boundary_closure_max_relative']<1e-12
    direct=np.linalg.norm(np.array([1,2,1])-np.array([4,3,1.5]))/m.sound_speed_m_s
    assert not energy[:int(direct*8000)].any()
    h=synthesize(m,(1,2,1),(4,3,1.5),8000,seed=5,config=cfg)
    assert h.metadata['status']=='unvalidated'
    again=synthesize(m,(1,2,1),(4,3,1.5),8000,seed=5,config=cfg)
    old=synthesize(m,(1,2,1),(4,3,1.5),8000,seed=5)
    assert np.array_equal(h.samples,again.samples)
    assert not np.array_equal(h.samples,old.samples)
    # After the existing crossfade the new early field cannot duplicate late energy.
    k=int(np.ceil((m.mixing_time()+cfg.crossfade_s/2)*8000))+cfg.filter_half_length
    assert np.array_equal(h.samples[k:],old.samples[k:])
    empty=RoomModel(shoebox_dims_m=(6,5,4))
    zero=synthesize(empty,(1,2,1),(4,3,1.5),8000,seed=5,config=cfg)
    assert np.array_equal(zero.samples,synthesize(empty,(1,2,1),(4,3,1.5),8000,seed=5).samples)


def test_diffuse_rain_against_lambert_plane_integral_and_visibility():
    """Independent first-bounce irradiance integral; frozen 1 dB engineering gate."""
    from scipy.integrate import quad
    a,s=uniform_bands(1.),uniform_bands(0.)
    a['z_min']=(.2,)*7;s['z_min']=(1.,)*7
    m=RoomModel(shoebox_dims_m=(40,40,20),absorption=a,scattering=s)
    cfg=SynthConfig(late=False,early_time_s=.05,air_absorption=False,
                    early_scattering=EarlyScattering(rays=4096,max_bounces=1))
    energy,_=scattering_energy(m,(20,20,1),(20,20,2),8000,config=cfg)
    length=.05*m.sound_speed_m_s
    radius=np.sqrt((length**2-9)*(length**2-1)/(4*length**2))
    expected=.8*quad(lambda r:4*r/((r*r+1)**1.5*(r*r+4)**1.5),0,radius)[0]
    assert max(abs(10*np.log10(energy.sum(axis=0)/expected)))<1.
    box=BoxObstacle(name='sealed',min_xyz=(2.9,0,0),size_xyz=(.2,5,4),absorption=(0.,)*7,scattering=(1.,)*7)
    blocked=RoomModel(shoebox_dims_m=(6,5,4),obstacles=(box,),scattering=uniform_bands(1.))
    energy,_=scattering_energy(blocked,(1,2,1),(4,3,1),8000,
        config=cfg.model_copy(update={'early_scattering':EarlyScattering(rays=256,max_bounces=3)}))
    assert not energy.any()
    device=Directivity(pattern='measured',profile=profile(),orientation=Orientation())
    with pytest.raises(ValueError,match='omni'):
        synthesize(blocked.model_copy(update={'source_directivity':device}),(1,2,1),(4,3,1),8000,
            config=SynthConfig(late=False,max_order=0,obstacle_fidelity='complex'))
