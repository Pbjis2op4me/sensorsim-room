"""Fast stage-2 claims; tolerances frozen in the parent-project synth/obstacle protocols (not distributed)."""
import json

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from sensorsim.acoustics.absorption import amplitude_factor
from sensorsim.acoustics.room.model import BoxObstacle, OCTAVE_HZ, SURFACES, RoomModel, uniform_bands
from sensorsim.acoustics.room.synth import Directivity, SynthConfig, image_paths, synthesize, synthesize_batch
from sensorsim.acoustics.room.augment import (AugmentationConfig, EQBand, additive_noise,
    augment_source, augment_receiver, doppler, pitch_shift, time_stretch)
from sensorsim.acoustics.room.dataset import (DryClip, PositionPair, SamplingSpec,
    generate_dataset, sample_positions, split_positions)

S, R = np.array([1.1, 1.3, 1.2]), np.array([4.2, 3.4, 1.7])


def room(**kwargs):
    return RoomModel(shoebox_dims_m=(6, 5, 3), **kwargs)


def test_model_roundtrip_formula_and_invalid_contract():
    m = room(provenance={'source': 'assumed'}, uncertainty={'alpha': {'bounds': [.1,.3]}})
    assert RoomModel.from_json(m.to_json()).model_hash == m.model_hash
    area, volume, c = 126., 90., m.sound_speed_m_s
    for method, loss in [('sabine', area*.2), ('eyring', -area*np.log(.8))]:
        assert m.t60_estimates(method, air_absorption=False) == pytest.approx(
            np.full(7, 24*np.log(10)*volume/(c*loss)), rel=1e-12)
    assert m.mixing_time() == max(12*volume/area, np.linalg.norm([6,5,3]))/c
    for kwargs in [dict(absorption={'x_min': (.2,)*7}), dict(humidity_pct=float('nan')),
                   dict(late_t60_s=(-1.,)*7), dict(unrecognised=True)]:
        with pytest.raises(ValueError):
            room(**kwargs)


def test_fixed_reference_speed_is_explicit_and_roundtrips():
    """D02: fixed published/calibration timing remains legal without retuning T."""
    m = room(reference_sound_speed_m_s=343.)
    assert m.sound_speed_m_s == 343.
    assert m.sound_speed_policy == 'fixed-reference/1'
    assert RoomModel.from_json(m.to_json()).sound_speed_m_s == 343.
    assert 'reference_sound_speed_m_s' not in room().model_dump(mode='json')


def test_direct_path_fractional_delay_and_air_level():
    m, fs = room(), 32000
    p = image_paths(m, S, R, max_order=0)
    distance = np.linalg.norm(R-S)
    assert p.delay_s[0] == pytest.approx(distance/m.sound_speed_m_s, rel=1e-12)
    exact = amplitude_factor(np.array(OCTAVE_HZ), distance, 20., 50.)/distance
    assert p.gain_octave[0] == pytest.approx(exact, rel=1e-12)
    h = synthesize(m, S, R, fs, config=SynthConfig(max_order=0, late=False))
    f = np.array(OCTAVE_HZ)
    response = np.exp(-2j*np.pi*f[:,None]*np.arange(len(h.samples))/fs) @ h.samples
    expected = exact*np.exp(-2j*np.pi*f*(distance/m.sound_speed_m_s+h.metadata['latency_samples']/fs))
    ratio = response/expected
    assert max(abs(20*np.log10(abs(ratio)))) < .1
    assert max(abs(np.angle(ratio)/(2*np.pi*f/fs))) < .02
    # A delay appears as exp(-i*omega*tau) under the forward DFT: the exp(+i*omega*t) convention.
    assert h.metadata['gain_budget']['phase_convention'] == 'exp(+iwt)'


def test_six_first_reflections_and_cardioid():
    absorption = {s: (0.1+0.1*i,)*7 for i,s in enumerate(SURFACES)}
    m = room(absorption=absorption)
    paths = image_paths(m, S, R, max_order=1, air_absorption=False)
    for axis in range(3):
        for side in range(2):
            image = S.copy()
            image[axis] = 2*side*m.shoebox_dims_m[axis]-S[axis]
            index = np.argmin(np.linalg.norm(paths.images_m-image,axis=1))
            distance = np.linalg.norm(image-R)
            assert paths.delay_s[index] == pytest.approx(distance/m.sound_speed_m_s, abs=1e-12)
            assert paths.gain_octave[index,0] == pytest.approx(np.sqrt(1-absorption[SURFACES[2*axis+side]][0])/distance)
    direction = (R-S)/np.linalg.norm(R-S)
    cardioid = image_paths(m, S, R, max_order=0, source_directivity=Directivity(pattern='cardioid',axis=tuple(-direction)))
    assert abs(cardioid.gain_octave[0]).max() < 1e-15


def test_early_crosscheck_pyroomacoustics():
    import pyroomacoustics as pra
    pra.constants.set('num_threads', 2)
    coefficients = {s: tuple(.08+.03*i+.01*j for j in range(7)) for i,s in enumerate(SURFACES)}
    m = room(absorption=coefficients)
    names = ('west','east','south','north','floor','ceiling')
    materials = {name: pra.Material({'coeffs':coefficients[s], 'center_freqs':OCTAVE_HZ})
                 for name,s in zip(names,SURFACES)}
    ref = pra.ShoeBox(m.shoebox_dims_m, fs=32000, materials=materials, max_order=3)
    ref.set_sound_speed(m.sound_speed_m_s)
    ref.add_source(S); ref.add_microphone(R); ref.image_source_model()
    own = image_paths(m, S, R, max_order=3, air_absorption=False)
    images = ref.sources[0].images.T
    index = np.argmin(np.linalg.norm(own.images_m[:,None,:]-images[None,:,:],axis=-1),axis=1)
    assert images.shape == own.images_m.shape
    assert np.max(abs(own.images_m-images[index])) < 1e-5
    bands = [int(np.flatnonzero(np.isclose(ref.octave_bands.centers, f))[0]) for f in OCTAVE_HZ]
    assert own.wall_pressure == pytest.approx(ref.sources[0].damping[bands][:,index].T, rel=2e-6)
    flat = pra.ShoeBox(m.shoebox_dims_m, fs=32000, materials=pra.Material(.2),max_order=3)
    flat.set_sound_speed(m.sound_speed_m_s); flat.add_source(S); flat.add_microphone(R)
    # PRA 0.10.1 defaults to an extra 10 Hz RIR high-pass, absent from our ISM.
    # Match the physical transfer before comparing; restore its process-global setting.
    hpf = pra.constants.get('rir_hpf_enable')
    try:
        pra.constants.set('rir_hpf_enable', False)
        flat.compute_rir()
    finally:
        pra.constants.set('rir_hpf_enable', hpf)
    h = synthesize(room(),S,R,32000,config=SynthConfig(max_order=3,late=False,air_absorption=False))
    a = h.samples[h.metadata['latency_samples']:]
    b = flat.rir[0][0][pra.constants.get('frac_delay_length')//2:]
    n=max(len(a),len(b));a=np.pad(a,(0,n-len(a)));b=np.pad(b,(0,n-len(b)))
    assert np.linalg.norm(a-b)/np.linalg.norm(b) < .02
    assert abs(np.argmax(abs(a))-np.argmax(abs(b))) <= 1


def test_batch_determinism_empty_obstacles_and_bounds():
    m=room(); cfg=SynthConfig()
    a=synthesize(m,S,R,8000,seed=7,config=cfg)
    b=synthesize(room(obstacles=()),S,R,8000,seed=7,config=cfg)
    assert np.array_equal(a.samples,b.samples)
    assert not np.array_equal(a.samples,synthesize(m,S,R,8000,seed=8).samples)
    batch=synthesize_batch(m,S,[R,R+[.1,0,0]],4000,seed=7,config=SynthConfig(max_order=1,late=False))
    assert len(batch)==2 and batch[0].metadata['seed'] != batch[1].metadata['seed']
    for cfg in [SynthConfig(max_order=1),SynthConfig(duration_s=.01)]:
        with pytest.raises(ValueError): synthesize(m,S,R,8000,config=cfg)
    with pytest.raises(ValueError): synthesize(m,S,S,8000)


def test_prescribed_late_octave_decay():
    target=np.array([.65,.60,.55,.50,.45,.40,.35])
    m=room(late_t60_s=tuple(target),scattering=uniform_bands(.2))
    rir=synthesize(m,S,R,32000,seed=1)
    start=int(np.ceil((m.mixing_time()+.006)*rir.fs))+rir.metadata['latency_samples']
    for centre,expected in zip(OCTAVE_HZ,target):
        sos=signal.butter(4,[centre/np.sqrt(2),centre*np.sqrt(2)],fs=rir.fs,btype='bandpass',output='sos')
        h=signal.sosfilt(sos,rir.samples.astype(float))[start:]
        e=np.cumsum(h[::-1]**2)[::-1]
        db=10*np.log10(np.maximum(e/e[0],1e-300));mask=(db<=-5)&(db>=-25)
        slope=np.polyfit(np.arange(len(h))[mask]/rir.fs,db[mask],1)[0]
        assert -60/slope==pytest.approx(expected,rel=.15)


def test_obstacle_geometry_reflection_and_energy():
    from sensorsim.acoustics.room.obstacles import obstacle_scene, folded_segments, first_reflections, apply_obstacles
    from sensorsim.scene.raycore import RayCore
    box=BoxObstacle(name='box',min_xyz=(2,2,0),size_xyz=(1,1,1.4),absorption=(.8,)*7)
    m=room(obstacles=(box,))
    scene=obstacle_scene(m)
    with RayCore(scene,backend='cpu') as rays:
        assert rays.any_hit([[1,2.5,.7],[1,2.5,2]],[[4,2.5,.7],[4,2.5,2]]).tolist()==[True,False]
        source=np.array([1.,1.,.7]);receiver=np.array([1.,3.,.7])
        _,records=first_reflections(m,source,receiver,rays=rays,scene=scene,config=SynthConfig())
        back=next(r for r in records if r['face']=='back')
        assert back['delay_s']==pytest.approx(np.linalg.norm([2,2,0])/m.sound_speed_m_s,abs=1e-12)
        assert back['point_m']==pytest.approx([2,2,.7],abs=1e-12)
    p=image_paths(m,[1,2.5,.7],[4,2.5,.7],max_order=1)
    ids,a,b=folded_segments(p.images_m,[4,2.5,.7],m.shoebox_dims_m)
    assert np.all(a>=-1e-12) and np.all(b<=np.array(m.shoebox_dims_m)+1e-12)
    assert np.bincount(ids,weights=np.linalg.norm(b-a,axis=1))==pytest.approx(p.distance_m,abs=1e-12)
    with RayCore(scene,backend='cpu') as rays:
        hits=np.bincount(ids,weights=rays.any_hit(a+1e-8*(b-a),b-1e-8*(b-a)),minlength=len(p.distance_m))>0
    # Direct, both x-wall echoes and the floor echo pass through this box;
    # both y-wall echoes and the ceiling echo pass around/above it.
    expected=(p.wall_hits.sum(axis=1)==0) | (p.wall_hits[:,0]>0) | (p.wall_hits[:,1]>0) | (p.wall_hits[:,4]>0)
    assert np.array_equal(hits,expected)
    adjusted,report=apply_obstacles(m,p,[1,2.5,.7],[4,2.5,.7],SynthConfig())
    assert report['blocked_paths']>0
    assert np.all(adjusted.gain_octave[:len(p.distance_m)] <= p.gain_octave+1e-12)
    assert np.all(m.t60_estimates() < room().t60_estimates())
    assert m.volume_m3==pytest.approx(88.6)
    with pytest.raises(ValueError): synthesize(m,[2.5,2.5,.5],R,8000)


def test_thin_obstacle_barrier_closed_form():
    from sensorsim.acoustics.room.obstacles import obstacle_scene, shortest_detour
    from sensorsim.scene.raycore import RayCore
    box=BoxObstacle(name='screen',min_xyz=(2.995,.01,0),size_xyz=(.01,4.98,2))
    m=room(obstacles=(box,)); scene=obstacle_scene(m)
    with RayCore(scene,backend='cpu') as rays:
        route=shortest_detour(scene,rays,np.array([[1,2.5,1]]),np.array([[5,2.5,1]]),0,m.sound_speed_m_s)
    length=2*np.sqrt(1.995**2+1)+.01
    assert route['length_m'][0]==pytest.approx(length,abs=1e-10)
    lam=m.sound_speed_m_s/np.array(OCTAVE_HZ)
    c3=(1+(5*lam/.01)**2)/(1/3+(5*lam/.01)**2)
    x=np.sqrt(2*np.pi*2*(length-4)/lam*c3)
    expected=np.minimum(5+20*np.log10(x/np.tanh(x)),25)
    assert route['attenuation_db'][0]==pytest.approx(expected,abs=.1)


def test_complex_obstacle_hook_preserves_transfer_phase_and_linear_support():
    """Hook guard, not a new BTM oracle: 0.03 relative spectral error / 0.5% energy."""
    from sensorsim.acoustics.room.obstacles import apply_obstacles
    fs, nfft = 8000, 2048
    box=BoxObstacle(name='screen',min_xyz=(2.49,1.,0.),size_xyz=(.02,2.,2.))
    m=room(obstacles=(box,),absorption=uniform_bands(1.))
    source,receiver=np.array([1.,2.,1.]),np.array([4.,2.,1.])
    cfg=SynthConfig(max_order=0,late=False,air_absorption=False,obstacle_fidelity='complex',
                    obstacle_fft_length=nfft)
    p=image_paths(m,source,receiver,max_order=0,air_absorption=False)
    f=np.fft.rfftfreq(nfft,1/fs)
    adjusted,report,spectra=apply_obstacles(m,p,source,receiver,cfg,frequency_hz=f)
    assert len(spectra)==1
    spectrum=spectra[0]
    assert spectrum.replaces_image_indices.tolist()==[0]
    assert report['phase_policy']=='complex-isolated-groups/1'
    assert np.max(abs(spectrum.transfer.imag))>1e-4
    h=synthesize(m,source,receiver,fs,config=cfg)
    actual=np.fft.rfft(h.samples,nfft)
    absolute=spectrum.transfer*np.exp(-2j*np.pi*f*(spectrum.time_origin_s+cfg.filter_half_length/fs))
    band=(f>=250)&(f<=3000)
    assert np.linalg.norm(actual[band]-absolute[band])/np.linalg.norm(absolute[band])<.03
    # Parseval on the explicitly bandlimited real spectrum, no clipped imaginary part.
    weights=np.ones_like(f);weights[1:-1]=2
    assert np.sum(h.samples.astype(float)**2)==pytest.approx(
        np.sum(weights*abs(spectrum.transfer)**2)/nfft,rel=.005)
    assert np.all(h.samples[:max(0,int(spectrum.time_origin_s*fs)-2)]==0)
    assert len(h.samples)>129
    with pytest.raises(ValueError,match='support'):
        synthesize(m,source,receiver,fs,config=cfg.model_copy(update={'obstacle_fft_length':128}))


@pytest.mark.parametrize('size',[(.02,1.,1.4),(1.,1.,1.4)])
def test_complex_obstacle_floor_groups_are_reciprocal(size):
    """Reciprocity of the complete group adapter, rtol=2e-6 float32 export budget."""
    box=BoxObstacle(name='box',min_xyz=(2.,2.,0.),size_xyz=size)
    m=room(obstacles=(box,))
    s,r=np.array([1.,2.4,.7]),np.array([4.,2.6,.9])
    cfg=SynthConfig(max_order=1,late=False,obstacle_fidelity='complex')
    forward=synthesize(m,s,r,8000,config=cfg)
    backward=synthesize(m,r,s,8000,config=cfg)
    np.testing.assert_allclose(forward.samples,backward.samples,rtol=2e-6,atol=2e-8)
    replacements=[i for group in forward.metadata['obstacles']['spectra'] for i in group['replaces_image_indices']]
    assert len(replacements)==len(set(replacements)) and len(replacements)==2
    with pytest.raises(ValueError,match='omni'):
        synthesize(m,s,r,8000,config=cfg.model_copy(update={'source_directivity':Directivity(pattern='cardioid')}))


def test_augment_physical_rates_seed_and_snr():
    fs=8000;t=np.arange(1600)/fs;x=.1*np.sin(2*np.pi*400*t)
    for y in [pitch_shift(x,fs,12),time_stretch(x,fs,2)]:
        assert len(y)==800
        assert np.fft.rfftfreq(len(y),1/fs)[np.argmax(abs(np.fft.rfft(y))) ]==pytest.approx(800,rel=.01)
    y,info=additive_noise(x,fs,((0.,12.),),seed=3,spectrum='pink')
    assert 20*np.log10(np.linalg.norm(x)/np.linalg.norm(y-x))==pytest.approx(12,abs=.02)
    config=AugmentationConfig(gain_db=(-2,2),tilt_db_octave=(-1,1),eq=(EQBand(frequency_hz=700,gain_db=3),),mic_response_std_db=1)
    a,params=augment_source(x,fs,config,seed=5);b,_=augment_source(x,fs,config,seed=5)
    assert np.array_equal(a,b) and params['gain_db'] != 0
    a,_=augment_receiver(a,fs,config,seed=5);assert np.isfinite(a).all()
    delayed=doppler(x,fs,.01,.01)
    assert delayed[:80]==pytest.approx(np.zeros(80),abs=1e-12)
    with pytest.raises(ValueError): additive_noise(np.zeros_like(x),fs,((0.,12.),),seed=1)


def test_dataset_wav_manifest_replay_and_clearance(tmp_path):
    fs=4000; m=room(obstacles=(BoxObstacle(name='desk',min_xyz=(2,2,0),size_xyz=(1,1,1)),))
    spec=SamplingSpec();pairs=sample_positions(m,spec,3,8)
    labels=split_positions(pairs,.34,seed=9)
    sets=[{tuple(p) for pair,label in zip(pairs,labels) if label==split for p in (pair.source_xyz_m,pair.receiver_xyz_m)} for split in ('train','val')]
    assert not sets[0]&sets[1] and all(sets)
    assert all(not m.obstacles[0].contains(p,.1) for pair in pairs for p in (pair.source_xyz_m,pair.receiver_xyz_m))
    x=.01*np.sin(2*np.pi*300*np.arange(80)/fs)
    other=.01*np.sin(2*np.pi*500*np.arange(80)/fs)
    clips=[DryClip('reference',x,fs),DryClip('reference',other,fs)]
    kwargs=dict(model=m,sampling=spec,clips=clips,augmentation=AugmentationConfig(),count=3,seed=4,fs=fs,
                synth_config=SynthConfig(max_order=1,late=False),target_positions=pairs,val_fraction=.34)
    p=generate_dataset(output_dir=tmp_path/'a',**kwargs)
    q=generate_dataset(output_dir=tmp_path/'b',**kwargs)
    assert p.read_bytes()==q.read_bytes()
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    sources=[{row['dry_source']['content_sha256'] for row in rows if row['split']==split} for split in ('train','val')]
    assert all(sources) and not sources[0]&sources[1]
    for row in rows:
        rate,pcm=wavfile.read(p.parent/row['file']);assert rate==fs and pcm.dtype==np.int16
        assert (q.parent/row['file']).read_bytes()==(p.parent/row['file']).read_bytes()
        assert row['model_hash']==m.model_hash and row['rir_metrics']['energy']>=0
        assert len(pcm)==len(x)+row['rir_metrics']['samples']-1
        assert row['export_scale']==1
    first=rows[0]
    h=synthesize(m,first['source_xyz_m'],first['receiver_xyz_m'],fs,
                 seed=first['rir']['seed'],config=kwargs['synth_config'])
    _,pcm=wavfile.read(p.parent/first['file'])
    dry=clips[first['source_catalog_index']].audio
    assert np.max(abs(pcm.astype(float)/32768-np.convolve(dry,h.samples))) <= 1/32768
    assert json.loads((p.parent/'run.json').read_text())['state']=='complete'
    with pytest.raises(FileExistsError): generate_dataset(output_dir=p.parent,**kwargs)
    shared=[PositionPair(source_xyz_m=(1,1,1),receiver_xyz_m=(2,2,2)),PositionPair(source_xyz_m=(2,2,2),receiver_xyz_m=(3,3,2))]
    with pytest.raises(ValueError): split_positions(shared)
