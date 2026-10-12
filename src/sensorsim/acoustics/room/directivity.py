"""Measured angular *pressure magnitudes*, with an explicit device frame.

No phase/delay or common device colour is applied by this octave renderer.
Tables are normalized by their measured +x response AFTER power integration.
Missing support is represented by None in JSON and NaN at query time; synthesis
rejects it. BRAS derived data retain CC-BY-SA-4.0 independently of code licensing.
"""
from __future__ import annotations

import csv
from functools import cached_property
import hashlib
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from scipy.spatial import cKDTree, SphericalVoronoi

from sensorsim.licensing import lineage_metadata
from sensorsim.core.messages import ModelValidity
from sensorsim.core.serialization import canonical_json

from sensorsim.acoustics.bands import OCTAVE_NOMINAL_125_8000_HZ

BANDS = OCTAVE_NOMINAL_125_8000_HZ
VALIDITY = ModelValidity(name="Measured octave angular pressure directivity",
    status="partially validated", freq_range_hz=(125.,8000.),
    conditions="On-axis-normalized far-field magnitude only; explicit fixed orientation. "
    "Power integration and analytic coordinate controls; no waveform phase, phone or "
    "cross-device qualification. Missing angular/band support must not be extrapolated.",
    citation="AES69-2022 FreeFieldDirectivityTF; BRAS v3 doi:10.14279/depositonce-6726.3")


class Orientation(BaseModel):
    """Degrees: yaw +x toward +y, pitch toward +z, roll about local +x.

    Device axes: +x front, +y left, +z up. Rotation Rz(yaw) Ry(-pitch) Rx(roll).
    The receiver query points FROM the receiver toward the arriving sound.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    yaw_deg: float = 0.
    pitch_deg: float = 0.
    roll_deg: float = 0.

    @property
    def matrix(self):
        y,p,r=np.deg2rad([self.yaw_deg,self.pitch_deg,self.roll_deg])
        cy,sy,cp,sp,cr,sr=np.cos(y),np.sin(y),np.cos(p),np.sin(p),np.cos(r),np.sin(r)
        return (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]]) @
                np.array([[cp,0,-sp],[0,1.,0],[sp,0,cp]]) @
                np.array([[1.,0,0],[0,cr,-sr],[0,sr,cr]]))

    def to_local(self, directions):
        return np.asarray(directions,float) @ self.matrix


class DirectivityProfile(BaseModel):
    """Portable normalized seven-octave balloon; no filesystem state at rendering.

    Three-neighbour inverse-chord-distance **power** interpolation; exact at a
    measured direction. Angular distance beyond max_angle_deg is unsupported.
    solid_angle_sr weights integrate directional power, never arithmetic dB.
    reference_gain documents the removed common response; it is NOT applied.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal['room-directivity/1'] = 'room-directivity/1'
    name: str
    directions_xyz: tuple[tuple[float,float,float], ...]
    pressure_gain: tuple[tuple[float | None, ...], ...]
    solid_angle_sr: tuple[float, ...]
    reference_gain: tuple[float | None, ...]
    max_angle_deg: float = Field(default=95.,gt=0.,le=180.)
    normalization: Literal['on-axis-pressure/1'] = 'on-axis-pressure/1'
    provenance: dict[str,JsonValue] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def _lineage(cls, data):
        if isinstance(data, dict):
            lineage = lineage_metadata(data)
            if lineage:
                data = {**data, 'provenance': {**data.get('provenance', {}), 'lineage': lineage}}
        return data

    @model_validator(mode='after')
    def _table(self):
        d=np.asarray(self.directions_xyz); g=np.asarray(self.pressure_gain,float)
        w=np.asarray(self.solid_angle_sr)
        if not 4 <= len(d) <= 100_000 or g.shape != (len(d),7) or w.shape != (len(d),):
            raise ValueError('directivity needs 4..100000 directions and seven octave columns')
        if not np.allclose(np.linalg.norm(d,axis=1),1.,rtol=0,atol=1e-8):
            raise ValueError('directivity directions must be unit vectors')
        if len(np.unique(np.round(d,10),axis=0)) != len(d):
            raise ValueError('duplicate directivity directions')
        if np.any(g[np.isfinite(g)]<0) or len(self.reference_gain)!=7:
            raise ValueError('pressure magnitudes must be nonnegative and reference has seven bands')
        if any(x is not None and x<=0 for x in self.reference_gain):
            raise ValueError('supported on-axis reference must be positive')
        frontal=np.argmin(np.linalg.norm(d-[1.,0,0],axis=1))
        supported=np.isfinite(np.asarray(self.reference_gain,float))
        if (np.linalg.norm(d[frontal]-[1.,0,0])>1e-6 or
                not np.allclose(g[frontal,supported],1.,rtol=0,atol=1e-10) or
                np.isfinite(g[:,~supported]).any()):
            raise ValueError('on-axis gain must be unity; unsupported reference bands must stay missing')
        if np.any(w<=0) or not np.isclose(w.sum(),4*np.pi,rtol=1e-8):
            raise ValueError('positive solid angles must sum to 4*pi')
        canonical_json(self.provenance)
        return self

    @cached_property
    def _tree(self):
        return cKDTree(self.directions_xyz)

    @cached_property
    def _power(self):
        return np.asarray(self.pressure_gain,float)**2

    def gain(self, local_directions):
        d=np.asarray(local_directions,float)
        if d.ndim!=2 or d.shape[1]!=3 or not np.isfinite(d).all() or np.any(np.linalg.norm(d,axis=1)==0):
            raise ValueError('directivity queries must be finite nonzero (N,3) vectors')
        d=d/np.linalg.norm(d,axis=1)[:,None]
        distance,index=self._tree.query(d,k=3,workers=1)
        weights=1/np.maximum(distance,1e-15)**2
        weights/=weights.sum(axis=1,keepdims=True)
        result=np.sqrt(np.sum(self._power[index]*weights[:,:,None],axis=1))
        exact=distance[:,0]<1e-10
        result[exact]=np.sqrt(self._power[index[exact,0]])
        result[distance[:,0]>2*np.sin(np.deg2rad(self.max_angle_deg)/2)]=np.nan
        return result

    @property
    def diffuse_power(self):
        return np.asarray(self.solid_angle_sr) @ self._power/(4*np.pi)


class Directivity(BaseModel):
    """Legacy analytic pattern or a declared balloon plus its full orientation.

    Legacy fields/serialization are unchanged. gain returns (N,) for analytic
    patterns, (N,7) for measured magnitudes. Use octave_gain for uniform shape.
    """
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)
    pattern: Literal['omni','cardioid','measured']='omni'
    axis: tuple[float,float,float]=(1.,0.,0.)
    profile: DirectivityProfile | None=Field(default=None,exclude_if=lambda v:v is None)
    orientation: Orientation | None=Field(default=None,exclude_if=lambda v:v is None)

    @model_validator(mode='after')
    def _valid(self):
        if np.linalg.norm(self.axis)==0: raise ValueError('directivity axis must be nonzero')
        if self.pattern=='measured':
            if self.profile is None or self.orientation is None:
                raise ValueError('measured directivity requires a profile and explicit orientation')
            if self.axis!=(1.,0.,0.): raise ValueError('measured profiles use orientation, not analytic axis')
        elif self.profile is not None or self.orientation is not None:
            raise ValueError('profile/orientation require pattern=measured; analytic patterns use axis')
        return self

    def gain(self,directions):
        if self.pattern=='measured':
            return self.profile.gain(self.orientation.to_local(directions))
        if self.pattern=='omni': return np.ones(len(directions))
        axis=np.asarray(self.axis)/np.linalg.norm(self.axis)
        return np.clip(.5*(1+directions @ axis),0.,1.)

    def octave_gain(self,directions):
        g=self.gain(directions)
        if not np.isfinite(g).all(): raise ValueError('unsupported measured directivity direction/band')
        return np.broadcast_to(g[:,None],(len(g),7)) if g.ndim==1 else g

    @property
    def diffuse_power(self):
        if self.pattern=='measured':
            p=self.profile.diffuse_power
            if not np.isfinite(p).all(): raise ValueError('unsupported measured diffuse directivity bands')
            return p
        return 1. if self.pattern=='omni' else 1/3


def _spherical(azimuth,elevation):
    a,e=np.deg2rad(azimuth),np.deg2rad(elevation)
    return np.stack([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)],axis=-1)


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _profile(directions,frequency,pressure,*,name,provenance,max_angle_deg=95.):
    """Integrate piecewise-constant spectral POWER by frequency-cell bandwidth.

    Samples describe logarithmic cells with geometric-midpoint edges. A NaN in
    any overlapping cell makes that octave unsupported. No dB/phase averaging.
    """
    f=np.asarray(frequency,float);d=np.asarray(directions,float);h=np.asarray(pressure,float)
    if f.ndim!=1 or len(f)<2 or not np.isfinite(f).all() or f[0]<=0 or np.any(np.diff(f)<=0):
        raise ValueError('positive increasing spectral frequencies required')
    if h.shape!=(len(d),len(f)): raise ValueError('spectral table shape does not match directions/frequencies')
    if np.any(np.isinf(h)) or np.any(h[np.isfinite(h)]<0): raise ValueError('invalid pressure magnitudes')
    d=d/np.linalg.norm(d,axis=1)[:,None]
    middle=np.sqrt(f[:-1]*f[1:]);edges=np.r_[f[0]**2/middle[0],middle,f[-1]**2/middle[-1]]
    values=[]
    for b in BANDS:
        low,high=b/np.sqrt(2),b*np.sqrt(2)
        width=np.maximum(0,np.minimum(edges[1:],high)-np.maximum(edges[:-1],low))
        use=width>0
        values.append(np.sqrt(np.sum(h[:,use]**2*width[use],axis=1)/(high-low))
                      if edges[0]<=low and edges[-1]>=high else np.full(len(d),np.nan))
    magnitude=np.array(values).T
    frontal=np.argmin(np.linalg.norm(d-np.array([1.,0,0]),axis=1))
    if np.linalg.norm(d[frontal]-[1,0,0])>1e-6: raise ValueError('measured on-axis (+x) reference required')
    reference=magnitude[frontal].copy();reference[reference<=0]=np.nan
    normalized=magnitude/reference
    weights=SphericalVoronoi(d).calculate_areas()
    def portable(x): return [float(v) if np.isfinite(v) else None for v in x]
    return DirectivityProfile(name=name,directions_xyz=d.tolist(),pressure_gain=[portable(x) for x in normalized],
        reference_gain=portable(reference),solid_angle_sr=weights.tolist(),max_angle_deg=max_angle_deg,
        provenance={**provenance,'spectral_reduction':'geometric-frequency-cell power integral/1',
                    'native_frequency_hz':f.tolist(),
                    'native_unsupported_directions_per_frequency':np.sum(~np.isfinite(h),axis=0).tolist(),
                    'phase':'not retained; magnitude-only octave renderer',
                    'common_response':'reference_gain removed once; never applied by synthesis'})


def load_bras_genelec(archive: str | Path, *, angular_step_deg: int=5) -> DirectivityProfile:
    """Stream the local 116 MB MPS member; never expand the 2.8 GB IR CSV.

    BRAS front pole: x=cos(theta), y=sin(theta)sin(phi), z=sin(theta)cos(phi).
    Default 5-degree sampling is opt-in approximation of the supplied 1-degree
    interpolated grid (original measurements 2 degrees). Data below 200 Hz were
    made omni by the database authors. No transfer to a different speaker/phone.
    """
    if not isinstance(angular_step_deg,int) or angular_step_deg<1 or 180%angular_step_deg:
        raise ValueError('angular step must be a positive integer divisor of 180')
    directions,pressure=[],[];digest=hashlib.sha256()
    seen={};duplicates=0;duplicate_delta=0.
    with ZipFile(archive) as z:
        members=[n for n in z.namelist() if '_MPS_' in n and n.endswith('.csv')]
        if len(members)!=1: raise ValueError('expected one BRAS MPS member')
        member=members[0]
        with z.open(member) as stream:
            header=stream.readline();digest.update(header)
            frequency=np.array([float(v) for v in header.decode().strip().split(',')[1:] if v.strip()])
            for line in stream:
                digest.update(line)
                if not line.strip(): continue
                key,values=line.split(b',',1)
                key=key.decode().strip()
                # The published archive repeats the entire table. Verify every
                # repeated row, including unsampled directions; never average
                # conflicting measurements or double-weight the angular grid.
                # Its decimal exports differ by at most the last printed place:
                # allow two 1e-7 complex-component rounding units, record delta.
                if key=='f in Hz':
                    if line.strip()!=header.strip(): raise ValueError('BRAS repeated frequency header changed')
                    continue
                if len(key)!=8 or key[0]!='P' or key[4]!='T': raise ValueError('invalid BRAS front-pole row')
                row_values=np.array([complex(v.replace(' ','').replace('i','j')) for v in
                                     values.decode().strip().split(',') if v.strip()])
                if key in seen:
                    if not np.allclose(row_values,seen[key],rtol=1e-12,atol=2e-7,equal_nan=True):
                        raise ValueError(f'conflicting BRAS duplicate row {key}')
                    finite=np.isfinite(row_values)&np.isfinite(seen[key])
                    duplicate_delta=max(duplicate_delta,float(np.max(abs(row_values[finite]-seen[key][finite]),initial=0.)))
                    duplicates+=1
                    continue
                seen[key]=row_values
                phi,theta=int(key[1:4]),int(key[5:8])
                if theta%angular_step_deg or phi%angular_step_deg or (theta in (0,180) and phi): continue
                p,t=np.deg2rad([phi,theta])
                directions.append([np.cos(t),np.sin(t)*np.sin(p),np.sin(t)*np.cos(p)])
                pressure.append(abs(row_values))
    return _profile(directions,frequency,pressure,name='BRAS Genelec 8020c',
        max_angle_deg=min(95.,2*angular_step_deg),provenance={
            'source':'https://doi.org/10.14279/depositonce-6726.3','licence':'CC-BY-SA-4.0',
            'lineage':{'bras_derived':True,'internal_validation_only':True,'restricted_assets':['bras-v3']},
            'attribution':'Aspoeck et al., BRAS v3','member':member,'member_sha256':digest.hexdigest(),
            'angular_step_deg':angular_step_deg,'coordinates':'BRAS front-pole; +x front +y left +z up',
            'duplicate_rows':duplicates,'duplicate_max_complex_delta':duplicate_delta,
            'duplicate_policy':'retain first; require agreement atol=2e-7 rtol=1e-12 (decimal export precision)',
            'below_200_hz':'database-imposed omnidirectionality'})


def load_balloon_csv(path: str | Path, *, name: str, provenance: dict | None=None) -> DirectivityProfile:
    """Long-form CSV: azimuth_deg,elevation_deg,frequency_hz,pressure_gain.

    Degrees use +x front, azimuth toward +y and elevation toward +z. Gains are
    linear pressure magnitudes at a common radius. Empty/NaN gains stay missing.
    The same loader is suitable for a declared source or receiver measurement.
    """
    rows={};freq=set()
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            key=(float(row['azimuth_deg']),float(row['elevation_deg']))
            b=float(row['frequency_hz']);freq.add(b)
            if b in rows.setdefault(key,{}): raise ValueError('duplicate balloon direction/frequency')
            rows[key][b]=float(row['pressure_gain']) if row['pressure_gain'].strip() else np.nan
    freq=sorted(freq)
    return _profile([_spherical(*k) for k in rows],freq,
        [[v.get(f,np.nan) for f in freq] for v in rows.values()],name=name,
        provenance={**(provenance or {}),'file_sha256':_file_hash(path),
                    'lineage':lineage_metadata(provenance or {}, {'file':str(Path(path)),'name':name})})


def load_sofa(path: str | Path, *, measurement_index: int | None=None) -> DirectivityProfile:
    """FreeFieldDirectivityTF 1.0/1.1, one selected source condition, all receivers.

    Raw room IRs/HRIRs are intentionally rejected. A receiver balloon can use the
    CSV API; HRIR selection needs an explicit separate calibration convention.
    TF phase is discarded with provenance, never advertised as a coherent filter.
    Moving emitter offsets or unequal radii require preprocessing, not guessing.
    """
    from netCDF4 import Dataset
    with Dataset(path) as d:
        if (getattr(d,'SOFAConventions','')!='FreeFieldDirectivityTF' or
                getattr(d,'SOFAConventionsVersion','') not in ('1.0','1.1') or getattr(d,'DataType','')!='TF'):
            raise ValueError('unsupported SOFA directivity convention/version; require FreeFieldDirectivityTF 1.0/1.1')
        if d['Data.Real'].dimensions!=('M','R','N') or d['Data.Imag'].dimensions!=('M','R','N'):
            raise ValueError('SOFA TF must use M,R,N axes')
        count=d['Data.Real'].shape[0]
        if measurement_index is None and count!=1: raise ValueError('select one SOFA measurement_index explicitly')
        index=0 if measurement_index is None else measurement_index
        if not isinstance(index,int) or not 0<=index<count: raise ValueError('invalid SOFA measurement_index')
        def coordinates(name):
            v=d[name];a=np.asarray(v[:],float)
            dims=list(v.dimensions)
            if 'M' in dims:
                axis=dims.index('M');a=np.take(a,index,axis=axis);dims.pop(axis)
            a=np.moveaxis(a,dims.index('C'),-1).reshape(-1,3)
            typ=getattr(v,'Type','cartesian');units=getattr(v,'Units','metre').lower().replace(' ','')
            if typ=='spherical' and units in ('degree,degree,metre','degree,degree,meter'):
                return _spherical(a[:,0],a[:,1])*a[:,2,None]
            if typ=='cartesian' and units in ('metre','meter'): return a
            raise ValueError(f'unsupported SOFA coordinate type/units for {name}')
        def frame(prefix):
            x=coordinates(prefix+'View')[0];z=coordinates(prefix+'Up')[0]
            x=x/np.linalg.norm(x);z=z/np.linalg.norm(z)
            if not np.isfinite([x,z]).all() or abs(x@z)>1e-6: raise ValueError('SOFA view/up must be orthogonal')
            return np.stack([x,np.cross(z,x),z],axis=1)
        if np.any(coordinates('EmitterPosition')): raise ValueError('noncoincident SOFA emitter needs preprocessing')
        listener=coordinates('ListenerPosition')[0];source=coordinates('SourcePosition')[0]
        receivers=coordinates('ReceiverPosition') @ frame('Listener').T+listener
        directions=(receivers-source) @ frame('Source')
        radius=np.linalg.norm(directions,axis=1)
        if np.any(radius<=0) or not np.allclose(radius,radius[0],rtol=1e-5):
            raise ValueError('SOFA directivity requires one common nonzero measurement radius')
        if getattr(d['N'],'Units','').lower()!='hertz': raise ValueError('SOFA frequencies must be hertz')
        if d['Data.Real'].shape[1]*d['Data.Real'].shape[2]>10_000_000:
            raise ValueError('SOFA selected transfer exceeds ten-million-cell budget; preprocess it')
        h=np.ma.filled(d['Data.Real'][index],np.nan)
        imag=np.ma.filled(d['Data.Imag'][index],np.nan)
        frequency=np.asarray(d['N'][:]);positive=frequency>0
        if np.any(frequency<0): raise ValueError('SOFA frequencies cannot be negative')
        return _profile(directions,frequency[positive],np.hypot(h,imag)[:,positive],name=getattr(d,'Title',Path(path).stem),
            provenance={'convention':'FreeFieldDirectivityTF','measurement_index':index,
                        'file_sha256':_file_hash(path),'DC':'ignored for octave integration',
                        'licence':getattr(d,'License',getattr(d,'GLOBAL_License','unspecified')),
                        'lineage':lineage_metadata({'file':str(Path(path))},
                            {k.removeprefix('GLOBAL_'):str(d.getncattr(k)) for k in d.ncattrs()
                             if k.removeprefix('GLOBAL_').lower() in
                             {'title','license','licence','databasename','origin','references','comment'}}),
                        'measurement_radius_m':float(radius[0])})
