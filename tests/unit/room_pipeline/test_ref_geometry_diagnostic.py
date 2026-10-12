"""Owner-patch regression: no relaxation of REF admission."""
from types import SimpleNamespace
import numpy as np
import pytest
from sensorsim.acoustics.room import pipeline
from sensorsim.acoustics.room.model import RoomModel, OCTAVE_HZ


def test_unresolved_ref_explains_geometry_and_preserves_rejection(tmp_path):
    module=pipeline
    np.save(tmp_path/'h.npy',np.zeros(64))
    bands={str(b):dict(excitation_supported=True,t30=dict(valid=True,rt60_s=.3,r_squared=.99)) for b in OCTAVE_HZ}
    records={(0,0):dict(take=0,is_reference=False,bands=bands,file='h.npy',source_xyz_m=(1,1,1),
        receiver_xyz_m=(2,1,1),metadata={})}
    def unresolved(**kwargs):
        raise ValueError('REF has no resolved direct/reflection separation')
    with pytest.raises(ValueError) as error:
        module._fit(SimpleNamespace(fs=48000,takes=[SimpleNamespace(role='fit')]),records,RoomModel(shoebox_dims_m=(4,3,2.5)),tmp_path,
                    pipeline.PipelineConfig(),unresolved)
    assert 'nearby surface' in str(error.value)
    assert 'omit REF' in str(error.value)
    assert 'effective-source' in str(error.value)
    assert isinstance(error.value.__cause__,ValueError)
