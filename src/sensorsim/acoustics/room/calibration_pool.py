"""Pool observations sharing a speaker/microphone transfer, retaining discrepancy."""
import hashlib
import numpy as np


def pool_playback(observations, *, uncertainty_components=None):
    """One likelihood for one geometry; repeats never shrink model discrepancy.

    This pools metrics, not waveforms: it prevents pseudoreplication but does not
    claim the SNR improvement of coherent signal-level averaging.
    """
    if not observations or not all(o.qualified for o in observations):
        raise ValueError('pool needs qualified observations')
    first = observations[0]
    if any((o.source_xyz_m,o.receiver_xyz_m)!=(first.source_xyz_m,first.receiver_xyz_m) for o in observations):
        raise ValueError('one pool per geometry')
    keys=sorted(set().union(*(set(o.values) for o in observations)))
    values={k:float(np.mean([o.values[k] for o in observations if k in o.values])) for k in keys}
    modular = {}
    if uncertainty_components is not None:
        from .calibration_discrepancy import decompose_uncertainty
        # Explicit units/covariances only. Do not reinterpret the old discrepancy
        # scales as repeat noise. Duplicate content and incomplete keys refuse.
        decomposition = decompose_uncertainty([o.values for o in observations],
            repeat_ids=[o.source_group for o in observations],
            content_sha256=[o.input_sha256 for o in observations],
            **uncertainty_components)
        values.update(decomposition['values'])
        modular = {'modular_uncertainty': decomposition}
    scales={k:max(o.uncertainty[k] for o in observations if k in o.values) for k in keys}
    return first.model_copy(update=dict(source_group='pooled-playback',
        input_sha256=hashlib.sha256(''.join(o.input_sha256 for o in observations).encode()).hexdigest(),
        values=values,uncertainty=scales,metadata={
            'method':'one shared-geometry likelihood; mean metrics, unreduced discrepancy',
            'observations':[o.model_dump(mode='json') for o in observations],
            'independent_geometry_count':1,'signal_averaging':False, **modular}))
