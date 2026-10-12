"""Both-view E1/E2 reports, shared by primary and explicitly diagnostic comparisons."""
import numpy as np

from .c2st import compare_representations, PASS_INTERPRETATION
from .realism import realism_report
from .stream import combine_status


def compare_views(reference, synthetic, *, seed=20261008, permutations=999, bootstrap_replicates=2000,
                  bootstrap_seeds=(20261008, 20261009, 20261010),
                  c2st_permutations=199, c2st_permutation_seed=None):
    reports = {}
    for key, name, bands in [('stats', 'band_statistics', len(reference['band_centres_hz'])),
                              ('embedding', 'pooled_embedding', None)]:
        reports[name] = realism_report(reference[key], synthetic[key],
            reference_groups=reference['groups'], rendered_groups=synthetic['groups'], n_bands=bands,
            seed=seed, permutations=permutations)
    e1 = {'status': combine_status(r['status'] for r in reports.values()), **reports,
          'reference_kind': reference['kind'], 'rendered_kind': 'proxy',
          'reference_groups': len(np.unique(reference['groups'])), 'rendered_groups': len(np.unique(synthetic['groups'])),
          'window_sampling': {'reference': reference['window_sampling'], 'rendered': synthetic['window_sampling']}}
    probes = compare_representations(reference['stats'], synthetic['stats'],
        reference['embedding'], synthetic['embedding'], reference['groups'], synthetic['groups'],
        fold_seed=seed, bootstrap_replicates=bootstrap_replicates, bootstrap_seeds=bootstrap_seeds,
        permutation_replicates=c2st_permutations,
        permutation_seed=seed if c2st_permutation_seed is None else c2st_permutation_seed)
    reps = {name: value.get('result', value) for name, value in probes.items()}
    values = np.asarray(reports['band_statistics']['band_smd']['absolute_band_smd'])
    order = np.argsort(-values, kind='stable')[:5]
    e2 = {'status': combine_status(r.get('outcome', 'inconclusive') for r in reps.values()),
          'reference_kind': reference['kind'], 'rendered_kind': 'proxy', 'representations': reps,
          'top5_bands_by_abs_smd': [{'index': int(i), 'centre_hz': float(reference['band_centres_hz'][i]),
                                   'absolute_smd': float(values[i])} for i in order]}
    e2['pass_interpretation'] = PASS_INTERPRETATION
    if e2['status'] == 'inconclusive' and any(r.get('below_chance_cv_instability') for r in reps.values()):
        e2['reason'] = 'inconclusive: CV instability'
    return e1, e2


GATE_ACCEPTANCE = {
    "status": "accepted on the TRUTH proxy",
    "scope": "20-replicate event-framed 6 s/2 s same-pipeline control; no real-take or deployment validation",
    "preregistration_sha256": "9d9e13513f5b72bbb7171fc2942221dfce8c945d560edd9868834b3d0f6fff7f",
    "tested_revision": "c1b09868a13199b2cfe692b2a56697cb09802e29",
    "measured_validation": False,
}
