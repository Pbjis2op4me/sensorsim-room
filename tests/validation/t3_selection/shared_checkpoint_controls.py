# SPDX-License-Identifier: Apache-2.0
"""Explicit shared-patch integration controls, adapted from verifier F2.

Run this file explicitly on a disposable candidate with the exact handoff patch.
These disclosed fixtures are synthetic proxy software evidence only.
"""
import copy
from hashlib import sha256

import numpy as np
import pytest

from tests.unit.nn.test_nn import manifest
from tests.unit.t3_selection.test_selection import make_bank, constraints


@pytest.mark.parametrize('mutation', [
    'matching', 'wrong_features', 'wrong_class_order', 'tampered_digest',
    'invalid_policy', 'missing_policy_field', 'changed_checkpoint', 'changed_without_selection',
])
def test_callback_binds_emitted_checkpoint(manifest, tmp_path, monkeypatch, mutation):
    from sensorsim import nn, edge
    from sensorsim.evaluation.selection import candidate_identity, cache_scores, score_candidate, select_candidate, digest
    root = tmp_path/'bank'
    root.mkdir()
    bank, _ = make_bank(root, classes=('high', 'low'), fs=16000)
    policy = dict(on=.8, smooth_hops=1, min_hops=2, off_ratio=.6, hop_s=.25)
    seen = []
    def selector(path, step):
        runtime = edge.Model(str(path))
        features = copy.deepcopy(runtime.meta['features'])
        c = candidate_identity(model_sha256=sha256(path.read_bytes()).hexdigest(), features=features,
            classes=runtime.classes, policy=policy, step=step, seed=1103,
            code_revision='disclosed-verifier-F2', model_path=path)
        seq = {}
        for row in bank.rows:
            x = np.zeros((1+(row['sample_count']-16000)//4000, 2))
            if row['kind'] == 'target':
                x[2:4, runtime.classes.index(row['annotations'][0]['label'])] = .9
            seq[row['id']] = x
        row = score_candidate(c, bank, constraints(), score_cache=cache_scores(c, bank, seq))
        ident = row['candidate']
        if mutation == 'wrong_features':
            ident['features']['fmin_hz'] = 600.
            ident['features_sha256'] = digest(ident['features'])
        elif mutation == 'wrong_class_order':
            ident['classes'] = ident['classes'][::-1]
        elif mutation == 'tampered_digest':
            ident['code_revision'] = 'tampered-without-updating-digest'
        elif mutation == 'invalid_policy':
            ident['policy']['min_hops']['high'] = 0
        elif mutation == 'missing_policy_field':
            del ident['policy']['off_ratio']
        elif mutation in ('changed_checkpoint', 'changed_without_selection'):
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            key = next(k for k in arrays if k != 'meta')
            arrays[key] = arrays[key] + .125
            np.savez(path, **arrays)
            ident['model_sha256'] = sha256(path.read_bytes()).hexdigest()
        if mutation != 'tampered_digest':
            ident['candidate_sha256'] = digest({k: v for k, v in ident.items() if k != 'candidate_sha256'})
        seen.append(row)
        if mutation == 'changed_without_selection':
            return dict(status='infeasible', selected=None, attempted=[row])
        return dict(select_candidate([row]), attempted=[row])
    def forbidden(*args, **kwargs):
        raise AssertionError('post-selection fitting reached')
    monkeypatch.setattr(nn, '_fit_tracker', forbidden)
    out = tmp_path/'output.npz'
    kwargs = dict(checkpoint_selector=selector, selection_mode='operational', steps=3, width=4,
        blocks=1, batch=8, workers=0, threads=1, val_windows=64, p_confuser_given_background=0., seed=1103)
    if mutation != 'matching':
        with pytest.raises(ValueError):
            nn.train(manifest, out, **kwargs)
        assert len(seen) == 1  # callback and actual scoring reached, not early argument rejection
        assert not out.exists()
        return
    report = nn.train(manifest, out, **kwargs)
    assert len(seen) == 1 and report['selection']['status'] == 'feasible'
    runtime = edge.Model(str(out))
    selected = report['selection']['selected']['candidate']
    assert selected['features'] == runtime.meta['features']
    assert selected['classes'] == runtime.classes
    assert all(runtime.meta['detector'][k] == v for k, v in selected['policy'].items())
    checkpoint = out.with_name('output.checkpoint-3.npz')
    with np.load(checkpoint) as before, np.load(out) as after:
        assert all(np.array_equal(before[k], after[k]) for k in before.files if k != 'meta')
    assert report['selected_checkpoint_sha256'] == sha256(checkpoint.read_bytes()).hexdigest()
    assert report['artifact_sha256'] == sha256(out.read_bytes()).hexdigest()
    assert report['tracker_order'] == 'policy frozen before scoring; no post-selection fitting'
