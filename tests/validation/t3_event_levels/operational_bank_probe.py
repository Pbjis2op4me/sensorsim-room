# SPDX-License-Identifier: Apache-2.0
"""Explicit cross-lane contract controls; require a pinned source snapshot.

Run separately with P3_BANK_PIN pointing at the reviewer's dependency manifest.
Missing dependencies/digests fail collection. This never certifies the bank's
selector or imports a moving worktree implicitly. All audio here is procedural.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim import event_levels as el
from sensorsim.acoustics.room.source_reference import SourceReference


PIN = json.loads(Path(os.environ["P3_BANK_PIN"]).read_text())
for filename, expected in PIN["files"].items():
    path = Path(PIN["root"])/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, "pinned B dependency digest changed"

# Resolve both modules from the snapshot; other imports stay on the candidate.
for leaf in ("operating_point", "selection"):
    name = "sensorsim.evaluation."+leaf
    path = Path(PIN["root"])/"src/sensorsim/evaluation"/(leaf+".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
selection = sys.modules["sensorsim.evaluation.selection"]


def bank_fixture(tmp_path):
    fs = 16000
    source = .01*np.sin(2*np.pi*137*np.arange(fs)/fs)
    parent = el.ParentEvent("train", "train-group", "train-session", "kettle", "unit", "cycle",
        "train", "synthetic_proxy", "source_only", source, SourceReference(),
        el.LevelReference("rms", "relative", 1., "Z", (0, fs), fs, "source-1m"))
    rows = []
    for i, role in enumerate(("train", "development", "audit")):
        pcm = np.rint(1000*np.sin(2*np.pi*(137+i*37)*np.arange(fs)/fs)).astype(np.int16)
        filename = role+".wav"
        # Only development audio exists. Training/audit identities are metadata;
        # neither path may be opened while validating development provenance.
        path = tmp_path/filename
        if role == "development":
            wavfile.write(path, fs, pcm)
        rows.append(dict(id=role, role=role, kind="target", parent_id=role,
            session_id=role+"-session", lineage_ids=[role+"-group"],
            parent_audio_sha256=[parent.content_id if role == "train" else hashlib.sha256(role.encode()).hexdigest()],
            audio=filename, file_sha256=hashlib.sha256(path.read_bytes() if path.exists() else role.encode()).hexdigest(),
            audio_sha256=selection.audio_digest(pcm.astype(np.float32)/32768), sample_rate_hz=fs, sample_count=fs,
            route="procedural-development", acoustic_stage="rendered_context", evidence_domain="synthetic proxy",
            panel="nominal", annotations=[dict(label="kettle", start_s=.1, end_s=.9)]))
    return parent, dict(schema="operational-bank/1", allocation_frozen=True, classes=["kettle"],
                        required_families=["slam"], rows=rows)


def load_bank(tmp_path, manifest):
    path = tmp_path/"bank.json"
    path.write_text(json.dumps(manifest))  # File digest differs from canonical content digest.
    return selection.build_validation_bank(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_pinned_bank_parent_content_session_contract(tmp_path, record_property):
    parent, manifest = bank_fixture(tmp_path)
    bank = load_bank(tmp_path, manifest)
    bank.verify()
    assert bank.content_sha256 == selection.digest(manifest)
    assert bank.manifest_sha256 != bank.content_sha256
    roles = el.role_bindings_from_bank(bank.manifest, expected_sha256=bank.content_sha256)
    prior = el.fit_level_prior([parent], mode=el.RELATIVE, role_bindings=roles,
                              relative_anchors={"train-session": el.RelativeAnchor(.25, "assumed")})
    assert prior.parents == (parent,)
    assert roles["session:audit-session"] == "audit"
    assert not (tmp_path/"audit.wav").exists()
    record_property("B_revision", PIN["revision"])
    record_property("B_files", PIN["files"])
    record_property("allocation_content_sha256", bank.content_sha256)
    # B's float32 rendered hash alone cannot substitute for source+rate identity.
    bad = copy.deepcopy(manifest)
    bad["rows"][0]["parent_audio_sha256"] = [bad["rows"][0]["audio_sha256"]]
    bad_roles = el.role_bindings_from_bank(bad, expected_sha256=selection.digest(bad))
    with pytest.raises(ValueError, match="train"):
        el.fit_level_prior([parent], mode=el.RELATIVE, role_bindings=bad_roles,
                          relative_anchors={"train-session": el.RelativeAnchor(.25, "assumed")})


@pytest.mark.parametrize("field", ["parent_id", "session_id", "lineage_ids", "parent_audio_sha256", "audio_sha256", "file_sha256"])
def test_b_and_p3_reject_same_cross_role_alias(tmp_path, field):
    _, manifest = bank_fixture(tmp_path)
    load_bank(tmp_path, manifest).verify()  # Known-correct control for every mutation.
    bad = copy.deepcopy(manifest)
    bad["rows"][-1][field] = copy.deepcopy(bad["rows"][0][field])
    with pytest.raises(ValueError, match="crosses roles"):
        load_bank(tmp_path, bad)
    with pytest.raises(ValueError, match="crosses roles"):
        el.role_bindings_from_bank(bad, expected_sha256=selection.digest(bad))
