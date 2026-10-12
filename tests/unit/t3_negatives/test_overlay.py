"""WP2 source admission controls; no false-alarm efficacy claim."""
import json
import pytest
from sensorsim import training_negatives as tn
from sensorsim.negatives.composer import Event
from sensorsim.negatives.procedural import generate, Source
from .conftest import sha


def test_overlay_parent_and_permission_match_and_audit_is_separate(factory):
    f=factory(procedural=True,count=1);entry=f.data['rows'][0]
    source=generate(fs=f.fs,**entry['source']['recipe'])
    event=Event(source,1.,(1.,1.,1.),70.)
    review=json.loads((f.root/entry['semantic_review']['file']).read_text())
    args=dict(lock=f.root/'split.json',lock_sha256=f.data['split_file_sha256'],purpose='train',
        parent_items=f.parents,scope=f.data['target_scope'],reviews=[review],
        source_records=[dict(procedural=entry['source']['recipe'])],root=f.root)
    result=tn.validate_overlay_sources([event],**args)
    assert result['parents'][0]['source_group']=='group-0'
    with pytest.raises(ValueError,match='authorized'):
        tn.validate_overlay_sources([event],**dict(args,purpose='audit'))
    with pytest.raises(ValueError,match='clearance'):
        tn.validate_overlay_sources([event],**dict(args,source_records=[{}]))
    bad=dict(entry['source']['recipe'],seed=99)
    with pytest.raises(ValueError,match='recipe'):
        tn.validate_overlay_sources([event],**dict(args,source_records=[dict(procedural=bad)]))


def test_recorded_overlay_runs_existing_rights_gate(factory):
    f=factory(count=1);entry=f.data['rows'][0]
    from sensorsim.negatives.clips import load_clip
    spec=entry['source'];source=load_clip(f.root,spec['asset'],register_path=f.root/spec['register']['file'],
                                          review=spec['review'],fs=f.fs)
    event=Event(source,1.,(1.,1.,1.),70.)
    review=json.loads((f.root/entry['semantic_review']['file']).read_text())
    args=dict(lock=f.root/'split.json',lock_sha256=f.data['split_file_sha256'],purpose='train',
        parent_items=f.parents,scope=f.data['target_scope'],reviews=[review],
        source_records=[dict(clearance=spec)],root=f.root)
    assert tn.validate_overlay_sources([event],**args)['parents'][0]['item_id']=='parent-0'
    spec['asset']['license_id']='CC-BY-NC-4.0'
    with pytest.raises(ValueError,match='licence|prohibited'):
        tn.validate_overlay_sources([event],**args)
