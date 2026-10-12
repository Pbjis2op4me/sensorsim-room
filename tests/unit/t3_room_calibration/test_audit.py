"""Development audit controls, including known-bad early-field and missing truth."""
import copy
import hashlib
import json

import pytest

from tools.room_calibration_audit import REQUIRED_KEYS, audit_products, interval_score, main, validate_partition


def document():
    truth = {k: .5 if k.endswith("_s") else 0. for k in REQUIRED_KEYS}
    units = {k: "s" if k.endswith("_s") else "dB" for k in REQUIRED_KEYS}
    def product(w):
        return dict(intervals={k:[v-w if units[k] == "s" else v-2*w,v,v+w if units[k] == "s" else v+2*w] for k,v in truth.items()},
                    nominal_interval=.9,units=units)
    cases = []
    for i in range(3):
        cases.append(dict(case_id=f"case-{i}", independent_group_id=f"group-{i}",
            truth_content_sha256=hashlib.sha256(f"truth-{i}".encode()).hexdigest(),
            partition=dict(dataset="synthetic",role="development",split_sha256="a"*64,independent_group_id=f"group-{i}"),
            truth=truth.copy(),units=units.copy(),
            candidate={p:product(.1) for p in ("parameter_only","actual_render","discrepancy_inclusive")},
            baseline={p:product(.2) for p in ("parameter_only","actual_render","discrepancy_inclusive")}))
    return dict(required_keys=list(REQUIRED_KEYS),score_scales={k:.5 if units[k] == "s" else 1. for k in REQUIRED_KEYS},
                seed=301,evidence_domain="synthetic proxy",cases=cases,
                historical_records=[dict(registration="historical-wp4",outcome="fail",source="reviewed historical record, not rerun")])


def test_correct_control_and_early_field_mutant_reach_actual_render_gate():
    doc = document(); good = audit_products(doc)
    assert good["products"]["actual_render"]["screen_outcome"] == "pass"
    assert good["products"]["discrepancy_inclusive"]["standardized_interval_score_improvement"] == pytest.approx(.5)
    assert good["registered_outcome"] == "inconclusive" and good["harness"] == "unqualified"
    for case in doc["cases"]:
        for key in REQUIRED_KEYS:
            if key.endswith("c50_db"):
                case["candidate"]["actual_render"]["intervals"][key] = [2.8,3.,3.2]
                # Deliberately broad discrepancy intervals can still cover truth.
                case["candidate"]["discrepancy_inclusive"]["intervals"][key] = [-4.,3.,10.]
    bad = audit_products(doc)
    assert bad["products"]["actual_render"]["screen_outcome"] == "fail"
    assert bad["products"]["actual_render"]["by_metric"]["c50_db"]["coverage_scored"] == 0
    assert bad["products"]["actual_render"]["by_metric"]["c50_db"]["median_absolute_error"] == 3.
    assert not bad["training_promotion"]
    assert bad["historical_records"] == good["historical_records"]


def test_denominators_preserve_missing_truth_and_missing_prediction():
    doc = document(); case = doc["cases"][0]
    case["truth"]["500.t30_s"] = None
    del case["candidate"]["actual_render"]["intervals"]["1000.t30_s"]
    out = audit_products(doc)["products"]["actual_render"]
    assert [out[k] for k in ("requested","truth_valid","predicted","scoreable","covered")] == [36,35,35,34,34]
    assert out["coverage_all_requested"] == pytest.approx(34/36)
    assert out["coverage_scored"] == 1.
    assert out["screen_outcome"] == "inconclusive"
    assert out["independent_group_count"] == 3


def test_score_never_drops_missing_baseline_or_uses_unequal_support():
    doc = document()
    del doc["cases"][0]["baseline"]["discrepancy_inclusive"]["intervals"]["500.t30_s"]
    out = audit_products(doc)["products"]["discrepancy_inclusive"]
    assert out["standardized_interval_score_improvement"] is None
    assert out["screen_outcome"] == "inconclusive"
    assert interval_score(2.,[.5,1.,1.5],.5) == 22.


@pytest.mark.parametrize("mutation", ["empty", "duplicate_content", "unit", "nonfinite", "missing_key", "wrong_domain", "filtered_requirements"])
def test_invalid_audit_inputs_refused(mutation):
    doc = document()
    if mutation == "empty": doc["cases"] = []
    if mutation == "duplicate_content": doc["cases"][1]["truth_content_sha256"] = doc["cases"][0]["truth_content_sha256"]
    if mutation == "unit": doc["cases"][0]["units"]["500.t30_s"] = "ms"
    if mutation == "nonfinite": doc["cases"][0]["truth"]["500.t30_s"] = float("nan")
    if mutation == "missing_key": del doc["cases"][0]["truth"]["500.t30_s"]
    if mutation == "wrong_domain": doc["evidence_domain"] = "deployment recording"
    if mutation == "filtered_requirements": doc["required_keys"] = ["500.t30_s"]
    with pytest.raises(ValueError): audit_products(doc)


@pytest.mark.parametrize("source,array,fmt", [(9,1,"official-hdf5-v2"),(7,6,"official-hdf5-v2"),(7,1,"sofa"),("7",1,"official-hdf5-v2")])
def test_locked_or_uncleared_dechorate_partition_refused(source,array,fmt):
    with pytest.raises(ValueError):
        validate_partition(dict(dataset="dEchorate",source_id=source,array_id=array,format=fmt,role="permitted-heldout",
            split_sha256="a"*64,independent_group_id="room",licence="CC-BY-4.0",use="internal-validation"))


def test_permitted_partition_and_missing_actual_hook_are_explicit():
    validate_partition(dict(dataset="dEchorate",source_id=7,array_id=1,format="official-hdf5-v2",role="permitted-heldout",
        split_sha256="a"*64,independent_group_id="room",licence="CC-BY-4.0",use="internal-validation"))
    doc = document()
    for case in doc["cases"]: case["candidate"]["actual_render"] = None
    out = audit_products(doc)["products"]["actual_render"]
    assert out["predicted"] == 0 and out["scoreable"] == 0
    assert out["screen_outcome"] == "inconclusive"


def test_cli_preserves_previous_attempt(tmp_path):
    path, out = tmp_path/"input.json", tmp_path/"audit.json"
    path.write_text(json.dumps(document()))
    assert main([str(path),"--output",str(out)]) == 0
    assert json.loads(out.read_text())["result_label"] == "proxy result"
    with pytest.raises(FileExistsError): main([str(path),"--output",str(out)])


def test_missing_required_prediction_blocks_even_where_truth_is_unavailable():
    doc = document()
    doc["cases"][0]["truth"]["500.t30_s"] = None
    del doc["cases"][0]["candidate"]["actual_render"]["intervals"]["500.t30_s"]
    summary = audit_products(doc)["products"]["actual_render"]
    assert summary["missing_prediction_on_truth_valid"] == 0
    assert summary["truth_available_fraction"] > .9
    assert summary["screen_outcome"] == "inconclusive"
