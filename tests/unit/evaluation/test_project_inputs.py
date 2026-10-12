import json

from sensorsim import project


def test_evaluation_directory_supports_external_take_filename_layout(tmp_path):
    root = tmp_path/"takes"
    root.mkdir()
    (root/"source").mkdir()
    for name in ("knock_01.wav", "doorbell_02.wav", "negative_dishes_03.wav"):
        (root/name).write_bytes(b"proxy")
    (root/"source"/"doorbell.wav").write_bytes(b"dry source")
    rows = project._evaluation_files(root, "measured")
    assert [(row["label"], row["group"]) for row in rows] == [
        ("doorbell", "doorbell_02"), ("knock", "knock_01"), ("negative_dishes", "negative_dishes_03")]
    assert all(row["data_kind"] == "measured" for row in rows)


def test_manifest_data_kind_and_absolute_paths_are_reported_safely(tmp_path):
    root = tmp_path/"evaluation"
    root.mkdir()
    recording = tmp_path/"proxy_take.wav"
    (root/"manifest.jsonl").write_text(json.dumps({"file": str(recording), "label": "knock",
        "data_kind": "proxy", "group": "proxy-parent"})+"\n")
    row = project._evaluation_files(root, "measured")[0]
    assert row["path"] == recording
    assert row["file"] == "proxy_take.wav"
    assert row["data_kind"] == "proxy"


def test_project_evaluate_cli_runs_the_evaluate_stage_directly(tmp_path, monkeypatch, capsys):
    cfg = {"root": str(tmp_path), "out": "out"}
    monkeypatch.setattr(project, "load_project", lambda path: cfg)
    def evaluate(given, out):
        assert given is cfg
        assert out == tmp_path/"out"
        return {"data_kind": "proxy", "status": "inconclusive"}
    monkeypatch.setattr(project, "_evaluate", evaluate)
    assert project.main(["evaluate", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"data_kind": "proxy", "status": "inconclusive"}
