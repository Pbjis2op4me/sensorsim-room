"""cr-tests-harness-05: failed-run recovery is bound to fresh attempt evidence."""
import json

import pytest

from tests.validation.bench_room_closedloop import runner


def _attempt(tmp_path):
    session = tmp_path/"session"
    session.mkdir()
    source = session/"take.wav"
    source.write_bytes(b"current delivered input")
    output = tmp_path/"pipeline"
    attempt = runner.begin_attempt(session, output)
    dry = output/"dry"
    dry.mkdir(parents=True)
    (dry/"take_0000_ch_00.wav").write_bytes(b"dry render")
    (dry/"take_0000_ch_00.json").write_text(json.dumps({"fs": 8000}))
    (dry/"take_0000_ch_00_dataset_source.wav").write_bytes(b"dataset source")
    (dry/"event_selection.json").write_text(json.dumps([
        {"take": 0, "channel": 0, "file": str(source), "event_index": 0}]))
    (output/"resolved_config.json").write_text(json.dumps({
        "input_sha256": {str(source): runner.file_hash(source)}}))
    return attempt, source, [{"type": "real", "file": "take.wav"}]


def test_recovery_needs_attempt_selection_receipt_and_complete_products(tmp_path):
    attempt, source, takes = _attempt(tmp_path)
    assert list(runner.recover_dry_takes(attempt, takes, 8000)) == [0]
    with pytest.raises(ValueError, match="sample rate"):
        runner.recover_dry_takes(attempt, takes, 8001)
    selection = attempt.output/"dry/event_selection.json"
    selection.unlink()
    assert runner.recover_dry_takes(attempt, takes, 8000) == {}
    selection.write_text(json.dumps([{"take": 0, "channel": 0,
                                      "file": str(tmp_path/"other"/source.name), "event_index": 0}]))
    with pytest.raises(ValueError, match="event selection"):
        runner.recover_dry_takes(attempt, takes, 8000)
    selection.write_text(json.dumps([{"take": 0, "channel": 0,
                                      "file": str(source), "event_index": 0}]))
    source.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="input bytes"):
        runner.recover_dry_takes(attempt, takes, 8000)


def test_reused_output_folder_cannot_begin_attempt(tmp_path):
    output = tmp_path/"pipeline"
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh output"):
        runner.begin_attempt(tmp_path, output)
    with pytest.raises(TypeError, match="PipelineAttempt"):
        runner.recover_dry_takes(output, [], 8000)
