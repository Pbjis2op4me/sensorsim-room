"""Regressions for defects fixed in this distribution.

Each test is derived from the reproduction attached to the parent project's
2026-10-03 code review finding of the same ID, and failed before its fix.
"""
from dataclasses import replace

import numpy as np
import pytest

from sensorsim.acoustics.room.fit import Observation, estimate_geometry


def test_cr_room_model_04_integer_xyz_matches_float_xyz():
    """Integer coordinates (as written in a YAML manifest) must not truncate wall images."""
    dims = np.array([5.4, 6.2, 3.7])
    source = (1, 1, 1)
    observations = []
    for i, receiver in enumerate(((2, 2, 1), (3, 1, 2), (3, 4, 2))):
        s, r = np.asarray(source, float), np.asarray(receiver, float)
        echoes = {}
        for axis, name in enumerate(("x_max", "y_max", "z_max")):
            image = s.copy()
            image[axis] = 2*dims[axis]-s[axis]
            echoes[name] = float((np.linalg.norm(image-r)-np.linalg.norm(s-r))/343.)
        observations.append(Observation(str(i), source, receiver, 48000, {},
                                        np.array(list(echoes.values())), {"wall_echoes_s": echoes}))
    floats = [replace(o, source_xyz_m=tuple(map(float, o.source_xyz_m)),
                      receiver_xyz_m=tuple(map(float, o.receiver_xyz_m))) for o in observations]
    expected = estimate_geometry(floats)
    actual = estimate_geometry(observations)
    np.testing.assert_allclose(actual["shoebox_dims_m"], expected["shoebox_dims_m"], atol=1e-8)
    np.testing.assert_allclose(actual["shoebox_dims_m"], dims, atol=1e-6)


def test_cr_room_dsp_02_output_buffer_aliasing_inputs_is_rejected():
    """An output buffer sharing the noise floor used to zero the result silently."""
    from sensorsim.acoustics.room.dry import extract_dry

    x = np.random.default_rng(3).normal(size=800)
    noise = np.zeros_like(x)
    kwargs = dict(recordings={"M": x}, rirs={"M": np.ones(1)}, noise_floor=noise, fs=8000, case="A")
    assert np.linalg.norm(extract_dry(**kwargs).samples) > 1  # independent buffer: a real signal
    with pytest.raises(ValueError, match="must not alias noise floors"):
        extract_dry(**kwargs, output_buffer=noise)
    # The RIRs are reread after output writes too: an RIR viewing the output buffer is rejected.
    shared = np.zeros(800)
    rir = shared[:1]
    rir[0] = 1.
    with pytest.raises(ValueError, match="must not alias RIRs"):
        extract_dry(**{**kwargs, "rirs": {"M": rir}}, output_buffer=shared)


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg to encode FLAC")
def test_cr_room_capture_01_flac_and_wav_get_the_same_clipping_decision(tmp_path):
    """Phone apps often save lossless FLAC; it must not bypass the integer-PCM clipping rule."""
    import subprocess
    from scipy.io import wavfile
    from sensorsim.acoustics.room.ingest import RecordingManifest, read_recording_audio, read_take

    pcm = np.zeros(1000, np.int16)
    pcm[[100, 200, 300]] = 32767  # three isolated positive ADC saturations
    wave, flac = tmp_path/"clipped.wav", tmp_path/"clipped.flac"
    wavfile.write(wave, 8000, pcm)
    subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(wave), "-threads", "2", str(flac)], check=True)
    np.testing.assert_array_equal(read_recording_audio(wave)[0], read_recording_audio(flac)[0])
    for path in (wave, flac):
        manifest = RecordingManifest(room_id="clipping-probe", fs=8000, takes=[dict(
            file=path, type="noise", receiver_xyz_m=(1, 1, 1), excitation={"type": "unknown"})])
        with pytest.raises(ValueError, match="clipping at digital full scale"):
            read_take(manifest, 0)


def _git_repo(path):
    import subprocess
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, text=True)
    run("init", "-q")
    (path / "f").write_text("x")
    run("add", "f")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "caller repo")
    return run("rev-parse", "HEAD").stdout.strip()


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="needs git")
def test_cr_room_capture_04_provenance_never_records_the_callers_repository(tmp_path, monkeypatch):
    """Run from inside another repo, or installed inside one: its HEAD must never be recorded."""
    from sensorsim.core import fingerprint
    from sensorsim.acoustics.room import session_sim

    caller_head = _git_repo(tmp_path / "company")
    monkeypatch.chdir(tmp_path / "company")             # route 1: caller's working directory
    assert session_sim._git_head() != caller_head
    assert fingerprint.source_revision()["commit"] != caller_head

    # route 2: an installed copy whose site-packages lives inside another repository
    installed = tmp_path / "company/.venv/lib/python3.12/site-packages/sensorsim/core/fingerprint.py"
    installed.parent.mkdir(parents=True)
    monkeypatch.setattr(fingerprint, "__file__", str(installed))
    assert fingerprint.source_revision()["commit"] is None


def test_ml02_every_class_with_two_recordings_reaches_validation(tmp_path):
    """Validation used to draw recording groups label-blind; a 3-class set could validate on 1 class."""
    import json
    from sensorsim.acoustics.room.augment import AugmentationConfig
    from sensorsim.acoustics.room.dataset import DryClip, SamplingSpec, generate_dataset
    from sensorsim.acoustics.room.model import RoomModel
    from sensorsim.acoustics.room.synth import SynthConfig

    fs = 8000
    t = np.arange(160) / fs
    clips = [DryClip(label, .01*np.sin(2*np.pi*f*(1+k/10)*t), fs, source_group=f"{label}-{k}")
             for label, f in (("beep", 500), ("hiss", 1100), ("whistle", 1900)) for k in (0, 1)]
    path = generate_dataset(RoomModel(shoebox_dims_m=(6, 5, 3)), SamplingSpec(), clips, AugmentationConfig(),
                            count=10, seed=0, output_dir=tmp_path/"ds", fs=fs, all_sources=True,
                            synth_config=SynthConfig(max_order=0, late=False, air_absorption=False))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for split in ("train", "val"):
        assert {r["label"] for r in rows if r["split"] == split} == {"beep", "hiss", "whistle"}
    groups = {s: {r["dry_source"]["source_group"] for r in rows if r["split"] == s} for s in ("train", "val")}
    assert not groups["train"] & groups["val"]  # still leakage-free by recording
