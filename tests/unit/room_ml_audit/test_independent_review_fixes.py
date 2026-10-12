"""Regression coverage for independent review findings on z-model."""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from sensorsim.acoustics.room.augment import AugmentationConfig, TrainingLevelPolicy
from sensorsim.acoustics.room.dataset import DryClip, PositionPair, SamplingSpec, generate_dataset, read_dry
from sensorsim.acoustics.room.directivity import Directivity, DirectivityProfile, Orientation
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.acoustics.room.synth import SynthConfig, early_paths, late_bands, synthesize


def _dataset(tmp_path, clip, augmentation=AugmentationConfig()):
    fs = 8000
    pair = PositionPair(source_xyz_m=(1., 1., 1.), receiver_xyz_m=(4., 1., 1.))
    return generate_dataset(RoomModel(shoebox_dims_m=(6, 5, 3)), SamplingSpec(), [clip],
        augmentation, 1, 0, tmp_path / "dataset", fs=fs,
        synth_config=SynthConfig(max_order=0, late=False, air_absorption=False),
        target_positions=[pair], val_fraction=0)


@pytest.mark.parametrize("name,clip,augmentation", [
    ("A", lambda x, t, fs: DryClip("A_sub_lsb", 2e-5*x, fs), AugmentationConfig()),
    ("B", lambda x, t, fs: DryClip("B_gain", .01*x, fs), AugmentationConfig(gain_db=(-60., -60.))),
    ("D", lambda x, t, fs: DryClip("D_1lsb", np.rint(x).astype(np.int16), fs), AugmentationConfig()),
])
def test_ml04_rejects_a_row_when_encoded_pcm_is_silent(tmp_path, name, clip, augmentation):
    fs = 8000
    x = np.sin(2*np.pi*500*np.arange(160)/fs)
    with pytest.raises(ValueError, match=rf"clip .*{name}.*PCM"):
        _dataset(tmp_path, clip(x, np.arange(160)/fs, fs), augmentation)


@pytest.mark.parametrize("clip_kind,augment", [("array_rms", "rms"), ("wav_rms", "rms"),
                                                ("array_gain", "gain")])
def test_ml04_preserves_quiet_sources_that_export_nonzero_pcm(tmp_path, clip_kind, augment):
    fs = 8000
    x = 1e-5*np.sin(2*np.pi*500*np.arange(160)/fs)
    if clip_kind == "wav_rms":
        src = tmp_path / "quiet.wav"
        wavfile.write(src, fs, x.astype(np.float32))
        clip = DryClip("quiet_float_wav", src, fs)
    else:
        clip = DryClip("quiet_source", x, fs)
    policy = (AugmentationConfig(training_level=TrainingLevelPolicy(mode="rms"))
              if augment == "rms" else AugmentationConfig(gain_db=(40., 40.)))
    path = _dataset(tmp_path, clip, policy)
    row = json.loads(path.read_text().splitlines()[0])
    _, pcm = wavfile.read(path.parent / row["file"])
    assert np.any(pcm)


def _balloon():
    return DirectivityProfile(name="review fixture",
        directions_xyz=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)),
        pressure_gain=((1.,)*7,(.2,.3,.4,.5,.6,.7,.8),(.4,)*7,(.5,)*7,(.6,)*7,(.7,)*7),
        solid_angle_sr=(4*np.pi/6,)*6, reference_gain=(1.,)*7)


@pytest.mark.parametrize("fs", [8000, 16000, 32000, 48000])
def test_measured_directivity_late_render_scales_each_band(fs):
    profile = _balloon()
    source = Directivity(pattern="measured", profile=profile, orientation=Orientation())
    receiver = Directivity(pattern="measured", profile=profile, orientation=Orientation(yaw_deg=90))
    omni = RoomModel(shoebox_dims_m=(6,5,3), mixing_time_s=.04, late_t60_s=(.2,)*7)
    directed = omni.model_copy(update={"source_directivity": source, "receiver_directivity": receiver})
    ref = synthesize(omni, (1.1,1.3,1.2), (4.2,3.7,1.7), fs, seed=3)
    got = synthesize(directed, (1.1,1.3,1.2), (4.2,3.7,1.7), fs, seed=3)
    expected = (source.diffuse_power * receiver.diffuse_power)[[b.index for b in late_bands(fs)]]
    ratio = np.asarray(got.metadata["late_broadband_variance_at_mixing"]) / np.asarray(
        ref.metadata["late_broadband_variance_at_mixing"])
    np.testing.assert_allclose(ratio, expected, rtol=1e-14, atol=1e-14)


def test_ml06_rejects_object_array_before_identity_hashing():
    x = np.array([.1, .2, .3], dtype=object)
    with pytest.raises(ValueError, match="dtype"):
        read_dry(DryClip("object", x, 8000), 8000)


def test_ml10_default_handoff_is_continuous_and_provenanced():
    room = RoomModel(shoebox_dims_m=(4,3,2.6))
    source = np.array([.25,.25,.5]); receiver = np.array([3.75,2.75,1.8])
    cfg = SynthConfig()
    # The returned handoff follows max(original, direct + fade/2 + 1/c).
    u = (receiver-source)/np.linalg.norm(receiver-source)
    distances = np.linspace(4.1348, 4.1448, 5)
    effective = []
    for distance in distances:
        r = source + u*distance
        rir = synthesize(room, source, r, 16000, seed=1)
        effective.append(rir.metadata["mixing_time_s"])
        adjustment = rir.metadata["mixing_time_adjustment"]
        assert adjustment["original_mixing_time_s"] == pytest.approx(room.mixing_time())
        assert adjustment["effective_mixing_time_s"] == pytest.approx(rir.metadata["mixing_time_s"])
        early_paths(room, source, r, cfg)
    assert max(np.diff(effective)) < .00075


def test_ml10_does_not_replace_explicit_early_handoff():
    explicit = RoomModel(shoebox_dims_m=(4,3,2.6), mixing_time_s=.005)
    with pytest.raises(ValueError, match="increase mixing_time_s"):
        synthesize(explicit, (.25,.25,.5), (3.75,2.75,1.8), 16000)
