# Room synthesis and augmentation (stage 2)

> **Note on references.** This module was extracted from a larger project.
> Mentions of audits, frozen protocols, benchmark tables and validation runs
> refer to documents in that parent project, which are not included in this
> distribution. They are named here for provenance, not as working links.

This pipeline renders mono dry sounds at new positions in a rectangular room with
short solid box obstacles. Numerical geometry and synthetic decay have proxy
controls; measured-room accuracy and recognition benefit remain unqualified.
Use [USAGE.md](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/USAGE.md) for the surveyed deployment-room workflow and
[README_FIT.md](README_FIT.md) for fitting and uncertainty limits.

```python
from sensorsim.acoustics.room.model import RoomModel, BoxObstacle, uniform_bands
from sensorsim.acoustics.room.synth import synthesize, SynthConfig

room = RoomModel(
    room_id="example", shoebox_dims_m=(6, 5, 3),
    absorption=uniform_bands(.2), scattering=uniform_bands(.1),
    obstacles=(BoxObstacle(name="desk", min_xyz=(2.5, 1.3, 0),
                          size_xyz=(1.2, .8, .85), material="wood"),),
    provenance={"coefficients": "assumed; not measured"},
)
rir = synthesize(room, (1.2, 1.7, .6), (4.1, 1.7, .6), 32000, seed=7)
assert RoomModel.from_json(room.to_json()).model_hash == room.model_hash
```

## Model contract and controls

`room-model/1` is a strict, JSON-serialisable Pydantic schema. The lower room corner is
(0,0,0) metres, z upwards. Six surfaces are `x_min/x_max/y_min/y_max/z_min/z_max`;
absorption and scattering each contain seven **energy** coefficients ordered
125, 250, 500, 1000, 2000, 4000, 8000 Hz. Defaults (.2 absorption, zero wall scattering)
are assumptions. `BoxObstacle` accepts the recording manifest's `name`, `min_xyz`,
`size_xyz`, `material`, with explicit seven-band absorption/scattering (shared by its
faces). Box defaults .2/.1 are assumptions; the material text never silently selects
coefficients. Boxes must fit; overlapping or touching **boxes** are rejected. Wall/floor
contacts are supported. No `ingest`, fitting or dry-extraction module is replaced.

`temperature_c`, `humidity_pct` drive the existing ISO 9613-1 routine at reference
pressure 101.325 kPa, and the sound speed: Cramer's humid-air formula
(`humid-air-cramer/1`, 0-30 C) by default, or 331.3*sqrt(1+T/273.15) m/s with
`sound_speed_model="dry-air-core/1"`; an explicit `reference_sound_speed_m_s` overrides both. `late_t60_s` and `mixing_time_s` override late decay independently of the
walls. Provenance and uncertainty are caller-supplied JSON; no invented probability
interpretation is attached to them. Hashes cover the complete canonical model JSON.

`t60_estimates('sabine'|'eyring')` uses free volume (solid-box volumes subtracted),
exposed area and area-weighted absorption, plus ISO energy loss `m=alpha_dB*ln(10)/10`:
`T60=24 ln(10) V/[c*(A+4mV)]`; `A=S*alpha_bar` or `-S*ln(1-alpha_bar)`.
Box/wall contact faces and their covered room patches are excluded. Scattering changes
specular coherence, not absorbed energy. Mixing defaults to the larger of three mean
free flights and the diagonal flight; this is a heuristic, not an inferred mixing time.
For late synthesis, if that handoff would start before a requested pair's direct arrival,
the renderer raises the effective mixing time just enough to leave the direct arrival
before the crossfade. This pair-specific adjustment is recorded in the rendered model.

## Renderer, reuse and limitations

The NumPy shoebox lattice exposes image positions and six-wall hit counts, supports
per-wall octave coefficients, and shares the project's continuous ISO attenuation.
This array interface is the future torch/HIP backend seam; currently `backend='numpy'`
is the only synthesis backend. This choice provides transparent path diagnostics and
custom obstacle integration, **not a measured speed advantage** over installed MIT
pyroomacoustics 0.10.1. That package supplies the independent early-path reference and
synthetic decay diagnostics. No package installation is needed.

`SynthConfig(max_order=3, late=False)` selects bounded early ISM; `early_time_s` is an
additional geometric image-arrival cutoff. Automatic order covers the time horizon.
Default hybrid synthesis enumerates all images through the crossfade end; an explicit
order/time that could leave a gap is rejected. Maximum image order is 60; maximum RIR
length is five million samples (errors are explicit, never silent truncation).

Per path, reflection pressure is the product of `sqrt((1-alpha)*(1-scattering))`,
then `1/r`, air attenuation and optional source/receiver cardioid gains. Seven wall
pressure values interpolate linearly in log frequency, held flat outside the centres;
ISO air loss is evaluated continuously in frequency. Short Kaiser-windowed spectral
kernels carry the exact fractional phase. Default half-length is 64 samples. Returned
samples contain **64 samples of DSP latency**, recorded in `metadata.latency_samples`;
subtract that latency from sample time for geometric TOAs. Finite bandlimited kernels
have pre-ringing. The path table's delay/level is exact geometrically, while the waveform
has the frozen finite-filter tolerances. Wall reflection phase is zero. The late tail
tiles DC to Nyquist at every rate (`late-band-coverage/2`, `synth.late_bands`): the lowest
octave is a low-pass, a top octave cut by Nyquist becomes a high-pass, and above a complete
top octave an extension band follows that octave's walls plus the air loss of the octave
above it (`late_band_t60`, `extension_gain_ratio`).

Late pressure uses independent unit-variance Gaussian noise, warmed for 0.5 s
through each causal synthesis filter, then multiplied by
`exp(-3 ln(10)*(t-tmix)/T60)*sin(pi*u/2)` **at the filter output** (R1).
The filters share `metrics.py`'s exact base-10 edges. Bandwidth enters through
filtering, with no second bandwidth factor or finite-draw RMS normalization.
The variance anchor remains absorption-only image energy in a 12 ms window about
mixing; early pressure uses `cos(pi*u/2)`. The shared `late.py` covariance operator
retains temporal correlation through the envelope and causal analysis filter,
including neighboring/edge bands, before energy binning. It cannot be replaced by
the old squared cascade convolution. R1 changes samples/RNG sequence and fitted
outputs; pre-R1 proxy results are historical. A repaired envelope is not an
accepted per-position continuous join; the registered join FAIL remains.
Scattering energy is represented by this diffuse tail, rather than counted as absorption;
it does not implement an early directional scattered field. Obstacle runs retain the
unoccluded shoebox image-energy anchor but use obstacle-inclusive Eyring decay: absolute
late level around furniture is an approximation, not an independently validated energy
balance. Overrides change the prescribed slope without fitting the anchor.
See the repository MATH.md and KNOWN_ISSUES.md for current operator limits and
the retained proxy failures. WP8E-NUM remains open.

Obstacle image lines are folded into physical room segments before batched
`Scene`/`RayCore` visibility queries. First-order finite outward-face reflections use
explicit box coefficients. A blocked leg uses the shortest clear roof/side route from
`scene_paths`, with the existing Kurze–Anderson and ISO C3/caps magnitude model. Added
length changes delay and air loss; insertion loss references the unblocked distance.
No gain from coherently adding several alternative routes. Paths with several blocked
legs or no clear one-box detour are dropped and counted in metadata. Diffraction retains
the original image directivity directions (documented approximation). No transmission,
obstacle-wall reflection chains, multiple-obstacle diffraction or edge phase. For a
`late=False` time cutoff, a detour can arrive after its original image cutoff; it is kept
with its actual longer time, avoiding an incorrectly early arrival.

`obstacle_backend='cpu'` is the deterministic reference; `'auto'` resolves to CPU.
The incomplete Vulkan backend has been removed (KNOWN_ISSUES export-vk-01);
requesting `'vulkan'` raises a clear `ValueError`. `iter_synthesize` streams paired/broadcast source-receiver batches;
`synthesize_batch` returns a list. Seeds are stable per batch index; duplicate positions
with different seeds give independent tails, not a spatially coherent microphone array.

Optional `Directivity(pattern='cardioid', axis=(...))` is frequency-independent pressure
`.5*(1+cos(theta))`; omni is default. Source axes are mirrored with images; receiver
axes stay in room coordinates. Diffuse tail power uses the spherical mean (1/3 per
cardioid). This does not model a real loudspeaker, head shadow or binaural response.
No modal, elastic furniture, non-diffuse-room or measured recognition-accuracy claim.

## Augmentation: physical meaning

| Transform | Meaning and limitations |
|---|---|
| Gain | Source/electronics pressure ratio; no implied absolute SPL calibration |
| Peaking EQ / spectral tilt | Source resonance/colour, then zero-phase tilt; finite filter edges and EQ phase retained |
| Pitch / stretch | Polyphase playback-speed changes: pitch and duration are coupled, including motor-speed proxy; bandwidth loss and edge transients; no independent pitch-preserving stretch |
| Doppler | Optional linear time-varying delay, with zero padding and linear interpolation; static RIR remains fixed, no changing reflected paths/range spreading/convective gain |
| Microphone response | Seeded independent octave gain offsets after room convolution; no AGC/head shadow |
| Noise | White/pink only; removed traffic/park shapes are refused; SNR schedule in dB against whole clean-clip RMS; no local-window or calibrated ambient-level claim |

Source transforms run before convolution; microphone/noise transforms run after it.
The combined pitch/stretch speed is `2**(semitones/12)*stretch_rate`. A positive gain or
mic EQ does not automatically normalise the signal. Silence with requested SNR is rejected.
All realised parameters and independent child seeds are recorded.

## Dataset and real-sound mode

```python
from sensorsim.acoustics.room.augment import AugmentationConfig
from sensorsim.acoustics.room.dataset import DryClip, PositionPair, SamplingSpec, generate_dataset

manifest = generate_dataset(
    room, SamplingSpec(), [DryClip("reference", "data/dry.wav")],
    AugmentationConfig(gain_db=(-6, -3)), count=2, seed=7,
    output_dir="runs/room-training", fs=32000,
    target_positions=[
        PositionPair(source_xyz_m=(1.2,1.7,.6), receiver_xyz_m=(4.1,1.7,.6)),
        PositionPair(source_xyz_m=(1.2,1.7,.6), receiver_xyz_m=(5.2,4.3,1.4)),
    ], val_fraction=0,  # both examples share a source position; cannot split them
)
```

An array from the implemented [room.dry](README_DRY.md) stage is accepted as
`DryClip(label, samples, fs, provenance=...)`; a WAV path is equally supported. Dryness is
caller-supplied: the generator cannot certify that source/speaker/microphone coloration
was removed. WAV input is mono; differing sample rates are explicitly polyphase-resampled.
No speech content is modelled by this job. Class labels are copied, never inferred.

Without `target_positions`, independent uniform positions are sampled inside wall
clearance/height bounds and outside expanded obstacle boxes. Box clearances are conservative
axis-wise distances. Position-disjoint splitting groups shared coordinates across **both**
source and receiver roles (one-micrometre rounding). A single connected component cannot
make a nonempty train/val split and raises; `val_fraction=0` explicitly disables validation.
This does not ensure room/source-clip or nearby-position separation; use additional held-out
rooms/recordings when evaluating recognition generalisation.

Outputs are SciPy **16-bit PCM WAV** (no 24-bit/FLAC claim), plus `manifest.jsonl` and
`run.json`. Full convolution tails are retained. Each row contains model hash, positions,
label, split, all resolved config, selected input identity/hash, child seeds, realised
augmentation, RIR metadata/diagnostics, environment/git fingerprint, export scale and
output hash. `peak_policy='error'` prevents silent clipping; `'scale'` explicitly allows
per-clip attenuation to 0.98 peak and records it (this changes absolute distance-level cues).
No per-clip normalisation is applied otherwise. Quantisation has no added dither.

Synthetic diagnostics use installed `pyroomacoustics.experimental.measure_rt60` after the
crossfade and SciPy octave filters, plus energy/peak/delay. They are **not** stage-1
noise-corrected ISO metrics and never substitute target T60 for a measured waveform value.
Unsupported/insufficient bands are null. Stage-1 `room.metrics` remains separately owned.
Datasets stream one clip at a time. Nonempty output folders are refused; failures retain
`state=incomplete` and `manifest.partial.jsonl`. Successful completion publishes the final
manifest. Identical seeds/config/input bytes and the same software environment give identical
output; changing git/environment fingerprints intentionally changes provenance bytes.

Rendering reuses exact late-envelope work and bounded image-lattice arrays;
per-band random draws and path order are retained. A 2026-10-10
end-to-end synthetic-proxy benchmark measured about 1.14× rendering and 1.06×
dataset-generation speed versus the pre-optimization tree. These descriptive
host timings do not establish acoustic accuracy.

CLI help: `.venv/bin/python -m sensorsim.acoustics.room.dataset --help`. Configs are strict
JSON, not a recording-manifest ingestion replacement. `--clips` is a JSON list of
`{"path","label","provenance"?,"source_group"?}` rows, paths relative to that list, each with exactly
one of `"source_reference": {"target": "source_1m", "units": "relative"}` (a raw source WAV) or
`"dry_report": "<mic>_report.json"` (a `room.dry` output).
`--positions` is a JSON list of `source_xyz_m`/`receiver_xyz_m` pairs.

Historical parent protocol citations: shoebox, obstacles, results. These are
model-based/proxy records, not current-main acceptance or distributed reports.

References: [Allen & Berkley (1979)](https://doi.org/10.1121/1.382599),
[pyroomacoustics paper](https://arxiv.org/abs/1710.04196),
[pyroomacoustics room/filter-delay documentation](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html),
[Välimäki et al. (2012)](https://doi.org/10.1109/TASL.2012.2189567),
[W3C Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/).
