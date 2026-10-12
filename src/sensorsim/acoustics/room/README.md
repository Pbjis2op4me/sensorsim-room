# Room recordings and acoustic metrics (stage 1)

> **Scope at main 159518b:** this is the optional legacy sweep/extraction path.
> The surveyed `room.yaml`, no-sweep playback calibration and chamber-target
> workflow is documented in the repository USAGE.md and has a **synthetic proxy rehearsal**.
> Route completion did not qualify panel selection, short calibration or deployment accuracy.
> Historical numeric capture/fit examples here are model-based/proxy evidence,
> not deployment recordings or post-R1 acoustic acceptance.


> **Note on references.** This module was extracted from a larger project.
> Mentions of audits, frozen protocols, benchmark tables and validation runs
> refer to documents in that parent project, which are not included in this
> distribution. They are named here for provenance, not as working links.

`load_manifest` accepts the user's `room_recording_kit/manifest_template.yaml`
format and the internal coordinate-based format. `fs_hz`, named speakers/mics,
room dimensions, box obstacles, recorder/speaker descriptions and take types
are converted to `RecordingManifest`. Zero-sized unfinished template rooms are
rejected. Coordinates include the room boundary; no obstacle acoustic model is
applied during ingestion. A mic name denotes one channel; use a list of mic
names for multichannel files. Internal `receiver_xyz_m: [x,y,z]` denotes
co-located channels, or supply one XYZ per channel. `mic_ids` and unit
`receiver_orientation_xyz` are optional. Calibration is
`calibration: {pa_per_fs: scalar_or_channel_list}`.

Paths are relative to the manifest, checked on load; WAV/FLAC headers must
match `fs` and channels. WAV uses the existing native reader; FLAC uses installed
FFmpeg with two threads and no seek (the shared reader's `-ss 0` drops short
FLACs). No soundfile dependency. `read_take` returns `(samples,channels)` float32
in FS, or Pa if calibrated. Headers are read eagerly, audio one take at a time.
FFT extraction holds a whole take in memory; this is offline measurement
processing, not a streaming simulator/long-session renderer.

```python
from sensorsim.acoustics.room import load_manifest, extract_take, analyze_octaves
manifest = load_manifest("data/my_room/manifest.yaml")
# Index a type='sweep' take (including REF), not noise or a real sound.
rir = extract_take(manifest, 1, rir_duration_s=1.0)
metrics = analyze_octaves(rir.samples[:, 0], rir.fs,
                          onset_sample=rir.metadata["direct_samples"][0])
print(metrics[1000].t20.rt60_s, metrics[1000].t20.reason)
```

Generic internal manifests use `room_id`, `fs`, optional `shoebox_dims_m`, and
`takes: [{file, source_xyz_m, receiver_xyz_m, excitation}]`. Excitation is
`{type: ess, duration_s, f_start_hz, f_end_hz, amplitude, fade_s}` (optional
separate `fade_in_s`/`fade_out_s` select the kit half-Hann convention),
`{type: known_signal, path: mono.wav}`, or `{type: unknown}`. Generic ESS takes
start at excitation sample zero; direct `deconvolve_ess` supports a known
acquisition offset. User-kit takes locate the entire repetition train.
Real/noise takes are retained for later stages, not deconvolved as sweeps.
`deconvolve_blind` raises `NotImplementedError`.

## Explicit held-out check sweeps

Both manifest formats accept `role: check` on a non-REF sweep. Such takes never
set fitted room parameters, direct anchors or pooled levels and cannot share a
file or source/receiver pair with fitting/REF/real takes. Explicit checks replace
the automatic receiver split; ordinary profiling sweeps then train. The pipeline
publishes a separate frozen statistical scorecard and predicted RIRs in `checks/`.
Missing/failed support remains visible. No check/QA sidecar preserves legacy
outputs; supplied capture QA is reported separately from completion. See the
contract and limits.

## Extraction and timing

`generate_ess` reproduces the kit's Farina phase and asymmetric half-Hann fades;
`generate_kit_playback` constructs all four sweeps with the exact silence layout.
The float waveform differs from PCM16 only by its documented dither/quantization.
`ess_inverse` uses `reverse(sweep)*exp(-t/L)` and stationary-phase **in-band gain**
normalisation; forcing the band-limited impulse peak to one would bias gain.
`deconvolve_ess` returns linear response and optional harmonic packets at
`-L*ln(order)`. Packet windows must not overlap; harmonics are not independently
identified nonlinear/Volterra kernels. Fades, playback band limits and
recording duration limit all extraction. `deconvolve_known` uses
`Y conj(X) / (|X|² + lambda max|X|²)`, default `lambda=1e-8`, and requires the
complete excitation plus decay. It cannot recover excitation spectral holes.

`deconvolve_repetitions` finds all four peaks, estimates constant clock drift
(up to 300 ppm), optionally resamples the take, then aligns and averages each
channel's repeat peaks. A take is refused if **any** channel clips, not only the
loudest. Legacy alignment searches each channel within ±30 ms of its own arrival
(the reference channel's detected arrival plus that channel's geometric delay, or
zero without geometry) and refuses a channel with no supported packet there
rather than cropping it. The opt-in `repeat_alignment="packet-medoid/1"` aligns
repeats to the medoid packet and refuses arrivals outside its 50 ms early window.
A `SweepKit` whose `period_s` is not a whole number of samples at `fs_hz` is
refused when it is built. Metadata records arrival samples, estimated drift,
latency and applied geometry alignment. Known XYZ puts each peak at
`round(distance/c*fs)`: the pipeline passes the speed of its `timing_policy` (Cramer's
humid-air speed at the configured temperature and humidity by default); the low-level
default is `c=343 m/s`.
**Strongest peak = direct sound is an assumption:** obstacle shadows and strong
reflections can violate it. Geometry-aligned timing is not an independent range
measurement. No AGC, changing Bluetooth latency, motion or nonlinear drift repair.
`trim_rir` retains a common channel crop and accumulates `time_origin_s` metadata.
Both integer and fractional direct-arrival indices move with the crop; adding
`time_origin_s` to either sample/fs recovers its original arrival time.
Onset thresholding and `early_reflections` produce candidates, not wall labels.

## REF handoff to dry extraction and fitting

`estimate_reference_eq(ref_rir, direct_sample=...)` returns `ReferenceEQ`:
`fs`, `nfft`, `frequency_hz`, complex `response`, complex `correction`, boolean
`supported`, and metadata. Use `correction` only on this exact FFT grid and
supported bins; regenerate for another `nfft`. Unsupported bins are zero.
It gates around REF, removes propagation delay and `1/r` relative to 1 m,
then forms a regularised complex inverse (default `1e-4`, 12 dB gain cap).
The default gate is -2.5..+10 ms; default support 200..8000 Hz, additionally
limited by inverse gate duration and transfer power relative to regularisation.

This estimates **speaker plus microphone together**, with unknown absolute gain.
At 0.3 m the speaker may be in the near field; early reflections, phone filtering,
low-frequency SNR and directionality contaminate it. No claim of separated mic
calibration, clean room-only RIR, or recovered physical source power. A real
source did not pass through the sweep speaker: later dry extraction must not
remove that speaker twice or assume REF identifies mic and speaker separately.
This stage supplies the RIR/REF handoff. [Dry extraction](README_DRY.md),
[stage-2 synthesis](README_SYNTH.md), and [room fitting](README_FIT.md) are
implemented. Their numerical and measured-position limits remain separate;
see the validation matrix for current evidence.

For near-table REF geometry the box bound includes same-side supporting-plane
image paths. It remains conservative for finite faces and edges. The 0.25 ms
guard is unchanged: a 0.3 m REF at 5 cm height above the tabletop still rejects.
More clearance may resolve the direct packet while leaving its short window
unable to support low octaves; successful admission is not full-band calibration.
See the second-pass audit.

## Metrics and evidence

`room_metrics` accepts a mono RIR and optional manual onset / noise power;
`analyze_octaves` processes broadband and all 125..8000 Hz bands (fs must support
all upper edges, e.g. 48 kHz). Exact base-10 centres and causal 12-pole Butterworth
bandpasses were checked in the parent project against IEC 61260-3 periodic response points
at 48 kHz; the standard's table and those checks are not included in this repository. No complete IEC 61260-1 conformance certification is claimed. Band metrics
follow `room-measured-metrics/2`: EDT/T20/T30 from the band filter applied to the
time-reversed response, C50/C80/D50/Ts from a zero-phase filter integrated from one filter
span before the onset (MATH.md); `policy="room-measured-metrics/1"` restores the causal
convention with its filter delay. DRR splits broadband pressure before filtering; pass raw
pressure with `band=...`, not a prefiltered RIR.

Schroeder integration uses iterative Lundeby crossing, stationary noise
subtraction (Chu), truncation, and fitted missing exponential energy/moment.
The final 10% must contain stationary background. No usable tail, nonconvergence
or missing dynamic range produces invalid fits with reasons (not invented RTs).
`noise_power=0` explicitly requests ordinary finite-record integration. EDT,
T20 and T30 extrapolate 0..-10, -5..-25 and -5..-35 dB fits to T60; diagnostics
include slope and R². Regression ends at least 10 dB above the noise crossing.
C50/C80, D50 and Ts use the same compensated energy. DRR defaults to `drr/2`: split
broadband pressure at +-2.5 ms around onset, filter the components independently,
and retain their full ring-out. `legacy_drr=True` requests the construct-invalid
low-band `drr/1` for historical comparisons. This remains an explicit window
convention, not an ISO 3382 parameter or physical path separator. The
definition and limits are authoritative.

Status: **partially validated**. The frozen synthetic 125 Hz ensemble T20 gate
fails (+4.32% against 2%); the other mean RT gates pass. Finite low-band RIRs
have much larger individual variation than ensemble-mean tolerance. Detailed
protocol,
kit protocol, and
results
separate numerical checks from measured-room validation. No existing listening-kit
case changes: this package is opt-in and no runtime propagation was modified.
