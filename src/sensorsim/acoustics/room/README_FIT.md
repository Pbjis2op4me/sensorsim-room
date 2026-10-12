# Room inversion (stage 3)

> **Note on references.** This module was extracted from a larger project.
> Mentions of audits, frozen protocols, benchmark tables and validation runs
> refer to documents in that parent project, which are not included in this
> distribution. They are named here for provenance, not as working links.

`fit.py` fits an **effective** `RoomModel` to measured RIRs at known coordinates.
It does not uniquely identify physical wall absorption, scattering, furnishings,
or separate speaker/microphone responses. Accuracy is bounded by the measured
position holdouts in the rerun.
The original audit used an older synthesizer.
Integrated-main and corrected-forward campaigns keep its gates and splits;
see the rerun's per-band tables and combined denominators. The fitter remains
unvalidated; meeting a late-decay or spectral gate alone is insufficient.
Historical dEchorate and MeshRIR fit figures are withheld. No retained
dataset-derived result establishes faithful relocation of arbitrary audio
or recognition improvement.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 10 .venv/bin/python -m sensorsim.acoustics.room.fit \
  --manifest /path/to/recordings/manifest.yaml --out /path/to/job/fit
```

Use the canonical recording-kit manifest, including named speaker/mic positions,
surface descriptions, box obstacles, ESS parameters and source-specific REF
takes. The existing stage-1 ingest/extraction code is reused. Noise and real-sound
takes are retained by ingest but do not become fitting observations. The output
directory must be empty: existing evidence is never overwritten by the CLI.
Scratch/cache environment variables should point into the job directory.

Outputs consumed by the pipeline:

* `room_model.json`: load with `RoomModel.from_json(path.read_text())`; it keeps
  all six surfaces and seven octave bands, boxes, late overrides, provenance and
  conditional bootstrap intervals.
* `fit_report.json`: full per-position/band signed errors and PASS/FAIL/UNAVAILABLE,
  exact split, input hashes, extraction/device metadata, environment/commit,
  optimizer outcomes, early echoes and fast-objective/waveform differences.
* `resolved_config.json`: the `FitConfig` used. The resolved ingest manifest is
  also included in the report.

`--bootstrap` defaults to 20; `--seed` to 20260930; `--rir-duration` to 1.5 s.
Choose duration from the acquisition protocol before comparison. A shorter RIR
that lacks a noise-only tail can make ISO metrics unavailable. A return code of
zero means the pipeline completed, **not** that measured validation passed.
The CLI prints the held-out gate counts.

`fit_room(..., bootstrap_workers=2)` or pipeline
`fit_options: {bootstrap_workers: 2}` parallelizes only the bootstrap refits;
the default is one worker and the main fit remains serial. The standalone fit
CLI has no worker flag. Spawned workers preserve ordered draws and optimizer
starts; errors propagate. Python callers need a guarded main entry point and
memory/thread budgets per process. See [USAGE.md](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/USAGE.md#execution-settings).

## API and held-out positions

```python
from sensorsim.acoustics.room.fit import FitConfig, observe, fit_room, predict, compare

config = FitConfig(bootstrap=20)
# measured_rirs: mono room.rir.RIR, each with known source and receiver XYZ.
features = [observe(rir, id=f"train-{i}", config=config, reference=ref_rir)
            for i, rir in enumerate(measured_rirs)]
result = fit_room(features, initial_room_model, config=config)
rendered = predict(result, new_source_xyz, new_receiver_xyz, 48000, id="test-1")
# REF support-only filtering is also applied to the rendered prediction.
prediction = observe(rendered, id="test-1", config=config, reference=ref_rir)
measurement = observe(heldout_rir, id="test-1", config=config,
                      reference=ref_rir, split="test")
rows = compare(result, measurement, prediction)
```

`fit_room` accepts only `split="train"`. Repeats/orientations at the same XYZ pair
may not cross a split. `compare` rejects held-out positions used for fitting.
The simple CLI reserves the last `ceil(N/4)` unique receiver coordinates in
lexicographic XYZ order and keeps all takes/sources at those receivers held out.
At least two receiver positions are required. It is a deterministic convenience
split, not a spatially balanced sampling design. Use the API and a committed
dataset-specific split for validation campaigns. Identical positions round to
1 micrometre for leakage checks.

Main's pipeline keyword contract is also supported:
`fit_room(manifest=manifest, rirs=mono_rirs, initial_model=model) -> RoomModel`.
Supply stage-1 RIRs with `recording_file` metadata, in channel order within each
recording. The adapter verifies coordinates and channel counts, selects REF,
applies material priors, and uses the same held-out receiver split as the CLI.
It preserves the initial model's air conditions, wall scattering and late settings;
boxes use manifest geometry, material absorption priors and default scattering.
The full comparison report is
in `model.provenance['pipeline_fit_report']`. This adapter requires manifest
dimensions; the CLI or feature API handles missing-dimension inference.

The measured runner is
`tests/validation/bench_room_fit/run.py`.
Its protocol was committed
before measured comparisons. It handles MeshRIR blocked receiver holdout and
BRAS simple-reflection fallback; the legacy dEchorate route is withdrawn. It needs
the existing `h5dump` executable, not a new h5py/netCDF installation. Audio remains
in the ignored evidence folders under the datasets' own licences.
`--recover-reports` checks saved config, input hashes and complete metric keys,
then writes separate `*.recovered.json` diagnostics without changing fitted
models or metric rows. It is intended for interrupted report export.

## What is fitted

Known geometry has priority. `estimate_geometry` can infer missing dimensions
from early excess TOAs in a known corner frame; the generic stage-1 manifest
supports `shoebox_dims_m: null`. With known dimensions and local coordinates it
can instead infer the lower-corner translation. It uses labelled first echoes
when `Observation.metadata['wall_echoes_s']` is supplied; otherwise it performs
bounded multistart candidate matching across at least three positions. Missing
planes, rank deficiency and >2 ms assignment RMS cause an explicit failure.
Competing solutions are reported, not hidden. Coordinates stored in the model's
`coordinate_translation_m` provenance are applied by `predict`; callers using
`synthesize` directly must apply that translation themselves.

Jointly across supported bands, robust bounded least squares fits shared coefficients for identically
described surfaces. `floor`, `ceiling`, `walls` descriptions map to `z_min`,
`z_max`, and the four side walls; explicit `x_min` etc. take priority. Conservative
aliases/exact catalogue IDs select mounting-dependent absorption profiles from
the existing library. Unmatched or ambiguous descriptions use a labelled broad
0.2 prior. Catalogue population uncertainty is unknown; the regularization
scale 0.3 is an algorithm setting, not measured material variability. The
4000 Hz endpoint held at 8000 Hz is explicitly unvalidated.

The objective matches normalized EDC, EDT/T20/T30, C50, DRR and candidate early
reflection energies/timing. D50 is reported but not independently weighted
because it is algebraically determined by C50. Scattering and obstacle
absorption remain fixed priors; obstacle geometry is supplied. Optional per-band
late overrides are regularized toward training decay estimates. Unobserved
bands retain priors, never acquire a validation claim. Multiple starts and
iteration-limit status are recorded.

`room-fit/2` caches the actual linear spectral path kernels from `synth.py`,
including wall interpolation, continuous-frequency air loss, fractional delay,
obstacles and the early crossfade. Early pressure is summed coherently before
band filtering and squaring. The stochastic tail filters stationary noise before applying its decay/crossfade
envelope (R1). The shared `late.py` operator propagates the stationary covariance
through this output envelope and the analysis filter, retaining all neighboring
and edge synthesis bands. A squared cascaded-impulse convolution is no longer the
correct forward operator. Native sample expectations precede energy binning.
The 1 ms energy grid is retained for long decays;
tail-free reflection fits use native samples. DRR and early-pulse windows retain
their actual sample boundaries. Geometry is cached, not inversely ray-traced.
Late fit levels also include the resolved source and receiver diffuse-power
product used by synthesis. Obstacle path groups with embedded floor images are
re-parameterized per octave as the fitted floor absorption changes.

The coupled fit keeps all parameter bounds, objective scales and bootstrap
rules. `starts.parameters` contains each usable band's group coefficients in
configured band order, followed by its log T60 when an override is fitted.
`joint_cost` is the cost of that whole parameter vector. Spectral gains are
calibrated on training positions using the shared forward prediction.
Early bases are capped at 64 MiB per position and 1.5 GiB total, with an explicit
error if exceeded. Synthetic consistency guards require median eight-seed
EDT/T30 agreement within 10% and C50 within 1 dB; measured gates remain 5%/1 dB.
Training fast-versus-waveform differences are still reported because finite
stochastic realizations and time discretization remain. REF support filtering,
source directivity and measured-room accuracy require their own evidence.

Whole training XYZ pairs are bootstrapped with the same joint multistart fit,
preserving repeated takes together.
With fewer than three independent pairs, no interval is issued. Failed bootstrap
fits remain counted. The intervals condition on echo associations, geometry,
grouping, fixed scattering/obstacles, device assumptions and this model family;
they exclude structural error and are not calibrated predictive intervals.

## Timing, devices and validity

Supply `RIR.metadata['direct_sample']` when independently known. Otherwise the
first thresholded onset and a peak within the next 1 ms are used. Strong echoes,
obstacle shadowing and device ringing can violate this assumption. Raw onset and
analysis onset are retained. Shape metrics use the direct arrival as origin;
echo errors are **excess** TOA, not emission-time validation.

REF uses the exact-grid bounded inverse from `rir.estimate_reference_eq`, centred
as a finite FIR and applied by linear convolution with explicit pre-roll. Predicted
RIRs receive the same supported-frequency mask. Incomplete octaves are unavailable.
The default inverse gain cap is 12 dB. The product speaker+mic is identified only
approximately: near-field effects, source directionality, early REF reflections
and different recorder processing remain confounders. Without REF the chain stays
in the measurement; a per-band scalar is fitted on **training positions only**
for relative spectral-energy comparison. This is not absolute SPL validation.
An octave outside the declared ESS passband is also unavailable, even when a
truncated or noisy extracted response contains energy there.

Final metrics reuse exact base-10 octave filters and noise-aware ISO decay fits
from `metrics.py`. Missing measurements are UNAVAILABLE; invalid predictions
against valid measurements are FAIL. No zero filling, low-band promotion or
noise-floor fallback creates passes. Synthetic noise-free responses explicitly
use ordinary Schroeder integration. DRR's +/-2.5 ms window is flagged when an
early reflection contaminates it. All required decay/clarity/definition gates
must pass before a position/band is called within the frozen targets.

CPU is the reference because the existing SciPy optimizer/NumPy synthesizer have
no AMD inversion backend. No GPU speed or readiness claim; the full failed
sandbox environment check is retained. Rerendering the room-synth demo changes
`01_near_hard`, `02_far_clear_hard`, `03_behind_desk_hard` and
`04_behind_desk_softer`; the dry reference is unchanged. New fitted-room and
pipeline-generated clips also change. The older parent listening kit uses a
separate acoustic-node/scene path. No recognition-utility or moving-array
coherence claim.

## No-sweep ensembles and current fit cost

The primary surveyed-room route is described in [USAGE.md](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/USAGE.md).
`room.yaml` enables it automatically; optional exact playback/recording pairs
update a broad calibration ensemble. Assumed or weak-decay project results use a
separate domestic render assumption, and prior conflicts block rendering.
WP4(a) acoustic calibration is **not accepted**; engineering gates do not establish
predictive coverage on unobserved physical rooms.

The modular late-discrepancy implementation produces a research-only ensemble;
training promotion is blocked. Its software controls do not close WP4(a).

A 2026-10-10 end-to-end synthetic-proxy benchmark left main fit time
essentially unchanged at about 27.5 s. Parallel bootstrap timing was incomplete;
the existence of workers is not a measured main-fit speedup.

R1 renderer and fitter form one operator change. Historical dEchorate fit-cost
and accuracy figures are withdrawn. The default-cost failure and invalid
parameter gate remain in the internal review record; no parameter-accuracy
acceptance follows. The geometry cache budget is 1024 MiB and late-energy budget
1118 MiB; these are cache bounds, not process RSS limits. The early-basis limit
above is separate. Affected experiments need requalification.
