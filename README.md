# sensorsim-room

Expand indoor knock, doorbell and kettle training data using recordings of the
actual targets, surveyed room geometry and an explicit deployment microphone.
The aim is in-distribution augmentation for a small CNN on a Raspberry Pi 5.

A `room.yaml` survey enables a no-sweep route: room dimensions, loudspeaker and
microphone coordinates, and per-class target sites. Exact playback files paired
with in-room recordings can update a room ensemble; without useful calibration,
rendering uses a labelled domestic-room assumption. Chamber takes pass intake QA
and any required listening approval before rendering. The legacy sweep workflow
also remains available. A numpy-only runtime and an offline 3D page are included.

> **Research software. Deployment accuracy is not established.** User target takes,
> target-free deployed-microphone exposure and a locked real evaluation set are
> still needed. A simulator, measured-RIR proxy or software-test pass cannot accept
> target-room recall or false alarms per hour. The workflow has a **synthetic proxy
> rehearsal**: intake, training, evaluation and export ran; panel selection fell
> back and 30-minute calibration failed the 1/h exposure gate. Follow
> [USAGE.md](USAGE.md#end-to-end-workflow) for the inputs and required approvals.

| Document | Purpose |
| --- | --- |
| [USAGE.md](USAGE.md) | Inputs, intake, training, evaluation and export |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Fixed defects, original failures and current limits |
| [MATH.md](MATH.md) | Implemented operators, decisions and assumptions |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Licence record, external data policy and divergence |
| [Sweep kit](scripts/room_sweep_kit_templates/README.md) | Optional legacy sweep protocol |

## Pipeline

```bash
python -m sensorsim.project run my_project
python -m sensorsim.project evaluate my_project
python -m sensorsim.project detect my_project recording.wav
python -m sensorsim.project export my_project device
python -m sensorsim.project visualize my_project paths.html --source X Y Z --receiver X Y Z  # positions optional only with measured sweep takes
```

| Stage | Current behavior and scope |
| --- | --- |
| Room | Surveyed no-sweep ensemble or legacy sweep fit. Assumed/weak calibration does not become a measured room. WP4(a) acoustic calibration is **not accepted**. |
| Targets | Near-dry chamber intake, source/session grouping, contamination and clipping checks, and a session clap/pop reference. These heuristics still need real-take validation. Playback files calibrate the room; they are not project training targets. |
| Dataset | Render target sites with the room ensemble; keep recorded backgrounds in their captured domain. Position and source grouping do not establish independence of a physical evaluation set. |
| Device | Default `pi5_inmp441/1` and active-sample SPL convention. Device-recorded background is added in floating point without another capture-chain pass. Random codecs/AGC belong to stress mode. The 60–80 dB event-level prior is assumed. |
| Negatives | Automatic procedural background-confuser pool with a capped sampling share. Impact/tonal families require receipts; metadata checks cannot authenticate authorship or recover discarded ancestry. |
| Training | CNN plus joint checkpoint/detector selection on a validation panel, or an explicit uncalibrated fixed-detector fallback. New dynamics knobs are off; partial windows and hard validation are refused in `project run`. |
| Evaluation | Matched training-stream framing, group-based feature tests, exact standalone MMD permutation ranks and C2ST; weak evidence at practical take counts. Inspect wrong-class alerts as well as recall. WP2 separately scores continuous target-free audio and whole nuisance contexts with the deployed event tracker. |
| Export | `edge.py`, model, licence/notices, integrity hashes and a deployment guide. The device needs numpy; PyTorch is a host training dependency. |

The pre-R1 negatives screen did not establish a usable benefit without recall loss.
Its best arm still triggered on about 59% of proxy nuisance contexts at the **fixed
0.5 threshold**, not a calibrated threshold. An E2 pass is weak evidence: its fixed
out-of-fold bootstrap can undercover, and a below-chance interval is inconclusive.
WP2's `scored` status means sufficient declared groups, not validated prevalence,
calibrated uncertainty, or compliance with a deployment false-alarm budget.

The 2026-10-10 window-placement screen was a registered **FAIL** on synthetic
proxies, confirmed by independent audit. Doorbell-as-knock confusions fell in two
of three seeds, but legacy-model seed variance was as large and nuisance triggers
rose. Legacy placement remains default; deployment-matched placement is opt-in.
P5 synthetic-proxy power qualification failed at 4 groups per class and passed at
6–20 groups for its registered procedural defects only; see [Verification](#verification).

**Known limitation: doorbells.** The default (legacy) window placement puts long
sounds at the start of every training window, which biases long events such as
doorbells. In a 2026-10-11 synthetic-proxy workflow check, every doorbell take
(0/6) was detected as a knock, while knock and kettle scored 6/6. This is
synthetic-proxy evidence only; check doorbell recall on your own recordings.

## Install

```bash
git clone https://github.com/Pbjis2op4me/sensorsim-room
cd sensorsim-room
pip install .                    # the full room workflow
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install '.[nn]'
                                 # + PyTorch (CPU build, no CUDA), to train the CNN recognizer
pip install '.[recognizer]'      # + scikit-learn, to train the baseline recognizer
pip install '.[test]'            # + test dependencies
pip install '.[sofa]'            # + netCDF4, to read measured directivity from SOFA files
```

Python 3.12 or 3.13 for training. The base install needs six dependencies: numpy, scipy, pydantic, PyYAML, pyroomacoustics and threadpoolctl. **Obstacles do not need torch**, and
nothing in the room workflow imports it; only CNN training does. The device that
runs the trained CNN needs Python 3.9 or newer and numpy, nothing else.

**Name collision:** the import package is `sensorsim`, the same name as the
project it was extracted from. Don't install both in one environment. If you
integrate this into a larger codebase, consider renaming the package, which
means rewriting its internal `sensorsim.` imports.


## Verification

```bash
python -m pytest -q -p no:cacheprovider  # default excludes slow tests
python -m pytest -q -m 'slow and not acceptance' -p no:cacheprovider
```

Frozen `acceptance` campaigns require their own registration, custody and verifier
signoff. The default suite includes two P5 controls that spawn up to eight worker
processes; they do not run the P5 acceptance campaign.

**P5 synthetic-proxy acceptance (A5d, revision `5b788c1`):** One frozen attempt completed 11,400 cases without retry. The registered null/power criteria **failed at 4 groups** and **passed at 6, 10, 15, 18 and 20 independent groups per class per side**. At the passing counts, null rejections were 24/28/40/37/29 of 1,000 and defect detections were 294–300 of 300, meeting pointwise one-sided 95% Clopper-Pearson upper-null <= 0.075 and lower-power >= 0.80 bounds. Power is specific to `original-procedural-mechanisms/1` with +2 dB peaking EQ at 3 kHz (Q 1.5), +6 dB scalar gain and 20 ms frame-order reversal. Observed wall time was 22 min 53.93 s, within the 2 h target and all frozen caps; future runtime is not guaranteed. This is a synthetic mechanism result, with **no acoustic-realism, equivalence or deployment claim**. See [known issues](KNOWN_ISSUES.md) for the four-group limit and retained failures.

**Documentation verification, 2026-10-10, code base 506315b (Python 3.12.14):**
2,110 passed, 29 slow tests deselected, 1 historical expected failure in 152.86 s.
The complete default suite ran with host IPC; no worker test was excluded.
Slow regressions and frozen acceptance campaigns were not run in this docs pass.

These are software regression results on the named tree and environment. They do
not requalify historical acoustic or training experiments. Keep `TMPDIR` short
(about 70 characters or less) for spawned workers; sandbox IPC can still block
workers independently of path length. Historical xfails preserve failures.

R1 changed rendered samples and the fitter's forward operator. Render-based
WP3/4/6/7 evidence and the integration screens from before R1 remain historical
until their affected controls are rerun on the merged candidate. Historical
dEchorate fit-cost figures are withdrawn. The parameter gate was
invalid; no parameter-accuracy acceptance follows. See
[the failure record](KNOWN_ISSUES.md#programme-history-and-review-decisions).

## Runtime

A 2026-10-10 end-to-end benchmark on synthetic proxy workloads measured
about **1.06× CPU training**, **1.14× rendering** and **1.06× dataset generation**
versus the pre-optimization tree. Main room fitting stayed about **27.5 s**;
`bootstrap_workers` parallelizes bootstrap refits. These are descriptive host
timings, not performance guarantees or acoustic validation.

Explicit worker/GPU configurations offer larger gains: generation up to about
**5× at eight workers** versus serial, and opt-in GPU training about **2.7×**
versus CPU on the benchmark host. GPU results are not bit-identical to CPU.
See [execution settings](USAGE.md#execution-settings) for supported entry points
and determinism limits.

## Accuracy and limits

Historical dEchorate accuracy figures are withdrawn. The HDF5 recompute is
inconclusive and supplies no replacement accuracy table.

The parent's same-model tier (2.58% / 7.60% T30) measures self-consistency,
not accuracy, and needs a harness that is not distributed, so it was not re-run.

- **No physical phone-and-laptop session has ever been validated.** Synthetic
  capture and measured-RIR proxy checks do not establish deployment accuracy.
- The room model remains an approximation; early and late decay can differ.
  Low frequencies near or below the room's Schroeder frequency need particular
  care because individual modes can dominate.
- The 8 kHz band is fitted when the recording supports it, but not through a
  30 cm `REF` correction, which stops at 8 kHz; it then keeps the prior (about
  28% in the model-based walkthrough). Two corrections of the band edges were
  tried in the walkthrough and not adopted: filling 250 Hz and 8 kHz from their
  measured T30, and fitting 8 kHz through a wider `REF` correction. Both fixed
  T30 there but made clarity worse, and EDT at 8 kHz: one exponential tail per
  band cannot follow a room whose early and late decays differ
  ([KNOWN_ISSUES.md](KNOWN_ISSUES.md)).
- **Absorption priors are uniform in this distribution.** The fitter starts from
  the broad 0.2 prior it already defines for unmatched materials (it warns once),
  then fits to your measurements. Supply a catalogue you have rights to, to
  restore per-material priors (`SENSORSIM_ACOUSTIC_CATALOGUE`; format in
  [THIRD_PARTY.md](THIRD_PARTY.md#absorption-catalogue)).
- **Band metrics follow `room-measured-metrics/2`:** decay times from a
  time-reversed band filter, clarity from a zero-phase filter. Against the closed
  form they are exact (T30, Ts) or within 0.2 dB of clarity from 250 Hz up, 1.1 dB
  at worst at 125 Hz ([MATH.md](MATH.md#clarity-c50-c80-and-d50-per-octave-band-roommetricspy)).
  The fit's internal objective still compares measured and predicted values
  under the earlier causal convention, the same on both sides.
- **Rendered training data is not real data.** Validation on renders is
  optimistic. Measure a recognizer on real recordings made with the deployment
  microphone.

It reproduces an approximation of a room. Recovered audio still has substantial
spectral and waveform error.


## Third-party parts and licence

This source distribution is [Apache-2.0](LICENSE). It ships no third-party audio,
measured data, data tables or weights. Procedural audio is generated at runtime;
user inputs and optional external manifests retain their own provenance and rights.
The licence and NOTICE accompany device exports.

Core dependencies are NumPy, SciPy, Pydantic, PyYAML, pyroomacoustics and threadpoolctl. CPU PyTorch is optional for CNN training. Their native binaries have additional terms: OpenBLAS,
libgfortran with the GCC exception, LGPL libquadmath, and MPL Eigen are relevant
to redistributed environments. No bundled environment has been cleared. See
[dependency notices](THIRD_PARTY.md#runtime-dependencies).

dEchorate v1/v2 HDF5 and MeshRIR CC BY 4.0 records are **internal validation only**
under this programme. BRAS is ShareAlike and is internal-only, never a source for
shipping renders or training products. Legacy dEchorate SOFA licence scope remains
unconfirmed and excluded. No NC, ShareAlike or unclear-licence collection is an
approved shipping input. The metadata guard refuses declared restricted lineage;
an empty match is not licence clearance. Independent release and licensing reviews
ran before publication, and this repository is a single clean snapshot: the
development history, which contains removed material, is not published.

Historical internal figures from the legacy dEchorate edition are withdrawn; dEchorate HDF5 is used only for internal validation from the user's own CC BY 4.0 download.
