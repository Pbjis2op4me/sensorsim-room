# Known issues

Documentation snapshot: first public release, 2026-10-12. Commit identifiers cited in these documents
(for example 506315b or 5b788c1) refer to the project's private development history, which is not
published. Programme outcomes below follow the reviewed records; historical numbers are not newly measured on this tree. All
acoustic/recognition experiments cited here are **synthetic proxy** or explicitly
**measured-RIR proxy**, never deployment recordings.

Every confirmed defect from the parent project's 2026-10-03 code review that
lands in code shipped here, plus the dataset-contract defects its test suite
tracks as expected failures (`xfail`). For each item: whether this distribution
fixed it, and whether the room workflow (capture → fit → render → dataset)
actually reaches it.

"Reached" was established by running the room pipeline tests (including an
obstacle scene) and a full session → fit → dataset run under `coverage`, then
checking whether the code containing the defect executed. Where the defect also
needs a specific input to trigger, that condition is stated.

## Fixed in this distribution

The port records report failing-before/passing-after regressions; this documentation
pass does not rerun those original revisions. The first five are in
`tests/unit/acoustics_room/test_export_regressions.py`. The others are in
`tests/unit/acoustics_room/test_room_signal_round2.py` (signal and capture),
`tests/unit/room_ml_audit/` (`test_round2_room_ports.py`, `test_independent_review_fixes.py`,
`test_dataset_contracts.py`: fit, renderer and dataset),
`tests/validation/bench_room_closedloop/test_dry_recovery_{contract,e2e}.py`, and one file per
direct-API fix (`test_pcm_strict_clipping.py`, `test_atomic_json_modes.py`, `test_raycore_integer_ranges.py`).

| ID | Where | Defect | Effect on you before the fix |
| --- | --- | --- | --- |
| cr-room-model-04 | `room/fit.py` `estimate_geometry` | Integer coordinates truncated wall-image positions | A manifest written with whole numbers (`[1, 1, 1]`) failed geometry inference: "rank deficient" |
| cr-room-capture-01 | `room/ingest.py` `read_take` | FLAC bypassed the integer-PCM clipping rule | Clipped phone recordings saved as FLAC were silently accepted |
| cr-room-dsp-02 | `room/dry.py` `extract_dry` | An output buffer aliasing an input was only checked against recordings | Passing the noise-floor array as `output_buffer` silently zeroed the result |
| cr-room-capture-04 | `room/session_sim.py`, `dataset.py`, `fit.py`, `dry.py` | Source revision read from the caller's working directory, or from whatever repository contained the venv | Crashes outside a git checkout; inside another repository, that repository's HEAD was written into every dataset as the simulator revision |
| ML02 | `room/dataset.py` `_source_pools` | Validation recordings were drawn without regard to label | A multi-class dataset could validate on one class, and a class with a single recording could land **only** in validation, where it is never learned. Now every class with two or more recordings is validated, and single-recording classes always train. The parent's xfail test, which asserts the impossible single-recording case, remains |
| cr-room-model-01 | `room/fit.py` `_expected_all` | The fit's late-tail energy ignored source and receiver directivity | Up to +9.5 dB late-power bias for cardioid devices. The fit now includes both devices' diffuse power |
| cr-room-model-02 | `room/fit.py` `_cache` | Floor-reflection contributions inside obstacle paths were fixed at fit start | Fitted absorption near obstacles was biased. They now follow the fitted floor absorption in each octave |
| cr-room-model-03 | `room/synth.py` `resolve_config` | A saved `SynthConfig` lost which fields were omitted, overridden or disabled | Non-default directivity, or an explicit `early_scattering=None`, did not replay identically |
| cr-room-dsp-03 | `room/ingest.py` `SweepKit` | A custom kit's period was not checked against the playback's sample grid | Invented clock drift and an unneeded resample. An off-grid period is now refused when the kit is built |
| cr-room-dsp-01 | `room/rir.py` `deconvolve_repetitions` | Multichannel takes were aligned on one channel's arrival | A far channel's direct sound could be cropped away. Each channel is now searched near its own arrival; a channel with no supported packet there is refused rather than misaligned |
| cr-room-dsp-04 | `room/rir.py` `deconvolve_repetitions` | Clipping was checked in the loudest channel only | A clipped quieter channel produced a distorted response. Every channel must now pass |
| cr-room-capture-02 | `room/capture_qa.py` `compare_levels` | The two-level pilot ignored each level's own consistency check | The pilot could pass with a failed level. Reports are now `capture-qa/2`; `/1` reports still load |
| cr-room-capture-03 | `room/capture_qa.py` `assess_harmonics` | Harmonic and fundamental ratios used mismatched windows | Wrong harmonic ratios in the pilot |
| ML10 | `room/synth.py` default crossfade | For far-apart pairs in a small room, the default early/late handoff could precede the direct arrival | Dataset generation raised instead of rendering the pair. A too-early **default** handoff is now raised continuously to direct delay + crossfade/2 + 1/c, and the RIR metadata records both times; an explicit too-early `mixing_time_s` still raises |
| ML03 | `room/dataset.py` | The same audio supplied under two labels was accepted | Now refused before generation, naming both labels |
| ML04 | `room/dataset.py` | Silent or sub-LSB recordings were exported as positives | A row whose final PCM16 is all zero is now refused, naming the clip. Quiet sources remain usable when their exported PCM is nonzero |
| ML06 | `room/dataset.py` | In-memory arrays with equal numbers but different dtype or scale hashed as the same content | Identity now uses dtype-tagged raw bytes; non-numeric dtypes are refused |
| D2 (export review) | `room/synth.py` | Measured-directivity late power was indexed past the octave table for the Nyquist extension band | Renders with a measured device crashed at 8, 32, 44.1 and 48 kHz |
| cr-acoustics-2-01 | `acoustics/materials.py` (module since removed) | `dataclasses.replace` on a material shared the original's transmission cache | The copy returned the original's coefficients. Direct API only |
| cr-acoustics-3-03 | `acoustics/audio_boundary.py` `encode_pcm16` | Strict clipping used `abs`, which overflows at the signed integer minimum | That sample escaped the clipping check. Direct API only; dataset export does not call it |
| cr-core-run-05 | `core/geometry.py` `wrap_deg` (function since removed) | Integer angle arrays wrapped in their own dtype | Unsigned wraparound and signed-minimum overflow. Direct API only |
| cr-scene-gpu-03 | `scene/raycore.py` | `uint64` mesh indices, IDs and exclusion IDs were narrowed to `int64` unchecked | Values above 2^63−1 wrapped to negative values or the −1 "none" sentinel. Direct API misuse only; now refused |

Not in the review, also fixed: the base install no longer imports PyTorch (the outdoor
propagation engine that needed it is no longer shipped), and `fit.fingerprint()` no longer raises in an installed copy. The validation benches in
`tests/validation` no longer crash at the end of each scenario: they hashed parent-project protocol
documents that do not ship, and required PyTorch's version metadata; both are now recorded as absent.
Ported from the parent on 2026-10-08: the fit forms device references from `REF` sweeps only (real
and noise takes recorded at `REF` had been treated as reference sweeps), and the closed-loop benches'
dry recovery is bound to a fresh attempt, the selected session file, its input digest and sample rate
(cr-tests-harness-05, for the two drivers that ship). `atomic_json` now gives new files the mode
`open` would and keeps an existing file's mode; it had published every JSON file as 0600.

Historical, pre-R1 model-based walkthrough findings (subsequent device/training
changes supersede these as current performance evidence):

- **Multi-part sounds:** the event detector merged bursts only 0.15 s apart, so
  a knock became several events and no hint could select it whole. New per-take
  manifest option `event_merge_gap_s` (default 0.15, unchanged behaviour).
- **Level mismatch in training:** renders are stored about 22 dB below real
  recordings, so the first recognizer learned "quiet = target" and detected 0/6
  real events. The redesigned recognizer scales targets into real-level
  background and removes per-window gain. On the same stream it detected 3/6 with
  the same stress-level target data, and 5/6 with quiet, protocol-compliant target
  recordings (0 false alarms in both).

### Programme correctness repairs

| IDs | Fixed behavior / regression scope |
| --- | --- |
| WP5-D1 | Sine-referenced floor and high-frequency A-weighting corrected; original failed acceptance retained below. |
| WP5-D2 | Device-recorded bed bypasses recapture; events use the configured absolute-SPL chain. |
| WP5-D3/D4 | Training/persistence share the mixer; profile, split, reference and level settings are forwarded and archived. |
| WP5-D5 | Capture QA checks native mono 16 kHz S16, independent decoder parity and sensitivity tolerance; physical bit conversion remains not checked. |
| WP5-D6/D7/D8 | ICS 60 Hz corner; modeled INMP441 FIR passband; factual datasheet citations and warning alias. Other-part high-frequency filters remain unvalidated. |
| WP4-F1/F2 | Hardware-anchor opt-in separates pre-roll from propagation delay; domestic assumptions are separate from broad calibration support. |
| WP6 N1–N5/N7, L2 | Session-aware dryness, playback exclusion, review of spectrally different strays, diagnostic replay, drop-folder filtering, custom knock kind and content-bound approval. |
| WP7 gain / trace | Active-level partial compensation and stress-span compensation coexist; integrated independent decomposition controls catch dither and dropped-event mutants. These do not enable partial/hard project modes. |
| WP9 packaging / sources | Registry/schema ship as package data; R10 labels and procedural ADPCM steps replace removed tables. Appendix A coordinates and Vulkan are removed; IMA-style index adaptation is disclosed. |
| Polish-1 | Root target intake reruns, portable listening receipts, explicit YAML-date/PCM errors, calibration warnings and the 3-hour deployment recipe passed scoped independent review. |
| Window confusion reporting | CNN evaluation counts every detected class, including simultaneous correct/wrong alerts; recall no longer hides wrong-class firing. |
| T3 integrated contracts | Matched-negative bank identity/parent grouping, operational checkpoint-policy binding, complete-scene geometry and research-room training refusal, paired cue guards and JSON-stable parent-level diagnostics are implemented and integration-reviewed. Scientific efficacy remains separate. |
| MMD exact arithmetic | Standalone independent-group MMD and the P5 family use exact dyadic block sums and inclusive rational ranks; partition/complement ties are retained. Earlier MMD-dependent results are historical. |
| Retraining (2026-10-11 workflow check) | Retraining an existing CNN project (a setting change, new recordings or `--force train`) stopped with "use a fresh output path for a selection run". The ledger now records each training artifact; a retrain removes only those, while unchanged, and refuses anything else (such as a calibrated copy) with a move-aside message. A ledger written before this fix proves only the model file: any other training artifact present must be moved aside before retraining. |
| Missing stage outputs | Fit/dataset folder digests and evaluation.json must match their ledger entries. For training, the model file must exist (a calibrated copy is allowed) and every recorded secondary artifact must match its hash. Missing secondary artifacts trigger safe retraining; changed or unrecorded secondary artifacts are refused with a move-aside message. A stage folder without its ownership marker is refused until moved aside and is never deleted. |
| Evaluation threads | Evaluation ignored `recognizer.threads`; in a plain environment on a busy host it was stopped at 622 s versus 30.9 s with two threads (tiny C2ST linear-algebra calls on full-width BLAS pools). Pools above `recognizer.threads` (default 4) are now lowered while evaluating, via the new `threadpoolctl` dependency. |
| Calibration holdout | The automatic seed-hashed holdout could take `noise`, the only broadband playback, leaving an assumed room with a warning that asked for an existing file. Broadband noise groups are never held out automatically. If too few non-noise groups exist for the automatic holdout (a third of the groups), the fit stops with the requested and available counts; declare `holdout_groups` to choose. The assumed-room warning names the actual reason. |

## Programme history and review decisions

These are attributed programme/reviewer results, not acceptance emitted by this
documentation lane. Engineering fixes, invalid harnesses, historical failures and
scientific claims are separate. “Pass” below retains its original narrow scope.

| Work / record | Original failure or limitation | Later result and surviving scope |
| --- | --- | --- |
| WP5 floor and D1–D8 | Frozen sine-referenced floor **FAIL**, +3.13 dB versus ±0.5 dB (IEC-style check +4.02 dB). Device-recorded backgrounds were rechained. | Fresh correction proxy **PASS**, median floor error +0.344 dB, post-S16 −86.6557 dBFS(A), versus −87 dBFS(A) prequantization. Explicit sine reference and event-only capture preserve the bed. Old `pi5_mems_unfiltered` results are historical. |
| WP2 stress coverage | Original **FAIL** 2/3 stress cases; at the 0.35 setting brute force 12.333/h lay below 13.93/h [12.570, 26.172]. | Repair **PASS** 3/3 plus independent 3/3, rate ratios 0.839434–1.071071. Its 66-parent stress bank is now **inconclusive** under the later 100-parent sufficiency rule. Reviewer pass probability was only ~0.69 for an unbiased estimate and ~0.64 at 20% under-reporting. |
| WP2 misspecification | Proxy baseline biases: mixture −19.81%, superposition −100%, drift +60%, splitting −61.54%, smoothing +233.33%. | Oracle mitigations used known simulator truth. The study does not show that household prevalence or these mitigations can be learned from available data. A large upper bound is not calibration. |
| WP1 framing | Initial registered pass used a mis-sized null/raw-render path. The subsequent frozen F2 test **FAIL**ed 2/3 length controls; review found no length/drift effect, but only ~2.4% reachability for its pass/pass/pass rule under the null. | Preserve the strict historical xfail. Fresh prospective controls reported 0/20 null failures and 20/20 +2/+6 dB defects for each gate; a later reviewer found 5/40 null failures. Matched training-stream framing and per-test power gates are implemented; untrained low-variance TRUTH controls are not target-room validation. |
| WP3 L4 | Initial 0.500 path AUC compared identical inputs; the dither check reused the production path as its own oracle. The later exact-null rule **FAIL**ed seed 1703: AUC 0.491946, CI [0.484084, 0.498071]. | Floating-point bed-plus-event addition removes label-correlated re-quantization bias. The later [0.45, 0.55] equivalence rule was **outcome-informed**, adopted after those seed results were known; it is not blind confirmation. Registered default/slow suite interruptions remain FAIL records despite subsequent host completion. |
| WP3 admission | Impact/tonal generators, reuse and shortcuts were not evidence of realism; provenance-only approval was insufficient. | Listening, recorded-family receipts, hashes, all-zero-label training and mining locks are implemented. Forged declarations remain a limit; procedural API checks do not validate household sound distributions. |
| WP4(a) | Historical dEchorate coverage figures are withdrawn; the initial failure remains in the internal record. | **Not accepted**: missing configurations and broad intervals prevent acoustic qualification. Engineering timing/conflict/domestic-prior checks do not close it. |
| WP4 engineering | Unanchored pre-roll implied a false direct-delay constraint; broad/weak calibration priors could become domestic render assumptions. | Final proxy engineering controls pass; its registered addendum stayed **inconclusive overall** because suite timeouts prevented completion. Project uses separate calibration/render ensembles, explicit assumed/weak/conflict states and no timing anchor from pre-roll alone. The lower-level selector caveat remains open. |
| WP6 | Wet takes could pass and dry resonant targets could hard-fail; playback could become a target at the wrong site. | Near-dry intake, session reference, review-only intrinsic decay, playback exclusion and content-bound approval are implemented. Frame-row forwarding merged separately. Gain replay is opt-in, training-only gain × take, excluded from acceptance and C2ST; no measured learning-curve claim. |
| WP7 | Observer missed the Pi path; old knob proposals could mask validation or drop every kettle. A later gain oracle measured whole-span RMS after WP3 had adopted active RMS. | Final gain/pairing repairs use the shared active-level contract; review2 amended the oracle and checked correct/bad controls. Knobs remain off. Later panel selection supersedes synthetic-mAP checkpoint ranking. |
| WP8 recheck / R1 | Original join statistic **FAIL**ed even an unjoined ISM control; retain that result but it cannot diagnose a renderer discontinuity. Noise auto-estimation biased noise-free metrics. | Historical dEchorate recheck figures are withdrawn. R1 warms stationary noise, filters first and applies the envelope afterward, with shared fitter covariance. |
| WP8 registered join | **Model-based proxy:** new control-qualified paired-reference gate **FAIL**, including +1.839 dB at 125 Hz (base +1.770 dB). | Review attributed substantial low-band bias to the reference/direct-dominated window, not solely R1. A later post-direct diagnostic had zero 250 Hz coverage and was **inconclusive**. Do not erase either failure or claim per-position continuity. |
| WP8 cost sequence | Historical dEchorate fit-cost figures are withdrawn; the w8d, w8e and default-cost w8f **FAIL** records remain in the internal review. | The parameter harness was invalid because it rejected correct controls; no parameter-accuracy acceptance follows. |
| WP9 removal / packaging | Original licence result **INCONCLUSIVE** on source completeness; first packaging addendum **FAIL**, register omitted from wheel. | R10-generated bands, procedural ADPCM-like steps, deleted Appendix A/Vulkan and packaged registry/schema address the identified source/package issues. Index adaptation is still IMA-style. Four stripped-ancestry probes remained undetectable in the 22-case attack. |
| WP9 matcher | Overbroad matching rejected legitimate names/paths, including common words. | w9d verifier accepted four scoped claims with benign training controls and mutants. Matching is field-sensitive; remaining false positives/misses and unauthenticated ancestry are listed above. This is not blanket licence clearance. |
| int-eval calibration | Old calibration applied the requested budget per class and returned its last grid threshold without an infeasibility status (review examples 3 and 6 events/h for a requested 1/h). | Combined start-alert budget, exhaustive grid evaluation and explicit infeasible/insufficient-evidence statuses are implemented. This selects development thresholds; it does not measure deployment error. |
| int-eval OVERLAY | Implementer's raw pass was **rejected** by the verifier: calibration excerpts and saved-model digests escaped the partition check. | int-eval-2 verifier accepted five claims in one frozen run (61 controls, 16 falsification probes): model digests, ranges and exact block matching cover both plain/overlay paths even with `calibration_streams: []`. Transformed/short audio remains outside the exact-hash guarantee. |
| int-train | Acceptance **not run / inconclusive**: the lowest panel nuisance-trigger fraction over all checkpoints and the ten detector policies was 0.50, so the 0.10 trigger budget was unreachable; absent default manifests broke an integration path. | int-train-2 adds usable defaults and a declared terminal fallback without weakening the budget. Its verifier accepted software integration claims, not a negative-audio benefit. |
| Negatives × partial screen | Twelve pre-R1 proxy runs: 11/12 panel selections infeasible; all screen arms used terminal models. Ungated negatives gave no nuisance improvement. | The "all" arm, whose impact/tonal negatives entered only through unreviewed `proxy_screen_only` receipts that production refuses, reduced nuisance triggers ~32–34 points versus ungated (partial windows off) with 7.3–11 points recall loss. Best nuisance rate ~59% at fixed 0.5, **not calibrated**. Partial windows restored kettle recall but raised swaps; boundary-effect power ~24–26%. No eligible production configuration was demonstrated. |
| T3 integration | Earlier negative-bank/parent grouping, selector identity, research-room training, cue-pairing and parent-diagnostic JSON controls failed independent review. | Repairs integrated through int-t3 and int-pel passed scoped software review; original rejections remain historical. Matched confuser cue pairing and full-parent training stay refused. No deployment benefit is accepted. |
| Window placement screen, 2026-10-10 | Legacy long-event onset placement creates a class shortcut. The opt-in deployment-matched policy was screened under a frozen synthetic-proxy protocol. | Registered **FAIL**, confirmed by independent audit (demonstrated nuisance-trigger harm). Doorbell-as-knock confusions fell in two of three seeds, but legacy-model seed variance was as large and nuisance triggers rose. Legacy remains default; `deployment/1` remains opt-in. |
| T3/P5 MMD ties | A disclosed four-group case returned p=0.003 after losing partition/complement ties; the independent rational result was p=0.028. | Exact repair merged via `921dadd`, independently reviewed for integration. Standalone MMD p-values and affected family decisions from earlier revisions are historical; no retrospective population-power acceptance follows. |
| P5 synthetic-proxy power | Earlier invalid control addresses, MMD ties, NumPy-alpha behavior, collector failures and A5b/A5c guard rejections remain historical failure records. | A5d completed one frozen acceptance attempt on `5b788c1`: **11,400/11,400 cases**, no retry. At 4 independent groups per class per side: **FAIL**, with 0/300 detections for each defect; the exact complement-symmetric first-step permutation resolution is insufficient. At 6/10/15/18/20 groups: **PASS** under the registered pointwise one-sided 95% Clopper-Pearson criteria (null upper bound <= 0.075; each defect lower bound >= 0.80). Null rejections were 24/28/40/37/29 of 1,000; detections were 294–300 of 300 per defect. This qualifies power only for `original-procedural-mechanisms/1` and +2 dB peaking EQ at 3 kHz (Q 1.5), +6 dB scalar gain and reversal of 320-sample frames at 16 kHz. Observed wall time was **22 min 53.93 s**, within all frozen caps and the 2 h target; this is descriptive timing, not a throughput guarantee. **Synthetic proxy only: no acoustic-realism, equivalence or deployment claim.** |
| Optimization round | Per-lane timings were partial workloads; some timing targets failed or remained inconclusive. The opt-render review found a surviving rounding mutant in its equivalence harness. | CPU route/worker controls supported integration; 2026-10-10 end-to-end timings are descriptive: training ~1.06×, render ~1.14×, dataset ~1.06×, main fit unchanged (~27.5 s). Parallel bootstrap timing was incomplete. Larger worker/GPU gains require their explicit configurations; none establish acoustic accuracy. |

R1 invalidates affected render-path receipts until controls run on the merged
candidate. Software regression completion is not a fresh scientific acceptance.
Historical dEchorate figures are withdrawn; walkthrough figures remain dated pre-R1 proxy results. The
original implementation/review records are cited by lane names in THIRD_PARTY.md;
they are internal evidence records, not links to unavailable parent documents.

## Removed from this distribution

| ID | Severity | Where | Why it is gone |
| --- | --- | --- | --- |
| cr-core-run-01 | **HIGH** | `core/configuration.py` | Hydra imported external packages before validating the config source. The room workflow never used scenario composition, so the module is not shipped, and neither is the scenario schema itself. |
| cr-scene-gpu-04 | MEDIUM | `scene/raycore.py`, `scene/raycore_vulkan.py` (file since removed) | The electro-optical ray adapters (`_ray_box_intervals`, `ray_closest`, `ray_shadowed`), their query contracts (`eo-entering-ray/1`, `eo-shadow-ray/1`, `optical-patch-ray/1`), the backend manifest's `eo` note and Vulkan query modes 3 and 4. Only the parent's camera code called them. The finite-segment queries that obstacle rendering uses are unchanged. |
| export-vk-01 | Removed | `scene/raycore.py` | The incomplete Vulkan backend and its undeclared `slangpy` import were removed. `auto` resolves to CPU; an explicit `vulkan` request raises a clear `ValueError`. CPU finite-segment queries retain their outputs. |
| audio-only-scope | Partially removed | `core/schema.py`, `core/messages.py`, `acoustics/scene_paths.py`, `acoustics/audio_boundary.py`; removed `core/model_registry.py`, `core/units.py`, `core/rng.py`, `library/`, `scene/building.py`, `scene/materials.py`, `acoustics/propagation.py`, `acoustics/interior_reverb.py`, `acoustics/materials.py`, `acoustics/ground.py`, `acoustics/contracts.py`, `acoustics/levels.py`, `acoustics/room/energy.py` | Drone-scenario, Remote ID, RF, camera and radar declarations were removed, and so was the outdoor moving-source scene simulator with its propagation engine, interior reverb, ground reflection and transmission-loss material presets (open findings cr-acoustics-1-01/02/03, cr-acoustics-2-04 and AD-02 went with them). `core/schema.py` retains only the obstacle-box geometry (`SceneOpening`, `SceneBox`) used by the room renderer, and `core/clock.py` only the `SampleTimeMap` that audio units carry (the recording-clock simulator `ClockSpec`/`SensorClock`/`TimeSynchronization` and the `SceneSpec` scenario wrapper are removed); `core/messages.py` retains `ModelValidity`; `acoustics/scene_paths.py` keeps the two obstacle diffraction routes the room renderer uses; `acoustics/audio_boundary.py` keeps the PCM16 adapters. Unused drone-era helpers also went: compass azimuth/elevation and angle-wrap utilities, the node pressure-stream type, the run-manifest writer, three outdoor barrier formulas, box-interior volume helpers and the graphics-stack probe in the environment record. The RF/EO model registry, material catalogues, the descriptive scene material-name map and library-backed buildings are not part of the audio training system. The selectable wifi_camera_g711 profile in iot_mic.py models microphone capture effects only (AGC, noise suppression, G.711 and packet loss); it does not simulate a camera, optical path or radio propagation. |


## Open — reached by the room workflow under specific conditions

The original port-review defects above are fixed. The programme still reaches these issues:

| ID / area | Open issue and next safe action |
| --- | --- |
| WP4(a) | **Not accepted.** Predictive acoustic calibration lacks adequate full-room coverage and narrow calibrated intervals. Missing reflective cases cannot be discarded. Collect appropriately long, supported recordings and qualify an independent oracle. |
| WP8 join / decay | R1 fixes envelope ordering, not arbitrary per-position continuity or non-exponential decay. The registered join FAIL remains; a later conservative post-direct diagnostic was inconclusive at low frequencies. The S08 decay-shape work is pending. |
| WP8 cost | Historical dEchorate fit-cost figures are withdrawn. The default-cost failure remains unresolved; fresh permitted inputs and a qualified cost protocol are required. |
| WP8E-NUM | Modal covariance numerics differ from a native real-SOS oracle. Historical fitted-dEchorate sensitivity figures are withdrawn; the numerical defect remains open. |
| WP1 / P5 | E2 bootstrap undercoverage and CV instability remain open. Peak framing can choose noise; there is no annotation-span adapter. A5d synthetic-proxy power qualification fails at 4 groups and passes at the tested 6/10/15/18/20 groups per class per side, only for the named procedural generator and registered effects. These results do not establish power for real takes or other defects, acoustic realism, deployment recall or measured false alarms. Changed code dependencies or acquisition/generator/effect scope require new qualification; future tuned statistical confirmation requires fresh cases under a new registration. |
| WP2 | Nonconstant bootstrap coverage is approximate (roughly 1.6× nominal alpha under skew); parent independence, prevalence and missing nuisance families remain assumptions. No measured deployment exposure has been supplied. |
| Training / negatives | Automatic panel often falls back; the proxy screen found no eligible nuisance improvement without recall loss. Impact/tonal admission and real-take learning curves remain unvalidated. Partial/hard modes stay refused. |
| Device / level | Assumed 60–80 dB target prior has no low tail unless the opt-in experiment is enabled. Source-take noise scales with target gain. Pi FIR taps/phase, other MEMS filters and hardware bit conversion are not physically verified. |
| Intake / background | Proxy dryness heuristics need real validation; project training still requires recorded backgrounds. Complete-scene geometry is now resolved before descendants are emitted; other generation failures can still leave an explicitly incomplete dataset. |
| Integration | The low-level render-ensemble selector can retain a supplied weak-decay posterior; the project route uses the domestic assumption. The 3-hour calibration recipe and insufficient-evidence refusal passed scoped polish-1 review. |
| Calibration disjointness | Exact hashes do not find gain/resample/dither variants or every short crop; legacy models may lack ancestry. Calibration now warns about missing recording IDs and nonnative sample rates; declarations still need review. Parent IDs are still declarations. |
| Licensing guard | Stripped/forged provenance cannot be cleared. Reviewed repairs cover BRAS DOI/archive identities, incidental folder names and versioned declared ancestry such as `dEchorate/v1`. Explicit provenance remains restricted even with slash/encoded spellings; source-file fields retain the narrower identifier rule. Distinctive identifiers can still false-positive. The v1 HDF5 source row is present in the packaged register; its evidence is historical and no fresh publisher verification is claimed. |
| Licensing / provenance (L1) | Historical dEchorate figures are **withdrawn**, and legacy SOFA validation loaders are removed. The HDF5 recompute is **inconclusive**; no replacement accuracy table is justified. The 2026-10-11 independent licensing review's findings were addressed in that review's snapshot; see [licences/README.md](licences/README.md). This does not clear later publication checks. |
| Multiprocessing | Keep TMPDIR about 70 characters or shorter. The review reproduced a 76-character socket-path failure and a 32-character pass; sandbox IPC can independently hang even with a short path. DataLoader has no timeout. |
| Training execution | CPU remains default. Opt-in `stream/2` preserves fixed-seed, fixed-batch CPU results across worker counts; legacy `stream/1` retains worker-dependent model RNG. GPU training is opt-in and non-bit-identical, available through `nn.train(device=...)` only; project/CLI device wiring is absent. |
| Model file hashes | `model.npz` records training wall time, so models with identical weights (for example `stream/2` at different worker counts) have different file hashes. Compare arrays, not file hashes. |
| Doorbell bias (default window placement) | Legacy placement puts long sounds at the start of every training window. The 2026-10-11 synthetic-proxy workflow check detected every doorbell take (0/6) as a knock; knock and kettle scored 6/6. The deployment-matched placement is opt-in and its screen FAILed. Check doorbell recall on your own recordings. |
| Research integration limits | T03 modular discrepancy/calibration is implemented but research-only, with training promotion blocked. T06 parent-event contracts and lower-level mixer/metadata hooks are implemented; the full-parent/context/confuser/window/persistence adapter remains absent. T05 paired known-playback-versus-render evaluation remains proposed. Legacy active-level normalization stays default. |
| Release / evidence | Synthetic proxy rehearsal completed the route, with fixed-detector fallback and insufficient 30-minute calibration exposure preserved as failures. Affected pre-R1 evidence reruns remain pending. Independent release and licensing reviews ran before publication; this repository is a single clean snapshot, and the development history (which contains removed material) is not published. |


## Open — not reached by the room workflow

None. The earlier entries concerned outdoor receivers, moving sources and building volumes;
that code is no longer shipped (see `audio-only-scope` under Removed).

## Workflow limits

| Area | Current limit / next action |
| --- | --- |
| Legacy target extraction | Automatic matching can miss noisy broadband events or include another sound in a decay tail. Prefer declared chamber intake and inspect/listen to its QA. |
| Model structure | Per-surface absorption is weakly identifiable; scattering and several geometry/device assumptions are fixed. Early/non-exponential decay, directional scattering and modal behavior remain approximations. |
| Legacy acquisition | Whole `real` takes in legacy manifests can contain silence or contamination. Use explicit event hints and the recording protocol. |
| Timing | Direct-path recovery assumes a detectable first arrival. An obstructed or colored speaker chain can violate that assumption. |
| Simulated devices | Stress codecs, AGC, noise suppression and independent packet loss are approximations, not validated physical devices. The Pi filter's vendor taps/phase are not measured. |
| Platforms | Spawn-mode Linux checks do not establish macOS/Windows operation. Memory and IPC remain environment-sensitive. Benchmark inference on the actual Pi. |
| Microphone input | The runtime requires PCM, not a GPIO/I2C loudness reading. `edge.py` reads integer-PCM WAV and refuses sample rates that are not an integer ratio of the model rate (for example 44.1 kHz for a 16 kHz model); record with `arecord -r 16000 -f S16_LE` or resample first. Check capture bit alignment and native rate independently; per-window DC removal does not repair incorrect PCM scaling. |
| Visualization | Third-order paths and a tail echogram do not depict every reflection or the physical diffraction route. |

## Plug-in data that does not ship

| Area | Limit | What to do |
| --- | --- | --- |
| Room-fit priors | Without an absorption catalogue, every surface starts from a uniform 0.2 prior (a warning says so once). | Install a catalogue you have the rights to: [THIRD_PARTY.md](THIRD_PARTY.md#absorption-catalogue). |
| Legacy dEchorate SOFA / BRAS CR2 benches | Historical internal figures from the legacy dEchorate edition are withdrawn; dEchorate HDF5 is used only for internal validation from the user's own CC BY 4.0 download. BRAS internal acquisition/scoring runs with restricted lineage, but CR2 lacks verified room dimensions and remains outside room-geometry qualification. | Retain BRAS's geometry exclusion and internal-only status. |
| Bench integrity | `tests/validation/pin_data.py` pins whatever you downloaded; the parent project's verified hashes are not included. | Download from the records named in THIRD_PARTY.md. |

## Accuracy limits (not defects)

See [README.md](README.md#accuracy-and-limits) and [USAGE.md](USAGE.md#accuracy).
Historical dEchorate accuracy figures are withdrawn. The HDF5 recompute is
inconclusive and does not establish current acoustic accuracy. Rendered validation remains
optimistic and no target-room deployment claim has been accepted.

Conventions measured against closed-form cases and open-source tools ([MATH.md](MATH.md)):

| Area | Limit | What to do |
| --- | --- | --- |
| Band metrics | `room-measured-metrics/2`: decay times from a time-reversed band filter (T30 exact on exponential decays; EDT within 3.8% with a direct sound up to +6 dB), clarity from a zero-phase filter integrated from one filter span before the onset (within 0.2 dB from 250 Hz up; 1.1 dB at worst at 125 Hz, T60 0.15 s). Numbers computed under the earlier causal convention (`/1`, still selectable) read low-band clarity up to 9 dB low. | Pass `policy=` to reproduce earlier numbers; label which convention a figure uses. |
| Fit objective | The fit still compares measured and predicted responses under `/1`, the same causal filter on both sides, so its residuals are consistent; reports and benchmarks use `/2`. The direct-to-reverberant ratio below 500 Hz is a weak quantity in any convention: its ±2.5 ms window is shorter than the band filter's response. | The default fit bands are 500 Hz to 4 kHz; 250 Hz and 8 kHz join as support bands when the recording supports them. |
| Band edges | The late tail is one exponential per band. Where a room decays fast early and slowly late (absorbing ceiling and floor, reflective walls), two corrections tried in the walkthrough fixed T30 at the band edges but made clarity worse there, and EDT at 8 kHz: filling 250 Hz and 8 kHz from their measured T30, and fitting 8 kHz through a `REF` correction widened past 8 kHz. Neither was adopted, so the edge bands keep their prior when a 30 cm `REF` correction does not cover them (it reaches neither 250 Hz nor the whole 8 kHz octave). | No edge-band correction is adopted; the edge bands keep the prior when a 30 cm REF correction does not cover them. A decay-only fit is a possible extension. |
| μ-law stand-in | A continuous μ = 255 curve, not G.711's segments: SNR within 3 dB of the real codec, samples up to 1084 LSB apart at full scale. | Treat simulated G.711 as an approximation. |
