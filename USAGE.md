# Usage: indoor recognition training

## End-to-end workflow

The rehearsal completed intake, rendering/training, evaluation and device export
on a **synthetic proxy** drop folder. It did not establish deployment accuracy:
the small stand-in panel fell back to a fixed detector, and 30 minutes of
calibration correctly failed the combined 1/h exposure requirement. Human listening
was represented by explicit automated fixture decisions, not real approvals.

Keep the four recording roles separate. Have the evaluation protocol allocate
independent development and locked evaluation takes before tuning. Example layout:

```text
my_project/
  room.yaml
  project.yaml
  calibration.yaml                    # optional known-playback settings
  dry/                                # development targets only
    knock_01.wav
    doorbell_01.wav
    kettle_01.wav
    session_reference_2026-10-09.wav
  played/noise.wav                    # exact loudspeaker playback
  room_recordings/noise__room.wav      # recorded speaker-room-mic response
  background/train/                   # independent target-free device sessions
  background/val/
  calibration/background.wav          # separate threshold-development allocation
  evaluate/                           # untouched target takes and target-free audio
    manifest.jsonl
    data_kind.txt                     # proxy for generated fixtures
  wp2/manifest.json                   # disjoint floor/challenge/overlay allocation
  out/                                # generated; never a dry target
```

1. Fill in `room.yaml`, including actual per-class `source_sites` below. The
   loudspeaker position describes calibration, not where a door or kettle emits.
2. Give every development target a `dataset.sources` row. Quote the session string
   and name its clap `session_reference_<session>.wav` in the target directory.
3. Run `python -m sensorsim.project run my_project` to produce `out/source_qa.json`.
   Listen to each crop, inspect flags, enter the current approval key and rerun.
   Missing sites stop before crop approvals; hard rejects cannot be approved away.
4. If using gated impact/tonal negatives, prepare and finalize listening receipts
   with the audition tool below and attach them to the frozen negative manifest.
5. Train with independent background sessions. Check whether detector selection
   reports `fixed_fallback`; a completed training run does not qualify a panel.
6. Convert edge inputs to integer PCM without changing gain. Calibrate on its
   separate allocation, evaluate on untouched recordings and run WP2 with its
   explicit manifest. Calibration refusal writes no replacement model.
7. Export the selected model and verify the six-file device folder. Keep source
   recordings, allocation manifests, approvals and evaluation evidence with the run.

## Install and run

```bash
git clone https://github.com/Pbjis2op4me/sensorsim-room
cd sensorsim-room
pip install .
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install '.[nn]'
python -m sensorsim.project status my_project
python -m sensorsim.project run my_project
```

The import package is `sensorsim`; do not install the parent project into the same
environment. Stage caches fingerprint inputs and implementations. Use
`--force fit`, `--force dataset`, `--force train` or `--force evaluate` when needed.
Keep `TMPDIR` short (about 70 characters or less) for multiprocessing socket paths.
This does not cure sandbox IPC restrictions. Training has no worker timeout; a
worker hang needs diagnosis, not an assertion that a shortened path fixed it.
For a constrained host set `recognizer: {type: cnn, threads: 2, workers: 0}`.
`threads` caps PyTorch during training and the native BLAS/OpenMP pools during
evaluation (default 4). Changing a recognizer setting, adding recordings or
`--force train` retrains in place: the previous run's model files are replaced only if
training wrote them and they are unchanged, so keep a calibrated model under another
name (as in the calibration recipe below) before retraining.

### Execution settings

Set `recognizer.workers` in `project.yaml` (or `--workers` in the standalone
`sensorsim.nn` CLI): `0` is serial; positive counts use generation workers.
The automatic count reserves two available CPUs and caps workers at eight.
For CPU model weights that reproduce across worker counts at a fixed seed,
batch size and environment, explicitly select the new stream:

```yaml
recognizer: {type: cnn, threads: 2, workers: 8, stream_version: stream/2}
```

`stream/1` remains default: it preserves its historical worker-dependent model
RNG. `stream/2` uses per-example seeds and a private loader RNG; it changes samples
and weights relative to `stream/1`. Changing streams is not an execution-only
replacement for a frozen experiment. Changing batch size also changes training.

GPU training is a Python API option: `nn.train(manifest, out, device="cuda",
threads=2, workers=0)` explicitly requires a CUDA/ROCm-capable PyTorch build.
`device="cpu"` is the default; `device="auto"` permits a recorded CPU fallback.
Import it with `from sensorsim import nn`. Neither `project.yaml` nor the
standalone training CLI currently accepts a `device` setting. GPU results are
not bit-identical to CPU; exported inference still uses NumPy on the CPU.

For the legacy sweep fit, `fit: {fit_options: {bootstrap_workers: 2}}` in
`project.yaml` enables parallel bootstrap refits. The API is
`fit_room(..., bootstrap_workers=2)`; the default is `1`. The main fit stays
serial. Ordered draws, optimizer starts and fitted outputs are preserved at a
fixed seed/environment; worker errors propagate. Python callers need a guarded
`if __name__ == "__main__":` entry point. Budget memory per process and native
threads separately. The fit CLI has no `--bootstrap-workers` flag.

A 2026-10-10 descriptive end-to-end timings on synthetic proxies were
about 1.06× for default-path CPU training, 1.14× for render and 1.06× for dataset
generation; main fit stayed about 27.5 s. Explicit generation-worker tests reached
about 5× at eight workers versus serial, and opt-in GPU training about 2.7× versus
CPU. These workload-specific host results do not measure Pi inference or acoustic
quality; per-lane microbenchmarks are not end-to-end speed claims.

For software regression checks use `python -m pytest -q -p no:cacheprovider` and
`python -m pytest -q -m 'slow and not acceptance' -p no:cacheprovider`.
The default suite includes two P5 controls that spawn up to eight worker processes.
Frozen acceptance campaigns require separate verifier authorization and custody.

## Surveyed room and optional noise playback

A root `room.yaml` is detected automatically. Do not add a `room_yaml` key to
`project.yaml`; unknown project keys are rejected. Example geometry **is assumed**:

```yaml
units: m
origin: room floor corner; z up
room: {length_x: 5.0, width_y: 4.0, height_z: 2.5}
microphone: {x: 3.0, y: 2.0, z: 1.2, tolerance_m: 0.05}
loudspeaker: {x: 1.0, y: 1.0, z: 1.0}
source_sites:
  knock: [{x: 0.02, y: 1.0, z: 1.2, spread_m: [0.0, 0.2, 0.2]}]
  doorbell: [{x: 0.05, y: 2.0, z: 1.5}]
  kettle: [{x: 4.0, y: 1.0, z: 0.9}]
prior: {count: 32, seed: 4108}
```

Replace these with your measurements and actual class labels. Sites and their
spreads must remain strictly inside the room, outside obstacles; microphone and
loudspeaker must be at least 5 cm apart. Door states and furnishing tags document
assumptions; they do not create openings or detailed furniture acoustics.

No sweep session is required. Put the exact played file at `played/noise.wav` and
the deployed microphone recording at `room_recordings/noise__room.wav`. A roughly
30 s broadband-noise playback with recording pre-roll and at least 2 s of decay
is a recommendation, not a validated minimum for every room. Other paired files
use the same `name.wav` / `name__room.wav` naming. A playback file is neither a dry
target nor a measured RIR. Calibration describes the speaker-room-microphone chain.

Optional `calibration.yaml` accepts `recording_offsets_s`,
`recording_offsets_hardware_anchored`, `impulses`, `holdout_groups` and
`evidence_label`. Pre-roll offsets alone do not measure direct-path delay. Claim a
hardware anchor only when DAC/ADC and buffering latency are independently known.
Unsupported excitation bands abstain; clock alignment precedes decay extraction.
Conflicts block rendering and request review; do not broaden the prior after a fail.

The fit outputs separate `calibration_ensemble.json` and `room_ensemble.json`.
Known-source calibration retains its posterior; assumed or weak-decay calibration
uses an explicit domestic render assumption in the project route. The broad
calibration prior is not the domestic render prior. Default domestic T30 ranges,
125 Hz through 8 kHz, are 0.35–1.20, 0.30–1.00, 0.25–0.90, 0.25–0.90,
0.20–0.75, 0.15–0.60 and 0.10–0.45 s. These are engineering assumptions, not
household population estimates. Custom calibration bounds do not override this
fallback. **ROOM IS ASSUMED, NOT MEASURED** remains the applicable warning.
WP4(a)'s acoustic calibration has not been accepted; read the coverage/missing-data
record in [KNOWN_ISSUES.md](KNOWN_ISSUES.md#programme-history-and-review-decisions).

The optional legacy sweep path uses `scripts/room_sweep_kit.py` and its
[recording protocol](scripts/room_sweep_kit_templates/README.md). A held-out
`role: check` take must not set fitted parameters, anchors or pooled levels.
Historical dEchorate fit-cost and accuracy figures are withdrawn. Inspect resource
limits when fitting; the default-cost failure remains in the internal review record.

## Chamber targets, QA and listening approvals

Follow [the capture protocol](src/sensorsim/sources/CAPTURE_PROTOCOL.md): aim for
at least 18 independent takes per class, native mono audio, fixed gain, close
capture (at most 30 cm), at least 0.5 s quiet lead and complete decay plus 1 s.
Do not normalize or denoise takes. Record a clap/pop in each chamber session.
Name the session reference `dry/session_reference_2026-10-09.wav` and use
`session: '2026-10-09'` on its source rows. Intake discovers only that exact
session's clap, hashes it and assesses its decay. Keep at least 0.2 s quiet before
the clap and 0.5 s after. Other names or quiet windows require an explicit
`session_reference: {file: clap.wav, session: '2026-10-09', evidence_label: measured,
quiet_window_s: [0.0, 0.2]}` relative to the targets directory. Discovery does not
approve any crop. Review its qualification and the take's original noise floor.

Declare `dataset.sources` in `project.yaml`. A surveyed project automatically
uses `dry/` when present, unless `targets` is explicit. Paths below are relative to
that directory; list every development target, not held-out evaluation parents.

```yaml
targets: dry
dataset:
  count: 60
  fs: 16000
  sources:
    - file: knock_01.wav
      label: knock
      source_group: doorA-session1-take01
      object_id: doorA
      session: '2026-10-09'
      capture_route: near_dry
      evidence_label: measured
      quiet_window_s: [0.0, 0.2]
recognizer: {type: cnn, workers: 0, threads: 2}
```

`targets: .` also works: prefix rows with `dry/` and explicitly reference the clap
as `dry/session_reference_2026-10-09.wav`. Intake and its cache fingerprint exclude
`out/`, `evaluate/`, `session/`, calibration playback and background folders,
including configured output/evaluation/background/session paths. Those excluded
files cannot be declared as dry targets. Keep other non-target WAVs, such as
threshold-calibration audio, outside a root-wide target intake or use `targets: dry`.

Use `evidence_label: proxy` on generated fixtures. Declare all target files and
keep object/session grouping intact across splits. `played/`, `room_recordings/`
and `background/` are excluded from target discovery. Project targets refuse
`known_digital` and loudspeaker-playback rows; keep these for calibration/comparison.
`in_situ` takes are evaluation-only and are not rerendered as dry targets.

Inspect `out/source_qa.json`; `python -m sensorsim.sources.intake my_project/out/source_qa.json`
prints reviewable approval keys. After listening, a source row can carry
`approval: {reviewer: NAME, decision: approve, approval_key: KEY}`.
The resulting report stores the approval under `listening_approval`.
Keys bind the audio and QA settings; do not reuse a key after editing them.
Clipping, contamination, truncation, low SNR/SIR and a session reference classified not dry are hard rejects that approval cannot waive; an unavailable session decay is only a review item.
For custom knock labels set `kind: knock`. `heterogeneous_hits: true` is a knock-only
review path for differing taps; it does not waive explicit contamination.

The 0.20 s take-decay heuristic is review-only: a resonant dry door or chime is not
proof of a wet room. A qualified session clap supplies the hard session gate; for knock-kind takes, decay above 1.5× the reference requests review. Short references may have no fitted
T20 and use a conservative comparison bound. These are proxy-developed heuristics,
not physically validated dryness classifiers. Gain replay (`sources.replay`, opt-in, not wired into `project run`) emits gain × a training take into the training split only; it adds no acoustic independence and is excluded from WP6 acceptance and C2ST evidence.

## Backgrounds, negatives and training settings

Supply target-free deployed-microphone background in independent recordings for
train and validation. The project still requires recorded background in both
splits; automatic procedural confusers do not implement a background-free workflow.
Recorded beds retain digital level and are not filtered/noised/quantized again.
Rendered events use the versioned Pi chain and an assumed active-sample 60–80 dB SPL
prior. Silent, very low-floor or nonnative S16 backgrounds warn; a software WAV
check cannot certify the hardware's 24/32-bit to 16-bit I2S conversion.

At defaults, 35% of examples are background-only. Conditional confuser probability
is 0.5 and `max_procedural_share` is 0.2 of all background opportunities, including
unchanged beds: an all-procedural pool has nominal 0.10 confuser probability per
background opportunity, about 3.5% of all examples. This is not a 20% cap among
emitted confusers. The automatic pool has HVAC/fan, fridge hum, running tap and
extractor hood families, four parents each. It uses available room geometry or a
labelled direct-field proxy when none is available. These are assumed generators,
not accepted recordings of household nuisances.

An explicit `negative_manifest` with its required `negative_manifest_sha256` replaces automatic setup. Sampling/mining uses frozen split locks and model-bound records;
audit/calibration recordings are not mining data. Sampling weights change draw
probability, not BCE loss. Background/confuser targets are all-zero labels.
Impact/tonal admission requires listening/label-boundary and recorded-family realism
receipts, including referenced file hashes. Tonal variation and held-out doorbell
checks constrain that admission but do not prove population non-inferiority.
Metadata can be forged: a valid-looking receipt is not independent authentication.

Opt-in `matched/1` mining additions require a `negative-context-bank/1` descriptor
and digest binding one checkpoint, detector policy, split and reviewed membership.
Old unbound additions are refused. WP2 also groups overlay descendants by their
locked original source groups; duplicated descendants do not add independence.

### Audition gated negative families

```bash
python -m sensorsim.negatives.audition generate negative_previews --seed 8100
```

These are unrendered proxy previews. For training approval, use the actual
receiver-domain candidate audio and an independent recorded-family reference.
Prepare an input manifest (paths relative to that manifest):

```json
{"schema":"negative-audition-input/1","rows":[{
  "item_id":"closure-01","family":"door_closure","file":"candidate.wav",
  "reference_file":"recorded_closure.wav","reference_kind":"recorded_family",
  "reference_evidence_domain":"deployment recording","reference_license_id":"user-owned"
}]}
```

```bash
python -m sensorsim.negatives.audition prepare inputs.json review
# Listen to candidate/reference pairs in review/clips/.
# Edit review/decisions.json: reviewer, approve/reject, reviewed_label_boundary, notes.
python -m sensorsim.negatives.audition finalize review
```

Pending decisions, changed audio and absent recorded references prevent approval.
No tool supplies a human listening decision. Finalization checks both originals
and audition copies. Each approved `realism_gate` in `review/receipts.json` uses
paths relative to the review folder and hashes the copied reference. A finalized
review folder can move as a unit. Put the `negative-training/1` manifest beside
`receipts.json`, or prefix/rebase its receipt paths relative to that manifest's
folder; the loader resolves paths relative to the training manifest. Attach the
gate to the matching item's provenance. A listening receipt alone does not
satisfy licensing, frozen splits, room identity or held-out tonal doorbell recall.

The pre-R1, 12-model proxy screen (three negative arms × partial on/off × two seeds,
1,000 steps) did not find an eligible improvement without recall loss. Ungated
negatives gave no nuisance reduction. The screen-only impact/tonal arm reduced
nuisance triggers roughly 32–34 percentage points against ungated negatives while
losing 7.3–11 points of recall; its best nuisance result remained about 59% at a
**fixed 0.5 threshold, not calibrated**. Partial windows restored kettle recall
but increased class swaps. The small screen had weak boundary-effect power and
cannot authorize production adoption. Recheck affected evidence after R1.

New settings `event_aware_masks`, `audibility_gate`, `mixstyle_same_label` and
`low_event_spl_tail` default off; `positive_fraction` defaults unset. The low-tail
experiment assigns 25% of events to 45–60 dB SPL, an assumption requiring real data.
Existing SpecAugment (0.5) and frequency MixStyle (0.4) remain active defaults.
`partial_windows: true`, `validation_mode: hard` and `proxy_screen: true` are
refused by `project run`; proxy-screen models are also refused for production export.

Experimental `window_policy: deployment/1` samples overlapping deployment-grid
positions and masks unknown class labels. It requires `p_specaugment: 0` and
`p_freq_mixstyle: 0`, with reviewed `window_cues` in rendered-file sample coordinates
(kettle: `boil`; optional knock/chime overrides: `identifying`). Automatic
source-to-render cue transport is not implemented; use the lower-level training
manifest. The 2026-10-10 synthetic-proxy screen registered **FAIL**: confusions fell
in two of three seeds, but legacy-model seed variance was as large and nuisance
triggers rose. Audit is pending; legacy placement remains default.

Cue-valid paired augmentation is a separate opt-in research path; confuser-enabled
pairing is refused until per-draw identity is supported. Parent-event level
contracts and lower-level mixing/metadata hooks are implemented, but parent-level
training and persistence remain refused pending the full-parent adapter. Legacy
active-sample normalization remains default. Modular late-room calibration is
research-only and cannot supply training data.

## Model selection

The default `selection_mode: panel` selects a checkpoint and detector jointly,
using validation parents/content disjoint from training. The automatic panel needs
at least ten independent validation groups per target class and at least one disjoint background group; otherwise it falls back
to the terminal checkpoint and the configured fixed detector. The small automatic
panel has only eight nuisance contexts and often has no feasible candidate.

Selection checks checkpoints every 250 steps and at the end against ten detector
policies. Feasibility requires nuisance triggers ≤0.10, zero background starts and
each class's recall ≥0.50; worst-class recall then macro recall ranks feasible
candidates. The selected detector is saved with the selected model. No feasible
candidate means an explicit **uncalibrated** terminal fallback, not a relaxed gate.
The default fixed policy has on=0.5, off ratio=0.6, smoothing=3 hops, minimum=2 hops
and hop=0.25 s. `diagnostic_final` explicitly requests an unqualified terminal model.
Synthetic validation mAP remains diagnostic; it is not the deployed selection gate.

The lower-level `selection_mode: operational` route requires a frozen
`checkpoint_selector` callback to `nn.train`. It validates checkpoint bytes,
feature configuration, class order and detector policy and counts every declared
nuisance family. It is not wired into `project run`; software integration checks
do not establish a recall or nuisance benefit.

## PCM copies for the edge runtime

Keep originals. The simplest edge input is mono signed PCM16 WAV at the model
rate (normally 16 kHz). The loader also accepts 8/16/24/32-bit integer PCM,
averages channels, and resamples integer-ratio rates such as 48 kHz to 16 kHz.
Float WAV is refused with a PCM export message. Intake supports float source
audio; edge evaluation has a narrower format contract.

For an edge-only copy using your installed ffmpeg:

```bash
ffmpeg -i original.wav -ac 1 -ar 16000 -c:a pcm_s16le edge-copy.wav
```

Do not normalize or denoise. Record conversion, original recording ID and sample
ranges. Convert once to model-rate PCM before splitting calibration, WP1, WP2
floor and overlay allocations; never count them as independent acquisitions.

## Evaluate: WP1

```bash
python -m sensorsim.project evaluate my_project
```

Place labelled files in `evaluate/`, preferably `manifest.jsonl` (or `takes.jsonl`)
with `file`, `label`, `group` and `data_kind`. A group is an independent take/parent,
not a crop. Declare generated data `data_kind: proxy`; unmarked files are silently treated as measured (only a standing `source_note` in evaluation.json reminds you); a folder-wide `evaluate/data_kind.txt` containing `proxy` also works. `background`, `negative` and `negative_...` denote negatives.
Label folders/named WAVs are also supported, but explicit provenance is preferable.

The CNN JSON schema is `sensorsim-room-evaluation/2` (the baseline remains `/1`).
It reports `comparison`, `training_stream`, `raw_render_diagnostic`,
`spike_in_sensitivity`, `gate_acceptance`, `data_kinds`, `source_note`, `recall`,
`feature_space_realism`, `sim_vs_real_discriminator`, `pipeline_symmetry`,
`recordings` and `limits`. The primary comparison invokes frozen `nn.make_example`
settings with matched framing. Raw renders are diagnostic. Strongest-energy peak
framing may anchor on noise; there is no target-span annotation adapter yet.

E1 uses matched-count whole-group label permutation for the joint MMD/band-SMD
decision; real-only half-splits are descriptive. E2 uses grouped cross-validation
and train-fold scaling. An interval wholly below AUC 0.5 is **inconclusive: CV
instability**. Every registered spike-in at every seed must be detected by the
corresponding E1/E2 sensitivity check before a pass survives. This can downgrade a
pass to inconclusive, not erase a fail. A refit group-permutation p-value is reported
but does not prove equivalence. The fixed-score bootstrap may undercover and passes
at realistic group counts are **weak evidence**, not distributional realism or
recognition validation. Identical-input comparisons establish route equivalence only.

Standalone MMD uses exact stored-kernel arithmetic and inclusive permutation ties
(`statistic_policy=exact-dyadic-symmetrized-blocks/1`,
`pvalue_policy=inclusive-exact-rational-plus-one/1`). Earlier MMD p-values and
MMD-dependent family decisions are historical. This repair does not replace the
separate joint E1 MMD/SMD calibration or paired baseline.

Recall can coexist with wrong-class alerts. Inspect `confusion_matrix`,
`wrong_class_takes`, `wrong_class_events` and per-take detected labels.
The separate `tools/p5_power_audit.py` qualification runner completed one frozen
synthetic-proxy campaign (A5d): **FAIL at 4 groups** per class per side and **PASS at
6/10/15/18/20** for the registered procedural defects only. Any new qualification
requires a frozen registration, independent custody, signed controls and fresh
external cases; a completed software suite cannot substitute for the campaign or
establish deployment realism.

## False-alarm evaluation: WP2

```bash
python -m sensorsim.evaluation.fa_rate --model my_project/out/model.npz --manifest my_project/wp2/manifest.json --out my_project/wp2/estimate.json
```

The schema is `sensorsim-room-fa-rate/3`. Start with target-free audio through the
actual deployment microphone. Report `start_alerts` (what live listening emits
immediately) separately from duration-qualified `confirmed_events`. Do not use the
second count to certify the first. No observed real exposure means no measured
false-alarm claim. A measured headline requires at least 30 minutes of continuous
real target-free recording; pooling short files does not establish continuity.
Short runs are smoke tests with no headline point estimate.

A manifest declares `source_kind`, `calibration_streams`, `floor_streams`,
`challenges`, `context_prevalence_per_hour`, `context_weights`,
`floor_time_fraction` and their source descriptions. Challenge rows carry `family`,
`context`, `context_id` and `parent_id`. Every WAV record needs `path`, stable
original `recording_id`, `sample_rate_hz` at the model rate, and `sample_range:
[start, end]` in that original (end excluded). The range length must equal the
exported excerpt's length. Keep all excerpts from one acquisition linked by parent.

For example, this **model-based/proxy** manifest declares one whole vacuum
context and a separate half-hour floor. Replace paths, IDs, exact ranges and
incidence assumptions with your allocated recordings; it is deliberately too
small to qualify the challenge bank. Change `source_kind` to `measured` only for
actual device recordings, and list calibration audio even if calibration failed.
Paths are relative to `wp2/manifest.json`.

```json
{
  "source_kind": "proxy",
  "calibration_streams": [],
  "floor_streams": [{"path":"floor.wav", "recording_id":"floor-session",
    "sample_rate_hz":16000, "sample_range":[0,28800000]}],
  "challenges": [{"path":"vacuum-episode.wav", "recording_id":"vacuum-session",
    "sample_rate_hz":16000, "sample_range":[0,480000],
    "family":"vacuum", "context":"isolated", "context_id":"vacuum-01",
    "parent_id":"vacuum-session"}],
  "context_prevalence_per_hour": {"vacuum":2},
  "context_weights": {"isolated":1},
  "floor_time_fraction": 1,
  "prevalence_source": "assumed",
  "context_weights_source": "assumed",
  "floor_time_fraction_source": "assumed"
}
```

A zero-alert half hour still has an upper bound near 6/h; one nuisance parent
remains inconclusive. Do not substitute duplicated files for independent groups.

Composite rates combine floor exposure with whole-context incidence, including
all repeated outputs in a long episode. They are model-based unless the audio,
incidence, weights and floor fraction are all measured. Declare incidence sensitivity;
default global/per-family multipliers include ×0.3 and ×3. Avoid counting a mixed
scene once as a whole and again as its constituent sources. `scored` requires
100 independent parents per family and 20 per stratum; that is a sample-count gate,
not calibration or a false-alarm budget pass. Constant counts still need an upper
bound. Nonconstant cluster-bootstrap intervals are approximate under skew.

At zero alarms, the standalone 95% Poisson floor bound is 2.996/exposure-hours:
about 3.0 h for ≤1/h. Composite multiplicity can require 5.1–5.7 h for the floor
component alone. A reported zero does not make nuisance contributions zero: in the WP2 review scenario (100 never-firing contexts per family, physical cap 31 outputs, a 3.5 h zero-event proxy floor and incidences 2/5/10 per hour) the primary composite upper bound was 24.27/h.
An optional outputs-per-trigger bound carries an explicit assumption and is
suppressed when contradicted by observations.

### Overlay on your own audio and disjointness

`overlay_path` declares `room` (room-model JSON), `receiver_xyz_m` and `device_id`.
A challenge's `overlay` specifies a target-free deployed-device `background` record,
`duration_s`, `seed`, optional `start_s`, `whole_episode: true` and `events`.
Each event supplies a complete procedural specification or a user-owned dry WAV,
`onset_s`, active receiver `spl_db`, `source_xyz_m` and `whole_episode: true`.
Ambiguous tonal/impact labels require explicit review. All sources mix as waveforms
through the shared room/device path; a captured bed is not rechained. Overlays
remain **synthetic proxy evidence**, even with a measured bed.

The complete episode, tail and tracker guards must fit within the bed; contexts
are limited to 60 s and 256 events. Do not chop longer episodes into independent
observations. Reusing a bed or source links parent clusters. The full supplied bed
extent is reserved, including unused samples: write separate nonoverlapping PCM
excerpts before allocating calibration, floor and overlay roles.

Convert once to mono PCM at the model rate **before calibration**. Keep original
IDs/ranges. The manifest must contain `calibration_streams`; `[]` means none declared
and does not disable saved-model ancestry checks. Model metadata stores calibration
digests and nonconstant one-second float32 block fingerprints. Exact copied blocks
with shifted boundaries can be found, but gain changes, resampling, dithering and
short excerpts may escape. Legacy digest-only models need complete declared inputs;
missing provenance is not proof of disjointness. `edge calibrate` warns when
`--recording-id` is omitted or the background rate differs from the model rate.
Resampled/requantized copies may not fingerprint-match, so declare original IDs
and ranges; `--sample-start` uses samples at the model rate.
Reviewer cost estimates were about 0.47 MB of metadata per calibration hour and
about 15 minutes for a large 300-context bank; these are workload-specific history.

## Threshold calibration, deployment and export

```bash
python -m sensorsim.edge calibrate my_project/out/model.npz my_project/calibration/background.wav --fa-per-hour 1 --budget-mode combined --recording-id development-day1 --sample-start 0 --out my_project/out/model_calibrated.npz
# Only after status feasible and saved true; preserve the original:
cp my_project/out/model.npz my_project/out/model_uncalibrated.npz
cp my_project/out/model_calibrated.npz my_project/out/model.npz
python -m sensorsim.project evaluate my_project
python -m sensorsim.evaluation.fa_rate --model my_project/out/model.npz --manifest my_project/wp2/manifest.json --out my_project/wp2/estimate.json
python -m sensorsim.project export my_project device
(cd device && sha256sum -c SHA256SUMS)
python device/edge.py listen device/model.npz target-free-audit.wav
```

The export folder must not exist or must be empty. Its six files are `edge.py`, `model.npz`,
`LICENSE`, `NOTICE`, `SHA256SUMS` and `DEPLOY.md`. Restricted lineage and research-only
proxy-screen models are refused. Inspect the selected detector and fallback status.
The generated guide records 3 hours for the combined 1/h budget. With only
30 minutes and zero alerts, the one-sided 95% upper bound is about 6/h; one hour
supports about 3/h. Insufficient exposure exits 2 with `insufficient_evidence`,
`saved: false` and no new calibrated model. Collect more separate development
audio. Three hours is necessary, not sufficient if alerts remain.
`project evaluate` and `project export` both read `out/model.npz`; after successful
calibration use the explicit copy step above. Do not rerun training between that
copy and evaluation/export, since training can replace the model.

`edge calibrate` defaults to a combined class budget in `start_alerts`, evaluates
the complete finite threshold grid (counts need not be monotone), then allocates
remaining budget deterministically. `--budget-mode per_class --event-unit
confirmed_events` requests a different claim explicitly. Status is `feasible`,
`infeasible` or `insufficient_evidence`; nonfeasible CLI runs exit 2 and write no
new model. The report includes exposure, observed and best-achievable counts.
At ≤1/h the zero-event exposure floor alone needs about 3 h. This is development
threshold selection, not a confidence bound after selection or a recall objective.
Check both recall and false alarms on untouched acquisitions after calibration.

## Accuracy

Historical dEchorate accuracy figures are withdrawn; the HDF5 recompute remains
inconclusive. See [README.md](README.md#accuracy-and-limits). TRUTH session
simulations are model-based proxies, not deployment recordings. No measured
target-room recall or false alarms per hour is established, and no negative-screen
result replaces a locked real evaluation.
