# Negative source tools

The negative library is wired into project/CNN training through explicit locked
input manifests. Eligible negatives default on; missing required inputs stop training. No third-party recordings or weights ship. Generated signals are
project-owned Apache-2.0 **proxies**, not validated household populations.

## Shared capture path

`training_device.mix_events` mixes receiver-domain events for either label.
The existing positive `nn.make_example` entry uses it through `mix_device_audio`.
`render_context` and `captured_context` use the same helper. The default event
level is uniform 60–80 dB SPL at the microphone, an explicit assumption shared
with positives. Override with `cfg={'event_spl_db': [low, high]}` or an explicit
`Event.spl_db`. A 10 ms energy envelope selects active samples within -25 dB of
its peak, inside the supplied span; silent padding does not set gain. This is a
level convention, not a measured perceptual threshold or family-level prior.

A `RenderPath` supplies the room realization, receiver, production synthesis
configuration and versioned absolute device profile. Pass the same `RoomDraw`
and already-varied `device_draw` to paired calls. If omitted, the path uses its
room with the supplied seed and draws device variation once. The positive
dataset's room sampling must use those same draws when integrating the library;
these tools do not change the dataset sampler. Labels never choose a renderer.

```python
from sensorsim.acoustics.room.model import RoomModel
from sensorsim.iot_mic import PROFILES
from sensorsim.negatives.composer import RenderPath, compose
path = RenderPath(RoomModel(shoebox_dims_m=(6, 5, 3)), (4, 3, 1.5),
                  PROFILES['pi5_inmp441/1'], 'pi5_inmp441/1')
scene = compose(path=path, seed=1701,
    family_rates_per_hour={'running_tap': 4},
    positions={'room_tone': ((2, 2, 1),),
               'running_tap': ((3, 1, 1),)})
```

This example's geometry, rates and source models are assumptions. It is a proxy
inspection example; procedural realism remains unvalidated. The former relative
`source_rms_db`, `event_rms_db`, `bed_rms_db` and legacy-profile API is superseded
by receiver `spl_db`, `event_spl_db`, `bed_spl_db` and the absolute profile.

On a synthetic pressure-reference bed, the combined scene goes through one device
capture including self-noise. For an already device-recorded background, only
new events get response/sensitivity; the recording retains its level and floor.
The rendered float increment is added without dither or requantization, with the
profile overload clip retained. Background-only windows remain identical. The
recording already owns its quantization error; a second quantization adds a
label-correlated artefact. Training and evaluation features accept float samples;
S16 conversion belongs at a PCM output boundary. Quiet-floor QA uses the nominal
profile, a 6 dB margin sustained for 1 s in 250 ms frames (conservative assumptions,
not a microphone calibration). Silent or non-native-S16 recordings still warn.

## Proxy forms and limits

`taxonomy.yaml` and `procedural.FAMILY_MODELS` name the actual forms and distinct
regions. Impact confusers contain broadband contact/body transients, weaker modal
ringing, per-hit envelopes and timing/level/mode jitter. A struck-chime model is
separate. Tonal and knock-like impact families all require explicit reviewed
label boundaries in `compose`; direct negative `Event` calls must also set
`reviewed_label_boundary=True` after review. The review gate is a human attestation, not a
proof that a sound cannot be the deployment doorbell.

Tonal composition requires `training_targets` (mono takes or a class-to-takes
mapping). Partial frequencies are estimated only from training spectra; draws
exclude +/-15% around every detected partial. Sweeps crossing a band are refused.
An unsatisfiable family fails closed. Direct `generate` calls without targets are
inspection proxies only. The exclusion does not establish real doorbell recall.

`assumptions.py` and each source's `parameter_assumptions` record fixed values
and their rationales, including water bands, motor regions and tonal forms.
Every numerical contact/mode/spacing/chime range is an **unsourced proxy
assumption**, per the S05 citation errata. No range is attributed to research.
Motor frequencies, fridge harmonic amplitudes and noise weights are also
assumptions, not measured compressor populations. Generate up to 60 s; kettle-side
sources in `compose` default to 10–30 s continuous forms without forced fades.
Durations are overrideable by family. Contexts are 10 s, events are Poisson at
caller-supplied rates, and timestamps/tail truncation are retained. These defaults
do not establish prevalence or in-distribution realism. Room-tone beds remain
point-source acoustic proxies rather than measured diffuse ambience.

## Recorded sources and licences

`load_clip(root, item, register_path=..., review=...)` checks exact WAV bytes,
register hashes, licence, and listening/target/rights review. User-owned, CC0 and
CC BY 4.0 only; NC, SA and unknown fail closed. External items additionally require
WP9's commercial-manifest validator; an unavailable gate is a rejection.
Attribution and transformations remain attached to source provenance.

User items require `registry_id`, `path`, `sha256`, `license_id: user-owned`,
`verified_on`, `owner` and `ownership_statement`. `user_register_row` additionally
requires a caller-supplied name and genuine ownership evidence URL; it creates
WP9-compatible `name`/`url` fields without inventing ownership. The register binds
the clip hash, date, evidence and disposition. `checked_on` records row validation
today (or an explicitly supplied validation date); `ownership_attested_on` retains
the original attestation. A historic attestation does not backdate the WP9 check.
Ownership does not expire after
30 days; invalid or future dates are rejected. External item and register checks
retain the explicit 30-day freshness rule. WP9 may impose additional requirements.

Review fields: `family`, `source_group`, `acoustic_stage` (`dry` or `captured`),
`reviewer`, `notes`, and true `listened`, `target_free`, `rights_cleared` flags.
Captured audio additionally needs `capture_device`, `profile_id` and boolean
`deployed_device`. Loader records native S16/rate status. Use
`captured_context(source, events=..., path=..., fs=..., seed=...)` to put either
label on the user's 10-second recorded background. Longer takes are explicitly
cropped with their source identity retained.

Only recordings attested as captured through the deployed profile bypass its
response. External captured audio preserves its existing room but passes through
the deployment device; it needs an explicit `captured_reference_spl_db` (the
capture amplitude reference in the peak-sine convention). Its original unknown
response remains an `external_capture_stress` domain. This does not remove an
old device response or establish in-distribution suitability. Dry external clips
instead use production room rendering. Never mark a wet clip dry to force entry.

## Persisted splits and mining

`SplitPlan.create` separates items, families, original source groups and duplicate
content. Save once. At pre-registration, freeze the raw file SHA-256 and lock
path as well as the payload receipt. `mine` and `reviewed_bank` take `split=path`
and `expected_split_sha256=<pre-registered raw file hash>`, re-read that file and
validate its payload/receipt. Passing a rebuilt in-memory plan is refused, and a
saved replacement cannot match the original registered file hash. Recomputing
the expected hash from a replacement would constitute a different study; this
integrity check cannot prevent an operator rewriting the pre-registration.

`FrozenModel` checks immutable checkpoint bytes before and after mining. The
whole pool is checked before scoring. Candidate windows do not overlap within
an item; listening reviews remain pending until a human confirms negative with
notes. A bank accepts one checkpoint only and rejects duplicate content windows.
Every cap is a fraction of **base training weight**: defaults are 5% per source,
10% per family and 20% combined. Apply once to the whole bank. Mined development
items are not independent validation after entering training.

`integration.authorize_audio` checks decoded content and locked Item identity,
for positive targets and backgrounds as well as confusers. `LockedEventPool` is
a content-checked adapter for receiver-domain confusers. It refuses missing or
false `training_eligible`. Tonal/impact admission requires a realism-gate receipt
bound to the audio hash and family, a passing recorded-family comparison, actual
reference/evidence files and reviewer, plus label-boundary review.
Generated sources are ineligible by default. Receipts attest to an external study;
the library verifies their file hashes and identity, not the truth of a human review.
The integrated trainer requires source identities and receiver audio from identified
target room draws; no undeclared room is synthesized during training.

## Evidence boundaries

The original WP3(b) PASS was a **label-independence unit check**, because it
rendered identical inputs twice. Its historical result remains on record; it is
not path-C2ST evidence. The earlier `review_fix.py` L4 PASS was **degenerate**:
it compared the same posterior-dither estimator against itself with another
random draw. Its historical FAIL/PASS and registration stay on record. The final
addendum uses `bench_negatives/final_review.py`: the reviewer's independent-RNG
path test and a true single capture of a known analog background. L4 reports both
bounded chance equivalence and the older exact-0.5-null rule; they are different
statistical decisions. See the committed final registration for frozen rules.

A path pass concerns routing/levels/quantization on neutral proxies. It does not
show procedural versus real-family agreement, trained-B score symmetry, target
recall or false-alarm improvement. WP3(a) and trained-B WP3(b) remain pending
integration; real-family gates await authorized data. No deployment FA/h is
measured. The lead-approved non-gated admission policy is an engineering proxy
policy; it does not replace recorded-family realism evidence.

The active-sample level convention replaces span averaging. Gappy knock takes
therefore receive less gain; the shift depends on each take's silent gaps. A
partial-window policy must use `active_power` for both full-take and crop power
and offset requested SPL by `10*log10(crop_power/full_power)`. Stress SNR keeps
its span-power convention. The shared `draw_confuser` hook is now used by both
the base and dynamics training routes.

## Training integration

The CNN now defaults to `p_confuser_given_background: 0.5` and
`max_procedural_share: 0.2`. These are engineering starting settings, not measured
optima. An enabled policy needs an explicit locked pool; missing inputs raise an
error. There is no implicit room, automatic realism approval, or download.

```yaml
recognizer:
  type: cnn
  p_confuser_given_background: 0.5
  max_procedural_share: 0.2
  negative_manifest: inputs/negatives.json
  negative_manifest_sha256: <sha256 of that file>
  selection_mode: panel
  selection_panel: inputs/panel.json
  selection_panel_sha256: <sha256 of that file>
  partial_windows: false
  validation_mode: synthetic
```

`negative-training/1` JSON contains `split_lock`, its `split_file_sha256`, and
`rows`. Each row contains `file`, `file_sha256`, `item` (the locked WP3 Item),
`room_draw_id`, and `provenance`. The receiver-domain file must use a room draw
identified by training targets. Provenance includes `acoustic_stage: receiver`,
`training_eligible: true`, `license_id`, `provenance` (`synthetic` or recorded
origin), and positive sampling `weight`. Non-project sources additionally bind a
local licence record with `licence_file` and `licence_sha256`.

Beds, water and motor families may use the lead-approved engineering admission
policy without a realism receipt. This is not recorded-family validation. Explicit
ineligibility still refuses a source. Tonal and impact families require a passing
`realism_gate` bound to family/content, reviewer, label-boundary review and actual
`evidence_file`/`reference_file` bytes with their SHA-256 fields. Tones additionally
need `doorbell_recall` (`file`, `sha256`, `reviewer`, `review_decision: accept`). That
reviewed development study must have at least ten independent held-out doorbell
parents, baseline recall at least 0.75 and an observed paired recall loss no greater
than 0.05. It is an admission screen, not a statistical noninferiority guarantee.
The study must bind the candidate confuser content and both compared model files
(`baseline_model_file` / `candidate_model_file`, relative to the study JSON, and
the corresponding SHA-256 fields); those files are also checked.
The tonal generator excludes carrier/sweep support within 15% of training-target
partials; broad exclusions may correctly make a family unavailable.

Weights control sampling, not BCE loss or deployment prevalence. Procedural mass
is capped per background-route opportunity, including unchanged recorded beds;
its unconditional probability also multiplies `p_confuser_given_background` and
`p_background_only`. With only procedural sources, unused mass stays ordinary
recorded background. The model metadata and DEPLOY table record family inventory,
weights and probabilities. Confusers pass through the shared mixer with zero target
labels, including the dynamics/context routes.

A mined addition retains its original locked development Item and `role: mining`.
Its manifest row binds `mining_bank_file`, `mining_bank_sha256` and `review_id` from
`reviewed_bank`; the trainer consumes that reviewed window and its returned weight.
Total, source and family mining caps remain 0.20, 0.05 and 0.10 of base weight.
Use `SplitPlan.create(calibration_families=...)` to lock calibration sources
separately; `purpose=calibrate` cannot authorize mining. Calibration, selection
and audit audio are never mining inputs. Selection excludes
all admitted mining parents and decoded content.

`training-panel/1` JSON contains `role: development`, `rows`, and a `sealed_audit`
identity inventory. A row has `audio`, `file_sha256`, `audio_sha256` (decoded float64
content hash), `source_group`, `background_groups`, `parent_audio_sha256`, `kind`
(target, nuisance, background), and `evidence_domain`. Targets add `label`,
`start_s`, `end_s`; nuisances add `family`. The audit inventory carries parent and
content identities only: training never opens its audio. Parent provenance needs
independent protocol review; hashes cannot prove different acquisition sessions.

Joint selection maximizes worst-class then macro recall at the configured total
confuser-trigger fraction, subject to zero background starts and a per-class recall
floor. Ties prefer fewer nuisance triggers/starts, later checkpoints, then grid
order. All checkpoint scores and detector policies are retained. An infeasible
search exports no selected model. `selection_mode: diagnostic_final` explicitly
uses the terminal checkpoint and makes no selection claim. Synthetic AP remains a
diagnostic. The project continues to refuse partial windows and hard validation
until an independently reviewed class-swap screen accepts them.

### Automatic defaults and research screens

`nn.train` creates a small ungated procedural pool when no negative manifest is
supplied. It uses dataset room models and geometry where available; a minimal
manifest without geometry explicitly records a direct-field proxy. Both routes
use the ordinary receiver-to-device mixer. Generated inputs and split metadata
live beside the model. These engineering proxies have no measured realism claim.

Without a selection manifest, training builds a development panel from disjoint
validation source groups. Too few eligible groups, missing held-out backgrounds,
or infeasible selection retain the terminal checkpoint at the configured fixed
detector and emit a note. Library defaults are on=0.5, off_ratio=0.6,
smooth_hops=3, min_hops=2, hop_s=0.25. This is not deployment calibration.

`proxy_screen=True` is a research-only flag on `nn.train`. A
`proxy_screen_only` receipt binds the family and decoded content, and explicitly
states that it is **not a human listening review**. It allows unreviewed impact
and tonal proxies solely for the comparative screen; it supplies neither
recorded-family realism nor target-boundary, spectral-exclusion or held-out
recall approval. Production admission, `project.run`, and device export refuse
these receipts/models. The low-level `nn.export` also refuses them unless its
explicit `screen_artifact=True` path is used with `proxy_screen=True` to retain a
research checkpoint. Never interpret such a checkpoint as a deployable model.

All procedural admission validates the declared family against generator/model
provenance, including the family parameter region for shared generators.
Training reports the actual confuser count and its shares of background and all
windows. Sampling probabilities are experimental weights, not household rates.
