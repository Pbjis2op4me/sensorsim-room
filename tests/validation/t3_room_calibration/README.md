# Room calibration development and verifier handoff

All bundled fixtures are procedural **proxy results**. No measured data, audio,
weights, learned discrepancy tables or third-party implementation ships here.
These tools do not issue acceptance receipts or approve training promotion.

The module `sensorsim.acoustics.room.calibration_discrepancy` provides:

- `decompose_uncertainty`: one geometry/state/chain pool, native seconds/dB
  inputs and explicit covariance in log(seconds)/dB coordinates. Only independent
  measurement repeats and simulator-mean noise shrink with their own counts.
  Shared playback and discrepancy remain unreduced. Duplicate hashes are refused.
- `late_block_update`: a covariance-weighted T30 likelihood with early-field
  diagnostics that cannot reweight it. Missing predictions or singular covariance
  abstain. Low ESS requests refinement. No wall-material inference is made.
- `refine_particles`: proposal callbacks with explicit log densities, ancestry,
  bounded doubling and prior/proposal correction. Each batch replaces its
  predecessor. The caller must supply a correct normalized proposal density in
  the same coordinates as the prior; this module does not derive a room-prior
  density or supply an unqualified physical proposal.
- `convergence_check`: two independent seeds and an independently converged
  reference, comparing 5/50/95% T30 quantiles in seconds. This is a diagnostic,
  not a verifier receipt.
- `collect_render_predictions` and `predictive_products`: separately report
  parameter-only, realized member-RIR and discrepancy-inclusive intervals.
  The render probe records imported code hashes and actual RIR digests. It uses
  production metrics and therefore is not an independent acoustic oracle.
  Actual renders retain their posterior weights and member identities; no
  discrepancy metric sample is presented as generated training audio.
- `update_room_ensemble`, `require_discrepancy_review`, `audit_scene_states`:
  opt-in integration adapters, full observation metadata, refusal of reused
  evidence and a training block until the owner implements verifier receipt
  binding. The adapter returns `ResearchRoomEnsemble` with version
  `room-research-ensemble/1`, not a training-capable `RoomEnsemble`. Legacy
  dataset calls refuse it before writing output; legacy ensemble JSON loaders
  reject its version. Use `ResearchRoomEnsemble.model_validate_json` to reload
  research diagnostics. No caller-supplied boolean bypass exists.

The owner-only integration patch is delivered outside the repository. Existing
prior, pool, dataset, pipeline, physical decay and numerical energy code remain
unchanged in this commit. The patch changes multiple-source scene receiver
jitter to one installation per scene. It fixes sites/member/split/seeds once,
then jointly rejects the receiver and sources within 10,000 attempts, yielding
no rows until the entire scene is legal. It adds explicit late-block opt-in pooling,
and blocks modular training. It must be reviewed and tested on the merged tree.

`tools/room_calibration_audit.py INPUT.json --output OUTPUT.json` accepts
`required_keys`, native-unit `score_scales`, a bootstrap `seed`, an explicit
`evidence_domain`, and `cases`. Each case supplies unique content and group IDs,
an explicit permitted `partition`, native `truth`/`units` and `candidate` and
`baseline` products. The executable fixture in `test_audit.document()` shows the
schema. Null truth is unavailable. Required predictions cannot be silently
filtered. Interval scores compare all truth-valid cells using the same fixed
scales. Output preserves all-requested and scored coverage denominators and
separate per-metric gates. Group bootstrap intervals are approximate, not proof
that positions or room configurations are independent homes.

Permitted data summaries must name synthetic, MeshRIR, official HDF5 dEchorate,
or deployment evidence. dEchorate is internal development validation only, with
[CC BY attribution](../../../THIRD_PARTY.md#audio-data-sources). The partition
guard refuses unsupported IDs and uncleared editions. It validates supplied metadata; it cannot
prove the origin of a caller's file. The verifier must freeze artifact/split
hashes and inspect the source independently before reading measured RIRs.
User recordings use the same schema with the deployment domain and explicit
known-playback or room-response kind. Dry targets and deployed background are
not accepted as room-response truth.

Development commands:

```sh
python -m pytest -q tests/unit/t3_room_calibration
python -m tests.validation.t3_room_calibration.development_controls --output development.json
```

The development runner fixes 32 independent conjugate-normal cases and seed
31001. Its reference is an analytic normal posterior. It also calculates
prospective exact-binomial reachability at 200 groups; it does not execute the
verifier's 200-case acceptance. The early C50 mutation tests modular feedback,
not physical early-field realism. Default tests also cover proposal correction,
missing inputs, units, constant arrays, duplicate identities and missing hooks.

Run `handoff_controls.py` explicitly only when testing the shared-file patch.
It intentionally fails on an unpatched tree and is outside default collection.
Also run `route_controls.py` on the patched tree to exercise procedural playback
through real extraction, pooling, update, research serialization and dataset
refusal, plus generated WAV/manifest coherence. These are disclosed development
fixtures, not untouched P4 acceptance cases. The F2 seeds 1/3 are the known-bad
old-sampler cases; 2/4 and ordinary geometry are the matched correct controls.
An impossible scene must yield zero descendants, with a bounded explicit error.

P4 remains unqualified until independent control review, custody, frozen permitted
measured partitions, actual-render coverage/width/bias gates and physical-room
convergence checks. Keep the historical WP4 coverage/width failure. P0 operational
nonregression and natural-positive recordings remain separate requirements.
