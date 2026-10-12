# Target capture and expansion

Record **18 or more varied, independent takes per class**: knock, doorbell and
kettle. With 18 groups the third reserved for RH and the separate validation
subset leave eight training groups for the 4/8/all learning curve. A crop or
playback of one performance is the same `source_group`. Keep object and session
identities across chamber, digital and room copies. Record separate presses,
knock patterns/strengths and kettle cycles; do not relabel excerpts as new takes.

For `near_dry`, use native-rate mono WAV, fixed gain, a microphone within 30 cm,
and **at least 0.5 seconds of untouched quiet lead**. Record until silent, then
one more second. Chimes can ring for many seconds. Do not normalize, denoise,
pitch shift or stretch microphone recordings. If a record-button click occupies
0–0.2 s, move `quiet_window_s` to an untouched quiet interval at the same gain.
This interval estimates the local floor, not a false-alarm rate.

In `dry/NOTES.md`, record chamber type, microphone, distance, gain and how the
sound was made. Optionally record **one clap or balloon pop per chamber session**
at the same capture distance and gain, with quiet lead and complete decay. List
it as `session_reference` in the source rows. Its input hash, session identity
and band decay rates stay in provenance. Close capture alone does not prove
dryness. No windowed ratio is interpreted as physical DRR.

Keep calibration/comparison digital files in `played/`; they are not project training targets.
Optional deployed-microphone playback recordings belong in `room_recordings/`
and use `*__room.wav`. Those paths are always refused as dry sources. In-situ
takes are evaluation/RH anchors on this branch; in-situ training is not wired.
Never convolve them with a second room. Keep room dimensions, microphone/source
coordinates and loudspeaker settings for the room and device stages.

## Project intake

Declare every target in `project.yaml` under `dataset.sources`; file names are
relative to the configured `targets` directory, normally in `dry/`.
The drop-folder paths `room_recordings/`, `background/` and `played/` are excluded
from target discovery and the unlisted-audio check. Explicit playback rows are
refused. Unlisted audio elsewhere still requires provenance. No third-party audio or response data belongs in the repository.

```yaml
dataset:
  sources:
    - file: dry/knock_01.wav
      label: door_knock
      kind: knock
      source_group: doorA-session1-performance01
      object_id: doorA
      session: session1
      capture_route: near_dry
      evidence_label: measured
      quiet_window_s: [0.0, 0.2]
      merge_gap_s: 0.7
      # Optional chamber qualification, shared by rows in this session:
      session_reference:
        file: dry/session1_clap.wav
        session: session1
        evidence_label: measured
        quiet_window_s: [0.0, 0.2]
```

`kind` controls source-specific QA independently of the class label. Valid kinds
are `knock`, `chime`, `kettle`, and `other`. If omitted, `knock`, `knocks`,
`knocking`, `door_knock`, `door_knocks`, and `door_knocking` resolve to knock
(case, spaces and hyphens are normalized); doorbell/chime resolve to chime.
Use explicit `kind: knock` for any custom knock label.

`merge_gap_s` joins compatible hit regions before event selection. It
is **0.7 s for kind knock**, 0.15 s for other kinds. `event_merge_gap_s` is an alias
(the legacy sweep-session manifest uses that name); supply only one. Raise it
for a deliberately slower sequence. `event_index` or `event_window_s` selects
an entire merged event; a window is an intersection hint, not a crop. Keep all
hits in one performance. Other sounds inside the resulting crop are still
contamination: merging is not a semantic classifier. `premerge_regions`,
`impact_onsets_samples`, and `impact_spectral_similarity` remain in the report.
An energy rise with octave-power cosine similarity below 0.5 to the first hit
is flagged as contamination; it cannot extend the last hit into the tail. This
40 ms spectral check is a development heuristic and can flag a changing knock
timbre or miss a similar-sounding contaminant. Listening remains mandatory.

Project intake refuses all `known_digital` and loudspeaker-playback target rows.
Playback recordings are for room calibration and comparison (the WP4 known-source
path); targets are rendered at their true sites. The low-level
`intake_known_digital` / `render_source` comparison APIs remain available, with
clipping and acoustic diagnostics; they do not authorize project training intake.
Digital origin is a declaration, not proof of acoustic dryness.

`evidence_label: proxy` is for generated test stand-ins; user recordings use
`measured`. `dataset.sources` replaces the legacy sweep-extracted source list.
This source intake does not fit a room or change the background stage.

## QA and listening approval

Intake writes `out/source_qa.json`. To list file, crop seconds, status and key:

```sh
python -m sensorsim.sources.intake out/source_qa.json
```

Listen at those bounds, check semantic isolation and the complete event, then
add the exact report key to the corresponding source row:

```yaml
      approval:
        reviewer: listener-name
        decision: approve
        approval_key: copy-the-current-report-key-after-listening
```

The key binds content, provenance, crop, settings, reference hash and decisions;
it excludes unstable last bits of floating diagnostics. Hard rejections cannot
be approved away. All chamber crops require listening. Known contamination,
low SIR/SNR, clipping, truncation and a wet session reference reject admission.
Unknown interference still needs human judgement.

Every crop has a noise-limited octave late-decay diagnostic (125, 500, 1000 and
2000 Hz). `t20_s` is the **T60 extrapolation from the −5 to −25 dB fit**, not a
20 dB interval or a DRR estimate. It starts after the last detected energy rise,
subtracts the quiet band floor, and stops at a sustained floor crossing. Missing
range or a poor fit is `unavailable`, never a long fit to the noise floor. The
500 Hz diagnostic flags values above **0.20 s** for review, never hard rejection.
Without a reference the review message is: "late decay above the dry-knock
heuristic, which may be a resonant door or a wet room; record a session clap to
decide automatically". This message recommends the session room gate; it does
not waive listening for semantic isolation.

A session impulse is qualified by usable mid-band decay rates at or below
0.20 s (or a labelled short-decay comparison bound). A long reference yields
`session_not_dry`, a hard rejection; ambiguous/multiple impulses require review.
For short claps, `t20_s` remains null where no fit is available: the per-band
`comparison_decay_s` carries the 0.10 s short-decay bound and a method label.
That bound describes a fast return to the floor, not a measured T20. Each
available take band is compared to 1.5 times its session comparison value,
without a 0.20 s floor. A slower take is reviewable as intrinsic source ringing.
A qualified session does not establish independent acoustic truth or source
likeness. Without a reference, source and room decay remain confounded.

Optional `dryness_window_s: [start, source_stop, tail_stop]` retains an
`early_tail_ratio_db` and tail fraction. The deprecated `estimated_drr_db` is
always null. High tail energy requires review, including for knocks; it cannot
separate intrinsic ringing from room decay. Never invent a source-stop time for
a ringing source. Automatic decay diagnostics also run without windows.

**Threshold provenance:** the 0.20 s heuristic was tuned on the same development
stand-ins used in its initial test. S05 and the citation-check errata provide no
transferable door-mode range. SNR/SIR, floor, merge and spectral thresholds are
also development heuristics. Independent proxy panels test implementation
behavior only. Real-take validation is pending; no population validity or
recognition/false-positive improvement is established.

## Loudspeaker and splits

For the low-level WP4 calibration/comparison API only, the loudspeaker response applies **only to exact digital files explicitly
played through that loudspeaker at its position**. Set `loudspeaker_played:
true` and `loudspeaker_position: [x, y, z]` in metres in that API, never in project targets. The default is false.
The renderer requires that exact source position and refuses mismatches; it
never moves filtered playback to a doorbell site or a random room position.
Chamber recordings of physical knocks/doorbells already include their emitter
response and cannot opt into this hook. Recordings of loudspeaker playback also
already include that response and are not inputs for another application.

Optional `loudspeaker` contains a user-owned `impulse`, `kind: measured`, and
`reference`. Alternatively supply `frequency_hz`, `gain_db` and `reference` for
a magnitude FIR approximation from DC to Nyquist. `gain_db` is **relative dB**,
limited to ±40 dB; when the curve gives SPL, supply its passband `reference_db`
to subtract first. This shapes a relative transfer, not absolute SPL calibration.
No curve ships. Full convolution tails and application identity are retained;
no automatic normalization or second speaker application is permitted.

Split `SourceTake` objects with `split_sources`, audit with `audit_splits`, then
render `AssignedSource` objects. When using `render_source`, provide
`source_position` for playback; it must match the declared loudspeaker position.
`as_dry_clip` preserves groups, hashes, fit lineage and declared train/val splits
in the real dataset path. Synthetic children require an explicit train split.
The project adapter's automatic split is train/val only: reserve RH outside its
inputs. Project evaluation still needs a group/hash audit before any WP6(b)
claim (the evaluation-lane proposal accompanies this handoff).

## Replay and real-data acceptance

Procedural sources remain **off and unavailable** until a generative model is
validated. The former "procedural" implementation was exactly gain times a
training take: its modal estimate plus full residual reconstructed the input.
It added no acoustic diversity and could not support a meaningful C2ST.

For explicitly requested gain resampling, use `sources.replay.fit_replay` and
`generate_replay`, both with `enabled=True`. Output is **gain × training take**,
labelled `mode: replay`, `generator: gain-replay/1`, `c2st_eligible: false`, and
`independent_realization: false`. Replays are training-only, retain parent and
fit lineage, and inherit the parent's capture route and response record. They
are excluded from diversity claims and WP6 acceptance/C2ST job preparation.
No speaker response can be added to a replay. Pitch and stretch stay off.

`evaluation.prepare_acceptance` prepares matched-budget 4/8/all-group jobs,
reserves RH and reports shortages. Pre-register and hash before experiments.
WP6(b/c), recognition and false-alarm benefits remain pending on real takes and
WP1/WP2/WP4 integration. Proxy tests establish contracts only.
