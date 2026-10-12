# Room recording kit: one phone and the laptop's built-in speakers

> **Scope at main 159518b:** this is the optional legacy sweep/extraction path.
> The surveyed `room.yaml`, no-sweep playback calibration and chamber-target
> workflow is documented in the repository USAGE.md and remains **pending rehearsal**.
> Historical numeric capture/fit examples here are model-based/proxy evidence,
> not deployment recordings or post-R1 acoustic acceptance.


Profile a rectangular room and its short obstacles, then record real sounds at
known positions. Move the **same phone between three positions** for each speaker
placement. Move the **same speaker between K placements**, named S1 through SK.
Every take is sequential, single-channel, and has its own unknown start and clock.
Add **one extra sweep at C1**, kept out of the fit, for your own accuracy check.

| File | Purpose |
|---|---|
| `sweep_playback_48k.wav` | Play once per position: v1 60 s (14 s spacing); optional v2 76 s (18 s spacing), four 10 s ESS sweeps |
| `level_check_pink_noise_48k.wav` | Rough volume setting; verify headroom with the actual sweep |
| `sweep_single_48k.wav`, `sweep_kit.json` | Exact analysis reference; keep the JSON beside your manifest |
| `manifest_template.yaml` | Coordinates, settings, filenames and optional event selection |
| `QA_sweep_minus6_48k.wav`, `capture_qa_assets.json` | Exactly -6 dB companion playback (PCM rounding only), with nominal/quiet/kit hashes |
| `capture_qa_template.yaml` | Separate seven-take pilot sidecar; never list the quiet file as a normal nominal sweep |

Use unprocessed mono PCM WAV or lossless FLAC with fixed gain. Turn OFF AGC,
noise suppression, echo cancellation, speaker EQ and loudness enhancement.
48 kHz is preferred; **native 44.1 kHz is accepted and converted by ingest**.
Never change a WAV header to pretend its samples have a different rate. Keep
original recordings. No phone app or Bluetooth path has been physically certified.

## Fixed orientation and playback protocol

- Use the laptop's **built-in speaker route (internal/wired), no Bluetooth**.
  Disable audio enhancements, spatial sound, EQ, loudness normalization and player
  effects. Set a moderate volume during the pilot, then keep it fixed for every
  sweep, REF and laptop-played sound. Record the volume and settings; do not
  normalize recordings afterward.
- At each S placement, put the laptop on a table/stand; record speaker height,
  front-facing wall/direction and lid angle. Keep that pose fixed while the phone
  visits P1–P3, REF and C1; do not turn the laptop to follow the phone. Mark the
  speaker reference point and use it consistently for XYZ.
- Put the phone on a stand with its **bottom mic pointing at the laptop** at
  every position. Keep mic height fixed where practical and record the actual
  height/XYZ and orientation; keep its mic hole uncovered and the same case/stand.
  Request the app's unprocessed source with AGC, noise suppression and echo
  cancellation off. Save available support-query evidence; otherwise record
  `unknown` rather than assuming that “raw” means unprocessed.
- Mark C1 before fitting: choose a position **not symmetric to P1–P3**, at least
  **1 m from walls**, with a clear path to S1 (behind obstacles only if that is the
  intended check). Record one complete `S1_C1_sweep.wav` with unchanged settings.
  Keep `role: check` on that take; do not add a training/real-sound take at the
  same S1/C1 pair. If the room cannot meet the clearance, note the actual distance.

<!-- LAPTOP-CHAIN EVIDENCE: condensed from the laptop-chain owner's protocol patch. -->
**Laptop/REF details:** mark both emitting drivers and their separation; use their
midpoint as the nominal source if both play. Keep the same mono route without
stereo widening. A close REF of two drivers is not a point-source calibration.
Keep weak/unsupported 125/250 Hz bands visible; do not boost bass to force a pass.

A nearby tabletop can invalidate REF: at source/mic height **5 cm above the
table**, with **0.3 m** horizontal separation, the table reflection arrives only
**0.047 ms** after the direct sound (image-source geometry at 343 m/s), below the
existing **0.25 ms** separation guard. **Raise the laptop so its speakers sit at least
15 cm above any table surface** (e.g. on a stack of books or a stand); at 15 cm the
reflection arrives **0.362 ms** later and REF works (image-source geometry at 343 m/s).
Pilot an open stand with more clearance;
if moving the laptop, update its XYZ and repeat all affected sweeps and sounds.
Do not omit the table geometry or relax that guard. Omitting an unusable REF
leaves the physical-source coloration unresolved.

Disable optional player/vendor effects, but **retain factory speaker protection**;
on Asahi Linux this includes its protection daemon. Fixed volume still needs
the two-level pilot. [Asahi audio-chain documentation](https://asahilinux.org/docs/sw/audio-userspace/).
On Android request `UNPROCESSED` and check
`PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED`; `VOICE_RECOGNITION` is the documented
fallback if unavailable, still requiring the pilot and honest support metadata.
[Android recording guidance](https://developer.android.com/media/platform/mediarecorder).
No particular laptop's directional response is calibrated by this protocol.
<!-- END LAPTOP-CHAIN EVIDENCE -->

## Recording sequence

1. Mark a floor corner as origin, x/y along the walls and z up, all in metres.
   Measure room dimensions, obstacle boxes (minimum corner, size, material),
   speaker cone coordinates and phone mic-hole coordinates. Record orientation.
   Note the room's temperature and relative humidity (a household thermometer and
   hygrometer will do); they set the speed of sound that anchors arrival times
   (`fit: {temperature_c, humidity_pct}` in `project.yaml`).
2. Choose three phone positions P1–P3 spanning near/far and useful obstacle
   shadows. Coordinates may be reused for each speaker placement. If positions
   change, create distinct position IDs such as S2P1 and record their coordinates.
3. Start at a moderate listening level; set volume and gain once after the pilot
   below. Record a full sweep at the **closest planned phone position** (or
   optional 0.3 m REF), then check useful decay/noise support at the far position.
   -12 to -6 dBFS recorded peaks are optional headroom guidance **only when input
   gain is adjustable**, never a minimum acceptance level. Do not raise speaker
   volume just to hit a recording peak; lower peaks can have adequate decay SNR.
   The -6 dBFS playback file provides digital headroom, not protection against
   speaker/phone compression. Pink noise alone does not certify sweep headroom.
4. Record about 15 s of same-gain quiet as `noise_floor.wav`; assign its actual
   mic position in the manifest. More position-specific floors are supported.
5. At each S{k}, move the phone to P1, P2, then P3. Start recording **before**
   playback, play the entire file, wait for the decay, then stop. Save as
   `S{k}_P{j}_sweep.wav`. No synchronized button presses or reliable timestamps
   are required. Several minutes of extra lead/trail are acceptable within the
   configured offline sample budget. Avoid talking/handling during playback.
   Keep the phone on its stand and the laptop fixed as specified above. At S1,
   also record the marked C1 check sweep before moving the laptop.
6. Record the real sound at each desired measured S/P pair. Each is a **new
   performance**, named `S{k}_P{j}_real_<label>.wav`. Leave quiet before and enough
   quiet after to cover the reverb. Keep source direction and recorder gain fixed.
   Do not average performances or transfer a sweep's clock to a real take.
   Keep laptop-played and physically made sounds in separate analysis runs:
   use pipeline `case: A` for playback through this laptop, `case: B` for physical
   sounds. The sweeps/check can be shared; select the corresponding real-take rows.
7. Fill and save `manifest.yaml`, remove unused rows, and retain
   `acquisition: sequential_phone`. Place `sweep_kit.json` beside it. Old M1/M2
   position names are still accepted; names do not imply multiple microphones.

**REF is strongly recommended when making sounds physically: it helps account
for the laptop/phone coloration in the sweep, because a physical sound does not
pass through the laptop speaker.** REF is a separate sweep take with the phone 0.3 m on-axis from the
speaker. Add `REF_S1` coordinates and `S1_REF_sweep.wav` (type REF), similarly for
other placements, keeping the same laptop pose, volume and phone mic direction.
It guides combined speaker/phone coloration and remains optional.
Without REF, inversion gives an effective source with inseparable device
coloration; it does not establish an anechoic physical source. Even with REF,
near-field effects and source directivity remain limitations.

The pipeline places **Your held-out accuracy check** prominently in `report.md`
and `report.json`. It predicts C1 from the fitted room and compares T30/EDT/T20,
C50/C80, D50, drr/2 DRR and recorder-relative level in the four primary octaves.
Check takes do not set absorption, device/reference corrections or level scale.
Failed/unavailable bands stay visible; one position does not validate the entire
room, the phone or physical-source directionality. Omit C1 and its take only when
no check was recorded; old manifests retain their previous behavior.

Analysis matches the declared kit, checks repeat spacing and chirp rate, and
requires at least three healthy repeats. A corrupted repeat is dropped with a
reason. Too few repeats, ambiguous playback trains or inconsistent clocks produce
an actionable rejection. Clipping requires re-recording at lower fixed gain.
Constant drift up to ±300 ppm is supported; changing clock/latency needs review.
The first supported compressed arrival may be a reflection. **Absolute emission time
is unobservable**: RIR time is imposed from known distance/c or reported relative.

Real-sound onset/offset is detected against the measured noise floor with pre-roll
and a decay-covering post-roll. Use the same `label` for related performances at
P1–P3. The pipeline compares their spectral and temporal signatures, accounting
for the measured room responses. It cannot always distinguish a repeated
nuisance from your intended sound, especially when events overlap or are quiet.

An optional hint identifies your intended event in **that WAV's recorded time**:
add `event_window_s: [12.3, 12.8]` to its take row, after listening to the file.
Choose a window intersecting just that event. It selects the full detected event
and its decay; these numbers do not crop the tail or assert an emission time.
Alternatively use `event_index: 2` for the third detected energy region (zero-based).
Use one hint form per take. A hint applies only to its own recording;
annotate each unclear take separately. Hints cannot recover an event
below the measured noise floor.

If selection remains ambiguous, that take is rejected and the resolved takes
continue through dry recovery and re-simulation. `dry/event_selection.json`
lists candidate bounds and per-file reasons; `dry/event_summary.json` counts
automatic, hinted and rejected takes. Default completion requires at least one
resolved take per label. Pipeline config `event_min_resolved_per_label` changes
that minimum; unmet coverage is reported as `partial` while retaining resolved
outputs. Listen to rejected files, add hints, then rerun into a new output folder.
Set `event_post_roll_s` to cover the observed
room decay. A truncated tail is flagged. A selected region can still contain
overlapping nuisance sounds; selection is not source separation.

## Built-in laptop speaker and one phone

Treat the laptop, its support and the phone as one fixed measurement chain. A
laptop can have little usable bass at 125 Hz; 250 Hz can also be weak. A finite
T30 or a detected sweep alone does not establish band support. Keep every failed
or unavailable band in the report; do not boost bass to force it to pass.

* **Volume:** begin at a moderate setting (25–50% is only a starting suggestion,
  not a calibrated safe/linear range). Use the actual sweep at the nearest and
  farthest positions and the seven-take two-level pilot below. At unchanged
  hardware volume, nominal versus the -6 dB companion should differ by 6 ±1 dB
  in each usable band. Lower volume and repeat the pilot if the output compresses,
  pumps or clips. Once selected, keep hardware/player volume fixed for sweeps,
  quiet and laptop-played real takes; save the exact settings.
* **Windows:** Settings > System > Sound > the laptop output > Audio enhancements
  Off; also turn Spatial sound Off and disable optional EQ/loudness/surround in
  the vendor app. Older Sound properties may call this Disable all enhancements.
  [Microsoft instructions](https://support.microsoft.com/en-us/windows/hardware/audio/fix-distorted-or-crackling-audio-in-windows).
* **macOS:** in the playback app disable EQ, normalization and spatial effects.
  In Music, turn Window > Equalizer Off and Settings > Playback > Sound Enhancer
  and Sound Check Off. These controls do not establish that internal speaker DSP
  is bypassed. [Apple EQ/Enhancer documentation](https://support.apple.com/guide/music/museb684a3de/mac),
  [Sound Check](https://support.apple.com/en-ca/109331).
* **Linux:** bypass user-added EasyEffects/PulseEffects EQ, bass enhancement,
  loudness and compressors; use a fixed playback route. Retain factory speaker
  protection, including Asahi speakersafetyd. Its processing/protection is not
  equivalent to optional user EQ; test linearity at the chosen volume.
  [Asahi chain documentation](https://asahilinux.org/docs/sw/audio-userspace/).
* **Geometry:** keep lid angle, laptop heading, height, table position and feet
  unchanged throughout each source placement. Mark the emitting driver positions;
  for a stereo pair record their separation and the midpoint used as the nominal
  source. Play the same mono file without widening. A reliably selected single
  driver can simplify geometry, but use that exact routing for every sweep and
  playback take. Close REF of a stereo pair is not a point-source calibration.
* **Phone:** use a stand with the selected mic hole clear of the clamp and pointed
  consistently toward the source. Record mic-hole XYZ, orientation, phone case,
  microphone route and app/OS version. Do not hold it, cover the mic, lay it on
  the table, or rotate it between the sweep and real take at a position.
* **App:** record native-rate mono PCM WAV or lossless FLAC; no normalization,
  denoise, voice isolation, AGC or echo cancellation. On Android request
  UNPROCESSED and verify PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED; if unavailable,
  VOICE_RECOGNITION is the documented fallback, still requiring the pilot.
  On iOS use an app exposing AVAudioSession measurement mode. A mode name alone
  is not physical validation. Keep gain fixed; preserve original files and native
  sample-rate headers. [Android](https://developer.android.com/media/platform/mediarecorder),
  [Apple measurement mode](https://developer.apple.com/documentation/avfaudio/avaudiosession/mode-swift.struct/measurement).

**Table and REF:** a 0.3 m take only helps when its direct response can be separated
from nearby reflections. With the acoustic source 5 cm above a tabletop, the
image-source table echo at a same-height 0.3 m REF is only about 0.047 ms later;
the current 0.25 ms separation guard correctly rejects that calibration. An
open stand with more clearance can help, but clearance is not a universal
height guarantee, and short direct windows still lose low-frequency support.
Pilot the intended arrangement before recording all takes. If moving/raising
the laptop, update its geometry and repeat the corresponding sweeps and sounds.
Do not remove the table from the manifest or relax clock/reflection gates to
make an invalid reference pass. Omitting REF permits the effective-source
workflow but leaves physical-source coloration unresolved.

**Real sounds:** label whether a take is (A) a digital sound played through the
same laptop or (B) a physical sound made at that source mark. For A, keep exactly
the sweep playback route and volume. For B, record the actual sound at the
optional 0.3 m REF as well as the room positions, with a matching REF sweep when
geometry permits. These are separate performances, never synchronized channels.
A usable REF helps cancel the sweep speaker's coloration; the phone response
remains, and different physical-source directions remain unverified. Do not
apply laptop EQ again to a recovered B source. Compare the separate close take
with the room-derived estimates instead of treating it as the same waveform.

## Seven-take physical repeatability pilot

Run this before real performances. It qualifies the particular phone/app/speaker
chain for further evaluation; phone-only repeatability never establishes accuracy.

1. Freeze the room, contents, doors, occupancy, source and mic-hole marks,
   orientations, case and stands. Fill `capture_identity` (exact device, OS build,
   app/version, requested/actual source, gain, mic route, processing settings).
   For Android raw-source support, save the actual support-query evidence when
   available. Otherwise keep `support_query: unknown`; an app's “raw” label is
   not proof. Record speaker firmware, player, connection/codec, volume and EQ.
   Confirm simultaneous playback on the same phone does not change its mic route.
2. Check the nearest and farthest planned placements with the actual sweep.
   Keep all hardware gains fixed for every pilot take. Do not normalize exports.
   Retain the exact kit JSON, playback assets and their hash sidecar.
3. At fixed **P1**, capture **A nominal, B nominal, L quieter, C nominal**, in
   that order, each containing the complete four-repeat playback. L uses
   `QA_sweep_minus6_48k.wav`; the hardware volume remains unchanged. The companion
   has identical sample count/rate/timing and is 6 dB below the nominal file.
   Its peak is about -12 dBFS. Start recording before playback; stop after decay.
4. Capture nominal at **P2, P3, then return to marked P1 for D**. Include the
   intended far and obstacle-shadow placements in P2/P3. These are **seven full
   playback takes** total, not seven repeats. Record 15 s same-gain quiet at
   each position and again after returning. Optional 0.3 m REF adds an eighth
   take; never increase fixed volume to rescue a weak REF or low-frequency band.
5. Copy `capture_qa_template.yaml` to `capture_qa.yaml`; replace example XYZ,
   filenames and noise files with the measured values. Add
   `capture_qa: capture_qa.yaml` to the production manifest if using the optional
   QA report integration. Pilot A/B/L/C/P2/P3/D rows remain in the sidecar, with
   L labelled `quiet`; it must not be analysed as the unchanged nominal excitation.
   The sidecar importer rejects mismatched asset hashes or digital scaling.

v1 playback alone takes seven minutes. Allow roughly 50–60 MB for mono PCM16
recordings including short margins and quiet; long margins or v2 need more.
Keep original physical recordings and identity/query evidence for reproducibility.
The generated companion adds about 5.8 MB (v1) or 7.3 MB (v2) to the playback kit.

Predeclare these gates; record every failed/unavailable band and excluded repeat:

| Check | Proposed evidence required |
|---|---|
| Extraction and timing | Existing clipping/clock gates, at least three healthy repeats; changing adjacent rates or local lags need review before correction/averaging |
| Two-level transfer | Nominal minus L = **6 ±1 dB per usable octave**, quiet-band SNR ≥20 dB; lower support is unavailable. Each level's own repeat consistency and planned-repeat status must also pass (`capture-qa/2`) |
| Fixed P1 repeatability | A/B/C level spread ≤1 dB; maximum pairwise `2*abs(Ta-Tb)/(Ta+Tb)` ≤5% separately for EDT, T20 and T30; C50 spread ≤1 dB |
| Replacement/session drift | D versus the A/B/C median, separately with the same budgets; P2/P3 measure coverage, not fixed-position repeatability |
| Per-band eligibility | All 125–8000 Hz octaves, with primary 500/1000/2000/4000 Hz separately; retain counts, noise crossings, endpoints, 10 dB noise margins and R² |
| Decay/distortion support | T20 needs approximately 35 dB and T30 45 dB usable range under the endpoint-margin rule; label them separately. Flag possible harmonic-tail overlap; packet ratios are not calibrated THD |

Pumping, disappearing weak tails, time-varying gain or clipping require review.
A lower level and another pilot may help, but no automatic denoising, arbitrary
time warp, indefinite volume increase or post-hoc tolerance relaxation is allowed.
Four repeats are correlated observations of one room/path, not four independent
rooms. Stable nonlinear processing can still pass consistency checks. Kit v2's
longer quiet gap does **not** increase harmonic separation: sweep duration and
frequency ratio are unchanged. Absolute emission time remains unobservable.

When `capture_qa` is supplied, the pipeline includes the QA status in `report.md`
and its per-take/repeat/band reasons in `report.json` under `capture_qa`. These
diagnostics do not change fitting admission or pipeline completion. Review a
failed/unavailable pilot before relying on the session; passing repeatability
does not establish physical accuracy.

## Separate reference-microphone accuracy comparison

When a calibrated omnidirectional mic/interface or room-acoustic instrument can
be borrowed, capture **reference–phone–reference at P1 and P3**, using the **same
consumer speaker, orientation and playback level** in an unchanged room.
Sequential measurements are acceptable; match mic-hole coordinates carefully.
This arm tests the capture chain. A later consumer-versus-omnidirectional speaker
comparison is a separate experiment; do not change speaker and mic together.

Compare T30 and EDT at 5%, C50 at 1 dB; report signed errors, absolute seconds,
reference variation, valid counts and all unsupported bands. Report T20 alongside
without renaming it T30. Use signed `100*(phone-reference)/reference` for decay
error and `phone-reference` dB for C50. If reference repeats vary beyond the budget, the result
is **inconclusive**, not a wider passing tolerance. An independent analyzer of
the same RIR checks algorithms only: match filter, onset and integration rules.
Only independent reference recordings can establish acquisition accuracy.
Neither this pilot nor optional REF alone identifies absolute SPL, room-only
transfer, speaker radiation or recognition-training accuracy.

## Kit versions

The generator accepts `--placements K` for K speaker placements and
`--templates-only` to write guides/templates without regenerating any playback.
For long reverberation, `--kit-version 2 --out NEW_FOLDER` generates an opt-in
76 s playback with 8 s quiet tails. Keep its matching `sweep_kit.json`; ingest
selects the declared period/tail from that file. Existing v1 kits remain valid.
The extra cost is 16 s per take, or 48 s for three positions at one speaker
placement. Typical 0.3–1 s rooms need no longer kit in the synthetic comparison.
Pipeline config `adaptive_rir_window: true` grows analysis beyond 1.5 s only
while supported decay bands remain unresolved, stopping at the acquired quiet
tail. It reports remaining unusable bands; even 8 s may be insufficient in a
very reverberant room. See `rirs` metadata `observation_window` and the report.
Send the completed folder to the room pipeline. Synthetic detector checks do not
establish physical room reconstruction or recognition accuracy.
