# Model-based join and precursor checks

Run with a frozen YAML preregistration and its SHA-256 file; write results outside
this repository. No audio or measured RIRs are loaded by this bench.

```bash
python -m tests.validation.bench_room_join.run \
  --preregistration "$JOB/preregistration.yaml" \
  --hash-file "$JOB/evidence/prereg_sha256.txt" \
  --controls-preregistration "$JOB/preregistration_controls.yaml" \
  --controls-hash-file "$JOB/evidence/prereg_controls_sha256.txt" \
  --out "$JOB/evidence/join.json"
```

The main registration supplies `geometry` (`dimensions_m`, `source_m`,
`receiver_m`), `join` (`fs`, `seeds: {start, count}`, `bands_hz`, `threshold_db`),
and `precursor` (`room_dimensions_m`, `source_m`, `receiver_m`, `fs`,
`fractional_samples`, `threshold_energy_fraction`, `threshold_peak_db`). The
original prose geometry in `splits.acceptance` is also parsed; omitted geometry
is an error. Room absorption and renderer settings are the shipped defaults.
Use the callable `join_trials` API with an explicit `RoomModel` for other priors.
All figures describe a model, including when dimensions come from a user survey.

The optional controls registration supplies `parent_sha256` (the main
registration digest), `fs`, `seeds: {start, count}`, and `bands_hz`. It runs
continuous filtered noise with a 0.5 s exponential decay and no join. Preserve
that result: a median absolute short-window difference includes noise variation
and need not approach zero as the number of seeds increases. Neither detrending
nor filtering makes this statistic an unbiased estimate of a join discontinuity.

The 10 ms step subtracts the exact exponential trend over the same sample
windows. Causal and zero-phase analysis results are saved separately; only the
causal statistic determines the registered join result. The precursor check uses
the summed direct waveform, excluding a 2 ms main-lobe guard. It also records
phase/magnitude errors against the geometric free-field transfer and descriptive
log-mel differences after trimming the precursor. The full-band gates are
0.1 dB magnitude and 0.02 sample phase-equivalent delay. These are engineering
checks, not listening tests or evidence about classifier false positives.

Existing evidence is never overwritten, and registration hashes are verified
before rendering. Kernel basis energies must not be added to infer full-band
precursor energy: the coherent basis pressures interfere and can cancel.

## Expected-energy criterion

The separate `expected` module supports a new registration after the historical
short-window failure. It does not rescore that old failure:

```sh
python -m tests.validation.bench_room_join.expected \
  --preregistration preregistration.yaml --hash-file prereg_sha256.txt \
  --mode controls --out controls.json
python -m tests.validation.bench_room_join.expected \
  --preregistration preregistration.yaml --hash-file prereg_sha256.txt \
  --mode join --out join.json
```

Run controls before modifying the renderer. Every built-in case is a **proxy
result / model-based**. The controls apply a known step to the analysed process;
they validate the estimator, not the entire waveform analysis chain. Use at least
200 independent tail seeds per position. The room run retains window powers for
every seed, the fixed named geometry and the sampled positions, rather than
discarding position failures behind a pooled statistic.

This statistic averages power before taking logarithms. Analysis-filter energy
delay is removed from the time axis. A paired unjoined image-source continuation
supplies the local energy trend; the prescribed exponential trend is also
reported descriptively. The continuation accounts for deterministic early
interference, which changing only the late random seed cannot sample. The
room-family estimate is conditional on the declared positions. Its seed-cluster
interval and its separate geometry/seed interval answer different questions:
Monte Carlo precision and sensitivity to position selection, respectively.
A family pass does not certify each individual position or a real room.

The version-1 bench uses the registered 20 ms windows, 2,000 bootstrap replicates,
a point threshold of 1 dB and a 1.5 dB interval limit. A change to that fixed
protocol needs a fresh registration. Controls, failures, missing metrics and
historical results must remain available. The bench refuses evidence overwrite
and verifies the hash before running.

The revised renderer warms each causal synthesis band with hidden noise, then
applies its pressure decay and sine fade. No individual draw is normalized.
The fitter propagates stationary synthesis-state covariance through that
time-varying envelope and the causal analysis filter. This retains temporal
correlation and adjacent-band leakage; convolving squared cascaded impulse
responses with input variance would model the previous ordering instead.

## Exact fitter cost and fair join diagnostic

Historical dEchorate fit-cost figures and their shipped registration are
withdrawn. Original failures remain in the internal review record. Freeze a
new external registration for the supplied inputs and the model-based join
diagnostic, then record its hash before running:

```sh
sha256sum preregistration.yaml > prereg_sha256.txt
python -m tests.validation.bench_room_join.fair \
  --preregistration preregistration.yaml --hash-file prereg_sha256.txt \
  --out fair_join.json
```

The diagnostic reports coherent ISM and exact independent-path energy
references together. The latter sums each analysed path's squared kernel;
it has no random-sign Monte Carlo error. The renderer expectation is coherent
early energy plus the shared stationary late covariance. Bootstrap intervals
resample geometry clusters. They describe these proxy room families, not
physical-room accuracy or recognition performance.

Before and centred windows start after the direct packet's analysis response.
Empty windows remain unavailable, including low bands in small rooms. Report
coverage alongside ratios. The primary statistic is the centred energy ratio
at 250 Hz–4 kHz; 125 Hz and 250 Hz also receive descriptive reporting. The
Waterhouse room-average factor is reported as a candidate hypothesis only;
the renderer does not apply it. Registered dimensions and coordinates can be
replaced by a new preregistered user-room family without editing the tool.

The timing harness takes an external RIR manifest and records both wall and
process CPU time, the full fitted model, bootstrap results, and input hashes:

```sh
python -m tests.validation.bench_room_join.fit_timing \
  --rir-manifest inputs.json --bootstrap 20 --seed 20260930 \
  --preregistration preregistration.yaml --hash-file prereg_sha256.txt \
  --out fit_timing.json
```

Use `--bootstrap 0` for the primary fit alone. The manifest has a
`model` object accepted by `RoomModel`, and an `observations` list containing
`file` (a mono WAV relative to the manifest), `source_xyz_m`, `receiver_xyz_m`
and optional `metadata`. Preserve the data split and input labels in the new
registration. Supply impulse responses, not ordinary target recordings.
