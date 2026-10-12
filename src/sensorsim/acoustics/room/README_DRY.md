# Dry-source extraction from measured sweeps

> **Scope at release-doc base 506315b:** this is the optional legacy sweep/extraction path.
> The surveyed `room.yaml`, no-sweep playback calibration and chamber-target
> workflow is documented in [USAGE.md](https://github.com/Pbjis2op4me/sensorsim-room/blob/main/USAGE.md) and has a **synthetic proxy rehearsal**.
> Route completion did not qualify panel selection, short calibration or deployment accuracy.
> Historical numeric capture/fit examples here are model-based/proxy evidence,
> not deployment recordings or post-R1 acoustic acceptance.


> **Note on references.** This module was extracted from a larger project.
> Mentions of audits, frozen protocols, benchmark tables and validation runs
> refer to documents in that parent project, which are not included in this
> distribution. They are named here for provenance, not as working links.

Historical dEchorate dry-extraction figures are withdrawn. Historical synthetic
dry-extraction figures are withdrawn pending source-provenance review. No result
here validates case B on a measured physical source or establishes full-band
anechoic recovery.

`dry.py` is an offline, deterministic, approximate dereverberator. It accepts arrays and
metadata independently of the other room stages. It does not turn arbitrary recordings into
verified anechoic sources. See the frozen protocol
and original benchmark results.

## What can be recovered

Write a sweep transfer as `H_i = S M_i R_i`: loudspeaker, microphone, propagation.
The on-axis REF direct response is `D = S M_ref g_ref`, where `g_ref` includes direct
travel/distance gain (REF nominally 0.3 m). REF is not an independent speaker calibration.
All sample rates, recording gains, geometry, orientations and RIR time origins must match.
Phone AGC, clipping, clock drift and nonlinear speakers invalidate this fixed linear model.

- **A — the same speaker plays the real recording:** `Y_i = H_i X + N_i`.
  Full inverse filtering targets the played digital signal `X`, including removal of the
  speaker and microphone response, conditional on their measured bandwidth and invertibility.
- **B — a physical source at S1:** `Y_i = M_i R_i X_rad + N_i`. Use the relative transfer
  `Q_i = H_i / D`, then recover `D/S * X_rad = M_ref g_ref X_rad`.
  Implementation avoids explicit division at REF notches: `G_i = D w_i H_i* /
  denominator`, using the conditioned denominator described below. In the ideal matched noiseless limit, the loudspeaker
  factor cancels; **the microphone response stays**. This is sound as captured on axis at
  REF, not source pressure at 1 m, not a response-free waveform. No extra distance scaling.

This factorization assumes the physical source excites each room path with the same relative
angular pattern as the speaker. Voices, machines and drones generally violate that assumption.
RIR ratios cannot recover the missing directional information. Every B band therefore carries
a `source_directivity_unverified` flag even at high SNR. A short direct window also loses LF
resolution; 125 Hz is explicitly uncertain. **The REF recording at 0.3 m is the best basis
for the dry sound:** its higher direct-to-reverberant ratio makes the inverse less dependent
on room cancellation. Record the actual sound there, at the same gain and orientation as the
REF sweep, as well as at the room mic positions. Retain its microphone
coloration consistently when augmenting training data. Do not apply the speaker response again
to a B estimate in synthesis. Separately known receiver calibration would be needed to change
microphone type or infer absolute Pa.

The recording-kit README's statement about removing the phone response is too strong for B.
This module's explicit case selection and report define the recoverable target.

## Separate takes versus simultaneous microphones

A phone moved from REF to M1 records different physical performances. The manifest CLI outputs
one estimate per take and recommends REF. It does not align or sum those waveforms. The array
API rejects multiple recordings unless `synchronized=True`; the CLI's `--synchronized` is an
explicit assertion of the same event on a common sample clock, with no gain changes. Equal
length alone does not prove synchronization. Sweep measurement offsets must already be removed,
leaving acoustic propagation delays intact. Do not align each RIR independently to its peak.

## Methods reviewed and default

| Method / primary source | Role and limitations |
|---|---|
| [Kirkeby & Nelson (1999), regularized inverse filters](https://resource.isvr.soton.ac.uk/staff/pubs/PubPDFs/Pub9229.pdf) | Frequency-dependent ridge limits inverse boosts at poorly observed frequencies. Single-channel special case is a Wiener-style regularized inverse, not exact inversion of deep notches. |
| [Boll (1979), spectral subtraction](https://course.ece.cmu.edu/~ece491/homework/Boll79.pdf) | Stationary noise evidence supports spectral subtraction. Our octave power moments and measured-RIR multichannel weighting are a custom adaptation; the paper does not validate this implementation. |
| [Miyoshi & Kaneda (1988), MINT](https://doi.org/10.1109/29.1509) | Exact finite multichannel inverses require sufficiently long filters and suitable coprime channels; identification error destroys that ideal. |
| [Kodrasi, Goetze & Doclo (2013)](https://inakodrasi.github.io/publications_pdfs/Kodrasi_ITASLP_2013.pdf) | Partial equalization and regularization improve robustness to RIR errors. Their P-MINT constrains the effective impulse response; our spectral ridge is not P-MINT or an implementation of their experiments. |
| [Lim & Naylor (2013)](https://www.commsp.ee.ic.ac.uk/~sap/uploads/publications/Lim2013.pdf) | Low-complexity robust multichannel equalization motivates avoiding a dense, ill-conditioned convolution-matrix pseudoinverse. |
| [Lebart, Boucher & Denbigh (2001)](https://www.ee.columbia.edu/~dpwe/papers/LebBD01-ssdereverv.pdf) | A statistical decaying tail supports late-reverb spectral suppression. Preserves noisy phase; cannot reconstruct arbitrary canceled early components. |
| [Habets (2007), thesis](https://research.tue.nl/en/publications/single-and-multi-microphone-speech-dereverberation-using-spectral/) | Spectral enhancement using late-reverb statistics and direct/reverberant structure; motivates T60/DRR-informed suppression, with diffuse-tail assumptions. |
| [nara_wpe](https://github.com/fgnt/nara_wpe) | MIT, NumPy WPE implementation available; not installed here. Proposed blind baseline: 32 ms STFT, 50% hop, delay 3 frames, 10 taps, 3 iterations, one common channel set. Delayed linear prediction targets late reverb and can remove predictable machine tones. Not evaluated and not required at runtime. |

Reuse: NumPy/SciPy FFT, Welch, overlap-add convolution and STFT do the numeric work. No new
package is installed. pyroomacoustics simulates rooms but is not this case-aware measured-RIR
extraction adapter. nara_wpe does not use measured sweep RIRs or implement the A/B distinction.
A small custom spectral estimator is justified by those missing interface/identification needs;
analytic tests check its cancellation and bounded-noise behavior. It is not a novel MINT solver.

**Revised defaults (`room-dry/2`):** the user's spectral-fidelity correction supersedes the
original estimator design; the original gates and failure records remain in the
internal audit. Full measured RIR; `ridge=0.001`, `ref_weight=4`, gain cap 20 dB. The default
noise amplitude floor changes from 0.2 to **0**: forcing a fixed noise floor into a very weak
band also changes its timbre. Nonpositive estimated signal power is flagged, never silently
claimed as recovered. Set `noise_suppression=False` to keep the conditioned noisy inverse.

Let `r_i` be octave direct/reverberant power ratio, clipped 0.1..10 and multiplied by four
for REF. Interpolate this reliability on log frequency; `w_i ∝ r_i/N_i`, normalized per bin,
with a frequency-local noise floor. Let `E=sum w_i|H_i|²` and `E_b` its local octave mean.
The supported inverse is `G_i=T w_i H_i*/max(E,0.001 E_b)`, where `T=1` for A and `T=D`
for B. It has unit relative transfer where the response is supported: weak observed source
power does not itself shrink the inverse. The same joint inverse-noise gain cap of 10 remains.
When **both** `E <0.001 max(E)` and octave SNR <6 dB, keep the original strong ridge
`max(0.001 max(E), E min(N/(P-N),100))` in `E+ridge`: a wholly weak, noisy sweep band may
be outside the speaker bandwidth. This exception prevents noise amplification below the
measured passband. It is explicitly reported as unsupported transfer.

Noise removal is a separate, stationary **power** estimate. For each filtered channel
`z_i=G_i*y_i`, estimate Welch spectra of `z_i` and identically filtered silence `G_i*n_i`,
in bounded batches, and subtract their signed powers before octave averaging or clipping.
With `q_i=w_i|H_i|²/E`, the wanted band power is estimated by
`S_b = mean_b(sum_i (PSD(z_i)-PSD(G_i*n_i))/r_i) / mean_b(sum_i q_i²/r_i)`.
Dividing by `r_i` avoids unintentionally squaring the DRR/REF prior in these power moments.
The remaining weighting is proportional to inverse noise variance times one reliability
prior. The coherent waveform still gives REF its original fourfold preference.

Apply constant octave amplitude gain `sqrt(clip(S_b/PSD_b(sum z_i),0,1))`, with an exact
identity when measured noise is zero. A Wiener amplitude fraction would square the desired
power fraction and suppress weak bands again. Keeping each band's gain local avoids leaking
a strong neighbouring band's power through interpolation. Linear centered FIR convolution
uses at least 4×RIR length and 2048 samples; there is no circular audio wraparound or peak
normalization. The gain cap is a filter bound, not a universal peak/sample bound.

This preserves supported transfer and approximate **long-term octave balance**, not unbiased
waveform phase, individual harmonics or every time-varying source envelope. Finite Welch
resolution, RIR errors and clipping of uncertain excess power still cause bias. Low-SNR band
power may pass a spectral gate while its phase is mostly noise. The band report is a diagnostic,
not a posterior confidence interval. Subtraction assumes stationary background and independent
channel noise. One `noise_floor.wav` may be reused as a proxy, with a warning; separate same-gain
silence per mic is better. Correlated noise needs a covariance-aware extension. No speech-content
model or truth-derived spectral correction is used.

Optional `early_ms=50` truncates/tapers the inversion model, preserving measured timing.
Optional `late_suppression=True` uses per-band decay `a=exp(-6 ln(10) hop/(fs T60))`, a 50 ms
prediction delay and a conservative DRR-based late-power fraction. Metadata T60 values override
the diagnostic RIR decay fit. This is an engineering approximation inspired by Lebart/Habets,
not their exact estimator; in full-inverse mode it can double-suppress. Both options stay off
by default and their perceptual quality remains unvalidated. No silent substitution when T60
is missing: that band's late subtraction is skipped and its T60 is null in diagnostics.

## Array API

```python
from sensorsim.acoustics.room.dry import extract_dry, DryConfig
result = extract_dry(
    {"REF_S1": real_ref}, {"REF_S1": rir_ref, "M1": rir_m1}, noise_floor,
    fs=48000, case="B", reference="REF_S1",
    metadata={"REF_S1": {"direct_start_sample": 42, "direct_end_sample": 180}},
)
dry, report = result.samples, result.report
```

Inputs are finite mono arrays; matching recordings must have equal length. Noise may be a
single array or a dict keyed like the recordings. `metadata[mic]` accepts `source_id`, direct
window `[direct_start_sample,direct_end_sample)`, and `t60_s` keyed by octave-center strings.
Direct windows are absolute indices in the RIR; the B target removes the REF direct-window
start delay. If omitted, a window from 1 ms before to 2.5 ms after the largest peak is used,
explicitly flagged as unverified (especially hazardous behind obstacles). Inspect these windows.
B needs a REF RIR even for a far-only recording. Matching mic response/orientation matters.

`DryResult` includes samples, fs, version, `ModelValidity` and a JSON-safe report. Per octave:
SNR, DRR, T60/weights for every channel, effective ridge, retained transfer, notch fraction and
reasons to distrust it. Combined SNR/DRR summary fields are the **best channel**, not an output
SNR estimate. Signed band signal-minus-noise power is summed **before** clipping for SNR;
rectifying each bin first would bias this diagnostic upward. Bands below 6 dB SNR, with local
notches or unsupported transfer, more than 20% inverse loss, gain clipping or unresolved direct
window are flagged. `retained_transfer` now includes gain-cap loss, before noise subtraction;
`noise_power_gain` states the separate subtraction. `unsupported_transfer_fraction`,
`recovery_status` (`conditioned`, `noise_limited`, `unresolved`) and `unresolved_bands_hz`
expose missing information; `no_resolved_signal_power` flags nonpositive excess power even
when a nonzero user-selected mask floor leaves residual noise. `conditioned` does not override
the other reasons or make case B directivity verified. RIR decay fitting
is a diagnostic -5..-25 dB regression, not ISO 3382/noise-corrected T20 certification.
Frequencies outside the reported octave bands are unassessed and must not be treated as reliable.

Input float arrays and output can be memory mapped (`output_buffer=`). PSD estimation and
convolution use bounded batches; default output is an in-memory array. Noise subtraction
makes a second filtering pass after estimating full-take band powers. It does not require a
second full-length output in RAM. The CLI memory-maps
16/32-bit PCM or float WAVs and uses temporary float files inside its output directory. Scratch
requires roughly 4 bytes/sample/channel plus output. Packed 24-bit WAV cannot be mmap-read by
SciPy: convert to 16/32-bit PCM first. This is offline (noncausal), not a real-time processor.

## CLI over the user's manifest folder

Use the kit's `fs_hz`, `takes.type` (`real`, `noise`, `sweep`), `speaker`, `mic`, `label`, `file`.
Do not pass raw 60 s sweep recordings as RIRs. Stage 1 must first extract RIR arrays, preserving calibration and time-origin metadata;
the independent index below hands those arrays to dry extraction without coupling the APIs.

`rirs.yaml` (paths relative to this index; `.npy` or mono WAV):

```yaml
schema_version: room-rir-index/1
fs_hz: 48000
source: S1
reference: REF_S1
rirs:
  REF_S1: {file: S1_REF_rir.npy, units: FS/FS, direct_start_sample: 42, direct_end_sample: 180}
  M1: {file: S1_M1_rir.npy, units: FS/FS, direct_start_sample: 280, direct_end_sample: 420}
```

`units` is `FS/FS` for uncalibrated takes and `Pa/FS` for calibrated ones. An index without
`schema_version` (the older form) still runs, with a warning that its units are assumed.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python -m sensorsim.acoustics.room.dry /path/to/manifest_folder \
  --rir-index /path/to/rirs.yaml --source S1 --label 'example sound' \
  --case B --output ./my_room_dry
```

Each take produces an unclipped float WAV and JSON report, plus `summary.json` with the
recommended REF output. Existing estimates are never overwritten. Input SHA256s, config,
source commit and environment versions accompany the output. Float WAV peaks above 1 are
preserved; lower gain for playback, retaining that gain as metadata. Clipping before extraction
cannot be repaired. The CLI does not synthesize sweeps or fit a room model.
