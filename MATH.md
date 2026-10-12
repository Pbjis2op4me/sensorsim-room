# Mathematical conventions and evidence

**Evidence scope.** Unless a section names a later programme review, the numerical
comparison tables below are historical **model-based/proxy** checks dated
2026-10-07, pre-R1. They were not rerun by the documentation pass. They do not
establish current-renderer room accuracy or deployment recognition.

The original comparison building blocks were run side by side with the established open-source implementation of
the same thing, on a case whose answer is known in closed form: a single reflection, an
exponential decay with a set T60, pure tones, a delayed impulse. A difference from a library
counts as an error only when the closed-form answer says ours is wrong; otherwise it is a
convention, documented here. The comparison scripts are in `tests/validation/math_compare/`.
They run the other libraries' code locally; none of it is copied into this package.

Versions (2026-10-07): numpy 2.5.3, scipy 1.18.1, pyroomacoustics 0.10.1, pyfar 0.8.1,
pyrato 1.1.0, librosa 1.0.0, torch 2.14.1+cpu, torchaudio 2.11.0+cpu, python-acoustics 0.2.6
(its ISO 9613-1 module loaded on its own: the package no longer imports with scipy 1.18), and
Python 3.12.14's `audioop` (μ-law only; ADPCM now uses a procedural comparison below).

## Summary

| Building block | Compared with | Result | Verdict |
| --- | --- | --- | --- |
| Exponential sweep and its inverse | pyfar | Waveform bit-identical; inverse gain 0.000 dB in band; recovered amplitudes equal the closed form | Identical |
| Image sources (positions, orders, wall damping) | pyroomacoustics | Same 63 images to third order; damping equal to 1e-7 | Identical |
| Fractional delay kernels | pyroomacoustics | Both within 0.012 dB of an ideal delay; ours flatter, theirs more phase-linear | Equivalent |
| Air absorption, ISO 9613-1 | python-acoustics, pyfar | Equal to machine precision; pyfar 0.8.1 is 27% off at 90 kPa (its error) | Identical |
| Speed of sound | Cramer (pyfar), ISO 9613-1, pyroomacoustics | Humid-air Cramer formula by default: equal to pyfar's to 6e-14 m/s | Identical |
| Sabine and Eyring T60 | pyroomacoustics | Same formulas; the small offset is the ratio of the two sound speeds | Identical |
| Late reverberant tail across the band | closed form (band power sum) | Covers DC to Nyquist at every rate, flat within 1.5 dB; was octave band-passes only | Fixed |
| T20, T30, EDT; Lundeby and Chu | pyrato, pyroomacoustics | All within 0.3% of a known T60, with and without a noise floor | Equivalent |
| Per-octave decay times | closed form | Time-reversed band filter: T30 exact at every band; the causal filter read 125 Hz T30 up to 7.3% off | Fixed |
| C50, C80, D50 per octave | closed form, pyrato | Zero-phase filter from before the onset: within 0.07 dB at 125 Hz (T60 0.5 s); the causal filter read 3.1 dB low | Fixed |
| Mel filterbank | librosa, torchaudio | Bit-identical to librosa (HTK, no normalization) | Identical |
| DC offset on the device input | closed form | Removed per window before framing; a raw MEMS offset no longer changes the features | Fixed |
| Log-mel features | librosa | Equal to 1e-4 log units under the same framing (our DC removal is the difference); librosa's defaults centre frames | Convention |
| Resampling on the device | scipy `resample_poly` | Was 1.5 dB down at 7 kHz, unlike the training data; **changed to match** | Fixed |
| ADPCM-like stress codec (G.726 stand-in) | Procedural proxy signals | IMA-style adaptation, project-generated step sizes, not bit-exact | Approximation, documented |
| μ-law (G.711 stand-in) | `audioop` G.711, torchaudio | Continuous law: SNR within 3 dB of segmented G.711 | Approximation, documented |
| BatchNorm folding | PyTorch `fuse_conv_bn_eval` | Equal to 6e-8 | Identical |
| Peaking and shelf filters | RBJ cookbook identities | Exact | Identical |

## Room acoustics

### Exponential sweep and deconvolution (`room/rir.py`)

- **Formula.** x(t) = A sin(2π f1 L (e^{t/L} − 1)) with L = T / ln(f2/f1) (Farina 2000). The
  instantaneous frequency is f1 e^{t/L}, so harmonic k arrives L ln k early: 231.4 ms for k = 2
  with a 2 s, 50 Hz–20 kHz sweep.
- **Inverse.** The time-reversed sweep weighted by e^{−t/L} (+6 dB per octave), divided by the
  stationary-phase gain A² fs² L / (4 f2). pyfar instead divides spectra with regularization outside
  the sweep band (`pyfar.dsp.deconvolve`).
- **Model-based check (historical).** Our sweep equals `pyfar.signals.exponential_sweep_time` sample for sample (difference
  0.0). Sweep times inverse has 0.000 dB mean gain from 100 Hz to 13.3 kHz, with 0.97 dB of Fresnel
  ripple for an unfaded sweep and −3.3 dB near f1. A direct sound of 1 and a reflection of 0.5 come
  back as 0.8307 and 0.4146; pyfar's division gives 0.8325 and 0.4144. The closed form for a
  50 Hz–20 kHz band-limited unit impulse is a peak of 2(f2 − f1)/fs = 0.8313, so both are right, and
  neither should be renormalized to a peak of one.

### Image sources (`room/synth.py`)

- **Formula.** Images at 2nL + (−1)^p s on the shoebox lattice, order Σ|2n − p|; each wall hit
  multiplies the pressure by √(1 − α_band) (and √(1 − s) for scattering); 1/r spreading.
- **Model-based check (historical).** Every wall and band was given a different absorption, with source and microphone
  off-centre, so a swapped wall or band would show. Our 63 images up to third order match
  pyroomacoustics' image list one to one: positions within 1.2e-6 m and per-band damping within
  9.8e-8 (pyroomacoustics stores them as 32-bit floats), and reflection orders are identical. The
  first-order floor image of a source at height z sits at −z with damping √(1 − α_floor), as
  expected. pyroomacoustics scales each image by 1/(4πr) when it builds the response
  (`soundsource.py`) where we use 1/r: a constant factor that the fitted levels absorb.

### Fractional delay (`room/synth.py`, `path_kernels`)

Our path kernel is a 129-tap Kaiser-windowed (β 8.6) sinc built in the frequency domain, so the
same kernel can carry octave-band absorption and air loss. pyroomacoustics uses an 81-tap
Hann-windowed sinc. Against the ideal e^{−jωd} for d = 0.1 to 0.9 samples:

| | DC gain | Ripple to 0.45 fs | max \|H − ideal\| | Delay error to 0.25 fs |
| --- | --- | --- | --- | --- |
| ours | 0.99920–0.99999 | ≤ 0.007 dB | ≤ 8.1e-4 | ≤ 3.0e-4 samples |
| pyroomacoustics | 0.99875–0.99999 | ≤ 0.012 dB | ≤ 1.4e-3 | ≤ 5.5e-6 samples |

Both are accurate far beyond what the room fit can resolve; 3e-4 samples is 6 ns at 48 kHz.
Neither rescales the taps to unit DC gain. Ours does not on purpose: the fitter caches these
kernels, and a spectrum-dependent rescale would colour each band.

### Air absorption (`acoustics/absorption.py`)

ISO 9613-1 equations 3–5 with Annex B for the saturation pressure, written out in the module.
Ours equals python-acoustics' implementation to within 4.4e-16 (relative) at 20 °C/50 %, 10 °C/80 %,
30 °C/20 %/90 kPa and −10 °C/70 %. It equals pyfar's `air_attenuation` at standard pressure, but
pyfar 0.8.1 differs by 27 % at 90 kPa: it computes the molar water-vapour concentration as
h = h_r (p_sat/p_r)(p_a/p_r), multiplying by p_a/p_r where ISO 9613-1 equation B.1 divides. Ours
and python-acoustics divide. The difference only appears away from 101.325 kPa.

### Speed of sound (`core/geometry.py`)

The default is Cramer's (1993) formula for humid air, with Davis's saturation vapour pressure and
enhancement factor for the water-vapour mole fraction (policy `humid-air-cramer/1`). It equals
pyfar's `speed_of_sound_cramer` to 6e-14 m/s at the same CO2 fraction, and is valid from 0 to 30 °C
and 75 to 102 kPa; outside that range the model refuses rather than extrapolates. At 20 °C and
50 % RH it gives 343.98 m/s. The earlier dry-air approximation c = 331.3 √(1 + T/273.15)
(343.21 m/s) stays selectable as `dry-air-core/1`; it is 0.77 m/s slow in 50 % humid air, which
over 3 m is 19.6 µs, about one sample at 48 kHz. pyroomacoustics uses 331.4 + 0.6T + 0.0124 RH
(344.02 m/s at 20 °C, 50 %), within 0.04 m/s of Cramer there.

The pipeline's timing anchor and the fitted room both use this speed, from `fit: {temperature_c,
humidity_pct}` (defaults 20 °C and 50 %). `timing_policy: dry-air-core/1` restores the dry formula
and `fixed-reference/343` a fixed 343 m/s. Humidity matters little at these distances; temperature
matters more: 5 °C moves the speed by 3 m/s, 0.08 ms over 3 m.

### Sabine and Eyring reverberation (`room/decay.py`)

T60 = 24 ln(10) V / [c (A + 4mV)], with A = Σ S α (Sabine) or −S ln(1 − ᾱ) (Eyring) and m the ISO
energy attenuation per metre. With air absorption off and both at 20 °C and 50 % RH, ours is 0.01 %
above pyroomacoustics' `rt60_theory` at α = 0.1, 0.3 and 0.6, exactly the ratio of the two sound
speeds (Cramer's 343.98 against pyroomacoustics' 344.02 m/s).

### Late reverberant tail across the band (`room/synth.py`)

The late bands tile DC to Nyquist: a low-pass at the bottom, octave bands,
and either a high-pass at Nyquist or an extension above the last complete octave.
The extension uses `1/T = 1/T_oct + c Δm/(6 ln 10)` and the squared air-loss ratio.
These band definitions are retained; R1 changes the time-varying envelope ordering.

For synthesis band j, let stationary unit-variance noise be ξ_j, synthesis filter
h_j, and envelope `a_j[n] = exp(-3 ln(10)*(t-tmix)/T_j) sin(πu/2)` across the
late fade. R1 emits `y_j[n] = a_j[n] (h_j * ξ_j)[n]`: **filter first, then
multiply by the envelope**. A hidden 0.5 s noise prefix warms the causal filter.
Early pressure has the cosine fade. The anchor remains absorption-only image
energy; there is no finite-draw RMS normalization or final envelope advance.

With analysis filter g, stationary covariance `R_j[k] = Σ_l h_j[l] h_j[l+k]`
and late anchor variance v_j, the expected analysis energy is

`E[z[n]^2] = Σ_j v_j Σ_p Σ_q g[p] g[q] a_j[n-p] a_j[n-q] R_j[q-p]`.

This retains temporal correlation and neighboring/edge synthesis bands. The old
squared cascade identity for enveloped *independent* input cannot be moved across
the filter and used here. `room/late.py` shares the covariance operator with the
fitter; native sample expectations are computed before energy binning. A modal
geometric sum accelerates exponential tails, with native recursion for ill-conditioned
cases. Geometry reuse changes cost, not the declared operator. Modal roundoff
(WP8E-NUM) remains open; fixed-energy comparisons do not certify optimizer parameters.

The w8f verifier reported 2,052 fixed-energy checks within 1.04e-13 of peak
(limit 1e-12), but rejected the parameter harness as unqualified by correct
controls. The registered join FAIL also remains. See the complete history in
KNOWN_ISSUES.md. Both synth and fit must change together; pre-R1 render-based
experiments require affected-path requalification.

### Decay times: Schroeder, T20, T30, EDT, noise (`room/metrics.py`)

Schroeder backward integration; least-squares fits over 0 to −10 dB (EDT), −5 to −25 dB (T20) and
−5 to −35 dB (T30), extrapolated to 60 dB; Lundeby's iteration for the noise crossing and Chu's
subtraction. pyroomacoustics' `measure_rt60` fits a line to the Schroeder curve from −5 dB down by
`decay_db` (in the log domain by default) and handles noise only by an optional `energy_thres`
truncation; pyrato fits by least squares and offers Chu and Lundeby. On white noise decaying with T60 0.600 s
(8 draws, 48 kHz):

| Estimator | No noise | Noise floor −70 dB |
| --- | --- | --- |
| ours T30 / T20 / EDT | 0.5984 / 0.5998 / 0.6000 | 0.5984 / 0.5998 / 0.6000 |
| pyroomacoustics T30 | 0.5984 | 0.5992 |
| pyrato T30 / EDT (Schroeder; Chu–Lundeby) | 0.6017 / 0.6003 | 0.5984 / 0.5999 |

All are within 0.3 % of the truth, and the spread between draws is about 0.3 %. Our
noise-compensated T30 equals pyrato's Chu–Lundeby value to four decimals. EDT starts at the
detected direct sound: with 20 ms of silence first, the onset is found 2 samples late and EDT reads
0.598 s.

**Per octave band** (`room-measured-metrics/2`), decay times come from the band filter applied to
the time-reversed response (Jacobsen and Rindel 1987): the filter's ringing then precedes each
event instead of lengthening the decay, which matters when the filter's bandwidth times the
reverberation time is small, as in low bands and short rooms. On exact expected energies (white noise with a known exponential decay,
T60 0.15–1.2 s, direct-to-reverberant ratio none to +6 dB, no random draws), the worst errors are:

| | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
| --- | --- | --- | --- | --- | --- | --- |
| T30, time-reversed filter (`/2`) | 0.00 % | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| T30, causal filter (`/1`) | 7.30 % | 3.57 | 1.81 | 0.85 | 0.40 | 0.20 |
| EDT, time-reversed filter (`/2`) | 3.78 % | 3.78 | 3.78 | 3.78 | 3.78 | 3.78 |
| EDT, causal filter (`/1`) | 63.8 % | 63.7 | 63.4 | 62.7 | 57.2 | 45.8 |

The remaining 3.8 % in EDT is the direct sound: in the closed form it is one sample at the onset,
and the reversed filter moves it to just before the onset, outside the decay curve. It matters only
when the direct sound dominates (+6 dB). The causal filter instead spreads it over the filter's
ringing, which the 0 to −10 dB fit reads as a fast early decay.

### Clarity C50, C80 and D50 per octave band (`room/metrics.py`)

ISO 3382 integrates the octave-filtered response from the direct sound. A causal filter delays
energy (group delay 14 ms at 125 Hz, 7 ms at 250 Hz), moving early energy past the 50 ms boundary.
`room-measured-metrics/2` filters forward and backward (zero phase, so nothing is delayed) and
integrates from one filter span before the onset, so the direct sound's spread-out energy is
counted as early. Against the closed form 10 log10(e^{13.8·0.05/T60} − 1), on exact expected
energies, the C50 error is:

| T60 | Convention | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 s | zero-phase (`/2`) | −0.17 dB | −0.04 | −0.01 | −0.00 | −0.00 |
| 0.3 s | causal (`/1`) | −4.25 | −1.98 | −0.96 | −0.47 | −0.24 |
| 0.5 s | zero-phase (`/2`) | −0.07 | −0.02 | −0.00 | −0.00 | −0.00 |
| 0.5 s | causal (`/1`) | −3.08 | −1.43 | −0.69 | −0.34 | −0.17 |
| 1.0 s | zero-phase (`/2`) | −0.03 | −0.01 | −0.00 | −0.00 | −0.00 |
| 1.0 s | causal (`/1`) | −2.36 | −1.09 | −0.52 | −0.26 | −0.13 |
| 2.0 s | zero-phase (`/2`) | −0.01 | −0.00 | −0.00 | −0.00 | −0.00 |
| 2.0 s | causal (`/1`) | −2.04 | −0.93 | −0.45 | −0.22 | −0.11 |

Over T60 0.15–1.2 s with a direct sound up to +6 dB, the worst `/2` errors are 1.12 dB (C50) and
1.29 dB (C80) at 125 Hz, 0.16 dB at 250 Hz and at most 0.04 dB above; the centre time Ts is exact.
The causal convention's worst case was 9.4 dB at 125 Hz. pyfar's default causal octave filter, of
higher order, under-reads more (−8.6 dB at 125 Hz, T60 0.5 s). pyrato's clarity formula is exact
(4.744 dB on the closed-form curve). Zero-phase filtering squares the band's magnitude response, so
the noise level is re-estimated on the filtered signal. `room-measured-metrics/1` remains
selectable (`policy=`) to reproduce earlier numbers.

**The room fit still compares under `/1`.** Its objective filters measured and predicted responses
with the same causal filter (the model side filters its early waveform and passes its late tail
through the filter's energy kernel), so like is compared with like; only reports and benchmarks
changed convention. The direct-to-reverberant ratio below 500 Hz is a weak construct in any
convention (the direct window is shorter than the band filter's response), one reason the fit's
default bands are 500 Hz to 4 kHz.

### Low-frequency limit: the Schroeder frequency

Below f_s = 2000 √(T/V) (T in seconds, V in m³), individual room modes dominate and a diffuse,
exponentially decaying tail stops describing the room; band decay values become position
dependent. Reports, `room_model.json` users (`RoomModel.modal_region()`) and render metadata state
f_s and the octave bands below it: about 170 Hz for a 61 m³ room with a mid-band T60 of 0.45 s, so
the 125 Hz band.

## Recognizer

### Mel filterbank and log-mel features (`edge.py`)

HTK mel scale 2595 log10(1 + f/700), triangles on the rfft frequency grid, no area normalization.
Bit-identical to `librosa.filters.mel(htk=True, norm=None)` (librosa's default is the Slaney scale
with area normalization). torchaudio's `melscale_fbanks(mel_scale="htk", norm=None)` matches to
7e-6, the precision of its 32-bit floats. Features: power spectrum of a symmetric Hann window
(`np.hanning`), frames starting at the window's first sample without padding, natural log with a
1e-10 floor, each window's mean removed. librosa with `center=False` and the same window agrees to
1e-4 log units (mean 6e-7); the difference is our per-window DC removal (below), and was 2e-6 before
it. Its defaults (centred frames padded with zeros, and a periodic Hann window) give 101 frames
instead of 97 per second and differ by up to 3.9 log units at the padded edges. Training and the
device share `edge.Features`, so they cannot diverge.

### DC offset on the device input (`edge.Features`), changed

A MEMS capsule read over I2S can carry a constant offset of a few percent of full scale. The modelled `pi5_inmp441/1` chain includes a DC blocker (3.7 Hz at 48 kHz), but other parts, I2S paths and recorded files may not remove it. A Hann window does not reject DC completely:
with 512-point frames at 16 kHz its main lobe reaches the first mel band (60 Hz), and the gain
normalization then shifts every band. The features now subtract each window's waveform mean before
framing (a DC block; the log-domain mean removal is separate). Before, an offset of 0.05 full scale
in a quiet room moved the features by up to 1.05 natural-log units; now it moves them by nothing
(`tests/unit/edge`). The default Pi training profile adds no DC offset. `iot_mic.random_chain` draws offsets up to ±0.05 in 30% of draws; it is used by the opt-in stress split, by legacy models evaluated without a recorded `device_split`, by the baseline recognizer's opt-in `mic_augment`, and by one fixed draw in `project evaluate`'s raw-render neutral probe. Those offsets still matter through asymmetric clipping.

### Resampling (`edge.resample`), changed

The device resampler was a 73-tap Kaiser (β 8) sinc with its cutoff at 0.95 of the lower Nyquist.
From 48 to 16 kHz it lost 1.5 dB at 7 kHz and 6.0 dB at 7.6 kHz, inside the mel range. The training
data is resampled with scipy's `resample_poly`, which loses only 0.25 dB and 2.4 dB there, so a
48 kHz WAV on the device gave top mel bands unlike training. It now applies the same filter as
`resample_poly`'s defaults: a Kaiser (β 5) sinc with ten zero crossings per side, cutoff at the
lower Nyquist, unit DC gain. It matches `resample_poly` to 4e-7 for 48, 32 and 8 kHz to 16 kHz.
Alias rejection is −14 dB at 8.5 kHz (it folds to 7.5 kHz, inside the mel range), −57 dB at 10 kHz
and −77 dB at 23 kHz. The old filter rejected more above 10 kHz but cut the top of the band. Live
capture with `arecord -r 16000` never resamples here.

### BatchNorm folding (`nn.export`)

w' = w γ/√(σ² + ε), b' = β − μ γ/√(σ² + ε), for convolutions without bias. With non-trivial
statistics this equals `torch.nn.utils.fusion.fuse_conv_bn_eval` within 6e-8; the exported numpy
network matches PyTorch within 1.2e-7 end to end.

## Simulated microphones (`iot_mic.py`)

- **Peaking and high-shelf filters** follow R. Bristow-Johnson's Audio EQ Cookbook. The exact
  identities hold: peak gain at f0 equal to the requested gain (to 1e-6 dB), 0 dB at DC and Nyquist,
  and the shelf at half its gain at f0, with 0 dB at DC and the full gain at Nyquist.
- **ADPCM-like stress codec** (stand-in for DECT's G.726): IMA-style adaptation, project-generated
  step sizes, not bit-exact. Steps are round(7·1.1^i), clipped at 32767; the project predictor's
  index-adaptation increments equal IMA/DVI's. This is not a G.726 implementation.
- **μ-law** (stand-in for G.711): a continuous μ = 255 curve on 255 levels, not G.711's eight linear
  segments. Against `audioop`'s G.711 on a 313 Hz sine:

  | Level | SNR ours | SNR G.711 | torchaudio μ-law |
  | --- | --- | --- | --- |
  | 0 dBFS | 37.6 dB | 39.5 | 37.7 |
  | −20 dBFS | 37.5 | 37.7 | 36.9 |
  | −30 dBFS | 37.2 | 34.2 | 37.1 |
  | −50 dBFS | 29.1 | 26.8 | 29.8 |

  Ours behaves like torchaudio's continuous law. It tracks G.711 within 3 dB of SNR, but sample
  values differ by up to 1084 LSB at full scale, where G.711's top segment has its coarsest steps.
  It is an approximation of the codec, labelled so in the module.
- **G.722** is modelled as a 50–7000 Hz band limit, not its sub-band ADPCM. **Packet loss** drops
  independent 20 ms frames.

## In-distribution levels and the Pi profile (WP5)

A digital full-scale sine has RMS `1/sqrt(2)`. Sine-referenced dBFS(A) is
`20 log10(sqrt(2) * RMS(A(x)))`; a desired floor L therefore needs weighted RMS
`10^(L/20)/sqrt(2)`. Omitting that factor produces the original ~3 dB error.
`pi5_inmp441/1` uses −26 dBFS sensitivity for a 94 dB SPL sine and −87 dBFS(A)
prequantization floor. The w5-device-fix acceptance run's **synthetic proxy** post-S16 floor was −86.6557 dBFS(A) (median of 3 seeds), median error +0.344 dB; its review measured −86.655 dBFS(A) over 30 seeds. Neither is a hardware measurement.

The profile separates the 60 Hz acoustic high-pass from a digital DC blocker of
`3.7 * fs/48000` Hz. Its modeled 129-tap causal linear-phase decimation FIR has
64 samples of delay and passband edge `0.423 fs`, **not** a −3 dB corner there;
it reaches roughly −6 dB at Nyquist. Vendor taps and phase are not known.

The in-distribution level operator measures power P over active event samples,
using a 10 ms energy gate at −25 dB within the span. A render referenced to L_ref
is scaled by `g = 10^((L-L_ref)/20) / sqrt(2 P)`. To preserve gain after a partial
crop, adjust the requested level by `10 log10(P_crop/P_full)` using the same
active-power function. The stress branch retains span-mean SNR. Silence and invalid
spans cannot supply a level. This shared convention is in `event_level.py` and
`training_device.py`; it is not whole-file RMS normalization.

Device-recorded beds use floating-point `b + D(event)` with clipping, without
another noise/filter/quantization pass on b. A synthetic SPL-referenced bed passes
through capture once. Take noise still scales with the event, and assumed SPL
priors cannot replace physical level calibration.

## Room-ensemble update (WP4)

The survey samples correlated log-uniform T30 marginals and maps them to effective
uniform absorption through the Eyring/air/obstacle budget. These are engineering
coverage assumptions, not identified wall materials or a population distribution.
For supported metric keys K, member i has prediction m_ik, observation o_k and
declared discrepancy σ_k. The implemented approximate update is

`L_i = mean_{k in K} ((m_ik-o_k)/σ_k)^2`,
`w'_i ∝ w_i exp(-L_i/2)`.

The mean (not a sum) tempers correlated metrics. At least two jointly supported
keys are required. Impulsive observations use T30 only; direct delay is admitted
only with independently anchored timing. Any observation outside every member's
support plus one declared discrepancy scale is a prior conflict. Effective size
`1/Σ_i(w'_i)^2 < min(3,N)` abstains. Repeated same-geometry playback shares model
discrepancy and cannot be treated as independent likelihood factors.

Predictive intervals are central 90% Gaussian-mixture intervals including the
declared discrepancies, not empirically calibrated predictive coverage. A 25%
relative T30 discrepancy alone imposes ~41.1% central-90% half-width, incompatible
with a 40% requirement even before ensemble spread. WP4(a) is **not accepted**;
missing reflective-room cells and broad intervals must stay in the denominator.

The opt-in modular discrepancy update separates uncertainty components and
early-field adequacy from late calibration. Its `room-research-ensemble/1` result
is research-only and refused by training; the legacy update above remains
available. No new predictive-coverage claim has been accepted.

## Feature realism and C2ST (WP1)

E1 compares the actual training stream and reference under matched framing.
Whole independent take groups are permuted between domains with their original
group counts preserved. RBF MMD² and per-band absolute SMD are jointly calibrated.
Each permutation's statistic is the maximum of normalized MMD² and the ranked
band-SMD ratio needed for 90% band coverage; observed and permuted samples share
symmetric marginal scales. With B permutations, `p=(1+#(T_perm >= T_obs))/(B+1)`;
`p <= 0.05` fails. Insufficient reference half-splits or fewer than five rendered
groups makes the result inconclusive. Real-only half-splits are descriptive,
not the acceptance null for full-size samples. Exchangeable independent groups
are an assumption; duplicate crops cannot create independent observations.

E2 fits an L2 logistic discriminator with grouped cross-validation and train-fold
scaling on log-mel summaries and embeddings. Its decision is ordered: if upper
95% AUC <0.5, inconclusive (CV instability); otherwise pass if upper ≤0.70 and
point ≤0.60; otherwise fail if lower ≥0.60; otherwise inconclusive. The fixed
out-of-fold score bootstrap omits refitting variability and can undercover.
A whole-group **refit** permutation p-value is reported separately; it does not
turn non-rejection into distributional equivalence. Each gate's own required
spike-ins must all be detected before that gate's pass survives. A weak sensitivity
result can downgrade pass to inconclusive but cannot erase fail.

The old framing FAIL remains: a pass/pass/pass rule had about 2.4% reachability
under the valid null. Later proxy controls qualify a narrower harness, not a
physical-room distribution or target recall. Identical-input controls only
establish route equivalence. Small-group E2 passes remain weak evidence.

Standalone independent-group MMD and the P5 family use biased, window-weighted
MMD² from the symmetrized stored group kernel. Finite binary64 block values are
accumulated as exact dyadic integers; unrounded rational statistics determine
inclusive ties in `p=(1+#(T_perm >= T_obs))/(B+1)`. Only displayed MMD² rounds.
Partition order, simultaneous block/count reindexing, transpose and side
interchange preserve the statistic for the same stored blocks/counts. Floating
kernel reconstruction and arbitrary group renaming are outside that invariant.
This does not replace the separate joint MMD/SMD calibration or paired baseline.
Earlier standalone MMD p-values and affected family decisions are historical.

P5's opt-in decision mode bounds the original fixed-budget p-values and stops
refits only when they cannot change the complete-family rejection event; it still
consumes the original random draws. Full diagnostics remain default. Independently
seeded cases can run in 1/2/4/8 processes and are collected in registered order.
These execution changes do not establish null calibration or defect sensitivity:
The A5d synthetic-proxy campaign **failed at four groups per domain**, as this arithmetic
predicts: the minimum exact two-sided MMD tail is 2/70, above the first Holm threshold
0.05/12. It passed at 6/10/15/18/20 groups for the registered procedural defects only
(see KNOWN_ISSUES).

Average precision now groups equal scores: with cumulative positives `T_g`,
cumulative examples `N_g` at each descending score group and `P` positives,
`AP = sum_g ((T_g - T_(g-1))/P) * (T_g/N_g)`. Unknown labels are excluded;
no positive denominator gives an undefined diagnostic. Earlier AP values retain
their original evaluator identity.

## False-alarm estimator and intervals (WP2)

For class c, floor count k_c over T hours, outside-context time fraction f_floor,
whole-context family incidence λ_f (contexts/hour), stratum weights w_fs summing
to one per family, and mean deployed-tracker output count μ_fsc:

`r_c = f_floor * k_c/T + Σ_f λ_f Σ_s w_fs μ_fsc`.

Compute separately for start alerts and confirmed events. Counts include repeated
outputs within each complete context; annotations of individual causes are only
diagnostics. Interactions belong in a mixed waveform/context and incidence, not
an additive sum of separately scored source probabilities.

For a standalone Poisson floor, the one-sided upper rate is
`χ²_(2(k+1),1-α)/(2T)`; the reported two-sided interval is the central Garwood
interval `[χ²_(2k,α/2), χ²_(2(k+1),1-α/2)]/(2T)` (lower is zero at k=0).
At k=0, `U=-ln(α)/T`: 2.995732/T at 95%. At T=1 h and k=1,2, one-sided upper
rates are 4.743865/h and 6.295794/h. About 3.0 h resolves a zero-count standalone
floor at 1/h. The composite allocates α/m to each of m = 1 + Σ_f|S_f| components
and uses the upper end of each component's central interval, a one-sided tail of
α/(2m): a zero-count floor needs `ln(2m/α)` hours per unit budget, 5.075 h (m=4)
or 5.635 h (m=7) at 1/h with f_floor = 1 (round up to 5.1/5.7 h), for the floor
alone. Selection on those data invalidates a naive post-selection confidence claim.

Nonconstant strata use parent-cluster bootstrap; floor blocks do not cross file
boundaries and retain the sparse Poisson envelope. For a constant observed count
c in n>1 IID parents with a declared population cap C, set
`q = 1 - α^(1/(n-1))`; the mean bound is `[c(1-q), c+(C-c)q]`.
For data-selected c, `Σ_bad p_c^n <= (1-q)^(n-1) Σ_bad p_c <= α`.
This qualifies the constant branch only, not the nonconstant bootstrap. One parent
or dependent constant observations require full support `[0,C]`. C is a physical
output-count cap, never the observed maximum. Skewed nonconstant data still
undercovers (review estimate roughly 1.6α).

The engineering sufficiency gate is 100 independent parents/family and 20/stratum.
A measured headline additionally needs 30 minutes of continuous measured exposure;
assumed prevalence/weights/floor fraction makes the composite model-based. Original
WP2 stress **FAIL** (2/3) and repaired **PASS** (3/3) remain alongside the later
sample-sufficiency inconclusive result. Low power (~0.69 pass under an unbiased
proxy, ~0.64 with 20% under-reporting) precludes a calibration claim.

## Compared in concept only

- **AGC** (`iot_mic.agc`): a 10 ms block level detector, gain moved toward a target with one-pole
  attack and release in dB, a maximum gain and a full-scale limiter. Production AGCs, such as
  WebRTC's, are more elaborate, with speech-gated adaptation; ours is a generic model of the gain
  pumping such devices produce.
- **Noise suppression** (`iot_mic.noise_suppress`): spectral subtraction (Boll 1979) with an
  over-subtraction factor, a spectral floor, and a noise estimate from the minimum of the smoothed
  power over about 1.5 s (minimum statistics in the spirit of Martin 2001). Production suppressors
  are more elaborate; this one reproduces the musical noise and gated onsets the recognizer must
  tolerate.
- **Dry recovery** (`room/dry.py`, `room/dereverb.py`): it uses the measured RIR. A regularized
  inverse (Kirkeby and Nelson 1999) plus statistical late-reverberation suppression (Lebart 2001;
  Habets 2009). Blind methods such as WPE (`nara_wpe`) need no RIR but more channels or longer
  observations; this module's evidence and limits are in `room/README_DRY.md`.
- **Sweep-train timing** (`room/sweep_detect.py`): matched-filter detection of the repeated sweep;
  clock drift from the median of the pairwise slopes of the repeat arrival times (a Theil–Sen
  estimate, robust to a misdetected repeat); sub-sample peaks by maximizing a Kaiser-windowed (β 6,
  ±32 samples) sinc interpolation of the correlation, which avoids the bias of the usual three-point
  parabola on sinc-shaped peaks. No open-source package does this for an unsynchronized phone and
  laptop; its accuracy was estimated in the historical, simulated pre-R1 walkthrough (KNOWN_ISSUES.md).

## Reproduce

```bash
pip install pyfar pyrato librosa torchaudio      # in a separate environment, with this package
python tests/validation/math_compare/room_chain.py           # images delay decay clarity bands sweep air speed reverb
python tests/validation/math_compare/recognizer_codecs.py    # mel logmel resample adpcm mulaw batchnorm eq
```

The `audioop` sections need Python 3.12 or older (`audioop` left the standard library in 3.13).
python-acoustics is optional; its ISO 9613-1 module is loaded on its own if the package is present.
