# Cue validity development controls (synthetic proxy)

This lane supplies deterministic correctness controls, not an independent
acceptance receipt or trained-policy efficacy study. Test fixtures are disclosed
development data. No third-party audio, RIRs or weights are included.

Run `python -m pytest tests/unit/t3_label_policy -q` with the package on PYTHONPATH.
The scalar crop oracle enumerates sample/frame boundaries separately from the
implementation. The mask control has two alternative sufficient cues: a time mask
removes the first and a frequency mask removes the second. Each alone is allowed;
their union must be rejected. A nontrivial safe mask and valid middle/late crops
must survive. BCE has a hand-calculated log(2) value and [-.25, 0, .25] gradient.
Guardrail boundary controls use 20 paired examples plus 5 examples of the other
class, explicitly only two source contents/parents, then remove one cell or one accepted
augmentation to cross 0.80/0.25. These are exact deterministic boundaries; there is
no stochastic power or household inference in this test. Every attempted candidate,
including retries, enters the augmentation-success denominator; a fifth retry
per successful example cannot masquerade as 100% proposal acceptance.

`check_handoff.py` is explicitly invoked on the integration owner's patched tree.
It proves mixer/policy/mask/loss routing, keeps validation unmasked, checks source
history and parent gain against an analytic sinusoid, and exports trace hashes.
Its tiny trainer control deliberately uses the same mocked corpus for train/val
and a declared observability hook: this is software routing evidence only.

Required reviewer work before acceptance:

1. Independently review the sufficient-cue annotations, candidate code, test
   collection and known-good/bad controls; test a plausible counterexample.
2. Freeze and take external custody of registration, candidate/patch/environment
   digests, fixture/split IDs and all required nodes. Run in fresh processes.
3. Apply the handoff hunks only after coordinating parent-level and negative
   integration owners, then rerun controls on the merged tree. This standalone
   module cannot establish that the trainer called it.
4. P6 downstream adoption still requires the matched four arms and P0 gates:
   >=0.05 equal-parent macro-recall gain; no additional missed parents per class;
   no additional real-background starts/confirmed outputs; adjusted one-sided
   95% nuisance ratios <=1.00 for both types. A zero comparator count gives an
   inconclusive ratio, without pseudocounts. Qualify interval null/error and
   minimum-defect power at the independent group count before opening the audit.
   No data/power, no default activation. Identical features test route equivalence
   only; these fixtures never claim distributional realism.

Support schema: `cue_valid/1`, `units: samples/mel_bins`, explicit `reviewed: true`,
`reviewer`, `annotation_id`, `source_id`, `source_sha256`, `evidence_domain`, exact
`feature_grid` (`fs`, `n_fft`, `hop`, `n_mels`, `fmin_hz`, `fmax_hz`), and `classes`.
Each class declares `state: present|verified_absent|unknown`; a present class has
`alternatives`, a list of sufficient conjunctions of `{start, stop, mel_bins}`.
Samples are rendered-source half-open intervals, bins are zero-based. Annotation
identity and review attestations are provenance, not cryptographic proof of human
correctness. All relevant framing/filter changes require re-review/rebinding.

The fix2 controls retain the independent verifier's positive-to-absent and
background-parent failures as disclosed regression fixtures. Newly verified
absence is forbidden from either a present or unknown baseline. Every target and
background row must match its annotation's parent, including rejecting missing
or empty parents. Rejected ledger appends leave no partial entry or reserved ID.

The amended handoff renders two examples from the same initial RNG state using
`same_draw_no_partial_no_masks/1`: an unaugmented comparator first, then the
training example. The comparator disables partial windows and feature masks but
keeps the source, bed, parent level and mixer draw. Their source/annotation hashes,
recorded bed-window hash, mixer RNG hash and requested level must match. Separate
rendered comparator labels supply the denominator. Source-positive annotations or
accepted candidate counts cannot substitute for it. A copied corpus prevents
comparator draws from inflating training telemetry, and the candidate's RNG stream
is unchanged. Each consumed example has a unique seed/batch/index ID. Labels in
the trace must exactly match the cells supplied to BCE.

Loss mass retains the registered unit-valid-class-cell interpretation, not a sum
of realized BCE values. A crop followed by a mask is one composed proposal; each
subsequent mask retry is another proposal. Crop and mask attempts remain separately
visible. The ledger records each final example once, including failed/skipped
proposals, and checks the unchanged 0.80/0.25 thresholds. The trainer saves the
trace and rejects a failed or inconclusive intervention guard before model export.
The no-partial/no-mask comparator arm has no eligible augmentation; its ledger
remains inconclusive rather than manufacturing an augmentation pass. Scientific
promotion always remains disabled pending independent downstream P6 acceptance.

The former random partial-window export fixture is preserved as an expected
rejection: its disclosed retention fractions are 0.70 and 0.5625. A fixed safe
crop provides the separate known-correct export control. Another control removes
all but one positive in each twenty to reproduce the reviewer's low-retention
failure. An unmocked WAV-manifest/mixer/features/trainer control uses analytic sine
bands, distinct train/validation parents and workers=0. These are synthetic proxy
software controls, not distributional or held-out efficacy evidence.

Confuser-enabled paired training currently fails explicitly: the negative owner
must supply a reviewed per-draw identity/provenance hook. A recorded bed's identity
cannot establish the overlaid confuser's identity. The legacy negative path is
unchanged; this limitation blocks a matched-negative P6 study until integration.
Worker-process tests need the verifier's host environment; do not treat sandbox
deselection as passing evidence. Any merged dependency or shared-hunk change
requires the affected controls to be rerun on that exact candidate.
