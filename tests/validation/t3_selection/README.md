# Operational selection development controls

These fixtures are disclosed **synthetic proxy software controls**. They are not
TRUTH/MEAS realism runs, fresh acceptance panels or deployment recordings. The
implementer does not issue acceptance receipts. The independent verifier must
qualify controls, freeze its own required collection and take custody before
acceptance. Run these ordinary regression controls with:

```sh
python -m pytest -q tests/unit/t3_selection tests/validation/t3_selection
```

`operating_point.replay_policy` calls the actual `edge.Tracker`, preserves state
across score chunks, and observes activation and qualification decisions. A
separate scalar queue/counter reference enumerates 2,916 five-hop sequences.
Correct fixtures and designated bad cases test pre-event alarms, one-hop starts,
multiple outputs, final flush, nonmonotone threshold splitting, stale caches,
missing hooks, incorrect units, role/content overlap, absent classes/families,
constant scores versus constant audio, and fixed-seed candidate selection.

The exact McNemar calculation is checked by independent integer-binomial
summation. With four discordant parents p<0.05 is unreachable; with five all
improvements it is 0.03125. At 20 independent discordances and conditional
improvement probability 0.8, the exact rejection probability is about 0.8042.
This conditional sensitivity calculation does not establish power at 20 total
parents or resolve actual-source transfer. Population independence and meaningful
effect sizes require a separate prospective protocol.

The bank schema is `operational-bank/1`. The module docstring specifies each row.
Supply identities for train/development/audit together before any experiment.
The optional `negative_training: {file, sha256}` reference consumes lane A's
`negative-training/2` inventory and its persisted split lock, excludes its
training/mining ancestry, and checks that both files remain unchanged. Include
that reference whenever the model uses lane A's pool. It is an exclusion adapter,
not a grant of training eligibility or a source of held-out negatives.

Use a list of `required_families`, or a role-keyed mapping when development and
audit reserve different families. Every target parent has one nominal complete
context; other views use diagnostic/playback/stress panels and add no primary
recall denominator. Source, session, decoded content and original parent content
identities are mandatory. Lossy re-encoding still requires preserved original
lineage; signal hashes do not prove independent acquisition. Never score a dry
target or an RIR as a continuous deployment recording.

`candidate_identity` binds model bytes, features, class order, seed, checkpoint,
code revision and full policy. `infer_scores` checks the model and calls
`edge.Model/window_scores`; `cache_scores` provides an explicitly synthetic
control route. Cached scores bind hop size as well as model/features/bank/role.
`score_candidate` retains every context and both endpoints; output provenance
contains no local model path. `search_operating_point` replays the finite grid
(up to 4,096 policies and 50 million replay rows by default) without dropping
contexts or assuming monotonic event counts. Resource violations raise errors.

`select_candidate` minimizes the declared ordering among feasible candidates:
negative worst recall, negative macro recall, nuisance starts, nuisance confirmed
outputs, earlier checkpoint, threshold tuple, identity. Mixed seeds and audit
scores cannot select. `calibrate_ap_selector` takes raw AP diagnostics and gives
the earliest AP winner the identical finite policy search. An infeasible AP
winner has no substitute. `checkpoint_callback` adapts a frozen development bank
to the integration-owner trainer hook; the selected checkpoint must be retained
without a later persistence fit or model refit.

`outer_screen` compares an already frozen candidate/comparator pair. It preserves
each comparator-successful parent, checks zero background outputs and the
conditional rate bound, and refuses extra nuisance outputs. The audit seal and
single-use policy remain the independent runner's responsibility. Two zero-event
hours give a 95% homogeneous-Poisson upper bound of about 1.497866/hour, which
cannot support 1/hour. No helper claims natural-event recall or rank transfer.

The F1 amendment validates nuisance families against each row's own role. A
role-keyed mapping must name train, development and audit. Additional families
must be declared in `optional_families` (a list for all roles, or a complete
role mapping); required and optional lists must be unique and disjoint within
each role. Optional families are reported explicitly, contribute every nominal
start/confirmation and parent trigger, and obey the same trigger-fraction limit.
Only their minimum parent-support floor is optional. Empty optional cells are
reported with zero parents and undefined fractions, never as measured zero risk.
All nominal nuisance totals and binary parent outcomes are aggregated directly
from rows, independently of the family-floor loop.

The F2 amendment adds `evaluate_checkpoint_selector`, which captures the actual
checkpoint bytes/features/class order before calling the selector, rejects
checkpoint replacement even on an infeasible result, verifies the full candidate
digest and canonical policy, and binds the result to the emitted model. The
integration owner must call this wrapper from the shared trainer hunk. It checks
identity and routing; callback-reported efficacy still requires independent
verification.

The shared probes intentionally use filenames outside normal pytest discovery:
run each explicit file on the candidate with the exact shared handoff applied:

```sh
python -m pytest -q tests/validation/t3_selection/shared_checkpoint_controls.py tests/validation/t3_selection/shared_handoff_controls.py
```

Do not combine those filenames with their parent directory in one collection:
pytest can omit the explicitly named nonstandard files in that case. Check the
collected node list against the frozen required set. The F1 regressions in
`tests/unit/t3_selection/test_review_families.py` are part of the default suite.
The controls preserve the disclosed verifier counterexamples: role omission in
both development and audit, false outer PASS, and wrong-feature callback. Added
callback controls reject class-order, digest, policy and checkpoint mutations;
the correct control preserves numerical weights, features and full policy.
These are reusable deterministic regressions, not fresh statistical observations.
