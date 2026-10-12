# WP8 room recheck

Historical internal figures from the legacy dEchorate edition are withdrawn;
dEchorate HDF5 is used only for internal validation from the user's own CC BY 4.0
download. See [dataset attribution and changes](../../../THIRD_PARTY.md#audio-data-sources).
The HDF5 recompute remains **inconclusive**: its requested source selection is
unavailable. No replacement acoustic table is justified.

Start internal development with the metadata-only feasibility check:

    python -m tests.validation.bench_room_recheck.preflight_v1 \
      --data-root DATA_ROOT --output JOB/evidence/preflight.json

The v1 reader verifies pinned file integrity, parses literal geometry assignments
without executing the publisher's script, and refuses unavailable selections.
Rate and source-axis interpretation depend on the recorded review because the
archive lacks those attributes. Short native responses can leave decay metrics
unavailable. The reader preserves its supported development partition; it does
not substitute another edition or silently drop missing groups.

`corrected.py` accepts a data root, an external preregistration, its SHA-256
record and a new evidence directory. It verifies the hash and source
availability before join/render work and refuses to overwrite an output:

    python -m tests.validation.bench_room_recheck.corrected \
      --data-root DATA_ROOT --preregistration JOB/preregistration.yaml \
      --prereg-sha256 JOB/evidence/prereg_sha256.txt --output JOB/evidence/run_1

The `sensorsim-room-wp8-hdf5/1` registration must declare
`data.edition: dechorate-v1-5562386`, geometry, seeds, split, windows, bootstrap
count, thresholds and decision rules. A changed experiment needs a new
registration. Historical registrations and failures remain in internal evidence.

The diagnostics compare join energy against paired image-source continuation,
normalize tail energy by isolated direct energy, compare early peaks against a
matched random-time null, and retain finite counts and unavailable decay/clarity
metrics. Automatic noise handling applies to measured RIRs; simulations declare
zero noise. These are **measured-RIR proxy** comparisons, not deployment results.
Model-only joins and rendering quantities are **model-based**.

`run.py` retains computation helpers and refusal-only compatibility entrypoints;
its historical command and `_load_pairs()` cannot read inputs or create output.
`source7_development` remains an explicitly selected descriptive development
route using the same external inputs and separate T2/WP8 splits. It cannot
satisfy the unavailable registered selection.

No archive data ship. Independent control signoff and external registration
custody are required before acceptance. The preflight emits feasibility evidence,
not an acceptance receipt. A qualified grouped C2ST and evaluation-owned
recognizer/take/background inputs remain separate requirements. These tools do
not establish user-room realism, recognition performance or measured false alarms.
