# WP1 acceptance records — proxy results

The historical registered repair acceptance **failed**. Do not describe that revision as a
validated in-distribution evaluation gate. The unmodified WP1(a–d) numerical
criteria still pass in the labelled historical raw-window diagnostic.

Registration: `preregistration.yaml`, SHA-256
`18867e5b7ef8ad6c0b6d636fd42236ace747a0848b4b9d2b8d98e8c8d30c2219`. Acceptance ran once at implementation
commit `7b65b91865213718cc76af5c81d80e2836b3579f`, with `JOB` unset.
No acceptance threshold changed and no acceptance rerun was performed.

| Proxy criterion | Result |
| --- | --- |
| E1 null calibration, 300 runs, group variance 0.2, window variance 0.8 | 22/300 rejections (7.33%); registered binomial count interval [8, 23]: pass |
| Shift 0.6 SD in 10 of 120 features | 100/100 rejections: pass |
| Independent TRUTH draws, 6 s versus 2 s, one event window/group | E1: fail, fail, pass; E2: fail, inconclusive, inconclusive: fail |
| Same framing with +6 dB at 3 kHz | E1 and E2 fail in all three replicates: pass |
| +2 dB EQ and PCM8 spike-ins, 15 groups per side | Each detected 3/3 seeds; conditional detection fraction 1.0: pass |
| Original WP1(a–d) controls | All four pass; JSON replay byte-identical |

Every number above is a proxy result. No user recordings or measured RIRs were
used. The frozen CNN fixture is untrained. Three seed detections are conditional
on the same retained groups, not independent replication power.

The failed record is retained. The valid 6 s versus 2 s same-pipeline control
shows **no length effect** in the independent re-review, with or without drift;
this is **not a drift cause**. The registered criterion required E1 and E2 passes
in all 3 replicates. At 15 single-window groups per side, the review estimates
only a **2.4%** probability of meeting that criterion under a true null (range
0.7–6.4%). The criterion was nearly unreachable; the harness was not invalidated.
The review's maximum paired |t| was 2.99 at 100 ppm and 2.92 at 0 ppm.
These are attributed independent proxy-review results, not reruns of this record.

The embedding AUC 0.298 [0.124, 0.498] is a small-sample CV artefact. The final
rule treats an upper AUC bound below 0.5 as "inconclusive: CV instability".
The review's exact interval for 22/300 is [4.65%, 10.89%], consistent with 5%.
Its fixed-score bootstrap excluded 0.5 in 12–14% of null runs; a refit group
permutation test rejected in 4/100 at 0.05. An E2 pass is weak evidence: single
views still passed in 10–40% of runs at true AUC 0.70–0.80, and in 52/100 runs
at 60 groups per side and true AUC 0.75. Those numbers are review evidence.

E1 and E2 must each detect every spike-in for every seed to retain their own pass.
Without that qualification the result is "inconclusive (insufficient power)".
A new prospective registration with fresh seeds is required for acceptance.
Before the new prospective run below, the gate was **implemented, gate not accepted**. The failed registration,
its thresholds, results and hash above remain unchanged.

## Final prospective registration — PASS, accepted on the TRUTH proxy

Registration `final_preregistration.yaml` was committed as `c94b024` before the
single execution at implementation revision `c1b09868a13199b2cfe692b2a56697cb09802e29`.
SHA-256: `9d9e13513f5b72bbb7171fc2942221dfce8c945d560edd9868834b3d0f6fff7f`.
The old failed record above is retained, and no old acceptance seed was rerun.

| Registered proxy criterion | New result |
| --- | --- |
| Null replicates with any E1/E2 fail ≤ 4/20 | **0/20**: pass |
| 300 paired takes, 120 band features, no Bonferroni exceedance | **0/120**, max absolute t **2.689831**, two-sided Student-t cutoff **3.569409** (299 df): pass |
| +2 dB / 3 kHz detected ≥ 19/20 | **20/20**, E1 and E2 each **20/20**: pass |
| +6 dB / 3 kHz detected 20/20 | **20/20**, E1 and E2 each **20/20**: pass |
| Refit-permutation null calibration, 100 runs, 15+15 groups, 99 permutations | **1/100** at p ≤ 0.05, within registered binomial count interval **[1, 10]**: pass |

Descriptive only: E1 null passes **20/20**; E2 null passes **5/20**, inconclusive
**15/20**, failures **0/20**. Per-test power qualification against the two
registered EQ controls leaves those counts unchanged. Five null per-view
intervals had upper bounds below 0.5 and were classified as CV instability.
No event window was clamped in this control. Both slow tests passed in **234.84 s**.

The exact two-sided Bonferroni cutoff follows the registered formula, rather than
the re-review's rough 3.3 approximation. The fixture is an untrained CNN and every
result is **proxy**, with **no measured results**. This accepts the event-framed
same-pipeline TRUTH control; it does not establish primary training-stream length
invariance, room/device realism, deployment power or measured false-alarm rate.
The below-chance, bootstrap-coverage, weak-pass and framing caveats remain in force.
