# Independent pilot code audit
Audited source commit: 7d67a404f34980a5fbc65fdf2a7b6cf5bdda25c6.
Reviewer started with fresh context and read only the declared project code and protocol.
This is a code/protocol audit, not a manuscript acceptance review.

The reviewer found fixed-coordinate migration labels, parent-before-window splits and the exact interval-refinement argument sound.
No input-label leakage, persisted credentials or explicit external output writes were found.

## Findings
1. P2: full-resolution switches partitioned only at migration endpoints. A confirmed two-tree fixture missed two ancestry transitions.
2. P2: checkpoint paths based only on run basename could collide.
3. P2: dataset tensor hashes were recorded but not verified by training, and run results lacked a dataset-manifest digest.
4. P2: storage preflight occurred only before the whole data-generation batch.
5. P2: GPU leases existed only in the launcher and could disappear while detached children remained alive.
6. P3: the initial pip inventory contained an inaccessible conda build-machine file URL.

The primary agent integrated fixes after this audit. Updated tests and source commits document validation.
The v1 jobs were already running with distinct checkpoint names and unchanged tensors. Their recorded source commit remains the original version.
Their full-resolution switch diagnostic must be corrected separately before use.

## Scientific gates
Stronger reference-conditioned baselines, capacity/tuning comparisons, paired parent-level uncertainty, inferred genealogies, demographic shifts and tract-level evaluation remain necessary before performance claims.
The declared pilot only measures oracle engineering behavior and headroom.
NLL uses an explicit probability floor of 1e-12. This affects hard nearest-reference votes and must be considered in calibration comparisons.
