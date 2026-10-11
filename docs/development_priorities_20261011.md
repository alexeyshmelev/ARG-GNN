# Development priorities
Updated after the user's 2026-10-11 research steering.

The immediate objective is higher local ancestry accuracy against correctly applied native ARGMix and ARGformer on the development task.

- Use FP32 model and solver arithmetic for new candidates. Use mixed precision where appropriate. FP64 is an optional narrow diagnostic for a demonstrated numerical problem, not a development sweep or ancestry scoring requirement.
- Use one optimizer seed per new candidate during architecture discovery. Do not launch further optimizer-seed repeats of answered pilots. Larger simulations and independent confirmation follow a convincing pilot.
- Defer the remaining full-parent numerical-equivalence matrix and the unlaunched weak-backbone Census lineage sweep. Preserve existing evidence.
- Keep essential biological input, target leakage, split, query identity, physical coverage and baseline fidelity checks. Replace extensive implementation equivalence matrices with basic correctness and finite-loss checks.
- Prioritize the persistence-trace experiment: whole observed trees linked by genuine persistent ancestral branches, compared with the same frozen emitter plus ordinary genomic smoothing, sample-only continuity and shuffled ancestral correspondence.
- Report accuracy to two decimal places and NLL to four in summaries. Retain normal native metric values in raw receipts without using extra digits as evidence of research progress.
- Continue genuine native competitor fits already running. Do not restart them for seed variation.
- Current storage authorization is 1 TB, with a 950 GB scheduling ceiling. This supersedes the earlier storage figure in the original instructions.

The persistence experiment has no measured predictive result yet. All pilot claims remain separate from final independent biological validation and original-paper empirical comparison.
