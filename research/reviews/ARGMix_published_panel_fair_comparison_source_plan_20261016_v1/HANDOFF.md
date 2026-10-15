# ARGMix published versus prospective paired panel

A scientifically fair new comparison is possible without an April archive. Exact table-number reproduction remains unavailable. This packet is source/config only, inactive, and creates no execution authority.

The smallest next phase retaining the paper's training budget and IID replicate count is a **new synthetic 1,611-diploid / 3,222-haplotype panel**, no BAA: five TRAIN parents (500,000 target rows total), one VAL parent (100,000), and three IID parents (100,000 each). Compare current-release ARGMix transformer+BiGRU with count/role-adapted released Keras AP. Add the five misspecified parents after their exact prospective draws are frozen. A one/one/one adapter check is engineering evidence only.

Every method must receive the same realized genotypes/error, Relate trees, sample dates/roles, target coordinates/labels, parent splits and test rows. Representations may differ. Current ARGMix is not the paper's 25-reference + 3-ancestor model; released AP is not the paper's unrecovered PyTorch implementation. Real identities, original seeds and the paper's undefined ± do not block this prospective comparison; they block exact-reproduction claims.

## Concrete successor

1. Adopt or amend the explicitly new integer-age synthetic roster, error rule, donor policy and target sampler in `prospective_panel_config.proposed.inactive.json`.
2. Supply an authenticated empirical chr22 map and freeze its build/hash/interval/coordinate transform. Physical span is still unknown. Do not substitute chr1 or the old uniform 2 Mb recipe.
3. Build count-generic generator/Relate and method adapters in a successor, preserving native components. Remove hardcoded 476/952/686/266/1600 ranges. Reuse the prior 77-seed registry.
4. For NF/Yam dual roles, pass explicit query IDs while retaining reference metadata; exclude both focal diploid haplotypes per query in every donor adapter. The checked current Rust loop includes every reference ID and does not enforce this. Dropping one nearest reference is insufficient.
5. Freeze finite fit configurations, exact shared target rows and source/dependency/command manifests; then apply root-owned admissions. No execution is part of this review.

## No-BAA caveat

In the pinned AP graph, BAA is an unsampled terminal branch with no migration or contribution into any retained sampled population. With all other parameters/dates fixed, zero BAA sampling and deletion of its unused population/event have the same **marginal stochastic law** for retained lineages. This does not imply identical seeded realizations or establish the April implementation. It does not extend to graphs where BAA contributes ancestry/migration. The stock script still fails on an empty age list at `max(baa_times)`; skip/delete the unused event explicitly.

## Scope and scale

The published panel has 594 possible donor haplotypes before focal exclusion, 2,972 query haplotypes and 250 pure-founder haplotypes. It is 3.384× the 952-haplotype development panel. Paper 29-node subgraphs have 841 attention pair positions/head/layer/example; current 16-node subgraphs have 256. These are positions, not FLOPs. The real-panel 3,720,746 variants are not a simulated site count. No timing/storage forecast can use a guessed chromosome length; formulas are in `scale_forecast.v1.json`.

The proposed dates/error/sampling/OOD rule are disclosed new choices. Remaining concrete work is the map/interval, acceptance/freezing of prospective choices, source adapters, finite native fit manifests and OOD parameter materialization. Earlier eleven archival uncertainties are not eleven universal blockers.

No simulations, model execution, guards, array/checkpoint reads, downloads, jobs, Git or GPU/process probes were performed. Only this new review folder is written.
