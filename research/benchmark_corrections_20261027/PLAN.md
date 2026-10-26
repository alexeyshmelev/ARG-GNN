# Benchmark additions, 27 October 2026

User request: recover ARGMix real sample metadata and implement the separate AncestralPaths four-path training condition. Preserve healthy jobs, existing six-path experiments and author source files. Only these two topics are in scope.

## Completed work and open scientific questions

- [x] Recover and reconcile public MesoNeo identities, date records and reference assignments. Four reference cohorts match published counts. The recovery attempt is complete, but the exact 1,611-person roster remains unresolved as detailed below.
- [x] Implement supplementary Note 3.1: exclude BAA reference contributions before choosing informative ancestors, retain observed genealogies, remove BAA query rows, train a fresh four-output CNN.
- [x] Integrate prepare, fit, prediction and independent scoring stages with immutable inputs and distinct outputs.
- [x] Test author arithmetic, BAA-only ancestors, original sample identities, class rejection, fresh training, checkpoint round trips and metric definitions.
- [x] Prepare a runnable separate experiment plan. Preserve current live campaign.
- [x] Update README and concise research records with implementation checks and the metadata recovery outcome. Git publication is recorded in the commit history. Unrelated working files are excluded from this change.

## Scientific definitions

The original six-path benchmark remains primary for its original task. The new condition predicts four paths directly. It never merges six-output predictions and never initializes from the six-output checkpoint.

Note 3.1 specifies exclusion from GNN distributions and labels, not regeneration of simulated genotypes or re-estimation of genealogies. The four-path features are recomputed from the same observed trees because dropping a cached feature column cannot recover ancestors skipped after reference exclusion.

The new sampling allocation retains all original non-BAA focal rows: 40,000 training examples across three parents and 8,000 examples per each of five test parents. The present-day cohort has 2,000 examples per test parent. These counts are a declared reconstruction because Note 3.1 does not specify its own sample counts. Exact original identities and positions remain shared across methods. No test truth enters feature selection or prediction.

Full training and full feature production are not launched while the new implementation is being checked. CPU smoke checks are engineering validation, not benchmark accuracy.

## Verification outcome

All 33 CPU tests passed. The task factory smoke check found a canonical-path failure caused by the existing Python symlink. The saved plan now uses the actual provisioned python3.11 binary. All 15 task factories pass. Full benchmark training remains unlaunched and no new ancestry score is reported. Independent review found no substantive scientific blocker.

## ARGMix recovery outcome

[The recovery report](../argmix_roster_recovery_20261027/public_metadata/REPORT.md) and source tables identify 116 references: 13 CHG, 42 EHG, 48 WHG and 13 Yamnaya. The published QC list contains 1,492 individuals, and its intersection with the painted sample list contains 992. Original source tables, dates and joins are retained and validated against 12 source hashes.

Open questions are explicit rather than filled with invented choices:

- [ ] Recover the 3 farmer and 39 other ancient exclusions from that intersection.
- [ ] Recover individual ARGMix ADMIXTURE assignments separating Anatolian and Neolithic farmer references.
- [ ] Resolve the exact 660-person modern panel. Documented filters on available resources yield different counts.
- [ ] Establish the simulation age source and conversion convention. The full metadata and painted repository differ in 166 mean dates. For 107 references, two tables within the published workbook disagree.

No replacement simulation roster was emitted and no existing simulated data changed. Matching cohort counts does not prove the entire original roster has been recovered.

## Remaining execution

- [ ] Admit and run the separate four-class AncestralPaths plan. This implementation task does not launch the full benchmark.
- [ ] Add a separate four-output TreeLoom fit on identical focal support before claiming a new matched comparison.

No new accuracy results are available from these additions. Existing jobs and historical six-class results are preserved.
