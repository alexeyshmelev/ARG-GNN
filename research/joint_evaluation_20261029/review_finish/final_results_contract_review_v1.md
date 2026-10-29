# Independent final scientific contract review

Reviewer: benchmark_audit_oct29. Date: 2026-10-29.

This statement reviews the scientific contracts and the independent audit implementation. It does not assert that the still-pending American and ARGMix production comparisons have completed or passed numerical review. Only a successful final_results_*_completed_v1.json receipt from the independent audit can make that claim.

## American comparison

The reviewed joint model is bound to freeze a203156e0de57b9a536d176542b84c1fea934f70f8200cb3a57a35129af060c7. Its prospective protocol retains 20 original TRAIN parents, 200,000 canonical midpoint targets, all 158,476 retained windows, the original complete trees and genotype context, and unknown query ancestry. The grouped-parent objective weights each canonical target uniformly in expectation, including every partial batch. The fixed 13 complete epochs require 128,869 optimizer updates and 2,600,000 target presentations. The final checkpoint alone authorizes held-out inputs.

All 15 held-out prediction products from each of the three methods must be authenticated before private ancestry is read. The common target keys include parent, raw record ID, observed node, original sample index, physical midpoint and the direct class order EUR/EAS/AFR. The independent audit does not call the production label join or metric helper: it independently joins half-open ancestry intervals using left boundaries and raw query-sample identity, reconstructs confusion counts and class metrics, and checks equal-parent means, sample standard deviations and paired differences.

Five validation chromosomes and ten TEST chromosomes remain separate. The final report must not substitute validation accuracy for TEST accuracy or select a new joint-model checkpoint from either set. Model development and checkpoint-isolation evidence remain bound to the frozen source, training plan, fit product and genuine producer waits.

ARGformer is the explicitly disclosed reference-only PCA10 nearest-class-centroid evaluation of author embeddings. It is not native top-one retrieval. ARGformer and FLARE predictions enter this comparison as unchanged one-hot hard-call transports. Their Brier and NLL values describe that transport, not their native posterior calibration; a zero probability for any true class produces positive-infinite NLL without clipping. FLARE's original rounded ancestry posteriors remain separate. Where native checkpoints were intentionally reclaimed, the audit authenticates retained producer metadata and reports that their bytes could not be rehashed; it does not claim to rerun native inference.

## ARGMix comparison

The reviewed joint model is bound to freeze c3303268253987f305d6cb298c625ccae9a669d4785c5181f5d8f38922c66634. The original recipe retains five TRAIN, one validation, three IID TEST and five misspecification TEST parents. The joint model uses all 500,000 TRAIN targets for five complete epochs, with 2,500,000 target presentations, complete native trees and genotypes, and the same corrected grouped-parent/tail objective. Its final epoch5 checkpoint precedes all nine held-out predictions. Validation is a separate postfit diagnostic and does not select that checkpoint.

The native ARGMix comparator retains its original policy: earliest highest validation example accuracy, a 40-epoch cap and patience8. This is legitimate validation selection, not TEST selection. The independent audit verifies the native fit's TRAIN/VAL identities, selected-model provenance and recorded epoch history; it must not force this comparator to use the joint model's fixed-epoch policy.

The exact original 100,000 query/tree targets per parent retain sampler order, semantic diploid/haplotype identity, inferred-tree index and physical midpoint. All 800,000 TEST targets are retained. The audit verifies both native prediction barriers, the paired barrier, every genuine prediction and scoring wait, and the direct class order Ana/CHG/WHG/EHG. It independently traverses each original truth tree to census350 using the original raw-sample mapping and physical target coordinate, then compares every rebuilt label with the original label product. Cohort counts, per-parent accuracy, IID/misspecification means and sample SD, and paired differences must agree. Confusion, class precision/recall/F1, Brier and NLL are additional independent diagnostics; the original ARGMix scorer reports accuracy only.

The 1611-diploid roster, ages and synthetic older-NF proxy are disclosed reconstruction choices. The proxy is not an identified Iceman, and the recipe does not establish recovery of original paper participants or random streams. The joint graph/genotype representation differs from ARGMix's original compact ancestry tokens and distances.

## Interpretation and completion requirements

These are comparisons on shared simulated benchmarks with documented paper-guided reconstruction choices. They do not establish exact replication of a published numerical table, training-seed uncertainty, universal superiority, or performance on real populations. Sample SD describes variation between independently simulated parents and is not a confidence interval. Results must retain their original endpoints and cohort definitions even when methods rank differently across groups or diagnostics.

A numerical PASS requires all expected scoring waits to be successful, every owned worker session to be absent, all completion and prediction hashes to authenticate, exact canonical coverage and class order, checkpoint and input-provenance checks, and independent metric agreement. Pending or malformed evidence cannot produce a successful final receipt. No training, GPU launch, source change, probability modification, target filtering or checkpoint selection is performed by the audit.

Prelaunch scientific evidence remains in american_v2_checks.json, american_review.md and argmix_final_checks_v1.json in this directory. The separate completed European direct-four result audit is european_four_completed_checks_v1.json; its primary present-day result must remain distinct from its overall result.
