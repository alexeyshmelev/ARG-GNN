# Independent prelaunch criteria for the complete joint model

29 October 2026. This review owns only this directory. Root owns integration,
qualification on the authorized GPU, training, deployment and Git. The reviewer
will not run GPU work, fit ancestry parameters, inspect new TEST payloads, alter
competitors or change simulation/inference recipes or healthy jobs.

## Evidence already established

The 28 October branch operator passed 18 numerical checks. Those checks are
accepted as prior evidence and will not be duplicated. They do not qualify an
observation adapter, complete initializer, focal decoder, training objective or
real-data throughput. The source design and independent review explicitly leave
these components open.

The earlier TreeLoom audit found legitimate observed-cohort information missing
relative to the author CNN, only 90 large optimizer updates despite 1.5 million
label presentations, and class collapse. These failures motivate input-policy
and optimization diagnostics. Existing TEST results were already inspected in
that historical audit; they must not guide architecture or hyperparameter
selection now. The earlier 128-draw fit/64-draw internal-development exercise
shows small-model learnability, not a benchmark result. The old focal-executor
GPU comparisons establish bounded functional parity only; strict one-step AdamW
parameter differences and fixed-order timing limitations remain intact.

## Required before starting the new benchmark fit

1. Freeze a complete, executable predictor. Record hashes for model, adapter,
   training harness, protocol, manifests, any compiled helper, and the exact
   environment. Declare ancestral measure, genomic support/normalization,
   coordinate sampling, duplicate policy, finite-root coverage, zero-length
   branches, forests, sampled-interior support or explicit rejection, full
   initializer, number of joint layers, and focal readout. Each mode must run
   the same registered observation-to-logit path during fitting and evaluation.
   If there are optimized paths, qualify the complete path, including gradients,
   against a slow reference before using them.

2. Demonstrate actual deep ancestry access. Retaining every node or running a
   complete-tree initializer is insufficient evidence that a new branch message
   arriving at a distant ancestor reaches a leaf-only focal decoder. Specify an
   all-depth path/readout or an equivalent dependency mechanism. Use a deep
   genealogy whose informative ancestor is beyond the local layer count. Hold
   the focal local neighborhood and leaf channel fixed, alter only a distant
   foreign ancestral state, and demonstrate the intended focal-logit dependence
   and finite nonzero gradient for a deterministic nondegenerate setting. Also
   verify that a distant reference observation can influence the initialized
   focal prediction. A fixed-hop truncation must be named as such; it must not
   be presented as complete ancestry context. Report optional unary/zero-length
   refinement sensitivity instead of claiming invariance without evidence.

3. Preserve input and identity semantics. Require a feature whitelist separating
   observed genotypes, inferred topology/dates, observed sample age/cohort, and
   allowed pure-reference labels from query local ancestry and simulation truth.
   Stable IDs are routing metadata only. Authenticate native node/sample roster
   mapping to original observations; merely attaching equal ID arrays to a
   permuted leaf matrix is insufficient. Verify a same-age roster permutation,
   native-node renumbering, target/sample position joins, and independent-parent
   namespacing. Query reference-class fields must remain unknown even when
   observed cohort predicts a subset of classes. No simulated ancestral IDs,
   source-population truth, label-dependent pooling or numeric ID embeddings.

4. Freeze the target population and split. Use explicit parent/chromosome lists
   with TRAIN/VAL roles, immutable source references and disjoint parent IDs.
   Never randomly split nearby targets from one chromosome into claimed
   independent validation. Context cannot cross parent or chromosome boundaries.
   If only a held-out TRAIN parent is ready, name it internal development and
   reserve its entire chromosome; do not call it the untouched benchmark TEST.
   Existing independent VAL can be used once its unchanged pipeline is qualified.
   Register which validation metric selects checkpoints or register a fixed final
   step in advance. No new TEST data or labels may be read for these choices.

5. Make sample weighting explicit. State the exact canonical training target
   sequence, duplicate multiplicities, per-target objective, class/cohort weights,
   and draw law. Retain benchmark sampling multiplicities unless a separately
   named condition changes them. Context quadrature weight is not target-loss
   weight. Context nodes have no additional loss. Multiple focal targets sharing
   a packet are each counted exactly once per intended draw, and repeated
   contexts do not multiply target weight. For fixed canonical-target batches,
   aggregate summed losses and divide by their registered target count, retaining
   that denominator across unequal microbatches. For uniform batches of S sites
   from M sites and N targets, use M/(S*N) times the summed target losses: dividing
   by the realized query count is a biased ratio estimator when site sizes vary.
   Select the fixed site group before resource-based microbatch splitting; retain
   its denominator throughout. Data bucketing/adaptive group sizes must not
   silently change inclusion probabilities, discard or preferentially resample
   expensive examples. Epoch metrics use total loss divided by total targets.
   The exact counterexample and original registry audit are recorded in
   REGISTRY_AND_WEIGHTING_REVIEW.json. This paragraph refines the initial draft,
   preserved as PRELAUNCH_CRITERIA.initial.md.

6. Match the architecture controls. A minimal direct joint-model comparison
   should use native-only, same-leaf and branch modes with identical observed
   packets, complete initializer, focal readout, common tensor initialization,
   target sequence, optimizer protocol and checkpoint policy. Seed equality
   alone does not guarantee matching common tensors when module construction
   consumes RNG in different order: compare initial tensor hashes or copy the
   common initialized tensors explicitly. Declare disabled/extra parameters and
   gradient-presence behavior. If only the candidate begins first, call it a
   candidate benchmark fit and defer architecture/correspondence claims until
   these controls are trained. The old frozen model changes schedule as well as
   carrier and cannot alone isolate the branch mechanism.

7. Qualify the complete model numerically on real TRAIN observations. Require
   finite logits, loss and all present gradients for each mode, including a
   mixed-size batch, a deep/unbalanced tree and edge cases the adapter supports.
   Check individual-versus-disjoint-batched outputs and full input/parameter VJPs;
   changing another independent window must not change the first output/gradient.
   For a claimed equivalent execution optimization, prospectively use FP32 logit
   tolerance atol 2e-5, rtol 2e-4 and full-gradient relative L2 at most 1e-4,
   identical gradient presence and finite values. These are measured engineering
   tolerances, not formal equivalence or optimizer-trajectory guarantees. They do
   not compare different architecture definitions. Test serialization/reload of
   the exact complete model and optimizer state on the training harness path.

8. Account for all work and retained information. GPU receipts must separately
   report observation reads, topology validation, overlap construction, immutable
   cache lookup/hit rate, complete initialization, transfer, forward, backward,
   zeroing and optimizer update. Also report end-to-end steady-state useful
   target presentations/second and optimizer updates/second for the registered
   effective batch, cold-start cost, peak host RSS, allocated/reserved device
   memory, node/edge/SNP counts and size distributions. Include at least one
   realistic high-depth/high-edge packet; balanced toys alone are insufficient.
   Do not drop full trees, remote ancestry or difficult examples to satisfy a
   timing gate. Cache immutable topology/observations only, with keys covering
   parent, topology, dates, ordered stable roster, measure and coordinate policy;
   learned states cannot be reused across optimizer updates. If preprocessing
   moves offline, report its time/storage and amortization explicitly.

9. Register a feasible launch envelope and meaningful learning diagnostics.
   Root must choose device/memory headroom, worker/thread counts and batch packing
   from measured workload before launch, preserving other jobs. Enforce finite
   checks and report failures with the exact target IDs; no silent label/packet
   skipping, tree truncation or automatic changed hyperparameters on retry.
   Record loss, per-class support/predicted counts, gradient norm/clipping,
   learning rate, effective targets, presentations and optimizer steps. Provide
   early TRAIN learning behavior, common-tensor/head changes and VAL behavior
   when available. All classes need not be predicted in the first few updates,
   but persistent collapse/nonlearning is an explicit diagnostic, not a success.
   Do not mechanically copy the CNN's update count or equate epochs, target
   presentations and useful optimizer steps. Select the schedule using TRAIN/VAL
   evidence and preserve failed attempts under distinct names.

## Needed for stronger claims, not a blanket barrier to a candidate fit

A branch-mechanism claim needs coverage/geometry and capacity matching, followed
by a branch-only alignment destruction control that preserves leaf identities,
observed age/cohort/reference strata and coverage as far as possible. The branch
operator can otherwise exploit inferred-root availability, source-clade size,
dating errors, cohort composition or generic smoothing. A gain after shuffling
branch correspondence would weaken the intended explanation.

A carrier-versus-schedule claim needs selected-cut and branch exchange under a
common schedule/readout, preferably crossed with the old schedule. A fine-cut
approximation can test integration resolution and cost. No claim of matching
historical ancestors follows from shared descendants and overlapping ages.

A biological performance claim needs held-out-parent NLL, class/cohort recall,
boundary behavior, short-tract recall and false switches where the registered
endpoint supports them. Do not trade erased short tracts for average accuracy
and call that improved local ancestry. Report uncertainty at the independent
parent/chromosome level; correlated targets are not independent replicates.

A performance advantage needs matched complete-model/end-to-end measurements,
repeated order or explicit fixed-order qualification, and convergence cost to a
registered TRAIN/VAL criterion. High sampled GPU utilization, layer-only timing,
or cached-batch speed does not establish sustained useful training throughput.
No arbitrary utilization percentage is itself a correctness or launch gate.

## Review artifacts to collect

- Frozen source/config/data hashes and complete input-policy/split statement.
- New adapter/full-predictor tests and deep-context witness, without repeating
  the already completed 18-test operator suite.
- Root's exact GPU numerical and complete-throughput receipts, plus failure logs.
- Initial common-tensor/readout checks, target-weighting audit, and launch config.
- Training progress/checkpoint receipts with source lineage and untouched TEST.

Pending status: criteria proposed before the 29 October implementation or GPU
measurements. Final pass/fail requires the concrete sources and receipts.
