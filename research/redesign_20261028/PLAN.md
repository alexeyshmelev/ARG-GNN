# TreeLoom redesign, 28 October 2026

The owner authorizes a substantive redesign following the accuracy audit. Preserve the existing TreeLoom, author baselines and scientific observations. Expensive continuation of the old fits remains held. New isolated implementation and TRAIN-only diagnostic experiments are allowed. Root integrates, schedules and publishes. This work does not yet establish an accuracy improvement or a novel operator.

## Questions and decision criteria

1. Can bounded groups of complete inferred trees support useful, frequent minibatch updates without global chromosome episode construction?
2. Does cross-tree transfer through the same leaves and multiple genealogical partitions improve held-out local ancestry over a simpler tree encoder with the same observed inputs and optimization opportunity?
3. Can physical or genetic coordinate sampling bound computation while preserving the meaning of irregular tree intervals and retaining short ancestry tracts?

Both repaired and redesigned conditions must receive legitimate cohort, reference and age information, with query ancestry truth excluded. All labels join through canonical example_index, never registry storage order. The biological simulation and native author methods stay fixed.

## Parallel ownership

| Owner | Deliverable |
|---|---|
| Architecture agent | Diagnose current learned operator, formulate bounded alternatives, implement isolated PyTorch/PyG prototype and structural tests. |
| Data agent | Exact observed TRAIN extraction, bounded windows or coordinate sets, stable sample identities, observed features and focal-target contracts. |
| Literature agent | Full-source reading with explicit scope, prior-work challenge, novelty limits, mathematical properties and falsification criteria. |
| Root | Register hypotheses, integrate contracts, review mathematical and empirical claims, qualify meaningful diagnostic runs, publish concise outcomes. |

## Candidate comparison

A minimal condition uses complete local trees and finite context with minibatch updates. This is a new finite-context model, not a numerically equivalent executor of the old chromosome model. The substantive candidate uses shared-leaf transfer between changing ancestral partitions, with multiple scales and explicit genomic distances. Tree-specific internal node IDs are never assumed to denote the same ancestor across different trees.

Coordinate sampling is under investigation: sample a bounded number of genomic coordinates from a registered neighborhood density, always include the focal tree, and retain every sampled tree in full. This avoids treating each serialized tree as an equal unit of genomic distance. Duplicated tree computation may be reused while retaining sampling weights. An unbiased linear integral estimator does not make nonlinear predictions or gradients unbiased. Consecutive-tree windows are a useful control, with their different context coverage disclosed.

All claims about subdivision invariance must specify the operator, feature field, support selection and nonlinear readout. A separable kernel, ordinary temporal GNN or a GRU after tree embeddings is not by itself a novelty claim. Prior transport, signature-kernel and shrinkage proposals in the ledger must not be relabelled as new discoveries.

## Sequence and gates

- [x] Preserve and read the completed accuracy audit.
- [x] Assign disjoint architecture, data and literature work to three agents.
- [x] Register the new bounded-learning hypothesis and explicit relation to prior work.
- [x] Select the mathematical formulation after adversarial literature review.
- [x] Extract exact small real TRAIN inputs without global episode recipes.
- [x] Verify feature, node, leaf, target and interval contracts.
- [x] Verify operator properties and batch isolation with CPU tests.
- [x] Integrate one real TRAIN forward/backward diagnostic and measure preparation, model time and memory separately.
- [x] Run a registered learning diagnostic and compare useful work per optimizer update. Any training accuracy is diagnostic only.
- [x] Freeze an internal development split by complete parent, never by neighboring windows. Use two original TRAIN parents for candidate fitting and the third for internal development, or the separate independent VAL after its inference is available. Native baselines fitted on all three TRAIN parents cannot be treated as held-out comparators on that third parent.
- [ ] Compare the simple and cross-tree candidates under matched observed inputs, targets and a declared tuning budget. Include no-cross-tree and one-tree controls.
- [ ] Advance only if development learning and actual end-to-end efficiency justify larger training. No TEST-driven selection.

The published comparison requires a separately frozen model and independent confirmation. Existing TEST results informed the failure audit and must not become an architecture-tuning set. No production queue is changed by this plan.

## Resource and reporting scope

All computation and files remain in /workspace-SR003.nfs2/ARG-GNN on anogena. Current owner authorization is GPUs 3, 4 and 5 and 1.2 TB project storage, superseding historical values in AGENTS.md. Initial work uses CPU only. GPU diagnostics require root scheduling and the prescribed process title. Preserve healthy ARGformer and every unrelated process. Do not promise constant utilization or a final accuracy gain before measuring them.

Report design decisions, failed hypotheses and measured results in this directory and the main README. Do not replace old evidence or silently alter competitor methods.

## Frozen implementation and current execution

The first executable candidate is `architecture/window_model_frozen_v1.py` (SHA256 e0528c8e0d01b69512fe588dc74ec966a3855489abf4160a8ef4b9111e5a4e12). The mathematical and literature assessments support testing shared-leaf alignment between ancestral-age partitions, while explicitly rejecting a broad new-kernel or first-joint-tree claim. The alternative preserving old recurrent memory remains a design specification, not an implemented comparison arm.

The independent technical review passed nine targeted tests. The coordinate packet preserves five complete trees, 9,515 nodes, all 952 leaves and 100 observed SNPs. Its untrained CPU forward/backward costs were 0.228/0.163 seconds with two threads. These are one-packet probes, not training throughput or ancestry accuracy.

Registered pilot: 128 uniformly selected fitting draws from original TRAIN000/001 and 64 internal development draws from TRAIN002. Both new-model arms use identical initialization, contexts, target order and 400 minibatch updates. The no-cross arm removes communication. A leaf-only control and a control with matched numbers of distance scales remain required for mechanism attribution. Final metrics use a fixed checkpoint, never development selection. TEST is excluded.

The data worker exported and validated all 192 registered packets using two workers. Root completed the learning harness and both fits. Independent review verified model/data contracts, source registration, target joins, objective and saved scoring. The old full-parent continuation and all native baselines remain unchanged.

The separate focal executor v2 processes only trees with focal predictions in the second within-tree pass. This removes operations with no path to a focal output. The symbolic predictor is unchanged, but floating-point operation ordering differs. Logit and gradient qualification passed. Strict CPU clipped-optimizer parameter qualification passed 28 of 33 cases, so exact optimizer equivalence is not established. V2 did not replace the frozen pilot implementation and is not deployed in the historical production queue.

## Completed learning diagnostic

Both frozen v1 arms finished 400 updates. Fitting/development accuracy: no-cross 79.69%/68.75%, cross-tree 77.34%/70.31%. Development macro F1 is 60.21%/65.82%, cross-entropy 0.5786/0.5824. The difference is one of 64 targets, without an independent replication. This supports learnability, not superiority. Root will not expand claims from this small diagnostic. See LEARNING_DIAGNOSTIC.md and root/learning_summary.json. A larger adequately fitted comparison, direct-input controls and leaf/capacity/partition controls remain unfinished.

A separate agent completed the requested sliding joint-graph convolution investigation in research/sliding_graph_convolution_20261028. Its mathematical formulation, prior-art assessment, receptive fields and CPU operator are preserved separately from the bounded prototype.

## Focal executor and GPU qualification completed

The separate executor v2 omits unused context trees in the second pass and in the no-cross arm. The independent review qualifies FP32 functional equivalence, not identical optimizer trajectories. CPU strict parameter tolerances fail in five of 33 cases, with small downstream differences retained in the report. All four registered GPU comparisons pass the prospective logit, full-gradient and gradient-presence tests. The cross-tree batch-16 observation improves from 280.61 to 263.58 ms, with 1.609 to 1.171 GB peak allocated GPU memory. No streaming training or sustained 80% utilization claim is supported. All eight GPU profile processes completed, with no optimizer updates. The initial missing-NVML import failure and readable attempt-2 launcher remain preserved.

The separate sliding branch-convolution operator is implemented and independently critiqued. Its 18 CPU tests pass. A complete ancestry model wrapper, matched initializer/readout, real-data preparation profile and controlled learning comparison are still required. The operator alone is not an ancestry result or a general novelty proof.

## Next discriminating experiment

The first bounded learning diagnostic and the separate sliding-operator investigation are complete. The scientific question remains open. Do not rerun the 128/64 pilot to select among new architectures.

1. Build the sliding model with the same observed SNP encoder, complete-tree initializer and focal ancestry readout as its controls. Keep a separate model identifier and preserve frozen v1.
2. Register a larger fitting/internal-development support drawn from the existing original TRAIN parents, with chromosome separation. Freeze sample IDs, observations, labels, optimizer opportunities and metric definitions before fitting. Keep TEST excluded.
3. Compare no cross-tree communication, same-leaf communication, selected-cut communication and branch-overlap communication. Separate the effect of synchronous layer scheduling from the effect of the correspondence. Match capacity or report a declared capacity control.
4. Measure real window preparation, worker supply, transfer, forward, backward and optimizer time. Bare-layer timings and fixed-batch GPU activity cannot substitute for full training throughput.
5. Use development accuracy, cross-entropy and class/boundary behavior together with cost to decide whether the additional branch coupling is worth keeping. A useful shared-leaf baseline is acceptable if more elaborate correspondence adds no reliable benefit.

These are pending research steps, not launched jobs or completed comparisons. The 18-check operator receipt qualifies its implemented numerical properties, not biological identity of inferred internal nodes or a new operator family.
