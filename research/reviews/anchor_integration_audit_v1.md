# Anchor integration audit v1

Date: 2026-10-10. Independent implementation/protocol audit of the fixed-anchor prototype. This is a validation-only research comparison, not a paper verdict or a predictive/novelty claim.

## Verdict

**No blocking findings in the final frozen snapshot below.** The finite-anchor mathematics, nonlinear source/readout separation, reference-inclusive normalization, observed-input boundary, integrity bindings and train/validation isolation are consistent with the specified protocol. Three initial P2 concerns were corrected during the audit and the new control paths were independently checked before this verdict.

Approval here means the bounded exploratory train/validation experiment can proceed. It does not establish novelty, benefit, biological correctness of inferred-tree agreement, or superiority over existing baselines.

## Scope and execution

I read AGENTS.md and operated through ssh anogena only in /workspace-SR003.nfs2/ARG-GNN. Every Python check sourced scripts/project_env.sh, used the existing project Python, and set CUDA_VISIBLE_DEVICES empty. No GPU job, package action, Git command or implementation edit was performed by this reviewer. This report is the only durable research output. Fresh CPU fixture scratch is inside .runtime/tmp; pytest caching was disabled. A failed shell-quoting attempt did not create the report and was corrected before this successful write.

Inspected: src/arggnn/age_interface.py, age_dataset.py, anchor_kernel.py, train_anchor.py and their four test modules; frozen configuration; age_conditioned_boundary_design_v1.md; age_geometry_qualification_v1.json; supporting boundary/kron interfaces, reference adapter, source loading, metrics, seeding and resource/launcher code. The root amended implementation/configuration while I audited the original snapshot, notified me when the new core was frozen, and I separately rechecked those changes. I did not edit implementation files.

## Initially actionable concerns and resolution

| Original severity | Final source location | Concern and resolution |
|---|---|---|
| P2 | src/arggnn/anchor_kernel.py:243; configs/anchor_inferred_cohort64_plain_pilot_v1.json comparison_note | Ordinary scalar mode uses K=(lambda+d)I, changing diagonal decay/gain as well as removing spatial propagation. The earlier statement that it removed cross-query propagation only was too strong. The final note correctly names an ordinary scalar temporal family, and adds same-diagonal and explicitly scoped gain-matched controls. On the first real inferred tile, diag(CQQ) ranged from 0.0217441 to 1.7096222, illustrating why identity decay is not matched full diagonal decay. Resolved. |
| P2 | src/arggnn/train_anchor.py:188–194,292–294 | The original active dependency receipt omitted metrics.py and train.py, despite their roles in checkpoint NLL, summaries, seeding and writing. The final receipt/drift list includes numerical, seeding, metric, adapter and resource dependencies; age dependencies are included when used. Resolved. |
| P2 | src/arggnn/train_anchor.py:216–217,230–263; src/arggnn/anchor_kernel.py:438–445 | The original result recorded only initial rates. The final selected model manifest records current rate/damping/mixture and raw values; epoch history records pre-clip gradient-norm means/maxima and maximum relative rate/damping movement. This makes the exploratory log/softplus comparison auditable. Resolved for pilot scope. Detailed modal duration*(lambda*mu+d) distributions would still be useful for diagnosing optimization, as suggested by design-note line 309. |

No additional P0/P1/P2 correctness issue was found in the inspected final snapshot. Depth changes active parameter count; matched local-capacity and same-anchor Transformer controls remain necessary before an exclusive mechanism/depth claim. The configuration explicitly limits the current stage to exploratory comparisons.

## Mathematical assessment

The fixed age potential implements Phi_s(t)=(1+a)t-a*s*atan(t/s). The scaled atan2 endpoint formula is the same integrated positive resistance for ordered nonnegative ages. With a=3, edge resistance lies between plain and four times plain. Minimization over unchanged interior nodes gives one-quarter*C0 <= Cs <= C0 in Loewner order. Endpoint telescoping supplies unary-subdivision invariance. Equal-age edges follow the explicit positive-resistance rejection policy, without an epsilon floor.

Age interfaces reuse the ordinary plain source bank, divide all branches by one full-boundary plain mean-diagonal scale, and retain fixed CPU-precomputed eigenpairs. matched_scalar_age multiplies both plain CQQ and CQR by the age/plain full-boundary mean-diagonal ratio. It therefore matches attenuation while retaining plain relative geometry. Plain repeated three times with distinct initial couplings [5,20,80] matches the declared multibranch temporal capacity. The final frozen study is plain-only and supplies no age-performance evidence.

The inclusive forward/exclusive reverse scans count each anchor self impulse once. Symmetric hold factors in reverse order yield T(y,x)^T for x<y, preserving the intended noncommuting chronological products rather than replacing them with an exponential of an integrated generator. Separate branch states and global nonnegative mixtures implement sums of kernel actions without online eigensolves.

The complete symmetric block kernel H and its global nonnegative mixtures are PSD by the covariance construction for symmetric positive generators. Rectangular CQR forcing and elementwise row normalization are not symmetric PSD Gram matrices. The manifest distinguishes the underlying block kernel from the normalized message. For gain_matched_diagonal, the PSD statement concerns its diagonal propagation kernel; the varying instantaneous source transform must not be relabeled as a single symmetric input-to-query Gram kernel.

Every layer evaluates shared bounded q/r maps only at fixed anchors. Query anchor states receive contextual updates and produce later-layer sources. References remain graph-local embeddings from the same plain encoder, fixed across kernel layers. Requested targets start from their containing-tree embeddings and receive each layer's exact-coordinate kernel action through the same update gate. They never become source impulses. Consequently, extra target coordinates, target permutation or duplicates cannot feed back into anchor context in exact arithmetic.

The denominator runs the same scans and mixtures with source 1Q-lambda*CQR*1R, so query and reference source mass is included. Nonpositive CQR makes the initial mass source at least one. Anchors have a positive self term; non-anchor readouts have strictly positive mass mathematically at finite rates, with explicit numerical underflow/nonpositivity rejection and no denominator floor.

Fixed physical midpoint anchors, cell-width masses and partial hold products preserve identical-record subdivision by the semigroup law. Telescoping age resistance and invariant selected leaf features preserve serial unary subdivision. The contract propagates inductively through nonlinear anchor layers. Graph serialization, packed-node storage, query order and arbitrary identity relabeling obey the intended invariance/equivariance. Changes in sample set, source roles, anchor count or tile endpoints are outside the contract.

## Additional diagonal controls

src/arggnn/anchor_kernel.py:198–218 and 239–253 correctly implement:

- diagonal: D=lambda*diag(CQQ)+d, original numerator/mass sources, independent query propagation.
- gain_matched_diagonal: transform every anchor numerator and matching mass by D*K_full^-1, then propagate independently under D.

The cached spectral source solve agrees with a dense solve and retains gradients. Since K_full is a positive definite M-matrix, its inverse is entrywise nonnegative mathematically; transforming numerator and mass identically preserves the normalized bounded-source interpretation.

D^-1*(D*K_full^-1)=K_full^-1 matches homogeneous continuous-forcing equilibrium gain. Likewise, integrating a homogeneous bidirectional impulse over the infinite coordinate domain gives the same factor 2*K_full^-1. It does not match arbitrary finite-anchor outputs, finite-domain boundaries or a changing-operator global equilibrium. The final source docstring, manifest and configuration explicitly state this restriction. The transformed control retains instantaneous cross-query source mixing and removes prior-state cross-query propagation; it should be interpreted with that distinction.

## Data integrity, split and resources

GraphSequence/model/operator bank inputs contain no targets. The source encoder uses only the selected observed leaf features and plain response. Raw parent-branch column 7 is excluded from the 47 selected fields. Reference statistics use observed topology, times and known reference categories. Query labels and nearest-TMRCA baseline probabilities remain external supervision and do not enter sources, generators or age sidecars. Label-counterfactual tests support this separation.

The boundary loader verifies manifest/preparation/source hashes, parent raw/prepared tensor/metadata binding, complete-parent admission, exact clipped tile features/geometry, exhaustive Q/R ordering, fixed parent-coordinate coverage and eigensystem reconstruction. Age sidecars bind the boundary manifest/preparation, each plain-bank hash, strict payload/provenance, implementation dependencies, artifact byte counts/hashes and exact plain-channel/partition/shared-scale correspondence. These checks support caches generated by the inspected frozen preparation code in a fresh process. They do not independently reconstruct an adversarially re-authored age operator from scratch.

The frozen boundary manifest hash matched 7f25874806535897d819b99d20c8f0da1b47f4fe10dde04280e7c42f6e1b9e09. Its source hash matched 611c7e32973c22bd35115b835279884f7b0c866136e91ae2c5f2a95daefd6fe7. Source protocol counts are train16/val4/test0. Train/validation sample identity sets have zero overlap. Boundary preparation admits train/validation only and splits parents before eight-way clipping. Runner evaluation reassembles each parent at fixed coordinates with exactly-once coverage and aligned query order, selects minimum mean parent-validation NLL, and never requests final-test data. Reused validation parents remain selection/exploration evidence.

The runner is sequential and has no DataLoader/process workers. It uses four CPU threads. Process title is set before CUDA inspection or contexts. CUDA entry requires an eligible physical GPU, matching inherited project lease and exactly one exposed device. Static launcher inspection shows scheduler locking, at-most-two project leases, eligible/free-device verification, inherited lease descriptor and child process-group signal forwarding. I did not exercise GPU behavior. The root retains physical-assignment and NVTOP responsibility. Model/cache/output paths remain repository-scoped, with storage/headroom guards.

## Bounded verification

Original snapshot: 76 targeted CPU tests passed in 9.59 seconds; the remaining tiny one-epoch runner test passed in 2.27 seconds. That integration test used a synthetic Git revision marker without running Git and redirected its checkpoint subtree into .runtime/tmp/anchor_runner_audit_v1.KCuG6f.

Final frozen snapshot: **109 tests passed in 10.18 seconds** across all four requested test modules. The same Git stub and checkpoint redirection kept every fixture write within .runtime/tmp/anchor_supplement_audit_v1.HzYUm9. These stubs affect provenance lookup/location only, preserving model training, checkpoint selection, parent predictions and result logic.

An independent real inferred train-parent-0/window-0 check verified bank checksum/dependency receipts, loaded [22,88,47] features and [22,64,64] CQQ, prepared 32 anchors/three plain branches, and ran depth-2 float32 forward/backward:

| Coupling | Finite output/all parameter gradients | Minimum row mass | Largest added-readout change in existing logits |
|---|---|---:|---:|
| Full | Yes | 0.218385279 | 3.5762787e-7 |
| Diagonal | Yes | 0.060273815 | 3.5762787e-7 |
| Gain-matched diagonal | Yes | 0.561825991 | 2.3841858e-7 |

All observed normalized message magnitudes remained below one. The tiny added-readout differences are consistent with float32 factor-splitting roundoff, not source feedback.

All four recorded qualification source hashes matched the inspected files. Qualification samples only five adjacent serialized graphs from one train parent per oracle/inferred input kind, not an unbiased panel and not coordinate-paired. Best-scalar residual ranges: oracle s350 0.114456–0.114511, s1400 0.022557–0.022702; inferred s350 0.151619–0.266044, s1400 0.037664–0.060141. Strict tau350 reference-grounded query counts were 1/64 in each sampled oracle graph and [0,0,1,0,0]/64 for inferred graphs. These establish local label-free non-scalar geometry/coverage observations only.

Tests and mathematical checks establish the bounded implementation contract. They do not establish predictive contribution or infer novelty.

## Final frozen source snapshot

- src/arggnn/age_interface.py: 15f9615a84a230386d8f71f5a317a46051c395f75408c79a6213f1b07f101f3d
- src/arggnn/age_dataset.py: b8585bf12c14201f46bda6b333c9dd23e8824c9af44cbe7173232f1c820e8a84
- src/arggnn/anchor_kernel.py: 2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083
- src/arggnn/train_anchor.py: 41c916217ece8dc541e7f5131dc3cb3dcdb839e58493c3e53ea4d3fe86c04cc3
- configs/anchor_inferred_cohort64_plain_pilot_v1.json: 4ef5da930cdcf2c327f7da036838685b911d03977b5be35107405020f49edf7d

This verdict must be reconsidered if these behavior-defining files change before the experiment.
