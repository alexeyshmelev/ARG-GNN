# Performance risks and scientific attribution

28 October 2026. Source-derived assessment of frozen v1, SHA256 `e0528c8e0d01b69512fe588dc74ec966a3855489abf4160a8ef4b9111e5a4e12`. This note introduces no new performance measurement or predictive result. The registered 400-update diagnostic remains on v1.

## What is measured

The real five-tree coordinate probe contains 9,515 nodes, 952 samples, 100 SNP occurrences and one focal row. On two CPU threads it measured 0.0651 s preparation, 0.2281 s structure forward, 0.1625 s backward and 1.019 GB peak process RSS. The efficient same-packet no_cross forward was 0.0631 s. These are single untrained calls, not steady-state training, GPU results or a 20-tree batch qualification (`REAL_COORDINATE_CPU_PROBE.json`).

The older 128-tree GPU profile found substantial copies and indexed backward operations, but used a different model/executor. It motivates examination of similar operations and cannot establish the new bottleneck quantitatively (`research/scalability_20261024/profiling/REPORT.md`).

## Actual bounded-executor costs

`window_model_frozen_v1.py:457` visits upward levels, then downward levels. Upward messages and learned transforms touch each actual node/edge a bounded number of times. However, every upward level calls out-of-place `index_copy` on full-node `child_message`, `total` and `up` fields (the first may be absent at a leaf-only level). Every downward level similarly copies the full-node `outside` field. Structure executes two sweeps.

With N node occurrences, width d, U upward levels and D downward levels, the implemented full-field traffic is O(N d (3U+D)) per sweep, in addition to O(N d^2) learned transforms. This is not the linear-work tree algorithm implied by counting only edge messages. Backward traverses the index-copy chains and full-size gather/scatter gradients too.

**This does not prove O(N d (U+D)) persistent activation storage.** Autograd need not save every full-field value when an operation's derivative needs only indices/shapes; gathered features and linear inputs can be compact. The code proves repeated full-size output construction and a deep indexed autograd graph. Exact saved-tensor residency, allocator reuse, transient peaks and backward traffic require a profile. Do not call all historical full-field versions simultaneously resident without that evidence.

For four five-tree packets with the same 1,903 nodes/tree as the probe, N=38,060 and one FP32 [N,32] field is 4,871,680 bytes. If U=D=50, two sweeps write roughly 1.95 GB through these full-field updates, before reads and backward. Fifty levels is an illustrative assumption, not a measured depth for the diagnostic. Balanced and highly imbalanced inferred trees have different depth distributions.

SNP activations can also dominate. `encode_snps` evaluates its MLP only at categorical allele rows, which is good, but gathers [J,H,d] values and materializes [J,H,13] observed-group membership (`:410`). If 20 selected trees contained 400 SNP occurrences, these two FP32 fields alone would occupy about 48.7 MB and 19.8 MB. Actual SNP count is variable; a fixed number of full trees does not bound it without the loader's explicit cap. A selected native interval can extend beyond the 50-kb coordinate-sampling radius. Its full SNP pool extends the actual observation support.

The 23,822 parameters and Adam state are small compared with these observation/activation fields. Parameter compression is not the immediate storage issue.

## GPU suitability and low-risk improvements

A 20-tree batch provides many leaf rows at the first levels, but only about 20 ancestral roots at the deepest levels. Narrow d=32 projections, a serial tree-depth dependency, and many small gather/linear/norm/scatter kernels can limit GPU occupancy. A GPU may help, but neither constant utilization nor acceleration over the measured small CPU case follows from the node count.

`context_kernel` (`:350`) creates separate per-window matrices, so kernel storage is O(sum_w T_w^2). At K=5 these matrices are tiny. It nevertheless repeats `torch.unique`, dynamic `nonzero` and `int(batch.context_kind[win])` for each cut/leaf kernel. On CUDA, conversion of a device scalar to Python forces synchronization; dynamic-index discovery can also synchronize. `EdgeSum` (`:311`) recreates arange/stack routing arrays at every depth. These are plausible launch/synchronization costs, not measured percentages.

The following are justified candidates, in priority order:

1. **Exploit exact output dependencies.** After the sole cross-tree exchange, second-pass context trees cannot affect focal outputs. Compute the first sweep for all K source trees but the second sweep only for complete trees owning requested focal rows. For one focal tree/window this reduces full-tree passes from 2K to K+1; K=5 gives six rather than ten. Compute temporal target rows and target-cut pooling only for those focal trees, retaining all source trees and source weights. In no_cross, only focal-tree encoding/sweeping is necessary. Qualify logits, gradients and optimizer updates before adoption.
2. **Cache observation-only execution data.** Window row lists/kinds, PyG edge indexes, degree, age geometry, cohort membership, and allele-frequency feature tables are fixed observations. Prepare them once. Gates, SNP MLP values and hidden states remain learned and must be recomputed after each update. This removes avoidable tiny operations without a new scientific approximation.
3. **Deduplicate within one forward only.** Repeated coordinate draws may select exactly the same topology, dates, SNP pool and metadata. Their first-sweep state can be computed once and gathered with correct backward multiplicity. The second-pass states generally differ by coordinate-specific messages and must not be deduplicated merely because topology matches. No learned-state cache may survive an optimizer update.
4. **Reduce full-field copy chains after profiling.** Store compact states by level or use a qualified custom tree forward/VJP with preallocated buffers and explicit dependencies. Naive in-place replacements can violate autograd saved-value rules. Prior level-local and tiny-checkpoint candidates were slower; their names or mathematical sparsity do not establish a faster implementation here. The downward PyG call currently aggregates one message into its own unique target, an identity operation that could be removed after parity while retaining actual upward PyG message passing.

Stable log-weight normalization is appropriate before broad length-scale tuning: a zero-mass forced focal plus tiny learned length can leave every positive-source exponential underflowed. The current defaults are finite in the recorded probes. Stable log normalization is a numerical successor requiring qualification, not evidence that v1 has already failed.

Measure synchronized complete forward/backward/update time, preparation, transfer, actual target presentations, host/device peaks and time to a development criterion. Compare with an optimized focal-tree-only baseline as well as the matched-packet control. Do not call no_cross's current all-K first sweep the fastest possible one-tree method.

## What cut transport adds, and what it does not

At scale s, the cut route is

    z_us = learned_projection(first_up_u[rep] + first_outside_u[rep], flags)
    m_ts = D_ts^-1 A_ts^T sum_u W_tu A_us z_us.

The leaf route uses the first leaf readout h_ui directly and then injects its distance-weighted difference at leaves. Cut transport accesses intermediate ancestral-node states that may have been compressed away by the width-32 leaf readout, pools via target descendant sets, and injects at ancestral representatives before the second sweep. These are different computations and inductive biases.

They do not introduce new observed information: both are deterministic functions of the same inferred forests, genotypes and metadata. No strict expressivity or information-theoretic superiority is established. If cut tokens were simple means of recoverable leaf features, much of the operation would reduce to extra projection/pooling of leaf information. The present nonlinear representative state is not generally that mean, but that fact alone does not prove useful ancestry signal.

Moreover, structure has three cut branches with three independently learned genomic lengths initialized at 2/10/50 kb, while leaf_only has one 10-kb length. It has more active parameters, access to additional internal states, and distinct injection locations. A gain over leaf_only would still not isolate multiscale clade correspondence. A cut with mostly singleton groups can act as another transformed leaf kernel, rather than a meaningful ancestral-group exchange. Report group sizes, fractions of leaves in nonsingleton groups, invalid-time singleton fraction, root saturation and overlap entropy by cut.

## Decisive controls

Prioritize a small sequence rather than a large search:

- **Information shortcuts:** metadata-only; focal SNP plus metadata; pooled SNP neighborhood plus metadata with no genealogy. The six-class BAA endpoint is strongly constrained by observed cohort, so recovery of BAA accuracy cannot by itself show a graph benefit. Keep legitimate cohort metadata in every principal comparison. Removing it is an information-policy ablation, not an allegedly fairer main baseline.
- **Local genealogy:** the optimized focal-tree encoder with the same focal SNP/cohort head. Compare actual topology with a controlled topology perturbation only if leaf observations, dates and target identity remain valid. Degraded incoherent trees are not a substitute for a well-specified nongraph control.
- **Cross-tree mechanism:** leaf_only, then a capacity/scale-matched multiscale leaf route before the three-cut claim. Also compare a same-tree local clade adapter to distinguish extra nonlinear local capacity from foreign-tree evidence.
- **Partition semantics:** replace cuts by singleton partitions or size-matched random partitions. If randomization is used, keep the same set of per-tree observations and source weights. Permute only the cross-tree leaf correspondence within observed metadata strata; do not simultaneously permute genotypes or targets. These diagnose descendant correspondence separately from generic smoothing.
- **Representation versus routing:** use cut tokens obtained by pooling first-sweep leaf states instead of direct internal states while retaining the same cut routing. This separates added internal-state access from the overlap transport itself.
- **Genomic context:** fixed source observations with distance weighting removed or coordinates permuted. The current kernel depends on absolute pairwise distance, not a directional sequence state. A left/right reversal is symmetric; averaging can cancel different signals on opposite sides of a boundary. A later directional or attention readout is a new hypothesis, not a promised fix.

Use independent development parents and then untouched confirmation. The registered 128 fitting draws, 64 internal-development draws and one seed can test learning and expose gross class collapse; they cannot identify these mechanisms or yield chromosome-level confidence intervals. Fixed cached coordinate draws define a fixed sampled-context empirical objective. Short-tract recall, boundary precision/recall and false switches need dense evaluation support. A kernel's constant/mass property is a primitive algebra identity, not conservation by the full gated nonlinear predictor.
