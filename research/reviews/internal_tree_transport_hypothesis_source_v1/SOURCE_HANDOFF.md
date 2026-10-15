**Minimal marker-grid reset-chain source handoff — source only**

Implemented `src/arggnn/marker_grid_reset_chain_gnn_v1.py`. It provides the full paired trainable model and observation preparation API, not an execution wrapper. Actual fitting remains conditional on root review, H046's result, a new registration and root's resource/execution closure. No hypothesis ID or live allocation was created.

**What is concrete.** The module instantiates the original 30,788-parameter H042 body with its actual two PyG MessagePassing modules. It directly reuses H044's `backbone_state_v1`, which runs H042's complete two tree sweeps, SNP encoder, all952 haplotype contexts and full bidirectional genomic GRU. It introduces no duplicate tree/context layers. The classifier still supplies whole-tree logits; its first32 state channels supply the exact second-sweep leaf context to the rate conditioner.

The 84-parameter marker head, 1,057-parameter graph conditioner, one base-rate parameter and three simplex logits give exactly31,933 parameters per arm. Both new output layers start at exact zero. The family API deep-copies every initial weight and creates independent reset AdamW optimizers. Both arms initially have identical chain outputs. Their original H042 emission values are preserved at initialization, while adding the common chain already changes their final predictions relative to H042.

Preparation consumes an existing H044-bound observation bundle and its exact original bin-edge array. Byte identities are checked before converting integral bp coordinates to int64. New evidence accumulation, learned operations and chain arithmetic are FP32. The qualification executed no FP64 arithmetic. The source accepts the original stored coordinate arrays solely for bound identity/integrality checks and integer conversion, not FP64 accumulation.

Each new-channel SNP belongs to one half-open scored bin. Interior-edge SNPs enter the right bin. Observed coverage tails outside scored bins remain in the original tree body. A usable new-channel site requires a called query and at least one callable reference in all four classes, matching H046's common-callability rule, while reusing H044 categorical/allele-cardinality tables. Missing bins emit exactly zero marker correction. This is stricter than H044's per-class missingness features and is now explicit in the design note. No target enters preparation or forward computation.

The transition uses the exact analytic reset-generator formula and stable `expm1`/`logaddexp` arithmetic. It reuses H039's balanced log-semiring scan with an explicit initial simplex prior. Query chunks contain the entire original position sequence. Carries cross all positions through balanced upsweep/downsweep; there are no position resets, truncated histories, per-position Python/CUDA loops or detached genomic gradients. The default checkpoints the full decode for each32-query block. Returned values are normalized log marginals of shape[P,686,4]. The unchanged `original_grid_span_nll_v1` is exported for the existing TRAIN-loss path. Existing probability sealing can apply the same `.softmax(-1)` used by H046, with that exact convention frozen for both arms before fitting.

**Actual small qualification.** `tiny_fp32_receipt_final_v1.json` binds the final source hash. With Torch2.6/PyG and FP32, the5-position/two-query fixture enumerated all1,024 ancestry paths. Analytic transition error versus a tiny FP32 matrix exponential was2.33e-9. Maximum log-marginal error was9.54e-7 and gradient error4.47e-7. Transition row-sum error was0. Disjoint-bin counts and missing-bin zeros matched the handmade expectation. Both arms had31,933 bit-identical initial weights and outputs, two inherited PyG modules, and exact query-chunk parity on this fixture. Initial rate-output gradients and base-logit gradients were finite and nonzero. No full-tree forward, real checkpoint load, optimizer update, scientific payload or GPU operation was performed. Reuse of the body method establishes source semantics, not actual full-tree runtime qualification.

**One bounded CPU cost probe.** A synthetic20,000-position×32-query scan took1.0390s forward and1.2177s backward with two CPU threads. Process peak RSS, including imports and the tiny checks, was1,473,544,192 bytes. Outputs and gradients were finite. This probe used ordinary autograd to expose its cost. The default model checkpoints query decodes, which reduces saved scan tables and adds recomputation. The cost receipt binds the pre-handoff source hash; the later changes added observation properties, a zero-head warm-start assertion and an exported existing loss, without changing the probed scan. The final tiny check was rerun against the final source, and the expensive cost probe was not repeated.

At686 queries there are22 query chunks. Multiplying the measured CPU block cost suggests roughly50s per arm for the scan alone, or about73s with one extra forward recomputation per block, if serial CPU scaling held. These are estimates, not measured full-model or GPU timings. GPU performance and complete-parent allocator peaks are unqualified. The balanced scan is O(P*Q*4^3), and source/evidence heads and unchanged GNN/BiGRU add work. This is plausibly implementable at the stated dimensions with query checkpointing, but the CPU probe does not certify a fit within root's limits.

**Inactive root integration recipe.** Do not invoke H046's frozen entrypoint with a different core. Root should create a separately named stage adapter using its existing observation reader, target barriers, cached-parent ownership, optimizer-step accounting, checkpoint manager and all96-seal evaluation path. Only these scientific hooks change:

```python
from arggnn.marker_grid_reset_chain_gnn_v1 import (
    prepare_marker_grid_reset_v1, make_matched_grid_reset_family_v1,
    original_grid_span_nll_v1,
)
# Observation preparation before TRAIN truth, once per admitted original parent:
prepared_cpu = prepare_marker_grid_reset_v1(original_H044_bundle, original_bin_edges_bp)
# Root supplies the authenticated original H042 pass30 payload and refs:
models, optimizers, proof = make_matched_grid_reset_family_v1(
    root_authenticated_payload, checkpoint_ref=root_H042_ref,
    completion_ref=root_H042_completion_ref)
# Within the existing root-owned admitted update boundary:
prepared = prepared_cpu.to(root_device)
model = models[arm].to(root_device)
log_marginal = model(prepared)
loss = original_grid_span_nll_v1(
    log_marginal, root_TRAIN_mass, prepared.markers.bin_widths_bp.to(root_device))
# Existing zero_grad/backward/clip1/AdamW/after-step accounting remains root-owned.
```

The complete proposed schedule is32 original TRAIN parents×30 passes=960 genuine full-parent FP32 updates per arm, independent reset AdamW lr.001/wd.0001/betas(.9,.999)/eps1e-8, seed101, six checkpoints5/10/15/20/25/30 and unchanged accuracy selection/support. Both complete arms'96 prediction products must be sealed before any VAL truth. No extra disposable qualification optimizer step is proposed: root's first actual scheduled parent update must count toward960 and qualify complete-model peaks/time. Do not reuse H046's admission, source pins, scope or checkpoint identity. There is currently no reviewed scientific CLI for this new model, so no real fit launch command is supplied or claimed executable.

For root's read-only source qualification after any further change, the existing project Python can run:

```sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$ARGGNN_PYTHON" research/reviews/internal_tree_transport_hypothesis_source_v1/tiny_fp32_and_cost_probe_v1.py --tiny-only
```

**Resources root still must freeze.** At the maximum original support, one full probability array is219.52MB and all96 RAM seals total21.074GB. Prepared five-channel CPU bin evidence is274.4MB per parent, or8.781GB for32 simultaneously cached TRAIN parents before original banks/markers. A31,933-weight FP32 model has127,732 parameter bytes, and two AdamW moments add255,464 bytes before serialization overhead. Saving12 weights-plus-moments checkpoints is about4.60MB of raw tensor data, excluding final checkpoints, logs, object overhead and any retained predictions. No new durable observed-feature cache is required. A32-query matrix scan table is40.96MB, with a163.84MB log-product broadcast; saved gradients/scratch multiply these sizes. Full-tree memory is additional. These are planning counts, not admissions or peak-memory bounds. Root must use actual counts and its current execution policy and must independently accept the complete-model first-step qualification.

**Prior and claim boundary.** H039 actually completed320 updates/arm and frozen evaluation, obtaining directed95.7119%, plain95.5635%, generic95.4964% on eight reused VAL parents/one seed. It already graph-conditions native-hold CRF transitions. RFMix already combines reference emissions and chain inference, and Neural Continuous-Time Markov Models already learns covariate-dependent rates. The reset generator, neural conditioning, PyG body and exact scan are inherited mechanisms. The only untested utility question is whether whole-tree-conditioned persistence on a SNP-resolved grid beats the identical graph signal used for own-state confidence in a fixed distance-only chain. Publication priority and efficacy remain unresolved. The generator is a conditional smoothing model, not an estimated biological recombination process, admixture clock or internal-ancestor identity.

Only this dedicated review folder and the uniquely named new source were written. Existing scientific/driver/guard/Git files were not edited. No real data, checkpoint, truth, GPU, admission, installation or Git operation occurred.
