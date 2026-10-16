# H050 memory learning-path source audit

2026-10-16. Source and existing JSON receipts only. No model imports, data/checkpoint payload reads, numerical runs, new tests or source changes.

**Finding: no obvious source-level learning blocker.** The 513 added scalars have a viable loss-gradient path. This does not establish that every scalar learned useful structure or that memory improves ancestry accuracy.

## Exact model and parameter coverage

The actual H050 TRAIN configuration, research/reviews/GPU0_shared_admission_draft_20261016_v1/bound_metadata_v1/H050_config.json, binds direct_readout_cut_memory_v1.py with SHA256 ed7b9b889ca703a76c48f104e3f695b29bab005da9c53c3090aa29ea0b9d4ec3. The CPU evaluation metadata points to that same TRAIN config.

CutMemoryV1 has U.weight of shape8x32, O.weight of shape32x8 and scalar ell_theta, totaling256+256+1=513. Only O is explicitly zero initialized. U retains nn.Linear initialization. The positive length scale is ell=100+99900*sigmoid(theta), initialized to5000bp with nonzero sigmoid derivative. Source: research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py:351-366.

family_from_base_state_v1 copies the full H045 body, accepts exactly the three absent memory keys, then checks every model parameter requires gradients. The three memory arms have independent parameter storage. integration_hooks_v1.py:35-43 transfers the models before constructing an independent AdamW over model.parameters() for each arm. No memory parameter is omitted or frozen.

## Zero output initialization is a delayed opening, not a dead double gate

Writing x for the representative first-sweep features, F=U*x and R=history(F,ell)-F, the injected residual is delta=O*R.

At O=0, the loss derivatives of U and ell through this branch are zero on the first update. The derivative with respect to O need not be zero when R and the downstream loss signal are nonzero. Once O moves, U and ell can receive loss gradients. There is no second zero factor or hard-zero learned gate trapping both sides.

The actual existing scheduled-step receipt, experiments/results/H050_GPU0_direct_readout_train_seed101_v1/first_real_parent_step_qualification.json, records finite FP32 gradients, full AdamW state coverage and changed memory.O.weight in all three memory arms after step1. Because O starts at zero, AdamW weight decay alone cannot explain that change. It supplies evidence that the residual output gate opened in the actual training path.

The same receipt records U and ell changes but does not record their gradient norms. Their first-step changes can arise from weight decay. They must not be described as evidence of loss-driven U/ell learning.

The older handmade_FP32_check_receipt_v2.json reports positive U/O/ell gradient L1 values on its synthetic fixture. Its source manually opens O before that backward. It supports conditional gradient connectivity, not automatic gate opening, actual full-data parameter movement or predictive utility.

## Loss path and gradient boundaries

- direct_readout_cut_memory_v1.py:139-147 computes learned first-sweep upward/outside states, calls memory, index-adds delta at cut representatives, then executes a fresh shared-weight whole-tree sweep and leaf projection.
- order2_cut_memory_v1.py:382-434 uses differentiable exp/expm1 decay, PyG sum/mean aggregation, indexing and non-reentrant checkpointing. The source does not detach force, history, length scale or output. Fixed topology/date/index preparation uses detached observed inputs intentionally.
- anonymous_clade_gnn_v1.py:110-131 passes the injected base through learned nonlinear PyG upward and recipient-excluded downward messages to leaf context. Its upward-plus-outside forcing is not an algebraic subtraction cancelling the injection.
- anonymous_clade_flag_control_v1.py:290-304 feeds context directly and through the complete BiGRU to the classifier. OrderedCladeEpisodeGNNV1 then selects supervised queries.
- train_ordered_clade_episode_v1.py:290-310 sets training mode, computes the ordinary exact-span loss with autocast disabled, calls backward, clips and steps. no_grad in integration_hooks_v1.py:58-63 belongs only to prediction. Diagnostic detach operations capture snapshots/logits and do not replace the tensor used by the loss.

## Limits

Single-hold inputs or temporally constant forcing can make history-minus-local exactly zero. Symmetries or saturated downstream activations can also yield zero gradients for particular parameters or examples. These are conditional properties, not a source proof of a universally dead branch. No all-scalars-nonzero-every-step requirement is appropriate.

This audit does not inspect trained memory values, claim useful learning by all513 scalars, or reinterpret a small performance difference as an implementation bug. Predictive utility remains the result of the registered paired evaluation.
