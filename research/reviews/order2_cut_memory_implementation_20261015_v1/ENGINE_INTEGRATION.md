# Fresh three-arm engine and driver integration

This packet supplies callable model, preparation, matched-family and count hooks. Root creates and owns a fresh execution producer, context, driver, registration, sources, resource plan and runtime capability. Original H048 sources remain frozen. **fresh_engine_integration_map_v1.json** pins every inspected original source and lists the exact original lines to adapt.

## Minimal callable substitutions

Add the repository root and src to the caller's normal import path, then:

~~~python
from research.reviews.order2_cut_memory_implementation_20261015_v1 import integration_hooks_v1 as hooks
from arggnn import marker_grid_reset_chain_gnn_v1 as grid_core
ARMS = hooks.ARMS
PASSES = hooks.PASSES
READOUT = dict(hooks.READOUT)
~~~

Use the original H048 module as grid_core for its unchanged grid decoder/loss contracts. Use the new module for the model/factory. The new module does not expose every old H048 core symbol.

At a CPU observation cache miss, replace the original engine's single grid preparation call:

~~~python
limits = hooks.parent_bounds_v1(bank, parent["positions"])
# Existing root prebuild guard checks the full new limits.
ctx["before_grid_build"](pid, bank, limits)
value, cut_metadata = hooks.prepare_parent_v1(original_h044_bundle, bin_edges_bp)
extra = hooks.extra_bytes_v1(value)
require(extra <= limits["extra_retained_tensor_upper_bytes"], "Full grid/flag/cut tensor bound")
# Store cut_metadata and identities of all value.tensors() in the immutable recipe.
# Existing LRU size is extra + value.markers.tensor_bytes(); evict both together.
~~~

The wrapper's validate, tensors, features_cpu, markers, bank, source_bank and to implement the original cache/device entrypoints. Grid marker features remain CPU; cut tables transfer once with the native parent. One prepared parent is shared by all three sequential arm steps. The root preparation_check_v1 must call the same new bound function, or its exact-declared-bound comparison will reject it.

Replace the original family helper:

~~~python
models, optimizers, proof, hashes = hooks.build_family_for_engine_v1(
    authenticated_h042_payload, device, guard,
    checkpoint_ref=WARM_CHECKPOINT, completion_ref=WARM_COMPLETION)
~~~

This verifies all three initial state hashes, all three pairwise learned-storage separations, all 32,446 trainable FP32 scalars, and fresh independent AdamW state. The caller's reader authenticates actual checkpoint/completion bytes; this packet reads no real checkpoint. Evaluation constructors are:

~~~python
model = hooks.evaluation_model_v1(arm, device)
model.load_state_dict(authenticated_new_state, strict=True)
~~~

Keep the original H046 generic complete-parent update and original exact-span loss. Its first scheduled-step full-body check must use name.startswith("common.backbone."). At initialization O=0: U and ell may have connected zero gradients on the first scheduled update; finite connected gradients are the appropriate qualification. No additional optimizer update is needed.

## All cardinality changes

The new protocol has 3 arms, 6 checkpoint passes, 18 actual checkpoint files, 8 original VAL parents and 144 complete original-grid validation products. Each arm receives 30 passes x32 complete-parent updates=960; total 2,880. Six checkpoint groups stay six.

Original producer **src/arggnn/train_marker_grid_reset_chain_v1.py**:

- Line17 becomes hooks.ARMS. Lines45-73 need the fresh registered protocol/schema/id,3-arm execution mode and32446 total/31933 common/513 memory parameters.
- Lines151-168 need the new factory and all 3 pairwise independence checks. Parameter-count literals at 159,161,206,267,401,518 become32446.
- Lines296-326 need a fresh phase reader/schema/id and the144-key allowlist. Lines329,342,353,367 update2-arm/96-product labels. Lines347-361 use fresh packet provenance, all144_original_grid_VAL_RAM_sealed_before_truth.json, a fresh all144 schema, complete_dense_prediction_products=144 and all144_sealed_before_any_VAL_truth=True.
- Line395 requires all 3 arms per group. ARMS-derived training, checkpoint and scoring loops include all3 after setting the fresh global.
- Lines419-440 contain a hard2-arm unpack at 424 and H048's transition-vs-confidence interpretation. Replace with descriptive true-minus-null and true-minus-leaf parent effects (hooks.paired_effects_v1) and separately registered decision criteria. The helper emits no threshold decision.
- Lines445-453 use the fresh reader/namespace and new model constructor. Line463 uses 144. Line466 must verify all 144 immutable grids before enabling any VAL truth. Line485 becomes all144_RAM_seals_before_any_VAL_truth; line520 becomes all144_products_before_VAL_truth.
- Lines492-553 need fresh capability/schema/id/provenance and joint_three_arm. All H048 runtime schema/status/stage names need a fresh condition namespace, including train/checkpoint/completion/failure/RAM names.

Original root sources **research/reviews/H048_GPU45_grid_root_source_v1/**:

- root_common_v1.py: worker imports57,105,131,150 point at the fresh worker. Line79's12 products/parent multiplier becomes18. Line88's12 checkpoint archive units becomes18 with a reviewed actual per-file allowance. Line93's all96_original_grid_probability_bytes becomes all144_original_grid_probability_bytes. Lines74-87 and104-127 include cut payload/preparation/cache bounds and larger recipe planning. Lines134-176 validate fresh three-arm TRAIN lineage/paths; line 170 means all 3.
- context_factory_v1.py: new root-common and worker imports, closure, schema and capability at4,11-28,63,66-71. Lines67,140,145 use joint_three_arm; line135 reads the all144 count-plan field; line142 becomes all144_evaluation_seals_before_truth.
- root_driver.py: fresh namespace/HERE/worker/CLI_MODULE/CLI/MAIN/output/job/key/schema at7-19,23,26,35,67,75,96,108-120. Line44 reads the all144 forecast. Lines43-50 use fresh cut-inclusive forecasts/caps. Lines54-57 bind fresh TRAIN lineage. Line108 uses joint_three_arm; line193 becomes three_arms_one_GPU=True. State 18 checkpoint files, 144 products and 2,880 total updates in the admission description at125.

The machine map contains original exact source hashes and every inspected literal/label occurrence. Numeric2,12 and96 have other uses: 2*n flag storage and96*sites preparation arithmetic remain. Source-class count 4, 686 queries, 952 leaves, 6 checkpoint groups, 8 VAL parents, 32 TRAIN parents and 30 passes stay unchanged. “Paired sample differences” is appropriate for each parent-matched contrast.

## Resources and artifact lineage

hooks.bounds_from_counts_v1(n,t,s,p) uses already authenticated count metadata without scientific arrays or Torch imports. It adds the complete cut-plan raw table upper89*t*952+952 bytes, preserves full grid/flag payloads, and lists rank 8 history/forcing and compact32-channel injection banks separately. It does not qualify total RSS, CUDA peak, allocator, autograd or throughput. Cut preparation's128*n+512*t*952+512*952+64MiB is explicitly an estimate. Its planning allowance must not be relabelled a proven RSS bound.

Complete raw validation RAM is18*sum(P_parent)*686*4*4 bytes, or 31,610,880,000 bytes if all 8 parents have 20,000 bins. This excludes banks, markers, Python, history, device and allocator overhead. hooks.raw_validation_probability_bytes_v1 provides the count from all 8 metadata counts.

Old H048's1MiB per-checkpoint unit,100MB output cap, fixed proof/recipe allowance, cache/preparation/residency caps and RAM cap require fresh review. New cut metadata lists up toT hold counts, episode starts and conditional-information values; the old 131072-byte per-parent recipe allowance needs a derived bound. The hooks expose additional 65536+256*T bytes per-parent planning allowance. Root binds the final output forecast and exact source closure.

Fresh checkpoints should bind the memory arm, rank 8, cut 350, ell limits, unchanged common graph_to_transition decoder, full readout config (including checkpoint_memory), exact preparation/cut recipe and all new/inherited source refs. The common H042 pass30 warm start remains the same authenticated original, followed by identical freshly seeded heads/memory and reset AdamW. No future H048-trained checkpoint is required.

## Boundaries of this handoff

The single synthetic CPU fixture verifies full-model PyG/GRU/decoder forward/backward math and two discriminating memory witnesses. Root owns actual observation bundle/device-transfer qualification, actual warm-start payload authentication, full 20,000-bin runtime/memory qualification, registration/criteria/resources and full 30-pass fits. This source packet supplies no execution producer, active registration, admission, lease, launcher or target reader. Registered outer/TEST support stays unopened and unchanged.
