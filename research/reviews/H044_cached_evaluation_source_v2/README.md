# H044 cached evaluation successor, source-only

Implemented only `src/arggnn/marker_resolution_input_cache_v2.py` and files in this owned folder. No frozen/live scientific core, trainer, evaluator, root context or driver was edited. This package has not imported Torch/project modules or read observations, arrays, targets or checkpoints. No guard, process/accounting inspection, admission, GPU work, install or Git operation was performed.

## What is concrete

`ImmutableMarkerBlocksV2` computes and retains exactly the original eleven contiguous query blocks (64 queries per block, final block46), separately for localized and expanded tree-pooled features. It calls the frozen `features_for_queries` and `pool_within_trees` without changing formulas, accumulation precision, casts, pooling, shape, stride or query order. Cache tensors are ordinary CPU FP32 tensors built under `no_grad`, with inference-mode tensors rejected. No learned state, model output, loss or target enters this module.

A frozen marker-bank subclass view returns those exact block objects from the two feature methods. It borrows every original observation field and is wrapped in the original typed H044 bundle. The original `MarkerResolutionReadoutV1.forward`, body, classifier, decoder, center blocking, state-dict layout and device transfers execute unchanged. Original bundle stamps cover all borrowed observations. Separate cache stamps cover every cached block and check the original binding digest. Normal Torch in-place mutations/replacements are rejected. NumPy/.data alias writes remain prohibited by the original contract.

`evaluate_cached_v2.py` copies the original evaluation control flow, with a small explicit diff in `evaluation_flow.diff`: source aliases, one cache build/bundle substitution per parent, release after that parent's12 arm/checkpoint forwards, and completion provenance. It calls original `DenseRAMMatrixV1.seal/verify`, original target reader and score/selection helpers. Parent order, arm order, checkpoint order, strict checkpoint provenance, eight VAL parents,96 products, probability hashes and the all96-before-VAL-truth barrier remain. No TRAIN path or optimizer exists in the producer. Original direct TRAIN wait0/drain and complete30pass/960updates are required before evaluation. Outer bindings are rejected.

The producer and cache are inert on import. Default producer invocation prints inactive metadata and does not load Torch. The inactive execution template cannot authorize work.

## Memory and scope

The hard maximum is eight VAL parents and21,073,920,000 retained feature bytes. A complete parent pair at P20,000/Q686/C4/F6 is2,634,240,000bytes. The provided producer releases each pair after all12 products for that parent, so only one pair is retained at a time, while all96 sealed probabilities retain the original21,073,920,000byte worst case. Feature construction also needs the original query-block prefix/window workspace (existing named worst-S allowance21.135GB) and marker/bank storage. CPU allocator/native peaks and GPU fit remain unqualified here. The existing96-channel H042 memory erratum remains applicable. No full-query repacking, target mass caching, hold aggregation, feature compression or GPU-resident cache is introduced.

For eight VAL parents, the source call counts become88 raw-block and88 pooled-block constructions, reused across the same96 predictions. Original evaluation would construct1056 raw blocks and528 pooled blocks. These counts are not a timing or speedup result. Bitwise output parity has not been executed or certified. Possible CPU reduction nondeterminism and alias discipline remain explicit limits.

## Existing closure blocks an old-input launch

`H044_marker_resolution_root_v1/context_factory_v2.py` hardcodes WORKER_SOURCE to the frozen trainer and directly binds the actual context, driver and worker in its consumer closure/reservation. Its build function also checks its own actual source hash. The frozen driver chooses that hardcoded worker. Therefore this package does not try to wrap, monkeypatch or relabel that context or reuse an old admitted input.

Root must create a fresh immutable evaluation-only context/driver/consumer successor. It can retain the original guard/launcher implementations and scientific config validation, but must reserve and validate the actual new producer, cache, context and driver sources, plus the full unchanged scientific/source closure and the existing repair/policy/authority/launcher pins. Keep the frozen scientific config/checkpoint references auditable. Put the new cache policy/source closure in separately bound successor metadata rather than changing old checkpoint provenance or pretending old source manifests authorize new code.

The new context must expose `build_context_v2(record,spec)` and return the original admitted evaluation context fields, plus `cached_execution_ref`, `cache_policy`, `cache_policy_ref`, and the original `train_actual_wait_ref`. It must validate the new execution-input schema `H044_cached_evaluation_root_input_v2`, actual worker/context/driver refs and consumer closure, current resource authority, active lease/title/device, exact original successful preparation, original complete TRAIN terminal evidence, eight VAL-only bindings and unchanged data/source identities before returning `admitted=True`. The worker independently checks original wait0/drain and complete960-update metadata. The original contexts are explicitly rejected by the new entry point.

No new root factory/driver has been fabricated here: their accounting/admission implementation remains root-owned. `execution_input_inactive.json` deliberately has null authority/context/driver/consumer/terminal refs. `source_refs.json` pins the implemented sources. `cache_policy_inactive.json` is policy content, not authorization.

## Inactive commands for root review

Source checks only, with no project-module import or model/data execution:

    PYTHONDONTWRITEBYTECODE=1 python3 research/reviews/H044_cached_evaluation_source_v2/source_checks_v2.py

Inert producer inspection:

    PYTHONDONTWRITEBYTECODE=1 python3 research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py

After root supplies the new exact consumer closure and admission, only the existing approved GPU launcher should invoke the new producer with its fresh input and SHA:

    PYTHONDONTWRITEBYTECODE=1 python3 research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py --execute --input ROOT_ISSUED_NEW_INPUT --expected-input-sha256 ROOT_ISSUED_SHA

The final line is a placeholder command shape, not a runnable or admitted command. None of these commands invokes a new fit. Original evaluation is still pending and should not be duplicated.

## Source checks and remaining checks

The static checker parses/compiles the two new modules without importing them, verifies their pinned frozen/cache hashes and stdlib-only top-level imports, checks names used in the copied evaluator, and compares its complete source body against the original with only the declared insertions allowed. This comparison preserves the original sealing-before-truth and scoring/selection code literally. It also checks that the execution template is inactive. These checks are source checks, not model/data/runtime tests. Root owns separate typed-interface/parity checks, review, closure, admission and execution. No accuracy or biological claim is made.
