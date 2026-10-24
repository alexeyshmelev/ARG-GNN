# Isolated exact executor integration

The approved successor integration is implemented and source-frozen. The healthy campaign, original runtime sources, inputs, model equations, optimizer settings and checkpoint files are unchanged. No GPU job was launched or Git commit made.

## Explicit dispatch and profiler API

Use integration.executors.build_executor(kind, model, bundle, metadata, execution="staged", layout_cache=cache, frozen_module=..., **original_constructor_kwargs), where kind is "original" or "integrated". The staged route returns the current StagedExecutionV1 or its isolated subclass. The bounded route similarly supports current prediction execution. Existing constructor type, CPU-observation and admitted-CUDA guards remain enforced.

The profiler's separately reviewed diagnostic slice may use the original staged constructor's frozen_module injection, binding the original model class and explicit diagnostic bundle class. Full-parent work uses the untouched frozen namespace. A slice is not claimed to have full-parent boundary/history equivalence. Give distinct diagnostic scopes an explicit metadata["layout_cache_scope"] identity.

SelectedFreshTreeLoom in runtime.py is an explicit successor subclass, constructed in a new authorized interpreter with executor_kind and an optional explicitly budgeted cache. Its train_step override preserves the registry/bundle/target checks and dispatches to integration.step.train_step. This matters because the current FreshTreeLoom.train_step directly imports step_v2; replacing runtime.execution.train_step_v1 alone would not redirect training. Prediction is routed through the same selected factory.

## Changed call sites

1. Executor construction selects the original constructor or integrated constructor. The latter obtains immutable LayoutPlan arrays from LayoutCache instead of rebuilding height/depth/block recipes.
2. Integrated _sites is the same global256-site chunk/selection/pooling/projection loop with m._site_chunk replaced by categorical_site_chunk(m,...).
3. Integrated _block explicitly copies immutable int32 local indices to torch.long on the admitted device; the other observation transfers are unchanged.
4. The isolated training step selects the factory and records executor/cache metadata. Its full-parent decoder backward, context.backward(context_gradient), all49-gradient validation, clipping and original AdamW step remain unchanged.
5. Explicit runtime dispatch and finally blocks release executor layout leases only after all decoder/trunk backward and optimizer work, or after prediction forward. An aborted step releases only because its graph will not be resumed.

The inherited first sweep, forcing, finite own-state proof, second sweep, context_forward, stable_leaf_history_v2, full genomic decoder and non-reentrant checkpoint helper are unchanged. Checkpoint closures retain the executor/model and resolve the integrated _sites method during recomputation. Ephemeral categorical MLP outputs are evaluated again using current parameters; no learned numeric state is cached.

## Layout cache ownership and memory

Construct LayoutCache(max_bytes=declared_additional_host_reserve, admit=optional_headroom_callback). This budget is additional host memory and must be included in the successor's declared reserve; it does not raise or reuse an observation-cache cap silently. The callback receives (working_bytes, stage) before cold build, restore-binding or lease growth. Existing process/RSS admission guards remain required.

The persistent cache owns only derived immutable index arrays, provenance keys, scalar tensor stamps and weak bank references. It does not own an observation bank, its tensors, model state or learned activations. Therefore the plan can survive deletion of an expanded CPU bank. A restored bank receives one content-digest bind; repeated access to the same immutable tensor storage checks pointer/version/shape stamps without hashing topology again. The original immutable alias discipline is required.

Retained charging covers immutable NumPy backing/header objects, plan/key/binding metadata and active executor tuple/control views. Active executor leases pin their entries: eviction and clear are rejected while a lease is in use, and a new allocation fails if idle eviction cannot fit the finite cap. Thus an evicted in-flight plan cannot disappear from accounting. Direct factory callers must invoke executor.release_layout() only after all backward/optimizer work and must not retain escaped layout views afterward. Runtime step/prediction adapters perform release in finally. One controlling execution thread is the intended cache owner.

Cold-build admission includes the eventual plan, full height/depth/assigned arrays and a generous128 bytes per largest-block node for simultaneous int64 recipes, sort scratch and conversion. This is an operational reservation, not a mathematical NumPy allocator peak proof. Caller headroom and measured process peaks remain necessary.

## Small CPU wiring evidence

cpu_smoke.json records one bounded smoke, with production CUDA guards intact. It tests cache build, fast hit, weak bank collection, restored-content bind, tensor-version rejection, active lease eviction/clear rejection, idle LRU eviction, cap enforcement and admission callback use.

The real original non-reentrant _cp helper invoked original and integrated _sites on a CPU diagnostic context, covering four real trees and the original256-site source chunk at1,600 haplotypes. Each closure ran twice, proving backward recomputation. Outputs matched exactly; six connected site/projection parameter gradients differed by at most5.59e-8 under1e-4 tolerance. The model still has49parameter tensors.

This is a wiring smoke, not a GPU or full-history/all49-gradient parity result. The already sealed categorical/layout component evidence remains applicable. Full forward/loss/decoder/trunk-backward/AdamW comparison, GPU/host peaks, cold/warm timing and campaign integration remain with the separately authorized profiler/runner work. Source hashes are sealed in artifact_manifest.json. Historical prototype metadata flags are preserved as historical evidence; the later user approval authorizes this isolated successor integration.
