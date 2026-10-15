# C3 optional bounded observation prefetch provider

Source-only packet, 2026-10-15. No live source, source pins, model, dataset, guard, admission, accounting or Git state was changed. This provider is inactive. Root owns any integration, resource authority and actual execution.

## What changed

prefetch_provider_v1.py supplies PrefetchSealedObservationProviderV1, a subclass of the existing SealedObservationProviderV1, and ObservationProviderFactoryV1. It uses the **same AmericanC3WholeTreeDatasetV1 and original _read validator in each CPU worker**. A worker processes one requested physical block at a time; the parent has at most one in-flight request per worker. The optional factory accepts the existing constructor arguments: plan_ref, parent_id, worker_tmp_directory and cache_bytes.

The new provider keeps the original get(index), seal(resource_check=...) and release() API. It preserves:
- Every original whole tree, all 1,600 samples, every prescribed focal, exact physical block IDs and the original block plan.
- The inherited complete-parent observation seal and its exact-once 10,000-focal coverage check.
- The trainer's global all-20 observation seal barrier before any private TRAIN resolver is reachable.
- The original complete-parent checkpointed step without changing its source or call arguments: forward/reverse carried states, CPU autograd edges, checkpoint recomputation, all-focal mean CE, complete-parent backward, one clipped optimizer update and final release.
- Original source/observation checks and canonical feature hashes. Every cache hit still rehashes every feature byte, including cache mutation detection.

The parent canonical hash uses a contiguous NumPy buffer view instead of array.tobytes(), avoiding that full-array Python bytes copy. Field order, dtype/shape encoding, feature bytes and metadata encoding remain identical. Workers retain the original hash/check implementation. This is a narrow copy-avoidance change, not a skipped validation.

An explicit cache requirement, cache_bytes >= max_block_bytes >= the frozen plan's feature planning bound, ensures that a row-only get(index) and the immediate checkpoint get(index) retain the same materialized CPU block. CPU prefetch proceeds while the caller consumes the previous block. It never yields a completed block just because that worker finished first. Unexpected/reverse/repeated checkpoint demands return the requested physical index; speculative requests may be discarded after completing their original checks.

No throughput improvement has been measured or claimed. Worker source loading, duplicated tree state, IPC and hashes may cost more than the overlap saves. Actual Torch/PyG shared-memory transport and a full training step remain unqualified.

## Bounded worker lifecycle and memory

bounded_workers_v1.py uses multiprocessing spawn with explicit duplex pipes. There is no feeder queue, parent result thread, task retry or unbounded future submission. Worker count must be 1..4, normally proposed as 2; actual count is min(num_workers, block_count). Each worker holds at most one task/result. Workers call Torch with one CPU thread, check that no CUDA context exists, hide CUDA before observation reading, use an explicit repository-local temporary child. A fresh pre-Torch abstract IPC bootstrap prevents filesystem resource-sharer socket nodes.

Workers start lazily during seal/read. The inherited seal releases them before returning; consequently all 20 seals complete without retaining 20 worker pools. During training only the current parent's provider starts workers. The current trainer invokes parents serially. Spawn receives only plan/parent/temporary-directory metadata, byte caps and the observation seal; private truth resolvers, target vectors, model and optimizer are never passed to workers.

Root must provide:
- num_workers.
- max_block_bytes, covering the maximum feature planning bound of every supplied frozen plan.
- cache_bytes through the existing scientific config, large enough for one whole block.
- worker_rss_cap_bytes.
- aggregate_rss_cap_bytes, covering the training parent and all exact owned CPU workers.
- A real root resource callback and explicit task/drain deadlines.

The constructor conservatively requires:
cache_bytes + (num_workers + 2) * max_block_bytes + num_workers * worker_rss_cap_bytes <= aggregate_rss_cap_bytes.

This is a planning requirement, not a claim that CPU native allocations cannot transiently exceed an RSS limit. Actual block tensor sizes and worker peak RSS are checked after building; the parent polls its peak RSS plus each live owned PID/start identity and worker peak/current RSS at <=0.1-second transport waits. Each worker has a separate RSS cap. Native transient allocations and Torch shared-memory behavior must be qualified in the actual authorized route. Complete-parent CPU autograd state still grows with the original block/focal count.

Normal release consumes/discards bounded outstanding replies, requires stop acknowledgements, joins all workers and requires zero exits. Failure cancels the pool without retry, closes channels and terminates/kills only its exact owned child Process objects if needed. All children are joined; the drain receipt records actual exit codes, forced PIDs and discarded speculative counts. A hidden prefetched failure aborts normal release. It cannot silently fall back to the original provider.

The worker drain receipt concerns these explicit worker children. It does not manufacture an outer launcher wait, storage close, historical process absence or complete Torch helper lifecycle certificate. Root's existing process-tree supervision remains required.

## Integration is explicit and unapplied

provider_factory.integration.unapplied.patch is a narrow two-file patch. Review copies are trainer.provider_factory.staged.py and execution_hooks.provider_factory.staged.py. They are not launchers.

The patch adds observation_provider_factory=None to run_train_v1 and run_train_stage_v1. The default constructs the original provider exactly as before. A supplied factory is explicitly forwarded, its actual source refs/bounds enter the protocol signature, and the optional provider's diagnostics are recorded after each complete parent step. Existing resumes with a different provider signature do not become silently compatible. The target joiner and global seal-before-truth order remain unchanged.

No patch touches checkpointed_complete_parent_step_v1 or model/target equations. Source validation confirms the complete-parent step call AST is identical.

The fresh abstract_ipc_bootstrap_v1.py adapts only the reviewed v12 IPC mechanism (source SHA 4550ead44f36eeb0fda8a817a2002082c3d067cf1a84fd817301532577d8b014). It uses the independent c3-gnn-prefetch-v1 namespace. It does not import the amendment, activate old authority, edit installed files, remove sockets or change any storage guard.

Root must install it in the future job interpreter BEFORE its first Torch/PyG import. root_entry.pre_torch_bootstrap.unapplied.patch demonstrates that exact insertion before line 33 of the unchanged current980 worker entry; its staged copy is review-only. The existing route and source pins remain untouched. A separate successor must pin this entry change too. If Torch was imported before an uninstalled bootstrap, provider import fails instead of installing it late.

Each spawned CPU process imports the same fresh bootstrap before importing Torch/PyG while unpickling the opener. It also hides CUDA before those imports. The bootstrap requires the exact reviewed stdlib connection source, preserves authkeys and Listener/Client authentication, delegates other address families, and accepts only bounded leading-NUL abstract AF_UNIX addresses. It rejects an existing resource-sharer listener or another wrapper. Explicit spawn is a new operational adaptation; v12's default-fork qualification is not inherited.

Before Torch/PyG in the root future entry:

```python
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.abstract_ipc_bootstrap_v1 import ensure_abstract_ipc_before_torch_v1
prefetch_ipc_bootstrap = ensure_abstract_ipc_before_torch_v1()
```

Canonical factory import after root authorizes scientific imports and adds the repository plus src to Python's import path:

```python
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.prefetch_provider_v1 import ObservationProviderFactoryV1
factory = ObservationProviderFactoryV1(
    **root_caps["observation_prefetch"], resource_check=runtime_guard)
```

Root then passes observation_provider_factory=factory through the patched run_train_stage_v1 call. CPU cache bytes still come from cfg["CPU_observation_cache_bytes"]. No monkeypatch, import hook, patched Dataset, changed graph grouping or installed-file edit is involved.

The current980 route remains **zero-worker and unchanged**. Coordination with h048_novelty_crossdomain_fresh_v1 confirmed:
- trainer SHA cce52c9f1776e46c7005509544bfb3743acaf860f256e546e3ac5fc7cb518df2.
- execution hook SHA 6546cc4498a8bcbac4919ae08ce75b23a0ac8e5b350da25e3fa31a967286bfa1.
- Current fast guard checks main-process RSS and explicitly assumes workers 0.

Enabling this provider therefore requires a separately reviewed successor: apply/repin trainer and hooks, add the provider/core source closure, bind explicit worker/RSS/transport/temp caps, forward the factory and retain root process-tree drain. Nothing in this packet changes current activation or authorizes that successor.

## Diagnostics and future timing

diagnostics() reports cache hits/misses, delivered worker blocks, demand wait, parent transport/cache hashing time, delivered worker read/hash time, configured bounds and the latest actual worker drain. Worker times are cumulative durations for delivered blocks, exclude discarded prefetch and can overlap across workers; they are not wall-clock speedup. No timing has been collected from real training.

After a root-authorized integration, compare actual total elapsed complete-parent time with matched source/config/data/device conditions and inspect aggregate RSS, worker drains and the complete-parent result. No scientific shortcut or extra mock benchmark is needed to make a speed claim.

## One handmade check

```sh
.runtime/env/bin/python -B research/reviews/C3_GNN_prefetch_provider_20261015_v1/handmade_coverage_v1.py
```

The single tiny fixture uses four blocks containing six handmade trees and six focal rows, including an empty-focal block. Two real spawned CPU transport workers return exact forward/reverse and repeated/nonmonotone physical demands. It checks bounded in-flight work, exact-once tree/focal coverage per directional scan, demanded-index retention even when all speculative cache candidates are skipped, graceful zero-exit drain and injected-failure drain. Each handmade spawned reader also installs the bootstrap and binds/closes an actual abstract AF_UNIX listener. No filesystem socket is used. It imports no Torch/PyG, dataset, model, truth or guard.

Within the same check, the original and new pure hash function ASTs are executed with tiny CPU array stubs, including scalar, empty and noncontiguous arrays. They produce identical canonical hashes/byte sizes, and a feature mutation changes the digest. This does not establish real PyG tensor resource-sharing transport or dense-gradient equivalence.

The actual handmade receipt and source validation are saved beside the source. AST/compile and original hash pins pass. The original dataset, stream trainer, scientific trainer and execution hook remain unchanged.
