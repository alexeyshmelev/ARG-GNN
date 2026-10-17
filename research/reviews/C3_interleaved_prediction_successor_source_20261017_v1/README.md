# Future C3 interleaved prediction adapter

Status: **source only, inactive, unexecuted**. This package implements only audit
action 1 for potential future original VAL000–002 successors. It does not
authorize replacing or interrupting any job. No model, data, array, checkpoint or
truth payload was read during development. No numerical equivalence, inference,
resource observation, admission, install, job, lifecycle or Git operation was run.

`prediction_adapter_v1.py` copies the effective checkpoint-schema bridge used
by VAL003/004. Its only edits are the module documentation, import of the sibling
scheduler, restriction to future `predict_val_000/001/002`, and a check that
the caller's five-parent contract retains the exact original order.
`six_candidate_scan_v1.py` is byte-identical to the existing scheduler
SHA256 `4da2db9c217b7e4cfe25f0b9c9cf6caeaf228b36e8ced978d7d7299af3607255`.
`source_map.v1.json` records exact source identities, call-route line ranges,
and the sealed audit's historical plan identities.

The adapter constructs six separate original models, strictly loads the six
original saved epochs 5/10/15/20/25/30, and applies the original nested
`scheduled_checkpoint_metadata_v1` validator. It preserves both original
complete traversals, original model/encoder/GRU/classifier bodies, all1600 graph
and recurrent sample columns, all1000 queries, whole30Mb parents, original
10000 focal records and their identity/order/class fields, every tree/site,
global-neighbor geometry, and state reset only at true parent endpoints. Raw
observed banks alone are shared. Each weight retains its own learned encoding,
directional state and focal forward history. Query-only recurrence, layout
reuse, broader kernels, caches, controllers and new model mathematics are absent.

## Scope of the expected saving

There is one dataset/loader per direction instead of a new pair per checkpoint.
Thus each parent has two whole-parent raw-feature traversals and two observation
loads, one per direction, rather than twelve. In each traversal, every planned
block is still constructed once. From the sealed audit's complete-route counts:

| Parent | Blocks | Serial raw block constructions | Future raw block constructions | Learned encoder calls, unchanged |
|---|---:|---:|---:|---:|
| VAL000 |2392|28704|4784|28704|
| VAL001 |2390|28680|4780|28680|
| VAL002 |2443|29316|4886|29316|
| Total |7225|86700|14450|86700|

VAL003/004 already use this scheduling. Their source/runtime need no change.
All-five raw counts would be96294 to24044 if only the serial routes were
separately elected for replacement. All-five learned encoder calls stay144264.
These are static successful-route counts, not observed progress or speedup.
Only raw construction/loading overhead is amortized. Learned tree/site pooling,
both GRU directions, classifier work and guarded writes remain. Wall-clock
benefit depends on their actual shares and contention and has not been measured.
Six models and six focal histories coexist, unlike the serial route: the
scheduler reserves7.68MB of FP32 focal forward state versus1.28MB at a time in the
serial route, excluding model parameters, probabilities and other state. Existing
VAL003/004 behavior supports source reuse, not a newly measured memory/time fit.

## Exact integration requirements owned by ROOT

This is a callable package, not an executable worker, builder or admission API.
There is deliberately no usable launch command in this package. Old admitted
inputs pin their existing worker/source/driver/plan closure and cannot be
silently redirected here.

1. Decide whether a successor is needed from genuine current product and
   lifecycle evidence. Source efficiency alone is not a reason to stop current
   jobs or rerun completed scientific cells. This package performs no such
   evidence collection or decision.
2. Create a fresh, separately reviewed worker/source closure and fresh key,
   preparation/input/packet/output namespaces. Preserve old inputs, source
   manifests, scopes, guards, caps, charges, deadlines, failures and products.
   Bind both new Python files plus the exact original scientific dependencies
   and exact chosen original plan/checkpoint/training-acceptance provenance.
   The source map is a review aid, not a new admission manifest or authority.
3. In that new guarded CPU worker, replace only the prediction dispatch with:

   ```python
   from research.reviews.C3_interleaved_prediction_successor_source_20261017_v1.prediction_adapter_v1 import predict_v1 as future_predict
   return future_predict(req, training_bundle, training, writer, fast, contract=c)
   ```

   Import this scientific module only after original input/storage/runtime/
   lifetime/CPU guards pass, as the existing guarded worker does. Supply the
   original five-parent contract order, bound `contract.read`, same-parent
   genuine closed-plan evidence, accepted immutable saved600 family, guarded
   writer with private charged TMP, and original callable `fast` guard.
   Keep32 CPU threads, zero CUDA, zero loader workers/no pinning,128GiB RSS,
   86400s scientific window,32MiB complete scope and original64-role ceiling
   unless separately authorized otherwise. These are inherited intended caps,
   not newly measured fits or granted resource authority.
4. Retain the existing genuine direct-wait, owned-drain, consumed outer wait,
   completion and ordinary-close requirements. Fresh admissions and ordinary
   closes must use ROOT's genuinely current verified child selector, not any
   historical source-map policy. No selector, admission or close is performed
   here. A separate review must join actual worker, builder, driver, source
   closure, concrete current accounting selector and fresh ROOT directive.
5. Launch only the exact one-shot argv emitted by a genuine fresh preparation0,
   after ROOT's actual review, authority, absence/capacity checks as applicable,
   and admission. Do not use an old admitted argv, replay/resume an old key,
   migrate a live process, reset an old clock or manufacture a successful wait.
   This implementation gives no automatic retry or execution permission.
6. Keep the unchanged all-five genuine closed-plan gate and original all30
   (five parents x six epochs) pretruth prediction-seal barrier. The unchanged
   matrix/evaluator/selection code receives ordinary prediction/seal fields.
   No VAL truth/resolver, scorer or TEST stage is wired in this package. TEST
   stays deferred until complete original-five-VAL selection.

## Completed products and any later election

All completed original products remain at their original paths and hashes with
their actual producer, direct/outer waits, drain evidence and provenance. This
package neither overwrites nor relabels them. It implements no election,
revocation, cancellation, retention, scope-credit or historical-completion API.

If ROOT later separately elects a successor, ROOT must retain the original
products and record the new decision/provenance in fresh metadata. Only genuine
current lifecycle evidence and explicit authority can determine whether an
unfinished cell may be superseded. Any permitted new predictions belong to
fresh outputs. The eventual matrix must bind exactly one authenticated chosen
seal per original (parent,epoch) cell and retain the other products as history,
with the same original registry/checkpoint/training identity. New results do
not become historical old-job successes or imply any credit. A separate review
must resolve actual completion-envelope and matrix joins; this source package
does not fabricate that lifecycle or bypass the all30 barrier.

## Bounded verification

Run only the stdlib static script when reviewing this source:

```text
/usr/bin/python3 -B /workspace-SR003.nfs2/ARG-GNN/research/reviews/C3_interleaved_prediction_successor_source_20261017_v1/static_tests_v1.py
```

It parses/compiles source without importing it, authenticates only recorded
source/compact-audit JSON files, checks scheduler byte identity and full adapter
AST equivalence after the four disclosed edits, executes only isolated metadata
guards on fabricated Python lists/dicts, and verifies static route counts.
It opens no registry/plan/model/checkpoint/array/truth payload and invokes no
runtime, launcher, accounting or process code. `static_tests.result.v1.json`
records the actual static outcome. Scientific numerical equivalence and scale
fit remain unexecuted and unproved here. The unchanged source bodies provide
source-backed scheduling equivalence, not an executed bitwise-equality claim.
