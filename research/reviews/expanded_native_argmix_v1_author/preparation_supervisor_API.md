# Root execution of H038 native ARGMix preparation

Supervisor: src/arggnn/prepare_expanded_native_argmix_v1.py.
Frozen scientific adapter: src/arggnn/expanded_native_argmix_v1.py, SHA00c8e57423d38b484bda3cfeb126c188add63c383e399df874a1a2e55e4bca23.
Accounting: src/arggnn/project_storage_terminal_success_scopes_v1.py, SHA2dfbcc2711fc8310fd8a8b9866ee84e9107e1d2c4356396e0ebc0e99fdb7c3b3, mode journal_terminal_success_scopes_credit_v1.

The new supervisor provides a direct root launch path. It prepares ap_expanded_train_000 through031, then ap_expanded_val_000 through007. It waits only for each registered original generation/Relate completion, never generates or reinfers data, then streams the assembled train.pt/val.pt. No GPU fit or outer work is included. Each completed parent has a separate durable product record. Partial output is preserved on failure; automatic replay/reuse is disabled.

## Root admission steps

1. Copy configdraft.json into a new root-owned native config, setting execution_guard_ready=true while preserving every native seed/model/optimizer/selection setting and five frozen scientific source pins. Freeze its path/SHA.
2. Copy preparation_runtime_draft.json into the root packet, updating source/job/runtime evidence and freezing its path/SHA. This is CPU-only, existing native Python3.12/Torch2.10.0+cu129. Existing build/CPU-library receipts are already resolved in the draft.
3. Use the frozen successor helper's apply_reserve_v1. Its required accounting bindings are implementation, launcher, cpu_overlay, frozen terminal_overlay and terminal_success_overlay, plus the original v9 consumer stub and all actual supervisor/scientific/config/runtime/producer/build receipts. The draft's source_sha256 is the exact15-Python-file closure. Bind the new active config/runtime records instead of their draft records.
4. Write active root storage with kind=project_budget_v9_storage_record, consumer_uses_v9_guard=true, accounting_mode=journal_terminal_success_scopes_credit_v1, reserved_new_bytes=50000000000 and the exact reservation. Initial ledger pins are the five post-reservation pins: legacy_ledger_sha256, supplemental_ledger_sha256, journal_tail_sha256, cpu_overlay_tail_sha256, terminal_overlay_tail_sha256.
5. Copy preparation_admission_draft.json into the root packet. Set status=active, execution_authorized=true, created/expires UTC, config_ref/runtime_ref and actual storage. Keep GPU_authorized=false, fit_authorized=false, outer_included=false and TRAIN_and_VAL_truth_preparation_authorized=true. The root-owned input must remain immutable during execution. Freeze its SHA.

The source does not create a reservation, mutate a ledger or activate any draft. Root remains responsible for source/config/runtime admission and launch.

## Direct command

Source scripts/project_env.sh, then execute:

```text
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m arggnn.prepare_expanded_native_argmix_v1 --execute --input <root-active-input.json> --expected-input-sha256 <SHA>
```

Without --execute, the CLI only reports source pins and performs no admission or preparation.

The supervisor uses the ordinary project interpreter. Each parent worker uses the existing qualified native interpreter at .runtime/official_argmix_abi210_v1/cu129/env/bin/python. The worker invokes the frozen adapter's load_config/admit_native/prepare_observed/prepare_native functions; assembly invokes its existing streaming assembler. The five-module return is consumed directly. No library install, version bypass or hidden source edit occurs.

Every worker receives CUDA_VISIBLE_DEVICES empty, one CPU thread, disabled bytecode and caches/TMP wholly under the admitted output's sandbox. The supervisor records its exact worker command, PID and process group before work. It monitors only that worker and explicitly listed /proc descendants, enforces total owned RSS and the real successor budget guard every2seconds, and terminates/waits for its own group on a breach or stop. Guard calls reserve actual compact writes and the frozen adapter's conservative native archive estimate. This adds execution containment only; native extraction, all samples/ages, first-SNP labels, chunking, seed50 and later native training settings are unchanged.

Outputs include started.json; logs/<parent>.launch.json and log; parents/<parent>/observed and native products; progress_01 through40.json; parent_products.json; assembled/native_prepared.json; assembled_product.json; completed.json. A failed.json retains completed parent refs and the honest failure. The completed.json native_prepared_ref is the direct input to the frozen native fit API.

## Conservative cap without a scan

preparation_storage_estimate.json reads only38 targeted original inference JSONs. Their native trees span852–1026 and sites are at most14067; VAL006/007 were pending at drafting. The stored numeric arrays require672768bytes per1024-row chunk. Per-parent upper chunks are686*(1+ceil(max(tree_count−1024,0)/512)); edge exclusion cannot increase this bound. Known parents use their exact metadata bounds; the two pending VAL parents have admitted bounds2048trees/20000sites. These are storage admission bounds, never data truncation.

All parent archives plus their assembly duplicate and conservative SNP/metadata allowance are bounded by42451517680bytes. The50GB cap leaves7548482320bytes for logs/cache/headers/workspace. Native array bytes are copied once into the assembled archives, and every original per-parent archive is retained. If a full parent exceeds its declared shape bound, the supervisor stops before preparation; root must revise resource admission rather than truncate or change the method.

Only parsing/import, exact parent order and CPU environment sanity were performed. No actual observation/truth arrays, native Rust/model, GPU, optimizer, storage authority, reservation, installation or commit was executed. Original adapter/audit artifacts are unchanged.
