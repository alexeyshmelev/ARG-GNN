# H042 memoized frozen evaluation source

The new entry point is evaluation-only. It imports the untouched train_anonymous_clade_flag_control_v1 module and calls its original run_context. run_evaluate, FlagRAMMatrix, FlagCache, checkpoint provenance, scoring and selection are unchanged functions/classes. Their source hashes are recorded in scientific_identity_receipt.json. All six checkpoints × eight VAL parents produce the same 48 RAM seals before any VAL truth. No H042 training source, active context, driver or configuration was edited.

The future context installs finalized transaction_memo_adapter_v2 (c972e9a7…, 256 MiB / 16,384 entries) before its first storage guard. It retains the original H042 lightweight lease/title/RSS/CUDA guard without adding per-step accounting. Every strict writer boundary remains unthrottled. Current credit-v3 policy/accounting/launcher sources remain unchanged.

## Root prerequisites and use

Admission requires the exact original H042 TRAIN completed.json and original admitted_train launch.json/actual_wait.json. The completion must show 30 passes, 960 updates, all six checkpoints, original provenance/scientific refs and no VAL or outer truth. The actual direct launcher wait must be zero and match the completion and TRAIN input. The main and consumer revalidate these bindings.

Use the separate root_driver.py with --admit --gpu 4 (or 5), then --launch with the same GPU. Root alone chooses between physical GPUs 4 and 5 and owns resource admission/execution. The fixed admitted_evaluate packet prevents a second admission on the other GPU.

The future evaluation configuration is copied from the actual TRAIN config. Its only permitted change is source_manifest_ref, extending the source closure with the new main/context/driver and mandatory memo helper. Restoring that one field must reproduce the TRAIN config exactly. The original source manifest, config and active TRAIN files remain immutable. The new helper and actual TRAIN terminal refs are direct bindings and transitively pinned. Original prefix pins, 64-consumer and 64 KiB metadata limits are preserved.

The new supervisor buffers at most 4 MB of log payload in RAM and does no budget/scope scans or charged log writes while its launcher runs. It preserves the actual return code even if terminal logging fails. It makes no descendant-drain or completed-credit claim.

## Qualification boundary

Sources AST-parse and static checks prove direct science reuse, unchanged lightweight guard, memo installation before storage validation and no prefix rebasing. Existing memo-helper handmade qualification was reused; no fixture campaign was duplicated. This task imported no live guard/model runtime, read no live guard state or H040/H042 VAL artifacts, created no admission and launched nothing.
