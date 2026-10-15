# H040 frozen evaluation v3 source handoff

This successor changes runtime I/O only. It installs the finalized transaction_memo_adapter_v2 helper (256 MiB / 16,384 entries) in the future consumer before configure/validate/first storage guard. No guard call is throttled. The current credit-v3 source, policy, journals, title, lease, RSS, CUDA and strict planned-write boundaries remain in use. Old v2 sources/results are untouched.

scientific_identity_receipt.json proves byte-identical run_evaluate, run_context, current_resource_config, target-view and all three checkpoint/provenance functions against current v2. RAMMatrix and selection still come from the identical original H040 base source. Both original 960-update TRAIN completions and direct waits are used unchanged. All 96 RAM seals and the pre-VAL truth gate are unchanged. No training is added.

## Required root evidence

The driver refuses admission unless supplied a bound runtime supersession evidence record. Inactive templates are provided. Root must create genuine active copies after the old v2 stops. Required evidence is:

- Exact original v2 launch.json and actual_wait.json, with a nonzero actual direct wait for the same launcher PID and no completion ref.
- Complete targeted root PID absence/revocation certificate, with every observation after the actual wait.
- Old completed.json and all96_VAL_RAM_sealed_before_truth.json both absent. The main validates this at admission, before launch and in the future consumer.
- Runtime-only repeat reason duplicate_guard_IO_before_VAL_truth, preserved old files, no VAL results used and no new training.

Root alone stops the old run, observes processes, admits and launches. This task did none of those actions and did not read old results or any live guard state.

## Future root commands

Use the existing project environment. The inert driver default only describes the deferred operation. For admission, pass both --supersession-ref PATH and --supersession-sha256 SHA with --admit. Launch uses --launch --gpu 5 after successful admission. The new driver binds the helper directly and in the mandatory source closure, plus the original v4 source and old source/terminal evidence. Existing 64-consumer and 64 KiB metadata bounds remain enforced.

The helper is installed only in the future model consumer. Root reserve/close mutations do not use this read/check memo context. runtime_memo_installation.json records its original source/function identities. There is no cache across storage operations.

Qualification: all successor sources AST-parse. Nine handmade metadata cases accept the valid old terminal barrier and reject missing/zero wait, live writer evidence, wrong old input, either existing terminal gate, wrong memo pin and VAL-conditioned repeat. No live guard/model imports or production execution occurred.
