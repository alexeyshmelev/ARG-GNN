# Native ARGMix preparation v3 containment API

Source: src/arggnn/prepare_expanded_native_argmix_v3.py.
Frozen scientific adapter: src/arggnn/expanded_native_argmix_v1.py, SHA00c8e57423d38b484bda3cfeb126c188add63c383e399df874a1a2e55e4bca23.
Storage helper: src/arggnn/project_storage_terminal_retired_scopes_v1.py, SHAbbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab.
Mode: journal_terminal_retired_scopes_credit_v1.

V3 repairs supervisor containment only. Native code/math/settings, 32 TRAIN then 8 VAL, all samples, 50GB cap and 16-file source closure remain unchanged. V1/v2 source and author artifacts are preserved byte for byte. The source retains its v1 admission schema and frozen adapter API.

Before Popen, the supervisor performs a strict boundary check for declared native archive growth plus the RAM log cap and compact metadata, then saves logs/<parent>.intent.json. Per-parent archive bound: 686*(1+ceil(max(maximum_native_trees-1024,0)/512))*672768+8000000 bytes. Assembly uses the sum across all40 parents plus2000000 bytes. Complete-parent metadata must still satisfy the tree/site shape limits. These bound native archives; the unchanged adapter separately guards actual observation, truth, metadata and native writes.

Worker stdout=PIPE and stderr=STDOUT. The supervisor reads a nonblocking pipe, retains output in RAM and forwards it to root stdout. The draft max_worker_log_bytes is67108864. Root stdout and stderr, including any upstream pipe capture, must be outside the native output scope. Direct /proc/self/fd/1 and2 file destinations are checked before spawning. While active, no supervisor scope file is written.

Own-worker/explicit-descendant RSS, expiry and boundary=False reservation checks run every2seconds. They do not inventory the changing scope. Native adapter actual-write callbacks retain boundary=True. On failure, root stops and waits for its owned worker group. An overflowing RAM log is a declared failure with an explicitly marked retained prefix.

Only after the worker and inherited group exit does root perform the final strict boundary and save .log, .launch.json and .exit.json. Launch PID/group metadata is held in RAM until then. If group exit cannot be established, no final scope write occurs. Ordinary failure preserves completed products and disables automatic replay.

## Root admission and command

1. Finish H038 fits and every terminal-success writer before activating retired credits. This packet applies no authority or retirement.
2. Copy research/reviews/expanded_native_argmix_v1_author/configdraft.json into a root-owned active config, setting execution_guard_ready=true and preserving every scientific pin/native setting.
3. Freeze an active CPU runtime from preparation_runtime_draft.json with the v3 source/job bindings and existing native Python3.12/Torch2.10.0+cu129 build/library evidence.
4. Use the retired helper's apply_reserve_v1 for the unchanged50000000000-byte cap. Reserve all six accounting refs, the original v9 consumer stub, all16 source pins and actual config/runtime/generation/registration/build/library refs. The draft lists22 bindings. Replace draft config/runtime refs with their active refs.
5. Bind the five post-reservation ledger pins: legacy_ledger_sha256, supplemental_ledger_sha256, journal_tail_sha256, cpu_overlay_tail_sha256 and terminal_overlay_tail_sha256.
6. Create an immutable root input from preparation_admission_draft.json with status=active, execution_authorized=true, created/expires UTC, active config/runtime refs and exact storage. Preserve max_worker_log_bytes and root_stdout_must_not_write_inside_native_output_scope=true. Keep GPU_authorized=false, fit_authorized=false, outer_included=false and TRAIN_and_VAL_truth_preparation_authorized=true.

After sourcing scripts/project_env.sh:

    CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m arggnn.prepare_expanded_native_argmix_v3 --execute --input <root-active-input.json> --expected-input-sha256 <SHA>

Without --execute the CLI only reports source pins. Source creates no reservation or authority.

The unchanged prior estimate is research/reviews/expanded_native_argmix_v1_author/preparation_storage_estimate.json, SHA09972ded205bce13b30df23a6fd6490dea1fbf4c2aa704affbf4f23660ad08b8, with42451517680 bytes structured output. At the draft cap,41 RAM-derived durable worker logs are bounded by2751463424 bytes. Actual-write/final scope checks remain authoritative.

Focused verification used three mocked worker lifecycles: success, reservation revocation and RAM-log overflow. They assert intent-before-spawn, lightweight active checks, and wait-before-durable-records. No actual worker/native launch, actual observation/truth arrays, fit, GPU, storage authority, broad tests, installation or commit occurred.
