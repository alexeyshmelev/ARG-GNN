# H039 TRAIN stage root helper

The helper admits one fresh 15,000,000-byte TRAIN-only scope, including the existing 2,000,000-byte qualification allowance. The worker uses three arms with 320 additional updates each and retains six actual model/optimizer checkpoints at passes 5 and 10. Evaluation is deferred; TRAIN target bindings contain exactly 32 parents. The original 40 observation-entry metadata refs retain cache lineage, while the worker decodes only 32 TRAIN bank and sidecar array sets.

Root admission on anogena:

```sh
cd /workspace-SR003.nfs2/ARG-GNN
.runtime/env/bin/python research/reviews/H039_train_stage_root_v1/root_train_stage_driver.py --admit --launcher scripts/launch_reserved_gpu_authorized_external_share_v1.py --max-cuda-reserved-bytes 50000000000
```

Explicit root launch after admission:

```sh
.runtime/env/bin/python research/reviews/H039_train_stage_root_v1/root_train_stage_driver.py --launch --gpu 2
```

Admission reads the current inactive fixed config and current official 46-source map. It verifies those hashes and the worker's exact closure, binds the actual root-selected launcher and the sharing authorization, writes runtime/phase-registration/TRAIN-binding/storage/input records under `admitted_v1/`, and applies one reservation. It installs the activated config at `configs/directed_clade_exchange_train_stage_v1.json`, the worker's required fixed path. It does not inspect GPU availability or launch a process.

The optional launcher choice defaults to `scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py`. With the explicit sharing wrapper, both the unchanged original launcher and actual sharing wrapper are reserved, together with `external_GPU_sharing_authorization.json`. The existing launcher CLI, guard, lock, title, runtime and wait mechanisms remain in use.

Launch verifies the admitted source/metadata/launcher pins, refreshes the same reservation's ledger pins, and invokes the selected module-form launcher. It saves actual output in `admitted_v1/launch.log` and records the actual direct launcher wait status in `admitted_v1/root_wait.json`. Failed launcher stderr is retained in the log and the wait receipt's output tail. Interrupted or failed scopes are preserved; there is no automatic admission or execution replay.

The `activated_config_draft.json`, `source_closure.json`, `TRAIN_target_bindings.json` and `phase_registration_draft.json` files are reviewable authoring snapshots. Actual admission derives its active records from current metadata. All authoring writes stayed in this directory. Author verification used source hashes, JSON structure and AST parsing only, without importing/running the driver or worker, reading arrays/checkpoints/truth, inspecting live GPU inventory, reserving storage or launching a job.
