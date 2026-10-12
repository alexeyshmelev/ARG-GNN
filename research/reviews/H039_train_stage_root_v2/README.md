# H039 TRAIN v2 root helper

V1 admission omitted the literal v9 consumer marker. Its immutable 15 MB reservation remains fully charged and its execution stays revoked. The v1 worker, config and root driver remain unchanged.

V2 uses a fresh job and output scope with the same 15 MB cap, 2 MB qualification allowance, three arms, 320 additional updates each, six frozen model/optimizer checkpoints at passes 5/10, and deferred evaluation. Its 46-source computational closure replaces the TRAIN v1 source with v2. Training, protocol, sidecar-reuse and phase-validation bodies were compared at the AST level; the optimization behavior is unchanged.

The v2 worker and root driver pin and configure the targeted marker-repair reader. Fresh reservation consumers explicitly include the actual repair helper, separate root repair authority, literal v9 consumer marker and new sharing launcher. The original six accounting bindings and frozen guards remain in use.

Root must first publish a genuine activated repair authority from research/reviews/H039_missing_marker_repair_v1/repair_authority.draft.json, including its barrier reference and future expiry. Then:

```sh
.runtime/env/bin/python research/reviews/H039_train_stage_root_v2/root_train_stage_driver.py --admit --repair-authority research/reviews/H039_missing_marker_repair_v1/repair_authority.json --expected-repair-authority-sha256 5899970a2cb345b508d4f523679bd703fcadb9b1b47ba9fd30c5ef482dc5961a
.runtime/env/bin/python research/reviews/H039_train_stage_root_v2/root_train_stage_driver.py --launch --gpu 2
```

The default selected launcher is scripts/launch_reserved_gpu_authorized_external_share_v2.py. The default CUDA reserved cap is 50,000,000,000 bytes. Admission does not inspect GPU inventory or launch; explicit launch uses the existing reserved-launcher CLI and saves actual launch.log/root_wait.json under admitted_v1. Failed scopes and errors remain; the helper does not replay either admission or execution automatically.

Admission installs the active config at configs/directed_clade_exchange_train_stage_v2.json because the worker requires that fixed path. All other admission metadata stays under this review directory. The worker decodes only 32 TRAIN banks/sidecars and has exactly 32 TRAIN target bindings; VAL arrays, truth and predictions remain deferred.

Author verification uses source hashes, JSON structure and AST parsing only. No driver/worker/helper imports, admission, live inventory inspection, arrays/checkpoint/truth reads, GPU launch, deletion or journal mutation occurred while authoring the v2 stage.
