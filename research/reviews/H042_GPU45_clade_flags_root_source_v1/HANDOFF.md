# H042 GPU4 exact-clade-flag root integration

This source-only integration uses the frozen anonymous_clade_flag_control_v1.py model and train_anonymous_clade_flag_control_v1.py worker. The control remains H040_exact_clade_flag_control_v1 / exact_clade_flags because those are the fixed worker API identities. The independent hypothesis ID is H042. Only the proposed config's hypothesis placeholder and operational status changed; all existing scientific fields remain identical.

The config preserves the selected H038 FP32 aligned seed101 pass30 warm start, original32TRAIN/eightVALparents,30passes,960full-parent updates, six checkpoints5/10/15/20/25/30, optimizer/reset policy, metric/selection contract, no early stopping and no VAL-dependent optimization. The prospective protocol predates H040 VAL. No H040 VAL results, probability arrays, labels, learned checkpoints or observation arrays were read during authoring. Existing handmade fixtures were not rerun.

context_factory_v1.py handles the exact anonymous_clade_flag_root_execution_input_v1 schema. It validates source/config/runtime/job closure, the current credit-v3 guard, exact root policy, marker repair authority, shared lease and process title before qualification. It restricts the worker to physicalGPU4, preserves the50GB CUDA and512GB RSS limits and original lightweight resource checks, and uses configs/census_tsinfer_fit_v1.json for configure_fit. That base file supplies runtime configuration; the H042 config supplies this model's scientific training protocol.

The source manifest pins original/template/scientific/runtime/launcher sources transitively. Each future admitted consumer manifest binds that source manifest, phase-only target metadata, original40observed entries and predecessor terminal evidence. The current guard and launcher-required wrappers/authorities remain direct reservation bindings with worker/config/runtime/context/driver/source-manifest/consumer-manifest. The driver checks the64binding cap. No old reservation is copied or reactivated.

Exact guard: concurrent_completed_scope_credits_v3 SHA ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef.
Exact root policy: activation_policy.json SHA a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85.
Exact new GPU45 launcher: launch_reserved_gpu_authorized_GPU45_completed_v1.py SHA d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d.
The driver passes exact ARGGNN_COMPLETED_CREDITS_POLICY_REF JSON to launcher and worker; storage.completed_overlay_policy_ref matches.

Root owns every operation below. Default invocation only prints source status. This task has not imported/configured the live guard, obtained reservations/admissions, run the driver/context/science, built CUDA, installed anything or used Git.

After root actually waits for H040tensor on GPU4, the driver requires actual_wait returncode0 and its exact pinned H040tensor completed.json reference, plus launch.json/actual_wait.json launcher identity and physicalGPU4. It does not inspect tensor scores or any H040VALoutcome. It never stops or changes the otherGPU5job. Root can review and run:

    .runtime/env/bin/python -B research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py --admit train
    .runtime/env/bin/python -B research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py --launch train

Each phase gets a fresh40,000,000byte aggregate output/auxiliary reservation. The original scientific output proposal remains32MB. Fresh worker_aux holds the launcher log, capped at4MB. The parent drains later stdout without retaining bytes past that logging cap and records received/retained/truncated counts. Scientific artifacts are never truncated or deleted. Planned retained log chunks use the actual current storage guard. Compact packet metadata outside bulk scopes stays within overlay charged metadata allowances. A direct proc.wait creates actual_wait.json with launcher return code and exact completion ref if present. A log guard error is recorded and produces a nonzero driver exit after actual wait even if the worker succeeded. No automatic retry.

A later root-reviewed evaluate admission requires:

    --admit evaluate --train-completion <H042-TRAIN-completed.json> --expected-train-completion-sha256 <exact digest>

Then root invokes --launch evaluate. Context supplies singular train_completion_ref. The frozen worker's48cell RAM seal before eightVALtargets and selection protocol are unchanged. Evaluation is a separate fresh guarded phase and has not run here.

Compilation and mechanical hash/config/source-closure checks are the only qualification. Runtime/lease/guard behavior is preserved by source reuse but remains unexecuted in this package.

Pre-admission review repair: read1(65536) streams flushed worker progress without waiting for a full64KB buffer. Both root driver and context require the exact successful tensor completion. pre_admission_review_fix.diff preserves the prior unadmitted source/metadata changes. Compilation was the only additional check; no driver, worker, guard or GPU execution occurred.
