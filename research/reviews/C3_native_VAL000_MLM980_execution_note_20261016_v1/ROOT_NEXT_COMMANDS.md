# Additive root activation note

Use the unchanged existing sealed driver below. No new launcher/code is introduced.

After genuine engineering completion/wait0/drain and full5ep verified counters, create engineering_acceptance.root_active_v1.json in this note folder from the original engineering_acceptance.inactive.json. Set active/native_GPU5_engineering_completed_and_reviewed true and bind actual engineering completion/wait refs. Leave authority reuse and checkpoint initialization false.

After root reviews successful engineering resources and actual host/GPU availability, create caps_mlm.resource_review.root_active_v1.json in this note folder from caps_mlm.resource_proposal.inactive.json, setting root_resource_caps_reviewed true. Keep active false and physical_gpu/runtime_ref/driver_source_ref null: existing --bind supplies them. The proposal is384GiB CPU RSS,50GB CUDA,259200s; all other16GiB reservation/14GiB phase settings are unchanged. Counts/source arithmetic does not prove fit or deadline completion.

Prefer physicalGPU5 after engineering releases it, preserving current GPU4 C3. Actual UUID and>=54000000000free bytes remain root checks. Set the original task variables below with GPU5, actual UUID and the two new reviewed files' actual SHA256 values. Do not guess engineering completion refs.

Fresh16GiB current980 headroom is an independent gate. Engineering completion does not itself release its charge; any engineering close must use root's separately authorized current accounting route. The v12 legacy close is disabled. No old v12/engineering authority or checkpoint is replayed. Launch only after actual new admit success, and retain genuine root-owned direct outer wait.


# Root commands after actual review

All commands run on anogena in /workspace-SR003.nfs2/ARG-GNN. Do not run the launcher directly, including help. Root supplies real reviewed metadata refs; these variables deliberately have no defaults.

First complete actual native GPU5 engineering, consume its genuine terminal wait0/drain, and write a new engineering acceptance JSON from the inactive template. Separately write a new reviewed caps JSON from the proposed inactive template. Leave their originals unchanged.

Before bind, supply these task variables: VAL000_MLM_GPU (4 or5), VAL000_MLM_GPU_UUID, VAL000_MLM_CAPS_REVIEW_PATH, VAL000_MLM_CAPS_REVIEW_SHA256, VAL000_MLM_ENGINEERING_PATH, VAL000_MLM_ENGINEERING_SHA256. Paths are canonical project-relative paths and hashes are actual file SHA256 values.

    cd /workspace-SR003.nfs2/ARG-GNN
    : "${VAL000_MLM_GPU:?Root actual eligible physical GPU required}"
    : "${VAL000_MLM_GPU_UUID:?Root actual GPU UUID required}"
    : "${VAL000_MLM_CAPS_REVIEW_PATH:?Root actual reviewed caps path required}"
    : "${VAL000_MLM_CAPS_REVIEW_SHA256:?Root actual reviewed caps SHA required}"
    : "${VAL000_MLM_ENGINEERING_PATH:?Root actual completed engineering acceptance path required}"
    : "${VAL000_MLM_ENGINEERING_SHA256:?Root actual completed engineering acceptance SHA required}"
    .runtime/env/bin/python -B research/reviews/C3_native_VAL000_MLM980_root_source_20261015_v1/root_native_driver_v1.py --bind --gpu "${VAL000_MLM_GPU}" --gpu-uuid "${VAL000_MLM_GPU_UUID}" --caps-review "${VAL000_MLM_CAPS_REVIEW_PATH}" --caps-review-sha256 "${VAL000_MLM_CAPS_REVIEW_SHA256}" --engineering-acceptance "${VAL000_MLM_ENGINEERING_PATH}" --engineering-acceptance-sha256 "${VAL000_MLM_ENGINEERING_SHA256}"
    .runtime/env/bin/python -B research/reviews/C3_native_VAL000_MLM980_root_source_20261015_v1/root_native_driver_v1.py --admit mlm
    setsid .runtime/env/bin/python -B research/reviews/C3_native_VAL000_MLM980_root_source_20261015_v1/root_native_driver_v1.py --launch mlm

Retain and consume the genuine outer launch terminal status. The driver must begin its own session and process group; it enforces that condition. Do not fabricate success from launch.json, progress, a checkpoint, or an output directory. On any failure preserve the entire namespace/charge and stop for root review.

After genuine completed terminal0 and all owned processes/groups drained, root copies root_close_approval.inactive.json to a new actual approval, supplies exact input/actual_wait refs, complete actual PID/group rosters, and explicitly revokes all old write/launch/resume capabilities. Only then:

    : "${VAL000_MLM_CLOSE_PATH:?Root actual terminal close approval path required}"
    : "${VAL000_MLM_CLOSE_SHA256:?Root actual terminal close approval SHA required}"
    .runtime/env/bin/python -B research/reviews/C3_native_VAL000_MLM980_root_source_20261015_v1/root_native_driver_v1.py --close mlm --root-approval "${VAL000_MLM_CLOSE_PATH}" --expected-root-approval-sha256 "${VAL000_MLM_CLOSE_SHA256}"

Default inert inspection commands (already passed) omit bind/admit/launch/close and execute:

    .runtime/env/bin/python -B research/reviews/C3_native_VAL000_MLM980_root_source_20261015_v1/root_native_driver_v1.py
    .runtime/env/bin/python -B research/reviews/C3_native_VAL000_MLM980_root_source_20261015_v1/entry_mlm980_v1.py
