# Conditional root-only shared native VAL000 sequence

Run on anogena in /workspace-SR003.nfs2/ARG-GNN. This packet is inactive source,
with no actual admission, guard/configure, pool action, GPU probe or native run.

1. Review source_handoff_v1.json, base scientific handoff, sealed GPU0 shared pool,
   exact21-source map, all11 pool metadata/source refs, current policy/actual
   absence,40GiB CUDA arithmetic and explicit7GiB profile/risk. After actual review
   create NEW GPU0_execution_approval.root_active_v1.json from the inactive
   template; fill root/active/timestamps and each explicit review flag truthfully.
   The source-review lifetime may be<=7days; actual stage/pool capabilities remain
   independently bounded and must cover full phase+drain inside genuine renewal.

2. Create NEW actual mlm_v1 GPU0 readiness approval from its inactive template.
   Fill active/root/actual execution-approval ref, IO review and absence review.
   Review caps_mlm.resource_proposal.inactive.json into a NEW actual caps file:
   set root_resource_caps_reviewed and smaller_operational_profile_reviewed true
   after reviewing actual resources. Keep active=false and null runtime/driver
   fields for bind. No old execution/storage authority or checkpoint is reused.

3. Bind then separately admit genuine native MLM using actual variables:

.runtime/env/bin/python -B research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/root_native_driver_v1.py --bind --gpu 0 --gpu-uuid GPU-ff06ca1f-403a-f2d0-b584-71e177452667 --caps-review "$NATIVE_VAL000_CAPS_REVIEW_PATH" --caps-review-sha256 "$NATIVE_VAL000_CAPS_REVIEW_SHA256" --gpu0-readiness "$NATIVE_VAL000_GPU0_READINESS_PATH" --gpu0-readiness-sha256 "$NATIVE_VAL000_GPU0_READINESS_SHA256"
.runtime/env/bin/python -B research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/root_native_driver_v1.py --admit mlm

4. Use the genuine admitted native input/storage/caps refs to fill the declared
   native job in pool_native_job.proposed.inactive.json. Its exact original args:
   ["--execute","--input",ACTUAL_INPUT_PATH,"--expected-input-sha256",ACTUAL_SHA].
   Runtime entry is this packet/shared_native_runtime_entry_v1.py and module is
   native_ARGformer_VAL000_mlm, with the genuine mlm_v1/entry_mlm980_v1.py ref.
   Job name must equal the new native storage/input identity. The native input
   directly declares actual operational pool/source/launcher/base/runtime/guard/
   policy/renewal refs. Keep native storage separate from controller/H050/C3.
   Runtime and module refs are direct native bindings; full required pool refs
   may use the directly bound native transitive closure with exact equal maps.

5. Root creates a NEW actual shared pool plan/capability and controller authority
   including that prebound native job. Do not mutate an already bound plan or
   capability. native_40GiB_operational_ceiling_authorized=true is required;
   native_72h_operational_deadline_authorized=true is required above24h. Only
   GPU0, max2concurrent, exact aggregate memory/headroom and real storage/expiry/
   renewal validation may admit it. The controller may queue native until a slot
   is available. After those actual reviews/admissions submit via:

.runtime/env/bin/python -B research/reviews/GPU0_shared_execution_source_20261016_v1/launch_gpu0_shared_v1.py run --execute --capability "$ACTUAL_GPU0_POOL_CAPABILITY_PATH" --sha256 "$ACTUAL_GPU0_POOL_CAPABILITY_SHA256" --job "$ACTUAL_NATIVE_LOGICAL_JOB"

   The adapter invokes the original root launch supervisor inside the pool's
   dedicated worker session. Never call native --launch outside that inherited
   context, and never take a second GPU lock or rewrite the pool lease.

6. Consume actual own-VAL000 MLM root/direct/native wait0/drain,5epochs and live/
   saved227990 counters. Only then bind/admit fit with this packet's top-level
   root_native_driver_v1.py --bind fit --gpu 0 --gpu-uuid GPU-ff06ca1f-403a-f2d0-b584-71e177452667 and actual --MLM/--MLM-wait
   refs/SHA pairs as in the base commands. The actual literal device arguments
   remain --gpu 0 --gpu-uuid GPU-ff06ca1f-403a-f2d0-b584-71e177452667.
   Fit uses native_ARGformer_VAL000_fit and top-level entry_postMLM_v1.py.
   Give it a NEW own reservation and pool declaration; submit via the same pool
   command with its own job. Its only initializer is the actual own MLM final.

7. Consume actual fit wait0/drain and saved2135 counters (independent live counter
   observation is unavailable). Then bind/admit embeddings with actual own MLM
   and fit completion/wait refs and use native_ARGformer_VAL000_embeddings with
   top-level entry_postMLM_v1.py under a NEW own reservation/pool declaration.
   Require real root/direct/native completion/drain and full10000 prediction seal
   before a separately authorized evaluator opens query truth. No TEST access.

All new result namespaces contain prediction_route_sharedGPU0. MLM packet is
mlm_v1/admitted_mlm_v1; post packets are admitted_fit_v1/admitted_embeddings_v1.
MLM TMP is .runtime/tmp/c3val0shm/m; post TMP .runtime/tmp/c3val0shpost/f or /r.
No future completion/checkpoint/prediction SHA is supplied. Keep every failure,
partial output and charge. Completed close needs separate actual root PID/group/
terminal/revocation approval through original --close; no automatic cleanup,
credit, retry, resume, source mutation or numerical fallback is authorized.
