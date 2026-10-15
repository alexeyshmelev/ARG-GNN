# H046 future evaluation: prospective980 IO successor

This is a separate source-only evaluation integration. Root owns admission,
GPU5 scheduling and execution after genuine original TRAIN completion and
launcher wait0. This packet creates no reservation, checkpoint or result.

The scientific configuration is byte-identical to the actual frozen v2 TRAIN
configuration, SHA 13d1b2fa0503a452c3bee7ba3b47bf3d702ade18a796c0f4c6e562e6f129f5a8.
The frozen worker checks TRAIN/evaluation configuration equality except resource
caps. Editing helper or source-manifest fields inside that config would violate
the unchanged checkpoint protocol. Therefore the new active980 guard, policy,
launcher and source closure live in an explicitly validated root IO envelope.
The old config helper fields remain immutable provenance. All actual storage
operations use the new IO envelope.

The separately pinned new launcher reuses the exact frozen full-storage lease
launcher, substituting only the storage mode/check. It retains the exact GPU4/5
authority, at-most-two project devices, external-process preservation, inherited
lease FD, process-title validation and 50 GB+4GB memory eligibility rules.
Admission mutations use original unmemoized v4 functions. Consumer checks retain
the original memo2 installation. Frozen files remain unchanged. The original consumer memo2 behavior
is retained.

All 81 original source pins remain exact. The extended transitive source closure
has 87 sources. New root_common, root_driver, context, canonical CLI and launcher
are directly bound. The prospective direct-binding shape is 40, below 64. Root
admission reads the actual live 980 binding roster and rechecks the limit.

The evaluation config, paired model, objective, checkpoint loading, original
8 VAL/native query support, original-grid scoring, metrics and selection rule
remain unchanged. The entire scientific context tail is byte-identical. The
same 8 VAL result is development selection evidence. No new heldout claim is made.
The frozen worker seals all 96 prediction products before opening any VAL truth.

The original 100 MB output cap, 512 GB CPU RSS cap, 50 GB CUDA cap, 4 MB log cap and 24-hour
admission expiry duration remain unchanged. No TRAIN cap, live process, original
launcher, old guard/policy or historical journal is patched.

## Immutable TRAIN lineage

New admission requires actual completed_fixed30pass_paired_TRAIN, both 960-step
arms, the genuine original launch and actual direct launcher wait0, no logging
or supervisor error, and exact original GPU5 TRAIN input/config/runtime refs.
It verifies all six pass groups and both arm checkpoint-reference names/hashes
as metadata. These 12 checkpoint refs are pinned transitively by the actual TRAIN
completion and consumer IO manifest. The unchanged scientific reader byte-checks
and validates each checkpoint at actual execution. This agent never reads them.
No fabricated completion, wait, checkpoint or drain claim is generated.

Source handoff does not itself certify TRAIN-scope closure. Root must confirm
genuine terminal completion, direct wait0 and released original lease before
launching evaluation. The original wait semantics and no-descendant-drain-claim
limitations are preserved.

## Root commands after genuine TRAIN completion

Run on anogena from /workspace-SR003.nfs2/ARG-GNN. Fresh evaluation packet:
research/reviews/H046_GPU45_flank_eval980_source_v1/admitted_evaluate.
Fresh output:
experiments/results/H046_GPU5_flank_evaluate_seed101_eval980_v1.
Existing scopes are never replayed.

~~~bash
.runtime/env/bin/python -B research/reviews/H046_GPU45_flank_eval980_source_v1/root_driver.py --admit evaluate --gpu 5   --train-completion experiments/results/H046_GPU5_flank_train_seed101_v2/completed.json   --expected-train-completion-sha256 "$(sha256sum experiments/results/H046_GPU5_flank_train_seed101_v2/completed.json | cut -d ' ' -f 1)"   --train-launch research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/launch.json   --expected-train-launch-sha256 a899af3d485339b26e1873eb13d7db0b514c4886b668a5e81127c29e8d20cd0e   --train-actual-wait research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/actual_wait.json   --expected-train-actual-wait-sha256 "$(sha256sum research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/actual_wait.json | cut -d ' ' -f 1)"
~~~

Review the returned admission metadata and actual TRAIN terminal lineage, then:

~~~bash
.runtime/env/bin/python -B research/reviews/H046_GPU45_flank_eval980_source_v1/root_driver.py --launch evaluate
~~~

The canonical worker is arggnn.H046_GPU45_flank_eval980_v1.
The actual launcher is scripts/launch_reserved_gpu_authorized_GPU45_980_v1.py.
The active adapter is prospective_980gb_v1 with actual root-active policy,
unchanged marker repair and exact GPU4/5 sharing authority.

integration_only.diff records the five source changes. source_static_receipt.json
records 25 bounded source/hash/AST and metadata-shape checks. No science or payload
qualification was rerun. No guard configuration/admission, model/GPU/native/Git
execution, target/checkpoint/focal/tree/query-truth payload read or actual TRAIN
completion assertion was performed by this agent.
