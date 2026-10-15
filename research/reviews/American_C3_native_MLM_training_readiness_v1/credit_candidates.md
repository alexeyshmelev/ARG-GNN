# Retained-file credit candidates for full MLM

This is a read-only source/compact-metadata review. No guard was imported, process inspected, product scope inventoried, or accounting/admission operation performed. No earned credit is asserted.

## Best candidates

| Scope | Charged cap | Completion and recorded terminal evidence | Retained-byte evidence |
|---|---:|---|---|
| H038/expanded-native-argmix-full32TRAIN8VAL-fit-v1 |7,000,000,000|Root completed.json SHA9e27f2c188d74ed652ecdf10f5fdf51d23e6ffd20742f35d9832bdefcf3c698a; native_argmix_GPU_fit_v1/root_wait.json SHAa8744fc52bc1a44bf6afb86cd36fce99022745a87f44f9856da200d4ca5f6971; pid389848/returncode0, actual_direct_launcher_wait=true|No current exact retained-byte certificate read. Original size_bounds.json caps fit128MB, checkpoint64MB, VAL outputs5,317,347,632B and workspace1GB; these are upper bounds, not measurements.|
| H038/mutation-routed-expanded-v1-aligned-seed101 |5,000,000,000|Completed SHAa36e449a5ec3042993419eaf1a61628122fa24f15de320f0ad02f94a2b087567; relaunch_v6_wait.json SHAe59cc0eac626179d6221f6b772b9b67450334ae1f6d405318d6d6917f0f43ff8; pid383019/returncode0|No retained-size measurement. Receipt records owned model pid383455, workers383567/383568, drained=true, exitcodes0/0.|
| H038/mutation-routed-expanded-v1-zero-concordance-seed101 |5,000,000,000|Completed SHA5b6235f649510f6b1a4074f11f89975cde3cf8867d9eb78834454e742f8ed1c3; relaunch_v6_wait.json SHAa6466009c193e440300f23f67c639c0dd771e4b72ce1238505cfe9ff0c680d9d; pid383020/returncode0|No retained-size measurement. Receipt records model pid383091, workers383178/383179, drained=true, exitcodes0/0.|
| H038/expanded-full-panel-pyg-preparation-v5 |4,000,000,000|Completed SHAadcbd391e769d83ac4d882b31e40742ed78556950b661ea4a18e569d7f1b1e15; launcher_exit.json SHA424144c6a7412e4581beea65c7e77e9081c83d213232708543007cd402dfa65b; pid379173/returncode0|Producer records1,672,000,826 compressed payload bytes,1,688,527,620 guarded planned bytes and owned_workers_drained=true. These are historical producer metadata, not the complete current stat certificate.|

All paths above lie below research/reviews/expanded_full_panel_v1_root, except the producer outputs documented with exact refs in credit_candidates.inactive.json. Three small terminal receipts do not carry an explicit actual_direct_launcher_wait flag; preserve their original content and have root verify their direct-wait provenance. Do not fabricate a successful wait or infer it from completion alone. The successful v6 relaunches are the candidate terminal records for the two arms; earlier initial/v5 launch waits are1 and must not be selected or relabeled. Root must include/revoke every old launch/resume capability for the same scope.

These four caps total21,000,000,000B; credit is that total minus actual retained bytes, with512KiB charged per close event. This total is not available credit.

Two additional positive C3 candidates are:
- H041/American-C3-observe-qualification100kb-20261015-v5, cap1,250,000,000B. completed.json SHAef752c4d350cf6c097f69704c64b35683c761415465be487cf3e5c729dd4076e. Original actual_wait.json SHA56460954c772e26abeda9c46197bd58d6c7ff9b955cf07fce11b24e2aa40e61c records actual_direct_worker_wait=true,pid2543659,returncode0. Storage is admission.json/storage in its admitted_qualification100kb_v5 directory.
- H041/American-C3-qualification100kb-20261015, cap1,000,000,000B. completed.json SHAa2819204b13275bdb57a1d8a1e9f573bb1e874a86af8830bd8039760700475ac. Use retry2_launch.json SHA0190fb61d50de1da3a39affc69b3a67238450e1b9fdf7d0e6fce1d666cc75c11 and retry2_actual_wait.json SHA0db4bc88037d666ddd9c01f9ecb7ae69d55fa58a83a7a0d75006b84619ec0ed2, pid2525258,returncode0,actual_direct_worker_wait=true. The first attempt was1. Storage is retry2_admission.json/storage.

Their directory roots are respectively research/reviews/American_C3_observation_root_execution_20261015_v5/admitted_qualification100kb_v5 and research/reviews/American_C3_root_execution_20261015_v1/admitted_qualification100kb_v2. Together the six caps total23,250,000,000B before retained artifacts.

## Feasibility

Using root's approximate1.67GB remaining, an18GiB reservation needs about17.66GB fresh net credit;20GiB needs about19.81GB;24GiB needs about24.10GB. Add512KiB for every closure and the future reserve event and keep strictly below950,000,000,000B. Root must use its fresh authoritative guard view, not these rounded calculations.

The four large candidates can cover18GiB only if their combined actual retained bytes are below roughly3.34GB, before metadata. Since PyGcache already records1.672GB payload, this leaves roughly1.67GB combined for nativeArgMix plus both arms. This cannot be confirmed without root's permitted exact completed-scope inventories.20GiB will generally require additional candidates, and still depends on actual retained totals.18-20GiB remains a review option, not an established admission.

The source/metadata survey found25,056,842,752B total charged caps across open supplemental/overlay reservations with an output-root completed.json, including several small unsupported schema-only/status cases and unverified waits. Even this deliberately generous union cannot support24GiB under the reported remaining amount while preserving PyGcache's1.672GB recorded payload: other retained files and metadata further reduce credits.24GiB was forecast headroom, not a scientific training requirement.

## Current primitive and exact root requirements

Use only existing concurrent_completed_scope_credits_v3.py apply_close_completed_v1 under the original mutex. Source SHAea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef; active policy SHAa0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85.

The request has exactly key,reservation_sha256,storage_ref,storage_locator,launch_ref,launch_pid_locator,launch_identity,direct_wait_ref,wait_pid_locator,wait_exit_locator,wait_identity,completed_ref,completed_identity,observation_ref. Each candidate JSON supplies actual storage/completion/launch/wait refs and locators for the first four. Root fills the exact reservation digest and genuine observation; do not treat the inactive review as authority.

Storage must contain the exact still-open reservation, output and original cap. The original producer completed_ref must be exactly output/completed.json, at most1MiB, with an exact completed-prefixed status identity, or one of the three allowlisted schema identities with its actual nonfuture completed_utc. PyGv5 is allowlisted. Original wait must resolve to integer0 for the exact launch PID.

Root's observation must bind the same key/reservation/storage/launch/wait/completion, list the complete root-owned launcher/supervisor/worker roster, certify honest targeted os.kill(pid,0) ESRCH observations, preserve every file, and revoke all old/original/resume/write capabilities. The listed PIDs above are evidence for root's audit, not a claimed complete roster. At apply the existing primitive rechecks each PID absent, performs one exact stat-only measure_scopes over every output and auxiliary scope, requires retained<=cap, rechecks absence and appends one immutable close event. Net credit is original cap minus actual measured retained bytes; every event costs524,288B. No delete, global scan/rebase, historical mutation, duplicate closure or substitute accounting mechanism.

Snapshot read: frozen v9 journal58 events, CPU56, terminal38; current v3 journal85 events, tailb62180d92ed6dc7b7b0a1b3995c0c4a517e63bb8ec975256537290f7e6543ec7. This was an unlocked metadata survey, not an authoritative live guard result. Already-closed scopes, including native v7 fits and full v3 corpus preparation, cannot be credited again.

## Largest excluded scopes

biological_mechanism_controls_v1/seed202 retains a52,321,048,576B charge. Its original exit.json SHA4786f2f68e3ed95582be6aaf868db9b36bf70e10839ce3e6d2935ba4c21708bf records launcher1230670,supervisor1230669,returncode0, but the required output-root completed.json is absent. Current primitive cannot close it using summary.json or exit.json as a substitute.

The second mechanisms303-cache-v2 reservation is52,321,048,576B and two census fit reservations are40GB each; none has its required output-root completed.json. Keep them charged. The anchor-parent40GB scope is already historically retired and ineligible for duplicate credit; its preserved matrix_failure.json records failed_partial_no_fallback. Native v6 scopes, C3 failed/unknown-exit attempts, and all other failures or unknown outcomes receive no proposed new credit. No historical evidence was modified.

The immutable legacy jobs ledger is not a candidate pool for this primitive: _state_locked obtains eligible reservation objects from the pinned supplemental/overlay readers, while legacy jobs are only occupied paths and admission exclusions. assert_open rejects a key absent from eligible reservations. No new primitive or legacy rebase is proposed.
