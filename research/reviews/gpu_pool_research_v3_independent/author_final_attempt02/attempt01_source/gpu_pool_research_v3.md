# Source-only GPU pool research extension v3

This separately versioned repair selects bootstrap v2, whose only behavioral change supports the two original hash-pinned metadata manifests through64MiB while preserving1MiB source/permit/review bounds. All existing pool v1/v2 sources, configurations and reviews stay frozen. No existing controller is modified.

This separate CLI uses the unchanged `arggnn.gpu_pool_v1.Controller` with an explicit fixed `modules=` argument. It retains the six original worker modules and adds only the sealed compact qualification runner and its separate CUDA bootstrap. It never rebinds the base module's globals. Direct runner submission through this CLI rejects. Its unchanged runner also requires an already qualified same-PID context, so a direct base IPC runner request cannot provide the required bootstrap.

The configuration is permanently source-only and grants no execution. Before start or submit the CLI verifies its exact configuration hash, the unchanged base controller and old launcher/guard hashes, the runner source/configuration/independent source-only PASS, and the bootstrap source/configuration. The bootstrap separately requires independent reviews of its exact bytes and this exact extension under a fresh root operational permit before creating a CUDA context.

All original scheduling behavior remains in the sealed base: canonical storage paths and shared scheduler/device locks, complete eligible inventory, the maximum of two physical devices, inherited identical lease FD, PID/start-tick/session/PGID ownership, descendant drain, native titles, device UUID and PID memory attribution, peak reservations, FIFO/reject policy, bounded logs, persistent storage arithmetic and source/output rechecks before Popen. The controller creates no CUDA context. Registry schema, stable `gpu_pool_v1:<pool>:gpu<physical>` lease job and Unix socket protocol remain v1.

Submit calls the original `pool.request` with the same exact `{v:1,op:"submit",spec:...}` object. Status, wait and shutdown delegate to the original CLI, including disk fallback and natural drain. Those maintenance commands remain available if a source pin changes. Existing active controllers keep their original allowlists, and a new CLI cannot extend them.

## Future root command shapes

These shapes are documentation only. This source stage starts no controller or worker, creates no live permit, reads no actual bank, changes no ledger or Git state, and preserves completed outputs, including seed202. Root must obtain independent extension/bootstrap PASS receipts, satisfy the frozen natural-drain prerequisite, verify the replaced controller and surviving lease holders are gone, and authoritatively prove GPU2 free before starting a fresh pool. An active GPU3 pool counts toward the two-device ceiling. This extension is not a third-device authorization.

After sourcing `scripts/project_env.sh`, a future root-owned foreground controller command has the same resource fields as v1:

```text
.runtime/env/bin/python -B scripts/gpu_pool_research_v3.py start \
  --pool ROOT_FRESH_POOL --gpu 2 --headroom-mib 8192 \
  --storage-snapshot ROOT_REVIEWED_STORAGE_SNAPSHOT \
  --storage-snapshot-sha256 ROOT_SNAPSHOT_SHA256 \
  --external-storage-reserve-bytes ROOT_ACCOUNTED_EXTERNAL_RESERVE_BYTES
```

The persistent storage ledger binding must match the audited snapshot/external reserve/ceiling. The extension performs no rebase or reconciliation. Root must maintain fresh whole-repository accounting and account for every current writer, ignored file and environment before activation.

The scientific output must be absent with an existing canonical parent at submission. Root issues a fresh permit for that exact output, logical job, controller identity and inherited lease. The worker creates its owned directory only after the base scheduler's final fresh-output/source check:

```text
.runtime/env/bin/python -B scripts/gpu_pool_research_v3.py submit \
  --pool ROOT_FRESH_POOL --job ROOT_DISTINCT_JOB \
  --model H020_compact_qualification --seed 101 \
  --output ROOT_FRESH_OUTPUT_SCOPE --peak-mib ROOT_REVIEWED_PEAK_MIB \
  --storage-bytes 1001048576 --log-limit-bytes 1048576 --policy reject \
  arggnn.anchor_compact_cuda_bootstrap_v2 \
  --config configs/anchor_compact_cuda_bootstrap_v2.json \
  --config-sha256 bf94468e5b8d613e027c1d46fbb7991a672870d5a6f0159ec9511af1d8285dac \
  --permit ROOT_FRESH_BOOTSTRAP_PERMIT --permit-sha256 ROOT_PERMIT_SHA256 \
  --output ROOT_FRESH_OUTPUT_SCOPE
```

The 1,001,048,576-byte base job reservation covers exactly a 1GB owned runner output/cache/log scope plus the base's bounded 1MiB pool log. Root's current reservation inventory splits those scopes without counting the same bytes twice. Controller overhead and other outstanding writers remain separately accounted. The bootstrap redirects same-PID stdout/stderr to `output/logs/worker.log`, places launch evidence under `output/.launch`, and redirects every frozen runner cache/temp/config variable inside the owned 1GB scope.

```text
.runtime/env/bin/python -B scripts/gpu_pool_research_v3.py status --pool ROOT_FRESH_POOL
.runtime/env/bin/python -B scripts/gpu_pool_research_v3.py wait --pool ROOT_FRESH_POOL --job ROOT_DISTINCT_JOB
.runtime/env/bin/python -B scripts/gpu_pool_research_v3.py shutdown --pool ROOT_FRESH_POOL
```

Bootstrap receipt issuance is described in [its contract](anchor_compact_cuda_bootstrap_v2.md). The old `scripts/launch_gpu.py` and `native_argformer_exec.py` remain byte-identical, so the frozen adapter source map requires no reconciliation change. The additional bootstrap/permit/extension pins live in separately bound launcher evidence. Scientific and numerical protocols remain unchanged.

The focused CPU fixtures exercise exact source/config pins, explicit constructor injection, literal IPC shape, direct/arbitrary-module rejection, protocol and job-field types, canonical output paths, and the inherited base source/output check for the added bootstrap. They start no live controller, acquire no device lock and launch no worker. Routine scheduler tests and scientific qualification remain covered by their separate existing reviews.

The independent pool receipt is a new artifact at research/reviews/gpu_pool_research_v3_independent/review_receipt.json. It must bind this v3 launcher/config and the exact v2 bootstrap/config with the same required hash field names. The author source/diff audit proves the launcher identical to v2 after normalizing version identities and configuration hashes. The author tests passed36 fixtures; no actual metadata, failed worker scope or live scheduling state was accessed.
