# Source-only diagnosis of the saved scheduler-lock observation

The supplied evidence proves that original C3 VAL000 worker PID3729481 held the scheduler flock while sleeping in an NFS RPC wait at 2026-10-16T23:31:47Z. It does **not** prove that it was reading or hashing scientific arrays inside that lock. No such source path was found in the traced original worker and its original native14v3 accounting chain. The exact metadata operation causing that particular RPC wait remains unknown.

The two saved genotype FDs both point to the same365,563,328-byte inode, with position equal to file size. That is compatible with the two retained read-only NumPy mmaps used by the forward and reverse datasets. The installed NumPy memmap implementation seeks to EOF at lines241-243, constructs mmap from that descriptor at291 and retains it at295. This source explanation is not a runtime stack trace or proof of FD provenance.

## Actual call paths

Scientific authentication occurs in AmericanC3WholeTreeDatasetV1.__getitem__ -> load_observation -> american_c3_common_v1.bound -> ref, which hashes the full genotype file in8MiB blocks, then returns to np.load(...,mmap_mode="r"). The dataset keeps this state. This path has no accounting mutex. Original zero-worker inference holds one forward and one reverse dataset. The outer worker clears their states only after the complete prediction seal returns.

At the end of inference, seal_prediction_v1 calls the supplied writer for predictions and the seal. Worker Writer.save -> original native14v3 PilotOutputV9.save -> budget_check_v9 -> with budget.v4.mutex(ROOT) -> check_storage_locked_v1. The source serializes before taking that guard and creates/writes its new product after the guard returns. The original fast per-block callback only checks authority expiry, time, RSS and zero CUDA. It does not call accounting.

The held accounting transaction reauthenticates sources, policies, authorities and journal evidence, recursively reconstructs accounting state, validates scope namespaces, rehashes direct consumer metadata and performs a stat-only owned-scope boundary inventory. Its generic v4.file_bytes rejects a file larger than32MiB before opening it. Its consumer bindings allow only .py/.sh/.json and retain the original64ref cap. The365,563,328-byte genotype cannot be content-read through that API. measure_scopes and dependency fingerprints use stat operations rather than payload reads.

The traced chain includes native14v3, Census-fit2 retention, H020v3, prospective980v1, completed-credit-v3, marker-repaired H039 capacity, retired/success/terminal/CPU overlays and underlying v9/v4 readers. An explicit historical255,488-byte NVTOP binary authentication exists in H020v3, unrelated to scientific arrays. Renewal authentication reads bounded JSON and exact Python sources. None establishes genotype-content I/O under the lock.

## Limits of the saved evidence

rchar=17,677,177,149 and syscr=611,876 are cumulative lifetime counters. read_bytes=28,672 does not attribute I/O to a particular FD or call stack. EOF positions can persist after mmap setup. NFS rpc_wait_bit_killable is compatible with path resolution, lstat, directory enumeration, open/read and filesystem operations. The evidence cannot choose among these, quantify lock-hold duration or demonstrate a metadata-memo speedup.

The earlier23:26 observation shows a different owner, H060v5 admission PID3834511, also in NFS RPC wait, with H061, native and controller waiters. Together these snapshots support shared-lock/NFS contention as an operational risk. They do not prove one continuous hold or one common leaf operation.

## Proposed future-source remedy

Do not move genotype hashing out of a lock on this evidence: the traced hashing already occurs outside it. For separately admitted future read-only consumers, the candidate is the existing pinned transaction-local metadata/namespace memo adapter. Install it only in a fresh consumer interpreter before guard use and consumer threads. Preserve original guard calls, policy, journal replay logic, scope inventory, lock path and identity checks. Root admission/close/retire processes remain unwrapped.

The candidate caches raw reads and selected canonical resolutions only during one held original mutex transaction, with256MiB and16,384entry caps. It performs the original no-follow checked SHA read on the first miss, falls back to the original operation on capacity exhaustion, revalidates canonical namespaces and leaf identities before the original lock exits, then clears its memo. Journal directory listings, fingerprints and scope inventories remain uncached. Individual immutable journal-file byte reads may be memoized within the transaction. Do not describe all journal bytes as uncached.

This can eliminate repeated metadata reads inside one transaction. It cannot eliminate every NFS metadata round trip or repair an existing blocked call. The original C3 acceptance contract installs no such adapter. Existing jobs retain their exact code and are not candidates for live patching, reload, restart or scope migration under this review.

ROOT's next step is to select one concrete fresh consumer successor and obtain a source review of its exact installation and role binding. Required fixtures should establish unchanged original outcomes, content/SHA rejection, namespace replacement rejection, changed leaf rejection before unlock, capacity fallback, no cache across lock operations and no installation in mutation roles. Optional future timing metadata should distinguish acquisition wait from held time and name the guard phase without scientific reads or another scheduler lock. No measured performance claim follows.

## Review boundaries

Only source text and named previously saved metadata records were read. No scientific file was opened or hashed, no guard imported, no process/resource/FD probe performed and no numerical work launched. Only this new diagnosis directory was written. No live source, journal, policy, process, resource, package, scientific payload or Git state changed.

Exact hashes, line excerpts and scope are in source_references.v1.json. Proposed implementation conditions are in proposed_fix_contract.v1.json. This review authorizes neither implementation nor execution.
