# Persistent bounded pool integration design

Status: executable source successor complete. Root must review actual metadata/caps and source handoff before any bind/admission. Persistent lifecycle remains unexecuted.

The separate root contract/driver/worker/shim, copied trainer/hooks, pure JSON registry validation and inactive request/cap templates are written. The installed additive shim has exact staged parity. Frozen old sources are unchanged.

Root chose one persistent2or4-worker pool instead of620 pools. New files persistent_workers_v1.py and persistent_provider_v1.py implement the successor inside this packet. The original provider/core remain frozen.

A single factory owns the lazy pool across20 original observation seals and600 complete-parent updates. Each parent provider remains an original SealedObservationProviderV1 subclass, retains the original seal API and original Dataset/_read source checks. The fixed complete-parent step call and gradient code stay unchanged.

Before each parent session, the manager drains speculative replies, binds exact parent ID/plan ref/actual optional observation seal and a fresh context serial/hash to every worker. Each read request/reply authenticates context serial/hash, exact parent/plan ref, block index and task serial. At most one request per worker remains in flight and demand delivery stays indexed.

At each original provider.release boundary, pending replies are checked/discarded, every worker clears that parent's Dataset/mmap/cache state and acknowledges the exact context. Workers remain alive, with one active parent context at a time. All20 seals finish before TRAIN truth resolution exactly as before.

Final success/failure releases every parent and drains the one persistent pool. Full per-parent release receipts plus actual process PIDs/start identities/exits and global drain are written through the admitted writer. Root outer process-tree RSS/drain supervision remains mandatory. Exhaustive actual owned PID roster remains below the unchanged256 guard limit. Missing/rejected context, worker failure, deadline, changed feature hash, RSS violation or incomplete drain aborts without fallback/retry.

The six caps remain root-bound and late: num_workers2or4, max_block_bytes, worker_rss_cap_bytes, aggregate_rss_cap_bytes, task_timeout_seconds, drain_timeout_seconds. max_block_bytes must cover every frozen original plan and fit the root-bound CPU cache. The exact sources/caps enter the protocol and root transitive closure.

The previously passed tiny Torch/PyG fixture establishes the abstract shared-tensor transport mechanism only. The new persistent/context lifecycle is unexecuted here, as root forbids new jobs/test campaigns. Validation was source/AST/default-inert only.
