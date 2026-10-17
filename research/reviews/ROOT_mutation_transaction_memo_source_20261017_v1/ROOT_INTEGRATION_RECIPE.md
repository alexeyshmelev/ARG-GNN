# Future fresh admission integration

This inactive source is not independently accepted and authorizes no operation. Use only after a separate SOURCE_PASS and a separately reviewed fresh admission integration. Preserve failed H06049766, every consumed token, key, file, charge and clock; never replay that admission.

Supported: the exact pinned Relate_failed64GiB_retention_v1 guard, original v4 and budget-v9 source in root_mutation_memo_v1.PINS. Only RootMutationMemo.read_state_v1 and reserve_v1 are supported. Both acquire the ORIGINAL actual mutex themselves. No close, retire, caller-held-lock assertion, consumer process, existing admitted job or live patch is supported.

## New source integration

1. In a NEW separately reviewed admission driver/manifest, pin root_mutation_memo_v1.py at its sealed path/SHA. Add that pair to the driver's already DIRECT source_manifest.source_sha256 map. Keep this integration manifest as an existing direct consumer role. This transitive join authenticates the memo without adding a65th role. The adapter requires that same manifest to remain directly bound before append. Preserve all returned current guard roles and original64 ceiling.
2. Authenticate/import this helper only in that new ROOT -B -s, single-thread interpreter, after the explicit legal Relate child is selected/configured and before any consumer memo. No existing consumer memo may be installed for mutations. The helper checks exact original code objects and source hashes and refuses altered/unsupported modules.
3. Build cache_refs from exact SHA refs in the integration's authenticated source closure, filtering to .py/.sh under src, scripts or research. No JSON, accounting/runtime/data/output paths are accepted for byte caching. This list is a source declaration; no new operational configuration or ledger is introduced.
4. Construct once for the new admission:

~~~python
memo = RootMutationMemo(
    g,
    integration_manifest_ref=request["source_manifest_ref"],
    adapter_ref=EXACT_SEALED_MEMO_REF,
    cache_refs=EXACT_AUTHENTICATED_CODE_REFS,
)
~~~

The integration manifest must contain source_sha256[EXACT_SEALED_MEMO_REF["path"]] equal to its SHA. Authenticate the helper and accepted independent review through the new driver's sealed closure before constructing it.

5. Replace only the admission's existing read-only critical section:

~~~python
# Replaces: with g.budget.v4.mutex(ROOT): state = g._state_locked(..., fresh=True)
state = memo.read_state_v1(
    expected_source_sha256=implementation["sha256"],
    expected_launcher_sha256=launcher["sha256"],
)
~~~

Continue the unchanged all-current-role union and original scope_spec preflight outside that completed transaction. Create the same exact prospective reserve authority, binding the actual returned prefix.

6. Replace only the existing g.apply_reserve_v1 call:

~~~python
receipt = memo.reserve_v1(
    authority_ref,
    expected_source_sha256=implementation["sha256"],
    expected_launcher_sha256=launcher["sha256"],
)
~~~

Do not wrap either call in another scheduler mutex. They acquire their own exact original mutex. The original pinned reserve body is AST-derived with only its outer with moved to the adapter; schema/authority/prefix/freshness/roles/64cap/overlap/ceiling/append/receipt code is retained. There is no new accounting successor or journal.

Each call starts a new cache AFTER lock acquisition. No cache survives between state and reserve. Only pinned source bytes and canonical results reached through cloned reference/scope_spec/successor_paths are reused. Schema predicates still execute each time. The original canonical, file_bytes and mutex objects remain unchanged, including mutation identity checks.

Before state returns or the original reserve append runs, cached unique sources and namespace paths receive original UNCACHED validation. The source helper and transitive integration manifest are also authenticated uncached. Caching is disabled before original append, fsync and post-write reads. All replacements are restored and cache bytes/paths cleared before original unlock on success or error. Stats are available in memo.counts, in memory only; no new receipt/process is required.

## Limits and later retained-credit work

Only .py/.sh byte authentication is cached. JSON policies, authorities, waits and proofs remain uncached. Journals, prefixes, namespace fingerprints, scope inventories, targeted absence and scientific/stat measurements are not cached. There is no lifetime/cross-operation cache, state forecast, skipped role, raised ceiling or fit claim. Real speedup and whether remaining uncached work fits any deadline are unmeasured.

V1 does not expose a caller-held-lock API: accepting a caller's lock assertion would not establish actual ownership. A future typed context could expose state only while an adapter-minted original mutex context is live, under a separate review.

V1 does not support apply_retire_stopped_v1. A later narrow adapter could preserve its exact pre-lock started/deadline/finite preamble and exact critical body without a new accounting lineage. It must leave actual proof/measurement work uncached and call the original finite() AFTER added uncached revalidation immediately before append. Existing finite() runs before _append and cannot by itself cover extra revalidation. Preserve300s including mutex wait; do not reset or extend it, bypass a timeout or claim this reserve memo resolves retirement stalls. Do not queue that operation behind another ROOT mutation.

Source-only check:

~~~sh
/usr/bin/python3 -B -s research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/source_check_v1.py
~~~

It compiles source and exercises actual v4 predicate code against handwritten in-memory readers. It imports no project module and calls no real guard/mutex or accounting operation.
