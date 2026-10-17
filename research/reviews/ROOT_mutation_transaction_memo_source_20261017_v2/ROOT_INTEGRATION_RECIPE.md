# Future fresh integration recipe — memo v2 candidate

This source is inactive and not independently accepted. Preserve v1 and every live/finished attempt, old key/token/charge/file/clock. No replay, caller-held-lock mutation, old interpreter patch, automatic retry or fit claim.

## New admission integration

1. After independent v2 SOURCE_PASS, use a NEW separately sealed/reviewed admission integration. Transitively pin this exact module plus its source manifest/handoff and accepted component review through the integration's already-DIRECT source manifest. Do not add a direct memo role or omit any guard; original64cap remains.
2. Authenticate helper/review before construction, after exact current Relate child configuration and before first staging put. Fresh single-thread ROOT -B -s nonoptimized interpreter only. Static cache_refs remain exact authenticated .py/.sh refs under src/scripts/research; dynamic metadata does not require adding direct roles.
3. RootMutationMemo signature remains:
~~~python
memo = RootMutationMemo(
    g,
    integration_manifest_ref=request["source_manifest_ref"],
    adapter_ref=EXACT_V2_SOURCE_REF,
    cache_refs=EXACT_AUTHENTICATED_CODE_REFS,
)
~~~
Only read_state_v1(expected_source_sha256=...,expected_launcher_sha256=...) and reserve_v1(authority_ref,expected_source_sha256=...,expected_launcher_sha256=...) are public. They acquire the ORIGINAL actual mutex themselves. Replace only original initial mutex/state and original reserve callable, plus constructor before first put; no outer lock, changed schema or reordered staging.
4. Exact original bound(ref) validates reference/SHA and parses every call. Eligible dynamic explicit-SHA .py/.sh/.json raw bytes enrol only after original initial bound/parse success. Every unique cache entry receives original uncached content/path validation before return/append; conflicting expected SHA fails. No JSON rawread/reference without expected SHA is cached. Returned JSON objects and all predicates are always recomputed.
5. Each transaction is cold. No replay state, raw journal/prefix/fingerprint, inventory/stat/process/measurement or numerical payload cache. All .runtime paths, filenames containing journal/prefix/fingerprint/inventory, and non .py/.sh/.json suffixes bypass bytecache. Original canonical/file_bytes/mutex identities remain unchanged.
6. Public reserve excludes authority_ref.path automatically. It reauthenticates the already-direct integration manifest and exact helper uncached before final validation; original append/fsync/postwrite reads run with caching disabled. All patches/cache bytes/names/exclusions disappear before original unlock.

## Narrow separately reviewed retirement helper

This module still exposes NO public retirement API. The private core remains:
~~~python
tx = _Transaction(
    root, v4, budget, allow,
    max_entries=2048, max_bytes=64*2**20,
    uncached_paths=tuple(CURRENT_OPERATION_PATHS),
)
~~~
allow is the existing static path→SHA code map. uncached_paths accepts contained relative or absolute path strings, at most2048. install(), finish(), close() protocol is unchanged. Construct/install only AFTER actual original mutex acquisition; always close before unlock.

Keep original retirement started/deadline/finite preamble before lock and original300s including mutex wait. Inside original lock, a future helper may read exact authority through ORIGINAL bound to identify current request refs before install; preserve original _authority validation later. Exclude current authority, revocation, observation/proof refs and all out-of-band helper instruction/review/manifest paths. Current computed target proof objects are never cached. Historical immutable exact-SHA guard/config/activation/completed metadata BYTES may cache; original parse/schema/policy/time/proof predicates rerun.

Actual PID absence, drain/identity checks, inventory, two-pass stat, final census/measurement, raw journals/prefixes/fingerprints and revocation observations remain original uncached. Before append: authenticate current ROOT instruction/review/manifest/helper/memo uncached; tx.finish(); call ORIGINAL finite(); then original append. Keep caching disabled through append. Finally tx.close() before original unlock. Never extend/reset deadline, use a fake mutex assertion, queue behind another mutation or claim300s fit from source checks.

## Source checks

~~~sh
/usr/bin/python3 -B -s research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2/source_check_v1.py
~~~

Only stdlib memo definitions and isolated actual AST predicates run with handwritten in-memory readers/mutexes. No real guard, journal, state, resource, process, payload, scientific or Git operation. Synthetic actual-bound/source-reader reduction is evidence about repeated byte authentication only; actual runtime speedup remains unmeasured.
