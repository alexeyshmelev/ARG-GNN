# Validation-cache addendum

Exact reviewed source: project_storage_closed_scopes_v1.py SHA256 **974bcf342965abb7e7e062547c68123de9bb9414bf9b1f97fd962d1488d3fdd7**. This supersedes the cache-free b168b8e1 review for new admissions. The unchanged full-reservation adapter and V9 core remain unchanged.

**Qualified verdict:** no blocking finding within the explicit immutable-journal/authority/dependency and cooperative no-future-writes contract. The code cache is not a fresh physical audit of other scopes.

## Exact cache behavior

The cache is process-local and keyed by the full reservation digest. Inside the existing canonical scheduler mutex, it fingerprints device/inode/type/link-count/size/mtime/ctime of the fixed legacy and supplemental ledgers, journal directory, V9 implementation/launcher, V4 config and every pinned consumer path. Duplicate paths are removed without scanning a tree or enumerating global processes.

The first call or any observed fingerprint change runs the unchanged check_reservation_locked(boundary=False, required_write_bytes=0). A before/after fingerprint mismatch rejects and does not admit the new cache entry. A hit returns the previously validated journal result with cached_journal_validation=True.

The adapter still checks its mode/active status/expiry, exact implementation/launcher/adapter byte pins, component membership and output/cap binding before lookup. The source callsite validator, output writer and discard rules are AST-identical to the original. For each requested boundary write, successor_paths() and the owned scope measurement run again, followed by the original measured-bytes-plus-planned-bytes cap rejection. The existing successor_paths()/scope_spec() also retains bound own-consumer metadata checks. A cached history result does not cache active output sizes.

## One fake metadata fixture

fake_cache_metadata_check.py uses the real adapter functions, with fake stat paths, fake mutex/core and fake active-scope measurement. No actual ledger/journal, scope, disk, descriptor or process inspection occurs.

Observed sequence:

- First validation: one core replay and one owned-scope measurement.
- Same metadata: cache hit, still one core replay and two owned-scope measurements.
- Pinned consumer-file metadata change: second core replay.
- Journal-directory metadata change: third replay, refreshing fake charge900→910GB.
- Fixed-ledger metadata change: fourth replay.
- On a warm cache, a write one byte over the original4GB active cap rejects and still remeasures the active scope.
- Missing launcher membership and changed adapter byte pin reject before lookup.
- A core rejection after a miss propagates.
- Fingerprint change during a core replay rejects.

All fingerprint reads in the fixture occur under the fake canonical mutex. Storage/consumer input metadata is unmodified. The fixture values are invented test metadata, **not actual project accounting**. The receipt reports all checks passed.

Reproduction:

~~~bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$ARGGNN_ROOT/src:$ARGGNN_ROOT" \
  PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python \
  research/reviews/project_storage_closed_scopes_v1_review_author/fake_cache_metadata_check.py
~~~

## Contract boundary

A journal-directory stat detects ordinary append/remove/rename namespace changes. It does **not** fingerprint the contents of existing event files. Likewise, historical event authority files and other immutable predecessor dependencies not present among the listed fingerprints are not newly inspected on a hit. Their certification/immutability is the stated root contract. Unauthorized in-place mutation of an existing event/authority while the directory metadata stays unchanged is outside this cache's detection claim. The original cache-free consumer invoked the core each time; this cache intentionally relies more strongly on that immutability between misses.

The canonical mutex coordinates cooperative writers; it does not prove absence of external NFS writers. Existing boundary=False/verify_closed=False semantics continue to rely on certified closure and future-write revocation rather than global descriptor/stat revalidation. No throughput improvement is measured here. Root should bind this exact source digest and the review receipt before H038 admission. No producer, scientific model, checkpoint, journal or root reservation was modified by this review.
