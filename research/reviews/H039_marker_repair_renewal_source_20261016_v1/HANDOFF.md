# Additive marker-repair expiry renewal for fresh interpreters

Source only. No root certificate was activated, no hook installed, and no guard,
probe, reserve, close, wait, signal, process, job, journal, or old file was changed.

The original repair authority expired at2026-10-15T21:36:04.459612+00:00.
The old source/authority/barrier and full15,000,000-byte malformed reservation
remain byte-for-byte pinned. A separate root certificate may extend only the
read-only repair-authority expiry view. Allocation/storage expiries remain
unchanged. No accounting credit, rebase, source/binding/history mutation, or
malformed-key future write/relaunch is authorized.

The frozen guards dynamically load private marker modules. The new interpreter
bootstrap routes SourceFileLoader.exec_module only when the exact old marker
path/hash is loaded, then wraps that private module's _bound. All other loads and
refs delegate unchanged. Only an exact read of the old authority returns a deep
copy with the separately certified expiry. Every such read revalidates current
certificate bytes, root scope, exact old authority/source/barrier and new source/
bootstrap pins, and real UTC time. The original _repair still executes its full
source/barrier/original-reservation/original-authority proof every time. No clock
or original proof method is replaced. Existing loaded consumers are never patched.

Root activation:
1. Independently review this source and its pool integration delta.
2. Create a NEW active certificate from renewal_certificate.proposed.inactive.json.
   Set active=true, root_authorizer=root_agent, genuine current UTC issued_at_utc,
   and explicit renewed_expires_at_utc (at most7days after issue). Do not overwrite
   the inactive proposal or old authority. Hash the new certificate.
3. For pool use, the shared route must bind that certificate and its exact source/
   bootstrap. Route expiry +600seconds must be strictly earlier than renewal
   expiry so original guard writes remain authorized throughout owned drain.
4. Future root admission installs the hook before loading original validators/
   guards. Every worker/controller reserve directly binds renewal source,
   bootstrap, and certificate in addition to actual pool/launcher/base/shim.
5. Worker argv uses the pinned bootstrap; it installs renewal before canonical
   pool/scientific imports. load_cap also installs idempotently for controller/
   worker fresh interpreters. No current process is patched.

Root can use the renewal API in a separately authorized NEW interpreter before
importing/configuring an original guard. Load marker_repair_renewal_v1.py by an
exact SHA-bound file spec, then call install_renewal_v1({path,sha256}) with the
fresh certificate. That API validates metadata and installs only this exact
source hook. It does not call configure/read/apply on a guard.

The old C3 migration gate remains pending: if the expired old guard prevents its
full terminal roster receipt after a genuine minimal wait and own drain, root
must prepare separately labelled successor evidence from actual tool/minimal wait,
exact failure control flow, and a fresh narrowly observed own roster/drain
certificate. This renewal does not invent that missing receipt or alter its gate.

Pure fixtures use fake loader/module/data objects only.17 renewal cases and41
pool lifecycle cases pass. The separate cache-C3 progress observer has8 stub cases:
one unchanged original helper call/arguments/return/exception; events at1/every10
successful whole-parent updates; logging errors cannot alter scientific returns.
No model, optimizer, parameter, RNG, Torch-method, or checkpoint state is read.

The pre-renewal pool lifecycle PASS is separately sealed at
research/reviews/GPU45_C3cache_H050_pool_independent_review_20261016_v1/
source_handoff.v1.json SHA1a6c875a3e1a569d4fc46fb7dcebc79f54871ddf62483c4a7af8bed4adbaf24e.
It does not cover this renewal/observer delta until reviewed independently.
