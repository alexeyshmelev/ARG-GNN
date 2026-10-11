# Supplemental project budget v3

The direct human authorization in research/resource_authorizations/project_storage_1TB_20261011.json permits up to1,000,000,000,000 decimal bytes. New scheduling keeps50GB headroom and requires the combined projection to remain strictly below950,000,000,000 bytes. This companion is source-only until independently reviewed and activated by a separate root reserve admission. It changes no legacy source, config, AGENTS.md or ledger. GPU, runtime, scientific and isolation guards remain in their existing authorities.

## Accounting and root protocol

The existing .runtime/gpup/storage_reservations.json remains at450GB. Every companion check reads its latest invariant-valid state under the existing canonical .runtime/locks/scheduler.lock and adds:

    historical baseline167336640512
    + unchanged external reserve130000000000
    + every latest legacy controller and job reserved_bytes
    + every supplemental reserved_bytes

The supplemental ledger is .runtime/storage/project_budget_v3.json. Its immutable header binds this implementation, configs/project_budget_v3.json, the human authorization and legacy schema/root/snapshot/baseline/external/450GB invariants. The companion never rebases the baseline, borrows the external reserve, lowers active charges or writes the legacy ledger. Existing active legacy work continues to use its450GB guard. New supplemental consumers do not submit the same output to the legacy pool. Scope equality, ancestor/descendant overlap, reused keys and overlap with live accounting/mutex paths fail closed.

The source-only default is:

    . scripts/project_env.sh
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B -m arggnn.project_budget_v3

Root first obtains a fresh independent review with kind project_budget_v3_independent_review, role independent_reviewer, verdict PASS, exact implementation/config/human-authorization references and actual_apply_performed=false. The author receipt supplies no independent verdict.

Root creates an activated copy of research/reviews/project_budget_v3_author/root_reserve_admission_template.json, fills the independent PASS reference, exact fresh legacy/supplemental prehashes, and exact source/runtime consumer metadata references for all four reservations, then hashes that root admission. Missing supplemental ledger is represented by supplemental_ledger_sha256=null only for initial creation. Root alone runs:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B -m arggnn.project_budget_v3 --reserve --expected-source-sha256 <reviewed-source> --expected-config-sha256 <reviewed-config> --admission <root-reserve-admission.json> --expected-admission-sha256 <root-admission-sha256>

The existing mutex is opened without creation, truncation or byte writes. Under it, reserve validates root/independent/human/source/runtime authority, exact initial ledger hashes, fresh exclusive scopes and combined capacity. It atomically creates or appends only the supplemental ledger, preserves all permanent old reservations, verifies the exact written value and rechecks unchanged legacy bytes. The root outer launcher must preserve stdout, stderr and exit evidence before execution, because an atomic replacement or post-write error can leave uncertainty. There is no automatic rollback. The root receipt's admitted_reservations entries are the full immutable values consumers must bind.

No template is active. No live ledger has been created or changed by author work.

## Fit consumer interface

The API in src/arggnn/project_budget_v3.py is:

    check_reservation(
        root,
        reservation=<full exact stored value from admitted_reservations>,
        expected_source_sha256=<companion source SHA256>,
        expected_config_sha256=<companion config SHA256>,
        initial_ledger_pins={
            "legacy_ledger_sha256": <exact first-read SHA256>,
            "supplemental_ledger_sha256": <exact first-read SHA256>,
        },   # optional on later checks
        boundary=True,
        required_write_bytes=0,
    )

It only reads and returns projected_bytes, limit_bytes=950000000000, hard_limit_bytes=1000000000000, legacy_ledger_sha256, supplemental_ledger_sha256, baseline_bytes, external_reserve_bytes, legacy_reserved_bytes, supplemental_reserved_bytes, the full reservation and its SHA256, scope_measurement and remaining_scope_bytes. At the first authorized fit read, root can bind both current ledger hashes. Later checks omit initial_ledger_pins and accept legitimate unrelated ledger additions if every invariant and the entire consumer's own reservation, source/runtime references and original root/independent authority remain unchanged. Both ledger hashes and current projection remain diagnostic evidence at each boundary. Neither ledger is written by this API.

Each native_argformer_tsinfer_fit_v1/<parent_id> reservation covers one fresh output root experiments/results/native_argformer_tsinfer_fit_v1/<parent_id> plus one exclusive short auxiliary directory .runtime/tmp/afts1/<first12 SHA256 of canonical absolute parent output>. Its20,000,000,000 bytes include all probe-memory, fit and retrieve stages, retained checkpoints, embeddings, seals, caches, temporary files and logs. The consumer must bind its separate scientific/runtime/GPU/root execution authority. A companion reserve does not authorize a fit or GPU use.

Boundary=True counts every directory and regular file across both scopes using max(apparent size,allocated bytes), without decoding file content. Directory identity is captured before listing and checked afterward; stat drift, symlinks, multiply linked files, nonregular entries and scan errors fail closed. Missing fresh scopes count zero. The actual sum plus optional required_write_bytes must not exceed the declared charge. Consumers call it before/after stage boundaries and supply conservative planned growth before bulk writes. It is a contractual check at boundaries, not an OS quota or exclusion of unknown writers. Scopes remain exclusively owned, and all legitimate accounting writers must honor scheduler.lock.

Any global apparent/allocated measurement is a separate diagnostic. It never substitutes for the historical baseline or triggers an automatic rebase. Future budget evolution needs a separately reviewed successor. Completed supplemental keys and charges remain permanent in this implementation.

The handmade tests exercise four20GB reservations, dynamic unrelated mutations, unchanged own authority, typed invariant drift, double counting, strict ceiling edges, combined output/short-temp boundaries, planned writes, namespace safety, real mutex contention and atomic failure. Initial harness cleanup denial is preserved. No author test reads actual products/arrays, live ledgers/registries/PIDs or GPU state, or runs installations or scientific work.


## Narrow immutable legacy-schema repair

The v2 independent FAIL and all v2 source/config/docs/author evidence remain unchanged. V3 adds only strict latest-legacy qualification and the shared accounting namespace exclusion. After normalizing versioned protocol/path names, only current and scope_spec have changed existing function ASTs; accounting_paths, legacy_key and legacy_state are the three small new helpers. Caps, human authority, legacy source/snapshot/baseline/external/450GB bindings, atomic supplemental writes, exact own reservation checks, boundary measurements and the consumer API retain the prior kernels.

The latest legacy ledger must contain exactly schema,root,snapshot_sha256,baseline_bytes,external_reserve_bytes,limit_bytes,controllers,jobs. Jobs contain exactly reserved_bytes and output. Controllers contain exactly reserved_bytes and pid, with positive integer pid; there is no invented controller output field. Charges are exact nonnegative integers as in the pinned pool source. Logical names are bounded compatible ASCII pool/job components, and output strings must be canonical and absolute. Every legacy output pair is tested for equality or ancestor/descendant overlap. All legacy and supplemental product scopes exclude the legacy ledger, supplemental storage namespace and scheduler mutex. Unknown top-level accounting collections or entry scope fields fail closed instead of being ignored.

Legitimate unrelated additions and completed-scope charge settlements remain accepted after first hash binding when those exact schemas/invariants, disjoint scopes and the consumer's entire own reservation remain unchanged. A legacy schema evolution needs a new separately reviewed companion. No live legacy state, authority or products were accessed by this repair, and no v3 reservation is active.
