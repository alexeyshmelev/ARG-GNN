# Supplemental project budget v4

The direct human authorization in research/resource_authorizations/project_storage_1TB_20261011.json permits up to1,000,000,000,000 decimal bytes. New scheduling keeps50GB headroom and requires the combined projection to remain strictly below950,000,000,000 bytes. This companion is source-only until independently reviewed and activated by a separate root reserve admission. It changes no legacy source, config, AGENTS.md or ledger. GPU, runtime, scientific and isolation guards remain in their existing authorities.

## Accounting and root protocol

The existing .runtime/gpup/storage_reservations.json remains at450GB. Every companion check reads its latest invariant-valid state under the existing canonical .runtime/locks/scheduler.lock and adds:

    historical baseline167336640512
    + unchanged external reserve130000000000
    + every latest legacy controller and job reserved_bytes
    + every supplemental reserved_bytes

The supplemental ledger is .runtime/storage/project_budget_v4.json. Its immutable header binds this implementation, configs/project_budget_v4.json, the human authorization and legacy schema/root/snapshot/baseline/external/450GB invariants. The companion never rebases the baseline, borrows the external reserve, lowers active charges or writes the legacy ledger. Existing active legacy work continues to use its450GB guard. New supplemental consumers do not submit the same output to the legacy pool. Scope equality, ancestor/descendant overlap, reused keys and overlap with live accounting/mutex paths fail closed.

The source-only default is:

    . scripts/project_env.sh
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B -m arggnn.project_budget_v4

Root first obtains a fresh independent review with kind project_budget_v4_independent_review, role independent_reviewer, verdict PASS, exact implementation/config/human-authorization references and actual_apply_performed=false. The author receipt supplies no independent verdict.

Root creates an activated copy of research/reviews/project_budget_v4_author/root_reserve_admission_template.json, fills the independent PASS reference, exact fresh legacy/supplemental prehashes, and exact source/runtime consumer metadata references for all four reservations, then hashes that root admission. Missing supplemental ledger is represented by supplemental_ledger_sha256=null only for initial creation. Root alone runs:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B -m arggnn.project_budget_v4 --reserve --expected-source-sha256 <reviewed-source> --expected-config-sha256 <reviewed-config> --admission <root-reserve-admission.json> --expected-admission-sha256 <root-admission-sha256>

The existing mutex is opened without creation, truncation or byte writes. Under it, reserve validates root/independent/human/source/runtime authority, exact initial ledger hashes, fresh exclusive scopes and combined capacity. It atomically creates or appends only the supplemental ledger, preserves all permanent old reservations, verifies the exact written value and rechecks unchanged legacy bytes. The root outer launcher must preserve stdout, stderr and exit evidence before execution, because an atomic replacement or post-write error can leave uncertainty. There is no automatic rollback. The root receipt's admitted_reservations entries are the full immutable values consumers must bind.

No template is active. No live ledger has been created or changed by author work.

## Fit consumer interface

The API in src/arggnn/project_budget_v4.py is:

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


## Tiny immutable combined-map-cap repair

All v2/v3 source/config/docs/author evidence and independent failure evidence remain unchanged. V4 changes only reserve after normalizing versioned path/protocol names. Before constructing or changing a proposal, it rejects len(existing reservations)+len(incoming batch)>4096. The4095+1 case can reach exactly4096 entries. The4096+1 case stops before proposal construction or any ledger write. No thresholds or scope/schema/own-reservation checks are relaxed.

The strict v3 legacy qualification remains unchanged: exact8-field header, jobs exactly{reserved_bytes,output}, controllers exactly{reserved_bytes,pid}, compatible bounded logical names, typed nonnegative integer charges, positive integer controller PID, canonical absolute outputs and all pairwise/accounting-path overlap rejections. Legitimate unrelated additions and settlements remain dynamically charged with exact own reservations and the same read-only API. No controller output field is invented.

The focused author tests add only4095+1 acceptance and4096+1 rejection to the61 preceding guarded tests. They inject an already-validated handmade latest-ledger view to isolate the cap and prove rejection before proposal construction. No actual ledgers, authority, GPU, arrays or scientific work are touched. Independent review and root activation remain pending.
