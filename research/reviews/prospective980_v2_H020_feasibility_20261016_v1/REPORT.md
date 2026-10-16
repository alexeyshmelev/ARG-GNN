# Prospective980 v2, one-key H020 feasibility

**Feasible as a small successor of the existing guard, with its own policy and journal over an exact frozen v1 prefix.** Existing admitted v1 jobs can finish unchanged. This is a design assessment, not implementation or activation.

## Reader and writer invariants

1. Preserve every v1 source, policy, journal event, authority and reservation byte. Existing v1 consumers continue using their original configured source/policy, original scopes and full caps. Their read/check operations and scientific output writes remain allowed.
2. At activation, root freezes the exact v1 _pins(view), projected_bytes and source/policy references under the original mutex. Revoke all future v1 reserve/close mutations cooperatively. Immutable v1 code does not mechanically consult a new policy, so this revocation must cover every authorized accounting writer.
3. Every v2 read calls the unchanged v1 state reader and requires that exact frozen prefix and projection. Any later v1 append or source/policy drift blocks v2. No configured v1 process changes policy or module globals. Current configure_v2 explicitly forbids an in-process policy swap.
4. All new events go only to the distinct v2 journal. V1 readers do not enumerate or replay that journal, so its new legacy proof cannot break their _proof checks. Current budget_check_v9 reads only its own configured v1 state. All old admitted caps remain included in v2's frozen base, covering their remaining writes.
5. V2 preserves the original mutex, owned-scope validation/measurement, no-overlap checks, marker/repair authorities, prefix checks, exclusive append, duplicate-close rejection and permanent revocation requirements. No old journal rebase, source mutation, marker manufacture, deletion or global disk scan.

## Minimal accounting and API

Let B be the authenticated frozen v1 projection, N the new v2 reservations, C certified v2 credits and e the v2 event count. Use B+N-C+1,048,576+524,288e bytes, retaining the existing 950 GB warning, strict 980 GB ceiling and human 1 TB limit. V1 metadata is already inside B. New metadata is charged separately. Activation must itself fit the ceiling before any credit.

Retain the existing configure/inspect, operation draft, reserve, normal completed-close, storage check/validation and guarded writer APIs in the new module. Preserve normal _proof semantics. Add only apply_retire_H020_seed202_v1 with a separate typed proof/event for the exact key biological_mechanism_controls_v1/seed202. No generic legacy allowlist or alternate completion path for other keys is needed.

## Exact H020 settlement requirements

Bind the original 52,321,048,576-byte reservation and digest, its original launch admission and all output/auxiliary paths. Authenticate the original receipt.json at its actual SHA and 10,169,131-byte length, without fabricating completed.json or relaxing the ordinary 1 MiB completion branch. Its pinned producer must establish completed_seven_controls_one_seed and the seven positive completed optimizer fits.

Bind original launch and supervisor source to actual child.wait return 0, matching launcher PID/start 1230670/435395396 and supervisor 1230669/435395390. Preserve any unknown outer-tool status. Root must supply the complete authentic known worker/descendant roster, including worker 1230680, fresh targeted identity/group absence and permanent original write/resume/relaunch revocation. These missing live facts were not established here.

Under the unchanged original v4 mutex/measurement, retain every output and auxiliary byte and credit only 52,321,048,576 minus the actual conservative retained total. Reject over-cap, missing-evidence, already-closed and forbidden-key cases. Preserve every historical artifact. The producer's old 23,372,410,880 accounted-write counter is not that measurement. Funding the pending stage depends on the resulting real projection, not the nominal allowance.

## Integration constraint

Retaining all 18 existing guard bindings plus the new guard/policy adds two unique direct refs. A future closure already at 64 would become 66 and must not be admitted. Resolve through reviewed physical source consolidation or a smaller valid closure, never omitted required sources. Current jobs/admissions remain untouched v1.

Source basis: prospective_980gb_v1.py SHA fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56, especially configure_v2:119, _bindings:160, _state_locked:269 and check_storage_locked_v1:392. Exact H020 evidence is linked in legacy_capacity_actual_evidence_options_20261016_v1/REPORT.md. No code, limits, policy, activation, admission, probe or accounting changes occurred.
