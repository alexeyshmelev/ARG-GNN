"""Root-only reserve admission with existing certified completed-scope credits, using original V9 journal."""
from pathlib import Path
import copy
from scripts import project_budget_v9 as budget
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/closed_scopes_admission_v1_root/admission.py"
def apply(root,authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    root=Path(root)
    with budget.v4.mutex(root):
        c,bindings=budget.reviewed(root,expected_source_sha256,expected_launcher_sha256)
        authority=budget.check_authority(root,authority_ref,"reserve",bindings)
        if (authority.get("accounting_mode")!="journal_closed_scopes_credit_v1"
                or authority.get("actual_reserve_helper")!=budget.v4.reference(root,SOURCE)):
            raise ValueError("Exact root full-reservation helper authority required")
        review=budget.v4.bound(root,authority["full_reservation_helper_review"],True)
        if review.get("verdict")!="PASS" or review.get("source")!=budget.v4.reference(root,SOURCE):
            raise ValueError("Exact independent reserve-helper PASS required")
        legacy,old,new,closed,before,events=budget.state(root,c,bindings,verify_closed=False)
        if authority.get("expected_before")!=budget.pins(before):
            raise ValueError("Exact fresh historical ledgers/journal pins required")
        request=authority["reservation"]
        adapter=budget.v4.reference(root,"src/arggnn/project_storage_closed_scopes_v1.py")
        if adapter not in request["consumer_bindings"]:
            raise ValueError("Only pinned closed-credit consumers can use this admission")
        paths=budget.successor_paths(root,request)
        occupied=[budget.v4.canonical(root,r["output"]) for r in legacy["jobs"].values()]
        for reservation in [*old["reservations"].values(),*new.values()]:
            occupied += budget.successor_paths(root,reservation,True) if reservation["key"] in new else budget.v4.scope_spec(root,reservation,True)
        if (len(old["reservations"])+len(new)>=4096
                or request["key"] in old["reservations"] or request["key"] in new or request["key"] in legacy["jobs"]
                or any(p.exists() for p in paths)
                or any(budget.v4.paths_overlap(a,b) for a in paths for b in occupied)):
            raise ValueError("New successor key/scopes must be fresh and exclusive")
        value={**copy.deepcopy(request),"root_admission":copy.deepcopy(authority_ref)}
        projected_new={**new,request["key"]:value}
        base={**before,"projected_bytes":before["v4_projected_bytes"]}
        after=budget.reconcile(base,old["reservations"],projected_new,closed)
        full_charge=after["projected_bytes"]
        if after["hard_limit_bytes"]!=1000000000000 or full_charge>=after["hard_limit_bytes"]:
            raise ValueError("Certified project reservation charge reaches human1TB limit")
        budget.reviewed(root,expected_source_sha256,expected_launcher_sha256)
        budget.check_authority(root,authority_ref,"reserve",bindings)
        event={"schema":"project_budget_v9_event","root":str(root),
            "sequence":before["journal_events"]+1,"previous_sha256":before["journal_tail_sha256"],
            "bindings":bindings,"operation":"reserve","authority":copy.deepcopy(authority_ref),"value":value}
        sha=budget.append_event(root,event,before,c,bindings)
        return {"kind":"closed_scopes_root_apply_receipt","operation":"reserve","before":before,
            "projected_bytes":after["projected_bytes"],"credited_project_charge_bytes":full_charge,
            "closed_credit_applied_to_full_charge":after["closed_allowance_credit_bytes"],"journal_tail_sha256":sha,"value":value,
            "historical_records_mutated":False,"global_process_scan_performed":False,
            "GPU_used":False,"scientific_execution_performed":False}
