"""Explicit immutable policy choice for future reviewed ROOT adapters only. Import inert."""
from pathlib import Path
import copy,importlib.util,json
import sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
from scripts import project_budget_v9 as budget
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/explicit_policy_selector_v1.py"
APPROVED_PATHS=("research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py",
    "research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py")
_SELECTED_REF=_SELECTED_VALUE=_GUARD=None
def require(ok,message):
    if not ok:raise ValueError(message)
def _read(record):
    raw=budget.v4.bound(ROOT,record);require(len(raw)<=65536,"Bounded explicit selector/policy/review metadata")
    return budget.v4.parse(raw)
def selector_draft_v1(*,source_ref,policy_ref,source_manifest_ref,measurement_source_ref,source_review_ref,
        repair_authority_ref,accounting_mode,journal,frozen_parent_lineage):
    return dict(schema="explicit_ROOT_storage_policy_selector_v1",root_authorizer="root_agent",active=False,
        selector_source_ref=budget.v4.reference(ROOT,SOURCE),source_ref=source_ref,policy_ref=policy_ref,
        source_manifest_ref=source_manifest_ref,measurement_source_ref=measurement_source_ref,source_review_ref=source_review_ref,
        repair_authority_ref=repair_authority_ref,accounting_mode=accounting_mode,journal=journal,
        frozen_parent_lineage=copy.deepcopy(frozen_parent_lineage),scheduling_ceiling_bytes=980000000000,
        human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000,
        no_fallback_discovery_upgrade_or_in_run_swap=True,future_ROOT_adapters_only=True,
        existing_adapters_runs_and_storage_records_unchanged=True)
def load_selected_guard_v1(selector_ref):
    global _SELECTED_REF,_SELECTED_VALUE,_GUARD
    require(isinstance(selector_ref,dict) and set(selector_ref)=={"path","sha256"},"Explicit path and SHA; no latest/env/discovery")
    value=_read(selector_ref)
    expected=selector_draft_v1(**{k:value.get(k) for k in ("source_ref","policy_ref","source_manifest_ref",
        "measurement_source_ref","source_review_ref","repair_authority_ref","accounting_mode","journal","frozen_parent_lineage")})
    expected["active"]=True;require(value==expected and set(value)==set(expected),"Exact immutable ROOT selector")
    require(value["source_ref"]["path"] in APPROVED_PATHS,"Only reviewed current native14v3 or exact two-key successor")
    for k in ("selector_source_ref","source_ref","source_manifest_ref","measurement_source_ref","repair_authority_ref"):
        budget.v4.bound(ROOT,value[k])
    policy=_read(value["policy_ref"]);review=_read(value["source_review_ref"])
    require(policy.get("active") is True and policy.get("root_authorizer")=="root_agent"
        and policy["source_ref"]==value["source_ref"] and policy["source_manifest_ref"]==value["source_manifest_ref"]
        and policy["measurement_source_ref"]==value["measurement_source_ref"] and policy["source_review_ref"]==value["source_review_ref"]
        and policy["marker_repair_authority_ref"]==value["repair_authority_ref"]
        and policy["journal"]==value["journal"] and policy["scheduling_ceiling_bytes"]==980000000000
        and policy["human_limit_bytes"]==1000000000000 and policy["warning_threshold_bytes"]==950000000000,
        "Selector matches exact active source/policy/guard/review/ceiling")
    lineage=dict(base_ref=policy["base_ref"],base_policy_ref=policy["base_policy_ref"],
        base_pins=policy["base_pins"],base_projected_bytes=policy["base_projected_bytes"],base_accounting_mode=policy["base_accounting_mode"])
    require(value["frozen_parent_lineage"]==lineage and review.get("verdict") in ("PASS","SOURCE_PASS","PASS_SOURCE_ONLY")
        and review.get("source_ref")==value["source_ref"] and review.get("source_manifest_ref")==value["source_manifest_ref"]
        and review.get("measurement_source_ref")==value["measurement_source_ref"],"Exact reviewed frozen-parent lineage")
    require(all(getattr(budget.v4,n).__module__==budget.v4.__name__ for n in ("canonical","file_bytes","mutex")),
        "Select/configure once before consumer memo; never unwrap or reinstall")
    if _SELECTED_REF is not None:
        require(_SELECTED_REF==selector_ref and _SELECTED_VALUE==value,"No lifetime selector/policy swap")
        require(_GUARD.budget is budget and _GUARD.MODE==value["accounting_mode"] and _GUARD.JOURNAL==value["journal"]
            and _GUARD._POLICY_REF==value["policy_ref"],"Exact configured selected module/policy identity retained")
        return _GUARD
    sp=importlib.util.spec_from_file_location("_explicit_ROOT_selected_storage_guard",ROOT/value["source_ref"]["path"])
    guard=importlib.util.module_from_spec(sp);sp.loader.exec_module(guard)
    require(guard.budget is budget and guard.MODE==value["accounting_mode"] and guard.JOURNAL==value["journal"],
        "Exact selected singleton/mode/journal")
    guard.configure_with_renewal_v1(value["repair_authority_ref"],policy_ref=value["policy_ref"])
    _SELECTED_REF=copy.deepcopy(selector_ref);_SELECTED_VALUE=copy.deepcopy(value);_GUARD=guard
    return guard
if __name__=="__main__":print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY_EXPLICIT_SELECTOR",no_selection_or_guard_configuration=True)))
