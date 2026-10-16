"""Actual registry assembly with public metadata and fake new authority only."""
from pathlib import Path
import importlib,json,types
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v9"
def run():
    c=importlib.import_module(HERE.replace("/",".")+".amendment_contract_v1")
    a=c.read(c.V8_AMENDMENT_REF);qualified=c.read(c.V8_QUALIFICATION_REF)
    c.validate_known_owner_scope_v6(a,qualified,c.read(c.V4_STDOUT_PROJECTION_REF))
    fake=dict(path="SYNTHETIC_ONLY_NEW_AMENDMENT",sha256="0"*64)
    ns=dict(c.entry_known_registry_v9.__globals__,ENTRY_REGISTRY_CACHE_V9=None,
        active_amendment_v1=lambda:({"additional_known_old_ROOT_launcher_source_refs":[]},None,None),
        current_amendment_ref_v1=lambda:fake)
    actual=types.FunctionType(c.entry_known_registry_v9.__code__,ns)
    registry=actual()
    assert len(registry["identities"])==75 and len(registry["process_groups"])==5
    assert dict(pid=3660358,start_ticks=484268347) in registry["identities"]
    assert dict(pid=3659453,start_ticks=None) in registry["identities"]
    assert registry["current_entry_absence_established_by_registry_alone"] is False
    assert registry["unknown_historical_descendants_claimed_listed"] is False
    assert registry["historical_v3_descendant_coverage"]=="UNKNOWN_TRUNCATED64"
    assert registry["new_v8_source_refs"]==[c.V8_TERMINAL_REF,c.V8_WAIT_REF,c.V8_LAUNCH_REF]
    assert actual()==registry
    return dict(schema="native_VAL000_v9_actual_registry_code_fixture",status="PASS_SOFTWARE_ONLY",
        historical_PID_rows=73,new_complete_known_PID_rows=75,known_groups=5,
        v8_worker_start_exact=True,v8_supervisor_start_honestly_null=True,
        unknown_v3_scope_unchanged=True,no_current_absence_inferred=True,
        actual_readonly_process_probes_guard_or_scientific_payload_access=False)
if __name__=="__main__":
    import sys
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit metadata-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))
