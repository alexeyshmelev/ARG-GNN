"""Marker-repaired successor for the one failed H039 10GB allowance.

Preserves original journals, failed artifacts and pinned sources. Import is inert.
Configure only after root's complete terminal barrier. No scope deletion.
""" 
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"
REPAIR_SOURCE="research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"
REPAIR_SHA="e4185dc51b285d8ad4eba224b8e1e15f3e1479f57c73b4ec53fff2b238d0ef2f"
ORIGINAL_SOURCE="research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"
ORIGINAL_SHA="7c0a35f82423eb824edf3b7dccbac31f139d50d2ee397d4d92515b6a0ae33c88"
_CONFIGURED=None
_REPAIR_REF=None

def configure_v2(repair_authority_ref):
    global _CONFIGURED,_REPAIR_REF,budget,MODE,JOURNAL
    from arggnn import project_storage_terminal_retired_scopes_v1 as pinned
    for path,sha in ((REPAIR_SOURCE,REPAIR_SHA),(ORIGINAL_SOURCE,ORIGINAL_SHA)):
        pinned.budget.v4.bound(ROOT,dict(path=path,sha256=sha))
    def load(name,path):
        s=importlib.util.spec_from_file_location(name,ROOT/path)
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    repair=load("_H039_capacity_v2_marker_repair",REPAIR_SOURCE)
    repair.configure_repair_v1(repair_authority_ref)
    original=load("_H039_capacity_v2_original",ORIGINAL_SOURCE)
    cpu,retired=repair._readers()
    def frozen():
        pinned.budget.v4.bound(ROOT,dict(path=pinned.SOURCE,sha256=original.PREVIOUS_SHA256))
        return cpu
    ns=repair._namespace(original,previous=retired,old=retired.old,SOURCE=SOURCE,
        _paths=repair._paths,_frozen=frozen,_CACHE=None)
    original_authority=ns["_authority"]
    def authority(ref,operation,bindings,*,fresh):
        value=original_authority(ref,operation,bindings,fresh=fresh)
        if fresh:
            repair._repair()
            if operation=="reserve":
                needed=[dict(path=REPAIR_SOURCE,sha256=REPAIR_SHA),repair_authority_ref,
                    dict(path=ORIGINAL_SOURCE,sha256=ORIGINAL_SHA),repair.MARKER_REF]
                if any(r not in value["reservation"]["consumer_bindings"] for r in needed):
                    raise ValueError("Actual successor, marker repair and original source bindings required")
        return value
    ns["_authority"]=authority
    original_check=ns["check_storage_locked_v1"]
    def checked(storage,*,first=False,boundary=False,required_write_bytes=0):
        repair._repair()
        if storage["reservation"].get("key")==repair.KEY:
            raise ValueError("Malformed historical scope remains fully charged and revoked")
        needed=[dict(path=REPAIR_SOURCE,sha256=REPAIR_SHA),repair_authority_ref,
            dict(path=ORIGINAL_SOURCE,sha256=ORIGINAL_SHA),repair.MARKER_REF]
        if any(r not in storage["reservation"]["consumer_bindings"] for r in needed):
            raise ValueError("Exact fresh marker-repaired capacity bindings required")
        return original_check(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)
    ns["check_storage_locked_v1"]=checked
    _CONFIGURED=ns;_REPAIR_REF=repair_authority_ref
    budget=repair.budget;MODE=ns["MODE"];JOURNAL=ns["JOURNAL"]
    global PilotOutputV9
    class PilotOutputV9(repair.PilotOutputV9):
        __init__=pinned._clone(repair.PilotOutputV9.__init__,budget_check_v9=budget_check_v9)
        save=pinned._clone(repair.PilotOutputV9.save,budget_check_v9=budget_check_v9)
        discard=pinned._clone(repair.PilotOutputV9.discard,budget_check_v9=budget_check_v9)

def _call(name,*args,**kwargs):
    if _CONFIGURED is None:raise ValueError("Root must configure marker-repaired capacity successor first")
    return _CONFIGURED[name](*args,**kwargs)

def _state_locked(*args,**kwargs):
    kwargs["extra"]=set(kwargs.get("extra",()))|{SOURCE,REPAIR_SOURCE,ORIGINAL_SOURCE,_REPAIR_REF["path"]}
    return _call("_state_locked",*args,**kwargs)

def inspect_v1(*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        return _state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)[2]

def _pins(view):return _call("_pins",view)
def retirement_request_v1(observation_ref):return _call("retirement_request_v1",observation_ref)
def apply_retire_failed_H039_v1(*args,**kwargs):return _call("apply_retire_failed_H039_v1",*args,**kwargs)
def apply_reserve_v1(*args,**kwargs):return _call("apply_reserve_v1",*args,**kwargs)
def check_storage_locked_v1(*args,**kwargs):return _call("check_storage_locked_v1",*args,**kwargs)
def budget_check_v9(*args,**kwargs):return _call("budget_check_v9",*args,**kwargs)
def validate_storage_v9(*args,**kwargs):return _call("validate_storage_v9",*args,**kwargs)
