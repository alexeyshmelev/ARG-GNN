"""Narrow SHA-bound reader repair for one missing-v9-marker H039 reservation.

Never edits historical journal/source/bindings, inventories a scope, or grants
write/relaunch to the malformed key. Its full15MB allowance remains charged.
Root must finish all old-guard consumers/supervisors before using this reader.
"""
from __future__ import annotations
import copy
from datetime import datetime,timezone
from pathlib import Path
from types import FunctionType,SimpleNamespace
from arggnn import project_storage_terminal_retired_scopes_v1 as previous

budget=previous.budget
ROOT=previous.ROOT
SOURCE='research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py'
MODE=previous.MODE
JOURNAL=previous.JOURNAL
PREVIOUS_REF=dict(path=previous.SOURCE,sha256='bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab')
MARKER_REF=dict(path='src/arggnn/project_budget_v9_consumer.py',
                sha256='c11a5543746899c8a387508e9877ce6ae2bf80fe4d3076c1dbe1cb8f7a43b859')
KEY='H039/directed-clade-exchange-train-stage-seed101-v1'
RESERVATION_SHA='e824b52b0bb9ff26e1b7d81b48cfe03c85ee58d5aab0120601b52f43f9c5acee'
AUTHORITY_REF=dict(path='research/reviews/H039_train_stage_root_v1/admitted_v1/reserve_authority.json',
                   sha256='c89b3bbeba69551185e38401992957ca4215895a0586a9017ee4aedb4bbea1cc')
RECEIPT_REF=dict(path='research/reviews/H039_train_stage_root_v1/admitted_v1/reserve_receipt.json',
                 sha256='de2bd69d5cad70030b4132fa15a1cd5597c0d3f56eb73467dcd297b94b726e23')
_REPAIR_REF=None
_READERS=None
_ORIGINAL_PATHS=None


def _ref(path):return budget.v4.reference(ROOT,path)
def _bound(ref):return budget.v4.bound(ROOT,ref,True)
def _pins(view):return previous._pins(view)


def repair_authority_draft_v1(barrier_ref=None):
    return dict(kind='H039_missing_marker_reader_repair_root_authority_v1',root_authorizer='root_agent',
        active=False,source_ref=_ref(SOURCE),predecessor_ref=PREVIOUS_REF,marker_ref=MARKER_REF,
        reservation_sha256=RESERVATION_SHA,original_authority_ref=AUTHORITY_REF,original_receipt_ref=RECEIPT_REF,
        retained_allowance_bytes=15000000,accounting_credit_bytes=0,historical_journal_mutation_authorized=False,
        historical_source_or_binding_mutation_authorized=False,global_rebase_authorized=False,
        malformed_reservation_future_writes_and_relaunch_revoked=True,
        all_old_guard_consumers_and_supervisors_terminal=True,
        all_new_consumers_and_launchers_use_marker_repair_reader=True,
        all_budget_writers_use_scheduler_mutex=True,activation_barrier_ref=barrier_ref,expires_at_utc=None)


def _repair():
    if _REPAIR_REF is None:raise ValueError('Configure exact root marker-repair authority first')
    value=_bound(_REPAIR_REF);expected=repair_authority_draft_v1(value.get('activation_barrier_ref'))
    expected.update(active=True);expected.pop('expires_at_utc')
    if any(budget.v4.encoded(value.get(k))!=budget.v4.encoded(v) for k,v in expected.items()):
        raise ValueError('Exact root narrow marker-reader repair authority required')
    expiry=datetime.fromisoformat(value['expires_at_utc'].replace('Z','+00:00'))
    if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('Expired marker-reader repair authority')
    budget.v4.bound(ROOT,PREVIOUS_REF);budget.v4.bound(ROOT,MARKER_REF)
    barrier=_bound(value['activation_barrier_ref'])
    fields={'kind','root_authorizer','known_root_owned_guard_pids','observations','roster_complete',
            'all_old_guard_consumers_and_supervisors_terminal','all_old_guard_future_writes_and_relaunch_revoked'}
    pids=barrier.get('known_root_owned_guard_pids')
    if (set(barrier)!=fields or barrier['kind']!='H039_missing_marker_old_guard_terminal_barrier_v1'
            or barrier['root_authorizer']!='root_agent'
            or any(barrier[k] is not True for k in ('roster_complete','all_old_guard_consumers_and_supervisors_terminal',
                                                  'all_old_guard_future_writes_and_relaunch_revoked'))
            or not isinstance(pids,list) or not pids or len(pids)>256
            or any(type(p) is not int or p<2 for p in pids) or len(set(pids))!=len(pids)
            or len(barrier['observations'])!=len(pids)):
        raise ValueError('Root complete targeted terminal/revocation barrier required')
    for pid,row in zip(pids,barrier['observations']):previous.old._observation(row,pid)
    receipt=_bound(RECEIPT_REF);own=receipt['value'];authority=_bound(AUTHORITY_REF)
    if (budget.digest(own)!=RESERVATION_SHA or own['key']!=KEY or own['reserved_bytes']!=15000000
            or own['root_admission']!=AUTHORITY_REF or own['auxiliary_scopes']!=[]
            or own['output']!=str(ROOT/'experiments/results/directed_clade_exchange_train_stage_v1_seed101')
            or any(r['path']==MARKER_REF['path'] for r in own['consumer_bindings'])
            or authority['operation']!='reserve'
            or authority['reservation']!={k:v for k,v in own.items() if k!='root_admission'}):
        raise ValueError('Exact malformed15MB reservation/authority proof required')
    return value,own


def configure_repair_v1(repair_authority_ref):
    """Read-only configuration. Root supplies a genuine barrier; no live action."""
    global _REPAIR_REF,_READERS,_ORIGINAL_PATHS
    _REPAIR_REF=copy.deepcopy(repair_authority_ref);_READERS=None;_ORIGINAL_PATHS=None
    _repair()


def _paths(own):
    if own.get('key')==KEY:
        _,original=_repair()
        if own!=original or budget.digest(own)!=RESERVATION_SHA:
            raise ValueError('Only exact immutable malformed reservation qualifies')
        effective=copy.deepcopy(own)
        effective['consumer_bindings'].append(copy.deepcopy(MARKER_REF))
        # Temporary parsing input only. Stored reservation/authority remain exact.
        return budget.successor_paths(ROOT,effective,True)
    if _ORIGINAL_PATHS is None:_readers()
    return _ORIGINAL_PATHS(own)


def _namespace(module,**changes):
    original=vars(module) if not isinstance(module,dict) else module
    ns=dict(original)
    for key,value in original.items():
        if isinstance(value,FunctionType) and value.__globals__ is original:
            method=FunctionType(value.__code__,ns,value.__name__,value.__defaults__,value.__closure__)
            method.__kwdefaults__=value.__kwdefaults__;ns[key]=method
    ns.update(changes)
    return ns


def _readers():
    global _READERS,_ORIGINAL_PATHS
    if _READERS is None:
        _repair();original=previous._frozen();_ORIGINAL_PATHS=original['_paths']
        cpu=_namespace(original,_paths=_paths,_CACHE=None)
        terminal=SimpleNamespace(**_namespace(previous.old,_cpu_reader=lambda:cpu,_paths=_paths,_CACHE=None))
        success=SimpleNamespace(**_namespace(previous.previous,old=terminal,_frozen=lambda:cpu,_paths=_paths,_CACHE=None))
        retired=SimpleNamespace(**_namespace(previous,old=terminal,previous=success,_frozen=lambda:cpu,_paths=_paths,_CACHE=None))
        _READERS=(cpu,retired)
    return _READERS


def _state_locked(source_sha,launcher_sha,*,extra=(),fresh=False):
    _repair();_,retired=_readers()
    extra=set(extra)|{SOURCE,MARKER_REF['path'],_REPAIR_REF['path']}
    result=retired._state_locked(source_sha,launcher_sha,extra=extra,fresh=fresh)
    reservations,closed,view,bindings,events=result
    # The malformed key is charged unchanged but cannot execute through new guards.
    view=dict(view,marker_repair_authority_ref=copy.deepcopy(_REPAIR_REF),
              malformed_reservation_retained_allowance_bytes=15000000,marker_repair_accounting_credit_bytes=0,
              historical_journal_or_binding_mutated=False)
    return reservations,closed,view,bindings,events


def inspect_v1(*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):return _state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)[2]


def _authority(ref,operation,bindings,*,fresh):
    value=previous._authority(ref,operation,bindings,fresh=fresh)
    if fresh:
        _repair()
        if operation!='reserve':raise ValueError('Repair adapter admits only fresh reservations')
        needed=[_ref(SOURCE),_REPAIR_REF,MARKER_REF]
        if any(r not in value['reservation']['consumer_bindings'] for r in needed):
            raise ValueError('Fresh repair-reader reservation must bind helper/repair authority/actual v9 marker')
    return value


def _append(operation,authority_ref,value,state):return previous._append(operation,authority_ref,value,state)


def _adapt(name):
    cpu,_=_readers()
    return previous._clone(cpu[name],SOURCE=SOURCE,MODE=MODE,JOURNAL=JOURNAL,_state_locked=_state_locked,
        _pins=_pins,_paths=_paths,_authority=_authority,_append=_append,
        budget_check_v9=budget_check_v9,check_storage_locked_v1=check_storage_locked_v1)


def apply_reserve_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    value=_adapt('apply_reserve_v1')(authority_ref,expected_source_sha256=expected_source_sha256,
                                   expected_launcher_sha256=expected_launcher_sha256)
    value['kind']='H039_marker_repair_fresh_reserve_receipt_v1'
    value['terminal_overlay_tail_sha256']=value.pop('cpu_overlay_tail_sha256')
    return value


def check_storage_locked_v1(storage,*,first=False,boundary=False,required_write_bytes=0):
    _repair()
    if storage['reservation'].get('key')==KEY:raise ValueError('Malformed historical scope remains fully charged and revoked')
    if any(r not in storage['reservation']['consumer_bindings'] for r in (_ref(SOURCE),_REPAIR_REF,MARKER_REF)):
        raise ValueError('Actual fresh marker-repair source/authority/marker binding required')
    return _adapt('check_storage_locked_v1')(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def budget_check_v9(storage,*,first=False,boundary=False,required_write_bytes=0):
    with budget.v4.mutex(ROOT):
        return check_storage_locked_v1(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def validate_storage_v9(spec,cfg,storage,*,source_ref,config_ref,runtime_ref):
    return _adapt('validate_storage_v9')(spec,cfg,storage,source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref)


def _writer_method(method):
    return previous._clone(method,budget_check_v9=budget_check_v9)


class PilotOutputV9(previous.previous.old.previous.PilotOutputV9):
    """Existing compact serialization and ownership with this reader per write."""
    __init__=_writer_method(previous.previous.old.previous.PilotOutputV9.__init__)
    save=_writer_method(previous.previous.old.previous.PilotOutputV9.save)
    discard=_writer_method(previous.previous.old.previous.PilotOutputV9.discard)
