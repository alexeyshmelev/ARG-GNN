"""Root-only CPU completion credits and compatible append-only V9 overlay.

No action at import. Original ledgers, journal, modules and GPU closure evidence
are unchanged. All consumers/launchers must honor this cooperative revocation.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import FunctionType
from scripts import project_budget_v9 as budget
from . import project_storage_closed_scopes_v1 as previous

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='src/arggnn/project_storage_cpu_closed_scopes_v1.py'
JOURNAL='.runtime/storage/project_budget_cpu_closed_scopes_v1'
MODE='journal_cpu_completed_scopes_credit_v1'
METADATA_CHARGE=65536
_CACHE=None


def _utc():return datetime.now(timezone.utc).isoformat()
def _ref(path):return budget.v4.reference(ROOT,path)
def _bound(ref):return budget.v4.bound(ROOT,ref,True)
def _pins(view):
    return {k:view[k] for k in ('legacy_ledger_sha256','supplemental_ledger_sha256',
        'journal_tail_sha256','cpu_overlay_tail_sha256')}


def _raw_reconcile(view,old,new,closed):
    """Original R->A arithmetic, delaying only its final ceiling test."""
    all_reservations={**old,**new}
    if set(old)&set(new) or not set(closed)<=set(all_reservations):
        raise ValueError('Historical reservation key/closure drift')
    credit=0
    for key,value in closed.items():
        reservation=all_reservations[key]
        if value['reservation_sha256']!=budget.digest(reservation):
            raise ValueError('Original closed reservation drift')
        actual=budget.v4.integer(value['measurement']['conservative_scope_bytes'],'retained bytes')
        allowance=budget.v4.integer(reservation['reserved_bytes'],'allowance',1)
        if actual>allowance:raise ValueError('Original closed scope exceeds allowance')
        credit+=allowance-actual
    added=sum(budget.v4.integer(r['reserved_bytes'],'allowance',1) for r in new.values())
    return {**view,'v4_projected_bytes':view['projected_bytes'],'v9_reserved_bytes':added,
        'closed_allowance_credit_bytes':credit,'projected_bytes':view['projected_bytes']+added-credit,
        'accounting_is_conservative_not_global_physical_measurement':True,
        'valid_under_cooperative_project_write_authority_contract':True,
        'external_NFS_writers_outside_observation':True}


_ORIGINAL_STATE=FunctionType(budget.state.__code__,dict(budget.state.__globals__,reconcile=_raw_reconcile),
    budget.state.__name__,budget.state.__defaults__,budget.state.__closure__)
_ORIGINAL_STATE.__kwdefaults__=budget.state.__kwdefaults__


def _fingerprint(extra=()):
    paths=[budget.v4.LEGACY,budget.v4.SUPPLEMENTAL,budget.JOURNAL_REL,budget.SOURCE_REL,
        budget.LAUNCHER_REL,'configs/project_budget_v4.json',SOURCE,JOURNAL,*extra]
    rows=[]
    for name in dict.fromkeys(paths):
        p=budget.v4.canonical(ROOT,name)
        try:s=p.lstat()
        except FileNotFoundError:rows.append((str(p),None));continue
        rows.append((str(p),s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
    return tuple(rows)


def _paths(reservation):
    is_v9=any(r.get('path')=='src/arggnn/project_budget_v9_consumer.py' for r in reservation['consumer_bindings'])
    paths=(budget.successor_paths(ROOT,reservation,'root_admission' in reservation) if is_v9
        else budget.v4.scope_spec(ROOT,reservation,'root_admission' in reservation))
    if any(budget.v4.paths_overlap(p,ROOT/JOURNAL) for p in paths):
        raise ValueError('Scientific scope overlaps CPU overlay journal')
    return paths


def _authority(ref,operation,bindings,*,fresh):
    value=_bound(ref)
    required=dict(kind='project_budget_cpu_overlay_root_admission_v1',root_authorizer='root_agent',
        active=True,operation=operation,accounting_mode=MODE,bindings=bindings,
        all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_cpu_overlay_guard=True,
        historical_records_mutation_authorized=False,global_rebase_authorized=False,
        scientific_execution_authorized=False,GPU_authorized=False)
    if any(value.get(k)!=v for k,v in required.items()):
        raise ValueError('Exact root CPU overlay authority required')
    if operation=='close' and any(value.get(k) is not True for k in (
        'scope_write_authority_revoked_for_all_cooperative_project_jobs',
        'closed_scope_future_writes_and_relaunch_revoked','external_NFS_writers_outside_observation')):
        raise ValueError('CPU closure requires truthful cooperative future-write revocation')
    if fresh:
        expiry=datetime.fromisoformat(value['expires_at_utc'].replace('Z','+00:00'))
        if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('Expired CPU overlay authority')
    return value


def _load_events(bindings):
    directory=budget.v4.canonical(ROOT,JOURNAL)
    if not directory.exists():return [],None
    if not directory.is_dir():raise ValueError('CPU overlay must be a canonical directory')
    names=sorted(p.name for p in directory.iterdir())
    if len(names)>4096:raise ValueError('CPU overlay event cap exceeded')
    events=[];tail=None;total=0
    for number,name in enumerate(names,1):
        if name!=f'{number:06d}.json':raise ValueError('CPU overlay gap/unknown entry')
        raw,sha=budget.v4.file_bytes(ROOT,directory/name);total+=len(raw)
        if total>32*1024*1024:raise ValueError('CPU overlay byte cap exceeded')
        event=budget.v4.parse(raw)
        if (set(event)!={'schema','sequence','previous_sha256','bindings','operation','authority','value'}
                or event['schema']!='project_budget_cpu_overlay_event_v1'
                or event['sequence']!=number or event['previous_sha256']!=tail
                or event['bindings']!=bindings or event['operation'] not in ('close','reserve')):
            raise ValueError('Immutable CPU overlay chain drift')
        authority=_authority(event['authority'],event['operation'],bindings,fresh=False)
        if authority.get('expected_before',{}).get('cpu_overlay_tail_sha256')!=tail:
            raise ValueError('CPU overlay authority predecessor drift')
        events.append(event);tail=sha
    return events,tail


def _state_locked(source_sha,launcher_sha,*,extra=(),fresh=False):
    global _CACHE
    bindings={'implementation':{'path':budget.SOURCE_REL,'sha256':source_sha},
        'launcher':{'path':budget.LAUNCHER_REL,'sha256':launcher_sha},'cpu_overlay':_ref(SOURCE)}
    dependencies=_CACHE[3] if _CACHE is not None else ()
    before=_fingerprint(sorted(set(extra)|set(dependencies)))
    if not fresh and _CACHE is not None and _CACHE[0]==bindings and _CACHE[1]==before:
        return copy.deepcopy(_CACHE[2])
    c,original_bindings=budget.reviewed(ROOT,source_sha,launcher_sha)
    legacy,old,new,gpu_closed,base,original_events=_ORIGINAL_STATE(ROOT,c,original_bindings,verify_closed=False)
    all_reservations={**old['reservations'],**new};events,tail=_load_events(bindings)
    cpu_closed={};overlay={};occupied=[budget.v4.canonical(ROOT,r['output']) for r in legacy['jobs'].values()]
    for r in all_reservations.values():occupied.extend(_paths(r))
    for event in events:
        value=event['value'];authority=_bound(event['authority'])
        if event['operation']=='reserve':
            request=authority['reservation']
            if value!={**request,'root_admission':event['authority']}:
                raise ValueError('CPU overlay reservation authority drift')
            key=value['key'];paths=_paths(value)
            if key in all_reservations or key in overlay or key in legacy['jobs'] or any(
                    budget.v4.paths_overlap(a,b) for a in paths for b in occupied):
                raise ValueError('CPU overlay reservation key/scope collision')
            overlay[key]=value;occupied.extend(paths)
        else:
            request=authority['closure'];key=request['key'];reservation=all_reservations.get(key) or overlay.get(key)
            if reservation is None or key in gpu_closed or key in cpu_closed:
                raise ValueError('Unknown/duplicate/GPU-closed CPU scope')
            if (value['request']!=request or value['reservation_sha256']!=budget.digest(reservation)
                    or value['measurement']['scope_paths']!=[str(p) for p in _paths(reservation)]
                    or value['worker_observation']['absent'] is not True
                    or value['completion_verified'] is not True):
                raise ValueError('CPU closure certificate/ownership drift')
            actual=budget.v4.integer(value['measurement']['conservative_scope_bytes'],'CPU retained bytes')
            if actual>reservation['reserved_bytes']:raise ValueError('CPU retained bytes exceed allowance')
            _completion(request,reservation,observe=False)
            cpu_closed[key]=value
    all_reservations.update(overlay)
    credit=sum(all_reservations[k]['reserved_bytes']-v['measurement']['conservative_scope_bytes'] for k,v in cpu_closed.items())
    added=sum(r['reserved_bytes'] for r in overlay.values())
    projected=base['projected_bytes']+added-credit+len(events)*METADATA_CHARGE
    if base['hard_limit_bytes']!=1000000000000 or not 0<=projected<min(base['limit_bytes'],base['hard_limit_bytes']):
        raise ValueError('CPU-credited scheduling/human ceiling exceeded')
    view={**base,'original_v9_projected_bytes':base['projected_bytes'],'projected_bytes':projected,
        'cpu_closed_allowance_credit_bytes':credit,'cpu_overlay_reserved_bytes':added,
        'cpu_overlay_metadata_charge_bytes':len(events)*METADATA_CHARGE,
        'cpu_overlay_tail_sha256':tail,'cpu_overlay_events':len(events),'accounting_mode':MODE,
        'global_process_inspection_performed':False,'original_journal_mutated':False}
    result=(all_reservations,set(gpu_closed)|set(cpu_closed),view,bindings,events)
    if before!=_fingerprint(sorted(set(extra)|set(dependencies))):
        raise ValueError('Accounting namespace changed during validation')
    discovered=set()
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'} and Path(value['path']).suffix in {'.json','.py','.sh'}:
                discovered.add(value['path'])
            for child in value.values():collect(child)
        elif isinstance(value,(tuple,list)):
            for child in value:collect(child)
    collect(list(all_reservations.values()));collect(original_events);collect(events)
    for event in original_events+events:collect(_bound(event['authority']))
    discovered.update(extra)
    after=_fingerprint(sorted(discovered))
    _CACHE=(bindings,after,copy.deepcopy(result),tuple(sorted(discovered)))
    return result


def inspect_v1(*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        return _state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)[2]


def _append(operation,authority_ref,value,state):
    _,_,view,bindings,events=state
    if len(events)>=4096:raise ValueError('CPU overlay event cap exceeded')
    event=dict(schema='project_budget_cpu_overlay_event_v1',sequence=len(events)+1,
        previous_sha256=view['cpu_overlay_tail_sha256'],bindings=bindings,operation=operation,
        authority=authority_ref,value=value)
    raw=budget.v4.encoded(event)+b'\n'
    if len(raw)>METADATA_CHARGE:raise ValueError('CPU overlay event exceeds charged metadata allowance')
    directory=budget.v4.canonical(ROOT,JOURNAL);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    path=directory/f'{len(events)+1:06d}.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    stored,sha=budget.v4.file_bytes(ROOT,path)
    if stored!=raw:raise ValueError('CPU overlay append uncertainty; preserve partial event')
    return sha


def _absent(pid):
    budget.v4.integer(pid,'recorded CPU worker PID',2)
    try:os.kill(pid,0)
    except ProcessLookupError:return dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=_utc())
    except PermissionError:raise ValueError('Recorded CPU worker identity cannot be certified') from None
    raise ValueError('Recorded CPU worker PID is still present, including zombie/reuse')


def _completion(request,reservation,*,observe=True):
    storage=_bound(request['storage_ref']);completion=_bound(request['completion_ref'])
    exit_value=_bound(request['launcher_exit_ref']);launch=_bound(request['launch_ref'])
    parent=request['parent_id'];paths=_paths(reservation)
    if (storage['reservation']!=reservation or request['key']!=reservation['key']
            or request['completion_ref']['path']!=str((paths[0]/'expanded_parent_completed.json').relative_to(ROOT))
            or request['launcher_exit_ref']['path']!=str(Path(request['launch_ref']['path']).parent/'launcher_exit.json')
            or completion.get('schema')!='expanded_full_panel_v1_parent_completion'
            or completion.get('parent_id')!=parent or completion.get('generation_completed') is not True
            or completion.get('Relate_completed') is not True or completion.get('GPU_used') is not False
            or completion.get('outer_truth_evaluated') is not False
            or exit_value.get('parent_id')!=parent or type(exit_value.get('returncode')) is not int
            or exit_value['returncode']!=0 or exit_value.get('CPU_only') is not True
            or launch.get('parent_id')!=parent or launch.get('device')!='cpu'
            or launch.get('CUDA_VISIBLE_DEVICES')!='' or launch.get('pid')!=exit_value.get('worker_pid')):
        raise ValueError('Actual successful same-parent CPU completion/launcher wait required')
    # The resume supervisor's exact five-field receipt omitted the child-wait flag.
    # Accept that historical omission only with independently bound native success.
    child_wait=exit_value.get('native_children_waited_by_original_inference_code')
    if child_wait is not True:
        if (child_wait is not None or set(exit_value)!={'CPU_only','completed_ref','parent_id','returncode','worker_pid'}
                or completion['source_ref']!=_ref('src/arggnn/prepare_expanded_full_panel_v1.py')
                or completion['original_producer_ref']!=_ref('src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py')
                or completion['inference_ref']['path']!=str((paths[0]/'relate/inference.json').relative_to(ROOT))):
            raise ValueError('Missing native-child wait flag requires exact historical resume schema and pinned producer')
        inference=_bound(completion['inference_ref'])
        codes=inference.get('outer_command_returncodes')
        if (inference.get('parent_id')!=parent or inference.get('status')!='post_five_iterations_inferred'
                or inference.get('truth_or_oracle_read') is not False
                or not isinstance(codes,list) or not codes or any(type(code) is not int or code!=0 for code in codes)):
            raise ValueError('Bound original native inference must record successful waited commands')
    # Older wait receipts with completed_ref=null are not rewritten or synthesized.
    if exit_value.get('completed_ref') is not None and exit_value['completed_ref']!=request['completion_ref']:
        raise ValueError('Recorded CPU wait completion binding differs')
    admission=_bound(launch['input_ref'])
    if (admission.get('schema')!='expanded_full_panel_v1_root_admission'
            or admission.get('root_authorizer')!='root_agent' or admission.get('execution_authorized') is not True
            or admission.get('parent_id')!=parent or admission.get('storage')!=storage
            or admission.get('output_directory')!=storage['output_directory']
            or admission.get('source_ref')!=completion['source_ref']
            or admission.get('config_content_sha256')!=completion['config_content_sha256']
            or admission.get('parent_registry_sha256')!=completion['parent_registry_sha256']
            or admission.get('outer_evaluation_authorized') is not False):
        raise ValueError('CPU completion must bind original admitted producer/registry/storage')
    return _absent(exit_value['worker_pid']) if observe else None


def apply_close_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        reservations,closed,view,bindings,_=state
        authority=_authority(authority_ref,'close',bindings,fresh=True)
        if authority['expected_before']!=_pins(view):raise ValueError('Fresh CPU overlay/historical pins required')
        request=authority['closure'];key=request['key']
        if key not in reservations or key in closed:raise ValueError('Unknown/already closed CPU key')
        reservation=reservations[key];observation=_completion(request,reservation)
        measurement=budget.v4.measure_scopes(ROOT,_paths(reservation))  # Exactly one owned-scope inventory.
        _absent(observation['pid'])
        actual=measurement['conservative_scope_bytes']
        if actual>reservation['reserved_bytes']:raise ValueError('CPU output exceeds original allowance')
        value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(reservation),
            measurement=measurement,worker_observation=observation,completion_verified=True,
            closed_utc=_utc(),future_writes_and_relaunch_revoked=True)
        projected=view['projected_bytes']-(reservation['reserved_bytes']-actual)+METADATA_CHARGE
        if not 0<=projected<min(view['limit_bytes'],view['hard_limit_bytes']):
            raise ValueError('CPU closure metadata charge exceeds scheduling ceiling')
        tail=_append('close',authority_ref,value,state)
        return dict(kind='project_budget_cpu_close_receipt_v1',key=key,retained_bytes=actual,
            allowance_bytes=reservation['reserved_bytes'],credit_bytes=reservation['reserved_bytes']-actual,
            projected_bytes=projected,
            cpu_overlay_tail_sha256=tail,value=value,original_journal_mutated=False,
            scope_measurements=1,global_process_scan_performed=False,GPU_used=False)


def apply_reserve_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        reservations,closed,view,bindings,_=state
        authority=_authority(authority_ref,'reserve',bindings,fresh=True)
        if authority['expected_before']!=_pins(view):raise ValueError('Fresh CPU overlay/historical pins required')
        request=authority['reservation'];paths=_paths(request)
        if (_ref(SOURCE) not in request['consumer_bindings'] or request['key'] in reservations
                or any(p.exists() for p in paths)):
            raise ValueError('Fresh exclusive CPU-overlay consumer reservation required')
        occupied=[]
        for reservation in reservations.values():occupied.extend(_paths(reservation))
        # The original replay already rejects overlap with legacy/accounting paths.
        legacy=budget.v4.read_json(ROOT,budget.v4.LEGACY)[0]
        occupied.extend(budget.v4.canonical(ROOT,r['output']) for r in legacy['jobs'].values())
        if request['key'] in legacy['jobs'] or any(budget.v4.paths_overlap(a,b) for a in paths for b in occupied):
            raise ValueError('CPU-overlay reservation key/scope overlaps charged history')
        projected=view['projected_bytes']+request['reserved_bytes']+METADATA_CHARGE
        if projected>=min(view['limit_bytes'],view['hard_limit_bytes']):raise ValueError('Credited admission exceeds scheduling ceiling')
        value={**copy.deepcopy(request),'root_admission':copy.deepcopy(authority_ref)}
        tail=_append('reserve',authority_ref,value,state)
        return dict(kind='project_budget_cpu_reserve_receipt_v1',projected_bytes=projected,
            cpu_overlay_tail_sha256=tail,value=value,original_journal_mutated=False,
            global_process_scan_performed=False,GPU_used=False)


def check_storage_locked_v1(storage,*,first=False,boundary=False,required_write_bytes=0):
    """Caller holds the original canonical scheduler mutex; never reacquire it."""
    if (storage.get('kind')!='project_budget_v9_storage_record' or storage.get('active') is not True
            or storage.get('consumer_uses_v9_guard') is not True or storage.get('accounting_mode')!=MODE):
        raise ValueError('Active CPU-overlay consumer authority required')
    if storage.get('expires_at_utc'):
        expiry=datetime.fromisoformat(storage['expires_at_utc'].replace('Z','+00:00'))
        if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('Expired storage authority')
    refs=storage['budget_components']
    previous._source_ref(refs['implementation'],budget.SOURCE_REL)
    previous._source_ref(refs['launcher'],budget.LAUNCHER_REL)
    previous._source_ref(refs['consumer_adapter'],SOURCE)
    own=storage['reservation'];budget.v4.integer(required_write_bytes,'planned growth')
    if required_write_bytes and not boundary:raise ValueError('Planned writes require owned-scope boundary')
    extra=[r['path'] for r in own['consumer_bindings']]
    reservations,closed,view,_,_=_state_locked(refs['implementation']['sha256'],refs['launcher']['sha256'],extra=extra)
    if (reservations.get(own['key'])!=own or own['key'] in closed
            or own['reserved_bytes']!=storage['reserved_new_bytes']
            or own['output']!=str(budget.v4.canonical(ROOT,storage['output_directory']))
            or any(r not in own['consumer_bindings'] for r in refs.values())):
        raise ValueError('Exact active admitted own reservation required; closed keys cannot write/relaunch')
    if first and storage['initial_ledger_pins']!=_pins(view):raise ValueError('Initial historical/CPU-overlay pins changed')
    measurement=budget.v4.measure_scopes(ROOT,_paths(own)) if boundary else None
    if measurement and measurement['conservative_scope_bytes']+required_write_bytes>own['reserved_bytes']:
        raise ValueError('Owned output exceeds reserved allowance')
    return dict(view,reservation=copy.deepcopy(own),scope_measurement=measurement,
        required_write_bytes=required_write_bytes,
        remaining_scope_bytes=own['reserved_bytes']-measurement['conservative_scope_bytes'] if measurement else None)

def budget_check_v9(storage,*,first=False,boundary=False,required_write_bytes=0):
    with budget.v4.mutex(ROOT):
        return check_storage_locked_v1(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def validate_storage_v9(spec,cfg,storage,*,source_ref,config_ref,runtime_ref):
    if (storage['job_id']!=spec['job_id'] or storage['output_directory']!=spec['output_directory']
            or not 0<storage['reserved_new_bytes']<=cfg['output_cap_bytes']
            or any(r not in storage['reservation']['consumer_bindings'] for r in (source_ref,config_ref,runtime_ref))):
        raise ValueError('Exact consumer/config/runtime/output bindings required')
    return budget_check_v9(storage,first=True,boundary=True)


def _writer_method(method):
    namespace=dict(method.__globals__,budget_check_v9=budget_check_v9)
    value=FunctionType(method.__code__,namespace,method.__name__,method.__defaults__,method.__closure__)
    value.__kwdefaults__=method.__kwdefaults__
    return value


class PilotOutputV9(previous.PilotOutputV9):
    """Same compact serialization, with an independent CPU-overlay guard namespace."""
    __init__=_writer_method(previous.PilotOutputV9.__init__)
    save=_writer_method(previous.PilotOutputV9.save)
    discard=_writer_method(previous.PilotOutputV9.discard)


def prepare_expanded_v1(admission_ref,*,config_ref,producer_ref,parent_id,stage='all'):
    """CPU entry point calls the unchanged original producer with this guard."""
    from . import prepare_expanded_full_panel_v1 as producer
    if (producer_ref!=_ref(producer.SOURCE) or config_ref['path']!='configs/expanded_full_panel_v1.json'
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise ValueError('Exact original producer/config and empty CUDA visibility required')
    cfg=_bound(config_ref);admission=_bound(admission_ref)
    if admission.get('storage_adapter_ref')!=_ref(SOURCE):raise ValueError('CPU-overlay adapter admission required')
    validate_storage_v9(admission,cfg,admission['storage'],source_ref=producer_ref,
        config_ref=config_ref,runtime_ref=admission['runtime_ref'])
    return producer.execute_parent_v1(cfg,parent_id,stage,admission,
        guard=lambda planned=0:budget_check_v9(admission['storage'],boundary=True,required_write_bytes=int(planned)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-expanded',action='store_true');parser.add_argument('--admission')
    parser.add_argument('--expected-admission-sha256');parser.add_argument('--config')
    parser.add_argument('--expected-config-sha256');parser.add_argument('--expected-producer-sha256')
    parser.add_argument('--parent');parser.add_argument('--stage',default='all',choices=('simulate','relate','all'))
    args=parser.parse_args(argv)
    if not args.prepare_expanded:
        print(json.dumps(dict(status='source_only_CPU_overlay_not_applied',source=_ref(SOURCE),accounting_mode=MODE),indent=2));return
    if not all((args.admission,args.expected_admission_sha256,args.config,args.expected_config_sha256,args.expected_producer_sha256,args.parent)):
        raise ValueError('Root pinned CPU producer arguments required')
    result=prepare_expanded_v1({'path':args.admission,'sha256':args.expected_admission_sha256},
        config_ref={'path':args.config,'sha256':args.expected_config_sha256},
        producer_ref={'path':'src/arggnn/prepare_expanded_full_panel_v1.py','sha256':args.expected_producer_sha256},
        parent_id=args.parent,stage=args.stage)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
