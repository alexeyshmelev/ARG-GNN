"""Prospective980 v2 with one exact historical H020 settlement branch.

Source-only until ROOT configures an independently reviewed active policy.
The original prospectivev1 source/policy/journal and admitted writers stay
unchanged. Every read authenticates its exact frozen parent pins/projection.
Only this new journal can receive the typed legacy proof. No fake completion.
"""
from __future__ import annotations
import copy,hashlib,importlib.util,io,json,os,stat
from datetime import datetime,timezone
from pathlib import Path
from scripts import project_budget_v9 as budget

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/prospective980_H020_legacy_source_20261016_v2/prospective_980gb_H020_v2.py'
JOURNAL='.runtime/storage/prospective_980gb_H020_v2'
MODE='prospective_980gb_H020_v2'
BASE_REF=dict(path='research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py',sha256='fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56')
BASE_POLICY_REF=dict(path='research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json',sha256='eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018')
BASE_MODE='prospective_980gb_v1'
WARNING_BYTES=950000000000
CEILING_BYTES=980000000000
HUMAN_LIMIT_BYTES=1000000000000
REPAIR_REF=dict(path='research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py',sha256='e4185dc51b285d8ad4eba224b8e1e15f3e1479f57c73b4ec53fff2b238d0ef2f')
CAPACITY_V1_REF=dict(path='research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py',sha256='7c0a35f82423eb824edf3b7dccbac31f139d50d2ee397d4d92515b6a0ae33c88')
MARKER_REF=dict(path='src/arggnn/project_budget_v9_consumer.py',sha256='c11a5543746899c8a387508e9877ce6ae2bf80fe4d3076c1dbe1cb8f7a43b859')
MALFORMED_KEY='H039/directed-clade-exchange-train-stage-seed101-v1'
META_CHARGE=524288
BOOTSTRAP_CHARGE=1048576
MAX_EVENTS=4096
MAX_JSON=65536
MAX_COMPLETED_JSON=1048576
COMPLETED_SCHEMAS=frozenset((
    'expanded_native_ap_driver_completed_v2',
    'expanded_native_argmix_preparation_v1_root_admission_completed',
    'expanded_pyg_observations_v5_completed',
))
_BASE=_POLICY_REF=_REPAIR_AUTHORITY_REF=None


def require(ok,message):
    if not ok:raise ValueError(message)
def _utc():return datetime.now(timezone.utc).isoformat()
def _ref(path):return budget.v4.reference(ROOT,path)
def _raw(record):return budget.v4.bound(ROOT,record)
def _json(record):
    raw=_raw(record);require(len(raw)<=MAX_JSON,'Bounded authority/evidence JSON required')
    return budget.v4.parse(raw)
def _completed_json(record):
    raw=_raw(record);require(len(raw)<=MAX_COMPLETED_JSON,'Bounded producer completed JSON <=1MiB required')
    return budget.v4.parse(raw)
def _expire(value):
    stamp=datetime.fromisoformat(value['expires_at_utc'].replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp>datetime.now(timezone.utc),'Unexpired root operation/storage authority required')
def _pointer(value,keys):
    require(isinstance(keys,list) and 1<=len(keys)<=4 and all(isinstance(k,str) and k for k in keys),'Explicit bounded metadata locator')
    for k in keys:value=value[k]
    return value
def _identity(value,expected):
    require(isinstance(expected,dict) and expected,'Nonempty exact root evidence identity required')
    require(all(k in value and budget.v4.encoded(value[k])==budget.v4.encoded(v) for k,v in expected.items()),'Bound evidence identity mismatch')


def _completed_identity(value,expected):
    _identity(value,expected)
    if isinstance(expected.get('status'),str) and expected['status'].startswith('completed'):
        return  # Preserve the original completed-status branch.
    require(expected.get('schema') in COMPLETED_SCHEMAS,'Exact allowlisted producer completed schema required')
    completed_utc=expected.get('completed_utc')
    require(isinstance(completed_utc,str),'Schema-only completion identity must bind actual completed_utc')
    stamp=datetime.fromisoformat(completed_utc.replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),
        'Schema-only completed_utc must be timezone-aware and nonfuture')


def charge(base_bytes,new_caps,certified_credits,event_count):
    for name,value in [('base',base_bytes),('new caps',new_caps),('credits',certified_credits),('events',event_count)]:
        budget.v4.integer(value,name)
    value=base_bytes+new_caps-certified_credits+BOOTSTRAP_CHARGE+META_CHARGE*event_count
    require(0<=value<CEILING_BYTES and value<HUMAN_LIMIT_BYTES,'Prospective980GB ceiling/1TB human limit exceeded')
    return value


def warning_fields(projected):
    return dict(warning_threshold_bytes=WARNING_BYTES,warning_active=projected>=WARNING_BYTES,
        warning_message='Projected project charge reached950GB warning; fixed980GB future ceiling and1TB user limit remain enforced' if projected>=WARNING_BYTES else None,
        admission_ceiling_bytes=CEILING_BYTES,human_limit_bytes=HUMAN_LIMIT_BYTES,
        remaining_to_admission_ceiling_bytes=CEILING_BYTES-projected,
        remaining_to_human_limit_bytes=HUMAN_LIMIT_BYTES-projected)

def check_prefix(pin,hashes,*,minimum_events=0):
    require(isinstance(pin,dict) and set(pin)=={'events','tail_sha256'},'Exact overlay prefix pin required')
    n=budget.v4.integer(pin['events'],'prefix event count')
    require(minimum_events<=n<=len(hashes) and pin['tail_sha256']==(hashes[n-1] if n else None),
        'Pinned prefix is not an append-only ancestor or predates this reservation')
    return True


def assert_open(key,reservations,closed):
    require(key in reservations and key not in closed and key!=MALFORMED_KEY,'Unknown/closed/revoked scope cannot write or relaunch')


def root_activation_draft_v1(*,base_pins,base_projected_bytes):
    """ROOT-only inert shape. No parent state is selected or read by this helper."""
    return dict(schema='prospective980_H020_v2_root_frozen_parent_activation_v1',root_authorizer='root_agent',active=False,
        source_ref=_ref(SOURCE),frozen_parent_ref=BASE_REF,frozen_parent_policy_ref=BASE_POLICY_REF,
        frozen_parent_pins=copy.deepcopy(base_pins),frozen_parent_projected_bytes=base_projected_bytes,new_journal=JOURNAL,
        all_future_parent_reserve_close_mutations_cooperatively_revoked=True,
        all_older_accounting_future_mutation_revocations_preserved=True,
        current_parent_admitted_writers_continue_unchanged_with_full_original_caps=True,
        old_parent_source_policy_journal_and_consumer_storage_records_unchanged=True,
        no_historical_mutation=True,no_deletion=True,no_rebase=True,physical_GPU_ids=[0])


def activation_policy_draft_v1(*,repair_authority_ref,base_pins,base_projected_bytes,implementation_ref,launcher_ref,root_activation_ref=None):
    """Inert draft. ROOT selects/authenticates the actual frozen parent at activation."""
    return dict(schema='prospective_980GB_H020_credit_policy_v2',root_authorizer='root_agent',active=False,
        source_ref=_ref(SOURCE),base_ref=BASE_REF,base_policy_ref=BASE_POLICY_REF,base_accounting_mode=BASE_MODE,
        repair_ref=REPAIR_REF,capacity_v1_ref=CAPACITY_V1_REF,marker_ref=MARKER_REF,
        marker_repair_authority_ref=repair_authority_ref,implementation_ref=implementation_ref,launcher_ref=launcher_ref,
        base_pins=copy.deepcopy(base_pins),base_projected_bytes=base_projected_bytes,journal=JOURNAL,
        root_activation_ref=root_activation_ref,
        scheduling_ceiling_bytes=CEILING_BYTES,warning_threshold_bytes=WARNING_BYTES,human_limit_bytes=HUMAN_LIMIT_BYTES,
        bootstrap_metadata_charge_bytes=BOOTSTRAP_CHARGE,per_event_metadata_charge_bytes=META_CHARGE,
        all_old_future_admissions_revoked=True,all_future_admissions_use_this_overlay=True,
        old_active_writers_continue_unchanged_fully_charged=True,old_journals_frozen_for_new_events=True,
        all_budget_writers_use_original_mutex=True,closed_scope_all_write_resume_capabilities_revoked=True,
        no_historical_mutation=True,no_rebase=True,no_deletion=True,
        cooperative_root_process_rosters_complete=True,external_NFS_writers_outside_observation=True,
        new_unscoped_metadata_within_charged_allowances=True,never_credit_keys=[MALFORMED_KEY],
        eligible_physical_GPU_ids=[0],all_nonzero_physical_GPU_ids_excluded=True,
        actual_GPU_assignment_and_GPU_launcher_guards_owned_by_root=True,
        only_legacy_settlement_key=LEGACY_KEY,legacy_outer_tool_status='UNKNOWN',historical_legacy_descendant_drain='UNKNOWN')


def configure_v2(repair_authority_ref,*,policy_ref=None):
    """Same one-argument producer entry, with exact policy ref supplied via env.

    ROOT may pass policy_ref explicitly instead. Configuration is read-only,
    but validates live root evidence through the unchanged base; agent never calls it.
    """
    global _BASE,_POLICY_REF,_REPAIR_AUTHORITY_REF
    if policy_ref is None:
        policy_ref=json.loads(os.environ['ARGGNN_COMPLETED_CREDITS_POLICY_REF'])
    require(_POLICY_REF is None or _POLICY_REF==policy_ref,'No policy swap inside a configured consumer')
    for r in (BASE_REF,BASE_POLICY_REF,REPAIR_REF,CAPACITY_V1_REF,MARKER_REF):_raw(r)
    _POLICY_REF=copy.deepcopy(policy_ref);_REPAIR_AUTHORITY_REF=copy.deepcopy(repair_authority_ref)
    _policy()
    spec=importlib.util.spec_from_file_location('_prospective980_H020_immutable_v1',ROOT/BASE_REF['path'])
    _BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(_BASE)
    _BASE.configure_v2(repair_authority_ref,policy_ref=BASE_POLICY_REF)


def _policy():
    require(_POLICY_REF is not None,'ROOT must configure exact active overlay policy')
    p=_json(_POLICY_REF)
    expected=activation_policy_draft_v1(repair_authority_ref=_REPAIR_AUTHORITY_REF,base_pins=p.get('base_pins'),
        base_projected_bytes=p.get('base_projected_bytes'),implementation_ref=p.get('implementation_ref'),
        launcher_ref=p.get('launcher_ref'),root_activation_ref=p.get('root_activation_ref'))
    expected['active']=True
    require(set(p)==set(expected) and p==expected,'Exact root source/policy/frozen-parent/revocation required')
    for r in (p['source_ref'],p['base_ref'],p['base_policy_ref'],p['repair_ref'],p['capacity_v1_ref'],p['marker_ref'],
              p['marker_repair_authority_ref'],p['implementation_ref'],p['launcher_ref']):_raw(r)
    require(p['implementation_ref']['path']==budget.SOURCE_REL and p['launcher_ref']['path']==budget.LAUNCHER_REL,
        'Original actual implementation/launcher required')
    budget.v4.integer(p['base_projected_bytes'],'fixed charged base projection')
    require(isinstance(p['base_pins'],dict) and set(p['base_pins'])=={'base_pins','completed_overlay_prefix'},
        'ROOT must select exact authentic prospectivev1 pins at activation')
    activation=_json(p['root_activation_ref'])
    required=dict(schema='prospective980_H020_v2_root_frozen_parent_activation_v1',root_authorizer='root_agent',active=True,
        source_ref=p['source_ref'],frozen_parent_ref=BASE_REF,frozen_parent_policy_ref=BASE_POLICY_REF,
        frozen_parent_pins=p['base_pins'],frozen_parent_projected_bytes=p['base_projected_bytes'],new_journal=JOURNAL,
        all_future_parent_reserve_close_mutations_cooperatively_revoked=True,
        all_older_accounting_future_mutation_revocations_preserved=True,
        current_parent_admitted_writers_continue_unchanged_with_full_original_caps=True,
        old_parent_source_policy_journal_and_consumer_storage_records_unchanged=True,
        no_historical_mutation=True,no_deletion=True,no_rebase=True,physical_GPU_ids=[0])
    require(set(activation)==set(required) and budget.v4.encoded(activation)==budget.v4.encoded(required),
        'Exact explicit ROOT frozen-parent activation required')
    return p


def _paths(own):
    require(_BASE is not None,'Configured immutable base required')
    # Preserve every frozen ancestor scope/marker rule and 64-direct-binding cap.
    paths=_BASE._paths(own)
    protected=[ROOT/JOURNAL,ROOT/SOURCE,budget.v4.canonical(ROOT,_POLICY_REF['path'])]
    require(not any(budget.v4.paths_overlap(a,b) for a in paths for b in protected),'Product scope overlaps overlay accounting/source/policy')
    return paths


def _bindings(base_bindings,p):
    result=copy.deepcopy(base_bindings)
    additions=dict(prospective_980_H020_overlay=_ref(SOURCE),prospective_980_H020_policy=_POLICY_REF,
        immutable_prospective_v1=BASE_REF,immutable_prospective_v1_policy=BASE_POLICY_REF)
    require(not set(result)&set(additions),'Never overwrite a frozen ancestor binding')
    result.update(additions)
    return result


def _pins(view):
    return dict(base_pins=copy.deepcopy(view['base_pins']),completed_overlay_prefix=copy.deepcopy(view['completed_overlay_prefix']))


def operation_draft_v1(operation,request,*,expected_before,bindings):
    require(operation in ('reserve','close_completed','retire_H020_seed202_v1'),'Only reserve, normal close or exact H020 retirement operations')
    return dict(schema='prospective_980GB_H020_root_operation_v2',root_authorizer='root_agent',active=False,
        operation=operation,accounting_mode=MODE,bindings=copy.deepcopy(bindings),expected_before=copy.deepcopy(expected_before),
        expires_at_utc=None,request=copy.deepcopy(request),all_budget_writers_use_original_mutex=True,
        all_old_future_admissions_revoked=True,all_future_consumers_use_this_overlay=True,
        historical_mutation_authorized=False,global_rebase_authorized=False,deletion_authorized=False,
        scientific_execution_authorized=False,GPU_authorized=False)


def _authority(record,operation,bindings,*,fresh):
    value=_json(record);expected=operation_draft_v1(operation,value.get('request'),expected_before=value.get('expected_before'),bindings=bindings)
    expected.update(active=True,expires_at_utc=value.get('expires_at_utc'))
    require(set(value)==set(expected) and value==expected,'Exact bound root operation authority required')
    if fresh:_expire(value)
    return value


def _proof(request,own,*,observe=False):
    fields={'key','reservation_sha256','storage_ref','storage_locator','launch_ref','launch_pid_locator','launch_identity',
        'direct_wait_ref','wait_pid_locator','wait_exit_locator','wait_identity','completed_ref','completed_identity','observation_ref'}
    require(set(request)==fields and request['key']==own['key'] and request['reservation_sha256']==budget.digest(own),
        'Exact completed reservation/evidence request required')
    storage=_json(request['storage_ref'])
    if request['storage_locator']:storage=_pointer(storage,request['storage_locator'])
    require(storage['reservation']==own and storage['reserved_new_bytes']==own['reserved_bytes']
        and own['output']==str(budget.v4.canonical(ROOT,storage['output_directory'])),'Actual stored reservation/output/cap binding required')
    launch=_json(request['launch_ref']);wait=_json(request['direct_wait_ref'])
    _identity(launch,request['launch_identity']);_identity(wait,request['wait_identity'])
    pid=_pointer(launch,request['launch_pid_locator']);budget.v4.integer(pid,'launcher PID',2)
    require(_pointer(wait,request['wait_pid_locator'])==pid and type(_pointer(wait,request['wait_exit_locator'])) is int
        and _pointer(wait,request['wait_exit_locator'])==0,'Actual direct launcher wait0 bound to its PID required')
    completion_path=budget.v4.canonical(ROOT,request['completed_ref']['path'])
    require(completion_path==Path(own['output'])/'completed.json','Actual producer output completed.json required')
    completion=_completed_json(request['completed_ref']);_completed_identity(completion,request['completed_identity'])
    obs=_json(request['observation_ref'])
    expected=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',
        key=own['key'],reservation_sha256=budget.digest(own),storage_ref=request['storage_ref'],
        launch_ref=request['launch_ref'],direct_wait_ref=request['direct_wait_ref'],completed_ref=request['completed_ref'],
        launcher_pid=pid,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
        completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
        external_NFS_writers_outside_observation=True)
    require(all(obs.get(k)==v for k,v in expected.items()),'Root exact completion/targeted roster/revocation certificate required')
    pids=obs.get('pids');observations=obs.get('observations')
    require(isinstance(pids,list) and 1<=len(pids)<=256 and pid in pids and len(set(pids))==len(pids)
        and all(type(x) is int and x>=2 for x in pids) and isinstance(observations,list) and len(observations)==len(pids),
        'Complete bounded launcher/supervisor/worker PID roster')
    for n,row in zip(pids,observations):
        require(set(row)=={'pid','absent','method','checked_utc'} and row['pid']==n and row['absent'] is True
            and row['method']=='targeted os.kill(pid,0) ESRCH','Honest targeted absent observation')
        stamp=datetime.fromisoformat(row['checked_utc'].replace('Z','+00:00'))
        require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'Nonfuture root observation')
    applied=[_absent(n) for n in pids] if observe else observations
    return dict(root_observation_ref=request['observation_ref'],pids=pids,PID_observations=applied,
        completed_ref=request['completed_ref'],direct_wait_ref=request['direct_wait_ref'],completion_verified=True)


def _absent(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=_utc())
    except PermissionError:raise ValueError('Targeted process absence cannot be certified') from None
    raise ValueError('Launcher/worker still present, zombie or PID reused; no completion credit')


def _measurement(value,own):
    require(set(value)=={'conservative_scope_bytes','entries','scope_paths','stat_only'} and value['stat_only'] is True
        and value['scope_paths']==[str(p) for p in _paths(own)],'Exact owned-scope stat certificate required')
    actual=budget.v4.integer(value['conservative_scope_bytes'],'retained completed bytes')
    budget.v4.integer(value['entries'],'scope entries')
    require(actual<=own['reserved_bytes'],'Completed scope exceeds cap; no credit')
    return actual


def _events(bindings):
    directory=budget.v4.canonical(ROOT,JOURNAL)
    if not directory.exists():return [],[]
    require(directory.is_dir(),'Canonical new overlay journal directory')
    before=directory.lstat();events=[];hashes=[];total=0
    names=sorted(p.name for p in directory.iterdir())
    require(len(names)<=MAX_EVENTS,'Overlay event bound')
    for i,name in enumerate(names,1):
        require(name==f'{i:06d}.json','Overlay gap/unknown entry')
        raw,sha=budget.v4.file_bytes(ROOT,directory/name);total+=len(raw)
        require(len(raw)<=MAX_JSON and total<=MAX_EVENTS*MAX_JSON,'Charged bounded new journal metadata')
        event=budget.v4.parse(raw)
        require(set(event)=={'schema','sequence','previous_sha256','bindings','operation','authority','value'}
            and event['schema']=='prospective_980GB_H020_event_v2' and type(event['sequence']) is int and event['sequence']==i
            and event['previous_sha256']==(hashes[-1] if hashes else None) and event['bindings']==bindings
            and event['operation'] in ('reserve','close_completed','retire_H020_seed202_v1'),'Exact immutable new-journal chain/bindings')
        events.append(event);hashes.append(sha)
    after=directory.lstat()
    require((before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns)==
        (after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns),'Journal changed during locked read')
    return events,hashes


def _state_locked(source_sha,launcher_sha,*,fresh=False):
    p=_policy();require(_BASE is not None,'Configure root base first')
    require((source_sha,launcher_sha)==(p['implementation_ref']['sha256'],p['launcher_ref']['sha256']),'Actual original budget source/launcher pins')
    base=_BASE._state_locked(source_sha,launcher_sha,fresh=fresh)
    require(len(base)==7,'Exact frozen prospectivev1 base state API')
    reservations,base_closed,view,base_bindings,_,_,_=base
    require(_BASE._pins(view)==p['base_pins'] and view['projected_bytes']==p['base_projected_bytes']
        and view['accounting_mode']==BASE_MODE and view['limit_bytes']==CEILING_BYTES
        and view['hard_limit_bytes']==HUMAN_LIMIT_BYTES,
        'Frozen prospectivev1 journal/admission base changed; no new charge admitted')
    reservations=copy.deepcopy(reservations);closed=set(base_closed);bindings=_bindings(base_bindings,p)
    legacy=budget.v4.read_json(ROOT,budget.v4.LEGACY)[0]
    occupied=[budget.v4.canonical(ROOT,v['output']) for v in legacy['jobs'].values()]
    for own in reservations.values():occupied.extend(_paths(own))
    events,hashes=_events(bindings);added=credit=0;reserved_at={}
    for i,event in enumerate(events):
        before=dict(base_pins=p['base_pins'],completed_overlay_prefix=dict(events=i,tail_sha256=hashes[i-1] if i else None))
        auth=_authority(event['authority'],event['operation'],bindings,fresh=False)
        require(auth['expected_before']==before,'Operation did not bind its actual prior new-journal prefix')
        request=auth['request'];value=event['value']
        if event['operation']=='reserve':
            paths=_paths(value);key=value['key']
            require(value==dict(request,root_admission=event['authority']) and key not in reservations and key not in legacy['jobs']
                and not any(budget.v4.paths_overlap(a,b) for a in paths for b in occupied),
                'Reservation authority/key/old-or-new scope overlap')
            require(all(r in value['consumer_bindings'] for r in bindings.values()),'All actual base/overlay/policy/marker bindings must be reserved')
            reservations[key]=value;occupied.extend(paths);added+=value['reserved_bytes'];reserved_at[key]=i+1
        elif event['operation']=='retire_H020_seed202_v1':
            key=request['key'];require(key==LEGACY_KEY,'Only exact H020 legacy key')
            assert_open(key,reservations,closed);own=reservations[key]
            proof=_legacy_proof(request,own)
            require(set(value)=={'request','reservation_sha256','measurement','proof','closed_utc','future_writes_and_relaunch_revoked'}
                and value['request']==request and value['reservation_sha256']==budget.digest(own)
                and value['future_writes_and_relaunch_revoked'] is True,'Typed legacy certificate drift')
            _legacy_recorded_proof(value['proof'],proof,value['closed_utc'])
            credit+=own['reserved_bytes']-_measurement(value['measurement'],own);closed.add(key)
        else:
            key=request['key'];assert_open(key,reservations,closed)
            require(key not in p['never_credit_keys'],'Policy forbids credit for this scope')
            own=reservations[key];proof=_proof(request,own)
            require(set(value)=={'request','reservation_sha256','measurement','proof','closed_utc','future_writes_and_relaunch_revoked'}
                and value['request']==request and value['reservation_sha256']==budget.digest(own)
                and value['future_writes_and_relaunch_revoked'] is True,'Completed credit/reservation certificate drift')
            recorded=value['proof']
            require(set(recorded)==set(proof) and all(recorded[k]==proof[k] for k in proof if k!='PID_observations'),
                'Bound completed evidence proof drift')
            require(len(recorded['PID_observations'])==len(proof['pids']),'Applied target absence roster drift')
            for row,root_row in zip(recorded['PID_observations'],proof['PID_observations']):
                stamp=datetime.fromisoformat(row['checked_utc'].replace('Z','+00:00'))
                root_stamp=datetime.fromisoformat(root_row['checked_utc'].replace('Z','+00:00'))
                require(set(row)=={'pid','absent','method','checked_utc'} and type(row['pid']) is int
                    and row['pid']==root_row['pid'] and row['absent'] is True and row['method']==root_row['method']
                    and stamp.tzinfo is not None and root_stamp<=stamp<=datetime.now(timezone.utc),
                    'Applied honest targeted absence must not predate root observation or be future')
            credit+=own['reserved_bytes']-_measurement(value['measurement'],own);closed.add(key)
        # Every historical prefix must itself have obeyed the same ceiling.
        charge(p['base_projected_bytes'],added,credit,i+1)
    projected=charge(p['base_projected_bytes'],added,credit,len(events))
    result=dict(view,projected_bytes=projected,base_pins=p['base_pins'],bindings=bindings,
        completed_overlay_prefix=dict(events=len(events),tail_sha256=hashes[-1] if hashes else None),
        completed_credit_bytes=credit,new_reserved_bytes=added,overlay_metadata_charge_bytes=BOOTSTRAP_CHARGE+len(events)*META_CHARGE,
        accounting_mode=MODE,limit_bytes=CEILING_BYTES,hard_limit_bytes=HUMAN_LIMIT_BYTES,
        ancestor_accounting_mode=BASE_MODE,ancestor_projected_bytes=view['projected_bytes'],
        ancestor_new_reserved_bytes=view['new_reserved_bytes'],ancestor_completed_credit_bytes=view['completed_credit_bytes'],
        old_active_caps_unchanged=True,old_journal_mutated=False,global_scan=False,
        **warning_fields(projected))
    return reservations,closed,result,bindings,events,hashes,reserved_at


def inspect_v1(*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):return _state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)[2]


def _append(operation,authority_ref,value,state):
    events,hashes=state[4:6];require(len(events)<MAX_EVENTS,'Overlay event cap')
    event=dict(schema='prospective_980GB_H020_event_v2',sequence=len(events)+1,previous_sha256=hashes[-1] if hashes else None,
        bindings=state[3],operation=operation,authority=authority_ref,value=value)
    raw=budget.v4.encoded(event)+b'\n';require(len(raw)<=MAX_JSON,'Event exceeds charged metadata bound')
    directory=budget.v4.canonical(ROOT,JOURNAL);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(directory/f'{len(events)+1:06d}.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    stored,sha=budget.v4.file_bytes(ROOT,directory/f'{len(events)+1:06d}.json')
    require(stored==raw,'Append uncertainty; preserve and audit, never retry blindly')
    return sha


def apply_reserve_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        auth=_authority(authority_ref,'reserve',state[3],fresh=True)
        require(auth['expected_before']==_pins(state[2]),'Fresh exact admission prefix required')
        own=auth['request'];paths=_paths(own)
        require(all(r in own['consumer_bindings'] for r in state[3].values()),'Every actual guard/policy/marker source required')
        require(own['key'] not in state[0] and own['key'] not in _policy()['never_credit_keys']
            and not any(p.exists() for p in paths),'New key and fresh exclusive paths')
        legacy=budget.v4.read_json(ROOT,budget.v4.LEGACY)[0]
        occupied=[budget.v4.canonical(ROOT,r['output']) for r in legacy['jobs'].values()]
        for reservation in state[0].values():occupied.extend(_paths(reservation))
        require(own['key'] not in legacy['jobs'] and not any(budget.v4.paths_overlap(a,b) for a in paths for b in occupied),
            'Never overlap/reuse old, closed or new reservations')
        projected=charge(_policy()['base_projected_bytes'],state[2]['new_reserved_bytes']+own['reserved_bytes'],
            state[2]['completed_credit_bytes'],len(state[4])+1)
        value=dict(copy.deepcopy(own),root_admission=copy.deepcopy(authority_ref));tail=_append('reserve',authority_ref,value,state)
        return dict(kind='concurrent_completed_scope_reserve_receipt_v1',value=value,projected_bytes=projected,
            initial_ledger_pins=dict(base_pins=state[2]['base_pins'],completed_overlay_prefix=dict(events=len(state[4])+1,tail_sha256=tail)),
            old_journal_mutated=False,**warning_fields(projected))


def apply_close_completed_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        auth=_authority(authority_ref,'close_completed',state[3],fresh=True)
        require(auth['expected_before']==_pins(state[2]),'Fresh exact credit prefix required')
        request=auth['request'];key=request['key'];assert_open(key,state[0],state[1])
        require(key not in _policy()['never_credit_keys'],'Uncreditable policy key')
        own=state[0][key];proof=_proof(request,own,observe=True)
        measurement=budget.v4.measure_scopes(ROOT,_paths(own))  # One exact completed owned-scope stat inventory.
        actual=_measurement(measurement,own)
        for pid in proof['pids']:_absent(pid)
        projected=charge(_policy()['base_projected_bytes'],state[2]['new_reserved_bytes'],
            state[2]['completed_credit_bytes']+own['reserved_bytes']-actual,len(state[4])+1)
        value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
            proof=proof,closed_utc=_utc(),future_writes_and_relaunch_revoked=True)
        tail=_append('close_completed',authority_ref,value,state)
        return dict(kind='concurrent_completed_scope_close_receipt_v1',key=key,value=value,retained_bytes=actual,
            credit_bytes=own['reserved_bytes']-actual,projected_bytes=projected,scope_measurements=1,
            completed_overlay_prefix=dict(events=len(state[4])+1,tail_sha256=tail),old_journal_mutated=False,artifacts_deleted=0,**warning_fields(projected))


def check_storage_locked_v1(storage,*,first=False,boundary=False,required_write_bytes=0):
    require(storage.get('kind')=='project_budget_v9_storage_record' and storage.get('active') is True
        and storage.get('consumer_uses_v9_guard') is True and storage.get('accounting_mode')==MODE,
        'Active new-overlay-only storage authority required')
    _expire(storage);refs=storage['budget_components'];p=_policy()
    require(refs['implementation']==p['implementation_ref'] and refs['launcher']==p['launcher_ref']
        and refs['consumer_adapter']==p['source_ref'] and storage['completed_overlay_policy_ref']==_POLICY_REF,'Exact source/launcher/policy/adapter guards')
    budget.v4.integer(required_write_bytes,'planned growth')
    require(not required_write_bytes or boundary,'Planned writes require exact owned-scope boundary')
    state=_state_locked(refs['implementation']['sha256'],refs['launcher']['sha256'])
    own=storage['reservation'];key=own['key'];assert_open(key,state[0],state[1])
    require(key in state[6] and state[0][key]==own and own['reserved_bytes']==storage['reserved_new_bytes']
        and own['output']==str(budget.v4.canonical(ROOT,storage['output_directory']))
        and all(r in own['consumer_bindings'] for r in [*refs.values(),*state[3].values()]),
        'Exact newly admitted, open source/runtime/cap/output reservation required')
    for record in own['consumer_bindings']:_raw(record)
    pins=storage['initial_ledger_pins']
    require(set(pins)=={'base_pins','completed_overlay_prefix'} and pins['base_pins']==p['base_pins'],'Exact immutable old base pins')
    check_prefix(pins['completed_overlay_prefix'],state[5],minimum_events=state[6][key])
    measurement=budget.v4.measure_scopes(ROOT,_paths(own)) if boundary else None
    if measurement:require(measurement['conservative_scope_bytes']+required_write_bytes<=own['reserved_bytes'],'Owned output exceeds fully charged new cap')
    return dict(state[2],reservation=copy.deepcopy(own),scope_measurement=measurement,required_write_bytes=required_write_bytes,
        remaining_scope_bytes=own['reserved_bytes']-measurement['conservative_scope_bytes'] if measurement else None)


def budget_check_v9(storage,*,first=False,boundary=False,required_write_bytes=0):
    with budget.v4.mutex(ROOT):return check_storage_locked_v1(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def validate_storage_v9(spec,cfg,storage,*,source_ref,config_ref,runtime_ref):
    require(storage['job_id']==spec['job_id'] and storage['output_directory']==spec['output_directory']
        and 0<storage['reserved_new_bytes']<=cfg['output_cap_bytes']
        and all(r in storage['reservation']['consumer_bindings'] for r in (source_ref,config_ref,runtime_ref)),
        'Actual producer/config/runtime/output bindings required')
    for record in (source_ref,config_ref,runtime_ref):_raw(record)
    return budget_check_v9(storage,first=True,boundary=True)


class PilotOutputV9:
    """Compatible compact writer; preserves files, including active save products."""
    def __init__(self,directory,storage,reader):
        self.path=budget.v4.canonical(ROOT,directory);self.storage=storage;self.reader=reader;self.used=0
        require(reader.root.resolve()==ROOT and self.path==ROOT/storage['output_directory'] and not self.path.exists(),'Fresh exclusive actual output')
        budget_check_v9(storage,boundary=True,required_write_bytes=8192);self.path.mkdir()
    def save(self,name,value,kind='json'):
        require(isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'Flat exclusive output name')
        buffer=io.BytesIO()
        if kind=='json':
            from arggnn.census_tsinfer_fit_v1 import _jsonable_v1
            data=budget.v4.encoded(_jsonable_v1(value))+b'\n'
        elif kind=='npz':
            import numpy as np
            np.savez_compressed(buffer,**value);data=buffer.getvalue()
        elif kind=='torch':
            import torch
            torch.save(value,buffer);data=buffer.getvalue()
        else:raise ValueError('Known json/npz/torch serialization required')
        planned=len(data)+8192;require(self.used+planned<=self.storage['reserved_new_bytes'],'Owned writer counter exceeds cap')
        budget_check_v9(self.storage,boundary=True,required_write_bytes=planned)
        require(planned<=os.statvfs(self.path).f_bavail*os.statvfs(self.path).f_frsize,'Actual filesystem free bytes required')
        path=self.path/name
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno());identity=os.fstat(f.fileno())
        named=path.lstat()
        require(stat.S_ISREG(named.st_mode) and named.st_nlink==1
            and (identity.st_dev,identity.st_ino)==(named.st_dev,named.st_ino)
            and named.st_size==len(data),'Created output pathname changed during save')
        self.used+=planned
        return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
    def discard(self,record):raise ValueError('Completed-credit overlay preserves files; discard is unsupported')


# This is a single immutable exception, never a general legacy allowlist.
LEGACY_KEY='biological_mechanism_controls_v1/seed202'
LEGACY_CAP=52321048576
LEGACY_OUTPUT=str(ROOT/'results/biological_mechanism_controls_v1_seed202')
LEGACY_AUX=str(ROOT/'.runtime/tmp/bio_mechanisms_seed202_v1')
LEGACY_RESERVATION_SHA='e6dc3f00771f4290cbcd2369f915e994c6cb635561f0c65f1fa88d0379a34a12'
LEGACY_RECEIPT_BYTES=10169131
LEGACY_RECEIPT_REF=dict(path='results/biological_mechanism_controls_v1_seed202/receipt.json',sha256='271ada740e7b5804129b3e44b2599748658374693635cd11b97d8585bfcfa7b8')
LEGACY_ROOT='experiments/results/biological_mechanisms_seed202_parallel_v1_root/'
LEGACY_REFS={
 'reserve':dict(path=LEGACY_ROOT+'reserve_receipt.json',sha256='9f8e14af3b4c9e48c4f2414c278121d41d209f55a565404bbbc38a73964ffc74'),
 'launch_admission':dict(path=LEGACY_ROOT+'launch_admission.json',sha256='2f68295741c679a4f26b995cbdba50f0e838acff9f60743a5c6560d735957254'),
 'launch':dict(path=LEGACY_ROOT+'launch.json',sha256='6d9591143204ad9e6580a43b245489d5e4cd3da554d7b36458bc4500d519c8e2'),
 'direct_wait':dict(path=LEGACY_ROOT+'exit.json',sha256='4786f2f68e3ed95582be6aaf868db9b36bf70e10839ce3e6d2935ba4c21708bf'),
 'supervisor_source':dict(path=LEGACY_ROOT+'launch_supervisor.py',sha256='6183a08441c0321b8fa6c524fe3b86f35102bc8390fb2e95637fb083b96ba13f'),
 'budget_launcher_source':dict(path='src/arggnn/budgeted_gpu_launcher_v1.py',sha256='d7eeaa568685ffc367242010a7035b6cc0a868493fa7dd279914189ab47358ff'),
 'gpu_launcher_source':dict(path='scripts/launch_gpu.py',sha256='f07324053d8e649e27d66cbca9799bb6ea237c92c247853e2d45e6d440a2c094'),
 'runner_source':dict(path='scripts/train_biological_mechanism_controls_v1.py',sha256='f19bdead17a135a42ad70513f1522cef2a9fd58dc159d04f189adbe08de67dec'),
 'producer_source':dict(path='src/arggnn/biological_mechanism_controls_v1.py',sha256='28f3e7d3f28a48eeee0944b2e4ad55e3ea6ebb314d8b0342bd785facb08dcbeb'),
 'config':dict(path='configs/biological_mechanism_controls_v1.json',sha256='67d7aba454d3cc57b11faa32151dd6af397210fa89d9b56b5fcd0014f662f1b4'),
 'concurrency':dict(path=LEGACY_ROOT+'concurrency_verified.json',sha256='ec97b4cd4a9e4689b1857a4628a1d9e237c07b4d46faf70032ae2ed47705f37f'),
 'source_roster_audit':dict(path='research/reviews/H020_legacy_write_roster_source_audit_20261016_v1/REPORT.md',sha256='ddc883a84c4be4edc2f6f2604007f5291b4d5db39bb0350f45a98e977dceae68'),
}
LEGACY_WRITERS=[
 dict(role='root_supervisor',pid=1230669,start_ticks=435395390),
 dict(role='budget_gpu_launcher',pid=1230670,start_ticks=435395396),
 dict(role='sole_seven_model_fit_writer',pid=1230680,start_ticks=435395427),
]
LEGACY_GROUPS=[1230670,1230680]  # source-derived child sessions, not recorded group observations
LEGACY_STEPS={
 'bio_mc_v1_scalar_age_d2':352,'bio_mc_v1_age_propagation_only_d2':352,
 'bio_mc_v1_age_source_only_d2':352,'bio_mc_v1_age_gainmatched_d2':480,
 'bio_mc_v1_noncontext_age_d2':352,'bio_mc_v1_age_d1':480,'bio_mc_v1_age_d1_plus_pointwise':352,
}


def consumer_bindings_v1(*groups):
    """Deduplicate exact refs only. Never omit guards or raise original 64-ref cap."""
    result=[]
    for group in groups:
        require(isinstance(group,(list,tuple)),'Explicit source reference sequence required')
        for record in group:
            _raw(record)
            if record not in result:result.append(copy.deepcopy(record))
    require(1<=len(result)<=64,'All required direct bindings must fit original 64-ref cap, otherwise consolidate real source')
    return result


def legacy_observation_draft_v1():
    """Inert schema only. ROOT must replace empty observations with actual fresh facts."""
    return dict(schema='prospective980_H020_seed202_targeted_root_observation_v1',root_authorizer='root_agent',active=False,
        key=LEGACY_KEY,reservation_sha256=LEGACY_RESERVATION_SHA,
        original_completion_ref=LEGACY_RECEIPT_REF,original_launch_ref=LEGACY_REFS['launch'],
        original_direct_wait_ref=LEGACY_REFS['direct_wait'],source_roster_audit_ref=LEGACY_REFS['source_roster_audit'],
        source_determined_scoped_writer_roots=copy.deepcopy(LEGACY_WRITERS),source_derived_session_groups=LEGACY_GROUPS[:],
        all_source_determined_scoped_writer_roots_listed=True,
        roster_is_cooperative_pinned_route_not_complete_historical_descendant_inventory=True,
        original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',
        actual_original_direct_launcher_wait0=True,
        all_original_scoped_write_resume_relaunch_authorities_permanently_cooperatively_revoked=True,
        preserve_all_original_files=True,external_NFS_writers_outside_observation=True,
        PID_observations=[],group_observations=[])


def legacy_request_draft_v1(observation_ref):
    return dict(key=LEGACY_KEY,reservation_sha256=LEGACY_RESERVATION_SHA,observation_ref=copy.deepcopy(observation_ref))


def _stamp(text):
    require(isinstance(text,str),'Timezone-aware evidence timestamp required')
    stamp=datetime.fromisoformat(text.replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'Nonfuture timezone-aware timestamp')
    return stamp


def _legacy_observations(value,*,fresh):
    expected=legacy_observation_draft_v1();expected['active']=True
    require(set(value)==set(expected),'Exact one-key ROOT roster/revocation certificate schema')
    require(all(budget.v4.encoded(value[k])==budget.v4.encoded(expected[k]) for k in expected if k not in ('PID_observations','group_observations')),
        'Exact original writer identities and honest unknown history required')
    rows=value['PID_observations'];groups=value['group_observations']
    require(isinstance(rows,list) and len(rows)==3 and isinstance(groups,list) and len(groups)==2,
        'Exact three writer roots and two source-derived child sessions required')
    for identity,row in zip(LEGACY_WRITERS,rows):
        require(set(row)=={'pid','start_ticks','absent','method','checked_utc'}
            and type(row['pid']) is int and row['pid']==identity['pid']
            and type(row['start_ticks']) is int and row['start_ticks']==identity['start_ticks']
            and row['absent'] is True and row['method']=='targeted os.kill(pid,0) ESRCH',
            'Exact PID/start identity and conservative ESRCH absence required')
        stamp=_stamp(row['checked_utc'])
        if fresh:require((datetime.now(timezone.utc)-stamp).total_seconds()<=300,'ROOT PID absence certificate older than five minutes')
    for pgid,row in zip(LEGACY_GROUPS,groups):
        require(set(row)=={'pgid','absent','method','checked_utc'} and type(row['pgid']) is int and row['pgid']==pgid
            and row['absent'] is True and row['method']=='targeted os.killpg(pgid,0) ESRCH',
            'Exact source-derived child-session group ESRCH required')
        stamp=_stamp(row['checked_utc'])
        if fresh:require((datetime.now(timezone.utc)-stamp).total_seconds()<=300,'ROOT group absence certificate older than five minutes')
    return rows,groups


def _legacy_pid_absent(identity):
    row=_absent(identity['pid'])
    return dict(pid=row['pid'],start_ticks=identity['start_ticks'],absent=True,method=row['method'],checked_utc=row['checked_utc'])


def _legacy_group_absent(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return dict(pgid=pgid,absent=True,method='targeted os.killpg(pgid,0) ESRCH',checked_utc=_utc())
    except PermissionError:raise ValueError('Targeted original session group absence cannot be certified') from None
    raise ValueError('Original session group still present or reused, no legacy credit')


def _legacy_proof(request,own,*,observe=False):
    require(set(request)=={'key','reservation_sha256','observation_ref'} and request['key']==LEGACY_KEY
        and request['reservation_sha256']==LEGACY_RESERVATION_SHA==budget.digest(own),
        'Exact one-key original reservation digest required')
    require(own['key']==LEGACY_KEY and own['reserved_bytes']==LEGACY_CAP and own['output']==LEGACY_OUTPUT
        and own['auxiliary_scopes']==[LEGACY_AUX],'Original full charge/output/auxiliary scopes cannot change')
    for ref in LEGACY_REFS.values():_raw(ref)
    reserve=_json(LEGACY_REFS['reserve']);admission=_json(LEGACY_REFS['launch_admission'])
    require(reserve['admitted_reservations'][LEGACY_KEY]==own and admission['reservation']==own,
        'Authentic original stored reservation and launch binding required, no aliases')
    launch=_json(LEGACY_REFS['launch']);wait=_json(LEGACY_REFS['direct_wait'])
    for value in (launch,wait):
        require(value['supervisor_pid']==1230669 and value['supervisor_start_ticks']==435395390
            and value['launcher_pid']==1230670 and value['launcher_start_ticks']==435395396,
            'Actual original supervisor/launcher PID/start identities required')
    require(wait['status']=='terminal' and type(wait['returncode']) is int and wait['returncode']==0
        and wait['watchdog_triggered'] is False and wait['command']==launch['command'],
        'Authentic original direct launcher wait0 with no watchdog required')
    worker=[r for r in _json(LEGACY_REFS['concurrency'])['workers'] if r.get('seed')==202]
    require(len(worker)==1 and worker[0]['pid']==1230680 and worker[0]['start_ticks']==435395427,
        'Contemporaneous sole scientific writer PID/start identity required')
    cfg=_json(LEGACY_REFS['config'])
    for path,sha in cfg['source_sha256'].items():_raw(dict(path=path,sha256=sha))
    raw=_raw(LEGACY_RECEIPT_REF)
    require(len(raw)==LEGACY_RECEIPT_BYTES,'Exact original 10,169,131-byte receipt required')
    completion=budget.v4.parse(raw)
    require(completion['kind']=='biological_mechanism_controls_fit_v1'
        and completion['status']=='completed_seven_controls_one_seed' and type(completion['seed']) is int
        and completion['seed']==202 and completion['config_sha256']==LEGACY_REFS['config']['sha256']
        and completion['source_sha256']==cfg['source_sha256'] and set(completion['models'])==set(LEGACY_STEPS)
        and completion['GPU_qualification']['pid']==1230680,'Actual original completed seven-control producer receipt required')
    fit_rows=[]
    for name,steps in LEGACY_STEPS.items():
        row=completion['models'][name]
        require(row['status']=='completed' and row['optimizer_fit_completed'] is True
            and type(row['optimizer_steps']) is int and row['optimizer_steps']==steps and steps>0,
            'Every original model must have its authentic completed positive optimizer fit')
        fit_rows.append(dict(model=name,optimizer_steps=steps,status='completed',optimizer_fit_completed=True))
    observation=_json(request['observation_ref']);pids,groups=_legacy_observations(observation,fresh=observe)
    if observe:
        pids=[_legacy_pid_absent(row) for row in LEGACY_WRITERS]
        groups=[_legacy_group_absent(pgid) for pgid in LEGACY_GROUPS]
    return dict(schema='prospective980_H020_original_receipt_direct_wait_proof_v1',
        root_observation_ref=request['observation_ref'],original_evidence_refs=copy.deepcopy(LEGACY_REFS),
        original_completion_ref=LEGACY_RECEIPT_REF,original_completion_bytes=LEGACY_RECEIPT_BYTES,
        original_completion_status='completed_seven_controls_one_seed',original_completed_fits=fit_rows,
        source_determined_scoped_writer_roots=copy.deepcopy(LEGACY_WRITERS),source_derived_session_groups=LEGACY_GROUPS[:],
        applied_PID_observations=copy.deepcopy(pids),applied_group_observations=copy.deepcopy(groups),
        original_direct_launcher_wait0=True,original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',
        future_original_scoped_write_resume_relaunch_authority_permanently_cooperatively_revoked=True,
        marker_manufactured=False,old_scope_mutated=False)


def _legacy_recorded_proof(recorded,expected,closed_utc):
    excluded={'applied_PID_observations','applied_group_observations'}
    require(set(recorded)==set(expected) and all(recorded[k]==expected[k] for k in expected if k not in excluded),
        'Typed original receipt/direct-wait/roster proof drift')
    closed=_stamp(closed_utc)
    for key in excluded:
        rows=recorded[key];root_rows=expected[key]
        require(isinstance(rows,list) and len(rows)==len(root_rows),'Applied exact legacy absence roster drift')
        for row,root_row in zip(rows,root_rows):
            require(set(row)==set(root_row) and all(row[k]==root_row[k] for k in row if k!='checked_utc'),
                'Applied legacy absent identity or method drift')
            require(_stamp(root_row['checked_utc'])<=_stamp(row['checked_utc'])<=closed,
                'Applied legacy absence predates ROOT observation or follows close')


def apply_retire_H020_seed202_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    """ROOT only. One exact proof and original owned-scope measure, no old writes."""
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        auth=_authority(authority_ref,'retire_H020_seed202_v1',state[3],fresh=True)
        require(auth['expected_before']==_pins(state[2]),'Fresh exact v2 prefix required')
        request=auth['request'];require(request['key']==LEGACY_KEY,'Only exact legacy H020 key')
        assert_open(LEGACY_KEY,state[0],state[1]);own=state[0][LEGACY_KEY]
        proof=_legacy_proof(request,own,observe=True)
        measurement=budget.v4.measure_scopes(ROOT,_paths(own))
        actual=_measurement(measurement,own)
        proof['applied_PID_observations']=[_legacy_pid_absent(identity) for identity in LEGACY_WRITERS]
        proof['applied_group_observations']=[_legacy_group_absent(pgid) for pgid in LEGACY_GROUPS]
        projected=charge(_policy()['base_projected_bytes'],state[2]['new_reserved_bytes'],
            state[2]['completed_credit_bytes']+LEGACY_CAP-actual,len(state[4])+1)
        value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
            proof=proof,closed_utc=_utc(),future_writes_and_relaunch_revoked=True)
        _legacy_recorded_proof(proof,_legacy_proof(request,own),value['closed_utc'])
        tail=_append('retire_H020_seed202_v1',authority_ref,value,state)
        return dict(kind='prospective980_H020_legacy_retirement_receipt_v1',key=LEGACY_KEY,value=value,
            retained_bytes=actual,credit_bytes=LEGACY_CAP-actual,projected_bytes=projected,scope_measurements=1,
            completed_overlay_prefix=dict(events=len(state[4])+1,tail_sha256=tail),
            old_journal_mutated=False,old_scope_mutated=False,artifacts_deleted=0,
            original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',**warning_fields(projected))


if __name__=='__main__':
    print(json.dumps(dict(status='source_only_no_configuration_or_state_read',source_ref=_ref(SOURCE),base_ref=BASE_REF,
        journal=JOURNAL,accounting_mode=MODE,live_operations_performed=False),indent=2))
