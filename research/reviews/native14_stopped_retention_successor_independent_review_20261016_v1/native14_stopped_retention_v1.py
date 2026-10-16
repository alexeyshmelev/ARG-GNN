"""Exact historical14GiB stopped/native retention successor over future frozen Census child.

Inactive source only until independently reviewed ROOT activation selects actual
fresh parent pins/projection. Existing admitted parent consumers remain unchanged.
No completion, historical drain, deletion, global rebase or capacity is invented.
"""
from __future__ import annotations
import copy,hashlib,importlib.util,io,json,os,stat
from datetime import datetime,timezone
from pathlib import Path
from scripts import project_budget_v9 as budget

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/native14_stopped_retention_successor_source_20261016_v1/native14_stopped_retention_v1.py'
JOURNAL='.runtime/storage/native14_stopped_retention_v1'
MODE='native14_stopped_retention_v1'
BASE_REF={'path': 'research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py', 'sha256': 'fab6bb67bc1f4be989df9bb16e7a3a17d24652ca8238617ca060a56cb8c89d43'}
BASE_POLICY_REF={'path': 'research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/activation_policy.root_active_v1.json', 'sha256': '72ff346c43f96ba8430411cfe6b2250480a3aa66fa8e3b8c904a625a0399932a'}
BASE_MODE='prospective_980gb_Census_fit2_v1'
REPAIR_REF=dict(path='research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py',sha256='e4185dc51b285d8ad4eba224b8e1e15f3e1479f57c73b4ec53fff2b238d0ef2f')
CAPACITY_V1_REF=dict(path='research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py',sha256='7c0a35f82423eb824edf3b7dccbac31f139d50d2ee397d4d92515b6a0ae33c88')
MARKER_REF=dict(path='src/arggnn/project_budget_v9_consumer.py',sha256='c11a5543746899c8a387508e9877ce6ae2bf80fe4d3076c1dbe1cb8f7a43b859')
RENEWAL_REF=dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py',sha256='491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468')
RENEWAL_CERT_REF=dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json',sha256='66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7')
AUDIT_REF={'path': 'research/reviews/old_native14GiB_retention_evidence_20261016_v1/evidence_audit.v1.json', 'sha256': '45721fc2cee4e048e151c28a5cfe488ad0a4335ec25cfa7c8b47afc91c0f0f65'}
WARNING_BYTES=950000000000
CEILING_BYTES=980000000000
HUMAN_LIMIT_BYTES=1000000000000
MALFORMED_KEY='H039/directed-clade-exchange-train-stage-seed101-v1'
META_CHARGE=524288
BOOTSTRAP_CHARGE=2097152
MAX_EVENTS=4096
MAX_JSON=65536
MAX_COMPLETED_JSON=1048576
COMPLETED_SCHEMAS=frozenset(('expanded_native_ap_driver_completed_v2','expanded_native_argmix_preparation_v1_root_admission_completed','expanded_pyg_observations_v5_completed'))
RETIRE_OPERATION='retire_exact_stopped_revoked_native_MLM_retention_v1'
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

def root_activation_draft_v1(*,base_pins,base_projected_bytes,permanent_revocation_ref=None,source_review_ref=None):
    return dict(schema="native14_root_actual_frozen_Census_activation_v1",root_authorizer="root_agent",active=False,
        source_ref=_ref(SOURCE),source_manifest_ref=_ref(MANIFEST),measurement_source_ref=_ref(MEASURE_SOURCE),
        frozen_parent_ref=BASE_REF,frozen_parent_policy_ref=BASE_POLICY_REF,
        frozen_parent_pins=copy.deepcopy(base_pins),frozen_parent_projected_bytes=base_projected_bytes,
        permanent_revocation_ref=permanent_revocation_ref,source_review_ref=source_review_ref,new_journal=JOURNAL,
        only_stopped_retention_key=NATIVE_KEY,
        actual_parent_prefix_and_projection_selected_under_original_mutex=True,
        all_future_parent_reserve_close_retire_mutations_cooperatively_revoked=True,
        all_older_accounting_future_mutation_revocations_preserved=True,
        current_parent_admitted_C3_Relate_native7GiB_writers_continue_unchanged_with_full_original_caps=True,
        old_parent_source_policy_journal_and_consumer_storage_records_unchanged=True,
        ROOT_accepts_exact_unexpired_stopped_revocation_branch_and_residual_historical_NFS_uncertainty=True,
        no_current_credit_or_capacity_implied_by_activation=True,no_historical_mutation=True,
        no_deletion_or_rename_or_normalization=True,no_rebase=True,physical_GPU_ids=[0])


def activation_policy_draft_v1(*,repair_authority_ref,base_pins,base_projected_bytes,implementation_ref,launcher_ref,
        root_activation_ref=None,permanent_revocation_ref=None,source_review_ref=None):
    return dict(schema="native14_exact_stopped_retention_policy_v1",root_authorizer="root_agent",active=False,
        source_ref=_ref(SOURCE),source_manifest_ref=_ref(MANIFEST),measurement_source_ref=_ref(MEASURE_SOURCE),
        base_ref=BASE_REF,base_policy_ref=BASE_POLICY_REF,base_accounting_mode=BASE_MODE,
        repair_ref=REPAIR_REF,capacity_v1_ref=CAPACITY_V1_REF,marker_ref=MARKER_REF,
        marker_repair_authority_ref=repair_authority_ref,implementation_ref=implementation_ref,launcher_ref=launcher_ref,
        renewal_ref=RENEWAL_REF,renewal_certificate_ref=RENEWAL_CERT_REF,evidence_audit_ref=AUDIT_REF,
        sealed_evidence_handoff_ref=EVIDENCE_SEAL_REF,framed_wait_extracts_ref=WAIT_EXTRACT_REF,
        base_pins=copy.deepcopy(base_pins),base_projected_bytes=base_projected_bytes,journal=JOURNAL,
        root_activation_ref=root_activation_ref,permanent_revocation_ref=permanent_revocation_ref,source_review_ref=source_review_ref,
        scheduling_ceiling_bytes=CEILING_BYTES,warning_threshold_bytes=WARNING_BYTES,human_limit_bytes=HUMAN_LIMIT_BYTES,
        bootstrap_metadata_charge_bytes=BOOTSTRAP_CHARGE,per_event_metadata_charge_bytes=META_CHARGE,
        all_old_future_admissions_revoked=True,all_future_admissions_use_this_overlay=True,
        old_active_writers_continue_unchanged_fully_charged=True,old_journals_frozen_for_new_events=True,
        all_budget_writers_use_original_mutex=True,closed_scope_all_write_resume_capabilities_revoked=True,
        no_historical_mutation=True,no_rebase=True,no_deletion_or_rename_or_normalization=True,
        external_NFS_writers_outside_observation=True,new_unscoped_metadata_within_charged_allowances=True,
        never_credit_keys=[MALFORMED_KEY,NATIVE_KEY],only_stopped_retention_keys=[NATIVE_KEY],
        unexpired_original_authority_accepted_only_with_exact_permanent_ROOT_revocation=True,
        failed_retirements_are_successful_completion=False,all_historical_descendants_or_old_writer_absence_claimed=False,
        original_v8_native_exit="UNKNOWN",historical_whole_descendant_drain="UNKNOWN",optimizer_updates="UNKNOWN",
        eligible_physical_GPU_ids=[0],all_nonzero_physical_GPU_ids_excluded=True,
        actual_GPU_assignment_and_GPU_launcher_guards_owned_by_root=True)


def configure_v2(repair_authority_ref,*,policy_ref=None):
    global _BASE,_POLICY_REF,_REPAIR_AUTHORITY_REF
    if policy_ref is None:
        policy_ref=json.loads(os.environ['ARGGNN_COMPLETED_CREDITS_POLICY_REF'])
    require(_POLICY_REF is None or _POLICY_REF==policy_ref,'No policy swap inside a configured consumer')
    for r in (BASE_REF,BASE_POLICY_REF,REPAIR_REF,CAPACITY_V1_REF,MARKER_REF,RENEWAL_REF,RENEWAL_CERT_REF,AUDIT_REF):_raw(r)
    _POLICY_REF=copy.deepcopy(policy_ref);_REPAIR_AUTHORITY_REF=copy.deepcopy(repair_authority_ref)
    _policy()
    sp=importlib.util.spec_from_file_location('_native14_future_frozen_Census_child',ROOT/BASE_REF['path'])
    _BASE=importlib.util.module_from_spec(sp);sp.loader.exec_module(_BASE)
    _BASE.configure_v2(repair_authority_ref,policy_ref=BASE_POLICY_REF)

def configure_with_renewal_v1(repair_authority_ref,*,policy_ref):
    """ROOT fresh interpreter, before any guard/memo configuration; never called by author."""
    _raw(RENEWAL_REF);_raw(RENEWAL_CERT_REF)
    sp=importlib.util.spec_from_file_location('_Census_fit2_original_renewal',ROOT/RENEWAL_REF['path'])
    renewal=importlib.util.module_from_spec(sp);sp.loader.exec_module(renewal)
    renewal.install_renewal_v1(RENEWAL_CERT_REF)
    return configure_v2(repair_authority_ref,policy_ref=policy_ref)

def _policy():
    require(_POLICY_REF is not None,"ROOT must configure exact active successor policy")
    p=_json(_POLICY_REF)
    expected=activation_policy_draft_v1(repair_authority_ref=_REPAIR_AUTHORITY_REF,
        base_pins=p.get("base_pins"),base_projected_bytes=p.get("base_projected_bytes"),
        implementation_ref=p.get("implementation_ref"),launcher_ref=p.get("launcher_ref"),
        root_activation_ref=p.get("root_activation_ref"),permanent_revocation_ref=p.get("permanent_revocation_ref"),
        source_review_ref=p.get("source_review_ref"))
    expected["active"]=True
    require(set(p)==set(expected) and budget.v4.encoded(p)==budget.v4.encoded(expected),"Exact inactive-until-ROOT source/policy/frozen-parent contract")
    for r in (p["source_ref"],p["source_manifest_ref"],p["measurement_source_ref"],p["base_ref"],p["base_policy_ref"],
        p["repair_ref"],p["capacity_v1_ref"],p["marker_ref"],p["marker_repair_authority_ref"],p["implementation_ref"],
        p["launcher_ref"],p["renewal_ref"],p["renewal_certificate_ref"],p["evidence_audit_ref"],
        p["sealed_evidence_handoff_ref"],p["framed_wait_extracts_ref"],p["source_review_ref"]):_raw(r)
    require(p["implementation_ref"]["path"]==budget.SOURCE_REL and p["launcher_ref"]["path"]==budget.LAUNCHER_REL,"Original actual budget implementation/launcher")
    budget.v4.integer(p["base_projected_bytes"],"Future ROOT-selected actual Census parent projection")
    require(isinstance(p["base_pins"],dict) and set(p["base_pins"])=={"base_pins","completed_overlay_prefix"},"Future actual current Census prefix, not a saved illustrative prefix")
    activation=_json(p["root_activation_ref"])
    required=root_activation_draft_v1(base_pins=p["base_pins"],base_projected_bytes=p["base_projected_bytes"],
        permanent_revocation_ref=p["permanent_revocation_ref"],source_review_ref=p["source_review_ref"])
    required["active"]=True
    require(set(activation)==set(required) and budget.v4.encoded(activation)==budget.v4.encoded(required),"Explicit future ROOT frozen-parent activation")
    review=_json(p["source_review_ref"])
    require(review.get("verdict") in ("PASS","SOURCE_PASS") and review.get("source_ref")==_ref(SOURCE)
        and review.get("source_manifest_ref")==_ref(MANIFEST) and review.get("measurement_source_ref")==_ref(MEASURE_SOURCE),
        "Independent complete source/measurement/manifest review")
    _native_revocation(p["permanent_revocation_ref"]);return p


def _bindings(base_bindings,p):
    result=copy.deepcopy(base_bindings);m=_native_manifest()
    additions=dict(native14_source=_ref(SOURCE),native14_manifest=_ref(MANIFEST),native14_retained_measurement=_ref(MEASURE_SOURCE),
        native14_policy=_POLICY_REF,native14_root_activation=p["root_activation_ref"],
        native14_permanent_revocation=p["permanent_revocation_ref"],native14_source_review=p["source_review_ref"])
    for i,r in enumerate(m["original_reservation"]["consumer_bindings"]):
        additions["native14_original_consumer_%02d"%i]=r
    require(not set(result)&set(additions),"Never overwrite/drop an ancestor or historical direct binding")
    result.update(additions)
    consumer_bindings_v1(list(result.values()))
    return result


def operation_draft_v1(operation,request,*,expected_before,bindings):
    require(operation in ("reserve","close_completed",RETIRE_OPERATION),"Ordinary reserve/close or one exact stopped-native retention operation")
    return dict(schema="native14_exact_root_operation_v1",root_authorizer="root_agent",active=False,
        operation=operation,accounting_mode=MODE,bindings=copy.deepcopy(bindings),expected_before=copy.deepcopy(expected_before),
        expires_at_utc=None,request=copy.deepcopy(request),all_budget_writers_use_original_mutex=True,
        all_old_future_admissions_revoked=True,all_future_consumers_use_this_overlay=True,
        historical_mutation_authorized=False,global_rebase_authorized=False,deletion_authorized=False,
        scientific_execution_authorized=False,GPU_authorized=False)




def _paths(own):
    require(_BASE is not None,'Configured immutable base required')
    # Preserve every frozen ancestor scope/marker rule and 64-direct-binding cap.
    paths=_BASE._paths(own)
    protected=[ROOT/JOURNAL,ROOT/SOURCE,budget.v4.canonical(ROOT,_POLICY_REF['path'])]
    require(not any(budget.v4.paths_overlap(a,b) for a in paths for b in protected),'Product scope overlaps overlay accounting/source/policy')
    return paths

def _pins(view):
    return dict(base_pins=copy.deepcopy(view['base_pins']),completed_overlay_prefix=copy.deepcopy(view['completed_overlay_prefix']))

def _authority(record,operation,bindings,*,fresh):
    value=_json(record);expected=operation_draft_v1(operation,value.get('request'),expected_before=value.get('expected_before'),bindings=bindings)
    expected.update(active=True,expires_at_utc=value.get('expires_at_utc'))
    require(set(value)==set(expected) and value==expected,'Exact bound root operation authority required')
    if fresh:
        require(all(getattr(budget.v4,n).__module__==budget.v4.__name__ for n in ('canonical','file_bytes','mutex')),
            'Fresh original unwrapped ROOT mutation interpreter required')
        _expire(value)
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
            and event['schema']=='native14_exact_stopped_retention_event_v1' and type(event['sequence']) is int and event['sequence']==i
            and event['previous_sha256']==(hashes[-1] if hashes else None) and event['bindings']==bindings
            and event['operation'] in ('reserve','close_completed',RETIRE_OPERATION),'Exact immutable new-journal chain/bindings')
        events.append(event);hashes.append(sha)
    after=directory.lstat()
    require((before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns)==
        (after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns),'Journal changed during locked read')
    return events,hashes

def _state_locked(source_sha,launcher_sha,*,fresh=False):
    p=_policy();require(_BASE is not None,'Configure root base first')
    require((source_sha,launcher_sha)==(p['implementation_ref']['sha256'],p['launcher_ref']['sha256']),'Actual original budget source/launcher pins')
    base=_BASE._state_locked(source_sha,launcher_sha,fresh=fresh)
    require(len(base)==7,'Exact frozen Census child seven-part state API')
    reservations,base_closed,view,base_bindings,_,_,_=base
    require(_BASE._pins(view)==p['base_pins'] and view['projected_bytes']==p['base_projected_bytes']
        and view['accounting_mode']==BASE_MODE and view['limit_bytes']==CEILING_BYTES
        and view['hard_limit_bytes']==HUMAN_LIMIT_BYTES,
        'Future frozen Census parent prefix/projection changed; no new charge or credit')
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
        elif event['operation']==RETIRE_OPERATION:
            key=request["key"];require(key==NATIVE_KEY,"Only exact historical14GiB stopped key")
            assert_open(key,reservations,closed);own=reservations[key]
            proof=_native_proof(request,own)
            require(set(value)=={"request","reservation_sha256","measurement","proof","closed_utc","future_writes_and_relaunch_revoked"}
                and value["request"]==request and value["reservation_sha256"]==budget.digest(own)
                and value["future_writes_and_relaunch_revoked"] is True,"Exact stopped-retention certificate")
            _native_recorded_proof(value["proof"],proof,value["closed_utc"])
            credit+=own["reserved_bytes"]-_native_measurement(value["measurement"],own);closed.add(key)
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
    event=dict(schema='native14_exact_stopped_retention_event_v1',sequence=len(events)+1,previous_sha256=hashes[-1] if hashes else None,
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



def _stamp(text):
    require(isinstance(text,str),'Timezone-aware evidence timestamp required')
    value=datetime.fromisoformat(text.replace('Z','+00:00'))
    require(value.tzinfo is not None and value<=datetime.now(timezone.utc),'Nonfuture evidence timestamp')
    return value

































MANIFEST='research/reviews/native14_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json'
MEASURE_SOURCE='research/reviews/native14_stopped_retention_successor_source_20261016_v1/retained_nodes_v1.py'
NATIVE_KEY='H041/American-C3-native-MLM-full30mb-mlm-fit-v8'
NATIVE_RESERVED=15032385536
NATIVE_EXPIRY='2026-10-18T11:47:27.217781+00:00'
NATIVE_ROOTS=['/workspace-SR003.nfs2/ARG-GNN/experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8', '/workspace-SR003.nfs2/ARG-GNN/research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/worker_aux', '/workspace-SR003.nfs2/ARG-GNN/.runtime/tmp/c3full8/f']
EVIDENCE_SEAL_REF={'path': 'research/reviews/old_native14GiB_retention_evidence_20261016_v1/source_handoff.v1.json', 'sha256': 'adb016daf477f352e0d32fed705fa6af50f5e3067bb1db39cd7078ccc75b5bd2'}
WAIT_EXTRACT_REF={'path': 'research/reviews/old_native14GiB_retention_evidence_20261016_v1/framed_wait_extracts.v1.json', 'sha256': 'e70b5532fe4babac653748941795d6f6c8e86ffa1136280f390945aea564717d'}

def _native_manifest():
    m=_completed_json(_ref(MANIFEST))
    require(m["schema"]=="native14_stopped_retention_source_manifest_v1"
        and m["exact_key"]==NATIVE_KEY and m["original_reserved_bytes"]==NATIVE_RESERVED
        and m["original_scope_paths"]==NATIVE_ROOTS and m["evidence_audit_ref"]==AUDIT_REF
        and m["sealed_evidence_handoff_ref"]==EVIDENCE_SEAL_REF
        and m["framed_wait_extracts_ref"]==WAIT_EXTRACT_REF,"Exact sealed source/evidence/three-parent contract")
    for r in m["runtime_verified_source_and_compact_metadata_refs"]:_raw(r)
    return m

def _native_helper():
    r=_ref(MEASURE_SOURCE);_raw(r)
    spec=importlib.util.spec_from_file_location("_native14_retained_readonly",ROOT/MEASURE_SOURCE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def stopped_revocation_draft_v1():
    return dict(schema="native14_exact_stopped_permanent_ROOT_revocation_v1",root_authorizer="root_agent",active=False,
        key=NATIVE_KEY,reservation_sha256=_native_manifest()["original_reservation_sha256"],
        source_manifest_ref=_ref(MANIFEST),original_scope_paths=NATIVE_ROOTS[:],original_expiry_unchanged=NATIVE_EXPIRY,
        revoked_at_utc=None,all_old_original_v8_v9_v10_v11_v12_launch_resume_write_routes_for_exact_key_permanently_cooperatively_revoked=True,
        all_original_supervisor_worker_driver_CLI_aliases_and_future_routes_for_exact_key_permanently_cooperatively_revoked=True,
        old_frozen_sources_do_not_mechanically_consult_new_revocation=True,
        ROOT_explicitly_accepts_historical_descendant_incompleteness_and_cooperative_basis=True,
        external_NFS_writers_outside_observation=True,historical_whole_descendant_drain="UNKNOWN",
        original_v8_native_exit="UNKNOWN",optimizer_updates="UNKNOWN",scientific_success_asserted=False,
        original_authority_not_marked_expired=True,new_certificate_does_not_modify_old_authorities=True,
        current_live_C3_Relate_native7GiB_caps_guards_and_writes_unchanged=True,
        preserve_all_files_and_scope_namespace=True,no_delete_rename_normalize_or_payload_read_authorized=True,
        no_other_key_retired=True)

def _native_revocation(record):
    v=_json(record);expected=stopped_revocation_draft_v1()
    expected.update(active=True,revoked_at_utc=v.get("revoked_at_utc"))
    require(set(v)==set(expected) and budget.v4.encoded(v)==budget.v4.encoded(expected),"Exact scoped permanent ROOT revocation/residual acceptance")
    require(_stamp(_completed_json(AUDIT_REF)["observed_utc"])<=_stamp(v["revoked_at_utc"]),"New permanent ROOT revocation cannot predate sealed stop/failure audit")
    return v

def _native_origin(own):
    m=_native_manifest();require(own==m["original_reservation"] and budget.digest(own)==m["original_reservation_sha256"]
        and own["key"]==NATIVE_KEY and own["reserved_bytes"]==NATIVE_RESERVED
        and [str(p) for p in _paths(own)]==NATIVE_ROOTS,"Exact original14GiB key, reservation and complete three parents")
    audit=_completed_json(AUDIT_REF);seal=_json(EVIDENCE_SEAL_REF);frames=_completed_json(WAIT_EXTRACT_REF)
    require(audit["original"]["reservation"]==own and audit["original"]["expires_at_utc"]==NATIVE_EXPIRY
        and audit["actual_stop"]["actual_tool_session"]==48073 and audit["actual_stop"]["actual_outer_tool_exit"]==1
        and audit["actual_stop"]["native_wait"]==audit["actual_stop"]["entry_wait"]==audit["actual_stop"]["launcher_wait"]==-15
        and audit["actual_stop"]["terminal_consumed"] is True
        and audit["normalization"]["subsequent_normalization_success_receipt_present"] is False
        and audit["lineage"]["v8"]["native_exit"]==audit["lineage"]["v8"]["optimizer_updates"]=="UNKNOWN"
        and seal["actual_retirement_or_credit"] is False,"Genuine failed stop, original unknown history and incomplete normalization")
    storage=_json(audit["original"]["storage_ref"])
    require(storage["reservation"]==own and storage["reserved_new_bytes"]==NATIVE_RESERVED
        and storage["expires_at_utc"]==NATIVE_EXPIRY,"Original authority remains unchanged; no expiry prerequisite invented")
    stop=_json(audit["actual_stop"]["root_actual_transition_ref"]);signal=_json(audit["actual_stop"]["signal_receipt_ref"])
    require(stop["native_tool"]["session"]==48073 and stop["native_tool"]["actual_exit_code"]==1
        and stop["native_tool"]["terminal_consumed"] is True
        and stop["native_tool"]["actual_native_wait"]==stop["native_tool"]["actual_entry_wait"]==stop["native_tool"]["actual_launcher_wait"]==-15
        and stop["native_tool"]["native_full_budget_completed"] is False
        and signal["native_supervisor_identity"]==audit["actual_stop"]["supervisor_identity"]
        and signal["exact_identity_revalidated"] is True and signal["signal_sent_through_pidfd"] is True
        and signal["signal"]=="SIGTERM","Actual exact pidfd stop and ROOT-consumed failed terminal")
    expected={("native",0):(2946584,"native_MLM_genuine_native_direct_wait_v9","actual_direct_child_wait"),
        ("launcher",0):(2944830,"native_MLM_genuine_entry_supervisor_wait_v9","actual_direct_supervisor_wait"),
        ("outer",0):(2944351,"native_MLM_genuine_launcher_direct_wait_v9","actual_direct_launcher_wait")}
    records={(r["role"],r["slot"]):r for r in frames["records"]}
    require(len(records)==6 and frames["stdout_tails_read"] is False,"Complete sealed bounded metadata extracts, no stdout payload")
    for key,(pid,schema,flag) in expected.items():
        record=records[key];v=record["value"]
        require(record["committed"] is True and v["pid"]==pid and v["schema"]==schema
            and v[flag] is True and v["returncode"]==-15,"Actual separately authenticated direct wait minus15")
    return m

def _kernel_endpoints():
    with Path("/proc/net/unix").open("rb") as f:data=f.read(16*2**20+1)
    require(len(data)<=16*2**20,"Bounded exact-target kernel endpoint inventory")
    return [row.split(maxsplit=7)[7] for row in data.splitlines()[1:] if len(row.split(maxsplit=7))==8]

def _native_socket_absent(path):
    require(str(path).encode() not in _kernel_endpoints(),"Exact retained socket remains a kernel endpoint")
    return True

def _native_targets(m):
    pids=[]
    for root in m["known_PID_start_rows"]:
        row=_absent(root["pid"])
        pids.append([root["pid"],root["start_ticks"],True,row["checked_utc"]])
    groups=[]
    for group in m["known_groups"]:
        try:os.killpg(group,0)
        except ProcessLookupError:groups.append([group,True,_utc()]);continue
        except PermissionError:raise ValueError("Exact numeric group absence denied") from None
        raise ValueError("Known old numeric group live or reused; preserve and block")
    endpoints=[];helper=_native_helper();kernel=_kernel_endpoints()
    require(not any(path.startswith(b"@arggnn-c3mlm-v12-") for path in kernel),"Source-bound abstract native v12 namespace still registered")
    for row in m["recorded_filesystem_endpoints"]:
        path=row["path"];require(str(ROOT/path).encode() not in kernel,"Exact historical filesystem kernel endpoint remains")
        try:
            s=helper.lstat_no_follow_v1(ROOT,ROOT/path)
            original=[s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
            require(stat.S_ISSOCK(s.st_mode) and original==row["original_lstat"],"Unexpected changed/non-socket historical endpoint; no cleanup or normalization")
            state="genuine_stale_socket_metadata_preserved"
        except FileNotFoundError:state="absent_path"
        endpoints.append([path,state,_utc()])
    return dict(PID_observations=pids,group_observations=groups,endpoint_observations=endpoints,
        abstract_namespace_observation=["@arggnn-c3mlm-v12-","no_exact_prefix_kernel_endpoint",_utc()])

def stopped_observation_draft_v1(*,revocation_ref):
    return dict(schema="native14_exact_stopped_targeted_ROOT_observation_v1",root_authorizer="root_agent",active=False,
        key=NATIVE_KEY,source_manifest_ref=_ref(MANIFEST),revocation_ref=copy.deepcopy(revocation_ref),
        observed_at_utc=None,PID_observations=[],group_observations=[],endpoint_observations=[],
        abstract_namespace_observation=[],historical_whole_descendant_drain="UNKNOWN",
        original_v8_native_exit="UNKNOWN",optimizer_updates="UNKNOWN",current_complete_old_writer_absence_claimed=False)

def _native_observation(record,*,fresh):
    v=_json(record);m=_native_manifest();p=_policy()
    expected=stopped_observation_draft_v1(revocation_ref=p["permanent_revocation_ref"])
    fields=("PID_observations","group_observations","endpoint_observations","abstract_namespace_observation")
    expected.update(active=True,observed_at_utc=v.get("observed_at_utc"),**{k:v.get(k) for k in fields})
    require(set(v)==set(expected) and budget.v4.encoded(v)==budget.v4.encoded(expected),"Exact current targeted ROOT observation schema")
    rev=_native_revocation(v["revocation_ref"]);start=_stamp(rev["revoked_at_utc"]);end=_stamp(v["observed_at_utc"])
    require(start<=end,"Actual revocation precedes targeted observation")
    _native_validate_targets({k:v[k] for k in fields},m,start,end,fresh=fresh)
    return v

def _native_validate_targets(facts,m,start,end,*,fresh):
    require(set(facts)=={"PID_observations","group_observations","endpoint_observations","abstract_namespace_observation"},"Exact compact targeted collections")
    expected_pids=[[r["pid"],r["start_ticks"],True] for r in m["known_PID_start_rows"]]
    expected_groups=[[g,True] for g in m["known_groups"]]
    endpoints=m["recorded_filesystem_endpoints"]
    require(len(facts["PID_observations"])==len(expected_pids) and len(facts["group_observations"])==len(expected_groups)
        and len(facts["endpoint_observations"])==len(endpoints),"Every retained known PID/group/endpoint is observed")
    rows=[]
    for actual,expected in zip(facts["PID_observations"],expected_pids):
        require(isinstance(actual,list) and len(actual)==4 and budget.v4.encoded(actual[:3])==budget.v4.encoded(expected),"Exact numeric PID/start-or-UNKNOWN ESRCH fact")
        rows.append(actual[-1])
    for actual,expected in zip(facts["group_observations"],expected_groups):
        require(isinstance(actual,list) and len(actual)==3 and budget.v4.encoded(actual[:2])==budget.v4.encoded(expected),"Exact numeric group ESRCH fact")
        rows.append(actual[-1])
    for actual,expected in zip(facts["endpoint_observations"],endpoints):
        require(isinstance(actual,list) and len(actual)==3 and actual[0]==expected["path"]
            and actual[1] in ("absent_path","genuine_stale_socket_metadata_preserved"),"Exact preserved historical endpoint observation")
        rows.append(actual[-1])
    abstract=facts["abstract_namespace_observation"]
    require(isinstance(abstract,list) and len(abstract)==3 and abstract[:2]==["@arggnn-c3mlm-v12-","no_exact_prefix_kernel_endpoint"],"Exact source-bound abstract namespace observation")
    rows.append(abstract[-1])
    for text in rows:
        stamp=_stamp(text);require(start<=stamp<=end,"Target observation order cannot be fabricated")
        if fresh:require((datetime.now(timezone.utc)-stamp).total_seconds()<=300,"Targeted observation older than five minutes")
    return True

def root_observe_stopped_v1(*,revocation_ref,expected_source_sha256,expected_launcher_sha256):
    """Future ROOT only. No writes; exact probes require separate actual authority."""
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        assert_open(NATIVE_KEY,state[0],state[1]);m=_native_origin(state[0][NATIVE_KEY])
        require(revocation_ref==_policy()["permanent_revocation_ref"],"Actual policy-bound permanent revocation")
        _native_revocation(revocation_ref);v=stopped_observation_draft_v1(revocation_ref=revocation_ref)
        v.update(active=True,**_native_targets(m),observed_at_utc=_utc());return v

def stopped_request_draft_v1(*,observation_ref):
    return dict(key=NATIVE_KEY,reservation_sha256=_native_manifest()["original_reservation_sha256"],
        observation_ref=copy.deepcopy(observation_ref))

def _native_proof(request,own,*,observe=False):
    require(set(request)=={"key","reservation_sha256","observation_ref"} and request["key"]==NATIVE_KEY
        and request["reservation_sha256"]==budget.digest(own),"Only one exact stopped native14 key")
    m=_native_origin(own);obs=_native_observation(request["observation_ref"],fresh=observe)
    fields=("PID_observations","group_observations","endpoint_observations","abstract_namespace_observation")
    facts=_native_targets(m) if observe else {k:copy.deepcopy(obs[k]) for k in fields}
    rev=_native_revocation(obs["revocation_ref"])
    _native_validate_targets(facts,m,_stamp(rev["revoked_at_utc"]),datetime.now(timezone.utc),fresh=observe)
    return dict(schema="native14_genuine_stopped_revoked_retention_proof_v1",key=NATIVE_KEY,
        reservation_sha256=budget.digest(own),source_manifest_ref=_ref(MANIFEST),sealed_evidence_handoff_ref=EVIDENCE_SEAL_REF,
        root_observation_ref=request["observation_ref"],permanent_revocation_ref=obs["revocation_ref"],
        original_expiry_unchanged=NATIVE_EXPIRY,original_v12_native_entry_launcher_waits=[-15,-15,-15],
        actual_outer_tool_session=48073,actual_outer_tool_exit=1,postdrain_marker_error="Expired marker-reader repair authority",
        historical_whole_descendant_drain="UNKNOWN",original_v8_native_exit="UNKNOWN",optimizer_updates="UNKNOWN",
        scientific_success_asserted=False,current_complete_old_writer_absence_claimed=False,
        retirement_kind="exact_stopped_permanently_revoked_retention_only",
        pre_measurement_targets=facts,post_measurement_targets=copy.deepcopy(facts))

def _native_measurement(value,own):
    m=_native_manifest()
    expected={"schema","conservative_scope_bytes","entries","scope_paths","stat_only","per_scope","inventory_sha256",
        "two_complete_passes_and_final_no_follow_revalidation","per_node_additional_metadata_charge_bytes","regular_payloads_opened",
        "symlink_targets_followed","unsafe_nodes_rejected","node_metadata_preserved","namespace_normalization_performed","cooperative_snapshot_not_atomic_quota"}
    require(set(value)==expected and value["schema"]=="native14_exact_stopped_retained_node_measurement_v1"
        and value["scope_paths"]==NATIVE_ROOTS and value["scope_paths"]==[str(p) for p in _paths(own)]
        and value["stat_only"] is True and value["two_complete_passes_and_final_no_follow_revalidation"] is True
        and value["per_node_additional_metadata_charge_bytes"]==4096
        and value["regular_payloads_opened"] is False and value["symlink_targets_followed"] is False
        and value["unsafe_nodes_rejected"] is True and value["node_metadata_preserved"] is True
        and value["namespace_normalization_performed"] is False and value["cooperative_snapshot_not_atomic_quota"] is True,
        "Exact full-parent readonly retained-node certificate; no normalization or followed target")
    actual=budget.v4.integer(value["conservative_scope_bytes"],"conservatively retained nodes")
    entries=budget.v4.integer(value["entries"],"retained nodes",3)
    require(entries<=1000000 and actual<=NATIVE_RESERVED and own["reserved_bytes"]==NATIVE_RESERVED,"Full retained charge stays under original14GiB; no fabricated credit")
    require(isinstance(value["inventory_sha256"],str) and len(value["inventory_sha256"])==64
        and all(c in "0123456789abcdef" for c in value["inventory_sha256"]),"Actual metadata inventory SHA256")
    require(len(value["per_scope"])==3 and sum(r["entries"] for r in value["per_scope"])==entries
        and sum(r["conservative_scope_bytes"] for r in value["per_scope"])==actual,"No charged parent omitted")
    for r,path in zip(value["per_scope"],NATIVE_ROOTS):
        require(set(r)=={"path","entries","conservative_scope_bytes","node_types"} and r["path"]==path
            and set(r["node_types"])=={"directory","regular","symlink","socket"},"Exact all-node parent summary")
        for number in [r["entries"],r["conservative_scope_bytes"],*r["node_types"].values()]:budget.v4.integer(number,"parent metadata count")
        require(r["entries"]>=1 and r["node_types"]["directory"]>=1 and sum(r["node_types"].values())==r["entries"]
            and r["conservative_scope_bytes"]>=4096*r["entries"],"Conservative per-node metadata charge")
    return actual

def _native_recorded_proof(recorded,expected,closed_utc):
    variable={"pre_measurement_targets","post_measurement_targets"}
    require(set(recorded)==set(expected) and all(recorded[k]==expected[k] for k in expected if k not in variable),"Immutable failed/unknown-history proof preserved")
    m=_native_manifest();rev=_native_revocation(recorded["permanent_revocation_ref"])
    start=_stamp(rev["revoked_at_utc"]);end=_stamp(closed_utc)
    for field in ("pre_measurement_targets","post_measurement_targets"):
        _native_validate_targets(recorded[field],m,start,end,fresh=False)
    for field in recorded["pre_measurement_targets"]:
        first=recorded["pre_measurement_targets"][field];last=recorded["post_measurement_targets"][field]
        rows=[(first,last)] if field=="abstract_namespace_observation" else list(zip(first,last))
        original=expected["pre_measurement_targets"][field]
        oldrows=[original] if field=="abstract_namespace_observation" else original
        for (a,b),old in zip(rows,oldrows):
            require(budget.v4.encoded(old[:-1])==budget.v4.encoded(a[:-1])==budget.v4.encoded(b[:-1]),
                "Known target identity/presence changed across certificate and measurement")
            require(_stamp(old[-1])<=_stamp(a[-1])<=_stamp(b[-1]),"Actual targeted observation chronology must extend the ROOT certificate")

def _native_no_completion():
    helper=_native_helper()
    for relative in ("experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8/completed.json",
        "experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8/amended_v12_attempt1/completed.json",
        "experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8/amended_v12_attempt1/mlm/completed.json"):
        try:helper.lstat_no_follow_v1(ROOT,ROOT/relative)
        except FileNotFoundError:continue
        raise ValueError("Unexpected native completion pathname; preserve and block stopped-only disposition")

def apply_retire_stopped_native14_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    """Future exact-key ROOT mutation; never invoked by source author/fixtures."""
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        auth=_authority(authority_ref,RETIRE_OPERATION,state[3],fresh=True)
        require(auth["expected_before"]==_pins(state[2]),"Exact fresh successor prefix before stopped retirement")
        request=auth["request"];require(request["key"]==NATIVE_KEY,"No generic stopped/failed key facility")
        assert_open(NATIVE_KEY,state[0],state[1]);own=state[0][NATIVE_KEY]
        # The full invocation role union includes both actual observation and operation refs.
        consumer_bindings_v1(list(state[3].values()),[request["observation_ref"],authority_ref])
        _native_no_completion()
        proof=_native_proof(request,own,observe=True);helper=_native_helper()
        measurement=helper.measure_retained_nodes_v1(ROOT,_paths(own),expected_paths=NATIVE_ROOTS,socket_is_absent=_native_socket_absent)
        actual=_native_measurement(measurement,own);_native_no_completion()
        proof["post_measurement_targets"]=_native_targets(_native_manifest())
        value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
            proof=proof,closed_utc=_utc(),future_writes_and_relaunch_revoked=True)
        _native_recorded_proof(proof,_native_proof(request,own),value["closed_utc"])
        projected=charge(_policy()["base_projected_bytes"],state[2]["new_reserved_bytes"],
            state[2]["completed_credit_bytes"]+NATIVE_RESERVED-actual,len(state[4])+1)
        _expire(auth)
        tail=_append(RETIRE_OPERATION,authority_ref,value,state)
        return dict(kind="native14_exact_stopped_retention_receipt_v1",key=NATIVE_KEY,value=value,retained_bytes=actual,
            credit_bytes=NATIVE_RESERVED-actual,projected_bytes=projected,scope_measurements=1,
            completed_overlay_prefix=dict(events=len(state[4])+1,tail_sha256=tail),
            old_journal_mutated=False,old_scope_mutated=False,artifacts_deleted=0,
            scientific_success_asserted=False,historical_whole_descendant_drain="UNKNOWN",optimizer_updates="UNKNOWN",
            **warning_fields(projected))

if __name__=='__main__':
    print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE_ROOT_REVIEW_ACTIVATION_REQUIRED',accounting_mode=MODE,retirement_key=NATIVE_KEY,live_operations_performed=False,credit_claimed=False)))
