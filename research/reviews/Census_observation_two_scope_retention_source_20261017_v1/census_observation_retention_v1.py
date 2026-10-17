"""Only two exact historical Census observation scopes over a frozen Census_observation parent.

Inactive source only until independently reviewed ROOT activation selects actual
fresh parent pins/projection. Existing admitted parent consumers remain unchanged.
No completion, historical drain, deletion, global rebase or capacity is invented.
"""
from __future__ import annotations
import copy,hashlib,importlib.util,io,json,os,posixpath,stat,time
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
from scripts import project_budget_v9 as budget

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py'
JOURNAL='.runtime/storage/Census_observation_two_scope_retention_v1'
MODE='Census_observation_two_scope_retention_v1'
BASE_REF={'path': 'research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py', 'sha256': 'b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164'}
BASE_POLICY_REF={'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json', 'sha256': 'fdec761f8cae04022093299955655f8ffa27166829653817165126d7ada6a65c'}
BASE_MODE='Relate_failed64GiB_retention_v1'
BASE_SELECTOR_REF={'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json', 'sha256': 'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}
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
RETIRE_OPERATION='retire_exact_Census_observation_retained_allowance_v1'
OPERATION_WALL_SECONDS=300
_BASE=_POLICY_REF=_REPAIR_AUTHORITY_REF=_SELECTOR_REF=_SELECTED_VALUE=None





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

def root_activation_draft_v1(*,base_pins,base_projected_bytes,permanent_revocation_ref=None,source_review_ref=None,
 prerequisite_admissions_ref=None,parent_selected_at_utc=None):
 return dict(schema="Census_observation_frozen_Relate64_ROOT_activation_v1",root_authorizer="root_agent",active=False,
  source_ref=_ref(SOURCE),source_manifest_ref=_ref(MANIFEST),measurement_source_ref=_ref(MEASURE_SOURCE),
  frozen_parent_ref=BASE_REF,frozen_parent_policy_ref=BASE_POLICY_REF,
  frozen_parent_selector_ref=BASE_SELECTOR_REF,frozen_parent_pins=copy.deepcopy(base_pins),
  frozen_parent_projected_bytes=base_projected_bytes,permanent_revocation_ref=permanent_revocation_ref,
  source_review_ref=source_review_ref,prerequisite_admissions_ref=prerequisite_admissions_ref,
  parent_selected_at_utc=parent_selected_at_utc,new_journal=JOURNAL,only_Census_observation_retention_keys=list(TARGET_KEYS),
  actual_parent_prefix_and_projection_selected_under_original_mutex=True,
  parent_pins_selected_after_H058v5_72068_and_FLARE_native_50131_consumed_admissions=True,
  ROOT_explicitly_releases_current_selector_for_this_child=True,
  all_future_parent_reserve_close_retire_mutations_cooperatively_revoked=True,
  all_older_accounting_future_mutation_revocations_preserved=True,
  existing_parent_consumers_continue_under_original_sources_policies_scopes_caps_and_guards=True,
  exact_Census_future_writers_require_separate_permanent_revocation=True,
  old_parent_source_policy_journal_and_consumer_storage_records_unchanged=True,
  no_current_credit_or_capacity_implied_by_activation=True,no_historical_mutation=True,
  no_deletion_or_rename_or_normalization=True,no_rebase=True,physical_GPU_ids=[0])


def activation_policy_draft_v1(*,repair_authority_ref,base_pins,base_projected_bytes,implementation_ref,launcher_ref,
 root_activation_ref=None,permanent_revocation_ref=None,source_review_ref=None,
 prerequisite_admissions_ref=None,parent_selected_at_utc=None):
 return dict(schema="Census_observation_exact_retained_allowance_policy_v1",root_authorizer="root_agent",active=False,
  source_ref=_ref(SOURCE),source_manifest_ref=_ref(MANIFEST),measurement_source_ref=_ref(MEASURE_SOURCE),
  selector_source_ref=_ref(SOURCE),base_ref=BASE_REF,base_policy_ref=BASE_POLICY_REF,base_selector_ref=BASE_SELECTOR_REF,
  base_accounting_mode=BASE_MODE,marker_repair_authority_ref=repair_authority_ref,
  implementation_ref=implementation_ref,launcher_ref=launcher_ref,base_pins=copy.deepcopy(base_pins),
  base_projected_bytes=base_projected_bytes,journal=JOURNAL,root_activation_ref=root_activation_ref,
  permanent_revocation_ref=permanent_revocation_ref,source_review_ref=source_review_ref,
  prerequisite_admissions_ref=prerequisite_admissions_ref,parent_selected_at_utc=parent_selected_at_utc,
  scheduling_ceiling_bytes=CEILING_BYTES,warning_threshold_bytes=WARNING_BYTES,human_limit_bytes=HUMAN_LIMIT_BYTES,
  bootstrap_metadata_charge_bytes=BOOTSTRAP_CHARGE,per_event_metadata_charge_bytes=META_CHARGE,
  never_credit_keys=[MALFORMED_KEY,"H041/American-C3-native-MLM-full30mb-mlm-fit-v8"],
  only_Census_observation_retention_keys=list(TARGET_KEYS),ordinary_completed_close_excluded_keys=list(ORDINARY_EXCLUSIONS),
  all_budget_writers_use_original_mutex=True,all_old_future_admissions_revoked=True,
  all_future_admissions_and_ordinary_closes_use_this_overlay_after_ROOT_activation=True,
  existing_parent_consumers_never_upgraded_or_rebound=True,old_active_writers_continue_unchanged_fully_charged=True,
  no_historical_mutation=True,no_rebase=True,no_deletion_or_rename_or_normalization=True,
  retention_events_assert_scientific_completion=False,external_NFS_writers_outside_observation=True,
  genuine_phase_success_and_direct_worker_wait0_preserved=True,
  unknown_supervisor_outer_wait_and_historical_descendant_drain_preserved=True,
  unknown_historical_descendants_preserved=True,live_jobs_are_separate_and_unchanged=True,
  operation_wall_seconds=OPERATION_WALL_SECONDS,direct_guard_additions="one_explicit_selector_capsule",
  eligible_physical_GPU_ids=[0])


def configure_v2(repair_authority_ref,*,policy_ref=None,selector_ref=None):
 global _BASE,_POLICY_REF,_REPAIR_AUTHORITY_REF,_SELECTOR_REF,_SELECTED_VALUE
 require(policy_ref is not None and selector_ref is not None,"Explicit reviewed policy AND selector capsule; no discovery")
 require(_POLICY_REF is None or (_POLICY_REF==policy_ref and _SELECTOR_REF==selector_ref),"No lifetime source/policy/selector swap")
 require(all(getattr(budget.v4,n).__module__==budget.v4.__name__ for n in("canonical","file_bytes","mutex")),
  "Configure exact guard before any consumer memo installation")
 _POLICY_REF=copy.deepcopy(policy_ref);_SELECTOR_REF=copy.deepcopy(selector_ref);_REPAIR_AUTHORITY_REF=copy.deepcopy(repair_authority_ref)
 _policy();_SELECTED_VALUE=copy.deepcopy(_selected_record(_SELECTOR_REF))
 if _BASE is None:
  sp=importlib.util.spec_from_file_location("_Census_observation_frozen_native7_parent",ROOT/BASE_REF["path"])
  _BASE=importlib.util.module_from_spec(sp);sp.loader.exec_module(_BASE)
  _BASE.configure_v2(repair_authority_ref,policy_ref=BASE_POLICY_REF,selector_ref=BASE_SELECTOR_REF)

def configure_with_renewal_v1(repair_authority_ref,*,policy_ref,selector_ref):
 """Future fresh ROOT interpreter only; never called by authoring or static checks."""
 _raw(RENEWAL_REF);_raw(RENEWAL_CERT_REF)
 sp=importlib.util.spec_from_file_location("_Census_observation_original_renewal",ROOT/RENEWAL_REF["path"])
 renewal=importlib.util.module_from_spec(sp);sp.loader.exec_module(renewal)
 renewal.install_renewal_v1(RENEWAL_CERT_REF)
 return configure_v2(repair_authority_ref,policy_ref=policy_ref,selector_ref=selector_ref)

def _policy():
 require(_POLICY_REF is not None and _SELECTOR_REF is not None,"Explicit independently reviewed ROOT activation and selector required")
 p=_json(_POLICY_REF)
 expected=activation_policy_draft_v1(repair_authority_ref=_REPAIR_AUTHORITY_REF,base_pins=p.get("base_pins"),
  base_projected_bytes=p.get("base_projected_bytes"),implementation_ref=p.get("implementation_ref"),launcher_ref=p.get("launcher_ref"),
  root_activation_ref=p.get("root_activation_ref"),permanent_revocation_ref=p.get("permanent_revocation_ref"),
  source_review_ref=p.get("source_review_ref"),
  prerequisite_admissions_ref=p.get("prerequisite_admissions_ref"),parent_selected_at_utc=p.get("parent_selected_at_utc"))
 expected["active"]=True
 require(p==expected and set(p)==set(expected),"Exact inactive-until-ROOT two-key Census observation policy")
 for r in(p["source_ref"],p["source_manifest_ref"],p["measurement_source_ref"],p["selector_source_ref"],p["base_ref"],
  p["base_policy_ref"],p["base_selector_ref"],p["marker_repair_authority_ref"],p["implementation_ref"],p["launcher_ref"],p["source_review_ref"]):_raw(r)
 require(p["implementation_ref"]["path"]==budget.SOURCE_REL and p["launcher_ref"]["path"]==budget.LAUNCHER_REL,
  "All original implementation/launcher guards remain")
 require(isinstance(p["base_pins"],dict)and set(p["base_pins"])=={"base_pins","completed_overlay_prefix"},"Actual ROOT Relate64 prefix, never forecast")
 budget.v4.integer(p["base_projected_bytes"],"actual frozen Relate64 projection")
 parent_selector=_json(BASE_SELECTOR_REF)
 require(parent_selector.get("source_ref")==BASE_REF and parent_selector.get("policy_ref")==BASE_POLICY_REF
  and parent_selector.get("accounting_mode")==BASE_MODE and parent_selector.get("active")is True,
  "Exact previously active Relate64 selector history remains authenticated")
 activation=_json(p["root_activation_ref"])
 exact=root_activation_draft_v1(base_pins=p["base_pins"],base_projected_bytes=p["base_projected_bytes"],
  permanent_revocation_ref=p["permanent_revocation_ref"],source_review_ref=p["source_review_ref"],
  prerequisite_admissions_ref=p["prerequisite_admissions_ref"],parent_selected_at_utc=p["parent_selected_at_utc"])
 exact["active"]=True
 require(activation==exact and set(activation)==set(exact),"Actual ROOT parent freeze after prerequisite admissions")
 rev=_json(p["permanent_revocation_ref"])
 require(rev==dict(schema="Census_observation_parent_future_mutation_revocation_v1",root_authorizer="root_agent",active=True,
  parent_source_ref=BASE_REF,parent_policy_ref=BASE_POLICY_REF,parent_selector_ref=BASE_SELECTOR_REF,
  actual_parent_pins=p["base_pins"],prerequisite_admissions_ref=p["prerequisite_admissions_ref"],
  parent_selected_at_utc=p["parent_selected_at_utc"],all_future_parent_reserve_close_retire_mutations_permanently_cooperatively_revoked=True,
  existing_parent_read_check_and_fully_charged_writers_unchanged=True,no_historical_mutation=True),
  "Exact future-parent-mutation revocation; existing read/check/writers unchanged")
 _prerequisites(p["prerequisite_admissions_ref"],p["parent_selected_at_utc"])
 review=_json(p["source_review_ref"])
 require(review.get("independent")is True and review.get("verdict")in("PASS_SOURCE_ONLY","SOURCE_PASS","PASS")
  and review.get("source_ref")==_ref(SOURCE)and review.get("source_manifest_ref")==_ref(MANIFEST)
  and review.get("measurement_source_ref")==_ref(MEASURE_SOURCE)and review.get("blockers")==[],
  "Independent complete source/manifest/existing-measurement review with no blockers")
 return p


def _bindings(base_bindings,p):
 # Every actual predecessor role stays DIRECT and unchanged. Only one new
 # DIRECT selector capsule is added; it transitively authenticates source,
 # policy, activation, revocation, independent review and sealed manifest.
 result=copy.deepcopy(base_bindings);_manifest();_selected_record(_SELECTOR_REF)
 additions=dict(Census_observation_explicit_selector_capsule=copy.deepcopy(_SELECTOR_REF))
 require(not set(result)&set(additions),"Never overwrite or drop any active ancestor guard role")
 result.update(additions);consumer_bindings_v1(list(result.values()))
 return result


def operation_draft_v1(operation,request,*,expected_before,bindings):
    require(operation in ("reserve","close_completed",RETIRE_OPERATION),"Ordinary reserve/close or only exact Census observation retained allowance")
    return dict(schema="Census_observation_exact_ROOT_operation_v1",root_authorizer="root_agent",active=False,
        operation=operation,accounting_mode=MODE,bindings=copy.deepcopy(bindings),expected_before=copy.deepcopy(expected_before),
        expires_at_utc=None,request=copy.deepcopy(request),all_budget_writers_use_original_mutex=True,
        all_old_future_admissions_revoked=True,all_future_consumers_use_this_overlay=True,
        historical_mutation_authorized=False,global_rebase_authorized=False,deletion_authorized=False,
        scientific_execution_authorized=False,GPU_authorized=False)




def _paths(own):
    require(_BASE is not None,'Configured immutable base required')
    # Preserve every frozen ancestor scope/marker rule and 64-direct-binding cap.
    paths=_BASE._paths(own)
    protected=[ROOT/JOURNAL,ROOT/SOURCE,budget.v4.canonical(ROOT,_POLICY_REF['path']),budget.v4.canonical(ROOT,_SELECTOR_REF['path'])]
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
            and event['schema']=='Census_observation_exact_retention_event_v1' and type(event['sequence']) is int and event['sequence']==i
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
    require(len(base)==7,'Exact frozen Relate64 child seven-part state API')
    reservations,base_closed,view,base_bindings,_,_,_=base
    require(_BASE._pins(view)==p['base_pins'] and view['projected_bytes']==p['base_projected_bytes']
        and view['accounting_mode']==BASE_MODE and view['limit_bytes']==CEILING_BYTES
        and view['hard_limit_bytes']==HUMAN_LIMIT_BYTES,
        'Frozen Relate64 parent prefix/projection changed; no new charge or credit')
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
            require(all(r in value['consumer_bindings'] for r in bindings.values()) and _ref(SOURCE) in value['consumer_bindings'],'All actual base/overlay/policy/marker plus new adapter bindings must be reserved')
            reservations[key]=value;occupied.extend(paths);added+=value['reserved_bytes'];reserved_at[key]=i+1
        elif event['operation']==RETIRE_OPERATION:
            key=request["key"];require(key in TARGET_KEYS,"Only the exact two Census observation keys")
            assert_open(key,reservations,closed);own=reservations[key]
            proof=_stopped_proof(request,own)
            require(set(value)=={"request","reservation_sha256","measurement","proof","closed_utc","future_writes_and_relaunch_revoked"}
                and value["request"]==request and value["reservation_sha256"]==budget.digest(own)
                and value["future_writes_and_relaunch_revoked"] is True,"Exact Census retained-allowance certificate")
            _recorded_stopped_proof(value["proof"],proof,value["closed_utc"])
            credit+=own["reserved_bytes"]-_retained_measurement(value["measurement"],own);closed.add(key)
        else:
            key=request['key'];assert_open(key,reservations,closed)
            require(key not in p['never_credit_keys'] and key not in ORDINARY_EXCLUSIONS,'No ordinary completed close for never-credit or exact stopped keys')
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
    event=dict(schema='Census_observation_exact_retention_event_v1',sequence=len(events)+1,previous_sha256=hashes[-1] if hashes else None,
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
        require(all(r in own['consumer_bindings'] for r in state[3].values()) and _ref(SOURCE) in own['consumer_bindings'],'Every actual guard/policy/marker plus new adapter source required')
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
        require(key not in _policy()['never_credit_keys'] and key not in ORDINARY_EXCLUSIONS,'No ordinary completed close for never-credit or exact stopped keys')
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


































MANIFEST="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/source_manifest.v1.json"
MEASURE_SOURCE='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/single_root_retained_nodes_v1.py'
SELECTOR_SOURCE=SOURCE
PARENT_STOPPED_KEYS=('native32/native-VAL000-full227990-borrowed32-GPU0-v1', 'H050/H050-direct-readout-GPU0-TRAIN-v1', 'ARGMix/prospective1611-accepted-TRAIN000-Relate-v4')
TARGET_KEYS=('census_tsinfer_observation_v1/census-tsinfer-inspect-v1', 'census_tsinfer_observation_v1/census-tsinfer-numerical-v1')
ORDINARY_EXCLUSIONS=TARGET_KEYS+PARENT_STOPPED_KEYS
FRESH_TARGET_SECONDS=120

def _manifest():
 m=_completed_json(_ref(MANIFEST))
 require(m.get("schema")=="Census_observation_exact_two_scope_retention_manifest_v1"
  and m.get("target_order")==list(TARGET_KEYS)and set(m["targets"])==set(TARGET_KEYS),
  "Only exact two sealed Census observation reservations")
 for path,sha in m["source_sha256"].items():_raw(dict(path=path,sha256=sha))
 for record in m["runtime_authenticated_historical_source_and_metadata_refs"]:_raw(record)
 require(m.get("predecessor_source_ref")==BASE_REF and m.get("predecessor_policy_ref")==BASE_POLICY_REF
  and m.get("predecessor_selector_ref")==BASE_SELECTOR_REF,"Exact immutable actual Relate64 predecessor refs")
 for key,t in m["targets"].items():
  storage=_json(t["storage_ref"]);own=t["original_reservation"]
  require(storage["schema"]=="census_tsinfer_observation_storage_v1" and storage["mode"]=="project_budget_v4"
   and storage["reservation"]==own and storage["reserved_new_bytes"]==t["reserved_bytes"]==20000000000
   and storage["expires_at_utc"]==t["original_expiry_unchanged"]and budget.digest(own)==t["original_reservation_sha256"]
   and own["key"]==key and own["reserved_bytes"]==20000000000 and own["auxiliary_scopes"]==[]
   and t["scope_paths"]==[own["output"]] and storage["output_directory"]==str(Path(own["output"]).relative_to(ROOT)),
   "Exact original20GB single root/cap/expiry/reservation; no synthetic auxiliary")
  require(own["consumer_bindings"]==t["all_original_binding_refs"]
   and all(r in m["runtime_authenticated_historical_source_and_metadata_refs"]for r in t["all_original_binding_refs"]),
   "Every exact original consumer binding remains runtime authenticated")
 return m

def _target(key):
 require(key in TARGET_KEYS and key not in(MALFORMED_KEY,"H041/American-C3-native-MLM-full30mb-mlm-fit-v8",*PARENT_STOPPED_KEYS),
  "Only these exact two Census observation keys; no other retention exception")
 return _manifest()["targets"][key]

def _historical_stop(t):
 """Authenticate genuine historical phase success; never invent a supervisor wait or drain."""
 own=t["original_reservation"];adm=_json(t["admission_ref"]);ev=_phase_json(t["phase_evidence_ref"])
 wait=_json(t["direct_worker_wait_ref"]);started=_json(t["started_ref"]);launch=_json(t["launch_ref"])
 storage=_json(t["storage_ref"]);cfg=_json(t["config_ref"])
 require(adm["schema"]=="census_tsinfer_observation_admission_v1" and adm["status"]=="active"
  and adm["phase"]==t["phase"] and adm["job_id"]==t["job_id"]
  and adm["output_directory"]==storage["output_directory"]
  and adm["expires_at_utc"]==t["original_expiry_unchanged"]
  and adm["fit_authorized"]is False and adm["query_truth_authorized"]is False
  and adm["refs"]["storage"]==t["storage_ref"] and adm["refs"]["config"]==t["config_ref"]
  and adm["refs"]["runtime"]==t["runtime_metadata_ref"],"Exact original admission phase/source/storage/expiry joins")
 require(storage["reservation"]==own and adm["source_sha256"]==t["original_source_sha256"]
  and adm["source_sha256"][t["source_ref"]["path"]]==t["source_ref"]["sha256"]
  and ev["source_sha256"]==adm["source_sha256"] and ev["admission_ref"]==t["admission_ref"]
  and ev["metadata_input_sha256"]==cfg["metadata_input_sha256"]
  and ev["observed_artifact_sha256"]==adm["observed_artifact_sha256"]==cfg["observed_artifact_sha256"],
  "Exact historical provenance maps preserved without scientific payload reads")
 require(ev["schema"]=="census_tsinfer_observation_evidence_v1" and ev["phase"]==t["phase"] and ev["job_id"]==t["job_id"]
  and ev["status"]==t["original_phase_status"] and ev["expected_cells"]==32 and len(ev["cells"])==32
  and ev["failed_cells"]==[] and len({(x["parent_id"],x["tile_id"])for x in ev["cells"]})==32
  and all(x["status"]==t["original_cell_status"]for x in ev["cells"])
  and ev["fixture_mode"]is False and ev["biological_targets_opened"]is False
  and ev["fit_performed"]is False and ev["GPU_used"]is False,"Original genuine32of32 phase success; no completion rewrite")
 worker,supervisor=t["known_PID_start_rows"][1],t["known_PID_start_rows"][0]
 require(wait["worker_pid"]==started["worker_pid"]==worker["pid"]
  and wait["worker_start_ticks"]==started["worker_start_ticks"]==worker["start_ticks"]
  and type(wait["returncode"])is int and wait["returncode"]==0 and wait["watchdog_triggered"]is False
  and started["supervisor_pid"]==supervisor["pid"]and started["supervisor_start_ticks"]==supervisor["start_ticks"]
  and launch["supervisor_pid"]==supervisor["pid"]and launch[t["launch_supervisor_start_field"]]==supervisor["start_ticks"],
  "Original true direct worker wait0 and exact known supervisor/worker identities")
 argv=started["argv"]
 require(argv[argv.index("--phase")+1]==t["phase"]and argv[argv.index("--admission")+1]==t["admission_ref"]["path"]
  and argv[argv.index("--expected-admission-sha256")+1]==t["admission_ref"]["sha256"]
  and t["known_process_groups"]==[worker["pid"]],"Exact original phase argv and source-derived worker session group")
 if t["phase"]=="inspect":
  require(t["original_phase_status"]=="inspection_complete_pending_root_freeze"
   and t["original_cell_status"]=="inspected_pending_root_freeze"
   and launch["supervisor_source_sha256"]==t["supervisor_source_ref"]["sha256"],"Original inspection/source identity")
  freeze=_phase_json(t["inspection_ROOT_freeze_ref"])
  require(freeze["inspection_evidence"]==t["phase_evidence_ref"]and freeze["inspection_terminal"]==t["direct_worker_wait_ref"]
   and freeze["status"]=="complete","Original ROOT inspection freeze authenticates genuine phase and wait")
 else:
  require(t["phase"]=="numerical" and t["original_phase_status"]=="numerical_complete_no_fitting_or_predictive_claim"
   and t["original_cell_status"]=="numerically_qualified" and launch["admission"]==t["admission_ref"]
   and launch["actual32_numerical_launched"]is True,"Original numerical admission and phase identity")
 require(_stamp(adm["created_at_utc"])<=_stamp(started["started_utc"])<=_stamp(ev["created_at_utc"])
  <=_stamp(wait["completed_utc"])<=_stamp(adm["expires_at_utc"])<datetime.now(timezone.utc),
  "Original expired interval and phase/wait chronology; no clock reset")
 require(t["historical_facts_preserved"]==dict(original_phase_status=t["original_phase_status"],direct_worker_wait_returncode=0,
  watchdog_triggered=False,supervisor_outer_OS_wait="UNKNOWN",historical_descendant_drain="UNKNOWN",
  complete_historical_descendant_roster="UNKNOWN",scientific_fit_or_ancestry_claim=False),
  "True phase success/direct0 and UNKNOWN supervisor waits/drain remain distinct")
 return True

def stopped_revocation_draft_v1(key):
 t=_target(key)
 return dict(schema="Census_observation_exact_permanent_writer_revocation_v1",root_authorizer="root_agent",active=False,key=key,
  source_manifest_ref=_ref(MANIFEST),reservation_sha256=t["original_reservation_sha256"],
  original_scope_paths=t["scope_paths"],original_expiry_unchanged=t["original_expiry_unchanged"],revoked_at_utc=None,
  all_original_launch_resume_retry_write_aliases_for_exact_key_permanently_cooperatively_revoked=True,
  original_read_check_routes_and_later_artifact_reads_remain_allowed=True,
  every_other_live_job_source_policy_scope_cap_guard_unchanged=True,
  old_frozen_sources_do_not_mechanically_consult_new_revocation=True,
  ROOT_accepts_cooperative_revocation_and_residual_unknown_historical_external_NFS_writers=True,
  unknown_historical_descendants_absent_claimed=False,original_expiry_preserved_without_clock_reset=True,
  preserve_all_files_and_namespaces=True,no_delete_rename_normalize_rebase_or_historical_mutation=True,
  scientific_success_relabelled=False)

def _revocation(record,key):
    v=_json(record);expected=stopped_revocation_draft_v1(key)
    expected.update(active=True,revoked_at_utc=v.get("revoked_at_utc"))
    require(v==expected and set(v)==set(expected),"Exact fresh permanent ROOT writer revocation required")
    require(_stamp(v["revoked_at_utc"])>=_stamp(_target(key)["earliest_revocation_utc"]),
        "Permanent revocation follows original expiry and genuine phase wait0")
    return v

def stopped_observation_draft_v1(key,*,revocation_ref):
    t=_target(key)
    return dict(schema="Census_observation_known_target_ROOT_observation_v1",root_authorizer="root_agent",active=False,key=key,
        source_manifest_ref=_ref(MANIFEST),reservation_sha256=t["original_reservation_sha256"],revocation_ref=revocation_ref,
        preserve_all_files=True,unknown_historical_descendants_absent_claimed=False,
        live_shared_controllers_not_claimed_absent=True,PID_absence_method="targeted os.kill(pid,0) ESRCH",
        group_absence_method="targeted os.killpg(pgid,0) ESRCH",PID_observations=[],group_observations=[],observed_at_utc=None)

def _validate_targets(facts,t,start,end,*,fresh):
    require(set(facts)=={"PID_observations","group_observations"},"Exact bounded known-target observations")
    require(len(facts["PID_observations"])==len(t["known_PID_start_rows"])
        and len(facts["group_observations"])==len(t["known_process_groups"]),"No known identity/group omitted or truncated")
    for expected,row in zip(t["known_PID_start_rows"],facts["PID_observations"]):
        require(isinstance(row,list) and len(row)==4 and row[:3]==[expected["pid"],expected["start_ticks"],True],
            "Actual numeric known-PID ESRCH; reused numeric PID blocks")
        stamp=_stamp(row[-1]);require(start<=stamp<=end,"Revocation and observation chronology")
        if fresh:require((datetime.now(timezone.utc)-stamp).total_seconds()<=FRESH_TARGET_SECONDS,"Fresh known PID observation required")
    for expected,row in zip(t["known_process_groups"],facts["group_observations"]):
        require(isinstance(row,list) and len(row)==3 and row[:2]==[expected,True],"Actual known-group ESRCH")
        stamp=_stamp(row[-1]);require(start<=stamp<=end,"Revocation and group chronology")
        if fresh:require((datetime.now(timezone.utc)-stamp).total_seconds()<=FRESH_TARGET_SECONDS,"Fresh known group observation required")

def _observe_targets(t):
    pids=[];groups=[]
    for row in t["known_PID_start_rows"]:
        actual=_absent(row["pid"])
        pids.append([row["pid"],row["start_ticks"],True,actual["checked_utc"]])
    for pgid in t["known_process_groups"]:
        try:os.killpg(pgid,0)
        except ProcessLookupError:groups.append([pgid,True,_utc()])
        else:raise ValueError("Known group exists/reused/unresolved; preserve and block")
    return dict(PID_observations=pids,group_observations=groups)

def _stopped_proof(request,own,*,observe=False):
 require(set(request)=={"key","reservation_sha256","revocation_ref","observation_ref"} and request["key"]==own["key"],
  "Exact one-of-two Census retained-allowance request")
 t=_target(own["key"])
 require(own==t["original_reservation"]and budget.digest(own)==request["reservation_sha256"]==t["original_reservation_sha256"]
  and [str(p)for p in _paths(own)]==t["scope_paths"],"No original scope/cap/binding drift")
 _historical_stop(t);rev=_revocation(request["revocation_ref"],own["key"])
 obs=_json(request["observation_ref"]);expected=stopped_observation_draft_v1(own["key"],revocation_ref=request["revocation_ref"])
 expected.update(active=True,observed_at_utc=obs.get("observed_at_utc"),
  PID_observations=obs.get("PID_observations"),group_observations=obs.get("group_observations"))
 require(obs==expected and set(obs)==set(expected),"Exact fresh ROOT known-target observations; historical unknowns preserved")
 start=_stamp(rev["revoked_at_utc"]);end=_stamp(obs["observed_at_utc"])
 rootfacts={k:obs[k]for k in("PID_observations","group_observations")}
 _validate_targets(rootfacts,t,start,end,fresh=observe)
 actual=_observe_targets(t)if observe else copy.deepcopy(rootfacts)
 _validate_targets(actual,t,start,datetime.now(timezone.utc),fresh=observe)
 return dict(schema="Census_observation_retained_allowance_proof_v1",key=own["key"],reservation_sha256=budget.digest(own),
  source_manifest_ref=_ref(MANIFEST),revocation_ref=request["revocation_ref"],observation_ref=request["observation_ref"],
  phase_evidence_ref=t["phase_evidence_ref"],direct_worker_wait_ref=t["direct_worker_wait_ref"],
  historical_facts_preserved=t["historical_facts_preserved"],scientific_success_relabelled=False,
  unknown_historical_descendants_absent_claimed=False,pre_measurement_targets=actual,post_measurement_targets=copy.deepcopy(actual))

def _retained_measurement(value,own):
    t=_target(own["key"])
    expected={"schema","conservative_scope_bytes","entries","scope_paths","stat_only","per_scope","inventory_sha256",
        "two_complete_passes_and_final_no_follow_revalidation","per_node_additional_metadata_charge_bytes",
        "regular_payloads_opened","symlink_targets_followed","unsafe_nodes_rejected","node_metadata_preserved",
        "namespace_normalization_performed","cooperative_snapshot_not_atomic_quota"}
    require(set(value)==expected and value["schema"]=="Census_observation_exact_single_root_retained_node_measurement_v1"
        and value["scope_paths"]==t["scope_paths"] and value["stat_only"] is True
        and value["two_complete_passes_and_final_no_follow_revalidation"] is True
        and value["per_node_additional_metadata_charge_bytes"]==4096 and value["regular_payloads_opened"] is False
        and value["symlink_targets_followed"] is False and value["unsafe_nodes_rejected"] is True
        and value["node_metadata_preserved"] is True and value["namespace_normalization_performed"] is False
        and value["cooperative_snapshot_not_atomic_quota"] is True,"Exact full single-root two-pass no-follow retained-node certificate")
    actual=budget.v4.integer(value["conservative_scope_bytes"],"entire stopped retained bytes")
    entries=budget.v4.integer(value["entries"],"entire retained node count",1)
    require(actual<=own["reserved_bytes"]==t["reserved_bytes"] and entries<=1000000
        and len(value["per_scope"])==1 and sum(r["conservative_scope_bytes"] for r in value["per_scope"])==actual
        and sum(r["entries"] for r in value["per_scope"])==entries,"No credit beyond full original reservation")
    digest=value["inventory_sha256"];require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),
        "Actual complete retained-node metadata inventory digest")
    for row,path in zip(value["per_scope"],t["scope_paths"]):
        require(set(row)=={"path","entries","conservative_scope_bytes","node_types"} and row["path"]==path
            and set(row["node_types"])=={"directory","regular","symlink","socket"}
            and row["node_types"]["socket"]==0,"Every charged parent; any retained socket blocks without cleanup")
        for number in [row["entries"],row["conservative_scope_bytes"],*row["node_types"].values()]:budget.v4.integer(number,"retained node summary")
        require(row["entries"]>=1 and row["node_types"]["directory"]>=1 and sum(row["node_types"].values())==row["entries"]
            and row["conservative_scope_bytes"]>=4096*row["entries"],"Conservative per-node entire-parent charges")
    return actual

def _recorded_stopped_proof(recorded,expected,closed_utc):
    variable={"pre_measurement_targets","post_measurement_targets"}
    require(set(recorded)==set(expected) and all(recorded[k]==expected[k] for k in expected if k not in variable),
        "Original failures/unknown history/revocation/evidence unchanged on replay")
    t=_target(recorded["key"]);start=_stamp(_revocation(recorded["revocation_ref"],recorded["key"])["revoked_at_utc"]);end=_stamp(closed_utc)
    for field in variable:_validate_targets(recorded[field],t,start,end,fresh=False)
    for field in ("PID_observations","group_observations"):
        for old,a,b in zip(expected["pre_measurement_targets"][field],recorded["pre_measurement_targets"][field],recorded["post_measurement_targets"][field]):
            require(a[:-1]==b[:-1]==old[:-1]
                and _stamp(old[-1])<=_stamp(a[-1])<=_stamp(b[-1]),
                "Recorded known-target observations extend ROOT evidence before and after measurement")

def apply_retire_stopped_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
 """Future explicit ROOT operation only; no author/CLI/fixture calls."""
 started=time.monotonic();deadline=started+OPERATION_WALL_SECONDS
 def finite():require(time.monotonic()<deadline,"Finite300second exact retention operation; preserve on timeout")
 with budget.v4.mutex(ROOT):
  finite();state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
  auth=_authority(authority_ref,RETIRE_OPERATION,state[3],fresh=True)
  require(auth["expected_before"]==_pins(state[2]),"Actual fresh child prefix before exact failed retirement")
  request=auth["request"];key=request["key"];require(key in TARGET_KEYS,"Only the exact two Census observation keys")
  assert_open(key,state[0],state[1]);own=state[0][key]
  consumer_bindings_v1(list(state[3].values()),[request["revocation_ref"],request["observation_ref"],authority_ref])
  proof=_stopped_proof(request,own,observe=True);finite()
  sp=importlib.util.spec_from_file_location("_Census_observation_existing_retained_nodes",ROOT/MEASURE_SOURCE)
  helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
  original_require=helper.require
  def bounded_require(ok,message):finite();return original_require(ok,message)
  helper.require=bounded_require
  measurement=helper.measure_retained_nodes_v1(ROOT,_paths(own),expected_paths=_target(key)["scope_paths"],
   socket_is_absent=lambda path:False)
  actual=_retained_measurement(measurement,own);finite()
  proof["post_measurement_targets"]=_observe_targets(_target(key));closed=_utc()
  _recorded_stopped_proof(proof,_stopped_proof(request,own),closed);finite()
  value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
   proof=proof,closed_utc=closed,future_writes_and_relaunch_revoked=True)
  projected=charge(_policy()["base_projected_bytes"],state[2]["new_reserved_bytes"],
   state[2]["completed_credit_bytes"]+own["reserved_bytes"]-actual,len(state[4])+1)
  _expire(auth);finite();tail=_append(RETIRE_OPERATION,authority_ref,value,state)
  return dict(kind="Census_observation_exact_retained_allowance_receipt_v1",key=key,value=value,retained_bytes=actual,
   credit_bytes=own["reserved_bytes"]-actual,projected_bytes=projected,scope_measurements=1,
   completed_overlay_prefix=dict(events=len(state[4])+1,tail_sha256=tail),old_journal_mutated=False,
   old_scope_mutated=False,artifacts_deleted=0,scientific_success_asserted=False,**warning_fields(projected))


def selector_draft_v1(*,policy_ref,source_review_ref,repair_authority_ref,frozen_parent_lineage):
 return dict(schema="explicit_ROOT_storage_policy_selector_v1",root_authorizer="root_agent",active=False,
  selector_source_ref=_ref(SOURCE),source_ref=_ref(SOURCE),policy_ref=policy_ref,source_manifest_ref=_ref(MANIFEST),
  measurement_source_ref=_ref(MEASURE_SOURCE),source_review_ref=source_review_ref,repair_authority_ref=repair_authority_ref,
  accounting_mode=MODE,journal=JOURNAL,frozen_parent_lineage=copy.deepcopy(frozen_parent_lineage),
  scheduling_ceiling_bytes=CEILING_BYTES,human_limit_bytes=HUMAN_LIMIT_BYTES,warning_threshold_bytes=WARNING_BYTES,
  no_fallback_discovery_upgrade_or_in_run_swap=True,future_ROOT_adapters_only=True,
  existing_adapters_runs_and_storage_records_unchanged=True,one_new_direct_guard_selector_capsule=True)

def _selected_record(record):
 require(isinstance(record,dict)and set(record)=={"path","sha256"},"Explicit immutable selector path/SHA; no discovery")
 v=_json(record);expected=selector_draft_v1(**{k:v.get(k)for k in("policy_ref","source_review_ref","repair_authority_ref","frozen_parent_lineage")})
 expected["active"]=True
 require(v==expected and set(v)==set(expected),"Exact reviewed Census_observation single-file selector capsule")
 for key in("selector_source_ref","source_ref","source_manifest_ref","measurement_source_ref","source_review_ref","repair_authority_ref"):_raw(v[key])
 p=_json(v["policy_ref"])
 require(p.get("active")is True and p.get("source_ref")==v["source_ref"]and p.get("source_manifest_ref")==v["source_manifest_ref"]
  and p.get("measurement_source_ref")==v["measurement_source_ref"]and p.get("source_review_ref")==v["source_review_ref"]
  and p.get("marker_repair_authority_ref")==v["repair_authority_ref"]and p.get("journal")==JOURNAL
  and p.get("base_ref")==BASE_REF and p.get("base_policy_ref")==BASE_POLICY_REF and p.get("base_selector_ref")==BASE_SELECTOR_REF
  and p.get("base_accounting_mode")==BASE_MODE and p.get("scheduling_ceiling_bytes")==CEILING_BYTES
  and p.get("human_limit_bytes")==HUMAN_LIMIT_BYTES and p.get("warning_threshold_bytes")==WARNING_BYTES,
  "Capsule authenticates exact selected policy/source/review/parent/ceiling")
 require(v["frozen_parent_lineage"]==dict(base_ref=BASE_REF,base_policy_ref=BASE_POLICY_REF,
  base_pins=p["base_pins"],base_projected_bytes=p["base_projected_bytes"],base_accounting_mode=BASE_MODE),
  "Actual frozen parent prefix/projection is never a planning forecast")
 if _POLICY_REF is not None:require(v["policy_ref"]==_POLICY_REF and v["repair_authority_ref"]==_REPAIR_AUTHORITY_REF,
  "Same configured exact lifetime policy/repair authority")
 return v

def load_selected_guard_v1(selector_ref):
 global _SELECTED_VALUE
 value=_selected_record(selector_ref)
 require(all(getattr(budget.v4,n).__module__==budget.v4.__name__ for n in("canonical","file_bytes","mutex")),
  "Select/configure once before consumer memo; no unwrap or reinstall")
 if _SELECTOR_REF is not None:
  require(_SELECTOR_REF==selector_ref and _SELECTED_VALUE==value,"No lifetime selector/source/policy swap")
 else:
  configure_with_renewal_v1(value["repair_authority_ref"],policy_ref=value["policy_ref"],selector_ref=selector_ref)
  _SELECTED_VALUE=copy.deepcopy(value)
 module=sys.modules.get(__name__)
 require(module is not None and module.budget is budget and module.MODE==MODE and module._POLICY_REF==value["policy_ref"],
  "Return this exact configured guard singleton; no old-guard fallback")
 return module

if __name__=="__main__":
 print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY_REVIEW_AND_ROOT_ACTIVATION_REQUIRED",mode=MODE,
  exact_failed_keys=TARGET_KEYS,live_operations=False,current_credit_or_capacity_claimed=False)))

MAX_PHASE_JSON=25165824

def _phase_json(record):
 raw=_raw(record)
 require(len(raw)<=MAX_PHASE_JSON,"Bounded original phase evidence JSON<=24MiB; never scientific payload")
 return budget.v4.parse(raw)

def prerequisite_admissions_draft_v1(*,H058v5_terminal_ref=None,FLARE_native_terminal_ref=None,released_at_utc=None):
 return dict(schema="Census_observation_parent_freeze_prerequisites_ROOT_v1",root_authorizer="root_agent",active=False,
  H058v5_admission_terminal_ref=H058v5_terminal_ref,FLARE_VAL001_native_admission_terminal_ref=FLARE_native_terminal_ref,
  ROOT_explicitly_releases_current_selector_after_both_admissions=True,released_at_utc=released_at_utc,
  no_parent_prefix_selected_by_this_record=True,no_credit_or_activation_by_this_record=True)

def _prerequisites(record,selected_at):
 v=_json(record)
 expected=prerequisite_admissions_draft_v1(H058v5_terminal_ref=PREREQUISITE_ADMISSIONS["H058v5"]["terminal_ref"],
  FLARE_native_terminal_ref=PREREQUISITE_ADMISSIONS["FLARE_VAL001_native"]["terminal_ref"],
  released_at_utc=v.get("released_at_utc"))
 expected["active"]=True
 require(v==expected and set(v)==set(expected),"Explicit later ROOT selector release; no release while ROOT keeps current selector")
 released=_stamp(v["released_at_utc"]);selected=_stamp(selected_at)
 require(released<=selected,"Actual frozen parent prefix selected only after ROOT release")
 for label,row in PREREQUISITE_ADMISSIONS.items():
  term=_json(row["terminal_ref"]);adm=_json(row["admitted_ref"]);inp=_json(row["input_ref"])
  receipt=_json(row["reserve_receipt_ref"]);storage=_json(row["storage_ref"])
  require(adm["input_ref"]==row["input_ref"]and adm["reserve_receipt_ref"]==row["reserve_receipt_ref"]
   and adm["consumer_binding_count"]==row["consumer_binding_count"]and inp["refs"]["storage"]==row["storage_ref"]
   and storage["reservation"]==receipt["value"]and receipt["kind"]=="concurrent_completed_scope_reserve_receipt_v1"
   and storage["reserved_new_bytes"]==receipt["value"]["reserved_bytes"]==row["reserved_bytes"]
   and storage["storage_policy_selector_ref"]==BASE_SELECTOR_REF,
   "Both original admitted/input/storage/receipt joins and unchanged exact parent selector")
  if label=="H058v5":
   require(term["schema"]=="ROOT_H058_GPU0_v5_actual_admission_and_execution_invocation_v24"
    and term["root_authorizer"]=="root_agent"and term["actual_admission_tool_session_id"]==72068
    and type(term["actual_admission_exit_code"])is int and term["actual_admission_exit_code"]==0
    and term["actual_admission_terminal_consumed"]is True
    and term["input_ref"]==row["input_ref"]and term["admitted_ref"]==row["admitted_ref"]
    and adm["whole_scope_reservation_bytes"]==row["reserved_bytes"]and _stamp(term["utc"])<=released,
    "Genuine consumed72068 admission0; H058 execution failure/wait status never upgraded")
  else:
   require(term["schema"]=="ROOT_actual_admission_terminal_consumed_v1"and term["tool_session_id"]==50131
    and type(term["outer_exit_code"])is int and term["outer_exit_code"]==0 and term["admission_only"]is True
    and term["input_ref"]==row["input_ref"]and term["consumer_binding_count"]==63
    and term["scientific_completion_claimed"]is False and inp["stage"]=="native"
    and inp["parent_id"]=="american_c3_val_001"and adm["storage"]==storage
    and _stamp(term["at_utc"])<=released,
    "Genuine consumed50131 native admission0 is not native execution completion")
 return v

PREREQUISITE_ADMISSIONS={'H058v5': {'terminal_ref': {'path': 'research/reviews/research_execution_ROOT_20261017_v24/H058_GPU0_v5_actual_admission_and_execution_invocation.root_v1.json', 'sha256': 'c4adb395e6360391ab71f92fda836f7ee2144310118fa80d6facee68b30d980c'}, 'admitted_ref': {'path': 'research/reviews/H058_GPU0_v5_ROOT_actual_root03_oct17/admitted.json', 'sha256': '4c84f72585e003df07c336e75f20f5f801220c7bba151d608457f7cb729a28ff'}, 'input_ref': {'path': 'research/reviews/H058_GPU0_v5_ROOT_actual_root03_oct17/input.json', 'sha256': 'e729c6cc967b71930dbd03064e49cf8d4fe62bcfe808e899d6c30cc33f9e5df5'}, 'storage_ref': {'path': 'research/reviews/H058_GPU0_v5_ROOT_actual_root03_oct17/storage.json', 'sha256': 'c21c0261c57630d1decf32576974702f47b9366db0b0998f4beb6b0624e34fdd'}, 'reserve_receipt_ref': {'path': 'research/reviews/H058_GPU0_v5_ROOT_actual_root03_oct17/reserve_receipt.json', 'sha256': '43cd745af70140f2a17a43b9735ada47a47f9231a32cea54be142d6cc5dd5e67'}, 'reserved_bytes': 268435456, 'consumer_binding_count': 64}, 'FLARE_VAL001_native': {'terminal_ref': {'path': 'research/reviews/research_execution_ROOT_20261017_v25/FLARE_VAL001_native_actual_admission_terminal_consumed.root_v1.json', 'sha256': '8aa02ce7e86b65665e05e8749f2b82d6d4a4e5552d4b53034ae8b7a3eff5d7a4'}, 'admitted_ref': {'path': 'research/reviews/FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17/admitted.json', 'sha256': '1f6a2e21727fac750be170ff6051a612a66e7def135512ddfc4e5b434d22d8ec'}, 'input_ref': {'path': 'research/reviews/FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17/input.json', 'sha256': 'a88d93a5abd7ce7e84850fbfa7b48a3a1eb82d75ca9d9518a8f29e786f44b970'}, 'storage_ref': {'path': 'research/reviews/FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17/storage.json', 'sha256': '539ba201576d5b5fe7523096b4166ec98215e785082112a57c352f51c48b1562'}, 'reserve_receipt_ref': {'path': 'research/reviews/FLARE_remaining_ROOT_actual_VAL001_native_bridgev2_root01_oct17/reserve_receipt.json', 'sha256': '54fee871d71063d6a80de23a7e95bde0820b49fe669ecf16a3c91193d2975a2b'}, 'reserved_bytes': 2005743919, 'consumer_binding_count': 63}}
