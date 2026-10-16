"""Exact two failed Census fit2 retention retirements over frozen active H020v3.

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
SOURCE='research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py'
JOURNAL='.runtime/storage/prospective_980gb_Census_fit2_v1'
MODE='prospective_980gb_Census_fit2_v1'
BASE_REF=dict(path='research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py',sha256='ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635')
BASE_POLICY_REF=dict(path='research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json',sha256='7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582')
BASE_MODE='prospective_980gb_H020_v3'
REPAIR_REF=dict(path='research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py',sha256='e4185dc51b285d8ad4eba224b8e1e15f3e1479f57c73b4ec53fff2b238d0ef2f')
CAPACITY_V1_REF=dict(path='research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py',sha256='7c0a35f82423eb824edf3b7dccbac31f139d50d2ee397d4d92515b6a0ae33c88')
MARKER_REF=dict(path='src/arggnn/project_budget_v9_consumer.py',sha256='c11a5543746899c8a387508e9877ce6ae2bf80fe4d3076c1dbe1cb8f7a43b859')
RENEWAL_REF=dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py',sha256='491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468')
RENEWAL_CERT_REF=dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json',sha256='66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7')
AUDIT_REF=dict(path='research/reviews/gpu3_Census_fit2_two_scope_retirement_evidence_20261016_v1/evidence_audit.v1.json',sha256='14bf31744492eea514e854766e4d3dd32a3bf53c4789aeb97b810cc8e1b15c2c')
GPU_POOL_REF=dict(path='src/arggnn/gpu_pool_v1.py',sha256='ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1')
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
RETIRE_OPERATION='retire_failed_Census_fit2_retention_v1'
_BASE=_POLICY_REF=_REPAIR_AUTHORITY_REF=None

FAILED_SCOPES={'budgeted_shared_pool_v3/gpu3_fits_v1/census-relate101-fit2-v1': {'auxiliary_scopes': [],
                                                                   'groups': [1313627, 1313689],
                                                                   'job': 'census-relate101-fit2-v1',
                                                                   'key': 'budgeted_shared_pool_v3/gpu3_fits_v1/census-relate101-fit2-v1',
                                                                   'output': '/workspace-SR003.nfs2/ARG-GNN/experiments/results/census_biological_fit_v2_gpu3_seed101_r001',
                                                                   'pool': 'gpu3_fits_v1',
                                                                   'refs': {'budget_admission': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/budget_admission.json',
                                                                                                 'sha256': 'e21b67c1cdad27fcafdc107696a5558727ec4dcf25bab3f70669ac1566b9004b'},
                                                                            'budget_receipt': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/budget_receipt.json',
                                                                                               'sha256': '7490ce18aa22b7373c90da1f27d84b42dfca2005cdf15c2ad91328a5c908a4e2'},
                                                                            'closed_registry': {'path': '.runtime/gpup/gpu3_fits_v1/registry.json',
                                                                                                'sha256': 'df36f2809c813ba6e01bfea13a49c1f86901126d68c58e9983ad5b4b3f9ed407'},
                                                                            'controller_started': {'path': 'experiments/results/gpu3_parallel_fits_v1_root/controller_started.json',
                                                                                                   'sha256': 'f62bae4fa5640fbdcb3d0dcbe26e16594ffc0100c35f4cabf6476159eca0fb0a'},
                                                                            'original_launch': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/launch.json',
                                                                                                'sha256': 'd9a07edc92fe16284530331ec6c702c5e5dd13ee8cb67b9589f9587ac4c7f5e8'},
                                                                            'paired_restart_reason': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/paired_restart_reason.json',
                                                                                                      'sha256': 'fa51a72ea45fa54df53d55719918c825a164bf5b75bb76a3210bf34ca51ff163'},
                                                                            'pool_admission': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/pool_admission.json',
                                                                                               'sha256': '7be887598da440d70d92cbeb1b59136887626f38167337738ccd4cd2586af100'},
                                                                            'supervisor_source': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/supervisor.py',
                                                                                                  'sha256': '6a3771fe56422102854decee46d483162433d87b89a04652d16524733eff660d'},
                                                                            'supervisor_started': {'path': 'experiments/results/gpu3_parallel_fits_v1_root/started.json',
                                                                                                   'sha256': '61a6f436c4ff8c68274051ae4eea2839c032ff29b3fc35b8c6f2de35f01d8ac6'},
                                                                            'supervisor_terminal': {'path': 'experiments/results/gpu3_parallel_fits_v1_root/exit.json',
                                                                                                    'sha256': '5cb0f612761bbedb96f951fc8433af9be9ddf4fc0a9c8d45225f8431d722c8fd'},
                                                                            'worker_admission': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/census_admission.json',
                                                                                                 'sha256': 'cbf9b9b2d4f08578b76b62fb5f095c111a8428c333e5b57e9872896d5e3f97ee'},
                                                                            'worker_launcher': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/census_launcher.json',
                                                                                                'sha256': '725f57c1c923c49d5aabb89fc2de6d2939aa8fe8e16936a5b81feb22b9f19179'},
                                                                            'worker_started': {'path': 'experiments/results/census_biological_fit_v2_gpu3_seed101_r001/started.json',
                                                                                               'sha256': 'e9fa88b0950ff205e03f7c119097279969047143bd1720b777014dfbdae6be37'},
                                                                            'worker_storage': {'path': 'research/reviews/gpu3_parallel_fits_v1_root/census_storage.json',
                                                                                               'sha256': '16b153e9ac41dec9390c6cac00278481dddae70ace9329fa9a2eb0b601a2acd3'}},
                                                                   'reservation_sha256': '23b4e0b0b9ebfe161094e6a2eeb9f2bb7b42ea1a7972d4936a7eb549e2923e6a',
                                                                   'reserved_bytes': 40000000000,
                                                                   'roots': [{'pid': 1313616,
                                                                              'role': 'root_supervisor',
                                                                              'start_ticks': 436595387},
                                                                             {'pid': 1313627,
                                                                              'role': 'pool_controller',
                                                                              'start_ticks': 436595455},
                                                                             {'pid': 1313689,
                                                                              'role': 'Census_scoped_writer',
                                                                              'start_ticks': 436596300},
                                                                             {'pid': 1313732,
                                                                              'role': 'recorded_worker_descendant',
                                                                              'start_ticks': 436596812}],
                                                                   'socket': '.runtime/gpup/gpu3_fits_v1/control.sock',
                                                                   'worker_pid': 1313689,
                                                                   'worker_start': 436596300},
 'budgeted_shared_pool_v3/gpu3_fits_v2/census-relate101-fit2-v2': {'auxiliary_scopes': [],
                                                                   'groups': [1317368, 1317433],
                                                                   'job': 'census-relate101-fit2-v2',
                                                                   'key': 'budgeted_shared_pool_v3/gpu3_fits_v2/census-relate101-fit2-v2',
                                                                   'output': '/workspace-SR003.nfs2/ARG-GNN/experiments/results/census_biological_fit_v2_gpu3_seed101_r002',
                                                                   'pool': 'gpu3_fits_v2',
                                                                   'refs': {'budget_admission': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/budget_admission.json',
                                                                                                 'sha256': '2001ce180e8475be9943683c7888f649c91e744ede0310dcf2b413816339bf04'},
                                                                            'budget_receipt': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/budget_receipt.json',
                                                                                               'sha256': '9f2119c7fbcdb383cff270dab068a97d3b0c8409ec4ae6695d3276450635b47d'},
                                                                            'closed_registry': {'path': '.runtime/gpup/gpu3_fits_v2/registry.json',
                                                                                                'sha256': '472abb1ba06b4e2dc82e1cc2452dc22089e1bb5dc75f743d7d55e7b8925cd151'},
                                                                            'controller_started': {'path': 'experiments/results/gpu3_parallel_fits_v2_root/controller_started.json',
                                                                                                   'sha256': '0abb8b431e89a0a09cecd6b41bc4753cf3e601c5cc21654a9e4c55ba61c3b3b3'},
                                                                            'original_launch': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/launch.json',
                                                                                                'sha256': 'aca2d3ac06f3ae0d51f97ac7874d94383b0fecb075196803c204f8f4076a0888'},
                                                                            'pool_admission': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/pool_admission.json',
                                                                                               'sha256': 'e2eb497d855218104892575f902e38115880652d880d330cbae483ed9abb8a49'},
                                                                            'supervisor_source': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/supervisor.py',
                                                                                                  'sha256': 'a081737317fb0f3d37363c9eb3c5e7b3cef9a68fc201e531838f6f7896eae30d'},
                                                                            'supervisor_started': {'path': 'experiments/results/gpu3_parallel_fits_v2_root/started.json',
                                                                                                   'sha256': 'e8005154afc0a0a8ebec76c83c66435f683f01415e27f309da3b9cc7b5298e60'},
                                                                            'supervisor_terminal': {'path': 'experiments/results/gpu3_parallel_fits_v2_root/exit.json',
                                                                                                    'sha256': 'ad42cd69996c0a4a62d877b730964843538a97bff29e3d437a7c1d1b093785a2'},
                                                                            'worker_admission': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/census_admission.json',
                                                                                                 'sha256': '24725b87aa564cfc6da9be0ea99e8b43044c31c084f38b14cb7a0b84c3b7965f'},
                                                                            'worker_launcher': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/census_launcher.json',
                                                                                                'sha256': 'f4fbb20f21f3e39661c8265e676c02bb9e77e1280ac07d0d5df14c8e0d0424fc'},
                                                                            'worker_started': {'path': 'experiments/results/census_biological_fit_v2_gpu3_seed101_r002/started.json',
                                                                                               'sha256': '3f2a9fbd1eb076404fb48d04d0fab3ff88e985b98cc21c56bfbde05ccfad2900'},
                                                                            'worker_storage': {'path': 'research/reviews/gpu3_parallel_fits_v2_root/census_storage.json',
                                                                                               'sha256': '58391656adb2a1162bce75f83aeee955f95742f3a0e2ab91e9a47ae94d1fa4aa'}},
                                                                   'reservation_sha256': '796a1afcd4869f309f17df858836a1fb2b3a7fdec2b8598f5d57cec453adedd3',
                                                                   'reserved_bytes': 40000000000,
                                                                   'roots': [{'pid': 1317352,
                                                                              'role': 'root_supervisor',
                                                                              'start_ticks': 436676816},
                                                                             {'pid': 1317368,
                                                                              'role': 'pool_controller',
                                                                              'start_ticks': 436676880},
                                                                             {'pid': 1317433,
                                                                              'role': 'Census_scoped_writer',
                                                                              'start_ticks': 436677698}],
                                                                   'socket': '.runtime/gpup/gpu3_fits_v2/control.sock',
                                                                   'worker_pid': 1317433,
                                                                   'worker_start': 436677698}}



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
    return dict(schema='prospective980_Census_fit2_root_frozen_H020v3_activation_v1',
        root_authorizer='root_agent',active=False,source_ref=_ref(SOURCE),
        frozen_parent_ref=BASE_REF,frozen_parent_policy_ref=BASE_POLICY_REF,
        frozen_parent_pins=copy.deepcopy(base_pins),frozen_parent_projected_bytes=base_projected_bytes,
        new_journal=JOURNAL,retirement_keys=sorted(FAILED_SCOPES),
        all_future_parent_reserve_close_retire_mutations_cooperatively_revoked=True,
        all_older_accounting_future_mutation_revocations_preserved=True,
        current_parent_admitted_writers_continue_unchanged_with_full_original_caps=True,
        old_parent_source_policy_journal_and_consumer_storage_records_unchanged=True,
        active_C3_H058_source_policy_admission_caps_unchanged=True,
        no_historical_mutation=True,no_deletion=True,no_rebase=True,physical_GPU_ids=[0])

def activation_policy_draft_v1(*,repair_authority_ref,base_pins,base_projected_bytes,implementation_ref,launcher_ref,root_activation_ref=None):
    return dict(schema='prospective980_Census_fit2_retention_policy_v1',root_authorizer='root_agent',active=False,
        source_ref=_ref(SOURCE),base_ref=BASE_REF,base_policy_ref=BASE_POLICY_REF,base_accounting_mode=BASE_MODE,
        repair_ref=REPAIR_REF,capacity_v1_ref=CAPACITY_V1_REF,marker_ref=MARKER_REF,
        marker_repair_authority_ref=repair_authority_ref,implementation_ref=implementation_ref,launcher_ref=launcher_ref,
        renewal_ref=RENEWAL_REF,renewal_certificate_ref=RENEWAL_CERT_REF,evidence_audit_ref=AUDIT_REF,
        base_pins=copy.deepcopy(base_pins),base_projected_bytes=base_projected_bytes,journal=JOURNAL,
        root_activation_ref=root_activation_ref,scheduling_ceiling_bytes=CEILING_BYTES,
        warning_threshold_bytes=WARNING_BYTES,human_limit_bytes=HUMAN_LIMIT_BYTES,
        bootstrap_metadata_charge_bytes=BOOTSTRAP_CHARGE,per_event_metadata_charge_bytes=META_CHARGE,
        all_old_future_admissions_revoked=True,all_future_admissions_use_this_overlay=True,
        old_active_writers_continue_unchanged_fully_charged=True,old_journals_frozen_for_new_events=True,
        all_budget_writers_use_original_mutex=True,closed_scope_all_write_resume_capabilities_revoked=True,
        no_historical_mutation=True,no_rebase=True,no_deletion=True,
        cooperative_root_process_rosters_complete=True,external_NFS_writers_outside_observation=True,
        new_unscoped_metadata_within_charged_allowances=True,never_credit_keys=[MALFORMED_KEY],
        eligible_physical_GPU_ids=[0],all_nonzero_physical_GPU_ids_excluded=True,
        actual_GPU_assignment_and_GPU_launcher_guards_owned_by_root=True,
        only_failed_retention_keys=sorted(FAILED_SCOPES),failed_retirements_are_successful_completion=False,
        original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN')

def configure_v2(repair_authority_ref,*,policy_ref=None):
    global _BASE,_POLICY_REF,_REPAIR_AUTHORITY_REF
    if policy_ref is None:
        policy_ref=json.loads(os.environ['ARGGNN_COMPLETED_CREDITS_POLICY_REF'])
    require(_POLICY_REF is None or _POLICY_REF==policy_ref,'No policy swap inside a configured consumer')
    for r in (BASE_REF,BASE_POLICY_REF,REPAIR_REF,CAPACITY_V1_REF,MARKER_REF,RENEWAL_REF,RENEWAL_CERT_REF,AUDIT_REF):_raw(r)
    _POLICY_REF=copy.deepcopy(policy_ref);_REPAIR_AUTHORITY_REF=copy.deepcopy(repair_authority_ref)
    _policy()
    sp=importlib.util.spec_from_file_location('_Census_fit2_frozen_H020v3',ROOT/BASE_REF['path'])
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
    require(_POLICY_REF is not None,'ROOT must configure exact active overlay policy')
    p=_json(_POLICY_REF)
    expected=activation_policy_draft_v1(repair_authority_ref=_REPAIR_AUTHORITY_REF,
        base_pins=p.get('base_pins'),base_projected_bytes=p.get('base_projected_bytes'),
        implementation_ref=p.get('implementation_ref'),launcher_ref=p.get('launcher_ref'),
        root_activation_ref=p.get('root_activation_ref'))
    expected['active']=True
    require(set(p)==set(expected) and budget.v4.encoded(p)==budget.v4.encoded(expected),'Exact source/policy/frozen-parent/revocation')
    for r in (p['source_ref'],p['base_ref'],p['base_policy_ref'],p['repair_ref'],p['capacity_v1_ref'],p['marker_ref'],
              p['marker_repair_authority_ref'],p['implementation_ref'],p['launcher_ref'],
              p['renewal_ref'],p['renewal_certificate_ref'],p['evidence_audit_ref']):_raw(r)
    require(p['implementation_ref']['path']==budget.SOURCE_REL and p['launcher_ref']['path']==budget.LAUNCHER_REL,
        'Original actual implementation/launcher required')
    budget.v4.integer(p['base_projected_bytes'],'actual fixed parent projection')
    require(isinstance(p['base_pins'],dict) and set(p['base_pins'])=={'base_pins','completed_overlay_prefix'},
        'Actual exact current H020v3 parent pins required, never an illustrative saved prefix')
    activation=_json(p['root_activation_ref'])
    required=root_activation_draft_v1(base_pins=p['base_pins'],base_projected_bytes=p['base_projected_bytes'])
    required['active']=True
    require(set(activation)==set(required) and budget.v4.encoded(activation)==budget.v4.encoded(required),'Explicit ROOT actual frozen-parent activation/revocation required')
    return p

def _bindings(base_bindings,p):
    result=copy.deepcopy(base_bindings)
    additions=dict(failed_fit2_overlay=_ref(SOURCE),failed_fit2_policy=_POLICY_REF,
        failed_fit2_frozen_parent=BASE_REF,failed_fit2_frozen_parent_policy=BASE_POLICY_REF,
        failed_fit2_root_activation=p['root_activation_ref'],failed_fit2_evidence_audit=AUDIT_REF,
        failed_fit2_original_renewal=RENEWAL_REF,failed_fit2_original_renewal_certificate=RENEWAL_CERT_REF)
    require(not set(result)&set(additions),'Never overwrite an ancestor binding')
    result.update(additions)
    require(len({r['path'] for r in result.values()})<=64,'All direct ancestor roles within unchanged64-ref ceiling')
    return result

def operation_draft_v1(operation,request,*,expected_before,bindings):
    require(operation in ('reserve','close_completed',RETIRE_OPERATION),'Only ordinary reserve/close or exact two-key failure retention')
    return dict(schema='prospective980_Census_fit2_root_operation_v1',root_authorizer='root_agent',active=False,
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
            and event['schema']=='prospective980_Census_fit2_event_v1' and type(event['sequence']) is int and event['sequence']==i
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
    require(len(base)==7,'Exact frozen H020v3 base state API')
    reservations,base_closed,view,base_bindings,_,_,_=base
    require(_BASE._pins(view)==p['base_pins'] and view['projected_bytes']==p['base_projected_bytes']
        and view['accounting_mode']==BASE_MODE and view['limit_bytes']==CEILING_BYTES
        and view['hard_limit_bytes']==HUMAN_LIMIT_BYTES,
        'Frozen H020v3 parent prefix/projection changed; no new charge or credit')
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
            key=request['key'];require(key in FAILED_SCOPES,'Only exact two failed Census keys')
            assert_open(key,reservations,closed);own=reservations[key]
            proof=_failed_proof(request,own)
            require(set(value)=={'request','reservation_sha256','measurement','proof','closed_utc','future_writes_and_relaunch_revoked'}
                and value['request']==request and value['reservation_sha256']==budget.digest(own)
                and value['future_writes_and_relaunch_revoked'] is True,'Exact failed retention certificate')
            _failed_recorded_proof(value['proof'],proof,value['closed_utc'])
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
    event=dict(schema='prospective980_Census_fit2_event_v1',sequence=len(events)+1,previous_sha256=hashes[-1] if hashes else None,
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

def _failed_spec(key):
    require(key in FAILED_SCOPES,'No generic failed-run credit or key alias')
    return FAILED_SCOPES[key]

def _stamp(text):
    require(isinstance(text,str),'Timezone-aware evidence timestamp required')
    value=datetime.fromisoformat(text.replace('Z','+00:00'))
    require(value.tzinfo is not None and value<=datetime.now(timezone.utc),'Nonfuture evidence timestamp')
    return value

def failed_revocation_draft_v1(key):
    s=_failed_spec(key)
    return dict(schema='Census_fit2_exact_failed_scope_permanent_revocation_v1',root_authorizer='root_agent',
        active=False,key=key,reservation_sha256=s['reservation_sha256'],output=s['output'],auxiliary_scopes=[],
        original_refs=copy.deepcopy(s['refs']),revoked_at_utc=None,
        all_original_scoped_write_resume_relaunch_capabilities_permanently_cooperatively_revoked=True,
        original_pool_start_and_submit_routes_for_this_scope_permanently_cooperatively_revoked=True,
        original_supervisor_and_direct_worker_CLI_aliases_for_this_scope_permanently_cooperatively_revoked=True,
        all_source_determined_scoped_writer_roots_listed=True,
        roster_is_pinned_cooperative_route_not_complete_historical_descendant_inventory=True,
        original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',
        scientific_success_asserted=False,optimizer_updates='UNKNOWN',preserve_all_original_files=True,
        no_other_reservation_retired=True,external_NFS_writers_outside_observation=True)

def _failed_revocation(record,s):
    value=_json(record);expected=failed_revocation_draft_v1(s['key'])
    expected.update(active=True,revoked_at_utc=value.get('revoked_at_utc'))
    require(set(value)==set(expected) and budget.v4.encoded(value)==budget.v4.encoded(expected),'Exact permanent ROOT two-key scoped revocation')
    _stamp(value['revoked_at_utc'])
    return value

def failed_observation_draft_v1(key,*,revocation_ref):
    s=_failed_spec(key)
    return dict(schema='Census_fit2_exact_failed_scope_targeted_ROOT_observation_v1',
        root_authorizer='root_agent',active=False,key=key,reservation_sha256=s['reservation_sha256'],
        revocation_ref=copy.deepcopy(revocation_ref),observed_at_utc=None,
        known_scoped_route_roots=copy.deepcopy(s['roots']),recorded_or_source_derived_session_groups=s['groups'][:],
        control_socket_path=s['socket'],original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',
        PID_observations=[],group_observations=[],endpoint_observations=[])

def _failed_pid_absent(root):
    row=_absent(root['pid'])
    return dict(pid=root['pid'],start_ticks=root['start_ticks'],absent=True,method=row['method'],checked_utc=row['checked_utc'])

def _failed_group_absent(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:
        return dict(pgid=pgid,absent=True,method='targeted os.killpg(pgid,0) ESRCH',checked_utc=_utc())
    except PermissionError:raise ValueError('Targeted exact old group absence denied') from None
    raise ValueError('Exact old group remains or numeric group reused; no retirement')

def _failed_endpoint_absent(relative):
    p=budget.v4.canonical(ROOT,relative)
    try:p.lstat()
    except FileNotFoundError:pass
    else:raise ValueError('Old control endpoint pathname still exists; preserve and block')
    with Path('/proc/net/unix').open('rb') as f:raw=f.read(16*2**20+1)
    require(len(raw)<=16*2**20,'Bounded kernel AF_UNIX endpoint inventory required')
    target=str(p).encode()
    require(not any(len(row.split())>=8 and row.split()[7]==target for row in raw.splitlines()[1:]),
        'Exact old kernel control endpoint still registered')
    return dict(path=relative,absent=True,method='targeted canonical lstat absent and /proc/net/unix exact-path absence',checked_utc=_utc())

def _failed_targets(s):
    return dict(PID_observations=[_failed_pid_absent(r) for r in s['roots']],
        group_observations=[_failed_group_absent(p) for p in s['groups']],
        endpoint_observations=[_failed_endpoint_absent(s['socket'])])

def _failed_origin(own):
    s=_failed_spec(own['key'])
    require(budget.digest(own)==s['reservation_sha256'] and own['reserved_bytes']==40000000000
        and own['output']==s['output'] and own['auxiliary_scopes']==[],'Exact immutable original40GB reservation/output/empty auxiliaries')
    _raw(AUDIT_REF);_raw(GPU_POOL_REF)
    for r in own['consumer_bindings']:_raw(r)
    for r in s['refs'].values():_raw(r)
    r=s['refs'];reserve=_json(r['budget_receipt']);budget_admission=_json(r['budget_admission'])
    admitted_rows=[row for row in budget_admission['reservations'] if row.get('key')==s['key']]
    require(own['root_admission']==r['budget_admission'] and reserve['admitted_reservations'][s['key']]==own
        and len(admitted_rows)==1 and admitted_rows[0]=={k:v for k,v in own.items() if k!='root_admission'},
        'Original budget admission and reserve receipt must bind exact reservation')
    a=_json(r['worker_admission']);storage=_json(r['worker_storage']);launcher=_json(r['worker_launcher'])
    started=_json(r['worker_started']);pool=_json(r['pool_admission'])
    job_rows=[j for j in pool['plan']['jobs'] if j['spec']['job']==s['job']]
    require(len(job_rows)==1 and job_rows[0]['reservation']==own and job_rows[0]['spec']['output']==s['output']
        and job_rows[0]['spec']['storage_bytes']==40000000000
        and job_rows[0]['spec']['module']=='arggnn.census_biological_fit_v2'
        and pool['plan']['pool']['name']==s['pool'],'Original exact pool route and full worker cap')
    require(a['job_id']==storage['job_id']==launcher['job_id']==s['job']
        and a['output_directory']==storage['output_directory']==str(Path(s['output']).relative_to(ROOT))
        and storage['reserved_new_bytes']==40000000000 and a['refs']['storage']==r['worker_storage']
        and a['refs']['launcher']==r['worker_launcher'] and started['admission_ref']==r['worker_admission']
        and started['job_id']==s['job'] and started['runtime']['pid']==s['worker_pid'],
        'Authentic original expired worker/storage/launcher/started lineage')
    expiry=[a['expires_at_utc'],storage['expires_at_utc'],launcher['expires_at_utc'],pool['expires_utc']]
    for text in expiry:_stamp(text)
    supervisor=_json(r['supervisor_started'])['supervisor'];controller=_json(r['controller_started'])['identity']
    launch=_json(r['original_launch'])
    require((supervisor['pid'],supervisor['start_ticks'])==(s['roots'][0]['pid'],s['roots'][0]['start_ticks'])
        and (controller['pid'],controller['start_ticks'])==(s['roots'][1]['pid'],s['roots'][1]['start_ticks'])
        and (launch['supervisor_pid'],launch['start_ticks'])==(supervisor['pid'],supervisor['start_ticks']),
        'Exact original supervisor/controller root identities')
    registry=_json(r['closed_registry'])
    jobs=[j for j in registry['jobs'] if j['logical_job']==s['job']]
    require(len(jobs)==1,'One exact recorded failed worker')
    job=jobs[0]
    require(registry['lifecycle']=='closed' and registry['pool']==s['pool'] and job['state']=='failed'
        and type(job['exit_code']) is int and job['exit_code']==-15
        and (job['pid'],job['leader_start_ticks'])==(s['worker_pid'],s['worker_start'])
        and job['requested']==job_rows[0]['spec'],'Preserve exact closed-registry/failed-minus15 lineage')
    terminal=_json(r['supervisor_terminal'])
    require(terminal['status']=='failed_partial_preserved' and terminal['error']=="FileNotFoundError(2, 'No such file or directory')",
        'Preserve original root supervisor failure, never successful completion')
    return s,expiry

def _failed_observation(record,s,expiry,*,fresh):
    value=_json(record)
    expected=failed_observation_draft_v1(s['key'],revocation_ref=value.get('revocation_ref'))
    expected.update(active=True,observed_at_utc=value.get('observed_at_utc'),
        PID_observations=value.get('PID_observations'),group_observations=value.get('group_observations'),
        endpoint_observations=value.get('endpoint_observations'))
    require(set(value)==set(expected) and budget.v4.encoded(value)==budget.v4.encoded(expected),'Exact known-route observation schema')
    rev=_failed_revocation(value['revocation_ref'],s);observed=_stamp(value['observed_at_utc'])
    require(all(_stamp(t)<=_stamp(rev['revoked_at_utc'])<=observed for t in expiry),'Old authorities expired before permanent revocation/observation')
    expected_roots=[dict(pid=r['pid'],start_ticks=r['start_ticks'],absent=True,method='targeted os.kill(pid,0) ESRCH') for r in s['roots']]
    expected_groups=[dict(pgid=p,absent=True,method='targeted os.killpg(pgid,0) ESRCH') for p in s['groups']]
    expected_endpoints=[dict(path=s['socket'],absent=True,method='targeted canonical lstat absent and /proc/net/unix exact-path absence')]
    for field,rows in (('PID_observations',expected_roots),('group_observations',expected_groups),('endpoint_observations',expected_endpoints)):
        actual=value[field];require(isinstance(actual,list) and len(actual)==len(rows),'Exact bounded absence roster')
        for row,shape in zip(actual,rows):
            require(set(row)==set(shape)|{'checked_utc'} and all(budget.v4.encoded(row[k])==budget.v4.encoded(v) for k,v in shape.items()),'Exact conservative absence fact')
            stamp=_stamp(row['checked_utc'])
            require(_stamp(rev['revoked_at_utc'])<=stamp<=observed,'Absence cannot predate revocation or postdate certificate')
            if fresh:require((datetime.now(timezone.utc)-stamp).total_seconds()<=300,'Target absence certificate older than five minutes')
    return value

def failed_request_draft_v1(key,*,observation_ref):
    s=_failed_spec(key)
    return dict(key=key,reservation_sha256=s['reservation_sha256'],observation_ref=copy.deepcopy(observation_ref))

def root_observe_failed_v1(key,*,revocation_ref,expected_source_sha256,expected_launcher_sha256):
    """ROOT only after explicit revocation; actual reads, no writes or accounting mutation."""
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        assert_open(key,state[0],state[1]);s,expiry=_failed_origin(state[0][key])
        rev=_failed_revocation(revocation_ref,s)
        require(all(_stamp(t)<=_stamp(rev['revoked_at_utc']) for t in expiry),'Expired original authorities required')
        value=failed_observation_draft_v1(key,revocation_ref=revocation_ref)
        value.update(active=True,**_failed_targets(s),observed_at_utc=_utc())
        return value

def _failed_proof(request,own,*,observe=False):
    require(set(request)=={'key','reservation_sha256','observation_ref'}
        and request['key']==own['key'] and request['reservation_sha256']==budget.digest(own),'Exact two-key failed retention request')
    s,expiry=_failed_origin(own)
    obs=_failed_observation(request['observation_ref'],s,expiry,fresh=observe)
    facts=_failed_targets(s) if observe else {k:copy.deepcopy(obs[k]) for k in ('PID_observations','group_observations','endpoint_observations')}
    if observe:
        require(not (Path(own['output'])/'completed.json').exists(),'Unexpected completion file blocks failed-only disposition')
    return dict(schema='Census_fit2_original_failed_retention_proof_v1',key=s['key'],
        reservation_sha256=s['reservation_sha256'],original_refs=copy.deepcopy(s['refs']),
        root_observation_ref=request['observation_ref'],permanent_revocation_ref=obs['revocation_ref'],
        original_controller_recorded_exit_code=-15,original_supervisor_status='failed_partial_preserved',
        original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',optimizer_updates='UNKNOWN',
        scientific_success_asserted=False,retirement_kind='failed_revoked_retention_only',
        known_roots=copy.deepcopy(s['roots']),known_groups=s['groups'][:],
        **facts,post_measurement_PID_observations=copy.deepcopy(facts['PID_observations']),
        post_measurement_group_observations=copy.deepcopy(facts['group_observations']),
        post_measurement_endpoint_observations=copy.deepcopy(facts['endpoint_observations']))

def _failed_recorded_proof(recorded,expected,closed_utc):
    collections=('PID_observations','group_observations','endpoint_observations')
    variable=set(collections)|{'post_measurement_'+k for k in collections}
    require(set(recorded)==set(expected) and all(recorded[k]==expected[k] for k in expected if k not in variable),
        'Original failed lineage/unknown-history proof drift')
    end=_stamp(closed_utc)
    for field in collections:
        previous=expected[field]
        for name in (field,'post_measurement_'+field):
            rows=recorded[name];require(isinstance(rows,list) and len(rows)==len(previous),'Applied exact absence count')
            for row,old in zip(rows,previous):
                require(set(row)==set(old) and all(row[k]==old[k] for k in old if k!='checked_utc')
                    and _stamp(old['checked_utc'])<=_stamp(row['checked_utc'])<=end,'Applied absence identity/method/time drift')
            previous=rows

def apply_retire_failed_fit2_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
        auth=_authority(authority_ref,RETIRE_OPERATION,state[3],fresh=True)
        require(auth['expected_before']==_pins(state[2]),'Exact fresh successor prefix before retirement')
        request=auth['request'];key=request['key'];_failed_spec(key)
        assert_open(key,state[0],state[1]);own=state[0][key]
        proof=_failed_proof(request,own,observe=True)
        measurement=budget.v4.measure_scopes(ROOT,_paths(own))
        actual=_measurement(measurement,own)
        post=_failed_targets(_failed_spec(key))
        for name,rows in post.items():proof['post_measurement_'+name]=rows
        require(not (Path(own['output'])/'completed.json').exists(),'Unexpected completion appeared; block failed-only credit')
        projected=charge(_policy()['base_projected_bytes'],state[2]['new_reserved_bytes'],
            state[2]['completed_credit_bytes']+own['reserved_bytes']-actual,len(state[4])+1)
        value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
            proof=proof,closed_utc=_utc(),future_writes_and_relaunch_revoked=True)
        _failed_recorded_proof(proof,_failed_proof(request,own),value['closed_utc'])
        tail=_append(RETIRE_OPERATION,authority_ref,value,state)
        return dict(kind='Census_fit2_failed_retention_retirement_receipt_v1',key=key,value=value,
            retained_bytes=actual,credit_bytes=own['reserved_bytes']-actual,projected_bytes=projected,scope_measurements=1,
            completed_overlay_prefix=dict(events=len(state[4])+1,tail_sha256=tail),
            old_journal_mutated=False,old_scope_mutated=False,artifacts_deleted=0,
            scientific_success_asserted=False,original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',
            **warning_fields(projected))


if __name__=='__main__':
    print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE_ROOT_ACTIVATION_REQUIRED',accounting_mode=MODE,
        journal=JOURNAL,retirement_keys=sorted(FAILED_SCOPES),capacity_recovered=False,live_operations_performed=False)))

