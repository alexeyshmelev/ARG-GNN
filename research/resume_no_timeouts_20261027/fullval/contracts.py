"""Exact full200Mb preregistration and ROOT owner-authorized operational gates."""
import json
from pathlib import Path
from research.efficiency_20261027.execution.european_short_val_runner.contracts_v3 import ROOT,need,contained,ref,read

HERE=Path(__file__).parent
RUNS=ROOT/'research/efficiency_20261027/root/runs'
AGGREGATE=RUNS/'european_independent_validation_200mb_v2_no_timeouts'
MODULE='research.resume_no_timeouts_20261027.fullval.worker'
GIB=2**30
HOST_CAP=128*GIB
RESERVATION=120*GIB+128*2**20
OPERATING=RESERVATION-4*GIB
MAX_SECONDS=None  # No aggregate, generation, native-stage or native-step elapsed-time cutoff.
HEADROOM=GIB
METADATA_CAP=256*2**20
ROOT_STATUS_MAX_AGE_SECONDS=120
GROWTH_INTERVAL_SECONDS=30
PROJECT_CAP=1_200_000_000_000
PROJECT_ADMISSION_CAP=PROJECT_CAP-5_000_000_000
RETAINED_ATTEMPTS_RESERVATION=16*GIB+128*2**20
RETAINED_PATH=HERE/'RETAINED_ATTEMPTS.json'
PLAN_REF=dict(path='research/efficiency_20261027/architecture/european_cohort_candidate/independent_validation_plan_v2.json',sha256='919d2feb0debf2071441aa1893a0f4b21537611789c158709901a26aea31321b')
NATIVE_CONFIG_REF=dict(path='configs/benchmarks/ap200mb_native_runtime_v2.json',sha256='a86720db22c176bf194ed1a3ec704e82126f4be14fb514fc4ac43c3c5176482e')
NATIVE_CHAIN_REF=dict(path='research/efficiency_20261027/execution/european_short_val_runner/NATIVE_CHAIN_REVIEW_v1.json',sha256='a4e811711325b87500d4661c16474ca45f83dcca0d440e54302f743eb763fd3a')
OWNER_AUTH_PATH=ROOT/'research/efficiency_20261027/root/owner_storage_authorization_1p2TB_v1.json'


def validate_plan():
    plan=read(PLAN_REF);recipe=read(plan['canonical_recipe_ref']);sampling=read(plan['sampling_plan_ref'])
    need(plan['parameters']==recipe['parameters'] and recipe['parameters']['length_bp']==200000000
         and recipe['parameters']['haplotypes']==952 and len(plan['parents'])==1,
         'Exact original complete200Mb recipe and one independent validation parent')
    parent=plan['parents'][0]
    need(parent['parent_id']=='ancestralpaths_european_validation_20261027_000'
         and parent['split']=='validation' and parent['seeds']==dict(ancestry=3519499615,mutations=1701692571,
             relate_initial=1802632633,relate_size_base=1864819694), 'Original preregistered parent and independent seeds')
    need(sampling['sequence_length_bp']==200000000 and sampling['validation_examples_per_parent']==10000
         and sampling['validation_parent_ids']==[parent['parent_id']], 'Original full-length10000-row validation sampling')
    for role in ('generator','registry','inference'):
        need(ref(plan[role]['source_ref']['path'])==plan[role]['source_ref'],'Frozen scientific source changed')
    return dict(plan=plan,recipe=recipe,parent=parent,sampling_plan=sampling)


def validate_native_config(config):
    need(config==read(NATIVE_CONFIG_REF),'Exact original full200Mb native runtime configuration')
    return config


def native_bindings():
    config=validate_native_config(read(NATIVE_CONFIG_REF))
    installation=read(config['relate']['installation']);release=contained(installation['release_root'])
    chain=read(NATIVE_CHAIN_REF)
    expected={str(release/key):digest for key,digest in config['relate']['tool_sha256'].items()}
    need(chain['passed'] is True and chain['encoded_executable_payloads'] is False
         and chain['native_config_ref']==NATIVE_CONFIG_REF and chain['reviewed_native_tools']==expected,
         'Original installed plaintext native chain review and exact seven tools')
    for path,digest in expected.items():need(ref(path)['sha256']==digest,'Native binary/script changed')
    return config,expected,NATIVE_CHAIN_REF


def owner_authorization():
    reference=ref(OWNER_AUTH_PATH);value=read(reference)
    need(value['schema']=='ARGGNN_owner_storage_authorization_v1' and value['authorized_limit_bytes']==PROJECT_CAP
         and value['scope']==str(ROOT)+' on anogena' and value['existing_jobs_unchanged'] is True
         and value['failed_attempt_evidence_preserved'] is True and value['automatic_deletion_authorized'] is False,
         'Explicit owner1.2TB authorization without deletion or live-job changes')
    return reference


def retained_attempts():
    from .aggregate_guard import measure
    reference=ref(RETAINED_PATH);value=read(reference)
    need(value['reservation_bytes']==RETAINED_ATTEMPTS_RESERVATION and len(value['attempts'])==3,
         'Retained short attempts plus failed full-size predecessor reservation')
    total=0
    for item in value['attempts']:
        scope=contained(item['directory']);need(scope.parent==RUNS and scope!=AGGREGATE,'Exact retained ROOT scope')
        terminal=read(item['actual_wait_ref'])
        need(terminal['actual_wait'] is True and terminal['actual_returncode']==1
             and terminal['owned_family_absent'] is True,'All predecessor attempts actually failed and were reaped')
        for evidence in item['evidence_refs']:need(ref(evidence['path'])==evidence,'Retained terminal evidence changed')
        total+=measure(scope)['charged_bytes']
    need(total<=RETAINED_ATTEMPTS_RESERVATION,'Retained predecessor scopes exceed their conservative combined allowance')
    return reference,total


def verify_sources(digest,review_ref):
    manifest_ref=ref(HERE/'source_pins.json');need(manifest_ref['sha256']==digest,'Exact full validation runner source manifest')
    manifest=read(manifest_ref)
    need(manifest['schema']=='European_full_VAL_ROOT_runner_sources_v1','Exact full validation source schema')
    for item in manifest['source_refs']:need(ref(item['path'])==item,'Pinned source/evidence changed: '+item['path'])
    need(owner_authorization() in manifest['source_refs'] and PLAN_REF in manifest['source_refs'],
         'Exact owner authorization and immutable full-size registration pinned')
    review=read(review_ref)
    need(review['schema']=='European_full_VAL_ROOT_runner_review_v1' and review['passed'] is True
         and review['source_manifest_ref']==manifest_ref and review['native_terminal_race_fix_reviewed'] is True
          and review.get('atomic_status_race_fix_reviewed') is True
          and review.get('elapsed_time_cutoffs_removed_reviewed') is True
          and review.get('campaign_transition_retry_reviewed') is True,
         'Independent exact full validation runner and native drain review required')
    return manifest_ref,validate_plan()


def validate_release(value,manifest_ref,review_ref,peers):
    need(value.get('schema')=='European_full_VAL_ROOT_execution_release_v1' and value.get('passed') is True
         and value.get('execution_authorized') is True and value.get('source_manifest_ref')==manifest_ref
         and value.get('independent_source_review_ref')==review_ref and value.get('plan_ref')==PLAN_REF
         and value.get('owner_storage_authorization_ref')==owner_authorization()
         and value.get('aggregate_directory')==str(AGGREGATE.relative_to(ROOT))
         and value.get('host_cap_bytes')==HOST_CAP and value.get('reservation_bytes')==RESERVATION
         and value.get('operating_stop_bytes')==OPERATING and value.get('max_seconds')==MAX_SECONDS
         and value.get('CUDA_VISIBLE_DEVICES')=='' and value.get('peer_directories')==peers
         and value.get('global_project_ceiling_bytes')==PROJECT_CAP
         and value.get('ROOT_live_global_storage_and_peer_admission_required') is True
         and value.get('model_fit_prediction_or_scoring_authorized') is False
         and value.get('ROOT_status_max_age_seconds')==ROOT_STATUS_MAX_AGE_SECONDS
         and value.get('growth_refresh_interval_seconds')==GROWTH_INTERVAL_SECONDS
         and value.get('retained_attempts_ref')==ref(RETAINED_PATH)
         and value.get('retained_attempts_reservation_in_other_bytes')==RETAINED_ATTEMPTS_RESERVATION,
         'Exact ROOT full-size execution release and retained resource envelope')
    need(type(peers) is list and len(peers)==len(set(peers)),'Distinct explicit current ROOT peers')
    retained={item['directory'] for item in read(ref(RETAINED_PATH))['attempts']}
    for peer in peers:need(contained(peer).parent==RUNS and contained(peer)!=AGGREGATE and peer not in retained,
                          'Retained predecessor scopes are counted once through other-project allowance')
    return value


def validate_worker_records(request,launch,argv,pid,ticks,parent,parent_ticks):
    expected=['.runtime/env/bin/python3.11','-B','-m',MODULE,*argv]
    need(request.get('command')==launch.get('actual_argv')==expected
         and launch.get('pid')==pid and launch.get('process_start_ticks')==ticks
         and request.get('supervisor_pid')==parent and request.get('supervisor_start_ticks')==parent_ticks,
         'Exact visible ROOT command and actual guardian/worker generations')
    need(request.get('host_cap_bytes')==HOST_CAP and request.get('output_cap_bytes')==RESERVATION
         and request.get('operating_stop_bytes')==OPERATING and request.get('max_seconds')==MAX_SECONDS
         and request.get('CUDA_VISIBLE_DEVICES')=='' and request.get('aggregate_directory')==str(AGGREGATE.relative_to(ROOT)),
         'Exact admitted CPU-only full-size output and no elapsed-time cutoff')
    return request
