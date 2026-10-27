"""Exact one-parent ROOT orchestration and readable native-source gates."""
import hashlib
import json
import os
from pathlib import Path
import stat
from research.efficiency_20261027.integration_candidate.european_short_development_val import short_plan_v1 as plan

ROOT=plan.ROOT
HERE=Path(__file__).parent
RUNS=ROOT/'research/efficiency_20261027/root/runs'
# One immutable aggregate location prevents accidental fresh16GiB reservations
# for a retry. This runner cannot retry,extend the registration,or remove output.
AGGREGATE=RUNS/'european_short_development_val_10mb_v2'
RETAINED_AGGREGATE=RUNS/'european_short_development_val_10mb_v1'
RETAINED_REF_PATH=HERE/'RETAINED_FAILED_ATTEMPT_v1.json'
ROOT_STATUS_MAX_AGE_SECONDS=120
GROWTH_INTERVAL_SECONDS=30
MODULE='research.efficiency_20261027.execution.european_short_val_runner.worker_v3'
GIB=2**30
HOST_CAP=64*GIB
RESERVATION=16*GIB
OPERATING=12*GIB
MAX_SECONDS=21600
METADATA_CAP=256*2**20
HEADROOM=GIB
SOURCE_REF=dict(path='research/efficiency_20261027/integration_candidate/european_short_development_val/source_manifest_v3.json',sha256='08e1875adac650366ba64cb3f5cc78fab849d15a875007140907307a0469b024')
SOURCE_REVIEW_REF=dict(path='research/efficiency_20261027/fidelity/EUROPEAN_SHORT_DEVELOPMENT_VAL_V3_SOURCE_REVIEW.json',sha256='ed6c539c3853b2f89e12080784ee1d2d1f9cf5f05eb64c6e89f9812456b1a91f')
NATIVE_CONFIG_REF=dict(path='configs/benchmarks/ap200mb_native_runtime_v2.json',sha256='a86720db22c176bf194ed1a3ec704e82126f4be14fb514fc4ac43c3c5176482e')


def need(ok,message):
    if not ok:raise ValueError(message)


def contained(path):
    path=ROOT/Path(path)
    need(path.is_relative_to(ROOT) and '..' not in path.parts and path.resolve()==path,'Canonical contained ordinary path')
    return path


def ref(path):
    path=contained(path);info=path.stat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'Ordinary reference file')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest())


def read(reference,cap=8*2**20):
    path=contained(reference['path']);need(path.stat().st_size<=cap and ref(path)==reference,'Exact bounded metadata reference')
    value=json.loads(path.read_text());need(ref(path)==reference,'Metadata remained immutable during read')
    return value


def verify_sources(digest,review_ref):
    manifest_ref=ref(HERE/'source_pins_v3.json');need(manifest_ref['sha256']==digest,'Exact orchestration manifest')
    manifest=read(manifest_ref)
    need(manifest.get('schema')=='European_short_VAL_ROOT_runner_sources_v3','Exact ROOT runner source schema')
    for item in manifest['source_refs']:need(ref(item['path'])==item,'Pinned source changed: '+item['path'])
    need(SOURCE_REF in manifest['source_refs'] and SOURCE_REVIEW_REF in manifest['source_refs'],'Original immutable stage review included')
    read(SOURCE_REF);old_review=read(SOURCE_REVIEW_REF)
    need(old_review.get('passed') is True,'Original v3 stage review still passed')
    review=read(review_ref)
    need(review.get('schema')=='European_short_VAL_ROOT_runner_review_v3' and review.get('passed') is True
         and review.get('source_manifest_ref')==manifest_ref,'Exact independently reviewed ROOT orchestration')
    registered=plan.validate_plan()
    need(registered['registration']['initial_parent_count']==1 and registered['registration']['additional_parent_admitted'] is False,
         'Only the one original prospectively registered10Mb parent')
    return manifest_ref,registered


def native_bindings():
    config=read(NATIVE_CONFIG_REF);plan.validate_native_config(config)
    installation=read(config['relate']['installation']);release=contained(installation['release_root'])
    chain_ref=ref(HERE/'NATIVE_CHAIN_REVIEW_v1.json');chain=read(chain_ref)
    need(chain.get('schema')=='European_short_VAL_plaintext_native_chain_review_v1' and chain.get('passed') is True
         and chain.get('native_config_ref')==NATIVE_CONFIG_REF and chain.get('encoded_executable_payloads') is False,
         'Exact reviewed plaintext native chain')
    expected={str(release/key):digest for key,digest in config['relate']['tool_sha256'].items()}
    need(chain.get('reviewed_native_tools')==expected,'Every declared native binary and script reviewed')
    for path,digest in expected.items():need(ref(path)['sha256']==digest,'Exact installed native tool hash')
    return config,expected,chain_ref


def validate_release(value,manifest_ref,review_ref,peers):
    need(value.get('schema')=='European_short_VAL_ROOT_execution_release_v1' and value.get('passed') is True
         and value.get('execution_authorized') is True and value.get('source_manifest_ref')==manifest_ref
         and value.get('independent_source_review_ref')==review_ref and value.get('plan_ref')==plan.PLAN_REF
         and value.get('aggregate_directory')==str(AGGREGATE.relative_to(ROOT))
         and value.get('host_cap_bytes')==HOST_CAP and value.get('reservation_bytes')==RESERVATION
         and value.get('operating_stop_bytes')==OPERATING and value.get('max_seconds')==MAX_SECONDS
         and value.get('CUDA_VISIBLE_DEVICES')=='' and value.get('peer_directories')==peers
         and value.get('global_project_ceiling_bytes')==1_000_000_000_000
         and value.get('ROOT_live_global_storage_and_peer_admission_required') is True
         and value.get('model_fit_prediction_or_scoring_authorized') is False
         and value.get('ROOT_status_max_age_seconds')==ROOT_STATUS_MAX_AGE_SECONDS
         and value.get('growth_refresh_interval_seconds')==GROWTH_INTERVAL_SECONDS
         and value.get('retained_attempt_ref')==ref(RETAINED_REF_PATH),
         'Exact ROOT one-parent authorization,aggregate allowance and current peer binding')
    need(type(peers) is list and len(peers)==len(set(peers)),'Explicit unique current peers;empty is allowed only by ROOT release')
    for peer in peers:need(contained(peer).parent==RUNS and contained(peer) not in (AGGREGATE,RETAINED_AGGREGATE),'Each exact other ROOT run peer; retained attempt shares this reservation')
    return value


def validate_worker_records(request,launch,argv,pid,ticks,parent,parent_ticks):
    expected=['.runtime/env/bin/python3.11','-B','-m',MODULE,*argv]
    need(request.get('command')==launch.get('actual_argv')==expected
         and launch.get('pid')==pid and launch.get('process_start_ticks')==ticks
         and request.get('supervisor_pid')==parent and request.get('supervisor_start_ticks')==parent_ticks,
         'Exact readable ROOT request/launch and actual live generations')
    need(request.get('host_cap_bytes')==HOST_CAP and request.get('output_cap_bytes')==RESERVATION
         and request.get('operating_stop_bytes')==OPERATING and request.get('max_seconds')==MAX_SECONDS
         and request.get('CUDA_VISIBLE_DEVICES')=='' and request.get('aggregate_directory')==str(AGGREGATE.relative_to(ROOT)),
         'Exact ROOT admitted CPU-only aggregate and finite budget')
    return request


def retained_attempt():
    """Authenticate the terminated predecessor and charge its actual retained bytes."""
    from research.efficiency_20261027.integration_candidate.european_short_development_val.aggregate_guard_v3 import measure
    reference=ref(RETAINED_REF_PATH);value=read(reference)
    need(value.get('aggregate_directory')==str(RETAINED_AGGREGATE.relative_to(ROOT)),
         'Only the fixed failed predecessor shares the original reservation')
    terminal=read(value['actual_wait_ref'])
    need(terminal.get('actual_wait') is True and terminal.get('actual_returncode')==1
         and terminal.get('owned_family_absent') is True,'Failed predecessor actually waited and owned family absent')
    for item in value['file_refs']:need(ref(item['path'])==item,'Retained failed evidence changed')
    charge=value['charged_bytes']
    need(type(charge) is int and 0<=charge<OPERATING,'Bounded retained predecessor charge')
    need(measure(RETAINED_AGGREGATE)['charged_bytes']==charge,'Retained predecessor real byte charge changed')
    return reference,charge
