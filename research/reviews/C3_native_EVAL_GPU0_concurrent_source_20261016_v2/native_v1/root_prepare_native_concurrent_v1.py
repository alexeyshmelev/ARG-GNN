"""ROOT-only operational bridge into unchanged sealed nativev3 admission.
Default/import performs no project import, probes, guard state or writes.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
from datetime import datetime,timezone
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/native_v1/root_prepare_native_concurrent_v1.py'
HERE='research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2'
POOL=HERE+'/controller_v1/gpu0_declared_pool_v1.py'
LAUNCHER=HERE+'/controller_v1/launch_declared_gpu0_v1.py'
V3='research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3'
PREPARE_REF={'path':V3+'/root_prepare_native_v3.py','sha256':'9eeef07a7da5cfab8d659e6a88181468c6df5726c5ecda551521b183c44a304a'}
def ref(path):
    p=ROOT/path
    if p.resolve()!=p or not p.is_file() or p.is_symlink():raise ValueError('Canonical existing source/metadata')
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    if ref(record['path'])!=record:raise ValueError('Exact source/metadata SHA')
    return json.loads((ROOT/record['path']).read_text())
def load(record,name):
    if ref(record['path'])!=record:raise ValueError('Exact source before private ROOT import')
    sp=importlib.util.spec_from_file_location(name,ROOT/record['path'])
    module=importlib.util.module_from_spec(sp)
    sys.modules[name]=module
    sp.loader.exec_module(module)
    return module

def admit_native_concurrent_v1(actual_policy_ref,actual_execution_approval_ref,
                               actual_capacity_guard_ref,actual_concurrent_source_handoff_ref,
                               actual_prior_controller_completion_ref):
    if Path.cwd()!=ROOT:raise ValueError('ROOT cwd')
    if actual_concurrent_source_handoff_ref['path']!=HERE+'/source_handoff.v2.json':
        raise ValueError('Exact fresh reviewed concurrent source handoff')
    handoff=read(actual_concurrent_source_handoff_ref)
    approval=read(actual_execution_approval_ref)
    if (approval.get('concurrent_execution_handoff_ref')!=actual_concurrent_source_handoff_ref
        or approval.get('actual_concurrent_pool_source_ref')!=ref(POOL)
        or approval.get('actual_concurrent_launcher_source_ref')!=ref(LAUNCHER)):
        raise ValueError('ROOT explicitly reviewed the actual operational-only bridge/pool')
    if handoff.get('controller_pool_ref')!=ref(POOL) or handoff.get('controller_bootstrap_ref')!=ref(LAUNCHER) or handoff.get('native_bridge_ref')!=ref(SOURCE):
        raise ValueError('Exact final source seal for actual bridge/controller/bootstrap')
    helper=load(PREPARE_REF,'_original_v3_native_ROOT_utilities')
    # Same original preparation steps and exclusive v3 metadata paths.
    helper.load(helper.RENEWAL_SOURCE_REF,'_native_concurrent_original_renewal').install_renewal_v1(helper.RENEWAL_CERT_REF)
    policy=read(actual_policy_ref)
    if policy.get('active') is not True or policy.get('source_ref')!=actual_capacity_guard_ref:
        raise ValueError('Actual reviewed ROOT active H020v3 source/policy')
    os.environ['ARGGNN_NATIVE_CAPACITY_GUARD_REF']=json.dumps(actual_capacity_guard_ref)
    os.environ['ARGGNN_COMPLETED_CREDITS_POLICY_REF']=json.dumps(actual_policy_ref)
    closed=load(handoff['controller_contract_ref'],'_native_concurrent_genuine_closed_handover')
    closed.validate_previous_controller_v1(actual_prior_controller_completion_ref)
    authority_rows=closed.family_authority_refs_v2()
    if authority_rows[1]!=actual_concurrent_source_handoff_ref or authority_rows[-1]!=actual_execution_approval_ref:
        raise ValueError('Bridge uses the same actual approval/seal authenticated by the full immutable core')
    verification=dict(schema='C3_native_concurrent_actual_full_core_verification_v2',
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        native_bridge_ref=ref(SOURCE),concurrent_execution_handoff_ref=authority_rows[1],
        core_source_manifest_ref=authority_rows[2],actual_execution_approval_ref=authority_rows[3],
        full_core_ref_count=read(authority_rows[2])['source_ref_count'],
        every_full_core_ref_authenticated=True,genuine_prior_controller_completion_ref=actual_prior_controller_completion_ref,
        scientific_or_GPU_work_executed=False)
    verification_path=ROOT/HERE/'native_v1'/'actual_core_verification.root_active_v2.json'
    with verification_path.open('x') as stream:
        json.dump(verification,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    c=helper.load(ref(V3+'/mlm_v1/root_native_contract_v1.py'),'_native_concurrent_root_contract')
    if actual_execution_approval_ref!=c.ref(c.GPU0_EXECUTION_APPROVAL_PATH):
        raise ValueError('Original fixed fresh nativev3 ROOT approval path')
    c.require_gpu0_execution_route_v1()
    old=read(ref(helper.OLD+'/mlm_v1/admitted_mlm_v1/input.json'))
    caps=read(old['caps_ref'])
    caps['driver_source_ref']=c.ref(c.DRIVER)
    caps_ref=helper.save('mlm_v1/caps_mlm.root_active_v3.json',caps)
    readiness=read(old['refs']['native_gpu0_readiness'])
    readiness['GPU0_execution_approval_ref']=actual_execution_approval_ref
    readiness_ref=helper.save('mlm_v1/GPU0_readiness_approval.root_active_v3.json',readiness)
    request=read(old['root_stage_request_ref'])
    request['caps_ref']=caps_ref
    request['refs']['native_gpu0_readiness']=readiness_ref
    request_ref=helper.save('mlm_v1/request_mlm.root_active_v3.json',request)
    driver=helper.load(c.ref(c.DRIVER),'_native_concurrent_unchanged_admission_driver')
    # These two private operational globals record the actual controller. The
    # narrow private unique wrapper below appends only authority metadata. All
    # original numerical code, corpus, caps, namespaces, charge and expiry stay exact.
    driver.c.SHARED_POOL_SOURCE=POOL
    driver.c.SHARED_POOL_LAUNCHER=LAUNCHER
    # Only the private metadata deduplicator is extended. The original admit
    # function and every original reserve/guard implementation remain exact.
    original_unique=driver.unique
    def unique_with_concurrent_authority(rows):
        consumers=original_unique([*rows,*authority_rows])
        if len(consumers)>64 or not all(r in consumers for r in authority_rows):
            raise ValueError('All four actual concurrent authorities must be direct before original reserve')
        return consumers
    driver.unique=unique_with_concurrent_authority
    try:
        driver.admit(request_ref)
    finally:
        driver.unique=original_unique
    admitted_ref=driver.c.ref(driver.c.phase_paths('mlm')[0]/'admitted.json')
    admitted=read(admitted_ref)
    spec=read(admitted['input_ref'])
    if spec['operational_pool_source_ref']!=ref(POOL) or spec['operational_launcher_ref']!=ref(LAUNCHER):
        raise ValueError('Actual native input must truthfully name the executed concurrent pool')
    closed.validate_native_direct_authority_v2(spec)
    return admitted_ref

if __name__=='__main__':
    print(json.dumps(dict(status='SOURCE_ONLY_ROOT_API_REQUIRES_ACTUAL_REVIEWED_REFS',admission=False,execution=False)))
