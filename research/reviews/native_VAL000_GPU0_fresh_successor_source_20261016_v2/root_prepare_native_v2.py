"""ROOT-only preparation of fresh native metadata/admission, default inert.
No project modules, guards, state, probes or jobs are loaded on import.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v2'
OLD='research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1'
RENEWAL_SOURCE_REF={'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py','sha256':'491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
RENEWAL_CERT_REF={'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json','sha256':'66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
def ref(name):
    p=ROOT/name
    if p.resolve()!=p or not p.is_file() or p.is_symlink():raise ValueError('Canonical actual source/metadata')
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    if ref(record['path'])!=record:raise ValueError('Exact source/metadata SHA')
    return json.loads((ROOT/record['path']).read_bytes())
def load(record,name):
    if ref(record['path'])!=record:raise ValueError('Exact source before private load')
    sp=importlib.util.spec_from_file_location(name,ROOT/record['path']);m=importlib.util.module_from_spec(sp)
    sys.modules[name]=m;sp.loader.exec_module(m);return m
def save(name,value):
    p=ROOT/HERE/name
    if p.resolve()!=p:raise ValueError('Canonical fresh root metadata')
    with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return ref(str(p.relative_to(ROOT)))
def admit_native_v2(actual_policy_ref,actual_execution_approval_ref,actual_capacity_guard_ref):
    if Path.cwd()!=ROOT:raise ValueError('Exact ROOT cwd')
    # The original marker-reader authority is expired. Install the approved
    # interpreter-local renewal BEFORE any marker consumer import/configure.
    load(RENEWAL_SOURCE_REF,'_fresh_native_root_original_renewal').install_renewal_v1(RENEWAL_CERT_REF)
    policy=read(actual_policy_ref)
    if policy.get('active') is not True or policy.get('source_ref')!=actual_capacity_guard_ref:raise ValueError('Actual independently reviewed ROOT active capacity source/policy')
    os.environ['ARGGNN_NATIVE_CAPACITY_GUARD_REF']=json.dumps(actual_capacity_guard_ref,separators=(',',':'))
    os.environ['ARGGNN_COMPLETED_CREDITS_POLICY_REF']=json.dumps(actual_policy_ref,separators=(',',':'))
    c=load(ref(HERE+'/mlm_v1/root_native_contract_v1.py'),'_fresh_native_root_contract')
    if actual_execution_approval_ref!=c.ref(c.GPU0_EXECUTION_APPROVAL_PATH):raise ValueError('Actual reviewed ROOT execution approval at fixed fresh path')
    c.require_gpu0_execution_route_v1()
    old_input=read(ref(OLD+'/mlm_v1/admitted_mlm_v1/input.json'))
    caps=read(old_input['caps_ref']);caps['driver_source_ref']=c.ref(c.DRIVER)
    caps_ref=save('mlm_v1/caps_mlm.root_active_v2.json',caps)
    readiness=read(old_input['refs']['native_gpu0_readiness'])
    readiness['GPU0_execution_approval_ref']=actual_execution_approval_ref
    readiness_ref=save('mlm_v1/GPU0_readiness_approval.root_active_v2.json',readiness)
    request=read(old_input['root_stage_request_ref']);request['caps_ref']=caps_ref
    request['refs']['native_gpu0_readiness']=readiness_ref
    request_ref=save('mlm_v1/request_mlm.root_active_v2.json',request)
    driver=load(c.ref(c.DRIVER),'_fresh_native_root_actual_driver')
    driver.admit(request_ref)  # ROOT actual reserve only after reviewed active capacity policy.
    return driver.c.ref(driver.c.phase_paths('mlm')[0]/'admitted.json')
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admit',action='store_true')
    p.add_argument('--policy-ref');p.add_argument('--execution-approval-ref');p.add_argument('--capacity-guard-ref');a=p.parse_args(argv)
    if not a.admit:
        print(json.dumps(dict(status='source_only_unconfigured',execution=False,admission=False)));return 0
    if not a.policy_ref or not a.execution_approval_ref or not a.capacity_guard_ref:raise ValueError('Exact active ROOT policy and reviewed execution approval refs')
    print(json.dumps(dict(actual_admitted_ref=admit_native_v2(json.loads(a.policy_ref),json.loads(a.execution_approval_ref),json.loads(a.capacity_guard_ref)))));return 0
if __name__=='__main__':raise SystemExit(main())
