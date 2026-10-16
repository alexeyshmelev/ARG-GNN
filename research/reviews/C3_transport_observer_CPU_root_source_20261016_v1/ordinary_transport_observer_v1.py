"""Deferred root CPU transport observation. No actions occur on import."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,resource,sys,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_transport_observer_CPU_root_source_20261016_v1"
SOURCE=HERE+"/ordinary_transport_observer_v1.py"
P_REF={"path":"research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/prepare_resume_v1.py","sha256":"8db30e2eca56925937f4b22f6657603b514211ffe1b6a209c999ddb2a939b438"}
E_REF={"path":"research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/finite_envelopes_v1.py","sha256":"06a1d9bbb0045d32f6975eb1e00864748078f8cc8d6f158ad22aefede43244e9"}
FJ_REF={"path":"research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/finite_jobs_v1.py","sha256":"8dcd1e5fd3b79e59baea6378d5eeb7ad485e59478cd9d8943aca4669bf9568b1"}
RENEWAL_REF={"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py","sha256":"491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468"}
RENEWAL_CERT_REF={"path":"research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json","sha256":"66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7"}
JOB="C3-transport-observer-CPU-v1"
OUTPUT="experiments/results/C3_transport_observer_CPU_root_v1"
CAP=8*2**20
TTL=1800
PROBE_WALL=120
RSS=2*2**30

def need(ok,message):
    if not ok:raise ValueError(message)

def ref(name):
    need(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
        "Contained relative source/metadata")
    p=ROOT/name
    need(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=2**20,
        "Canonical bounded source/metadata")
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def load(record,label):
    need(ref(record["path"])==record,"Exact reviewed source before import")
    spec=importlib.util.spec_from_file_location(label,ROOT/record["path"])
    module=importlib.util.module_from_spec(spec)
    sys.modules[label]=module
    spec.loader.exec_module(module)
    return module

def preflight_v1(request_ref,p,e,fj,c,guard_bindings):
    """Actual saved lineage only. No probe or accounting operation."""
    request,rows=fj.validate_transport_unknown_request_v1(request_ref)
    need(request["original_external_tool_session_id"]==23022
        and request["original_external_tool_exit_status"]=="UNKNOWN"
        and request["original_external_tool_exit_code"] is None
        and request["original_controller_outer_wait0_claimed"] is False,
        "Original controller23022 remains UNKNOWN")
    refs=[ref(SOURCE),P_REF,E_REF,FJ_REF,p.GPU0_ADMISSION_REF,c.ref(c.SOURCE),c.BASE_HELPER,
        *e.base_common_refs_v1(),*fj.resume_transition_refs_v1(request_ref),
        fj.TRANSPORT_HELPER_REF,fj.HANDLE_LOSS_REF,fj.OLD_C3_CUDA_REF,fj.OLD_H050_CUDA_REF,
        fj.FAILED_CLI_VALIDATOR_REF,request_ref,
        *[request[k] for k in ("prior_capability_ref","prior_ready_ref","final_registry_ref",
            "explicit_old_capability_revocation_ref","H050_failed_CLI_certificate_ref")],
        *guard_bindings]
    # The exact existing certificate validator authenticates its own complete
    # frozen H050 lineage. The exact original migration reader authenticates
    # every manifest/intent/bounds/qualification/terminal/source reference.
    refs=p.unique(refs)
    need(len(refs)<=64,"Complete actual observer source/lineage/current-guard union exceeds64")
    return request,refs

def execute_v1(request_ref):
    control_started=time.monotonic()
    need(Path.cwd().resolve()==ROOT and Path(__file__).resolve()==ROOT/SOURCE,
        "Exact root observer source and repository cwd")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit CPU-only observer")
    need(not any(n in sys.modules for n in ("torch","torch_geometric","numpy")),
        "No numerical package in metadata observer")
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    # Renewal must precede every guard-bearing helper import.
    renewal=load(RENEWAL_REF,"_transport_observer_exact_renewal")
    renewal.install_renewal_v1(RENEWAL_CERT_REF)
    p=load(P_REF,"_transport_observer_reviewed_preparation")
    e=load(E_REF,"_transport_observer_reviewed_envelopes")
    fj=load(FJ_REF,"_transport_observer_reviewed_finite_contract")
    c=p.original_contract_v1()
    guard=c.original_guard()
    admission_started=time.monotonic()
    packet=ROOT/HERE/"actual_admission_v1"
    output=ROOT/OUTPUT
    need(not packet.exists() and not output.exists(),"Fresh one-use observer admission/output")
    with guard.budget.v4.mutex(ROOT):
        state=guard._state_locked(c.CORE,c.BASE,fresh=True)
        view,bindings=state[2],state[3]
    request,consumers=preflight_v1(request_ref,p,e,fj,c,list(bindings.values()))
    metadata_auth_finished=time.monotonic()
    created=datetime.now(timezone.utc)
    expires=created+timedelta(seconds=TTL)
    expiry=expires.isoformat()
    reserve=dict(key="H041/"+JOB,output=str(output),auxiliary_scopes=[],
        reserved_bytes=CAP,consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",reserve,
        expected_before=guard._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry)
    packet.mkdir()
    metadata_bytes=0
    def admission_save(name,value):
        nonlocal metadata_bytes
        data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        need(len(data)<=65536 and metadata_bytes+len(data)<=131072,
            "Original64KiB file and128KiB total admission metadata bounds")
        result=c.save(packet,name,value)
        metadata_bytes+=len(data)
        return result
    authority_ref=admission_save("reserve_authority.json",authority)
    receipt=guard.apply_reserve_v1(authority_ref,
        expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
    receipt_ref=admission_save("reserve_receipt.json",receipt)
    components=dict(implementation=c.ref("scripts/project_budget_v9.py"),
        launcher=c.ref("scripts/launch_reserved_gpu_v6.py"),consumer_adapter=c.GUARD_REF)
    storage=e.storage_from_actual_reserve_v1(reserve,receipt_ref,job_id=JOB,
        expires_at_utc=expiry,actual_budget_components=components,actual_policy_ref=c.POLICY_REF)
    storage_ref=admission_save("storage.json",storage)
    need(storage["reserved_new_bytes"]==CAP and storage["reservation"]["auxiliary_scopes"]==[]
        and storage["reservation"]["consumer_bindings"]==consumers
        and all(r in consumers for r in (ref(SOURCE),P_REF,E_REF,FJ_REF)),
        "Exact separately charged source-bound observer8MiB/noaux scope")
    admission_elapsed=time.monotonic()-admission_started
    initial_guard_started=time.monotonic()
    guard.budget_check_v9(storage,boundary=True,required_write_bytes=8192)
    need(not output.exists(),"Fresh observer output after original guarded admission")
    output.mkdir()
    guard.budget_check_v9(storage,boundary=True)
    initial_guard_elapsed=time.monotonic()-initial_guard_started
    # The120s probe clock starts AFTER expensive original admission/initial
    # guards. Those actual control times are separately reported below.
    probe_started=time.monotonic()
    def resource_check(event):
        need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only metadata probe")
        need(time.monotonic()-probe_started<PROBE_WALL and time.time()<expires.timestamp(),
            "Actual120s probe and1800s observer authority")
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RSS,
            "Actual peak self RSS at most2GiB")
    resource_check("before_actual_observer")
    observation=fj.observe_transport_unknown_closed_v1(request_ref,resource_check=resource_check)
    resource_check("after_actual_observer")
    probe_elapsed=time.monotonic()-probe_started
    after_probe_guard_started=time.monotonic()
    guard.budget_check_v9(storage,boundary=True)
    after_probe_guard_elapsed=time.monotonic()-after_probe_guard_started
    data=(json.dumps(observation,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(data)<=65536,"Original exclusive observation JSON limit64KiB")
    before_write_guard_started=time.monotonic()
    guard.budget_check_v9(storage,boundary=True,required_write_bytes=len(data)+8192)
    before_write_guard_elapsed=time.monotonic()-before_write_guard_started
    write_started=time.monotonic()
    observation_ref=c.save(output,"observation.json",observation)
    write_elapsed=time.monotonic()-write_started
    after_write_guard_started=time.monotonic()
    guard.budget_check_v9(storage,boundary=True)
    after_write_guard_elapsed=time.monotonic()-after_write_guard_started
    need(not any(n in sys.modules for n in ("torch","torch_geometric","numpy")),
        "Observer performed no numerical imports")
    print(json.dumps(dict(schema="C3_transport_observer_actual_control_result_v1",
        request_ref=request_ref,observation_ref=observation_ref,storage_ref=storage_ref,
        observer_source_ref=ref(SOURCE),observer_pid=os.getpid(),
        charged_scope=OUTPUT,reserved_bytes=CAP,auxiliary_scopes=[],
        complete_direct_bindings=len(consumers),admission_metadata_bytes=metadata_bytes,
        authority_expires_at_utc=expiry,
        control_before_request_auth_finished_seconds=metadata_auth_finished-control_started,
        admission_seconds=admission_elapsed,initial_guard_seconds=initial_guard_elapsed,
        probe_seconds=probe_elapsed,after_probe_guard_seconds=after_probe_guard_elapsed,
        before_write_guard_seconds=before_write_guard_elapsed,write_seconds=write_elapsed,
        after_write_guard_seconds=after_write_guard_elapsed,
        total_control_seconds=time.monotonic()-control_started,
        original_external_tool_session_id=23022,original_external_tool_exit_status="UNKNOWN",
        original_external_tool_exit_code=None,original_controller_outer_wait0_claimed=False,
        actual_external_tool_terminal_not_inferred=True,no_model_or_truth_or_numerics=True)),
        flush=True)
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--request")
    parser.add_argument("--expected-request-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive",source=SOURCE)))
        return 0
    need(args.request and args.expected_request_sha256,
        "Exact actual saved future transport request path/SHA required")
    return execute_v1(dict(path=args.request,sha256=args.expected_request_sha256))

if __name__=="__main__":
    try:raise SystemExit(main())
    except SystemExit:raise
    except BaseException as error:
        print(json.dumps(dict(error=type(error).__name__,message=str(error)[:2048],
            actual_probe_success_not_asserted=True)),flush=True)
        raise SystemExit(1)
