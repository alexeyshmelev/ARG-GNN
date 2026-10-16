"""Root-only VAL001 operational bootstrap around the unchanged legacy driver."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess as _real_subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_GNN_VAL001_plan_metadata_20261016_v1"
LEGACY="research/reviews/C3_GNN_current980_executable_root_20261015_v1"
PHASE="plan_val_001"
WRAPPER=HERE+"/root_operational_bootstrap_v1.py"
STARTUP=HERE+"/startup/sitecustomize.py"
RUNTIME=HERE+"/operational_runtime_metadata.v1.json"
REQUEST=LEGACY+"/request_plan_val_001.root_active_v1.json"
INPUT=LEGACY+"/admitted_plan_val_001_v1/input.json"
BASE_PATH=os.pathsep.join((str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")))
CHILD_PATH=str(ROOT/HERE/"startup")+os.pathsep+BASE_PATH
ENV_PATH="ARGGNN_C3_VAL001_ROOT_REQUEST_PATH"
ENV_SHA="ARGGNN_C3_VAL001_ROOT_REQUEST_SHA256"
DRIVER_REF=dict(path=LEGACY+"/root_driver_v1.py",sha256="95f5c8fcbf096f9f648f6ff293152cc81b5adb36586b129c80e4e6dd1326015c")
ENTRY_REF=dict(path=LEGACY+"/worker_entry_v1.py",sha256="5129f3efbe4bfdbddd2361b9c60ccbd920f582e88cfdcaf48fcd47266ec6671a")
CONTRACT_REF=dict(path=LEGACY+"/root_contract_v1.py",sha256="fe94412381ba50c427d0357ea98edfd1e002b8ab27b281d6958c3999d6dcfe4d")
RENEWAL_REF=dict(path="research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py",sha256="491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468")
CERT_REF=dict(path="research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json",sha256="66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7")
SCIENCE_RUNTIME_REF=dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")

def _must(ok,message):
    if not ok:raise ValueError(message)

def _path(name):
    p=ROOT/name
    _must(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
        and p.suffix in (".py",".json"),"Canonical contained source/metadata only")
    return p

def _ref(name):
    p=_path(name)
    _must(p.stat().st_size<=1048576,"Bounded source/metadata")
    return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def _read(record):
    _must(isinstance(record,dict) and set(record)=={"path","sha256"} and record["path"].endswith(".json"),"Exact JSON ref")
    _must(_ref(record["path"])==record,"Bound metadata changed")
    return json.loads(_path(record["path"]).read_text())

def _load(record,name):
    _must(_ref(record["path"])==record and record["path"].endswith(".py"),"Exact source before private import")
    spec=importlib.util.spec_from_file_location(name,ROOT/record["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def checked_request(record):
    _must(record["path"]==REQUEST,"Only exact VAL001 legacy root request")
    req=_read(record)
    _must(req.get("schema")=="C3_GNN_current980_request_v1" and req.get("active") is True
        and req.get("root_authorizer")=="root_agent" and req.get("phase")==PHASE
        and req.get("physical_gpu") is None and req.get("GPU_UUID") is None
        and req.get("native_MLM_dependency") is False and req.get("TRAIN_dependency") is False
        and req.get("runtime_ref")==SCIENCE_RUNTIME_REF,"Exact active observation-only CPU request")
    _must(req["bindings"]["parent_id"]=="american_c3_val_001"
        and set(req["bindings"])=={"parent_id","plan_config","native_preparation_ref","preparation_wait_ref"},
        "No TRAIN/model/truth capability")
    wrapper_ref,startup_ref=_ref(WRAPPER),_ref(STARTUP)
    _must(req.get("runtime_bootstrap_ref")==startup_ref and req.get("operational_wrapper_ref")==wrapper_ref
        and req.get("renewal_source_ref")==RENEWAL_REF and req.get("renewal_certificate_ref")==CERT_REF,
        "Explicit same-request helper/renewal pins")
    runtime_ref=req["operational_runtime_ref"]
    _must(runtime_ref["path"]==RUNTIME,"Exact separate operational runtime metadata")
    runtime=_read(runtime_ref)
    expected=dict(schema="C3_GNN_VAL001_operational_bootstrap_metadata_v1",phase=PHASE,
        CPU_only=True,physical_gpu=None,GPU_UUID=None,TRAIN_dependency=False,native_MLM_dependency=False,
        runtime_bootstrap_ref=startup_ref,operational_wrapper_ref=wrapper_ref,legacy_driver_ref=DRIVER_REF,
        legacy_worker_ref=ENTRY_REF,legacy_contract_ref=CONTRACT_REF,science_runtime_ref=SCIENCE_RUNTIME_REF,
        renewal_source_ref=RENEWAL_REF,renewal_certificate_ref=CERT_REF,
        original_child_PYTHONPATH=BASE_PATH,actual_child_PYTHONPATH=CHILD_PATH,
        only_operational_override="prepend_dedicated_startup_directory_to_exact_legacy_CPU_child_PYTHONPATH",
        legacy_launch_wait_and_owned_drain_codeobjects_unchanged=True,
        global_subprocess_patch=False,installed_or_frozen_source_edits=False,source_only=True)
    _must(runtime==expected,"Exact narrow operational metadata")
    for ref in (DRIVER_REF,ENTRY_REF,CONTRACT_REF,RENEWAL_REF,CERT_REF,SCIENCE_RUNTIME_REF):
        _must(_ref(ref["path"])==ref,"Pinned legacy/renewal/runtime bytes")
    _must(os.environ.get(ENV_PATH)==record["path"] and os.environ.get(ENV_SHA)==record["sha256"],
        "Explicit task-scoped root request environment")
    return req,runtime

def checked_input(record,request_ref):
    _must(record["path"]==INPUT,"Exact legacy VAL001 input")
    spec=_read(record)
    admitted=_read(_ref(LEGACY+"/admitted_plan_val_001_v1/admitted.json"))
    _must(admitted["input_ref"]==record and admitted["driver_ref"]==DRIVER_REF
        and spec.get("schema")=="C3_GNN_current980_active_input_v1" and spec.get("status")=="active"
        and spec.get("root_authorizer")=="root_agent" and spec.get("execution_authorized") is True
        and spec.get("GPU_authorized") is False and spec.get("phase")==PHASE
        and spec.get("physical_gpu") is None and spec.get("execution_gpu_uuid") is None
        and spec.get("root_stage_request_ref")==request_ref and spec.get("root_driver_ref")==DRIVER_REF
        and spec.get("main_source_ref")==ENTRY_REF,"Exact admitted CPU input/request/driver")
    return spec

def install_renewal():
    renewal=_load(RENEWAL_REF,"_C3_VAL001_exact_renewal")
    _must(renewal.install_renewal_v1(CERT_REF)==CERT_REF,"Exact existing renewal installed")

class _PrivateSubprocessProxy:
    """Only this private driver's Popen gets one additive child PYTHONPATH."""
    def __init__(self,input_ref,request_ref):
        self.input_ref=input_ref;self.request_ref=request_ref;self.used=False;self.event=None
    def __getattr__(self,name):return getattr(_real_subprocess,name)
    def Popen(self,command,**kwargs):
        expected=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/ENTRY_REF["path"]),
            "--execute","--input",self.input_ref["path"],"--expected-input-sha256",self.input_ref["sha256"]]
        _must(not self.used and command==expected,"Only exact original CPU worker argv/input once")
        _must(set(kwargs)=={"cwd","env","stdout","stderr","bufsize","start_new_session"}
            and kwargs["cwd"]==ROOT and kwargs["stdout"]==_real_subprocess.PIPE
            and kwargs["stderr"]==_real_subprocess.STDOUT and kwargs["bufsize"]==0
            and kwargs["start_new_session"] is True,"Exact legacy Popen options")
        env=kwargs["env"]
        _must(isinstance(env,dict) and env.get("CUDA_VISIBLE_DEVICES")==""
            and env.get("PYTHONPATH")==BASE_PATH and env.get("PYTHONDONTWRITEBYTECODE")=="1"
            and env.get("PYTHONNOUSERSITE")=="1" and env.get(ENV_PATH)==self.request_ref["path"]
            and env.get(ENV_SHA)==self.request_ref["sha256"],"Exact CPU/startup/request environment")
        actual_env=dict(env);actual_env["PYTHONPATH"]=CHILD_PATH
        actual_kwargs=dict(kwargs);actual_kwargs["env"]=actual_env
        self.used=True
        self.event=dict(command=command,input_ref=self.input_ref,root_stage_request_ref=self.request_ref,
            original_child_PYTHONPATH=BASE_PATH,actual_child_PYTHONPATH=CHILD_PATH,
            only_environment_change="PYTHONPATH",real_subprocess_Popen_object_returned=True)
        proc=_real_subprocess.Popen(command,**actual_kwargs)
        self.event["pid"]=proc.pid
        return proc

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--admit",action="store_true");mode.add_argument("--launch",action="store_true")
    mode.add_argument("--close",action="store_true")
    parser.add_argument("--request");parser.add_argument("--expected-request-sha256")
    parser.add_argument("--root-approval");parser.add_argument("--expected-root-approval-sha256")
    args=parser.parse_args(argv)
    if not any((args.admit,args.launch,args.close)):
        print(json.dumps(dict(status="source_only_VAL001_operational_bootstrap",actual_admission=False,actual_execution=False)));return 0
    _must(args.request and args.expected_request_sha256,"Explicit root request path/SHA")
    request_ref=dict(path=args.request,sha256=args.expected_request_sha256)
    req,runtime=checked_request(request_ref)
    _must(args.close or (args.root_approval is None and args.expected_root_approval_sha256 is None),"Approval args only for close")
    install_renewal()
    driver=_load(DRIVER_REF,"_C3_VAL001_unchanged_legacy_driver")
    _must(Path(driver.c.__file__)==ROOT/CONTRACT_REF["path"] and driver.SOURCE==DRIVER_REF["path"],
        "Exact original imported contract/driver identities")
    codeobjects={name:getattr(driver,name).__code__ for name in ("admit","launch","close","main")}
    proxy=None
    if args.launch:
        input_ref=_ref(INPUT);checked_input(input_ref,request_ref)
        proxy=_PrivateSubprocessProxy(input_ref,request_ref)
        driver.subprocess=proxy
    action="admit" if args.admit else ("launch" if args.launch else "close")
    original_args=["--"+action,PHASE]
    if args.close:
        _must(args.root_approval and args.expected_root_approval_sha256,"Genuine root terminal/revocation approval required")
        original_args+=["--root-approval",args.root_approval,"--expected-root-approval-sha256",args.expected_root_approval_sha256]
    rc=driver.main(original_args)
    _must(all(getattr(driver,name).__code__ is code for name,code in codeobjects.items()),
        "Original admit/launch/wait/drain/close/main codeobjects preserved")
    if proxy is not None:
        _must(proxy.used and proxy.event is not None,"Actual exact Popen invocation required")
        receipt=dict(schema="C3_GNN_VAL001_root_actual_operational_launch_receipt_v1",
            root_authorizer="root_agent",phase=PHASE,legacy_driver_ref=DRIVER_REF,
            operational_wrapper_ref=_ref(WRAPPER),runtime_bootstrap_ref=_ref(STARTUP),
            operational_runtime_ref=req["operational_runtime_ref"],renewal_source_ref=RENEWAL_REF,
            renewal_certificate_ref=CERT_REF,root_stage_request_ref=request_ref,
            original_driver_returncode=rc,actual_child_invocation=proxy.event,
            original_launch_ref=_ref(LEGACY+"/admitted_plan_val_001_v1/launch.json"),
            original_wait_ref=_ref(LEGACY+"/admitted_plan_val_001_v1/actual_wait.json"),
            legacy_launch_wait_and_owned_drain_codeobjects_unchanged=True,global_subprocess_patch=False)
        receipt_ref=driver.c.save(ROOT/HERE,"root_launch_bootstrap_receipt.actual_v1.json",receipt)
        print(json.dumps(dict(operational_launch_receipt_ref=receipt_ref)),flush=True)
    return rc

if __name__=="__main__":raise SystemExit(main())
