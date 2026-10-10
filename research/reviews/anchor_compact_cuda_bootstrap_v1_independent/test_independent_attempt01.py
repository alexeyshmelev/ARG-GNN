"""Independent source-contract attacks. Live state and scientific arrays are never read."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pytest
from arggnn import anchor_compact_cuda_bootstrap_v1 as boot

PROJECT = Path("/workspace-SR003.nfs2/ARG-GNN")
OWN = PROJECT / "research/reviews/anchor_compact_cuda_bootstrap_v1_independent"
spec = importlib.util.spec_from_file_location("independent_boot_fixture", OWN / "test_focused_replay_v1.py")
scaffold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaffold)
admission_fixture = scaffold.admission_fixture

def fail_before_context(f):
    with mock.patch.object(boot, "load_cuda_modules") as load, mock.patch.object(boot, "prepare_scope") as prepare:
        with pytest.raises((ValueError, RuntimeError, KeyError, TypeError)):
            f.admit()
    load.assert_not_called()
    prepare.assert_not_called()
    assert not f.output.exists()

@pytest.mark.parametrize("key,value", [
    ("kind","other"),("project_root","/other"),("bootstrap_config",{}),
    ("root_delegate_live_evidence_issuance",False),("fit_allowed",True),("truth_allowed",True),
    ("pool","../p"),("pool","p"*33),("logical_job","../j"),("output_scope","other"),
    ("issued_unix_seconds",float("nan")),("expires_unix_seconds",float("nan")),
    ("expires_unix_seconds",True),("issued_unix_seconds",-1),
])
def test_independent_root_guards(admission_fixture,key,value):
    f=admission_fixture; f.permit[key]=value; fail_before_context(f)

@pytest.mark.parametrize("key,value", [
    ("kind","other"),("status","active"),("actual_execution_admitted",True),
    ("root_delegated_live_evidence_issuance_requires_fresh_permit",1)
])
def test_independent_protocol_content(admission_fixture,key,value):
    f=admission_fixture; f.config[key]=value; fail_before_context(f)

@pytest.mark.parametrize("record,key,value", [
    ("bootstrap","bootstrap_config_sha256","0"*64),
    ("bootstrap","actual_execution_admitted",0),
    ("extension","verdict","BLOCKED"),
    ("extension","launcher_sha256","0"*64),
    ("extension","bootstrap_sha256","0"*64),
    ("extension","actual_execution_admitted",0),
    ("experiment","kind","other"),("experiment","status","inactive"),
    ("experiment","authorizer","author"),("experiment","project_root","/other"),
    ("experiment","fit_allowed",0),("experiment","truth_allowed",True),
    ("experiment","repeat_or_new_condition_reason",""),("experiment","expires_unix_seconds",999.0),
])
def test_independent_review_experiment_fields(admission_fixture,record,key,value):
    f=admission_fixture
    bound={"bootstrap":f.permit["independent_bootstrap_review"],
           "extension":f.permit["pool_extension"]["independent_review"],
           "experiment":f.permit["experiment_admission"]}[record]
    f.records[bound["path"]][key]=value; fail_before_context(f)

@pytest.mark.parametrize("record,path",[
    ("bootstrap_source","src/arggnn/other.py"),
    ("independent_bootstrap_review","research/reviews/other/review_receipt.json"),
    ("extension_source","scripts/other.py"),("extension_config","configs/other.json"),
    ("extension_review","research/reviews/other/review_receipt.json"),
    ("ledger","research/other.md"),("decisions","research/other.json")])
def test_independent_exact_record_paths(admission_fixture,record,path):
    f=admission_fixture
    if record.startswith("extension_"):
        bound=f.permit["pool_extension"][{"extension_source":"source","extension_config":"config","extension_review":"independent_review"}[record]]
    elif record in ("ledger","decisions"):
        bound=f.experiment[record]
    else: bound=f.permit[record]
    bound["path"]=path; fail_before_context(f)

@pytest.mark.parametrize("key",["allowed_modules","bootstrap_source","bootstrap_config"])
def test_independent_extension_config_join(admission_fixture,key):
    f=admission_fixture; cfg=f.records[f.permit["pool_extension"]["config"]["path"]]
    if key=="allowed_modules": cfg[key]=sorted(boot.pool.MODULES | {boot.RUNNER})
    else: cfg["bootstrap_worker"][key.removeprefix("bootstrap_")]={"path":"other","sha256":"0"*64}
    fail_before_context(f)

@pytest.mark.parametrize("key,value", [
    ("schema","other"),("pool","other"),("lifecycle","draining"),("physical_gpu",3),
])
def test_independent_registry_domain(admission_fixture,key,value):
    f=admission_fixture; f.registry[key]=value; fail_before_context(f)

@pytest.mark.parametrize("key,value",[("pid",124),("uid",999999),("start_ticks",78)])
def test_independent_registry_controller_identity(admission_fixture,key,value):
    f=admission_fixture; f.registry["controller"][key]=value; fail_before_context(f)

@pytest.mark.parametrize("mode",["duplicate","absent","logical_name","args","argv_hash","pid_bool","ticks_bool"])
def test_independent_registry_complete_job_and_argv(admission_fixture,mode):
    f=admission_fixture; job=f.registry["jobs"][0]
    if mode=="duplicate": f.registry["jobs"].append(copy.deepcopy(job))
    elif mode=="absent": f.registry["jobs"]=[]
    elif mode=="logical_name": job["logical_job"]="other"
    elif mode=="args": job["requested"]["args"][-1]="other"
    elif mode=="argv_hash": job["argv"][job["argv"].index("--permit-sha256")+1]="0"*64
    elif mode=="pid_bool": job["pid"]=True
    else: job["leader_start_ticks"]=True
    fail_before_context(f)

@pytest.mark.parametrize("key,value", [
    ("job","other"),("model","other"),("seed",202),("peak_mib",0),("peak_mib",True),
    ("policy","queue"),("log_limit_bytes",1),("storage_bytes",1000000000),("extra",True)
])
def test_independent_bound_requested_fields(admission_fixture,key,value):
    f=admission_fixture
    f.permit["pool_submission"][key]=value
    f.registry["jobs"][0]["requested"][key]=value
    fail_before_context(f)

@pytest.mark.parametrize("target,key,value",[
    ("own","ppid",124),("own","uid",999999),("own","sid",123),("own","pgid",123),
    ("own","pid",123),("controller","start_ticks",78),("controller","state","Z"),
    ("permit","pid",True),("permit","uid",False),("permit","start_ticks",False),
    ("permit","extra",1)
])
def test_independent_parent_session_live_identity(admission_fixture,target,key,value):
    f=admission_fixture
    obj={"own":f.own,"controller":f.controller,"permit":f.permit["controller_identity"]}[target]
    obj[key]=value; fail_before_context(f)

@pytest.mark.parametrize("mode",["changed_lease","wrong_launcher","wrong_job"])
def test_independent_inherited_lease_join(admission_fixture,mode,monkeypatch):
    f=admission_fixture
    if mode=="changed_lease":
        monkeypatch.setattr(boot.guard,"validate_lease",lambda:dict(f.lease,lease_inode=123))
    elif mode=="wrong_launcher": f.lease["launcher_metadata"]["launcher_pid"]=124
    else: f.lease["launcher_metadata"]["job"]="gpu_pool_v1:other:gpu2"
    fail_before_context(f)

@pytest.mark.parametrize("mode",["reordered_devices","duplicates","empty","leased_device_missing",
    "uuid","own_context","duplicate_gpu","duplicate_uuid","missing_inventory","compute_missing",
    "compute_uid","compute_start","compute_duplicate"])
def test_independent_complete_inventory(admission_fixture,mode,monkeypatch):
    f=admission_fixture
    if mode=="reordered_devices": f.permit["project_physical_gpus"]=[3,2]
    elif mode=="duplicates": f.permit["project_physical_gpus"]=[2,2]
    elif mode=="empty": f.permit["project_physical_gpus"]=[]
    elif mode=="leased_device_missing": f.permit["project_physical_gpus"]=[3]
    elif mode=="uuid": f.rows[0]["uuid"]="GPU-00000000-0000-0000-0000-000000000099"
    elif mode=="own_context": f.rows[0]["compute"]=[dict(pid=os.getpid(),used_mib=1)]
    elif mode=="duplicate_gpu": f.rows.append(copy.deepcopy(f.rows[0]))
    elif mode=="duplicate_uuid": f.rows[1]["uuid"]=f.rows[0]["uuid"]
    elif mode=="missing_inventory": f.rows.pop(1)
    else:
        identity=dict(pid=123,uid=os.getuid(),start_ticks=77,physical_gpu=2,uuid=f.lease["visible_gpu_uuid"])
        f.permit["project_compute_identities"]=[identity]
        if mode!="compute_missing": f.rows[0]["compute"]=[dict(pid=123,used_mib=1)]
        if mode=="compute_uid": f.controller["uid"]+=1
        if mode=="compute_start": f.controller["start_ticks"]+=1
        if mode=="compute_duplicate": f.permit["project_compute_identities"].append(copy.deepcopy(identity))
    fail_before_context(f)

@pytest.mark.parametrize("key,value",[
    ("kind","other"),("status","inactive"),("authorizer","author"),("project_root","/other"),
    ("runner_config",{}),("issued_unix_seconds",True),("issued_unix_seconds",1001),
    ("issued_unix_seconds",879),("reservation_scope",[]),
    ("raw_tensors_or_checkpoints_retained",True),("accounting_scope","partial"),
    ("warning_threshold_bytes",450000000001),("hard_limit_bytes",500000000001),
    ("current_reservation_inventory",[]),("reservation_expires_unix_seconds",float("nan")),
    ("observed_repository_bytes",-1),
])
def test_independent_storage_fields(admission_fixture,key,value):
    f=admission_fixture; f.permit["storage_fields"][key]=value; fail_before_context(f)

@pytest.mark.parametrize("mode",["no_log","wrong_log_scope","duplicate_log","duplicate_owned",
    "duplicate_id","empty_id","wrong_log_bytes","wrong_own_bytes","status","float_bytes","traversal",
    "at_warning"])
def test_independent_owned1gb_plus1mib_split(admission_fixture,mode):
    f=admission_fixture; s=f.permit["storage_fields"]; reservations=s["outstanding_growth_reservations"]
    if mode=="no_log": reservations.pop()
    elif mode=="wrong_log_scope": reservations[1]["scope"]=[str(f.output/"logs/pool.log")]
    elif mode in ("duplicate_log","duplicate_owned"):
        new=copy.deepcopy(reservations[1 if mode=="duplicate_log" else 0]);new["id"]="new";reservations.append(new)
    elif mode=="duplicate_id": reservations[1]["id"]=reservations[0]["id"]
    elif mode=="empty_id": reservations[1]["id"]=""
    elif mode=="wrong_log_bytes": reservations[1]["bytes"]-=1
    elif mode=="wrong_own_bytes": reservations[0]["bytes"]+=1
    elif mode=="status": reservations[1]["status"]="released"
    elif mode=="float_bytes": reservations[1]["bytes"]=float(reservations[1]["bytes"])
    elif mode=="traversal": reservations[1]["scope"]=["../outside"]
    else: s["observed_repository_bytes"]=450000000000-sum(r["bytes"] for r in reservations)
    s["current_reservation_inventory"]=copy.deepcopy(reservations)
    s["projected_repository_peak_bytes"]=s["observed_repository_bytes"]+sum(r["bytes"] for r in reservations)
    fail_before_context(f)

def test_initial_launching_registry_window_requires_parent_and_exact_argv(admission_fixture):
    f=admission_fixture; f.registry["jobs"][0].update(state="launching",pid=None,leader_start_ticks=None)
    assert f.admit()["lease"]==f.lease
    f.own["ppid"]=124;fail_before_context(f)

@pytest.mark.parametrize("mode",["config","permit","parent","lease","storage","registry","inventory"])
def test_independent_immediate_precontext_recheck(admission_fixture,mode,monkeypatch):
    f=admission_fixture; admission=f.admit()
    if mode=="config": f.config["status"]="changed"
    elif mode=="permit": f.permit["expires_unix_seconds"]=1000.0
    elif mode=="parent": f.own["sid"]=123
    elif mode=="lease": monkeypatch.setattr(boot.guard,"validate_lease",lambda:dict(f.lease,lease_inode=99))
    elif mode=="storage": admission["permit"]["storage_fields"]["reservation_active"]=False
    elif mode=="registry": f.registry["jobs"][0]["argv"][-1]="changed"
    else: f.rows[0]["compute"]=[dict(pid=999,used_mib=1)]
    with mock.patch.object(boot,"initialize_context") as context:
        with pytest.raises((ValueError,RuntimeError)): boot.recheck_inputs(admission,initial=True)
    context.assert_not_called()

@pytest.mark.parametrize("mode",["extra_key","python","torch","python_path","binary_hash","driver"])
def test_independent_fixed_runtime_before_context(admission_fixture,mode,monkeypatch):
    f=admission_fixture; runtime=copy.deepcopy(f.permit["runtime_expected"]); cfg=copy.deepcopy(f.config)
    exe=Path(runtime["python_executable"]);exe.parent.mkdir(parents=True);exe.write_bytes(b"handmade interpreter hash fixture")
    runtime["python_executable_sha256"]=hashlib.sha256(exe.read_bytes()).hexdigest()
    cfg["expected_runtime"]={k:v for k,v in runtime.items() if k!="driver_version"}
    monkeypatch.setattr(boot.platform,"python_version",lambda:runtime["python"])
    monkeypatch.setattr(boot.sys,"executable",str(exe))
    monkeypatch.setattr(boot.importlib.metadata,"version",lambda name:runtime["torch"])
    query=mock.Mock(return_value=SimpleNamespace(stdout="fixed\n"))
    monkeypatch.setattr(boot.subprocess,"run",query)
    # Restore the actual guard overwritten by the scaffold.
    source=ast.parse((PROJECT/boot.SOURCE).read_text())
    node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="check_runtime")
    scope=dict(boot.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),str(PROJECT/boot.SOURCE),"exec"),scope)
    if mode=="extra_key": runtime["extra"]="bad"
    elif mode=="python": cfg["expected_runtime"]["python"]="other"
    elif mode=="torch": cfg["expected_runtime"]["torch"]="other"
    elif mode=="python_path": monkeypatch.setattr(boot.sys,"executable",str(exe)+"other")
    elif mode=="binary_hash": exe.write_bytes(b"changed")
    else: runtime["driver_version"]="other"
    with pytest.raises(ValueError):scope["check_runtime"](runtime,f.lease,cfg)
    if mode!="driver":query.assert_not_called()
    else:assert query.call_args.args[0]==["nvidia-smi","--id="+f.lease["visible_gpu_uuid"],"--query-gpu=driver_version","--format=csv,noheader"]

@pytest.mark.parametrize("mode",["title_failure","policy_mismatch","metadata_failure","recheck_failure",
    "owned_cap_failure","nvtop_failure"])
def test_independent_pipeline_failure_reachability(admission_fixture,mode,monkeypatch):
    admission,events,written,torch,live=scaffold.mocked_run(admission_fixture,monkeypatch)
    _,policy,runner=boot.load_cuda_modules()
    if mode=="title_failure": monkeypatch.setattr(boot.guard,"apply_and_validate_title",mock.Mock(side_effect=RuntimeError("title")))
    elif mode=="policy_mismatch": policy.configure_deterministic_development=lambda:{"strict":False}
    elif mode=="metadata_failure": monkeypatch.setattr(boot,"validate_adapter_fields",mock.Mock(side_effect=ValueError("metadata")))
    elif mode=="recheck_failure": monkeypatch.setattr(boot,"recheck_inputs",mock.Mock(side_effect=ValueError("pins")))
    elif mode=="owned_cap_failure": runner.check_owned_storage=mock.Mock(side_effect=ValueError("cap"))
    else: monkeypatch.setattr(boot.subprocess,"run",mock.Mock(side_effect=OSError("nvtop")))
    events.clear()
    with pytest.raises((ValueError,RuntimeError,OSError)):boot.run_bootstrap(admission)
    assert "runner" not in events
    assert "domains" not in events
    if mode!="nvtop_failure":assert "context" not in events

@pytest.mark.parametrize("key,value",[
    ("logical_device_count",2),("python","other"),("torch","other"),("torch_cuda","other"),
    ("driver_version","other"),("python_executable","other"),("python_executable_sha256","0"*64)
])
def test_independent_postcontext_runtime_no_runner(admission_fixture,key,value,monkeypatch):
    admission,events,written,torch,live=scaffold.mocked_run(admission_fixture,monkeypatch)
    live[key]=value
    with pytest.raises(ValueError):boot.run_bootstrap(admission)
    assert "context" in events and "runner" not in events
    assert set(written)=={"nvtop_initial.json"}

def test_actual_domain_body_accepts_generated_same_pid_shapes(admission_fixture,monkeypatch):
    """Use frozen runner validation body without importing scientific/Torch code."""
    f=admission_fixture;admission,events,written,torch,live=scaffold.mocked_run(f,monkeypatch)
    _,policy,runner=boot.load_cuda_modules()
    devtree=ast.parse((PROJECT/"src/arggnn/development_execution.py").read_text())
    deterministic=ast.literal_eval(next(n.value for n in devtree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="DETERMINISTIC_POLICY" for t in n.targets)))
    policy.DETERMINISTIC_POLICY=deterministic
    policy.configure_deterministic_development=lambda:events.append("policy") or deterministic
    live.update(device="cuda",execution_policy=deterministic)
    tree=ast.parse((PROJECT/"src/arggnn/anchor_compact_qualification_runner_v1.py").read_text())
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="validate_domains"))
    node.body=[n for n in node.body if not isinstance(n,ast.ImportFrom)]
    scope=dict(require=boot.require,ROOT=f.output.parent,math=math,os=os,
        sealed=SimpleNamespace(PROCESS_TITLE=boot.guard.PROCESS_TITLE),
        DETERMINISTIC_POLICY=deterministic,canonical_uuid=boot.guard.canonical_uuid,
        canonical_path=lambda value:boot.pool.canonical_path(f.output.parent,value))
    exec(compile(ast.Module(body=[node],type_ignores=[]),"frozen_validate_domains_body","exec"),scope)
    runner.validate_domains=scope["validate_domains"]
    events.clear()
    assert boot.run_bootstrap(admission)==0
    runtime=written["runtime_admission.json"];storage=written["storage_admission.json"];launcher=written["launcher_admission.json"]
    for mutate in ("missing_own","wrong_own_kind","duplicate_own","third_device","excluded_device"):
        changed=copy.deepcopy(live)
        if mutate=="missing_own":changed["nvtop_snapshot"][2]["processes"]=[]
        elif mutate=="wrong_own_kind":changed["nvtop_snapshot"][2]["processes"][0]["kind"]="graphics"
        elif mutate=="duplicate_own":changed["nvtop_snapshot"][2]["processes"].append(copy.deepcopy(changed["nvtop_snapshot"][2]["processes"][0]))
        elif mutate=="third_device":
            changed["nvtop_snapshot"] += [dict(processes=[dict(pid=99,kind="compute",cmdline=boot.guard.PROCESS_TITLE)]),dict(processes=[dict(pid=98,kind="compute",cmdline=boot.guard.PROCESS_TITLE)])]
        else:changed["nvtop_snapshot"][0]={"processes":[dict(pid=99,kind="compute",cmdline=boot.guard.PROCESS_TITLE)]}
        changed["nvtop_snapshot_record"]=launcher["nvtop_snapshot"]
        with pytest.raises(ValueError):scope["validate_domains"](runtime,storage,launcher,config_record=boot.RUNNER_CONFIG,source_sha256=runtime["source_sha256"],output=f.output,live=changed,now=1000.0,initial=True)

def test_stdout_stderr_and_all_owned_cache_scopes(tmp_path,monkeypatch):
    monkeypatch.setattr(boot,"ROOT",tmp_path);output=tmp_path/"own"
    with mock.patch.object(boot.os,"dup2") as redirect:boot.prepare_scope(output)
    assert [call.args[1] for call in redirect.call_args_list]==[1,2]
    assert (output/"logs/worker.log").stat().st_size==0
    for name,relative in boot.CACHE_PATHS.items():assert os.environ[name]==str(output/relative)
    assert (output/"logs/worker.log").stat().st_mode & 0o777 == 0o600
    record=boot.write_evidence(output,"bounded.json",{"mock":True})
    data=Path(record["path"]).read_bytes()
    with pytest.raises(FileExistsError):boot.write_evidence(output,"bounded.json",{"replacement":True})
    assert Path(record["path"]).read_bytes()==data
    assert record["sha256"]==hashlib.sha256(data).hexdigest()

def test_source_imports_and_same_pid_runner_call_are_fixed():
    tree=ast.parse((PROJECT/boot.SOURCE).read_text())
    for n in tree.body:
        if isinstance(n,ast.Import):assert all(alias.name.split(".")[0] not in {"torch","numpy"} for alias in n.names)
        if isinstance(n,ast.ImportFrom):assert n.module not in {"torch","anchor_compact_qualification_runner_v1"}
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden={"Popen","execv","execve","fork","system","rmtree","unlink","remove"}
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_bootstrap")
    runnercalls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="runner" and n.func.attr=="main"]
    assert len(runnercalls)==1
