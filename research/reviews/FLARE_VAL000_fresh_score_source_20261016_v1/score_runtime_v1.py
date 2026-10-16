"""Deferred fresh CPU FLARE VAL000 score. No import-time project/scientific work.
No reservation or launch authority is granted by this source packet.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,runpy,subprocess,sys,time

ROOT = '/workspace-SR003.nfs2/ARG-GNN'
SOURCE = 'research/reviews/FLARE_VAL000_fresh_score_source_20261016_v1/score_runtime_v1.py'
REPAIR_REF = {'path': 'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json', 'sha256': 'f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db'}
RENEWAL_REF = {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
CERT_REF = {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
RENEWAL_BOOTSTRAP_REF = {'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/bootstrap_renewal_v1.py', 'sha256': 'f4ae97eeaf244373d1fcd7922e2235d08225468fe5006449d189cc461317fff2'}
RENEWAL_BARRIER_REF = {'path': 'research/reviews/H039_missing_marker_repair_v1/old_guard_terminal_barrier_20261015.json', 'sha256': '85a69aa907ed5c0c80ca5c51232a58ef67a212bceca71b367f1446acdaf8459f'}
EVALUATOR_REF = {'path': 'src/arggnn/american_c3_flare_evaluate_v1.py', 'sha256': 'ce7567d0f71054cfd96c9a44727e20501ad6972e4af7340de7ccc231a9e8aed7'}
CLOSURE = [{'path': 'src/arggnn/american_c3_common_v1.py', 'sha256': '615d3892f1df7e9de39e57991eab883b2eca1af9d13e424661541aac4b2d2abf'}, {'path': 'src/arggnn/american_c3_flare_evaluate_v1.py', 'sha256': 'ce7567d0f71054cfd96c9a44727e20501ad6972e4af7340de7ccc231a9e8aed7'}, {'path': 'src/arggnn/american_c3_flare_common_v1.py', 'sha256': '7dd0b7fc49ace60a6c19a8e7f57b3c9ad7dbeb145810e558ea141668677c511a'}, {'path': 'src/arggnn/american_c3_evaluate_v1.py', 'sha256': '1ada0ab40fe9f198b89a1953bb7c0688e68f1dd44b34dc118600ac25eefc70d3'}]
TEMPLATE_REF = {'path': 'research/reviews/FLARE_VAL000_fresh_score_source_20261016_v1/score_config.inactive.json', 'sha256': 'f1ff16e4628b2fc8c5a2360e3b5bfa9ad65039a9264b6023f5c7c76bcfd7ba78'}
UPSTREAM_REF = {'path': 'research/reviews/FLARE_VAL000_fresh_score_source_20261016_v1/upstream_metadata.json', 'sha256': '293122bc624d96610bd954b525bdb1cfb2f90801de171ea585287d2417f47a02'}
SEAL_REF = {'path': 'data/FLARE_val000_seal_namespace_980_root_v1/completed.json', 'sha256': '71967126b892dfab60761994eacc8f72ec91b02e9729387f6b262e21bc8677bc'}
SIM_REF = {'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/simulation/completed.json', 'sha256': '5a603f261fc711fdf22312633ddfe727cb2ba34bc6030c8e7cbce62b92bc0df3'}
OLD_WAIT_REF = {'path': 'research/reviews/FLARE_val000_seal_namespace_980_root_v1/worker_aux/actual_wait_minimal.json', 'sha256': '4228b4685de805020a7d7df27493c455e4bfa4b629947146e9ba8a428b8780e7'}
OLD_OUTER_REF = {'path': 'research/reviews/FLARE_val000_seal_namespace_980_root_v1/worker_aux/outer_wait.json', 'sha256': '6fe013b7e8c3c46a3438d8ae4d314c03686e9faac86e2168f5053ef311572b04'}
PREDICTION_REF = {'path': 'data/FLARE_val000_seal_namespace_980_root_v1/predictions.npz', 'sha256': '118efa1a079c72525513ad1ec49c7b6b8ac33dc8e2feb50ff335dfad287ed5c0'}
FOCAL_REF = {'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/focals.npz', 'sha256': '220ab48127b41b9a47fdaf831d0c9e4b8b4449eed6e99a5ae14f26234bd36ab4'}
TRUTH_REF = {'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/simulation/truth/c3_pulse_ancestry.npz', 'sha256': '831397e1cecdfde0c68b9c562e4c7947505b3809f834bc704a9f8ca3c12d4352'}
RUNTIME_REF = {'path': 'research/reviews/FLARE_val000_seal_namespace_980_root_v1/runtime.json', 'sha256': 'e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602'}
CAP = 8388608
AUX_ALLOWANCE = 524288
LOG_LIMIT = 65536
OUTPUT = 'experiments/results/FLARE_C3_VAL000_common_score_H020980_20261016_v1'
AUXILIARY = 'research/reviews/FLARE_VAL000_score_H020980_root_20261016_v1/worker_aux'

ROOT=Path(ROOT)
def require(ok,message):
    if not ok:raise ValueError(message)
def local(name):
    p=Path(name);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT) and ".." not in p.parts,"Canonical contained path")
    return p
def reference(name):
    p=local(name);require(p.is_file() and not p.is_symlink(),"Regular exact source/metadata")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def metadata(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"}
        and Path(record["path"]).suffix==".json"
        and not any(x in ("truth","oracle") for x in Path(record["path"]).parts),"Bound JSON metadata only")
    p=local(record["path"]);raw=p.read_bytes()
    require(len(raw)<=1048576 and hashlib.sha256(raw).hexdigest()==record["sha256"],"Bounded exact metadata")
    return json.loads(raw)
def exact(record):
    require(reference(record["path"])==record,"Exact required source bytes")
def bootstrap_renewal():
    # MUST precede any marker consumer import/configure, in parent AND child.
    exact(RENEWAL_REF)
    spec=importlib.util.spec_from_file_location("_FLARE_score_fresh_renewal",local(RENEWAL_REF["path"]))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.install_renewal_v1(CERT_REF)
    return module
def admission(record):
    spec=metadata(record);cfg=metadata(spec["config_ref"]);template=metadata(TEMPLATE_REF)
    template["execution_enabled"]=True
    require(cfg==template,"Exact unchanged FLARE score config")
    require(spec.get("schema")=="FLARE_VAL000_fresh_CPU_score_admission_v1"
        and spec.get("root_authorizer")=="root_agent" and spec.get("execution_authorized") is True
        and spec.get("GPU_authorized") is False and spec.get("worker_entry_ref")==reference(SOURCE)
        and spec["source_ref"]==EVALUATOR_REF and spec["source_closure_refs"]==CLOSURE
        and isinstance(spec["storage_adapter_ref"],dict)
        and spec["accounting_mode"].startswith("prospective_980gb_H020_v")
        and spec["accounting_mode"]!="prospective_980gb_H020_v2"
        and spec["marker_repair_authority_ref"]==REPAIR_REF
        and spec["renewal_source_ref"]==RENEWAL_REF and spec["renewal_certificate_ref"]==CERT_REF,
        "Fresh exact source/CPU/renewal authority")
    expiry=datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    require(expiry.tzinfo is not None and datetime.now(timezone.utc)<expiry,"Fresh score allocation expiry")
    require(spec["output_directory"]==OUTPUT and spec["auxiliary_directory"]==AUXILIARY
        and spec["runtime_ref"]==RUNTIME_REF,"New score namespaces and qualified CPU interpreter")
    for r in CLOSURE:exact(r)
    sealed=metadata(SEAL_REF);sim=metadata(SIM_REF);upstream=metadata(UPSTREAM_REF)
    require(sealed["status"]=="sealed_C3_native_FLARE_calls_and_raw_posteriors"
        and sealed["prediction_ref"]==PREDICTION_REF and sealed["focal_registry_ref"]==FOCAL_REF
        and sealed["parent_id"]=="american_c3_val_000" and sealed["records"]==10000
        and sealed["class_order"]==["EUR","EAS","AFR"]
        and sealed["native_AN1_AN2_calls_used_unchanged"] is True
        and sealed["one_hot_is_native_call_transport"] is True
        and sealed["rounded_ANP_renormalized"] is False
        and sealed["raw_rounded_ANP_modified"] is False
        and sealed["common_focal_registry_unchanged"] is True,
        "Actual already-sealed original native calls/common focal support")
    require(sim["status"]=="completed_C3_simulation_reconstruction"
        and sim["parent_id"]=="american_c3_val_000" and sim["private_truth_ref"]==TRUTH_REF
        and sim["class_order"]==["EUR","EAS","AFR"],"Truth capability from simulation metadata only")
    wait=metadata(OLD_WAIT_REF);outer=metadata(OLD_OUTER_REF)
    require(wait["actual_direct_wait"] is True and wait["returncode"]==0
        and outer["actual_wait_ref"]==OLD_WAIT_REF and outer["returncode"]==0
        and outer["observed_owned_groups_ESRCH"] is True and outer["error"] is None
        and upstream["old_external_tool_terminal"]=="UNKNOWN"
        and upstream["old_external_final_guard"]=="UNKNOWN","Preserve genuine old evidence and uncertainty")
    storage=spec["storage"];bindings=storage["reservation"]["consumer_bindings"]
    required=[*CLOSURE,reference(SOURCE),spec["config_ref"],RUNTIME_REF,spec["storage_adapter_ref"],
        spec["completed_overlay_policy_ref"],REPAIR_REF,RENEWAL_REF,CERT_REF,
        RENEWAL_BOOTSTRAP_REF,RENEWAL_BARRIER_REF,TEMPLATE_REF,UPSTREAM_REF,
        SEAL_REF,SIM_REF,OLD_WAIT_REF,OLD_OUTER_REF,*spec["additional_required_refs"]]
    require(len(bindings)<=64 and all(r in bindings for r in required),"Complete <=64 direct reference closure")
    require(TRUTH_REF not in bindings and not any(
        any(part in ("truth","oracle") for part in Path(r["path"]).parts) for r in bindings),
        "Storage guard must not open private truth before evaluator verifies predictions")
    require(storage["accounting_mode"]==spec["accounting_mode"] and storage["budget_components"]["consumer_adapter"]==spec["storage_adapter_ref"]
        and storage["completed_overlay_policy_ref"]==spec["completed_overlay_policy_ref"]
        and storage["reserved_new_bytes"]==CAP and storage["reservation"]["reserved_bytes"]==CAP
        and storage["output_directory"]==OUTPUT
        and storage["reservation"]["auxiliary_scopes"]==[str(ROOT/AUXILIARY)],
        "Fresh small new-overlay reservation covers output and actual wait/log scope")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and Path.cwd().resolve()==ROOT,"Explicit CPU-only repository process")
    return spec,cfg
def guard(spec):
    record=spec["storage_adapter_ref"];exact(record)
    s=importlib.util.spec_from_file_location("_FLARE_score_actual_H020980_guard",local(record["path"]))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.configure_v2(REPAIR_REF,policy_ref=spec["completed_overlay_policy_ref"])
    require(m.MODE==spec["accounting_mode"] and m.SOURCE==record["path"],"Only actual root-bound activated H020980 successor")
    return m
def worker(record):
    bootstrap_renewal()
    spec,_=admission(record)
    # common.cli loads/configures the unchanged guard after this bootstrap.
    os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(spec["completed_overlay_policy_ref"])
    sys.argv=["arggnn.american_c3_flare_evaluate_v1","--execute","--admission",record["path"],
              "--expected-admission-sha256",record["sha256"]]
    runpy.run_module("arggnn.american_c3_flare_evaluate_v1",run_name="__main__",alter_sys=True)
    return 0
def precharged_json(directory,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536 and Path(name).name==name,"Bounded exclusive actual lifecycle receipt")
    path=directory/name
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return reference(path)
def launch(record):
    # Root must consume THIS new outer command's real terminal separately.
    bootstrap_renewal()
    spec,cfg=admission(record);m=guard(spec);storage=spec["storage"]
    m.validate_storage_v9(spec,cfg,storage,source_ref=EVALUATOR_REF,
        config_ref=spec["config_ref"],runtime_ref=RUNTIME_REF)
    aux=local(AUXILIARY)
    require(not aux.exists() and not local(OUTPUT).exists(),"Never replay or overwrite score scope")
    # Precharge bounded launch/wait/stdout persistence before the worker exists.
    m.budget_check_v9(storage,boundary=True,required_write_bytes=AUX_ALLOWANCE)
    aux.mkdir()
    runtime=metadata(RUNTIME_REF);env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONPATH=".:src:scripts",PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
    argv=[runtime["executable"],"-B",str(ROOT/SOURCE),"--worker",
        "--admission",record["path"],"--expected-admission-sha256",record["sha256"]]
    started=datetime.now(timezone.utc).isoformat();clock=time.monotonic()
    proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        launched=precharged_json(aux,"launch.json",dict(schema="FLARE_VAL000_fresh_score_launch_v1",
            admission_ref=record,pid=proc.pid,pgid=proc.pid,root_supervisor_pid=os.getpid(),
            argv=argv,started_utc=started))
        kept=bytearray();discarded=0
        while True:
            chunk=proc.stdout.read(65536)
            if not chunk:break
            take=min(len(chunk),LOG_LIMIT-len(kept));kept.extend(chunk[:take]);discarded+=len(chunk)-take
    finally:
        if proc.stdout is not None:proc.stdout.close()
        rc=proc.wait()
    try:os.killpg(proc.pid,0);drained=False
    except ProcessLookupError:drained=True
    log=aux/"stdout.txt"
    with log.open("xb") as stream:stream.write(kept);stream.flush();os.fsync(stream.fileno())
    waited=precharged_json(aux,"actual_wait.json",dict(
        schema="FLARE_VAL000_fresh_score_actual_direct_wait_v1",admission_ref=record,
        launch_ref=launched,pid=proc.pid,returncode=rc,actual_direct_wait=True,
        owned_group_ESRCH=drained,stdout_ref=reference(log),discarded_stdout_bytes=discarded,
        elapsed_seconds=time.monotonic()-clock,waited_utc=datetime.now(timezone.utc).isoformat(),
        old_external_tool_terminal="UNKNOWN",old_external_final_guard="UNKNOWN",
        new_external_tool_terminal="ROOT_MUST_CONSUME_ACTUAL_OUTER_TOOL_TERMINAL"))
    print(json.dumps(dict(actual_new_score_wait_ref=waited,returncode=rc,owned_group_ESRCH=drained)),flush=True)
    require(rc==0 and drained,"New score worker must genuinely finish and drain")
    done_ref=reference(local(OUTPUT)/"completed.json");done=metadata(done_ref)
    require(done["status"]=="completed_external_C3_score"
        and done["method"]==cfg["method"] and done["parent_id"]=="american_c3_val_000"
        and done["records"]==10000 and done["class_order"]==["EUR","EAS","AFR"]
        and done["prediction_ref"]==PREDICTION_REF and done["focal_registry_ref"]==FOCAL_REF
        and done["external_truth_ref"]==TRUTH_REF and done["probability_floor"] is False
        and done["probabilities_modified"] is False
        and done["author_Table1_exact_reproduction"] is False,"Unchanged common score completion")
    # Preserve the actual direct wait above even if a later guard check fails.
    m.budget_check_v9(storage,boundary=True)
    print(json.dumps(dict(score_completion_ref=done_ref,new_actual_wait_ref=waited,
        final_storage_guard_passed=True,outer_tool_status="PENDING_ROOT_TERMINAL_CONSUMPTION")),flush=True)
    return 0
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--launch",action="store_true");mode.add_argument("--worker",action="store_true")
    p.add_argument("--admission");p.add_argument("--expected-admission-sha256");a=p.parse_args(argv)
    if not (a.launch or a.worker):
        print(json.dumps(dict(status="source_only_inactive_no_guard_or_scoring_import")));return 0
    require(a.admission and a.expected_admission_sha256,"Explicit separately activated root score admission")
    record=dict(path=a.admission,sha256=a.expected_admission_sha256)
    return worker(record) if a.worker else launch(record)
if __name__=="__main__":raise SystemExit(main())
