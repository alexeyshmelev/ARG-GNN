"""Author source/AST/RAM checks only. No actual guard, probe, accounting or science."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,sys
ROOT=Path(".")
REL="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
SIM="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3"
OBS="research/reviews/frozen_TRAIN001_observe_ready_20261017_v2"
checks=[]
def check(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
def reject(call,label):
    try:call()
    except (ValueError,KeyError,TypeError,AssertionError):checks.append(label);return
    raise AssertionError("Accepted invalid fixture: "+label)
def need(value,message):
    if not value:raise ValueError(message)
def rr(path):return dict(path=str(path),sha256=hashlib.sha256(str(path).encode()).hexdigest())
def actual_or_RAM_ref(path):
    if str(path).startswith("ram/"):return rr(path)
    p=Path(path)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def extract(file,names,namespace):
    tree=ast.parse(Path(file).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
    check(set(n.name for n in nodes)==set(names),"AST functions present "+str(sorted(names)))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),file+"#RAM-only","exec"),namespace)
    return namespace
manifest=json.loads(Path(REL+"/source_manifest.v1.json").read_text())
for name in ("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py"):
    ast.parse(Path(REL+"/"+name).read_text());checks.append(name+" parses")
check(manifest["original_consumer_ceiling"]==64 and manifest["full_original_current_guard_roles"]==47,"Original64ceiling/all47roles")
check(manifest["condition"]["stage_caps"]["infer"]["reservation_bytes"]==13000000000,"Inference13GBwhole")
check(manifest["condition"]["stage_caps"]["native_paths"]["reservation_bytes"]==16500000000,"Native16.5GBwhole")
for stage,primary,wall in [("infer",12500000000,10800),("native_paths",16000000000,7200)]:
    c=manifest["condition"]["stage_caps"][stage]
    check(manifest["condition"]["stage_templates"][stage]["output_cap_bytes"]==primary,"Original primary cap "+stage)
    check(c["reservation_bytes"]>=primary+c["aux_bytes"] and c["wall_seconds"]==wall
        and c["rss_bytes"]==68719476736 and c["CPU_threads"]==8 and c["cleanup_seconds"]==30
        and c["terminal_evidence_margin_seconds"]==150 and c["work_cutoff_margin_seconds"]==300,"Original complete lifecycle "+stage)
obs=json.loads(Path(OBS+"/source_manifest.v1.json").read_text())
check(manifest["condition"]["stage_templates"]["infer"]==obs["condition"]["inference_template_inactive"],"Unchanged inference template")
native=json.loads(Path("research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1/config.json").read_text())
native.update(execution_enabled=False,observation_ref=None)
check(manifest["condition"]["stage_templates"]["native_paths"]==native,"Unchanged native template")
# Function suites see only explicit RAM metadata, never a guard or operating-system API.
ram={}
condition=manifest["condition"]
w=dict(need=need,read=lambda r:copy.deepcopy(ram[r["path"]]),metadata_ref=actual_or_RAM_ref,local=lambda p:p,Path=Path,
    PARENT="american_c3_train_002",STAGE="infer",MANIFEST=REL+"/source_manifest.v1.json",SOURCE=REL+"/worker_entry_v1.py",
    ALLOWED_PARENTS=tuple("american_c3_train_%03d"%i for i in range(1,20)),
    SIM_BATCH_SOURCE=SIM+"/worker_entry_v1.py",IDENTITY_SOURCE="src/arggnn/american_c3_gnn_targets_v1.py")
ram[w["MANIFEST"]]=manifest
extract(REL+"/worker_entry_v1.py",["stage_status","registered_row","scientific_config","canonical_ref_shape",
    "observation_metadata","predecessor_capsule","pathlib_packet"],w)
check(w["stage_status"]("infer")=="completed_current_C3_inference","Inference exact status")
check(w["stage_status"]("native_paths")=="completed_native_C3_path_preparation","Native exact status")
reject(lambda:w["stage_status"]("train"),"No training stage")
for index in range(1,20):
    parent="american_c3_train_%03d"%index;r=w["registered_row"](parent)
    check(r["path_sampling_seed"]==92001+index,"Original registered parent/path seed "+parent)
reject(lambda:w["registered_row"]("american_c3_train_000"),"TRAIN000 no rerun")
reject(lambda:w["registered_row"]("american_c3_val_000"),"No VAL")
reject(lambda:w["registered_row"]("american_c3_test_000"),"No TEST")
simulation_template=json.loads(Path("research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/scientific_condition.v5.json").read_text())["original_full30mb_template"]
def simulation_cfg(parent):
    r=w["registered_row"](parent)
    return dict(simulation_template,execution_enabled=True,parent_id=parent,split="train",ancestry_seed=r["ancestry_seed"],mutation_seed=r["mutation_seed"])
w["coordinator_support"]=lambda:SimpleNamespace(scientific_config_for_parent=lambda p,a:simulation_cfg(p))
def fixture(stage="infer",parent="american_c3_train_002"):
    w["PARENT"]=parent;w["STAGE"]=stage
    prior_stage="simulate"if stage=="infer"else"infer";output="ram/results/"+parent+"/"+prior_stage
    packet="ram/packet/"+parent+"/"+prior_stage;inp_ref=rr(packet+"/input.json")
    refs=dict(input_ref=inp_ref,config_ref=rr(packet+"/config.json"),completion_ref=rr(output+"/completed.json"),
        wait_ref=rr(packet+"/worker_aux/actual_wait.json"),outer_ref=rr(packet+"/outer.json"),
        observation_ref=rr(output+"/observed/manifest.json"),roster_ref=rr(output+"/observed/roster.json"))
    req_ref=rr(packet+"/request.json")
    expected_manifest=(condition["original_simulation_batch_manifest_ref"]if stage=="infer"else
        condition["original_inference_manifest_ref"]if parent.endswith("001")else actual_or_RAM_ref(w["MANIFEST"]))
    expected_source_metadata=json.loads(Path(expected_manifest["path"]).read_text())
    ram[expected_manifest["path"]]=expected_source_metadata
    source=actual_or_RAM_ref(SIM+"/worker_entry_v1.py")if stage=="infer"else(condition["original_TRAIN001_inference_source_ref"]if parent.endswith("001")else actual_or_RAM_ref(w["SOURCE"]))
    sources={r["path"]:r["sha256"]for r in expected_source_metadata["refs"]}
    cfg=simulation_cfg(parent)if stage=="infer"else dict(condition["stage_templates"]["infer"],execution_enabled=True,observation_ref=rr("ram/original_sim_obs"))
    inp=dict(parent_id=parent,stage=prior_stage,phase=prior_stage+"_"+parent.removeprefix("american_c3_"),
        GPU_authorized=False,query_truth_access_authorized=False,TEST_access_authorized=False,
        refs=dict(config=refs["config_ref"]),root_stage_request_ref=req_ref,worker_auxiliary_scope=packet+"/worker_aux",
        output_directory=output,root_driver_ref=source,main_source_ref=source,source_sha256=sources,
        storage=dict(reservation=dict(key="H041/"+parent+"/"+prior_stage)))
    req=dict(scientific_config=cfg,source_sha256=inp["source_sha256"],source_manifest_ref=expected_manifest)
    done=dict(parent_id=parent,status="completed_C3_simulation_reconstruction"if stage=="infer"else"completed_current_C3_inference",
        observation_ref=refs["observation_ref"])
    wait=dict(schema="American_C3_full_original_CPU_actual_wait_v1",input_ref=inp_ref,completed_ref=refs["completion_ref"],parent_id=parent,
        phase=inp["phase"],actual_direct_worker_wait=True,actual_direct_launcher_wait=True,actual_OS_wait_status=0,returncode=0,
        all_owned_processes_and_groups_drained=True,operational_failure=None,cleanup_errors=[],adopted_waits=[],group_observations=[dict(absent=True)])
    outer=dict(schema="ROOT_batch_unit_actual_OS_terminal_v1",root_authorizer="root_agent",actual_supervisor_OS_terminal_consumed=True,
        actual_exit_code=0,UI_tool_terminal_claimed=False,actual_worker_wait_ref=refs["wait_ref"],input_ref=inp_ref,
        completed_ref=refs["completion_ref"],all_owned_processes_and_groups_drained=True)
    obs=dict(schema="american_c3_observed_v1",parent_id=parent,split="train",sequence_length_bp=30000000,sample_unit="haploid",
        class_order=["EUR","EAS","AFR"],counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,
        roster_ref=refs["roster_ref"],genealogy_kind="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG"if stage=="infer"else"inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
        sites_and_sample_identities_verified=True,source_observation_ref=rr("ram/original_sim_obs"),migrations_and_internal_population_labels_removed=True)
    samples=[]
    for pop,count,cls in[("AFR",200,2),("EUR",200,0),("ASIA",200,1),("ADMIX",1000,None)]:
        for j in range(count):samples.append(dict(sample_index=len(samples),sample_id=f"{pop}_{j//2:05d}",haplotype=j%2,age_generations=0,
            role="query"if pop=="ADMIX"else"pure_source_reference",observed_reference_class=cls,observed_node_id=len(samples)))
    roster=dict(parent_id=parent,class_order=["EUR","EAS","AFR"],samples=samples)
    values=dict(input_ref=inp,config_ref=cfg,completion_ref=done,wait_ref=wait,outer_ref=outer,observation_ref=obs,roster_ref=roster)
    accept=rr(packet+"/accepted.json");receipt=rr(packet+"/receipt.json");revref=rr(packet+"/revocation.json")
    ram[receipt["path"]]=dict(key=inp["storage"]["reservation"]["key"])
    ram[accept["path"]]=dict(parent_id=parent,actual_direct_OS_wait0=True,all_owned_processes_and_groups_drained=True,receipt_ref=receipt)
    ram[revref["path"]]=dict(input_ref=inp_ref,completed_ref=refs["completion_ref"],direct_wait_ref=refs["wait_ref"],
        actual_outer_terminal_ref=refs["outer_ref"],all_original_launch_resume_retry_write_routes_permanently_revoked=True)
    ram[req_ref["path"]]=req
    for k,r in refs.items():ram[r["path"]]=copy.deepcopy(values[k])
    capsule=dict(schema="TreeLoom_preparation_predecessor_capsule_v1",parent_id=parent,next_stage=stage,refs=refs,metadata=values,
        close_accepted_ref=accept,permanent_revocation_ref=revref,query_truth_opened=False)
    cr=rr(packet+"/capsule.json");ram[cr["path"]]=capsule;return cr,capsule
for stage in("infer","native_paths"):
    cr,cap=fixture(stage);check(w["predecessor_capsule"](cr,condition)==cap["refs"],"All seven genuine RAM predecessor roles accepted "+stage)
for name,mutate in [
 ("missing role",lambda x:x["refs"].pop("roster_ref")),
 ("snapshot differs",lambda x:x["metadata"]["config_ref"].update(mutation_rate=.3)),
 ("wrong parent",lambda x:x.update(parent_id="american_c3_train_003")),
 ("truth opened",lambda x:x.update(query_truth_opened=True)),
 ("worker nonzero",lambda x:x["metadata"]["wait_ref"].update(returncode=1)),
 ("worker undrained",lambda x:x["metadata"]["wait_ref"].update(all_owned_processes_and_groups_drained=False)),
 ("phase differs",lambda x:x["metadata"]["input_ref"].update(phase="infer_train_002")),
 ("source differs",lambda x:x["metadata"]["input_ref"].update(root_driver_ref=rr("ram/other_source.py"))),
 ("outer UI claim",lambda x:x["metadata"]["outer_ref"].update(UI_tool_terminal_claimed=True)),
 ("outer nonzero",lambda x:x["metadata"]["outer_ref"].update(actual_exit_code=1)),
 ("roster shortened",lambda x:x["metadata"]["roster_ref"]["samples"].pop()),
 ("roster reorder",lambda x:x["metadata"]["roster_ref"]["samples"].reverse()),
 ("roster label",lambda x:x["metadata"]["roster_ref"]["samples"][0].update(observed_reference_class=0)),
 ("oracle native input",lambda x:x["metadata"]["observation_ref"].update(genealogy_kind="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG"))]:
    stage="native_paths"if name=="oracle native input"else"infer"
    cr,cap=fixture(stage);mutate(cap)
    # Synchronize the original RAM records except snapshot-diff, so semantic failures are exercised.
    if name!="snapshot differs":
        for k,r in cap["refs"].items():ram[r["path"]]=copy.deepcopy(cap["metadata"][k])
    ram[cr["path"]]=cap;reject(lambda:w["predecessor_capsule"](cr,condition),"Predecessor rejects "+name)
cr,cap=fixture("infer")
original_request=ram[cap["metadata"]["input_ref"]["root_stage_request_ref"]["path"]]
one_key=next(iter(original_request["source_sha256"]));original_request["source_sha256"].pop(one_key)
cap["metadata"]["input_ref"]["source_sha256"]=copy.deepcopy(original_request["source_sha256"])
ram[cap["refs"]["input_ref"]["path"]]=copy.deepcopy(cap["metadata"]["input_ref"]);ram[cr["path"]]=cap
reject(lambda:w["predecessor_capsule"](cr,condition),"Incomplete predecessor source map rejected even when input/request agree")
cr,cap=fixture("native_paths","american_c3_train_001")
# Genuine ROOT single observation terminal aliases may be supplied for existing001 only.
cap["metadata"]["outer_ref"]=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,
    input_ref=cap["refs"]["input_ref"],completed_ref=cap["refs"]["completion_ref"],actual_worker_wait_ref=cap["refs"]["wait_ref"],
    all_future_writes_permanently_revoked=True,output_frozen=True)
ram[cap["refs"]["outer_ref"]["path"]]=cap["metadata"]["outer_ref"];ram[cr["path"]]=cap
check(w["predecessor_capsule"](cr,condition)==cap["refs"],"Existing001 genuine ROOT observation terminal reuse")
# Direct binding count uses actual original immutable40 baseline role identities, not a source estimate.
original=json.loads(Path("research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/ROOT_actual_TRAIN001_root02_oct17_v5/input.json").read_text())
baseline=original["storage"]["reservation"]["consumer_bindings"][:40]
check(len({r["path"]for r in baseline})==40,"Authentic original47role40unique identity projection")
# rr uses actual original baseline identity for overlap paths.
actual_by_path={r["path"]:r for r in baseline}
def direct_ref(p):return actual_by_path.get(str(p),rr(p))
ns=dict(need=need,c=SimpleNamespace(ref=direct_ref,PIDFD=rr("pidfd"),OWNED=rr("owned"),NAMESPACE_REF=rr("namespace"),V4_REF=rr("v4")),
    SOURCE=REL+"/worker_entry_v1.py",MANIFEST=REL+"/source_manifest.v1.json",
    OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1",
    PATHS="src/arggnn/american_c3_paths_v1.py",AUTHOR=".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py",
    IDENTITY_SOURCE="src/arggnn/american_c3_gnn_targets_v1.py",SELECTOR=actual_by_path["research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json"],
    CHILD=rr("child"),SIM_BATCH_SOURCE=SIM+"/worker_entry_v1.py")
ns["c"].POLICY_REF=rr("policy")
extract(REL+"/worker_entry_v1.py",["direct_roles"],ns)
q=dict(source_review_ref=rr("review"),prior_capsule_ref=rr("capsule"),batch_control_input_ref=rr("simcontrol"),pipeline_authority_ref=rr("authority"))
roles=ns["direct_roles"](rr("request"),q,rr("config"),rr("request"),rr("runtime"))
check(len(roles)==25,"All25direct stage roles")
check(len({json.dumps(r,sort_keys=True)for r in baseline+roles})==64,"Original47roles plus25direct roles exactly64unique")
# Controller flow uses RAM process/accounting doubles only, with real scheduler run_batch body.
flow_ram={};events=[];saved={};root_input=rr("pipelineinput");authority_ref=rr("pipelineauthority")
a=dict(run_directory="ram/run",simulation_batch_control_input_ref=rr("simcontrol"),
    TRAIN000_readiness_reuse=dict(parent_id="american_c3_train_000"),TRAIN000_simulation_completion_ref=rr("sim000"),
    TRAIN001_simulation_completion_ref=rr("sim001"),TRAIN001_inference_predecessor_capsule_ref=rr("infer001"),
    expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=40)).isoformat())
flow_ram[root_input["path"]]=dict(schema="TreeLoom_TRAIN_preparation_admin_input_v1",source_manifest_ref=rr(REL+"/source_manifest.v1.json"),
    simulation_batch_control_input_ref=a["simulation_batch_control_input_ref"],authority_ref=authority_ref,
    storage=dict(reservation=dict(key="admin",reserved_bytes=33554432)))
flow_ram["infer001"]=dict(parent_id="american_c3_train_001",next_stage="native_paths",close_accepted_ref=rr("close001"))
class NullContext:
    def __enter__(self):return self
    def __exit__(self,*a):pass
class Dir:
    def is_dir(self):return True
    def iterdir(self):return iter([])
ww=SimpleNamespace(ALLOWED_PARENTS=tuple("american_c3_train_%03d"%i for i in range(1,20)),
    validate_pipeline_authority=lambda r:a,accounting_transaction=lambda r:NullContext())
gg=SimpleNamespace()
state_value=({"admin":dict(key="admin",reserved_bytes=33554432)}, {}, {"limit_bytes":10**12,"projected_bytes":0}, None,None,None)
def put(ctx,name,value):
    need(name not in saved,"No duplicate immutable flow artifact");saved[name]=copy.deepcopy(value);return rr(name)
def simulated(ctx,parent):
    events.append(("simulation_ready",parent))
    return dict(parent_id=parent,stage="infer",unit_key=parent.removeprefix("american_c3_")+"_infer",prior_capsule_ref=rr(parent+"_sim"),simulation_completion_ref=rr(parent+"_simdone"))
def unitrequest(ctx,u,lane,capacity,ram):events.append(("request",u["parent_id"],u["stage"]));return dict(u)
def call(ctx,commands,seconds,rss,name):
    roots=[]
    for module,flag,record in commands:
        if flag=="--admit-ref":
            unit=saved[record["path"]];ir=rr(unit["unit_key"]+"_input");flow_ram[ir["path"]]=unit;root=dict(execution_started=False,input_ref=ir)
        elif flag=="--execute-ref":
            unit=flow_ram[record["path"]];events.append(("execute",unit["parent_id"],unit["stage"]));root=dict(input_ref=record)
        else:root=dict(record=record)
        roots.append(root)
    return dict(roots=roots)
def accept(ctx,u,r):
    events.append(("closed",u["parent_id"],u["stage"]));return rr(u["unit_key"]+"_closed")
def downstream(ctx,u,ir,sup,accepted):
    need(("closed",u["parent_id"],"infer")in events,"Inference actual close precedes native")
    return dict(parent_id=u["parent_id"],stage="native_paths",unit_key=u["parent_id"].removeprefix("american_c3_")+"_native_paths",
        prior_capsule_ref=rr("infercapsule"),inference_close_accepted_ref=accepted,simulation_completion_ref=u["simulation_completion_ref"])
def ready(ctx,u,ir,sup,accepted):
    need(("closed",u["parent_id"],"native_paths")in events,"Native actual close precedes readiness")
    return dict(parent_id=u["parent_id"])
def seals(ctx,rows,sim):
    check(len(rows)==len(sim)==20,"Flow all20 before barrier")
    check(set(rows)==set(sim)=={"american_c3_train_%03d"%i for i in range(20)},"Flow exact original20roster")
    return rr("readiness"),rr("truth_handles")
fake_signal=SimpleNamespace(SIGINT=2,SIGTERM=15,signal=lambda *a:None)
flow=dict(read=lambda r:flow_ram[r["path"]],qualified=lambda r:(ww,a,manifest),guard=lambda m:gg,state=lambda g,m:state_value,
    need=need,ref=rr,MANIFEST=REL+"/source_manifest.v1.json",local=lambda p:Dir(),SimpleNamespace=SimpleNamespace,
    put=put,original_owned=lambda m:None,signal=fake_signal,CANCEL=[None],datetime=datetime,timezone=timezone,
    time=SimpleNamespace(time=lambda:datetime.now(timezone.utc).timestamp(),sleep=lambda n:None),
    simulated_candidate=simulated,wait_wave_capacity=lambda ctx,candidates:candidates[:2],unit_request=unitrequest,
    available_memory=lambda:2**50,WORKER="worker",CLOSE="close",argv=lambda *a:a,group_call=call,terminal=lambda r:r,
    prepare_close=lambda ctx,u,ir,sup:dict(unit=u),accept_close=accept,downstream_candidate=downstream,readiness_row=ready,
    seal_all20=seals,json=json,
    close_when_ready=lambda ctx,u,ca:call(ctx,[("close","--authorization-ref",ca)],1500,4*2**30,u["unit_key"]+".close_OS.json"),
    preoperation_queue=lambda ctx,key,v:events.append(("prequeue",key)))
extract(REL+"/batch_ROOT_v1.py",["run_batch"],flow)
result=flow["run_batch"](root_input)
check(result["status"]=="PREPARATION_COMPLETE"and result["returncode"]==0,"Complete37stage flow actual-close doubles")
executed=[e[1:]for e in events if e[0]=="execute"]
expected={(f"american_c3_train_{i:03d}",stage)for i in range(1,20)for stage in("infer","native_paths")if not(i==1 and stage=="infer")}
check(len(executed)==37 and set(executed)==expected,"Exactly original37missing stages, no repetitions")
check(len(set(executed))==37,"Every stage attempted once")
check(not any(p.endswith("000")or(p.endswith("001")and s=="infer")for p,s in executed),"No TRAIN000 or TRAIN001 inference rerun")
check(all(len(v["completed"])<=2 for k,v in saved.items()if k.startswith("wave_")),"At most2original roots per wave")
# Fail after original first admission, stop with preserved failure and no replay.
saved.clear();events.clear();old_call=flow["group_call"]
def fail(ctx,commands,seconds,rss,name):
    if any(x[1]=="--execute-ref"for x in commands):raise ValueError("RAM genuine worker failure")
    return old_call(ctx,commands,seconds,rss,name)
flow["group_call"]=fail
result=flow["run_batch"](root_input)
check(result["returncode"]==1 and result["status"]=="PREPARATION_STOPPED_NO_REPLAY","Failure stops controller")
check("failed_preserved.json"in saved and not any(k=="all20_training_readiness.json"for k in saved),"No failed barrier or inferred credit")

# One queued sibling does not consume an admission attempt or replay an admitted sibling.
saved.clear();events.clear();flow["group_call"]=call;first_busy=[True]
def queue_first(ctx,commands,seconds,rss,name):
    if commands[0][1]=="--admit-ref"and first_busy[0]:
        first_busy[0]=False
        return dict(roots=[dict(status="QUEUED_BEFORE_OPERATION",accounting_operation_started=False,execution_started=False)])
    return call(ctx,commands,seconds,rss,name)
flow["group_call"]=queue_first
result=flow["run_batch"](root_input)
check(result["returncode"]==0,"Busy coordinator pre-operation queue completes without replay")
check(len([e for e in events if e[0]=="execute"])==37 and len({e[1:]for e in events if e[0]=="execute"})==37,
    "Queued sibling causes zero repeated science")
check(len([e for e in events if e[0]=="prequeue"])==1,"Exact pre-operation queue marker consumed")
check(any(len(v["completed"])==1 for k,v in saved.items()if k.startswith("wave_")),"Genuinely admitted partial wave executes")
# Source wrapper accepts only before-yield timeout as a queue, all after-yield errors propagate.
class Busy:
    def __enter__(self):raise ValueError("Finite shared ROOT accounting coordinator wait")
    def __exit__(self,*a):pass
class Acquired:
    def __enter__(self):return self
    def __exit__(self,*a):pass
qstub=dict(batch_control_input_ref=rr("simcontrol"),created_at_utc=datetime.now(timezone.utc).isoformat())
admit_calls=[]
ad=dict(read=lambda r:qstub,accounting_transaction=lambda r:Busy(),datetime=datetime,timezone=timezone,
    admit_under_coordinator=lambda r:admit_calls.append(r))
extract(REL+"/worker_entry_v1.py",["admit"],ad)
check(ad["admit"](rr("request"))["status"]=="QUEUED_BEFORE_OPERATION"and not admit_calls,"Acquire timeout returns queue before any operation")
ad["accounting_transaction"]=lambda r:Acquired()
def afteryield(r):raise ValueError("Finite shared ROOT accounting coordinator wait")
ad["admit_under_coordinator"]=afteryield
reject(lambda:ad["admit"](rr("request")),"Same failure after coordinator yield is never queued/replayed")
qstub["created_at_utc"]=(datetime.now(timezone.utc)-timedelta(seconds=245)).isoformat()
check(ad["admit"](rr("request"))["immutable_request_refresh_required"]is True,"Stale capacity exits before operation with fresh immutable request required")
# Actual scheduler AST chooses available1or2 original slots with different original stage caps.
capacity_ctx=SimpleNamespace(w=SimpleNamespace(validate_pipeline_authority=lambda r:a),authority_ref=authority_ref,
    authority=a,g=gg,manifest=manifest)
available=[0];sleeps=[]
def capacity_sleep(n):sleeps.append(n);available[0]=1
scheduler=dict(need=need,CANCEL=[None],datetime=datetime,time=SimpleNamespace(time=lambda:datetime.now(timezone.utc).timestamp(),sleep=capacity_sleep),
    state=lambda g,m:({"admin":{}},{},{"limit_bytes":70000000000,"projected_bytes":0}),
    available_memory=lambda:128*2**30,available_original_slots=lambda ctx:available[0],json=json)
extract(REL+"/batch_ROOT_v1.py",["wait_wave_capacity"],scheduler)
candidates=[dict(stage="infer",unit_key="train_002_infer"),dict(stage="native_paths",unit_key="train_001_native_paths")]
selected=scheduler["wait_wave_capacity"](capacity_ctx,candidates)
check(len(selected)==1 and sleeps==[30],"Zero original slots queues before storage admission, then exactly1slot runs")
available[0]=2
check(len(scheduler["wait_wave_capacity"](capacity_ctx,candidates))==2,"Two free original slots and70GB allows13+16.5GB")
scheduler["available_memory"]=lambda:64*2**30
check(len(scheduler["wait_wave_capacity"](capacity_ctx,candidates))==1,"Available64GiBRSS permits one original stage")
# Additional focal seal uses unchanged identity_sha and only explicit RAM arrays/file doubles.
import numpy as np
identity_source=Path("src/arggnn/american_c3_gnn_targets_v1.py").read_text()
records=dict(record_ids=np.arange(10000,dtype=np.int64),sample_nodes=np.arange(10000,dtype=np.int64),
    sample_indices=np.arange(10000,dtype=np.int64),positions_bp0=np.arange(10000,dtype=np.float64),
    parent_id=np.asarray("american_c3_train_002"),class_order=np.asarray(["EUR","EAS","AFR"]))
class FocalArrays:
    def __enter__(self):return records
    def __exit__(self,*a):pass
payload=b"RAM_FOCALS_NO_FILE"
focal_rr=dict(path="ram/focals.npz",sha256=hashlib.sha256(payload).hexdigest())
class FocalPath:
    def stat(self):return SimpleNamespace(st_size=len(payload))
    def read_bytes(self):return payload
class IdentityPath:
    def read_text(self):return identity_source
local_double=lambda p:IdentityPath()if p=="src/arggnn/american_c3_gnn_targets_v1.py"else FocalPath()
written={}
writer=SimpleNamespace(json=lambda name,value:written.update({name:value})or rr(name))
published=dict(need=need,canonical_ref_shape=lambda r:r,local=local_double,hashlib=hashlib,json=json,metadata_ref=rr,
    read=lambda r:dict(refs=[rr("src/arggnn/american_c3_gnn_targets_v1.py")]),
    MANIFEST=REL+"/source_manifest.v1.json",PARENT="american_c3_train_002",IDENTITY_SOURCE="src/arggnn/american_c3_gnn_targets_v1.py")
extract(REL+"/worker_entry_v1.py",["publish_focal_seal"],published)
prior_load=np.load;np.load=lambda p,allow_pickle=False:FocalArrays()
try:published["publish_focal_seal"](dict(focals_ref=focal_rr,observation_ref=rr("observed")),writer)
finally:np.load=prior_load
orig={}
for n in ast.parse(identity_source).body:
    if(isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=="IDENTITY"for v in n.targets))or(isinstance(n,ast.FunctionDef)and n.name=="identity_sha"):
        exec(compile(ast.Module(body=[n],type_ignores=[]),"original#RAM","exec"),dict(np=np,hashlib=hashlib,json=json),orig)
# identity helper resolves globals from its own compilation namespace.
original_id=dict(np=np,hashlib=hashlib,json=json)
nodes=[n for n in ast.parse(identity_source).body if(isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=="IDENTITY"for v in n.targets))or(isinstance(n,ast.FunctionDef)and n.name=="identity_sha")]
exec(compile(ast.Module(body=nodes,type_ignores=[]),"original_identity#RAM","exec"),original_id)
seal=written["focal_seal.json"]
check(seal["focal_identity_sha256"]==original_id["identity_sha"](records),"Published focal hash equals exact original trainer identity helper")
check(seal["query_truth_opened"]is False and seal["focal_records"]==10000,"Focal seal is label blind all10000records")
check(set(written)=={"focal_seal.json"},"Original native completion remains untouched")


# Real seal_all20 body sees bounded metadata only. Its truth payload handles are never opened.
barrier_ram={};barrier_written={};barrier_events=[];state_calls=[]
allparents=["american_c3_train_%03d"%i for i in range(20)];barrier_rows={};simrefs={};closed_keys={};tails=set()
for parent in allparents:
    obsref=rr(parent+"/observed.json");nativeref=rr(parent+"/native.json");focalref=rr(parent+"/focals.npz")
    barrier_ram[obsref["path"]]=dict(parent_id=parent,sequence_length_bp=30000000,query_ancestry_truth_in_inputs=False,
        genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG")
    barrier_ram[nativeref["path"]]=dict(parent_id=parent,status="completed_native_C3_path_preparation",observation_ref=obsref,
        focals_ref=focalref,query_focal_count=10000,query_ancestry_truth_opened=False)
    row=dict(parent_id=parent,observation_ref=obsref,native_preparation_ref=nativeref,focal_registry_ref=focalref,
        focal_identity_sha256="0"*64,observation_only=True,all_native_trees_nodes_sites_samples_retained=True)
    for stage,key in[("infer","inference_close_accepted_ref"),("native","native_close_accepted_ref")]:
        closeref=rr(parent+"/"+stage+"_closed");receiptref=rr(parent+"/"+stage+"_receipt");tail=parent+stage
        barrier_ram[closeref["path"]]=dict(parent_id=parent,actual_direct_OS_wait0=True,all_owned_processes_and_groups_drained=True,receipt_ref=receiptref)
        barrier_ram[receiptref["path"]]=dict(key=parent+stage,completed_overlay_prefix=dict(tail_sha256=tail))
        row[key]=closeref;closed_keys[parent+stage]={};tails.add(tail)
    barrier_rows[parent]=row;simrefs[parent]=rr(parent+"/simdone")
    barrier_ram[simrefs[parent]["path"]]=dict(parent_id=parent,status="completed_C3_simulation_reconstruction",private_truth_ref=rr("PRIVATE_TRUTH/"+parent))
def barrier_read(r):
    need(not r["path"].startswith("PRIVATE_TRUTH"),"No truth payload boundary")
    barrier_events.append(("metadata",r["path"]));return copy.deepcopy(barrier_ram[r["path"]])
def barrier_put(ctx,name,value):
    barrier_events.append(("publish",name));need(name not in barrier_written,"Exclusive seal")
    barrier_written[name]=value;return rr(name)
def barrier_state(g,m):state_calls.append(1);return ({},closed_keys,{},None,None,tails)
bs=dict(need=need,read=barrier_read,put=barrier_put,state=barrier_state)
extract(REL+"/batch_ROOT_v1.py",["seal_all20"],bs)
bc=SimpleNamespace(g=gg,manifest=manifest,w=SimpleNamespace(canonical_ref_shape=lambda r:r))
readyref,truthref=bs["seal_all20"](bc,barrier_rows,simrefs)
check(len(state_calls)==1,"One fresh original state snapshot authenticates all40close events")
check(barrier_written["all20_training_readiness.json"]["TRAIN_parent_ids"]==allparents,"Readiness exact20order")
check(barrier_written["all20_training_readiness.json"]["parents"]==barrier_rows,"Readiness exact agreed row schema")
check(barrier_written["all20_training_truth_handles.json"]["class_order"]==["EUR","EAS","AFR"],"Truth aggregate exact class order")
check(all(set(v)=={"parent_id","truth_ref"}for v in barrier_written["all20_training_truth_handles.json"]["parents"].values()),"Truth aggregate exact handle rows")
check(barrier_events.index(("publish","all20_training_readiness.json"))<barrier_events.index(("metadata",simrefs[allparents[0]]["path"])),
    "All20readiness published before original truth-handle metadata publication")
check(not any(e[1].startswith("PRIVATE_TRUTH")for e in barrier_events),"Zero private truth payload opens")
for label,change in[
    ("missing20thparent",lambda rows:rows.pop(allparents[-1])),
    ("oracle row",lambda rows:barrier_ram[rows[allparents[1]]["observation_ref"]["path"]].update(genealogy_kind="ORACLE")),
    ("native count9999",lambda rows:barrier_ram[rows[allparents[2]]["native_preparation_ref"]["path"]].update(query_focal_count=9999)),
    ("unaccepted close",lambda rows:closed_keys.pop(allparents[3]+"infer"))]:
    rows=copy.deepcopy(barrier_rows);old_ram=copy.deepcopy(barrier_ram);old_keys=copy.deepcopy(closed_keys);barrier_written.clear()
    change(rows);reject(lambda:bs["seal_all20"](bc,rows,simrefs),"Barrier rejects "+label)
    barrier_ram.clear();barrier_ram.update(old_ram);closed_keys.clear();closed_keys.update(old_keys)
# Exact close acquisition queue is equally strict and never invokes original close after timeout.
close_calls=[]
close_ns=dict(support=lambda:None,read=lambda r:dict(batch_control_input_ref=rr("simcontrol")),
    accounting_transaction=lambda r:Busy(),close_under_coordinator=lambda r:close_calls.append(r))
extract(REL+"/close_unit_ROOT_v1.py",["close_once"],close_ns)
check(close_ns["close_once"](rr("closeauth"))["status"]=="QUEUED_BEFORE_OPERATION"and not close_calls,"Close acquire queue before original invocation")
close_ns["accounting_transaction"]=lambda r:Acquired()
close_ns["close_under_coordinator"]=afteryield
reject(lambda:close_ns["close_once"](rr("closeauth")),"Close post-yield errors never retried")

# Static boundaries make the absence of actual operations explicit.
btext=Path(REL+"/batch_ROOT_v1.py").read_text()
wtext=Path(REL+"/worker_entry_v1.py").read_text()
ctext=Path(REL+"/close_unit_ROOT_v1.py").read_text()
check("science.execute(cfg,writer)"in wtext,"Original unchanged scientific callable remains")
check("publish_focal_seal(read(result),writer)"in wtext,"Bounded label-blind separate focal identity seal")
check("simulator.execute("not in wtext and "torch.optim"not in wtext,"No new simulation or optimization stage")
check("g.apply_close_completed_v1(authority_ref"in ctext,"Original ordinary completed close apply retained")
check("with accounting_transaction(auth["in ctext and "with accounting_transaction(q["in wtext,"Both mutation wrappers serialized")
check("with w.accounting_transaction(a["in btext,"Admin admission serialized in same coordinator")
check("self_probe_v1"in btext and 'd.execute(record)'in wtext,"Original lifecycle executes only future actual ROOT invocation")
check('UI_tool_terminal_claimed=False'in btext,"Future terminal claims direct OS evidence only")
check('truth_payloads_read=False'in btext and 'readiness_ref=barrier'in btext,"Truth handles only after readiness")
result=dict(schema="TreeLoom_preparation_author_AST_RAM_checks_v1",passed=True,check_count=len(checks),checks=checks,
    source_AST_RAM_only=True,actual_guard_imported=False,actual_probes=False,actual_accounting=False,scientific_execution=False)
print(json.dumps(result,sort_keys=True,indent=2))
