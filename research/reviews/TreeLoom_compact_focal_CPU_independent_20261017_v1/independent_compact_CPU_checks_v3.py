"""Narrow independent compact CPU control review. Explicit RAM metadata only."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,re,sys
B=Path("research/reviews/TreeLoom_compact_focal_CPU_continuation_source_20261017_v1")
O=Path("research/reviews/frozen_TRAIN002_019_inference_only_source_20261017_v1")
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
checks=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def need(ok,label):
 if not ok:raise ValueError(label)
def reject(fn,label):
 try:fn()
 except(ValueError,KeyError,TypeError):checks.append(label);return
 raise AssertionError("Invalid control accepted "+label)
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rr(p):return dict(path=str(p),sha256=hashlib.sha256(str(p).encode()).hexdigest())
ram={}
def read(r):
 if r["path"]in ram:return copy.deepcopy(ram[r["path"]])
 need(ref(r["path"])==r,"Immutable known source/metadata hash")
 return json.loads(Path(r["path"]).read_text())
def mr(p):return rr(p)if str(p).startswith("ram/")else ref(p)
def extract(name,file,ns):
 n=next(n for n in trees[file].body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),str(B/file)+"#independentRAM","exec"),ns)
mref=ref(B/"source_manifest.v1.json");m=read(mref)
check(mref["sha256"]=="629e0162c97e9bdd8c590b848617748401b06c8fe1318bc2c2b0a744d4605e55","Exact final compact CPU manifest")
files=["worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py","build_ROOT_authority_v1.py"]
hashes=["859249e3d9b1c1410f69cee0a2018c91165cf2b9d6d431b43796cad185fea05c","00bb947458b397412ddd2cf36ef1fe8394a914193e7087bf8aecf83e647e239e","611ede14cd35c42281f344153e0c6e7e5ea4d7e9be0ac148f02da724cfa8499e","a9b61fb91ef9a3578de2d9e2f16ebef978d2cf3aa285a0803466e93753b45e46"]
trees={}
for name,h in zip(files,hashes):
 check(ref(B/name)["sha256"]==h,"Exact final "+name)
 trees[name]=ast.parse((B/name).read_text())
for r in m["refs"]:check(ref(r["path"])==r,"Full source metadata pin "+r["path"])
expected_changes={
"worker_entry_v1.py":{"initialize","source_paths","direct_roles","validate_completion","stage_status","scientific_config","validate_pipeline_authority","select_parent_from_record","predecessor_capsule","execute_in_admitted_context_v1","worker"},
"batch_ROOT_v1.py":{"qualified","admit_batch","available_original_slots","simulated_candidate","publish_parent_ready","run_batch"},
"close_unit_ROOT_v1.py":{"metadata_join"}}
for name,expected in expected_changes.items():
 old=ast.parse((O/name).read_text());of={n.name:ast.dump(n,include_attributes=False)for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 nf={n.name:ast.dump(n,include_attributes=False)for n in trees[name].body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 changed={k for k in of if k in nf and of[k]!=nf[k]}
 check(changed==expected,"Only declared control functions changed "+name)
 check(set(nf)-set(of)==({"validate_compact_done"}if name=="worker_entry_v1.py"else set())and set(of)-set(nf)==({"publish_focal_seal"}if name=="worker_entry_v1.py"else set()),"Exact added/removed control functions "+name)
for key in["close_pins","coordinator_source_ref","owned_ref","pidfd_ref"]:
 check(m[key]==read(ref(O/"source_manifest.v1.json"))[key],"Inherited original lifetime/accounting pin "+key)
core_review=read(m["compact_component_review_ref"])
check(core_review["independent"]is True and core_review["verdict"]=="SOURCE_PASS"and core_review["blockers"]==[]and core_review["source_manifest_ref"]==m["compact_component_manifest_ref"],"Actual compact core independent SOURCE_PASS inherited")
cap=m["condition"]["stage_caps"]["compact_focals"]
check(cap==dict(CPU_threads=8,aux_bytes=16*2**20,cleanup_seconds=30,log_bytes=4*2**20,private_work_bytes=2**20,reservation_bytes=18*2**20,rss_bytes=64*2**30,terminal_evidence_margin_seconds=150,wall_seconds=7200,work_cutoff_margin_seconds=300),"Exact whole18MiB and finite original7200second64GiB work caps")
check(2**21+cap["aux_bytes"]==cap["reservation_bytes"]and cap["log_bytes"]+cap["private_work_bytes"]+11*2**20==cap["aux_bytes"],"Whole primary plus logs/private/metadata/terminal allocation adds exactly")
check(m["admin_unique_union"]==m["scientific_unit_unique_union"]==64 and m["direct_unit_role_count"]==25,"Final declared binding ceiling remains64, directroles25")
# Exact candidate constants and narrow pure metadata functions.
wn=dict(need=need,read=read,metadata_ref=mr,local=lambda p:ROOT/p,ROOT=ROOT,Path=Path,datetime=datetime,timezone=timezone,sys=sys)
for n in trees["worker_entry_v1.py"].body:
 if isinstance(n,ast.Assign)and all(isinstance(v,ast.Name)for v in n.targets):
  if not any(v.id in["context"]for v in n.targets):
   try:exec(compile(ast.Module(body=[n],type_ignores=[]),"worker-constants#RAM","exec"),wn)
   except NameError:pass
for name in["stage_status","registered_row","scientific_config","validate_pipeline_authority","validate_compact_done","predecessor_capsule","observation_metadata","pathlib_packet"]:
 extract(name,"worker_entry_v1.py",wn)
now=datetime.now(timezone.utc)
control=rr("ram/control");sim_auth=rr("ram/sim_authority")
ram[control["path"]]=dict(authority_ref=sim_auth);ram[sim_auth["path"]]=dict(expires_at_utc=(now+timedelta(days=2)).isoformat())
wn["coordinator_support"]=lambda:SimpleNamespace(validate_batch_authority=lambda r:need(r==sim_auth,"Exact RAM original coordinator authority"))
check(wn["stage_status"]("compact_focals")==wn["COMPACT_STATUS"],"Compact stage has honest distinct status")
for bad in["infer","native_paths"]:reject(lambda bad=bad:wn["stage_status"](bad),"Excluded stage rejects "+bad)
for idx in[1,7,19]:
 p="american_c3_train_%03d"%idx;cfg=wn["scientific_config"](p,"compact_focals",rr("ram/observed"))
 check(cfg["path_sampling_seed"]==92001+idx and cfg["query_focal_count"]==10000 and cfg["output_cap_bytes"]==2**21 and cfg["execution_enabled"]is True,"Exact original config extractor "+p)
reject(lambda:wn["registered_row"]("american_c3_train_000"),"Retained TRAIN000 never compactly re-executed")
# The final emitted review shape is used verbatim by both builder and qualified.
review=dict(schema="TreeLoom_compact_focal_CPU_independent_source_review_v1",independent=True,verdict="SOURCE_PASS",blockers=[],source_only=True,operational_authority=False,source_ref=ref(B/"worker_entry_v1.py"),candidate_manifest_ref=mref,source_manifest_ref=mref,reviewed_sources=[ref(B/n)for n in files],reviewed_source_refs=[ref(B/n)for n in files],scope="Narrow compact CPU operational delta. Inherits core, original supervisor/coordinator/ordinary close. RAM evidence only.",actual_operations=False,scientific_execution=False)
reviewref=rr("ram/final_emitted_review");ram[reviewref["path"]]=review
template=read(ref(B/"ROOT_authority.template.INACTIVE.v1.json"));a=copy.deepcopy(template)
a.update(created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(days=1)).isoformat(),active=True,source_only=False,root_authorizer="root_agent",source_manifest_ref=mref,source_review_ref=reviewref,scope_id="independent_ram",CPU_lanes=[list(range(8)),list(range(8,16))],simulation_batch_control_input_ref=control,all_other_compact_controller_routes_revoked=True,run_directory=str(B)+"/ROOT_actual_pipeline_independent_ram",admission_directory=str(B)+"/ROOT_pipeline_admission_independent_ram")
a["slot_directory"]=a["run_directory"]+"/producer_slots";a["all20_adapter_output_directory"]=a["run_directory"]+"/all20_ready_control"
ar=rr("ram/authority");ram[ar["path"]]=a
check(wn["validate_pipeline_authority"](ar)==a,"Final explicit compact authority validates")
for key,bad in[("stages",["native_paths"]),("maximum_producers",3),("stage_reservation_bytes",dict(compact_focals=16*2**20)),("all20_adapter_output_cap_bytes",262145),("dedicated_compact_producer_slots",False),("all_other_compact_controller_routes_revoked",False),("slot_directory","ram/outside_slots"),("all20_adapter_output_directory","ram/outside_ready"),("GPU_authorized",True),("query_truth_access_authorized",True),("automatic_retry",True)]:
 bad_a=copy.deepcopy(a);bad_a[key]=bad;ram[ar["path"]]=bad_a
 reject(lambda:wn["validate_pipeline_authority"](ar),"Reject expanded compact authority "+key)
ram[ar["path"]]=a
# Actual initializer, but only inert helper module objects and RAM metadata. No slot call.
fakec=ModuleType("independent_inert_contract");faked=ModuleType("independent_inert_driver")
fakec.read=read;fakec.ref=ref;fakec.PIDFD=rr("ram/pidfd");fakec.OWNED=rr("ram/owned");fakec.contextlib=SimpleNamespace(contextmanager=lambda f:f)
fakec.local=lambda p:ROOT/p;fakec.require=need;fakec.load=lambda *x:None;fakec.put=lambda *x:None;fakec.current=lambda *x:None
initns=dict(wn);initns.update(c=None,d=None,_selected_authority_ref=ar,PARENT="american_c3_train_001",STAGE="compact_focals",PHASE="compact_focals_train_001",SLOT_REL=a["slot_directory"],guard=lambda *x:None,_memo=None)
for name in["source_paths","validate_request","phase_paths","direct_roles","input_spec","validate_completion"]:initns.setdefault(name,lambda *x:None)
loads=[]
def inert_load(path,sha,name):
 loads.append((path,sha));return fakec if path.endswith("root_contract_v1.py")else faked
initns["_load_initial"]=inert_load
extract("initialize","worker_entry_v1.py",initns)
c,d=initns["initialize"]()
check(c is fakec and d is faked and d.c is fakec,"Actual initializer wires original inert helpers")
check(any(type(v)is str and v.startswith(a["slot_directory"]+"/cpu_slot_")for v in c.cpu_slot.__code__.co_consts),"Actual original slot AST receives exact dedicated charged path")
check(len(loads)==2 and loads[0][1]==wn["OLD_CONTRACT_SHA"]and loads[1][1]==wn["OLD_DRIVER_SHA"],"Actual initializer retains exact original helper pins")
check(c.STAGES==("compact_focals",)and c.STATUSES[initns["PHASE"]]==wn["COMPACT_STATUS"],"Actual initializer consumes honest compact stage condition")
# Compact completion and original-format seal control validator.
parent="american_c3_train_007";ob=rr("ram/observed");sealref=rr("ram/seal");focals=rr("ram/focals.npz");identity="a"*64
done=dict(schema="TreeLoom_compact_original_focal_preparation_completion_v1",status=wn["COMPACT_STATUS"],parent_id=parent,observation_ref=ob,genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",class_order=["EUR","EAS","AFR"],query_focal_count=10000,path_sampling_seed=92008,full_original_path_traversal=True,reference_paths_counted_in_global_record_ids=True,native_corpus_materialized=False,native_dataset_files_written=0,native_vocabulary_materialized=False,native_train_val_split_executed=False,sequence_to_tree_occurrence_mapping_materialized=False,query_ancestry_truth_opened=False,focal_seal_ref=sealref,focals_ref=focals,focal_identity_sha256=identity)
ram[sealref["path"]]=dict(schema="TreeLoom_original_native_focal_identity_seal_v1",parent_id=parent,observation_ref=ob,focal_registry_ref=focals,focal_identity_sha256=identity,focal_records=10000,query_truth_opened=False,preparation_kind="compact_original_identity_focals_without_native_corpus")
check(wn["validate_compact_done"](done,parent,ob)==done,"Honest compact completion and exact original-format seal accept")
for key,bad in[("status","completed_native_C3_path_preparation"),("native_corpus_materialized",True),("query_focal_count",9999),("path_sampling_seed",92009),("native_dataset_refs",{}),("query_ancestry_truth_opened",True),("full_original_path_traversal",False)]:
 value=copy.deepcopy(done);value[key]=bad
 reject(lambda value=value:wn["validate_compact_done"](value,parent,ob),"Invalid compact completion rejects "+key)
oldid=ram[sealref["path"]]["focal_identity_sha256"];ram[sealref["path"]]["focal_identity_sha256"]="b"*64
reject(lambda:wn["validate_compact_done"](done,parent,ob),"Changed original focal identity seal rejects")
ram[sealref["path"]]["focal_identity_sha256"]=oldid
# Actual builder metadata path with independent explicit RAM TRAIN001 provenance.
bn=dict(Path=Path,datetime=datetime,timezone=timezone,json=json,re=re,need=need,read=read,ref=mr,REL=str(B),MANIFEST=str(B/"source_manifest.v1.json"))
for n in trees["build_ROOT_authority_v1.py"].body:
 if isinstance(n,ast.Assign)and all(isinstance(v,ast.Name)for v in n.targets):
  exec(compile(ast.Module(body=[n],type_ignores=[]),"builder-constants#RAM","exec"),bn)
bn.update(BATCH=control,TRAIN001_INPUT=rr("ram/TRAIN001_input"),TRAIN001_SUPERVISOR=rr("ram/TRAIN001_supervisor"))
inp_ref=bn["TRAIN001_INPUT"];sw_ref=bn["TRAIN001_SUPERVISOR"]
req_ref=rr("ram/TRAIN001_request");dr=rr("ram/TRAIN001_out/completed.json");wr=rr("ram/TRAIN001_aux/actual_wait.json")
out=rr("ram/TRAIN001_outer");rev=rr("ram/TRAIN001_rev");closed=rr("ram/TRAIN001_closed");receipt=rr("ram/TRAIN001_receipt");obs=rr("ram/TRAIN001_observed");simdone=rr("ram/TRAIN001_simdone")
ram[inp_ref["path"]]=dict(parent_id="american_c3_train_001",stage="infer",root_stage_request_ref=req_ref,output_directory="ram/TRAIN001_out",worker_auxiliary_scope="ram/TRAIN001_aux",refs=dict(config=rr("ram/TRAIN001_cfg")),storage=dict(reservation=dict(key="RAM/original_infer")),root_driver_ref=rr("ram/original_infer_source"))
ram[req_ref["path"]]=dict(prior=dict(completion_ref=simdone),source_manifest_ref=rr("ram/original_infer_manifest"))
ram[dr["path"]]=dict(status="completed_current_C3_inference",observation_ref=obs)
ram[wr["path"]]=dict(parent_id="american_c3_train_001",input_ref=inp_ref,completed_ref=dr,actual_direct_worker_wait=True,actual_direct_launcher_wait=True,actual_OS_wait_status=0,returncode=0,all_owned_processes_and_groups_drained=True,operational_failure=None,cleanup_errors=[],adopted_waits=[],supervisor_pid=7654)
ram[sw_ref["path"]]=dict(actual_direct_wait=True,actual_returncode=0,pid=7654,input_ref=inp_ref)
ram[out["path"]]=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,input_ref=inp_ref,completed_ref=dr,actual_worker_wait_ref=wr,actual_supervisor_wait_ref=sw_ref,all_owned_processes_and_groups_drained=True)
ram[rev["path"]]=dict(input_ref=inp_ref,direct_wait_ref=wr,completed_ref=dr,actual_outer_terminal_ref=out,all_original_launch_resume_retry_write_routes_permanently_revoked=True)
ram[closed["path"]]=dict(parent_id="american_c3_train_001",actual_direct_OS_wait0=True,all_owned_processes_and_groups_drained=True,receipt_ref=receipt)
ram[receipt["path"]]=dict(key="RAM/original_infer")
ram[obs["path"]]=dict(parent_id="american_c3_train_001",sequence_length_bp=30000000,query_ancestry_truth_in_inputs=False,roster_ref=rr("ram/roster"))
ram[simdone["path"]]=dict(parent_id="american_c3_train_001",status="completed_C3_simulation_reconstruction")
inferadmin=rr("ram/infer_admin");inferauth=rr("ram/infer_auth")
ram[inferadmin["path"]]=dict(authority_ref=inferauth,source_manifest_ref=bn["INFERENCE_MANIFEST"])
ram[inferauth["path"]]=dict(source_manifest_ref=bn["INFERENCE_MANIFEST"],simulation_batch_control_input_ref=control,stages=["infer"],inference_only=True,run_directory="ram/infer_run")
params=dict(schema="ROOT_compact_focal_CPU_explicit_build_parameters_v1",created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(minutes=10)).isoformat(),root_authorizer="root_agent",active=True,build_only=True,admission_or_execution_authorized=False,operational_source_review_ref=reviewref,simulation_batch_control_input_ref=control,inference_only_admin_input_ref=inferadmin,CPU_lanes=a["CPU_lanes"],actual_initial_capacity=dict(created_at_utc=now.isoformat(),root_authorizer="root_agent",actual_current_observation=True,source_only_estimate=False,admissible_new_bytes=2**30),scope_id="independent_ram",controller_expires_at_utc=a["expires_at_utc"],all_other_compact_controller_routes_revoked=True,TRAIN001_inference_input_ref=inp_ref,TRAIN001_inference_outer_terminal_ref=out,TRAIN001_inference_permanent_revocation_ref=rev,TRAIN001_inference_close_accepted_ref=closed)
pr=rr("ram/build_parameters");ram[pr["path"]]=params
for name in["train001_ready","build_values"]:extract(name,"build_ROOT_authority_v1.py",bn)
p,namespace,ready,built=bn["build_values"](pr)
check(ready["input_ref"]==inp_ref and ready["actual_original_worker_and_supervisor_OS_wait0"]is True,"Actual builder authenticates explicit RAM original TRAIN001 wait/close provenance")
check(built["source_review_ref"]==reviewref and built["source_manifest_ref"]==mref and built["slot_directory"]==built["run_directory"]+"/producer_slots","Actual builder emits exact review/manifest/dedicated slot joins")
for where,key,bad in[(ram[wr["path"]],"returncode",1),(ram[sw_ref["path"]],"actual_returncode",1),(ram[out["path"]],"actual_terminal_consumed",False),(ram[rev["path"]],"all_original_launch_resume_retry_write_routes_permanently_revoked",False),(ram[receipt["path"]],"key","RAM/wrong")]:
 old=copy.deepcopy(where[key]);where[key]=bad
 reject(lambda:bn["build_values"](pr),"Builder rejects changed genuine provenance "+key);where[key]=old
badreview=copy.deepcopy(review);badreview["reviewed_sources"]=review["reviewed_sources"][:-1];ram[reviewref["path"]]=badreview
reject(lambda:bn["build_values"](pr),"Builder rejects missing fourth reviewed source")
ram[reviewref["path"]]=review
# Exact qualified consumes the very same review dictionary intended for sealing.
built["TRAIN001_inference_ready_ref"]=rr("ram/builtTRAIN001ready");ram[ar["path"]]=built
qn=dict(read=read,ref=mr,raw=lambda r:need(mr(r["path"])==r,"Exact source hash"),load=lambda r,name:SimpleNamespace(validate_pipeline_authority=wn["validate_pipeline_authority"]),need=need,Path=Path,ROOT=ROOT,sys=sys,PYTHON=ROOT/".runtime/env/bin/python",REL=str(B),MANIFEST=str(B/"source_manifest.v1.json"),WORKER=str(B/"worker_entry_v1.py"),SOURCE=str(B/"batch_ROOT_v1.py"),CLOSE=str(B/"close_unit_ROOT_v1.py"),local=lambda p:ROOT/p,os=SimpleNamespace(sched_getaffinity=lambda pid:set(range(16))))
extract("qualified","batch_ROOT_v1.py",qn)
returned=qn["qualified"](ar)
check(returned[1]==built,"Exact qualified accepts actual emitted final-review shape and builder authority in RAM")
ram[reviewref["path"]]=badreview
reject(lambda:qn["qualified"](ar),"Exact qualified rejects incomplete final-review shape")
ram[reviewref["path"]]=review
run=next(n for n in trees["batch_ROOT_v1.py"].body if isinstance(n,ast.FunctionDef)and n.name=="run_batch")
rt=ast.unparse(run)
check("while len(rows) < 19"in rt and "len(completed) == len(attempted) == 19"in rt,"Exactly19compact units, retained000 not repeated")
check("validate_pipeline_authority"not in rt,"Empty frontier loop never replays full authority validation")
while_node=next(n for n in ast.walk(run)if isinstance(n,ast.While))
early=while_node.body[:3]
check(not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in["state","guard","qualified"]for x in early for n in ast.walk(x)),"Absent-frontier loop checks cached clock and known paths only")
check(rt.index("accepted = accept_close")<rt.index("publish_parent_ready"),"Compact frontier published only after genuine inherited accepted close")
dispatch=ast.unparse(next(n for n in trees["worker_entry_v1.py"].body if isinstance(n,ast.FunctionDef)and n.name=="execute_in_admitted_context_v1"))
check("science = c.load(CORE_REF"in dispatch and "science.execute(cfg, writer)"in dispatch and "american_c3_paths_v1 as science"not in dispatch,"Only exact reviewed compact component executes, never full native corpus")
result=dict(schema="TreeLoom_compact_CPU_independent_narrow_checks_v1",passed=True,check_count=len(checks),checks=checks,author_fixtures_reused=False,exact_initializer_with_inert_helpers=True,exact_builder_and_qualified_final_review_shape_in_RAM=True,actual_guard_imported=False,actual_operations=False,actual_CPU_affinity_slot_lock_GPU_resource_probes=False,scientific_execution=False,actual_runtime_success_claimed=False)
print(json.dumps(dict(result=result,final_review_shape=review),sort_keys=True,indent=2))
