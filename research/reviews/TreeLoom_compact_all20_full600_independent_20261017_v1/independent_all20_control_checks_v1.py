"""Independent narrow compact all20/full600 metadata adapter review."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,re
B=Path("research/reviews/TreeLoom_compact_all20_full600_control_source_20261017_v1")
O=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1")
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
 raise AssertionError("Invalid metadata accepted "+label)
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rr(p):return dict(path=str(p),sha256=hashlib.sha256(str(p).encode()).hexdigest())
ram={};reads=[]
def read(r):
 reads.append(r["path"])
 if r["path"]in ram:return copy.deepcopy(ram[r["path"]])
 need(ref(r["path"])==r,"Immutable metadata SHA")
 return json.loads(Path(r["path"]).read_text())
def mr(p):return rr(p)if str(p).startswith("ram/")else ref(p)
mref=ref(B/"source_manifest.v1.json");m=read(mref)
check(mref["sha256"]=="4fff9b0b0aef5533caf1db8f48c4a9b8c51f592c97a85f2fed7b6223f3e367ed","Exact sealed all20/full600 manifest")
expected={"readiness_ROOT_v1.py":"4d196298b6f2b9a4892b846cba52c9ba38925e6926a1fe89cbbdb3deeff4d6f8","full600_ROOT_compact_continuation_v1.py":"c138e2e00b9be29d74cbec1a5a6c293d3ccc965198d907eb10137b3aaaad2ba7"}
trees={}
for name,h in expected.items():
 check(ref(B/name)["sha256"]==h,"Exact sealed "+name);trees[name]=ast.parse((B/name).read_text())
for key in["inherited_owner4_control_review_ref","inherited_original_full600_review_ref"]:
 prior=read(m[key]);check(prior["independent"]is True and prior["verdict"]=="SOURCE_PASS"and prior["blockers"]==[],"Inherited independent SOURCE_PASS "+key)
prior=ast.parse((O/"full600_ROOT_continuation_owner4_v1.py").read_text())
of={n.name:n for n in prior.body if isinstance(n,ast.FunctionDef)}
nf={n.name:n for n in trees["full600_ROOT_compact_continuation_v1.py"].body if isinstance(n,ast.FunctionDef)}
check(set(of)==set(nf),"No added or removed full600 control functions")
changed={k for k in of if ast.dump(of[k],include_attributes=False)!=ast.dump(nf[k],include_attributes=False)}
check(changed=={"_request","_all20","_admit","continue_v1"},"Only four declared full600 operational functions changed")
for name in set(of)-changed:check(ast.dump(of[name],include_attributes=False)==ast.dump(nf[name],include_attributes=False),"Inherited full600 gate unchanged "+name)
newadmit=copy.deepcopy(nf["_admit"])
for n in ast.walk(newadmit):
 if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="dict":
  n.keywords=[k for k in n.keywords if k.arg not in["compact_control_source_review_ref","all20_metadata_builder_parameters_ref","all20_compact_control_manifest_ref"]]
check(ast.dump(newadmit,include_attributes=False)==ast.dump(of["_admit"],include_attributes=False),"Original admission and sole scientific parent argv unchanged outside three existing-bundle fields")
def extract(name,file,ns):
 n=next(n for n in trees[file].body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),str(B/file)+"#independentRAM","exec"),ns)
# Original compact config helper is extracted only, no arrays/models/science.
core=Path("research/reviews/TreeLoom_compact_original_focal_source_20261017_v1/compact_focals_v1.py")
ca=ast.parse(core.read_text());cn=dict(read=read,require=need)
names={"TRAIN","GENEALOGY","ORIGINAL_LIMITS","FOCALS","PRIMARY_OUTPUT_CAP_BYTES","CONFIG_SCHEMA","ORIGINAL_PIPELINE_MANIFEST","STATUS","COMPLETION_SCHEMA"}
for n in ca.body:
 if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in names for t in n.targets):
  exec(compile(ast.Module(body=[n],type_ignores=[]),"core-constant#RAM","exec"),cn)
n=next(n for n in ca.body if isinstance(n,ast.FunctionDef)and n.name=="make_config_v1")
exec(compile(ast.Module(body=[n],type_ignores=[]),"exact-core-config#RAM","exec"),cn)
focal=SimpleNamespace(**{k:cn[k]for k in names if k in cn},CLASSES=("EUR","EAS","AFR"),make_config_v1=cn["make_config_v1"])
def raw(r):need(mr(r["path"])==r,"Declared known source/metadata pin")
c=SimpleNamespace(require=need,read=read,ref=mr,raw=raw,local=lambda p:ROOT/p,ROOT=ROOT,MANIFEST=str(O/"source_manifest.v1.json"),active=lambda v:need(datetime.fromisoformat(v["expires_at_utc"])>datetime.now(timezone.utc),"Explicit current RAM authority"))
rn=dict(c=c,focal=focal,Path=Path,datetime=datetime,timezone=timezone,json=json,re=re)
for n in trees["readiness_ROOT_v1.py"].body:
 if isinstance(n,ast.Assign)and all(isinstance(t,ast.Name)for t in n.targets):
  exec(compile(ast.Module(body=[n],type_ignores=[]),"readiness-constants#RAM","exec"),rn)
for name in["short_ref_v1","source_pass_v1","closed_v1","observation_v1","simulation_v1","compact_parent_v1","native000_v1","parameters_v1","build_values","verify_values_v1"]:
 extract(name,"readiness_ROOT_v1.py",rn)
parents=rn["PARENTS"]
review=dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_only=True,operational_authority=False,candidate_manifest_ref=mref,source_manifest_ref=mref,reviewed_sources=[ref(B/n)for n in expected],actual_operations=False,scientific_execution=False)
reviewref=rr("ram/final_adapter_review");ram[reviewref["path"]]=review
producer_manifest=m["compact_CPU_manifest_ref"];producer=read(producer_manifest)
producer_review=dict(path="research/reviews/TreeLoom_compact_focal_CPU_independent_20261017_v1/review.compact_CPU.v1.json",sha256="017ac88d680837c6b2aa92e95539a10024db9f50920adc0f4591325c751371ab")
roster_rows=[dict(sample_index=i,role="pure_source_reference"if i<600 else"query",observed_reference_class=2 if i<200 else0 if i<400 else1 if i<600 else None)for i in range(1600)]
frontier_refs={};simulation_refs={};fixture={}
for index,parent in enumerate(parents):
 prefix="ram/"+parent
 ob=rr(prefix+"/observation.json");roster=rr(prefix+"/roster.json");prep=rr(prefix+"/output/completed.json");focals=rr(prefix+"/focals.npz");seal=rr(prefix+"/seal.json")
 sim=rr(prefix+"/simulation.json");private=rr(prefix+"/truth/local.npz");identity=hashlib.sha256(parent.encode()).hexdigest()
 ram[ob["path"]]=dict(schema="american_c3_observed_v1",parent_id=parent,split="train",sequence_length_bp=30000000,genealogy_kind=focal.GENEALOGY,class_order=list(focal.CLASSES),query_ancestry_truth_in_inputs=False,sites_and_sample_identities_verified=True,tree_ref=rr(prefix+"/observed.trees"),genotypes_ref=rr(prefix+"/genotypes.npz"),positions_ref=rr(prefix+"/positions.npz"),alleles_ref=rr(prefix+"/alleles.json"),roster_ref=roster)
 ram[roster["path"]]=dict(parent_id=parent,class_order=list(focal.CLASSES),samples=copy.deepcopy(roster_rows))
 ram[sim["path"]]=dict(parent_id=parent,status="completed_C3_simulation_reconstruction",private_truth_ref=private);simulation_refs[parent]=sim
 inferinput=rr(prefix+"/infer_input.json");inferclose=rr(prefix+"/infer_close.json");inferreceipt=rr(prefix+"/infer_receipt.json")
 ram[inferinput["path"]]=dict(parent_id=parent,stage="infer",storage=dict(reservation=dict(key=prefix+"/infer")))
 ram[inferclose["path"]]=dict(parent_id=parent,actual_direct_OS_wait0=True,all_owned_processes_and_groups_drained=True,receipt_ref=inferreceipt)
 if index!=1:ram[inferclose["path"]]["stage"]="infer"
 ram[inferreceipt["path"]]=dict(key=prefix+"/infer")
 compactclose=rr(prefix+"/focal_close.json");focalreceipt=rr(prefix+"/focal_receipt.json")
 ram[compactclose["path"]]=dict(parent_id=parent,stage="compact_focals",actual_direct_OS_wait0=True,all_owned_processes_and_groups_drained=True,receipt_ref=focalreceipt)
 ram[focalreceipt["path"]]=dict(key=prefix+"/compact")
 if index==0:
  ram[prep["path"]]=dict(status="completed_native_C3_path_preparation",parent_id=parent,observation_ref=ob,focals_ref=focals,query_focal_count=10000,genealogy_kind=focal.GENEALOGY,class_order=list(focal.CLASSES),query_ancestry_truth_opened=False)
  metadata=rr(prefix+"/metadata.json");ram[metadata["path"]]=dict(parent_id=parent,observation_ref=ob,native_preparation_ref=prep,focal_registry_ref=focals,focal_identity_sha256=identity,observation_only=True,all_native_trees_nodes_sites_samples_retained=True)
  fixture[parent]=dict(ob=ob,prep=prep,focals=focals,metadata=metadata,inferclose=inferclose,focalclose=compactclose)
  continue
 inp=rr(prefix+"/input.json");config=rr(prefix+"/config.json");request=rr(prefix+"/request.json");wait=rr(prefix+"/wait.json");supervisor=rr(prefix+"/supervisor.json");outer=rr(prefix+"/outer.json");rev=rr(prefix+"/revocation.json");inferready=rr(prefix+"/infer_ready.json")
 cfg=focal.make_config_v1(parent,ob,execution_enabled=True);ram[config["path"]]=cfg
 ram[request["path"]]=dict(source_manifest_ref=producer_manifest,source_review_ref=producer_review,source_sha256={r["path"]:r["sha256"]for r in producer["refs"]})
 ram[inp["path"]]=dict(parent_id=parent,stage="compact_focals",phase="compact_focals_"+parent.removeprefix("american_c3_"),GPU_authorized=False,query_truth_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False,root_stage_request_ref=request,source_sha256=ram[request["path"]]["source_sha256"],refs=dict(config=config),output_directory=prefix+"/output",storage=dict(reservation=dict(key=prefix+"/compact")))
 ram[prep["path"]]=dict(schema=focal.COMPLETION_SCHEMA,status=focal.STATUS,parent_id=parent,observation_ref=ob,genealogy_kind=focal.GENEALOGY,class_order=list(focal.CLASSES),focals_ref=focals,focal_seal_ref=seal,focal_identity_sha256=identity,query_focal_count=10000,path_sampling_seed=92001+index,full_original_path_traversal=True,reference_paths_counted_in_global_record_ids=True,native_corpus_materialized=False,native_dataset_files_written=0,native_vocabulary_materialized=False,native_train_val_split_executed=False,sequence_to_tree_occurrence_mapping_materialized=False,query_ancestry_truth_opened=False)
 ram[seal["path"]]=dict(schema="TreeLoom_original_native_focal_identity_seal_v1",parent_id=parent,observation_ref=ob,focal_registry_ref=focals,focal_identity_sha256=identity,focal_records=10000,query_truth_opened=False)
 ram[wait["path"]]=dict(schema="American_C3_full_original_CPU_actual_wait_v1",parent_id=parent,phase=ram[inp["path"]]["phase"],input_ref=inp,completed_ref=prep,actual_direct_worker_wait=True,actual_direct_launcher_wait=True,actual_OS_wait_status=0,returncode=0,all_owned_processes_and_groups_drained=True,cleanup_errors=[],operational_failure=None,adopted_waits=[],supervisor_pid=7000+index)
 ram[supervisor["path"]]=dict(actual_direct_wait=True,actual_OS_wait_status=0,actual_returncode=0,all_owned_processes_and_groups_drained=True,pid=7000+index,input_ref=inp,completed_ref=prep,direct_worker_wait_ref=wait)
 ram[outer["path"]]=dict(schema="ROOT_batch_unit_actual_OS_terminal_v1",root_authorizer="root_agent",actual_supervisor_OS_terminal_consumed=True,actual_exit_code=0,UI_tool_terminal_claimed=False,actual_worker_wait_ref=wait,actual_supervisor_wait_ref=supervisor,input_ref=inp,completed_ref=prep,all_owned_processes_and_groups_drained=True)
 ram[rev["path"]]=dict(input_ref=inp,completed_ref=prep,direct_wait_ref=wait,actual_outer_terminal_ref=outer,all_original_launch_resume_retry_write_routes_permanently_revoked=True)
 ram[inferready["path"]]=dict(parent_id=parent,observation_ref=ob,inference_close_accepted_ref=inferclose,simulation_completion_ref=sim,query_truth_opened=False,output_frozen=True,input_ref=inferinput)
 readyref=rr(prefix+"/ready.json");frontier_refs[parent]=readyref
 ram[readyref["path"]]=dict(schema="TreeLoom_compact_focal_parent_ready_v1",parent_id=parent,focal_preparation_kind=rn["COMPACT"],query_truth_opened=False,output_frozen=True,source_manifest_ref=producer_manifest,compact_input_ref=inp,compact_config_ref=config,observation_ref=ob,compact_completed_ref=prep,native_preparation_ref=prep,focal_registry_ref=focals,focal_seal_ref=seal,focal_identity_sha256=identity,direct_wait_ref=wait,actual_supervisor_wait_ref=supervisor,actual_outer_terminal_ref=outer,permanent_revocation_ref=rev,focal_preparation_close_accepted_ref=compactclose,inference_ready_ref=inferready,inference_close_accepted_ref=inferclose,simulation_completion_ref=sim)
 fixture[parent]=dict(ob=ob,prep=prep,focals=focals,seal=seal,wait=wait,supervisor=supervisor,outer=outer,rev=rev,close=compactclose,request=request,roster=roster,ready=readyref)
now=datetime.now(timezone.utc)
p=dict(schema="ROOT_TreeLoom_compact_all20_build_parameters_v1",root_authorizer="root_agent",active=True,build_only=True,admission_or_execution_authorized=False,created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(minutes=10)).isoformat(),source_manifest_ref=mref,source_review_ref=reviewref,compact_preparation_manifest_ref=producer_manifest,compact_preparation_source_review_ref=producer_review,compact_parent_ready_refs=frontier_refs,TRAIN000_preparation_metadata_ref=fixture[parents[0]]["metadata"],TRAIN000_inference_close_accepted_ref=fixture[parents[0]]["inferclose"],TRAIN000_native_close_accepted_ref=fixture[parents[0]]["focalclose"],TRAIN000_simulation_completion_ref=simulation_refs[parents[0]])
pr=rr("ram/all20_parameters.json");ram[pr["path"]]=p
returned,ready,truth=rn["build_values"](pr)
check(list(ready["parents"])==parents and ready["TRAIN_parent_ids"]==parents,"Exact ordered20 independent TRAIN parents")
check(ready["parents"][parents[0]]["focal_preparation_kind"]==rn["NATIVE"]and all(ready["parents"][v]["focal_preparation_kind"]==rn["COMPACT"]for v in parents[1:]),"Retained native000 and19honest compact preparations")
check(ready["all20_complete_observation_metadata_authenticated"]is True and ready["compact_native_corpus_claimed"]is False,"Metadata barrier honest about native corpus")
check(ready["original_barrier_semantics"]=="all20_metadata_before_this_full_training_truth;whole_parent_encode_before_own_target_join","Exact original barrier semantics, no extra encoding pass")
check(truth["truth_payloads_read"]is False and all(truth["parents"][v]["truth_ref"]==ram[simulation_refs[v]["path"]]["private_truth_ref"]for v in parents),"Original TRAIN truth handles joined without opening")
rrdy=rr("ram/ready_all20.json");tr=rr("ram/truth_all20.json");truth=dict(truth,readiness_ref=rrdy)
check(rn["verify_values_v1"](rrdy,ready,tr,truth)==p,"Exact consumer reauthenticates producer builder output")
chosen=fixture[parents[3]]
for which,key,bad in[
("prep","status","completed_native_C3_path_preparation"),("prep","native_corpus_materialized",True),("prep","query_focal_count",9999),
("prep","path_sampling_seed",123),("prep","query_ancestry_truth_opened",True),("prep","native_dataset_refs",{}),
("seal","focal_identity_sha256","f"*64),("wait","returncode",1),("supervisor","actual_returncode",1),
("outer","actual_supervisor_OS_terminal_consumed",False),("rev","all_original_launch_resume_retry_write_routes_permanently_revoked",False),
("close","stage","native_paths"),("ob","sequence_length_bp",2000000)]:
 doc=ram[chosen[which]["path"]];had=key in doc;old=copy.deepcopy(doc.get(key));doc[key]=bad
 reject(lambda:rn["build_values"](pr),"Changed compact provenance rejects "+which+"."+key)
 if had:doc[key]=old
 else:del doc[key]
doc=ram[chosen["close"]["path"]];old=doc.pop("stage")
reject(lambda:rn["build_values"](pr),"New compact close cannot omit exact stage")
doc["stage"]=old
old=ram[chosen["request"]["path"]]["source_sha256"];ram[chosen["request"]["path"]]["source_sha256"]={}
reject(lambda:rn["build_values"](pr),"Incomplete producer source closure rejects")
ram[chosen["request"]["path"]]["source_sha256"]=old
badroles=ram[chosen["roster"]["path"]]["samples"];old=badroles[600]["observed_reference_class"];badroles[600]["observed_reference_class"]=0
reject(lambda:rn["build_values"](pr),"Query ancestry label entering observed roster rejects")
badroles[600]["observed_reference_class"]=old
old=ram[simulation_refs[parents[2]]["path"]]["private_truth_ref"];ram[simulation_refs[parents[2]]["path"]]["private_truth_ref"]=rr("ram/test_000/truth/local.npz")
reject(lambda:rn["build_values"](pr),"NonTRAIN truth handle rejects")
ram[simulation_refs[parents[2]]["path"]]["private_truth_ref"]=old
old=ram[pr["path"]]["compact_parent_ready_refs"];ram[pr["path"]]["compact_parent_ready_refs"]=dict(list(old.items())[:-1])
reject(lambda:rn["build_values"](pr),"Missing nineteenth compact frontier rejects")
ram[pr["path"]]["compact_parent_ready_refs"]=old
badready=copy.deepcopy(ready);badready["compact_native_corpus_claimed"]=True
reject(lambda:rn["verify_values_v1"](rrdy,badready,tr,truth),"Modified emitted readiness rejects exact consumer rejoin")
badtruth=copy.deepcopy(truth);badtruth["parents"][parents[2]]["truth_ref"]=rr("ram/wrong/truth/local.npz")
reject(lambda:rn["verify_values_v1"](rrdy,ready,tr,badtruth),"Modified truth handle aggregate rejects")
# Existing immutable build outputs can be checked after build directive expiry, without authorizing new writes.
oldtime=(ram[pr["path"]]["created_at_utc"],ram[pr["path"]]["expires_at_utc"])
ram[pr["path"]]["created_at_utc"]=(now-timedelta(hours=1)).isoformat();ram[pr["path"]]["expires_at_utc"]=(now-timedelta(minutes=50)).isoformat()
reject(lambda:rn["build_values"](pr),"Expired directive cannot produce fresh readiness")
check(rn["build_values"](pr,require_active=False)[1]==ready,"Immutable existing readiness revalidation survives build directive expiry")
ram[pr["path"]]["created_at_utc"],ram[pr["path"]]["expires_at_utc"]=oldtime
# Actual changed full600 joins, with no guard/launcher imports or operations.
full=dict(c=c,compact=SimpleNamespace(**{k:rn[k]for k in["source_pass_v1","verify_values_v1","READINESS_SCHEMA","COMPACT","NATIVE"]}),datetime=datetime,timezone=timezone,re=re,math=__import__("math"),Path=Path)
for n in trees["full600_ROOT_compact_continuation_v1.py"].body:
 if isinstance(n,ast.Assign)and all(isinstance(t,ast.Name)for t in n.targets):
  try:exec(compile(ast.Module(body=[n],type_ignores=[]),"full600-constants#RAM","exec"),full)
  except NameError:pass
for name in["_short_ref","_request","_all20"]:extract(name,"full600_ROOT_compact_continuation_v1.py",full)
qualification=dict(acceptance_ref=rr("ram/qualification_acceptance"),qualified_input=dict(parents={parents[0]:{k:ready["parents"][parents[0]][k]for k in["observation_ref","native_preparation_ref","focal_registry_ref"]}}))
joined=full["_all20"](rrdy,ready,tr,truth,qualification)
check(list(joined["parents"])==parents and joined["parents"][parents[0]]==qualification["qualified_input"]["parents"][parents[0]],"Actual full600 all20 join retains exact qualified TRAIN000 refs")
check(all(set(row)=={"observation_ref","native_preparation_ref","focal_registry_ref"}for row in joined["parents"].values()),"Frozen trainer parent dictionaries unchanged")
check(joined["barrier"]["schema"]=="frozen_American_full30Mb_all20_observation_barrier_v2"and joined["barrier"]["this_full_training_truth_opened"]is False and joined["barrier"]["qualification_weights_optimizer_or_rng_reused"]is False,"Actual full600 preserves original fresh-state barrier flags")
badq=copy.deepcopy(qualification);badq["qualified_input"]["parents"][parents[0]]["focal_registry_ref"]=rr("ram/wrong_focals.npz")
reject(lambda:full["_all20"](rrdy,ready,tr,truth,badq),"Substituted qualified TRAIN000 rejects")
# Explicit new request uses the exact final review shape and inherited owner qualification pins.
q=read(ref(B/"ROOT_full600_compact_continuation_request.INACTIVE.v1.json"))
q.update(created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(days=1)).isoformat(),wait_deadline_at_utc=(now+timedelta(hours=12)).isoformat(),instance="independent_ram",automatic_single_parent_launch_authorized=False,continuation_manifest_ref=mref,continuation_source_review_ref=reviewref,source_manifest_ref=m["trainer_manifest_ref"],qualification_input_ref=rr("ram/qualification_input"),accounting_batch_input_ref=rr("ram/accounting_input"))
ram[q["accounting_batch_input_ref"]["path"]]={};qi=q["qualification_input_ref"];qb=rr("ram/qualification_binding")
ram[qi["path"]]=dict(phase="qualification",source_manifest_ref=q["source_manifest_ref"],controller_binding_ref=qb)
ram[qb["path"]]=dict(owner4_service_manifest_ref=full["OWNER4_MANIFEST"],owner4_service_source_ref=full["OWNER4_SOURCE"])
qref=rr("ram/continuation_request");ram[qref["path"]]=q
check(full["_request"](qref)[0]==q,"Exact new full600 request consumes actual final review shape in RAM")
old=ram[reviewref["path"]]["candidate_manifest_ref"];ram[reviewref["path"]]["candidate_manifest_ref"]=rr("ram/wrongmanifest");old2=ram[reviewref["path"]]["source_manifest_ref"];ram[reviewref["path"]]["source_manifest_ref"]=rr("ram/wrongmanifest")
reject(lambda:full["_request"](qref),"Incorrect new adapter review manifest rejects")
ram[reviewref["path"]]["candidate_manifest_ref"]=old;ram[reviewref["path"]]["source_manifest_ref"]=old2
check(not any(p.endswith((".npz",".trees"))for p in reads),"No tree/genotype/focal/truth scientific payload opened or hashed")
buildnode=next(n for n in trees["readiness_ROOT_v1.py"].body if isinstance(n,ast.FunctionDef)and n.name=="build")
buildtext=ast.unparse(buildnode)
check("33554432"in buildtext and "authority['all20_adapter_source_ref'] == c.ref(SOURCE)"in buildtext and "guard.budget_check_v9"in buildtext,"Readiness writes require exact charged compact admin and declared adapter")
check("reserve"in buildtext.split("def build",1)[-1]or "guard.budget_check_v9"in buildtext,"Readiness producer uses current guard write boundary")
result=dict(schema="TreeLoom_compact_all20_full600_independent_checks_v1",passed=True,check_count=len(checks),checks=checks,author_fixtures_reused=False,actual_guard_imported=False,actual_operations=False,actual_resource_process_socket_lock_GPU_probes=False,scientific_execution=False,real_payloads_opened_or_hashed=False,full600_fit_or_runtime_claimed=False)
print(json.dumps(dict(result=result,final_review_shape=review),sort_keys=True,indent=2))
