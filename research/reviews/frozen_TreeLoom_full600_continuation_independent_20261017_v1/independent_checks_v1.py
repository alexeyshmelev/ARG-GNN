"""Independent exact-code/AST/RAM admin-continuation review. No production imports/probes."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace,FunctionType,CodeType
from datetime import datetime,timezone,timedelta
from contextlib import contextmanager,redirect_stdout
import argparse,ast,copy,contextlib,hashlib,io,json,math,re
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");HERE="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1"
S=HERE+"/full600_ROOT_continuation_v1.py";M=HERE+"/full600_ROOT_continuation_manifest.v1.json";TRAIN=HERE+"/source_manifest.v1.json"
REVIEW="research/reviews/frozen_TreeLoom_bounded_training_independent_20261017_v1/review.qualification.v1.json"
PARENTS=["american_c3_train_%03d"%i for i in range(20)];checks=[]
def need(ok,msg="Independent RAM assertion"):
 if not ok:raise ValueError(msg)
def yes(n,ok=True):need(ok,n);checks.append(n)
def deny(n,call,kind=(ValueError,KeyError,TypeError)):
 try:call()
 except kind:yes(n);return
 raise AssertionError("Expected reject: "+n)
def realref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def realdoc(p):return json.loads((ROOT/p).read_text())
source=(ROOT/S).read_text();tree=ast.parse(source);compiled=compile(source,S,"exec")
def fn(path,name,e):
 text=(ROOT/path).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
 f=FunctionType(next(c for c in compile(text,path,"exec").co_consts if isinstance(c,CodeType)and c.co_name==name),e,name,tuple(ast.literal_eval(x)for x in node.args.defaults))
 f.__kwdefaults__={k.arg:ast.literal_eval(v)for k,v in zip(node.args.kwonlyargs,node.args.kw_defaults)if v is not None};return f
manifest=realdoc(TRAIN);h=realdoc(HERE+"/HANDOFF.full600_ROOT_continuation.v1.json")
yes("sealed_continuation_handoff_exact",realref(HERE+"/HANDOFF.full600_ROOT_continuation.v1.json")["sha256"]=="7b4bff7b064f9459c64d1ecbd56fe2d507bfbbe62dfa5f4d98c059521032b38d")
yes("sealed_continuation_source_manifest_exact",realref(S)["sha256"]=="237a1d99d6ac18bcc4972644a0b5eaa9f30a3ebcbd2ab65640efb27c97a189ac"and realref(M)["sha256"]=="79e2db93fe9d1f0b46afce999385f3b450646065ced2c6d5b75783ac21b6e810")
yes("prior_whole_trainer_review_exact",realref(REVIEW)["sha256"]=="ed77e7885a9b7403d778dc483ac2074e1b3f4bd4794eaa996f7488a8fcca001f"and realdoc(REVIEW)["verdict"]=="SOURCE_PASS"and realdoc(REVIEW)["independent"]is True and realdoc(REVIEW)["blockers"]==[]and realdoc(REVIEW)["candidate_manifest_ref"]==realref(TRAIN))
yes("immutable_trainer_manifest_exact",realref(TRAIN)["sha256"]=="35760763a7f0608ed56cd62565ec555d4fcae7f622ea6bfb9dd5280257d20fa0")
for path,sha in manifest["source_sha256"].items():need(realref(path)["sha256"]==sha)
yes("all85_trainer_source_compact_refs_unchanged",len(manifest["source_sha256"])==85)
for rr in manifest["executed_new_source_refs"]:need(realref(rr["path"])==rr)
yes("all9_executed_trainer_math_lifecycle_sources_unchanged",len(manifest["executed_new_source_refs"])==9)
worker=(ROOT/HERE/"worker_v1.py").read_text()
yes("fresh_seed_model_AdamW_no_qualification_warmstart","torch.manual_seed(70001);model=frozen.AmericanTreeLoomC3FullV1()"in worker and "optimizer=torch.optim.AdamW("in worker and "not optimizer.state"in worker and "torch.load("not in worker and "load_state_dict("not in worker)
yes("fixed600_final30_and_six_checkpoint_schedule","else range(1,31)"in worker and "epoch in(5,10,15,20,25,30)"in worker and 'else 600)'in worker and "fixed_final_epoch30_before_scored_benchmark_truth"in worker)
yes("all20_metadata_before_target_resolver_or_truth_join",worker.index('for parent_id,value in q["parents"].items()')<worker.index("target_join=TrainTargetJoinV1")<worker.index("targets=target_join.join"))
yes("continuation_no_science_import_new_supervisor_or_consumer_memo",not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(k in ast.unparse(n)for k in("torch","numpy","Census"))for n in tree.body)and "subprocess"not in source and "install_consumer_v1"not in source and source.count("os.execve(")==1)
class P(PurePosixPath):
 def resolve(self):return self
 def exists(self):return False
 def is_file(self):return True
class Pending(Exception):pass
class ReadyOnly(Exception):pass
class Repo:
 def __init__(self):self.rows={};self.reads=[];self.raws=[];self.saved=[]
 def save(self,p,v):
  p=str(p);need(p not in self.rows,"Exclusive RAM metadata");self.rows[p]=copy.deepcopy(v);self.saved.append(p);return self.ref(p)
 def ref(self,p):
  p=str(p);p=p[len(str(ROOT))+1:]if p.startswith(str(ROOT)+"/")else p
  if p not in self.rows:return realref(p)
  return dict(path=p,sha256=hashlib.sha256(json.dumps(self.rows[p],sort_keys=True).encode()).hexdigest())
 def read(self,r):
  need(not r["path"].endswith((".npz",".npy",".pt",".trees")),"No scientific payload read")
  self.reads.append(r["path"]);need(self.ref(r["path"])==r,"Exact RAM/source ref")
  return copy.deepcopy(self.rows[r["path"]])if r["path"]in self.rows else realdoc(r["path"])
 def raw(self,r):
  need(not r["path"].endswith((".npz",".npy",".pt",".trees")),"No scientific payload raw")
  self.raws.append(r["path"]);need(self.ref(r["path"])==r,"Exact RAM/source raw ref")
  return json.dumps(self.rows[r["path"]],sort_keys=True).encode()if r["path"]in self.rows else(ROOT/r["path"]).read_bytes()
def fixture():
 repo=Repo();now=datetime.now(timezone.utc)
 c=SimpleNamespace(ROOT=P(str(ROOT)),HERE=HERE,MANIFEST=TRAIN,SOURCE=HERE+"/worker_v1.py",GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667",require=need,
  read=repo.read,ref=repo.ref,raw=repo.raw,local=lambda p:P(str(ROOT))/p if not P(str(p)).is_absolute()else P(str(p)))
 active_env=dict(datetime=datetime,timezone=timezone,require=need)
 c.active=fn(HERE+"/contract_v1.py","active",active_env)
 for key in("GUARD",):
  n=next(n for n in ast.parse((ROOT/HERE/"contract_v1.py").read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==key for t in n.targets));setattr(c,key,ast.literal_eval(n.value))
 e=dict(c=c,HERE=HERE,SOURCE=S,MANIFEST=M,FULL_TEMPLATE=HERE+"/ROOT_full600_input.INACTIVE.v1.json",BARRIER_TEMPLATE=HERE+"/ROOT_full600_observation_barrier.INACTIVE.v1.json",
  REVIEW=realref(REVIEW),POLICY=h["actual_baseline_reserve_authority_ref"],SELECTOR=realref("research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json"),
  ENDPOINT=realref("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/empty_gpu0_controller_v1.py"),COORDINATOR=realref("research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/worker_entry_v1.py"),
  PARENTS=PARENTS,ADMIN_CAP=384*1024,RESERVATION_BYTES=64*2**20,Pending=Pending,ReadyOnly=ReadyOnly,datetime=datetime,timezone=timezone,math=math,re=re,json=json,Path=P,copy=copy)
 e["POLICY"]=realref("research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json")
 e["os"]=SimpleNamespace(waitstatus_to_exitcode=lambda s:s>>8)
 e["time"]=SimpleNamespace(time=lambda:datetime.now(timezone.utc).timestamp())
 for n in tree.body:
  if isinstance(n,ast.FunctionDef):e[n.name]=fn(S,n.name,e)
 directory=P(str(ROOT))/HERE/"RAM_continuation"
 e["_save"]=lambda directory,name,value:repo.save(str(directory.relative_to(P(str(ROOT))))+"/"+name,value)
 clock=dict(root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(hours=8)).isoformat())
 q=realcopy=realdoc(HERE+"/ROOT_full600_continuation_request.INACTIVE.v1.json")
 q.update(clock,instance="independent_ram",continuation_manifest_ref=realref(M),source_manifest_ref=realref(TRAIN),source_review_ref=realref(REVIEW),
  storage_policy_selector_ref=e["SELECTOR"],accounting_coordinator_source_ref=e["COORDINATOR"],automatic_single_parent_launch_authorized=True,poll_seconds=5,wait_deadline_at_utc=(now+timedelta(hours=1)).isoformat())
 qualified_input=repo.save(HERE+"/RAM_qualification_input.json",dict(phase="qualification",source_manifest_ref=realref(TRAIN),output_directory=HERE+"/RAM_qualification_output"))
 completed=repo.save(HERE+"/RAM_qualification_done.json",dict(input_ref=qualified_input,phase="qualification",complete_parent_updates=1,trainable_tensors=49,
  qualification_weights_are_not_final_training_weights=True,qualification_warmstart_or_optimizer_or_rng_reuse=False,VAL_truth_read=False,TEST_truth_read=False,
  elapsed_seconds=100.,peak_cuda_reserved_bytes=2**30,peak_cpu_RSS_bytes=2*2**30))
 wait=repo.save(HERE+"/RAM_qualification_wait.json",dict(scientific_input_ref=qualified_input,actual_direct_returncode=0,actual_raw_wait=0,all_owned_processes_drained=True,pipe_eof=True,failure=None,cleanup_errors=[],adopted_waits_truncated=False,adopted_waits=[]))
 outer=repo.save(HERE+"/RAM_qualification_outer.json",dict(actual_exit_code=0,actual_terminal_consumed=True))
 parity=repo.save(HERE+"/RAM_parity.json",dict(passed=True,all49_raw_and_clipped_gradients_and_AdamW_parameters_moments_compared=True,nonzero_memory_gradient=True))
 acceptance=repo.save(HERE+"/RAM_acceptance.json",dict(schema="ROOT_frozen_TreeLoom_actual_qualification_acceptance_v1",root_authorizer="root_agent",actual_outer_terminal_consumed=True,actual_exit_code=0,
  all_owned_processes_drained=True,source_manifest_ref=realref(TRAIN),synthetic_all49_parity_passed=True,full30Mb_TRAIN000_disposable_optimizer_update_verified=True,qualification_weights_reused=False,
  qualification_input_ref=qualified_input,completed_ref=completed,actual_parent_wait_ref=wait,actual_outer_terminal_ref=outer,synthetic_parity_ref=parity))
 q["qualification_input_ref"]=qualified_input
 readiness=dict(schema="TreeLoom_all20_training_readiness_v1",all20_complete=True,query_truth_opened=False,TRAIN_parent_ids=PARENTS,parents={})
 truth=dict(schema="TreeLoom_all20_training_truth_handles_v1",TRAIN_parent_ids=PARENTS,class_order=["EUR","EAS","AFR"],query_truth_opened=False,parents={})
 for parent in PARENTS:
  observation=repo.save(HERE+"/RAM_"+parent+".observation.json",dict(parent_id=parent,split="train",sequence_length_bp=30000000,query_ancestry_truth_in_inputs=False))
  focal=dict(path="experiments/results/RAM_"+parent+"/focals.npz",sha256="f"*64)
  prepared=repo.save(HERE+"/RAM_"+parent+".prepared.json",dict(parent_id=parent,observation_ref=observation,focals_ref=focal,query_ancestry_truth_opened=False))
  accepted=repo.save(HERE+"/RAM_"+parent+".close.json",dict(parent_id=parent,accepted=True))
  readiness["parents"][parent]=dict(parent_id=parent,observation_ref=observation,native_preparation_ref=prepared,focal_registry_ref=focal,focal_identity_sha256="a"*64,observation_only=True,all_native_trees_nodes_sites_samples_retained=True,inference_close_accepted_ref=accepted,native_close_accepted_ref=accepted)
  truth["parents"][parent]=dict(parent_id=parent,truth_ref=dict(path="experiments/results/RAM_"+parent+"/truth/query.npz",sha256="b"*64))
 readiness_ref=repo.save(HERE+"/RAM_all20_readiness.json",readiness);truth_ref=repo.save(HERE+"/RAM_all20_truth_handles.json",truth)
 batch=repo.save(HERE+"/RAM_batch_input.json",dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1"))
 q["accounting_batch_input_ref"]=batch
 for k,r in(("future_qualification_acceptance_path",acceptance),("future_all20_readiness_path",readiness_ref),("future_all20_truth_handles_path",truth_ref)):q[k]=r["path"]
 q["future_launch_authority_path"]=HERE+"/RAM_launch.json"
 qr=repo.save(HERE+"/RAM_request.json",q);q["_request_ref"]=qr
 qual=e["_qualification"](acceptance,repo.read(acceptance),q);data=e["_all20"](readiness_ref,readiness,truth_ref,truth,qual)
 owner=repo.save(HERE+"/RAM_owner.json",dict(clock,expires_unix=(now+timedelta(hours=4)).timestamp(),source_refs={e["ENDPOINT"]["path"]:e["ENDPOINT"]},jobs={},physical_gpu=0,gpu_uuid=c.GPU0_UUID,protected_envelope_bytes=0,storage_policy_selector_ref=e["SELECTOR"],capacity_policy_ref=e["POLICY"]))
 ready=repo.save(HERE+"/RAM_owner_ready.json",dict(root_authorizer="root_agent",actual_status_terminal_consumed=True,actual_status_tool_exit_code=0,controller_capability_ref=owner,controller_pid=123,controller_start_ticks=456,source_ref=e["ENDPOINT"],owner_parent_still_live=True,
  actual_status=dict(controller=dict(pid=123,start_ticks=456),capability_ref=owner,jobs=[],lifecycle="running",safety_error=None,ambiguous_descendants_or_lease_holders=[])))
 runtime=repo.save(HERE+"/RAM_runtime.json",dict(source_ref=realref(c.SOURCE),CUDA_VISIBLE_DEVICES=c.GPU0_UUID,python=str(ROOT/".runtime/env/bin/python")))
 rq=repo.save(HERE+"/RAM_runtime_qualification.json",dict(python=str(ROOT/".runtime/env/bin/python"),ROOT_qualified_existing_Torch_PyG_runtime=True,runtime_ref=runtime,cpu_affinity=list(range(8))))
 launch=dict(clock,schema="ROOT_TreeLoom_full600_launch_authority_v1",continuation_request_ref=qr,source_manifest_ref=realref(TRAIN),qualification_acceptance_ref=acceptance,all20_readiness_ref=readiness_ref,all20_truth_handles_ref=truth_ref,
  automatic_single_parent_launch_authorized=True,all_other_GPU0_borrower_routes_revoked=True,preserve_controller_until_owned_drain=True,qualification_weights_optimizer_or_rng_reused=False,
  controller_capability_ref=owner,controller_ready_ref=ready,runtime_ref=runtime,runtime_qualification_ref=rq,max_seconds=3600,
  measured_runtime_resource_plan=dict(schema="ROOT_TreeLoom_full600_measured_runtime_resource_plan_v1",ROOT_approved=True,qualification_completed_ref=completed,actual_qualification_elapsed_seconds=100.,actual_qualification_peak_cuda_reserved_bytes=2**30,actual_qualification_peak_cpu_RSS_bytes=2*2**30,all20_parent_dimensions_reviewed=True,length_only_extrapolation=False,persistent_preparation_or_tensor_cache_assumed=False,full_optimizer_updates=600,basis="Independent RAM only, no actual fit claim",estimated_full600_work_seconds=3000.))
 launch["expires_at_utc"]=(now+timedelta(hours=2)).isoformat();lr=repo.save(q["future_launch_authority_path"],launch)
 bound=e["_launch"](lr,launch,q,qual,data)
 return SimpleNamespace(repo=repo,e=e,c=c,q=q,qr=qr,qualified=qual,data=data,bound=bound,acceptance=acceptance,readiness=readiness,readiness_ref=readiness_ref,truth=truth,truth_ref=truth_ref,launch=launch,launch_ref=lr,directory=directory,batch=batch,now=now)
f=fixture();q2,m2=f.e["_request"](f.qr)
yes("actual_request_exact_sealed_source_coordinator_and_trainer_join",q2["source_manifest_ref"]==realref(TRAIN)and q2["accounting_coordinator_source_ref"]==f.e["COORDINATOR"])
yes("all20_barrier_constructed_without_truth_payload_reads",len(f.data["parents"])==20 and len(f.data["train_truth_refs"])==20 and f.data["barrier"]["prior_disposable_TRAIN000_qualification_genuine_truth_opened"]is True and f.data["barrier"]["this_full_training_truth_opened"]is False and not any(p.endswith((".npz",".npy",".pt",".trees"))for p in f.repo.reads+f.repo.raws))
for label,field,value in(("unconsumed","actual_outer_terminal_consumed",False),("nonzero","actual_exit_code",1),("undrained","all_owned_processes_drained",False),("warmstart","qualification_weights_reused",True)):
 f=fixture();a=f.repo.read(f.acceptance);a[field]=value;deny("qualification_rejects_"+label,lambda:f.e["_qualification"](f.acceptance,a,f.q))
for label,mut in(("missing_parent",lambda r:r["parents"].pop(PARENTS[-1])),("wrong_order",lambda r:r.update(TRAIN_parent_ids=list(reversed(PARENTS)))),("truth_in_readiness",lambda r:r.update(query_truth_opened=True)),("missing_native_nodes",lambda r:r["parents"][PARENTS[1]].update(all_native_trees_nodes_sites_samples_retained=False))):
 f=fixture();r=copy.deepcopy(f.readiness);mut(r);deny("all20_rejects_"+label,lambda:f.e["_all20"](f.readiness_ref,r,f.truth_ref,f.truth,f.qualified))
f=fixture();t=copy.deepcopy(f.truth);t["parents"][PARENTS[0]]["truth_ref"]["path"]="experiments/results/val_000/truth/query.npz"
deny("truth_handles_reject_VAL_path_without_open",lambda:f.e["_all20"](f.readiness_ref,f.readiness,f.truth_ref,t,f.qualified))
for label,mut,kind in[
 ("unimplemented_cache",lambda l:l["measured_runtime_resource_plan"].update(persistent_preparation_or_tensor_cache_assumed=True),(ValueError,)),
 ("not_all20_dimensions",lambda l:l["measured_runtime_resource_plan"].update(all20_parent_dimensions_reviewed=False),(ValueError,)),
 ("length_only",lambda l:l["measured_runtime_resource_plan"].update(length_only_extrapolation=True),(ValueError,)),
 ("wrong_measured_elapsed",lambda l:l["measured_runtime_resource_plan"].update(actual_qualification_elapsed_seconds=1.),(ValueError,)),
 ("work_does_not_fit",lambda l:l["measured_runtime_resource_plan"].update(estimated_full600_work_seconds=3600.),(ReadyOnly,)),
 ("owner_expiry_extension",lambda l:l.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=5)).isoformat()),(ReadyOnly,)),
 ("wrong_all20_ref",lambda l:l.update(all20_readiness_ref={}),(ValueError,))]:
 f=fixture();l=copy.deepcopy(f.launch);mut(l);deny("launch_rejects_"+label,lambda:f.e["_launch"](f.launch_ref,l,f.q,f.qualified,f.data),kind)
# Complete exact admission core with genuine original47/40baseline, original merge/draft/scopeSpec.
def admission(case="success"):
 f=fixture();trace=[];held=[False];locked=[False];calls=dict(state=0,apply=0,scope=0);e=f.e;c=f.c
 baseline=realdoc(h["actual_baseline_reserve_authority_ref"]["path"])["bindings"]
 need(realref(h["actual_baseline_reserve_authority_ref"]["path"])==h["actual_baseline_reserve_authority_ref"])
 @contextmanager
 def coordinator(rr):
  need(rr==f.batch and not held[0]);held[0]=True;trace.append("coordinator-enter")
  try:yield
  finally:held[0]=False;trace.append("coordinator-exit")
 @contextmanager
 def mutex(root):
  need(held[0]and not locked[0]);locked[0]=True;trace.append("mutex-enter")
  try:yield
  finally:locked[0]=False;trace.append("mutex-exit")
 def canonical(root,path):return P(path)
 def file_bytes(*args):return b"",None
 for fun in(mutex,canonical,file_bytes):fun.__module__="independent_RAM_original_v4"
 v4=SimpleNamespace(__name__="independent_RAM_original_v4",mutex=mutex,canonical=canonical,file_bytes=file_bytes,encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":")).encode())
 se=dict(Path=P,canonical=canonical,bound=lambda root,r:f.repo.raw(r),integer=lambda x,n,m:need(type(x)is int and x>=m),
  accounting_paths=lambda root:[root/".runtime/storage",root/".runtime/locks/scheduler.lock"],paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents)
 original_scope=fn("src/arggnn/project_budget_v4.py","scope_spec",se)
 def scope(root,r):calls["scope"]+=1;return original_scope(root,r)
 v4.scope_spec=scope
 g=SimpleNamespace(MODE="Census_observation_two_scope_retention_v1",MAX_JSON=65536,META_CHARGE=524288,budget=SimpleNamespace(v4=v4))
 ge=dict(copy=copy,require=need,MODE=g.MODE,RETIRE_OPERATION="RAM",_raw=f.repo.raw)
 for name in("consumer_bindings_v1","operation_draft_v1"):setattr(g,name,fn(c.GUARD["path"],name,ge))
 g.load_selected_guard_v1=lambda r:g
 def capture(*args,**kw):need(held[0]and locked[0]and kw==dict(fresh=True));calls["state"]+=1;return({},set(),{},baseline,[],[])
 g._state_locked=capture;g._pins=lambda v:dict(base_pins={},completed_overlay_prefix=dict(events=0,tail_sha256=None))
 def apply(rr,**kw):
  need(held[0]and not locked[0]);calls["apply"]+=1;trace.append("apply")
  if case=="uncertain_apply":raise ValueError("RAM unknown original apply; preserve")
  a=f.repo.read(rr);return dict(value=dict(a["request"],root_admission=rr),initial_ledger_pins=dict(RAM=True))
 g.apply_reserve_v1=apply
 def load(rr,name):
  need(rr in(c.GUARD,e["COORDINATOR"]),"Only actual guard/coordinator source loaded in RAM")
  return g if rr==c.GUARD else SimpleNamespace(accounting_transaction=coordinator)
 c.load=load
 ce=dict(CURRENT_INPUT_REF=None,read=f.repo.read,raw=f.repo.raw,ref=f.repo.ref,local=c.local,active=c.active,require=need,
  MANIFEST=TRAIN,HERE=HERE,SOURCE=c.SOURCE,ROOT=c.ROOT,GPU_PHASES=("qualification","full_training"),GPU0_UUID=c.GPU0_UUID,
  BASE_POOL=realdoc(HERE+"/ROOT_full600_input.INACTIVE.v1.json")["base_pool_ref"],datetime=datetime)
 c.validate_input=fn(HERE+"/contract_v1.py","validate_input",ce)
 if case=="pre_capture_no_fit":e["_remaining_fits"]=lambda a,p:False
 elif case=="post_apply_no_fit":
  e["_remaining_fits"]=lambda a,p:calls["apply"]==0
 if case in("pre_capture_no_fit","uncertain_apply"):
  deny("admission_preserves_"+case,lambda:e["_admit"](f.q,manifest,f.directory,f.qualified,f.data,f.bound),(ReadyOnly,ValueError))
  yes("no_false_reserve_success_"+case,calls["apply"]==(1 if case=="uncertain_apply"else 0))
  return
 r=e["_admit"](f.q,manifest,f.directory,f.qualified,f.data,f.bound);actual=f.repo.read(r["input_ref"]);storage=f.repo.read(actual["storage_ref"]);bindings=storage["reservation"]["consumer_bindings"]
 yes("complete_original64_source_scope_and_contract_"+case,calls==dict(state=1,apply=1,scope=1)and len(baseline)==47 and len(bindings)==64 and all(rr in bindings for rr in baseline.values())and c.GUARD in bindings and trace.index("mutex-exit")<trace.index("apply")<trace.index("coordinator-exit"))
 yes("admitted_input_all20_fresh_full600_primary_AUX64MiB_"+case,actual["phase"]=="full_training"and list(actual["parents"])==PARENTS and len(actual["train_truth_refs"])==20 and storage["reservation"]["reserved_bytes"]==64*2**20 and not any("checkpoint"in k.lower()for k in actual))
 if case=="success":yes("success_has_only_pending_fresh_parent_argv",r["status"]=="FULL600_ADMITTED_SINGLE_PARENT_EXEC_NEXT"and r["automatic_exec_allowed"]is True and r["execute_argv"][1:5]==["-B","-s","-m",HERE.replace("/",".")+".parent_v1"])
 else:yes("postreserve_window_failure_honestly_admitted_no_launch",r["status"]=="FULL600_ADMITTED_WINDOW_INSUFFICIENT_NO_LAUNCH"and r["automatic_exec_allowed"]is False)
 deny("original65th_binding_rejected_"+case,lambda:g.consumer_bindings_v1(bindings,[realref(HERE+"/ROOT_full600_launch_authority.INACTIVE.v1.json")]))
admission();admission("post_apply_no_fit");admission("pre_capture_no_fit");admission("uncertain_apply")
# Complete continuation status machine; external wait/admit/exec and mkdir are RAM.
def continuation(case):
 f=fixture();saved=[];calls=[];e=f.e
 class D(P):
  def exists(self):return False
  def mkdir(self,**kw):calls.append("mkdir")
 c=f.c;c.local=lambda p:D(str(ROOT))/p if not P(str(p)).is_absolute()else D(str(p))
 e["_request"]=lambda r:(copy.deepcopy(f.q),manifest)
 e["_save"]=lambda directory,name,v:(saved.append(copy.deepcopy(v))or dict(path=name,sha256="0"*64))
 def wait(q,d):
  calls.append("wait")
  if case=="pending":raise Pending("RAM finite metadata deadline")
  if case=="ready_only":return f.qualified,f.data,None,"RAM actual owner/time insufficient"
  return f.qualified,f.data,f.bound,None
 e["_wait"]=wait
 def admit(*args):
  calls.append("admit")
  if case=="before_reserve_no_fit":raise ReadyOnly("RAM no reservation before capture")
  return dict(status="FULL600_ADMITTED_WINDOW_INSUFFICIENT_NO_LAUNCH"if case=="admitted_no_exec"else"FULL600_ADMITTED_SINGLE_PARENT_EXEC_NEXT",automatic_exec_allowed=case!="admitted_no_exec",execute_argv=["RAM_python","-B","-s","-m","existing.parent"],input_ref={})
 e["_admit"]=admit
 class ExecReplaced(Exception):pass
 def execve(path,argv,env):
  calls.append("exec");need(calls==["mkdir","wait","admit","exec"]and env["PYTHONDONTWRITEBYTECODE"]=="1"and env["PYTHONNOUSERSITE"]=="1");raise ExecReplaced()
 e["os"]=SimpleNamespace(environ={},execve=execve)
 if case=="exec":
  with redirect_stdout(io.StringIO()):
   try:e["continue_v1"](f.qr)
   except ExecReplaced:pass
   else:raise AssertionError("Expected exactexec")
  yes("fresh_exec_replaces_mutation_interpreter_once",calls.count("exec")==1);return
 with redirect_stdout(io.StringIO()):r,code=e["continue_v1"](f.qr)
 expected={"pending":"PENDING_GENUINE_PREREQUISITES_DEADLINE","ready_only":"READY_DATA_NO_ADMISSION","before_reserve_no_fit":"READY_DATA_NO_ADMISSION","admitted_no_exec":"FULL600_ADMITTED_WINDOW_INSUFFICIENT_NO_LAUNCH"}[case]
 yes("honest_status_"+case,r["status"]==expected and "exec"not in calls and calls[0:2]==["mkdir","wait"])
 if case in("pending","ready_only"):yes("no_admit_"+case,"admit"not in calls)
continuation("pending");continuation("ready_only");continuation("before_reserve_no_fit");continuation("admitted_no_exec");continuation("exec")
# Actual default main inert, no production request path.
e=dict(argparse=argparse,json=json,__doc__="RAM")
with redirect_stdout(io.StringIO())as output:r=fn(S,"main",e)([])
yes("default_inactive_no_wait_admit_or_execute",r==0 and json.loads(output.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",wait=False,admission=False,execution=False))
print(json.dumps(dict(schema="TreeLoom_full600_continuation_independent_exact_code_AST_RAM_checks_v1",status="INDEPENDENT_PASS",count=len(checks),checks=checks,
 source_ref=realref(S),source_manifest_ref=realref(M),trainer_manifest_ref=realref(TRAIN),trainer_review_ref=realref(REVIEW),
 actual_guard_runtime_imports_configurations_resource_process_GPU_probes_admissions_launches_closes=0,scientific_payload_truth_checkpoint_reads_or_hashes=0,
 actual_fit_current_capacity_or_scientific_success_claimed=False),sort_keys=True,indent=2))
