"""Independent source/AST/RAM review. No production modules or runtime probes."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
from typing import Callable
from contextlib import redirect_stdout
import argparse,ast,copy,hashlib,io,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1"
OUT="research/reviews/frozen_TreeLoom_bounded_training_independent_20261017_v1"
MANIFEST=BASE+"/source_manifest.v1.json";CHECKS=[]
def need(ok,msg="Independent check"):
 if not ok:raise ValueError(msg)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(path):return json.loads((ROOT/path).read_text())
def fn(path,name,env):
 s=(ROOT/path).read_text();n=next(n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),path+"#independent_RAM","exec"),env);return env[name]
def yes(name):CHECKS.append(name)
def rejects(name,call):
 try:call()
 except(ValueError,KeyError,TypeError):yes(name);return
 raise AssertionError("Expected rejection: "+name)
need(ref(BASE+"/HANDOFF.qualification.v1.json")["sha256"]=="25a56d2ddf8810b5ce22e5283cb30ae104682e1620d5985859978458c94ad4ff")
need(ref(MANIFEST)["sha256"]=="35760763a7f0608ed56cd62565ec555d4fcae7f622ea6bfb9dd5280257d20fa0")
m=doc(MANIFEST)
for p,sha in m["source_sha256"].items():
 need(ref(p)["sha256"]==sha)
 if p.endswith(".py"):compile((ROOT/p).read_bytes(),p,"exec")
for rr in m["executed_new_source_refs"]:need(ref(rr["path"])==rr)
need(len(m["source_sha256"])==85 and len(m["executed_new_source_refs"])==9)
yes("sealed_complete85pins_and_nine_actual_modules_authenticated_and_compiled")
cfg=doc(BASE+"/scientific_config.FROZEN.v1.json")
need(m["scientific_parameters"]==32132 and m["trainable_tensors"]==49)
need(cfg["memory_arm"]=="stable_leaf_memory"and cfg["total_optimizer_updates"]==600 and cfg["epochs"]==30)
yes("fixed_H050_32132parameters_49tensors_stable_leaf_and600updates")
# Original equations and the bounded complete route were independently read.
bounded=(ROOT/BASE/"bounded_execution_v1.py").read_text()
need("bank.to("not in bounded and "bundle.to("not in bounded)
need(".detach("not in bounded.replace("loss.detach()","loss"))
need('torch.func.functional_call(self.history,{"memory.ell_theta":theta}'in bounded)
need('return self.memory._history(forcing,spans,plan)'in bounded)
yes("CPU_global_observations_and_original_global_memory_with_connected_ell_copy")
need('(left//256)*256'in bounded and 'end=min(start+256,len(site_hold))'in bounded)
need('holds,inverse=torch.unique_consecutive(global_hold,return_inverse=True)'in bounded)
need('selected=torch.nonzero((holds>=a)&(holds<b)'in bounded)
yes("original_global256site_chunks_reduced_before_selected_hold_gather_preserving_ties")
need('totals=m.aggregation(weights,index=index,dim_size=episodes.episode_size.numel(),dim=0)'in bounded
 and 'minimum>0'in bounded and 'torch.finfo(torch.float32).max'in bounded)
yes("own_state_zero_foreign_shortcut_has_original_CUDA_weight_denominator_and_finite_bound")
need('context.split(self.hap_tile,dim=1)'in bounded and 'direction="forward"'in bounded and 'direction="reverse"'in bounded)
need('forward.sum()*0.+reverse.sum()*0.'in bounded)
need('result=result.index_copy(0,rows.to(self.device),logits)'in bounded)
yes("full_time_GRUs_tile_only_independent_haplotypes_and_preserve_all_focal_rows_and_zero_grad_paths")
need('loss.backward()'in bounded and 'clip_grad_norm_(params,1.'in bounded and bounded.count("optimizer.step()")==1)
need('reduction="mean"'in bounded and 'group["lr"]==.001'in bounded and 'group["weight_decay"]==.0001'in bounded)
yes("one_original_all10000_mean_CE_globalclip1_same49tensor_AdamW_step")
parity=(ROOT/BASE/"parity_v1.py").read_text()
need('torch.nn.init.normal_(adapter.O.weight'in parity and 'memory_nonzero='in parity)
need('for key in("step","exp_avg","exp_avg_sq")'in parity)
need('p.grad,q.grad'in parity and 'p.grad,other[name].grad'in parity and 'torch.testing.assert_close(na,nb'in parity)
yes("actual_parity_compares_FP32_logits_CE_all49_raw_clipped_gradients_clipnorm_AdamW_params_and_moments_nonzero_memory")
worker=(ROOT/BASE/"worker_v1.py").read_text()
need(worker.index("install_before_torch_v1")<worker.index("import torch"))
need(worker.index("run_parity_v1(resource_check=tick)")<worker.index("torch.manual_seed(70001);model"))
need('c.require(not optimizer.state'in worker and 'targets=target_join.join('in worker and 'train_step_v1(model,bundle,targets'in worker)
need('qualification_warmstart_or_optimizer_or_rng_reuse=False'in worker and 'VAL_truth_read=False,TEST_truth_read=False'in worker)
yes("actual_worker_gates_title_before_Torch_parity_before_fresh_genuine_TRAIN000_step_no_state_reuse_VAL_TEST")
inv=(ROOT/BASE/"gpu0_inventory_v1.py").read_text()
need("subprocess"not in inv.replace("subprocess calls","")and"nvmlDeviceGetComputeRunningProcesses_v3"in inv)
need("Never reuse a native handle after fork"in inv)
yes("new_periodic_exact_GPU0_NVML_inventory_has_no_helper_processes_or_roster_pruning")

# Actual contract callable with independent RAM documents.
q0=doc(BASE+"/ROOT_qualification_input.INACTIVE.v1.json")
now=datetime.now(timezone.utc);mr=ref(MANIFEST)
def contract_fixture():
 q=copy.deepcopy(q0);clock=dict(created_at_utc=(now-timedelta(seconds=2)).isoformat(),expires_at_utc=(now+timedelta(hours=1)).isoformat())
 q.update(clock,root_authorizer="root_agent",status="active",execution_authorized=True,template_only=False,
  drain_expires_unix=(now+timedelta(hours=1)).timestamp(),cpu_affinity=list(range(8)),source_manifest_ref=mr,
  output_directory="experiments/results/TreeLoom_frozen_bounded_RAM",worker_auxiliary_scope=".runtime/tmp/TreeLoom_frozen_bounded_RAM",ROOT_record_directory=".runtime/tmp/TreeLoom_frozen_bounded_RAM",
  source_review_ref=dict(path="RAM_review.json",sha256="1"*64),runtime_ref=dict(path="RAM_runtime.json",sha256="2"*64),
  runtime_qualification_ref=dict(path="RAM_runtime_qualified.json",sha256="3"*64),storage_ref=dict(path="RAM_storage.json",sha256="4"*64))
 q["caps"]["max_seconds"]=600;ir=dict(path="RAM_input.json",sha256="5"*64)
 docs={ir["path"]:q,mr["path"]:m,q["config_ref"]["path"]:cfg,
 q["source_review_ref"]["path"]:dict(independent=True,verdict="SOURCE_PASS",blockers=[],candidate_manifest_ref=mr),
 q["runtime_ref"]["path"]:dict(python="RAM/python",source_ref=ref(BASE+"/worker_v1.py"),CUDA_VISIBLE_DEVICES=q["gpu_uuid"]),
 q["runtime_qualification_ref"]["path"]:dict(python="RAM/python",ROOT_qualified_existing_Torch_PyG_runtime=True,cpu_affinity=list(range(8))),
 q["storage_ref"]["path"]:{}}
 env=dict(ROOT=ROOT,HERE=BASE,MANIFEST=MANIFEST,SOURCE=BASE+"/worker_v1.py",GPU0_UUID=q["gpu_uuid"],GPU_PHASES=("qualification","full_training"),
  BASE_POOL=q["base_pool_ref"],CURRENT_INPUT_REF=None,datetime=datetime,timezone=timezone,require=need,
  read=lambda rr:copy.deepcopy(docs[rr["path"]]),ref=ref,raw=lambda rr:None,local=lambda p:SimpleNamespace(is_file=lambda:True))
 fn(BASE+"/contract_v1.py","active",env);call=fn(BASE+"/contract_v1.py","validate_input",env)
 return q,docs,ir,call,env
q,docs,ir,call,env=contract_fixture();need(call(ir)[0]["phase"]=="qualification")
yes("actual_complete_qualification_input_contract_RAM_positive")
for name,mut in[
 ("reject_no_ROOT_execution",lambda q,d:q.update(execution_authorized=False)),
 ("reject_wrong_GPU",lambda q,d:q.update(physical_gpu=5)),
 ("reject_nonfixed_process_title",lambda q,d:q.update(process_title="python")),
 ("reject_truth_on_VAL_TEST",lambda q,d:q.update(query_VAL_TEST_truth_access_authorized=True)),
 ("reject_replay",lambda q,d:q.update(automatic_retry_or_resume=True)),
 ("reject_other_GPU_borrower_routes",lambda q,d:q.update(all_other_GPU0_borrower_routes_revoked=False)),
 ("reject_not_disposable",lambda q,d:q.update(qualification_is_disposable_and_not_benchmark_result=False)),
 ("reject_missing_parent",lambda q,d:q.update(parents={})),
 ("reject_unqualified_runtime",lambda q,d:d[q["runtime_qualification_ref"]["path"]].update(ROOT_qualified_existing_Torch_PyG_runtime=False)),
 ("reject_growth_of_allocator_cap",lambda q,d:q["caps"].update(max_cuda_reserved_bytes=65*2**30)),
 ("reject_changed_primary_cap",lambda q,d:q["caps"].update(output_cap_bytes=64*2**20)),
 ("reject_wrong_review_binding",lambda q,d:d[q["source_review_ref"]["path"]].update(candidate_manifest_ref={})),
 ("reject_review_blocker",lambda q,d:d[q["source_review_ref"]["path"]].update(blockers=["RAM"])),
 ("reject_unscoped_full_training",lambda q,d:q.update(phase="full_training"))]:
 q,d,ir,call,e=contract_fixture();mut(q,d);rejects(name,lambda:call(ir))
# Scope honest full-training barrier and fresh-state requirement are source enforced.
contract=(ROOT/BASE/"contract_v1.py").read_text()
need("all20_observations_sealed_before_any_TRAIN_truth"not in contract)
need('barrier["all20_observations_sealed_before_this_full_training_truth"]is True'in contract)
need('barrier["prior_disposable_TRAIN000_qualification_genuine_truth_opened"]is True'in contract)
need('barrier["qualification_weights_optimizer_or_rng_reused"]is False'in contract)
yes("honest_prior_disposable_TRAIN000_truth_exception_and_separate_fresh_full_training_barrier")

# Real guard_v1 order with RAM guard and memo boundary doubles.
events=[];q,d,ir,call,e=contract_fixture();q=call(ir)[0]
storage=dict(reservation=dict(reserved_bytes=64*2**20,output=str(ROOT/q["output_directory"]),
 auxiliary_scopes=[str(ROOT/q["worker_auxiliary_scope"])],
 consumer_bindings=[ref(BASE+"/worker_v1.py"),ref(MANIFEST),q["config_ref"],q["runtime_ref"]]))
g=SimpleNamespace(MODE="Census_observation_two_scope_retention_v1",_SELECTOR_REF=q["storage_policy_selector_ref"],budget=SimpleNamespace(v4=object()))
def select(rr):events.append("selected");return g
def install(*a,**k):need(events==["selected"]and k["consumer_role"]=="read_only_consumer");events.append("memo");return object()
def validate(*a,**k):need(events==["selected","memo"]);events.append("validate")
g.load_selected_guard_v1=select;g.validate_storage_v9=validate
ge=dict(_SELECTED_GUARD=None,_CONSUMER_MEMO=None,GUARD={},ROOT=ROOT,HERE=BASE,SOURCE=BASE+"/worker_v1.py",MANIFEST=MANIFEST,
 require=need,ref=ref,local=lambda p:ROOT/p,load=lambda rr,name:g if name=="_frozen_TreeLoom_exact_Census"else SimpleNamespace(install_consumer_v1=install))
need(fn(BASE+"/contract_v1.py","guard_v1",ge)(q,storage)is g and events==["selected","memo","validate"])
yes("actual_guard_select_then_existing_readonly_memo_then_first_original_validation_RAM")

# Actual full argv title check treats NVML/comm truncation as display-only.
def titlecase(wrong=False):
 env=dict(require=need,os=SimpleNamespace(getpid=lambda:123,getuid=lambda:1000),
  base_v1=lambda:SimpleNamespace(proc_identity=lambda p:dict(uid=1000,state="S",start_ticks=7)),
  Path=lambda p:SimpleNamespace(read_bytes=lambda:b"wrong\0"if wrong else b"AI Auto Research Pipeline Test\0",
   read_text=lambda:"AI Auto Researc\n"),GPU0_UUID=q0["gpu_uuid"])
 return fn(BASE+"/borrowed_gpu0_v1.py","actual_title_v1",env)(123,"AI Auto Researc")
need(titlecase()["actual_full_argv0"]=="AI Auto Research Pipeline Test");yes("actual_title_RAM_full_argv_verified_despite_truncated_NVML")
rejects("actual_title_RAM_rejects_wrong_full_argv",lambda:titlecase(True))
# Separate target join rejects prohibited splits before resolver or private path.
te=dict(TRAIN=tuple("american_c3_train_%03d"%i for i in range(20)),REGISTRY_REF={},Callable=Callable,require=need,
 authoritative_registry_v1=lambda rr:{p:{}for p in("american_c3_train_000",)},registry=lambda rr:None)
cls=fn(BASE+"/targets_v1.py","TrainTargetJoinV1",te);seen=[]
join=cls(lambda p:seen.append(p),max_truth_uncompressed_bytes=100,train_truth_refs={"american_c3_train_000":dict(path="experiments/results/RAM_truth.npz",sha256="0"*64)},allowed_parent_ids=("american_c3_train_000",))
for parent,split in(("american_c3_val_000","val"),("american_c3_test_000","test"),("american_c3_train_001","train")):
 rejects("private_truth_resolver_not_reached_"+parent,lambda p=parent,s=split:join.join(p,split=s,plan={},observation_seal={}))
need(seen==[]);yes("no_prohibited_split_or_unqualified_parent_reaches_private_truth_resolver")
target=(ROOT/BASE/"targets_v1.py").read_text()
need('np.searchsorted(intervals[:,1],positions,side="right")'in target and 'np.arange(600,1600,dtype=np.int64)'in target)
need('labels.dtype==np.uint8'in target and 'intervals[-1,1]==30000000'in target)
yes("original_complete30Mb_contiguous_half_open_truth_and1000query_columns_no_fallback")
# All actual entrypoints are inert without their explicit flags.
for path in("parent_v1.py","worker_v1.py"):
 ee=dict(argparse=argparse,json=json,__doc__="Independent RAM",
  c=SimpleNamespace(need=need,require=need),execute_v1=lambda *a:(_ for _ in()).throw(ValueError("unexpected execute")))
 stream=io.StringIO()
 with redirect_stdout(stream):need(fn(BASE+"/"+path,"main",ee)([])==0)
 need(json.loads(stream.getvalue())["status"]=="SOURCE_ONLY_INACTIVE");yes("actual_default_CLI_inert_"+path)

# Full actual parent supervisor through RAM OS/lease/lifecycle boundaries.
def parent_case(case):
 events=[];clock=[0.];raw_status=256 if case=="nonzero"else 0;reaped=[False];putrows={};signal_rows=[]
 def mono():clock[0]+=.01;return clock[0]
 nowstamp=datetime.now(timezone.utc).timestamp()
 cap=dict(job_id="RAMjob",cpu_affinity=list(range(8)),output_directory="experiments/results/TreeLoom_frozen_bounded_RAM",
  worker_auxiliary_scope=".runtime/tmp/TreeLoom_frozen_bounded_RAM",ROOT_record_directory=".runtime/tmp/TreeLoom_frozen_bounded_RAM",
  runtime_ref={},base_pool_ref={},caps=dict(max_seconds=600,max_cpu_rss_bytes=384*2**30),drain_expires_unix=nowstamp+3600,
  created_at_utc=datetime.fromtimestamp(nowstamp-1,timezone.utc).isoformat())
 class P:
  def __init__(self,p):self.p=str(p)
  def __truediv__(self,p):return P(self.p+"/"+str(p))
  def __str__(self):return self.p
  def exists(self):return False
  @property
  def parent(self):return self
  def is_dir(self):return True
  def mkdir(self,**k):events.append("mkdir")
  def read_text(self):return"0 0"
 class PathDouble:
  def __new__(cls,p):return P(p)
  @staticmethod
  def cwd():return ROOT
 class Stream:
  def fileno(self):return 9
  def close(self):events.append("stream_close")
 row=dict(pid=200,start=7,ppid=1,group=200,session=200,state="R")
 class Tree:
  def __init__(self,p):self.known={200:row}
  def refresh(self):
   if case=="refresh_error":raise ValueError("RAM refresh failure")
  def identities_absent(self):return reaped[0]and case!="unknown"
  def evidence(self):return dict(owned_identities=[dict(pid=200,start=7)])
 owned=SimpleNamespace(_children=lambda p:[],_stat=lambda p:row,_same=lambda a,b:False,OwnedTree=Tree,
  signal_identity=lambda r,s:signal_rows.append((r["pid"],s)))
 def waitpid(pid,flags):
  if pid==-1:raise ChildProcessError()
  if case=="unknown":return 0,0
  reaped[0]=True;return 200,raw_status
 def write(fd,data):need(data==b"G"and"launch"in events);events.append("gate");return 1
 osfake=SimpleNamespace(sched_setaffinity=lambda *a:None,getpid=lambda:1,close=lambda fd:events.append(("close",fd)),
  pipe=lambda:(10,11),set_blocking=lambda *a:None,read=lambda *a:b"",write=write,environ={},
  waitid=lambda *a:object(),waitpid=waitpid,waitstatus_to_exitcode=lambda x:x>>8,
  P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8,sysconf=lambda *a:4096)
 pidfd=SimpleNamespace(self_probe_v1=lambda:events.append("pidfd_probe_RAM"))
 sy=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1,optimize=0),modules={"pidfd_backend_v1":pidfd})
 guard=SimpleNamespace(budget_check_v9=lambda *a,**k:events.append("guard_RAM"))
 def put(cap,name,value):putrows[name]=copy.deepcopy(value);events.append("launch"if name=="actual_launch.json"else name);return dict(path=name,sha256="0"*64)
 c=SimpleNamespace(ROOT=ROOT,HERE=BASE,GPU0_UUID=q0["gpu_uuid"],need=need,validate_cap_v1=lambda r:(cap,None,None,{}),
  guard_v1=lambda *a:guard,local=lambda p:P(p),read=lambda rr:dict(python="RAM_python"),
  put=put,load=lambda rr,name:owned if name=="_empty_owner_original_owned_lifecycle"else pidfd if name=="pidfd_backend_v1"
   else SimpleNamespace(enable_child_subreaper=lambda:events.append("subreaper_RAM")))
 def popen(command,**kw):
  need(kw["start_new_session"]is True and kw["pass_fds"]==(10,4,5))
  need(kw["env"]["CUDA_VISIBLE_DEVICES"]==q0["gpu_uuid"])
  need(command[command.index("-m")+1]==BASE.replace("/",".")+".worker_v1")
  events.append("spawn_RAM");return SimpleNamespace(pid=200,stdout=Stream(),returncode=None)
 ee=dict(c=c,ROOT=ROOT,SOURCE=BASE+"/parent_v1.py",PIDFD={},OWNED={},OWNED_ROSTER_CAP=64,
  Path=PathDouble,os=osfake,sys=sy,json=json,put=put,
  borrowed=SimpleNamespace(attach_v1=lambda *a:(4,5,dict(physical_lease_metadata=dict(job="RAMlease")))),
  subprocess=SimpleNamespace(Popen=popen,PIPE=1,STDOUT=2),
  signal=SimpleNamespace(SIGINT=2,SIGTERM=15,SIGKILL=9,signal=lambda *a:None),
  time=SimpleNamespace(monotonic=mono,time=lambda:nowstamp,sleep=lambda s:clock.__setitem__(0,clock[0]+s)),
  select=SimpleNamespace(select=lambda *a:([1],[],[])))
 stream=io.StringIO()
 with redirect_stdout(stream):rc=fn(BASE+"/parent_v1.py","execute_v1",ee)(dict(path="RAM_input.json",sha256="0"*64))
 need(events.index("subreaper_RAM")<events.index("spawn_RAM")<events.index("launch")<events.index("gate"))
 need(("close",4)in events and("close",5)in events)
 wait=putrows["actual_wait.json"]
 if case=="success":
  need(rc==0 and wait["actual_raw_wait"]==0 and wait["all_owned_processes_drained"]is True and wait["pipe_eof"]is True)
 else:need(rc==1)
 if case=="refresh_error":need(signal_rows,"Retained identities still signaled after refresh failure")
 if case=="unknown":need(wait["actual_raw_wait"]is None and wait["all_owned_processes_drained"]is False)
 need(wait["actual_tool_outer_terminal_consumed"]is False and wait["storage_credit_claimed"]is False)
 return wait
for case in("success","nonzero","unknown","refresh_error"):
 parent_case(case);yes("actual_parent_complete_RAM_"+case+"_gate_wait_finite_cleanup_retained_identity")
print(json.dumps(dict(schema="frozen_TreeLoom_bounded_independent_source_AST_RAM_checks_v1",status="PASS",count=len(CHECKS),checks=CHECKS,
 candidate_manifest_ref=ref(MANIFEST),source_ref=ref(BASE+"/parent_v1.py"),checks_source_ref=ref(OUT+"/independent_checks_v1.py"),
 actual_guard_Torch_PyG_models_or_resource_process_socket_lock_GPU_imports_or_probes=0,
 actual_payload_truth_reads_or_admission_or_science=False,actual_fit_parity_update_score_claimed=False),sort_keys=True,indent=2))
