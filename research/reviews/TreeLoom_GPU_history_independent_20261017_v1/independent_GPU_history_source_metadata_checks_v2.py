"""Independent whole-route GPU-history source and metadata checks. No Torch import."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,math,re,time
B=Path("research/reviews/TreeLoom_GPU_history_source_20261017_v1")
O=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1")
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
checks=[];ram={};reads=[]
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
def mr(p):return rr(p)if str(p).startswith("ram/")else ref(p)
def read(r):
 reads.append(r["path"])
 if r["path"]in ram:return copy.deepcopy(ram[r["path"]])
 need(ref(r["path"])==r,"Exact immutable source/metadata SHA")
 return json.loads(Path(r["path"]).read_text())
def raw(r):
 need(not r["path"].endswith((".pt",".pkl",".npz",".trees")),"No scientific payload read/hash")
 need(mr(r["path"])==r,"Exact source/metadata SHA")
 return b"RAM"if r["path"]in ram else Path(r["path"]).read_bytes()
mref=ref(B/"source_manifest.v1.json");m=read(mref)
check(mref["sha256"]=="0b0013f0907ef1b9a2b7b37dda1e59230111761d19a55821e9c6489adde5c4eb","Exact final GPU-history whole-route manifest")
trees={}
for r in m["executed_new_source_refs"]:
 check(ref(r["path"])==r,"Exact new source "+Path(r["path"]).name)
 trees[Path(r["path"]).name]=ast.parse(Path(r["path"]).read_text())
for name in["parent_v1.py","borrowed_gpu0_v1.py","gpu0_inventory_v1.py","targets_v1.py"]:
 a=(B/name).read_text().replace(str(B),str(O)).replace(str(B).replace("/","."),str(O).replace("/","."))
 check(ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse((O/name).read_text()),include_attributes=False),"Original ownership/resource/target module unchanged "+name)
cpu=read(m["tiny_CPU_history_check_ref"])
check(cpu["passed"]is True and cpu["device"]=="cpu"and cpu["holds"]==9 and cpu["haplotypes"]==1600 and cpu["window_holds"]==3,"Authenticated existing bounded CPU FP32 fixture")
check(cpu["checks"]["forward"]["max_absolute_error"]==0 and cpu["checks"]["forcing_gradient"]["max_absolute_error"]==0 and cpu["checks"]["ell_theta_gradient"]["max_absolute_error"]<=2e-6,"Existing fixture forward/forcing exact and ell error within declared FP32 tolerance")
check(cpu["optimizer_updates"]==cpu["scientific_data_reads"]==0 and cpu["bitwise_GPU_parity_claimed"]is False,"Fixture makes no optimizer/data or GPU parity claim")
# Static numerical algebra audit, no mathematical execution.
history=(B/"stable_leaf_history_v1.py").read_text();ht=ast.parse(history)
hclass=next(n for n in ht.body if isinstance(n,ast.ClassDef)and n.name=="_Direction")
back=ast.unparse(next(n for n in hclass.body if isinstance(n,ast.FunctionDef)and n.name=="backward"))
forward=ast.unparse(next(n for n in hclass.body if isinstance(n,ast.FunctionDef)and n.name=="forward"))
check("range(len(ctx.windows) - 1, -1, -1)"in back and "g[0 if ctx.reverse else -1].add_(carry)"in back,"Cross-window VJP reverses traversal and adds carry at exact terminal state")
check("carry = None if incoming is None else vjp[3]"in back and "torch.autograd.grad(out, inputs, g, allow_unused=True)"in back,"Incoming state receives full next-window VJP, no truncation")
check("grad_force[lo:hi].copy_(vjp[0].to('cpu'))"in back,"Disjoint complete forcing slices receive local VJP exactly once")
check("ctx.boundaries.append(None if state is None else state.to('cpu'))"in forward and "result[a:b].copy_(out.to('cpu'))"in forward,"Forward carries exact state across windows and retains complete CPU result")
window=ast.unparse(next(n for n in ht.body if isinstance(n,ast.FunctionDef)and n.name=="_window"))
check("forcing.index_select(0, groups.reshape(-1))"in window and ".unbind(0)"in window,"Forcing gather/scatter is local per window, not per hold over global forcing")
stable=ast.unparse(next(n for n in ht.body if isinstance(n,ast.FunctionDef)and n.name=="stable_leaf_history_v1"))
check("rho = torch.exp(-spans / ell)"in stable and "source = -torch.expm1(-spans / ell)"in stable and "read - forcing"in stable and "memory._mean"in stable,"Original FP32 coefficient functions and CPU mean/residual pooling preserved")
check("theta.sigmoid()"in stable and "plan.haplotypes == 1600"in stable and "forcing.shape[1] == 8"in stable,"Original trainable ell transform and complete1600x8 history shape preserved")
# Independently source-check only new cold preparation delta, inherit its actual bounded author fixture honestly.
prep=m["preparation_reuse_source_manifest_ref"];pm=read(prep)
prepfixture=next(r for r in pm["own_refs"]if r["path"].endswith("/tiny_fixture_checks.result.v1.json"))
pf=read(prepfixture)
check(pf["status"]=="PASS"and pf["all_tensor_fields_bitwise_equal"]==87 and pf["global_recipe_metadata_equal"]is True and pf["connected_and_isolated_zero_SNP_holds_checked"]is True,"Authenticated existing87field preparation fixture, not an independent numeric rerun")
pbase=Path(prep["path"]).parent
opt=ast.parse((pbase/"optimized_preparation_v1.py").read_text());orig=ast.parse(Path("research/reviews/frozen_benchmark_training_ready_20261017_v1/american_full30mb_prepare_v1.py").read_text())
get=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
for recipe in ["build_clade_episode_arrays_v1","prepare_cut_plan_v1"]:
 a=next(n for n in ast.walk(opt)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id==recipe)
 z=next(n for n in ast.walk(orig)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id==recipe)
 check(ast.dump(a,include_attributes=False)==ast.dump(z,include_attributes=False),"Original preparation global recipe unchanged "+recipe)
obs=(pbase/"optimized_observations_v1.py").read_text()
check("nodes=np.asarray(list(tree.nodes()),dtype=np.int64)"in obs and 'stamp[nodes]=epoch'in obs and "raw_parent=tree.parent_array[nodes]"in obs and "age=np.asarray(node_time[nodes],dtype=np.float64)"in obs,"Vectorized native workspace preserves original tree traversal/order/parent/time metadata")
check("if lo==hi:"in obs and "support[:,4]=1."in obs,"Zero-SNP shortcut preserves original empty branch/root support")
# Original full600 loop and learning conditions survive profile/history additions.
wt=trees["worker_v1.py"];run=get(wt,"_run_v1")
rt=ast.unparse(run)
check("range(1, epochs + 1)"in rt and "epoch_order_v1(epoch, seed=cfg['seed'])"in rt and "600"in rt,"Original epoch order and600 complete-parent training condition retained")
check("TrainTargetJoinV1"in rt and rt.index("load_parent(parent_id)")<rt.index("target_join.join(parent_id"),"Complete observed parent prepared/sealed before own genuine TRAIN target join")
check("run_tiny_v1(device='cuda:0'"in rt and "cold_prepare_complete_parent"in rt and "preparation_cache_plan_ref"not in rt,"Executable worker performs tiny GPU check then direct complete cold preparation, no cache producer")
check("resource_check=tick"in rt and "caps['max_cuda_reserved_bytes']"in rt and "caps['max_cpu_rss_bytes']"in rt,"Actual worker retains finite resource checks through cold preparation and training")
prof=(B/"bounded_profiler_v1.py").read_text()
check("interval_seconds!=5"in prof and "len(self.stages)>=48"in prof and "len(self.stacks)<24"in prof and "len(rows)<8"in prof and "len(data)>49152"in prof,"Profiler has finite5s/stage/stack/frame/48KiB bounds")
check('tmp=self.folder/"profile.next.json";dst=self.folder/"profile.latest.json"'in prof and "os.replace(tmp,dst)"in prof,"Only one aggregate profile plus one bounded transient")
# Actual stdlib builder/contract joins, no guard or runtime operations.
cn=dict(Path=Path,datetime=datetime,timezone=timezone,copy=copy,json=json,time=time,require=need)
for n in trees["contract_v1.py"].body:
 if isinstance(n,ast.Assign):exec(compile(ast.Module(body=[n],type_ignores=[]),"contract-constants#RAM","exec"),cn)
cn.update(read=read,raw=raw,ref=mr,local=lambda p:ROOT/p)
def extract(name,file,ns):
 fn=get(trees[file],name);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(B/file)+"#independentRAM","exec"),ns)
for fn in["active","validate_input"]:extract(fn,"contract_v1.py",cn)
c=SimpleNamespace(**cn)
bn=dict(c=c,Path=Path,datetime=datetime,timezone=timezone,copy=copy,json=json,re=re)
for n in trees["qualification_ROOT_builder_v1.py"].body:
 if isinstance(n,ast.FunctionDef)and n.name=="R":exec(compile(ast.Module(body=[n],type_ignores=[]),"builder-R#RAM","exec"),bn)
 if isinstance(n,ast.Assign):exec(compile(ast.Module(body=[n],type_ignores=[]),"builder-constants#RAM","exec"),bn)
for fn in["_interval","_request","_controls","_runtime_source_binding_v1","_actual_input_v1","_direct_bindings_v1"]:extract(fn,"qualification_ROOT_builder_v1.py",bn)
review=dict(schema="TreeLoom_GPU_history_independent_whole_source_review_v1",independent=True,verdict="SOURCE_PASS",blockers=[],candidate_manifest_ref=mref,source_manifest_ref=mref,reviewed_sources=m["executed_new_source_refs"],source_only=True,operational_authority=False,actual_operations=False,scientific_execution=False)
reviewref=rr("ram/final_history_review");ram[reviewref["path"]]=review
ownerref=dict(path="research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4/ROOT_admission_records_root01/capability.json",sha256="ef51d487f764c81f4cf7977e47dccc4050848ac653977b96eca7049ca2bd0745")
owner=read(ownerref)
readyref=dict(path="research/reviews/research_execution_ROOT_20261017_v34/native_owner_v4_live_ready.root_v1.json",sha256="315fbd95ee1b521c0cb80125579a969de2c75ba619320e5e53b4fa7a1910e3d2")
now=datetime.now(timezone.utc);end=now+timedelta(seconds=4000)
q=read(ref(B/"ROOT_qualification_admission_request.INACTIVE.v1.json"))
q.update(status="active",template_only=False,execution_authorized=True,created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),max_seconds=3600,instance="independent_ram",source_manifest_ref=mref,source_review_ref=reviewref,controller_capability_ref=ownerref,controller_ready_ref=readyref,accounting_batch_input_ref=dict(path="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/ROOT_batch_admission_root01_oct17/batch_input.json",sha256="dfd6ced48bf7ccfe8ebf9d58e3ffb2ab5193e8a586882d873e3bd4eb9279c092"))
qr=rr("ram/history_request");ram[qr["path"]]=q
actualq,actualowner,ready,actualm,qualification,truth=bn["_request"](qr)
check(actualq==q and actualowner==owner,"Exact builder request consumes finalreview and authenticated ef51 owner capability metadata")
check(ready["controller_pid"]==270948 and ready["controller_start_ticks"]==493592520,"Actual archived ready metadata joins same owner identity, no liveness claim")
def saveRAM(directory,name,value):
 r=rr("ram/"+name);ram[r["path"]]=copy.deepcopy(value);return r
bn["_save"]=saveRAM
binding,attachment=bn["_controls"](q,owner,ready)
bindingref=saveRAM(None,"controller_binding.json",binding);attachmentref=saveRAM(None,"attachment.json",attachment)
rref,rqualref=bn["_runtime_source_binding_v1"](None,qualification)
check(read(rref)["source_ref"]==ref(B/"worker_v1.py")and read(rref)["new_runtime_probe_performed"]is False,"Runtime is truthfully rebound to new executed worker without new probe")
actual=bn["_actual_input_v1"](q,qualification,truth,output="experiments/results/independent_RAM_history",auxiliary=".runtime/tmp/independent_RAM_history",job="RAMhistory",request_ref=qr,binding_ref=bindingref,attachment_ref=attachmentref,runtime_ref=rref,runtime_qualification_ref=rqualref,drain_expires_unix=bn["_interval"](q,owner))
storage_ref=rr("ram/history_storage");ram[storage_ref["path"]]={};actual["storage_ref"]=storage_ref
ir=rr("ram/history_input");ram[ir["path"]]=actual
check(cn["validate_input"](ir)[0]==actual,"Actual new qualification input passes physical new contract in RAM")
template=read(ref(bn["TEMPLATE"]))
check(actual["parents"]==template["parents"]and actual["config_ref"]==template["config_ref"],"Exact original full30MbTRAIN000 observation/focals and frozen scientific config retained")
check(actual["caps"]["max_cuda_reserved_bytes"]==24*2**30 and actual["caps"]["max_cpu_rss_bytes"]==128*2**30 and actual["caps"]["reservation_bytes"]==64*2**20,"New24GiB/128GiB bounds are downward failstops, whole charge unchanged")
check(binding["owner4_service_manifest_ref"]==owner["source_manifest_ref"]and binding["endpoint_source_ref"]==bn["ENDPOINT"]and attachment["downstream_max_cuda_reserved_bytes"]==24*2**30,"New builder retains dual owner4/v3 binding and exact24GiB attachment")
for key,bad in[("history_holds_per_window",127),("profile_interval_seconds",10),("preparation_cache_plan_ref",rr("ram/cache")),("preparation_cold_qualification",False),("all_other_GPU0_borrower_routes_revoked",False)]:
 v=copy.deepcopy(q);v[key]=bad;ram[qr["path"]]=v
 reject(lambda:bn["_request"](qr),"New request rejects altered field "+key)
ram[qr["path"]]=q
old=ram[reviewref["path"]]["reviewed_sources"];ram[reviewref["path"]]["reviewed_sources"]=[r for r in old if not r["path"].endswith("/parent_v1.py")]
reject(lambda:bn["_request"](qr),"Whole review must explicitly include actual new parent source")
ram[reviewref["path"]]["reviewed_sources"]=old
for key,bad in[("max_cuda_reserved_bytes",64*2**30),("max_cpu_rss_bytes",384*2**30)]:
 old=ram[ir["path"]]["caps"][key];ram[ir["path"]]["caps"][key]=bad
 reject(lambda:cn["validate_input"](ir),"Execution contract rejects old resource ceiling "+key);ram[ir["path"]]["caps"][key]=old
direct=bn["_direct_bindings_v1"](q,actual,qr,bindingref,attachmentref,rref,rqualref)
baseline=read(dict(path="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_records_root01_oct17/reserve_authority.json",sha256="737442dc4280177a278ba27d433ebcb0e3777896712c83499aa71194f1bc6be4"))["bindings"]
unique={(r["path"],r["sha256"])for r in list(baseline.values())+direct}
check(len(direct)==18 and len(unique)<=64,"Actual47role baseline plus18direct records fits original64binding ceiling")
prepare=get(trees["qualification_ROOT_builder_v1.py"],"prepare_v1")
coord=next(n for n in ast.walk(prepare)if isinstance(n,ast.With)and any(isinstance(i.context_expr,ast.Call)and isinstance(i.context_expr.func,ast.Attribute)and i.context_expr.func.attr=="accounting_transaction"for i in n.items))
calls=[n.func.attr for n in ast.walk(coord)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
check("_state_locked"in calls and "operation_draft_v1"in calls and "apply_reserve_v1"in calls,"Actual v3 coordinator still spans original capture/authority/apply")
pt=ast.unparse(prepare)
check("c.HERE.replace('/', '.') + '.parent_v1'"in pt and "automatic_launch=False"in pt.replace(" ",""),"Builder returns new parent entrypoint only, no automatic scientific launch")
check(all(not path.endswith((".pt",".pkl",".npz",".trees"))for path in reads),"No real observations/checkpoints/focals/truth payload read or hashed")
result=dict(schema="TreeLoom_GPU_history_independent_source_metadata_checks_v1",passed=True,check_count=len(checks),checks=checks,
 author_numeric_fixtures_authenticated_not_rerun=True,independent_numerical_campaign=False,exact_final_review_shape_and_actual_builder_contract_RAM_join=True,
 actual_guard_imported=False,actual_resources_GPU_process_socket_lock_probes=False,actual_accounting_or_admission=False,
 real_scientific_payloads_read_or_hashed=False,scientific_execution=False,actual_GPU_numerical_performance_or_fit_claimed=False)
print(json.dumps(dict(result=result,final_review_shape=review),sort_keys=True,indent=2))
