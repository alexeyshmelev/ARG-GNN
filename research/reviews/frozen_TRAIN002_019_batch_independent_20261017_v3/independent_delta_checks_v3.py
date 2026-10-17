"""Independent v3 scheduler-delta source/AST/RAM review; no production module execution."""
from pathlib import Path
from types import FunctionType,CodeType,SimpleNamespace
from datetime import datetime,timezone,timedelta
from contextlib import redirect_stdout,nullcontext
import ast,copy,contextlib,io,json,hashlib
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OLD="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v2"
NEW="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3"
PREVIOUS="research/reviews/frozen_TRAIN002_019_batch_independent_20261017_v2/review.v2.json"
checks=[]
def need(ok,msg="Independent RAM condition"):
 if not ok:raise ValueError(msg)
def yes(n,ok=True):need(ok,n);checks.append(n)
def deny(n,call):
 try:call()
 except (ValueError,TypeError,KeyError):yes(n);return
 raise AssertionError("Expected rejection "+n)
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def doc(p):return json.loads((ROOT/p).read_text())
def normalize(s):return s.replace(NEW,OLD).replace(NEW.replace("/","."),OLD.replace("/","."))
def module(p):return ast.parse((ROOT/p).read_text())
def nodes(p):return {n.name:n for n in module(p).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def callable_(p,name,env):
 s=(ROOT/p).read_text();code=compile(s,p,"exec");n=nodes(p)[name]
 f=FunctionType(next(c for c in code.co_consts if isinstance(c,CodeType)and c.co_name==name),env,name,tuple(ast.literal_eval(x)for x in n.args.defaults))
 f.__kwdefaults__={a.arg:ast.literal_eval(v)for a,v in zip(n.args.kwonlyargs,n.args.kw_defaults)if v is not None}
 return f
prev=doc(PREVIOUS)
yes("exact_previous_independent_SOURCE_PASS",ref(PREVIOUS)["sha256"]=="51680661467218314dcd04c313371d3f8081cda6e2921a7e817b6e99e6ffcf98"and prev["verdict"]=="SOURCE_PASS"and prev["independent"]is True and prev["blockers"]==[])
for n in ("worker_entry_v1.py","close_unit_ROOT_v1.py"):
 yes("full_AST_unchanged_except_namespace_"+n,dump(ast.parse(normalize((ROOT/NEW/n).read_text())))==dump(module(OLD+"/"+n)))
B=NEW+"/batch_ROOT_v1.py";M=NEW+"/source_manifest.v1.json"
oldnodes=nodes(OLD+"/batch_ROOT_v1.py");newnodes=nodes(B)
yes("all_controller_function_names_preserved",set(oldnodes)==set(newnodes))
for name in oldnodes:
 if name in ("wait_wave_capacity","run_batch"):continue
 yes("unchanged_controller_function_"+name,dump(ast.parse(normalize(ast.unparse(newnodes[name]))))==dump(ast.parse(ast.unparse(oldnodes[name]))))
oldtree=module(OLD+"/batch_ROOT_v1.py");newtree=ast.parse(normalize((ROOT/B).read_text()))
oldglobals=[n for n in oldtree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))]
newglobals=[n for n in newtree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))]
yes("controller_global_setup_unchanged_except_namespace",dump(ast.Module(body=oldglobals,type_ignores=[]))==dump(ast.Module(body=newglobals,type_ignores=[])))
m=doc(M);oldm=doc(OLD+"/source_manifest.v1.json")
for rr in m["refs"]:need(ref(rr["path"])==rr)
yes("all_v3_manifest_refs_authenticated",len(m["refs"])==15)
for key in ("allowed_registered_rows","caps_proposed","close_pins","controller_runtime_ref","maximum_open_unit_reservations","maximum_producers","original_binding_ceiling","original_current_guard_roles","primary_scientific_cap_bytes","synchronization_pool","owned_ref","pidfd_ref"):
 yes("manifest_frozen_contract_"+key,m[key]==oldm[key])
for r in m["refs"][3:]:need(r in oldm["refs"])
yes("all_transitive_science_guard_pool_refs_unchanged")
recipe=(ROOT/NEW/"ROOT_RECIPE.v1.md").read_text()
yes("both_real_recipe_commands_use_v3",recipe.count("-m "+NEW.replace("/",".")+".batch_ROOT_v1")==2)
yes("manual_accounting_coordinator_still_explicit","EVERY manual original accounting capture+authority+apply"in recipe and "ROOT must exclusively serialize the initial32MiB admin admission"in recipe)
# The exact capacity loop only sees RAM state/RSS/original-slot counters.
def capacity_case(name,counts,remaining=10**12,ram=10**12,desired=2,expiry=200000,sleep_step=1):
 idx=[0];clock=[0.];events=[]
 ctx=SimpleNamespace(authority_ref={},authority=dict(expires_at_utc=datetime.fromtimestamp(expiry,timezone.utc).isoformat()),w=SimpleNamespace(validate_batch_authority=lambda r:events.append("authority")),g=object(),manifest=m)
 def state(g,mm):
  events.append("state");return(None,None,dict(limit_bytes=remaining,projected_bytes=0))
 def slots(ctx):return counts[min(idx[0],len(counts)-1)]
 def sleep(sec):events.append("queued_sleep");idx[0]+=1;clock[0]+=sleep_step
 env=dict(need=need,datetime=datetime,time=SimpleNamespace(time=lambda:clock[0],sleep=sleep),CANCEL=[None],state=state,available_memory=lambda:ram,available_original_slots=slots,json=json)
 cap=io.StringIO()
 if expiry<=16200 or (all(x==0 for x in counts)and sleep_step>=expiry):
  with redirect_stdout(cap):deny("capacity_finite_stop_"+name,lambda:callable_(B,"wait_wave_capacity",env)(ctx,desired))
  yes("capacity_stop_no_operation_"+name,not any(x in("admit","science")for x in events));return
 with redirect_stdout(cap):r=callable_(B,"wait_wave_capacity",env)(ctx,desired)
 yes("capacity_exact_choice_"+name,r==(remaining,ram,min(desired,counts[-1],remaining//(7000000000+65536),ram//68719476736)))
 if counts[0]==0:
  first=json.loads(cap.getvalue().splitlines()[0]);yes("zero_slot_queued_before_any_admission",first["status"]=="QUEUED_NOT_ADMITTED"and first["no_attempt_consumed"]is True and events[:3]==["authority","state","queued_sleep"])
capacity_case("two_slots",[2])
capacity_case("one_slot",[1])
capacity_case("storage_one",[2],remaining=7000000000+65536)
capacity_case("RSS_one",[2],ram=68719476736)
capacity_case("last_parent_one",[2],desired=1)
capacity_case("zero_then_one",[0,1])
capacity_case("zero_expiry",[0],sleep_step=200000)
capacity_case("no_headroom",[2],expiry=16200)
# Full actual run_batch: exact registered18 and fake only external boundaries.
parents=["american_c3_train_%03d"%i for i in range(2,20)]
def flow(schedule,fail_wave=None,invalid=None):
 events=[];saved={};groupnames=[];open_units=[];admitted=[];waves=[];cursor=[0];wave_index=[0];peak=[0]
 authority=dict(run_directory=NEW+"/ROOT_actual_batch_RAM",scope_id="RAM")
 own=dict(key="RAMadmin",reserved_bytes=33554432)
 inp=dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",authority_ref={},source_manifest_ref=ref(M),storage=dict(reservation=own))
 ir=dict(path="RAM_batch_input",sha256="0"*64)
 class Dir:
  def is_dir(self):return True
  def iterdir(self):return[SimpleNamespace(name="accounting_coordinator.lock")]
 def save(ctx,name,v):
  need(name not in saved,"Exclusive unique records with variable waves");saved[name]=copy.deepcopy(v);return dict(path=name,sha256="0"*64)
 def wait(ctx,wanted):
  need(not open_units,"All previous wave closes accepted before new capacity/admission")
  selected=invalid if invalid is not None else min(wanted,schedule[min(wave_index[0],len(schedule)-1)])
  events.append("capacity:"+str(selected));return(10**12,10**12,selected)
 def group(ctx,commands,seconds,rss,name):
  need(name not in groupnames,"Unique actual group evidence name");groupnames.append(name)
  if ".admission_"in name:
   parent=name.split(".")[0];need(parent not in admitted,"No replay");admitted.append(parent);open_units.append(parent);peak[0]=max(peak[0],len(open_units));need(peak[0]<=2)
   events.append("admit:"+parent);return dict(roots=[dict(stdout=json.dumps(dict(execution_started=False,input_ref=dict(path=parent,sha256="0"*64))))])
  if ".simulation_"in name:
   wave=[cmd[2]for cmd in commands];waves.append(wave);need(1<=len(wave)<=2 and len(wave)==len(open_units)and rss==len(wave)*68719476736)
   events.append("simulate:"+str(len(wave)));wave_index[0]+=1
   if wave_index[0]==fail_wave:raise ValueError("RAM unknown/nonzero child preserved")
   return dict(roots=[dict(stdout="{}")for cmd in commands])
  parent=name.split(".")[0];need(parent in open_units);events.append("close:"+parent);return dict(roots=[dict(stdout="{}")])
 def accept(ctx,parent,r):
  need(parent in open_units);open_units.remove(parent);events.append("accept:"+parent);return dict(path=parent,sha256="0"*64)
 env=dict(ROOT=ROOT,MANIFEST=M,WORKER=NEW+"/worker_entry_v1.py",CLOSE=NEW+"/close_unit_ROOT_v1.py",need=need,
  read=lambda r:inp,qualified=lambda r:(SimpleNamespace(accounting_transaction=lambda rr:nullcontext(),ALLOWED_PARENTS=tuple(parents)),authority,m),
  guard=lambda mm:SimpleNamespace(budget_check_v9=lambda *a,**k:None),state=lambda g,mm:({own["key"]:own},set(),dict(limit_bytes=10**12,projected_bytes=0)),
  local=lambda p:Dir(),put=save,original_owned=lambda mm:object(),SimpleNamespace=SimpleNamespace,
  signal=SimpleNamespace(SIGINT=2,SIGTERM=15,signal=lambda *a:None),CANCEL=[None],ref=ref,wait_wave_capacity=wait,available_memory=lambda:10**12,
  unit_request=lambda *a:{},group_call=group,argv=lambda module,flag,rr:[module,flag,rr["path"]],terminal=lambda r:json.loads(r["stdout"]),
  prepare_close=lambda ctx,parent,ir,sup:dict(path=parent,sha256="0"*64),accept_close=accept,json=json)
 with redirect_stdout(io.StringIO()):r=callable_(B,"run_batch",env)(ir)
 label="_".join(map(str,schedule))
 if invalid is not None:
  yes("controller_rejects_invalid_wave_count_"+str(invalid),r["returncode"]==1 and admitted==[]and "failed_preserved.json"in saved);return
 if fail_wave:
  yes("controller_failure_no_replay_or_later_wave_"+label,r["returncode"]==1 and admitted==waves[0]and not any(x.startswith("close:")for x in events)and "failed_preserved.json"in saved);return
 yes("controller_all18_once_in_order_"+label,r["returncode"]==0 and admitted==parents and saved["completed.json"]["attempted_parents"]==parents and [p for wave in waves for p in wave]==parents)
 yes("controller_all_genuine_closes_before_next_wave_"+label,not open_units and len([x for x in events if x.startswith("accept:")])==18)
 yes("controller_unique_simulation_and_completion_names_"+label,len([n for n in groupnames if n.endswith(".simulation_OS.json")])==len(waves)and len(set(groupnames))==len(groupnames) and len([n for n in saved if n.startswith("wave_")and n.endswith(".completed.json")])==len(waves))
 yes("controller_at_most_two_open_"+label,peak[0]<=2)
 if schedule==[1]:yes("single_slot_eighteen_one_parent_waves",len(waves)==18 and all(len(w)==1 for w in waves))
 if schedule==[2]:yes("two_slot_nine_two_parent_waves",len(waves)==9 and all(len(w)==2 for w in waves))
 if len(schedule)>1:yes("transition_uses_both_one_and_two_without_skipping",any(len(w)==1 for w in waves)and any(len(w)==2 for w in waves))
flow([1]);flow([2]);flow([1,2,1,2,2,1]);flow([1],fail_wave=1);flow([1],invalid=0);flow([1],invalid=3);flow([1],invalid=True)
# Exact unchanged aggregate monitor, actual original OwnedTree class, RAM one root.
def one_root_monitor():
 rows={};reaped=set();pipes={};clock=[0.];signals=[]
 def tick():clock[0]+=.01;return clock[0]
 def row(p):return None if p in reaped else rows.get(p)
 def children(p):return [k for k,v in rows.items()if k not in reaped and v["ppid"]==p]
 def wait(p,f):
  if p in reaped:raise ChildProcessError()
  reaped.add(p);return(p,0)
 class S:
  def fileno(self):return 100
  def close(self):pass
 def popen(cmd,**kw):
  need(kw["start_new_session"]and kw["env"]["CUDA_VISIBLE_DEVICES"]=="")
  rows[100]=dict(pid=100,ppid=1,group=100,session=100,start=1000,state="R");pipes[100]=[b'{}\n',b""]
  return SimpleNamespace(pid=100,stdout=S(),returncode=None)
 def read(p,n):return pipes[p].pop(0)if pipes[p]else b""
 def group(p,s):need(s==0);raise ProcessLookupError()
 osfake=SimpleNamespace(getpid=lambda:1,environ={},read=read,waitpid=wait,WNOHANG=1,waitstatus_to_exitcode=lambda s:s>>8,killpg=group,set_blocking=lambda *a:None,sysconf=lambda s:4096)
 ownedsource=(ROOT/m["owned_ref"]["path"]).read_text();env=dict(os=osfake,_stat=row,_children=children)
 oc=compile(ownedsource,"owned_RAM","exec");env["_same"]=FunctionType(next(c for c in oc.co_consts if isinstance(c,CodeType)and c.co_name=="_same"),env,"_same")
 cl=next(n for n in ast.parse(ownedsource).body if isinstance(n,ast.ClassDef)and n.name=="OwnedTree");exec(compile(ast.Module(body=[cl],type_ignores=[]),"original_OwnedTree_RAM","exec"),env)
 owned=SimpleNamespace(_children=children,_stat=row,_same=env["_same"],OwnedTree=lambda p:env["OwnedTree"](p,owner=1,read_stat=row,read_children=children),signal_identity=lambda r,s:signals.append((r["pid"],s)))
 class P:
  def __new__(cls,v):return SimpleNamespace(read_text=lambda:"0 1")
 e=dict(ROOT=ROOT,need=need,os=osfake,subprocess=SimpleNamespace(Popen=popen,PIPE=1,STDOUT=2),Path=P,time=SimpleNamespace(time=tick,monotonic=tick,sleep=lambda s:clock.__setitem__(0,clock[0]+s)),signal=SimpleNamespace(SIGTERM=15,SIGKILL=9),CANCEL=[None])
 r=callable_(B,"run_group",e)([["RAM"]],owned=owned,deadline_epoch=1000,rss_cap=10**7)
 yes("actual_original_OwnedTree_one_root_real_wait_EOF_groups_RAMLifecycle",r["success"]and len(r["roots"])==1 and r["roots"][0]["actual_OS_wait_status"]==0 and r["roots"][0]["actual_direct_wait"]is True and r["roots"][0]["all_owned_processes_and_groups_drained"]is True and r["UI_tool_terminal_claimed"]is False)
one_root_monitor()
print(json.dumps(dict(schema="TRAIN002_019_v3_independent_scheduler_delta_checks_v1",status="INDEPENDENT_PASS",count=len(checks),checks=checks,
 source_manifest_ref=ref(M),reviewed_sources=[ref(NEW+"/"+n)for n in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py")],
 previous_independent_review_ref=ref(PREVIOUS),actual_guard_imports_configurations_resource_process_slot_GPU_probes_admissions_launches_closes=0,scientific_payload_truth_checkpoint_reads_or_hashes=0,
 actual_scientific_execution=False),sort_keys=True,indent=2))
