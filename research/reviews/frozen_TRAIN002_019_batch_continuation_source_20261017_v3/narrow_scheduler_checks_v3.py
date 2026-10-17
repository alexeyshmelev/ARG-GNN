"""Narrow v3 scheduling delta and RAM integration. No actual operations."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from contextlib import redirect_stdout,nullcontext
import ast,copy,hashlib,io,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OLD="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v2"
REL="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3"
B=REL+"/batch_ROOT_v1.py";M=REL+"/source_manifest.v1.json";CHECKS=[]
def need(ok,msg="Narrow RAM check"):
 if not ok:raise ValueError(msg)
def ref(path):
 return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def fn(path,name,env):
 s=(ROOT/path).read_text();n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),path+"#narrow_RAM","exec"),env);return env[name]
def yes(name):CHECKS.append(name)
def normalized_tree(path):
 text=(ROOT/path).read_text().replace(OLD,"CANONICAL_BATCH").replace(REL,"CANONICAL_BATCH")
 return ast.parse(text)
for name in("worker_entry_v1.py","close_unit_ROOT_v1.py"):
 need(ast.dump(normalized_tree(OLD+"/"+name),include_attributes=False)==ast.dump(normalized_tree(REL+"/"+name),include_attributes=False))
yes("worker_and_original_close_exact_v2_AST_after_namespace_normalization")
old=normalized_tree(OLD+"/batch_ROOT_v1.py");new=normalized_tree(B)
changed=[]
oa={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
na={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
need(set(oa)==set(na))
for name in oa:
 if ast.dump(oa[name],include_attributes=False)!=ast.dump(na[name],include_attributes=False):changed.append(name)
need(changed==["wait_wave_capacity","run_batch"])
yes("only_queue_selection_and_cursor_loop_functions_changed")
manifest=json.loads((ROOT/M).read_text())
for rr in manifest["refs"]:need(ref(rr["path"])==rr)
need(manifest["maximum_producers"]==2 and manifest["maximum_open_unit_reservations"]==2
 and manifest["caps_proposed"]==json.loads((ROOT/OLD/"source_manifest.v1.json").read_text())["caps_proposed"])
yes("all_sources_pinned_original_caps_pool_and_two_open_limit_unchanged")
def queue_case(slots=(1,),capacity=100000000000,ram=200000000000,count=2,cancel=False):
 at=[0];sleep=[];cancelled=[None];checks=[0]
 epoch=datetime.now(timezone.utc).timestamp()
 ctx=SimpleNamespace(w=SimpleNamespace(validate_batch_authority=lambda rr:checks.__setitem__(0,checks[0]+1)),
  authority_ref={},authority=dict(expires_at_utc=datetime.fromtimestamp(epoch+48*3600,timezone.utc).isoformat()),
  g=object(),manifest=manifest)
 def pause(seconds):
  sleep.append(seconds);at[0]+=1
  if cancel:cancelled[0]=15
 env=dict(need=need,CANCEL=cancelled,time=SimpleNamespace(time=lambda:epoch,sleep=pause),
  datetime=datetime,state=lambda *a:(None,None,dict(limit_bytes=capacity,projected_bytes=0)),
  available_memory=lambda:ram,available_original_slots=lambda c:slots[min(at[0],len(slots)-1)],json=json)
 stdout=io.StringIO()
 with redirect_stdout(stdout):
  try:value=fn(B,"wait_wave_capacity",env)(ctx,count)
  except ValueError:
   need(cancel and sleep and "QUEUED_NOT_ADMITTED"in stdout.getvalue());return None,sleep
 return value,sleep
need(queue_case((1,))[0][2]==1);yes("one_original_slot_selects_one_before_admission")
need(queue_case((2,))[0][2]==2);yes("two_original_slots_and_capacity_select_two")
value,slept=queue_case((0,1));need(value[2]==1 and slept==[30]);yes("zero_slots_queue_without_attempt_then_one_actual_available_slot")
need(queue_case((2,),count=1)[0][2]==1);yes("last_single_remaining_parent_selects_one")
need(queue_case((2,),capacity=7000000000+65536)[0][2]==1);yes("existing_storage_fits_one_of_two_selects_one")
need(queue_case((2,),ram=68719476736)[0][2]==1);yes("existing_RSS_fits_one_of_two_selects_one")
need(queue_case((0,),cancel=True)[0]is None);yes("zero_slots_cancel_preserves_not_admitted_no_replay")
def integration(pattern):
 parents=tuple("american_c3_train_%03d"%i for i in range(2,20));events=[];saved={};open_units=set();wave_sizes=[];simnames=[]
 own=dict(key="RAMadmin",reserved_bytes=33554432)
 ir=dict(path="RAM_input.json",sha256="0"*64)
 a=dict(scope_id="RAM",run_directory=REL+"/ROOT_actual_batch_RAM")
 inp=dict(schema="ROOT_TRAIN002_019_batch_admitted_input_v1",source_manifest_ref=ref(M),authority_ref={},storage=dict(reservation=own))
 class Dir:
  def is_dir(self):return True
  def iterdir(self):return[SimpleNamespace(name="accounting_coordinator.lock")]
 def put(ctx,name,value):
  need(name not in saved,"Unique wave/request record");saved[name]=copy.deepcopy(value);return dict(path=name,sha256="0"*64)
 def available(ctx,count):
  need(not open_units,"Prior wave must genuinely close before any new admission")
  n=min(count,pattern[len(wave_sizes)%len(pattern)]);wave_sizes.append(n);return 0,0,n
 def group(ctx,commands,seconds,rss,name):
  if ".admission_"in name:
   parent=name.split(".")[0];need(parent not in open_units);open_units.add(parent);need(len(open_units)<=2)
   events.append("admit:"+parent);result=dict(execution_started=False,input_ref=dict(path=parent,sha256="0"*64))
  elif ".simulation_"in name:
   need(len(commands)in(1,2)and len(commands)==len(open_units));simnames.append(name)
   return dict(roots=[dict(parent=c[0],stdout="{}")for c in commands])
  else:events.append("close:"+name.split(".")[0]);result=dict(status="ORDINARY_CLOSE_APPLIED")
  return dict(roots=[dict(stdout=json.dumps(result))])
 def accept(ctx,parent,result):
  need(parent in open_units);open_units.remove(parent);events.append("accepted:"+parent)
  return dict(path=parent,sha256="0"*64)
 g=SimpleNamespace(budget_check_v9=lambda *a,**k:None)
 env=dict(MANIFEST=M,WORKER=REL+"/worker_entry_v1.py",CLOSE=REL+"/close_unit_ROOT_v1.py",need=need,read=lambda rr:inp,
  qualified=lambda rr:(SimpleNamespace(ALLOWED_PARENTS=parents,accounting_transaction=lambda rr:nullcontext()),a,manifest),
  guard=lambda m:g,state=lambda g,m:({own["key"]:own},set(),dict(limit_bytes=1000000000000,projected_bytes=0)),
  local=lambda p:Dir(),put=put,original_owned=lambda m:object(),SimpleNamespace=SimpleNamespace,
  signal=SimpleNamespace(SIGINT=2,SIGTERM=15,signal=lambda *a:None),CANCEL=[None],ref=ref,
  wait_wave_capacity=available,available_memory=lambda:200000000000,unit_request=lambda ctx,parent,lane,capacity,ram:dict(parent_id=parent),
  group_call=group,argv=lambda module,flag,rr:[rr["path"],module,flag],terminal=lambda r:json.loads(r["stdout"]),
  prepare_close=lambda ctx,parent,ir,sup:dict(path=parent,sha256="0"*64),accept_close=accept,json=json)
 stream=io.StringIO()
 with redirect_stdout(stream):result=fn(B,"run_batch",env)(ir)
 need(result["returncode"]==0 and not open_units and "completed.json"in saved)
 need(saved["completed.json"]["attempted_parents"]==list(parents))
 need([r["parent_id"]for r in saved["completed.json"]["completed"]]==list(parents))
 need(len(simnames)==len(set(simnames))and len(simnames)==len(wave_sizes))
 return wave_sizes
need(integration((1,))==[1]*18);yes("actual_controller_all18_single_slot_waves_once_inorder_unique_records_and_genuine_closes")
need(integration((2,))==[2]*9);yes("actual_controller_all18_two_slot_waves_once_inorder_unique_records_and_genuine_closes")
sizes=integration((1,2));need(sum(sizes)==18 and 1 in sizes and 2 in sizes)
yes("actual_controller_all18_variable_waves_no_offset_skip_or_duplicate_no_more_than_two_open")
recipe=(ROOT/REL/"ROOT_RECIPE.v1.md").read_text()
need(recipe.count(REL.replace("/",".")+".batch_ROOT_v1")==2
 and OLD.replace("/",".")+".batch_ROOT_v1"not in recipe
 and "source_20261017_v1.batch_ROOT_v1"not in recipe)
yes("both_ROOT_recipe_commands_point_to_actual_v3_module")
print(json.dumps(dict(status="NARROW_SOURCE_RAM_PASS",checks=CHECKS,count=len(CHECKS),changed_functions=changed,
 source_manifest_ref=ref(M),source_refs=[ref(REL+"/"+n)for n in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py")],
 actual_imports_guards_probes_requests_admissions_or_science=0),sort_keys=True,indent=2))
