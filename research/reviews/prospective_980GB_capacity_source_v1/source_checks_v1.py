"""Focused source/mocked accounting semantics; never configure a real guard."""
from pathlib import Path
from types import SimpleNamespace as NS
from datetime import datetime,timezone
import ast,copy,hashlib,json,importlib.util,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/prospective_980GB_capacity_source_v1"
OLD=ROOT/"research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py"
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(name,call,kind=ValueError):
    try:call()
    except kind:checks.append(name);return
    raise AssertionError(name+" accepted")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def integer(v,label,minimum=0):
    if type(v) is not int or v<minimum:raise ValueError(label)
    return v
def canonical(root,path):
    p=Path(path);return p if p.is_absolute() else root/p
def overlap(a,b):return a==b or a in b.parents or b in a.parents
def no_probe(*args):raise AssertionError("Actual process probe forbidden in tests")
def fake_ref(p):return {"path":str(p),"sha256":hashlib.sha256(str(p).encode()).hexdigest()}
source=(HERE/"prospective_980gb_v1.py").read_text();old_source=OLD.read_text()
tree=ast.parse(source)
def fn_dump(text,name):
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.dump(node,include_attributes=False)
check("current-v3 source remains exact",sha(OLD)=="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
unchanged=["_proof","_completed_identity","_identity","_absent","_measurement","check_prefix","assert_open","check_storage_locked_v1","validate_storage_v9","PilotOutputV9"]
for name in unchanged:check("original proof/storage/writer AST unchanged "+name,fn_dump(source,name)==fn_dump(old_source,name))
budget=NS(v4=NS(integer=integer,canonical=canonical,paths_overlap=overlap,read_json=lambda *a:({"jobs":{}},None),LEGACY="mock-legacy",encoded=encoded),
    SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py",digest=lambda v:hashlib.sha256(encoded(v)).hexdigest())
ns={"Path":Path,"copy":copy,"hashlib":hashlib,"json":json,"stat":NS(),"io":NS(),"datetime":datetime,"timezone":timezone,
    "os":NS(kill=no_probe,environ={}),"budget":budget}
# AST module has no imports or default CLI branch; every external accounting/IO
# dependency below is explicitly replaced. No production module is imported.
nodes=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef,ast.ClassDef))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<mocked-prospective-guard>","exec"),ns)
ns["_ref"]=fake_ref;ns["_raw"]=lambda r:r
check("separate source mode and journal",ns["MODE"]=="prospective_980gb_v1" and ns["JOURNAL"]==".runtime/storage/prospective_980gb_v1" and ns["SOURCE"]!=str(OLD.relative_to(ROOT)))
check("fixed950 warning980 ceiling1TB user limit",(ns["WARNING_BYTES"],ns["CEILING_BYTES"],ns["HUMAN_LIMIT_BYTES"])==(950000000000,980000000000,1000000000000))
for value,expected in [(949999999999,False),(950000000000,True),(979999999999,True)]:
    w=ns["warning_fields"](value)
    check("explicit warning at "+str(value),w["warning_active"] is expected and bool(w["warning_message"]) is expected)
    check("remaining safety arithmetic at "+str(value),w["remaining_to_human_limit_bytes"]==1000000000000-value)
check("charge includes bootstrap and every event",ns["charge"](900000000000,1234,500,2)==900000000000+1234-500+1048576+2*524288)
check("above950 future charge is allowed with warning",ns["charge"](960000000000,0,0,0)>950000000000)
reject("exact980 charge rejected",lambda:ns["charge"](980000000000-1048576,0,0,0))
reject("over980 charge rejected",lambda:ns["charge"](980000000000,0,0,0))
reject("negative metadata/cap rejected",lambda:ns["charge"](900000000000,-1,0,0))
old_bindings={"completed_overlay":ns["BASE_REF"],"completed_policy":ns["BASE_POLICY_REF"],"marker":fake_ref("marker")}
ns["_POLICY_REF"]=fake_ref("new-policy");new_bindings=ns["_bindings"](old_bindings,{})
check("old binding dictionary never overwritten",all(new_bindings[k]==v for k,v in old_bindings.items()) and old_bindings=={"completed_overlay":ns["BASE_REF"],"completed_policy":ns["BASE_POLICY_REF"],"marker":fake_ref("marker")})
check("four distinct new source/policy/base bindings",set(new_bindings)-set(old_bindings)=={"prospective_980_overlay","prospective_980_policy","immutable_completed_v3","immutable_completed_v3_policy"})
reject("new binding collision rejected",lambda:ns["_bindings"]({"prospective_980_overlay":fake_ref("x")},{}))
# Policy validation is exact, including immutable v3 policy and constants.
base_view={"projected_bytes":949889515999,"limit_bytes":950000000000,"hard_limit_bytes":1000000000000,"accounting_mode":ns["BASE_MODE"],
    "base_pins":{"old-journal":"frozen"},"completed_overlay_prefix":{"events":106,"tail_sha256":"base106"},
    "new_reserved_bytes":119432142080,"completed_credit_bytes":119395652522}
base_pins=ns["_pins"](base_view)
policy=ns["activation_policy_draft_v1"](repair_authority_ref=fake_ref("repair"),base_pins=base_pins,base_projected_bytes=base_view["projected_bytes"],
    implementation_ref=fake_ref(budget.SOURCE_REL),launcher_ref=fake_ref(budget.LAUNCHER_REL))
check("inactive root policy draft binds exact old reader and policy",policy["active"] is False and policy["base_ref"]==ns["BASE_REF"] and policy["base_policy_ref"]==ns["BASE_POLICY_REF"])
policy["active"]=True;ns["_REPAIR_AUTHORITY_REF"]=fake_ref("repair");ns["_json"]=lambda _:policy
check("exact active mock policy validates",ns["_policy"]()==policy)
for key,value in [("scheduling_ceiling_bytes",990000000000),("warning_threshold_bytes",960000000000),("active",False),("no_rebase",False)]:
    original=policy[key];policy[key]=value
    reject("policy drift "+key+" rejected",ns["_policy"]);policy[key]=original
# Configure only with mocked module loader; explicit old policy must win over new env.
calls=[]
base_mod=NS(configure_v2=lambda repair,**kw:calls.append((repair,kw)))
loader=NS(exec_module=lambda _:None)
ns["importlib"]=NS(util=NS(spec_from_file_location=lambda *a:NS(loader=loader),module_from_spec=lambda _:base_mod))
ns["_POLICY_REF"]=None;ns["os"].environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(fake_ref("new-policy"))
ns["configure_v2"](fake_ref("repair"))
check("base configuration receives original policy explicitly",calls==[(fake_ref("repair"),{"policy_ref":ns["BASE_POLICY_REF"]})])
reject("configured consumer cannot swap new policy",lambda:ns["configure_v2"](fake_ref("repair"),policy_ref=fake_ref("different-policy")))
# Entire current-base prefix must remain exact; no real ledger/scope access.
old_own={"key":"old/live","output":str(ROOT/"mock/old-live"),"auxiliary_scopes":[],"reserved_bytes":14000000000,"consumer_bindings":[]}
closed_own={"key":"old/closed","output":str(ROOT/"mock/old-closed"),"auxiliary_scopes":[],"reserved_bytes":2000000000,"consumer_bindings":[]}
base_res={x["key"]:x for x in [old_own,closed_own]}
base_state=[base_res,{"old/closed"},base_view,old_bindings,[],[],{}]
base=NS(_state_locked=lambda *a,**kw:base_state,_pins=ns["_pins"],
    _paths=lambda own:[Path(own["output"]),*[Path(p) for p in own.get("auxiliary_scopes",[])]])
ns["_BASE"]=base;ns["_POLICY_REF"]=fake_ref("new-policy");ns["_policy"]=lambda:policy
ns["_events"]=lambda _: ([],[])
first=ns["_state_locked"](policy["implementation_ref"]["sha256"],policy["launcher_ref"]["sha256"],fresh=True)
check("all original reservations and closed keys retained",first[0]==base_res and first[1]=={"old/closed"})
check("base projection charged exactly once plus bootstrap",first[2]["projected_bytes"]==base_view["projected_bytes"]+1048576)
check("old counters disclosed while child counters start zero",first[2]["ancestor_new_reserved_bytes"]==base_view["new_reserved_bytes"] and first[2]["ancestor_completed_credit_bytes"]==base_view["completed_credit_bytes"] and first[2]["new_reserved_bytes"]==first[2]["completed_credit_bytes"]==0)
first[0]["old/live"]["reserved_bytes"]=1
check("mock replay never mutates base reservation objects",base_res["old/live"]["reserved_bytes"]==14000000000)
original=base_view["completed_overlay_prefix"]
base_view["completed_overlay_prefix"]={"events":107,"tail_sha256":"changed"}
reject("ancestor prefix drift rejected even before new admission",lambda:ns["_state_locked"](policy["implementation_ref"]["sha256"],policy["launcher_ref"]["sha256"]))
base_view["completed_overlay_prefix"]=original
original=base_view["projected_bytes"];base_view["projected_bytes"]+=1
reject("ancestor projection drift rejected",lambda:ns["_state_locked"](policy["implementation_ref"]["sha256"],policy["launcher_ref"]["sha256"]))
base_view["projected_bytes"]=original
# Replay a genuine-shaped new reserve with only mock authority/record dependencies.
bindings=ns["_bindings"](old_bindings,policy)
new_own={"key":"new/VAL","output":str(ROOT/"mock/new-val"),"auxiliary_scopes":[],"reserved_bytes":29000000000,"consumer_bindings":list(bindings.values())}
ar=fake_ref("mock-reserve-authority")
auth={"request":new_own,"expected_before":{"base_pins":policy["base_pins"],"completed_overlay_prefix":{"events":0,"tail_sha256":None}}}
event={"authority":ar,"operation":"reserve","value":dict(new_own,root_admission=ar)}
ns["_authority"]=lambda *a,**kw:auth;ns["_events"]=lambda _:([event],["newhash1"])
state=ns["_state_locked"](policy["implementation_ref"]["sha256"],policy["launcher_ref"]["sha256"])
check("29GB new reserve fits980 while preserving charged base",state[2]["projected_bytes"]==949889515999+29000000000+1048576+524288)
check("new reserve warning explicit with20GB safety",state[2]["warning_active"] is True and state[2]["remaining_to_human_limit_bytes"]>20000000000)
check("original caps remain full after new reserve",state[0]["old/live"]["reserved_bytes"]==14000000000 and "old/closed" in state[1])
new_own["reserved_bytes"]=31000000000;event["value"]["reserved_bytes"]=31000000000
reject("new cap exceeding980 rejected during replay",lambda:ns["_state_locked"](policy["implementation_ref"]["sha256"],policy["launcher_ref"]["sha256"]))
new_own["reserved_bytes"]=29000000000;event["value"]["reserved_bytes"]=29000000000
# Unchanged close predicate against fully mocked exact metadata: no signal operation.
proof_ns=dict(ns);proof_ns["_json"]=lambda ref:records[ref["path"]];proof_ns["_completed_json"]=lambda ref:records[ref["path"]]
proof_ns["_raw"]=lambda r:r
cr=fake_ref(str(ROOT/"mock/new-val/completed.json"));sr=fake_ref("mock-storage");lr=fake_ref("mock-launch");wr=fake_ref("mock-wait");orr=fake_ref("mock-observation")
request={"key":new_own["key"],"reservation_sha256":budget.digest(new_own),"storage_ref":sr,"storage_locator":[],
 "launch_ref":lr,"launch_pid_locator":["pid"],"launch_identity":{"pid":501},"direct_wait_ref":wr,"wait_pid_locator":["pid"],"wait_exit_locator":["returncode"],
 "wait_identity":{"pid":501,"returncode":0,"actual_direct_wait":True},"completed_ref":cr,"completed_identity":{"status":"completed_mock"},"observation_ref":orr}
obs={"schema":"concurrent_completed_scope_targeted_root_observation_v1","root_authorizer":"root_agent","key":new_own["key"],
 "reservation_sha256":budget.digest(new_own),"storage_ref":sr,"launch_ref":lr,"direct_wait_ref":wr,"completed_ref":cr,"launcher_pid":501,
 "all_root_owned_launcher_supervisor_worker_pids_listed":True,"direct_wait_observed":True,
 "completed_scope_all_old_original_resume_write_capabilities_revoked":True,"preserve_all_files":True,"external_NFS_writers_outside_observation":True,
 "pids":[501],"observations":[{"pid":501,"absent":True,"method":"targeted os.kill(pid,0) ESRCH","checked_utc":"2000-01-01T00:00:00+00:00"}]}
records={sr["path"]:{"reservation":new_own,"reserved_new_bytes":new_own["reserved_bytes"],"output_directory":new_own["output"]},
 lr["path"]:{"pid":501},wr["path"]:{"pid":501,"returncode":0,"actual_direct_wait":True},cr["path"]:{"status":"completed_mock"},orr["path"]:obs}
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_proof")
exec(compile(ast.Module(body=[node],type_ignores=[]),"<mock-proof>","exec"),proof_ns)
check("unchanged close predicate accepts exact mock wait0 proof",proof_ns["_proof"](request,new_own)["completion_verified"] is True)
records[wr["path"]]["returncode"]=1
reject("wait1 still cannot credit",lambda:proof_ns["_proof"](request,new_own))
records[wr["path"]]["returncode"]=0
obs["completed_scope_all_old_original_resume_write_capabilities_revoked"]=False
reject("unrevoked scope still cannot credit",lambda:proof_ns["_proof"](request,new_own))
obs["completed_scope_all_old_original_resume_write_capabilities_revoked"]=True
bad=copy.deepcopy(request);bad["completed_ref"]=fake_ref(str(ROOT/"mock/new-val/expanded_parent_completed.json"))
reject("unsupported original H038 filename remains rejected",lambda:proof_ns["_proof"](bad,new_own))
# Original full-science configs and metadata-only VAL command binding.
old_packet=ROOT/"research/reviews/American_C3_VAL30mb_shared_source_v1"
for name in ["simulation.inactive.json","inference.inactive.json","native_paths.inactive.json","shared_contract.inactive.json","storage_forecast.json"]:
    check("VAL science/contract bytes unchanged "+name,(HERE/name).read_bytes()==(old_packet/name).read_bytes())
sp=importlib.util.spec_from_file_location("_new_VAL_IO_metadata_only",HERE/"val_root_packet_980_v1.py")
val=importlib.util.module_from_spec(sp);sp.loader.exec_module(val)
check("VAL default plan has no invented active policy",val.stage_plan("simulate")["activation_policy_ref"] is None)
check("VAL simulation still full30Mb original seeds",val.configuration("simulate")["sequence_length_bp"]==30000000 and val.configuration("simulate")["ancestry_seed"]==31001)
reject("VAL admit blocked before active root policy",lambda:val.admit("simulate",None))
pins=val.frozen();stage="simulate";packet,out,job=val.identity(stage)
spec={"parent_id":val.PARENT,"stage":stage,"job_id":job,"source_ref":pins[val.PRODUCERS[stage]],"execution_authorized":True,"GPU_authorized":False,
 "completed_overlay_policy_ref":{"path":val.POLICY,"sha256":"actual-root-policy-placeholder-for-mock-only"}}
val.frozen=lambda:pins
val.read=lambda _:spec
cmd=val.command(stage,{"path":str((packet/"admission.json").relative_to(ROOT)),"sha256":"mock"})
check("VAL child receives exact actual prospective policy env",json.loads(cmd["environment"]["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==spec["completed_overlay_policy_ref"])
check("VAL scientific worker module unchanged","arggnn.american_c3_simulate_v1" in cmd["argv"])
# Root activation source remains source-only by default; an inactive barrier fails
# before any base inspection or import when tested with mocked metadata.
act_source=(HERE/"root_activate_980_v1.py").read_text();at=ast.parse(act_source)
barrier_node=next(n for n in at.body if isinstance(n,ast.FunctionDef) and n.name=="barrier")
fake_barrier=json.loads((HERE/"activation_barrier.inactive.json").read_text())
act_ns={"read":lambda _:fake_barrier,"require":ns["require"],"ref":fake_ref,"GUARD":ns["SOURCE"]}
exec(compile(ast.Module(body=[barrier_node],type_ignores=[]),"<inactive-barrier>","exec"),act_ns)
reject("inactive activation barrier rejected",lambda:act_ns["barrier"]({}))
for path in HERE.glob("*.py"):compile(path.read_text(),str(path),"exec")
check("all new Python sources compile",True)
check("no production guard or scientific module imported",all(x not in sys.modules for x in ["numpy","torch","tskit","_root_prospective980_actual_v3_base","_prospective980_immutable_completed_v3"]))
check("original guard and original VAL packet untouched",sha(OLD)=="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef" and sha(old_packet/"root_packet_v1.py")=="8043262b3c1f15b7a26a0af270d1f6dd163baecf02158b25705e749e3f61621e")
result={"schema":"prospective980_focused_source_and_mocked_semantics_v1","status":"PASS","checks_passed":len(checks),"checks":checks,
 "actual_guard_configuration_or_state_inspection":False,"actual_journal_read_or_mutation":False,"actual_admission_credit_process_GPU_or_scientific_operation":False,
 "guard_source_sha256":sha(HERE/"prospective_980gb_v1.py"),"VAL_packet_source_sha256":sha(HERE/"val_root_packet_980_v1.py"),
 "root_activation_source_sha256":sha(HERE/"root_activate_980_v1.py")}
(HERE/"source_checks_result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({"checks_passed":len(checks),"result_sha256":sha(HERE/"source_checks_result.json")}))
