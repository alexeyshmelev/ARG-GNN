"""Native VAL004 ordinary-close author checks. No guard or real operations."""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import ast, contextlib, copy, hashlib, importlib.util, io, json, sys, types

D=Path("research/reviews/FLARE_VAL004_native_current_Census_close_source_20261017_v1")
OLD=Path("research/reviews/FLARE_VAL003_native_current_Census_close_source_20261017_v1")
sp=importlib.util.spec_from_file_location("_inactive_native_VAL004_close_check",D/"close_ROOT_v1.py")
m=importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
positive=[]
negative=[]
def check(ok,label):
    if not ok:raise AssertionError(label)
    positive.append(label)
def reject(fn,label):
    try:fn()
    except (ValueError,KeyError,TypeError,AssertionError):
        negative.append(label)
        return
    raise AssertionError("Accepted "+label)
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
p=m.pins()
joined=m.metadata_join(p)
inp,wait,done,launch,storage,own,pids,groups=joined
check(wait["pid"]==176516 and wait["supervisor_pid"]==175310
      and pids==groups==[175310,176516,178698,181613]
      and launch["ownership"]["owned_identities"]==[dict(pid=176516,start=492327635)],
      "Genuine native worker/start and entire recorded worker/supervisor/child roster")
check(p["producer_driver"]==p["producer_worker"]==inp["root_driver_ref"]==inp["main_source_ref"],
      "Actual native embedded worker shares exact authenticated driver source")
check(digest(own)==p["reservation_sha256"]=="dd677f0fbad49e800cef8c86c9759f576c6a847bc4d6f1a5fe3ed9477ad67e33"
      and own["reserved_bytes"]==2003235369 and len(own["consumer_bindings"])==63,
      "Whole unchanged original2003235369byte reservation and63bindings")
for ref in m.read(m.ref(str(D/"source_manifest.v1.json")))["source_refs"]:
    m.raw(ref)
check(True,"All31deduplicated selected source/immutable-metadata manifest refs authenticate")
with contextlib.redirect_stdout(io.StringIO()) as output:
    status=m.main([])
check(status==0 and json.loads(output.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False),
      "Default actual module entry is inert")
source=(D/"close_ROOT_v1.py").read_text()
delta=json.loads((D/"source_delta.v1.json").read_text())
normal=source
for old,new in reversed(delta["exact_namespace_substitutions"]):
    normal=normal.replace(new,old)
getfn=lambda text,name:ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
for name in ["close_once","current","request_value"]:
    check(getfn(normal,name)==getfn((OLD/"close_ROOT_v1.py").read_text(),name),
          "Original "+name+" whole AST unchanged after exact native substitutions")
check('len(state[3])==47' in source and 'complete==own["consumer_bindings"]and len(complete)<=64' in source
      and 'g._state_locked' in source and 'fresh=True' in source
      and 'g.apply_close_completed_v1' in source,
      "All47returned roles, original63union under64 and unchanged fresh ordinary-close API retained")

real_read=m.read
documents={name:real_read(value)for name,value in p.items()
           if isinstance(value,dict) and set(value)=={"path","sha256"} and name not in
           ["producer_driver","producer_worker","producer_manifest","producer_review","guard","implementation","launcher"]}
def changed(name,fn):
    value=copy.deepcopy(documents[name])
    fn(value)
    return value
cases=[
 ("wait",lambda v:v.update(returncode=1),"native worker wait1"),
 ("wait",lambda v:v.update(actual_OS_wait_status=256),"native raw OSwait nonzero"),
 ("wait",lambda v:v.update(actual_direct_worker_wait=False),"unknown direct worker wait"),
 ("wait",lambda v:v.update(actual_direct_launcher_wait=False),"unknown direct launcher wait"),
 ("wait",lambda v:v.update(all_owned_processes_and_groups_drained=False),"incomplete original owned drain"),
 ("wait",lambda v:v["group_observations"][0].update(absent=False),"nonabsent original worker group"),
 ("wait",lambda v:v.update(supervisor_process_group=175309),"pre-setsid group cannot replace final supervisor group"),
 ("outer",lambda v:v.update(actual_tool_session_id=40430),"wrong consumed outer session"),
 ("outer",lambda v:v.update(actual_exit_code=1),"nonzero consumed outer exit"),
 ("outer",lambda v:v.update(actual_worker_wait_ref={}),"wrong outer worker wait ref"),
 ("outer",lambda v:v.update(actual_direct_wait_ref={}),"wrong outer direct supervisor wait ref"),
 ("outer",lambda v:v.update(actual_direct_worker_returncode=1),"wrong outer worker status"),
 ("completed",lambda v:v.update(status="completed_C3_FLARE_observation_conversion"),"conversion cannot replace native completion"),
 ("completed",lambda v:v.update(query_ancestry_truth_opened=True),"private truth claim rejected"),
 ("direct_supervisor_wait",lambda v:v.update(actual_returncode=1),"nonzero actual direct supervisor wait"),
 ("direct_spawn",lambda v:v.update(pid=175311),"wrong direct supervisor PID"),
 ("direct_spawn",lambda v:v.update(proc_stat="invented alternative stat"),"invented unobserved stat field rejected"),
 ("direct_spawn",lambda v:v.update(actual_Popen=False),"nonactual spawn rejected"),
 ("direct_spawn",lambda v:v["actual_post_Popen_stat_observation"].update(start_jiffies=492310848),"wrong genuine supervisor start"),
 ("direct_spawn",lambda v:v["actual_post_Popen_stat_observation"].update(pgrp=175310),"pre-setsid group must retain genuine175309"),
 ("direct_spawn",lambda v:v["actual_post_Popen_stat_observation"].update(session=175310),"pre-setsid session cannot be relabelled final"),
 ("direct_spawn",lambda v:v["actual_post_Popen_stat_observation"].update(observed_after_Popen=False),"post-Popen observation role required"),
 ("admission_receipt",lambda v:v.update(consumer_binding_count=62),"dropped native admission binding"),
 ("admission_receipt",lambda v:v.update(execution_argv=[]),"wrong admitted execution argv"),
 ("admission_direct_wait",lambda v:v.update(actual_returncode=1),"nonzero actual admission direct wait"),
 ("admission_invocation",lambda v:v.update(ROOT_all_mutations_serialized=False),"unserialized actual admission"),
 ("admission_outer",lambda v:v.update(actual_terminal_consumed=False),"unconsumed actual admission outer"),
 ("admission_outer",lambda v:v.update(actual_tool_session_id=14773),"wrong admission outer session"),
 ("issued_once",lambda v:v.update(actual_once_execution_argv=[]),"wrong issued-once execution argv"),
 ("issued_once",lambda v:v.update(no_automatic_retry=False),"issued-once no-retry condition required"),
 ("count",lambda v:v.update(num_sites=229642),"changed original229643sites"),
 ("conversion",lambda v:v.update(query_haplotypes=999),"reduced original query support"),
 ("conversion",lambda v:v.update(common_focal_records=9999),"reduced original focal support"),
 ("conversion",lambda v:v.update(native_expected_markers=26531),"changed native marker count"),
 ("native_wait",lambda v:v.update(returncode=1),"nonzero original native child wait"),
 ("storage",lambda v:v["reservation"].update(reserved_bytes=2003235368),"changed original reservation cap"),
 ("storage",lambda v:v["reservation"]["consumer_bindings"].pop(),"dropped original consumer binding"),
]
for name,fn,label in cases:
    value=changed(name,fn)
    with patch.object(m,"read",lambda ref,name=name,value=value:
                      copy.deepcopy(value)if ref==p[name]else real_read(ref)):
        reject(lambda:m.metadata_join(p),label)
check("proc_stat"not in documents["direct_spawn"] and "later_supervisor_observation"not in p
      and "issued_once"in p and "admission_outer"in p,
      "Genuine VAL004 post-Popen/issued-once/admission-outer schemas are consumed directly")
post=documents["direct_spawn"]["actual_post_Popen_stat_observation"]
check(post["pid"]==175310 and post["start_jiffies"]==492310847
      and post["pgrp"]==post["session"]==175309
      and documents["wait"]["supervisor_process_group"]==175310 and 175309 not in groups,
      "Pre-setsid group175309 remains separate and is never a final-owned group target")
# Execute only the original pure proof/union definitions with RAM boundaries.
guard_tree=ast.parse(Path(p["guard"]["path"]).read_text())
class FixedDateTime(datetime):
    @classmethod
    def now(cls,tz=None):return cls(2100,1,1,tzinfo=timezone.utc)
env=dict(require=m.need,datetime=FixedDateTime,timezone=timezone,Path=Path,ROOT=m.ROOT,copy=copy)
for name in ["_pointer","_identity","_completed_identity","_proof","consumer_bindings_v1"]:
    node=next(n for n in guard_tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<original_RAM_"+name+">","exec"),env)
def integer(value,label,minimum):
    if type(value)is not int or value<minimum:raise ValueError(label)
env["budget"]=types.SimpleNamespace(digest=digest,v4=types.SimpleNamespace(
    encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),
    integer=integer,canonical=lambda root,path:Path(path)if Path(path).is_absolute()else root/path))
obsref=dict(path="RAM/native_VAL004_observation.json",sha256="6"*64)
rows=[dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc="2099-12-31T00:00:00+00:00")for pid in pids]
obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
    key=own["key"],reservation_sha256=p["reservation_sha256"],storage_ref=p["storage"],
    launch_ref=p["launch"],direct_wait_ref=p["wait"],completed_ref=p["completed"],
    launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,
    direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,
    preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=rows)
ram={json.dumps(p[name],sort_keys=True):copy.deepcopy(value)for name,value in
     [("storage",storage),("launch",launch),("wait",wait),("completed",done)]}
ram[json.dumps(obsref,sort_keys=True)]=obs
env["_json"]=lambda ref:copy.deepcopy(ram[json.dumps(ref,sort_keys=True)])
env["_completed_json"]=env["_json"]
env["_absent"]=lambda pid:copy.deepcopy(next(row for row in rows if row["pid"]==pid))
request=m.request_value(p,own,launch,obsref)
for observe in [False,True]:
    proof=env["_proof"](request,own,observe=observe)
    check(proof["pids"]==pids and proof["completion_verified"]is True,
          "Exact original ordinary proof accepts genuine native metadata in RAM observe="+str(observe))
for label,mutate in [
    ("wrong reservation digest",lambda v:v.update(reservation_sha256="0"*64)),
    ("changed original direct wait identity",lambda v:v["wait_identity"].update(returncode=1)),
    ("changed native completed identity",lambda v:v["completed_identity"].update(status="completed_wrong")),
]:
    bad=copy.deepcopy(request)
    mutate(bad)
    reject(lambda bad=bad:env["_proof"](bad,own),label)
obkey=json.dumps(obsref,sort_keys=True)
old=copy.deepcopy(ram[obkey])
ram[obkey]["observations"][0]["absent"]=False
reject(lambda:env["_proof"](request,own),"original proof rejects non-absent RAM target")
ram[obkey]=copy.deepcopy(old)
ram[obkey]["observations"][0]["checked_utc"]="2101-01-01T00:00:00+00:00"
reject(lambda:env["_proof"](request,own),"original nonfuture-only timestamp rejects future RAM observation")
ram[obkey]=old
allow={json.dumps(ref,sort_keys=True)for ref in own["consumer_bindings"]}
extras=[dict(path="RAM/extra_"+str(i),sha256=str(i+1)*64)for i in range(2)]
allow.update(json.dumps(ref,sort_keys=True)for ref in extras)
def raw_ram(ref):
    if json.dumps(ref,sort_keys=True)not in allow:raise ValueError("Unknown RAM ref")
    return b"RAM_only"
env["_raw"]=raw_ram
union=env["consumer_bindings_v1"](own["consumer_bindings"],own["consumer_bindings"][:47])
check(union==own["consumer_bindings"] and len(union)==63,
      "Original exact-ref union deduplicates63with47RAM subset, no current role-membership claim")
check(len(env["consumer_bindings_v1"](own["consumer_bindings"],extras[:1]))==64,
      "Original union permits64total refs")
reject(lambda:env["consumer_bindings_v1"](own["consumer_bindings"],extras),"Original union rejects65refs")
check(not any(name.startswith("_ROOT_FLARE_VAL004_native_exact_current_Census_close")for name in sys.modules),
      "No current guard module imported")
result=dict(schema="native_VAL004_ordinary_close_author_RAM_checks_v1",verdict="AUTHOR_SOURCE_CHECKS_ONLY",
    positive_count=len(positive),negative_count=len(negative),checks=positive,negative_cases=negative,
    actual_immutable_metadata_join=True,exact_original_proof_RAM_observe_both=True,
    fixed_2100_fixture_time=True,current_state_or_absence_asserted=False,
    all47_runtime_role_requirement_preserved=True,actual_current47roles_read=False,
    guard_imports=0,actual_process_or_lock_or_state_or_measurement_operations=0,
    close_or_admission_operations=0,scientific_payloads_opened=0,truth_reads=0,Git_actions=0)
with(D/"source_checks.result.v1.json").open("x")as stream:
    stream.write(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in ["checks","negative_cases"]},sort_keys=True))
