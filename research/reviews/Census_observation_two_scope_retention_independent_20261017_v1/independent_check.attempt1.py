"""Independent source-only Census review. Audited author RAM fixture baseline plus independently authored boundary cases.
No real guard imports/state, process probes, scanner, admission or resource operations.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,contextlib,copy,hashlib,io,json,types,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/Census_observation_two_scope_retention_source_20261017_v1"
AUTHOR_REF={"path":str((D/"ram_integration_check_v1.py").relative_to(ROOT)),"sha256":"316c7bed1768aff7e244e23143b49e617ea01a4940f200b041942c2a7e160349"}
raw=(ROOT/AUTHOR_REF["path"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==AUTHOR_REF["sha256"]
# Reuse only the reviewed RAM execution infrastructure and its disclosed baseline checks.
# Its complete module import replacement is the single project budget import.
g={"__name__":"independent_Census_audited_fixture_baseline"}
out=io.StringIO()
with contextlib.redirect_stdout(out):exec(compile(raw,"<audited_source_RAM_baseline>","exec"),g)
baseline=json.loads(out.getvalue())
e=g["env"];helper=g["helper"];fs=g["measurefs"];ramos=g["ram_os"];manifest=g["manifest"]
checks=[];negatives=[];real_hashes=[];deferred=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(call,label,types_=(ValueError,KeyError,TypeError,AssertionError,FileNotFoundError,PermissionError,OSError)):
 try:call()
 except types_ as err:negatives.append({"case":label,"exception":type(err).__name__});return
 raise AssertionError("Accepted invalid independent fixture: "+label)
def mutate(record,path,value,call,phase=False):
 field="_phase_json" if phase else "_json";original=e[field]
 def reader(r):
  obj=copy.deepcopy(original(r))
  if r==record:
   target=obj
   for k in path[:-1]:target=target[k]
   target[path[-1]]=copy.deepcopy(value)
  return obj
 e[field]=reader
 try:reject(call,"metadata:"+record["path"]+":"+str(path))
 finally:e[field]=original
def reset_journal():
 g["journal"].clear();g["directories"].clear();g["clock"].value=0.0
def apply(op):
 return e["apply_retire_stopped_v1"](op,expected_source_sha256=g["S"],expected_launcher_sha256=g["L"])
def state():return e["_state_locked"](g["S"],g["L"],fresh=True)
def ref(path):
 raw=path.read_bytes();return {"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(raw).hexdigest()}
# Independent pin closure. Do not open the mutable original ledger or any payload.
pinmap={p:h for p,h in manifest["source_sha256"].items()}
for record in manifest["runtime_authenticated_historical_source_and_metadata_refs"]:
 if record["path"] in pinmap:check(pinmap[record["path"]]==record["sha256"],"No conflicting repeated reference "+record["path"])
 pinmap[record["path"]]=record["sha256"]
for path,sha in pinmap.items():
 if path.startswith(".runtime/storage/"):
  deferred.append({"path":path,"reason":"Original live/guard state is outside source-only review; runtime pin remains required"});continue
 p=ROOT/path
 check(p.suffix in (".py",".json",".md",".sh",".patch"),"Only source/immutable JSON metadata hash "+path)
 check(p.stat().st_size<=33554432,"Bounded metadata/source "+path)
 check(ref(p)["sha256"]==sha,"Actual source/metadata SHA "+path);real_hashes.append(path)
for t in manifest["targets"].values():
 for field in ("phase_evidence_ref","inspection_ROOT_freeze_ref"):
  p=ROOT/t[field]["path"];check(p.stat().st_size<=25165824,"Actual phase JSON within24MiB "+field)
  check(ref(p)==t[field],"Exact original nonstandard evidence pin "+t["phase"]+":"+field)
check(len(manifest["inherited_parent_source_pins_exact"])==17,"Exact17 inherited source pins")
check(len(manifest["inherited_parent_metadata_refs_exact"])==73,"Exact73 inherited ordered refs")
check(baseline["parent_roles_preserved"]==46 and len(g["parentbindings"])==46,"Actual46 role metadata snapshot")
check(baseline["actual_guard_imports_or_state_calls"]==0,"Audited baseline no real guard/state")
source=(D/"census_observation_retention_v1.py").read_text()
tree=ast.parse(source)
check(all(n.module!="torch" for n in tree.body if isinstance(n,ast.ImportFrom)),"No top-level numerical import")
# Source-inspect unchanged v4 no-follow loader without importing it.
v4src=(ROOT/"src/arggnn/project_budget_v4.py").read_text()
v4tree=ast.parse(v4src)
loader_nodes=[n for n in v4tree.body if isinstance(n,ast.FunctionDef)and n.name in ("file_bytes","bound")]
check(len(loader_nodes)==2,"Original v4 loader definitions located without import")
loader_text="\n".join(ast.get_source_segment(v4src,n)for n in loader_nodes)
check("O_NOFOLLOW" in loader_text and "32" in loader_text,"Original no-follow32MiB loader retained")
# Exact review, policy and prerequisite business predicates run on supplied decoded metadata.
for path,value in [
 (["independent"],False),(["verdict"],"SOURCE_BLOCKED"),(["blockers"],["independent-blocker"]),
 (["source_ref"],e["BASE_REF"]),(["source_manifest_ref"],e["BASE_POLICY_REF"]),
 (["measurement_source_ref"],e["BASE_REF"])]:
 mutate(g["review"],path,value,e["_policy"])
for key,row in e["PREREQUISITE_ADMISSIONS"].items():
 tr=row["terminal_ref"]
 if key=="H058v5":
  for field,value in [("actual_admission_tool_session_id",72069),("actual_admission_exit_code",True),
   ("actual_admission_exit_code",1),("actual_admission_terminal_consumed",False),("admitted_ref",e["BASE_REF"])]:
   mutate(tr,[field],value,e["_policy"])
 else:
  for field,value in [("tool_session_id",50132),("outer_exit_code",False),("outer_exit_code",1),
   ("admission_only",False),("scientific_completion_claimed",True)]:
   mutate(tr,[field],value,e["_policy"])
 for field,value in [("consumer_binding_count",65),("input_ref",e["BASE_REF"])]:
  mutate(row["admitted_ref"],[field],value,e["_policy"])
 for field,value in [("reserved_new_bytes",row["reserved_bytes"]-1),("storage_policy_selector_ref",e["BASE_REF"])]:
  mutate(row["storage_ref"],[field],value,e["_policy"])
mutate(g["preref"],["released_at_utc"],"2026-10-11T00:00:00+00:00",e["_policy"])
mutate(g["preref"],["active"],False,e["_policy"])
for field,value in [("active",False),("scheduling_ceiling_bytes",1000000000000),("base_selector_ref",e["BASE_REF"])]:
 mutate(g["policyref"],[field],value,e["_policy"])
for field,value in [("active",False),("accounting_mode","fallback"),("source_ref",e["BASE_REF"]),
 ("frozen_parent_lineage",{}),("no_fallback_discovery_upgrade_or_in_run_swap",False)]:
 mutate(g["selectorref"],[field],value,lambda:e["_selected_record"](g["selectorref"]))
# Real phase/wait/provenance checks with original evidence.json, never synthesized completed.json.
for t in manifest["targets"].values():
 for field,value in [("returncode",False),("worker_start_ticks",0),("worker_pid",2)]:
  mutate(t["direct_worker_wait_ref"],[field],value,lambda t=t:e["_historical_stop"](t))
 for field,value in [("fixture_mode",True),("biological_targets_opened",True),("fit_performed",True),
  ("GPU_used",True),("expected_cells",31),("cells",[]),("admission_ref",e["BASE_REF"])]:
  mutate(t["phase_evidence_ref"],[field],value,lambda t=t:e["_historical_stop"](t),True)
 bad=copy.deepcopy(t);bad["historical_facts_preserved"]["historical_descendant_drain"]="COMPLETE"
 reject(lambda bad=bad:e["_historical_stop"](bad),"Cannot promote historical drain "+t["phase"])
# Actual finite operation and known-target predicates, with IO-only process doubles.
reset_journal();keys=list(manifest["target_order"]);op,req=g["prepare_operation"](keys[0])
for record,field,value in [(req["revocation_ref"],"active",False),
 (req["revocation_ref"],"unknown_historical_descendants_absent_claimed",True),
 (req["revocation_ref"],"preserve_all_files_and_namespaces",False),
 (req["observation_ref"],"PID_observations",[]),(req["observation_ref"],"group_observations",[])]:
 mutate(record,[field],value,lambda:apply(op))
# A valid-chronology but >120s old proof must fail freshness.
def stale_reader(record):
 obj=copy.deepcopy(original_json(record))
 if record==req["revocation_ref"]:obj["revoked_at_utc"]=g["stamp"](-300)
 if record==req["observation_ref"]:
  obj["observed_at_utc"]=g["stamp"](-121)
  for f in ("PID_observations","group_observations"):
   for row in obj[f]:row[-1]=g["stamp"](-121)
 return obj
original_json=e["_json"];e["_json"]=stale_reader
try:reject(lambda:apply(op),"Stale121s proof with valid revocation chronology")
finally:e["_json"]=original_json
for name in ("kill","killpg"):
 original=getattr(ramos,name)
 def denied(*args):raise PermissionError("RAM denied metadata/probe")
 setattr(ramos,name,denied)
 try:reject(lambda:apply(op),"Unexpected permission in "+name+" fails closed")
 finally:setattr(ramos,name,original)
original_mutex=g["v4"].mutex
@contextlib.contextmanager
def delayed_mutex(root):
 g["clock"].value=301.0;yield
delayed_mutex.__module__=g["v4"].__name__;g["v4"].mutex=delayed_mutex
try:reject(lambda:apply(op),"Pre-mutex300s deadline includes mutex wait")
finally:g["v4"].mutex=original_mutex;g["clock"].value=0.0
check(not g["journal"],"Rejected metadata/process/deadline cases append zero events")
# Unchanged strict accounting/cap and direct binding operands.
check(e["charge"](e["CEILING_BYTES"]-e["BOOTSTRAP_CHARGE"]-1,0,0,0)==e["CEILING_BYTES"]-1,"Ceiling minus one byte accepts")
reject(lambda:e["charge"](e["CEILING_BYTES"]-e["BOOTSTRAP_CHARGE"],0,0,0),"Exact980GB blocked")
reject(lambda:e["charge"](0,True,0,0),"Boolean accounting operand blocked")
refs=[{"path":"RAM/bound64/"+str(i),"sha256":"b"*64}for i in range(65)]
for r in refs:g["records"][r["path"]]=r
check(len(e["consumer_bindings_v1"](refs[:64],refs[:64]))==64,"Exact64 refs deduplicate without dropping roles")
reject(lambda:e["consumer_bindings_v1"](refs),"65 actual bound refs blocked")
# Full operation -> full scanner -> actual append -> actual journal replay, independently asserted.
receipts=[]
for key in keys:
 op,req=g["prepare_operation"](key);receipt=apply(op);receipts.append(receipt)
 check(receipt["credit_bytes"]==20000000000-receipt["retained_bytes"],"Exact unused allowance "+key)
 facts=receipt["value"]["proof"]["historical_facts_preserved"]
 check(facts["direct_worker_wait_returncode"]==0 and facts["supervisor_outer_OS_wait"]=="UNKNOWN"
  and facts["historical_descendant_drain"]=="UNKNOWN","Original success and historical unknowns preserved "+key)
 check(receipt["scientific_success_asserted"]is False and receipt["old_scope_mutated"]is False
  and receipt["artifacts_deleted"]==0,"No science/files changed "+key)
 reject(lambda op=op:apply(op),"Consumed immutable prefix/key replay "+key)
final=state()
check(len(final[4])==2 and final[2]["completed_credit_bytes"]==sum(r["credit_bytes"]for r in receipts),"Exactly two retained events replay")
check(len(final[3])==47 and all(final[3][k]==v for k,v in g["parentbindings"].items()),"All46 current parent roles remain exact plus one")
for receipt in receipts:
 own=manifest["targets"][receipt["key"]]["original_reservation"];good=receipt["value"]["measurement"]
 for field,value in [("conservative_scope_bytes",own["reserved_bytes"]+1),("entries",0),
  ("scope_paths",["/dummy"]),("regular_payloads_opened",True),("inventory_sha256","not-a-digest")]:
  bad=copy.deepcopy(good);bad[field]=value
  reject(lambda bad=bad,own=own:e["_retained_measurement"](bad,own),"Invalid retained certificate:"+field)
 bad=copy.deepcopy(good);bad["per_scope"][0]["node_types"]["socket"]=1
 reject(lambda bad=bad,own=own:e["_retained_measurement"](bad,own),"Any socket summary blocks")
 proof=receipt["value"]["proof"]
 for field,value in [("unknown_historical_descendants_absent_claimed",True),("scientific_success_relabelled",True)]:
  bad=copy.deepcopy(proof);bad[field]=value
  reject(lambda bad=bad,proof=proof,receipt=receipt:e["_recorded_stopped_proof"](bad,proof,receipt["value"]["closed_utc"]),"Replay cannot upgrade:"+field)
# Corrupt a last RAM event: full original chain/parser/replay must reject semantic drift.
journal=g["journal"];last=sorted(journal)[-1];original_bytes=journal[last]
for path,value in [(["schema"],"invented"),(["bindings"],{}),(["sequence"],99),
 (["value","proof","historical_facts_preserved","supervisor_outer_OS_wait"],0),
 (["value","measurement","per_scope",0,"node_types","socket"],1)]:
 ev=json.loads(original_bytes);target=ev
 for k in path[:-1]:target=target[k]
 target[path[-1]]=value;journal[last]=g["encoded"](ev)+b"\n"
 try:reject(state,"Full journal replay rejects "+str(path))
 finally:journal[last]=original_bytes
# Additional full scanner cases. Changes affect virtual filesystem IO only.
one=manifest["targets"][keys[0]]["scope_paths"][0]
snapshot=copy.deepcopy(fs.nodes)
measure=lambda:helper.measure_retained_nodes_v1(g["MPath"](ROOT),[g["MPath"](one)],expected_paths=[one],socket_is_absent=lambda p:False)
helper.require=lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg))
def fs_case(change,label):
 fs.nodes=copy.deepcopy(snapshot);change()
 try:reject(measure,label)
 finally:fs.nodes=copy.deepcopy(snapshot)
leaf=one+"/nested/preserved.bin"
for field,value in [("st_blocks",-1),("st_size",-1),("st_nlink",2),("st_dev",2)]:
 fs_case(lambda field=field,value=value:setattr(fs.nodes[leaf],field,value),"Scanner unsafe numeric/node:"+field)
fs_case(lambda:fs.add(one+"/control\nname","regular"),"Control character entry blocks")
fs_case(lambda:fs.add(one+"/huge_link","symlink",link="x"*65537),"Over64KiB stored symlink text blocks")
fs_case(lambda:setattr(fs.nodes[one+"/stored_link"],"st_size",0),"Stored symlink metadata length mismatch blocks")
fs_case(lambda:setattr(fs.nodes[str(Path(one).parent)],"st_dev",2),"Cross-device ancestor blocks")
def deep():
 p=one
 for i in range(65):p+="/deep"+str(i);fs.add(p)
fs_case(deep,"Depth65 rejected under original64 bound")
# A metadata change visible only during final per-node revalidation.
orig_stat=fs.stat;counts={}
def late_stat(p,dir_fd=None,follow_symlinks=False):
 v=orig_stat(p,dir_fd=dir_fd,follow_symlinks=follow_symlinks);full=fs.path(p,dir_fd)
 if full==leaf:
  counts[full]=counts.get(full,0)+1
  if counts[full]>=5:v.st_size+=1
 return v
fs.stat=late_stat
try:reject(measure,"Final no-follow leaf identity revalidation catches post-second-pass change")
finally:fs.stat=orig_stat
# max(size,allocated blocks*512)+4096 is tested using nonzero allocated blocks.
fs.nodes[leaf].st_blocks=8
allocation=measure()
check(allocation["conservative_scope_bytes"]==g["receipts"][0]["retained_bytes"]-123+4096,"Scanner conservative allocated-block charge")
fs.nodes=copy.deepcopy(snapshot)
check(fs.payload_opens==0,"Every baseline and independent scanner case opened zero regular/link target payloads")
check(not fs.fds,"Virtual scanner closes every descriptor on success and rejection")
# Actual 24MiB phase loader bound, not a shortened substituted constant.
oversize={"path":"RAM/oversized_phase.json","sha256":"c"*64}
g["records"][oversize["path"]]=oversize
g["metadata"][oversize["path"]]=b" "*(e["MAX_PHASE_JSON"]+1)
reject(lambda:e["_phase_json"](oversize),"Actual24MiB phase metadata cap rejects oversized input")
del g["metadata"][oversize["path"]]
result={"schema":"Census_observation_independent_source_RAM_review_checks_v1","status":"PASS_SOURCE_AND_RAM_ONLY",
 "reused_audited_author_RAM_baseline_ref":AUTHOR_REF,"reused_baseline_positive_checks":baseline["positive_checks"],
 "reused_baseline_negative_fixtures":baseline["negative_fixture_count"],
 "independently_authored_positive_checks":len(checks),"independently_authored_negative_fixtures":len(negatives),
 "checks":checks,"negative_cases":negatives,"actual_source_and_immutable_metadata_hashes_verified":real_hashes,
 "runtime_only_pins_deferred_without_reading_state":deferred,
 "full_candidate_business_predicates_unchanged":True,"full_single_root_scanner_virtual_FS_only":True,
 "parent_roles_preserved":46,"new_direct_selector_capsule":1,"original_direct_ref_ceiling":64,
 "actual_guard_state_process_GPU_scanner_admission_journal_operations":0,
 "actual_scientific_payload_checkpoint_prediction_truth_reads_or_hashes":0,
 "actual_retention_or_credit":0,"RAM_retention_events":2}
print(json.dumps(result,sort_keys=True,indent=2))
