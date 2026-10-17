"""Independent exact native-v12 source/metadata/RAM ordinary close review; zero actual probes."""
from pathlib import Path
import hashlib
HERE="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1"
p=Path(HERE)/"native_MLM_v12_close_checks_v1.py"
assert hashlib.sha256(p.read_bytes()).hexdigest()=="cb1cfdf625166c9df110375d40b85bc521789dccea7756d2fd47418e06837371"
s=p.read_text()
start=r"""
 independent_cases=[]
 def independent(ok,name):
  check(ok,"Independent: "+name);independent_cases.append(name)
 handoff=realref(c.HERE+"/HANDOFF.native_MLM_v12_close.v1.json")
 independent(handoff["sha256"]=="8d64907d7aedfb31c619bfe59613efd185030e697866d60a6876ea38fa2375b9","Exact sealed handoff")
 manifest=realread(realref(b.MANIFEST))
 independent(realref(b.MANIFEST)["sha256"]=="9065b508581155bee1fdcefbeb306a7aa18b7125b88f97c75e7f5faf8292830c","Exact sealed manifest")
 independent(realref(b.SOURCE)["sha256"]=="fba77efd62446069d34a551cc306aa3819a91f66d2e2d2505cf5cf42e5960229","Exact sealed source")
 for record in manifest["source_refs"]:
  independent(realref(record["path"])==record,"Authenticate original source/compact metadata "+record["path"])
 independent(len(manifest["source_refs"])==14,"Complete14source and evidence refs")
 original_read=c.read
 corruptions=[
  ("storage","Excluded original H041 key",lambda v:v["reservation"].update(key="H041/American-C3-native-MLM-full30mb-mlm-fit-v8"),"Exact separate original v12"),
  ("storage","Wrong reservation cap",lambda v:v["reservation"].update(reserved_bytes=7*2**30),"Exact separate original v12"),
  ("storage","Wrong auxiliary scope",lambda v:v["reservation"].update(auxiliary_scopes=["/RAMwrong"]),"Exact separate original v12"),
  ("storage","Omitted original consumer bindings",lambda v:v["reservation"].update(consumer_bindings=[]),"Exact separate original v12"),
  ("wait","Boolean raw wait zero",lambda v:v.update(actual_OS_wait_status=False),"Genuine exact direct worker"),
  ("wait","Nonzero direct worker wait",lambda v:v.update(returncode=1),"Genuine exact direct worker"),
  ("wait","Incomplete owned drain",lambda v:v.update(all_owned_processes_and_groups_drained=False),"Genuine exact direct worker"),
  ("wait","Historical operational failure erased",lambda v:v.update(operational_failure=None),"Genuine exact direct worker"),
  ("wait","Truncated owned roster",lambda v:v["ownership"].update(owned_roster_truncated=True),"Original complete retained50identity"),
  ("wait","Duplicated PID hides helper",lambda v:v["ownership"]["owned_identities"][-1].update(pid=v["ownership"]["owned_identities"][0]["pid"]),"All51 actual"),
  ("wait","Recorded group absence false",lambda v:v["group_observations"][0].update(absent=False),"All51 actual"),
  ("outer","Supervisor outer falsely zero",lambda v:v.update(actual_tool_terminal_exit_code=0),"Historical actual supervisor outer1"),
  ("outer","Historical failure preservation erased",lambda v:v.update(outer_failure_preserved=False),"Historical actual supervisor outer1"),
  ("completed","Wrong original optimizer count",lambda v:v.update(actual_optimizer_updates=227989),"Genuine completed five-epoch worker"),
  ("completed","Wrong original parent",lambda v:v.update(parent_id="american_c3_train_000"),"Genuine completed five-epoch worker"),
  ("acceptance","Full MLM fit relabel",lambda v:v["saved_MLM_eligibility"].update(full_fit_completion_claimed=True),"Genuine completed five-epoch worker"),
  ("acceptance","Checkpoint acceptance substitution",lambda v:v.update(saved_MLM_checkpoint_ref=dict(path="RAMwrong.pt",sha256="0"*64)),"Genuine completed five-epoch worker")]
 for key,label,mutate,message in corruptions:
  record=b.P[key];value=copy.deepcopy(original_read(record));mutate(value)
  c.read=lambda r,value=value,record=record:copy.deepcopy(value)if r==record else original_read(r)
  try:
   try:b.metadata_v1()
   except ValueError as error:independent(message in str(error),"Reject "+label+" at exact metadata boundary")
   else:raise AssertionError("Accepted independently corrupted metadata: "+label)
  finally:c.read=original_read
 for label,change in [
  ("v12 never-credit policy",lambda v:v["never_credit_keys"].append(b.KEY)),
  ("Original H041 exclusion removed",lambda v:v["never_credit_keys"].remove("H041/American-C3-native-MLM-full30mb-mlm-fit-v8")),
  ("Stopped native32 exclusion removed",lambda v:v["ordinary_completed_close_excluded_keys"].remove("native32/native-VAL000-full227990-borrowed32-GPU0-v1"))]:
  value=copy.deepcopy(original_read(b.POLICY));change(value)
  c.read=lambda r,value=value:copy.deepcopy(value)if r==b.POLICY else original_read(r)
  try:
   try:b.metadata_v1()
   except ValueError as error:independent("Current original policy" in str(error),"Reject "+label)
   else:raise AssertionError("Accepted wrong original policy: "+label)
  finally:c.read=original_read
"""
s=s.replace(' meta=b.metadata_v1()',' meta=b.metadata_v1()\n'+start,1)
extra=r"""
 # Independent after-entry failures never return READY/no-close or replay.
 c.load=lambda r,name:g if r==b.GUARD else coord
 original_apply=g.apply_close_completed_v1
 trace.clear();prior_calls=dict(calls)
 def after_entry_error(r,**kwargs):
  calls["apply"]+=1;raise ValueError("Finite shared ROOT accounting coordinator wait")
 g.apply_close_completed_v1=after_entry_error
 afterq=dict(q,instance="independent_afterentry")
 afterref=add("RAM/independent_afterentry_authority.json",afterq)
 try:
  try:b.close_once_v1(afterref)
  except ValueError as error:independent(str(error)=="Finite shared ROOT accounting coordinator wait","Same coordinator-shaped error after entry propagates")
  else:raise AssertionError("Post-entry failure falsely returned READY")
  independent(calls["apply"]==prior_calls["apply"]+1 and calls["measure"]==prior_calls["measure"] and calls["append"]==prior_calls["append"],"One failed apply attempt and no automatic replay")
  prefix=c.HERE+"/ROOT_native_MLM_v12_close_independent_afterentry/"
  independent(not any(path==prefix+"ready_no_close.json"or path==prefix+"ordinary_close_receipt.json"or path==prefix+"result.json"for path in records),"Post-entry uncertainty retains attempt with no READY/receipt/result")
 finally:g.apply_close_completed_v1=original_apply
 # An original strict measurement failure must leave no append or claimed credit.
 trace.clear();prior_calls=dict(calls);original_measure=v4.measure_scopes
 def strict_namespace_error(root,paths):
  calls["measure"]+=1;raise ValueError("unsafe scope namespace/hardlink")
 v4.measure_scopes=strict_namespace_error
 strictq=dict(q,instance="independent_unsafe")
 strictref=add("RAM/independent_unsafe_authority.json",strictq)
 try:
  try:b.close_once_v1(strictref)
  except ValueError as error:independent(str(error)=="unsafe scope namespace/hardlink","Original strict namespace error propagates unchanged")
  else:raise AssertionError("Unsafe namespace produced receipt")
  independent(calls["measure"]==prior_calls["measure"]+1 and calls["append"]==prior_calls["append"] and calls["apply"]==prior_calls["apply"]+1,"Strict measurement failure invokes original apply once and appends zero events")
  prefix=c.HERE+"/ROOT_native_MLM_v12_close_independent_unsafe/"
  independent(not any(path in {prefix+"ready_no_close.json",prefix+"ordinary_close_receipt.json",prefix+"result.json"}for path in records),"Unsafe namespace has no credit/receipt/success/READY claim")
 finally:v4.measure_scopes=original_measure
 # Exact v4 stat inventory against synthetic directory metadata; no real lstat/scandir.
 import stat as stat_module
 inventory_source=Path("src/arggnn/project_budget_v4.py").read_text()
 inventory_node=next(n for n in ast.parse(inventory_source).body if isinstance(n,ast.FunctionDef)and n.name=="measure_scopes")
 canonical_node=next(n for n in ast.parse(inventory_source).body if isinstance(n,ast.FunctionDef)and n.name=="canonical")
 fs={};stat_counts={};mutation=None
 root_path="/RAM_PROJECT"
 class InventoryPath(PurePosixPath):
  def resolve(self):
   row=fs.get(str(self))
   return InventoryPath("/RAM_PROJECT/elsewhere")if row and stat_module.S_ISLNK(row["st_mode"])else self
  def exists(self):return str(self)in fs
  def lstat(self):
   key=str(self);stat_counts[key]=stat_counts.get(key,0)+1;row=copy.deepcopy(fs[key])
   if mutation==key and stat_counts[key]>=2:row["st_mtime_ns"]+=1
   return SimpleNamespace(**row)
 class Scan:
  def __init__(self,directory):self.directory=str(directory)
  def __enter__(self):return [SimpleNamespace(name=InventoryPath(key).name)for key in fs if str(InventoryPath(key).parent)==self.directory]
  def __exit__(self,*a):pass
 ie=dict(Path=InventoryPath,stat=stat_module,os=SimpleNamespace(scandir=lambda p:Scan(p)))
 for node in (canonical_node,inventory_node):exec(compile(ast.Module(body=[node],type_ignores=[]),"original_v4#independent-RAM-inventory","exec"),ie)
 def setup_inventory(kind="regular"):
  fs.clear();stat_counts.clear()
  for i,(key,mode,nlink,size,blocks)in enumerate([
   (root_path+"/output",stat_module.S_IFDIR|0o700,2,32,8),
   (root_path+"/output/retained.pt",stat_module.S_IFREG|0o600,1,1234,16),
   (root_path+"/aux",stat_module.S_IFDIR|0o700,2,32,8)]):
   fs[key]=dict(st_dev=1,st_ino=i+10,st_mode=mode,st_nlink=nlink,st_size=size,st_blocks=blocks,st_mtime_ns=1,st_ctime_ns=1)
  row=fs[root_path+"/output/retained.pt"]
  if kind=="symlink":row["st_mode"]=stat_module.S_IFLNK|0o777
  if kind=="hardlink":row["st_nlink"]=2
  if kind=="socket":row["st_mode"]=stat_module.S_IFSOCK|0o600
  return [InventoryPath(root_path+"/output"),InventoryPath(root_path+"/aux")]
 paths=setup_inventory();measured=ie["measure_scopes"](InventoryPath(root_path),paths)
 independent(measured["conservative_scope_bytes"]==16384 and measured["entries"]==3 and measured["stat_only"]is True,"Original regular inventory uses max logical/allocated bytes and directory stat")
 for kind in ("symlink","hardlink","socket"):
  paths=setup_inventory(kind)
  try:ie["measure_scopes"](InventoryPath(root_path),paths)
  except ValueError:independent(True,"Exact original inventory rejects synthetic "+kind)
  else:raise AssertionError("Unsafe synthetic inventory accepted: "+kind)
 for key in (root_path+"/output",root_path+"/output/retained.pt"):
  paths=setup_inventory();mutation=key
  try:ie["measure_scopes"](InventoryPath(root_path),paths)
  except ValueError:independent(True,"Exact original inventory rejects changing metadata "+key)
  else:raise AssertionError("Changing synthetic inventory accepted: "+key)
  finally:mutation=None
 independent(not any(str(getattr(mod,"__file__","")).endswith("census_observation_retention_v1.py")for mod in list(sys.modules.values())),"No actual current guard module imported")
"""
s=s.replace(' resultrecord=dict(schema=',extra+'\n resultrecord=dict(independent_cases=independent_cases,independent_case_count=len(independent_cases),schema=')
s=s.replace(' if __name__==', ' if __name__==')
# Keep main as a callable so the returned structured record can be saved by this checker.
namespace={"__name__":"__independent_native_v12__"}
exec(compile(s,str(p)+"#independent-challenges","exec"),namespace)
result=namespace["main"]()
import json
print("INDEPENDENT_RESULT_JSON")
print(json.dumps(result,sort_keys=True,indent=2))
