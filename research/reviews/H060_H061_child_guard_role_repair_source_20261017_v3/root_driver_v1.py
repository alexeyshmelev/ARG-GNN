"""Inactive common contract+driver successor; adds exact selected child source, no science change."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H060_H061_child_guard_role_repair_source_20261017_v3"
SOURCE=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
def need(ok,message):
 if not ok:raise ValueError(message)
def local(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and ".."not in p.parts and p.resolve()==p,"Canonical contained source/metadata")
 return p
def metadata_ref(path):
 p=local(path);st=p.lstat()
 need(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=32*2**20,"Compact regular source/metadata")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):
 need(isinstance(record,dict)and set(record)=={"path","sha256"}and metadata_ref(record["path"])==record,"Exact immutable source/metadata SHA")
 return local(record["path"]).read_bytes()
def read(record):return json.loads(raw(record))
def bindings():return read(metadata_ref(HERE+"/source_bindings.v1.json"))
def load(record,name):
 raw(record);sp=importlib.util.spec_from_file_location(name,local(record["path"]))
 module=importlib.util.module_from_spec(sp);sys.modules[name]=module;sp.loader.exec_module(module);return module
def append_selected_source(roles,source,bound_source):
 """Pure, order-preserving addition. Actual unchanged consumer API enforces <=64."""
 need(isinstance(roles,list)and isinstance(source,dict)and set(source)=={"path","sha256"}
  and source==bound_source,"Exact metadata-selected and immutable lifetime-bound source only")
 result=list(roles)
 if source not in result:result.append(copy.deepcopy(source))
 return result
def contract(consumer):
 """Actual new contract and driver share THIS implementation file, saving one real role."""
 spec=bindings()["consumers"].get(consumer);need(spec is not None,"Exact H060 or H061 consumer")
 c=load(spec["contract_ref"],"_child_source_repair_"+consumer+"_contract")
 # Isolated module globals only. Frozen files/modules/guards are never patched.
 # Original functions keep their original globals in this fresh private module.
 c.HERE=c.REL=HERE;c.SOURCE=c.DRIVER=SOURCE;c.ENTRY=ENTRY;c.MANIFEST=MANIFEST;c.HANDOFF=HANDOFF
 c.binding=lambda:read(spec["scientific_binding_ref"])
 original_roles=c.direct_roles
 def repaired_roles(request_ref,q,cfgref,closure):
  roles=original_roles(request_ref,q,cfgref,closure)
  chosen=c.selected(q["storage_policy_selector_ref"])
  c.require(c.SELECTOR_REF==q["storage_policy_selector_ref"]and c._CHOSEN==chosen
   and c.GUARD_REF==chosen["source_ref"],"Same exact lifetime-selected source before role union")
  return append_selected_source(roles,chosen["source_ref"],c.GUARD_REF)
 c.direct_roles=repaired_roles
 original_validate=c.validate_request
 def validate_request(record):
  q=c.read(record)
  capsule=c.read(q["storage_policy_selector_ref"]);child=bindings()["selected_child_source_ref"]
  c.require(capsule.get("source_ref")==child and capsule.get("selector_source_ref")==child,
   "Fresh admissions only through exact selected Relate child; no older-guard fallback")
  c.require(q.get("guard_role_repair_consumer")==consumer
   and q.get("source_manifest_ref")==c.ref(MANIFEST),"Genuine ROOT fresh successor request for this consumer")
  c.require(all(q.get(key)!=spec["excluded_"+key]for key in("preparation_directory","packet_directory","output_directory")),
   "Never retry/replay old H060 failure or old H061 preparation namespaces")
  return original_validate(record)
 c.validate_request=validate_request
 return c
def driver(consumer,c):
 d=load(bindings()["consumers"][consumer]["driver_ref"],"_child_source_repair_"+consumer+"_driver")
 # Reuse the exact reviewed admission/execution functions, including pre-authority
 # union+scope_spec and all finite lifecycle/scientific completion predicates.
 for name,value in dict(c=c,ROOT=ROOT,HERE=HERE,SOURCE=SOURCE,PIDFD=c.PIDFD,OWNED=c.OWNED,
  need=c.require,local=c.local,ref=c.ref,read=c.read,load=c.load,put=c.put,current=c.current,guard=c.guard).items():
  setattr(d,name,value)
 return d
def execute_or_admit(consumer,record,admission):
 c=contract(consumer);d=driver(consumer,c)
 result=d.admit(record)if admission else d.execute(record)
 if admission:
  # This exact emitted successor module requires its consumer discriminator.
  result["execution_argv"]+=["--consumer",consumer]
  result["guard_role_repair_consumer"]=consumer
 return result
def consumer_for_admitted(record):
 s=read(record);matches=[k for k,v in bindings()["consumers"].items()if s.get("parent_id")==v["parent_id"]]
 need(len(matches)==1 and s.get("root_driver_ref")==metadata_ref(SOURCE)
  and s.get("main_source_ref")==metadata_ref(ENTRY),"Exact admitted common successor identity")
 return matches[0]
def worker(record,gate_fd):
 consumer=consumer_for_admitted(record);c=contract(consumer)
 w=load(bindings()["consumers"][consumer]["worker_ref"],"_child_source_repair_"+consumer+"_worker")
 w.c=c;w.ROOT=ROOT
 if consumer=="H061":
  # The original bridge also holds a contract global for predecessor acceptance.
  # Load its exact bytes privately at the ORIGINAL after-gate import site, then
  # bind the same new contract. No runner function/algorithm is changed.
  scientific=c.binding()
  def import_runner(name):
   c.require(name==scientific["runner_module"],"Only exact pinned H061 scientific bridge")
   runner=load(scientific["runner_source_ref"],"_child_source_repair_H061_scientific_bridge")
   runner.c=c
   return runner
  w.importlib=types.SimpleNamespace(import_module=import_runner)
 # Original worker validates the full new admitted input and real gate before
 # every scientific import; exact budgets, runner, settings and waits unchanged.
 return w.worker(record,gate_fd)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--consumer",choices=("H060","H061"))
 group=p.add_mutually_exclusive_group();group.add_argument("--admit-ref");group.add_argument("--execute-ref");a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False,guard_imported=False)));return 0
 need(a.consumer is not None and Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,
  "Exact ROOT consumer/-B/-s/repository interpreter")
 result=execute_or_admit(a.consumer,json.loads(a.admit_ref or a.execute_ref),a.admit_ref is not None)
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
