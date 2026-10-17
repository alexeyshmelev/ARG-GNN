"""Inactive H058 fresh72h authority successor; original scientific/runtime caps unchanged."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,stat,sys
from datetime import datetime
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H058_fresh_CPU_lifetime_source_20261017_v2"
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
def authority_interval(q):
 created=datetime.fromisoformat(q["created_at_utc"]);expires=datetime.fromisoformat(q["expires_at_utc"])
 need(created.tzinfo is not None and expires.tzinfo is not None
  and(expires-created).total_seconds()==259200,"Exactly separate finite fresh72h ROOT authority; no old-clock reset")
def contract():
 b=bindings();c=load(b["contract_ref"],"_H058_fresh72h_original_contract")
 selector=load(b["selector_protocol_contract_ref"],"_H058_fresh72h_reviewed_selector_protocol")
 # This genuine common implementation contains new contract+driver identities.
 c.HERE=c.REL=HERE;c.SOURCE=c.DRIVER=SOURCE;c.ENTRY=ENTRY;c.MANIFEST=MANIFEST;c.HANDOFF=HANDOFF
 c.binding=lambda:read(b["scientific_binding_ref"])
 def selected(record):
  value=selector.selected(record)
  c.require(value["source_ref"]==b["selected_child_source_ref"]
   and value["selector_source_ref"]==b["selected_child_source_ref"]
   and value["accounting_mode"]=="Relate_failed64GiB_retention_v1","Exact current reviewed Relate child only")
  return value
 def bind_selector(record):
  chosen=selected(record);selector.bind_selector(record)
  for name in("SELECTOR_REF","_CHOSEN","SELECTOR","GUARD_REF","POLICY_REF"):setattr(c,name,copy.deepcopy(getattr(selector,name)))
  return chosen
 def guard(consumer):
  c.require(c.SELECTOR_REF==selector.SELECTOR_REF and c._CHOSEN==selector._CHOSEN
   and c.GUARD_REF==selector.GUARD_REF and c.POLICY_REF==selector.POLICY_REF,
   "Immutable reviewed selected singleton/source/policy before memo")
  return selector.guard(consumer)
 c.selected=selected;c.bind_selector=bind_selector;c.guard=guard
 original_roles=c.direct_roles
 def direct_roles(request_ref,q,cfgref,closure):
  roles=original_roles(request_ref,q,cfgref,closure)
  # Dynamic SELECTOR slot now is the actual child source. Preserve the original
  # static selector source too; no actual ancestor or original direct role is lost.
  for record in(c.DEPS["SELECTOR"],c.GUARD_REF):
   if record not in roles:roles.append(copy.deepcopy(record))
  c.require(c.GUARD_REF==selected(q["storage_policy_selector_ref"])["source_ref"]
   and c.GUARD_REF in roles,"Actual exact selected child source remains direct")
  return roles
 c.direct_roles=direct_roles
 original_put=c.put
 def put(path,value,cap=2**20):
  if Path(path).name=="input.json"and value.get("schema")=="H058_full_parent_CPU_admitted_input_v1":
   c.require(value.get("root_driver_ref")==c.ref(SOURCE),"Only new successor admitted identity")
   value=dict(value,hypothesis_id="H058")
  return original_put(path,value,cap=cap)
 c.put=put
 def accept_failure(record):
  v=c.read(record)
  c.require(v.get("schema")=="ROOT_H058_unstarted_failure_acceptance_v1"
   and v.get("root_authorizer")=="root_agent"and v.get("accepted")is True
   and v.get("original_input_ref")==b["original_input_ref"]
   and v.get("actual_wait_ref")==b["original_actual_wait_ref"]
   and v.get("worker_log_ref")==b["original_worker_log_ref"]
   and v.get("outer_tool_session")==7922 and v.get("outer_tool_terminal_consumed")is True
   and v.get("outer_exit_code")==1 and v.get("direct_worker_returncode")==1
   and v.get("owned_drain")is True and v.get("scientific_startup_reached")is False
   and v.get("optimizer_updates")==0 and v.get("old_launch_resume_write_authority_revoked")is True
   and v.get("old_files_key_charge_and_expiry_unchanged")is True
   and v.get("condition")==b["condition"],"Genuine ROOT acceptance of exact unstarted failure, not training repetition")
  c.raw(v["actual_outer_terminal_ref"])
 original_validate=c.validate_request
 def validate_request(record):
  q=c.read(record);authority_interval(q)
  c.require(q.get("fresh_lifetime_condition")==b["condition"]
   and q.get("cpu_ids")==b["cpu_ids"]and q.get("caps")==b["caps"]
   and q.get("source_manifest_ref")==c.ref(MANIFEST),"Only fresh same-science exact CPU160–191 successor")
  for fresh_key in("preparation_directory","packet_directory","output_directory"):
   fresh=c.local(q[fresh_key])
   for old_path in b["excluded_namespaces"].values():
    prior=c.local(old_path)
    c.require(fresh!=prior and not fresh.is_relative_to(prior)and not prior.is_relative_to(fresh),
     "All new namespaces disjoint from every old preparation/charged namespace")
  accept_failure(q["prior_failure_acceptance_ref"])
  directive=c.read(q["root_directive_ref"])
  c.require(directive.get("fresh_lifetime_condition")==q["fresh_lifetime_condition"]
   and directive.get("prior_failure_acceptance_ref")==q["prior_failure_acceptance_ref"]
   and directive.get("fresh_authority_seconds")==259200,
   "Separate ROOT directive explicitly binds unfinished-cell condition/failure and72h authority")
  # Original current(q,86700+600) stays intact at EVERY original boundary.
  return original_validate(record)
 c.validate_request=validate_request
 original_input=c.input_spec
 def input_spec(record):
  result=original_input(record)
  c.require(result[0].get("hypothesis_id")=="H058","Exact admitted identity needed by unchanged CPU context factory")
  return result
 c.input_spec=input_spec
 return c
def driver(c):
 d=load(bindings()["driver_ref"],"_H058_fresh72h_original_driver")
 for name,value in dict(c=c,ROOT=ROOT,HERE=HERE,SOURCE=SOURCE,PIDFD=c.PIDFD,OWNED=c.OWNED,
  need=c.require,local=c.local,ref=c.ref,read=c.read,load=c.load,put=c.put,current=c.current,guard=c.guard).items():
  setattr(d,name,value)
 return d
def worker(record,gate_fd):
 c=contract();w=load(bindings()["worker_ref"],"_H058_fresh72h_original_worker")
 w.c=c;w.ROOT=ROOT
 return w.worker(record,gate_fd)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument("--admit-ref");g.add_argument("--execute-ref");a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False,guard_imported=False)));return 0
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact ROOT repository -B -s interpreter")
 c=contract();d=driver(c);record=json.loads(a.admit_ref or a.execute_ref)
 result=d.admit(record)if a.admit_ref else d.execute(record)
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
