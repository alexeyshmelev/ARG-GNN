"""Small source/config/serialization/direct-role checks. No scientific payload or guard modules."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,io,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v1"
def load(path,label):
 sp=importlib.util.spec_from_file_location(label,ROOT/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def reject(fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):return
 raise AssertionError("Expected rejection")
def fake(label):return dict(path="SYNTHETIC_ONLY/"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
def main():
 c=load(HERE+"/root_contract_v1.py","_TreeLoom_small_checks_contract")
 w=load(HERE+"/worker_entry_v1.py","_TreeLoom_small_checks_worker")
 b=c.binding();cfg=copy.deepcopy(b["inactive_defaults"])
 for key in b["operational_config_keys"]:
  if key not in ("execution_enabled","CPU_observation_cache_bytes"):cfg[key]=1
 cfg["execution_enabled"]=True;cfg["max_truth_uncompressed_bytes"]=b["inherited_max_truth_uncompressed_bytes"];c.validate_config(cfg)
 bad=dict(cfg,additional_updates_per_arm=1);reject(lambda:c.validate_config(bad))
 bad=dict(cfg,max_cached_parent_bytes=97*2**30);reject(lambda:c.validate_config(bad))
 buffer=w.CappedBuffer(64);assert buffer.write(b"x"*64)==64;reject(lambda:buffer.write(b"x"))
 # Drive exact torch serialization cap branch using only a synthetic serializer.
 class FakeTorch:
  @staticmethod
  def save(value,stream):stream.write(b"x"*65)
 writer=object.__new__(w.BoundedWriter);writer.fast=lambda e:None;writer.names=set();writer.torch_files=0;writer.torch=FakeTorch
 old=w.c.CAPS["checkpoint_serialized_bytes"];w.c.CAPS["checkpoint_serialized_bytes"]=64
 try:reject(lambda:writer.save("checkpoint_pair_epoch_005.pt",{},kind="torch"))
 finally:w.c.CAPS["checkpoint_serialized_bytes"]=old
 # No filesystem or guard object exists in that rejection fixture.
 output_upper=7*c.CAPS["checkpoint_serialized_bytes"]+c.CAPS["total_JSON_bytes"]+(7+8+1)*8192
 assert output_upper<=c.CAPS["output_bytes"]
 assert c.CAPS["output_bytes"]+c.CAPS["aux_bytes"]+c.CAPS["root_packet_bytes"]==c.CAPS["reservation_bytes"]==32*2**20
 parent=c.read(c.ref("research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json"))["current_guard_roles"]
 childpath="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py"
 ns=dict(copy=copy,require=c.require,_raw=lambda r:b"SYNTHETIC",_ref=c.ref,_manifest=lambda:None,
  SOURCE=childpath,MEASURE_SOURCE=childpath.rsplit("/",1)[0]+"/retained_nodes_v1.py",
  MANIFEST=childpath.rsplit("/",1)[0]+"/source_manifest.v1.json",SELECTOR_SOURCE=c.SELECTOR["path"],_POLICY_REF=c.POLICY_REF)
 tree=ast.parse((ROOT/childpath).read_text())
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in("_bindings","consumer_bindings_v1")],type_ignores=[]),"<exact_pure_guard_bindings>","exec"),ns)
 child=ns["_bindings"](parent,dict(root_activation_ref=fake("activation"),source_review_ref=fake("childreview"),permanent_revocation_ref=fake("revoke")))
 fn=next(n for n in ast.parse((ROOT/HERE/"root_contract_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="direct_roles")
 env=dict(vars(c));env["binding"]=lambda:b;env["ref"]=lambda p:fake("handoff"if p==c.HANDOFF else"manifest")if p in(c.HANDOFF,c.MANIFEST)else c.ref(p)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),"<exact_operation_roles>","exec"),env)
 q={k:fake(k)for k in("source_review_ref","scientific_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref",
  "registration_ref","registration_decisions_ref","registration_ledger_ref","training_context_ref","storage_policy_selector_ref")}
 direct=env["direct_roles"](fake("request"),q,fake("config"),fake("closure"))
 union=ns["consumer_bindings_v1"](list(child.values()),direct)
 assert len(union)<=64 and all(r in union for r in child.values())
 assert len(ns["consumer_bindings_v1"](union,[fake("extra"+str(i))for i in range(64-len(union))]))==64
 reject(lambda:ns["consumer_bindings_v1"](union,[fake("overflow"+str(i))for i in range(65-len(union))]))
 assert not any(n in sys.modules for n in("torch","numpy","torch_geometric","scripts.project_budget_v9","arggnn.project_budget_v4"))
 return dict(schema="TreeLoom_matched_CPU_small_source_checks_v1",status="PASS_SOURCE_ONLY",full_config_preserved=True,
  reduced_budget_and_over96GiB_cache_reject=True,actual_serializer_buffer_64_accepts65_rejects_before_guard_or_disk=True,
  checkpoint_serialization_preflight_exercised_with_synthetic_serializer=True,max7serialized_checkpoint_bytes=7*c.CAPS["checkpoint_serialized_bytes"],
  aggregate_JSON_bytes=c.CAPS["total_JSON_bytes"],prospective_output_upper_bytes=output_upper,output_cap_bytes=c.CAPS["output_bytes"],
  whole_output_aux_admin_reservation_bytes=c.CAPS["reservation_bytes"],physical_fit_proven=False,
  returned_guard_raw_keys=len(child),returned_unique_refs=len({json.dumps(r,sort_keys=True)for r in child.values()}),
  required_operation_refs=len(direct),complete_source_model_unique_refs=len(union),every_returned_guard_preserved=True,
  exact64_accepts65_rejects=True,all_policies_runtimes_metadata_guard_state_and_qualifications_synthetic=True,
  actual_guard_model_data_truth_checkpoint_probe_admission_jobs_or_Git=False)
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))
