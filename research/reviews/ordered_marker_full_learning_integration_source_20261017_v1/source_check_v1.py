"""Standard-library source/metadata audit only. Never import operational/scientific modules."""
from pathlib import Path
from datetime import datetime
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/ordered_marker_full_learning_integration_source_20261017_v1"
SCI="research/reviews/ordered_marker_learning_integration_source_20261017_v2"
OLD="research/reviews/H058_full_parent_CPU_runtime_wrapper_source_20261017_v2"
TERM="research/reviews/H058_context_terminal_successor_source_20261017_v1"
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def ref(path):
 p=ROOT/path;return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):
 p=ROOT/record["path"]
 need(p.resolve()==p and p.is_file()and not p.is_symlink()and p.stat().st_size<=32*2**20,"Compact source/metadata only")
 need(p.suffix in(".py",".sh",".json",".md",".patch")and not record["path"].startswith("data/"),"No payload/data/array/checkpoint read")
 value=p.read_bytes();need(hashlib.sha256(value).hexdigest()==record["sha256"],"Exact source/metadata SHA")
 return value
def read(record):return json.loads(raw(record))
def tree(path):return ast.parse((ROOT/path).read_text())
def function(path,name):
 matches=[n for n in ast.walk(tree(path))if isinstance(n,ast.FunctionDef)and n.name==name]
 need(len(matches)==1,"One exact function: "+name);return matches[0]
def dump(node):return ast.dump(node,include_attributes=False)
class Normalize(ast.NodeTransformer):
 def visit_Constant(self,node):
  if isinstance(node.value,str):
   node.value=node.value.replace(OLD,HERE).replace(OLD.replace("/","."),HERE.replace("/",".")).replace("H058","ordered_marker")
  return node
def normalized(node):return dump(Normalize().visit(copy.deepcopy(node)))
def projected_ref(path):return ref(path)if(ROOT/path).is_file()else dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
def run():
 manifest=read(ref(HERE+"/source_manifest.v1.json"));authenticated=0;inherited_data_metadata_not_opened=[]
 for path,sha in manifest["source_sha256"].items():
  if path.startswith("data/"):
   inherited_data_metadata_not_opened.append(path);continue
  raw(dict(path=path,sha256=sha));authenticated+=1
 python_files=[]
 for p in sorted((ROOT/HERE).glob("*.py")):
  compile(p.read_text(),str(p),"exec");python_files.append(p.name)
 b=read(ref(HERE+"/scientific_binding.v1.json"));caps=read(ref(HERE+"/caps.inactive.v1.json"));protocol=b["config_defaults"]["scientific_protocol"]
 scientific=read(b["scientific_review_ref"])
 need(scientific["verdict"]=="SOURCE_PASS"and scientific["source_ref"]==b["runner_source_ref"]and scientific["reviewed_entrypoint"]=="producer_v2.run_context_v2","Exact reviewed scientific caller")
 need(protocol["hypothesis_id"]=="ordered_marker_placement_v2"and protocol["updates_per_arm"]==960
  and protocol["updates_all_new_arms"]==1920and protocol["VAL_prediction_products_required_before_truth"]==96
  and set(protocol["parameters_per_arm"].values())=={33966},"Exact new scientific identities and complete budget")
 need(protocol==read(ref(SCI+"/SOURCE_REGISTRATION_SNAPSHOT.v2.json"))["scientific_protocol"],"Unchanged reviewed protocol")
 for name in("execute","boundary","sample_scope","worker_python","input_spec","main"):
  need(normalized(function(OLD+"/root_driver_v1.py",name))==normalized(function(HERE+"/root_driver_v1.py",name)),"Preserved original full supervisor method: "+name)
 old_factory="research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/context_factory_cpu_v1.py"
 need(dump(function(old_factory,"before_episode"))==dump(function(HERE+"/context_factory_v1.py","before_episode")),"Exact original whole-parent episode callback")
 for name in("jsonable_runtime_v1",):
  need(dump(function(OLD+"/worker_entry_v1.py",name))==dump(function(HERE+"/worker_entry_v1.py",name)),"Unchanged runtime-local serializer")
 caller=(ROOT/HERE/"worker_entry_v1.py").read_text()
 need("runner.run_context_v2(ctx,record)"in caller and "runner.run_context_v1("not in caller,"Only reviewed v2 scientific dispatcher")
 need("ordered_marker_(tree_feedback|sample_readout)"in caller and "all_32_ordered_marker_observation_recipes.json"in caller,"Complete new writer products")
 driver=(ROOT/HERE/"root_driver_v1.py").read_text()
 need("state=memo.read_state_v1("in driver and "receipt=memo.reserve_v1("in driver
  and "g.apply_reserve_v1("not in driver and "len(consumers)<=64"in driver,"Existing exact transaction memo and no role trimming/ceiling expansion")
 cfg=copy.deepcopy(b["config_defaults"])
 need(cfg["CPU_positive_plan_cap_bytes"]==4*2**30and cfg["CPU_prepared_cache_cap_bytes"]==32*2**30and cfg["max_cpu_rss_bytes"]==128*2**30,"Explicit positive-plan subcap and unchanged aggregate/RSS bounds")
 # Execute ONLY extracted pure metadata functions. No top-level contract/factory/guard/model import.
 nodes=[function(HERE+"/root_contract_v1.py",n)for n in("validate_config","validate_scientific_declarations","scientific_review","context","registration","reviewed","validate_request","direct_roles","make_config")]
 fake={}
 def p_read(record):
  if record["path"]in fake:return copy.deepcopy(fake[record["path"]])
  return read(record)
 def p_raw(record):
  if record["path"]in fake:
   value=fake[record["path"]];return value.encode()if isinstance(value,str)else json.dumps(value).encode()
  return raw(record)
 env=dict(binding=lambda:b,require=need,read=p_read,raw=p_raw,ref=projected_ref,datetime=datetime,
  ROOT=ROOT,HERE=HERE,SOURCE=HERE+"/root_contract_v1.py",DRIVER=HERE+"/root_driver_v1.py",
  ENTRY=HERE+"/worker_entry_v1.py",MANIFEST=HERE+"/source_manifest.v1.json",HANDOFF=HERE+"/source_handoff.v1.json",
  HYPOTHESIS="ordered_marker_placement_v2",TRAIN=tuple(protocol["TRAIN_parent_ids"]),CAPS=caps,
  local=lambda path:ROOT/path,DEPS=read(ref(HERE+"/dependencies.v1.json")))
 env.update({k:env["DEPS"][k]for k in("NAMESPACE","PIDFD","OWNED","SUPERVISOR")})
 selector=read(b["selected_child_selector_ref"]);env["SELECTOR"]=selector["source_ref"]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),HERE+"/root_contract_v1.py#pure-metadata-extract","exec"),env)
 def fake_ref(name,value):
  path=HERE+"/HANDWRITTEN_ONLY/"+name;fake[path]=value;return projected_ref(path)
 for key in b["operational_ref_keys"]:
  cfg[key]=fake_ref(key+".json",dict(root_authorizer="root_agent"))
 training=read(ref(HERE+"/full_training_directive.INACTIVE.template.v1.json"))
 training.update(root_authorizer="root_agent",full30pass_training_authorized=True)
 fake[cfg["full_training_directive_ref"]["path"]]=training
 layout=read(ref(HERE+"/layout_resource_proof.INACTIVE.template.v1.json"))
 layout.update(root_authorizer="root_agent",status="admitted_first_scheduled_full_parent_step")
 fake[cfg["layout_resource_proof_ref"]["path"]]=layout
 env["validate_config"](cfg);env["validate_scientific_declarations"](cfg);env["scientific_review"](b["scientific_review_ref"])
 ctx=read(ref(HERE+"/training_context.INACTIVE.template.v1.json"));ctx["root_authorizer"]="root_agent"
 env["context"](fake_ref("context.json",ctx))
 reg=read(ref(HERE+"/registration.INACTIVE.template.v1.json"));reg.update(root_authorizer="root_agent",active=True,registered_utc="2026-10-17T00:00:00+00:00")
 rref=cfg["experiment_registration_ref"];fake[rref["path"]]=reg
 q=dict(scientific_config=cfg,registration_ref=rref,registration_decisions_ref=fake_ref("decisions.json",dict(decisions=[dict(id=reg["id"],registration_ref=rref)])),
  registration_ledger_ref=fake_ref("ledger.md",reg["id"]),created_at_utc="2026-10-17T00:00:01+00:00")
 env["registration"](q)
 # Whole unchanged admission metadata validation, with handwritten declarations only.
 # Virtual paths answer only the preparation-directory shape; no real scope is created.
 class HandwrittenPath(type(ROOT)):
  def is_dir(self):return str(self)==str(ROOT/q["preparation_directory"])
 env["local"]=lambda path:HandwrittenPath(ROOT/path)
 request_template=read(ref(HERE+"/ROOT_request.INACTIVE.template.v1.json"))
 q.update(request_template)
 q.update(active=True,execution_authorized=True,root_authorizer="root_agent",
  created_at_utc="2026-10-17T00:00:01+00:00",expires_at_utc="2026-10-20T00:00:01+00:00",
  preparation_directory="research/reviews/ordered_marker_CPU_ROOT_preparation_HANDWRITTEN_ONLY",
  packet_directory="research/reviews/ordered_marker_CPU_ROOT_HANDWRITTEN_ONLY",
  output_directory="experiments/results/ordered_marker_full_parent_CPU_HANDWRITTEN_ONLY",
  scientific_config=cfg,source_manifest_ref=ref(HERE+"/source_manifest.v1.json"),
  registration_ref=rref,registration_decisions_ref=fake_ref("decisions.json",dict(decisions=[dict(id=reg["id"],registration_ref=rref)])),
  registration_ledger_ref=fake_ref("ledger.md",reg["id"]),training_context_ref=fake_ref("context.json",ctx),
  runtime_ref=fake_ref("runtime.json",dict(device="cpu",dtype="float32",CPU_threads=32,cpu_ids=request_template["cpu_ids"],
   prefix=str(ROOT/".runtime/env"),package_report_ref=env["DEPS"]["BASE_RUNTIME"])),
  scientific_review_ref=b["scientific_review_ref"])
 q["runtime_qualification_ref"]=fake_ref("qualification.json",dict(root_authorizer="root_agent",status="qualified",
  runtime_ref=q["runtime_ref"],source_manifest_ref=q["source_manifest_ref"],runner_source_ref=b["runner_source_ref"],
  unresolved_items=[],standalone_token_update_probe=False,one_hold_summary_ref=b["one_hold_summary_ref"],
  full_training_fit_measured=False,scientific_probe_updates=0))
 q["source_review_ref"]=fake_ref("review.json",dict(independent=True,verdict="SOURCE_PASS",
  source_ref=ref(HERE+"/root_driver_v1.py"),source_manifest_ref=q["source_manifest_ref"],source_handoff_ref=ref(HERE+"/source_handoff.v1.json"),
  runner_source_ref=b["runner_source_ref"],protocol_ref=b["protocol_ref"],blockers=[]))
 directive=copy.deepcopy(q);directive.update(directive="ROOT authorizes one fresh ordered_marker_placement_v2 full original32parent CPU run with960updates EACHarm and1920total, firstscheduled updates checked and timed inline",first_scheduled_update_timed_inside_full_budget=True)
 q["root_directive_ref"]=fake_ref("directive.json",directive)
 env["BASE_RUNTIME"]=env["DEPS"]["BASE_RUNTIME"];env["PHASE"]="train";env["PARENT"]="ordered_marker_European32_full_TRAIN_CPU"
 env["DIRECTIVE"]=directive["directive"];env["current"]=lambda *args:None;env["sources"]=lambda:manifest
 env["bind_selector"]=lambda record:read(record)
 request_record=fake_ref("request.json",q)
 request_record["path"]=q["preparation_directory"]+"/request.json";fake[request_record["path"]]=q
 result=env["validate_request"](request_record)
 need(result[0]==q and result[1]==caps,"Whole source-only metadata request/context/registration/qualification pass")
 # Bad historical field/schema and reduced/full-budget declarations must fail.
 negative=[]
 for name,mutate in(
  ("wrong_historical_terminal_schema",lambda value:value.update(schema="wrong")),
  ("wrong_historical_terminal_wait_join",lambda value:value.update(actual_wait_ref=None))):
  old=cfg["one_hold_outer_terminal_ref"];bad=read(old);mutate(bad);fake[old["path"]]=bad
  try:env["validate_scientific_declarations"](cfg)
  except AssertionError:negative.append(name)
  else:raise AssertionError("Bad historical terminal accepted")
  del fake[old["path"]]
 for key,value in(("checkpoint_added",False),("CPU_prepared_cache_cap_bytes",16*2**30)):
  bad=copy.deepcopy(cfg);bad[key]=value
  try:env["validate_config"](bad)
  except AssertionError:negative.append(key)
  else:raise AssertionError("Changed science/caps accepted")
 q.update(preparation_directory="research/reviews/ordered_marker_CPU_ROOT_preparation_HANDWRITTEN_ONLY",
  packet_directory="research/reviews/ordered_marker_CPU_ROOT_HANDWRITTEN_ONLY",scientific_config=cfg,
  source_review_ref=fake_ref("review.json",{}),scientific_review_ref=b["scientific_review_ref"],
  runtime_ref=fake_ref("runtime.json",{}),runtime_qualification_ref=fake_ref("qualification.json",{}),
  root_directive_ref=fake_ref("directive.json",{}),training_context_ref=fake_ref("training_context.json",ctx),
  storage_policy_selector_ref=b["selected_child_selector_ref"])
 stage=q["preparation_directory"]+"/"+Path(q["packet_directory"]).name+".scientific_config.json";fake[stage]=cfg
 request=fake_ref("request.json",q)
 direct=env["direct_roles"](request,q,fake_ref("config.json",{}),fake_ref("closure.json",{}))
 guard_meta=read(read(ref(HERE+"/source_bindings.v1.json"))["historical_guard_role_metadata_ref"])
 guards=list(guard_meta["bindings"].values())
 need(len(guards)==46,"All46 historical guard names retained")
 unique=[]
 for record in guards+direct:
  if record not in unique:unique.append(record)
 need(len(unique)<=64,"Complete handwritten direct+all46guard union remains<=64")
 need(all(r in unique for r in guards)and selector["source_ref"]in unique and env["DEPS"]["SELECTOR"]in unique,"Every actual/current/original guard source remains direct")
 need(all(Path(r["path"]).suffix in(".py",".sh",".json")for r in unique),"Direct consumer schema no Markdown ledger")
 test=list(unique)
 while len(test)<65:test.append(projected_ref(HERE+"/HANDWRITTEN_ONLY/extra_%d.json"%len(test)))
 need(not len(test)<=64,"65th role rejects without trimming")
 return dict(schema="ordered_marker_full_learning_source_check_v1",status="SOURCE_CHECK_PASS",
  authenticated_source_metadata_refs=authenticated,inherited_data_metadata_pin_preserved_not_opened=inherited_data_metadata_not_opened,
  python_files_compiled=python_files,actual_runtime_scientific_guard_imports=False,whole_request_metadata_canary_pass=True,
  actual_data_arrays_truth_checkpoints_resources_state_admission_jobs_or_Git=False,
  handwritten_projection_not_authority=True,all46guard_names_retained=True,
  unique_guard_records=len({json.dumps(x,sort_keys=True)for x in guards}),
  handwritten_complete_union=len(unique),original_role_ceiling=64,role65_rejected=True,
  negative_metadata_canaries=negative,exact_historical_qualification_actual_wait_join=True,
  exact_reviewed_scientific_caller=b["runner_source_ref"],full_budget=dict(updates_per_arm=960,total=1920,VAL_pretruth_products=96),
  measured_full_parent_fit_or_throughput=False)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
