"""Independent source-only review; standard library and fabricated metadata only."""
from pathlib import Path
from datetime import datetime
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/ordered_marker_full_learning_integration_source_20261017_v1"
SCI="research/reviews/ordered_marker_learning_integration_source_20261017_v2"
OUT="research/reviews/ordered_marker_full_learning_integration_independent_20261017_v1"
OLD="research/reviews/H058_full_parent_CPU_runtime_wrapper_source_20261017_v2"
def need(v,msg):
 if not v:raise AssertionError(msg)
def digest(b):return hashlib.sha256(b).hexdigest()
def ref(path):
 p=ROOT/path;return dict(path=path,sha256=digest(p.read_bytes()))
def raw(r):
 p=ROOT/r["path"]
 need(not r["path"].startswith("data/") and p.suffix in (".py",".sh",".json",".md",".patch"),"source/compact metadata only")
 need(p.is_file()and p.resolve()==p and not p.is_symlink()and p.stat().st_size<=32*2**20,"canonical compact source")
 b=p.read_bytes();need(digest(b)==r["sha256"],"source seal mismatch "+r["path"]);return b
def read(r):return json.loads(raw(r))
def function(path,name):
 nodes=[n for n in ast.walk(ast.parse((ROOT/path).read_text()))if isinstance(n,ast.FunctionDef)and n.name==name]
 need(len(nodes)==1,"unique function "+name);return nodes[0]
def execute_functions(path,names,env):
 nodes=[copy.deepcopy(function(path,n))for n in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path+"#independent-extracted-only","exec"),env)
 return env
def normalized(node):
 class N(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str):n.value=n.value.replace(OLD,HERE).replace(OLD.replace("/","."),HERE.replace("/",".")).replace("H058","ordered_marker")
   return n
 return ast.dump(N().visit(copy.deepcopy(node)),include_attributes=False)
def rejected(call,msg):
 try:call()
 except (AssertionError,ValueError,KeyError,TypeError):return
 raise AssertionError("invalid metadata accepted: "+msg)
manifest=read(ref(HERE+"/source_manifest.v1.json"))
authenticated=[];skipped=[];compiled=[]
for p,sha in manifest["source_sha256"].items():
 if p.startswith("data/"):skipped.append(p);continue
 b=raw(dict(path=p,sha256=sha));authenticated.append(p)
 if p.endswith(".py"):compile(b,p,"exec");compiled.append(p)
for p in (ROOT/HERE).glob("*.py"):compile(p.read_bytes(),str(p),"exec")
need(ref(HERE+"/source_manifest.v1.json")["sha256"]=="5293ed8adc9cebf259c41a18c2e618257850d58e3fc7d23715efdedd118f2fb9","final manifest")
need(ref(HERE+"/source_handoff.v1.json")["sha256"]=="cb309e728a6303a6ce277b7ce6715f9c4e9ffb3a202ac0a64ed4ba2d3fe1573d","final handoff")
need(ref(HERE+"/root_driver_v1.py")["sha256"]=="e3ca25facd26f866bda5d64c7f9aed31524c16415094908ff294bc0504b811ce","final driver")
preserved=["execute","boundary","sample_scope","worker_python","input_spec","main"]
for n in preserved:
 need(normalized(function(HERE+"/root_driver_v1.py",n))==normalized(function(OLD+"/root_driver_v1.py",n)),"lifecycle changed "+n)
need(ast.dump(function(HERE+"/context_factory_v1.py","before_episode"),include_attributes=False)==ast.dump(function("research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/context_factory_cpu_v1.py","before_episode"),include_attributes=False),"episode callback changed")
b=read(ref(HERE+"/scientific_binding.v1.json"));caps=read(ref(HERE+"/caps.inactive.v1.json"));protocol=b["config_defaults"]["scientific_protocol"]
need(protocol==read(ref(SCI+"/SOURCE_REGISTRATION_SNAPSHOT.v2.json"))["scientific_protocol"],"protocol identity")
review=read(b["scientific_review_ref"])
need(review["verdict"]=="SOURCE_PASS"and review["blockers"]==[] and review["source_ref"]==b["runner_source_ref"] and review["source_manifest_ref"]==b["source_manifest_ref"]and review["source_handoff_ref"]==b["source_handoff_ref"]and review["reviewed_entrypoint"]=="producer_v2.run_context_v2","actual scientific review identity")
need(protocol["updates_per_arm"]==960 and protocol["updates_all_new_arms"]==1920 and protocol["VAL_prediction_products_required_before_truth"]==96 and set(protocol["parameters_per_arm"].values())=={33966},"full science")
need(caps["reservation_bytes"]==32*2**20 and caps["max_cached_parent_bytes"]==32*2**30 and caps["rss_bytes"]==128*2**30 and caps["CPU_threads"]==32 and caps["CUDA_bytes"]==0 and caps["scientific_seconds"]==86400 and caps["wall_seconds"]==86700,"exact caps")
fake={}
def fixture(name,value):
 path=OUT+"/FABRICATED_METADATA_ONLY/"+name
 fake[path]=copy.deepcopy(value);return dict(path=path,sha256=digest(path.encode()))
def mread(r):
 return copy.deepcopy(fake[r["path"]])if r["path"]in fake else read(r)
def mraw(r):
 if r["path"]in fake:
  v=fake[r["path"]];return v.encode()if isinstance(v,str)else json.dumps(v).encode()
 return raw(r)
env=dict(binding=lambda:b,read=mread,raw=mraw,ref=ref,require=need,CAPS=caps,TRAIN=tuple(protocol["TRAIN_parent_ids"]),HYPOTHESIS="ordered_marker_placement_v2",datetime=datetime)
execute_functions(HERE+"/root_contract_v1.py",["validate_config","validate_scientific_declarations","context","scientific_review","registration"],env)
cfg=copy.deepcopy(b["config_defaults"])
for k in b["operational_ref_keys"]:cfg[k]=fixture(k+".json",dict(root_authorizer="root_agent"))
directive=read(ref(HERE+"/full_training_directive.INACTIVE.template.v1.json"));directive.update(root_authorizer="root_agent",full30pass_training_authorized=True)
fake[cfg["full_training_directive_ref"]["path"]]=directive
layout=read(ref(HERE+"/layout_resource_proof.INACTIVE.template.v1.json"));layout.update(root_authorizer="root_agent",status="admitted_first_scheduled_full_parent_step")
fake[cfg["layout_resource_proof_ref"]["path"]]=layout
env["validate_config"](cfg);env["validate_scientific_declarations"](cfg);env["scientific_review"](b["scientific_review_ref"])
negative=[]
for key,val in (("checkpoint_backbone",False),("checkpoint_added",False),("CPU_prepared_cache_cap_bytes",16*2**30),("CPU_positive_plan_cap_bytes",8*2**30),("max_cpu_rss_bytes",256*2**30)):
 v=copy.deepcopy(cfg);v[key]=val;rejected(lambda v=v:env["validate_config"](v),key);negative.append(key)
for key,val in (("updates_per_arm",959),("updates_all_new_arms",1919),("passes",29),("VAL_prediction_products_required_before_truth",95)):
 v=copy.deepcopy(cfg);v["scientific_protocol"][key]=val;rejected(lambda v=v:env["validate_config"](v),key);negative.append(key)
for key,val in (("schema","incorrect"),("actual_wait_ref",None),("actual_exit_code",1),("actual_terminal_consumed",False)):
 r=cfg["one_hold_outer_terminal_ref"];v=read(r);v[key]=val;fake[r["path"]]=v
 rejected(lambda:env["validate_scientific_declarations"](cfg),"historical terminal "+key);negative.append("historical_terminal_"+key);del fake[r["path"]]
for key,val in (("returncode",1),("actual_direct_worker_wait",False),("all_owned_processes_and_groups_drained",False),("operational_failure","retained failure")):
 r=cfg["one_hold_direct_wait_ref"];v=read(r);v[key]=val;fake[r["path"]]=v
 rejected(lambda:env["validate_scientific_declarations"](cfg),"historical worker "+key);negative.append("historical_worker_"+key);del fake[r["path"]]
v=copy.deepcopy(directive);v["extra_probe_updates_authorized"]=True;fake[cfg["full_training_directive_ref"]["path"]]=v
rejected(lambda:env["validate_scientific_declarations"](cfg),"disposable probe");negative.append("extra_probe");fake[cfg["full_training_directive_ref"]["path"]]=directive
v=copy.deepcopy(layout);v["full_training_fit_measured"]=True;fake[cfg["layout_resource_proof_ref"]["path"]]=v
rejected(lambda:env["validate_scientific_declarations"](cfg),"measured fit claim");negative.append("full_fit_claim");fake[cfg["layout_resource_proof_ref"]["path"]]=layout
ctx=read(ref(HERE+"/training_context.INACTIVE.template.v1.json"));ctx["root_authorizer"]="root_agent"
cr=fixture("context.json",ctx);env["context"](cr)
v=copy.deepcopy(ctx);v["TRAIN_parent_ids"]=v["TRAIN_parent_ids"][:-1];fake[cr["path"]]=v
rejected(lambda:env["context"](cr),"missing original parent");negative.append("missing_parent");fake[cr["path"]]=ctx
reg=read(ref(HERE+"/registration.INACTIVE.template.v1.json"));reg.update(active=True,root_authorizer="root_agent",registered_utc="2026-10-17T00:00:00+00:00")
rr=cfg["experiment_registration_ref"];fake[rr["path"]]=reg
q=dict(scientific_config=cfg,registration_ref=rr,registration_decisions_ref=fixture("decisions.json",dict(decisions=[dict(id=reg["id"],registration_ref=rr)])),registration_ledger_ref=fixture("ledger.md",reg["id"]),created_at_utc="2026-10-17T00:00:01+00:00")
env["registration"](q)
for k,v in (("separate_from_H058_alignment_null",False),("TEST_authorized",True),("id","H058")):
 bad=copy.deepcopy(reg);bad[k]=v;fake[rr["path"]]=bad
 rejected(lambda:env["registration"](q),"registration "+k);negative.append("registration_"+k)
fake[rr["path"]]=reg
# Exercise the REAL selected child's extracted role constructor on all historical role names,
# then add the exact new direct integration roles. No guard is imported or state read.
deps=read(ref(HERE+"/dependencies.v1.json"));sb=read(ref(HERE+"/source_bindings.v1.json"))
guardmeta=read(sb["historical_guard_role_metadata_ref"]);guards=list(guardmeta["bindings"].values())
need(len(guards)==46,"46 complete historical role names")
renv=dict(require=need,copy=copy,_raw=lambda r:need(set(r)=={"path","sha256"},"exact fixture ref"))
child=b["selected_child_source_ref"]["path"];execute_functions(child,["consumer_bindings_v1"],renv)
roleenv=dict(binding=lambda:b,ref=ref,MANIFEST=HERE+"/source_manifest.v1.json",SOURCE=HERE+"/root_contract_v1.py",DRIVER=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",HANDOFF=HERE+"/source_handoff.v1.json",SELECTOR=b["selected_child_source_ref"],DEPS=deps,make_config=lambda q,caps:dict(scientific_config_ref=fixture("science.json",cfg)),CAPS=caps)
for k in ("NAMESPACE","PIDFD","OWNED","SUPERVISOR"):roleenv[k]=deps[k]
execute_functions(HERE+"/root_contract_v1.py",["direct_roles"],roleenv)
rq={k:fixture(k+".json",{})for k in ("source_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref")}
rq.update(scientific_review_ref=b["scientific_review_ref"],storage_policy_selector_ref=b["selected_child_selector_ref"])
roles=roleenv["direct_roles"](fixture("request.json",{}),rq,fixture("config.json",{}),fixture("closure.json",{}))
union=renv["consumer_bindings_v1"](guards,roles)
need(len(union)==63 and all(r in union for r in guards+roles),"complete real constructor role union")
need(all(Path(r["path"]).suffix in(".py",".sh",".json")for r in union),"original direct role schema")
sixtyfour=renv["consumer_bindings_v1"](union,[fixture("role64.json",{})]);need(len(sixtyfour)==64,"64 role acceptance")
rejected(lambda:renv["consumer_bindings_v1"](sixtyfour,[fixture("role65.json",{})]),"real constructor 65th role");negative.append("real_child_role65")
need(renv["consumer_bindings_v1"](union,union)==union,"only exact-reference deduplication")
out=dict(schema="ordered_marker_full_learning_independent_source_checks_v1",status="PASS",authenticated_source_metadata_refs=len(authenticated),compiled_source_files=len(compiled),inherited_data_metadata_pin_not_opened=skipped,exact_supervisor_methods_preserved=preserved,exact_original_episode_callback=True,real_selected_child_constructor_used=True,historical_guard_role_names=len(guards),unique_historical_guard_records=len({json.dumps(v,sort_keys=True)for v in guards}),complete_projected_role_union=len(union),role64_accepts=True,role65_rejects=True,negative_fixture_count=len(negative),negative_fixtures=negative,no_runtime_science_guard_imports_or_payload_truth_checkpoint_resources_accounting_admission_jobs_installs_Git=True,source_manifest_ref=ref(HERE+"/source_manifest.v1.json"),source_handoff_ref=ref(HERE+"/source_handoff.v1.json"),source_ref=ref(HERE+"/root_driver_v1.py"))
(ROOT/OUT/"checks.v1.json").write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
print(json.dumps(out,sort_keys=True))

