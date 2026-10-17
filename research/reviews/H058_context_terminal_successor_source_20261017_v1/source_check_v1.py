"""Source/AST and compact metadata only. Never run an entrypoint or import a guard/model."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timedelta,timezone
import ast,copy,hashlib,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H058_context_terminal_successor_source_20261017_v1"
NEW=ROOT/HERE
OLD="research/reviews/H058_transaction_admission_source_20261017_v1"
SCI="research/reviews/H058_full_parent_CPU_execution_source_20261017_v1"
PREP="research/reviews/H058_CPU_ROOT_preparation_tx72_20261017_v1"
PACKET="research/reviews/H058_CPU_ROOT_tx72_20261017_v1"
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def fn(text,name):
 return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def isolate(nodes,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<source-ast-only>","exec"),scope)
def negative(label,call,out):
 try:call()
 except(ValueError,KeyError,AssertionError,TypeError)as error:out.append(dict(case=label,error=str(error)));return
 raise AssertionError("Malformed metadata accepted: "+label)
class Reader:
 root=ROOT
 def __init__(self):
  self.memory={};self.overrides={};self.authenticated={};self.source_reads=0
 def read(self,r,**kwargs):
  need(isinstance(r,dict)and set(r)=={"path","sha256"},"Exact metadata ref")
  if r["path"]in self.memory:
   data=self.memory[r["path"]]
   need(hashlib.sha256(data).hexdigest()==r["sha256"],"Exact handwritten memory fixture");return data
  p=ROOT/r["path"]
  need(p.suffix in(".json",".py",".md",".patch",".sh"),"NO PAYLOAD/TRUTH/CHECKPOINT READ")
  data=p.read_bytes();need(len(data)<=4*2**20 and hashlib.sha256(data).hexdigest()==r["sha256"],"Bound compact metadata/source")
  self.authenticated[r["path"]]=r;self.source_reads+=1;return data
 def decode_json(self,r):
  authenticated=json.loads(self.read(r))
  return copy.deepcopy(self.overrides.get(r["path"],authenticated))
 def add_memory(self,name,value):
  data=(json.dumps(value,sort_keys=True)+"\n").encode();p="HANDWRITTEN_METADATA_ONLY/"+name+".json";self.memory[p]=data
  return dict(path=p,sha256=hashlib.sha256(data).hexdigest())

def check():
 b=json.loads((NEW/"source_bindings.v1.json").read_text());bind=json.loads((NEW/"scientific_binding.v1.json").read_text())
 current=(NEW/"root_driver_v1.py").read_text();prior=(ROOT/OLD/"root_driver_v1.py").read_text()
 producer=(NEW/"producer_v1.py").read_text();old_producer=(ROOT/SCI/"producer_v1.py").read_text()
 for p in NEW.glob("*.py"):compile(p.read_text(),str(p),"exec")
 # Whole producer equality after reversing only current source/dependency identities and exact terminal predicate.
 restored=producer.replace("from "+SCI.replace("/",".")+".protocol_v1 import *","from .protocol_v1 import *").replace("from "+SCI.replace("/",".")+" import model_v1 as model,observations_v1 as observations,matrix_v1 as matrix,diagnostics_v1 as diagnostics","from . import model_v1 as model,observations_v1 as observations,matrix_v1 as matrix,diagnostics_v1 as diagnostics")
 restored=restored.replace('HERE="'+HERE+'"','HERE="'+SCI+'"').replace('MANIFEST=HERE+"/producer_manifest.v1.json"','MANIFEST=HERE+"/source_manifest.v1.json"')
 old_check=fn(old_producer,"validate_context_v1");new_check=fn(producer,"validate_context_v1")
 message="Genuine successful one-hold arithmetic qualification and consumed direct/outer waits"
 def qual_call(node):
  return next(n.value for n in node.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(a,ast.Constant)and a.value==message for a in n.value.args))
 old_gate=qual_call(old_check).args[0];new_gate=qual_call(new_check).args[0]
 need(len(old_gate.values)==15and len(new_gate.values)==16,"Original14plus exact schema/actual-wait join")
 need([dump(n)for n in old_gate.values[:14]]==[dump(n)for n in new_gate.values[:14]],"All original14conjuncts exact")
 for expr,node in [('terminal.get("schema")=="actual_root_tool_terminal_v1"',new_gate.values[14]),('terminal.get("actual_wait_ref")==cfg["one_hold_direct_wait_ref"]',new_gate.values[15])]:
  need(dump(ast.parse(expr,mode="eval").body)==dump(node),"Only exact terminal schema and actual wait")
 restored_tree=ast.parse(restored);restored_gate=qual_call(fn(restored,"validate_context_v1")).args[0]
 restored_gate.values=copy.deepcopy(old_gate.values)
 # Replace the matching gate in the tree, not a detached parse.
 gate_in_tree=qual_call(next(n for n in restored_tree.body if isinstance(n,ast.FunctionDef)and n.name=="validate_context_v1")).args[0]
 gate_in_tree.values=copy.deepcopy(old_gate.values)
 need(dump(restored_tree)==dump(ast.parse(old_producer)),"Whole original producer numeric/algorithm AST retained")
 # Reverse only declared fresh operational identity/failure joins; all current memo and lifecycle code exact.
 tree=ast.parse(current.replace(HERE,OLD));contract=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="contract")
 contract.body=[n for n in contract.body if not(isinstance(n,ast.FunctionDef)and n.name=="accept_context_failure")]
 roles=next(n for n in contract.body if isinstance(n,ast.FunctionDef)and n.name=="direct_roles")
 loop=next(n for n in roles.body if isinstance(n,ast.For));need(len(loop.iter.elts)==4,"Exactly2new retained direct identities");loop.iter.elts=loop.iter.elts[:2]
 validate=next(n for n in contract.body if isinstance(n,ast.FunctionDef)and n.name=="validate_request")
 body=[]
 for n in validate.body:
  text=ast.unparse(n)
  if isinstance(n,ast.Expr)and ("accept_context_failure("in text or "Exact one context-terminal correction condition"in text):continue
  if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(a,ast.Constant)and a.value=="Separate ROOT directive explicitly binds unfinished-cell condition/failure and72h authority"for a in n.value.args):
   n.value.args[0].values=[v for v in n.value.args[0].values if "context_terminal_"not in ast.unparse(v)]
  body.append(n)
 validate.body=body
 need(dump(tree)==dump(ast.parse(prior)),"Whole current tx/memo/72h/science/lifecycle AST retained except explicit new failure joins")
 need((NEW/"worker_entry_v1.py").read_bytes()==(ROOT/OLD/"worker_entry_v1.py").read_bytes(),"Exact current worker")
 for name in("_make_root_mutation_memo","_memo_admission_node","_install_memo_admit","authority_interval","driver","worker"):
  need(dump(fn(current,name))==dump(fn(prior,name)),"Current helper exact "+name)
 oldbind=json.loads((ROOT/b["base_directory"]/"scientific_binding.v1.json").read_text())
 for k,v in oldbind.items():
  if k not in("runner_module","runner_source_ref","source_manifest_ref","source_handoff_ref","config_defaults"):need(bind[k]==v,"Original scientific binding "+k)
 for k,v in oldbind["config_defaults"].items():
  if k not in("producer_ref","source_manifest_ref"):need(bind["config_defaults"][k]==v,"Unchanged scientific configuration "+k)
 # Authentic historical complete context: only read source/metadata and never call factory/episodes/models.
 reader=Reader();packet=reader.decode_json(ref(PACKET+"/input.json"));op=reader.decode_json(packet["refs"]["config"])
 cfgref=op["scientific_config_ref"];cfg=reader.decode_json(cfgref)
 oldinput=reader.decode_json(oldbind["original_H050_input_ref"]);targets=reader.decode_json(oldbind["original_TRAIN_bindings_ref"])["parents"]
 protocol=(ROOT/SCI/"protocol_v1.py").read_text();scope=dict(Path=Path,ROOT=ROOT,os=SimpleNamespace(environ={"CUDA_VISIBLE_DEVICES":""}))
 # Protocol is only inert constants and metadata helpers; no module import.
 protocol_nodes=[n for n in ast.parse(protocol).body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 isolate(protocol_nodes,scope)
 scope.update(SOURCE=SCI+"/producer_v1.py",MANIFEST=SCI+"/source_manifest.v1.json",CAP_NAMES=tuple(ast.literal_eval(next(n.value for n in ast.parse(old_producer).body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="CAP_NAMES"for x in n.targets)))),actual_ref=ref)
 callbacks={"metadata_guard":0,"guard_factory":0,"episodes":0}
 def actual_guard_factory(*args,**kwargs):callbacks["guard_factory"]+=1;raise AssertionError("REAL GUARD FACTORY MUST NEVER BE TOUCHED")
 def metadata_only_boundary():callbacks["metadata_guard"]+=1
 def no_episode(*args,**kwargs):callbacks["episodes"]+=1;raise AssertionError("NO SCIENTIFIC EPISODE")
 ctx=dict(root_authorizer="root_agent",admitted=True,hypothesis_id="H058",phase="train",guard=metadata_only_boundary,before_episode_build=no_episode,
  cfg=cfg,config_ref=cfgref,reader=reader,device=SimpleNamespace(type="cpu"),physical_gpu=None,entry_refs=oldinput["observation_entry_refs"],bindings=targets,observation_source_refs=oldinput["observation_cache_source_refs"])
 isolate([old_check],scope)
 old_failure=[];negative("original_exact_whole_context",lambda:scope["validate_context_v1"](ctx),old_failure)
 need(old_failure[0]["error"]==message,"Original whole context reproduces actual error")
 isolate([new_check],scope)
 need(scope["validate_context_v1"](ctx)==cfg,"Repaired exact whole historical context passes all later metadata predicates")
 actual_qualification_refs={k:cfg[k]for k in("one_hold_completion_ref","one_hold_direct_wait_ref","one_hold_outer_terminal_ref")}
 completion=reader.decode_json(cfg["one_hold_completion_ref"]);actual_qualification_refs["receipt_ref"]=completion["qualification_ref"]
 actual_terminal=reader.decode_json(cfg["one_hold_outer_terminal_ref"])
 need("direct_wait_ref"not in actual_terminal and actual_terminal["schema"]=="actual_root_tool_terminal_v1"and actual_terminal["actual_wait_ref"]==cfg["one_hold_direct_wait_ref"],"Genuine exact actual historical schema")
 # Fresh source identity join is exercised in memory only; no actual ROOT declarations are written.
 source_cfg=copy.deepcopy(cfg);source_cfg.update(producer_ref=bind["runner_source_ref"],source_manifest_ref=bind["source_manifest_ref"])
 directive=reader.decode_json(cfg["full_training_directive_ref"]);directive.update(CPU_entry_ref=bind["runner_source_ref"],SOURCE_ONLY_HANDWRITTEN_IDENTITY_FIXTURE_NOT_AUTHORITY=True)
 source_cfg["full_training_directive_ref"]=reader.add_memory("full_training_directive_identity_projection",directive)
 source_cfg_ref=reader.add_memory("scientific_config_identity_projection",source_cfg)
 source_ctx=dict(ctx,cfg=source_cfg,config_ref=source_cfg_ref)
 scope.update(SOURCE=HERE+"/producer_v1.py",MANIFEST=HERE+"/producer_manifest.v1.json")
 need(scope["validate_context_v1"](source_ctx)==source_cfg,"Current source-bound whole context passes with genuine historical qualification")
 negatives=[]
 qualification_records=dict(completed=cfg["one_hold_completion_ref"],receipt=completion["qualification_ref"],wait=cfg["one_hold_direct_wait_ref"],terminal=cfg["one_hold_outer_terminal_ref"])
 mutations=[
  ("completed","status","wrong"),("receipt","status","FAIL"),("receipt","optimizer_updates",1),("receipt","schema","wrong"),("receipt","parent_id","wrong"),
  ("completed","actual_wait_ref",{}),("wait","completed_ref",{}),("wait","returncode",1),("wait","actual_direct_worker_wait",False),
  ("wait","all_owned_processes_and_groups_drained",False),("wait","operational_failure","failed"),("terminal","actual_exit_code",1),
  ("terminal","actual_terminal_consumed",False),("terminal","completed_ref",{}),("terminal","schema","wrong"),("terminal","actual_wait_ref",{})]
 for obj,key,val in mutations:
  r=qualification_records[obj];changed=reader.decode_json(r);changed[key]=val;reader.overrides[r["path"]]=changed
  negative("whole_context:"+obj+":"+key,lambda:scope["validate_context_v1"](source_ctx),negatives);reader.overrides.clear()
 # Construct exact isolated new contract with guard factory hard-failing; no real guard source imported.
 deps=json.loads((ROOT/b["base_directory"]/"dependencies.v1.json").read_text());selector_value=reader.decode_json(b["actual_selector_metadata_ref"])
 selector=SimpleNamespace(SELECTOR_REF=None,_CHOSEN=None,SELECTOR=None,GUARD_REF=None,POLICY_REF=None,guard=actual_guard_factory)
 def selected(r):need(r==b["actual_selector_metadata_ref"],"Exact selected metadata");return copy.deepcopy(selector_value)
 def select(r):
  v=selected(r);selector.SELECTOR_REF=r;selector._CHOSEN=v;selector.SELECTOR=v["selector_source_ref"];selector.GUARD_REF=v["source_ref"];selector.POLICY_REF=v["policy_ref"];return v
 selector.selected=selected;selector.bind_selector=select
 P=lambda name:dict(path="HANDWRITTEN_METADATA_ONLY/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
 core=SimpleNamespace(require=need,DEPS=deps,put=lambda *a,**k:None,validate_request=lambda r:("ORIGINAL_VALIDATION_REACHED",r),input_spec=lambda r:None,
  read=reader.decode_json,raw=reader.read,ref=ref,local=lambda p:ROOT/p,CAPS=b["caps"],SELECTOR_REF=None,_CHOSEN=None,SELECTOR=None,GUARD_REF=None,POLICY_REF=None)
 role_scope=core.__dict__;role_scope.update(NAMESPACE=deps["NAMESPACE"],PIDFD=deps["PIDFD"],OWNED=deps["OWNED"],SUPERVISOR=deps["SUPERVISOR"],
  binding=lambda:bind,MANIFEST=HERE+"/source_manifest.v1.json",SOURCE=HERE+"/root_driver_v1.py",DRIVER=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",HANDOFF=HERE+"/source_handoff.v1.json",
  make_config=lambda q,caps:{"scientific_config_ref":P("scientific_config")})
 contract_text=(ROOT/b["contract_ref"]["path"]).read_text();isolate([fn(contract_text,"direct_roles")],role_scope)
 env=dict(Path=Path,copy=copy,datetime=datetime,need=need,HERE=HERE,SOURCE=HERE+"/root_driver_v1.py",ENTRY=HERE+"/worker_entry_v1.py",MANIFEST=HERE+"/source_manifest.v1.json",HANDOFF=HERE+"/source_handoff.v1.json",
  bindings=lambda:b,read=reader.decode_json,load=lambda r,name:core if r==b["contract_ref"]else selector)
 isolate([fn(current,"contract"),fn(current,"authority_interval")],env);core=env["contract"]();core.bind_selector(b["actual_selector_metadata_ref"])
 q={k:P(k)for k in("source_review_ref","scientific_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref","context_terminal_failure_acceptance_ref")}
 q["storage_policy_selector_ref"]=b["actual_selector_metadata_ref"]
 # Current manifest/handoff are not inputs to their own content hashes: synthetic source refs only in this role fixture.
 def role_ref(path):
  if str(path)in(HERE+"/source_manifest.v1.json",HERE+"/source_handoff.v1.json"):return dict(path=str(path),sha256=hashlib.sha256(str(path).encode()).hexdigest())
  return ref(path)
 core.ref=role_ref;core.__dict__["ref"]=role_ref
 direct=core.direct_roles(P("request"),q,P("config"),P("closure"))
 guards=reader.decode_json(b["actual_guard_role_metadata_ref"])["bindings"]
 union_scope=dict(copy=copy,require=need,_raw=lambda r:need(isinstance(r,dict)and set(r)=={"path","sha256"},"Exact memory roles"))
 child=(ROOT/b["selected_child_source_ref"]["path"]).read_text();isolate([fn(child,"consumer_bindings_v1")],union_scope)
 combined=union_scope["consumer_bindings_v1"](list(guards.values()),direct)
 need(len(combined)==64and all(r in combined for r in guards.values())and b["original_runner_source_ref"]in combined and bind["runner_source_ref"]in combined and deps["SELECTOR"]in combined and b["selected_child_source_ref"]in combined,"Complete unchanged/current64roles; no trimming")
 negative("role65",lambda:union_scope["consumer_bindings_v1"](combined+[P("extra65")]),negatives)
 need(b["transaction_memo"]==json.loads((ROOT/OLD/"source_bindings.v1.json").read_text())["transaction_memo"],"Exact actual current memo-v1 component")
 need(b["transaction_memo"]["source_ref"]["sha256"]=="2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98","No unrequested memo swap")
 created=datetime(2030,1,1,tzinfo=timezone.utc);env["authority_interval"](dict(created_at_utc=created.isoformat(),expires_at_utc=(created+timedelta(hours=72)).isoformat()))
 negative("not72h",lambda:env["authority_interval"](dict(created_at_utc=created.isoformat(),expires_at_utc=(created+timedelta(hours=71)).isoformat())),negatives)
 # Exercise the complete new failure-history wrapper with genuine old acceptance and current consumed failure metadata.
 old_request=reader.decode_json(packet["root_stage_request_ref"])
 accepted_latest=dict(b["context_terminal_failure"],schema="ROOT_H058_context_terminal_failure_acceptance_v1",root_authorizer="root_agent",accepted=True,
  old_launch_resume_write_authority_revoked=True,old_files_key_charge_clocks_and_failure_unchanged=True,qualification_success_preserved_without_repeat=True,SOURCE_ONLY_MEMORY_NOT_AUTHORITY=True)
 latest_ref=reader.add_memory("latest_failure_acceptance_projection",accepted_latest)
 request=dict(source_manifest_ref=role_ref(HERE+"/source_manifest.v1.json"),caps=b["caps"],cpu_ids=b["cpu_ids"],fresh_lifetime_condition=b["condition"],
  context_terminal_successor_condition=b["context_terminal_condition"],context_terminal_failure_acceptance_ref=latest_ref,prior_failure_acceptance_ref=old_request["prior_failure_acceptance_ref"],
  preparation_directory="research/reviews/H058_CPU_ROOT_preparation_HANDWRITTEN_NEW",packet_directory="research/reviews/H058_CPU_ROOT_HANDWRITTEN_NEW",output_directory="experiments/results/H058_full_parent_CPU_HANDWRITTEN_NEW",
  created_at_utc="2030-01-01T00:00:00+00:00",expires_at_utc="2030-01-04T00:00:00+00:00")
 directive=dict(fresh_lifetime_condition=b["condition"],prior_failure_acceptance_ref=request["prior_failure_acceptance_ref"],fresh_authority_seconds=259200,
  context_terminal_successor_condition=b["context_terminal_condition"],context_terminal_failure_acceptance_ref=latest_ref,SOURCE_ONLY_MEMORY_NOT_AUTHORITY=True)
 request["root_directive_ref"]=reader.add_memory("latest_directive_projection",directive)
 request_ref=reader.add_memory("latest_request_projection",request)
 need(core.validate_request(request_ref)[0]=="ORIGINAL_VALIDATION_REACHED","Both original7922and genuine consumed69608history gates precede unchanged original validation")
 for field in list(b["context_terminal_failure"])+["accepted","old_launch_resume_write_authority_revoked","old_files_key_charge_clocks_and_failure_unchanged","qualification_success_preserved_without_repeat"]:
  bad=copy.deepcopy(accepted_latest);bad[field]={"changed":True};reader.overrides[latest_ref["path"]]=bad
  negative("latest_failure_acceptance:"+field,lambda:core.validate_request(request_ref),negatives);reader.overrides.clear()
 need(callbacks["guard_factory"]==callbacks["episodes"]==0,"Guard factory and scientific episode never touched")
 need(not any(n in sys.modules for n in("torch","numpy","torch_geometric","tskit")),"No numerical imports")
 return dict(schema="H058_exact_context_terminal_successor_source_checks_v1",status="PASS_SOURCE_ONLY",independent=False,source_only=True,
  producer_ref=ref(HERE+"/producer_v1.py"),driver_ref=ref(HERE+"/root_driver_v1.py"),
  whole_original_producer_AST_except_exact_join_and_source_import_identities=True,all14qualification_predicates_exact=True,
  exact_current_transaction_driver_AST_except_identity_and_two_failure_history_joins=True,current_memo_v1_exact=True,
  exact_original_worker_bytes=True,full_science_protocol_caps_and72h_unchanged=True,
  whole_actual_historical_context_original_error_reproduced=old_failure,
  whole_actual_historical_context_with_repaired_join_passes=True,current_identity_whole_context_with_genuine_qualification_passes=True,
  latest_failure_acceptance_with_actual_ROOT_terminal_passes=True,genuine_qualification_refs=actual_qualification_refs,guard_factory_calls=callbacks["guard_factory"],scientific_episode_calls=callbacks["episodes"],metadata_callback_calls=callbacks["metadata_guard"],
  guard_names=len(guards),distinct_guard_roles=len({(r["path"],r["sha256"])for r in guards.values()}),complete_role_union=len(combined),all_original_guard_direct_scientific_roles_preserved=True,
  authenticated_context_metadata_and_sources=list(reader.authenticated.values()),negative_cases=negatives,
  actual_guard_state_mutex_resource_process_payload_truth_checkpoint_scientific_job_admission_restart_or_Git=False,
  fresh_operational_declarations_in_test="HANDWRITTEN_MEMORY_IDENTITY_PROJECTION_ONLY_NOT_AUTHORITY",
  no_actual_entrypoint_guard_or_model_import=True)
if __name__=="__main__":print(json.dumps(check(),indent=2,sort_keys=True))
