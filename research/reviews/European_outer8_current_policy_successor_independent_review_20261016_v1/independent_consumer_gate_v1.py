"""Exact successor admitted_v3 on virtual metadata, CPU context and synthetic guard only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,builtins,contextlib,copy,hashlib,importlib.util,json,re,sys,types
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
OLD="research/reviews/European_outer8_preparation_source_20261016_v3"
B="research/reviews/European_outer8_current_policy_successor_independent_review_20261016_v1"
def load(path,label):
 sp=importlib.util.spec_from_file_location(label,R/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def exact_function(path,name,ns):
 compiled=compile((R/path).read_bytes(),str(R/path),"exec",dont_inherit=True,optimize=0)
 code=next(x for x in compiled.co_consts if isinstance(x,types.CodeType)and x.co_name==name)
 return types.FunctionType(code,ns,name)
def fake(label):
 return dict(path=B+"/SYNTHETIC_consumer_"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
def rejection(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError)as e:return str(e)
 raise AssertionError("Expected complete consumer rejection")
def normalize_function(path,name):
 tree=ast.parse((R/path).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 for n in ast.walk(node):
  if isinstance(n,ast.ImportFrom)and n.module:
   n.module=n.module.rsplit(".",1)[-1];n.level=0
 return ast.dump(node,include_attributes=False)
assert normalize_function(C+"/check_prediction_barrier_v1.py","run_v3")==normalize_function(OLD+"/check_prediction_barrier_v2.py","run_v3")
models=[]
c0=load(C+"/prepare_outer000_v1.py","_independent_consumer_model_source")
parent=c0.read(c0.ref("research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json"))["current_guard_roles"]
child_ns=dict(copy=copy,require=c0.need,_raw=lambda r:b"SYNTHETIC",_ref=c0.ref,_manifest=lambda:None,
 SOURCE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py",
 MEASURE_SOURCE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/retained_nodes_v1.py",
 MANIFEST="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json",
 SELECTOR_SOURCE=c0.SELECTOR["path"],_POLICY_REF=fake("policy"))
# Pure binding producer has defaults, so compile definitions as AST rather than reconstructing bare functions.
txt=(R/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py").read_text()
nodes=[n for n in ast.parse(txt).body if isinstance(n,ast.FunctionDef)and n.name in ("consumer_bindings_v1","_bindings")]
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<pure_child_binding_definitions>","exec"),child_ns)
child=child_ns["_bindings"](parent,dict(root_activation_ref=fake("activation"),source_review_ref=fake("child_review"),permanent_revocation_ref=fake("revocation")))
for mode,returned,expected in (("native14_stopped_retention_v3",parent,{"validate":53,"score":56}),
 ("native7_H050_stopped_retention_v1",child,{"validate":60,"score":63})):
 for stage in ("validate","score"):
  c=load(C+"/prepare_outer000_v1.py","_independent_consumer_"+mode+"_"+stage)
  docs={};events=[]
  def put(label,value,identifier=None):
   raw=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
   rr=identifier or dict(path=B+"/SYNTHETIC_consumer_"+mode+"_"+stage+"_"+label+".json",sha256=hashlib.sha256(raw).hexdigest())
   docs[rr["path"]]=(rr,copy.deepcopy(value),raw);return rr
  raw0=c.raw;ref0=c.ref
  def raw(rr):
   if rr["path"]in docs:
    er,ev,eb=docs[rr["path"]];assert er==rr;return eb
   assert rr["path"].startswith(("research/reviews/","configs/","src/arggnn/"))and not rr["path"].startswith(("data/","experiments/results/"))
   return raw0(rr)
  c.raw=raw;c.read=lambda rr:copy.deepcopy(docs[rr["path"]][1])if rr["path"]in docs else json.loads(raw(rr))
  c.ref=lambda path:docs[str(path)][0]if str(path)in docs else ref0(path)
  now=datetime.now(timezone.utc)
  inv=c.read(c.ref(c.METHODS))
  freeze=put("freeze",dict(schema="European_outer8_method_freeze_v1",active=True,root_authorizer="root_agent",
   identity_inventory_ref=c.ref(c.METHODS),method_ids=inv["primary_method_ids"],all_methods_fixed_before_first_outer_generation=True,
   H058_automatically_included=False,outer_data_or_truth_used_for_selection=False,frozen_at_utc=(now-timedelta(hours=2)).isoformat()))
  reg_decisions=fake("decisions_"+mode+"_"+stage);ledger=put("ledger",{"entry":"SYNTHETIC_HYPOTHESIS independent registration"})
  # registration reads plain UTF-8 ledger; JSON encoded text remains valid text for the literal presence checks.
  reg=put("registration",dict(schema="European_outer8_prospective_registration_v1",root_authorizer="root_agent",status="registered",
   condition="European_outer8_frozen_predictor_independent_confirmation_v1",hypothesis_id="SYNTHETIC_HYPOTHESIS",linked_hypothesis_ids=["H038","H050"],
   method_freeze_ref=freeze,protocol_ref=c.ref(c.PROTOCOL),registry_ref=c.ref(c.REGISTRY),
   decisions_snapshot_ref=reg_decisions,ledger_snapshot_ref=ledger,registered_at_utc=(now-timedelta(hours=1)).isoformat()))
  rd=c.read(reg);decision={k:rd[k]for k in ("condition","linked_hypothesis_ids","method_freeze_ref","protocol_ref","registry_ref")}
  decision.update(id="SYNTHETIC_HYPOTHESIS",outer_tuning_authorized=False,repeat_reason="SYNTHETIC independent confirmation")
  put("decisions",dict(decisions=[decision]),identifier=reg_decisions)
  choice=c.APPROVED[mode];policy=choice["policy_ref"]or fake("policy")
  policy_review=put("policy_review",dict(status="SYNTHETIC_SOURCE_REVIEW"))
  measure="research/reviews/"+("native14_stopped_retention_successor_source_20261016_v3"if mode.startswith("native14")else"native7_H050_stopped_retention_successor_source_20261016_v1")+"/retained_nodes_v1.py"
  selector=put("selector",dict(schema="explicit_ROOT_storage_policy_selector_v1",active=True,root_authorizer="root_agent",
   selector_source_ref=c.SELECTOR,accounting_mode=mode,source_ref=choice["source_ref"],source_manifest_ref=choice["manifest_ref"],policy_ref=policy,
   measurement_source_ref=c.ref(measure),source_review_ref=policy_review,repair_authority_ref=c.REPAIR,
   no_fallback_discovery_upgrade_or_in_run_swap=True,future_ROOT_adapters_only=True,scheduling_ceiling_bytes=980000000000,
   human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000))
  entry=c.ref(C+"/check_prediction_barrier_v1.py");scientific=c.ref(OLD+"/source_manifest.v1.json");operational=c.ref(c.MANIFEST)
  entry_review=put("entry_review",dict(verdict="SOURCE_PASS",source_ref=entry))
  runtime=put("runtime",dict(schema="European_outer8_CPU_consumer_runtime_v3",device="cpu",entry_source_ref=entry,stage=stage,interpreter=str(R/".runtime/env/bin/python")))
  qualified=put("qualification",dict(schema="European_outer8_CPU_consumer_runtime_qualification_v3",root_authorizer="root_agent",status="qualified",
   unresolved_items=[],runtime_ref=runtime,source_manifest_ref=scientific,operational_manifest_ref=operational))
  leaf="validation"if stage=="validate"else"scoring";packet="research/reviews/European_outer8_ROOT_actual_SYNTHETIC_independent/"+leaf
  intent_ref=fake("intent_"+mode+"_"+stage);intent_ref["path"]=packet+"/root_request.json"
  q=dict(schema="European_outer8_actual_validation_ROOT_request_v3"if stage=="validate"else"European_outer8_actual_scoring_ROOT_request_v2",
   status="active",root_authorizer="root_agent",entry_source_ref=entry,source_manifest_ref=scientific,operational_manifest_ref=operational,
   config_ref=c.ref(c.PROTOCOL),stage=stage,output_directory=packet+"/results",storage_adapter_ref=choice["source_ref"],
   output_cap_bytes=64*2**20,observed_matrix_validation_authorized=True,outer_scoring_authorized=stage=="score",root_request_ref=intent_ref,
   runtime_ref=runtime,runtime_qualification_ref=qualified,method_freeze_ref=freeze,entry_review_ref=entry_review,
   bundle_ref=fake("bundle_"+mode+"_"+stage),job_id="SYNTHETIC_NO_ACTUAL_JOB",created_at_utc=(now-timedelta(minutes=1)).isoformat(),
   expires_at_utc=(now+timedelta(hours=20)).isoformat(),policy_selector_ref=selector,registration_ref=reg,
   registration_decisions_ref=reg_decisions,registration_ledger_ref=ledger,scorer_review_ref=fake("scorer_review"),
   actual_integration_qualification_ref=fake("integration"))
  req=fake("request_"+mode+"_"+stage);req["path"]=packet+"/input.json"
  refs=c.consumer_direct_refs_v1(q,stage)
  complete=child_ns["consumer_bindings_v1"](list(returned.values()),refs)
  assert len(complete)==expected[stage]
  q["storage"]=dict(completed_overlay_policy_ref=policy,reservation=dict(consumer_bindings=copy.deepcopy(complete)))
  intent_keys=("stage","entry_source_ref","source_manifest_ref","config_ref","runtime_ref","method_freeze_ref","entry_review_ref",
   "runtime_qualification_ref","bundle_ref","job_id","output_directory","output_cap_bytes","created_at_utc","expires_at_utc","outer_scoring_authorized",
   "operational_manifest_ref","policy_selector_ref","registration_ref","registration_decisions_ref","registration_ledger_ref")
  intent={k:copy.deepcopy(q[k])for k in intent_keys}
  intent.update(schema="European_outer8_CPU_consumer_ROOT_intent_v3",status="active",root_authorizer="root_agent")
  put("intent",intent,identifier=intent_ref);put("request",q,identifier=req)
  actual_roles=copy.deepcopy(returned)
  @contextlib.contextmanager
  def mutex(root):
   assert root==R;events.append("synthetic_mutex_enter")
   try:yield
   finally:events.append("synthetic_mutex_exit")
  def state(impl,launcher,*,fresh):
   assert fresh is False;events.append("synthetic_state_fresh_false");return [None,None,{},copy.deepcopy(actual_roles)]
  def validate_storage(spec,cfg,storage,**kw):
   assert spec==c.read(req)and kw==dict(source_ref=entry,config_ref=q["config_ref"],runtime_ref=runtime)
   events.append("synthetic_validate_storage")
  def budget_check(storage,*,boundary,required_write_bytes):
   assert boundary is True;events.append(("synthetic_boundary",required_write_bytes));return dict(status="SYNTHETIC_PASS")
  impl=put("impl",{});launcher=put("launcher",{})
  g=types.SimpleNamespace(budget=types.SimpleNamespace(SOURCE_REL=impl["path"],LAUNCHER_REL=launcher["path"],v4=types.SimpleNamespace(mutex=mutex)),
   MODE=mode,_POLICY_REF=policy,_state_locked=state,consumer_bindings_v1=child_ns["consumer_bindings_v1"],
   validate_storage_v9=validate_storage,budget_check_v9=budget_check)
  c.guard=lambda sr,consumer:(c.need(sr==selector and consumer is True,"Synthetic exact selected consumer")or g)
  c.sources=lambda:(events.append("source_closure_preverified_no_deferred_reads")or [])
  # Virtualize scientific manifest content only. Its real hash and closure were independently checked separately.
  sourceonly=put("source_only_manifest_member",dict(synthetic=True))
  put("scientific_manifest",dict(refs=[sourceonly]),identifier=scientific)
  class Reader:
   def __init__(self,guard):self.guard=guard;self.reads={}
   def read(self,rr,*,maximum=512*2**20):
    self.guard();data=raw(rr);assert len(data)<=maximum;self.reads[rr["path"]]=rr["sha256"];self.guard();return data
   def decode_json(self,rr):return json.loads(self.read(rr,maximum=32*2**20))
  validator=types.SimpleNamespace(ROOT=R,HERE=OLD,need=c.need,ref=c.ref,ObservedReaderV2=Reader)
  original_ns=dict(Path=Path,re=re,need=c.need)
  validator.record=exact_function(OLD+"/outer_prediction_validator_v2.py","record",original_ns)
  validator.record.__defaults__=(None,)
  original_ns["record"]=validator.record
  validator.reviewed_source=exact_function(OLD+"/outer_prediction_validator_v2.py","reviewed_source",original_ns)
  class VirtualPath:
   def __init__(self,path):self.path=str(path)
   @staticmethod
   def cwd():return R
   def resolve(self):return self.path
   def __truediv__(self,part):return VirtualPath(self.path+"/"+str(part))
   def __str__(self):return self.path
   def __eq__(self,other):return self.path==str(other)
  validator.ROOT=VirtualPath(R)
  def imports(name,globals=None,locals=None,fromlist=(),level=0):
   if name=="pathlib":return types.SimpleNamespace(Path=VirtualPath)
   if name=="sys":return types.SimpleNamespace(dont_write_bytecode=True,executable=str(R/".runtime/env/bin/python"))
   if level==1 and name==""and "prepare_outer000_v1"in fromlist:return types.SimpleNamespace(prepare_outer000_v1=c)
   if name==OLD.replace("/",".")+".outer_prediction_validator_v2":return validator
   raise AssertionError("Forbidden actual import "+name)
  ns=dict(__builtins__=dict(vars(builtins),__import__=imports),re=re)
  ns["direct_refs_v3"]=exact_function(C+"/check_prediction_barrier_v1.py","direct_refs_v3",ns)
  admitted=exact_function(C+"/check_prediction_barrier_v1.py","admitted_v3",ns)
  baseline=copy.deepcopy(docs)
  def restore():
   docs.clear();docs.update(copy.deepcopy(baseline));actual_roles.clear();actual_roles.update(copy.deepcopy(returned));events.clear()
  def change_doc(rr,key,value):docs[rr["path"]][1][key]=copy.deepcopy(value)
  aq,reader,boundary=admitted(req,stage)
  assert aq==q and events[:4]==["synthetic_mutex_enter","synthetic_state_fresh_false","synthetic_mutex_exit","synthetic_validate_storage"]
  boundary(0);boundary(q["output_cap_bytes"])
  for bad in (-1,1.0,q["output_cap_bytes"]+1):rejection(lambda:boundary(bad))
  negative=[]
  for label,fn in (
   ("old_new_manifest_conflation",lambda:change_doc(req,"source_manifest_ref",operational)),
   ("operational_manifest_conflation",lambda:change_doc(req,"operational_manifest_ref",scientific)),
   ("old_entry_substitution",lambda:change_doc(req,"entry_source_ref",c.ref(OLD+"/check_prediction_barrier_v2.py"))),
   ("separate_scoring_authority",lambda:change_doc(req,"outer_scoring_authorized",stage!="score")),
   ("intent_field_mismatch",lambda:change_doc(intent_ref,"output_cap_bytes",q["output_cap_bytes"]-1)),
   ("runtime_stage_mismatch",lambda:change_doc(runtime,"stage","score"if stage=="validate"else"validate")),
   ("qualification_inactive",lambda:change_doc(qualified,"status","inactive_template")),
   ("qualification_unresolved",lambda:change_doc(qualified,"unresolved_items",["pending"])),
   ("qualification_wrong_scientific_manifest",lambda:change_doc(qualified,"source_manifest_ref",operational)),
   ("qualification_wrong_operational_manifest",lambda:change_doc(qualified,"operational_manifest_ref",scientific)),
   ("registration_inactive",lambda:change_doc(reg,"status","inactive_template")),
   ("freeze_inactive",lambda:change_doc(freeze,"active",False))):
   restore();fn();negative.append(dict(name=label,message=rejection(lambda:admitted(req,stage))))
  restore()
  missing_required=refs[0]
  docs[req["path"]][1]["storage"]["reservation"]["consumer_bindings"].remove(missing_required)
  negative.append(dict(name="missing_required_direct_role",message=rejection(lambda:admitted(req,stage))))
  assert "synthetic_state_fresh_false"not in events
  restore()
  extra_returned=fake("new_required_actual_guard_"+mode+"_"+stage);actual_roles["SYNTHETIC_new_actual_returned_guard"]=extra_returned
  negative.append(dict(name="missing_actually_returned_guard",message=rejection(lambda:admitted(req,stage))))
  assert "synthetic_state_fresh_false"in events and "synthetic_validate_storage"not in events
  restore()
  # The exact same complete role helper admits exactly64 and refuses65 with all new actual guard refs direct.
  added=[]
  for i in range(64-expected[stage]):
   rr=fake("extra_"+mode+"_"+stage+"_"+str(i));added.append(rr);actual_roles["SYNTHETIC_extra_"+str(i)]=rr
  docs[req["path"]][1]["storage"]["reservation"]["consumer_bindings"].extend(added)
  admitted(req,stage)
  rr=fake("overflow_"+mode+"_"+stage);actual_roles["SYNTHETIC_overflow"]=rr
  docs[req["path"]][1]["storage"]["reservation"]["consumer_bindings"].append(rr)
  negative.append(dict(name="complete65_role_overflow",message=rejection(lambda:admitted(req,stage))))
  models.append(dict(policy=mode,stage=stage,returned_keys=len(returned),complete_unique_refs=expected[stage],
   exact64_passes=True,complete65_rejects=True,negative_cases=negative,
   exact_new_admitted_and_direct_role_functions=True,exact_original_record_and_reviewed_source_functions=True,
   all_guard_runtime_registration_manifest_reader_payloads_synthetic=True))
assert not any(n in sys.modules for n in ("scripts.project_budget_v9","arggnn.project_budget_v4","torch","numpy","tskit"))
result=dict(schema="European_outer8_successor_complete_consumer_synthetic_check_v1",status="PASS_SOURCE_ONLY",scientific_run_function_unchanged_after_import_normalization=True,
 models=models,complete_flows=len(models),negative_case_count=sum(len(x["negative_cases"])for x in models),
 no_actual_guard_configure_state_mutex_resource_probe_accounting_admission_data_model_truth_jobs_or_Git=True)
with(R/B/"independent_consumer_gate.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=result["status"],complete_flows=len(models),negative_cases=result["negative_case_count"])))
