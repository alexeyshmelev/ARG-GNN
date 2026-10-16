"""Generation request/admit/input and default entry points in an entirely virtual filesystem."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,contextlib,copy,hashlib,importlib.util,io,json,sys,types
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
B="research/reviews/European_outer8_current_policy_successor_independent_review_20261016_v1"
def load(path,label):
 sp=importlib.util.spec_from_file_location(label,R/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def fake(label):return dict(path=B+"/SYNTHETIC_generation_"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError)as e:return str(e)
 raise AssertionError("Expected generation rejection")
def rel(path):
 s=str(path);return s[len(str(R))+1:]if s.startswith(str(R)+"/")else s
# Source-derived parent/child role models and exact pure64-ref helper.
c0=load(C+"/prepare_outer000_v1.py","_independent_generation_model")
parent=c0.read(c0.ref("research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json"))["current_guard_roles"]
ns=dict(copy=copy,require=c0.need,_raw=lambda r:b"SYNTHETIC",_ref=c0.ref,_manifest=lambda:None,
 SOURCE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py",
 MEASURE_SOURCE="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/retained_nodes_v1.py",
 MANIFEST="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json",
 SELECTOR_SOURCE=c0.SELECTOR["path"],_POLICY_REF=fake("policy"))
p=R/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py"
nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in ("consumer_bindings_v1","_bindings")]
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<pure_child_bindings>","exec"),ns)
child=ns["_bindings"](parent,dict(root_activation_ref=fake("activation"),source_review_ref=fake("child_review"),permanent_revocation_ref=fake("revocation")))
flows=[]
for mode,returned,count in (("native14_stopped_retention_v3",parent,52),("native7_H050_stopped_retention_v1",child,59)):
 c=load(C+"/prepare_outer000_v1.py","_independent_generation_"+mode)
 docs={};events=[];paths=set();actual_roles=copy.deepcopy(returned)
 def put(label,value,identifier=None):
  raw=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
  rr=identifier or dict(path=B+"/SYNTHETIC_generation_"+mode+"_"+label+".json",sha256=hashlib.sha256(raw).hexdigest())
  docs[rr["path"]]=(rr,copy.deepcopy(value),raw);return rr
 raw0=c0.raw;ref0=c0.ref
 def raw(rr):
  if rr["path"]in docs:
   er,ev,eb=docs[rr["path"]];assert rr==er;return eb
  assert rr["path"].startswith(("research/reviews/","configs/","src/arggnn/"))and not rr["path"].startswith(("data/","experiments/results/"))
  return raw0(rr)
 c.raw=raw;c.read=lambda rr:copy.deepcopy(docs[rr["path"]][1])if rr["path"]in docs else json.loads(raw(rr))
 c.ref=lambda path:docs[rel(path)][0]if rel(path)in docs else ref0(path)
 c.sources=lambda:(events.append("source_closure_preverified_no_deferred_reads")or [])
 c.os=types.SimpleNamespace(sched_getaffinity=lambda pid:(c.need(pid==0,"Synthetic local affinity")or set(range(8))))
 c.sys=types.SimpleNamespace(dont_write_bytecode=True)
 class VPath:
  def __init__(self,path):self.path=rel(path)
  def __str__(self):return str(R/self.path)
  def __truediv__(self,part):return VPath(self.path+"/"+str(part))
  def exists(self):return self.path in paths or self.path in docs
  def mkdir(self):assert not self.exists();paths.add(self.path);events.append("synthetic_packet_mkdir")
 class P:
  @staticmethod
  def cwd():return R
 c.Path=P;c.local=lambda path:VPath(path)
 def vput(path,value,cap=2**20):
  name=rel(path);assert name not in docs
  raw=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode();assert len(raw)<=cap
  rr=dict(path=name,sha256=hashlib.sha256(raw).hexdigest());docs[name]=(rr,copy.deepcopy(value),raw)
  events.append(("virtual_publish",name));return rr
 c.put=vput
 now=datetime.now(timezone.utc)
 inv=c.read(c.ref(c.METHODS))
 freeze=put("freeze",dict(schema="European_outer8_method_freeze_v1",active=True,root_authorizer="root_agent",
  identity_inventory_ref=c.ref(c.METHODS),method_ids=inv["primary_method_ids"],all_methods_fixed_before_first_outer_generation=True,
  H058_automatically_included=False,outer_data_or_truth_used_for_selection=False,frozen_at_utc=(now-timedelta(hours=2)).isoformat()))
 decisions=fake("decisions_"+mode);ledger=put("ledger",{"entry":"SYNTHETIC_HYPOTHESIS independent registration"})
 regdoc=dict(schema="European_outer8_prospective_registration_v1",root_authorizer="root_agent",status="registered",
  condition="European_outer8_frozen_predictor_independent_confirmation_v1",hypothesis_id="SYNTHETIC_HYPOTHESIS",linked_hypothesis_ids=["H038","H050"],
  method_freeze_ref=freeze,protocol_ref=c.ref(c.PROTOCOL),registry_ref=c.ref(c.REGISTRY),decisions_snapshot_ref=decisions,ledger_snapshot_ref=ledger,
  registered_at_utc=(now-timedelta(hours=1)).isoformat())
 registration=put("registration",regdoc)
 decision={k:regdoc[k]for k in ("condition","linked_hypothesis_ids","method_freeze_ref","protocol_ref","registry_ref")}
 decision.update(id="SYNTHETIC_HYPOTHESIS",outer_tuning_authorized=False,repeat_reason="SYNTHETIC independent confirmation")
 put("decisions",dict(decisions=[decision]),identifier=decisions)
 policyreview=put("policyreview",dict(synthetic=True));choice=c.APPROVED[mode];policy=choice["policy_ref"]or fake("policy")
 measure="research/reviews/"+("native14_stopped_retention_successor_source_20261016_v3"if mode.startswith("native14")else"native7_H050_stopped_retention_successor_source_20261016_v1")+"/retained_nodes_v1.py"
 selected=put("selector",dict(schema="explicit_ROOT_storage_policy_selector_v1",active=True,root_authorizer="root_agent",
  selector_source_ref=c.SELECTOR,accounting_mode=mode,source_ref=choice["source_ref"],source_manifest_ref=choice["manifest_ref"],policy_ref=policy,
  measurement_source_ref=c.ref(measure),source_review_ref=policyreview,repair_authority_ref=c.REPAIR,
  no_fallback_discovery_upgrade_or_in_run_swap=True,future_ROOT_adapters_only=True,scheduling_ceiling_bytes=980000000000,
  human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000))
 cfg=c.read(c.CONFIG)
 runtime=put("runtime",dict(schema="European_outer000_CPU_runtime_v1",device="cpu",CUDA_VISIBLE_DEVICES="",
  interpreter=str(R/".runtime/env/bin/python"),config_runtime=cfg["runtime"],native_threads=8,cpu_ids=list(range(8))))
 qualified=put("qualified",dict(schema="European_outer000_runtime_qualification_v1",root_authorizer="root_agent",status="qualified",
  runtime_ref=runtime,operational_manifest_ref=c.ref(c.MANIFEST),config_ref=c.CONFIG,producer_ref=c.PRODUCER,unresolved_items=[]))
 source_review=put("source_review",dict(verdict="SOURCE_PASS",source_manifest_ref=c.ref(c.MANIFEST)))
 q=dict(schema="European_outer000_preparation_ROOT_request_v1",status="active",root_authorizer="root_agent",
  generation_and_native_Relate_authorized=True,GPU_authorized=False,outer_scoring_authorized=False,
  source_manifest_ref=c.ref(c.MANIFEST),caps=c.CAPS,parent_id=c.PARENT,output_directory=c.OUTPUT,
  packet_directory="research/reviews/European_outer000_ROOT_actual_SYNTHETIC_"+mode,automatic_retry_or_reseed=False,
  created_at_utc=(now-timedelta(minutes=1)).isoformat(),expires_at_utc=(now+timedelta(hours=20)).isoformat(),
  cpu_ids=list(range(8)),runtime_ref=runtime,runtime_qualification_ref=qualified,source_review_ref=source_review,method_freeze_ref=freeze,
  policy_selector_ref=selected,registration_ref=registration,registration_decisions_ref=decisions,registration_ledger_ref=ledger)
 req=put("request",q);impl=put("impl",{});launcher=put("launcher",{});pins=dict(SYNTHETIC="prior_exact_pins")
 @contextlib.contextmanager
 def mutex(root):
  assert root==R;events.append("synthetic_mutex_enter")
  try:yield
  finally:events.append("synthetic_mutex_exit")
 def state(i,l,*,fresh):
  assert i==impl["sha256"]and l==launcher["sha256"]and fresh is True;events.append("synthetic_state_fresh_true")
  return [None,None,pins,copy.deepcopy(actual_roles)]
 def draft(op,own,*,expected_before,bindings):
  assert op=="reserve"and expected_before==pins and bindings==actual_roles
  assert own["reserved_bytes"]==c.CAPS["reservation_bytes"]and own["output"]==str(R/c.OUTPUT)
  assert all(rr in own["consumer_bindings"]for rr in actual_roles.values())
  events.append("synthetic_draft_reserve")
  return dict(schema="SYNTHETIC_RESERVE_AUTHORITY",value=copy.deepcopy(own),expected_before=copy.deepcopy(expected_before))
 def apply(ar,*,expected_source_sha256,expected_launcher_sha256):
  assert expected_source_sha256==impl["sha256"]and expected_launcher_sha256==launcher["sha256"]
  auth=c.read(ar);assert auth["active"]is True and auth["expires_at_utc"]==q["expires_at_utc"]
  events.append("synthetic_apply_reserve");return dict(value=auth["value"],initial_ledger_pins=copy.deepcopy(pins))
 g=types.SimpleNamespace(budget=types.SimpleNamespace(SOURCE_REL=impl["path"],LAUNCHER_REL=launcher["path"],v4=types.SimpleNamespace(mutex=mutex)),
  MODE=mode,_POLICY_REF=policy,_state_locked=state,_pins=lambda value:copy.deepcopy(value),
  consumer_bindings_v1=ns["consumer_bindings_v1"],operation_draft_v1=draft,apply_reserve_v1=apply)
 c.guard=lambda sr,consumer:(c.need(sr==selected and consumer is False,"Synthetic exact generation guard without memo")or g)
 baseline=copy.deepcopy(docs)
 def restore():
  docs.clear();docs.update(copy.deepcopy(baseline));paths.clear();events.clear();actual_roles.clear();actual_roles.update(copy.deepcopy(returned))
 def change(rr,key,val):docs[rr["path"]][1][key]=copy.deepcopy(val)
 c.request(req);result=c.admit(req)
 assert result["admission_only"]is True and events.count("synthetic_apply_reserve")==1
 spec,aq,acfg=c.input_spec(result["input_ref"])
 assert aq==q and acfg==cfg and spec["runtime_ref"]==runtime
 assert spec["outer_evaluation_authorized"]is False and spec["allowed_stages"]==["all"]
 assert spec["storage"]["completed_overlay_policy_ref"]==policy and spec["storage"]["reserved_new_bytes"]==4*2**30
 assert len(spec["storage"]["reservation"]["consumer_bindings"])==count
 assert len([x for x in events if isinstance(x,tuple)and x[0]=="virtual_publish"])==5
 assert c.OUTPUT not in paths
 negative=[]
 for label,fn in (
  ("inactive_root_request",lambda:change(req,"status","inactive_template")),
  ("gpu_authority_substitution",lambda:change(req,"GPU_authorized",True)),
  ("outer_scoring_substitution",lambda:change(req,"outer_scoring_authorized",True)),
  ("retry_authorization_substitution",lambda:change(req,"automatic_retry_or_reseed",True)),
  ("scientific_cap_substitution",lambda:change(req,"caps",dict(c.CAPS,output_bytes=4000000001))),
  ("wrong_parent_substitution",lambda:change(req,"parent_id","ap_expanded_outer_001")),
  ("unavailable_cpu_identity",lambda:change(req,"cpu_ids",list(range(7))+[99])),
  ("wrong_source_manifest_review",lambda:change(source_review,"source_manifest_ref",c.ref(c.BASE_HERE+"/source_manifest.v1.json"))),
  ("inactive_generation_qualification",lambda:change(qualified,"status","inactive_template")),
  ("runtime_cpu_mismatch",lambda:change(runtime,"cpu_ids",list(range(7)))),
  ("registration_absent",lambda:change(registration,"status","inactive_template")),
  ("outer_tuning_registered",lambda:docs[decisions["path"]][1]["decisions"][0].update(outer_tuning_authorized=True))):
  restore();fn();negative.append(dict(name=label,message=reject(lambda:c.admit(req))))
  assert "synthetic_apply_reserve"not in events
 restore()
 for i in range(64-count):actual_roles["SYNTHETIC_extra_"+str(i)]=fake("extra_"+mode+"_"+str(i))
 limitresult=c.admit(req)
 assert len(c.read(limitresult["input_ref"])["storage"]["reservation"]["consumer_bindings"])==64
 restore()
 for i in range(65-count):actual_roles["SYNTHETIC_extra_"+str(i)]=fake("extra_"+mode+"_"+str(i))
 negative.append(dict(name="complete65_role_overflow",message=reject(lambda:c.admit(req))))
 assert "synthetic_apply_reserve"not in events
 flows.append(dict(policy=mode,returned_keys=len(returned),complete_unique_refs=count,exact64_passes=True,complete65_blocks_before_reserve=True,
  one_virtual_reserve_exact_pins_original_caps=True,request_and_input_checks_pass=True,negative_cases=negative))
# Defaults are executed with no arguments and redirected stdout; no source/guard function runs.
prep=load(C+"/prepare_outer000_v1.py","_independent_inactive_preparation")
consumer=load(C+"/check_prediction_barrier_v1.py","_independent_inactive_consumer")
def forbidden(*a,**k):raise AssertionError("Inactive default made forbidden call")
prep.guard=prep.sources=prep.read=prep.request=prep.admit=prep.execute=forbidden
args=sys.argv;sys.argv=["independent_inactive_default"]
try:
 out=io.StringIO()
 with contextlib.redirect_stdout(out):code=prep.main()
 value=json.loads(out.getvalue());assert code==0 and value==dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)
finally:sys.argv=args
consumer.admitted_v3=consumer.run_v3=forbidden
out=io.StringIO()
with contextlib.redirect_stdout(out):code=consumer.main([])
value=json.loads(out.getvalue());assert code==2 and value==dict(status="INACTIVE_SOURCE_ONLY",actual_execution=False,observed_matrix_validated=False,scoring_authorized=False)
assert not any(name in sys.modules for name in ("torch","numpy","tskit","scripts.project_budget_v9","arggnn.project_budget_v4"))
result=dict(schema="European_outer8_successor_virtual_generation_and_defaults_v1",status="PASS_SOURCE_ONLY",flows=flows,complete_flows=len(flows),
 negative_case_count=sum(len(x["negative_cases"])for x in flows),both_defaults_inactive_no_source_reads_or_guardcalls=True,
 exact_real_candidate_request_admit_and_input_functions=True,all_filesystem_publication_affinity_guard_state_reservation_and_authority_synthetic=True,
 no_actual_data_outer_path_array_model_truth_probe_accounting_configure_admission_jobs_or_Git=True)
with(R/B/"independent_generation_gate.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=result["status"],complete_flows=len(flows),negative_cases=result["negative_case_count"],inactive_defaults=True)))
