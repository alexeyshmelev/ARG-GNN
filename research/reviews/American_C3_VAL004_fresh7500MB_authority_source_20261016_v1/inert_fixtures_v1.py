"""Source-only fake metadata/selector checks. No actual guard, payload or resource call."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,hashlib,importlib.util,json,sys
R=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_VAL004_fresh7500MB_authority_source_20261016_v1"
sp=importlib.util.spec_from_file_location("_VAL004_fresh7500_FAKE_contract",R/REL/"root_contract_v1.py")
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
checks=[]
def ok(name,value):
 if not value:raise AssertionError(name)
 checks.append(dict(name=name,passed=True))
def reject(name,fun):
 try:fun()
 except (ValueError,KeyError,TypeError):checks.append(dict(name=name,passed=True,rejected=True));return
 raise AssertionError(name)
def fake_ref(path):
 p=str(path)
 if p.startswith(str(R)+"/"):p=p[len(str(R))+1:]
 return dict(path=p,sha256=hashlib.sha256(p.encode()).hexdigest())
db={}
def fake_raw(record):
 if record["path"] not in db:raise KeyError(record["path"])
 expected,value=db[record["path"]]
 if record!=expected:raise ValueError("FAKE exact metadata digest mismatch")
 return json.dumps(value).encode()
def fake_read(record,cap=0):return json.loads(fake_raw(record))
def add(path,value,reference=None):
 rr=fake_ref(path) if reference is None else reference
 db[rr["path"]]=(rr,copy.deepcopy(value));return rr
c.ref=fake_ref;c.read=fake_read;c.raw=fake_raw
for choice in c.APPROVED_GUARDS:
 for rr in choice.values():add(rr["path"],{},rr)
add(c.SELECTOR_SOURCE_REF["path"],{},c.SELECTOR_SOURCE_REF)
review=add("SYNTHETIC_ONLY/review.json",{})
selector=fake_ref("SYNTHETIC_ONLY/selector.json")
policy=add("SYNTHETIC_ONLY/policy.json",dict(c.APPROVED_GUARDS[1],source_review_ref=review))
value=dict(c.APPROVED_GUARDS[1],active=True,root_authorizer="root_agent",selector_source_ref=c.SELECTOR_SOURCE_REF,
 repair_authority_ref=c.REPAIR_REF,source_review_ref=review,policy_ref=policy,
 accounting_mode="native7_H050_stopped_retention_v1",journal=".runtime/storage/native7_H050_stopped_retention_v1")
add(selector["path"],value)
ok("known exact child selector metadata passes",c.selector_value(selector)==value)
unknown=copy.deepcopy(value);unknown["source_ref"]["sha256"]="0"*64
rr=add("SYNTHETIC_ONLY/unknown_source.json",unknown)
reject("unknown source digest rejects before guard",lambda:c.selector_value(rr))
bad=copy.deepcopy(value);bad["policy_ref"]["sha256"]="f"*64
rr=add("SYNTHETIC_ONLY/unknown_policy_digest.json",bad)
reject("unknown policy digest rejects",lambda:c.selector_value(rr))
bad=copy.deepcopy(value);bad["selector_source_ref"]["sha256"]="a"*64
rr=add("SYNTHETIC_ONLY/unknown_selector_source.json",bad)
reject("unknown selector source digest rejects",lambda:c.selector_value(rr))
calls=[]
class FakeSelector:
 def load_selected_guard_v1(self,record):
  calls.append(record)
  return type("Guard",(),dict(MODE=value["accounting_mode"],JOURNAL=value["journal"]))()
c._SELECTOR_MODULE=FakeSelector()
c.bind_selector(selector)
ok("only explicit selector passed to fake guard",calls==[selector])
reject("lifetime selector swap rejects",lambda:c.bind_selector(fake_ref("SYNTHETIC_ONLY/swap.json")))
ok("swap did not call fake guard",calls==[selector])
c.CAPS=dict(wall_seconds=7200)
now=datetime(2026,10,17,0,0,tzinfo=timezone.utc)
add(c.CERT_REF["path"],dict(renewed_expires_at_utc="2026-10-22T21:58:57.468498+00:00"),c.CERT_REF)
authority=dict(schema="VAL004_fresh7500_explicit_ROOT_authority_v1",active=True,root_authorizer="root_agent",
 parent_id="american_c3_val_004",allowed_stages=list(c.STAGES),adapter_manifest_ref=fake_ref(c.MANIFEST),
 source_review_ref=review,storage_policy_selector_ref=selector,historical_expired_authority_utc=c.ORIGINAL_EXPIRY_UTC,
 old_v3_and_down8_future_routes_revoked=True,only_never_admitted_original_VAL004_stages=True,
 no_replay_resume_or_old_clock_reset=True,native_accounting_failstop_bytes=7500000000,
 native_scientific_output_cap_bytes=16000000000,native_scientific_wall_seconds=7200,maximum_CPU_producers=2,
 all30_C3_prediction_seals_before_VAL_truth=True,query_truth_access_authorized=False,GNN_optimization_authorized=False,
 TEST_access_authorized=False,created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=24)).isoformat(),
 frozen_infer_prerequisite_ref=fake_ref(c.REL+"/frozen_infer_prerequisite.v1.json"))
checkauth=lambda v:c.authority_metadata(v,review,selector,"native_paths",now=now)
ok("separate exact24h authority accepts",checkauth(authority)==authority)
for key,val in [("parent_id","american_c3_val_003"),("maximum_CPU_producers",3),("native_scientific_wall_seconds",2000),
 ("native_scientific_output_cap_bytes",7500000000),("native_accounting_failstop_bytes",8000000000),
 ("old_v3_and_down8_future_routes_revoked",False),("all30_C3_prediction_seals_before_VAL_truth",False)]:
 bad=copy.deepcopy(authority);bad[key]=val
 reject("authority rejects "+key,lambda bad=bad:checkauth(bad))
bad=copy.deepcopy(authority);bad["created_at_utc"]="2026-10-16T19:00:00+00:00";bad["expires_at_utc"]="2026-10-17T19:00:00+00:00"
reject("historical clock reset rejects",lambda:checkauth(bad))
bad=copy.deepcopy(authority);bad["expires_at_utc"]=c.ORIGINAL_EXPIRY_UTC
reject("old expiry replay rejects",lambda:checkauth(bad))
bad=copy.deepcopy(authority);bad["expires_at_utc"]=(now+timedelta(hours=2)).isoformat()
reject("shortened authority rejects",lambda:checkauth(bad))
# Exact original inferred prerequisite, represented only by synthetic metadata.
parent="american_c3_val_004"; previous,folder,out,key=c.predecessor_paths_v1(parent,"native_paths")
ok("native prerequisite points at original v3 infer",previous=="infer" and folder.endswith("admitted_val004_infer_v3") and out.endswith("infer_native14v3_v3"))
cp=fake_ref(out+"/completed.json");wr=fake_ref(folder+"/worker_aux/actual_wait.json")
outer=fake_ref(folder+"/outer_terminal.root_actual_v1.json");close=fake_ref(folder+"/closed/close_receipt.json")
obs=fake_ref(out+"/observed/manifest.json");inp=fake_ref(folder+"/input.json")
sim=fake_ref("SYNTHETIC_ONLY/simulation_observed.json")
row=dict(simulation_observation_ref=sim)
c.existing_simulation=lambda p: dict(simulation_observation_ref=sim) if p==parent else (_ for _ in ()).throw(ValueError("wrong parent"))
m=dict(parent_id=parent,split="val",sequence_length_bp=30000000,sample_unit="haploid",
 class_order=["EUR","EAS","AFR"],counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),
 query_ancestry_truth_in_inputs=False,genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
 sites_and_sample_identities_verified=True,source_observation_ref=sim)
done=dict(status="completed_current_C3_inference",observation_ref=obs)
wait=dict(completed_ref=cp,actual_direct_worker_wait=True,returncode=0,parent_id=parent,phase="infer_val_004",
 input_ref=inp,all_owned_processes_and_groups_drained=True,operational_failure=None)
term=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,completed_ref=cp,direct_wait_ref=wr)
closed=dict(key=key,value=dict(future_writes_and_relaunch_revoked=True,proof=dict(completed_ref=cp,direct_wait_ref=wr)))
for rr,v in [(cp,done),(wr,wait),(outer,term),(close,closed),(obs,m)]:add(rr["path"],v,rr)
frozen=dict(completion_ref=cp,wait_ref=wr,outer_ref=outer,close_ref=close,observation_ref=obs,closed_key=key)
add(c.REL+"/frozen_infer_prerequisite.v1.json",frozen)
ok("genuine same-parent infer prerequisite passes fake evidence",c.actual_prior(parent,"native_paths")==frozen)
for name,rr,badvalue in [("missing genuine completion",cp,None),("wrong-parent wait",wr,dict(wait,parent_id="american_c3_val_003")),
 ("nonzero direct wait",wr,dict(wait,returncode=1)),("missing drain",wr,dict(wait,all_owned_processes_and_groups_drained=False)),
 ("unknown outer",outer,dict(term,actual_exit_code=None)),("unclosed infer",close,dict(closed,key="wrong"))]:
 saved=db.pop(rr["path"])
 if badvalue is not None:add(rr["path"],badvalue,rr)
 reject(name,lambda:c.actual_prior(parent,"native_paths"))
 db[rr["path"]]=saved
p=c.predecessor_paths_v1(parent,"plan")
ok("plan requires only fresh native paths",p[1].endswith("admitted_val004_native_paths_fresh7500_v1") and
 p[2].endswith("full30mb_native_paths_fresh7500_v1") and p[3].endswith("-native_paths-fresh7500-v1"))
c.PARENT=parent
for suffix in ["native_paths","plan"]:
 c.STAGE=suffix
 keys=c.never_admitted_stage_keys_v1()
 ok("all three immutable/new keys distinct "+suffix,len(set(keys))==3)
 for key in keys:reject("already admitted key rejects "+key,lambda key=key:c.require_never_admitted_v1({key:{}}))
 c.require_never_admitted_v1({})
# Complete concrete parent/child cardinality model, never a real guard call.
real=json.loads((R/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/inert_fixtures.final.v1.json").read_text())
state=real["full_role_models"]["guard_state"]
# Earlier fixture manifest pin predates final sealed manifest. Use final concrete known pin.
for row in state:
 if row["path"]==c.APPROVED_GUARDS[1]["source_manifest_ref"]["path"]:row.update(c.APPROVED_GUARDS[1]["source_manifest_ref"])
basis=json.loads((R/REL/"scientific_basis.v1.json").read_text())
add(c.BASIS,basis)
c.STAGE="native_paths"
q=dict(caps_ref=fake_ref("SYNTHETIC_ONLY/caps.json"),source_review_ref=review,
 root_authority_ref=fake_ref("SYNTHETIC_ONLY/root_authority.json"),storage_policy_selector_ref=selector)
for stage in ["native_paths","plan"]:
 c.STAGE=stage
 direct=c.direct_roles(fake_ref("SYNTHETIC_ONLY/request_"+stage+".json"),q,
 fake_ref("SYNTHETIC_ONLY/config_"+stage+".json"),fake_ref("SYNTHETIC_ONLY/closure_"+stage+".json"),
 basis["native_runtime_ref" if stage=="native_paths" else "plan_runtime_ref"])
 rows={r["path"]:r for r in [*state,*direct]}
 c.raw=lambda r:b"FAKE_BOUND_SOURCE"
 union=c.unique(list(rows.values()))
 ok("complete returned38 plus own direct roles fit64 "+stage,len(state)==38 and len(union)<=64)
 extra=[fake_ref("SYNTHETIC_ONLY/extra_"+str(i)+".json") for i in range(65-len(union))]
 reject("65th role rejects "+stage,lambda:c.unique([*union,*extra]))
 globals().setdefault("unions",{})[stage]=dict(count=len(union),refs=union,parent_roles_preserved=all(r in union for r in state))

# Full request/input boundary with injected metadata, no actual guard factory.
c.raw=fake_raw;c.read=fake_read;c.PARENT=parent
realread=lambda name:json.loads((R/REL/name).read_text())
for name in ["readiness.v1.json","scientific_basis.v1.json","caps.template.inactive.v1.json",
 "native_paths.scientific_config.template.inactive.v1.json","plan.scientific_config.template.inactive.v1.json"]:
 add(REL+"/"+name,realread(name))
for key0 in ["native_paths_config_ref","plan_config_ref","native_runtime_ref","plan_runtime_ref"]:
 rr=basis[key0];add(rr["path"],json.loads((R/rr["path"]).read_text()),rr)
capsfile=realread("caps.template.inactive.v1.json")
rr=capsfile["plan_original_caps_ref"];add(rr["path"],json.loads((R/rr["path"]).read_text()),rr)
add(c.MANIFEST,dict(refs=[fake_ref("src/arggnn/american_c3_gnn_train_v1.py")]))
add(review["path"],dict(verdict="SOURCE_PASS",blockers=[],source_manifest_ref=fake_ref(c.MANIFEST)),review)
fake_source="SYNTHETIC_ONLY/authenticated_source.py";add(fake_source,{})
c.source_paths=lambda:(fake_source,)
c.active=lambda v,margin=0: c.require(v["created_at_utc"]==authority["created_at_utc"] and
 v["expires_at_utc"]==authority["expires_at_utc"],"FAKE injected active clock")
c.validate_root_authority=lambda record,rev,sel,stage:c.authority_metadata(c.read(record),rev,sel,stage,now=now)
authref=add("SYNTHETIC_ONLY/authority.json",authority)
# Fresh native completion is synthetic, while source algorithms are never called.
_,nfolder,nout,nkey=c.predecessor_paths_v1(parent,"plan")
ncp=fake_ref(nout+"/completed.json");nwr=fake_ref(nfolder+"/worker_aux/actual_wait.json")
nouter=fake_ref(nfolder+"/outer_terminal.root_actual_v1.json");nclose=fake_ref(nfolder+"/closed/close_receipt.json")
ndone=dict(status="completed_native_C3_path_preparation",observation_ref=obs,parent_id=parent,
 query_focal_count=10000,query_ancestry_truth_opened=False,native_split_seed=42,native_train_fraction=.9,
 class_order=["EUR","EAS","AFR"],focals_ref=fake_ref("SYNTHETIC_ONLY/label_blind_focals.json"))
nwait=dict(wait,completed_ref=ncp,phase="native_paths_val_004",input_ref=fake_ref(nfolder+"/input.json"))
nterm=dict(term,completed_ref=ncp,direct_wait_ref=nwr)
nclosed=dict(key=nkey,value=dict(future_writes_and_relaunch_revoked=True,proof=dict(completed_ref=ncp,direct_wait_ref=nwr)))
for rr,v in [(ncp,ndone),(nwr,nwait),(nouter,nterm),(nclose,nclosed)]:add(rr["path"],v,rr)
for stage in ["native_paths","plan"]:
 c.select(parent,stage)
 prior=c.actual_prior(parent,stage)
 cfg=c.scientific_config(parent,stage,prior,True)
 orig=c.read(basis["native_paths_config_ref" if stage=="native_paths" else "plan_config_ref"])
 if stage=="native_paths":
  ok("unchanged16GB and all native count ceilings preserved",cfg["output_cap_bytes"]==16000000000 and
    all(cfg[k]==orig[k] for k in orig if k not in ["execution_enabled","observation_ref","path_sampling_seed"])
    and cfg["path_sampling_seed"]==93005 and cfg["query_focal_count"]==10000)
 else:
  ok("plan full1600 and original tree/block limits preserved",cfg["expected_haplotypes"]==1600 and
    all(cfg[k]==orig[k] for k in orig if k not in ["execution_enabled","observation_ref","native_preparation_ref","focal_registry_ref"]))
 capsref=add("SYNTHETIC_ONLY/"+stage+"_caps.json",c.worker_caps(stage))
 request=dict(schema="American_C3_VAL004_fresh7500_request_v1",active=True,root_authorizer="root_agent",
  parent_id=parent,stage=stage,phase=c.PHASE,storage_policy_selector_ref=selector,root_authority_ref=authref,
  source_review_ref=review,created_at_utc=authority["created_at_utc"],expires_at_utc=authority["expires_at_utc"],
  execution_authorized=True,source_review_passed=True,adapter_manifest_ref=fake_ref(c.MANIFEST),
  guard_ref=c.GUARD_REF,policy_ref=c.POLICY_REF,maximum_CPU_producers=2,
  query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,
  all30_C3_prediction_seals_before_VAL_truth=True,reservation_only_downward_failstop_authorized=True,
  accounting_reservation_bytes=c.CAPS["reservation_bytes"],prior=prior,scientific_config=cfg,
  bindings=dict(parent_id=parent,plan_config=cfg) if stage=="plan" else dict(parent_id=parent,scientific_config=cfg),
  caps_ref=capsref,runtime_ref=c.RUNTIME_REF)
 requestref=add("SYNTHETIC_ONLY/"+stage+"_request.json",request)
 ok("full fake request boundary "+stage,c.validate_request(requestref)[0]==request)
 packet,output=c.phase_paths(c.PHASE)
 configref=add(str(packet/"config.json"),c.make_config(request,c.read(capsref)))
 sources={fake_source:fake_ref(fake_source)["sha256"]}
 closureref=add(str(packet/"source_and_input_closure.json"),dict(source_sha256=sources,
  root_stage_request_ref=requestref,bindings=request["bindings"],native_MLM_dependency=False))
 direct=c.direct_roles(requestref,request,configref,closureref,c.RUNTIME_REF)
 allroles=list({row["path"]:row for row in [*state,*direct]}.values())
 storage=dict(reservation=dict(reserved_bytes=c.CAPS["reservation_bytes"],consumer_bindings=allroles),
  reserved_new_bytes=c.CAPS["reservation_bytes"],completed_overlay_policy_ref=c.POLICY_REF,
  budget_components=dict(consumer_adapter=c.GUARD_REF),storage_policy_selector_ref=selector)
 storageref=add(str(packet/"storage.json"),storage)
 spec=dict(root_stage_request_ref=requestref,phase=c.PHASE,parent_id=parent,root_driver_ref=fake_ref(c.DRIVER),
  output_directory=output,storage=storage,created_at_utc=authority["created_at_utc"],expires_at_utc=authority["expires_at_utc"],
  runtime_ref=c.RUNTIME_REF,selected_policy_ref=c.POLICY_REF,storage_policy_selector_ref=selector,root_authority_ref=authref,
  GPU_authorized=False,refs=dict(storage=storageref,config=configref,runtime=c.RUNTIME_REF),source_sha256=sources,
  transitive_source_closure_ref=closureref)
 inputref=add(str(packet/"input.json"),spec)
 ok("full fake admitted input boundary "+stage,c.input_spec(inputref)[0]==spec)
 for name,changed in [("wrong driver",dict(spec,root_driver_ref=fake_ref("SYNTHETIC_ONLY/wrong_driver.py"))),
  ("selector swap",dict(spec,storage_policy_selector_ref=fake_ref("SYNTHETIC_ONLY/wrong_selector.json")))]:
  add(inputref["path"],changed,inputref);reject("input rejects "+name+" "+stage,lambda:c.input_spec(inputref))
 add(inputref["path"],spec,inputref)
 badstorage=copy.deepcopy(storage);badstorage["reservation"]["consumer_bindings"]=[r for r in allroles if r!=selector]
 add(storageref["path"],badstorage,storageref);bad=copy.deepcopy(spec);bad["storage"]=badstorage
 add(inputref["path"],bad,inputref)
 reject("missing selector direct role rejects "+stage,lambda:c.input_spec(inputref))
 add(storageref["path"],storage,storageref);add(inputref["path"],spec,inputref)
 bad=copy.deepcopy(request);bad["scientific_config"]["output_cap_bytes"]=1
 add(requestref["path"],bad,requestref)
 reject("changed scientific recipe rejects "+stage,lambda:c.validate_request(requestref))
 add(requestref["path"],request,requestref)
reject("VAL003 never enters new source",lambda:c.select("american_c3_val_003","native_paths"))

result=dict(schema="VAL004_fresh7500_source_only_fake_checks_v1",source_only=True,
 no_actual_guard_probe_measurement_accounting_admission_job_payload_or_Git=True,
 checks=checks,passed=len(checks),full_role_models=unions,
 actual_capacity_and_peak_fit_unproved=True,no_ancestry_accuracy_claim=True)
print(json.dumps(result,sort_keys=True))
