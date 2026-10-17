
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,collections,copy,hashlib,io,json,os,stat,types
R=Path("/workspace-SR003.nfs2/ARG-GNN")
D=R/"research/reviews/C3_prepared_feature_future_integration_source_20261017_v3"
O=R/"research/reviews/C3_prepared_feature_future_integration_independent_20261017_v3"
V2=R/"research/reviews/C3_prepared_feature_future_integration_source_20261017_v2"
V5=R/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v5"
positive=[];negative=[]
def require(ok,label):
 if not ok:raise ValueError(label)
def check(ok,label):
 if not ok:raise AssertionError(label)
 positive.append(label)
def reject(fn,label):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError,FileExistsError) as e:
  negative.append(dict(case=label,message=str(e)[:1000]));return
 raise AssertionError("Unexpected acceptance: "+label)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else R/p
 b=p.read_bytes()
 return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(b).hexdigest())
def meta(row):
 require(ref(row["path"])==row,"Exact immutable metadata "+row["path"])
 return json.loads((R/row["path"]).read_bytes())
def node(tree,name):
 return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def install(tree,names,env):
 for name in names:
  exec(compile(ast.Module(body=[copy.deepcopy(node(tree,name))],type_ignores=[]),"<exact_AST_RAM_"+name+">","exec"),env)
handoff_ref=dict(path=str((D/"source_handoff.v3.json").relative_to(R)),sha256="a3c6d30121c87067bb0baa9c070ab52b15d0996e22abdec2b723598e3271ad29")
manifest_ref=dict(path=str((D/"source_manifest.v1.json").relative_to(R)),sha256="f5485f673d9821d241ae16ff45269fe075b2a767ee4421f0fe9ecddb152ff03c")
h=meta(handoff_ref);m=meta(manifest_ref);pins=meta(ref(D/"pins.v1.json"))
refs=m["executed_source_refs"]+m["source_authentication_refs"]
check(len(refs)==489 and len({r["path"]for r in refs})==489,"Exactly 489 distinct sealed source/metadata refs")
total=0
for row in refs:
 p=R/row["path"]
 require(p.suffix in (".py",".json",".md",".patch"),"Source/metadata closure excludes payloads")
 b=p.read_bytes();require(len(b)<=32*2**20,"Bounded exact source/metadata")
 require(hashlib.sha256(b).hexdigest()==row["sha256"],"Exact closure SHA "+row["path"]);total+=len(b)
check(True,"Every source-closure SHA authenticated without imports")
audit=meta(h["audit_manifest_ref"])
for row in audit["refs"]:meta(row) if Path(row["path"]).suffix==".json"else require(ref(row["path"])==row,"Exact author audit source")
check(len(audit["refs"])==13,"All 13 separate author audit refs authenticated")
check(h["source_manifest_ref"]==manifest_ref and h["source_only"]is True and h["activation_or_admission_authorized"]is False,"Inactive handoff joins exact sealed manifest")
check(len(m["executed_source_refs"])==5 and pins["feature_component"]in m["executed_source_refs"],"Five genuine executed files including accepted preprocessing")
trees={n:ast.parse((D/n).read_text())for n in ("contract_v1.py","execution_v1.py","worker_v1.py")}
v2trees={n:ast.parse((V2/n).read_text().replace(str(V2.relative_to(R)),str(D.relative_to(R))).replace(str(V2.relative_to(R)).replace("/","."),str(D.relative_to(R)).replace("/",".")))for n in trees}
changed={};unchanged=[]
for name,t in trees.items():
 old={n.name:n for n in v2trees[name].body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 diff=[]
 for n in t.body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
   if n.name not in old or ast.dump(n,include_attributes=False)!=ast.dump(old[n.name],include_attributes=False):diff.append(n.name)
   else:unchanged.append(name+":"+n.name)
 changed[name]=diff
check(set(changed["contract_v1.py"])=={"prior_v5_failure_resolution_values_v3","new_stage_values_v1","registration_values_v5","validate_request_v1","record_budget_v1","per_parent_source_policy_v6","config_v1"},"Contract changes exactly declared v3 definitions")
check(set(changed["execution_v1.py"])=={"OwnedLifecycleV8","observe_groups_v8","_put_supervisor_record_v5"},"Execution changes exactly lifecycle and caller AUX repair")
check(changed["worker_v1.py"]==["run_matrix_v1"],"Worker delta only matrix provenance; cache and writer repairs retained from v2")
emb=ast.literal_eval(next(n.value for n in trees["execution_v1.py"].body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=="_DIRECT_SOURCE_V6"))
embhash=ast.literal_eval(next(n.value for n in trees["execution_v1.py"].body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=="_DIRECT_SOURCE_V6_SHA"))
expected=(V5/"direct_gpu0_v1.py").read_text().replace(str(V5.relative_to(R)),str(D.relative_to(R))).replace(str(V5.relative_to(R)).replace("/","."),str(D.relative_to(R)).replace("/","."))
check(emb==expected and hashlib.sha256(emb.encode()).hexdigest()==embhash,"Embedded direct attachment byte-exact after namespace substitution and self-hashed")
calls=lambda t,name:[ast.dump(n,include_attributes=False)for n in ast.walk(t)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id==name]
oldworker=ast.parse((V5/"worker_v1.py").read_text())
for name in ("AmericanC3WholeTreeGNNV1","seal_shared_val_prediction_v1","seal_validation_matrix_v1"):
 check(calls(trees["worker_v1.py"],name)==calls(oldworker,name),"Original model/predictor/barrier call AST "+name)
v5input=meta(pins["preserved_v5_VAL000_input"]);v5storage=meta(v5input["refs"]["storage"]);v5manifest=meta(pins["preserved_v5_manifest"])
bindings=v5storage["reservation"]["consumer_bindings"]
replaced=[r for r in bindings if r not in v5manifest["executed_source_refs"]]+m["executed_source_refs"]
check(all(r in bindings for r in v5manifest["executed_source_refs"]),"All five historical executed files really bound")
check(len({(r["path"],r["sha256"])for r in bindings})==len({(r["path"],r["sha256"])for r in replaced})==64,"Exact static five-to-five replacement retains 64 direct refs")
check(all(r in replaced for r in bindings if r not in v5manifest["executed_source_refs"]),"Every other historical direct ref preserved")
source=(D/"contract_v1.py").read_text()
check("list(state[3].values()),direct_refs_v1(q,request_ref,config_ref)"in source,"Actual preparation unions every current returned role")
gt=ast.parse((R/pins["pins"]["CENSUS_GUARD"]["path"]).read_text())
gs=ast.unparse(node(gt,"consumer_bindings_v1"))
check("len(result) <= 64"in gs and "_raw(record)"in gs,"Selected guard hashes all refs and retains original 64 limit")
selector=meta(pins["pins"]["CENSUS_SELECTOR"]);verification=meta(pins["pins"]["CENSUS_VERIFICATION"])
check(selector["source_ref"]==pins["pins"]["CENSUS_GUARD"]and selector["accounting_mode"]=="Census_observation_two_scope_retention_v1"and verification["actual_direct_guard_roles"]==47,"Exact explicit Census route and published 47-role receipt; no fresh observation")
component=meta(pins["feature_review"]);component_manifest=meta(pins["feature_manifest"])
ram={};key=lambda row:json.dumps(row,sort_keys=True)
def put(name,value):
 row=dict(path="research/reviews/C3_prepared_feature_v7_ROOT_preparation_RAMreview/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
 ram[key(row)]=copy.deepcopy(value);return row
def rread(row):return copy.deepcopy(ram[key(row)])
for name in ("feature_review","feature_manifest","prior_v5_stop"):ram[key(pins[name])]=meta(pins[name])
env=dict(require=require,pins_v1=lambda:copy.deepcopy(pins),read=rread)
install(trees["contract_v1.py"],["accepted_feature_component_values_v2"],env)
check(env["accepted_feature_component_values_v2"]()==(component,component_manifest),"Genuine candidate_ref and candidate_manifest_ref consumer joins accepted component")
for field,value in [("candidate_ref",{}),("candidate_manifest_ref",{}),("schema","wrong"),("independent",False),("verdict","PASS"),("blockers",["block"])]:
 k=key(pins["feature_review"]);before=copy.deepcopy(ram[k]);ram[k][field]=value
 reject(env["accepted_feature_component_values_v2"],"Component review rejects "+field);ram[k]=before
for alias in ("source_ref","component_ref"):
 k=key(pins["feature_review"]);before=copy.deepcopy(ram[k]);ram[k].pop("candidate_ref");ram[k][alias]=pins["feature_component"]
 reject(env["accepted_feature_component_values_v2"],"No legacy alias substitution for "+alias);ram[k]=before
k=key(pins["feature_manifest"]);before=copy.deepcopy(ram[k]);ram[k]["component_ref"]={}
reject(env["accepted_feature_component_values_v2"],"Component manifest rejects mismatched component");ram[k]=before
stop=ram[key(pins["prior_v5_stop"])];base=datetime.fromisoformat(stop["utc"]);request_time=base+timedelta(seconds=15)
class FixedDatetime:
 fromisoformat=staticmethod(datetime.fromisoformat)
 @staticmethod
 def now(tz):return request_time+timedelta(seconds=5)
V=tuple("american_c3_val_%03d"%i for i in range(5));E=(5,10,15,20,25,30)
OWN=dict(pending=1024,discoveries=2**28,groups=65536,adopted_waits=2**28,history_rows=16)
env.update(datetime=FixedDatetime,timezone=timezone,VAL=V,EPOCHS=E,OWNED_LIMITS_V8=OWN,PARENT="American_C3_short20_TRAIN_original5VAL",CLASSES=("EUR","EAS","AFR"),MANIFEST=manifest_ref["path"],DRIVER=str((D/"execution_v1.py").relative_to(R)),ref=ref)
install(trees["contract_v1.py"],["active","prior_v5_failure_resolution_values_v3","owned_drain_values_v2","new_stage_values_v1","_complete_plan_metadata_v6","validate_all_original_plan_joins_v6","per_parent_source_policy_v6"],env)
training=put("training",{});env["tref_v1"]=lambda:training
election=meta(meta(v5input["root_stage_request_ref"])["cell_election_ref"])
check(sum(len(v["retained_prediction_seal_refs"])for v in election["parents"].values())==27,"Original election retains all 27 cells")
check({p:v["pending_epochs"]for p,v in election["parents"].items()}=={p:([30]if p in V[:3]else[])for p in V},"Only original three epoch30 cells unfinished")
for parent,row in election["parents"].items():
 for sref in row["retained_prediction_seal_refs"]:
  seal=meta(sref);require(seal["parent_id"]==parent and seal["records"]==10000 and seal["private_truth_opened"]is False,"Immutable retained seal full support")
check(True,"All 27 immutable retained seal metadata authenticated without payloads")
def add_resolution(q,i):
 proof=dict(schema="ROOT_C3_v5_known_targets_poststop_observation_v1",root_authorizer="root_agent",actual_observation=True,stopped_failure_ref=pins["prior_v5_stop"],old_v5_input_ref=pins["preserved_v5_VAL000_input"],known_root_observations=[dict(pid=66065,expected_start=491497125,present=False),dict(pid=70270,expected_start=491547684,present=False),dict(pid=66030,expected_start=None,present=False)],known_positive_group_observations=[dict(pgid=66065,absent=True),dict(pgid=70270,absent=True)],only_listed_known_targets_observed=True,historical_descendant_drain_proven=False,storage_close_or_credit_authorized=False,utc=(base+timedelta(seconds=10)).isoformat())
 q["prior_v5_stop_ref"]=pins["prior_v5_stop"];q["prior_v5_known_targets_ref"]=put("proof"+str(i),proof)
 resolution=dict(schema="C3_prepared_feature_v7_ROOT_prior_v5_resolution_v1",root_authorizer="root_agent",status="active",created_at_utc=(base+timedelta(seconds=10)).isoformat(),expires_at_utc=q["expires_at_utc"],stopped_failure_ref=pins["prior_v5_stop"],known_targets_ref=q["prior_v5_known_targets_ref"],old_v5_input_ref=pins["preserved_v5_VAL000_input"],old_v5_source_manifest_ref=pins["preserved_v5_manifest"],old_v5_execution_ref=pins["preserved_v5_execution"],all_old_v5_future_execution_resume_retry_write_routes_revoked=True,old_v5_historical_descendant_drain="UNPROVEN",residual_historical_uncertainty_accepted_for_new_phase_only=True,does_not_certify_old_drain_or_cleanup_success=True,new_exact_phase_adoption_authorized=True,all_old_files_and_full_original_charge_preserved=True,storage_close_or_credit_authorized=False,old_authority_admission_or_output_reuse_authorized=False,state_or_output_migration_authorized=False,automatic_retry_authorized=False)
 for name in ("source_manifest_ref","source_review_ref","phase","job_id","output_directory","worker_auxiliary_scope","cell_election_ref"):resolution[name]=copy.deepcopy(q[name])
 q["prior_v5_resolution_ref"]=put("resolution"+str(i),resolution)
 return proof,resolution
def hist(n,rows=None):return dict(count=n,first=list(rows or []),last=list(rows or []),ordered_chain_sha256="a"*64,full_historical_roster_serialized=False)
def good_wait():
 row=dict(pgid=100,absent=True)
 return dict(schema="C3_remaining_cells_actual_wait_v2",owned_roster_policy_ref=ref(env["DRIVER"]),owned_roster_policy=OWN,ownership=dict(complete_owned_roster_known=True,owned_roster_truncated=False,current_owned_identities=[],limits=OWN,retired_exact_identities_confirmed_absent_or_replaced=True,discovered_identity_history=hist(1),retired_identity_history=hist(1),reused_numeric_PID_history=hist(0),group_count=1,retirement_reasons=dict(absent=1,replaced=0)),group_absence_summary=dict(all_owned_groups_observed_absent=True,scan_finished_within_deadline=True,errors=[],expected_group_count=1,observed_group_count=1,absent_group_count=1,history=hist(1,[row])),adopted_waits=dict(hist(0),nonzero_returncodes=0),owned_identity_absence=True,group_observations_summarized=True,group_observations=[row,row],returncode=0,actual_direct_worker_wait=True,actual_OS_wait_status=0,all_owned_processes_and_groups_drained=True,complete_owned_roster_known=True,operational_failure=None)
cases=[]
for i,parent in enumerate(V[:3]):
 out="RAM/output"+str(i);inp=put("input"+str(i),{})
 seal=put("seal"+str(i),dict(epoch=30,parent_id=parent,training_completion_ref=training,private_truth_opened=False,prediction_ref=dict(path=out+"/predictions.npz",sha256="f"*64)))
 seal["path"]=out+"/seal.json";ram[key(seal)]=dict(epoch=30,parent_id=parent,training_completion_ref=training,private_truth_opened=False,prediction_ref=dict(path=out+"/predictions.npz",sha256="f"*64))
 result=put("result"+str(i),dict(status="sealed_C3_original_remaining_cells",parent_id=parent,epochs=[30],training_completion_ref=training,cell_election_ref=dict(path="RAM/election",sha256="e"*64),private_truth_opened=False,prediction_seal_refs=[seal]))
 sr=put("review"+str(i),dict(independent=True,verdict="SOURCE_PASS",blockers=[],candidate_manifest_ref=manifest_ref))
 q=dict(source_manifest_ref=manifest_ref,source_review_ref=sr,phase="predict_val_"+parent[-3:],parent_id=parent,pending_epochs=[30],job_id="C3prepared-v7-RAM"+str(i),output_directory=out,worker_auxiliary_scope="research/reviews/C3_prepared_feature_GPU0_ROOT_v7_RAM"+str(i)+"/worker_aux",cell_election_ref=dict(path="RAM/election",sha256="e"*64),created_at_utc=request_time.isoformat(),expires_at_utc=(request_time+timedelta(hours=1)).isoformat())
 add_resolution(q,i);qr=put("request"+str(i),q);ram[key(inp)]=dict(root_stage_request_ref=qr,source_manifest_ref=manifest_ref,phase=q["phase"],output_directory=out)
 done=put("done"+str(i),dict(status="completed_C3_remaining_cells",parent_id=env["PARENT"],original_VAL_parent_id=parent,child_input_ref=inp,actual_execution_source_manifest_ref=manifest_ref,cell_election_ref=q["cell_election_ref"],epochs=[30],training_completion_ref=training,query_ancestry_truth_opened=False,result_ref=result))
 wr=put("wait"+str(i),dict(good_wait(),completed_ref=done,input_ref=inp,adapter_manifest_ref=manifest_ref))
 outer=put("outer"+str(i),dict(root_authorizer="root_agent",actual_exit_code=0,actual_terminal_consumed=True,actual_wait_ref=wr,actual_direct_worker_returncode=0,actual_tool_session_id=i+1))
 ev=dict(input_ref=inp,completion_ref=done,wait_ref=wr,outer_ref=outer)
 check(env["new_stage_values_v1"](ev,parent,q["cell_election_ref"],[30])=={30:seal},"Full exact new stage + real drain + historical resolution join "+parent)
 check(env["prior_v5_failure_resolution_values_v3"](q,historical=False)[0]==stop,"Fresh finite resolution checked with fixed RAM clock "+parent)
 cases.append(dict(parent=parent,evidence=ev,q=q,review=sr,request=qr))
for case in cases:
 ev=case["evidence"];q=case["q"];parent=case["parent"]
 muts=[("wrong manifest",ev["input_ref"],lambda v:v.update(source_manifest_ref=pins["preserved_v5_manifest"])),("wrong review",case["review"],lambda v:v.update(candidate_manifest_ref={})),("nonzero worker",ev["wait_ref"],lambda v:v.update(returncode=1,actual_OS_wait_status=256)),("unconsumed outer",ev["outer_ref"],lambda v:v.update(actual_terminal_consumed=False)),("wrong ownership source",ev["wait_ref"],lambda v:v.update(owned_roster_policy_ref={})),("truncated roster",ev["wait_ref"],lambda v:v["ownership"].update(owned_roster_truncated=True)),("nonzero adopted wait",ev["wait_ref"],lambda v:v["adopted_waits"].update(nonzero_returncodes=1)),("missing group proof",ev["wait_ref"],lambda v:v["group_absence_summary"].update(observed_group_count=0)),("resolution credit",q["prior_v5_resolution_ref"],lambda v:v.update(storage_close_or_credit_authorized=True)),("resolution phase",q["prior_v5_resolution_ref"],lambda v:v.update(phase="matrix")),("old drain invention",q["prior_v5_resolution_ref"],lambda v:v.update(old_v5_historical_descendant_drain="DRAINED")),("proof full historical drain",q["prior_v5_known_targets_ref"],lambda v:v.update(historical_descendant_drain_proven=True))]
 for label,row,mutate in muts:
  before=copy.deepcopy(ram[key(row)]);mutate(ram[key(row)])
  reject(lambda:env["new_stage_values_v1"](ev,parent,q["cell_election_ref"],[30]),label+" "+parent);ram[key(row)]=before
case=cases[0];ev=copy.deepcopy(case["evidence"]);ram[key(pins["preserved_v5_VAL000_input"])]=rread(ev["input_ref"]);ev["input_ref"]=pins["preserved_v5_VAL000_input"]
reject(lambda:env["new_stage_values_v1"](ev,case["parent"],case["q"]["cell_election_ref"],[30]),"Stopped v5 input cannot become fresh successful VAL000")
reject(lambda:env["new_stage_values_v1"](cases[1]["evidence"],V[1],cases[1]["q"]["cell_election_ref"],[5,30]),"Retained epoch cannot be replayed")
plans={};bounds={};be=dict(c=SimpleNamespace(require=require))
install(trees["worker_v1.py"],["feature_cache_bounds_v6"],be)
chosen={}
for parent,record in pins["original_plan_refs"].items():
 plan=meta(record);plans[parent]=plan;ram[key(record)]=plan;env["_complete_plan_metadata_v6"](plan,parent_id=parent,length_bp=30000000)
 if parent in V[:3]:bounds[parent]=be["feature_cache_bounds_v6"](plan)
 chosen[parent]={epoch:put(parent+str(epoch),dict(parent_id=parent,epoch=epoch,block_index_ref=record,observation_ref=plan["observation_ref"],focal_registry_ref=plan["focal_registry_ref"],records=10000,private_truth_opened=False))for epoch in E}
check(env["validate_all_original_plan_joins_v6"](chosen)==pins["original_plan_refs"],"All five complete original plans and 30 source-bound seals join")
bad=copy.deepcopy(chosen);del bad[V[4]][30]
reject(lambda:env["validate_all_original_plan_joins_v6"](bad),"29-cell matrix rejected")
for field in ("block_index_ref","observation_ref","focal_registry_ref"):
 r=chosen[V[2]][30];before=copy.deepcopy(ram[key(r)]);ram[key(r)][field]=pins["original_plan_refs"][V[1]]if field=="block_index_ref"else plans[V[1]][field]
 reject(lambda:env["validate_all_original_plan_joins_v6"](chosen),"Cross-parent "+field+" rejected");ram[key(r)]=before
check(all(v["cache_tensor_upper_bound_bytes"]<=64*2**30 and v["cache_tensor_cap_bytes"]==64*2**30 and v["layout_tensor_cap_bytes"]==32*2**20 and v["auxiliary_tensor_cap_bytes"]==8*2**20 and v["feature_temporary_allowance_bytes"]==64*2**20 and v["original_CPU_RSS_cap_bytes"]==128*2**30 and v["full_RSS_fit_measured"]is False for v in bounds.values()),"All three full-parent cache bounds preserve original caps; whole RSS unmeasured")
# Re-execute audited author stdlib-only RAM fixtures, deleting only final report/print expressions.
author={}
for filename in ("source_checks_v3.py","failure_lifecycle_checks_v3.py"):
 t=ast.parse((D/filename).read_text())
 class NoReports(ast.NodeTransformer):
  def visit_Expr(self,n):
   if isinstance(n.value,ast.Call)and ((isinstance(n.value.func,ast.Attribute)and n.value.func.attr=="write_text")or(isinstance(n.value.func,ast.Name)and n.value.func.id=="print")):return None
   return self.generic_visit(n)
 t=NoReports().visit(t);ast.fix_missing_locations(t);ae={"__name__":"__RAM_author_fixture__"}
 exec(compile(t,"<audited_author_RAM_"+filename+">","exec"),ae)
 author[filename]=ae["result"]
 if filename.startswith("failure"):life=ae
check(all(v["operational_module_imports"]==0 and v["GPU_operations"]==0 for v in author.values()),"Repeated author stdlib fixtures remained RAM-only; no sealed outputs rewritten")
# Independent extra lifecycle cases reuse only the author's RAM backend, not its conclusions.
b=life["Backend"]([dict(pid=100,start=10,group=100,session=100,state="R")]);le=life["lenv"];t=le["OwnedLifecycleV8"](b,100)
b.stats[100]["group"]=-1;t.cleanup_read_stat_v9(100);b.stats[100]["group"]=200;t.cleanup_read_stat_v9(100)
check(not t.complete and t.groups=={100,200}and 100 in t.known,"Invalid observation remains incomplete after later positive group; exact identity retained")
oldlimit=le["OWNED_LIMITS_V8"]["groups"];le["OWNED_LIMITS_V8"]["groups"]=2
b.stats[100]["group"]=300;actual=t.cleanup_read_stat_v9(100)
check(actual["pid"]==100 and not t.complete and t.groups=={100,200},"Group-cap latch preserves cleanup read of known PID/start")
reject(lambda:t.read_stat_v9(100),"Strict group-cap overflow remains fail-closed")
le["OWNED_LIMITS_V8"]["groups"]=oldlimit
for raw in (-2,0,False,"",None):
 bt=life["Backend"]([dict(pid=100,start=10,group=raw,session=100,state="Z")]);tt=le["OwnedLifecycleV8"](bt,100);before=len(life["killcalls"])
 le["observe_groups_v8"](tt,10.0)
 check(not tt.complete and len(life["killcalls"])==before and 100 in tt.known,"Independent invalid group no-syscall/identity latch "+repr(raw))
life["reset"]();P=life["P"];directory=P("RAM/packet");aux=directory/"worker_aux";life["mk"](str(aux),"dir")
budget=life["budget"];used=4096+4096+100+4096
budget(directory,454656-used,worker_auxiliary_scope=aux)
check(True,"Exact final allowed caller metadata byte boundary with charged AUX node")
reject(lambda:budget(directory,454656-used+1,worker_auxiliary_scope=aux),"First byte above unchanged caller ROOT cap")
reject(lambda:budget(directory,True,worker_auxiliary_scope=aux),"Boolean metadata allowance rejected")
check(life["result"]["unproven_historical_drain_remains_unproven"]is True,"Repeated exact caller failed wait remains separate from scientific drain/credit")
# Exact JSON serializer and nested real shared/predictor callers with opaque RAM NPZ delegation.
it=ast.parse((D/"interface_checks_v2.py").read_text())
fe=dict(io=io,types=types,os=os,json=json,hashlib=hashlib,require=require,fs={},opened=[],syncs=[],fdnext=[100],descriptors={})
install(it,["MemoryStream","RamPath","RamOS","ram_ref"],fe)
rp=fe["RamPath"];common=ast.parse((R/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py").read_text())
ce=dict(require=require,Path=rp,json=json,os=fe["RamOS"],ROOT=rp(""),ref=fe["ram_ref"])
install(common,["save"],ce)
fixture_input=dict(path="RAM/input",sha256="a"*64);boundaries=[]
class C:
 MANIFEST=manifest_ref["path"];require=staticmethod(require);local=staticmethod(rp);save=staticmethod(ce["save"]);ref=staticmethod(ref)
 current_amendment_ref_v1=staticmethod(lambda:fixture_input);pins_v1=staticmethod(lambda:copy.deepcopy(pins))
 @staticmethod
 def active_amendment_v1():boundaries.append("active")
 @staticmethod
 def charged_census_v1(*args,**kwargs):boundaries.append("census")
class G:
 def budget_check_v9(self,*args,**kwargs):boundaries.append("budget")
we=dict(c=C,Path=rp,json=json);install(trees["worker_v1.py"],["GuardedWriter"],we)
writer=we["GuardedWriter"](dict(output_directory="RAM/current"),{},G(),{})
training_fixture=dict(trained_parent_ids=["RAM/train"]*20,train_sequence_lengths_bp=[2000000]*20,protocol_sha256="p"*64)
training_ref=dict(path="RAM/training",sha256="t"*64);checkpoint=dict(path="RAM/checkpoint",sha256="c"*64)
class Dummy:
 def __init__(self):
  self.class_order=("EUR","EAS","AFR");self.hidden_dim=32;self.expected_haplotypes=1600;self.donor_cohorts=2;self.donor_channels=6;self.site_chunk_size=256;self.checkpoint_sites=True;self.site_mlp=[SimpleNamespace(in_features=15)];self.leaf_projection=[SimpleNamespace(in_features=37)];self.forward_gru=self.reverse_gru=SimpleNamespace(input_size=35);self.classifier=[SimpleNamespace(out_features=3)]
 def __call__(self,*a,**k):raise AssertionError("No model forward")
drains=[];delegated=[]
class Delegate:
 def save(self,name,value,kind=None):
  delegated.append((name,kind))
  if kind=="npz":return dict(path="RAM/opaque_predictions.npz",sha256="f"*64)
  return writer.save(name,value,kind=kind)
def noop(*a,**k):return None
se=dict(require=require,VAL=V,TEST=(),CLASSES=("EUR","EAS","AFR"),AmericanC3WholeTreeGNNV1=Dummy,DEVELOPMENT_BRANCH="RAM/disclosure",LENGTH=30000000,eligible_checkpoint_v1=lambda*a:(copy.deepcopy(training_fixture),dict(epoch=30,global_step=600)),validate_shared_plan_v1=noop,validate_shared_metadata_v1=noop,require_baseline_v1=noop,validate_plan_v1=noop,streamed_focal_predictions=lambda*a,**k:({"opaque_RAM_prediction":True},dict(training_metadata={"opaque":True})),registry=lambda*r:{},verify_prediction=noop,identity_sha=lambda*r:"i"*64,shutdown_dataloader=lambda l:(drains.append(l.dataset.direction)or dict(drained=True)))
install(ast.parse((R/"src/arggnn/american_c3_gnn_predict_isolated_v2.py").read_text()),["seal_prediction_v1"],se)
install(ast.parse((R/"src/arggnn/american_c3_gnn_shared30mb_isolated_v2.py").read_text()),["seal_shared_val_prediction_v1"],se)
plan=dict(parent_id=V[0],config={},focal_registry_ref={})
loaders=[SimpleNamespace(_iterator=None,dataset=SimpleNamespace(plan_ref={},plan=plan,direction=d))for d in("forward","reverse")]
rr=se["seal_shared_val_prediction_v1"](Dummy(),*loaders,parent_id=V[0],checkpoint_ref=checkpoint,training_completion_ref=training_ref,writer=Delegate(),device="RAM",runtime_guard=noop)
saved=json.loads(fe["fs"][rr["path"]])
check(delegated[-1][1]=="json"and delegated[-2][1]=="npz"and drains==["forward","reverse"],"Exact full shared wrapper/predictor passes explicit JSON to actual writer and drains both mocked loaders")
check(saved["CPU_preprocessing_component_ref"]==pins["feature_component"]and saved["actual_execution_source_manifest_ref"]==manifest_ref and saved["actual_ROOT_phase_input_ref"]==fixture_input and saved["bitwise_original_feature_equivalence_claimed"]is False,"Serialized JSON retains phase/source/adaptation and avoids false bitwise claim")
check(saved["training_2Mb_to_30Mb_length_transfer"]is True and saved["original_full30Mb_matched_training_claimed"]is False and saved["complete_original_5VAL_selection_required_before_TEST"]is True and "training_metadata"not in saved,"Original shared disclosure and metadata-only serializer contract preserved")
check(len(fe["opened"])==len(fe["syncs"])==1 and boundaries.count("census")==boundaries.count("budget")==2,"Exact common O_EXCL/NOFOLLOW/0600/fsync JSON path and guarded pre/post boundaries executed in RAM")
for label,fn in [("existing output",lambda:writer.save(delegated[-1][0],{},"json")),("flat path",lambda:writer.save("../x.json",{},"json")),("nonfinite JSON",lambda:writer.save("bad.json",{"bad":float("nan")},"json")),("8MiB bound",lambda:writer.save("large.json",{"x":"x"*(8*2**20)},"json")),("unknown kind",lambda:writer.save("bad.yaml",{},"yaml"))]:reject(fn,"JSON writer "+label)
# Execution routes and source-only limits remain explicit.
check("--supervise-once"in source and "g.apply_reserve_v1"in source and "automatic_execution=False"in source,"Preparation emits one-shot caller only after actual reservation")
check("record_budget_v1(directory,len(data)+8192,worker_auxiliary_scope=local(s[\"worker_auxiliary_scope\"]))"in(D/"execution_v1.py").read_text(),"Only actual caller passes exact validated charged AUX allowance")
check(component["component_only"]is True and component["whole_future_route_admissible"]is False and component["independent_evidence"]["tolerance_widened"]is False,"Component evidence stays separate; no widened tolerance or whole-route fit")
result=dict(schema="C3_prepared_feature_future_integration_independent_checks_v3",status="SOURCE_CHECKS_PASS",independent_positive=len(positive),independent_negative=len(negative),checks=positive,negative_cases=negative,source_manifest_ref=manifest_ref,source_handoff_ref=handoff_ref,sealed_source_ref_count=len(refs),sealed_source_metadata_bytes=total,author_RAM_fixture_results={k:dict(positive=v["positive"],negative=v["negative"],schema=v["schema"])for k,v in author.items()},author_interface_original_result_ref=h["interface_checks_ref"],exact_full_shared_JSON_chain=True,NPZ_payload_is_opaque_RAM_delegate_no_serialization=True,all_three_new_stages_use_real_full_drain_and_resolution_AST=True,source_delta=changed,unchanged_definition_count=len(unchanged),cache_bounds=bounds,static_direct_slots=64,published_guard_roles=47,fresh_union_or_runtime_probed=False,operational_imports=0,scientific_payload_reads=0,numerical_imports=0,model_calls=0,optimizer_calls=0,process_resource_guard_state_probes=0,actual_subprocesses=0,signals=0,admissions=0,accounting_mutations=0,GPU_operations=0,Git_actions=0,active_ROOT_resolution_or_proof_created=False)
(O/"independent_checks.result.v3.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status=result["status"],positive=len(positive),negative=len(negative),author_RAM=result["author_RAM_fixture_results"],source_count=len(refs),changed=changed,result_ref=ref(O/"independent_checks.result.v3.json")),sort_keys=True))
