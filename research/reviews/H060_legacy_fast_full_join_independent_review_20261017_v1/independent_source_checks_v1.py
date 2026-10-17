"""Independent stdlib source/metadata/RAM checks; no project/runtime imports or operations."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import ast,copy,difflib,hashlib,json,os,stat
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H060_legacy_fast_full_join_independent_review_20261017_v1"
BASE="research/reviews/H060_legacy_fast_full_training_source_20261017_v1"
OLD="research/reviews/H060_fast_observed_full_training_source_20261017_v1"
checks=[];negative=[]
def need(v,m):
 if not v:raise ValueError(m)
def rr(path):
 p=ROOT/path;s=p.lstat()
 need(p.resolve()==p and p.is_relative_to(ROOT)and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=32*2**20,"Compact canonical regular source/metadata")
 return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(path):return json.loads((ROOT/path).read_text())
def loadmeta(record):need(rr(record["path"])==record,"Exact metadata ref");return read(record["path"])
def tree(path):return ast.parse((ROOT/path).read_text(),filename=path)
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
def run(t,names,scope):
 body=[copy.deepcopy(fn(t,n))for n in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),"<independent-RAM-metadata-only>","exec"),scope)
def reject(call,label):
 try:call()
 except(ValueError,KeyError,TypeError,AttributeError):negative.append(label);return
 raise AssertionError("Accepted invalid RAM metadata: "+label)
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def P(label):return dict(path="RAM_ONLY/"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
m=read(BASE+"/source_manifest.v1.json");b=read(BASE+"/scientific_binding.v1.json");bind=read(BASE+"/source_bindings.v1.json");p=read(OLD+"/source_manifest.v1.json");sp=read(BASE+"/scientific_source_pins.v2.json");ps=read(OLD+"/scientific_source_pins.v2.json")
need(rr(BASE+"/source_manifest.v1.json")["sha256"]=="08e106a4306f97a4091cfcc4512b5cac824371cdaa7192fcac7c3bf9e6634271","Requested outer manifest")
need(rr(BASE+"/source_handoff.v1.json")["sha256"]=="389646d9f1f929867e2a4339c2a3bf81c60f521093e654ee863ac84cdd135815","Requested handoff")
need(rr(BASE+"/audit_manifest.v1.json")["sha256"]=="769068faf0148ebb9ad59b20203a458e00e8f04eafcf4dc9fc8756417ce2b4e0","Requested audit")
total=parsed=0
for path,sha in m["source_sha256"].items():
 need(rr(path)["sha256"]==sha,"Complete sealed source "+path)
 total+=(ROOT/path).stat().st_size
 if path.endswith(".py"):tree(path);parsed+=1
oldscience={r["path"]:r["sha256"]for r in ps["refs"]};newscience={r["path"]:r["sha256"]for r in sp["refs"]}
need(len(m["source_sha256"])==423 and len(p["source_sha256"])==367 and all(m["source_sha256"].get(k)==v for k,v in p["source_sha256"].items()),"All367 operational pins retained")
need(len(newscience)==83 and len(oldscience)==76 and all(newscience.get(k)==v for k,v in oldscience.items())and all(m["source_sha256"].get(k)==v for k,v in newscience.items()),"All76 scientific pins retained inside exact83/423 closures")
audit=read(BASE+"/audit_manifest.v1.json")
for path,sha in audit["source_sha256"].items():need(rr(path)["sha256"]==sha,"Every package audit ref")
checks.append("Complete423/83 closures and367/76 predecessors,34 package hashes, source ASTs authenticate")
rt=tree(BASE+"/root_driver_v1.py");ot=tree(OLD+"/root_driver_v1.py");st=tree(BASE+"/parent_startup_legacy_fast_v1.py");runner=tree(BASE+"/warmstart_train_v2.py");orunner=tree(OLD+"/warmstart_train_v2.py")
for name in [n.name for n in ot.body if isinstance(n,ast.FunctionDef)]:
 if name not in("fast_scientific_review","contract"):need(same(fn(rt,name),fn(ot,name)),"Unchanged operational "+name)
for name in[n.name for n in orunner.body if isinstance(n,ast.FunctionDef)]:
 if name!="_fast_preparation_refs_v1":need(same(fn(runner,name),fn(orunner,name)),"Unchanged runner "+name)
need((ROOT/BASE/"worker_entry_v1.py").read_bytes()==(ROOT/OLD/"worker_entry_v1.py").read_bytes(),"Common worker byte identical")
for name,original in(("root_driver_v1.py",OLD+"/root_driver_v1.py"),("warmstart_train_v2.py",OLD+"/warmstart_train_v2.py"),("parent_startup_legacy_fast_v1.py","research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1/parent_startup_fast_v1.py")):
 oldtext=(ROOT/original).read_text();newtext=(ROOT/BASE/name).read_text();recorded=(ROOT/BASE/(name+".review.diff")).read_text()
 need(any("".join(difflib.unified_diff(oldtext.splitlines(True),newtext.splitlines(True),fromfile=original,tofile=BASE+"/"+name,n=n))==recorded for n in range(3,5)),"Exact supplied diff "+name)
checks.append("Unchanged lifecycle/memo/common worker/full training/cache/model/optimizer/checkpoint code; exact disclosed diffs")
# Original contract predicates and exact admission code used only as source.
ct=tree(bind["consumers"]["H060"]["contract_ref"]["path"]);dt=tree(bind["consumers"]["H060"]["driver_ref"]["path"]);caps=read(BASE+"/caps.inactive.v1.json")
need(caps==read("research/reviews/H060_isolated_CPU_runtime_source_20261017_v1/caps.inactive.v1.json")==bind["consumers"]["H060"]["caps"],"All original caps exact")
need(read(BASE+"/scientific_config.INACTIVE.template.v1.json")==read(OLD+"/scientific_config.INACTIVE.template.v1.json"),"All scientific config exact")
checks.append("Original full600each/1200+accepted600/20parents/60pretruth config, all caps unchanged")
# Execute the startup body with fake array/provider/bundle/source/guard values only.
assigns={t.id:n.value for n in st.body if isinstance(n,ast.Assign)for t in n.targets if isinstance(t,ast.Name)}
paths=ast.literal_eval(assigns["REQUIRED_FAST_SOURCE_PATHS"]);producerpaths=ast.literal_eval(assigns["LEGACY_PRODUCER_SOURCE_PATHS"])
need(len(paths)==14 and len(set(paths))==14 and len(producerpaths)==8 and set(producerpaths)<=set(paths),"Complete14 preparation/eight producer source paths")
required=[dict(path=p,sha256=m["source_sha256"][p])for p in paths]
need(required==b["fast_preparation_source_refs"],"Binding preserves exact14 path/SHA order")
imports=lambda t:[(n.module,tuple(x.name for x in n.names))for n in t.body if isinstance(n,ast.ImportFrom)]
provider_module="research.reviews.H060_throughput_source_analysis_20261017_v1.observation_provider_candidate_v1"
need((provider_module,("SealedObservationProviderV1",))in imports(st)and(provider_module,("SealedObservationProviderV1",))in imports(runner),"Same actual candidate provider import")
need(any(n.module=="parent_startup_legacy_fast_v1"and n.level==1 for n in runner.body if isinstance(n,ast.ImportFrom)),"Runner imports local actual startup")
class Array:
 def __init__(self,dtype="int8",size=4,strides=(4,1)):self.dtype=dtype;self.size=size;self.strides=strides
class Provider:
 def __init__(self):
  self.dataset=SimpleNamespace(_state=None,_pid=None,direction="forward");self.cache={};self.used=0;self.seal_record=None;self.releases=0;self.seals=0
  self.plan=dict(parent_id="ram_parent",trees=3,sites=4,observation_ref=P("obs"),focal_registry_ref=P("focals"),config=dict(native_preparation_ref=P("native")))
 def release(self):self.dataset._state=None;self.dataset._pid=None;self.cache.clear();self.used=0;self.releases+=1
 def seal(self,resource_check):
  need(self.dataset._state is state and self.dataset._pid==42,"Actual fresh state handoff in RAM");self.seals+=1
  self.seal_record=dict(focal_identity_sha256="fresh_focal",fresh_actual_seal=True);self.release();return dict(self.seal_record)
class OtherProvider(Provider):pass
state={"genotype":Array(),"allele_count":Array(strides=(1,))}
events=[];loads=[];preps=[];corrupt=[False]
def bound(record):need(not corrupt[0]and record in required,"Source identity/set must remain exact")
def load_obs(*a):loads.append(a);return state
def prepare(value,**kw):need(value is state,"Same single loaded state");preps.append(kw);return SimpleNamespace(native_holds=3,original_sites=4,focal_identity_sha256="fresh_focal"),dict(fresh_observation=True)
scope=dict(require=need,hashlib=hashlib,json=json,os=SimpleNamespace(getpid=lambda:42),np=SimpleNamespace(ndarray=Array,int8="int8"),SealedObservationProviderV1=Provider,validate_plan_v1=lambda p,**kw:need(kw["length_bp"]==2000000,"Full2Mb"),bound=bound,REQUIRED_FAST_SOURCE_PATHS=paths,LEGACY_PRODUCER_SOURCE_PATHS=producerpaths,load_observation=load_obs,prepare_from_loaded_observation_v1=prepare,fast_global_identity_v1=lambda bundle:dict(fresh_hash=True),json_sha=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())
run(st,["_check_fast_sources_v1","seal_then_prepare_one_parent_v1"],scope)
cfg=dict(max_native_nodes=1,max_native_holds=1,max_episode_payload_bytes=1)
def call(provider=None):
 return scope["seal_then_prepare_one_parent_v1"](provider or Provider(),cfg,lambda e:events.append(e),fast_source_refs=required)
provider=Provider();legacy,bundle,global_seal=call(provider)
need(len(loads)==len(preps)==provider.seals==1 and provider.releases==2,"One state load/genuine seal/unchanged prepare/final release")
need(legacy["fresh_actual_seal"] is True and legacy["legacy_block_seal_producer_unchanged"] is False and global_seal["legacy_block_seal_producer_unchanged"] is False and len(legacy["legacy_block_seal_producer_source_refs"])==8 and global_seal["legacy_block_seal_producer_source_refs"]==legacy["legacy_block_seal_producer_source_refs"]and global_seal["legacy_block_seal_sha256"]==scope["json_sha"](legacy),"New genuine false-producer seals and source-bound exact digest")
reject(lambda:call(OtherProvider()),"Startup wrong provider exact type")
for key,value in(("seal_record",{}),("used",1),("cache",{"old":True})):
 bad=Provider();setattr(bad,key,value);reject(lambda:call(bad),"Nonfresh provider "+key)
for field,value in(("genotype",Array("uint16")),("genotype",Array(strides=(-4,1))),("allele_count",Array(strides=(-1,)))):
 oldvalue=state[field];state[field]=value;bad=Provider();reject(lambda:call(bad),"Reviewed domain "+field+repr(value.strides)+value.dtype);need(bad.seals==0 and bad.releases==1,"Domain failure before seal and genuine final release");state[field]=oldvalue
reject(lambda:scope["_check_fast_sources_v1"](required[:-1]),"Missing preparation source")
badrefs=copy.deepcopy(required);badrefs[-1]["sha256"]="0"*64;reject(lambda:scope["_check_fast_sources_v1"](badrefs),"Changed preparation SHA")
corrupt[0]=True;bad=Provider();reject(lambda:call(bad),"Source identity invalid before state load");corrupt[0]=False
checks.append("RAM startup positive path and negative type/freshness/stride/dtype/source cases; real false-producer new seals and one load/full prepare/release")
# Exact new scientific review and generic wrapper interfaces, using metadata only.
review=read(BASE+"/ROOT_scientific_review.INACTIVE.template.v1.json")
review.update(status="SOURCE_PASS",independent_review=True,independent=True,verdict="SOURCE_PASS",blockers=[],candidate_outer_manifest_ref=rr(BASE+"/source_manifest.v1.json"),source_ref=rr(BASE+"/root_driver_v1.py"),source_manifest_ref=rr(BASE+"/source_manifest.v1.json"),source_handoff_ref=rr(BASE+"/source_handoff.v1.json"),runner_source_ref=b["runner_source_ref"],protocol_ref=b["protocol_ref"])
record=P("whole_review")
def meta(r):return copy.deepcopy(review)if r==record else loadmeta(r)
c=SimpleNamespace(binding=lambda:b,read=meta,require=need)
env=dict(metadata_ref=rr,MANIFEST=BASE+"/source_manifest.v1.json")
run(rt,["fast_scientific_review"],env);env["fast_scientific_review"](c,record)
for key,bad in(("schema","OLD_REVIEW"),("status","UNREVIEWED"),("independent_review",False),("candidate_outer_manifest_ref",P("old")),("legacy_support_review_ref",P("other")),("candidate_source_refs",[])):
 prior=review[key];review[key]=bad;reject(lambda:env["fast_scientific_review"](c,record),"Scientific review "+key);review[key]=prior
generic=dict(read=meta,binding=lambda:b,require=need,ref=rr,DRIVER=BASE+"/root_driver_v1.py",MANIFEST=BASE+"/source_manifest.v1.json",HANDOFF=BASE+"/source_handoff.v1.json")
run(ct,["reviewed"],generic);generic["reviewed"](record)
prior=review["source_handoff_ref"];review["source_handoff_ref"]=P("stalehandoff");reject(lambda:generic["reviewed"](record),"Generic review stale handoff");review["source_handoff_ref"]=prior
checks.append("Both complete scientific/generic consumer interfaces accept exact future review and reject old/incomplete substitutions")
# Exact pure config rejects scientific reduction and operational-cap violations.
configscope=dict(binding=lambda:b,require=need,CAPS=caps);run(ct,["validate_config"],configscope)
config=read(BASE+"/scientific_config.INACTIVE.template.v1.json");config["execution_enabled"]=True;configscope["validate_config"](config)
for key,bad in(("additional_updates_per_arm",599),("epochs",29),("inherited_saved_logical_updates",599),("CPU_observation_cache_bytes",1),("max_cached_parent_bytes",caps["max_cached_parent_bytes"]+1),("max_native_nodes",0),("precision","fp64")):
 before=config[key];config[key]=bad;reject(lambda:configscope["validate_config"](config),"Full config "+key);config[key]=before
checks.append("Original config rejects fewer updates/epochs, changed warmstart/precision, block cache and invalid/oversized caps")
# Derive the full role set independently from original contract and new append-only repair.
deps=read("research/reviews/H060_isolated_CPU_runtime_source_20261017_v1/dependencies.v1.json")
q={k:P(k)for k in("source_review_ref","scientific_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref")}
q["storage_policy_selector_ref"]=bind["actual_selector_metadata_ref"]
roleenv=dict(binding=lambda:b,ref=rr,MANIFEST=BASE+"/source_manifest.v1.json",SOURCE=BASE+"/root_driver_v1.py",DRIVER=BASE+"/root_driver_v1.py",ENTRY=BASE+"/worker_entry_v1.py",HANDOFF=BASE+"/source_handoff.v1.json",DEPS=deps,**{k:deps[k]for k in("NAMESPACE","PIDFD","OWNED","SUPERVISOR")})
run(ct,["direct_roles"],roleenv);roles=roleenv["direct_roles"](P("request"),q,P("config"),P("closure"))
appendenv=dict(need=need,copy=copy);run(rt,["append_selected_source"],appendenv);roles=appendenv["append_selected_source"](roles,bind["selected_child_source_ref"],bind["selected_child_source_ref"])
guards=loadmeta(bind["actual_H060_failed_authority_metadata_ref"])["bindings"]
unique=lambda rows:list({(r["path"],r["sha256"]):r for r in rows}.values())
union=unique([*guards.values(),*roles])
need(len(guards)==46 and len(unique(guards.values()))==39 and len(union)==61 and all(r in union for r in guards.values()),"Full61role union retains every actual historical role")
guardtree=tree(bind["selected_child_source_ref"]["path"]);consumer=dict(require=need,copy=copy,_raw=lambda r:need(set(r)=={"path","sha256"}and len(r["sha256"])==64,"RAM exact role"))
run(guardtree,["consumer_bindings_v1"],consumer);need(consumer["consumer_bindings_v1"](list(guards.values()),roles)==union,"Actual unchanged consumer union")
reject(lambda:consumer["consumer_bindings_v1"](union+[P("extra"+str(i))for i in range(4)]),"Actual consumer65role overflow")
reject(lambda:appendenv["append_selected_source"](roles,P("oldchild"),bind["selected_child_source_ref"]),"Wrong current child append")
need(all(Path(r["path"]).suffix in(".py",".sh",".json")for r in union),"No direct Markdown roles")
checks.append("Independent61/64 full original current API union; all46named39distinct guards retained;65roles/wrong child reject")
# The exact three AST substitutions only; no guard/state/mutation operation executes.
memoenv=dict(need=need,ast=ast,copy=copy);run(rt,["_memo_admission_node"],memoenv);original_admit=fn(dt,"admit");derived=memoenv["_memo_admission_node"](original_admit,"H060")
text=ast.unparse(derived)
need(text.index("memo = _make_root_mutation_memo(g, q)")<text.index("cfgref = put")<text.index("g.budget.v4.scope_spec")<text.index("memo.reserve_v1")<text.index("packet.mkdir"),"Original ordering plus memo before staging and fail-before-reserve")
need(not any(isinstance(n,ast.With)for n in ast.walk(derived))and text.count("memo.read_state_v1")==1 and text.count("memo.reserve_v1")==1,"No added outer mutex, one exact state/reserve transaction")
bad=copy.deepcopy(original_admit);bad.body.append(ast.parse("with g.budget.v4.mutex(ROOT):\n pass").body[0]);reject(lambda:memoenv["_memo_admission_node"](bad,"H060"),"Unexpected additional mutex")
bad=copy.deepcopy(original_admit);bad.body=[n for n in bad.body if not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="g"for t in n.targets))];reject(lambda:memoenv["_memo_admission_node"](bad,"H060"),"Missing original guard seam")
checks.append("Three exact memo AST changes only; constructor precedes staging, all-role schema before reserve, no outer mutex; malformed seams reject")
result=dict(schema="H060_legacy_fast_independent_whole_source_checks_v1",independent=True,source_only=True,status="PASS_SOURCE_RAM_ONLY",checks=checks,negative_cases=negative,negative_count=len(negative),verified_operational_refs=423,verified_scientific_refs=83,retained_old_operational_refs=367,retained_old_scientific_refs=76,parsed_python_files=parsed,source_metadata_bytes=total,package_audit_files=len(audit["source_sha256"]),preparation_refs=14,producer_refs=8,role_union_count=61,role_ceiling=64,actual_current_guard_resources_state_absence_or_fit_observed=False,numerical_tests_repeated=False,prior_exact_component_review=b["legacy_support_review_ref"],project_runtime_guard_science_import_or_process_GPU_admission_journal_mutex_install_Git=False,consumer_compatible_review_metadata=review)
(ROOT/HERE/"independent_checks.result.v1.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status=result["status"],checks=len(checks),negative=len(negative),result_ref=rr(HERE+"/independent_checks.result.v1.json"))))
