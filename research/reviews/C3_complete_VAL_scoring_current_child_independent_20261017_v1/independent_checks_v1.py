"""Independent complete C3 current-child contract review. Source/JSON + RAM IO only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace,ModuleType
import argparse,ast,builtins,copy,hashlib,importlib.util,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/C3_complete_VAL_scoring_current_child_source_20261017_v1"
B="research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v2"
OUT="research/reviews/C3_complete_VAL_scoring_current_child_independent_20261017_v1"
checks=[];ioevents=[];actualreads=set();RAM={};LATEST={};rawcache={}
def ck(name,ok=True):
    if not ok:raise AssertionError(name)
    checks.append(dict(name=name,status="PASS"))
def reject(name,action,part=None):
    try:action()
    except (ValueError,KeyError,TypeError,AssertionError) as e:
        if part is not None:assert part in str(e),(name,type(e).__name__,str(e))
        ck(name)
    else:raise AssertionError("Forbidden acceptance: "+name)
def data(path):
    p=ROOT/path
    assert p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=32*2**20,str(p)
    assert p.suffix not in (".pt",".npz",".npy",".trees",".vcf",".gz"),str(p)
    return p.read_bytes()
def text(p):return data(p).decode()
def js(p):return json.loads(data(p))
def actualref(p):
    p=str(p);p=str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else p
    return dict(path=p,sha256=hashlib.sha256(data(p)).hexdigest())
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
def dump(n):return ast.dump(n,include_attributes=False)
def norm(n):return dump(n).replace(D,B)
def emit(nodes,ns,label):
    z=ast.Module(body=copy.deepcopy(nodes),type_ignores=[]);ast.fix_missing_locations(z)
    exec(compile(z,"<independent_RAM_"+label+">","exec"),ns)
def imported(path,name):
    sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp)
    sys.modules[name]=m;sp.loader.exec_module(m);return m
expected={D+"/handoff.v1.json":"693a7a9418bcd23ead62244836ef5fc04c1bccddfa160a13d093e23f78f20971",D+"/sealed_manifest.v1.json":"add7ce2e7575dff9f5737c00d6a3d4f483e346eda269e24e181f94174de68aa8",D+"/source_manifest.v1.json":"69ea033ae2213a85945158ad0ecc31fef826beda45ac724c86a636a593b2339c",D+"/checks.result.v2.json":"2f13a7a9ca279e99d03855649d234810e87473868325b234559c7db8157d9817"}
for p,s in expected.items():ck("assigned_seal:"+p,actualref(p)["sha256"]==s)
manifest=js(D+"/source_manifest.v1.json");prior=js(B+"/source_manifest.v1.json")
ck("all115_predecessor_pins_preserved_in134",len(prior["source_sha256"])==115 and len(manifest["source_sha256"])==134 and all(manifest["source_sha256"].get(p)==s for p,s in prior["source_sha256"].items()))
for p,s in manifest["source_sha256"].items():ck("source_closure:"+p,actualref(p)["sha256"]==s)
for r in js(D+"/sealed_manifest.v1.json")["files"]:
    ck("sealed_artifact:"+r["path"],actualref(r["path"])["sha256"]==r["sha256"])
names=("__init__.py","acceptance_contract_v1.py","root_driver_v1.py","worker_entry_v1.py","prepare_evaluation_root_v1.py")
trees={n:ast.parse(text(D+"/"+n)) for n in names}
old={n:ast.parse(text(B+"/"+n)) for n in names if (ROOT/B/n).is_file()}
for n in names:compile(trees[n],"<parse_only_"+n+">","exec");ck("parse_compile:"+n)
for n in ("root_driver_v1.py","worker_entry_v1.py"):ck("entire_AST_namespace_only:"+n,norm(trees[n])==dump(old[n]))
ct=trees["acceptance_contract_v1.py"];ot=old["acceptance_contract_v1.py"]
unchanged=("plan_producer_lineage_v1","plan_ready_v1","training_ready_v1","actual_stage_v1","prediction_metadata_v1","matrix_metadata_v1","evidence_request_fields_v1","validate_input_v1","required_direct_refs_v1")
for n in unchanged:ck("unchanged_complete_function:"+n,norm(fn(ct,n))==dump(fn(ot,n)))
for n in ("matrix_v1","score_v1"):ck("unchanged_scientific_worker:"+n,norm(fn(trees["worker_entry_v1.py"],n))==dump(fn(old["worker_entry_v1.py"],n)))
# Import complete metadata contracts only. Original business predicates remain callable and unmodified.
c=imported(D+"/acceptance_contract_v1.py","_independent_C3_complete_contract")
builder=imported(D+"/prepare_evaluation_root_v1.py","_independent_C3_complete_builder")
helper=imported(c.PREP_HELPER_REF["path"],"_independent_C3_original_preparation_helper")
original_ref,original_read,original_local=c.ref,c.read,c.local
business_ids={n:getattr(c,n) for n in ("manifest_v1","select_storage_metadata_v1","training_ready_v1","plan_ready_v1","stage_source_v1","evidence_request_fields_v1","validate_request","validate_input_v1","required_direct_refs_v1","original_guard")}
ABSENT={c.legacy.TRAIN_OUTPUT+"/completed.json",c.legacy.TRAIN_OUTPUT+"/observation_prefetch_all_worker_drains.json"}
ramdirs=set()
class RamPath(type(Path())):
    def exists(self):ioevents.append(("RAM_exists",str(self)));return str(self) in ramdirs or str(self.relative_to(ROOT)) in LATEST
    def is_symlink(self):return False
    def mkdir(self,*a,**kw):assert str(self) not in ramdirs;ramdirs.add(str(self))
    def resolve(self,*a,**kw):return self
def local(p):
    a=Path(p);r=str(a.relative_to(ROOT)) if a.is_absolute() else str(a)
    if r in ABSENT or "_RAM_ONLY_" in r or "RAM_ONLY" in r:return RamPath(ROOT/r)
    return original_local(p)
def put(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    r=dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
    RAM[(path,r["sha256"])]=copy.deepcopy(value);LATEST[path]=r;rawcache[(path,r["sha256"])]=raw
    return r
def fixture(label,value):return put("RAM_ONLY/"+label+".json",value)
def ref(p):
    p=str(p);p=str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else p
    if p in LATEST:return copy.deepcopy(LATEST[p])
    assert Path(p).suffix in (".py",".json",".sh"),"No payload hashing: "+p
    if p not in rawcache:rawcache[p]=original_ref(p)
    return copy.deepcopy(rawcache[p])
CACHE={}
def read(r,cap=33554432):
    assert isinstance(r,dict) and set(r)=={"path","sha256"},"Exact reference"
    key=r["path"],r["sha256"]
    if key in RAM:return copy.deepcopy(RAM[key])
    assert Path(r["path"]).suffix==".json","No payload read: "+r["path"]
    assert ref(r["path"])==r
    if key not in CACHE:CACHE[key]=original_read(r,cap)
    actualreads.add(r["path"]);return copy.deepcopy(CACHE[key])
def save(directory,name,value):
    assert Path(name).name==name
    return put(str((Path(directory)/name).relative_to(ROOT)),value)
c.ref=ref;c.read=read;c.save=save;c.local=local
c.legacy.ref=ref;c.legacy.read=read;c.legacy.local=local
ck("whole_manifest_accepts",c.manifest_v1()==manifest)
ck("matrix_score_only",c.PHASES==builder.PHASES==("matrix","score") and c.DIRECTIVE==builder.DIRECTIVE)
c.select_storage_metadata_v1(c.ACTUAL_SELECTOR_REF)
ck("exact_Relate64_metadata_only_no_guard_import",c.SELECTED_MODE=="Relate_failed64GiB_retention_v1" and c._GUARD is None and c._SELECTOR_MODULE is None)
reject("wrong_selector_hash_no_fallback",lambda:c.select_storage_metadata_v1(dict(c.ACTUAL_SELECTOR_REF,sha256="0"*64)))
reject("historical_selector_no_fallback",lambda:c.select_storage_metadata_v1(c.LEGACY_MANIFEST_REF))
bundle,training=c.training_ready_v1(c.SAVED_ACCEPTANCE_REF)
ck("real_saved600_complete_training_metadata",training["complete_parent_updates"]==600 and training["focal_exposures"]==6000000)
tr=bundle["training_completion_ref"];candidates={x["epoch"]:x["checkpoint_ref"] for x in training["checkpoint_candidates"]}
inputs={**c.OLD_INPUT_REFS,**c.SCHEMA2_INPUT_REFS};plans={};pe={}
for phase,r in inputs.items():
    p=c.VAL[int(phase[-3:])];s=read(r);rq=read(s["root_stage_request_ref"])
    pe[p]=rq["plan_evidence"];plans[p]=c.plan_ready_v1(pe[p],p)
    ck("actual_closed_full_plan:"+p,plans[p]["focal_records"]==10000 and plans[p]["sequence_length_bp"]==30000000)
    c.stage_source_v1(dict(input_ref=r,root_driver_ref=s["root_driver_ref"]),phase);ck("actual_fixed_prediction_source_route:"+phase)
# Real completed VAL004 JSON supplies the schema; no product array/checkpoint is read.
realenv=read(ref("experiments/results/C3_full30Mb_eval_actual_plan_lineage_checkpoint_schema_predict_val_004_root01_oct17/completed.json"))
sixbase=read(realenv["result_ref"]);sealbase=read(sixbase["prediction_seal_refs"][0])
waitbase=read(ref("research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_checkpoint_schema_predict_val_004_root01_oct17/actual_wait.json"))
review=read(ref(D+"/ROOT_review.INACTIVE.template.v1.json"))
reject("inactive_review_does_not_authorize",lambda:c.integration_review_v1(ref(D+"/ROOT_review.INACTIVE.template.v1.json")))
review.update(independent=True,verdict="SOURCE_PASS")
reviewref=fixture("complete_source_review",review)
builder.builder_review_v1(review);c.integration_review_v1(reviewref);ck("whole_review_schema_real_builder_and_contract")
def opaque(label):return dict(path="RAM_ONLY/"+label+".npz",sha256=hashlib.sha256(label.encode()).hexdigest())
def stage(phase,rr,ir):
    spec=read(ir);env=copy.deepcopy(realenv);env.update(stage=phase,status=c.legacy.STATUSES[phase],training_completion_ref=tr,input_ref=ir,root_driver_ref=spec["root_driver_ref"],result_ref=rr)
    cr=fixture(phase+"_completion",env);w=copy.deepcopy(waitbase)
    w.update(driver_ref=spec["root_driver_ref"],input_ref=ir,completed_ref=cr,returncode=0,actual_OS_wait_status=0,actual_direct_launcher_wait=True,actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None)
    wr=fixture(phase+"_wait",w);outer=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,completed_ref=cr,actual_wait_ref=wr,input_ref=ir,driver_ref=spec["root_driver_ref"])
    return dict(completion_ref=cr,actual_wait_ref=wr,outer_terminal_ref=fixture(phase+"_outer",outer))
predictions={};allseals=[];pseals={}
for p in c.VAL:
    prs=[]
    for e in c.EPOCHS:
        seal=copy.deepcopy(sealbase);seal.update(parent_id=p,epoch=e,training_completion_ref=tr,checkpoint_ref=candidates[e],focal_registry_ref=plans[p]["focal_registry_ref"],prediction_ref=opaque(p+"_"+str(e)),private_truth_opened=False,probabilities_modified=False,records=10000,protocol_sha256=training["protocol_sha256"],purity_filter=False)
        sr=fixture(p+"_"+str(e)+"_seal",seal);prs.append(sr);allseals.append(sr)
    six=copy.deepcopy(sixbase);six.update(parent_id=p,training_completion_ref=tr,prediction_seal_refs=prs,private_truth_opened=False)
    predictions[p]=stage("predict_val_"+p[-3:],fixture(p+"_six",six),inputs["predict_val_"+p[-3:]]);pseals[p]=prs
simulations={}
for p in c.VAL:
    obs=read(plans[p]["observation_ref"]);simdir=Path(obs["source_observation_ref"]["path"]).parent.parent
    simulations[p]=ref(str(simdir/"completed.json"));sim=read(simulations[p]);assert sim["parent_id"]==p
def evidence(phase):
    v=dict(schema="C3_full30Mb_eval_actual_stage_evidence_v1",phase=phase,ROOT_artifact_acceptance_ref=c.SAVED_ACCEPTANCE_REF,closed_plan_evidence_by_parent=copy.deepcopy(pe))
    if phase=="matrix":v["VAL_predictions"]=copy.deepcopy(predictions)
    else:v.update(matrix=copy.deepcopy(matrix_stage),VAL_simulation_completion_refs=copy.deepcopy(simulations))
    return v
# Exercise complete original builder entry to its first forbidden guard-load IO boundary.
# No business function is replaced; only metadata IO, filesystem and interpreter identity are RAM fixtures.
class ResourceBoundary(Exception):pass
def load_boundary(r,label):
    if r==c.SELECTOR_SOURCE_REF:raise ResourceBoundary("First guard module load; no resource operation executed")
    raise AssertionError("Unexpected production load: "+str(r))
c.load=load_boundary
real_builder_load=builder.load
def builder_load(r,label):
    if r["path"]==D+"/acceptance_contract_v1.py":
        assert ref(r["path"])==r;return c
    if r==c.PREP_HELPER_REF:return helper
    return real_builder_load(r,label)
builder.load=builder_load;builder.reference=ref
def bound(r):
    key=r["path"],r["sha256"]
    if key in rawcache:return rawcache[key]
    assert ref(r["path"])==r
    return data(r["path"])
builder.bound=bound
class BPath(type(Path())):
    def resolve(self,*a,**kw):
        if str(self)==str(ROOT/".runtime/env/bin/python"):return self
        return super().resolve(*a,**kw)
builder.Path=BPath;builder.ROOT=BPath(ROOT);builder.sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1),executable=str(ROOT/".runtime/env/bin/python"))
expiry=(datetime.now(timezone.utc)+timedelta(hours=26)).isoformat()
def generated(phase,ev):
    er=fixture(phase+"_builder_evidence",ev)
    args=SimpleNamespace(root_directive=builder.DIRECTIVE,phase=phase,instance="REVIEW_RAM_ONLY_"+phase,selector_path=c.ACTUAL_SELECTOR_REF["path"],selector_sha256=c.ACTUAL_SELECTOR_REF["sha256"],builder_review_path=reviewref["path"],builder_review_sha256=reviewref["sha256"],main_review_path=c.LEGACY_MAIN_REVIEW_REF["path"],main_review_sha256=c.LEGACY_MAIN_REVIEW_REF["sha256"],evidence_path=er["path"],evidence_sha256=er["sha256"],expires_at_utc=expiry)
    try:builder.prepare_root_v1(args)
    except ResourceBoundary as e:
        tb=e.__traceback__;frame=None
        while tb:
            if tb.tb_frame.f_code.co_name=="prepare_root_v1":frame=tb.tb_frame
            tb=tb.tb_next
        assert frame is not None
        saved={k:copy.deepcopy(frame.f_locals[k]) for k in ("request","request_ref","spec","caps","config","config_ref","job","output","expiry_text")}
        ck("whole_original_builder_generated_request_spec_reaches_first_resource_boundary:"+phase)
        return saved
    raise AssertionError("Builder should stop before forbidden guard import")
generated_data={}
g=generated("matrix",evidence("matrix"));generated_data["matrix"]=g
ck("original_builder_real_schema_matrix_request",c.validate_request(g["request_ref"])[0]==g["request"])
# Use original pure metadata storage constructor; no guard state/reservation is called.
oldstorage=read(read(inputs["predict_val_004"])["refs"]["storage"])
def admitted(g):
    spec=copy.deepcopy(g["spec"]);req=g["request"];required=c.required_direct_refs_v1(spec,req)
    rv=copy.deepcopy(oldstorage["reservation"]);rv.update(output=str(ROOT/spec["output_directory"]),auxiliary_scopes=[str(ROOT/(spec["packet_directory"]+"/worker_aux")),str(ROOT/(spec["packet_directory"]+"/worker_tmp"))],consumer_bindings=required)
    fake_reserved=dict(value=rv,initial_ledger_pins=copy.deepcopy(oldstorage["initial_ledger_pins"]))
    constants=SimpleNamespace(MODE=c.SELECTED_MODE,budget=SimpleNamespace(SOURCE_REL=oldstorage["budget_components"]["implementation"]["path"],LAUNCHER_REL=oldstorage["budget_components"]["launcher"]["path"]))
    st=helper.storage_v1(c,constants,g["job"],g["output"],fake_reserved,g["expiry_text"]);st["storage_policy_selector_ref"]=c.SELECTOR_REF
    spec["refs"]["storage"]=fixture(g["request"]["phase"]+"_generated_storage",st)
    ir=put(spec["packet_directory"]+"/input.json",spec)
    ck("original_generated_schema_whole_input:"+g["request"]["phase"],c.validate_input_v1(ir)[0]==spec)
    an=next(n for n in fn(trees["prepare_evaluation_root_v1.py"],"prepare_root_v1").body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="admitted_ref" for t in n.targets))
    adm=dict(c=c,packet=c.local(spec["packet_directory"]),input_ref=ir,args=SimpleNamespace(phase=req["phase"]),reserve_ref=fixture(req["phase"]+"_reserved_RAM_ONLY",fake_reserved),storage=st,config_ref=g["config_ref"],caps_ref=req["caps_ref"],roles=required,evidence_ref=req["actual_stage_evidence_ref"],copy=copy)
    emit([an],adm,"actual_builder_admitted_metadata_statement")
    return ir,spec,len(required)
matrix_input,matrix_spec,matrix_roles=admitted(g)
# Original generated matrix product, original worker wrapper and original score matrix parser.
pn=dict(require=c.require,read=read,bound=None)
for p,wanted in (("src/arggnn/american_c3_gnn_train_v1.py",{"SCHEMA","CHECKPOINT_EPOCHS"}),("src/arggnn/american_c3_gnn_targets_v1.py",{"REGISTRY_REF","TRAIN","VAL"})):
    tree=ast.parse(text(p));emit([n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets)],pn,"actual_source_constants")
payloadbounds=[]
def RAMbound(r):
    assert isinstance(r,dict) and set(r)=={"path","sha256"}
    payloadbounds.append(copy.deepcopy(r))
    return b"RAM_ONLY_OPAQUE_NO_NUMERICAL_PAYLOAD"
pn["bound"]=RAMbound
predict_tree=ast.parse(text("src/arggnn/american_c3_gnn_predict_isolated_v2.py"))
emit([fn(predict_tree,"completed_training_v1"),fn(predict_tree,"seal_validation_matrix_v1")],pn,"original_matrix_business")
sn=dict(pn);selection_tree=ast.parse(text("src/arggnn/american_c3_gnn_selection_isolated_v2.py"))
emit([fn(selection_tree,"validation_matrix_v1"),fn(selection_tree,"score_validation_matrix_v1"),fn(selection_tree,"select_original_val_checkpoint_v1")],sn,"original_score_metadata_business")
class Writer:
    def __init__(self,phase):self.phase=phase
    def save(self,name,value,kind="json"):
        assert kind=="json","No array or checkpoint creation"
        return fixture(self.phase+"_original_"+name.removesuffix(".json"),value)
events=[]
def fast(event):events.append(event)
def workerimport(name,globals=None,locals=None,fromlist=(),level=0):
    if name=="arggnn.american_c3_gnn_predict_isolated_v2":return SimpleNamespace(seal_validation_matrix_v1=pn["seal_validation_matrix_v1"])
    if name=="arggnn.american_c3_gnn_selection_isolated_v2":return SimpleNamespace(**{k:sn[k] for k in fromlist})
    raise AssertionError("Forbidden numerical/project import: "+name)
wn=dict(c=c,Path=Path,__builtins__=dict(vars(builtins),__import__=workerimport))
emit([fn(trees["worker_entry_v1.py"],"matrix_v1"),fn(trees["worker_entry_v1.py"],"score_v1")],wn,"exact_original_worker_matrix_score")
matrix_ref=wn["matrix_v1"](g["request"],bundle,Writer("matrix"),fast)
m=read(matrix_ref)
ck("original_worker_and_original_generated_matrix_schema30",m["status"]=="sealed_C3_all30_original_VAL_predictions" and len(m["cells"])==30 and len(payloadbounds)==60)
ck("original_generated_matrix_passes_original_parser",len(sn["validation_matrix_v1"](matrix_ref)[2])==30)
c.matrix_metadata_v1(SimpleNamespace(**vars(c)),m,tr,candidates,plans);ck("original_generated_matrix_passes_successor_business_parser")
# Generate completion through the original execute completion AST; no numerical execute imports.
last=next(n for n in reversed(fn(trees["worker_entry_v1.py"],"execute").body) if isinstance(n,ast.Return))
wrap=ast.FunctionDef(name="generated_completion",args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[copy.deepcopy(last)],decorator_list=[])
en=dict(writer=Writer("envelope"),c=c,req=g["request"],record=matrix_input,result=matrix_ref,bundle=bundle);emit([wrap],en,"exact_original_completed_envelope_statement")
generated_completion=en["generated_completion"]()
matrix_stage=stage("matrix",matrix_ref,matrix_input);matrix_stage["completion_ref"]=generated_completion
# Rebind future RAM direct wait/outer consistently to the exact generated completion.
w=read(matrix_stage["actual_wait_ref"]);w["completed_ref"]=generated_completion;matrix_stage["actual_wait_ref"]=fixture("matrix_generated_wait",w)
o=read(matrix_stage["outer_terminal_ref"]);o.update(completed_ref=generated_completion,actual_wait_ref=matrix_stage["actual_wait_ref"]);matrix_stage["outer_terminal_ref"]=fixture("matrix_generated_outer",o)
gscore=generated("score",evidence("score"));generated_data["score"]=gscore
score_input,score_spec,score_roles=admitted(gscore)
ck("original_generated_schema_whole_score_request",c.validate_request(gscore["request_ref"])[0]==gscore["request"])
class ScienceBoundary(Exception):pass
def scorefast(event):
    events.append(event)
    if event=="external_original_VAL_score_cell":raise ScienceBoundary("Stop before numerical original evaluator/truth payload access")
try:wn["score_v1"](gscore["request"],bundle,Writer("score"),scorefast)
except ScienceBoundary:ck("whole_original_score_worker_reaches_post30_pre_numerical_boundary")
else:raise AssertionError("Scoring must not execute")

# Run complete driver main -> exact authenticated reused supervisor launch to its first guard-load IO boundary.
dn=dict(c=c,ROOT=ROOT,SOURCE=D+"/root_driver_v1.py",STATUSES=c.STATUSES,Path=Path,argparse=argparse,ast=ast,json=json,os=SimpleNamespace(getpgrp=lambda:700,getpid=lambda:700,getsid=lambda pid:700))
supnode=next(n for n in trees["root_driver_v1.py"].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SUPERVISOR_FUNCTIONS" for t in n.targets))
emit([supnode,fn(trees["root_driver_v1.py"],"bind_reviewed_supervisor_v1"),fn(trees["root_driver_v1.py"],"main")],dn,"complete_original_driver_main_and_supervisor_binder")
base_tree=ast.parse(text(c.BASE_DRIVER_REF["path"]))
selected=[fn(base_tree,n) for n in dn["SUPERVISOR_FUNCTIONS"]]
cattrs={n.attr for f in selected for n in ast.walk(f) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=="c"}
# GPU-only branches are ineligible for these two phases; launcher_ref_v1 belongs only to that unreachable branch.
ck("all_reachable_reused_supervisor_contract_attributes_present",all(hasattr(c,n) for n in cattrs-{"launcher_ref_v1"}))
for phase,ir in (("matrix",matrix_input),("score",score_input)):
    c._ACTIVE_SPEC=None
    try:dn["main"](["--launch","--input",ir["path"],"--expected-input-sha256",ir["sha256"]])
    except ResourceBoundary:ck("whole_generated_driver_main_reaches_first_resource_boundary:"+phase)
    else:raise AssertionError("Driver must stop before resources")
    c._ACTIVE_SPEC=None
# Full-contract negatives, with changed evidence and directive identities kept consistent.
def changed_evidence(phase,change):
    g=generated_data[phase];req=copy.deepcopy(g["request"]);ev=read(req["actual_stage_evidence_ref"]);change(ev)
    er=fixture("negative_"+phase+"_evidence",ev);dr=read(req["ROOT_directive_ref"]);dr["actual_stage_evidence_ref"]=er
    req["actual_stage_evidence_ref"]=er;req["ROOT_directive_ref"]=fixture("negative_"+phase+"_directive",dr)
    return c.validate_request(fixture("negative_"+phase+"_request",req))
def changed_request(phase,change):
    req=copy.deepcopy(generated_data[phase]["request"]);change(req)
    return c.validate_request(fixture("negative_"+phase+"_request_only",req))
def bad_matrix(change):
    v=copy.deepcopy(m);change(v)
    return changed_evidence("score",lambda ev:ev.update(matrix=stage("matrix",fixture("bad_generated_matrix",v),matrix_input)))
for phase in c.PHASES:
    reject("whole_"+phase+"_rejects_new_prediction_phase",lambda phase=phase:changed_request(phase,lambda r:r.update(phase="predict_val_004")),"New matrix/score admissions only")
    reject("whole_"+phase+"_rejects_TEST_capability",lambda phase=phase:changed_request(phase,lambda r:r.update(TEST_truth_capability={})))
    reject("whole_"+phase+"_rejects_old_selector",lambda phase=phase:changed_request(phase,lambda r:r.update(storage_policy_selector_ref=c.LEGACY_MANIFEST_REF)))
    reject("whole_"+phase+"_rejects_missing_closed_plan",lambda phase=phase:changed_evidence(phase,lambda ev:ev["closed_plan_evidence_by_parent"].pop(c.VAL[-1])))
    reject("whole_"+phase+"_rejects_changed_caps",lambda phase=phase:changed_request(phase,lambda r:r.update(caps_ref=fixture("badcap",dict(c.caps_v1(phase),reservation_cap_bytes=1)))))
    reject("whole_"+phase+"_rejects_wrong_source_manifest",lambda phase=phase:changed_request(phase,lambda r:r.update(source_manifest_ref=c.SCHEMA2_MANIFEST_REF)))
reject("whole_matrix_rejects24of30",lambda:changed_evidence("matrix",lambda ev:ev["VAL_predictions"].update({c.VAL[-1]:stage("predict_val_004",fixture("incomplete_six",dict(sixbase,parent_id=c.VAL[-1],prediction_seal_refs=[])),inputs["predict_val_004"])})))
reject("whole_matrix_rejects_missing_producer",lambda:changed_evidence("matrix",lambda ev:ev["VAL_predictions"].pop(c.VAL[-1])))
reject("whole_matrix_rejects_truth_capability_even_with30",lambda:changed_evidence("matrix",lambda ev:ev.update(VAL_truth_capability=simulations)))
for count in (24,29):reject("whole_score_rejects_%dof30"%count,lambda count=count:bad_matrix(lambda v:v.update(cells=v["cells"][:count])))
reject("whole_score_rejects_duplicate_cell",lambda:bad_matrix(lambda v:v["cells"].__setitem__(-1,copy.deepcopy(v["cells"][0]))))
reject("whole_score_rejects_wrong_checkpoint",lambda:bad_matrix(lambda v:v["cells"][0].update(checkpoint_ref=candidates[30])))
reject("whole_score_rejects_early_truth_flag",lambda:bad_matrix(lambda v:v.update(all_predictions_sealed_before_VAL_truth=False)))
reject("whole_score_rejects_foreign_parent_truth_metadata",lambda:changed_evidence("score",lambda ev:ev["VAL_simulation_completion_refs"].update({c.VAL[0]:simulations[c.VAL[1]]})))
reject("whole_score_rejects_missing_truth_metadata",lambda:changed_evidence("score",lambda ev:ev["VAL_simulation_completion_refs"].pop(c.VAL[0])))
def terminal_change(phase,kind,key,value):
    def change(ev):
        e=copy.deepcopy(ev["VAL_predictions"][c.VAL[0]] if phase=="matrix" else ev["matrix"])
        obj=read(e[kind]);obj[key]=value;e[kind]=fixture("bad_"+kind+key,obj)
        if kind=="actual_wait_ref":
            outer=read(e["outer_terminal_ref"]);outer["actual_wait_ref"]=e[kind];e["outer_terminal_ref"]=fixture("consistent_bad_outer",outer)
        if phase=="matrix":ev["VAL_predictions"][c.VAL[0]]=e
        else:ev["matrix"]=e
    return changed_evidence(phase,change)
for phase in c.PHASES:
    for kind,key,value in (("actual_wait_ref","returncode",False),("actual_wait_ref","returncode",1),("actual_wait_ref","all_owned_processes_and_groups_drained",False),("outer_terminal_ref","actual_exit_code",False),("outer_terminal_ref","actual_exit_code",1),("outer_terminal_ref","actual_terminal_consumed",False)):
        reject("whole_terminal_guard:"+phase+":"+kind+":"+key+":"+str(value),lambda phase=phase,kind=kind,key=key,value=value:terminal_change(phase,kind,key,value))
def changed_input(ir,change):
    spec=read(ir);change(spec);return c.validate_input_v1(fixture("bad_whole_input",spec))
def changed_storage(spec,change):
    st=read(spec["refs"]["storage"]);change(st);spec["refs"]["storage"]=fixture("badstorage",st)
for phase,ir in (("matrix",matrix_input),("score",score_input)):
    reject("whole_input_missing_required_role:"+phase,lambda ir=ir:changed_input(ir,lambda s:changed_storage(s,lambda st:st["reservation"]["consumer_bindings"].pop())))
    reject("whole_input65roles:"+phase,lambda ir=ir:changed_input(ir,lambda s:changed_storage(s,lambda st:st["reservation"].update(consumer_bindings=[dict(path="RAM_ONLY/role%d.json"%i,sha256="0"*64) for i in range(65)]))))
    reject("whole_input_old_source_map:"+phase,lambda ir=ir:changed_input(ir,lambda s:s.update(source_sha256=prior["source_sha256"])))
# Scope limits and source-selective supervisor remain unchanged; no current returned-role/state call was made.
ck("mandatory_roles_only_not_live_union",matrix_roles==score_roles==53)
ck("zero_guard_module_import_state",c._GUARD is None and c._SELECTOR_MODULE is None)
ck("no_business_predicate_replaced",all(getattr(c,n) is f for n,f in business_ids.items()))
ck("no_numerical_or_guard_module_import",not any(n=="torch" or n.startswith(("torch.","torch_geometric","numpy","arggnn.")) for n in sys.modules))
ck("full30_barrier_before_first_scientific_cell_event","before_external_original_VAL_scoring" in events and "external_original_VAL_score_cell" in events)
for p,s in expected.items():ck("seal_unchanged:"+p,actualref(p)["sha256"]==s)
for p,s in manifest["source_sha256"].items():ck("postreview_source_unchanged:"+p,actualref(p)["sha256"]==s)
result=dict(schema="C3_current_child_independent_complete_entry_AST_RAM_checks_v1",status="PASS",count=len(checks),checks=checks,source_ref=actualref(D+"/prepare_evaluation_root_v1.py"),source_manifest_ref=actualref(D+"/source_manifest.v1.json"),handoff_ref=actualref(D+"/handoff.v1.json"),sealed_manifest_ref=actualref(D+"/sealed_manifest.v1.json"),production_refs=[actualref(D+"/"+n) for n in names],actual_small_JSON_reads=sorted(actualreads),mandatory_direct_roles=dict(matrix=matrix_roles,score=score_roles),actual_returned_role_union_not_measured=True,no_business_function_stubs=True,full_original_builder_and_driver_to_forbidden_resource_boundary=True,original_matrix_output_schema_passes_original_and_successor_parsers=True,full_original_score_worker_to_pre_numerical_boundary=True,current_all30_or_scientific_success_claimed=False,real_guard_state_process_GPU_resource_admission_launch_Git_calls=False,payload_reads_or_hashes=False,numerical_or_Torch_PyG_imports=False,RAM_future_metadata_never_persisted=True)
with (ROOT/OUT/"checks.result.v1.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status="PASS",count=len(checks),checks_ref=actualref(OUT+"/checks.result.v1.json"),matrix_mandatory_roles=matrix_roles,score_mandatory_roles=score_roles),sort_keys=True))

