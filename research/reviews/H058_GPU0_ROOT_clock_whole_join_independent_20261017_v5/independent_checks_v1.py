"""Independent v5 complete-source clock compatibility review, AST/RAM only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import ModuleType,SimpleNamespace,FunctionType
import ast,builtins,copy,hashlib,io,json,stat,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/H058_GPU0_full_training_successor_source_20261017_v5"
B="research/reviews/H058_GPU0_full_training_successor_source_20261017_v4"
OUT="research/reviews/H058_GPU0_ROOT_clock_whole_join_independent_20261017_v5"
NAMES=("__init__.py","common_v1.py","context_factory_v1.py","direct_gpu0_v1.py","observations_gpu_v1.py","producer_v1.py","root_driver_v1.py")
checks=[]
def ck(name,ok=True):
    if not ok:raise AssertionError(name)
    checks.append(dict(name=name,status="PASS"))
def reject(name,call,kind=Exception,message=None):
    try:call()
    except kind as e:
        if message is not None:assert message in str(e),(name,type(e).__name__,str(e))
        ck(name)
    else:raise AssertionError("Forbidden acceptance: "+name)
def data(p):
    p=ROOT/p
    assert p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=32*2**20,str(p)
    assert p.suffix not in (".pt",".npz",".npy",".trees",".vcf",".gz"),str(p)
    return p.read_bytes()
def text(p):return data(p).decode()
def j(p):return json.loads(data(p))
def ref(p):return dict(path=p,sha256=hashlib.sha256(data(p)).hexdigest())
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
def dump(t):return ast.dump(t,include_attributes=False)
def norm(t):return dump(t).replace("H058_GPU0_full_training_successor_source_20261017_v5","H058_GPU0_full_training_successor_source_20261017_v4").replace("H058_GPU0_v5_","H058_GPU0_v4_").replace("H058-GPU0-v5-","H058-GPU0-v4-")
def emit(nodes,ns,label):
    t=ast.Module(body=copy.deepcopy(nodes),type_ignores=[]);ast.fix_missing_locations(t)
    exec(compile(t,"<independent_RAM_"+label+">","exec"),ns)
def refs(x):
    if isinstance(x,dict):
        if set(x)=={"path","sha256"}:yield x
        else:
            for y in x.values():yield from refs(y)
    elif isinstance(x,list):
        for y in x:yield from refs(y)
def mutate(obj,key,value,call,name,kind=ValueError):
    old=copy.deepcopy(obj.get(key));present=key in obj;obj[key]=value
    try:reject(name,call,kind)
    finally:
        if present:obj[key]=old
        else:obj.pop(key,None)
expected={D+"/source_handoff.v5.json":"c74660a75003cb7ca81eb3752797bbe46361cbde2699105da4528a3ad5985c3c",D+"/source_manifest.v1.json":"67e4f59fc0dba5e4596cdd8e672fc4eef00f57b6d36345153b87a4c763bb001b",D+"/MANIFEST.v5.json":"21f2fc17edac6e58597bd67858a980e0cd9389fd0c15a211f0dff24ab19b314c",D+"/root_driver_v1.py":"762ed34189ac5605acfab8118f898f09b1bee5303edfa58f58cbe02b5ad61394","research/reviews/research_execution_ROOT_20261017_v23/H058_GPU0_v4_actual_outer_terminal_consumed.root_v1.json":"fcc274d414a33b8aa7c9d84d71aa46ea106cabeb41e268af4d1b61470ab53c1c"}
for p,s in expected.items():ck("assigned_seal:"+p,ref(p)["sha256"]==s)
agg=j(D+"/source_manifest.v1.json");oldagg=j(B+"/source_manifest.v1.json")
ck("aggregate320_distinct_exact_map",len(agg["refs"])==len({r["path"] for r in agg["refs"]})==320 and agg["source_sha256"]=={r["path"]:r["sha256"] for r in agg["refs"]})
for r in agg["refs"]:ck("aggregate:"+r["path"],ref(r["path"])==r)
for r in oldagg["refs"]:ck("preserved_v4_closure:"+r["path"],r in agg["refs"])
for version,base in (("v4",B),("v5",D)):
    for r in j(base+"/MANIFEST."+version+".json")["files"]:ck("sealed_artifact:"+r["path"],ref(r["path"])["sha256"]==r["sha256"] and len(data(r["path"]))==r["bytes"])
for p,s in j(D+"/sealed_v4_baseline.v5.json").items():ck("baseline_v4:"+p,ref(p)["sha256"]==s)
for name in ("source_handoff.v5.json","SOURCE_MAP.v5.json","dependencies.v1.json","producer_handoff.v1.json","producer_manifest.v1.json"):
    for r in refs(j(D+"/"+name)):ck("join:"+name+":"+r["path"],ref(r["path"])==r)
protocol=ast.parse(text("research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/protocol_v1.py"))
warmnode=next(n for n in protocol.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="WARM_CHECKPOINT_REF" for t in n.targets))
warmref=eval(compile(ast.Expression(warmnode.value),"<frozen_protocol_source_ref>","eval"),{"__builtins__":{},"dict":dict})
for p in sorted((ROOT/D).glob("*INACTIVE*.json")):
    for r in refs(j(str(p.relative_to(ROOT)))):
        if Path(r["path"]).suffix==".pt":ck("warmstart_ref_static_only_no_checkpoint_read",r==warmref)
        else:ck("template:"+p.name+":"+r["path"],ref(r["path"])==r)
trees={n:ast.parse(text(D+"/"+n)) for n in NAMES};old={n:ast.parse(text(B+"/"+n)) for n in NAMES}
for n in NAMES:
    compile(trees[n],"<parse_only_"+n+">","exec");ck("all7_compile:"+n)
    ck("all7_sealed_source:"+n,ref(D+"/"+n) in agg["refs"])
    if n not in ("common_v1.py","context_factory_v1.py"):ck("five_entire_AST_namespace_only:"+n,norm(trees[n])==dump(old[n]))
c=trees["common_v1.py"];context=trees["context_factory_v1.py"];driver=trees["root_driver_v1.py"]
# Restore exactly two added request calls and appended authenticated compatibility section.
original_len=len(old["common_v1.py"].body)
prefix=copy.deepcopy(c);appended=prefix.body[original_len:];prefix.body=prefix.body[:original_len]
request=fn(prefix,"validate_request_v1")
added=[n for n in request.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) in ("validate_prior_v4_terminal_v5","validate_root_fit_clock_v5")]
ck("exact_two_added_request_calls",[ast.unparse(n) for n in added]==["validate_prior_v4_terminal_v5(prior)","validate_root_fit_clock_v5(q, cfg)"])
for n in added:request.body.remove(n)
ck("complete_original_common_AST_preserved",norm(prefix)==dump(old["common_v1.py"]))
ck("new_section_only_expected_functions_assignments",all(isinstance(n,(ast.Assign,ast.FunctionDef)) for n in appended))
newfunctions=[n.name for n in appended if isinstance(n,ast.FunctionDef)]
ck("exact_new_function_inventory",newfunctions==["fit_clock_acceptance_v5","validate_root_fit_clock_v5","fit_source_authenticator_v5","root_fit_clock_view_v5","root_resources_view_v5","validate_prior_v4_terminal_v5"])
z=copy.deepcopy(context);build=fn(z,"build_context_v1")
# Revert exact root callback assignment and call, exact private resource assignment/call and CPU callback.
found=0
for i,n in enumerate(build.body):
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="check_fit_active" for t in n.targets):
        ck("exact_context_bound_clock_constructor",ast.unparse(n.value)=="root_fit_clock_view_v5(record, spec, q, cfg, fit)")
        ck("context_immediate_clock_call",ast.unparse(build.body[i+1])=="check_fit_active(spec)")
        build.body[i:i+2]=ast.parse("fit._check_active_v1(spec)").body;found+=1;break
ck("one_context_clock_join",found==1)
class RestoreContext(ast.NodeTransformer):
    def visit_If(self,node):
        self.generic_visit(node)
        index=next((i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="checked_resources" for t in n.targets)),None)
        if index is not None:
            ck("exact_context_resource_constructor",ast.unparse(node.body[index].value)=="root_resources_view_v5(h036, fit, check_fit_active)")
            ck("exact_context_resource_closure",ast.unparse(node.body[index+1])=="def resources():\n    checked_resources(spec, qualified_lease)")
            node.body[index:index+2]=ast.parse("def resources():\n    h036.lightweight_resources_v1(spec,qualified_lease)").body
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=="check_fit_active":node.func=ast.Attribute(value=ast.Name(id="fit",ctx=ast.Load()),attr="_check_active_v1",ctx=ast.Load())
        return node
z=RestoreContext().visit(z);ast.fix_missing_locations(z)
ck("complete_original_context_AST_preserved",norm(z)==dump(old["context_factory_v1.py"]))
# All nonclock resource/lease/science/finite-budget paths survive through exact AST preservation.
ck("no_latency_source_in_production",all("guard_latency_minimal_reuse" not in text(D+"/"+n) for n in NAMES))
# Constants and complete new helpers isolated; all external I/O supplied by fixtures.
ns=dict(Path=Path,json=json,types=types)
emit([n for n in c.body if isinstance(n,ast.Assign)],ns,"source_constants")
emit([fn(c,n) for n in ("require","active","require_work_window_v2","fit_clock_acceptance_v5","validate_root_fit_clock_v5","fit_source_authenticator_v5","root_fit_clock_view_v5","root_resources_view_v5","validate_prior_v4_terminal_v5")],ns,"complete_new_helpers")
oq=j(ns["CLOCK_ROOT_REQUEST_V4"]["path"]);ospec=j(ns["CLOCK_ROOT_INPUT_V4"]["path"]);ocfg=j(oq["config_ref"]["path"]);odirective=j(ns["CLOCK_ROOT_DIRECTIVE_V4"]["path"])
outer=j(ns["PRIOR_V4_ACTUAL_OUTER_REF"]["path"]);wait=j(ns["PRIOR_V4_ACTUAL_WAIT_REF"]["path"]);log=j(outer["worker_log_ref"]["path"])
for key in ("CLOCK_ROOT_REQUEST_V4","CLOCK_ROOT_INPUT_V4","CLOCK_ROOT_DIRECTIVE_V4","FIT_ACTIVE_SOURCE_V5","H036_RESOURCE_SOURCE_V5","SOURCE_AUTHENTICATOR_V5","PRIOR_V4_ACTUAL_OUTER_REF","PRIOR_V4_ACTUAL_WAIT_REF"):ck("new_constant_sealed:"+key,ref(ns[key]["path"])==ns[key])
start=datetime.fromisoformat(oq["created_at_utc"]);end=datetime.fromisoformat(oq["expires_at_utc"])
clockstate={"now":datetime.fromisoformat(outer["utc"])}
class FixedDT(datetime):
    @classmethod
    def now(cls,tz=None):return clockstate["now"] if tz else clockstate["now"].replace(tzinfo=None)
ns.update(datetime=FixedDT,timezone=timezone,time=SimpleNamespace(time=lambda:clockstate["now"].timestamp()),ref=ref)
ck("actual_request_input_directive_exact_clock_strings",all(oq[k]==ospec[k]==odirective[k] for k in ("created_at_utc","expires_at_utc")) and ospec["root_stage_request_ref"]==ns["CLOCK_ROOT_REQUEST_V4"])
ck("actual_interval96369_801392_seconds",(end-start).total_seconds()==96369.801392 and start<clockstate["now"]<end)
fit_text=text(ns["FIT_ACTIVE_SOURCE_V5"]["path"]);fit_tree=ast.parse(fit_text)
legacy_ns=dict(datetime=FixedDT,timezone=timezone,_utc=datetime.fromisoformat)
emit([fn(fit_tree,"_check_active_v1")],legacy_ns,"actual_legacy_clock")
reject("actual_v4_admitted_input_reproduces_original_duration_failure",lambda:legacy_ns["_check_active_v1"](ospec),ValueError,"exceeds24 hours")
ck("actual_v4_worker1_outer1_drain_EOF_no_completion",outer["actual_exit_code"]==outer["actual_direct_worker_returncode"]==wait["returncode"]==1 and outer["actual_terminal_consumed"] is True and wait["all_owned_processes_and_groups_drained"] is True and wait["pipe_EOF_observed"] is True and wait["completed_ref"] is None and outer["actual_new_optimizer_updates"]==0)
ck("actual_v4_log_fails_initial_context_clock_before_producer","Fit admission expired or exceeds24 hours" in log["tail"] and "fit._check_active_v1(spec)" in log["tail"] and "producer.run_context_v1" not in log["tail"])
# RAM future request and authentic admission constructor, no old clock rewritten.
q=j(D+"/ROOT_request.INACTIVE.template.v1.json");q.update(status="active",execution_authorized=True,source_review_passed=True,fit_clock_compatibility_v5=ns["fit_clock_acceptance_v5"]())
qref=dict(path=q["preparation_directory"]+"/ROOT_request.json",sha256="a"*64)
cfg=copy.deepcopy(ocfg);cfg.update(producer_ref=ref(D+"/producer_v1.py"),source_manifest_ref=ref(D+"/producer_manifest.v1.json"),full_training_directive_ref=dict(path="RAM/new_directive.json",sha256="b"*64))
q["config_ref"]=dict(path="RAM/new_config.json",sha256="c"*64)
directive=copy.deepcopy(odirective);directive["GPU_entry_ref"]=cfg["producer_ref"]
manifest=j(D+"/producer_manifest.v1.json")
cn=dict(q=q,cfg=cfg,request_ref=qref,producer=cfg["producer_ref"],ref=ref,DRIVER=D+"/root_driver_v1.py",CONTEXT=D+"/context_factory_v1.py",closure=dict(path="RAM/closure.json",sha256="d"*64),runtime=dict(path="RAM/runtime.json",sha256="e"*64),storage_ref=dict(path="RAM/storage.json",sha256="f"*64),auxiliary=q["packet_directory"]+"/worker_aux",source_refs_v1=lambda:agg["refs"],manifest=manifest)
constructor=next(n for n in fn(driver,"admit_v1").body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="s" for t in n.targets))
emit([constructor],cn,"exact_actual_admit_s_dict_only")
s=cn["s"];record=dict(path=q["packet_directory"]+"/input.json",sha256="0"*64)
records={qref["path"]:q,q["config_ref"]["path"]:cfg,cfg["full_training_directive_ref"]["path"]:directive,record["path"]:s}
def read(r):
    if r["path"] in records:return records[r["path"]]
    assert ref(r["path"])==r
    return j(r["path"])
ns["read"]=read
ck("exact_actual_admit_constructor_preserves_all_original_clock_strings",all(s[k]==q[k]==ospec[k] for k in ("created_at_utc","expires_at_utc")))
ck("new_directive_same_original_clocks",all(directive[k]==s[k] for k in ("created_at_utc","expires_at_utc")))
ck("full_clock_validator_accepts_exact_actual_interval",ns["validate_root_fit_clock_v5"](q,cfg) is directive)
# Full existing source authenticator, actual function bytecode, entirely RAM filesystem.
sourcecache={str(ROOT/ns[k]["path"]):data(ns[k]["path"]) for k in ("FIT_ACTIVE_SOURCE_V5","H036_RESOURCE_SOURCE_V5","SOURCE_AUTHENTICATOR_V5")}
class RP:
    def __init__(self,p):self.p=str(p.p if isinstance(p,RP) else p)
    def __truediv__(self,p):return RP(self.p+"/"+p)
    def __str__(self):return self.p
    def __eq__(self,p):return self.p==str(p)
    def resolve(self):return self
    @property
    def parents(self):return [RP(str(p)) for p in Path(self.p).parents]
    def lstat(self):return st(self.p)
def st(p):return SimpleNamespace(st_dev=1,st_ino=1,st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=len(sourcecache[str(p)]),st_mtime_ns=1,st_ctime_ns=1)
fds={};counter=[100]
def RAMopen(p,flags):
    counter[0]+=1;fds[counter[0]]=str(p);return counter[0]
class RAMFile(io.BytesIO):
    def __init__(self,fd):super().__init__(sourcecache[fds[fd]]);self.fd=fd
    def fileno(self):return self.fd
ram_os=SimpleNamespace(O_RDONLY=0,O_NOFOLLOW=1,open=RAMopen,fdopen=lambda fd,mode:RAMFile(fd),fstat=lambda fd:st(fds[fd]))
def exact_module(path,name,names,globals_extra):
    m=ModuleType(name);m.__file__=str(ROOT/path)
    m.__dict__.update(globals_extra)
    whole=compile(data(path),str(ROOT/path),"exec",dont_inherit=True,optimize=0)
    for n in names:
        codes=[x for x in whole.co_consts if isinstance(x,types.CodeType) and x.co_name==n]
        assert len(codes)==1
        setattr(m,n,FunctionType(codes[0],m.__dict__,n))
    return m
auth=exact_module(ns["SOURCE_AUTHENTICATOR_V5"]["path"],"_RAM_exact_auth",["source_bytes","functions_from_source"],dict(ROOT=RP(ROOT),Path=RP,ast=ast,stat=stat,hashlib=hashlib,os=ram_os,need=ns["require"]))
ns["load"]=lambda r,name:auth
events=[];rss=[0];reserved=[0];leasebad=[False];logicalbad=[False]
def logical(lease,spec):
    events.append("logical")
    if logicalbad[0]:raise ValueError("RAM original logical guard")
fit=exact_module(ns["FIT_ACTIVE_SOURCE_V5"]["path"],"arggnn.census_tsinfer_fit_v1",["_check_active_v1"],dict(datetime=FixedDT,timezone=timezone,_utc=datetime.fromisoformat,require_logical_job_v1=logical))
ck("fit_has_no_torch_before_original_late_science_initialization",not hasattr(fit,"torch"))
def leasecheck(lease):
    events.append("lease")
    if leasebad[0]:raise ValueError("RAM original lease guard")
def importer(name,globals=None,locals=None,fromlist=(),level=0):
    assert name=="biological_candidate" and level==1 and fromlist==("check_fit_lease",)
    return SimpleNamespace(check_fit_lease=leasecheck)
h036=exact_module(ns["H036_RESOURCE_SOURCE_V5"]["path"],"arggnn.train_mutation_routed_whole_tree_gnn_v1",["lightweight_resources_v1"],dict(fit=fit,resource=SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda who:SimpleNamespace(ru_maxrss=rss[0])),__package__="arggnn",__builtins__=dict(vars(builtins),__import__=importer)))
savedfit=fit._check_active_v1;savedresource=h036.lightweight_resources_v1;savedglobals=dict(h036.__dict__)
checked=ns["root_fit_clock_view_v5"](record,s,q,cfg,fit)
checked(s);ck("complete_bound_callback_passes_real_constructor_input")
private=ns["root_resources_view_v5"](h036,fit,checked)
ck("private_bytecode_identity_exact_original",private.__code__ is savedresource.__code__)
ck("all_private_globals_identical_except_fit",private.__globals__ is not savedresource.__globals__ and all(private.__globals__[k] is v for k,v in savedglobals.items() if k!="fit"))
ck("no_global_original_module_or_function_patch",h036.lightweight_resources_v1 is savedresource and fit._check_active_v1 is savedfit and savedresource.__globals__["fit"] is fit)
ck("private_creation_does_not_require_early_fit_torch",not hasattr(fit,"torch"))
fit.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:reserved[0]))
private(s,{"RAM":"lease"})
ck("all_original_nonclock_lease_and_logical_checks_execute",events==["lease","logical"])
late=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:reserved[0]));fit.torch=late
ck("late_fit_torch_is_dynamic_original_module_delegation",private.__globals__["fit"].torch is late)
private(s,{})
reject("original_fit_still_rejects_actual_long_input",lambda:fit._check_active_v1(s),ValueError,"exceeds24 hours")
reject("original_h036_still_rejects_actual_long_input",lambda:savedresource(s,{}),ValueError,"exceeds24 hours")
leasebad[0]=True;reject("private_original_lease_guard_cannot_be_bypassed",lambda:private(s,{}),ValueError,"lease guard");leasebad[0]=False
logicalbad[0]=True;reject("private_original_logical_guard_cannot_be_bypassed",lambda:private(s,{}),ValueError,"logical guard");logicalbad[0]=False
rss[0]=s["max_cpu_rss_bytes"]//1024+1;reject("private_original_RSS_guard_cannot_be_bypassed",lambda:private(s,{}),ValueError,"RSS cap");rss[0]=0
reserved[0]=s["max_cuda_reserved_bytes"]+1;reject("private_original_CUDA_allocator_guard_cannot_be_bypassed",lambda:private(s,{}),ValueError,"CUDA reservation cap");reserved[0]=0
reject("equal_unbound_spec_copy_rejected",lambda:checked(copy.deepcopy(s)),ValueError)
for key,val in (("job_id","wrong"),("output_directory","wrong"),("created_at_utc",(start+timedelta(seconds=1)).isoformat()),("expires_at_utc",(end+timedelta(seconds=1)).isoformat()),("max_cpu_rss_bytes",1),("max_cuda_reserved_bytes",1),("main_source_ref",None),("root_context_module_ref",None),("producer_ref",None),("physical_gpu",1),("device","cpu")):
    mutate(s,key,val,lambda:checked(s),"frozen_exact_spec_rejects:"+key)
for key,val in (("created_at_utc",(start+timedelta(seconds=1)).isoformat()),("expires_at_utc",(end+timedelta(seconds=1)).isoformat()),("job_id",oq["job_id"]),("adapter_manifest_ref",None),("fit_clock_compatibility_v5",{})):
    mutate(q,key,val,lambda:ns["validate_root_fit_clock_v5"](q,cfg),"root_clock_rejects:"+key)
for key,val in (("expires_at_utc",(end+timedelta(seconds=1)).isoformat()),("status","inactive"),("full30pass_training_authorized",False),("extra_probe_updates_authorized",True),("scientific_protocol",{}),("GPU_entry_ref",None),("physical_gpu",1),("device","cpu")):
    mutate(directive,key,val,lambda:ns["validate_root_fit_clock_v5"](q,cfg),"new_directive_guard:"+key)
mutate(q["caps"],"max_seconds",86401,lambda:ns["validate_root_fit_clock_v5"](q,cfg),"no_invented_longer_scientific_budget")
mutate(cfg,"scientific_protocol",{},lambda:ns["validate_root_fit_clock_v5"](q,cfg),"unchanged_scientific_protocol_required")
# New bound callback freezes entire request and configuration too.
mutate(q,"arbitrary_mutation",1,lambda:checked(s),"bound_callback_rejects_any_q_mutation")
mutate(cfg,"arbitrary_mutation",1,lambda:checked(s),"bound_callback_rejects_any_cfg_mutation")
clockstate["now"]=end-timedelta(seconds=300)
reject("original_absolute_cutoff300_blocks_callback",lambda:checked(s),ValueError,"Scientific work stops")
clockstate["now"]=end
reject("original_absolute_expiry_blocks_callback",lambda:checked(s),ValueError,"Current actual ROOT authority")
clockstate["now"]=start-timedelta(seconds=1)
reject("future_creation_blocks_callback",lambda:checked(s),ValueError)
clockstate["now"]=datetime.fromisoformat(outer["utc"])
# Authenticator rejects exact source/hash/code/globals/module linkage drift without importing production.
fit._check_active_v1=lambda spec:None
reject("source_auth_rejects_modified_fit_callable",lambda:ns["root_fit_clock_view_v5"](record,s,q,cfg,fit),ValueError,"Original unmodified callable")
fit._check_active_v1=savedfit
h036.__dict__["fit"]=SimpleNamespace()
reject("resource_original_fit_linkage_required",lambda:ns["root_resources_view_v5"](h036,fit,checked),ValueError,"Exact unchanged original fit")
h036.__dict__["fit"]=fit
h036.lightweight_resources_v1=lambda *args:None
reject("source_auth_rejects_modified_H036_callable",lambda:ns["root_resources_view_v5"](h036,fit,checked),ValueError,"Original unmodified callable")
h036.lightweight_resources_v1=savedresource
origfitpath=fit.__file__;fit.__file__="RAM/wrong.py"
reject("source_auth_rejects_wrong_fit_module_path",lambda:ns["root_fit_clock_view_v5"](record,s,q,cfg,fit),ValueError,"module pathname")
fit.__file__=origfitpath
fitpath=str(ROOT/ns["FIT_ACTIVE_SOURCE_V5"]["path"]);origbytes=sourcecache[fitpath];sourcecache[fitpath]=origbytes+b"\n"
reject("source_auth_rejects_source_byte_SHA_drift",lambda:ns["root_fit_clock_view_v5"](record,s,q,cfg,fit),ValueError,"Exact stable source SHA")
sourcecache[fitpath]=origbytes
# Full actual v4 terminal gate; immutable actual failure stays failure.
prior=j(D+"/prior_CPU_disposition.INACTIVE.template.v1.json");prior["prior_GPU_v4_files_charges_clocks_and_failure_preserved"]=True
ns["validate_prior_v4_terminal_v5"](prior);ck("full_actual_consumed8959_gate_passes_only_genuine_lineage")
records[ns["PRIOR_V4_ACTUAL_WAIT_REF"]["path"]]=copy.deepcopy(wait);rw=records[ns["PRIOR_V4_ACTUAL_WAIT_REF"]["path"]]
records[ns["PRIOR_V4_ACTUAL_OUTER_REF"]["path"]]=copy.deepcopy(outer);ro=records[ns["PRIOR_V4_ACTUAL_OUTER_REF"]["path"]]
for key,val in (("returncode",0),("actual_direct_worker_wait",False),("all_owned_processes_and_groups_drained",False),("owned_identity_absence",False),("pipe_EOF_observed",False),("completed_ref",{}),("input_ref",None)):
    mutate(rw,key,val,lambda:ns["validate_prior_v4_terminal_v5"](prior),"prior_actual_wait_guard:"+key)
for key,val in (("actual_terminal_consumed",False),("actual_direct_worker_returncode",0),("successful_outer_terminal_claimed",True),("automatic_retry_authorized",True),("completed_ref",{}),("actual_wait_ref",None),("tool_session_id",0),("actual_exit_code",None)):
    mutate(ro,key,val,lambda:ns["validate_prior_v4_terminal_v5"](prior),"prior_actual_outer_guard:"+key)
# Actual admission call order requires new gates before guard selection/reservation.
def calls(t,name):return [n for n in ast.walk(t) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
request=fn(c,"validate_request_v1");admit=fn(driver,"admit_v1")
ck("prior_terminal_gate_before_science_and_clock_gate",calls(request,"validate_prior_v4_terminal_v5")[0].lineno<calls(request,"science_v1")[0].lineno<calls(request,"validate_root_fit_clock_v5")[0].lineno)
ck("admission_validates_all_request_gates_before_guard_and_reserve",calls(admit,"validate_request_v1")[0].lineno<calls(admit,"guard_v1")[0].lineno<calls(admit,"memo.reserve_v1")[0].lineno)
build=fn(context,"build_context_v1")
ck("context_source_clock_view_before_storage_validation",calls(build,"root_fit_clock_view_v5")[0].lineno<calls(build,"g.validate_storage_v9")[0].lineno)
ck("original_stage_work_cutoff_body_preserved",norm(fn(driver,"execute_v1"))==dump(fn(old["root_driver_v1.py"],"execute_v1")))
# Derive current complete direct role union from exact source expression only.
direct_constants=dict(ns)
emit([n for n in trees["direct_gpu0_v1.py"].body if isinstance(n,ast.Assign)],direct_constants,"direct_constants")
for key in ("source_review_ref","direct_attachment_authority_ref","native_resource_disposition_ref"):
    q[key]=dict(path="RAM/"+key+".json",sha256="9"*64)
rn=dict(ns,q=q,cfg=cfg,producer=cfg["producer_ref"],request_ref=qref,runtime=cn["runtime"],closure=cn["closure"],direct=SimpleNamespace(CURRENT_CAP=direct_constants["CURRENT_CAP"]))
for n in admit.body:
    group=[]
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="needed" for t in n.targets):group=[n]
    elif isinstance(n,ast.AugAssign) and ast.unparse(n.target)=="needed":group=[n]
    elif isinstance(n,ast.If):group=[x for x in n.body if isinstance(x,ast.AugAssign) and ast.unparse(x.target)=="needed"]
    if group:emit(group,rn,"exact_direct_role_expression")
# Fill only future active refs absent in inactive template, with disjoint RAM identities.

capturepath="research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json"
capture=j(capturepath);named=capture["all_actual_parent_bindings"];parent={r["path"]:r for r in named.values()}
selected=dict(parent);selected[ns["SELECTOR_REF"]["path"]]=ns["SELECTOR_REF"]
union=dict(selected)
for r in rn["needed"]:
    assert isinstance(r,dict)
    assert r["path"] not in union or union[r["path"]]==r
    union[r["path"]]=r
ck("complete_actual_source_role_union39_plus25_equals64",len(named)==45 and len(parent)==38 and len(selected)==39 and len(union)==64 and len(set(union)-set(selected))==25)
ck("compatibility_acceptance_is_embedded_in_already_direct_request",q.get("fit_clock_compatibility_v5") is not None and qref in rn["needed"] and not any(r["path"].startswith("RAM/fit_clock") for r in rn["needed"]))
# Every prior predicate/source guard remains unchanged except explicit authenticated clock hooks.
ck("live_native_controller_capacity_lease_entire_AST_preserved",norm(trees["direct_gpu0_v1.py"])==dump(old["direct_gpu0_v1.py"]))
ck("science_producer_and_observation_entire_AST_preserved",norm(trees["producer_v1.py"])==dump(old["producer_v1.py"]) and norm(trees["observations_gpu_v1.py"])==dump(old["observations_gpu_v1.py"]))
ck("unchanged_native_resource_disposition_validator",norm(fn(c,"validate_native_resource_disposition_v1"))==dump(fn(old["common_v1.py"],"validate_native_resource_disposition_v1")))
ck("unchanged_authenticated_admitted_input_validator",norm(fn(c,"validate_input_v1"))==dump(fn(old["common_v1.py"],"validate_input_v1")))
# Mutable RAM fixtures never altered any genuine old object bytes or assigned seals.
for p,h in expected.items():ck("postcheck_seal_unchanged:"+p,ref(p)["sha256"]==h)
for p,h in j(D+"/sealed_v4_baseline.v5.json").items():ck("postcheck_old_artifact_unchanged:"+p,ref(p)["sha256"]==h)
roles=dict(frozen_named=45,frozen_distinct=38,current_selected_distinct=39,additional_distinct=25,total=64,actual_live_state_evidence=False,direct_role_paths=sorted(union),capture_ref=ref(capturepath))
result=dict(schema="H058_GPU0_v5_independent_complete_source_clock_AST_RAM_checks_v1",status="PASS",count=len(checks),checks=checks,source_ref=ref(D+"/root_driver_v1.py"),source_manifest_ref=ref(D+"/source_manifest.v1.json"),source_handoff_ref=ref(D+"/source_handoff.v5.json"),production_refs=[ref(D+"/"+n) for n in NAMES],five_namespace_only_files=[n for n in NAMES if n not in ("common_v1.py","context_factory_v1.py")],semantic_change_files=["common_v1.py","context_factory_v1.py"],actual_ROOT_duration_seconds=(end-start).total_seconds(),original_created_at_utc=oq["created_at_utc"],original_expires_at_utc=oq["expires_at_utc"],actual_consumed_v4_outer_ref=ns["PRIOR_V4_ACTUAL_OUTER_REF"],role_arithmetic=roles,production_module_imports=False,real_guard_resource_operations=False,process_GPU_inspection=False,numerical_runtime_data_truth_checkpoint_payload_reads=False,admission_launch_Git=False,RAM_authenticator_source_reads_only=True)
with (ROOT/OUT/"checks.result.v1.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status="PASS",count=len(checks),result_ref=ref(OUT+"/checks.result.v1.json"),role_arithmetic={k:v for k,v in roles.items() if k!="direct_role_paths"},actual_ROOT_duration_seconds=(end-start).total_seconds()),sort_keys=True))

