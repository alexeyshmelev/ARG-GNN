"""Source/AST/RAM metadata regression only. No runtime/scientific/guard imports."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace,ModuleType
import ast,builtins,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/H058_GPU0_full_training_successor_source_20261017_v5"
V4=D.replace("_v5","_v4");OUT=ROOT/D
checks=[]
def check(name,ok=True):
    assert ok,name
    checks.append(dict(name=name,status="PASS"))
def rejects(name,action):
    try:action()
    except (ValueError,KeyError,TypeError,AttributeError):check(name)
    else:raise AssertionError("Expected fail-closed rejection: "+name)
def text(p):return (ROOT/p).read_text()
def readjson(p):return json.loads(text(p))
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def fn(tree,name):return copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name))
def dump(n):return ast.dump(n,include_attributes=False)
def normalize(n):return dump(n).replace("_20261017_v5","_20261017_v4").replace("H058_GPU0_v5_","H058_GPU0_v4_").replace("H058-GPU0-v5-","H058-GPU0-v4-")
# Reproduce all sealed-v4 source tests with only their output location redirected.
v4tests=ast.parse(text(V4+"/source_checks_v4.py"))
for n in v4tests.body:
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="OUT"for t in n.targets):n.value=ast.parse("ROOT/DV5",mode="eval").body
v4tests.body=v4tests.body[:-1]
class ResultPath(ast.NodeTransformer):
    def visit_Constant(self,n):
        if n.value=="checks.result.v1.json":return ast.copy_location(ast.Constant(value="reproduced_v4_checks.result.v1.json"),n)
        return n
v4tests=ResultPath().visit(v4tests);ast.fix_missing_locations(v4tests)
prior_ns=dict(DV5=D,__name__="_reproduced_sealed_v4_source_only")
exec(compile(v4tests,"<sealed_v4_tests_output_relocated_only>","exec"),prior_ns)
prior_count=len(prior_ns["rows"]);check("sealed_v4_tests_reproduced",prior_count==812)
# Entire seven-file production body preservation after explicit clock joins.
names=("__init__.py","common_v1.py","context_factory_v1.py","direct_gpu0_v1.py","observations_gpu_v1.py","producer_v1.py","root_driver_v1.py")
for name in names:
    a=ast.parse(text(D+"/"+name));b=ast.parse(text(V4+"/"+name))
    if name=="common_v1.py":
        # Every appended v5 helper/constant follows the complete original module.
        check("new_clock_helpers_appended_after_complete_original_common",len(a.body)>len(b.body))
        a.body=a.body[:len(b.body)]
        request=next(n for n in a.body if isinstance(n,ast.FunctionDef)and n.name=="validate_request_v1")
        request.body=[n for n in request.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id in("validate_root_fit_clock_v5","validate_prior_v4_terminal_v5"))]
    if name=="context_factory_v1.py":
        src=text(D+"/"+name)
        src=src.replace('reader=fit.ConsumerReaderV1();check_fit_active=root_fit_clock_view_v5(record,spec,q,cfg,fit)\n    check_fit_active(spec)','reader=fit.ConsumerReaderV1();fit._check_active_v1(spec)')
        src=src.replace('checked_resources=root_resources_view_v5(h036,fit,check_fit_active)\n        def resources():checked_resources(spec,qualified_lease)','def resources():h036.lightweight_resources_v1(spec,qualified_lease)')
        src=src.replace('check_fit_active(spec)','fit._check_active_v1(spec)');a=ast.parse(src)
    check("complete_original_production_AST_recovered:"+name,normalize(a)==dump(b))
common=ast.parse(text(D+"/common_v1.py"));ns=dict(Path=Path,json=json,types=types)
for n in common.body:
    if isinstance(n,ast.Assign):
        try:exec(compile(ast.Module(body=[n],type_ignores=[]),"<source_constants_only>","exec"),ns)
        except NameError:pass
source_auth=text(ns["SOURCE_AUTHENTICATOR_V5"]["path"])
check("existing_source_authenticator_checks_exact_code_globals_module_and_source",all(s in source_auth for s in("original.__code__==codes[0]","original.__globals__ is module.__dict__","fn.__module__==module.__name__","source_bytes(record)")))
for name in("require","active","require_work_window_v2","fit_clock_acceptance_v5","validate_root_fit_clock_v5","root_fit_clock_view_v5","root_resources_view_v5","validate_prior_v4_terminal_v5"):
    exec(compile(ast.Module(body=[fn(common,name)],type_ignores=[]),"<isolated_exact_v5_metadata_functions>","exec"),ns)
# The actual v4 ROOT request -> actual admitted input joins reproduce the failure.
original_q=readjson(ns["CLOCK_ROOT_REQUEST_V4"]["path"]);original_s=readjson(ns["CLOCK_ROOT_INPUT_V4"]["path"])
original_cfg=readjson(original_q["config_ref"]["path"]);original_directive=readjson(ns["CLOCK_ROOT_DIRECTIVE_V4"]["path"])
start=datetime.fromisoformat(original_q["created_at_utc"]);end=datetime.fromisoformat(original_q["expires_at_utc"])
now=start+timedelta(minutes=10)
class FixedDatetime(datetime):
    @classmethod
    def now(cls,tz=None):return now if tz else now.replace(tzinfo=None)
ns.update(datetime=FixedDatetime,timezone=timezone,time=SimpleNamespace(time=lambda:now.timestamp()),ref=ref)
check("actual_ROOT_request_exact_SHA5840e2b5",ref(ns["CLOCK_ROOT_REQUEST_V4"]["path"])==ns["CLOCK_ROOT_REQUEST_V4"])
check("actual_admitted_input_exact_SHA67c31e1a",ref(ns["CLOCK_ROOT_INPUT_V4"]["path"])==ns["CLOCK_ROOT_INPUT_V4"])
check("actual_input_was_derived_from_exact_ROOT_request",original_s["root_stage_request_ref"]==ns["CLOCK_ROOT_REQUEST_V4"]and all(original_s[k]==original_q[k]for k in("job_id","phase","output_directory","created_at_utc","expires_at_utc")))
check("actual_request_input_and_directive_genuine_clock_fields_identical",all(original_q[k]==original_s[k]==original_directive[k]for k in("created_at_utc","expires_at_utc")))
check("actual_new_ROOT_duration_exceeds_legacy24h",(end-start).total_seconds()>86400)
fit_ast=ast.parse(text(ns["FIT_ACTIVE_SOURCE_V5"]["path"]))
legacy_ns=dict(datetime=FixedDatetime,timezone=timezone,_utc=datetime.fromisoformat)
exec(compile(ast.Module(body=[fn(fit_ast,"_check_active_v1")],type_ignores=[]),"<exact_original_fit_clock_only>","exec"),legacy_ns)
rejects("actual_admitted_v4_spec_reproduces_original_duration_rejection",lambda:legacy_ns["_check_active_v1"](original_s))
check("actual_original_ROOT_clock_is_current_at_failure_regression_time",start<=now<end)
# Build a RAM v5 ROOT request and derive its input through the exact admit s=dict statement.
q=readjson(D+"/ROOT_request.INACTIVE.template.v1.json")
q.update(status="active",execution_authorized=True,source_review_passed=True,fit_clock_compatibility_v5=ns["fit_clock_acceptance_v5"]())
config=copy.deepcopy(original_cfg);config.update(producer_ref=ref(D+"/producer_v1.py"),source_manifest_ref=ref(D+"/producer_manifest.v1.json"),full_training_directive_ref=dict(path="RAM_ONLY_NOT_ACTUAL/v5_directive.json",sha256="a"*64))
q["config_ref"]=dict(path="RAM_ONLY_NOT_ACTUAL/v5_config.json",sha256="b"*64)
request_ref=dict(path="RAM_ONLY_NOT_ACTUAL/v5_ROOT_request.json",sha256="c"*64)
new_directive=copy.deepcopy(original_directive);new_directive["GPU_entry_ref"]=config["producer_ref"]
input_statement=next(n for n in fn(ast.parse(text(D+"/root_driver_v1.py")),"admit_v1").body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="s"for t in n.targets))
manifest=readjson(D+"/producer_manifest.v1.json");aggregate=readjson(D+"/source_manifest.v1.json")
construct=dict(q=q,request_ref=request_ref,producer=config["producer_ref"],ref=ref,DRIVER=D+"/root_driver_v1.py",CONTEXT=D+"/context_factory_v1.py",cfg=config,runtime=dict(path="RAM_ONLY_NOT_ACTUAL/runtime.json",sha256="d"*64),closure=dict(path="RAM_ONLY_NOT_ACTUAL/closure.json",sha256="e"*64),storage_ref=dict(path="RAM_ONLY_NOT_ACTUAL/storage.json",sha256="f"*64),source_refs_v1=lambda:aggregate["refs"],manifest=manifest,auxiliary=q["packet_directory"]+"/worker_aux")
exec(compile(ast.Module(body=[input_statement],type_ignores=[]),"<exact_admit_input_dict_only_no_guard_or_reserve>","exec"),construct)
s=construct["s"];record=dict(path=q["packet_directory"]+"/input.json",sha256="0"*64)
records={request_ref["path"]:q,q["config_ref"]["path"]:config,config["full_training_directive_ref"]["path"]:new_directive,record["path"]:s}
def read(r):
    if r["path"]in records:return records[r["path"]]
    assert ref(r["path"])==r
    return readjson(r["path"])
ns["read"]=read
check("actual_exact_admit_constructor_preserves_original_genuine_clocks",all(s[k]==q[k]==original_s[k]for k in("created_at_utc","expires_at_utc")))
check("explicit_new_ROOT_clock_view_accepts_exact_actual_interval",ns["validate_root_fit_clock_v5"](q,config)==new_directive)
# Only metadata-clock callback and resource bytecode run with RAM stubs.
events=[]
fit=ModuleType("_RAM_ONLY_NOT_ACTUAL_fit");fit._check_active_v1=legacy_ns["_check_active_v1"]
fit.require_logical_job_v1=lambda lease,spec:events.append("original_logical_lease_check")
fit.torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:0))
auth_calls=[]
class RAMAuth:
    def functions_from_source(self,module,r,names):
        auth_calls.append((r,names))
        return {name:(getattr(module,name),fn(fit_ast if r==ns["FIT_ACTIVE_SOURCE_V5"]else resource_ast,name))for name in names}
ns["fit_source_authenticator_v5"]=lambda:RAMAuth()
checked=ns["root_fit_clock_view_v5"](record,s,q,config,fit)
check("fit_original_active_callable_authenticated_at_view_creation",auth_calls[-1][0]==ns["FIT_ACTIVE_SOURCE_V5"])
checked(s);check("new_exact_admitted_input_passes_isolated_bound_clock")
rejects("equal_but_unbound_input_copy_rejected",lambda:checked(copy.deepcopy(s)))
for key,value in(("job_id","other"),("output_directory","other"),("created_at_utc",(start+timedelta(seconds=1)).isoformat()),("expires_at_utc",(end+timedelta(seconds=1)).isoformat()),("max_cuda_reserved_bytes",1)):
    old=s[key];s[key]=value
    rejects("bound_resource_clock_rejects_spec_mutation:"+key,lambda:checked(s));s[key]=old
for label,target,key,value in(("q_clock_extension",q,"expires_at_utc",(end+timedelta(seconds=1)).isoformat()),("q_creation_rewrite",q,"created_at_utc",(start+timedelta(seconds=1)).isoformat()),("legacy_job",q,"job_id",original_q["job_id"]),("directive_expiry_rewrite",new_directive,"expires_at_utc",(end+timedelta(seconds=1)).isoformat()),("extra_probe",new_directive,"extra_probe_updates_authorized",True),("scientific_budget_change",new_directive,"scientific_protocol",{}),("unaccepted_clock_view",q,"fit_clock_compatibility_v5",{})):
    old=target[key];target[key]=value
    rejects("authority_metadata_rejects:"+label,lambda:ns["validate_root_fit_clock_v5"](q,config));target[key]=old
old=q["caps"];q["caps"]=dict(old,max_seconds=old["max_seconds"]+1)
rejects("ROOT_authority_rejects_arbitrary_increased_work_cap",lambda:ns["validate_root_fit_clock_v5"](q,config));q["caps"]=old
now=end-timedelta(seconds=300)
rejects("same_original300second_work_cutoff_enforced",lambda:checked(s))
now=end;rejects("same_original_absolute_ROOT_expiry_enforced",lambda:checked(s));now=start+timedelta(minutes=10)
resource_ast=ast.parse(text(ns["H036_RESOURCE_SOURCE_V5"]["path"]))
h036=ModuleType("_RAM_ONLY_NOT_ACTUAL_h036")
def importer(name,globals=None,locals=None,fromlist=(),level=0):
    assert name=="biological_candidate"and level==1
    return SimpleNamespace(check_fit_lease=lambda lease:events.append("original_held_lease_check"))
resource_ns=dict(fit=fit,__builtins__=dict(vars(builtins),__import__=importer),__package__="RAM_ONLY",resource=SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda mode:SimpleNamespace(ru_maxrss=0)))
exec(compile(ast.Module(body=[fn(resource_ast,"lightweight_resources_v1")],type_ignores=[]),"<exact_original_resource_bytecode_RAM>","exec"),resource_ns)
h036.lightweight_resources_v1=resource_ns["lightweight_resources_v1"]
private=ns["root_resources_view_v5"](h036,fit,checked)
check("isolated_resources_original_bytecode_is_exact_identity",private.__code__ is h036.lightweight_resources_v1.__code__)
check("original_resource_globals_and_fit_module_not_mutated",private.__globals__ is not h036.lightweight_resources_v1.__globals__ and h036.lightweight_resources_v1.__globals__["fit"]is fit and fit._check_active_v1 is legacy_ns["_check_active_v1"])
check("isolated_private_view_changes_only_fit_global",all(private.__globals__[k]is v for k,v in resource_ns.items()if k!="fit"))
private(s,{"RAM":"lease"});check("private_resources_preserve_original_lease_and_logical_checks",events==["original_held_lease_check","original_logical_lease_check"])
late=SimpleNamespace(cuda=SimpleNamespace(max_memory_reserved=lambda:0));fit.torch=late
check("private_fit_view_preserves_late_original_science_initialization",private.__globals__["fit"].torch is late)
resource_ns["resource"].getrusage=lambda mode:SimpleNamespace(ru_maxrss=s["max_cpu_rss_bytes"]//1024+1)
rejects("unchanged_original_CPU_RSS_failstop",lambda:private(s,{}))
resource_ns["resource"].getrusage=lambda mode:SimpleNamespace(ru_maxrss=0)
fit.torch.cuda.max_memory_reserved=lambda:s["max_cuda_reserved_bytes"]+1
rejects("unchanged_original_CUDA_allocator_failstop",lambda:private(s,{}));fit.torch.cuda.max_memory_reserved=lambda:0
rejects("original_unmodified_resource_path_still_rejects_actual_long_input",lambda:h036.lightweight_resources_v1(s,{}))
rejects("original_legacy_clock_still_rejects_expired_old_fit",lambda:fit._check_active_v1(dict(created_at_utc="2020-01-01T00:00:00+00:00",expires_at_utc="2020-01-02T00:00:00+00:00")))
# Genuine ROOT consumed8959 metadata and all historical files stay immutable; adversarial changes are RAM only.
wait_ref=ns["PRIOR_V4_ACTUAL_WAIT_REF"];wait=read(wait_ref)
check("genuine_v4_worker1_drain_EOF_no_completion_metadata",wait["returncode"]==1 and wait["all_owned_processes_and_groups_drained"]is True and wait["pipe_EOF_observed"]is True and wait["completed_ref"]is None)
outer_ref=ns["PRIOR_V4_ACTUAL_OUTER_REF"]
outer=copy.deepcopy(read(outer_ref))
check("genuine_ROOT_consumed8959_outer1_record_exact",ref(outer_ref["path"])==outer_ref and outer["actual_exit_code"]==1 and outer["actual_terminal_consumed"]is True)
records[outer_ref["path"]]=outer
prior=dict(prior_GPU_v4_actual_wait_ref=wait_ref,prior_GPU_v4_actual_outer_terminal_ref=outer_ref,prior_GPU_v4_files_charges_clocks_and_failure_preserved=True)
ns["validate_prior_v4_terminal_v5"](prior);check("prior_gate_accepts_genuine_consumed8959_metadata")
for key,value in(("actual_terminal_consumed",False),("actual_direct_worker_returncode",0),("tool_session_id",1),("successful_outer_terminal_claimed",True),("automatic_retry_authorized",True)):
    old=outer[key];outer[key]=value
    rejects("prior_terminal_gate_rejects:"+key,lambda:ns["validate_prior_v4_terminal_v5"](prior));outer[key]=old
# Source closure, inactive templates, no new direct role and no live imports.
for r in aggregate["refs"]:check("exact_aggregate_source_metadata:"+r["path"],ref(r["path"])==r)
check("all_sealed_v4_artifacts_unchanged",all(ref(p)["sha256"]==sha for p,sha in readjson(D+"/sealed_v4_baseline.v5.json").items()))
for p in sorted((ROOT/D).glob("*.py")):ast.parse(p.read_text())
check("no_scientific_runtime_guard_module_import",not any(n=="torch"or n.startswith(("torch.","numpy","arggnn","composer"))for n in sys.modules))
result=dict(schema="H058_GPU0_v5_ROOT_clock_source_AST_RAM_checks_v1",status="PASS_SOURCE_AST_RAM_ONLY",count=len(checks)+prior_count,v5_checks=len(checks),sealed_v4_reproduced_checks=prior_count,checks=checks,actual_ROOT_duration_seconds=(end-start).total_seconds(),actual_ROOT_request_ref=ns["CLOCK_ROOT_REQUEST_V4"],actual_admitted_input_ref=ns["CLOCK_ROOT_INPUT_V4"],original_ROOT_clocks_unchanged=True,real_runtime_guard_process_GPU_payload_tensor_or_numerical_probe=False,source_manifest_ref=ref(D+"/source_manifest.v1.json"))
with (OUT/"checks.result.v5.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in result.items()if k!="checks"},sort_keys=True))
