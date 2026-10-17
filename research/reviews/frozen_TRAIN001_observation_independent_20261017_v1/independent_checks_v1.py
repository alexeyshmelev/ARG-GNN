"""Independent source/AST/RAM review. No live guard import, process probe or science."""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace, ModuleType
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager, redirect_stdout
import ast, copy, hashlib, io, json, os, stat, sys, contextlib
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE="research/reviews/frozen_TRAIN001_observe_ready_20261017_v1"
OUT="research/reviews/frozen_TRAIN001_observation_independent_20261017_v1"
SOURCE=BASE+"/worker_entry_v1.py"
MANIFEST=BASE+"/source_manifest.v1.json"
AUTHOR=BASE+"/author_RAM_checks_v1.py"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
CHILD="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py"
SELECTOR="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json"
OBS="src/arggnn/american_c3_observe_v3.py"
CHECKS=[]
READS=[]
def need(value,msg="Independent assertion"):
    if not value: raise AssertionError(msg)
def ref(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    need(p.is_relative_to(ROOT)and ".."not in p.parts)
    READS.append(str(p.relative_to(ROOT)))
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(path): return json.loads((ROOT/path).read_text())
def extract(path,name,env):
    text=(ROOT/path).read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),path+"#independent_AST","exec"),env)
    return env[name]
def passed(name): CHECKS.append(name)
def rejects(name,fn):
    try: fn()
    except (ValueError,KeyError,AssertionError,TypeError): passed(name);return
    raise AssertionError("Expected rejection: "+name)

# Only author data-fixture and AST-extraction helpers are loaded.
# No author check function or verdict is executed or reused.
allowed={"require","ref","doc","v5_scope","function","original_refs","fixture"}
atree=ast.parse((ROOT/AUTHOR).read_text())
authorenv={"__name__":"independent_fixture_support"}
nodes=[n for n in atree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))
       or isinstance(n,ast.FunctionDef)and n.name in allowed]
exec(compile(ast.Module(body=nodes,type_ignores=[]),AUTHOR+"#data_fixture_only","exec"),authorenv)
fixture=authorenv["fixture"]
m=doc(MANIFEST)
expected={
"SOURCE_HANDOFF.v1.json":"60a0f988b850bdf8313e4ff97fbe7d5788669d0ed241551d66f9eb0d394c40cf",
"worker_entry_v1.py":"d672b38e5d317ad9844104e1e0f741c6e338ac0f33b39e7d0f897f124a01a912",
"source_manifest.v1.json":"ff208ebdf593811aacd876257aaa5c329d904b18522fec97bd306b5f58bea1a5",
"author_RAM_checks_v1.py":"025d7816beb03e061559c7234121ff324ec2eb6f677fc87f9ef725a86e504bbd",
"author_RAM_checks.result.v1.json":"c914bfe1645aab3c586f6c1639acf4b7aaa07d2cc005c43e9605a34e78157861",
"ROOT_RECIPE.v1.md":"b8adadf607a37a7f8ddf77246297c489d95bc185975cecb7c36a93bf26d19a32",
"ROOT_request.template.inactive.v1.json":"050f4ee1fce0eaf87feb1f6fb410d5adc947d27814ca6918c2ad3f67f82574c9"}
for name,digest in expected.items():need(ref(BASE+"/"+name)["sha256"]==digest,name)
passed("sealed_primary_sources_and_handoff_authenticated")
for rr in m["refs"]:need(ref(rr["path"])==rr,rr["path"])
need(len(m["refs"])==13)
passed("all_thirteen_direct_actual_source_refs_authenticated")
condition=m["condition"]
for key in ("inference_config_provenance_ref","inference_runtime_ref"):
    rr=condition[key];need(ref(rr["path"])==rr)
original=doc(condition["inference_config_provenance_ref"]["path"])
need(condition["inference_template_inactive"]==dict(original,execution_enabled=False,observation_ref=None))
passed("scientific_inference_configuration_equals_original_except_inactive_binding")
sim=condition["simulation_stage"]
need(ref(sim["source_manifest_provenance_ref"]["path"])==sim["source_manifest_provenance_ref"])
need(sim["source_sha256"]=={r["path"]:r["sha256"]for r in doc(sim["source_manifest_provenance_ref"]["path"])["refs"]})
need(ref(sim["worker_source_ref"]["path"])==sim["worker_source_ref"])
passed("predecessor_source_map_matches_sealed_v5_simulator")
sf=fixture();need(sf.scope["simulation_prerequisites"](sf.prior,sf.condition)==sf.prior)
q,caps,rt=sf.scope["validate_request"](sf.qr)
need(caps["reservation_bytes"]==13000000000 and q["scientific_config"]["output_cap_bytes"]==12500000000)
need(caps["reservation_bytes"]>=12500000000+caps["aux_bytes"])
passed("complete_positive_parent001_predecessor_and_request_join")

# Independent domain mutations cover source, scientific settings, terminal
# evidence and exact metadata identities. Author fixture reads are RAM data.
mutations=[
("reject_source_map_drift","input_ref",lambda x:x["source_sha256"].update(extra="0"*64)),
("reject_parent001_driver_swap","input_ref",lambda x:x["root_driver_ref"].update(path="not_the_driver.py")),
("reject_simulator_GPU_authority","input_ref",lambda x:x.update(GPU_authorized=True)),
("reject_changed_mutation_seed","config_ref",lambda x:x.update(mutation_seed=1021003)),
("reject_changed_mutation_model","config_ref",lambda x:x.update(mutation_rate=1e-8)),
("reject_nonfull_sequence","config_ref",lambda x:x.update(sequence_length_bp=1000000)),
("reject_nonoriginal_sample_counts","config_ref",lambda x:x["sample_counts"].update(AFR=100)),
("reject_wrong_class_order","completion_ref",lambda x:x.update(class_order=["AFR","EUR","EAS"])),
("reject_unknown_completion","completion_ref",lambda x:x.update(status="UNKNOWN")),
("reject_unknown_wait_status","wait_ref",lambda x:x.update(actual_OS_wait_status=None)),
("reject_nonzero_launcher_wait","wait_ref",lambda x:x.update(actual_direct_launcher_wait=False)),
("reject_nonzero_worker_wait","wait_ref",lambda x:x.update(returncode=1)),
("reject_wait_operational_failure","wait_ref",lambda x:x.update(operational_failure="failed")),
("reject_absent_group_evidence","wait_ref",lambda x:x.update(group_observations=[])),
("reject_undrained_group","wait_ref",lambda x:x.update(group_observations=[dict(absent=None)])),
("reject_nonzero_adopted_child","wait_ref",lambda x:x.update(adopted_waits=[dict(returncode=1)])),
("reject_unconsumed_ROOT_outer","outer_ref",lambda x:x.update(actual_terminal_consumed=False)),
("reject_nonzero_ROOT_outer","outer_ref",lambda x:x.update(actual_exit_code=1)),
("reject_replay_capability_not_revoked","outer_ref",lambda x:x.update(all_prior_launch_resume_write_capabilities_revoked=False)),
("reject_unfrozen_simulated_outputs","outer_ref",lambda x:x.update(simulation_outputs_frozen=False)),
("reject_roster_wrong_diploid_identity","roster_ref",lambda x:x["samples"][1].update(sample_id="other")),
("reject_roster_duplicate_sample_node","roster_ref",lambda x:x["samples"][-1].update(observed_node_id=0)),
("reject_query_truth_label","roster_ref",lambda x:x["samples"][-1].update(observed_reference_class=1)),
("reject_observation_truth_input","observation_ref",lambda x:x.update(query_ancestry_truth_in_inputs=True)),
("reject_observation_payload_escape","observation_ref",lambda x:x["alleles_ref"].update(path="private_truth.jsonl")),
("reject_observation_bad_payload_digest_shape","observation_ref",lambda x:x["positions_ref"].update(sha256="z"*64))]
for title,key,mut in mutations:
    f=fixture();mut(f.rows[f.prior[key]["path"]])
    rejects(title,lambda f=f:f.scope["simulation_prerequisites"](f.prior,f.condition))
f=fixture();p=dict(f.prior);p.pop("outer_ref")
rejects("reject_missing_seventh_predecessor_ref",lambda:f.scope["simulation_prerequisites"](p,f.condition))
f=fixture();p=dict(f.prior,unexpected=f.prior["input_ref"])
rejects("reject_extra_undeclared_predecessor_ref",lambda:f.scope["simulation_prerequisites"](p,f.condition))
for title,mut in [
("reject_nonindependent_review",lambda q,f:f.rows[q["source_review_ref"]["path"]].update(independent=False)),
("reject_review_wrong_manifest",lambda q,f:f.rows[q["source_review_ref"]["path"]].update(source_manifest_ref={})),
("reject_predecessor_binding_swap",lambda q,f:q["bindings"].update(simulation_prerequisites={})),
("reject_reduced_primary_cap",lambda q,f:q["scientific_config"].update(output_cap_bytes=12000000000)),
("reject_enlarged_scientific_cap",lambda q,f:q["scientific_config"].update(output_cap_bytes=13000000000)),
("reject_missing_directive",lambda q,f:q.update(current_Census_original_reserve_execute_authorized=False)),
("reject_capacity_source_estimate",lambda q,f:q["actual_capacity"].update(source_only_estimate=True)),
("reject_future_capacity",lambda q,f:q["actual_capacity"].update(created_at_utc=(datetime.now(timezone.utc)+timedelta(seconds=60)).isoformat())),
("reject_insufficient_whole_scope_capacity",lambda q,f:q["actual_capacity"].update(admissible_new_bytes=12999999999)),
("reject_insufficient_owned_RSS_capacity",lambda q,f:q["actual_capacity"].update(available_owned_RSS_bytes=1)),
("reject_scope_escape",lambda q,f:q.update(output_directory=q["output_directory"]+"/../../escape")),
("reject_wrong_CPU_affinity",lambda q,f:q["actual_capacity"].update(cpu_ids=list(range(1,9)))),
("reject_expiring_authority",lambda q,f:q.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(seconds=900)).isoformat())),
("reject_native_MLM_dependency",lambda q,f:q.update(native_MLM_dependency=True)),
("reject_automatic_retry",lambda q,f:q.update(automatic_retry=True))]:
    f=fixture();mut(f.rows[f.qr["path"]],f)
    rejects(title,lambda f=f:f.scope["validate_request"](f.qr))

# Independently assemble and authenticate full current guard role closure.
f=fixture()
capture=doc("research/reviews/Census_observation_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json")
roles=dict(capture["all_actual_parent_bindings"])
need(len(roles)==46)
roles["Census_observation_explicit_selector_capsule"]=ref(SELECTOR)
need(len(roles)==47)
for rr in roles.values():need(ref(rr["path"])==rr)
need(len({json.dumps(r,sort_keys=True)for r in roles.values()})==40)
cfgpath=f.q["packet_directory"]+"/config.json"
cfg=f.q["scientific_config"]
f.rows[cfgpath]=copy.deepcopy(cfg)
cr=dict(path=cfgpath,sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
f.by_path[cfgpath]=cr
direct=f.scope["direct_roles"](f.qr,f.q,cr,f.qr,f.q["runtime_ref"])
need(len(direct)==25)
bindings=[]
for rr in list(roles.values())+direct:
    if rr not in bindings:bindings.append(copy.deepcopy(rr))
need(len(bindings)==64 and all(r in bindings for r in roles.values()))
need(all(Path(r["path"]).suffix in (".py",".sh",".json")for r in bindings))
passed("independent_full47_role_40_unique_plus25_direct_union64")

# Full actual scope_spec/file_bytes AST against a wholly RAM filesystem.
blobs={}
for rr in bindings:
    data=json.dumps(f.rows[rr["path"]],sort_keys=True).encode()if rr["path"]in f.rows else(ROOT/rr["path"]).read_bytes()
    need(hashlib.sha256(data).hexdigest()==rr["sha256"])
    blobs[rr["path"]]=data
fds={};number=[20]
class RAMPath(PurePosixPath):
    def resolve(self):return self
    def lstat(self):
        data=blobs[str(self.relative_to(RAM_ROOT))]
        return SimpleNamespace(st_mode=stat.S_IFREG|0o400,st_nlink=1,st_size=len(data),st_dev=1,
            st_ino=int(hashlib.sha256(str(self).encode()).hexdigest()[:12],16),st_mtime_ns=1,st_ctime_ns=1)
RAM_ROOT=RAMPath(str(ROOT))
class RAMFile(io.BytesIO):
    def fileno(self):return self.fd
def ram_open(path,flags):
    need(flags==os.O_RDONLY|os.O_NOFOLLOW)
    p=RAMPath(path);p.lstat();fd=number[0];number[0]+=1;fds[fd]=p;return fd
def ram_fdopen(fd,mode):
    need(mode=="rb");stream=RAMFile(blobs[str(fds[fd].relative_to(RAM_ROOT))]);stream.fd=fd;return stream
env=dict(Path=RAMPath,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock",
    MAX_JSON=32*2**20,json=json,hashlib=hashlib,stat=stat,os=SimpleNamespace(
        open=ram_open,fdopen=ram_fdopen,fstat=lambda fd:fds[fd].lstat(),O_RDONLY=os.O_RDONLY,O_NOFOLLOW=os.O_NOFOLLOW))
for name in ("encoded","integer","canonical","file_bytes","parse","paths_overlap","accounting_paths","bound","scope_spec"):
    extract("src/arggnn/project_budget_v4.py",name,env)
reserve=dict(key="H041/TreeLoom-frozen-full30Mb-TRAIN001-infer-independent",
    output=str(RAM_ROOT/f.q["output_directory"]),auxiliary_scopes=[str(RAM_ROOT/f.q["packet_directory"]/"worker_aux")],
    reserved_bytes=13000000000,consumer_bindings=bindings)
need(len(env["scope_spec"](RAM_ROOT,reserve))==2)
passed("original_complete_scope_spec_and_all64_file_bytes_authenticated_in_RAM")
for title,mut in [
("reject_scope_65th_binding",lambda r:r["consumer_bindings"].append(copy.deepcopy(r["consumer_bindings"][0]))),
("reject_scope_lock_binding",lambda r:r["consumer_bindings"][0].update(path=OLD+"/cpu_slot_0.lock")),
("reject_scope_array_binding",lambda r:r["consumer_bindings"][0].update(path="not_metadata.npy")),
("reject_scope_changed_source_digest",lambda r:r["consumer_bindings"][0].update(sha256="0"*64)),
("reject_scope_bool_charge",lambda r:r.update(reserved_bytes=True)),
("reject_scope_overlap",lambda r:r.update(auxiliary_scopes=[r["output"]+"/child"])),
("reject_scope_accounting_overlap",lambda r:r.update(output=str(RAM_ROOT/".runtime/storage/not_product"))),
("reject_scope_extra_field",lambda r:r.update(extra=True))]:
    r=copy.deepcopy(reserve);mut(r)
    rejects(title,lambda r=r:env["scope_spec"](RAM_ROOT,r))
# Event construction is extracted separately. No append/open/syscall is called.
childtree=ast.parse((ROOT/CHILD).read_text())
append=next(n for n in childtree.body if isinstance(n,ast.FunctionDef)and n.name=="_append")
assignment=next(n for n in append.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="event"for t in n.targets))
ar=dict(path=f.q["packet_directory"]+"/reserve_authority.json",sha256="0"*64)
eventenv=dict(state=(None,None,None,roles),events=[],hashes=[],operation="reserve",authority_ref=ar,
    value=dict(copy.deepcopy(reserve),root_admission=ar))
exec(compile(ast.Module(body=[assignment],type_ignores=[]),"RAM_original_event_assignment","exec"),eventenv)
eventbytes=len(env["encoded"](eventenv["event"])+b"\n")
need(eventbytes<65536)
passed("actual_Census_event_assignment_for_full64_bindings_fits64KiB")

# Concrete initialize path with modules represented solely by RAM doubles.
initenv=dict(ROOT=ROOT,REL=BASE,OLD=OLD,MANIFEST=MANIFEST,SOURCE=SOURCE,DRIVER=SOURCE,WORKER=SOURCE,
    PARENT="american_c3_train_001",PHASE="infer_train_001",STAGE="infer",SELECTOR=ref(SELECTOR),CHILD=ref(CHILD),
    OLD_CONTRACT_SHA=ref(OLD+"/root_contract_v1.py")["sha256"],OLD_DRIVER_SHA=ref(OLD+"/root_driver_v1.py")["sha256"],
    c=None,d=None,need=need)
for name in ("source_paths","validate_request","phase_paths","direct_roles","input_spec","validate_completion","guard"):
    initenv[name]=object()
contract=SimpleNamespace(ROOT=ROOT,contextmanager=contextmanager,contextlib=contextlib,read=lambda rr:doc(rr["path"]),ref=ref,
    require=need,local=lambda p:ROOT/p,raw=lambda rr:None,load=lambda *a:None,put=lambda *a:None,current=lambda *a:None,
    PIDFD={},OWNED={})
sentinel=lambda:None
driver=SimpleNamespace(execute=sentinel)
calls=[]
def load_initial(path,digest,name):
    need(ref(path)["sha256"]==digest);calls.append(path)
    return contract if path.endswith("root_contract_v1.py")else driver
initenv["_load_initial"]=load_initial
c,d=extract(SOURCE,"initialize",initenv)()
need(d.execute is sentinel and d.validate_completion is initenv["validate_completion"])
need(c.PARENTS==("american_c3_train_001",)and c.STAGES==("infer",))
need(c.validate_request is initenv["validate_request"]and c.input_spec is initenv["input_spec"])
need(c.cpu_slot.__wrapped__.__code__.co_filename.endswith("#unchanged-shared-slot"))
need(len(calls)==2 and c.GUARD_REF==ref(CHILD))
passed("concrete_initialize_preserves_original_execute_and_wires_exact_actual_context")
# AST invariant: original supervisor, hard gate, shared slot and callable path.
text=(ROOT/SOURCE).read_text();tree=ast.parse(text)
need(not any(isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=="d"
    and n.attr=="execute"and isinstance(n.ctx,ast.Store)for n in ast.walk(tree)))
need("Original shared slot is not genuinely held"in text and "os.getpid()==os.getpgrp()==os.getsid(0)"in text)
need("inference.execute(cfg,Writer("in text)
passed("no_replaced_supervisor_execute_and_actual_original_inference_route")
science=(ROOT/OBS).read_text();science_tree=ast.parse(science)
execute_node=next(n for n in science_tree.body if isinstance(n,ast.FunctionDef)and n.name=="execute")
keys={n.slice.value for n in ast.walk(execute_node)if isinstance(n,ast.Subscript)and isinstance(n.value,ast.Name)
    and n.value.id=="m"and isinstance(n.slice,ast.Constant)}
need({"genotypes_ref","positions_ref","alleles_ref","roster_ref"}.issubset(keys)and "tree_ref"not in keys)
need("truth_ref"not in keys and "private_truth_ref"not in keys)
need("filter_sites=False,filter_individuals=False,filter_populations=False,map_nodes=True"in science)
passed("original_inference_uses_observed_genotypes_not_simulated_tree_or_truth")
runtime=doc(condition["inference_runtime_ref"]["path"])
wp=extract(OLD+"/root_driver_v1.py","worker_python",dict(ROOT=ROOT,read=lambda r:runtime,need=need))
need(wp(dict(stage="infer",runtime_ref={}))==str(ROOT/".runtime/envs/biological_tsinfer_v1/bin/python"))
passed("original_worker_python_resolves_qualified_inference_runtime")

# Independently invoke the concrete callable bridge through RAM-only module doubles.
f=fixture();bridge=dict(f.scope);events=[]
cfg=copy.deepcopy(f.q["scientific_config"]);unchanged=copy.deepcopy(cfg)
cr=dict(path=BASE+"/RAM_ONLY_independent_cfg.json",sha256="1"*64)
bind=[ref(p)for p in (SOURCE,OBS,"src/arggnn/american_c3_simulate_v1.py","src/arggnn/american_c3_common_v1.py",CHILD)]
storage=dict(job_id="RAM_ONLY",output_directory="RAM_ONLY",reserved_new_bytes=13000000000,
    reservation=dict(reserved_bytes=13000000000,consumer_bindings=bind+[cr,f.q["runtime_ref"]]))
spec=dict(root_authorizer="root_agent",execution_authorized=True,GPU_authorized=False,source_ref=ref(SOURCE),config_ref=cr,
    runtime_ref=f.q["runtime_ref"],root_stage_request_ref=f.qr,storage=storage,job_id="RAM_ONLY",output_directory="RAM_ONLY")
prelude=extract(CHILD,"validate_storage_v9",dict(require=need,_raw=lambda rr:events.append(("authenticate",rr)),
    budget_check_v9=lambda *a,**kw:events.append("current_guard_RAM")))
def validate(s,operational,store,**kwargs):
    need(operational==dict(unchanged,output_cap_bytes=13000000000)and operational is not cfg)
    events.append("operational13GB");return prelude(s,operational,store,**kwargs)
g=SimpleNamespace(validate_storage_v9=validate,budget_check_v9=lambda *a,**kw:events.append("writer_current_guard"))
class Writer:
    def __init__(self,s,adapter):
        need(s is spec)
        adapter.budget_check_v9(storage,boundary=True,required_write_bytes=8192)
        self.adapter=adapter
result=dict(path="RAM_ONLY_complete.json",sha256="0"*64)
def original_execute(c,w):
    need(c is cfg and c==unchanged and c["output_cap_bytes"]==12500000000)
    events.append("original_science12p5GB");w.adapter.budget_check_v9(storage,boundary=True,required_write_bytes=4096)
    return result
pkg=ModuleType("arggnn");module=ModuleType("arggnn.american_c3_observe_v3")
common=ModuleType("arggnn.american_c3_common_v1")
module.execute=original_execute;module.__file__=str(ROOT/OBS);pkg.american_c3_observe_v3=module
common.Writer=Writer;common.runtime_report=lambda:dict(packages=runtime["packages"])
modules={"arggnn":pkg,"arggnn.american_c3_observe_v3":module,"arggnn.american_c3_common_v1":common}
prior_modules={k:sys.modules.get(k)for k in modules}
class FakePath:
    def __new__(cls,p):return Path(p)
    @staticmethod
    def cwd():return ROOT
bridge.update(Path=FakePath,sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1),
    prefix=str(ROOT/".runtime/envs/biological_tsinfer_v1")),os=SimpleNamespace(environ={"CUDA_VISIBLE_DEVICES":""}),
    read=lambda rr:cfg if rr==cr else f.read(rr),metadata_ref=ref,CENSUS_CHILD=ref(CHILD),
    local=lambda p:ROOT/p,INFERENCE=OBS,GENERATOR="src/arggnn/american_c3_simulate_v1.py",
    COMMON="src/arggnn/american_c3_common_v1.py")
extract(SOURCE,"_OriginalWriterCurrentGuard",bridge)
fn=extract(SOURCE,"execute_in_admitted_context_v1",bridge)
try:
    sys.modules.update(modules)
    need(fn(cr,spec=spec,current_guard=g,current_guard_source_ref=ref(CHILD),resource_check=events.append)==result)
finally:
    for k,v in prior_modules.items():
        if v is None:sys.modules.pop(k,None)
        else:sys.modules[k]=v
need(cfg==unchanged and events.index("operational13GB")<events.index("original_science12p5GB"))
need(events.count("before_original_Writer_current_Census_boundary")==2)
need(events.count("after_original_Writer_current_Census_boundary")==2)
passed("independent_concrete_bridge_RAM_original_guard13GB_science12p5GB_unchanged")
passed("all_original_Writer_boundaries_reenter_resource_check_and_current_guard_RAM")

# Default invocation must not initialize a framework or acquire any runtime resource.
import argparse
clienv=dict(argparse=argparse,__doc__="RAM only",json=json,initialize=lambda:(_ for _ in ()).throw(AssertionError("initialized")))
cli=extract(SOURCE,"main",clienv)
stream=io.StringIO()
with redirect_stdout(stream):need(cli([])==0)
need(json.loads(stream.getvalue())==dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False))
passed("default_CLI_inactive_without_initialization_or_any_runtime_access")
summary=dict(current_guard_roles=47,current_unique_refs=40,direct_stage_roles=25,union_unique_refs=64,reserve_event_bytes=eventbytes,
    whole_scope_reservation_bytes=13000000000,scientific_cfg_cap_bytes=12500000000)
print(json.dumps(dict(schema="frozen_TRAIN001_observation_independent_AST_RAM_checks_v1",status="PASS",count=len(CHECKS),checks=CHECKS,
    summary=summary,source_ref=ref(SOURCE),source_manifest_ref=ref(MANIFEST),
    independent_checks_source_ref=ref(OUT+"/independent_checks_v1.py"),
    support_reuse="Only author RAM fixture construction and AST-extraction helpers. No author check functions or conclusions were run or copied.",
    actual_guard_imports_calls_or_resource_process_socket_lock_GPU_probes=0,actual_admission_or_scientific_execution=False,
    payload_or_truth_reads=False,source_metadata_reads_only=True),indent=2,sort_keys=True))
