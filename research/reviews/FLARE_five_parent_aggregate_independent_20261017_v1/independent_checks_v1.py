"""Independent source-AST RAM review. No production module or guard import."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from contextlib import contextmanager
import ast,copy,hashlib,json,argparse,io
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SRC=ROOT/"research/reviews/native_benchmark_readiness_20261017_v1/aggregate_cli_v1.py"
TREE=ast.parse(SRC.read_text())
RESULTS=[]

def check(name,fn):
    fn();RESULTS.append(name)
def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError("Expected source predicate rejection")
def fixture(stage="freeze"):
    scope=dict(Path=Path,datetime=datetime,timezone=timezone,argparse=argparse,
               json=json,re=__import__("re"),sys=SimpleNamespace(),read=lambda _:None)
    names={"need","finite","controls","direct_roles","call_science","initial_storage_transaction","main"}
    selected=[n for n in TREE.body if isinstance(n,ast.Assign)or isinstance(n,ast.FunctionDef)and n.name in names]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SRC)+"#AST_RAM","exec"),scope)
    rows={};refs={}
    def put(path,value):
        data=json.dumps(value,sort_keys=True).encode()
        rr=dict(path=path,sha256=hashlib.sha256(data).hexdigest())
        rows[path]=copy.deepcopy(value);refs[path]=rr;return rr
    def raw(rr):
        assert rr==refs[rr["path"]]
        return json.dumps(rows[rr["path"]],sort_keys=True).encode()
    def read(rr):raw(rr);return copy.deepcopy(rows[rr["path"]])
    def reference(path):return refs[str(path)]
    scope.update(raw=raw,read=read,reference=reference)
    put(scope["SOURCE"],dict(source="RAM"))
    config=put(scope["CONFIG"],scope["CFG"])
    for name in ["GUARD","SELECTOR","POLICY","SCIENCE","SCIENCE_MANIFEST"]:
        r=scope[name];rows[r["path"]]={};refs[r["path"]]=r
    r=scope["SCIENCE_REVIEW"];rows[r["path"]]=dict(status="SOURCE_PASS",blocking_findings=[],source_ref=scope["SCIENCE"],source_manifest_ref=scope["SCIENCE_MANIFEST"]);refs[r["path"]]=r
    manifest=put(scope["MANIFEST"],dict(source_ref=reference(scope["SOURCE"]),scientific_source_ref=scope["SCIENCE"],scientific_manifest_ref=scope["SCIENCE_MANIFEST"],scientific_review_ref=scope["SCIENCE_REVIEW"],source_refs=[reference(scope["SOURCE"]),config]))
    review=put("RAM/review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_ref=reference(scope["SOURCE"]),source_manifest_ref=manifest))
    now=datetime.now(timezone.utc);start=(now-timedelta(seconds=20)).isoformat();end=(now+timedelta(minutes=25)).isoformat()
    runtime=put("RAM/runtime.json",dict(runtime=True));context=put("RAM/context.json",dict(context=True))
    seals={p:put("RAM/"+p+".json",dict(parent=p))for p in scope["MISSING"]}
    master=dict(schema="ROOT_FLARE_original5VAL_aggregate_master_contract_v1",root_authorizer="root_agent",active=True,
        source_ref=reference(scope["SOURCE"]),source_manifest_ref=manifest,config_ref=config,guard_ref=scope["GUARD"],selector_ref=scope["SELECTOR"],policy_ref=scope["POLICY"],source_review_ref=review,
        created_at_utc=start,expires_at_utc=end,remaining_seal_refs=seals,cpu_id=7,context_ref=context,runtime_ref=runtime,instance="INDEPENDENT_RAM",
        job_id="FLARE-original5VAL-aggregate-INDEPENDENT_RAM",output_directory="experiments/results/FLARE_original5VAL_aggregate_INDEPENDENT_RAM",packet_directory="research/reviews/FLARE_original5VAL_ROOT_INDEPENDENT_RAM")
    mr=put("RAM/master.json",master)
    q=dict(schema="ROOT_FLARE_original5VAL_aggregate_stage_request_v1",root_authorizer="root_agent",active=True,stage="freeze",ROOT_owns_direct_wait_and_group_monitor=True,
        created_at_utc=start,expires_at_utc=end,master_contract_ref=mr,truth_access_authorized=False,score_authority_ref=None,truth_map_ref=None)
    roles={}
    for i in range(47):
        roles[str(i)]=put("RAM/guard_role_%d.json"%i,dict(role=i))
    binding=list(roles.values())+scope["direct_roles"](q,master)
    storage=dict(job_id=master["job_id"],output_directory=master["output_directory"],reserved_new_bytes=scope["CFG"]["output_cap_bytes"],
        reservation=dict(reserved_bytes=scope["CFG"]["output_cap_bytes"],key="H041/"+master["job_id"],output=str(ROOT/master["output_directory"]),auxiliary_scopes=[str(ROOT/master["packet_directory"])],consumer_bindings=binding),
        accounting_mode="Census_observation_two_scope_retention_v1",storage_policy_selector_ref=scope["SELECTOR"],completed_overlay_policy_ref=scope["POLICY"],budget_components=dict(consumer_adapter=scope["GUARD"],implementation=roles["0"],launcher=roles["1"]),created_at_utc=start,expires_at_utc=end)
    sr=put("RAM/storage.json",storage);q["storage_ref"]=sr
    freeze_ref=put("RAM/freeze.json",q)
    if stage=="score":
        barrier=put("RAM/barrier.json",dict(barrier=True))
        done=put("RAM/freeze.completed.json",dict(status="completed_FLARE_original5VAL_freeze_v1",master_contract_ref=mr,result_ref=barrier,input_ref=freeze_ref,pid=12345))
        wait=put("RAM/direct_wait.json",dict(actual_direct_wait=True,actual_returncode=0,pid=12345))
        term=put("RAM/outer_terminal.json",dict(schema="ROOT_FLARE_original5VAL_freeze_terminal_consumed_v1",root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,all_owned_processes_and_groups_drained=True,actual_tool_session_id=77,completed_ref=done,direct_wait_ref=wait,input_ref=freeze_ref))
        handles=put("RAM/truth_handles.json",dict(schema="ROOT_FLARE_original5VAL_VAL_only_truth_handles_v1",root_authorizer="root_agent",VAL_truth_refs={p:dict(path="RAM_ONLY_"+p,sha256="0"*64)for p in scope["VAL"]},TEST_authorized=False))
        q=dict(q,stage="score",truth_access_authorized=True,freeze_terminal_ref=term,barrier_ref=barrier,score_authority_ref=put("RAM/score_authority.json",dict(score=True)),truth_map_ref=handles)
    qr=put("RAM/"+stage+".request.json",q)
    def mutate(rr,key,value):rows[rr["path"]][key]=value
    return SimpleNamespace(s=scope,q=q,qr=qr,master=master,storage=storage,rows=rows,put=put,read=read,mutate=mutate,roles=roles)

f=fixture();check("freeze_complete_original_conditions_accept",lambda:f.s["controls"](f.qr,"freeze"))
f=fixture("score");check("separate_genuine_freeze0_drained_chain_accept",lambda:f.s["controls"](f.qr,"score"))
cases=[
("freeze_denies_truth", "freeze","q","truth_access_authorized",True),
("freeze_denies_score_authority","freeze","q","score_authority_ref",{}),
("ROOT_monitor_required","freeze","q","ROOT_owns_direct_wait_and_group_monitor",False),
("no_phase_expiry_extension","freeze","q","expires_at_utc",(datetime.now(timezone.utc)+timedelta(minutes=59)).isoformat()),
("master_must_be_active","freeze","master","active",False),
("all_four_remaining_seals_required","freeze","master","remaining_seal_refs",{}),
("one_real_CPU_integer","freeze","master","cpu_id",True),
("fixed_32MiB_cap","freeze","storage","reserved_new_bytes",33554433),
("prior_outer1_blocks_score","score","terminal","actual_exit_code",1),
("unconsumed_prior_outer_blocks_score","score","terminal","actual_terminal_consumed",False),
("undrained_prior_freeze_blocks_score","score","terminal","all_owned_processes_and_groups_drained",False),
("direct_wait1_blocks_score","score","wait","actual_returncode",1),
("unknown_direct_wait_blocks_score","score","wait","actual_direct_wait",False),
("different_wait_PID_blocks_score","score","wait","pid",9876),
("different_barrier_blocks_score","score","done","result_ref",dict(path="RAM/bad",sha256="0"*64)),
("freeze_score_stage_substitution_blocks","score","freeze","stage","score"),
("freeze_truth_in_chain_blocks","score","freeze","truth_access_authorized",True)]
for name,stage,target,key,value in cases:
    f=fixture(stage)
    rr=f.qr if target=="q"else f.q["master_contract_ref"]if target=="master"else f.q["storage_ref"]if target=="storage"else f.q["freeze_terminal_ref"]
    if target in ["wait","done","freeze"]:
        term=f.read(f.q["freeze_terminal_ref"]);rr=term[{"wait":"direct_wait_ref","done":"completed_ref","freeze":"input_ref"}[target]]
    f.mutate(rr,key,value)
    check(name,lambda f=f,stage=stage:rejects(lambda:f.s["controls"](f.qr,stage)))
f=fixture()
f.storage["reservation"]["consumer_bindings"].append(f.storage["reservation"]["consumer_bindings"][0])
f.rows[f.q["storage_ref"]["path"]]=f.storage
check("duplicate_binding_rejected",lambda:rejects(lambda:f.s["controls"](f.qr,"freeze")))

for stage in ["freeze","score"]:
    f=fixture(stage);events=[]
    class Science:
        def freeze_remaining_predictions_v1(self,context,seals,**kw):
            assert set(seals)==set(f.s["MISSING"])and context==f.master["context_ref"]
            assert set(kw)=={"max_npz_uncompressed_bytes","writer","runtime_guard"}
            events.append("freeze_no_truth_resolver");return {"barrier":True}
        def score_five_original_VAL_v1(self,barrier,**kw):
            assert barrier==f.q["barrier_ref"]
            assert kw["max_truth_uncompressed_bytes"]==kw["max_npz_uncompressed_bytes"]==128*2**20
            for p in f.s["VAL"]:assert kw["resolve_val_truth"](p)["path"]=="RAM_ONLY_"+p
            rejects(lambda:kw["resolve_val_truth"]("american_c3_test_000"))
            events.append("score_VAL_only");return {"score":True}
    check("exact_"+stage+"_scientific_call_signature_and_caps",lambda:f.s["call_science"](stage,f.q,f.master,Science(),object(),lambda *a:None,read_document=f.read))
    assert events==(["freeze_no_truth_resolver"]if stage=="freeze"else["score_VAL_only"])

f=fixture();events=[]
@contextmanager
def mutex(root):
    assert root==ROOT;events.append("lock")
    try:yield
    finally:events.append("unlock")
guard_source=ROOT/"research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py"
gt=ast.parse(guard_source.read_text())
fn=next(n for n in gt.body if isinstance(n,ast.FunctionDef)and n.name=="consumer_bindings_v1")
gscope=dict(require=f.s["need"],_raw=lambda r:None,copy=copy)
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(guard_source)+"#original_union_RAM","exec"),gscope)
def state(*args,**kw):
    assert kw==dict(fresh=True)and events[-1]=="lock";events.append("fresh47")
    return (None,None,None,f.roles,None,None,None)
def locked(*args,**kw):
    assert kw==dict(first=True,boundary=True)and events[-1]=="fresh47";events.append("original_locked_checker")
    return dict(bindings=f.roles)
g=SimpleNamespace(budget=SimpleNamespace(v4=SimpleNamespace(mutex=mutex)),_state_locked=state,consumer_bindings_v1=gscope["consumer_bindings_v1"],check_storage_locked_v1=locked)
check("original_mutex_fresh47_union_and_original_locked_check",lambda:f.s["initial_storage_transaction"](f.q,f.master,f.storage,g))
assert events==["lock","fresh47","original_locked_checker","unlock"]
# Exact immutable actual40-unique lineage, no live state or guard probe.
capture=ROOT/"research/reviews/Census_observation_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json"
assert hashlib.sha256(capture.read_bytes()).hexdigest()=="58ee37959e97f2cb300a03ff7167fda10d7dbdd3a36a1dd947476f7e4cef0004"
actual=list(json.loads(capture.read_text())["all_actual_parent_bindings"].values())+[f.s["SELECTOR"]]
direct=f.s["direct_roles"](f.q,f.master)
union=gscope["consumer_bindings_v1"](actual,direct)
assert len(actual)==47 and len(union)<=64 and all(r in union for r in actual+direct)
RESULTS.append("all47_actual_lineage_roles_plus_complete7_stage_roles_fit64")
science=ROOT/"research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py"
st=ast.parse(science.read_text())
score=next(n for n in st.body if isinstance(n,ast.FunctionDef)and n.name=="score_five_original_VAL_v1")
cell_loop=next(i for i,n in enumerate(score.body)if isinstance(n,ast.For)and any(isinstance(v,ast.Call)and isinstance(v.func,ast.Name)and v.func.id=="_cell"for v in ast.walk(n)))
truth_loop=next(i for i,n in enumerate(score.body)if isinstance(n,ast.For)and any(isinstance(v,ast.Call)and isinstance(v.func,ast.Name)and v.func.id=="resolve_val_truth"for v in ast.walk(n)))
assert cell_loop<truth_loop
RESULTS.append("unchanged_science_checks_all5_payloads_before_any_truth_resolver_AST")
freeze=next(n for n in st.body if isinstance(n,ast.FunctionDef)and n.name=="freeze_remaining_predictions_v1")
assert not any(isinstance(n,ast.Name)and n.id in ["resolve_val_truth","_truth"]for n in ast.walk(freeze))
RESULTS.append("unchanged_freeze_has_no_truth_function_or_resolver_AST")
execute=next(n for n in TREE.body if isinstance(n,ast.FunctionDef)and n.name=="execute")
assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="setrlimit"for n in ast.walk(execute))
assert "actual_direct_wait_and_ROOT_outer_consumption_pending" in ast.dump(execute)
RESULTS.append("actual_CLI_retains_RLIMITs_and_terminal_pending_not_fabricated_wait_AST")
# Exercise the original complete reservation schema, including allowed ref suffixes.
from pathlib import PurePosixPath
class RAMPath(PurePosixPath):
    def resolve(self):return self
budget_source=ROOT/"src/arggnn/project_budget_v4.py"
budget_tree=ast.parse(budget_source.read_text())
names={"integer","canonical","paths_overlap","accounting_paths","bound","scope_spec"}
nodes=[n for n in budget_tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
known={r["path"]:r["sha256"]for r in union}
bg=dict(Path=RAMPath,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock",
    file_bytes=lambda root,path:(b"RAM_BYTES",known[str(path)]),parse=json.loads)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(budget_source)+"#full_scope_spec_RAM","exec"),bg)
reserve=dict(key="H041/"+f.master["job_id"],output=str(ROOT/f.master["output_directory"]),
    auxiliary_scopes=[str(ROOT/f.master["packet_directory"])],reserved_bytes=33554432,consumer_bindings=copy.deepcopy(union))
check("full_original_scope_spec_accepts_complete_actual_role_union",lambda:bg["scope_spec"](RAMPath(str(ROOT)),reserve))
for suffix in [".lock",".npz"]:
    bad=copy.deepcopy(reserve);bad["consumer_bindings"].append(dict(path="RAM/disallowed"+suffix,sha256="0"*64))
    check("full_original_scope_spec_rejects_"+suffix,lambda bad=bad:rejects(lambda:bg["scope_spec"](RAMPath(str(ROOT)),bad)))
# Actual public CLI dispatch without operational execute.
from contextlib import redirect_stdout
for stage in ["freeze","score"]:
    ff=fixture(stage);seen=[]
    ff.s["execute"]=lambda rr,st:(seen.append((rr,st))or dict(returncode=0,fixture_only=True))
    with redirect_stdout(io.StringIO()):
        code=ff.s["main"](["--"+stage+"-ref",json.dumps(ff.qr)])
    assert code==0 and seen==[(ff.qr,stage)]
    RESULTS.append("actual_"+stage+"_CLI_dispatch_exact_ref_no_dummy_context")
with redirect_stdout(io.StringIO()):
    ff=fixture();ff.s["execute"]=lambda *a:(_ for _ in ()).throw(AssertionError("Default must be inert"))
    assert ff.s["main"]([])==0
RESULTS.append("actual_default_CLI_inert_before_guard_or_science")
print(json.dumps(dict(schema="FLARE_aggregate_independent_AST_RAM_checks_v1",count=len(RESULTS),checks=RESULTS,
 actual_lineage_roles=len(actual),actual_unique_lineage=len({r["path"]for r in actual}),stage_direct_roles=len(direct),complete_unique_union=len(union),
 all47_roles_preserved=True,guard_imported_or_configured=False,production_module_imported=False,
 scientific_arrays_truth_or_checkpoint_read=False,process_resource_or_accounting_probe=False),indent=2,sort_keys=True))
