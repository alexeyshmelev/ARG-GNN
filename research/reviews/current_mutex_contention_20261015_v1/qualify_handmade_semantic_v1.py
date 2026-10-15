"""One focused handmade semantic scenario. No production guard/module import."""
from pathlib import Path
from types import ModuleType,SimpleNamespace,CodeType,FunctionType
import ast,contextlib,fcntl,hashlib,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/current_mutex_contention_20261015_v1"
V4=ROOT/"src/arggnn/project_budget_v4.py"
V9=ROOT/"scripts/project_budget_v9.py"
GUARD=ROOT/"research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py"
BASE=ROOT/"research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py"
NEW=HERE/"consumer_namespace_memo_v1.py"
RUN=HERE/"handmade_semantic_fixture_v4"

def source_namespace(path,name,functions=None,extra=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body if functions is None else [
        n for n in tree.body if
        isinstance(n,ast.ImportFrom) and n.module=="__future__"
        or isinstance(n,ast.FunctionDef) and n.name in functions]
    if functions is not None:
        assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==set(functions)
    m=ModuleType(name);m.__file__=str(path)
    if extra:m.__dict__.update(extra)
    if functions is None:
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec",
            dont_inherit=True,optimize=0),m.__dict__)
    else:
        # Preserve full-module compiler decisions without executing any import
        # or module body from a production guard.
        full=compile(path.read_bytes(),str(path),"exec",dont_inherit=True,optimize=0)
        for node in nodes:
            if not isinstance(node,ast.FunctionDef):continue
            matches=[c for c in full.co_consts if isinstance(c,CodeType) and c.co_name==node.name]
            assert len(matches)==1
            defaults=tuple(ast.literal_eval(d) for d in node.args.defaults) or None
            fn=FunctionType(matches[0],m.__dict__,node.name,defaults)
            fn.__kwdefaults__={a.arg:ast.literal_eval(d)
                for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults) if d is not None} or None
            for decorator in reversed(node.decorator_list):
                fn=eval(compile(ast.Expression(decorator),str(path),"eval",
                    dont_inherit=True,optimize=0),m.__dict__)(fn)
            m.__dict__[node.name]=fn
    return m

def fixture_v4(directory,name):
    names={"encoded","integer","canonical","file_bytes","parse","read_json","reference",
        "bound","mutex","paths_overlap","accounting_paths","scope_spec","measure_scopes"}
    return source_namespace(V4,name,names,dict(Path=Path,contextlib=contextlib,fcntl=fcntl,
        hashlib=hashlib,json=json,os=os,stat=stat,ROOT=directory,
        LOCK="fixture_scheduler.lock",LEGACY="fixture_legacy.json",MAX_JSON=32*1024*1024))

def profile_original_canonical(m,body):
    code=m.canonical.__code__;count=0
    def profiler(frame,event,arg):
        nonlocal count
        if event=="call" and frame.f_code is code:count+=1
    sys.setprofile(profiler)
    try:body()
    finally:sys.setprofile(None)
    return count

assert not RUN.exists()
RUN.mkdir()
for directory in [RUN/"consumer",RUN/"mutation_role"]:
    directory.mkdir();(directory/"fixture_scheduler.lock").write_text("")
d=RUN/"consumer"
(d/"src/arggnn").mkdir(parents=True)
(d/"src/arggnn/project_budget_v9_consumer.py").write_text("fixture marker\n")
(d/"metadata.json").write_text('{"fixture":17}\n')
for n in ("owned","aux","alternate"):(d/n).mkdir()
(d/"owned/payload.txt").write_text("handmade")
v4=fixture_v4(d,"_handmade_consumer_v4")
v9=source_namespace(V9,"_handmade_consumer_v9",{"successor_paths"},dict(v4=v4,JOURNAL_REL=".runtime/storage/fixture_v9"))
base=source_namespace(BASE,"_handmade_base_memo2")
candidate=source_namespace(NEW,"_handmade_namespace_candidate")
# Isolate only dynamic dependency loading: execute the exact memo2 source in
# this handmade namespace, never import/configure a production guard.
candidate._load_pinned_memo_v1=lambda:base
source_sha=hashlib.sha256(V4.read_bytes()).hexdigest()
bindings=[v4.reference(d,"metadata.json"),
    v4.reference(d,"src/arggnn/project_budget_v9_consumer.py")]
reservation=dict(key="fixture/owned",output=str(d/"owned"),auxiliary_scopes=[str(d/"aux")],
    reserved_bytes=1048576,consumer_bindings=bindings)
original={n:getattr(v4,n) for n in ("canonical","file_bytes","mutex")}
original_functions=dict(scope_spec=v4.scope_spec,successor_paths=v9.successor_paths)

# Capture the real mutation API AST unchanged in an independent, unmodified
# root-role namespace. Calls stop before state/authority/write work.
mv4=fixture_v4(RUN/"mutation_role","_handmade_root_mutation_v4")
class FixtureStateStop(Exception):pass
seen=[]
def stop_state(*args,**kwargs):
    seen.append("original_mutation_state_entry")
    raise FixtureStateStop()
mutation=source_namespace(GUARD,"_handmade_original_mutation",{"apply_reserve_v1","apply_close_completed_v1"},
    dict(ROOT=RUN/"mutation_role",budget=SimpleNamespace(v4=mv4),_state_locked=stop_state))
mutation_identity={n:getattr(mutation,n) for n in ("apply_reserve_v1","apply_close_completed_v1")}
mutation_io={n:getattr(mv4,n) for n in ("canonical","file_bytes","mutex")}

measurements=[]
def repeated_body():
    with v4.mutex(d):
        for _ in range(8):paths=v9.successor_paths(d,reservation)
        measurements.append(v4.measure_scopes(d,paths))
h=base.install(v4,expected_source_path=V4,expected_source_sha256=source_sha)
# The profiler counts original canonical code, even after module replacement.
def count_body(body):
    code=original["canonical"].__code__;count=0
    def profile(frame,event,arg):
        nonlocal count
        if event=="call" and frame.f_code is code:count+=1
    sys.setprofile(profile)
    try:body()
    finally:sys.setprofile(None)
    return count
base_canonical=count_body(repeated_body)
base_counts=h.last_operation_counts();h.uninstall()
h=candidate.install_consumer_v1(v4,v9,consumer_role="read_only_consumer",
    expected_source_path=V4,expected_source_sha256=source_sha)
candidate_canonical=count_body(repeated_body)
candidate_counts=h.last_operation_counts()
assert measurements[0]==measurements[1]
assert candidate_canonical<base_canonical,(base_canonical,candidate_canonical)
assert len(h.receipt["additional_namespace_callers"])==2
assert v4.scope_spec is original_functions["scope_spec"]
assert v9.successor_paths is original_functions["successor_paths"]

# The deliberately stale namespace hit must fail at unchanged transaction-end
# validation before guard authority is returned to a caller's next write.
returned_authority=False;mutation_rejection=None
try:
    with v4.mutex(d):
        v9.successor_paths(d,reservation)
        (d/"owned").rename(d/"owned_preserved")
        (d/"owned").symlink_to(d/"alternate",target_is_directory=True)
    returned_authority=True
except ValueError as exc:mutation_rejection=str(exc)
assert not returned_authority and mutation_rejection
assert h._state() is None and h.last_operation_counts()["success"] is False

# Same focused scenario proves the separate original root-mutation route still
# dispatches unchanged API and original lock/I/O objects, never the consumer cache.
for name,fn in mutation_identity.items():
    assert getattr(mutation,name) is fn
    assert all(getattr(mv4,n) is obj for n,obj in mutation_io.items())
    try:
        fn(dict(path="fixture-only.json",sha256="0"*64),
            expected_source_sha256="0"*64,expected_launcher_sha256="0"*64)
    except FixtureStateStop:pass
    else:raise AssertionError("Handmade original mutation call did not reach its original state entry")
assert len(seen)==2
h.uninstall()
assert all(getattr(v4,n) is obj for n,obj in original.items())
assert all(getattr(mv4,n) is obj for n,obj in mutation_io.items())

receipt=dict(schema="consumer_namespace_memo_focused_handmade_semantic_receipt_v1",
    passed=True,scenario="eight original scope+marker parses, fresh complete inventory, then in-transaction scope symlink mutation",
    source_AST_only=True,production_guard_or_module_imports=False,
    fixture_only_original_mutation_calls_stop_before_state_or_write=True,
    memo2_original_canonical_calls=base_canonical,candidate_original_canonical_calls=candidate_canonical,
    memo2_counts=base_counts,candidate_counts=candidate_counts,
    same_complete_scope_measurement=measurements[0],
    end_transaction_namespace_mutation_rejected=mutation_rejection,
    returned_guard_authority_after_mutation=returned_authority,
    exact_additional_callers=h.receipt["additional_namespace_callers"],
    mutation_API_call_objects_unchanged=True,mutation_role_original_IO_and_mutex_objects_unchanged=True,
    mutation_fixture_original_dispatches=len(seen),consumer_original_objects_restored=True,
    real_guard_configured=False,real_accounting_queried_or_mutated=False,
    real_payloads_or_processes_accessed=False,performance_speedup_claim=False)
with (HERE/"handmade_semantic_receipt.v1.json").open("x") as f:
    json.dump(receipt,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(receipt,sort_keys=True,indent=2))
