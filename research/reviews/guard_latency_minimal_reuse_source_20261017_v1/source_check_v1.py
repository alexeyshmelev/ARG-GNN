"""AST/source-only verification; imports no guard or installed project module."""
from pathlib import Path
import ast,copy,hashlib,json
BASE=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=BASE/"research/reviews/guard_latency_minimal_reuse_source_20261017_v1"
PINS={
"driver":("research/reviews/H058_GPU0_full_training_successor_source_20261017_v4/root_driver_v1.py","695499bd2bb3077808dd979012ab35e1f04e7f0a667cd209dd108e0ce90c82a8"),
"common":("research/reviews/H058_GPU0_full_training_successor_source_20261017_v4/common_v1.py","4b95ed276053dacb8fa28c80b7df78845f661204e8d1bf42f2b60da44879cc81"),
"context":("research/reviews/H058_GPU0_full_training_successor_source_20261017_v4/context_factory_v1.py","640a574a11d4119953acf96057667313e6739d75aeb5bf553889a618706cde34"),
"guard":("research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py","b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164"),
"base":("research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py","2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98"),
"consumer":("research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py","17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f"),
"txmemo":("research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py","c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364"),
}
sources={}
for key,(path,sha) in PINS.items():
    raw=(BASE/path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sha,("Changed source pin",path)
    sources[key]=raw.decode()
same=lambda a,b:ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
node=lambda src,name:next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)and n.name==name)
helper=(OUT/"startup_reuse_v1.py").read_text()
scope={"__name__":"_source_only_startup_AST_checks","__file__":str(OUT/"startup_reuse_v1.py")}
exec(compile(helper,str(OUT/"startup_reuse_v1.py"),"exec"),scope)
derive=scope["derive_validate_node_v1"]
original=node(sources["guard"],"validate_storage_v9")
merged=derive(original)
cases=[]
def passed(name):cases.append(dict(name=name,status="PASS"))
assert all(same(a,b)for a,b in zip(original.body[:-1],merged.body[:-1]))
passed("Every original storage/spec/source assertion and raw authentication statement unchanged")
expected=copy.deepcopy(original)
expected.name="_same_validation_with_planned_growth"
expected.args.kwonlyargs.append(ast.arg(arg="required_write_bytes"));expected.args.kw_defaults.append(None)
expected.body[-1].value.keywords.append(ast.keyword(arg="required_write_bytes",value=ast.Name(id="required_write_bytes",ctx=ast.Load())))
assert same(expected,merged)
compile(ast.Module(body=[merged],type_ignores=[]),"<AST_validation_not_executed>","exec")
passed("Only mandatory planned-growth parameter and original budget-check keyword added")
assert "first=True" in ast.unparse(merged.body[-1])and "boundary=True" in ast.unparse(merged.body[-1])
passed("Original first and boundary flags retained")
for label,mutate in [
("changed return",lambda n:setattr(n.body[-1],"value",ast.Constant(value=None))),
("extra body statement",lambda n:n.body.insert(0,ast.Pass())),
("wrong function",lambda n:setattr(n,"name","other")),
("preexisting growth parameter",lambda n:n.args.kwonlyargs.append(ast.arg(arg="required_write_bytes")))]:
    changed=copy.deepcopy(original);mutate(changed)
    try:derive(changed)
    except ValueError:passed("Reject "+label)
    else:raise AssertionError("Accepted incompatible AST:"+label)
driver=sources["driver"]
execute=node(driver,"execute_v1")
unparse=lambda n:ast.unparse(n)
start=next(i for i,n in enumerate(execute.body)if unparse(n)=="g = guard_v1()")
block=execute.body[start:start+8]
assert len(block)==8
assert unparse(block[1]).startswith("g.validate_storage_v9(")
assert unparse(block[2]).startswith("out, aux =")
assert unparse(block[3]).startswith("require(not out.exists()")
assert unparse(block[4])=="paths = worker_env_paths_v1(s)"
assert unparse(block[5]).startswith("directories = sorted(")
assert unparse(block[6])=="g.budget_check_v9(storage, boundary=True, required_write_bytes=(2 + len(directories)) * 8192)"
assert unparse(block[7])=="aux.mkdir()"
passed("Exact adjacent two-check startup region; only path/absence planning between checks")
planning=block[2:6]
for n in planning:
    assert not any(isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr in("mkdir","write_text","write_bytes","open","Popen","apply_reserve_v1")for x in ast.walk(n))
paths=node(sources["common"],"worker_env_paths_v1")
assert len(paths.body)==2 and isinstance(paths.body[0],ast.Assign)and isinstance(paths.body[1],ast.Return)
passed("Moved intervening plan has no original write, GPU, lease or accounting operation")
check=node(sources["guard"],"check_storage_locked_v1")
checks=ast.unparse(check)
assert "budget.v4.integer(required_write_bytes" in checks and "not required_write_bytes or boundary" in checks
assert "_state_locked(" in checks and "assert_open(" in checks and "budget.v4.measure_scopes(" in checks
assert "measurement['conservative_scope_bytes'] + required_write_bytes <= own['reserved_bytes']" in checks
passed("Original fresh accounting replay/open reservation/full scope measure/growth bound remain sole resource path")
assert "memo.read_state_v1(" in driver and "memo.reserve_v1(" in driver
assert 'return _SELECTOR_API.guard(consumer)' in sources["common"]
assert 'root_mutation_installation_authorized=False,cross_operation_cache=False' in sources["consumer"]
passed("Existing mutation and consumer memo adoption recognized; no duplicate installation proposed")
assert 'self._validate_snapshot(state)' in sources["txmemo"]and 'state["raw"].clear();state["canonical"].clear()' in sources["txmemo"]
passed("Existing consumer transaction-final validation and cache destruction unchanged")
assert "No nested or unsupported mutation memo" in sources["base"]and 'mode in("state","reserve")' in sources["base"]
passed("ROOT read-state/reserve APIs remain distinct; no cross-operation state or prefix cache")
helper_tree=ast.parse(helper)
calls_outside_functions=[n for n in helper_tree.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)]
assert not calls_outside_functions
assert "RootMutationMemo(" not in helper and "_append(" not in helper and ".install(" not in helper
passed("New helper has no guard import/install/accounting operation at import; uses source-auth API only")
result=dict(schema="guard_latency_minimal_reuse_source_check_v1",status="PASS_SOURCE_AST_ONLY",case_count=len(cases),cases=cases,source_refs={k:dict(path=p,sha256=s)for k,(p,s)in PINS.items()},guard_imported=False,guard_executed=False,derived_guard_body_executed=False,original_mutex_or_journals_touched=False,process_inspected=False,timing_or_numerical_probe=False,live_source_modified=False,latency_gain_measured=False,independent_review=False)
p=OUT/"source_check.result.v1.json"
if p.exists():raise ValueError("Immutable source-check result exists")
p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status=result["status"],case_count=len(cases),result_ref=dict(path=str(p.relative_to(BASE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))))
