"""Minimal deferred migration runtime over the unchanged ordinary root CPU monitor.
No source-author execution. No science, guard, admission or process probe on import.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys,types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_full_state_migration_CPU_runtime_source_20261016_v2"
SOURCE=HERE+"/ordinary_migration_cpu_v1.py"
STATIC_CONTRACT_REF={'path': 'research/reviews/C3_full_state_migration_CPU_runtime_source_20261016_v2/contract.source_only.v1.json', 'sha256': '5499c4cbb0cc80bb3cc59f3c0f04bc320d2604b8ba9947cd7d06c5b1b62a0055'}
ORDINARY_REF={'path': 'research/reviews/H050_one_FP32_CPU_forward_qualification_source_20261016_v1/ordinary_root_cpu_v1.py', 'sha256': 'bd1ffa9b3a93a219958a62a3ffa1ef04e2aeacf1f997587b3268bdf7f5c74ae8'}
METADATA_REF={'path': 'research/reviews/C3_full_state_migration_CPU_runtime_source_20261016_v2/request_metadata_v1.py', 'sha256': '73428bd37f722539b0b93a6ddea915c9382e2506879f7c6a48d3aa729a105afc'}
_REQUEST_REF=None
_BASE=None
_METADATA=None

def need(ok,message):
    if not ok:raise ValueError(message)

def load(record,label):
    p=ROOT/record["path"]
    need(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=2**20
        and hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],"Exact existing bounded source before private import")
    m=types.ModuleType(label);m.__file__=str(p);sys.modules[label]=m
    exec(compile(p.read_bytes(),str(p),"exec"),m.__dict__)
    return m

def contract():
    need(_REQUEST_REF is not None and Path.cwd().resolve()==ROOT and Path(__file__).resolve()==ROOT/SOURCE,
        "Fresh explicit fixed root CPU route and actual request")
    c=_BASE.read(STATIC_CONTRACT_REF)
    need(c["schema"]=="C3_full_state_migration_CPU_static_source_contract_v1"
        and c["execution_authorized"] is False and c["ordinary_CPU_runtime_ref"]==ORDINARY_REF
        and c["metadata_ref"]==METADATA_REF,"Exact source-only operational contract")
    request=_BASE.read(_REQUEST_REF)
    need(request["static_contract_ref"]==STATIC_CONTRACT_REF
        and request["actual_runtime_source_ref"]==_BASE.ref(__file__),"Exact independently sealed root request and runtime")
    _METADATA.validate_root_request_v1(request,c)
    rows=[*c["source_binding_basis_refs"],STATIC_CONTRACT_REF,request["migration_intent_ref"],*request["intent_metadata_refs"]]
    c=dict(c,source_binding_basis_refs=_BASE.unique(rows),root_request=request,
        migration_intent_ref=request["migration_intent_ref"],checkpoint_byte_plan=request["checkpoint_byte_plan"])
    need(len(c["source_binding_basis_refs"])+5<=64,"Final ordinary runtime/config/closure/request/source binding union fits64 before admission metadata")
    need(c["complete_execution_scope_cap_bytes"]==8*2**20,"Fixed8MiB complete migration scope")
    return c

def _generated_functions_v1(original_source):
    tree=ast.parse(original_source)
    texts={n.name:ast.get_source_segment(original_source,n) for n in tree.body
        if isinstance(n,ast.FunctionDef) and n.name in ("worker","execute")}
    worker=texts["worker"]
    replacements=[('result=module.run_single_CPU_qualification_v1(resource_check=resource_check)', 'result=module.run_migration_v1(c["root_request"],resource_check=resource_check,guard=m,storage=storage)'), ('fixture_passed=result["fixture_result"]["passed"]', 'migration_passed=result["passed"]'), ('need(cfg==spec["fixed_config"] and cfg["output_cap_bytes"]==CAP and cfg["wrapper_ref"]==c["wrapper_ref"],"Exact fixture/caps/config")', 'need(cfg==spec["fixed_config"] and cfg["output_cap_bytes"]==CAP and cfg["wrapper_ref"]==c["wrapper_ref"],"Exact fixture/caps/config")\n    need(cfg["migration_intent_ref"]==c["migration_intent_ref"] and cfg["checkpoint_byte_plan"]==c["checkpoint_byte_plan"],"Exact actual migration intent and checkpoint-byte plan")')]
    for old,new in replacements:
        need(worker.count(old)==1,"One scientific callback/result adaptation")
        worker=worker.replace(old,new,1)
    execute=texts["execute"]
    substitutions=[('c["minimal_handoff_ref"],', ''), ('contract_ref=CONTRACT_REF)\n    config_ref=meta', 'contract_ref=CONTRACT_REF,migration_intent_ref=c["migration_intent_ref"],checkpoint_byte_plan=c["checkpoint_byte_plan"])\n    config_ref=meta'), ('out.mkdir();aux.mkdir()', 'aux.mkdir()'), ('need(result["fixture_result"]["passed"] is True and result["fixture_result"]["precision"]=="fp32"\n         and result["fixture_result"]["optimizer_steps"]==0\n         and result["fixture_result"]["all_four_arms"] is True\n         and result["fixture_result"]["no_truth_read"] is True\n         and result["fixture_result"]["no_optimization"] is True,"Genuine one FP32 fixture result")', 'need(result.get("passed") is True and result.get("optimizer_updates_performed")==0\n         and result.get("actual_migration_tool_terminal_consumed") is False\n         and result.get("actual_migration_tool_exit_code") is None\n         and result.get("migration_intent_ref")==c["migration_intent_ref"]\n         and result.get("original_model_optimizer_cursor_RNG_order_history_unchanged") is True\n         and result.get("original_sources_and_checkpoint_files_untouched") is True,\n         "Genuine original migration manifest remains unconsumed until actual outer tool0")'), ('status="completed_single_FP32_CPU_fixture"', 'status="completed_CPU_migration_pending_actual_tool_terminal"'), ('actual_fixture_passed=True', 'actual_migration_passed=True')]
    for old,new in substitutions:
        need(execute.count(old)==1,"One narrow ordinary-runtime adaptation: "+old[:90])
        execute=execute.replace(old,new,1)
    return worker.replace("H050_one","C3_migration"),execute.replace("H050_one","C3_migration").replace("Unextended300s lifetime","Unextended1800s lifetime")

def setup(request_ref):
    global _REQUEST_REF,_BASE,_METADATA
    need(_BASE is None,"One fresh monitor or worker interpreter")
    _REQUEST_REF=request_ref
    _METADATA=load(METADATA_REF,"_C3_migration_exact_metadata")
    _BASE=load(ORDINARY_REF,"_C3_migration_private_original_CPU_runtime")
    _BASE.__file__=str(ROOT/SOURCE)
    _BASE.CONTRACT_REF=_REQUEST_REF
    _BASE.HERE=ROOT/HERE
    _BASE.ADMIN=ROOT/HERE/"actual_admission_v1"
    _BASE.CAP=8*2**20
    _BASE.RESULT_CAP=65536
    _BASE.WALL=1800
    _BASE.contract=contract
    text=(ROOT/ORDINARY_REF["path"]).read_text()
    worker,execute=_generated_functions_v1(text)
    # Functions are compiled in this separately imported original module only.
    # No installed/live runtime, guard or old scientific module is mutated.
    exec(compile(worker+"\n\n"+execute,"<C3_original_CPU_monitor_scientific_callback_only>","exec"),_BASE.__dict__)
    return _BASE

def main(argv=None):
    os.chdir(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute",action="store_true");mode.add_argument("--worker",action="store_true")
    parser.add_argument("--request");parser.add_argument("--expected-request-sha256")
    parser.add_argument("--input");parser.add_argument("--expected-sha256")
    parser.add_argument("--monitor-parent-pid",type=int);parser.add_argument("--monitor-read-fd",type=int)
    args=parser.parse_args(argv)
    if args.worker:
        # Read only the sealed small input needed to identify its actual request.
        # Original worker still verifies inherited monitor handshake before imports.
        ir=dict(path=args.input,sha256=args.expected_sha256)
        meta=load(METADATA_REF,"_C3_migration_worker_request_locator")
        spec=meta.read(ir);request_ref=spec["contract_ref"]
        return setup(request_ref).worker(args)
    need(args.request and args.expected_request_sha256,"Explicit actual post-terminal root request required")
    return setup(dict(path=args.request,sha256=args.expected_request_sha256)).execute()

if __name__=="__main__":
    try:raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        print(json.dumps(dict(error=type(exc).__name__,message=str(exc)[:2048],
            actual_migration_pass_not_asserted=True)),flush=True)
        raise SystemExit(1)
