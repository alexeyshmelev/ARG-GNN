"""Inactive source-only startup consolidation. No IO, guard or installation on import.
Future reviewed H058 driver only. Reuses exact reviewed source-authenticator;
the original selected budget_check and consumer transaction memo stay in use.
"""
from pathlib import Path
import ast,copy
SOURCE="research/reviews/guard_latency_minimal_reuse_source_20261017_v1/startup_reuse_v1.py"
BASE_REF=dict(path="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py",sha256="2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98")
GUARD_REF=dict(path="research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py",sha256="b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164")

def require(ok,message):
    if not ok:raise ValueError(message)

def derive_validate_node_v1(node):
    """Keep every original validation statement; only extend its final growth check."""
    require(isinstance(node,ast.FunctionDef) and node.name=="validate_storage_v9" and not node.decorator_list,"Exact original validation function")
    require(len(node.body)==3 and isinstance(node.body[-1],ast.Return),"Original three-statement validation body")
    original_return=ast.parse("return budget_check_v9(storage,first=True,boundary=True)").body[0]
    same=lambda a,b:ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    require(same(node.body[-1],original_return),"Exact original final budget check")
    require(all(a.arg!="required_write_bytes" for a in node.args.kwonlyargs),"No preexisting growth parameter")
    derived=copy.deepcopy(node);derived.name="_same_validation_with_planned_growth"
    derived.args.kwonlyargs.append(ast.arg(arg="required_write_bytes"))
    derived.args.kw_defaults.append(None)
    derived.body[-1].value.keywords.append(ast.keyword(arg="required_write_bytes",value=ast.Name(id="required_write_bytes",ctx=ast.Load())))
    return ast.fix_missing_locations(derived)

def validate_storage_with_growth_v1(guard,authenticated_base,*,base_ref,spec,cfg,storage,source_ref,config_ref,runtime_ref,required_write_bytes):
    """One original current resource transaction, immediately before first AUX mkdir.

    The caller's sealed future integration must authenticate this helper and the
    exact existing base module before calling. No RootMutationMemo is constructed
    here and no ROOT mutation API is called in a consumer process.
    """
    require(base_ref==BASE_REF,"Only exact existing reviewed source-authenticator")
    # Reuse its original stable no-follow SHA and code-object authentication.
    authenticated_base.functions_from_source(authenticated_base,BASE_REF,("source_bytes","functions_from_source"))
    original,node=authenticated_base.functions_from_source(guard,GUARD_REF,("validate_storage_v9",))["validate_storage_v9"]
    derived=derive_validate_node_v1(node)
    namespace=dict(original.__globals__)
    exec(compile(ast.Module(body=[derived],type_ignores=[]),SOURCE+"#same-original-validation-with-growth","exec",dont_inherit=True,optimize=0),namespace)
    return namespace[derived.name](spec,cfg,storage,source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref,required_write_bytes=required_write_bytes)

if __name__=="__main__":
    print("INACTIVE_SOURCE_ONLY_NO_GUARD_OR_OPERATION")
