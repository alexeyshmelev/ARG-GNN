"""Consumer-only namespace extension to the frozen transaction memo2.

Import is inert. Explicit future-consumer installation adds only the exact
scope_spec list-comprehension and successor_paths generator-comprehension
canonical callers. No guard, journal, accounting or process action at import.
"""
from pathlib import Path
from types import CodeType
import hashlib,importlib.util,marshal

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py"
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",
    sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",
    sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
V9_REF=dict(path="scripts/project_budget_v9.py",
    sha256="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e")

def _source_v1(record):
    path=ROOT/record["path"]
    if path.resolve()!=path or ROOT not in path.parents:
        raise ValueError("Exact canonical project source required")
    raw=path.read_bytes()
    if len(raw)>262144 or hashlib.sha256(raw).hexdigest()!=record["sha256"]:
        raise ValueError("Pinned original source bytes changed")
    return path,raw

def _load_pinned_memo_v1():
    path,_=_source_v1(MEMO_REF)
    spec=importlib.util.spec_from_file_location("_current_mutex_frozen_memo2",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def _original_parent_v1(module,name,record):
    path,raw=_source_v1(record)
    if Path(module.__file__).resolve()!=path:
        raise ValueError("Exact original namespace module pathname required")
    fn=getattr(module,name,None)
    if (fn is None or fn.__module__!=module.__name__
        or hasattr(fn,"__wrapped__") or Path(fn.__code__.co_filename).resolve()!=path):
        raise ValueError("Unwrapped original namespace parent required")
    compiled=compile(raw,str(path),"exec",dont_inherit=True,optimize=0)
    matches=[c for c in compiled.co_consts if isinstance(c,CodeType) and c.co_name==name]
    if len(matches)!=1 or fn.__code__!=matches[0]:
        raise ValueError("Exact SHA-source-derived parent code object required")
    return fn

def _child_v1(parent,name,line):
    matches=[c for c in parent.__code__.co_consts
        if isinstance(c,CodeType) and c.co_name==name and c.co_firstlineno==line]
    if len(matches)!=1:
        raise ValueError("Exact single original namespace comprehension required")
    code=matches[0]
    return code,dict(parent=parent.__qualname__,code_name=name,
        source_file=code.co_filename,first_line=line,
        code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest())

def install_consumer_v1(v4_module,budget_module,*,consumer_role,
                        expected_source_path,expected_source_sha256):
    """Only a separately admitted future read-only consumer may call this.

    Root reserve/close processes never install this adapter. Their original
    mutex, canonical, file_bytes and mutation call objects remain unwrapped.
    The returned object is the unchanged memo2 handle and supports its original
    last_operation_counts()/uninstall() lifecycle.
    """
    if consumer_role!="read_only_consumer":
        raise ValueError("Explicit read-only consumer role required; no mutation installation")
    v4_path,_=_source_v1(V4_REF)
    if Path(expected_source_path)!=v4_path or expected_source_sha256!=V4_REF["sha256"]:
        raise ValueError("Exact original v4 source pin required")
    if getattr(budget_module,"v4",None) is not v4_module:
        raise ValueError("Original budget and consumer must share the exact v4 module")
    scope=_original_parent_v1(v4_module,"scope_spec",V4_REF)
    successor=_original_parent_v1(budget_module,"successor_paths",V9_REF)
    scope_code,scope_record=_child_v1(scope,"<listcomp>",182)
    successor_code,successor_record=_child_v1(successor,"<genexpr>",54)
    base=_load_pinned_memo_v1()
    handle=base.TransactionMemoAdapter(v4_module,
        expected_source_path=v4_path,expected_source_sha256=V4_REF["sha256"])
    original_callers=handle.namespace_callers
    handle.namespace_callers=frozenset((*original_callers,scope_code,successor_code))
    if len(handle.namespace_callers)!=len(original_callers)+2:
        raise ValueError("Exactly two additional namespace callers required")
    handle.receipt.update(namespace_extension_schema="resource_guard_consumer_namespace_memo_v1",
        namespace_extension_ref=dict(path=SOURCE,
            sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()),
        original_memo2_ref=dict(MEMO_REF),original_budget_v9_ref=dict(V9_REF),
        consumer_role=consumer_role,additional_namespace_callers=[scope_record,successor_record],
        transaction_end_namespace_validation="unchanged original memo2 uncached validation before guard returns",
        journals_fingerprints_scope_inventories_uncached=True,
        root_mutation_installation_authorized=False,cross_operation_cache=False)
    return handle.install()
