"""Exact reuse of frozen native-v7 runtime readers only, never C4 execution."""
from pathlib import Path
import ast,copy,hashlib,importlib,importlib.machinery,importlib.util,json,os,re,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v2/mlm_v1"
def _contract():
    path=ROOT/REL/"root_native_contract_v1.py"
    spec=importlib.util.spec_from_file_location("_C3_native_runtime_contract",path)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c
def runtime_functions():
    c=_contract();source=c.local(c.LEGACY_RUNTIME_REF["path"])
    c.require(c.ref(source)==c.LEGACY_RUNTIME_REF,"Frozen existing runtime-helper source")
    names={"canonical","exact","sha","confined","reference","bound","read_json","require_fields",
        "native_executable","import_executable_name","distribution_name","native_alias_records",
        "native_alias_contract","native_alias_cycles","native_import_bytes","native_inventory_scan",
        "native_distributions","parse_runtime_inventory","actual_scientific_modules","additive_helper_paths","load_exact_helpers"}
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    c.require({n.name for n in nodes}==names,"Only exact existing runtime-reader definitions")
    ns=dict(ROOT=ROOT,Path=Path,copy=copy,hashlib=hashlib,importlib=importlib,json=json,
        os=os,re=re,stat=stat,sys=sys)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source)+":unchanged_runtime_readers","exec"),ns)
    return c,ns
def validate_native_runtime_v1(runtime_ref,runtime):
    c,ns=runtime_functions();policy=c.read(runtime["legacy_inventory_policy_ref"],cap=33554432)
    old=c.read(runtime["legacy_runtime_bindings_ref"],cap=33554432)
    inventory=c.read(runtime["native_inventory_ref"],cap=33554432)
    prefix=c.local(runtime["native_prefix"])
    c.require(c.local(policy["runtime"]["prefix"])==prefix
        and old["environment_inventory_receipt"]==runtime["native_inventory_ref"]
        and old["python_executable_sha256"]==runtime["native_python_ref"]["sha256"]
        and c.ref(Path(sys.executable).resolve())==runtime["native_python_ref"]
        and Path(sys.prefix).resolve()==prefix,"Actual isolated interpreter and exact existing native inventory")
    checked=ns["parse_runtime_inventory"](policy,inventory,old)
    before=ns["native_inventory_scan"](policy,prefix)
    expected=(inventory["import_file_sha256"],inventory["directory_symlinks"],inventory["absent_executable_symlinks"])
    c.require(ns["exact"](before,expected)
        and ns["native_distributions"](prefix)==inventory["distributions"],
        "Complete actual native import bytes/aliases/distributions before model imports")
    ns["load_exact_helpers"](policy,old)
    receipt=dict(status="passed",runtime_ref=runtime_ref,native_inventory_ref=runtime["native_inventory_ref"],
        legacy_reader_source_ref=c.LEGACY_RUNTIME_REF,actual_prefix=str(prefix),
        complete_import_files=len(inventory["import_file_sha256"]),no_model_constructed=True)
    def verify_imports():
        actual=ns["actual_scientific_modules"](policy,inventory["distributions"])
        c.require(ns["exact"](actual,inventory["scientific_modules"])
            and ns["exact"](ns["native_inventory_scan"](policy,prefix),before),
            "Actual imported native module origins/versions/bytes and post-import inventory")
    return receipt,verify_imports
