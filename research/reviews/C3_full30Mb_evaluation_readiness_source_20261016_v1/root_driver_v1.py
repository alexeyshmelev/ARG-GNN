"""ROOT entry. Exact reviewed CPU supervisor functions reused without body edits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,collections,hashlib,importlib.util,json,os,selectors,signal,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1"
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/HERE),
 str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),
 str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12")]
import evaluation_contract_v1 as c
SOURCE=HERE+"/root_driver_v1.py"
STATUSES=c.STATUSES
SUPERVISOR_FUNCTIONS=("_stat","_tree","_absent_group","_harvest_owned_roster","_signal_owned","launch")
def bind_reviewed_supervisor_v1():
    c.require(c.ref(c.BASE_DRIVER_REF["path"])==c.BASE_DRIVER_REF,"Exact reviewed source before CPU supervisor reuse")
    source=c.local(c.BASE_DRIVER_REF["path"]).read_text();tree=ast.parse(source)
    nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    c.require(all(name in nodes for name in SUPERVISOR_FUNCTIONS),"Exact finite original supervisor APIs")
    # Compile only exact original function ASTs. No import of old policy/admission/close entrypoints.
    selected=ast.Module(body=[nodes[name] for name in SUPERVISOR_FUNCTIONS],type_ignores=[])
    ast.fix_missing_locations(selected)
    namespace=dict(globals())
    exec(compile(selected,str(c.local(c.BASE_DRIVER_REF["path"])),"exec"),namespace)
    return namespace["launch"]
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--launch",action="store_true")
    p.add_argument("--input");p.add_argument("--expected-input-sha256");a=p.parse_args(argv)
    if not a.launch:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE_ROOT_INPUT_AND_REVIEW_REQUIRED",phases=list(c.PHASES),
            accounting_invoked=False,actual_execution=False,GPU_attachment_implemented=False)));return 0
    c.require(a.input and a.expected_input_sha256,"Exact separately admitted ROOT input reference")
    spec,req,caps=c.set_active_spec_v1(dict(path=a.input,sha256=a.expected_input_sha256))
    packet=c.local(spec["packet_directory"]);admitted=c.read(c.ref(packet/"admitted.json"))
    c.require(admitted["input_ref"]==dict(path=a.input,sha256=a.expected_input_sha256)
        and admitted["driver_ref"]==c.ref(SOURCE),"Actual separate admission packet, no old-controller submission")
    return bind_reviewed_supervisor_v1()(spec["phase"])
if __name__=="__main__":raise SystemExit(main())
