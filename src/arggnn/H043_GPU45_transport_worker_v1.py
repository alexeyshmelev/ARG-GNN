"""H043 paired scientific CLI. Inert without --execute and exact active capability."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H043_GPU45_transport_root_source_v1 import root_common_v1 as common
SOURCE="src/arggnn/H043_GPU45_transport_worker_v1.py"
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_no_execution",hypothesis_id="H043",
            paired_arms_one_GPU=True,root_selects_physical_GPU=[4,5])));return 0
    common.require(args.input and args.expected_input_sha256,"Pinned exact root input required")
    record=dict(path=args.input,sha256=args.expected_input_sha256);spec=common.read(record)
    common.require(spec.get("schema")=="H043_paired_transport_root_input_v1"
        and spec.get("status")=="active" and spec.get("root_authorizer")=="root_agent"
        and spec.get("hypothesis_id")=="H043" and spec.get("arm")=="joint_pair"
        and spec.get("physical_gpu") in (4,5),"Active root H043 paired capability required")
    for path,digest in spec["source_sha256"].items():
        common.require(common.ref(path)["sha256"]==digest,"Actual source closure changed")
    common.require(spec["worker_cli_ref"]==common.ref(SOURCE),"Actual CLI must be bound")
    ref=spec["root_context_module_ref"];common.require(common.ref(ref["path"])==ref,"Pinned root context")
    ms=importlib.util.spec_from_file_location("_H043_root_context",ROOT/ref["path"])
    module=importlib.util.module_from_spec(ms);ms.loader.exec_module(module)
    context=module.build_context_v1(record,spec)
    from arggnn.train_anonymous_exact_clade_transport_v1 import run_context
    run_context(context,record);return 0
if __name__=="__main__":raise SystemExit(main())
