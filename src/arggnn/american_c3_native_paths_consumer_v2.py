"""Fresh CPU consumer wrapper; unchanged native C3 CLI/author corpus semantics."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
SOURCE="src/arggnn/american_c3_native_paths_consumer_v2.py"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--admission");parser.add_argument("--expected-admission-sha256")
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive_native_paths_consumer",CPU_only=True,
            science_imported=False,guard_imported=False)));return 0
    from research.reviews.American_C3_native_paths_root_v2 import native_paths_contract_v2 as c
    c.require(args.admission and args.expected_admission_sha256,"Exact active root CPU admission required")
    spec=c.read(dict(path=args.admission,sha256=args.expected_admission_sha256))
    c.require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and spec["root_authorizer"]=="root_agent" and spec["execution_authorized"] is True
        and spec["GPU_authorized"] is False and spec["parent_id"]=="american_c3_train_000"
        and spec["split"]=="train" and spec["consumer_wrapper_ref"]==c.ref(SOURCE)
        and spec["exact_native_path_count_ref"]==c.EXACT_COUNT_REF
        and spec["source_ref"]==c.PATHS_REF and spec["source_closure_refs"]==c.NATIVE_CLOSURE
        and spec["integration_source_closure_refs"]==c.closure_v1(),"Exact fresh TRAIN-only CPU producer/wrapper closure")
    c.require(json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==c.POLICY_REF,
              "Same exact current policy must reach common CLI adapter")
    now=datetime.now(timezone.utc)
    c.require(datetime.fromisoformat(spec["created_at_utc"])<=now<datetime.fromisoformat(spec["expires_at_utc"]),
              "Unexpired root CPU capability")
    cfg=c.read(spec["config_ref"]);expected,bound=c.configuration_v1(active=True)
    c.require(cfg==expected,"Frozen complete native corpus config; no cap fallback")
    counts=c.validate_counts_v1();c.validate_exact_count_v2(counts)
    for record in [*c.closure_v1(),c.COUNTS_REF,c.EXACT_COUNT_REF,c.RUNTIME_REF,c.POLICY_REF,c.REPAIR_REF]:
        c.require(c.ref(record["path"])==record,"Frozen source/count/runtime/authority changed")
    c.validate_close_receipt_v1(spec["completed_inference_close_receipt_ref"])
    bindings=spec["storage"]["reservation"]["consumer_bindings"]
    c.require(all(r in bindings for r in [*c.closure_v1(),c.COUNTS_REF,c.EXACT_COUNT_REF,c.RUNTIME_REF,c.POLICY_REF,
        c.REPAIR_REF,spec["config_ref"],spec["completed_inference_close_receipt_ref"]]),
        "Actual wrapper/integration/count/runtime/config/close receipt must be reserved")
    from . import american_c3_common_v1 as common
    from . import american_c3_paths_v1 as paths
    observed=common.read(c.OBSERVATION_REF)
    c.require(observed["parent_id"]=="american_c3_train_000" and observed["split"]=="train"
        and observed["sequence_length_bp"]==30000000. and observed["class_order"]==["EUR","EAS","AFR"]
        and observed["query_ancestry_truth_in_inputs"] is False
        and observed["tree_ref"]==counts["tree_ref"],"Exact current TRAIN inferred observation; no truth")
    c.require(common.runtime_report()==common.read(c.RUNTIME_REF),"Exact qualified shared CPU runtime metadata")
    # Install only into this fresh consumer before common.cli loads its original adapter.
    from . import project_budget_v4 as v4
    c.require(c.ref(c.V4_REF["path"])==c.V4_REF,"Originalv4 source required")
    ms=importlib.util.spec_from_file_location("_C3_native_paths_consumer_memo2",ROOT/c.MEMO_REF["path"])
    memo=importlib.util.module_from_spec(ms);ms.loader.exec_module(memo)
    adapter=memo.install(v4,expected_source_path=ROOT/c.V4_REF["path"],
                         expected_source_sha256=c.V4_REF["sha256"])
    common.cli(paths.SOURCE,paths.DEFAULTS,paths.execute,
               dependencies=(paths.AUTHOR.relative_to(ROOT).as_posix(),))
    return 0
if __name__=="__main__":raise SystemExit(main())
