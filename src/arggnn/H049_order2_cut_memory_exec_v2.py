"""Canonical H049 GPU dispatcher; frozen scientific producer/context are reused.

The inherited worker_cli_ref names the frozen scientific CLI. The separate
canonical_dispatcher_ref binds this actual launched arggnn module explicitly.
Default is inert. No globals, guards or scientific functions are patched.
"""
from __future__ import annotations
import argparse,hashlib,importlib,json,sys
from pathlib import Path

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/H049_order2_cut_memory_exec_v2.py"
PRODUCER_MODULE="research.reviews.H049_order2_cut_memory_root_source_20261015_v1.producer_v1"
PRODUCER_REF={'path': 'research/reviews/H049_order2_cut_memory_root_source_20261015_v1/producer_v1.py', 'sha256': '445a616937458fe738c516226ab8f851268040ee79fa39519380ba9b2cdfc521'}
CONTEXT_REF={'path': 'research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v2/context_factory_v2.py', 'sha256': 'e1c58d1bcdece6bd8de7fff092369411e0fccc63e671882effa6984c2d50d3e1'}
SUCCESSOR_DRIVER_PATH="research/reviews/H049_order2_cut_memory_gpu_exec_source_20261015_v2/root_driver.py"


def require(ok,message):
    if not ok:
        raise ValueError(message)


def actual_ref_v1(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file(),"Canonical existing source")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def bound_v1(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"}
        and actual_ref_v1(record["path"])==record,"Exact pinned root/source bytes")
    return (ROOT/record["path"]).read_bytes()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_canonical_dispatcher_not_executed",
            hypothesis_id="H049",phases=["train","evaluate"],canonical_module="arggnn.H049_order2_cut_memory_exec_v2",
            scientific_producer_ref=PRODUCER_REF,frozen_context_ref=CONTEXT_REF,
            full_passes=30,updates_per_arm=960,checkpoint_files=18,VAL_pretruth_seals=144,
            Torch_imported=("torch" in sys.modules))));return 0
    require(args.input and args.expected_input_sha256,"Pinned active H049 root capability required")
    record=dict(path=args.input,sha256=args.expected_input_sha256)
    spec=json.loads(bound_v1(record))
    require(spec.get("schema")=="H049_marker_grid_root_input_v1"
        and spec.get("hypothesis_id")=="H049" and spec.get("phase") in ("train","evaluate")
        and spec.get("arm")=="joint_three_arm" and spec.get("status")=="active"
        and spec.get("root_authorizer")=="root_agent" and spec.get("physical_gpu") in (4,5),
        "Fresh active H049 GPU capability")
    cfg=json.loads(bound_v1(spec["refs"]["config"]))
    own=actual_ref_v1(SOURCE);driver=cfg.get("GPU_execution_successor_driver_ref")
    require(spec.get("canonical_dispatcher_ref")==cfg.get("canonical_dispatcher_ref")==own,
        "Actual canonical dispatcher bound by input and exact config")
    require(isinstance(driver,dict) and driver.get("path")==SUCCESSOR_DRIVER_PATH
        and spec.get("root_driver_ref")==driver,"Exact additive GPU root driver")
    bound_v1(driver);bound_v1(PRODUCER_REF);bound_v1(CONTEXT_REF)
    require(spec.get("producer_ref")==spec.get("main_source_ref")==spec.get("worker_cli_ref")==PRODUCER_REF
        and cfg.get("producer_ref")==PRODUCER_REF
        and spec.get("root_context_module_ref")==CONTEXT_REF,
        "Unchanged underlying scientific producer and context")
    require(spec["source_sha256"].get(SOURCE)==own["sha256"]
        and spec["source_sha256"].get(driver["path"])==driver["sha256"],
        "Actual canonical dispatcher and driver in complete source closure")
    producer=importlib.import_module(PRODUCER_MODULE)
    require(producer.SOURCE==PRODUCER_REF["path"],"Frozen scientific producer namespace")
    return producer.main(["--execute","--input",record["path"],
        "--expected-input-sha256",record["sha256"]])

if __name__=="__main__":raise SystemExit(main())
