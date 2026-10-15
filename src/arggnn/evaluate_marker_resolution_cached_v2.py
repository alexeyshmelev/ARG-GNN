"""Canonical H044 evaluation-only entry for the unchanged approved launcher.

Source-bound trampoline. Default is inactive and imports no project runtime.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/evaluate_marker_resolution_cached_v2.py"
CONTEXT="research/reviews/H044_cached_evaluation_root_v2/context_factory_v2.py"
PRODUCER_REF={'path': 'research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py', 'sha256': '73a00a99de9f0ff75bfb49e863a6da84e1fa9e6d2e39a76913c3e7b4ceb0dde1'}
CACHE_REF={'path': 'src/arggnn/marker_resolution_input_cache_v2.py', 'sha256': 'a1b12a31b1890b84e9033da216ce107bb3ce6c47f687876487514f010e3b1987'}

def require(value,message):
    if not value:raise ValueError(message)

def actual(path):
    file=ROOT/path
    require(file.resolve()==file and file.is_relative_to(ROOT) and file.is_file(),"Canonical project source required")
    return dict(path=path,sha256=hashlib.sha256(file.read_bytes()).hexdigest())

def load_source(name,record):
    require(actual(record["path"])==record,"Pinned source changed")
    module_spec=importlib.util.spec_from_file_location(name,ROOT/record["path"])
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed",source=SOURCE,phase="evaluate",
            producer_ref=PRODUCER_REF,cache_ref=CACHE_REF,optimization_performed=False)))
        return 0
    require(args.input and args.expected_input_sha256,"Fresh exact root-issued evaluation input required")
    file=Path(args.input);file=file if file.is_absolute() else ROOT/file
    require(file.resolve()==file and file.is_relative_to(ROOT),"Canonical root-owned input required")
    data=file.read_bytes();require(hashlib.sha256(data).hexdigest()==args.expected_input_sha256,"Exact input SHA required")
    record=dict(path=str(file.relative_to(ROOT)),sha256=args.expected_input_sha256);spec=json.loads(data)
    require(spec.get("schema")=="H044_cached_evaluation_root_input_v2"
        and spec.get("status")=="active" and spec.get("root_authorizer")=="root_agent"
        and spec.get("phase")=="evaluate" and spec.get("hypothesis_id")=="H044"
        and spec.get("evaluation_authorized") is True and spec.get("fit_authorized") is False
        and spec.get("main_source_ref")==actual(SOURCE)
        and spec.get("cache_producer_source_ref")==PRODUCER_REF
        and spec.get("cache_source_ref")==CACHE_REF
        and spec.get("root_context_module_ref",{}).get("path")==CONTEXT,
        "Actual canonical worker, producer/cache and evaluation-only root capability required")
    require(actual(PRODUCER_REF["path"])==PRODUCER_REF and actual(CACHE_REF["path"])==CACHE_REF,
        "Frozen producer/cache sources changed")
    context=load_source("_H044_cached_evaluation_context_v2",spec["root_context_module_ref"])
    ctx=context.build_context_v2(record,spec)
    require(ctx.get("cached_execution_ref")==record and ctx.get("cache_policy_ref")==spec["cache_policy_ref"],
        "Context must preserve exact new execution capability")
    producer=load_source("_H044_cached_evaluation_producer_v2",PRODUCER_REF)
    return producer.run_context_v2(ctx,record)

if __name__=="__main__":raise SystemExit(main())
