"""Future ROOT-admitted observed validation/scoring; default makes no reads or calls."""
import argparse,json,re

def direct_refs_v3(q,request_ref,stage):
    from . import prepare_outer000_v1 as source
    from .outer_prediction_validator_v2 import HERE,ref
    refs=[ref(HERE+"/source_manifest.v1.json"),ref(HERE+"/check_prediction_barrier_v2.py"),
        ref(HERE+"/prepare_outer000_v1.py"),ref(HERE+"/protocol.v1.json"),
        ref(HERE+"/outer_prediction_validator_v2.py"),ref(HERE+"/outer_adapters_v2.py"),
        ref(HERE+"/h050_observed_bridge_v3.py"),ref(HERE+"/outer8_registry.v1.json"),
        ref(HERE+"/frozen_method_ids.v1.json"),source.NAMESPACE,source.V4,
        q["root_request_ref"],q["runtime_ref"],q["method_freeze_ref"],q["entry_review_ref"],
        q["runtime_qualification_ref"],q["bundle_ref"]]
    if stage=="score":
        refs.extend([ref(HERE+"/score_outer_v2.py"),q["scorer_review_ref"],q["actual_integration_qualification_ref"]])
    return refs

def admitted_v3(request_ref,stage):
    """Consume an existing real reservation; never draft/apply an admission."""
    from pathlib import Path
    import sys
    from . import prepare_outer000_v1 as source
    from .outer_prediction_validator_v2 import ROOT,HERE,need,record,ref,ObservedReaderV2,reviewed_source
    need(stage in ("validate","score") and Path.cwd()==ROOT and sys.dont_write_bytecode,"Explicit ROOT CPU consumer runtime")
    need(Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),"Qualified project interpreter")
    path=str(record(request_ref))
    leaf="validation" if stage=="validate" else "scoring"
    need(re.fullmatch(r"research/reviews/European_outer8_ROOT_actual_[A-Za-z0-9_-]+/"+leaf+r"/input\.json",path) is not None,
        "Canonical compact actual ROOT request path")
    q=source.read(request_ref);source.current(q)
    expected=("European_outer8_actual_validation_ROOT_request_v3" if stage=="validate" else "European_outer8_actual_scoring_ROOT_request_v2")
    need(q["schema"]==expected and q["status"]=="active" and q["root_authorizer"]=="root_agent"
        and q["entry_source_ref"]==ref(HERE+"/check_prediction_barrier_v2.py")
        and q["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json")
        and q["config_ref"]==ref(HERE+"/protocol.v1.json") and q["stage"]==stage
        and q["output_directory"]==path.rsplit("/",1)[0]+"/results"
        and q["storage_adapter_ref"]==source.GUARD
        and q["storage"]["completed_overlay_policy_ref"]==source.POLICY
        and type(q["output_cap_bytes"]) is int and 0<q["output_cap_bytes"]<=64*2**20,
        "Exact current ROOT consumer admission/source/config/output")
    need((stage=="validate" and q["observed_matrix_validation_authorized"] is True and q["outer_scoring_authorized"] is False)
        or (stage=="score" and q["outer_scoring_authorized"] is True),"Separate observed validation and target-scoring authority")
    intent=source.read(q["root_request_ref"]);record(q["root_request_ref"],path.rsplit("/",1)[0]+"/root_request.json")
    need(intent["schema"]=="European_outer8_CPU_consumer_ROOT_intent_v3" and intent["status"]=="active"
        and intent["root_authorizer"]=="root_agent" and all(intent[k]==q[k] for k in
        ("stage","entry_source_ref","source_manifest_ref","config_ref","runtime_ref","method_freeze_ref","entry_review_ref",
         "runtime_qualification_ref","bundle_ref","job_id","output_directory","output_cap_bytes","created_at_utc","expires_at_utc",
         "outer_scoring_authorized")),"Pre-reservation actual ROOT intent binds this input; no circular self-hash")
    runtime=source.read(q["runtime_ref"]);qualified=source.read(q["runtime_qualification_ref"])
    need(runtime["schema"]=="European_outer8_CPU_consumer_runtime_v3" and runtime["device"]=="cpu"
        and runtime["entry_source_ref"]==q["entry_source_ref"] and runtime["stage"]==stage
        and runtime["interpreter"]==str(ROOT/".runtime/env/bin/python")
        and qualified["schema"]=="European_outer8_CPU_consumer_runtime_qualification_v3"
        and qualified["root_authorizer"]=="root_agent" and qualified["status"]=="qualified"
        and qualified["unresolved_items"]==[] and qualified["runtime_ref"]==q["runtime_ref"]
        and qualified["source_manifest_ref"]==q["source_manifest_ref"],"Genuine source-bound CPU consumer/runtime qualification")
    refs=direct_refs_v3(q,request_ref,stage)
    need(all(r in q["storage"]["reservation"]["consumer_bindings"] for r in refs),"All actual consumer roles in real child reservation")
    g=source.guard(True)
    g.validate_storage_v9(q,dict(output_cap_bytes=q["output_cap_bytes"]),q["storage"],
        source_ref=q["entry_source_ref"],config_ref=q["config_ref"],runtime_ref=q["runtime_ref"])
    def guard(planned=0):
        source.current(q)
        need(type(planned) is int and 0<=planned<=q["output_cap_bytes"],"Finite admitted output write")
        return g.budget_check_v9(q["storage"],boundary=True,required_write_bytes=planned)
    reader=ObservedReaderV2(guard)
    reviewed_source(q["entry_source_ref"],q["entry_review_ref"],reader)
    manifest=reader.decode_json(q["source_manifest_ref"])
    for r in manifest["refs"]:reader.read(r,maximum=32*2**20)
    return q,reader,guard

def run_v3(request_ref,stage):
    from .outer_prediction_validator_v2 import PARENTS,METHODS,need,validate_matrix
    from .outer_adapters_v2 import _publish_json
    from . import prepare_outer000_v1 as source
    q,reader,guard=admitted_v3(request_ref,stage)
    if stage=="validate":
        bundle,parents,products=validate_matrix(q["bundle_ref"],reader)
        need(bundle["method_freeze_ref"]==q["method_freeze_ref"] and len(parents)==8
            and set(products)==set(METHODS)|{"ARGformer_top5_prespecified_secondary"},"Complete actual fixed matrix")
        result=dict(schema="European_outer8_observed_matrix_validation_v3",status="VALIDATED_ACTUAL_40_PRIMARY_8_TOP5",
            observed_matrix_validated=True,primary_cells=40,prespecified_top5_cells=8,biological_parents=len(PARENTS),
            method_ids=list(METHODS),bundle_ref=q["bundle_ref"],method_freeze_ref=q["method_freeze_ref"],
            source_manifest_ref=q["source_manifest_ref"],request_ref=request_ref,
            scoring_authorized=False,truth_read=False)
        del products
    else:
        from .score_outer_v2 import score_outer_v2
        result=score_outer_v2(request_ref,guard=guard)
    output=source.local(q["output_directory"])
    need(output.is_dir(),"ROOT-created actually reserved result directory")
    published=_publish_json(output/("validation.json" if stage=="validate" else "scores.json"),result,guard)
    return dict(status="ACTUAL_VALIDATION_PUBLISHED" if stage=="validate" else "ACTUAL_SCORES_PUBLISHED",
        result_ref=published,request_ref=request_ref,scoring_authorized=stage=="score")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-ref",help="JSON exact path/SHA of genuine ROOT observed validation request")
    modes.add_argument("--score-ref",help="JSON exact path/SHA of genuine separate ROOT scoring request")
    args=parser.parse_args(argv)
    if not args.validate_ref and not args.score_ref:
        print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY",actual_execution=False,observed_matrix_validated=False,scoring_authorized=False)))
        return 2
    stage="validate" if args.validate_ref else "score"
    result=run_v3(json.loads(args.validate_ref or args.score_ref),stage)
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0

if __name__=="__main__":raise SystemExit(main())
