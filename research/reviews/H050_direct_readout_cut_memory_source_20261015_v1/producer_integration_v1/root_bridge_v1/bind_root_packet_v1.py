"""Declarative root-only metadata binding. No guard, admission, lease or launch calls."""
import argparse,json
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1.bridge_common_v1 import *
SOURCE=BINDER
BOUND=HERE+"/bound_root_v1"


def derive_genuine_preparation_v1(reader,actual_wait_ref):
    """Read successful terminal metadata and derive exact refs without directory scans."""
    wait=reader.decode_json(actual_wait_ref)
    require(wait.get("actual_direct_launcher_wait") is True and wait.get("returncode")==0
        and wait.get("owned_group_drained") is True and wait.get("error") is None
        and is_ref_v1(wait.get("completed_ref")) and is_ref_v1(wait.get("launch_ref")),"Actual successful preparation wait0/drain")
    completed=reader.decode_json(wait["completed_ref"])
    require(wait["completed_ref"]["path"]=="experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5/completed.json","Actual completed v5 output only")
    inp=reader.decode_json(completed["input_ref"]);prep_cfg=reader.decode_json(completed["config_ref"])
    admission_path=str(Path(wait["launch_ref"]["path"]).parent/"admitted.json")
    return dict(cut_preparation_completion_ref=wait["completed_ref"],cut_preparation_actual_wait_ref=actual_wait_ref,
        cut_preparation_admission_ref=ref(admission_path),cut_preparation_input_ref=completed["input_ref"],
        cut_preparation_config_ref=completed["config_ref"],cut_preparation_runtime_ref=inp["refs"]["runtime"],
        cut_preparation_launch_ref=wait["launch_ref"],cut_preparation_registration_ref=prep_cfg["registration_ref"],
        preparation_producer_ref=completed["producer_ref"],cut_preparation_codec_ref=completed["codec_ref"],
        cut_preparation_source_manifest_ref=completed["source_manifest_ref"])


def bind_root_packet_v1(request_ref):
    reader=MetadataReaderV1();r=reader.decode_json(request_ref)
    require(r.get("schema")=="H050_root_metadata_activation_request_v1" and r.get("status")=="active"
        and r.get("root_authorizer")=="root_agent" and r.get("authorize_metadata_activation") is True,"Explicit root authorization to bind active H050 metadata")
    require(r["inactive_config_ref"]["path"]==HERE+"/config.proposed.inactive.json"
        and r["bridge_source_manifest_ref"]["path"]==HERE+"/source_manifest.bridge.v1.json","Exact reviewed inactive bridge packet")
    cfg=reader.decode_json(r["inactive_config_ref"]);worker.validate_protocol(cfg,active=False)
    frozen=reader.decode_json(r["bridge_source_manifest_ref"])
    require(frozen.get("schema")=="H050_root_bridge_exact_source_manifest_v1" and frozen.get("hypothesis_id")=="H050","Reviewed bridge source closure")
    for path,sha in frozen["source_sha256"].items():reader.read(dict(path=path,sha256=sha))
    require(all(frozen["source_sha256"].get(p)==ref(p)["sha256"] for p in (SOURCE,DRIVER,CONTEXT,HERE+"/bridge_common_v1.py",worker.SOURCE)),"Exact reviewed bridge and producer bytes")
    caps=r.get("resource_caps",{});require(set(caps)==set(worker.RESOURCE_CAPS+EXTRA_CAPS)
        and all(type(v) is int and v>0 for v in caps.values()),"Only complete explicit root-reviewed H050 operational cap set")
    require(is_ref_v1(r.get("root_resource_review_ref")),"Root's actual cap review memo ref")
    reader.read(r["root_resource_review_ref"])
    actual=derive_genuine_preparation_v1(reader,r["preparation_actual_wait_ref"])
    require(actual["cut_preparation_registration_ref"]==cfg["cut_preparation_registration_ref"],"Unchanged frozen H049 preparation registration")
    cfg.update(caps);cfg.update(actual)
    cfg.update(execution_enabled=True,root_resource_caps_reviewed=True,registration_ref=r["registration_ref"],
        status="root_bound_active_metadata_not_admitted",future_completion_not_asserted=False,
        root_resource_review_ref=r["root_resource_review_ref"],canonical_dispatcher_ref=ref(worker.SOURCE),
        root_context_module_ref=ref(CONTEXT),root_driver_ref=ref(DRIVER))
    validate_config_v1(cfg,reader)
    prep_manifest=reader.decode_json(actual["cut_preparation_source_manifest_ref"])
    sources=dict(frozen["source_sha256"])
    for path,sha in prep_manifest["source_sha256"].items():
        require(path not in sources or sources[path]==sha,"Same exact source bytes across experiment and actual preparation")
        sources[path]=sha;reader.read(dict(path=path,sha256=sha))
    original=reader.decode_json(cfg["original_H038_input_ref"]);prep_spec=preparation_spec_v1(cfg,sources,original)
    marker=framework.completed_marker_preparation_v1(reader,prep_spec)
    authenticate_actual_cut_preparation_v1(reader,prep_spec,cfg)
    counts=reader.decode_json(cfg["count_manifest_ref"])
    for phase in ("train","evaluate"):
        require(type(r.get("reserved_bytes_"+phase)) is int,"Explicit whole-scope reservation bytes for "+phase)
        validate_plan_caps_v1(count_plan_v1(counts,phase,marker,cfg),cfg,r["reserved_bytes_"+phase])
    gpu=r.get("physical_gpu_train");require(type(gpu) is int and gpu in (4,5),"Root-selected TRAIN GPU4/5")
    folder=ROOT/BOUND;require(not folder.exists(),"Fresh metadata binding; no overwrite/replay");folder.mkdir()
    metadata=dict(frozen.get("metadata_refs",{}),registration=cfg["registration_ref"],
        preparation_registration=cfg["cut_preparation_registration_ref"],root_resource_review=r["root_resource_review_ref"],
        actual_preparation_completion=cfg["cut_preparation_completion_ref"],actual_preparation_wait=cfg["cut_preparation_actual_wait_ref"],
        actual_preparation_source_manifest=cfg["cut_preparation_source_manifest_ref"])
    manifest=save(folder,"source_manifest.root_bound.json",dict(schema="H050_root_bridge_exact_source_manifest_v1",
        hypothesis_id="H050",source_sha256=sources,metadata_refs=metadata,
        bridge_source_manifest_ref=r["bridge_source_manifest_ref"],actual_preparation_source_manifest_ref=cfg["cut_preparation_source_manifest_ref"],
        legacy_operational_receipt_schemas=list(LEGACY_RECEIPT_SCHEMAS)))
    cfg["source_manifest_ref"]=manifest
    config_ref=save(folder,"config_train_evaluate.root_active.json",cfg)
    exact_sources_v1(reader,cfg)
    train_request=save(folder,"admission_request.train.root_active.json",dict(schema="H050_root_admission_request_v1",status="active",
        root_authorizer="root_agent",phase="train",config_ref=config_ref,physical_gpu=gpu,reserved_bytes=r["reserved_bytes_train"],
        train_completion_ref=None,train_launch_ref=None,train_actual_wait_ref=None))
    pending_eval=save(folder,"admission_request.evaluate.pending_train.json",dict(schema="H050_root_admission_request_v1",status="inactive_pending_successful_H050_TRAIN",
        root_authorizer="root_agent",phase="evaluate",config_ref=config_ref,physical_gpu=None,reserved_bytes=r["reserved_bytes_evaluate"],
        train_completion_ref=None,train_launch_ref=None,train_actual_wait_ref=None))
    receipt=save(folder,"metadata_binding.json",dict(schema="H050_root_metadata_binding_v1",request_ref=request_ref,
        config_ref=config_ref,source_manifest_ref=manifest,train_request_ref=train_request,pending_evaluate_request_ref=pending_eval,
        actual_preparation_success_authenticated=True,all40_recipe_metadata_authenticated=True,
        guard_called=False,admitted=False,launched=False,CUDA_initialized=False))
    result=dict(metadata_binding_ref=receipt,config_ref=config_ref,train_request_ref=train_request,pending_evaluate_request_ref=pending_eval)
    print(json.dumps(result),flush=True);return result


def bind_evaluate_v1(request_ref):
    reader=MetadataReaderV1();r=reader.decode_json(request_ref)
    require(r.get("schema")=="H050_root_evaluation_binding_request_v1" and r.get("status")=="active"
        and r.get("root_authorizer")=="root_agent","Explicit root evaluation metadata binding")
    cfg=reader.decode_json(r["config_ref"]);validate_config_v1(cfg,reader)
    require(r["config_ref"]["path"]==BOUND+"/config_train_evaluate.root_active.json","Same immutable TRAIN/EVAL config")
    wait=reader.decode_json(r["train_actual_wait_ref"])
    spec=dict(phase="evaluate",train_actual_wait_ref=r["train_actual_wait_ref"],
        train_launch_ref=wait.get("launch_ref"),train_completion_ref=wait.get("completed_ref"))
    validate_train_lineage_v1(reader,spec,cfg)
    require(type(r.get("physical_gpu")) is int and r["physical_gpu"] in (4,5),"Root-selected EVALUATE GPU4/5")
    require(type(r.get("reserved_bytes")) is int and r["reserved_bytes"]>0,"Explicit positive EVALUATE whole-scope reservation bytes")
    marker=reader.decode_json(cfg["preparation_completion_ref"])
    validate_plan_caps_v1(count_plan_v1(reader.decode_json(cfg["count_manifest_ref"]),"evaluate",marker,cfg),cfg,r.get("reserved_bytes"))
    result=save(ROOT/BOUND,"admission_request.evaluate.root_active.json",dict(schema="H050_root_admission_request_v1",status="active",
        root_authorizer="root_agent",config_ref=r["config_ref"],physical_gpu=r["physical_gpu"],reserved_bytes=r["reserved_bytes"],**spec))
    print(json.dumps(dict(evaluate_request_ref=result)),flush=True);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("describe","bind","bind-evaluate"),nargs="?",default="describe")
    p.add_argument("--execute",action="store_true");p.add_argument("--record");p.add_argument("--sha256");args=p.parse_args(argv)
    if args.action=="describe":print(json.dumps(dict(status="inactive_source_only",hypothesis_id="H050",metadata_only=True,no_guard_or_launch=True)));return 0
    require(args.execute and args.record and args.sha256,"Explicit root metadata-binding execution and exact pinned request")
    record=dict(path=args.record,sha256=args.sha256)
    if args.action=="bind":bind_root_packet_v1(record)
    else:bind_evaluate_v1(record)
    return 0

if __name__=="__main__":raise SystemExit(main())
