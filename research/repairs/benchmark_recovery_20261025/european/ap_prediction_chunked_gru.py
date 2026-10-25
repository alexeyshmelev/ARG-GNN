"""Prediction-only AP repair: reuse final90-update fit with bounded GRU calls.

The saved model, native observations, sampled row identities, complete tree/
history execution, class order, GPU3/4/5 binding and product schemas are kept.
Only the new prediction model receives the same recurrent execution adapter
already used by the completed AP fit. Original source files are not modified.
"""
from collections import OrderedDict
from arggnn.benchmarking.paper_argmix import paper_argmix_ap_prediction as baseline
from arggnn.benchmarking.paper_argmix.paper_argmix_ap_prediction import (
    g,c,original,european,memory,predict_logits)
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_gru

MODULE = "research.repairs.benchmark_recovery_20261025.european.ap_prediction_chunked_gru"
SCHEMA = "AP_native6_chunked_GRU_prediction_repair_v1"
PROTOCOL = baseline.PROTOCOL
GRU_CHUNK_HOLDS = 1024


def requirements():
    prior = baseline.requirements()
    sources = [*prior["source_refs"],c.file_ref(__file__),c.file_ref(european_gru.__file__)]
    return dict(baseline=prior,source_refs=list({r["path"]:r for r in sources}.values()),
        recurrent_execution="same_full_length_bidirectional_GRU_with_connected_chunk_carry",
        recurrent_chunk_holds=GRU_CHUNK_HOLDS,training_repeated=False)


def install_model_execution(model):
    return european_gru.install_chunked_gru(model,chunk_holds=GRU_CHUNK_HOLDS)


def build_spec(**kwargs):
    spec = baseline.build_spec(**kwargs)
    spec.update(schema=SCHEMA,requirements=requirements())
    return spec


def run_predictions(spec,*,resource_check):
    g.need(spec["schema"]==SCHEMA and spec["requirements"]==requirements()
        and spec["prediction_spec"]["protocol_id"]==PROTOCOL,
        "Exact source-pinned AP prediction-only repair required")
    prediction = spec["prediction_spec"]
    plan = original._validate(prediction,fit=False)
    torch,check,writer,binding = baseline.setup(spec,resource_check)
    complete = original._fit_completion(prediction["fit_completion_ref"],
        protocol_id=PROTOCOL,sampling_plan_ref=prediction["sampling_plan_ref"],
        train_count=len(plan["train_parent_ids"]))
    barrier = c.read_ref(complete["observation_barrier_ref"])
    g.need(barrier.get("sampling_plan_ref")==prediction["sampling_plan_ref"]
        and set(barrier.get("parents",{}))==set(plan["train_parent_ids"])
        and barrier.get("all_original_TRAIN_observations_sealed_before_targets") is True
        and barrier.get("private_truth_opened") is False,
        "All original TRAIN observations must precede the saved fit")
    path = european._bound(complete["checkpoint_ref"])
    g.need(path.stat().st_size<=8*2**20,"Finite complete final checkpoint required")
    saved = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    model = european.build_model(PROTOCOL)
    original._validate_saved(saved,model,protocol_id=PROTOCOL,
        sampling_plan_ref=prediction["sampling_plan_ref"],
        updates=30*len(plan["train_parent_ids"]),
        barrier_ref=complete["observation_barrier_ref"])
    del saved
    model = model.to(device="cuda:0",dtype=torch.float32)
    model = install_model_execution(model)
    cache,predictions = OrderedDict(),{}
    load = original._observed_loader(prediction,cache,check,fit=False)
    for parent in plan["test_parent_ids"]:
        if parent not in prediction["heldout_inputs"]:
            continue
        bundle,metadata = load(parent)
        memory._admit_parent_headroom(cache,memory_envelope=prediction["memory_envelope"],
            stage="prediction",rss_bytes=memory._current_rss_bytes,
            resource_check=check,event=lambda *args:None,keep_parent=parent)
        logits = predict_logits(model,bundle,metadata,resource_check=check,
            gpu_uuid=binding["gpu_uuid"],gpu_uuid_map=spec["gpu_uuid_map"])
        n,classes = bundle.focal_hold.numel(),len(model.output_classes)
        g.need(logits.shape==(n,classes) and logits.dtype==torch.float32
            and torch.isfinite(logits).all().item(),
            "Every original registered held-out logit required")
        probabilities = logits.softmax(-1).to("cpu").numpy()
        item = prediction["heldout_inputs"][parent]
        records = european._sampled_records(item["sampled_registry_ref"],
            observation_ref=item["inference_ref"],parent_id=parent,split="test",
            max_metadata_bytes=prediction["max_metadata_bytes"])
        prediction_ref = writer.npz("predictions/"+parent+".npz",
            example_indices=records["record_ids"],probabilities=probabilities)
        product = dict(parent_id=parent,split="test",
            registry_ref=item["sampled_registry_ref"],predictions_ref=prediction_ref,
            rows=n,native_path_names=list(model.output_classes),
            schema="european_treeloom_prediction_product_v1",method="treeloom",
            **item,checkpoint_ref=complete["checkpoint_ref"],
            fit_completion_ref=prediction["fit_completion_ref"],
            focal_identity_sha256=bundle.focal_identity_sha256,
            query_ancestry_truth_opened=False,
            all_native_trees_nodes_sites_samples_retained=True,
            static_executor_source_ref=c.file_ref(
                "src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_executor.py"),
            recurrent_execution_source_ref=c.file_ref(european_gru.__file__),
            recurrent_chunk_holds=GRU_CHUNK_HOLDS,
            prediction_wrapper_source_ref=c.file_ref(__file__),
            campaign_gpu_binding=binding)
        predictions[parent] = writer.json("products/"+parent+".json",product)
        del logits,probabilities,records,bundle,metadata
        check("AP_chunked_GRU_heldout_prediction_sealed")
    torch.cuda.synchronize(0)
    result = dict(schema="european_treeloom_prediction_completion_v1",
        status="completed_all_requested_sampled_predictions",protocol_id=PROTOCOL,
        sampling_plan_ref=prediction["sampling_plan_ref"],
        fit_completion_ref=prediction["fit_completion_ref"],predictions=predictions,
        query_ancestry_truth_opened=False,
        actual_worker_wait_and_owned_cleanup_still_required=True,
        campaign_gpu_binding=binding,static_executor=True,
        recurrent_execution_source_ref=c.file_ref(european_gru.__file__),
        recurrent_chunk_holds=GRU_CHUNK_HOLDS,training_repeated=False)
    ref = writer.json("predictions_completed.json",result)
    g.need(not writer.pending and set(predictions)==set(prediction["heldout_inputs"]),
        "Every requested AP parent must finish")
    return dict(result,completion_ref=ref)


def prediction_factory(**kwargs):
    spec = build_spec(**kwargs)
    return dict(module=MODULE,function="run_predictions",spec=spec,
        python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])
