"""Prediction-only execution adapter with exact parsing and bounded lookahead.

The trained model, recurrent execution, sampled targets, logits, softmax and
product construction remain the original qualified prediction implementation.
Only our observed preprocessing and its scheduling change. No author module or
currently running worker is patched.
"""
from collections import OrderedDict
from arggnn.benchmarking.paper_argmix import paper_argmix_ap_prediction as baseline
from arggnn.benchmarking.paper_argmix.paper_argmix_ap_prediction import (
    g,c,original,european,memory,predict_logits)
from research.repairs.benchmark_recovery_20261025.european import ap_prediction_chunked_gru as previous
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_gru
from . import adapter, parser, preparation, prefetch
from .prefetch import OneParentLookahead, SerializedCheck
from ..tree import observations, episodes, cut, kernels

MODULE = "research.preprocessing_optimization_20261026.observed.prediction"
SCHEMA = "AP_native6_exact_preprocessing_lookahead_prediction_v1"
PROTOCOL = previous.PROTOCOL
GRU_CHUNK_HOLDS = previous.GRU_CHUNK_HOLDS
install_model_execution = previous.install_model_execution


def requirements():
    prior = previous.requirements()
    sources = [*prior["source_refs"], *[c.file_ref(module.__file__) for module in
        (adapter,parser,preparation,prefetch,observations,episodes,cut,kernels)],
        c.file_ref("research/preprocessing_optimization_20261026/tree/kernels.cpp"),
        c.file_ref("research/preprocessing_optimization_20261026/tree/libstructural.so"),
        c.file_ref(__file__)]
    return dict(previous=prior,source_refs=list({r["path"]:r for r in sources}.values()),
        observed_parser="exact_vectorized_with_original_scalar_fallback",
        tree_preparation="qualified_serial_compiled_structural_kernels",
        next_parent_prefetch_limit=1,completed_parent_cache_bytes=0,
        scientific_model_and_prediction_algebra_unchanged=True,
        native_competitor_source_modified=False,training_repeated=False)


def _validate_execution_options(spec):
    envelope = spec["prediction_spec"]["memory_envelope"]
    g.need(type(spec["prefetch_next_parent"]) is bool,
           "Explicit next-parent prefetch setting required")
    g.need(spec["prediction_spec"]["observation_cache_bytes"] == 0,
           "This one-pass prediction route retains no completed-parent cache")
    g.need(envelope["max_rss_bytes"] <= 512*2**30
           and envelope["preparation_headroom_bytes"] >= 128*2**30
           and envelope["prediction_headroom_bytes"] >= 64*2**30,
           "Qualified complete-parent preparation/prediction reserves within512GiB required")


def combined_headroom_available(current, envelope):
    return current + envelope["preparation_headroom_bytes"] + envelope["prediction_headroom_bytes"] <= envelope["max_rss_bytes"]


def build_spec(*,prefetch_next_parent=True,**kwargs):
    spec = previous.build_spec(**kwargs)
    spec.update(schema=SCHEMA,requirements=requirements(),
                prefetch_next_parent=prefetch_next_parent)
    _validate_execution_options(spec)
    return spec


def _load_observed_parent(spec,parent,check,torch):
    item = spec["heldout_inputs"][parent]
    def prepare():
        state = adapter.load_relate_observation(
            item["inference_ref"],item["observed_vcf_ref"],protocol_id=spec["protocol_id"],
            max_sites=spec["max_sites"],max_metadata_bytes=spec["max_metadata_bytes"],resource_check=check)
        return preparation.prepare_observation(
            state,protocol_id=spec["protocol_id"],parent_id=parent,split="test",
            source_refs=state["source_refs"],sampled_registry_ref=item["sampled_registry_ref"],
            resource_check=check,max_sampled_metadata_bytes=spec["max_metadata_bytes"],
            **spec["preparation_limits"])
    # A separate empty cache avoids shared mutable cache state between the
    # prediction and its single background producer. No previous bank is kept.
    return memory._load_cached_parent(parent,OrderedDict(),prepare,
        observation_cache_bytes=0,memory_envelope=spec["memory_envelope"],
        rss_bytes=memory._current_rss_bytes,tensor_bytes=lambda bundle:memory._cpu_bytes(bundle,torch),
        resource_check=check,event=lambda *args:None)


def run_predictions(spec,*,resource_check):
    g.need(spec["schema"]==SCHEMA and spec["requirements"]==requirements()
        and spec["prediction_spec"]["protocol_id"]==PROTOCOL,
        "Exact source-pinned AP prediction-only repair required")
    prediction = spec["prediction_spec"]
    plan = original._validate(prediction,fit=False)
    _validate_execution_options(spec)
    serialized = SerializedCheck(resource_check)
    torch,check,writer,binding = baseline.setup(spec,serialized)
    check = SerializedCheck(check, serialized.lock)
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
    parents = tuple(parent for parent in plan["test_parent_ids"]
                    if parent in prediction["heldout_inputs"])
    def load(parent, callback):
        return _load_observed_parent(prediction, parent, callback, torch)
    def admit_overlap():
        if not spec["prefetch_next_parent"]:
            return False
        check("TreeLoom_prefetch_combined_RSS_admission")
        envelope = prediction["memory_envelope"]
        current = memory._current_rss_bytes()
        g.need(type(current) is int and current >= 0, "Actual current RSS required")
        return combined_headroom_available(current, envelope)
    with OneParentLookahead(parents, load, resource_check=check,
                            admit_overlap=admit_overlap) as lookahead:
        for parent in parents:
            bundle,metadata = lookahead.take(parent)
            memory._admit_parent_headroom(cache,memory_envelope=prediction["memory_envelope"],
                stage="prediction",rss_bytes=memory._current_rss_bytes,
                resource_check=check,event=lambda *args:None,keep_parent=parent)
            lookahead.start_next()
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
