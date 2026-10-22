"""Independent original AP 90-update fit with explicit full-device VRAM permission.

The historical fitter has no resume API and failed before a completed update.
Its qualified inputs, preparation barrier, numerical loop and checkpoint policy
are preserved; only GPU setup uses the owner's full-device memory permission.
"""
import copy
from . import full_vram_v2 as full_vram
from research.repairs.benchmark_recovery_20261020.training.candidate_common import (
    need, read_ref, file_ref,
)
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_stage as original
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_stage import (
    OrderedDict, os, INPUTS, SETTINGS, _validate, _observed_loader, _need,
    _c4_training_targets, _training_targets, fixed_training_loop, _validate_saved,
    generation, memory, install_chunked_gru, european, common,
)

MODULE = "research.repairs.benchmark_full_vram_20261022.training.ap_fit_v2"
SCHEMA = "european_full_device_fit_wrapper_v1"
ORIGINAL_JOB_REF = dict(
    path=".runtime/benchmarking/workflow_20261022_parallel_resource_overlay_v1/jobs/AP__all_train__treeloom_fit.json",
    sha256="ecf117aed7d0eebb5e482be544e10608f03e9c1cdbe9c690ffac9472c6ba81c2")
ORIGINAL_STAGE_REF = dict(
    path="research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_stage.py",
    sha256="357f193e16fb929955ea40954efccc23635b9f57f2b9b428a8eaa77725946a58")
ORIGINAL_COMMON_REF = dict(
    path="research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/common.py",
    sha256="430d24a4513ec4fe3b1555cfbf3263ac5ff024b1597767a5e48e4b419696aa77")


def source_refs():
    for ref in (ORIGINAL_JOB_REF, ORIGINAL_STAGE_REF, ORIGINAL_COMMON_REF):
        need(file_ref(ref["path"]) == ref, "Original AP fit source and job remain unchanged")
    return [ORIGINAL_JOB_REF, ORIGINAL_STAGE_REF, ORIGINAL_COMMON_REF,
            file_ref(__file__), *full_vram.source_refs()]


def _require_original(scientific, qualification_ref):
    historical = read_ref(ORIGINAL_JOB_REF)["call"]["spec"]
    need(historical["candidate_source_refs"] == common.source_refs(),
         "Every original qualified AP numerical and preparation source remains pinned")
    expected = copy.deepcopy(historical["scientific_spec"])
    for key in ("output_directory", "physical_gpu", "gpu_uuid"):
        expected[key] = scientific[key]
    need(scientific == expected and qualification_ref == historical["qualification_ref"],
         "All original AP inputs, science, budgets, storage and qualification must be preserved")
    return historical


def fit_factory(*, qualification_ref, **kwargs):
    need("resume_checkpoint_ref" not in kwargs, "The original AP fitter has no resume argument")
    call = original.fit_factory(qualification_ref=qualification_ref, **kwargs)
    wrapped = call["spec"]
    _require_original(wrapped["scientific_spec"], qualification_ref)
    scientific = wrapped["scientific_spec"]
    wrapped.update(schema=SCHEMA, gpu_memory_policy=full_vram.POLICY,
        full_device_vram_permission=full_vram.permission(scientific["physical_gpu"], scientific["gpu_uuid"]),
        predecessor_job_ref=ORIGINAL_JOB_REF, successor_source_refs=source_refs())
    refs = [*call["source_refs"], *source_refs()]
    return dict(module=MODULE, function="run_fit", spec=wrapped,
        python=call["python"], source_refs=list({ref["path"]:ref for ref in refs}.values()))


def run_fit(spec, *, resource_check, acquire_gpu0):
    need(set(spec) == {"schema", "scientific_spec", "qualification_ref", "candidate_source_refs",
            "gpu_memory_policy", "full_device_vram_permission", "predecessor_job_ref", "successor_source_refs"}
         and spec["schema"] == SCHEMA and spec["predecessor_job_ref"] == ORIGINAL_JOB_REF
         and spec["successor_source_refs"] == source_refs(), "Exact full-device AP successor wrapper required")
    scientific = spec["scientific_spec"]
    full_vram.validate_permission(dict(**scientific, gpu_memory_policy=spec["gpu_memory_policy"],
        full_device_vram_permission=spec["full_device_vram_permission"]))
    _require_original(scientific, spec["qualification_ref"])
    historical_wrapper = {key:spec[key] for key in (
        "scientific_spec", "qualification_ref", "candidate_source_refs")}
    historical_wrapper["schema"] = original.FIT_WRAPPER_SCHEMA
    original._validate_wrapper(historical_wrapper, original.FIT_WRAPPER_SCHEMA)
    return _run_scientific_fit(scientific, resource_check=resource_check, acquire_gpu0=acquire_gpu0,
        full_device_vram_permission=spec["full_device_vram_permission"])


def _run_scientific_fit(spec,*,resource_check,acquire_gpu0,full_device_vram_permission):
    plan = _validate(spec,fit=True)
    torch,check,writer = common._cpu_setup(spec,resource_check)
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    parents = tuple(plan["train_parent_ids"])
    cache,seals,events = OrderedDict(),{},[]
    load = _observed_loader(spec,cache,check,fit=True)
    # No target metadata/array is opened until every complete observed parent seals.
    for parent in parents:
        bundle,metadata = load(parent)
        seals[parent] = dict(parent_id=parent,**{key:spec["train_inputs"][parent][key] for key in INPUTS},
            focal_identity_sha256=bundle.focal_identity_sha256,native_holds=bundle.native_holds,
            all_native_trees_nodes_sites_samples_retained=True,private_truth_opened=False)
        del bundle,metadata
    barrier = dict(schema="european_treeloom_all_observation_barrier_v1",
        sampling_plan_ref=spec["sampling_plan_ref"],parents=seals,
        all_original_TRAIN_observations_sealed_before_targets=True,private_truth_opened=False)
    barrier_ref = writer.json("observation_barrier.json",barrier)
    _need(callable(acquire_gpu0), "Actual late assigned GPU acquisition callback required")
    acquire_gpu0()
    full_vram.enable_gpu(torch, dict(**spec, gpu_memory_policy=full_vram.POLICY,
        full_device_vram_permission=full_device_vram_permission), writer.path, check)
    model = install_chunked_gru(european.build_model(spec["protocol_id"]).to(device="cuda:0",dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8,maximize=False,amsgrad=False,foreach=False,fused=False)
    _need(not optimizer.state,"Fresh independent European optimizer required")
    def join(parent,bundle,metadata):
        _need(bundle.parent_id == parent and bundle.split == "train"
              and bundle.focal_identity_sha256 == seals[parent]["focal_identity_sha256"],
              "Original full observed parent seal required before targets")
        memory._admit_parent_headroom(cache,memory_envelope=spec["memory_envelope"],stage="training",
            rss_bytes=memory._current_rss_bytes,resource_check=check,event=lambda *args:None,keep_parent=parent)
        target_loader = _c4_training_targets if spec["protocol_id"] == "ancestralpaths_paper200mb_c4" else _training_targets
        return target_loader(spec["train_inputs"][parent]["training_targets_ref"],parent,bundle,model,
                                 max_metadata_bytes=spec["max_metadata_bytes"])
    previous = [None]
    def checkpoint(epoch,updates):
        payload = dict(schema="european_treeloom_full_checkpoint_v1",protocol_id=spec["protocol_id"],
            sampling_plan_ref=spec["sampling_plan_ref"],observation_barrier_ref=barrier_ref,
            epoch=epoch,optimizer_updates=updates,settings=copy.deepcopy(SETTINGS),
            class_order=list(model.output_classes),model_state=_cpu_snapshot(model.state_dict()),
            optimizer_state=_cpu_snapshot(optimizer.state_dict()),
            checkpoint_selection="fixed_final_epoch30",TEST_truth_read=False)
        path = writer.path/"checkpoints"/("epoch_%02d.pt"%epoch)
        path.parent.mkdir(parents=True,exist_ok=True)
        check("European_TreeLoom_checkpoint")
        with path.open("xb") as stream:
            torch.save(payload,stream)
            stream.flush();os.fsync(stream.fileno())
        _need(path.stat().st_size <= 8*2**20,"Bounded complete European model+optimizer checkpoint required")
        ref = generation._reference(path)
        if previous[0] is not None:
            old,old_ref = previous[0]
            _need(old.parent == path.parent and not old.is_symlink()
                  and generation._reference(old) == old_ref,"Rotate only exact prior new-scope checkpoint")
            old.unlink()
        previous[0] = (path,ref)
        return ref
    def record_event(value):
        events.append(value)
        writer.json("updates/update_%03d.json" % value["optimizer_updates"], value)
    result = fixed_training_loop(parents=parents,load_parent=load,join_targets=join,
        step=lambda bundle,targets,metadata:european.train_sampled_step(
            model,bundle,targets,optimizer,metadata,resource_check=check),
        optimizer_counts=lambda:memory._optimizer_counts(optimizer),checkpoint=checkpoint,
        check=check,event=record_event)
    check("European_TreeLoom_verify_final_saved_state")
    saved = torch.load(generation._path(result["checkpoint_ref"]["path"]),
                       map_location="cpu",weights_only=True,mmap=True)
    cpu_model = european.build_model(spec["protocol_id"])
    _validate_saved(saved,cpu_model,protocol_id=spec["protocol_id"],sampling_plan_ref=spec["sampling_plan_ref"],
                    updates=result["optimizer_updates"],barrier_ref=barrier_ref)
    del saved,cpu_model
    writer.json("progress.json",dict(events=events))
    torch.cuda.synchronize(0)
    result.update(schema="european_treeloom_fit_completion_v1",status="completed_full_parent_sampled_fit",
        protocol_id=spec["protocol_id"],sampling_plan_ref=spec["sampling_plan_ref"],
        observation_barrier_ref=barrier_ref,class_order=list(model.output_classes),
        checkpoint_selection="fixed_final_epoch30",TEST_truth_read=False,
        actual_worker_wait_and_owned_cleanup_still_required=True)
    completion_ref = writer.json("training_completed.json",result)
    _need(not writer.pending,"Every European fit artifact must actually finish")
    return dict(result,completion_ref=completion_ref)
