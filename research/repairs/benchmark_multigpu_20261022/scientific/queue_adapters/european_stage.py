"""Full European30-pass TreeLoom fit and sampled-support held-out prediction.

ROOT's workflow owns the assigned GPU lease, process/RSS/storage/wall supervision and actual
waits. No launcher or policy is installed. Native TRAIN labels stay separate from
observed bank preparation; held-out callables accept no truth/target reference.
"""
from __future__ import annotations
from collections import OrderedDict
import copy
import json
import os
import random
import zipfile

from arggnn.benchmarking import contracts as c
from . import european_observations as european
from . import common
from .european_gru import install_chunked_gru
from arggnn.benchmarking import treeloom_training as memory
from arggnn.benchmarking import treeloom_prediction as gpu
from arggnn.benchmarking import american_preparation as generation
from arggnn.benchmarking import european_sampling as sampling

FIT_SCHEMA = "european_treeloom_full_fit_spec_v1"
PREDICT_SCHEMA = "european_treeloom_prediction_spec_v1"
INPUTS = ("inference_ref","observed_vcf_ref","sampled_registry_ref")
SETTINGS = dict(epochs=30,model_seed=101,parent_order_seed=101,
    precision="fp32",learning_rate=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8,
    gradient_clip_norm=1.,checkpoint_epochs=[5,10,15,20,25,30],
    checkpoint_selection="fixed_final_epoch30",validation_used_for_optimization=False,
    loss="mean_cross_entropy_per_registered_sampled_example_preserving_repeats",
    history_holds_per_window=128,TEST_truth_access=False)


def _need(ok,message):
    if not ok:
        raise c.ContractError(message)


def requirements():
    result = european.requirements()
    return dict(result,source_refs=[*result["source_refs"],c.file_ref(__file__),
        c.file_ref(memory.__file__),c.file_ref(sampling.__file__)],
        settings=copy.deepcopy(SETTINGS),checkpoint_peak_files=2,
        max_single_checkpoint_bytes=8*2**20,query_truth_in_observations=False)


def _parent_plan(ref,protocol_id):
    gpu._public_ref(ref)
    plan = c.read_ref(ref)
    _need(protocol_id in european.PROTOCOLS, "Supported explicit European task required")
    train,test = plan.get("train_parent_ids"),plan.get("test_parent_ids")
    _need(isinstance(train,list) and isinstance(test,list) and train and test
          and len(set(train+test)) == len(train)+len(test),
          "Distinct independent TRAIN/held-out parent identities required")
    for parent in train+test:
        import re
        _need(isinstance(parent,str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",parent),
              "Safe immutable parent identity required")
    if protocol_id in ("ancestralpaths_paper200mb_native6", "ancestralpaths_paper200mb_c4"):
        _need(plan.get("schema") == "ap_paper_sampled_plan_v1"
              and len(train) == 3 and len(test) == 5
              and plan.get("native_path_names") == list(european.NATIVE_SIX_CLASSES)
              and plan.get("training_total") == 50000 and plan.get("testing_per_parent") == 10000,
              "Exact threeTRAIN/fiveTEST paper-size native6 sampled plan required")
    else:
        _need(plan.get("schema") == "european_treeloom_parent_plan_v1"
              and plan.get("protocol_id") == protocol_id
              and plan.get("class_order") == ["Ana","CHG","WHG","EHG"],
              "Separately frozen four-class parent/support plan required")
    return plan


def _input_shapes(inputs,parents,*,training):
    expected = {*INPUTS,"training_targets_ref"} if training else set(INPUTS)
    _need(type(inputs) is dict and inputs and (set(inputs) == set(parents) if training
          else set(inputs) <= set(parents)), "Only registered actual prepared parents required")
    for item in inputs.values():
        _need(type(item) is dict and set(item) == expected,
              "Separate TRAIN target reference; held-out inputs accept observations only")
        for name,ref in item.items():
            # Targets contain already authorized TRAIN labels; observed inputs and
            # all held-out refs remain public and cannot carry private truth.
            if name == "training_targets_ref":
                memory.fresh._ref(ref)
            else:
                gpu._public_ref(ref)


def _limits(*,max_sites,max_metadata_bytes,preparation_limits,max_output_bytes,parents):
    _need(type(max_sites) is int and max_sites > 0
          and type(max_metadata_bytes) is int and max_metadata_bytes > 0
          and type(preparation_limits) is dict
          and set(preparation_limits) == {"max_nodes","max_holds","max_episode_payload_bytes"}
          and all(type(v) is int and v > 0 for v in preparation_limits.values())
          and type(max_output_bytes) is int and max_output_bytes >= 4*2**20+parents*2**20,
          "Finite complete-parent site/metadata/preparation/output limits required")
    return dict(max_sites=max_sites,max_metadata_bytes=max_metadata_bytes,
                preparation_limits=copy.deepcopy(preparation_limits),max_output_bytes=max_output_bytes)


def build_fit_spec(*,protocol_id,sampling_plan_ref,train_inputs,output_directory,max_rss_bytes,
                   preparation_headroom_bytes,training_headroom_bytes,observation_cache_bytes,
                   max_sites,max_metadata_bytes,preparation_limits,physical_gpu,gpu_uuid,max_output_bytes=64*2**20):
    plan = _parent_plan(sampling_plan_ref,protocol_id)
    _input_shapes(train_inputs,plan["train_parent_ids"],training=True)
    out = generation._path(output_directory)
    _need(out.is_relative_to(c.ROOT/"experiments/results") and not out.exists(),"Fresh dedicated fit scope required")
    limits = _limits(max_sites=max_sites,max_metadata_bytes=max_metadata_bytes,
                     preparation_limits=preparation_limits,max_output_bytes=max_output_bytes,
                     parents=len(train_inputs))
    envelope = memory._memory_envelope(max_rss_bytes=max_rss_bytes,
        preparation_headroom_bytes=preparation_headroom_bytes,
        training_headroom_bytes=training_headroom_bytes,observation_cache_bytes=observation_cache_bytes)
    return dict(schema=FIT_SCHEMA,protocol_id=protocol_id,sampling_plan_ref=copy.deepcopy(sampling_plan_ref),
        train_inputs=copy.deepcopy(train_inputs),output_directory=str(out.relative_to(c.ROOT)),
        memory_envelope=envelope,observation_cache_bytes=observation_cache_bytes,**limits,
        settings=copy.deepcopy(SETTINGS),optimizer_updates=30*len(train_inputs),
        requirements=requirements(),private_truth_in_observations=False,
        **common.gpu_binding.target(physical_gpu,gpu_uuid))


def _fit_completion(ref,*,protocol_id,sampling_plan_ref,train_count):
    gpu._public_ref(ref)
    complete = c.read_ref(ref)
    _need(complete.get("schema") == "european_treeloom_fit_completion_v1"
          and complete.get("status") == "completed_full_parent_sampled_fit"
          and complete.get("protocol_id") == protocol_id
          and complete.get("sampling_plan_ref") == sampling_plan_ref
          and complete.get("completed_epochs") == complete.get("selected_epoch") == 30
          and complete.get("optimizer_updates") == 30*train_count
          and complete.get("checkpoint_selection") == "fixed_final_epoch30"
          and complete.get("TEST_truth_read") is False,
          "Genuine same-plan full30-pass final European fit completion required")
    return complete


def build_prediction_spec(*,protocol_id,sampling_plan_ref,fit_completion_ref,heldout_inputs,
                          output_directory,max_rss_bytes,preparation_headroom_bytes,
                          prediction_headroom_bytes,observation_cache_bytes,max_sites,
                          max_metadata_bytes,preparation_limits,physical_gpu,gpu_uuid,max_output_bytes=64*2**20):
    plan = _parent_plan(sampling_plan_ref,protocol_id)
    _input_shapes(heldout_inputs,plan["test_parent_ids"],training=False)
    _fit_completion(fit_completion_ref,protocol_id=protocol_id,
                    sampling_plan_ref=sampling_plan_ref,train_count=len(plan["train_parent_ids"]))
    out = generation._path(output_directory)
    _need(out.is_relative_to(c.ROOT/"experiments/results") and not out.exists(),"Fresh held-out prediction scope required")
    limits = _limits(max_sites=max_sites,max_metadata_bytes=max_metadata_bytes,
        preparation_limits=preparation_limits,max_output_bytes=max_output_bytes,parents=len(heldout_inputs))
    envelope = memory._memory_envelope(max_rss_bytes=max_rss_bytes,
        preparation_headroom_bytes=preparation_headroom_bytes,
        training_headroom_bytes=prediction_headroom_bytes,observation_cache_bytes=observation_cache_bytes)
    envelope["prediction_headroom_bytes"] = envelope.pop("training_headroom_bytes")
    return dict(schema=PREDICT_SCHEMA,protocol_id=protocol_id,
        sampling_plan_ref=copy.deepcopy(sampling_plan_ref),fit_completion_ref=copy.deepcopy(fit_completion_ref),
        heldout_inputs=copy.deepcopy(heldout_inputs),output_directory=str(out.relative_to(c.ROOT)),
        memory_envelope=envelope,observation_cache_bytes=observation_cache_bytes,**limits,
        requirements=requirements(),query_ancestry_truth_opened=False,
        checkpoint_selection="fixed_final_epoch30",**common.gpu_binding.target(physical_gpu,gpu_uuid))


def _validate(spec,*,fit):
    keys = ("protocol_id","sampling_plan_ref","output_directory","observation_cache_bytes",
            "max_sites","max_metadata_bytes","preparation_limits","max_output_bytes","physical_gpu","gpu_uuid")
    kwargs = {key:spec[key] for key in keys}
    envelope = spec["memory_envelope"]
    kwargs.update(max_rss_bytes=envelope["max_rss_bytes"],
                  preparation_headroom_bytes=envelope["preparation_headroom_bytes"])
    if fit:
        kwargs.update(train_inputs=spec["train_inputs"],
                      training_headroom_bytes=envelope["training_headroom_bytes"])
        expected = build_fit_spec(**kwargs)
    else:
        kwargs.update(fit_completion_ref=spec["fit_completion_ref"],heldout_inputs=spec["heldout_inputs"],
                      prediction_headroom_bytes=envelope["prediction_headroom_bytes"])
        expected = build_prediction_spec(**kwargs)
    _need(expected == spec,"Exact frozen European numerical stage/source specification required")
    return _parent_plan(spec["sampling_plan_ref"],spec["protocol_id"])


def _observed_loader(spec,cache,check,*,fit):
    inputs = spec["train_inputs"] if fit else spec["heldout_inputs"]
    def load(parent):
        item = inputs[parent]
        def prepare():
            state = european.load_relate_observation(
                item["inference_ref"],item["observed_vcf_ref"],protocol_id=spec["protocol_id"],
                max_sites=spec["max_sites"],max_metadata_bytes=spec["max_metadata_bytes"],resource_check=check)
            return european.prepare_observation(
                state,protocol_id=spec["protocol_id"],parent_id=parent,split="train" if fit else "test",
                source_refs=state["source_refs"],sampled_registry_ref=item["sampled_registry_ref"],
                resource_check=check,max_sampled_metadata_bytes=spec["max_metadata_bytes"],
                **spec["preparation_limits"])
        return memory._load_cached_parent(parent,cache,prepare,
            observation_cache_bytes=spec["observation_cache_bytes"],memory_envelope=spec["memory_envelope"],
            rss_bytes=memory._current_rss_bytes,tensor_bytes=lambda bundle:memory._cpu_bytes(bundle,torch_module()),
            resource_check=check,event=lambda *args:None)
    return load


def torch_module():
    import torch
    return torch


def _training_targets(target_ref,parent,bundle,model,*,max_metadata_bytes):
    """Only already prepared TRAIN targets; this module never opens private truth."""
    import numpy as np
    from arggnn.benchmarking.targets import SampledFocalTargets
    metadata = european._json_ref(target_ref,max_metadata_bytes)
    expected_classes = list(model.output_classes)
    _need(metadata.get("schema") in ("ap_paper_TRAIN_native_targets_v1","european_TRAIN_sampled_targets_v1")
          and metadata.get("parent_id") == parent and metadata.get("split") == "train"
          and metadata.get("registry_ref") == bundle.sampled_registry_ref
          and metadata.get("no_label_dependent_selection") is True
          and metadata.get("native_path_names",metadata.get("class_order")) == expected_classes,
          "Same-parent canonical TRAIN sampled target receipt required after observation barrier")
    path = european._bound(metadata["arrays_ref"])
    with zipfile.ZipFile(path) as archive:
        _need(archive.namelist() == ["native_labels_0based.npy"]
              and sum(item.file_size for item in archive.infolist()) <= 2*2**20,
              "Only bounded canonical TRAIN label vector required")
    with np.load(path,allow_pickle=False) as archive:
        labels = archive["native_labels_0based"].copy()
    n = bundle.focal_hold.numel()
    _need(labels.dtype == np.int64 and labels.shape == (n,) and metadata.get("rows") == n
          and ((labels >= 0) & (labels < len(expected_classes))).all(),
          "Every canonical sampled TRAIN draw requires one actual target")
    query_lookup = {int(value):i for i,value in enumerate(bundle.bank.query_column)}
    return SampledFocalTargets(readout_rows=tuple(map(int,bundle.focal_hold.tolist())),
        query_columns=tuple(query_lookup[int(value)] for value in bundle.focal_sample_column),
        labels=tuple(map(int,labels)),class_order=tuple(expected_classes),parent_id=parent,split="train",
        sampled_registry_ref=bundle.sampled_registry_ref)


def fixed_training_loop(*,parents,load_parent,join_targets,step,optimizer_counts,checkpoint,check,event):
    """Thirty complete passes and actual counter proof; tiny backends can test it."""
    _need(parents and len(parents) == len(set(parents)),"Distinct complete TRAIN parents required")
    updates,refs = 0,{}
    for epoch in range(1,31):
        order = list(parents)
        random.Random(101+epoch).shuffle(order)
        for parent in order:
            check("European_TreeLoom_before_complete_parent")
            bundle,metadata = load_parent(parent)
            targets = join_targets(parent,bundle,metadata)
            result = step(bundle,targets,metadata)
            updates += 1
            counts = optimizer_counts()
            _need(len(counts) == 49 and set(counts) == {updates}
                  and result.get("parent_id") == parent
                  and result.get("one_complete_parent_optimizer_update") is True,
                  "One full-parent sampled update and all49 actual Adam counters required")
            event(dict(epoch=epoch,optimizer_updates=updates,**result))
            del bundle,metadata,targets
        if epoch in SETTINGS["checkpoint_epochs"]:
            refs[str(epoch)] = checkpoint(epoch,updates)
    _need(updates == 30*len(parents) and set(refs) == {"5","10","15","20","25","30"},
          "Every complete parent in every pass and fixed final checkpoint required")
    return dict(optimizer_updates=updates,completed_epochs=30,selected_epoch=30,
                checkpoint_ref=refs["30"],checkpoint_event_refs=refs)


def _validate_saved(saved,model,*,protocol_id,sampling_plan_ref,updates,barrier_ref):
    import torch
    _need(saved.get("schema") == "european_treeloom_full_checkpoint_v1"
          and saved.get("protocol_id") == protocol_id
          and saved.get("sampling_plan_ref") == sampling_plan_ref
          and saved.get("observation_barrier_ref") == barrier_ref
          and saved.get("epoch") == 30 and saved.get("optimizer_updates") == updates
          and saved.get("settings") == SETTINGS and saved.get("class_order") == list(model.output_classes)
          and saved.get("checkpoint_selection") == "fixed_final_epoch30"
          and saved.get("TEST_truth_read") is False,
          "Exact final30-pass same-endpoint European checkpoint required")
    wanted,state = model.state_dict(),saved["model_state"]
    _need(set(state) == set(wanted),"All exact frozen European model state keys required")
    for key,value in state.items():
        _need(torch.is_tensor(value) and value.device.type == "cpu" and value.dtype == torch.float32
              and value.shape == wanted[key].shape and bool(torch.isfinite(value).all()),
              "Finite exact European saved model tensor required: "+key)
    optimizer,parameters = saved["optimizer_state"],list(model.parameters())
    groups,states = optimizer["param_groups"],optimizer["state"]
    _need(len(parameters) == len(states) == 49 and len(groups) == 1,
          "All49 actual European Adam parameter states required")
    group = groups[0]
    ids = group["params"]
    _need(len(ids) == 49 and len(set(ids)) == 49 and set(ids) == set(states)
          and group["lr"] == .001 and group["weight_decay"] == .0001
          and tuple(group["betas"]) == (.9,.999) and group["eps"] == 1e-8,
          "Original European AdamW settings required")
    for index,parameter in zip(ids,parameters):
        item = states[index]
        value = item["step"].item() if torch.is_tensor(item["step"]) else item["step"]
        _need(value == updates,"Every actual saved Adam counter must equal all30passes")
        for name in ("exp_avg","exp_avg_sq"):
            value = item[name]
            _need(torch.is_tensor(value) and value.device.type == "cpu" and value.dtype == torch.float32
                  and value.shape == parameter.shape and bool(torch.isfinite(value).all()),
                  "Exact finite saved European Adam moments required")
        _need(bool((item["exp_avg_sq"] >= 0).all()),"Nonnegative Adam second moments required")
    model.load_state_dict(state,strict=True)
    return model


def _run_scientific_fit(spec,*,resource_check,acquire_gpu0):
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
    common.finish_gpu_setup(torch, spec)
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


def _run_scientific_predictions(spec,*,resource_check):
    plan = _validate(spec,fit=False)
    torch,check,writer = common._setup_gpu(spec,resource_check)
    complete = _fit_completion(spec["fit_completion_ref"],protocol_id=spec["protocol_id"],
        sampling_plan_ref=spec["sampling_plan_ref"],train_count=len(plan["train_parent_ids"]))
    barrier = c.read_ref(complete["observation_barrier_ref"])
    _need(barrier.get("sampling_plan_ref") == spec["sampling_plan_ref"]
          and set(barrier.get("parents",{})) == set(plan["train_parent_ids"])
          and barrier.get("all_original_TRAIN_observations_sealed_before_targets") is True
          and barrier.get("private_truth_opened") is False,
          "All original TRAIN observations must precede saved fit")
    path = european._bound(complete["checkpoint_ref"])
    _need(path.stat().st_size <= 8*2**20,"Finite full saved checkpoint required")
    saved = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    model = european.build_model(spec["protocol_id"])
    _validate_saved(saved,model,protocol_id=spec["protocol_id"],sampling_plan_ref=spec["sampling_plan_ref"],
        updates=30*len(plan["train_parent_ids"]),barrier_ref=complete["observation_barrier_ref"])
    del saved
    model = install_chunked_gru(model.to(device="cuda:0",dtype=torch.float32))
    cache,predictions = OrderedDict(),{}
    load = _observed_loader(spec,cache,check,fit=False)
    for parent in plan["test_parent_ids"]:
        if parent not in spec["heldout_inputs"]:
            continue
        bundle,metadata = load(parent)
        memory._admit_parent_headroom(cache,memory_envelope=spec["memory_envelope"],stage="prediction",
            rss_bytes=memory._current_rss_bytes,resource_check=check,event=lambda *args:None,keep_parent=parent)
        logits = european.predict_logits(model,bundle,metadata,resource_check=check)
        n,classes = bundle.focal_hold.numel(),len(model.output_classes)
        _need(logits.shape == (n,classes) and logits.dtype == torch.float32
              and bool(torch.isfinite(logits).all()),"Every original sampled held-out logit required")
        probabilities = logits.softmax(-1).to("cpu").numpy()
        item = spec["heldout_inputs"][parent]
        records = european.sampled_records(item["sampled_registry_ref"],protocol_id=spec["protocol_id"],
            observation_ref=item["inference_ref"],parent_id=parent,split="test",
            max_metadata_bytes=spec["max_metadata_bytes"])
        prediction_ref = writer.npz("predictions/"+parent+".npz",
            example_indices=records["record_ids"],probabilities=probabilities)
        product = dict(parent_id=parent,split="test",registry_ref=item["sampled_registry_ref"],
            predictions_ref=prediction_ref,rows=n,native_path_names=list(model.output_classes),
            schema="european_treeloom_prediction_product_v1",method="treeloom",
            **item,checkpoint_ref=complete["checkpoint_ref"],fit_completion_ref=spec["fit_completion_ref"],
            focal_identity_sha256=bundle.focal_identity_sha256,query_ancestry_truth_opened=False,
            all_native_trees_nodes_sites_samples_retained=True)
        predictions[parent] = writer.json("products/"+parent+".json",product)
        del logits,probabilities,records,bundle,metadata
        check("European_TreeLoom_heldout_prediction_sealed")
    torch.cuda.synchronize(0)
    result = dict(schema="european_treeloom_prediction_completion_v1",
        status="completed_all_requested_sampled_predictions",protocol_id=spec["protocol_id"],
        sampling_plan_ref=spec["sampling_plan_ref"],fit_completion_ref=spec["fit_completion_ref"],
        predictions=predictions,query_ancestry_truth_opened=False,
        actual_worker_wait_and_owned_cleanup_still_required=True)
    ref = writer.json("predictions_completed.json",result)
    _need(not writer.pending and set(predictions) == set(spec["heldout_inputs"]),"Every requested held-out parent must finish")
    return dict(result,completion_ref=ref)



C4_CLASSES = ("Ana", "CHG", "WHG", "EHG")


def _c4_training_targets(ref,parent,bundle,model,*,max_metadata_bytes):
    import numpy as np
    from arggnn.benchmarking.targets import SampledFocalTargets
    metadata = european._json_ref(ref,max_metadata_bytes)
    _need(metadata.get("schema") == "european_release_C4_sampled_targets_v1"
          and metadata.get("parent_id") == parent and metadata.get("split") == "train"
          and metadata.get("registry_ref") == bundle.sampled_registry_ref
          and metadata.get("source_class_order") == list(C4_CLASSES)
          and tuple(model.output_classes) == C4_CLASSES,
          "Same-parent source-verified canonical C4TRAIN labels required after observed barrier")
    path = european._bound(metadata["arrays_ref"])
    with zipfile.ZipFile(path) as archive:
        _need(archive.namelist() == ["source_labels_0based.npy"]
              and sum(item.file_size for item in archive.infolist()) <= 2*2**20,
              "Only bounded canonical C4TRAIN label vector required")
    with np.load(path,allow_pickle=False) as archive:
        labels = archive["source_labels_0based"].copy()
    n = bundle.focal_hold.numel()
    _need(labels.dtype == np.int64 and labels.shape == (n,) and metadata.get("rows") == n
          and ((labels >= 0) & (labels < 4)).all(),
          "Every actual sampled C4TRAIN example must keep its canonical label")
    query_lookup = {int(value):i for i,value in enumerate(bundle.bank.query_column)}
    return SampledFocalTargets(readout_rows=tuple(map(int,bundle.focal_hold.tolist())),
        query_columns=tuple(query_lookup[int(value)] for value in bundle.focal_sample_column),
        labels=tuple(map(int,labels)),class_order=C4_CLASSES,parent_id=parent,split="train",
        sampled_registry_ref=bundle.sampled_registry_ref)



QUAL_SCHEMA = "european_readable_CPU_qualification_spec_v1"
FIT_WRAPPER_SCHEMA = "european_readable_full_fit_wrapper_v1"
PREDICT_WRAPPER_SCHEMA = "european_readable_prediction_wrapper_v1"


def qualification_factory(*, protocol_id, sampling_plan_ref, parent_id, observed_inputs,
        output_directory, max_sites, max_metadata_bytes, preparation_limits,
        max_rss_bytes=128*2**30, max_seconds=7200, max_output_bytes=16*2**20):
    plan = _parent_plan(sampling_plan_ref, protocol_id)
    _need(protocol_id in ("ancestralpaths_paper200mb_native6", "ancestralpaths_paper200mb_c4")
          and parent_id == plan["train_parent_ids"][0] and set(observed_inputs) == set(INPUTS),
          "Exact supported European protocol, first TRAIN parent and observations only required")
    for ref in observed_inputs.values():
        gpu._public_ref(ref)
    _need(type(max_rss_bytes) is int and 0 < max_rss_bytes <= 128*2**30
          and type(max_seconds) is int and 0 < max_seconds <= 14400
          and 5*2**20 <= max_output_bytes <= 16*2**20,
          "Finite full observed-parent CPU qualification bounds required")
    _limits(max_sites=max_sites, max_metadata_bytes=max_metadata_bytes,
        preparation_limits=preparation_limits, max_output_bytes=max_output_bytes, parents=1)
    output = c.contained_path(output_directory)
    _need(output.is_relative_to(c.ROOT/"experiments/results") and output.name == "native" and not output.exists(),
          "Fresh supplied qualification task/native directory required")
    spec = dict(schema=QUAL_SCHEMA, protocol_id=protocol_id, parent_id=parent_id,
        sampling_plan_ref=copy.deepcopy(sampling_plan_ref), observed_inputs=copy.deepcopy(observed_inputs),
        output_directory=str(output.relative_to(c.ROOT)), max_sites=max_sites,
        max_metadata_bytes=max_metadata_bytes, preparation_limits=copy.deepcopy(preparation_limits),
        max_rss_bytes=max_rss_bytes, max_seconds=max_seconds, max_output_bytes=max_output_bytes-2**20,
        candidate_source_refs=common.source_refs(), CPU_only=True)
    return common.descriptor("european_stage", "run_qualification", spec, requirements()["source_refs"])


def _require_pass(ref, scientific_spec, observed_inputs=None):
    from .qualification import contract
    result = c.read_ref(ref)
    _need(result.get("schema") == "european_readable_complete_parent_CPU_qualification_v1"
          and result.get("status") == "passed_full_observed_parent_CPU_preparation"
          and result.get("protocol_id") == scientific_spec["protocol_id"]
          and result.get("sampling_plan_ref") == scientific_spec["sampling_plan_ref"]
          and result.get("candidate_source_refs") == common.source_refs()
          and result.get("preparation_contract") == contract(scientific_spec)
          and result.get("all_native_trees_nodes_sites_samples_retained") is True
          and result.get("sampled_rows_and_repeated_draws_preserved") is True
          and result.get("private_truth_opened") is False and result.get("training_labels_opened") is False
          and result.get("model_instantiated") is False and result.get("CUDA_called") is False
          and result.get("GPU_fit_qualified") is False and result.get("full_parent_preparation_count") == 1
          and result.get("disk_cache_bytes") == result.get("scientific_outputs_created") == 0
          and 0 < result.get("observed_peak_rss_bytes", 0) <= 128*2**30
          and 0 < result.get("elapsed_seconds", 0) <= 14400
          and 0 < result.get("observed_tensor_bytes", 0) <= 128*2**30
          and isinstance(result.get("complete_signature_sha256"), str)
          and len(result["complete_signature_sha256"]) == 64,
          "Actual same-source same-input full CPU qualification required")
    plan = _parent_plan(scientific_spec["sampling_plan_ref"], scientific_spec["protocol_id"])
    _need(result.get("parent_id") == plan["train_parent_ids"][0], "Exact qualified first TRAIN parent required")
    if observed_inputs is not None:
        _need(result.get("observed_inputs") == observed_inputs, "Exact qualified observed input refs required")
    return result


def run_qualification(spec, *, resource_check):
    from .qualification import qualify
    _need(spec["schema"] == QUAL_SCHEMA and spec["CPU_only"] is True
          and spec["candidate_source_refs"] == common.source_refs(), "Exact readable CPU qualification source spec required")
    result = qualify(spec, resource_check=resource_check)
    ref = c.file_ref(c.contained_path(spec["output_directory"])/"qualification.json")
    _need(c.read_ref(ref) == result, "Actual qualification receipt must match the returned result")
    _require_pass(ref, spec, spec["observed_inputs"])
    return ref


def fit_factory(*, qualification_ref, **kwargs):
    spec = build_fit_spec(**kwargs)
    plan = _parent_plan(spec["sampling_plan_ref"], spec["protocol_id"])
    first = spec["train_inputs"][plan["train_parent_ids"][0]]
    _require_pass(qualification_ref, spec, {key: first[key] for key in INPUTS})
    wrapped = dict(schema=FIT_WRAPPER_SCHEMA, scientific_spec=spec,
        qualification_ref=copy.deepcopy(qualification_ref), candidate_source_refs=common.source_refs())
    return common.descriptor("european_stage", "run_fit", wrapped, spec["requirements"]["source_refs"])


def prediction_factory(*, qualification_ref, **kwargs):
    spec = build_prediction_spec(**kwargs)
    _require_pass(qualification_ref, spec)
    wrapped = dict(schema=PREDICT_WRAPPER_SCHEMA, scientific_spec=spec,
        qualification_ref=copy.deepcopy(qualification_ref), candidate_source_refs=common.source_refs())
    return common.descriptor("european_stage", "run_predictions", wrapped, spec["requirements"]["source_refs"])


def _validate_wrapper(spec, schema):
    _need(spec == dict(schema=schema, scientific_spec=spec["scientific_spec"],
          qualification_ref=spec["qualification_ref"], candidate_source_refs=common.source_refs()),
          "Exact readable European source wrapper required")
    scientific = spec["scientific_spec"]
    first = None
    if schema == FIT_WRAPPER_SCHEMA:
        plan = _parent_plan(scientific["sampling_plan_ref"], scientific["protocol_id"])
        first = {key: scientific["train_inputs"][plan["train_parent_ids"][0]][key] for key in INPUTS}
    _require_pass(spec["qualification_ref"], scientific, first)
    return scientific


def run_fit(spec, *, resource_check, acquire_gpu0):
    scientific = _validate_wrapper(spec, FIT_WRAPPER_SCHEMA)
    return _run_scientific_fit(scientific, resource_check=resource_check, acquire_gpu0=acquire_gpu0)


def run_predictions(spec, *, resource_check):
    scientific = _validate_wrapper(spec, PREDICT_WRAPPER_SCHEMA)
    return _run_scientific_predictions(scientific, resource_check=resource_check)
