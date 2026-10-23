"""Representative fresh TreeLoom fit:20 complete30Mb parents x30 passes.

Source/metadata planning is inert. The scientific callable runs only inside the
current admitted GPU3/4/5 worker. It imports no legacy controller, launches nothing,
opens only registered TRAIN truth, and never predicts/scores held-out parents.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass
import copy
import gc
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
import time

from arggnn.benchmarking import contracts as c
from . import runtime_v2 as fresh

SCHEMA = "treeloom_fresh_full_training_v1"
PASSES = (5, 10, 15, 20, 25, 30)
SETTINGS = dict(epochs=30, parents_per_epoch=20, optimizer_updates=600,
                model_seed=70001, parent_order_seed=70002, precision="fp32",
                learning_rate=.001, weight_decay=.0001, betas=[.9, .999], eps=1e-8,
                gradient_clip_norm=1., history_holds_per_window=1024,
                checkpoint_epochs=list(PASSES), checkpoint_retention=1,
                checkpoint_selection="fixed_final_epoch30", early_stopping=False,
                validation_used_for_optimization=False, TEST_truth_access=False)
LIMITS = dict(max_nodes=400000000, max_holds=100000,
              max_episode_payload_bytes=8*2**30)


def _need(ok, message):
    if not ok:
        raise c.ContractError(message)


EXECUTION_VARIANT = "full_parent_parallel_history_v2"
SOURCE_PINS = "research/benchmark_rebuild_20261024/american/treeloom_speed/source_pins.json"

def requirements():
    manifest = c.read_json(SOURCE_PINS)
    refs = manifest["source_refs"]
    for reference in refs:
        _need(c.file_ref(reference["path"]) == reference, "Pinned static source changed")
    return dict(source_refs=refs + [c.file_ref(SOURCE_PINS)],
                python=".runtime/env/bin/python", adapter="native_PyTorch_PyG",
                settings=copy.deepcopy(SETTINGS), model_parameters=32132,
                trainable_tensors=49, focal_targets_per_parent=10000,
                sequence_length_bp=30000000, output_scope_cap_bytes=64*2**20,
                max_single_checkpoint_bytes=8*2**20, checkpoint_peak_files=2,
                entire_observation_bank_remains_CPU=True,
                analytical_parent_peak_RSS_proven=False,
                explicit_parent_preparation_and_training_RSS_reserves_required=True,
                execution_variant=EXECUTION_VARIANT, full_device=True,
                GNN_holds_per_block=128, GRU_haplotypes_per_tile=64,
                history_holds_per_window=1024, gradient_truncation=False,
                scientific_payloads_opened=False, jobs_started=0)


def _bindings(registry, train_inputs):
    parents = registry.ids("train")
    _need(len(parents) == 20 and len(set(parents)) == 20
          and type(train_inputs) is dict and set(train_inputs) == set(parents),
          "Exactly twenty distinct registered TRAIN parents required")
    for parent, item in train_inputs.items():
        _need(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", parent) is not None,
              "Registered parent ID must be a safe metadata filename")
        _need(type(item) is dict and set(item) == {*fresh.INPUT_REFS, "truth_ref"},
              "Separate actual observed inputs and same-parent TRAIN truth refs required")
        for ref in item.values():
            fresh._ref(ref)
        observed = c.read_ref(item["observation_ref"])
        prepared = c.read_ref(item["native_preparation_ref"])
        # Compact focal metadata is sufficient. No MLM/path corpus or fabricated
        # full-native-preparation completion is required or implied.
        _need(registry.split_for(parent) == "train"
              and observed["parent_id"] == prepared["parent_id"] == parent
              and observed["sequence_length_bp"] == 30000000
              and observed["split"] == "train"
              and observed["query_ancestry_truth_in_inputs"] is False
              and prepared["observation_ref"] == item["observation_ref"]
              and prepared["focals_ref"] == item["focal_registry_ref"]
              and prepared["class_order"] == observed["class_order"] == list(fresh.CLASSES)
              and prepared["genealogy_kind"] == observed["genealogy_kind"]
              and prepared["query_ancestry_truth_opened"] is False,
              "Every actual full observed parent/focal identity must bind before truth")


def build_spec(*, parent_registry_ref, train_inputs, output_directory,
               max_rss_bytes, preparation_headroom_bytes, training_headroom_bytes,
               observation_cache_bytes=128*2**30, max_truth_uncompressed_bytes=8*2**30,
               resume_checkpoint_ref=None, resume_evidence_ref=None):
    registry = fresh.FreshParentRegistry.from_ref(parent_registry_ref)
    _bindings(registry, train_inputs)
    validate_resume_metadata(registry, resume_checkpoint_ref, resume_evidence_ref)
    out = c.contained_path(output_directory)
    _need(out.resolve() == out and out.is_relative_to(c.ROOT/"experiments/results")
          and not out.exists(), "Fresh dedicated training output required")
    _need(type(observation_cache_bytes) is int and observation_cache_bytes >= 0
          and type(max_truth_uncompressed_bytes) is int and max_truth_uncompressed_bytes > 0,
          "Explicit finite observation cache and TRAIN truth payload bounds required")
    memory = _memory_envelope(max_rss_bytes=max_rss_bytes,
                              preparation_headroom_bytes=preparation_headroom_bytes,
                              training_headroom_bytes=training_headroom_bytes,
                              observation_cache_bytes=observation_cache_bytes)
    _need(training_headroom_bytes > max_truth_uncompressed_bytes,
          "Training reserve must also cover complete-parent autograd beyond TRAIN truth")
    return dict(schema=SCHEMA, parent_registry_ref=copy.deepcopy(parent_registry_ref),
                train_inputs=copy.deepcopy(train_inputs), output_directory=str(out.relative_to(c.ROOT)),
                observation_cache_bytes=observation_cache_bytes, memory_envelope=memory,
                max_truth_uncompressed_bytes=max_truth_uncompressed_bytes,
                preparation_limits=copy.deepcopy(LIMITS), settings=copy.deepcopy(SETTINGS),
                requirements=requirements(), private_truth_in_observations=False,
                resume_checkpoint_ref=copy.deepcopy(resume_checkpoint_ref),
                resume_evidence_ref=copy.deepcopy(resume_evidence_ref))


def _validate_spec(spec):
    expected = build_spec(parent_registry_ref=spec["parent_registry_ref"],
                          train_inputs=spec["train_inputs"],
                          output_directory=spec["output_directory"],
                          max_rss_bytes=spec["memory_envelope"]["max_rss_bytes"],
                          preparation_headroom_bytes=spec["memory_envelope"]["preparation_headroom_bytes"],
                          training_headroom_bytes=spec["memory_envelope"]["training_headroom_bytes"],
                          observation_cache_bytes=spec["observation_cache_bytes"],
                          max_truth_uncompressed_bytes=spec["max_truth_uncompressed_bytes"],
                          resume_checkpoint_ref=spec["resume_checkpoint_ref"],
                          resume_evidence_ref=spec["resume_evidence_ref"])
    _need(spec == expected, "Exact complete600-update fresh training specification required")
    return fresh.FreshParentRegistry.from_ref(spec["parent_registry_ref"])


def _optimizer_counts(optimizer):
    counts = []
    for item in optimizer.state.values():
        _need(isinstance(item, dict) and {"step","exp_avg","exp_avg_sq"} <= set(item),
              "Actual Adam moments and counters required for all49 tensors")
        step = item["step"].item() if hasattr(item["step"], "item") else item["step"]
        _need(type(step) in (int,float) and step == int(step), "Actual integral Adam counter required")
        counts.append(int(step))
    return counts


def _target_binding(target, parent, bundle, registry, train_inputs):
    receipt = target.receipt
    _need(target.parent_id == parent and target.focal_registry_ref == bundle.focal_registry_ref
          and target.focal_identity_sha256 == bundle.focal_identity_sha256
          and receipt.get("parent_id") == parent and receipt.get("split") == "train"
          and receipt.get("parent_registry_ref") == registry.reference
          and receipt.get("truth_ref") == train_inputs[parent]["truth_ref"]
          and receipt.get("focal_registry_ref") == bundle.focal_registry_ref
          and receipt.get("focal_identity_sha256") == bundle.focal_identity_sha256
          and receipt.get("records") == 10000
          and receipt.get("labels_in_observation_bank") is False,
          "Exact same-parent registered TRAIN target/focal join required before optimization")


def train_loop(*, registry, runtime, model, optimizer, train_inputs, observation_barrier,
               load_parent, join_targets, checkpoint, event, resource_check,
               optimizer_counts=_optimizer_counts, start_steps=0, initial_checkpoint_refs=None):
    """Fixed numerical schedule; callbacks permit a tiny fake-backend test."""
    parents = tuple(registry.ids("train"))
    _need(len(parents) == 20 and len(set(parents)) == 20
          and observation_barrier.get("parent_registry_ref") == registry.reference
          and observation_barrier.get("all20_observations_sealed_before_truth") is True
          and observation_barrier.get("private_truth_opened") is False
          and set(observation_barrier.get("parents", {})) == set(parents),
          "Complete all20 observed identity barrier required before fitting")
    _need(all(callable(f) for f in (load_parent,join_targets,checkpoint,event,resource_check,optimizer_counts)),
          "Actual admitted numerical and output callbacks required")
    _need(type(start_steps) is int and 0 <= start_steps <= 600, "Durable complete-parent resume count required")
    checkpoints = dict(initial_checkpoint_refs or {})
    _need(set(checkpoints) == {str(e) for e in PASSES if e*20 <= start_steps},
          "Every earlier fixed checkpoint event must remain referenced after resume")
    steps = start_steps
    for epoch in range(1,31):
        order = registry.epoch_order(epoch)
        _need(len(order) == 20 and set(order) == set(parents), "Every complete TRAIN parent once per epoch")
        for index, parent in enumerate(order):
            if (epoch-1)*20 + index < start_steps:
                continue
            resource_check("TreeLoom_before_complete_parent")
            bundle, metadata = load_parent(parent)
            seal = observation_barrier["parents"][parent]
            _need(registry.split_for(parent) == "train" and bundle.parent_id == parent
                  and bundle.split == "train" and bundle.sequence_length_bp == 30000000
                  and metadata.get("parent_id") == parent
                  and metadata.get("declared_split") == "train"
                  and metadata.get("parent_registry_ref") == registry.reference
                  and metadata.get("all_native_trees_nodes_sites_samples_retained") is True
                  and metadata.get("private_truth_opened") is False
                  and seal.get("focal_identity_sha256") == bundle.focal_identity_sha256
                  and all(getattr(bundle,k) == train_inputs[parent][k] == seal[k] for k in fresh.INPUT_REFS),
                  "Full same-parent observed bank and frozen focal barrier required")
            bundle.validate()
            target = join_targets(parent, bundle, metadata)
            _target_binding(target, parent, bundle, registry, train_inputs)
            result = runtime.train_step(model, bundle, target, optimizer, metadata,
                                        resource_check=resource_check)
            _need(result.get("parent_id") == parent
                  and result.get("one_complete_parent_optimizer_update") is True
                  and result.get("native_focals") == 10000,
                  "One genuine complete-parent all-focal update required")
            steps += 1
            counts = optimizer_counts(optimizer)
            _need(len(counts) == 49 and set(counts) == {steps}, "All49 actual Adam counters must match the600-update schedule")
            checkpoint(epoch, steps, model, optimizer, latest=True)
            event("complete_parent_update", dict(epoch=epoch, optimizer_updates=steps, **result))
            del bundle, metadata, target
        if epoch in PASSES and epoch*20 > start_steps:
            resource_check("TreeLoom_before_checkpoint")
            checkpoints[str(epoch)] = checkpoint(epoch, steps, model, optimizer)
    _need(steps == 600 and set(checkpoints) == {str(e) for e in PASSES}, "Complete30passes600updates and all six checkpoint events required")
    return dict(complete_parent_updates=steps, completed_epochs=30,
                checkpoint_event_refs=checkpoints, checkpoint_ref=checkpoints.get("30"),
                selected_epoch=30, checkpoint_selection="fixed_final_epoch30",
                VAL_truth_read=False, TEST_truth_read=False, ancestry_accuracy=None)


def _cpu_bytes(value, torch):
    storages = {}
    def visit(item):
        if torch.is_tensor(item):
            _need(item.device.type == "cpu" and not item.requires_grad,
                  "Only immutable observed CPU tensors may be reused")
            storage = item.untyped_storage()
            storages[(storage.data_ptr(),storage.nbytes())] = storage.nbytes()
        elif is_dataclass(item):
            for field in fields(item): visit(getattr(item,field.name))
        elif isinstance(item,dict):
            for child in item.values(): visit(child)
        elif isinstance(item,(tuple,list)):
            for child in item: visit(child)
    visit(value)
    return sum(storages.values())



def _memory_envelope(*, max_rss_bytes, preparation_headroom_bytes,
                     training_headroom_bytes, observation_cache_bytes):
    """Declared conservative reserves, not an analytical Python/allocator peak proof."""
    values = (max_rss_bytes, preparation_headroom_bytes, training_headroom_bytes)
    _need(all(type(v) is int and v > 0 for v in values)
          and type(observation_cache_bytes) is int and observation_cache_bytes >= 0,
          "Explicit positive process RSS and whole-parent working reserves required")
    _need(observation_cache_bytes + max(preparation_headroom_bytes, training_headroom_bytes)
          < max_rss_bytes,
          "RSS envelope must leave headroom beyond retained observations and parent reserve")
    return dict(max_rss_bytes=max_rss_bytes,
                preparation_headroom_bytes=preparation_headroom_bytes,
                training_headroom_bytes=training_headroom_bytes,
                reserve_basis="declared_conservative_whole_parent_working_set_not_measured_peak",
                analytical_parent_peak_RSS_proven=False,
                no_parent_reduction_or_partial_update=True)


def _current_rss_bytes():
    # This runner is admitted on Linux. Read current RSS, never ru_maxrss: freed
    # cache may reduce current usage even when the historical high-water mark stays.
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError) as error:
        raise c.ContractError("Current Linux process RSS is required for parent admission") from error
    _need(pages >= 0 and type(page_size) is int and page_size > 0,
          "Valid current process RSS required")
    return pages * page_size


def _drop_cache_entry(cache, parent):
    record = cache.pop(parent)
    size = record[2]
    del record
    # Observation metadata may contain cycles. Actual RSS is re-read afterwards;
    # retained allocator arenas are never assumed to have been released.
    gc.collect()
    return size


def _admit_parent_headroom(cache, *, memory_envelope, stage, rss_bytes,
                           resource_check, event, keep_parent=None):
    reserve = memory_envelope[stage + "_headroom_bytes"]
    ceiling = memory_envelope["max_rss_bytes"]
    while True:
        resource_check("TreeLoom_" + stage + "_RSS_admission")
        current = rss_bytes()
        _need(type(current) is int and current >= 0, "Actual current RSS required")
        if current + reserve <= ceiling:
            return
        victim = next((p for p in cache if p != keep_parent), None)
        _need(victim is not None,
              "Complete-parent " + stage + " RSS reserve unavailable after cache eviction; "
              "a larger admitted envelope is required, never a reduced parent")
        size = _drop_cache_entry(cache, victim)
        event("observation_cache_evicted", dict(parent_id=victim,
              observed_tensor_bytes=size, admission_stage=stage))


def _load_cached_parent(parent, cache, prepare, *, observation_cache_bytes,
                        memory_envelope, rss_bytes, tensor_bytes, resource_check, event):
    if parent in cache:
        bundle, metadata, size = cache.pop(parent)
        cache[parent] = (bundle, metadata, size)
        return bundle, metadata
    # Reserve the *complete cold preparation* before opening the observation or
    # constructing any new bank. Post-construction eviction is too late.
    _admit_parent_headroom(cache, memory_envelope=memory_envelope, stage="preparation",
                           rss_bytes=rss_bytes, resource_check=resource_check, event=event)
    bundle, metadata = prepare()
    resource_check("TreeLoom_after_complete_parent_preparation")
    _need(rss_bytes() <= memory_envelope["max_rss_bytes"], "Complete-parent preparation RSS cap exceeded")
    size = tensor_bytes(bundle)
    _need(type(size) is int and size >= 0, "Actual immutable observed tensor bytes required")
    while cache and sum(value[2] for value in cache.values()) + size > observation_cache_bytes:
        _drop_cache_entry(cache, next(iter(cache)))
    if size <= observation_cache_bytes:
        cache[parent] = (bundle, metadata, size)
    return bundle, metadata


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""): h.update(block)
    return dict(path=str(path.relative_to(c.ROOT)), sha256=h.hexdigest())


def run_training(spec, *, resource_check):
    """Full fresh PyG training inside the current admitted GPU3/4/5 worker."""
    _need(callable(resource_check), "Current worker resource callback required")
    resource_check("TreeLoom_before_fresh_training")
    registry = _validate_spec(spec)
    worker_resource_check = resource_check
    def resource_check(*args, **kwargs):
        worker_resource_check(*args, **kwargs)
        _need(_current_rss_bytes() <= spec["memory_envelope"]["max_rss_bytes"],
              "Declared training process RSS cap exceeded")
    resource_check("TreeLoom_explicit_RSS_envelope")
    _need(os.environ.get("ARGGNN_PHYSICAL_GPU") in ("3", "4", "5")
          and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
          and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
          and os.environ.get("ARGGNN_GPU_MEMORY_POLICY") == "full_device",
          "One ROOT-verified visible physicalGPU3/4/5 UUID and full-device policy required")
    out = c.contained_path(spec["output_directory"])
    out.mkdir()
    (out/"plans").mkdir()
    (out/"checkpoints").mkdir()
    for name in ("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","TORCH_HOME","TRITON_CACHE_DIR","CUDA_CACHE_PATH"):
        path=out/(".tmp" if name in ("TMPDIR","TMP","TEMP") else ".cache/"+name.lower())
        path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    tempfile.tempdir=str(out/".tmp")
    os.environ["CUDA_CACHE_MAXSIZE"]=str(16*2**20)
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    import torch
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    _need(torch.cuda.device_count() == 1, "One admitted visible GPU3/4/5 required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = fresh.FreshTreeLoom(registry)
    runtime.status_path = out/"training_live.json"
    plans, seals, cache = {}, {}, OrderedDict()
    def write_json(path, value):
        resource_check("TreeLoom_metadata_write")
        with path.open("x") as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        return c.file_ref(path)
    def event(name,value):
        resource_check("TreeLoom_progress")
        raw=(json.dumps(dict(event=name,**value),sort_keys=True,allow_nan=False)+"\n").encode()
        path=out/"progress.jsonl"
        _need(len(raw)<=65536 and (path.stat().st_size if path.exists() else 0)+len(raw)<=2*2**20,
              "Bounded600-update progress")
        with path.open("ab") as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
    def prepare_parent(parent):
        resource_check("TreeLoom_observed_parent")
        item=spec["train_inputs"][parent]
        config=dict(native_preparation_ref=item["native_preparation_ref"],class_order=list(fresh.CLASSES),
                    expected_haplotypes=1600,reference_counts=[200,200,200],
                    input_reference_class_mask=None,input_reference_sample_indices=None)
        state=load_observation(item["observation_ref"],item["focal_registry_ref"],config)
        bundle,meta=runtime.prepare_observation(
            state,split="train",resource_check=resource_check,
            **{k:item[k] for k in fresh.INPUT_REFS},**spec["preparation_limits"])
        del state
        _need(bundle.bank.parent.device.type=="cpu" and bundle.parent_id==parent
              and bundle.sequence_length_bp==30000000, "Complete observed bank remains CPU")
        if parent not in plans:
            plan=dict(parent_id=parent,sequence_length_bp=30000000,**{k:item[k] for k in fresh.INPUT_REFS},
                      native_holds=bundle.native_holds,original_sites=bundle.original_sites,
                      parent_registry_ref=registry.reference,observation_only=True,private_truth_opened=False)
            ref=write_json(out/"plans"/(parent+".json"),plan)
            plans[parent]=dict(plan,plan_ref=ref)
            seals[parent]=dict(parent_id=parent,plan_ref=ref,parent_registry_ref=registry.reference,
                               **{k:item[k] for k in fresh.INPUT_REFS},
                               observation_only=True,all_blocks_sealed_before_truth=True,
                               focal_identity_sha256=bundle.focal_identity_sha256)
            event("observed_parent_sealed",dict(parent_id=parent,sealed=len(plans),private_truth_opened=False))
        _need(bundle.focal_identity_sha256==seals[parent]["focal_identity_sha256"]
              and all(getattr(bundle,k)==seals[parent][k] for k in fresh.INPUT_REFS),
              "Reused/reprepared observation must preserve the original complete focal seal")
        return bundle,meta
    def load_parent(parent):
        resource_check("TreeLoom_before_observed_parent_cache")
        return _load_cached_parent(
            parent, cache, lambda: prepare_parent(parent),
            observation_cache_bytes=spec["observation_cache_bytes"],
            memory_envelope=spec["memory_envelope"], rss_bytes=_current_rss_bytes,
            tensor_bytes=lambda bundle: _cpu_bytes(bundle,torch),
            resource_check=resource_check, event=event)
    # No target resolver exists until every original complete observation seals.
    for parent in registry.ids("train"):
        bundle,meta=load_parent(parent)
        del bundle,meta
    barrier=dict(schema="treeloom_fresh_all20_observation_barrier_v1",
                 parent_registry_ref=registry.reference,train_parent_ids=list(registry.ids("train")),
                 parents=seals,all20_observations_sealed_before_truth=True,private_truth_opened=False)
    barrier_ref=write_json(out/"observation_barrier.json",barrier)
    def resolve_truth(parent):
        _need(registry.split_for(parent)=="train", "No held-out truth resolver")
        return spec["train_inputs"][parent]["truth_ref"]
    joiner=fresh.FreshTrainTargetJoin(registry,resolve_truth,train_inputs=spec["train_inputs"],
                                     max_truth_uncompressed_bytes=spec["max_truth_uncompressed_bytes"])
    def join(parent,bundle,meta):
        # Keep the active bundle: evicting its cache entry cannot free the
        # reference held by train_loop. Reserve truth loading plus the complete
        # CPU autograd working set before either begins.
        _admit_parent_headroom(cache, memory_envelope=spec["memory_envelope"], stage="training",
                               rss_bytes=_current_rss_bytes, resource_check=resource_check,
                               event=event, keep_parent=parent)
        return joiner.join(parent,split="train",plan=plans[parent],observation_seal=seals[parent])
    model=runtime.build_model().to(device="cuda:0",dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,
                                maximize=False,amsgrad=False,foreach=False,fused=False,
                                capturable=False,differentiable=False)
    _need(not optimizer.state,"Fresh optimizer object required before strict state restore")
    start_steps = 0
    checkpoint_history = {}
    if spec["resume_checkpoint_ref"] is not None:
        resume_ref = spec["resume_checkpoint_ref"]
        resume_path = c.contained_path(resume_ref["path"])
        _need(resume_path.stat().st_size <= 8*2**20 and _hash(resume_path) == resume_ref,
              "Exact bounded durable resume checkpoint required")
        saved = torch.load(resume_path, map_location="cpu", weights_only=True, mmap=True)
        start_steps = validate_resume_checkpoint(saved, model, registry, spec)
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_random_state(saved["random_state"])
        checkpoint_history.update(saved.get("checkpoint_event_refs", {}))
        if start_steps % 100 == 0:
            checkpoint_history.setdefault(str(start_steps//20), resume_ref)
        _need(set(checkpoint_history) == {str(e) for e in PASSES if e*20 <= start_steps},
              "All earlier periodic checkpoint event hashes must be preserved")
        _need(len(_optimizer_counts(optimizer)) == 49 and set(_optimizer_counts(optimizer)) == {start_steps},
              "All49 loaded optimizer counters must preserve the serialized boundary")
        del saved
        event("resumed_saved_optimizer_boundary", dict(complete_parent_updates=start_steps,
              resume_checkpoint_ref=resume_ref, resume_evidence_ref=spec["resume_evidence_ref"],
              original_optimizer_moments_preserved=True))
    previous=[None]
    def checkpoint(epoch,steps,actual_model,actual_optimizer,latest=False):
        resource_check("TreeLoom_checkpoint_write")
        payload=dict(schema="treeloom_fresh_full_checkpoint_v1",epoch=epoch,complete_parent_updates=steps,
                     model_seed=70001,parent_registry_ref=registry.reference,observation_barrier_ref=barrier_ref,
                     settings=copy.deepcopy(SETTINGS),contract=actual_model.checkpoint_contract(),
                     execution_variant=EXECUTION_VARIANT, execution_requirements=spec["requirements"],
                     resume_checkpoint_ref=spec["resume_checkpoint_ref"],resume_evidence_ref=spec["resume_evidence_ref"],
                     resume_schema="treeloom_exact_complete_update_resume_v1",
                     schedule_identity=schedule_identity(schedule),
                     last_completed_parent=schedule[steps-1][1],next_schedule_index=steps,
                     checkpoint_event_refs=dict(checkpoint_history),random_state=random_state(),
                     model_state=_cpu_snapshot(actual_model.state_dict()),
                     optimizer_state=_cpu_snapshot(actual_optimizer.state_dict()),
                     VAL_truth_read=False,TEST_truth_read=False,checkpoint_selection="fixed_final_epoch30")
        path=out/"checkpoints"/("latest.pt.next" if latest else "epoch_%02d.pt"%epoch)
        with path.open("xb") as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        _need(path.stat().st_size<=8*2**20,"Finite full model/optimizer checkpoint")
        if latest:
            target=out/"checkpoints/latest.pt"
            _need(not target.is_symlink(), "Ordinary own-scope latest checkpoint required")
            os.replace(path,target)
            ref=_hash(target)
            write_json_replace(out/"latest_checkpoint.json",dict(checkpoint_ref=ref,
                complete_parent_updates=steps,epoch=epoch,parent_id=schedule[steps-1][1]))
            return ref
        ref=_hash(path)
        if previous[0] is not None:
            old,old_ref=previous[0]
            _need(old.parent==path.parent and not old.is_symlink() and _hash(old)==old_ref,
                  "Rotate only the exact previous new-scope checkpoint")
            old.unlink()
        previous[0]=(path,ref)
        checkpoint_history[str(epoch)]=ref
        event("checkpoint_saved",dict(epoch=epoch,optimizer_updates=steps,checkpoint_ref=ref,
                                     retention="keep1_new_scope_only"))
        return ref
    schedule = [(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
    started=time.monotonic()
    result=train_loop(registry=registry,runtime=runtime,model=model,optimizer=optimizer,
                      train_inputs=spec["train_inputs"],observation_barrier=barrier,
                      load_parent=load_parent,join_targets=join,checkpoint=checkpoint,event=event,
                      resource_check=resource_check,start_steps=start_steps,initial_checkpoint_refs=checkpoint_history)
    if start_steps == 600:
        result["checkpoint_ref"] = spec["resume_checkpoint_ref"]
    resource_check("TreeLoom_verify_final_saved_state")
    saved=torch.load(c.contained_path(result["checkpoint_ref"]["path"]),map_location="cpu",weights_only=True,mmap=True)
    saved_states=saved["optimizer_state"]["state"]
    counts=[int(v["step"].item()) for v in saved_states.values()]
    _need(saved["epoch"]==30 and saved["complete_parent_updates"]==600
          and len(saved_states)==49 and set(counts)=={600}
          and all(v["step"].item()==600 for v in saved_states.values())
          and len(saved["model_state"])>0,
          "Actual final saved model and all49 optimizer counters at600 required")
    del saved
    torch.cuda.synchronize(0)
    result.update(schema="treeloom_fresh_full_training_completion_v1",
                  status="completed_full20parents30passes600updates",
                  parent_registry_ref=registry.reference,observation_barrier_ref=barrier_ref,
                  execution_variant=EXECUTION_VARIANT,requirements=spec["requirements"],
                  resumed_from_complete_parent_updates=start_steps,
                  resume_checkpoint_ref=spec["resume_checkpoint_ref"],resume_evidence_ref=spec["resume_evidence_ref"],
                  final_saved_optimizer_parameter_states=49,final_saved_optimizer_steps=[600],
                  checkpoint_retention="only_final_epoch30_file_retained_earlier_event_hashes_preserved",
                  observed_cache_cap_bytes=spec["observation_cache_bytes"],
                  memory_envelope=copy.deepcopy(spec["memory_envelope"]),
                  elapsed_training_seconds=time.monotonic()-started,
                  actual_worker_wait_and_owned_cleanup_still_required=True)
    result["completion_ref"] = write_json(out/"training_completed.json",result)
    return result


def validate_resume_metadata(registry, checkpoint_ref, evidence_ref):
    if checkpoint_ref is None and evidence_ref is None:
        return None
    _need(checkpoint_ref is not None and evidence_ref is not None,
          "Resume requires both exact durable checkpoint and provenance evidence")
    fresh._ref(checkpoint_ref)
    evidence = c.read_ref(evidence_ref)
    _need(evidence.get("schema") == "treeloom_durable_optimizer_boundary_v2"
          and evidence.get("checkpoint_ref") == checkpoint_ref
          and evidence.get("parent_registry_ref") == registry.reference
          and evidence.get("durable_complete_parent_optimizer_boundary") is True
          and evidence.get("no_inflight_update_in_serialized_state") is True
          and type(evidence.get("complete_parent_updates")) is int
          and 1 <= evidence["complete_parent_updates"] < 600
          and evidence.get("VAL_truth_read") is False and evidence.get("TEST_truth_read") is False,
          "Exact same-registry durable optimizer-boundary provenance required")
    refs = evidence.get("source_refs")
    _need(isinstance(refs, list) and refs, "Actual original training source pins required")
    for reference in refs:
        _need(c.file_ref(reference["path"]) == reference, "Original saved-fit source changed")
    return evidence


def validate_saved_state(saved, model, registry, *, expected_steps=None):
    import torch
    steps = saved.get("complete_parent_updates")
    _need(type(steps) is int and 1 <= steps <= 600
          and (expected_steps is None or steps == expected_steps),
          "Actual serialized complete-parent optimizer count required")
    # The only accepted legacy settings difference is the execution partition.
    settings = dict(saved.get("settings", {}))
    legacy_window = settings.get("history_holds_per_window")
    _need(legacy_window in (128, 1024), "Recognized exact-history execution partition required")
    settings["history_holds_per_window"] = 1024
    _need(saved.get("schema") == "treeloom_fresh_full_checkpoint_v1"
          and saved.get("epoch") == (steps+19)//20
          and saved.get("model_seed") == 70001
          and saved.get("parent_registry_ref") == registry.reference
          and settings == SETTINGS
          and saved.get("contract") == model.checkpoint_contract()
          and saved.get("checkpoint_selection") == "fixed_final_epoch30"
          and saved.get("VAL_truth_read") is False and saved.get("TEST_truth_read") is False,
          "Exact selected model/scientific settings and registered TRAIN checkpoint required")
    expected, state = model.state_dict(), saved["model_state"]
    _need(type(state) is dict and set(state) == set(expected), "All exact model tensors required")
    for key, wanted in expected.items():
        actual = state[key]
        _need(torch.is_tensor(actual) and actual.device.type == "cpu"
              and actual.dtype == wanted.dtype == torch.float32 and actual.shape == wanted.shape
              and bool(torch.isfinite(actual).all()), "Exact finite FP32 model tensor: " + key)
    optimizer = saved["optimizer_state"]
    groups, states = optimizer["param_groups"], optimizer["state"]
    _need(len(groups) == 1 and len(states) == 49, "All49 saved Adam states required")
    group = groups[0]
    ids = group["params"]
    _need(len(ids) == 49 and len(set(ids)) == 49 and set(states) == set(ids)
          and group["lr"] == .001 and group["weight_decay"] == .0001
          and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
          and all(group.get(key, False) is False for key in
                  ("maximize", "amsgrad", "foreach", "fused", "capturable", "differentiable")),
          "Original complete49-parameter FP32 AdamW state required")
    parameters = list(model.parameters())
    _need(len(parameters) == 49 and sum(p.numel() for p in parameters) == 32132,
          "Original49-tensor32132-parameter model required")
    for index, parameter in zip(ids, parameters):
        item = states[index]
        step = item["step"].item() if torch.is_tensor(item["step"]) else item["step"]
        _need(step == steps, "All49 actual Adam counters must equal serialized update count")
        for name in ("exp_avg", "exp_avg_sq"):
            value = item[name]
            _need(torch.is_tensor(value) and value.device.type == "cpu"
                  and value.dtype == torch.float32 and value.shape == parameter.shape
                  and bool(torch.isfinite(value).all()), "Exact finite saved Adam moments required")
        _need(bool((item["exp_avg_sq"] >= 0).all()), "Nonnegative saved Adam second moments required")
    return steps


def validate_resume_checkpoint(saved, model, registry, spec):
    evidence = validate_resume_metadata(registry, spec["resume_checkpoint_ref"], spec["resume_evidence_ref"])
    steps = validate_saved_state(saved, model, registry, expected_steps=evidence["complete_parent_updates"])
    schedule = [(epoch, parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
    _need(saved.get("resume_schema") == "treeloom_exact_complete_update_resume_v1"
          and saved.get("schedule_identity") == schedule_identity(schedule)
          and saved.get("last_completed_parent") == schedule[steps-1][1]
          and saved.get("next_schedule_index") == steps
          and "random_state" in saved,
          "Exact original schedule boundary and saved random states required")
    barrier = c.read_ref(saved["observation_barrier_ref"])
    parents = registry.ids("train")
    _need(barrier.get("parent_registry_ref") == registry.reference
          and barrier.get("train_parent_ids") == list(parents)
          and set(barrier.get("parents", {})) == set(parents)
          and barrier.get("all20_observations_sealed_before_truth") is True
          and barrier.get("private_truth_opened") is False,
          "Original all20 observed TRAIN barrier required for resume")
    for parent in parents:
        seal = barrier["parents"][parent]
        _need(seal.get("parent_id") == parent and seal.get("parent_registry_ref") == registry.reference
              and seal.get("observation_only") is True
              and seal.get("all_blocks_sealed_before_truth") is True
              and all(seal.get(key) == spec["train_inputs"][parent][key] for key in fresh.INPUT_REFS),
              "Same original all20 observed parent identities required for resume")
        plan = c.read_ref(seal["plan_ref"])
        _need(plan.get("parent_id") == parent
              and all(plan.get(key) == spec["train_inputs"][parent][key] for key in fresh.INPUT_REFS),
              "Original observed plan reference required")
    return steps


def schedule_identity(schedule):
    return hashlib.sha256(json.dumps(schedule,separators=(",",":")).encode()).hexdigest()


def random_state():
    import random
    import numpy as np
    import torch
    state = np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(name=state[0],keys=state[1].tolist(),
        position=state[2],has_gauss=state[3],cached_gaussian=state[4]),
        torch_CPU=torch.get_rng_state(),
        torch_CUDA=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_random_state(saved):
    import random
    import numpy as np
    import torch
    random.setstate(saved["python"])
    state=saved["numpy"]
    np.random.set_state((state["name"],np.asarray(state["keys"],dtype=np.uint32),
        state["position"],state["has_gauss"],state["cached_gaussian"]))
    torch.set_rng_state(saved["torch_CPU"])
    if saved["torch_CUDA"]:
        _need(torch.cuda.is_initialized() and len(saved["torch_CUDA"]) == 1,
              "One admitted visible device required for saved CUDA RNG restore")
        torch.cuda.set_rng_state_all(saved["torch_CUDA"])


def write_json_replace(path, value):
    temporary=path.with_name(path.name+".next")
    _need(not path.is_symlink() and not temporary.exists(), "Own ordinary metadata staging path required")
    with temporary.open("x") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return c.file_ref(path)
