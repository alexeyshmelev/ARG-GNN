"""Representative fresh TreeLoom fit:20 complete30Mb parents x30 passes.

Source/metadata planning is inert. The scientific callable runs only inside the
current admitted GPU0 worker. It imports no legacy controller, launches nothing,
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

from . import contracts as c
from . import treeloom
from . import treeloom_fresh as fresh

SCHEMA = "treeloom_fresh_full_training_v1"
PASSES = (5, 10, 15, 20, 25, 30)
SETTINGS = dict(epochs=30, parents_per_epoch=20, optimizer_updates=600,
                model_seed=70001, parent_order_seed=70002, precision="fp32",
                learning_rate=.001, weight_decay=.0001, betas=[.9, .999], eps=1e-8,
                gradient_clip_norm=1., history_holds_per_window=128,
                checkpoint_epochs=list(PASSES), checkpoint_retention=1,
                checkpoint_selection="fixed_final_epoch30", early_stopping=False,
                validation_used_for_optimization=False, TEST_truth_access=False)
LIMITS = dict(max_nodes=400000000, max_holds=100000,
              max_episode_payload_bytes=8*2**30)


def _need(ok, message):
    if not ok:
        raise c.ContractError(message)


def requirements():
    manifest = treeloom.verify_implementation()
    refs = [dict(path=p, sha256=h) for p,h in manifest["source_sha256"].items()]
    refs += [treeloom.MANIFEST_REF, c.file_ref(treeloom.__file__),
             c.file_ref(fresh.__file__), c.file_ref(Path(__file__).resolve()),
             c.file_ref("src/arggnn/american_c3_gnn_dataset_v1.py"),
             c.file_ref("src/arggnn/american_c3_gnn_train_v1.py")]
    return dict(source_refs=list({r["path"]:r for r in refs}.values()),
                python=".runtime/env/bin/python", adapter="native_PyTorch_PyG",
                settings=copy.deepcopy(SETTINGS), model_parameters=32132,
                trainable_tensors=49, focal_targets_per_parent=10000,
                sequence_length_bp=30000000, output_scope_cap_bytes=64*2**20,
                max_single_checkpoint_bytes=8*2**20,
                checkpoint_peak_files=2, entire_observation_bank_remains_CPU=True,
                analytical_parent_peak_RSS_proven=False,
                explicit_parent_preparation_and_training_RSS_reserves_required=True,
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
               observation_cache_bytes=128*2**30, max_truth_uncompressed_bytes=8*2**30):
    registry = fresh.FreshParentRegistry.from_ref(parent_registry_ref)
    _bindings(registry, train_inputs)
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
                requirements=requirements(), private_truth_in_observations=False)


def _validate_spec(spec):
    expected = build_spec(parent_registry_ref=spec["parent_registry_ref"],
                          train_inputs=spec["train_inputs"],
                          output_directory=spec["output_directory"],
                          max_rss_bytes=spec["memory_envelope"]["max_rss_bytes"],
                          preparation_headroom_bytes=spec["memory_envelope"]["preparation_headroom_bytes"],
                          training_headroom_bytes=spec["memory_envelope"]["training_headroom_bytes"],
                          observation_cache_bytes=spec["observation_cache_bytes"],
                          max_truth_uncompressed_bytes=spec["max_truth_uncompressed_bytes"])
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
               optimizer_counts=_optimizer_counts):
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
    steps, checkpoints = 0, {}
    for epoch in range(1,31):
        order = registry.epoch_order(epoch)
        _need(len(order) == 20 and set(order) == set(parents), "Every complete TRAIN parent once per epoch")
        for parent in order:
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
            event("complete_parent_update", dict(epoch=epoch, optimizer_updates=steps, **result))
            del bundle, metadata, target
        if epoch in PASSES:
            resource_check("TreeLoom_before_checkpoint")
            checkpoints[str(epoch)] = checkpoint(epoch, steps, model, optimizer)
    _need(steps == 600 and set(checkpoints) == {str(e) for e in PASSES}, "Complete30passes600updates and all six checkpoint events required")
    return dict(complete_parent_updates=steps, completed_epochs=30,
                checkpoint_event_refs=checkpoints, checkpoint_ref=checkpoints["30"],
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
    """Full fresh PyG training inside the current admitted GPU0 worker."""
    _need(callable(resource_check), "Current worker resource callback required")
    resource_check("TreeLoom_before_fresh_training")
    registry = _validate_spec(spec)
    worker_resource_check = resource_check
    def resource_check(*args, **kwargs):
        worker_resource_check(*args, **kwargs)
        _need(_current_rss_bytes() <= spec["memory_envelope"]["max_rss_bytes"],
              "Declared training process RSS cap exceeded")
    resource_check("TreeLoom_explicit_RSS_envelope")
    _need(os.environ.get("ARGGNN_PHYSICAL_GPU") == "0", "Current admitted physicalGPU0 required")
    out = c.contained_path(spec["output_directory"])
    out.mkdir()
    (out/"plans").mkdir()
    (out/"checkpoints").mkdir()
    for name in ("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","TORCH_HOME","TRITON_CACHE_DIR","CUDA_CACHE_PATH"):
        path=out/(".tmp" if name in ("TMPDIR","TMP","TEMP") else ".cache/"+name.lower())
        path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    tempfile.tempdir=str(out/".tmp")
    os.environ["CUDA_CACHE_MAXSIZE"]=str(16*2**20)
    import torch
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    _need(torch.cuda.device_count() == 1, "One admitted visible GPU0 required")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES","0"))
    total = torch.cuda.get_device_properties(0).total_memory
    _need(0 < cap <= min(74*2**30,total), "Finite current GPU0 allocator cap required")
    torch.cuda.set_per_process_memory_fraction(cap/total,0)
    runtime = fresh.FreshTreeLoom(registry)
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
    _need(not optimizer.state,"Fresh independent optimizer required")
    previous=[None]
    def checkpoint(epoch,steps,actual_model,actual_optimizer):
        resource_check("TreeLoom_checkpoint_write")
        payload=dict(schema="treeloom_fresh_full_checkpoint_v1",epoch=epoch,complete_parent_updates=steps,
                     model_seed=70001,parent_registry_ref=registry.reference,observation_barrier_ref=barrier_ref,
                     settings=copy.deepcopy(SETTINGS),contract=actual_model.checkpoint_contract(),
                     model_state=_cpu_snapshot(actual_model.state_dict()),
                     optimizer_state=_cpu_snapshot(actual_optimizer.state_dict()),
                     VAL_truth_read=False,TEST_truth_read=False,checkpoint_selection="fixed_final_epoch30")
        path=out/"checkpoints"/("epoch_%02d.pt"%epoch)
        with path.open("xb") as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        _need(path.stat().st_size<=8*2**20,"Finite full model/optimizer checkpoint")
        ref=_hash(path)
        if previous[0] is not None:
            old,old_ref=previous[0]
            _need(old.parent==path.parent and not old.is_symlink() and _hash(old)==old_ref,
                  "Rotate only the exact previous new-scope checkpoint")
            old.unlink()
        previous[0]=(path,ref)
        event("checkpoint_saved",dict(epoch=epoch,optimizer_updates=steps,checkpoint_ref=ref,
                                     retention="keep1_new_scope_only"))
        return ref
    started=time.monotonic()
    result=train_loop(registry=registry,runtime=runtime,model=model,optimizer=optimizer,
                      train_inputs=spec["train_inputs"],observation_barrier=barrier,
                      load_parent=load_parent,join_targets=join,checkpoint=checkpoint,event=event,
                      resource_check=resource_check)
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
                  final_saved_optimizer_parameter_states=49,final_saved_optimizer_steps=[600],
                  checkpoint_retention="only_final_epoch30_file_retained_earlier_event_hashes_preserved",
                  observed_cache_cap_bytes=spec["observation_cache_bytes"],
                  memory_envelope=copy.deepcopy(spec["memory_envelope"]),
                  elapsed_training_seconds=time.monotonic()-started,
                  actual_worker_wait_and_owned_cleanup_still_required=True)
    write_json(out/"training_completed.json",result)
    return result
