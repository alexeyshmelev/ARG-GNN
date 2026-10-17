"""No scientific imports before the actual parent gate, full source/input and lease/title checks."""
from pathlib import Path
import hashlib, json, os, resource, stat, sys, time
from . import contract_v1 as c
ROOT = c.ROOT

def stable_reference_v1(path):
    path = c.local(path)
    before = path.lstat()
    c.require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'Own single-link regular scientific product')
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)

    def stamp(s):
        return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
    try:
        c.require(stamp(os.fstat(fd)) == stamp(before), 'Stable scientific output open identity')
        total = 0
        while True:
            block = os.read(fd, 8 * 2 ** 20)
            if not block:
                break
            total += len(block)
            h.update(block)
        c.require(total == before.st_size and stamp(os.fstat(fd)) == stamp(before) == stamp(path.lstat()), 'Scientific output stable during streaming authentication')
    finally:
        os.close(fd)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest())

def check_external_prerequisites_v1(q):
    c.election_values_v1(q['cell_election_ref'])
    if q['phase'] in c.GPU_PHASES:
        c.peer_values_v1(q['peer_launch_capsule_ref'])
    return True

class GuardedWriter:

    def __init__(self, s, q, g, storage):
        self.path = c.local(s['output_directory'])
        self.s = s
        self.q = q
        self.g = g
        self.storage = storage

    def boundary(self, event, *, required_write_bytes=0):
        c.active_amendment_v1()
        c.charged_census_v1(self.s, required_write_bytes=required_write_bytes)
        self.g.budget_check_v9(self.storage, boundary=True, required_write_bytes=required_write_bytes)
        c.active_amendment_v1()

    def save(self, name, value, kind=None):
        c.require(type(name) is str and Path(name).name == name, 'Flat own output only')
        path = self.path / name
        c.require(not path.exists(), 'Exclusive new scientific output, no replay')
        if kind is None or kind == "json":
            if name == 'completed.json':
                c.require(value.get('child_input_ref', c.current_amendment_ref_v1()) == c.current_amendment_ref_v1(), 'Actual completion belongs to this exact phase input')
                value = dict(value, child_input_ref=c.current_amendment_ref_v1())
            value = dict(value, actual_execution_source_manifest_ref=c.ref(c.MANIFEST), actual_ROOT_phase_input_ref=c.current_amendment_ref_v1())
            if value.get("status")in("sealed_C3_original_heldout_predictions","sealed_C3_original_remaining_cells"):
                value=dict(value,CPU_preprocessing_component_ref=c.pins_v1()["feature_component"],
                  CPU_preprocessing_adaptation="accepted_v3_compensated_FP32",bitwise_original_feature_equivalence_claimed=False)
            encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
            c.require(len(encoded) <= 8 * 2 ** 20, 'Bounded true scientific metadata')
            self.boundary('before_JSON_write', required_write_bytes=len(encoded) + 8192)
            result = c.save(self.path, name, value)
        elif kind == 'npz':
            import numpy as np
            c.require(all((not a.dtype.hasobject for a in value.values())), 'Numerical truth-free output arrays')
            size = sum((a.nbytes for a in value.values()))
            bound = size + (size + 99) // 100 + 2 ** 20
            self.boundary('before_NPZ_write', required_write_bytes=bound + 8192)
            with path.open('xb') as stream:
                np.savez_compressed(stream, **value)
            c.require(path.stat().st_size <= bound, 'Actual serializer output bound')
            result = stable_reference_v1(path)
        else:
            raise ValueError('Only actual small JSON/NPZ writes through guarded writer. Author tensor/checkpoint writes stay original.')
        self.boundary('after_scientific_write')
        return result

def completion_v1(writer, input_ref, s, q, result, *, status, epochs):
    return writer.save('completed.json', dict(status=status, parent_id=c.PARENT, original_VAL_parent_id=q['parent_id'], child_input_ref=input_ref, stage=s['phase'], epochs=epochs, result_ref=result, training_completion_ref=c.tref_v1(), cell_election_ref=q['cell_election_ref'], query_ancestry_truth_opened=False, actual_worker_wait_and_consumed_outer_still_pending=True, no_storage_close_or_credit=True))


def feature_cache_bounds_v6(plan):
    c.require(plan["haplotypes"]==1600 and plan["focal_records"]==10000
      and plan["sequence_length_bp"]==30000000 and plan["all_trees_and_samples_retained"]is True,
      "Complete original C3 support")
    upper=256*sum(b["nodes"]for b in plan["blocks"])+32*plan["trees"]*1600+1856*plan["sites"]
    c.require(upper<=64*2**30,"Complete CPU feature tensor bound")
    return dict(cache_tensor_upper_bound_bytes=upper,cache_tensor_cap_bytes=64*2**30,
      layout_tensor_cap_bytes=32*2**20,auxiliary_tensor_cap_bytes=8*2**20,
      feature_temporary_allowance_bytes=64*2**20,Python_allowance_bytes=2*2**30,
      cached_blocks=len(plan["blocks"]),original_CPU_RSS_cap_bytes=128*2**30,full_RSS_fit_measured=False)

def prepared_loaders_v6(plan_ref,fast):
    import copy,types
    from arggnn import american_c3_gnn_dataset_isolated_v2 as ds
    from research.reviews.C3_CPU_prepared_features_candidate_20261017_v1 import prepared_cpu_features_v3 as prep
    c.require(c.ref(prep.__file__)==c.pins_v1()["feature_component"],"Exact accepted feature source")
    plan=c.read(plan_ref);bounds=feature_cache_bounds_v6(plan)
    layouts=prep.ImmutableTreeLayoutCacheV1(max_bytes=32*2**20,max_entries=32)
    cache=prep.ImmutableBlockFeatureCacheV1(layout_cache=layouts,max_tensor_bytes=64*2**30,max_entries=len(plan["blocks"]))
    shared=dict(state=None,ts_identity=None,frames={},aux_bytes=0,builds=0,hits=0)
    names=("parent_id","plan_ref","block_index","start_tree","stop_tree","sequence_length_bp","complete_parent",
      "genomic_geometry","focal_rows","focal_hold","focal_sample_column","observations_receipt","worker_pid")
    def source_signature(index,state):
        a=plan["blocks"][index]["start_tree"];b=plan["blocks"][index]["stop_tree"];breaks=state["breaks"]
        lo,hi=ds.np.searchsorted(state["positions"],[breaks[a],breaks[b]])
        rows=ds.np.flatnonzero((state["focal_tree"]>=a)&(state["focal_tree"]<b))
        return prep._digest_values([("spans_with_neighbors",state["spans"][max(0,a-1):min(len(state["spans"]),b+1)]),
          ("focal_rows",rows),("focal_holds",state["focal_tree"][rows]),
          ("focal_samples",state["focals"]["sample_indices"][rows]),("focal_positions",state["focals"]["positions_bp0"][rows]),
          ("focal_nodes",state["focals"]["sample_nodes"][rows]),
          ("intervals",ds.np.stack((breaks[a:b],breaks[a+1:b+1]),1)),
          ("genotype",state["genotype"][lo:hi]),("allele_count",state["allele_count"][lo:hi]),
          ("site_positions",state["positions"][lo:hi]),("reference_class",state["labels"]),
          ("sample_age",state["ages"]),("query_columns",state["query_columns"])])
    def auxiliary_signature(metadata):
        return prep._digest_values((name,metadata[name].numpy())for name in ds.BLOCK_TENSORS)
    class PreparedDataset(ds.AmericanC3WholeTreeDatasetV1):
        def __getitem__(self,index):
            fast("before_prepared_CPU_block")
            if shared["state"]is None:
                state=ds.load_observation(self.plan["observation_ref"],self.plan["focal_registry_ref"],self.plan["config"])
                c.require(len(state["query_columns"])==1000 and int((state["labels"]<0).sum())==1000,"Original1000masked queries")
                shared["state"]=state;shared["ts_identity"]=id(state["ts"])
            state=shared["state"];c.require(id(state["ts"])==shared["ts_identity"],"Same immutable tskit sequence")
            self._state=state;self._pid=os.getpid()
            logical=int(index)if self.direction=="forward"else len(self)-1-int(index)
            key=(plan["observation_ref"]["sha256"],plan_ref["sha256"],logical);signature=source_signature(logical,state)
            if logical in shared["frames"]:
                frame=shared["frames"][logical];record=cache.values[key]
                c.require(signature==frame["input_signature"] and prep._bank_digest(record["bank"])==record["bank_digest"]
                  and auxiliary_signature(frame["metadata"])==frame["aux_signature"],"Exact cached observation/focal/geometry content")
                shared["hits"]+=1
                # PyG Data.to mutates its wrapper, so construct a fresh wrapper.
                return ds.AmericanC3BlockV1(record["bank"],**copy.copy(frame["metadata"]))
            a=plan["blocks"][logical]["start_tree"];b=plan["blocks"][logical]["stop_tree"]
            keys=[(plan["observation_ref"]["sha256"],plan_ref["sha256"],i)for i in range(a,b)]
            def encode(trees,*args,**kwargs):
                n=max(len(t.parent)for t in trees);tile=min(32,(64*2**20-96*n)//(512*(n+1601)))
                c.require(tile>=1,"Complete tree within fixed scratch allowance")
                return cache.get_or_prepare(key,trees,*args,tree_keys=keys,site_tile=tile,max_working_bytes=64*2**20,**kwargs)
            namespace=dict(ds.AmericanC3WholeTreeDatasetV1.__getitem__.__globals__,encode_block=encode)
            block=types.FunctionType(ds.AmericanC3WholeTreeDatasetV1.__getitem__.__code__,namespace)(self,index)
            c.require(len(cache.values)==shared["builds"]+1,"Full parent cache without eviction")
            metadata={name:(getattr(block,name).clone()if name in ds.BLOCK_TENSORS else copy.deepcopy(getattr(block,name)))for name in names}
            shared["aux_bytes"]+=sum(metadata[name].numel()*metadata[name].element_size()for name in ds.BLOCK_TENSORS)
            c.require(shared["aux_bytes"]<=8*2**20,"All original focal/geometry tensors within fixed cap")
            shared["frames"][logical]=dict(input_signature=signature,metadata=metadata,aux_signature=auxiliary_signature(metadata))
            shared["builds"]+=1;fast("after_prepared_CPU_block");return block
    datasets=[PreparedDataset(plan_ref,direction=d,worker_tmp_directory=Path(os.environ["TMPDIR"]))for d in("forward","reverse")]
    loaders=[GuardedLoaderV1(ds.make_dataloader(d,num_workers=0,pin_memory=False),fast)for d in datasets]
    return datasets,loaders,dict(bounds=bounds,shared=shared,features=cache,layouts=layouts)

def feature_receipt_v6(value):
    s=value["shared"]
    c.require(s["builds"]==s["hits"]==value["bounds"]["cached_blocks"],"All original blocks built once/reused once")
    return dict(schema="C3_v6_observed_CPU_feature_reuse_v1",component_ref=c.pins_v1()["feature_component"],
      component_review_ref=c.pins_v1()["feature_review"],bounds=value["bounds"],
      prepared_blocks=s["builds"],reused_reverse_blocks=s["hits"],actual_feature_tensor_bytes=value["features"].bytes,
      actual_auxiliary_tensor_bytes=s["aux_bytes"],layout_tensor_bytes=value["layouts"].bytes,
      original30Mb_1600samples_1000queries_10000focals=True,all_original_trees_sites_and_recurrence_retained=True,
      learned_embeddings_cached=False,FP32_compensated_support_adaptation=True,full_parent_speedup_claimed=False,
      inherited_C3_block_requires_all_sampled_nodes_to_be_leaves=True)

def run_prediction_v1(input_ref, s, q, cfg, storage, ticket, native):
    g = c.guard_v1(q)
    writer = GuardedWriter(s, q, g, storage)

    def fast(event='C3_remaining_boundary'):
        c.active_amendment_v1()
        c.require_work_window_v2(s)
        native.apply_and_validate_title()
        lease = native.validate_lease()
        c.require(lease['physical_gpu'] == 0 and lease['visible_gpu_uuid'] == c.GPU0_UUID, 'Exact GPU0 lease/title on every block boundary')
        c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= c.CAPS['max_cpu_rss_bytes'], 'Bounded original128GiB worker RSS')
        if 'torch' in sys.modules:
            c.require(sys.modules['torch'].cuda.max_memory_reserved(0) <= 8 * 2 ** 30, 'Strict8GiB allocator remains bounded')
    fast('before_science_import')
    c.require(not writer.path.exists(), 'Fresh remaining-cell output')
    writer.boundary('before_output_mkdir', required_write_bytes=8192)
    writer.path.mkdir()
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 2 ** 20, 8 * 2 ** 20))
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
    import numpy as np, torch, torch_geometric
    c.require(str(torch.__version__) == '2.6.0+cu124' and str(torch_geometric.__version__) == '2.7.0' and (torch.cuda.device_count() == 1), 'Exact original shared Torch/PyG and one visible leased device')
    device = torch.device('cuda', 0)
    torch.cuda.set_per_process_memory_fraction(8 * 2 ** 30 / torch.cuda.get_device_properties(device).total_memory, device)
    from arggnn.american_c3_common_v1 import runtime_report
    expected = dict(c.read(c.RUNTIME_REF), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    c.require(runtime_report() == expected, 'Original shared runtime, disclosed GPU visibility')
    from arggnn.american_c3_gnn_shared30mb_isolated_v2 import MODEL_KWARGS, validate_shared_plan_v1, validate_shared_metadata_v1, seal_shared_val_prediction_v1
    from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
    from arggnn.american_c3_gnn_dataset_isolated_v2 import AmericanC3WholeTreeDatasetV1, make_dataloader
    from arggnn.american_c3_gnn_predict_isolated_v2 import completed_training_v1
    (bundle, training) = c.training_v1()
    training_ref = bundle['training_completion_ref']
    c.require(completed_training_v1(training_ref) == training, 'Same accepted immutable original saved600 family')
    (_, retained, pending) = c.election_values_v1(q['cell_election_ref'])
    parent = q['parent_id']
    plan_ref = q['plan_evidence']['plan_ref']
    plan = c.read(plan_ref)
    validate_shared_plan_v1(plan, parent_id=parent)
    validate_shared_metadata_v1(plan['config'], parent_id=parent)
    c.require(plan['haplotypes'] == 1600 and plan['sequence_length_bp'] == 30000000 and (plan['focal_records'] == 10000), 'All original30Mb/1600/10000 support')
    with torch.device('cpu'):
        model = AmericanC3WholeTreeGNNV1(**MODEL_KWARGS)
    model = model.to(device)
    seals = []
    for candidate in training['checkpoint_candidates']:
        if candidate['epoch'] not in q['pending_epochs']:
            continue
        fast('before_original_pending_cell')
        c.require(candidate['epoch'] in pending[parent] and candidate['epoch'] not in retained[parent], 'No answered cell may be rerun')
        datasets, loaders, feature_cache = prepared_loaders_v6(plan_ref, fast)
        try:
            seals.append(seal_shared_val_prediction_v1(model, *loaders, parent_id=parent, checkpoint_ref=candidate['checkpoint_ref'], training_completion_ref=training_ref, writer=writer, device=device, runtime_guard=fast))
            writer.save('prepared_CPU_features_epoch%03d.json'%candidate['epoch'], feature_receipt_v6(feature_cache))
        finally:
            for d in datasets:
                d._state = None
                d._pid = None
            del feature_cache
    del model
    result = writer.save(parent + '_remaining_prediction_seals.json', dict(status='sealed_C3_original_remaining_cells', parent_id=parent, epochs=q['pending_epochs'], training_completion_ref=training_ref, prediction_seal_refs=seals, retained_prediction_seal_refs=list(retained[parent].values()), cell_election_ref=q['cell_election_ref'], private_truth_opened=False, full1600_graph_and_recurrent_columns=True, common_focal_records=10000, tree_encoding_passes=2, device='cuda:0', CPU_bitwise_equivalence_claimed=False, all30_before_truth_still_required=True, TEST_deferred=True))
    return completion_v1(writer, input_ref, s, q, result, status='completed_C3_remaining_cells', epochs=q['pending_epochs'])

class GuardedLoaderV1:

    def __init__(self, loader, check):
        (self.loader, self.check) = (loader, check)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        self.check('before_original_loader_iteration')
        for block in self.loader:
            self.check('before_original_whole_tree_inference_block')
            yield block
            self.check('after_original_whole_tree_inference_block')

def new_stage_values_v1(e, parent, election, pending):
    return c.new_stage_values_v1(e, parent, election, pending)

def run_matrix_v1(input_ref, s, q, cfg, storage):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules and ('torch_geometric' not in sys.modules), 'CPU-only fresh matrix entry')
    (_, retained, pending) = c.election_values_v1(q['cell_election_ref'])
    needed = {p for p in c.VAL if pending[p]}
    c.require(set(q['new_stage_evidence']) == needed, 'Exactly actual new pending-parent stages, no fallback or missing cells')
    chosen = {p: dict(retained[p]) for p in c.VAL}
    for parent in needed:
        new = new_stage_values_v1(q['new_stage_evidence'][parent], parent, q['cell_election_ref'], pending[parent])
        c.require(not set(chosen[parent]) & set(new), 'Exactly one chosen producer per original cell')
        chosen[parent].update(new)
    c.require(all((set(chosen[p]) == set(c.EPOCHS) for p in c.VAL)), 'All original5parents x6epochs before truth')
    original_plans = c.validate_all_original_plan_joins_v6(chosen)
    g = c.guard_v1(q)
    writer = GuardedWriter(s, q, g, storage)
    c.require(not writer.path.exists(), 'Fresh metadata matrix output')
    writer.boundary('before_matrix_mkdir', required_write_bytes=8192)
    writer.path.mkdir()
    sys.path.insert(0, str(ROOT / 'src'))
    from arggnn.american_c3_gnn_predict_isolated_v2 import seal_validation_matrix_v1
    refs = [chosen[p][epoch] for p in c.VAL for epoch in c.EPOCHS]
    writer.boundary('before_unchanged_original30cell_barrier')
    result = seal_validation_matrix_v1(refs, training_completion_ref=c.tref_v1(), writer=writer)
    writer.save('cellwise_producer_provenance.json', dict(cell_election_ref=q['cell_election_ref'], retained_cells=retained, new_stage_evidence=q['new_stage_evidence'], one_authenticated_chosen_seal_per_original_cell=True, original_nonzero_waits_preserved=True, original_matrix_barrier_source_unchanged=True, private_truth_opened=False, per_parent_source_policy=c.per_parent_source_policy_v6(), all_original_plan_refs=original_plans, fresh_same_source_VAL000_VAL001_VAL002_disclosed=True, prior_v5_stopped_failure_ref=q["prior_v5_stop_ref"], prior_v5_known_targets_ref=q["prior_v5_known_targets_ref"], prior_v5_resolution_ref=q["prior_v5_resolution_ref"], old_v5_historical_descendant_drain="UNPROVEN", old_v5_full_charge_preserved=True, state_or_output_migration=False))
    return completion_v1(writer, input_ref, s, q, result, status='completed_C3_remaining_cellwise_matrix', epochs=[])
