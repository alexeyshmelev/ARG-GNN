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
        if kind is None:
            if name == 'completed.json':
                c.require(value.get('child_input_ref', c.current_amendment_ref_v1()) == c.current_amendment_ref_v1(), 'Actual completion belongs to this exact phase input')
                value = dict(value, child_input_ref=c.current_amendment_ref_v1())
            value = dict(value, actual_execution_source_manifest_ref=c.ref(c.MANIFEST), actual_ROOT_phase_input_ref=c.current_amendment_ref_v1())
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
        datasets = [AmericanC3WholeTreeDatasetV1(plan_ref, direction=d, worker_tmp_directory=Path(os.environ['TMPDIR'])) for d in ('forward', 'reverse')]
        loaders = [GuardedLoaderV1(make_dataloader(d, num_workers=0, pin_memory=False), fast) for d in datasets]
        try:
            seals.append(seal_shared_val_prediction_v1(model, *loaders, parent_id=parent, checkpoint_ref=candidate['checkpoint_ref'], training_completion_ref=training_ref, writer=writer, device=device, runtime_guard=fast))
        finally:
            for d in datasets:
                d._state = None
                d._pid = None
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
    writer.save('cellwise_producer_provenance.json', dict(cell_election_ref=q['cell_election_ref'], retained_cells=retained, new_stage_evidence=q['new_stage_evidence'], one_authenticated_chosen_seal_per_original_cell=True, original_nonzero_waits_preserved=True, original_matrix_barrier_source_unchanged=True, private_truth_opened=False))
    return completion_v1(writer, input_ref, s, q, result, status='completed_C3_remaining_cellwise_matrix', epochs=[])
