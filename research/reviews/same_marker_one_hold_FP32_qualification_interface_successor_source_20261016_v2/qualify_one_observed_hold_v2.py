"""One observed TRAIN000 hold, FP32 arithmetic qualification, ROOT calls only.

Import/default is inert stdlib. No optimizer, labels, VAL/TEST, checkpoint,
full-parent numerical bank/layout builder, CUDA or output-file writer exists.
ROOT supplies its already reviewed CPU guard and compact exclusive writer.
"""
from pathlib import Path
import bisect, hashlib, importlib.util, json, os, resource, sys, time, zipfile

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
PARENT = 'ap_expanded_train_000'
MANIFEST = {'path':'data/expanded_full_panel_pyg_cache_v5/cache/manifest.json','sha256':'6144bcc237cf2916fa8b4c90964f6633fa56989d73e2c8f4295675d2239469e8'}
ARRAY = {'path':'data/expanded_full_panel_pyg_cache_v5/cache/ap_expanded_train_000.npz','sha256':'9f3e7c7cfa4e13b43105457b49184e907723c3e41f18b538fd8c0aa8cf3520b4'}
ENTRY = {'path':'data/expanded_full_panel_v1/ap_expanded_train_000/full_panel_observed_entry.json','sha256':'6e95dd61543187aa9e69db01ee4449f5e4cd1ac4f8b708c79d465b02527173a4'}
CORE = {'path':'research/reviews/biological_operator_next_fresh_20261016_v1/packed_csr_v2/same_marker_packed_csr_v2.py','sha256':'081acadf8f55adec0a5d16c6430c30fe5771ee4a307f5387d32138e1659a5e61'}
NULL = {'path':'research/reviews/biological_operator_next_fresh_20261016_v1/source_handoff/same_marker_allele_support_v1.py','sha256':'a933d39033312acbccc74f63e5c76789b220ae530d983d2935d81fd681e37490'}
PLAN = {'path':'research/reviews/same_marker_allele_support_independent_review_20261016_v1/QUALIFICATION_PLAN.md','sha256':'462221416f1111448defb283c6f52a2b217575e807c8f37d001756e0d8295494'}
ATOL, RTOL, LOG_ATOL, GRAD_ATOL, GRAD_RTOL, FD_STEP = 5e-5, 2e-5, 1e-3, 1e-5, 2e-3, 1e-3

def require(ok, message):
    if not ok:
        raise ValueError(message)

def authenticated_path(record, max_bytes):
    require(set(record) == {'path','sha256'}, 'Exact explicit source/observation ref')
    p = ROOT/record['path']
    require(p.resolve() == p and p.is_file() and not p.is_symlink()
            and not Path(record['path']).is_absolute() and p.is_relative_to(ROOT)
            and '..' not in Path(record['path']).parts and p.stat().st_size <= max_bytes,
            'Canonical bounded fixed observed/source path')
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for raw in iter(lambda: f.read(1048576), b''):
            digest.update(raw)
    require(digest.hexdigest() == record['sha256'], 'Source/observed SHA mismatch: '+record['path'])
    return p

def read_json(record):
    return json.loads(authenticated_path(record, 128*2**20).read_text())

def source_module(record, name):
    path = authenticated_path(record, 2**20)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_qualification_v1(*, check_resources_v1, save_compact_receipt_v1):
    """Called once inside ROOT's actual600s/8GiB CPU admission/supervisor.

    check_resources_v1() is ROOT's original guarded resource check. The writer
    exclusively saves ONE small JSON receipt inside that actual reserved scope.
    Neither callback grants training or extra data authority. Failure is raised
    after the real partial receipt, so the supervisor receives failure status.
    """
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'ROOT cwd and explicit CPU-only execution')
    require(callable(check_resources_v1) and callable(save_compact_receipt_v1), 'ROOT existing guard/writer callbacks required')
    started = time.monotonic()
    receipt = dict(schema='same_marker_one_observed_TRAIN000_FP32_qualification_v1', parent_id=PARENT,
        seed=101, precision='FP32', maximum_seconds=600, owned_RSS_cap_bytes=8*2**30,
        fixed_source_refs=dict(plan=PLAN,packed_core=CORE,original_null_builder=NULL),
        fixed_observed_refs=dict(manifest=MANIFEST,archive=ARRAY,entry=ENTRY),
        optimizer_updates=0, ancestry_targets_read=False, checkpoint_reads=False,
        VAL_TEST_payload_reads=False, full_parent_layout_created=False, CUDA_used=False,
        tolerance=dict(ordinary_atol=ATOL, ordinary_rtol=RTOL, log_atol=LOG_ATOL,
                       gradient_atol=GRAD_ATOL, gradient_rtol=GRAD_RTOL, finite_difference_step=FD_STEP), checks={})
    def guard():
        require(time.monotonic()-started <= 600, 'One600s qualification budget exhausted')
        check_resources_v1()
    def fail(message):
        raise ValueError(message)
    def record_close(name, actual, expected, *, atol=ATOL, rtol=RTOL):
        guard()
        require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32,
                name+': exact shape/FP32 contract')
        require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), name+': nonfinite value')
        error = (actual-expected).abs()
        allowed = atol+rtol*expected.abs()
        receipt['checks'][name] = dict(max_abs_error=float(error.max()) if error.numel() else 0.,
            violations=int((error > allowed).sum()), elements=error.numel(), atol=atol, rtol=rtol)
        require(not bool((error > allowed).any()), name+': frozen tolerance violated')
    def summary_close(name, actual, expected):
        ordinary = list(range(8))+list(range(16,32))+list(range(40,48))
        masses = list(range(8,16))+list(range(32,40))
        record_close(name+'_ordinary',actual[:,ordinary],expected[:,ordinary])
        record_close(name+'_log_mass',actual[:,masses],expected[:,masses],atol=LOG_ATOL)
    def finite(name, value):
        require(value.dtype == torch.float32 and value.device.type == 'cpu'
                and bool(torch.isfinite(value).all()), name+': finite CPU FP32 required')
    def describe(value):
        value = value.detach().flatten()
        return dict(count=value.numel(), mean=float(value.mean()) if value.numel() else 0.,
            minimum=float(value.min()) if value.numel() else 0.,
            p05=float(torch.quantile(value,.05)) if value.numel() else 0.,
            median=float(value.median()) if value.numel() else 0.,
            p95=float(torch.quantile(value, .95)) if value.numel() else 0.,
            maximum=float(value.max()) if value.numel() else 0.)
    try:
        guard()
        authenticated_path(PLAN, 2**20)
        import numpy as np
        import torch
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(8)
        torch.manual_seed(101)
        torch.use_deterministic_algorithms(True)
        manifest = read_json(MANIFEST)
        require(manifest['schema'] == 'mutation_routed_full_panel_observed_cache_v1'
            and manifest['observations_only'] is True and manifest['targets_cached'] is False
            and manifest['pickle_used'] is False and manifest['receipt']['targets_read'] is False,
            'Authenticated observation-only bank contract')
        require(manifest['entry_refs'][PARENT] == ENTRY and manifest['parents'][PARENT]['array_ref'] == ARRAY,
            'Exact registered TRAIN000 archive/entry only')
        rows = [r for r in manifest['registry'] if r['parent_id'] == PARENT]
        require(len(rows) == 1 and rows[0]['split'] == 'train' and rows[0]['downstream_role'] == 'TRAIN', 'One original TRAIN parent')
        for source_ref in manifest['source_refs'].values():
            authenticated_path(source_ref, 2**20)
        meta = manifest['parents'][PARENT]
        provenance = meta['observations_receipt']
        require(provenance['entry_ref'] == ENTRY and provenance['parent_id'] == PARENT
            and provenance['split'] == 'train' and provenance['targets_read'] is False
            and provenance['observations_only'] is True and provenance['input_samples'] == 952
            and provenance['actual_query_readouts'] == 686 and provenance['labeled_pure_proxies'] == 266,
            'Original complete observed receipt binding')
        entry = read_json(ENTRY)
        require(entry['parent_id'] == PARENT and entry['split'] == 'train'
            and entry['observations_only'] is True and entry['truth_or_oracle_read'] is False,
            'Exact original observation lineage')
        observation = read_json(entry['refs']['observation_manifest'])
        inference = read_json(entry['refs']['inference_receipt'])
        roster = read_json(entry['refs']['source_roster'])
        derived = read_json(entry['refs']['derived_roster'])
        coordinates = read_json(entry['refs']['site_coordinates'])
        authenticated_path(entry['original_native_inference_receipt'], 16*2**20)
        require(observation['parent_id'] == inference['parent_id'] == PARENT
            and observation['census_or_ancestry_annotations_present'] is False
            and observation['cleaned_oracle_exported'] is False and inference['truth_or_oracle_read'] is False
            and inference['native_sample_dates_unchanged'] is True
            and inference['coordinate_contract'] == entry['coordinate_contract']
            and inference['original_observed_sample_manifest'] == entry['refs']['source_roster']
            and inference['observed_sample_manifest'] == entry['refs']['derived_roster']
            and inference['inferred_tree'] == entry['refs']['native_tree']
            and inference['native_inference_receipt'] == entry['original_native_inference_receipt'],
            'Observed source/derived roster/site/native-inference provenance')
        require(len(roster) == len(derived) == 952 and [r['sample_index'] for r in roster] == list(range(952)),
                'All original952 phased observed columns in declared order')
        native_mapping = inference['sample_index_to_inferred_tree_node_id']
        require(type(native_mapping) is list and len(native_mapping) == 952
            and all(type(row) is dict and set(row) == {'sample_index','inferred_tree_node_id'}
                and type(row['sample_index']) is int and type(row['inferred_tree_node_id']) is int
                and row['inferred_tree_node_id'] >= 0 for row in native_mapping)
            and [row['sample_index'] for row in native_mapping] == list(range(952)),
            'Exact original952 typed ordered native sample mapping')
        for old, new, mapped in zip(roster, derived, native_mapping):
            require(all(old[k] == new[k] for k in ['sample_id','sample_index','haplotype','age_generations','role','observed_reference_class'])
                and type(old['sample_index']) is int and type(new['sample_index']) is int
                and old['sample_index'] == new['sample_index'] == mapped['sample_index']
                and new['source_observed_node_id'] == old['observed_node_id']
                and type(new['inferred_tree_node_id']) is int
                and new['inferred_tree_node_id'] == mapped['inferred_tree_node_id'],
                'Exact observed sample/date/phase join')
        archive_path = authenticated_path(ARRAY, 128*2**20)
        require(archive_path.stat().st_size == meta['compressed_bytes'], 'Exact compressed observed byte count')
        with zipfile.ZipFile(archive_path) as z:
            require(sum(r.file_size for r in z.infolist()) <= 2**30, 'One bounded observed archive expansion')
        spec = meta['model_spec']
        with np.load(archive_path, allow_pickle=False) as archive:
            wanted = {k for value in spec.values() for k in (value if isinstance(value,list) else [value])}
            wanted.update(v['array_key'] for v in meta['scoring'].values() if 'array_key' in v)
            require(set(archive.files) == wanted, 'Only original observation/scoring-coordinate fields in archive')
            # Integer bank arrays only. No from_arrays, full bank loader, forest
            # layout builder, tree decoder, scoring target or model state.
            names = ['parent','sample_index','site_hold','genotype','reference_class','sample_age','query_column']
            arrays = {name:archive[spec[name]] for name in names}
            parent, samples, site_hold, full_gt = (arrays[k] for k in ['parent','sample_index','site_hold','genotype'])
            rc, age, queries = (arrays[k] for k in ['reference_class','sample_age','query_column'])
            require(parent.ndim == 1 and samples.ndim == 2 and samples.shape[1] == 952
                and full_gt.shape == (len(site_hold),952) and len(queries) == 686,
                'Complete original tree/sample/site/query shapes')
            require(all(a.dtype.kind in 'iu' for k,a in arrays.items() if k != 'sample_age')
                and np.isfinite(age).all() and ((full_gt >= -1)&(full_gt < 4)).all()
                and np.array_equal(queries, [r['sample_index'] for r in roster if r['role'] == 'query'])
                and np.array_equal(rc, [-1 if r['observed_reference_class'] is None else r['observed_reference_class'] for r in roster])
                and np.array_equal(age, [r['age_generations'] for r in roster]), 'Original called categorical/observed role/date joins')
            groups = 4*(age >= 350).astype(np.int64)+rc
            refs = np.flatnonzero(rc >= 0)
            require(len(refs) == 266 and np.all(rc[queries] == -1)
                and [int(np.count_nonzero(groups[refs] == g)) for g in range(8)] == [48,18,90,76,2,10,12,10],
                'All original pure references and unknown queries')
            prep = provenance['preprocessing']
            intervals = prep['intervals_bp0']
            require(len(intervals) == len(samples) and all(a[1] == b[0] for a,b in zip(intervals,intervals[1:])),
                    'Original contiguous native genomic hold order')
            ends = [r[1] for r in intervals]
            original_sites = coordinates['rows']
            covered_sites = [r for r in original_sites if intervals[0][0] <= r['tree_position_0based'] < intervals[-1][1]]
            require(len(covered_sites) == len(site_hold) and len(full_gt) == len(covered_sites)
                and np.array_equal(site_hold, [bisect.bisect_right(ends,r['tree_position_0based']) for r in covered_sites]),
                'Exact original SNP index/coordinate/native-hold membership')
            counts = np.bincount(site_hold, minlength=len(samples))
            chosen = None
            rejected = dict(site_count=0,nodes=0,polymorphism=0)
            for hold, count in enumerate(counts.tolist()):
                guard()
                if not 2 <= count <= 16:
                    rejected['site_count'] += 1
                    continue
                # Union of the952 actual sample-to-root chains. At most4096
                # nodes are retained before ANY packed preparation is invoked.
                native = set()
                for sample in samples[hold].tolist():
                    node, steps = int(sample), 0
                    while node >= 0 and node not in native:
                        require(0 <= node < len(parent) and steps <= 4096, 'Observed tree chain bounds/cycle')
                        native.add(node)
                        node, steps = int(parent[node]), steps+1
                lo, hi = min(native), max(native)+1
                if len(native) > 4096:
                    rejected['nodes'] += 1
                    continue
                require(len(native) == hi-lo and np.all((parent[lo:hi] == -1)|((parent[lo:hi] >= lo)&(parent[lo:hi] < hi))),
                    'Complete contiguous original tree component, no vertex/unary deletion')
                index = np.flatnonzero(site_hold == hold)
                window = full_gt[index]
                informative_site = any(np.any(row[queries] >= 0) and any(
                    len(set(row[refs[groups[refs] == g]].tolist())-{-1}) >= 2 for g in range(8)) for row in window)
                if not informative_site:
                    rejected['polymorphism'] += 1
                    continue
                chosen = (hold,lo,hi,index,window.copy())
                break
            require(chosen is not None, 'INCONCLUSIVE: no first eligible observed TRAIN000 hold, no parent/seed search')
            hold, lo, hi, indices, gt_np = chosen
            lp = parent[lo:hi].copy()
            lp[lp >= 0] -= lo
            sample = samples[hold:hold+1].copy()-lo
            require(np.count_nonzero(lp < 0) == 1 and len(set(sample[0].tolist())) == 952, 'One complete original tree root/leaf set')
            degree = np.bincount(lp[lp >= 0], minlength=len(lp))
            require(set(np.flatnonzero(degree == 0).tolist()) == set(sample[0].tolist()), 'All and only952 original sampled leaves')
            geometry_np = archive[spec['edge_geometry']][lo:hi].copy()
            support_np = archive[spec['branch_support_true']][lo:hi].copy()
        # Full-parent integer arrays never become a parent layout or model bank.
        del arrays, parent, samples, site_hold
        rc_t = torch.as_tensor(rc.copy(),dtype=torch.long)
        age_t = torch.as_tensor(age.copy(),dtype=torch.float32)
        query_t = torch.as_tensor(queries.copy(),dtype=torch.long)
        gt = torch.as_tensor(gt_np,dtype=torch.int16)
        local_parent = torch.as_tensor(lp,dtype=torch.long)
        local_sample = torch.as_tensor(sample,dtype=torch.long)
        core = source_module(CORE, '_same_marker_frozen_packed_core')
        null_source = source_module(NULL, '_same_marker_original_null_builder_only')
        ids = {}
        diploid = torch.tensor([ids.setdefault(r['sample_id'],len(ids)) for r in roster],dtype=torch.long)
        phase = torch.tensor([r['haplotype'] for r in roster],dtype=torch.long)
        permutation, null_receipt = null_source.build_alignment_null_v1(rc_t,age_t,diploid,phase,seed=101)
        require(torch.equal(permutation[query_t],query_t), 'Null fixes all query columns')
        for identity in torch.unique(diploid).tolist():
            pair = torch.nonzero(diploid == identity,as_tuple=False).flatten()
            require(len(pair) == 2 and len(torch.unique(diploid[permutation[pair]])) == 1
                and torch.equal(phase[permutation[pair]],phase[pair]), 'Null preserves complete diploid sequence/phase pairs')
        group_t = 4*(age_t >= 350).long()+rc_t
        # Bookkeeping verifies every original marker in TRAIN000, without a
        # feature/model evaluation outside the single selected hold.
        moved_counts = []
        perm_np = permutation.numpy()
        for group in range(8):
            columns = refs[groups[refs] == group]
            moved = columns[perm_np[columns] != columns]
            for allele in [-1,0,1,2,3]:
                require(np.array_equal((full_gt[:,columns] == allele).sum(1),
                    (full_gt[:,perm_np[columns]] == allele).sum(1)), 'Null preserves every per-marker group allele/call count')
            require(all(roster[int(permutation[c])]['haplotype'] == roster[int(c)]['haplotype'] for c in columns), 'Null phase pairing')
            called = int((full_gt[:,columns] >= 0).sum())
            moved_called = int((full_gt[:,moved] >= 0).sum())
            moved_counts.append(dict(group=group,references=len(columns),moved_haplotypes=len(moved),
                all_parent_called_entries=called,all_parent_moved_called_entries=moved_called,
                selected_hold_called_entries=int((gt_np[:,columns] >= 0).sum()),
                selected_hold_moved_called_entries=int((gt_np[:,moved] >= 0).sum())))
        del full_gt
        def layout(parent_value=local_parent,sample_value=local_sample,route=None):
            return core.prepare_packed_parent_v2(parent=parent_value,sample_index=sample_value,
                site_hold=torch.zeros(gt.shape[0],dtype=torch.long),reference_class=rc_t,sample_age=age_t,
                query_column=query_t,genotype=gt,route_permutation=route).holds[0]
        aligned = layout()
        null = layout(route=permutation)
        def verify_join(tree,parent_value,sample_value):
            native = tree.native_node.long()
            inverse = torch.empty_like(native)
            inverse[native] = torch.arange(len(native))
            expected = parent_value[native]
            expected = torch.where(expected >= 0,inverse[expected.clamp_min(0)],-1)
            require(torch.equal(tree.parent.long(),expected)
                and torch.equal(native[tree.sample_node.long()],sample_value[0])
                and torch.equal(native[tree.query_node.long()],sample_value[0,query_t]), 'Packed/original topology and sampled query-index join')
        verify_join(aligned,local_parent,local_sample)
        def tensor_sha(x):
            return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
        receipt['observation_binding'] = dict(manifest_ref=MANIFEST,array_ref=ARRAY,entry_ref=ENTRY,
            source_refs=manifest['source_refs'],roster_ref=entry['refs']['source_roster'],derived_roster_ref=entry['refs']['derived_roster'],
            site_coordinate_ref=entry['refs']['site_coordinates'],inference_ref=entry['refs']['inference_receipt'],
            selected_original_hold=hold,interval_bp0=intervals[hold],original_global_node_range=[lo,hi],
            original_covered_site_indices=indices.tolist(),original_site_ids=[covered_sites[int(i)]['site_id'] for i in indices],
            original_site_coordinates_bp0=[covered_sites[int(i)]['tree_position_0based'] for i in indices],
            nodes=len(lp),samples=952,queries=686,references=266,SNPs=gt.shape[0],rejected_before_first=rejected,
            local_parent_sha256=tensor_sha(local_parent),sample_join_sha256=tensor_sha(local_sample),genotype_slice_sha256=tensor_sha(gt),
            complete_nodes_unary_edges_samples_queries_references_SNPs_retained=True,selection_uses_observations_only=True)
        receipt['null'] = dict(builder_receipt=null_receipt,permutation_sha256=tensor_sha(permutation),
            exact_seed=101,reseeded=False,sequence_join='target_reference_node <- original_source_column, same map at every SNP',
            per_stratum_called_movement=moved_counts,called_moved_mass_computed_by_this_runner=True,
            called_moved_mass_units='called haplotype-by-SNP entries, not independent donors',
            all_original_per_marker_group_allele_call_counts_preserved=True)
        v = len(lp)
        root = int(torch.nonzero(local_parent < 0,as_tuple=False)[0])
        initial = torch.ones((v,),dtype=torch.float32)*torch.tensor(.95,dtype=torch.float32).log()
        nonuniform = -.05-.002*(torch.arange(v,dtype=torch.float32)%17)
        nonroot = torch.nonzero(local_parent >= 0,as_tuple=False).flatten()
        selected_edges = nonroot[:8].tolist()
        receipt['preselected_gradient_edges_original_global'] = [lo+i for i in selected_edges]
        def evidence(logs,genotypes=gt,tree=aligned,block=16):
            return core.PackedSameMarkerEvidenceV2(block)(logs.index_select(0,tree.native_node.long()),genotypes,tree)
        def marker(logs,genotypes=gt,tree=aligned):
            return core.PackedSameMarkerEvidenceV2(16).marker_features_v2(logs.index_select(0,tree.native_node.long()),genotypes,tree)
        # Explicit path enumeration in ORIGINAL supplied integer node IDs.
        # No up/outside/segment/layout-parent helper is called by this oracle.
        pv = local_parent.tolist()
        original_samples = local_sample[0].tolist()
        ref_columns = refs.tolist()
        oracle_queries = queries[:8].tolist()
        paths = []
        for qc in oracle_queries:
            ancestors, node = {}, original_samples[qc]
            chain = []
            while node >= 0:
                ancestors[node] = len(chain)
                if pv[node] >= 0:
                    chain.append(node)
                node = pv[node]
            qpaths = []
            for rc_column in ref_columns:
                node, donor_chain = original_samples[rc_column], []
                while node not in ancestors:
                    require(node >= 0, 'Actual donor/query unique path has a common root')
                    donor_chain.append(node)
                    node = pv[node]
                qpaths.append(chain[:ancestors[node]]+donor_chain)
            paths.append(qpaths)
        def oracle(logs,genotypes=gt,route=None):
            weights = torch.stack([torch.stack([logs[torch.tensor(path,dtype=torch.long)].sum() for path in row]) for row in paths])
            out = torch.zeros((gt.shape[0],8,24),dtype=torch.float32)
            mask = torch.zeros_like(out,dtype=torch.bool)
            assigned = genotypes if route is None else genotypes[:,route]
            for site in range(gt.shape[0]):
                for qpos,qc in enumerate(oracle_queries):
                    allele = int(genotypes[site,qc])
                    if allele < 0:
                        continue
                    mask[site,qpos,16:] = True
                    for group in range(8):
                        donor = [i for i,c in enumerate(ref_columns) if int(group_t[c]) == group and int(assigned[site,c]) >= 0]
                        if not donor:
                            continue
                        selected = torch.tensor(donor,dtype=torch.long)
                        log_d = torch.logsumexp(weights[qpos,selected],0)
                        matching = [i for i in donor if int(assigned[site,ref_columns[i]]) == allele]
                        ratio = torch.exp(torch.logsumexp(weights[qpos,torch.tensor(matching,dtype=torch.long)],0)-log_d) if matching else logs.new_zeros(())
                        frequency = logs.new_tensor(len(matching)/len(donor))
                        out[site,qpos,group] = ratio-frequency
                        out[site,qpos,8+group] = (log_d-logs.new_tensor(len(donor)).log()).clamp_max(0)
                        out[site,qpos,16+group] = 1
                        mask[site,qpos,group] = mask[site,qpos,8+group] = True
            return out,mask,weights
        def pool(features,mask):
            total = torch.where(mask,features,torch.zeros_like(features)).sum(0)
            count = mask.sum(0)
            high = torch.where(mask,features,-torch.inf).amax(0)
            return torch.cat((total/count.clamp_min(1),torch.where(count > 0,high,torch.zeros_like(high))),-1)
        oracle_results = {}
        for name,logs in [('initial_gamma095',initial),('fixed_nonuniform',nonuniform),('lower_bound_underflow',torch.full_like(initial,-16.))]:
            guard()
            with torch.no_grad():
                expected,expected_mask,weights = oracle(logs)
                actual,actual_mask = marker(logs)
                require(torch.equal(actual_mask[:,:8],expected_mask),name+': exact independent validity mask')
                record_close(name+'_contrast',actual[:,:8,:8],expected[:,:,:8])
                record_close(name+'_relative_log_mass',actual[:,:8,8:16],expected[:,:,8:16],atol=LOG_ATOL)
                record_close(name+'_availability',actual[:,:8,16:],expected[:,:,16:])
                summary_close(name+'_pooled',evidence(logs)[:8],pool(expected,expected_mask))
                oracle_results[name] = dict(absolute_path_products_underflowed=int((weights.exp() == 0).sum()),path_count=weights.numel())
        require(oracle_results['lower_bound_underflow']['absolute_path_products_underflowed'] > 0,
                'INCONCLUSIVE: selected observed paths did not exercise FP32 product underflow')
        receipt['path_oracle'] = oracle_results
        with torch.no_grad():
            pair = None
            _,_,weights = oracle(nonuniform)
            for site in range(gt.shape[0]):
                for qi,qc in enumerate(oracle_queries):
                    qa = int(gt[site,qc])
                    if qa < 0:
                        continue
                    for group in range(8):
                        same = [i for i,c in enumerate(ref_columns) if int(group_t[c]) == group and int(gt[site,c]) == qa]
                        other = [i for i,c in enumerate(ref_columns) if int(group_t[c]) == group and int(gt[site,c]) >= 0 and int(gt[site,c]) != qa]
                        candidates = [(a,b) for a in same for b in other if abs(float(weights[qi,a]-weights[qi,b])) > 1e-5]
                        if candidates:
                            pair = (site,qi,group,*candidates[0])
                            break
                    if pair is not None:break
                if pair is not None:break
            require(pair is not None, 'INCONCLUSIVE: no observed opposite-allele pair with distinct fixed path weights')
            site,qi,group,a,b = pair
            pair_gt = gt.clone()
            pair_gt[:,torch.tensor(ref_columns)] = -1
            for i in [a,b]:pair_gt[site,ref_columns[i]] = gt[site,ref_columns[i]]
            value,_ = marker(nonuniform,pair_gt)
            expected = torch.sigmoid(weights[qi,a]-weights[qi,b])-.5
            record_close('two_opposite_donors_weight_ratio',value[site,qi,group].reshape(1),expected.reshape(1))
            receipt['opposite_pair'] = dict(site_index=int(indices[site]),query_column=oracle_queries[qi],group=group,
                reference_columns=[ref_columns[a],ref_columns[b]],edge_log_weight_difference=float(weights[qi,a]-weights[qi,b]))
        gen = torch.Generator(device='cpu').manual_seed(101)
        allele_map = torch.stack([torch.randperm(4,generator=gen) for _ in range(gt.shape[0])])
        relabelled = torch.where(gt >= 0,allele_map.gather(1,gt.long().clamp_min(0)),-1).to(gt.dtype)
        with torch.no_grad():
            summary_close('per_SNP_joint_allele_relabel',evidence(nonuniform,relabelled),evidence(nonuniform))
        mapping = torch.randperm(v,generator=gen)
        new_parent = torch.empty_like(local_parent)
        new_parent[mapping] = torch.where(local_parent >= 0,mapping[local_parent.clamp_min(0)],-1)
        new_sample = mapping[local_sample]
        reordered = layout(new_parent,new_sample)
        verify_join(reordered,new_parent,new_sample)
        new_logs = torch.empty_like(nonuniform)
        new_logs[mapping] = nonuniform
        with torch.no_grad():
            summary_close('joint_node_child_order_relabel',evidence(new_logs,tree=reordered),evidence(nonuniform))
            aligned_features,aligned_mask = marker(nonuniform)
            null_features,null_mask = marker(nonuniform,tree=null)
            initial_features,_ = marker(initial)
            informative = torch.zeros((gt.shape[0],686,8),dtype=torch.bool)
            for group in range(8):
                cols = torch.tensor(refs[groups[refs] == group],dtype=torch.long)
                moved = cols[permutation[cols] != cols]
                polymorphic = torch.stack([gt[:,cols].eq(a).any(1) for a in range(4)]).sum(0) >= 2
                moved_called = gt[:,moved].ge(0).any(1) if len(moved) else torch.zeros(gt.shape[0],dtype=torch.bool)
                informative[:,:,group] = polymorphic[:,None]&moved_called[:,None]&gt[:,query_t].ge(0)
            require(torch.equal(aligned_mask,null_mask), 'Null preserves public validity/call bookkeeping')
            differences = (aligned_features[:,:,:8]-null_features[:,:,:8]).abs()
            responses = (aligned_features[:,:,:8]-initial_features[:,:,:8]).abs()
            above_alignment = differences > 10*(ATOL+RTOL*null_features[:,:,:8].abs())
            above_response = responses > 10*(ATOL+RTOL*initial_features[:,:,:8].abs())
            receipt['alignment'] = dict(preselected_rule='all observed polymorphic group/site and called query entries with moved called reference support',
                informative_query_group_site_entries=int(informative.sum()),
                signed_aligned_minus_null=describe((aligned_features[:,:,:8]-null_features[:,:,:8])[informative]),
                absolute_aligned_minus_null=describe(differences[informative]),
                absolute_fixed_edge_response=describe(responses[informative]),
                alignment_above_10x_tolerance=int((above_alignment&informative).sum()),
                edge_response_above_10x_tolerance=int((above_response&informative).sum()),
                per_group=[dict(group=g,informative=int(informative[:,:,g].sum()),
                    aligned_minus_null=describe(differences[:,:,g][informative[:,:,g]])) for g in range(8)])
        coefficients = torch.randn((686,24),generator=gen,dtype=torch.float32)
        coefficients /= coefficients.abs().sum()
        def smooth(logs,block=16):
            return (evidence(logs,block=block)[:,:24]*coefficients).sum()
        gradients = []
        for block in [1,16]:
            logs = nonuniform.clone().requires_grad_()
            output = evidence(logs,block=block)
            finite('block forward',output)
            gradient = torch.autograd.grad((output[:,:24]*coefficients).sum(),logs)[0]
            finite('smooth edge gradients',gradient)
            gradients.append(gradient.detach())
            if block == 1:
                first_output = output.detach()
            else:
                summary_close('block1_vs16_forward',output.detach(),first_output)
        record_close('block1_vs16_smooth_edge_gradients',gradients[0],gradients[1],atol=GRAD_ATOL,rtol=GRAD_RTOL)
        require(float(gradients[1][root]) == 0., 'Root has no attenuation edge or gradient')
        finite_differences = []
        with torch.no_grad():
            for edge in selected_edges:
                plus,minus = nonuniform.clone(),nonuniform.clone()
                plus[edge] += FD_STEP
                minus[edge] -= FD_STEP
                derivative = (smooth(plus)-smooth(minus))/(2*FD_STEP)
                record_close('central_difference_edge_'+str(lo+edge),derivative.reshape(1),gradients[1][edge].reshape(1),atol=GRAD_ATOL,rtol=GRAD_RTOL)
                finite_differences.append(dict(original_global_edge_child=lo+edge,autograd=float(gradients[1][edge]),central_difference=float(derivative)))
        receipt['finite_differences'] = dict(edges=finite_differences,FP32_cancellation_limit_disclosed=True,step=FD_STEP,no_posthoc_tolerance_changes=True)
        all_ref_mask = gt.clone()
        all_ref_mask[:,torch.tensor(ref_columns)] = -1
        group_mask = gt.clone()
        group_columns = torch.tensor(refs[groups[refs] == 0],dtype=torch.long)
        group_mask[:,group_columns] = -1
        query_mask = gt.clone()
        query_mask[:,query_t[:2]] = -1
        full_coefficients = torch.randn((686,48),generator=gen,dtype=torch.float32)
        full_coefficients /= full_coefficients.abs().sum()
        for name,genotypes in [('unmasked',gt),('all_references_empty',all_ref_mask),('one_group_empty',group_mask),('two_query_columns_empty',query_mask),('lower_bound_underflow',gt)]:
            guard()
            logs = (torch.full_like(nonuniform,-16.) if name == 'lower_bound_underflow' else nonuniform.clone()).requires_grad_()
            value = evidence(logs,genotypes)
            finite(name+' summaries',value)
            gradient = torch.autograd.grad((value*full_coefficients).sum(),logs,allow_unused=True)[0]
            unused = gradient is None
            if unused:gradient = torch.zeros_like(logs)
            finite(name+' full_mean_max_gradients',gradient)
            if name == 'all_references_empty':require(bool((value == 0).all()), 'All-empty reference evidence must be zero')
            if name == 'one_group_empty':require(bool((value[:,[0,8,16,24,32,40]] == 0).all()), 'Complete empty group must have zero evidence')
            if name == 'two_query_columns_empty':require(bool((value[:2] == 0).all()), 'Missing queries have zero evidence')
            receipt['checks'][name+'_finite_full_summary_gradient'] = dict(max_abs_gradient=float(gradient.abs().max()),unused_zero_gradient=unused)
        # A fixed probe context is not a model checkpoint/state or training step.
        adapter = core.PackedSameMarkerAdapterV2(16)
        upward = torch.randn((v,32),generator=gen,dtype=torch.float32,requires_grad=True)
        geometry = torch.as_tensor(geometry_np,dtype=torch.float32).requires_grad_()
        support = torch.as_tensor(support_np,dtype=torch.float32).requires_grad_()
        residual = adapter.forward_native_hold_v2(aligned,upward,geometry,support,gt)
        with torch.no_grad():
            native = aligned.native_node.long()
            edge_features = torch.cat((geometry[native],support[native]),-1)
            states = upward[native]
            gate_score = adapter.gate.projection(torch.cat((edge_features,states,states[aligned.parent.long().clamp_min(0)]),-1)).squeeze(-1)
            actual_log_gamma = core.LOG_MAX+(core.LOG_MIN-core.LOG_MAX)*gate_score.sigmoid()
            record_close('adapter_initial_log_gamma095',actual_log_gamma,initial[native])
        finite('zero adapter residual',residual)
        require(bool((residual == 0).all()), 'Original zero-output adapter must return exact zero')
        probe = torch.randn(residual.shape,generator=gen,dtype=torch.float32)
        probe /= probe.abs().sum()
        variables = [adapter.gate.projection.weight,adapter.gate.projection.bias,adapter.input.weight,adapter.output.weight,upward,geometry,support]
        grad = torch.autograd.grad((residual*probe).sum(),variables,allow_unused=True)
        norms = []
        for i,(value,derivative) in enumerate(zip(variables,grad)):
            derivative = torch.zeros_like(value) if derivative is None else derivative
            finite('zero adapter gradient '+str(i),derivative)
            norms.append(float(derivative.abs().max()))
            if i != 3:require(bool((derivative == 0).all()), 'Expected zero gate/input/state gradient at zero output')
        require(norms[3] > 0, 'Nontrivial fixed linear probe must expose output-projection gradient')
        receipt['zero_adapter'] = dict(exact_residual_zero=True,gradient_max_abs=norms,
            variable_order=['gate_weight','gate_bias','input_weight','output_weight','upward','geometry','support'],
            output_projection_gradient_nonzero=True,zero_gate_gradient_is_expected_not_unlearnability=True)
        meaningful = receipt['alignment']['alignment_above_10x_tolerance'] > 0 and receipt['alignment']['edge_response_above_10x_tolerance'] > 0
        receipt.update(status='PASS' if meaningful else 'INCONCLUSIVE_NO_DEMONSTRABLE_EXTRA_ALIGNMENT',
            all_arithmetic_assertions_passed=True,extra_alignment_demonstrated=meaningful,
            ancestry_improvement_established=False,full_parent_feasibility_established=False,training_authorized=False)
        guard()
        receipt['worker_self_peak_RSS_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        receipt['owned_group_RSS_is_enforced_and_recorded_by_ROOT_supervisor'] = True
        require(receipt['worker_self_peak_RSS_bytes'] <= 8*2**30, 'Worker self peak RSS exceeded original8GiB')
        receipt['elapsed_seconds'] = time.monotonic()-started
        save_compact_receipt_v1(receipt)
        require(meaningful, 'INCONCLUSIVE: same frozen hold has no qualifying >10x extra alignment/edge response')
        return receipt
    except Exception as exc:
        receipt.update(status='FAIL' if not str(exc).startswith('INCONCLUSIVE') else 'INCONCLUSIVE',
            error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-started,
            ancestry_improvement_established=False,full_parent_feasibility_established=False,training_authorized=False)
        # ROOT's exclusive writer must not be invoked twice after a terminal
        # inconclusive receipt. Guard/write failure remains a real failure.
        if 'all_arithmetic_assertions_passed' not in receipt:
            save_compact_receipt_v1(receipt)
        raise

if __name__ == '__main__':
    print(json.dumps(dict(status='SOURCE_ONLY_ROOT_CALLBACK_EXECUTION_REQUIRED',execution=False,numerical_work=False)))
