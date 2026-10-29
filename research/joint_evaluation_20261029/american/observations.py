"""Original C3 inferred marginals and canonical midpoint focal adapter.

All active native nodes, all1600 observed samples, and every SNP inside each
selected complete tree interval are retained. Only immutable CPU observations
are cached. Ancestry truth is not accepted by this module.
"""
from collections import OrderedDict
import json
import numpy as np
import tskit
from research.redesign_20261028.data.window_data import uniform_coordinate_context
from research.joint_model_20261029.model.joint_model_v1 import BASE_KEYS, CONTEXT_KEYS
from .common import CLASSES, IDENTITY, path, read, verify, load_npz, require, json_data


class ObservationSource:
    def __init__(self, parent_id, item, tree_cache_bytes=128 * 2**20):
        self.parent_id, self.item = parent_id, item
        require(set(item['binding']) == {'observation_ref', 'focal_registry_ref', 'native_preparation_ref'},
                'Public observation-only worker binding')
        self.observed = read(item['binding']['observation_ref'])
        require(self.observed['parent_id'] == parent_id and self.observed['query_ancestry_truth_in_inputs'] is False,
                'Exact unknown-query observation parent')
        refs = item['observed_refs']
        require(all(self.observed[key] == value for key, value in refs.items()), 'Original observation refs unchanged')
        self.check_stats()
        roster = read(refs['roster_ref'])
        self.panel = roster['samples']
        require(roster['parent_id'] == parent_id and roster['class_order'] == list(CLASSES), 'Roster identity')
        require(len(self.panel) == 1600 and [r['sample_index'] for r in self.panel] == list(range(1600)),
                'All original1600 samples in native order')
        self.samples = np.asarray([r['observed_node_id'] for r in self.panel], dtype=np.int64)
        self.ids = np.asarray([f"{parent_id}:{r['sample_id']}:{r['haplotype']}" for r in self.panel])
        require(len(set(self.samples)) == len(set(self.ids)) == 1600, 'Unique native sample/haplotype identities')
        self.reference = np.asarray([-1 if r['observed_reference_class'] is None else r['observed_reference_class']
                                      for r in self.panel], dtype=np.int64)
        self.query = np.asarray([r['role'] == 'query' for r in self.panel], dtype=bool)
        require(np.array_equal(self.query, np.arange(1600) >= 600) and (self.reference[600:] == -1).all() and
                np.array_equal(np.bincount(self.reference[:600], minlength=3), [200, 200, 200]), '600 known references;1000 unknown query ancestries')
        self.age = np.asarray([r['age_generations'] for r in self.panel], dtype=np.float64)
        self.cohort = np.where(self.query, 0, self.reference + 1)
        self.metadata = np.zeros((1600, 17), dtype=np.float32)
        self.metadata[np.arange(1600), self.cohort] = 1
        self.metadata[np.arange(1600), 9 + np.where(self.query, 4, self.reference)] = 1
        self.metadata[:, 14] = np.log1p(self.age) / np.log(10001.)
        self.metadata[:, 15], self.metadata[:, 16] = self.query, ~self.query
        self.records = load_npz(item['binding']['focal_registry_ref'])
        require(set(self.records) == set(IDENTITY) | {'first_occurrence_intervals_bp'}, 'Exact canonical observation-only focal fields')
        r = self.records
        require(r['parent_id'].item() == parent_id and r['class_order'].tolist() == list(CLASSES), 'Focal parent/class binding')
        require(all(r[k].dtype == np.int64 and r[k].shape == (10000,) for k in ('record_ids', 'sample_nodes', 'sample_indices')) and
                len(np.unique(r['record_ids'])) == 10000 and np.all(np.diff(r['record_ids']) > 0), '10000 stable ascending original raw-path IDs')
        require(r['positions_bp0'].dtype == np.float64 and r['positions_bp0'].shape == (10000,) and
                np.isfinite(r['positions_bp0']).all() and ((r['positions_bp0'] >= 0) & (r['positions_bp0'] < 30000000)).all(), 'Exact canonical midpoint positions')
        require(((r['sample_indices'] >= 600) & (r['sample_indices'] < 1600)).all() and
                np.array_equal(r['sample_nodes'], self.samples[r['sample_indices']]), 'Canonical focal sample/native-node identity')
        intervals = r['first_occurrence_intervals_bp']
        require(intervals.dtype == np.float64 and intervals.shape == (10000, 2) and
                np.array_equal(intervals.mean(1), r['positions_bp0']), 'Unchanged first-occurrence midpoint rule')
        require(len(set(zip(r['positions_bp0'], r['sample_indices']))) == 10000, 'Unique coordinate/sample focal ownership')
        self.ts = tskit.load(path(refs['tree_ref']))
        self.walk = self.ts.first()
        require(self.ts.sequence_length == 30000000 and np.array_equal(self.ts.samples(), self.samples), 'Full original native tree sequence/sample order')
        self.positions = np.load(verify(refs['positions_ref']), mmap_mode='r', allow_pickle=False)
        self.genotype = np.load(path(refs['genotypes_ref']), mmap_mode='r', allow_pickle=False)
        require(self.positions.dtype == np.float64 and np.array_equal(self.positions, self.ts.sites_position) and
                self.genotype.dtype == np.int8 and self.genotype.shape == (len(self.positions), 1600), 'All original observed SNP rows/sample columns')
        with verify(refs['alleles_ref']).open() as stream:
            self.alleles = np.asarray([len(json.loads(line)) for line in stream], dtype=np.int8)
        require(self.alleles.shape == self.positions.shape and ((self.alleles >= 1) & (self.alleles <= 4)).all(), 'Exact original allele categories')
        self.cache_limit, self.cache_bytes, self.cache = tree_cache_bytes, 0, OrderedDict()

    def check_stats(self):
        for key, reference in self.item['observed_refs'].items():
            info = path(reference).stat()
            require(dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns) == self.item['source_stats'][key], 'Immutable observed input stat changed')

    def plan(self):
        r = self.records
        positions, inverse = np.unique(r['positions_bp0'], return_inverse=True)
        sites = []
        for rank, focal in enumerate(positions):
            indices = np.flatnonzero(inverse == rank)
            rng = np.random.default_rng(np.random.SeedSequence([20261029, self.item['parent_ordinal'], rank]))
            sites.append(dict(parent_id=self.parent_id, window_index=rank, position_local_bp=float(focal),
                 example_indices=indices.tolist(), sample_indices=r['sample_indices'][indices].tolist(),
                 context=json_data(uniform_coordinate_context(rng, float(focal), 50000., 5, self.ts.sequence_length))))
        return sites

    def tree(self, coordinate):
        self.walk.seek(coordinate)
        key = self.walk.index
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        tree = self.walk
        native = np.sort(np.asarray(list(tree.nodes()), dtype=np.int64))
        lookup = {int(node): index for index, node in enumerate(native)}
        parents = np.asarray([lookup.get(tree.parent(int(node)), -1) for node in native], dtype=np.int64)
        leaves = np.asarray([lookup[int(node)] for node in self.samples], dtype=np.int64)
        times = self.ts.nodes_time[native]
        require(np.array_equal(times[leaves], self.age), 'Native observed sample ages')
        interval = np.asarray(tree.interval, dtype=np.float64)
        lo, hi = np.searchsorted(self.positions, interval, side='left')
        result = dict(parent=parents, time=times.copy(), leaves=leaves, interval=interval,
                      positions=np.array(self.positions[lo:hi]), genotypes=np.array(self.genotype[lo:hi]),
                      alleles=self.alleles[lo:hi].copy(), native_ids=native)
        size = sum(value.nbytes for value in result.values())
        while self.cache and self.cache_bytes + size > self.cache_limit:
            _, (_, old) = self.cache.popitem(last=False)
            self.cache_bytes -= old
        if size <= self.cache_limit:
            self.cache[key] = (result, size)
            self.cache_bytes += size
        return result

    def extract(self, site):
        require(site['parent_id'] == self.parent_id, 'Exact focal parent')
        indices = np.asarray(site['example_indices'], dtype=np.int64)
        require(len(indices) > 0 and len(np.unique(indices)) == len(indices) and ((indices >= 0) & (indices < 10000)).all(), 'Unique complete focal row IDs')
        r = self.records
        require(np.array_equal(r['sample_indices'][indices], site['sample_indices']) and
                (r['positions_bp0'][indices] == site['position_local_bp']).all(), 'Exact focal row/sample/coordinate join')
        context = site['context']
        selected = [self.tree(float(x)) for x in context['coordinates']]
        by_interval = {tuple(tree['interval']): tree for tree in selected}
        unique = [by_interval[key] for key in sorted(by_interval)]
        mapping = {tuple(tree['interval']): index for index, tree in enumerate(unique)}
        ownership = np.asarray([mapping[tuple(tree['interval'])] for tree in selected], dtype=np.int64)
        parts = {key: [] for key in ('parent', 'node_time', 'node_tree_index', 'leaf_node_index',
                                    'site_tree_index', 'site_position_local_bp', 'site_genotypes', 'site_allele_count')}
        offset = 0
        for index, tree in enumerate(unique):
            parts['parent'].append(np.where(tree['parent'] >= 0, tree['parent'] + offset, -1))
            parts['node_time'].append(tree['time'])
            parts['node_tree_index'].append(np.full(len(tree['parent']), index, dtype=np.int64))
            parts['leaf_node_index'].append((tree['leaves'] + offset)[None])
            parts['site_tree_index'].append(np.full(len(tree['positions']), index, dtype=np.int64))
            parts['site_position_local_bp'].append(tree['positions'])
            parts['site_genotypes'].append(tree['genotypes'])
            parts['site_allele_count'].append(tree['alleles'])
            offset += len(tree['parent'])
        packet = {key: np.concatenate(value) for key, value in parts.items()}
        focal = site['position_local_bp']
        index = np.searchsorted(self.positions, focal)
        exact = index < len(self.positions) and self.positions[index] == focal
        focal_g = np.array(self.genotype[index:index + 1]) if exact else np.full((1, 1600), -1, dtype=np.int8)
        focal_a = self.alleles[index:index + 1].copy() if exact else np.ones(1, dtype=np.int8)
        packet.update(leaf_features=self.metadata.copy(), tree_interval_local_bp=np.asarray([tree['interval'] for tree in unique]),
            tree_window_index=np.zeros(len(unique), dtype=np.int64), tree_core_mask=np.arange(len(unique)) == ownership[0],
            focal_tree_index=ownership[:1], focal_position_local_bp=np.asarray([focal]),
            focal_site_index=np.zeros(len(indices), dtype=np.int64), focal_sample_index=np.asarray(site['sample_indices'], dtype=np.int64),
            focal_genotypes=focal_g, focal_allele_count=focal_a, reference_class=self.reference.copy(), sample_age=self.age.copy(),
            observed_cohort=self.cohort.copy(), query_mask=self.query.copy(), sample_id=self.ids.copy(),
            tree_sample_id=np.broadcast_to(self.ids, (len(unique), len(self.ids))).copy(),
            context_coordinate_local_bp=np.asarray(context['coordinates']), context_tree_index=ownership,
            context_window_index=np.zeros(5, dtype=np.int64), context_quadrature_weight=np.asarray(context['weights']),
            context_proposal_density_per_bp=np.asarray(context['density']), context_forced_focal=np.asarray(context['forced']),
            window_context_domain_local_bp=np.asarray([context['domain']]), window_context_kind=np.asarray([context['kind']], dtype=np.int8))
        require(set(packet) == set(BASE_KEYS + CONTEXT_KEYS), 'Exact frozen model observation whitelist')
        return packet
