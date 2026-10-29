"""Complete Relate trees and unchanged ARGMix target rows, without truth access.

Coordinates in the model are local to the clipped native Relate coverage.
Registry coordinates stay in the original rebased chromosome for scoring.
Genotypes come only from the observed packed genotype file, never converted
Relate mutations. All selected native nodes and interval SNPs are retained.
"""
from collections import OrderedDict
import csv
import numpy as np
import tskit
from research.redesign_20261028.data.window_data import uniform_coordinate_context
from research.joint_model_20261029.model.joint_model_v1 import BASE_KEYS, CONTEXT_KEYS
from .common import CLASSES, COHORTS, RECIPE_REF, CAMPAIGN, path, ref, read, verify, require, json_data


class PackedGenotypes:
    def __init__(self, filename, sites, samples):
        self.samples = samples
        width = (samples + 7) // 8
        require(filename.stat().st_size == sites * width, 'Exact packed observed genotype dimensions')
        self.values = np.memmap(filename, mode='r', dtype=np.uint8, shape=(sites, width))

    def __getitem__(self, indices):
        return np.unpackbits(self.values[indices], axis=-1, bitorder='little', count=self.samples).astype(np.int8)


class ObservationSource:
    def __init__(self, parent_id, seal_ref, *, tree_cache_bytes=128 * 2**20):
        self.parent_id = parent_id
        self.seal = read(seal_ref)
        require(self.seal['schema'] == 'paper_treeloom1611_observation_seal_v1' and
                self.seal['parent_id'] == parent_id and self.seal['private_truth_opened'] is False,
                'Original same-parent observation-only seal')
        require(self.seal['config_ref']['sha256'] == RECIPE_REF['sha256'], 'Exact unchanged ARGMix recipe')
        recipe = read(RECIPE_REF)
        parents = {r['parent_id']: r for r in recipe['parents']}
        self.parent_ordinal = list(parents).index(parent_id)
        require(parents[parent_id]['split'] == self.seal['split'], 'Parent split identity')
        self.bridge = read(self.seal['bridge_ref'])
        self.observed = read(self.seal['observed_manifest_ref'])
        self.support = read(self.seal['support_ref'])
        require(self.bridge['parent_id'] == self.observed['parent_id'] == self.support['parent_id'] == parent_id and
                self.bridge['truth_read'] is False and self.observed['contains_genealogy_or_founder_truth'] is False and
                self.support['truth_or_labels_read'] is False and
                self.bridge['observed_manifest_ref'] == self.seal['observed_manifest_ref'] and
                self.support['inferred_bridge_ref'] == self.seal['bridge_ref'], 'Exact public observation chain')
        require(self.support['examples'] == 100000 and
                self.support['tree_step'] == parents[parent_id]['tree_sampling_step'], 'Original sampled support definition')
        self.panel = read(self.bridge['roster_ref'])
        original = read(self.observed['observed_files']['sample_manifest.json'])
        require(len(original) == len(self.panel) == 3222 and all(
            all(row[k] == old[k] for k in old) for row, old in zip(self.panel, original)), 'Unchanged original sample metadata')
        require([r['sample_index'] for r in self.panel] == list(range(3222)), 'Original sample column order')
        self.samples = np.asarray([r['inferred_tree_node_id'] for r in self.panel], dtype=np.int64)
        self.ids = np.asarray([parent_id + ':' + r['stable_haplotype_id'] for r in self.panel])
        require(len(set(self.samples)) == len(set(self.ids)) == 3222, 'Unique leaf identities')
        self.reference = np.asarray([-1 if r['observed_reference_class'] is None else r['observed_reference_class']
                                     for r in self.panel], dtype=np.int64)
        self.query = np.asarray([r['query_eligible'] for r in self.panel], dtype=bool)
        require(self.query.sum() == 2972 and np.count_nonzero(self.reference >= 0) == 250 and
                (self.reference[self.query] == -1).all() and
                (self.reference[~self.query] >= 0).all(), 'Exactly250 known pure founders and2972 unknown queries')
        require(all(self.reference[i] == CLASSES.index(row['observed_cohort'])
                    for i, row in enumerate(self.panel) if not self.query[i]), 'Reference class order')
        self.age = np.asarray([r['age_generations'] for r in self.panel], dtype=np.float64)
        self.cohort = np.asarray([COHORTS[r['observed_cohort']] for r in self.panel], dtype=np.int64)
        self.metadata = np.zeros((3222, 17), dtype=np.float32)
        self.metadata[np.arange(3222), self.cohort] = 1
        self.metadata[np.arange(3222), 9 + np.where(self.query, 4, self.reference)] = 1
        self.metadata[:, 14] = np.log1p(self.age) / np.log(10001.)
        self.metadata[:, 15], self.metadata[:, 16] = self.query, ~self.query
        self.source_refs = [self.bridge['tree_ref'], self.bridge['roster_ref'], self.seal['registry_ref'],
                            self.support['targets_ref'], self.observed['observed_files']['positions.u4'],
                            self.observed['observed_files']['genotypes.packbits']]
        for source in self.source_refs:
            verify(source)
        self.source_stats = {str(path(r)): (path(r).stat().st_size, path(r).stat().st_mtime_ns) for r in self.source_refs}
        self.ts = tskit.load(path(self.bridge['tree_ref']))
        self.walk = self.ts.first()
        self.offset = float(self.bridge['local_to_rebased_offset_bp'])
        require(np.array_equal(self.ts.samples(), self.samples) and
                np.array_equal(self.ts.nodes_time[self.samples], self.age), 'Native sample/date identity')
        require(self.bridge['observed_coverage_bp0'] == [self.offset, self.offset + self.ts.sequence_length] and
                self.offset >= 0 and self.offset + self.ts.sequence_length <= recipe['map']['length_bp'],
                'No genomic extrapolation beyond inferred coverage')
        rebased_positions = np.fromfile(path(self.observed['observed_files']['positions.u4']), dtype='<u4')
        self.positions = rebased_positions.astype(np.float64) - self.offset
        require(len(self.positions) == self.observed['sites'] == self.ts.num_sites and
                np.array_equal(self.positions, self.ts.sites_position), 'All original SNP coordinates join the native sites')
        self.genotype = PackedGenotypes(path(self.observed['observed_files']['genotypes.packbits']), len(self.positions), 3222)
        self.alleles = np.full(len(self.positions), 2, dtype=np.int8)
        with np.load(path(self.seal['registry_ref']), allow_pickle=False) as archive:
            self.records = {k: archive[k].copy() for k in archive.files}
        r = self.records
        require(set(r) == {'record_ids','sample_nodes','sample_indices','positions_bp0','parent_id','class_order'} and
                r['parent_id'].item() == parent_id and r['class_order'].tolist() == list(CLASSES), 'Original canonical target fields')
        require(np.array_equal(r['record_ids'], np.arange(100000)) and
                r['sample_indices'].shape == (100000,) and r['positions_bp0'].shape == (100000,) and
                ((r['sample_indices'] >= 0) & (r['sample_indices'] < 3222)).all(), 'All original100000 target rows')
        require(self.query[r['sample_indices']].all() and
                np.array_equal(r['sample_nodes'], self.samples[r['sample_indices']]), 'Canonical queries match native leaves')
        self.local_positions = r['positions_bp0'] - self.offset
        require(((self.local_positions >= 0) & (self.local_positions < self.ts.sequence_length)).all(), 'Targets inside inferred coverage')
        with path(self.support['targets_ref']).open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        require(len(rows) == 100000, 'Every original sampler row')
        for i, row in enumerate(rows):
            sample = int(row['sample_index'])
            require(int(row['target_row']) == i and sample == r['sample_indices'][i] and
                    row['sample_id'] == self.panel[sample]['sample_id'] and
                    int(row['haplotype']) == self.panel[sample]['haplotype'] and
                    float(row['local_midpoint_bp']) == self.local_positions[i] and
                    float(row['rebased_midpoint_bp']) == r['positions_bp0'][i], 'Exact target row/sample/coordinate join')
        self.native_target_tree = np.asarray([int(row['inferred_tree_index']) for row in rows], dtype=np.int64)
        self.cache_limit, self.cache_bytes, self.cache = tree_cache_bytes, 0, OrderedDict()

    def check_stats(self):
        for filename, expected in self.source_stats.items():
            info = path(filename).stat()
            require((info.st_size, info.st_mtime_ns) == expected, 'Immutable observed source changed')

    def plan(self):
        positions, inverse = np.unique(self.local_positions, return_inverse=True)
        sites = []
        order = np.argsort(inverse, kind='stable')
        offsets = np.r_[0, np.cumsum(np.bincount(inverse))]
        for rank, focal in enumerate(positions):
            indices = order[offsets[rank]:offsets[rank+1]]
            self.walk.seek(float(focal))
            require((self.native_target_tree[indices] == self.walk.index).all() and
                    float(focal) == np.mean(self.walk.interval), 'Original unshifted marginal-tree midpoint')
            rng = np.random.default_rng(np.random.SeedSequence([20261029, self.parent_ordinal, rank]))
            sites.append(dict(parent_id=self.parent_id, window_index=rank, position_local_bp=float(focal),
                example_indices=indices.tolist(), sample_indices=self.records['sample_indices'][indices].tolist(),
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
                      positions=np.array(self.positions[lo:hi]), genotypes=self.genotype[lo:hi],
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
        require(len(indices) > 0 and len(np.unique(indices)) == len(indices) and
                ((indices >= 0) & (indices < 100000)).all(), 'Unique focal row indices')
        require(np.array_equal(self.records['sample_indices'][indices], site['sample_indices']) and
                (self.local_positions[indices] == site['position_local_bp']).all(), 'Focal identity and coordinates')
        context = site['context']
        selected = [self.tree(float(x)) for x in context['coordinates']]
        by_interval = {tuple(tree['interval']): tree for tree in selected}
        unique = [by_interval[key] for key in sorted(by_interval)]
        mapping = {tuple(tree['interval']): index for index, tree in enumerate(unique)}
        ownership = np.asarray([mapping[tuple(tree['interval'])] for tree in selected], dtype=np.int64)
        parts = {key: [] for key in ('parent','node_time','node_tree_index','leaf_node_index',
                'site_tree_index','site_position_local_bp','site_genotypes','site_allele_count')}
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
        focal_g = self.genotype[index:index+1] if exact else np.full((1,3222), -1, dtype=np.int8)
        focal_a = self.alleles[index:index+1].copy() if exact else np.ones(1, dtype=np.int8)
        packet.update(leaf_features=self.metadata.copy(), tree_interval_local_bp=np.asarray([tree['interval'] for tree in unique]),
            tree_window_index=np.zeros(len(unique), dtype=np.int64), tree_core_mask=np.arange(len(unique)) == ownership[0],
            focal_tree_index=ownership[:1], focal_position_local_bp=np.asarray([focal]),
            focal_site_index=np.zeros(len(indices), dtype=np.int64), focal_sample_index=np.asarray(site['sample_indices'], dtype=np.int64),
            focal_genotypes=focal_g, focal_allele_count=focal_a, reference_class=self.reference.copy(), sample_age=self.age.copy(),
            observed_cohort=self.cohort.copy(), query_mask=self.query.copy(), sample_id=self.ids.copy(),
            tree_sample_id=np.broadcast_to(self.ids,(len(unique),3222)).copy(),
            context_coordinate_local_bp=np.asarray(context['coordinates']), context_tree_index=ownership,
            context_window_index=np.zeros(5, dtype=np.int64), context_quadrature_weight=np.asarray(context['weights']),
            context_proposal_density_per_bp=np.asarray(context['density']), context_forced_focal=np.asarray(context['forced']),
            window_context_domain_local_bp=np.asarray([context['domain']]), window_context_kind=np.asarray([context['kind']],dtype=np.int8))
        require(set(packet) == set(BASE_KEYS + CONTEXT_KEYS), 'Exact frozen-model observation whitelist')
        return packet
