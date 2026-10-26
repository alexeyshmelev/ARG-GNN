"""Create reviewable candidate copies. Originals stay byte-for-byte unchanged."""
from pathlib import Path
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/preprocessing_optimization_20261026/tree'
source = (ROOT / 'research/reviews/TreeLoom_preparation_reuse_source_20261017_v1/optimized_observations_v1.py').read_text()
source = source.replace('donor_frequencies, _layout_with_isolated_samples)', 'donor_frequencies)')
source = source.replace('def native_workspace_v1(ts):', 'from .kernels import _layout_with_isolated_samples, _level_arrays, child_levels\n\n\ndef native_workspace_v1(ts):')
old = '''    levels=_level_arrays(height);rank=np.empty(len(parent),dtype=np.int64)
    for level in levels:rank[level]=np.arange(len(level))
    groups=[[] for _ in levels]
    for node in child:groups[int(height[parent[node]])].append(int(node))
    children=tuple(np.asarray(v,dtype=np.int64) for v in groups)
    ranks=tuple(rank[parent[v]] for v in children)'''
assert old in source
source = source.replace(old, '    levels,children,ranks=child_levels(parent,height)')
(OUT / 'observations.py').write_text(source)
source = (ROOT / 'research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/episode_preparation_v1.py').read_text()
start = source.index('def _tree_clades(')
end = source.index('def build_clade_episode_arrays_v1(')
source = source[:start] + 'from .kernels import _forest_index, _tree_clades\n\n' + source[end:]
source = source.replace('from .structural_index_v1 import', 'from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.structural_index_v1 import')
old = '''    if np.count_nonzero(np.asarray(parent)<0)==len(sample_index):
        _require(tuple(isolated_hold_indices)==(),"Connected input has no isolated-hold declaration")
        return _episodes.build_clade_episode_arrays_v1(parent,sample_index,spans,max_payload_bytes=max_payload_bytes)
'''
assert old in source
source = source.replace(old, '')
(OUT / 'episodes.py').write_text(source)
source = (ROOT / 'research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/cut_preparation_v1.py').read_text()
source = source.replace('from .structural_index_v1 import', 'from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.structural_index_v1 import')
source = source.replace('_direction_layouts=_original._direction_layouts', 'from .kernels import cut_index, cut_groups, _direction_layouts')
old = '''    if int((bank.parent<0).sum())==len(bank.sample_index):
        _require(tuple(isolated_hold_indices)==(),"Connected input has no isolated-hold declaration")
        return _original.prepare_cut_plan_v1(bank,expected_haplotypes=expected_haplotypes,
            max_nodes=max_nodes,max_holds=max_holds)
'''
assert old in source
source = source.replace(old, '')
start = source.index('    head = np.full(n, -1, dtype=np.int64)')
end = source.index('    ordered=[classify_sample_roots', start)
source = source[:start] + '    head, sibling, owner, cut = cut_index(p, times, roots)\n' + source[end:]
start = source.index('    representatives, group_counts, root_flags = [], [], []')
end = source.index('    _require(len(set(representatives))', start)
source = source[:start] + '''    above = ages > HORIZON
    representatives, group_counts, root_flags, offsets, local_groups, global_groups = cut_groups(p, times, cut, samples, above)
''' + source[end:]
(OUT / 'cut.py').write_text(source)
