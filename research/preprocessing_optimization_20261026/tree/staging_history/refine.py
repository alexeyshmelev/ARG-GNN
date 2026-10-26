"""Maintain current runtime memory safeguards in the candidate."""
from pathlib import Path
root=Path('/workspace-SR003.nfs2/ARG-GNN')
out=root/'research/preprocessing_optimization_20261026/tree'
p=out/'kernels.py'
s=p.read_text().replace('return reps[:total].tolist(), counts[:total].tolist(), flags[:total].astype(bool).tolist(), offsets.tolist(), local, global_groups',
    'return reps[:total].copy(), counts[:total].copy(), flags[:total].astype(bool), offsets.tolist(), local, global_groups')
p.write_text(s)
s=(root/'research/repairs/benchmark_recovery_20261020/runtime/cuts.py').read_text()
s=s.replace('_direction_layouts=ix.direction_layouts', 'from .kernels import cut_index, cut_groups, _direction_layouts')
s=s.replace('ix.root_and_cut(bank, p, times, roots, HORIZON)', 'cut_index(p, times, roots)')
s=s.replace('    representatives, group_counts, root_flags, offsets, local_groups, global_groups, above = ix.cut_groups(\n        p, times, samples, ages, cut, HORIZON)',
'''    above = ages > HORIZON
    representatives, group_counts, root_flags, offsets, local_groups, global_groups = cut_groups(
        p, times, cut, samples, above)''')
(out/'cut.py').write_text(s)
p=out/'kernels.cpp'
s=p.read_text()
s=s.replace('   if(previous<0){\n    int64_t maximum=0;', '   std::vector<int64_t> exact_counts(h,0);\n   if(previous<0){\n    int64_t maximum=0;')
s=s.replace('counts[e*h+k]+=1.0f;', '++exact_counts[k];')
s=s.replace('   previous=e;\n  }', '   for(int64_t k=0;k<h;++k)counts[e*h+k]=static_cast<float>(exact_counts[k]);\n   previous=e;\n  }')
p.write_text(s)
