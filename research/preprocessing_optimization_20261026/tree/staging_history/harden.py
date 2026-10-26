from pathlib import Path
out=Path('/workspace-SR003.nfs2/ARG-GNN/research/preprocessing_optimization_20261026/tree')
p=out/'kernels.cpp'
s=p.read_text()
s=s.replace('  std::vector<int64_t> head(n,-1), next(n,-1), stack, preorder;', '''  if(n<=0 || n>=int64_t(1)<<31 || h<=0)return 8;
  for(int64_t i=0;i<n;++i)if(p[i]<-1 || p[i]>=n)return 8;
  for(int64_t i=0;i<h;++i)if(samples[i]<0 || samples[i]>=n)return 8;
  std::vector<int64_t> head(n,-1), next(n,-1), stack, preorder;''')
s=s.replace('  stack.push_back(root);\n  while', '  if(root<0 || root>=n)return 8;\n  stack.push_back(root);\n  while')
s=s.replace('  std::vector<int64_t> trail;\n  *steps=0;', '''  if(n<=0 || n>=int64_t(1)<<31)return 8;
  for(int64_t i=0;i<n;++i)if(p[i]<-1 || p[i]>=n)return 8;
  std::vector<int64_t> trail;
  *steps=0;''')
s=s.replace('  std::unordered_map<int64_t,int64_t> sampled;', '''  if(n<=0 || root<0 || root>=n || h<=0 || h>=int64_t(1)<<31)return 8;
  for(int64_t i=0;i<h;++i)if(samples[i]<0 || samples[i]>=n)return 8;
  std::unordered_map<int64_t,int64_t> sampled;''')
s=s.replace('  while(!stack.empty()) {\n   int64_t u=stack.back(),par=', '  int64_t edges_visited=0;\n  while(!stack.empty()) {\n   int64_t u=stack.back(),par=')
s=s.replace('   for(int64_t c=head[u];c>=0;c=sibling[c]){stack.push_back(c);stack_parent.push_back(local);++degree[local];}', '''   for(int64_t c=head[u];c>=0;c=sibling[c]){
    if(c>=n || ++edges_visited>n)return 8;
    stack.push_back(c);stack_parent.push_back(local);++degree[local];
   }''')
s=s.replace('  for(int64_t c=0;c<n;++c)if(p[c]>=0)', '''  if(n<=0 || n>=int64_t(1)<<31 || root_count<=0 || root_count>n)return 8;
  for(int64_t i=0;i<n;++i)if(p[i]<-1 || p[i]>=n)return 8;
  for(int64_t i=0;i<root_count;++i)if(roots[i]<0 || roots[i]>=n)return 8;
  for(int64_t c=0;c<n;++c)if(p[c]>=0)''')
s=s.replace('int cut_groups(const int64_t* p,const float* times,const int64_t* cut,', 'int cut_groups(const int64_t* p,const float* times,const int64_t* cut,int64_t n,')
s=s.replace('  int64_t total=0;offsets[0]=0;', '''  if(n<=0 || t<=0 || h<=0)return 8;
  int64_t total=0;offsets[0]=0;''')
s=s.replace('    int64_t u=samples[at*h+i],r=above[i]?u:cut[u];', '''    int64_t u=samples[at*h+i];if(u<0 || u>=n)return 8;
    int64_t r=above[i]?u:cut[u];if(r<0 || r>=n)return 8;''')
s=s.replace('  int64_t previous=-1;', '''  if(g<=0 || h<=0 || h>=int64_t(1)<<31)return 8;
  for(int64_t i=0;i<g*h;++i)if(groups[i]<0 || groups[i]>=h)return 8;
  int64_t previous=-1;''')
p.write_text(s)
p=out/'kernels.py'
s=p.read_text()
s=s.replace("_LIB.cut_groups.argtypes = [_I64, _F32, _I64, _I64,", "_LIB.cut_groups.argtypes = [_I64, _F32, _I64, _L, _I64,")
s=s.replace("    row = np.ascontiguousarray(row, dtype=np.int64)\n    h = len(row)", '''    row = np.ascontiguousarray(row, dtype=np.int64)
    h = len(row)
    if (row.ndim!=1 or h<1 or not 0<=root<len(parent)
            or np.asarray(head).shape!=np.asarray(sibling).shape!=(len(parent),)
            or np.asarray(head).shape!=(len(parent),)):
        raise ValueError('Exact bounded tree root and adjacency shapes required')''')
s=s.replace("    n = len(parent)\n    head = np.full(n, -1, dtype=np.int64)", '''    n = len(parent)
    if (np.asarray(parent).shape!=np.asarray(times).shape or np.asarray(parent).ndim!=1
            or np.asarray(roots).ndim!=1):
        raise ValueError('Exact cut node/time/root shapes required')
    head = np.full(n, -1, dtype=np.int64)''')
s=s.replace("    t, h = samples.shape\n    local =", '''    t, h = samples.shape
    if (np.asarray(parent).shape!=np.asarray(times).shape or np.asarray(parent).shape!=np.asarray(cut).shape
            or np.asarray(above).shape!=(h,) or np.asarray(parent).ndim!=1):
        raise ValueError('Exact cut parent/time/group shapes required')
    local =''')
s=s.replace('        np.ascontiguousarray(cut), np.ascontiguousarray(samples).reshape(-1),', '        np.ascontiguousarray(cut), len(parent), np.ascontiguousarray(samples).reshape(-1),')
s=s.replace('    g = len(groups)\n    order = tuple(order)', '''    g = len(groups)
    if groups.shape!=(g,h) or g<1 or h<1:
        raise ValueError('Exact nonempty direction-layout group shapes required')
    order = tuple(order)''')
s += '''


def validate_bank_levels(bank, parent):
    """Preserve the current recovery runtime's supplied-level rejection guards."""
    roots=np.flatnonzero(parent<0)
    owner=np.full(len(parent),-1,dtype=np.int32)
    owner[roots]=np.arange(len(roots),dtype=np.int32)
    for tensor in bank.downward_levels:
        nodes=tensor.detach().numpy()
        if not ((parent[nodes]>=0).all() and (owner[parent[nodes]]>=0).all()):
            raise ValueError("Existing depth levels must retain parent-before-child order")
        owner[nodes]=owner[parent[nodes]]
    if not (owner>=0).all():
        raise ValueError("All original nodes must be rooted and covered")
'''
p.write_text(s)
p=out/'episodes.py'
s=p.read_text().replace('    from research.repairs.benchmark_recovery_20261020.runtime.episodes import load_verified_arrays',
'''    from research.repairs.benchmark_recovery_20261020.runtime.episodes import load_verified_arrays
    from .kernels import validate_bank_levels
    validate_bank_levels(bank, bank.parent.numpy())''')
p.write_text(s)
p=out/'cut.py'
s=p.read_text().replace('from .kernels import cut_index, cut_groups, _direction_layouts', 'from .kernels import cut_index, cut_groups, _direction_layouts, validate_bank_levels')
s=s.replace('    head, sibling, owner, cut = cut_index(p, times, roots)', '    validate_bank_levels(bank,p)\n    head, sibling, owner, cut = cut_index(p, times, roots)')
p.write_text(s)
