"""Complete nonrecurrent genealogy model with immutable sparse routing.

New architecture, not equivalent to the 28 October cavity model. Observed SNP
encoding/readout follow its common-information contract. No ancestry labels are
accepted. Sample IDs validate routing and never enter trainable features.
"""
from dataclasses import dataclass, fields
from collections import defaultdict, deque
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

VERSION='joint_complete_sparse_v1_20261029'
RADIUS_BP=50000.
AGE_SCALE=100.
ROOT_ATOM=1.
LEAF_ATOM=1.
BASE_KEYS=('parent','node_time','node_tree_index','leaf_node_index','leaf_features',
 'tree_interval_local_bp','tree_window_index','tree_core_mask','focal_tree_index',
 'focal_sample_index','focal_position_local_bp','focal_genotypes','focal_allele_count',
 'reference_class','sample_age','observed_cohort','query_mask','site_tree_index',
 'site_position_local_bp','site_genotypes','site_allele_count','focal_site_index',
 'sample_id','tree_sample_id')
CONTEXT_KEYS=('context_coordinate_local_bp','context_tree_index','context_window_index',
 'context_quadrature_weight','context_proposal_density_per_bp','context_forced_focal',
 'window_context_domain_local_bp','window_context_kind')
MODEL_KEYS=('model_coordinate_bp','model_coordinate_weight','model_context_kind')


def require(ok,message):
    if not bool(ok):raise ValueError(message)


def load_observations(path):
    """Read only observation whitelist; labels/example_index are never accessed."""
    with np.load(Path(path),allow_pickle=False) as archive:
        extra=CONTEXT_KEYS if 'context_tree_index' in archive.files else MODEL_KEYS
        return {key:archive[key].copy() for key in BASE_KEYS+extra}


def expand_coordinates(arrays):
    a={k:np.asarray(v) for k,v in arrays.items()}
    require(set(a)==set(BASE_KEYS+CONTEXT_KEYS),'Exact compact native observation whitelist')
    source=a['context_tree_index'].astype(np.int64);win=a['context_window_index'].astype(np.int64)
    coord=a['context_coordinate_local_bp'].astype(np.float64);weight=a['context_quadrature_weight'].astype(np.float64)
    forced=a['context_forced_focal'].astype(bool);density=a['context_proposal_density_per_bp'].astype(np.float64)
    domain=a['window_context_domain_local_bp'];kind=a['window_context_kind']
    require(source.ndim==1 and all(x.shape==source.shape for x in (win,coord,weight,forced,density)),'Coordinate vector shapes')
    require(np.isfinite(coord).all() and np.isfinite(weight).all() and np.isfinite(density).all(),'Finite coordinates/weights')
    require(kind.ndim==1 and len(kind)>0 and np.all(kind==1),'Coordinate windows only')
    require(domain.shape==(len(kind),2) and np.isfinite(domain).all() and (domain[:,1]>domain[:,0]).all(),'Positive context domains')
    require(((win>=0)&(win<len(kind))).all() and ((source>=0)&(source<len(a['tree_window_index']))).all(),'Context ownership bounds')
    require(np.array_equal(a['tree_window_index'][source],win),'Source trees belong to same window')
    require(((coord>=domain[win,0])&(coord<domain[win,1])).all(),'Coordinate inside context domain')
    intervals=a['tree_interval_local_bp']
    require(((coord>=intervals[source,0])&(coord<intervals[source,1])).all(),'Coordinate inside complete source tree')
    for w in range(len(kind)):
        ix=win==w;random=ix&~forced
        require(ix.sum()==5 and forced[ix].sum()==1 and random.sum()==4,'Exactly one focal and four random draws per site')
        require(np.all(weight[ix&forced]==0) and np.all(density[ix&forced]==0),'Focal source mass is zero')
        require(np.allclose(weight[random],.25) and np.allclose(density[random],1./(domain[w,1]-domain[w,0])),'Uniform coordinate draw mass/density')
    out={key:np.array(a[key],copy=True) for key in BASE_KEYS}
    parts={key:[] for key in ('parent','node_time','node_tree_index','leaf_node_index','site_tree_index','site_position_local_bp','site_genotypes','site_allele_count')}
    offset=0
    for occurrence,tree in enumerate(source):
        nodes=np.flatnonzero(a['node_tree_index']==tree)
        remap=np.full(len(a['parent']),-1,dtype=np.int64);remap[nodes]=np.arange(len(nodes))+offset
        parent=a['parent'][nodes]
        require(((parent<0)|np.isin(parent,nodes)).all(),'Native parents stay within tree')
        parts['parent'].append(np.where(parent>=0,remap[parent.clip(min=0)],-1))
        parts['node_time'].append(a['node_time'][nodes]);parts['node_tree_index'].append(np.full(len(nodes),occurrence))
        parts['leaf_node_index'].append(remap[a['leaf_node_index'][tree]][None])
        site=np.flatnonzero(a['site_tree_index']==tree)
        parts['site_tree_index'].append(np.full(len(site),occurrence))
        for key in ('site_position_local_bp','site_genotypes','site_allele_count'):parts[key].append(a[key][site])
        offset+=len(nodes)
    for key,value in parts.items():out[key]=np.concatenate(value,axis=0)
    out['tree_interval_local_bp']=intervals[source].copy();out['tree_window_index']=win.copy()
    out['tree_sample_id']=a['tree_sample_id'][source].copy();out['tree_core_mask']=forced.copy()
    focal=[]
    for tree,position in zip(a['focal_tree_index'],a['focal_position_local_bp']):
        w=int(a['tree_window_index'][tree]);hit=np.flatnonzero((win==w)&forced&(source==tree)&np.isclose(coord,position,atol=1e-8,rtol=0))
        require(len(hit)==1,'Unique forced focal occurrence')
        focal.append(int(hit[0]))
    out['focal_tree_index']=np.asarray(focal,dtype=np.int64)
    out.update(model_coordinate_bp=coord,model_coordinate_weight=weight,model_context_kind=kind.copy())
    return out


def csr(rows,cols,values,shape):
    index=torch.tensor([rows,cols],dtype=torch.long)
    return torch.sparse_coo_tensor(index,torch.tensor(values,dtype=torch.float32),shape).coalesce().to_sparse_csr()


def block_csr(matrices):
    rows=[torch.zeros(1,dtype=torch.long)];cols=[];values=[];row_count=col_count=nnz=0
    for matrix in matrices:
        rows.append(matrix.crow_indices()[1:]+nnz);cols.append(matrix.col_indices()+col_count);values.append(matrix.values())
        row_count+=matrix.size(0);col_count+=matrix.size(1);nnz+=matrix._nnz()
    return torch.sparse_csr_tensor(torch.cat(rows),torch.cat(cols),torch.cat(values),size=(row_count,col_count))


@dataclass(frozen=True)
class Batch:
    parent:torch.Tensor
    node_time:torch.Tensor
    node_tree:torch.Tensor
    leaves:torch.Tensor
    metadata:torch.Tensor
    reference:torch.Tensor
    cohort:torch.Tensor
    query:torch.Tensor
    tree_window:torch.Tensor
    coordinate:torch.Tensor
    coordinate_weight:torch.Tensor
    site_tree:torch.Tensor
    site_genotype:torch.Tensor
    site_alleles:torch.Tensor
    focal_tree:torch.Tensor
    focal_position:torch.Tensor
    focal_genotype:torch.Tensor
    focal_alleles:torch.Tensor
    focal_site:torch.Tensor
    focal_sample:torch.Tensor
    geometry:torch.Tensor
    descendant:torch.Tensor
    ancestor:torch.Tensor
    native:tuple
    cross:tuple
    coupling_mass:torch.Tensor
    statistics:dict

    @property
    def target_count(self):return len(self.focal_sample)
    @property
    def window_count(self):return self.metadata.shape[0]
    def to(self,device,non_blocking=False):
        def move(value):
            if isinstance(value,torch.Tensor):return value.to(device,non_blocking=non_blocking)
            if isinstance(value,tuple):return tuple(move(x) for x in value)
            return value
        return Batch(**{f.name:move(getattr(self,f.name)) for f in fields(self)})
    def pin_memory(self):
        def pin(value):
            if isinstance(value,torch.Tensor):
                if value.layout==torch.sparse_csr:
                    return torch.sparse_csr_tensor(value.crow_indices().pin_memory(),value.col_indices().pin_memory(),value.values().pin_memory(),size=value.shape)
                return value.pin_memory()
            if isinstance(value,tuple):return tuple(pin(x) for x in value)
            return value
        return Batch(**{f.name:pin(getattr(self,f.name)) for f in fields(self)})


def prepared_to_state(batch):
    require(type(batch) is Batch,'Prepared Batch required')
    return {'version':VERSION,'radius_bp':RADIUS_BP,'age_scale':AGE_SCALE,'root_atom':ROOT_ATOM,'leaf_atom':LEAF_ATOM,
            'batch':{f.name:list(getattr(batch,f.name)) if isinstance(getattr(batch,f.name),tuple) else getattr(batch,f.name) for f in fields(batch)}}


def prepared_from_state(state):
    require(set(state)=={'version','radius_bp','age_scale','root_atom','leaf_atom','batch'},'Exact cache state schema')
    require((state['version'],state['radius_bp'],state['age_scale'],state['root_atom'],state['leaf_atom'])==(VERSION,RADIUS_BP,AGE_SCALE,ROOT_ATOM,LEAF_ATOM),'Cache scientific configuration mismatch')
    value=dict(state['batch']);require(set(value)=={f.name for f in fields(Batch)},'Exact cached Batch fields')
    value['native']=tuple(value['native']);value['cross']=tuple(value['cross'])
    return Batch(**value)


def prepare_observations(arrays, *, max_path_entries=2000000, max_witness_steps=20000000, max_cross_nnz=4000000):
    """Validate observations and cache immutable sparse maps once per site window."""
    if set(arrays)==set(BASE_KEYS+CONTEXT_KEYS):arrays=expand_coordinates(arrays)
    require(set(arrays)==set(BASE_KEYS+MODEL_KEYS),'Exact expanded observation whitelist; labels forbidden')
    a={k:np.asarray(v) for k,v in arrays.items()}
    def integer(key):
        require(np.issubdtype(a[key].dtype,np.integer),'Integer field: '+key)
        return a[key].astype(np.int64)
    p=integer('parent');nt=integer('node_tree_index');leaves=integer('leaf_node_index')
    age=a['node_time'].astype(np.float64);n=len(p)
    require(p.ndim==1 and n>0 and age.shape==p.shape and nt.shape==p.shape,'Native node vectors')
    require(leaves.ndim==2 and len(leaves)>0 and leaves.shape[1]>0,'Complete tree sample map')
    t,h=leaves.shape;tw=integer('tree_window_index')
    require(tw.shape==(t,) and (tw>=0).all(),'Window routing')
    w=int(tw.max())+1;require(np.array_equal(np.unique(tw),np.arange(w)),'Contiguous window labels')
    require(np.isfinite(age).all() and (age>=0).all() and ((p>=-1)&(p<n)).all(),'Valid ages and parents')
    require(((nt>=0)&(nt<t)).all() and ((leaves>=0)&(leaves<n)).all() and np.unique(leaves).size==t*h,'Native/sample bounds and uniqueness')
    require(np.array_equal(nt[leaves],np.broadcast_to(np.arange(t)[:,None],(t,h))),'Samples belong to declared trees')
    branch=p>=0
    require((nt[p[branch]]==nt[branch]).all() and (age[p[branch]]>=age[branch]).all(),'Parents are in same tree and no younger')
    def sample_field(key,trailing=None):
        value=a[key];single=(h,) if trailing is None else (h,trailing)
        if value.shape==single:value=np.broadcast_to(value,(w,)+single)
        require(value.shape==(w,)+single,'Sample metadata shape: '+key)
        return value
    sample_ids=sample_field('sample_id');tree_ids=a['tree_sample_id']
    require(tree_ids.shape==(t,h) and np.array_equal(tree_ids,sample_ids[tw]),'Explicit ordered tree/sample identity mismatch')
    require(all(len(set(row.tolist()))==h for row in sample_ids),'Unique parent/sample/haplotype identities required')
    metadata=sample_field('leaf_features',17).astype(np.float32)
    reference=sample_field('reference_class').astype(np.int64);cohort=sample_field('observed_cohort').astype(np.int64)
    query=sample_field('query_mask').astype(bool);sample_age=sample_field('sample_age').astype(np.float64)
    require(np.isfinite(metadata).all() and np.isfinite(sample_age).all(),'Finite sample metadata')
    require(((reference>=-1)&(reference<4)).all() and ((cohort>=0)&(cohort<9)).all(),'Observed class/cohort range')
    require((reference[query]==-1).all(),'Query ancestry must be unknown')
    require(np.allclose(age[leaves],sample_age[tw],atol=1e-6,rtol=1e-6),'Sample age agreement')
    intervals=a['tree_interval_local_bp'].astype(np.float64);x=a['model_coordinate_bp'].astype(np.float64)
    weight=a['model_coordinate_weight'].astype(np.float64);kind=integer('model_context_kind')
    require(intervals.shape==(t,2) and np.isfinite(intervals).all() and (intervals[:,1]>intervals[:,0]).all(),'Positive finite tree intervals')
    require(x.shape==(t,) and weight.shape==(t,) and np.isfinite(x).all() and np.isfinite(weight).all() and (weight>=0).all(),'Finite coordinate quadrature')
    require(kind.shape==(w,) and (kind==1).all(),'Coordinate model only')
    require(((x>=intervals[:,0])&(x<intervals[:,1])).all(),'Coordinate belongs to complete tree')
    require(all(np.isclose(weight[tw==j].sum(),1.) for j in range(w)),'Source quadrature mass one per window')
    degree=np.bincount(p[branch],minlength=n)
    require(set(np.flatnonzero(degree==0))==set(leaves.ravel()),'All and only terminal nodes are sampled')
    remaining=degree.copy();queue=deque(np.flatnonzero(degree==0).tolist());order=[];desc_count=np.zeros(n,dtype=np.int64)
    desc_count[leaves.ravel()]=1
    while queue:
        u=queue.popleft();order.append(u)
        if p[u]>=0:
            v=int(p[u]);desc_count[v]+=desc_count[u];remaining[v]-=1
            if remaining[v]==0:queue.append(v)
    require(len(order)==n,'Cycle in native parent forest')
    require(min(max_path_entries,max_witness_steps,max_cross_nnz)>0,'Positive execution bounds')
    depth=np.zeros(n,dtype=np.int64)
    for u in reversed(order):
        if p[u]>=0:depth[u]=depth[p[u]]+1
    path_entries=int((depth[leaves]+1).sum())
    require(path_entries<=max_path_entries,f'Complete window path map exceeds execution bound: nodes={n}, trees={t}, entries={path_entries}, limit={max_path_entries}; no truncation')
    witness_bound=0
    for ti in range(t):
        for sj in np.flatnonzero((tw==tw[ti])&(weight>0)&(np.abs(x-x[ti])<RADIUS_BP)):
            witness_bound+=int(depth[leaves[ti]].sum()+depth[leaves[sj]].sum())
    require(witness_bound<=max_witness_steps,f'Complete window witness work exceeds execution bound: {witness_bound}>{max_witness_steps}; no truncation')
    duration=np.zeros(n);duration[branch]=age[p[branch]]-age[branch]
    eta=np.zeros(n);eta[branch]=np.log1p((age[p[branch]]-age[branch])/(AGE_SCALE+age[branch]))
    masses=desc_count*eta
    paths=[];dr=[];dc=[];dv=[];ar=[];ac=[];av=[]
    for ti,row in enumerate(leaves):
        tree_paths=[]
        for si,leaf in enumerate(row):
            u=int(leaf);path=[]
            while u>=0:path.append(u);u=int(p[u])
            tree_paths.append(path)
            atom=np.array([eta[u]+(LEAF_ATOM if u==leaf else 0.)+(ROOT_ATOM if p[u]<0 else 0.) for u in path])
            atom=atom/atom.sum()
            for u,value in zip(path,atom):
                dr.append(u);dc.append(ti*h+si);dv.append(1./desc_count[u])
                if value>0:ar.append(ti*h+si);ac.append(u);av.append(value)
        paths.append(tree_paths)
    descendant=csr(dr,dc,dv,(n,t*h));ancestor=csr(ar,ac,av,(t*h,n))
    child=np.flatnonzero(branch);parent=p[branch]
    native=(csr(parent.tolist(),child.tolist(),(1./degree[parent]).tolist(),(n,n)),
            csr(child.tolist(),parent.tolist(),np.ones(len(child)).tolist(),(n,n)))
    cross_rows=[[] for _ in range(4)];cross_cols=[[] for _ in range(4)];cross_values=[[] for _ in range(4)]
    cross_count=0
    def add(rel,target,source,value):
        nonlocal cross_count
        if value<=0:return
        cross_count+=1
        require(cross_count<=max_cross_nnz,f'Complete window cross adjacency exceeds {max_cross_nnz} entries; no truncation')
        cross_rows[rel].append(int(target));cross_cols[rel].append(int(source));cross_values[rel].append(float(value))
    for ti in range(t):
        for sj in np.flatnonzero((tw==tw[ti])&(weight>0)):
            delta=x[sj]-x[ti];alpha=weight[sj]*max(0.,1.-abs(delta)/RADIUS_BP)
            if alpha<=0:continue
            sides=((0,.5),(1,.5)) if delta==0 else (((0,1.),) if delta<0 else ((1,1.),))
            counts=defaultdict(int)
            for pa,pb in zip(paths[ti],paths[sj]):
                i=j=0
                while i<len(pa)-1 and j<len(pb)-1:
                    u,v=pa[i],pb[j];ea=age[p[u]];eb=age[p[v]]
                    if min(ea,eb)>max(age[u],age[v]):
                        counts[(u,v)]+=1
                        require(len(counts)<=max_cross_nnz,f'Complete tree-pair overlap exceeds {max_cross_nnz} entries; no truncation')
                    if ea<=eb:i+=1
                    if eb<=ea:j+=1
            for side,fraction in sides:
                for u,v in zip(leaves[ti],leaves[sj]):add(side,u,v,alpha*fraction)
                for (u,v),intersection in counts.items():
                    lower=max(age[u],age[v]);upper=min(age[p[u]],age[p[v]])
                    overlap=np.log1p((upper-lower)/(AGE_SCALE+lower))
                    specificity=intersection/(desc_count[u]+desc_count[v]-intersection)
                    add(2+side,u,v,alpha*fraction*intersection*overlap*specificity/masses[u])
    cross=tuple(csr(cross_rows[r],cross_cols[r],cross_values[r],(n,n)) for r in range(4))
    coupling_mass=torch.cat([torch.sparse.mm(m,torch.ones((n,1))) for m in cross],1)
    require(torch.isfinite(coupling_mass).all() and (coupling_mass>=0).all() and (coupling_mass.sum(1)<=2.00001).all(),'Bounded effective coupling mass')
    st=integer('site_tree_index');sg=integer('site_genotypes');sa=integer('site_allele_count')
    ft=integer('focal_tree_index');fg=integer('focal_genotypes');fa=integer('focal_allele_count')
    fs=integer('focal_sample_index');fu=integer('focal_site_index');fp=a['focal_position_local_bp'].astype(np.float64)
    require(st.ndim==1 and ((st>=0)&(st<t)).all() and sg.shape==(len(st),h),'Site observation routing')
    sp=a['site_position_local_bp']
    require(sp.shape==st.shape and np.isfinite(sp).all() and ((sp>=intervals[st,0])&(sp<intervals[st,1])).all(),'Observed SNP belongs to native interval')
    require(ft.ndim==1 and ((ft>=0)&(ft<t)).all() and fg.shape==(len(ft),h) and fp.shape==ft.shape,'Unique focal observations')
    require(fs.ndim==1 and fu.shape==fs.shape and ((fs>=0)&(fs<h)).all() and ((fu>=0)&(fu<len(ft))).all(),'Canonical focal output routing')
    require(((fp>=intervals[ft,0])&(fp<intervals[ft,1])).all() and np.isfinite(fp).all(),'Focal belongs to tree')
    require(np.allclose(fp,x[ft],rtol=0,atol=1e-8) and (weight[ft]==0).all(),'Focal outputs use forced zero-mass coordinate')
    core=a['tree_core_mask']
    require(core.shape==(t,) and core[ft].all(),'Focal core mask')
    require(query[tw[ft[fu]],fs].all(),'Only query targets allowed')
    require(len(set(zip(fu.tolist(),fs.tolist())))==len(fs),'Each target has unique ownership within batch')
    for genotype,allele in ((sg,sa),(fg,fa)):
        require(allele.shape==(len(genotype),) and ((allele>=1)&(allele<=4)).all(),'Observed allele counts')
        require(((genotype>=-1)&(genotype<allele[:,None])).all(),'Called or missing categorical genotypes')
    sampled=np.zeros(n);sampled[leaves.ravel()]=1
    geometry=np.stack([np.log1p(age)/math.log(10001.),np.log1p(duration)/math.log(10001.),np.log1p(desc_count)/math.log(h+1),(p<0).astype(float),sampled],1)
    tensor=lambda value,dtype=None:torch.as_tensor(np.array(value,copy=True),dtype=dtype)
    lengths=[len(path) for group in paths for path in group]
    matrices=(descendant,ancestor)+native+cross
    sparse_bytes=sum(m.crow_indices().numel()*8+m.col_indices().numel()*8+m.values().numel()*4 for m in matrices)
    stats={'nodes':n,'trees':t,'samples':h,'windows':w,'max_path_nodes':max(lengths),'mean_path_nodes':float(np.mean(lengths)),
           'nnz':[m._nnz() for m in matrices],'sparse_bytes':sparse_bytes,'path_entries':path_entries,'witness_step_bound':witness_bound,'radius_bp':RADIUS_BP,'age_scale':AGE_SCALE,
           'site_observation_min_bp':float(a['site_position_local_bp'].min()) if len(st) else None,
           'site_observation_max_bp':float(a['site_position_local_bp'].max()) if len(st) else None}
    return Batch(tensor(p),tensor(age,torch.float32),tensor(nt),tensor(leaves),tensor(metadata),tensor(reference),tensor(cohort),tensor(query),
                 tensor(tw),tensor(x),tensor(weight),tensor(st),tensor(sg),tensor(sa),tensor(ft),tensor(fp),tensor(fg),tensor(fa),tensor(fu),tensor(fs),
                 tensor(geometry,torch.float32),descendant,ancestor,native,cross,coupling_mass,stats)


def collate_prepared(batches):
    require(len(batches)>0 and all(type(b) is Batch for b in batches),'Nonempty prepared batches required')
    h=batches[0].leaves.shape[1]
    require(all(b.leaves.shape[1]==h and b.parent.device.type=='cpu' for b in batches),'CPU equal-panel batches required')
    names=[f.name for f in fields(Batch) if f.name not in ('descendant','ancestor','native','cross','statistics')]
    pieces={key:[] for key in names};no=to=wo=uo=0
    for b in batches:
        for key in names:
            value=getattr(b,key)
            if key=='parent':value=torch.where(value>=0,value+no,value)
            elif key=='leaves':value=value+no
            elif key in ('node_tree','site_tree','focal_tree'):value=value+to
            elif key=='tree_window':value=value+wo
            elif key=='focal_site':value=value+uo
            pieces[key].append(value)
        no+=len(b.parent);to+=len(b.leaves);wo+=b.window_count;uo+=len(b.focal_tree)
    out={key:torch.cat(value,0) for key,value in pieces.items()}
    out.update(descendant=block_csr([b.descendant for b in batches]),ancestor=block_csr([b.ancestor for b in batches]),
               native=tuple(block_csr([b.native[j] for b in batches]) for j in range(2)),
               cross=tuple(block_csr([b.cross[j] for b in batches]) for j in range(4)),statistics={'collated_windows':wo,'components':[b.statistics for b in batches]})
    return Batch(**out)


class SparseSum(MessagePassing):
    """PyG fused message_and_aggregate: no Python edge-by-channel message tensor."""
    def __init__(self):super().__init__(aggr='add',node_dim=0)
    def forward(self,adj_t,x):
        return self.propagate(adj_t,x=x,size=(adj_t.size(1),adj_t.size(0)))
    def message_and_aggregate(self,adj_t,x):
        return torch.sparse.mm(adj_t,x)


class JointLayer(nn.Module):
    def __init__(self,width):
        super().__init__();self.self_map=nn.Linear(width,width)
        self.native_maps=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(2))
        self.cross_maps=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(4))
        self.cross_gate=nn.Parameter(torch.full((4,),-2.))
        self.norm=nn.LayerNorm(width);self.aggregate=SparseSum()
    def forward(self,x,batch,mode):
        z=self.self_map(x)
        for linear,matrix in zip(self.native_maps,batch.native):z=z+self.aggregate(matrix,linear(x))
        count={'no_cross':0,'leaf_only':2,'structure':4}[mode]
        for j in range(count):
            projected=self.cross_maps[j](x)
            delta=self.aggregate(batch.cross[j],projected)-batch.coupling_mass[:,j,None]*projected
            z=z+self.cross_gate[j].sigmoid()*delta
        return self.norm(x+F.gelu(z))


class JointAncestryModel(nn.Module):
    """Complete native-tree ancestry predictor, with shared site-level activations."""
    def __init__(self,width=64,layers=3,classes=6):
        super().__init__();require(width>=2 and layers>=1 and classes>=2,'Model dimensions')
        self.width=width;self.depth=layers;self.classes=classes;d=width
        self.site_mlp=nn.Sequential(nn.Linear(29,d),nn.GELU(),nn.Linear(d,d),nn.Tanh())
        self.site_pool=nn.Sequential(nn.Linear(2*d+2,d),nn.Tanh())
        self.leaf_projection=nn.Sequential(nn.Linear(d+17,d),nn.Tanh())
        self.node_initializer=nn.Sequential(nn.Linear(d+5,d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.layers=nn.ModuleList(JointLayer(d) for _ in range(layers))
        self.path_readout=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.LayerNorm(d))
        self.classifier=nn.Sequential(nn.Linear(2*d+17,d),nn.GELU(),nn.Linear(d,classes))
        self.aggregate=SparseSum()

    def model_id(self,mode='structure'):
        require(mode in ('no_cross','leaf_only','structure'),'Declared mode')
        return f'{VERSION}_{mode}_d{self.width}_L{self.depth}_c{self.classes}'

    def encode_snps(self,genotype,allele_count,windows,batch):
        """Common observed cohort/reference encoder, once per unique SNP row."""
        j,h=genotype.shape;d=self.width
        if not j:return self.site_mlp[0].weight.new_zeros((0,h,d))
        called=genotype>=0;safe=genotype.clamp_min(0)
        alleles=F.one_hot(safe,4).to(self.site_mlp[0].weight.dtype)*called[:,:,None]
        ref=F.one_hot(batch.reference.clamp_min(0),4).to(alleles.dtype)*(batch.reference>=0)[:,:,None]
        cohorts=F.one_hot(batch.cohort,9).to(alleles.dtype)
        membership=torch.cat((ref,cohorts),-1).index_select(0,windows)
        counts=torch.einsum('jha,jhg->jga',alleles,membership)
        total=counts.sum(-1,keepdim=True)
        frequency=(counts+.5)/(total+.5*allele_count[:,None,None]);available=(total>0).to(alleles.dtype)
        logfreq=torch.where(available.bool(),frequency.log(),torch.zeros_like(frequency)).transpose(1,2)
        mask=available.transpose(1,2).expand(-1,4,-1)
        panel=alleles.sum(1);panel=panel/panel.sum(-1,keepdim=True).clamp_min(1)
        panel_log=panel.clamp_min(1e-12).log();entropy=-(panel*panel_log).sum(-1)
        features=torch.cat((logfreq,mask,panel_log[:,:,None],entropy[:,None,None].expand(-1,4,1),torch.ones((j,4,1),device=genotype.device,dtype=alleles.dtype)),-1)
        encoded=self.site_mlp(features)
        return encoded.gather(1,safe[:,:,None].expand(j,h,d))*called[:,:,None]

    def encode_leaves(self,batch):
        t,h=batch.leaves.shape;d=self.width
        values=self.encode_snps(batch.site_genotype,batch.site_alleles,batch.tree_window[batch.site_tree],batch)
        called=(batch.site_genotype>=0).to(values.dtype)
        sums=values.new_zeros((t,h,d)).index_add(0,batch.site_tree,values)
        counts=values.new_zeros((t,h,1)).index_add(0,batch.site_tree,called[:,:,None])
        maxima=values.new_full((t,h,d),-torch.inf)
        if len(values):
            selected=torch.where(called[:,:,None].bool(),values,torch.full_like(values,-torch.inf))
            maxima=maxima.scatter_reduce(0,batch.site_tree[:,None,None].expand_as(selected),selected,reduce='amax',include_self=True)
        maxima=torch.where(counts>0,maxima,torch.zeros_like(maxima))
        pooled=self.site_pool(torch.cat((sums/counts.clamp_min(1),maxima,counts.log1p()/10,(counts==0).to(values.dtype)),-1))
        return self.leaf_projection(torch.cat((pooled,batch.metadata[batch.tree_window]),-1)).reshape(t*h,d)

    def initialize_nodes(self,batch):
        descendant_mean=self.aggregate(batch.descendant,self.encode_leaves(batch))
        return self.node_initializer(torch.cat((descendant_mean,batch.geometry),-1))

    def read_leaves(self,nodes,batch):
        own=nodes[batch.leaves.reshape(-1)];path=self.aggregate(batch.ancestor,nodes)
        return self.path_readout(torch.cat((own,path),-1)).reshape(len(batch.leaves),batch.leaves.shape[1],self.width)

    def forward(self,batch,mode='structure'):
        require(type(batch) is Batch and mode in ('no_cross','leaf_only','structure'),'Prepared Batch and declared mode required')
        nodes=self.initialize_nodes(batch)
        for layer in self.layers:nodes=layer(nodes,batch,mode)
        leaf=self.read_leaves(nodes,batch)
        focal=self.encode_snps(batch.focal_genotype,batch.focal_alleles,batch.tree_window[batch.focal_tree],batch)
        site=batch.focal_site;sample=batch.focal_sample;tree=batch.focal_tree[site]
        context=leaf[tree,sample];snp=focal[site,sample];meta=batch.metadata[batch.tree_window[tree],sample]
        return self.classifier(torch.cat((context,snp,meta),-1))
