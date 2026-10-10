"""Exact observation-only H021 native/serialized join, source-only interface.

No array loader, truth capability, scaler fitting or training entry point. The
root must admit and hash observed files before passing decoded inputs. Native
float64 dates and producer row positions certify anonymous interiors; equal
age, equal clade and inverted float32 ages are never identity mechanisms.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import torch
from torch import Tensor
from .data import GraphSequence
from .reference_encoder import ReferenceFeatureAdapter
from .census_cut_exchange_v1 import ObservedDatedTreeV1, prepare_whole_tree_encoder_inputs_v1, prepare_census_exchange_bank_v1, permuted_group_control_v1, clade_turnover_rho_v1
from .census_tree_encoder_v1 import prepare_retained_observed_forest_v1

ROOT=Path(__file__).resolve().parents[2]
SEQUENCE_FIELDS={f.name for f in fields(GraphSequence)}
SOURCE_ROLES={'native_observed_ts','serialized_whole_observed','observed_roster','observed_grid','producer_receipt'}


def fingerprint_v1(value):
    if isinstance(value,Tensor):
        if value.device.type!='cpu' or value.requires_grad:
            raise ValueError('Certification requires fixed CPU observed tensors')
        array=value.detach().contiguous().numpy()
        return {'dtype':str(value.dtype),'shape':list(value.shape),'stride':list(value.stride()),'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
    if is_dataclass(value):
        return {'type':type(value).__name__,'fields':{f.name:fingerprint_v1(getattr(value,f.name)) for f in fields(value)}}
    if isinstance(value,Mapping):return {str(k):fingerprint_v1(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(tuple,list)):return [fingerprint_v1(v) for v in value]
    if isinstance(value,(set,frozenset)):return [fingerprint_v1(v) for v in sorted(value)]
    if value is None or isinstance(value,(str,int,float,bool)):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite certification metadata')
        return value
    raise ValueError('Unsupported observation certificate type')


def fingerprint_sha256_v1(value):
    return hashlib.sha256(json.dumps(fingerprint_v1(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_census_biological_config_v1():
    return json.loads((ROOT/'configs/census_biological_bank_v1.json').read_text())


def _config(config):
    pinned=load_census_biological_config_v1()
    if config is None:config=pinned
    if config.get('fixture_mode') is True:
        allowed={'role_counts','references_per_class','fixture_mode'}
        if any(config.get(k)!=v for k,v in pinned.items() if k not in allowed):raise ValueError('Fixture may alter role counts only')
    elif config!=pinned:raise ValueError('Pinned join configuration changed')
    for path,expected in pinned['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned source drift: '+path)
    return dict(config)


@dataclass(frozen=True)
class ObservedSampleRoleV1:
    actual_id: int
    native_sample_node: int
    role: str
    age_generations: float
    observed_reference_class: int | None


def _roster(ts,rows,config):
    if not rows or any(not isinstance(row,ObservedSampleRoleV1) for row in rows):raise ValueError('Only explicit observed role rows are accepted')
    if any(type(row.actual_id) is not int or type(row.native_sample_node) is not int or not isinstance(row.age_generations,(int,float)) or isinstance(row.age_generations,bool) for row in rows):raise ValueError('Actual/native sample IDs and dates need explicit numeric types')
    ids=[row.actual_id for row in rows];native=[row.native_sample_node for row in rows]
    if (len(ids)!=len(set(ids)) or len(native)!=len(set(native)) or min(ids)<0
            or set(native)!=set(map(int,ts.samples()))):raise ValueError('Exact actual/native sample roster differs')
    if {role:sum(row.role==role for row in rows) for role in config['role_counts']}!=config['role_counts'] or any(row.role not in config['role_counts'] for row in rows):raise ValueError('Require scored/context/pure-reference role counts')
    for row in rows:
        age=float(ts.node(row.native_sample_node).time)
        if not math.isfinite(age) or age<0 or age!=row.age_generations:raise ValueError('Direct native sample date disagrees with observed roster')
        if row.role=='pure_source_reference':
            if type(row.observed_reference_class) is not int or row.observed_reference_class not in range(4):raise ValueError('Pure reference needs observed class')
        elif row.observed_reference_class is not None or age>=config['census']:raise ValueError('All Q must be strictly younger350 with no reference class')
    refs=[row for row in rows if row.role=='pure_source_reference']
    if [sum(row.observed_reference_class==c for row in refs) for c in range(4)]!=[config['references_per_class']]*4:raise ValueError('Pure reference class counts differ')
    dynamic=[row for row in rows if row.role!='pure_source_reference'];scored=[row for row in rows if row.role=='query']
    landmarks=[]
    for c in range(4):
        panel=[row.native_sample_node for row in refs if row.observed_reference_class==c]
        landmarks.extend(panel[i] for i in np.linspace(0,len(panel)-1,4,dtype=int))
    return dynamic,scored,refs,landmarks


def _bindings(bindings):
    if set(bindings)!=SOURCE_ROLES:raise ValueError('Exact observation source-role whitelist required')
    for role,record in bindings.items():
        if set(record)!={'path','sha256'} or not isinstance(record['path'],str) or record['path'].startswith('/') or '..' in Path(record['path']).parts:
            raise ValueError('Source records require project-relative observed path/hash')
        digest=record['sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Observed source SHA256 required')
    return dict(bindings)


@dataclass(frozen=True)
class CertifiedCensusBiologicalBankV1:
    inputs: tuple
    bank: object
    forest: object
    native_trees: tuple
    stored_sequence: GraphSequence
    provenance: dict
    config: dict
    certificate_sha256: str


def _content(cert):
    return {'inputs':cert.inputs,'bank':cert.bank,'forest':cert.forest,'native_trees':cert.native_trees,
            'stored_sequence':cert.stored_sequence,'provenance':cert.provenance,'config':cert.config}


def validate_census_biological_certificate_v1(cert,expected_certificate_sha256):
    if not expected_certificate_sha256 or cert.certificate_sha256!=expected_certificate_sha256 or fingerprint_sha256_v1(_content(cert))!=expected_certificate_sha256:
        raise ValueError('Changed/stale native,feature,donor,cut,forest,source or role certificate')
    _config(cert.config)
    return cert


def prepare_census_biological_bank_v1(ts,serialized_observation:Mapping,roster:Sequence[ObservedSampleRoleV1],
        *,left_bp:float,right_bp:float,offset_bp:float,positions_bp:Tensor,original_grid_indexes:Tensor,
        source_bindings:Mapping,parent_id:str,split:str,tile_id:int,config=None):
    """Join admitted decoded observations; do not open project files or targets.

    Reproduce the pinned producer's full native postorder x48/edges/landmarks
    first. Only after exact full parity may branch-column deletion and anonymous
    unary suppression occur. No whole-TS million-node F47 table is constructed.
    """
    config=_config(config);bindings=_bindings(source_bindings)
    dynamic,scored,refs,landmarks=_roster(ts,roster,config)
    if split not in {'train','val'} or not isinstance(parent_id,str) or not isinstance(tile_id,int) or tile_id not in range(8):raise ValueError('Development parent/split/tile metadata required')
    if (not 0<=left_bp<right_bp<=config['normalization_length_bp'] or not math.isfinite(offset_bp)
            or positions_bp.device.type!='cpu' or positions_bp.dtype!=torch.float64 or positions_bp.ndim!=1 or not positions_bp.numel()
            or original_grid_indexes.device.type!='cpu' or original_grid_indexes.dtype!=torch.long or original_grid_indexes.shape!=positions_bp.shape
            or not torch.equal(positions_bp,50.+100.*original_grid_indexes.double())
            or not bool(((positions_bp>=left_bp)&(positions_bp<right_bp)).all())
            or not bool((original_grid_indexes[1:]>original_grid_indexes[:-1]).all())):raise ValueError('Exact original100bp centers/order and physical tile required')
    all_centers=torch.arange(50.,config['normalization_length_bp'],100.,dtype=torch.float64)
    expected_indexes=torch.nonzero((all_centers>=left_bp)&(all_centers<right_bp),as_tuple=False).flatten()
    if not torch.equal(original_grid_indexes,expected_indexes) or not tile_id*config['tile_bp']<=left_bp<right_bp<=(tile_id+1)*config['tile_bp']:
        raise ValueError('All original centers and original tile identity must be preserved')
    if set(serialized_observation)!={'sequence','labels_in_input'} or serialized_observation['labels_in_input'] is not False:
        raise ValueError('Observation whitelist excludes targets/truth/labels')
    if set(serialized_observation['sequence'])!=SEQUENCE_FIELDS:raise ValueError('Exact observation GraphSequence fields required')
    stored=GraphSequence.from_dict(serialized_observation['sequence'])
    if any(isinstance(getattr(stored,f.name),Tensor) and (getattr(stored,f.name).device.type!='cpu' or getattr(stored,f.name).requires_grad) for f in fields(stored)):
        raise ValueError('Stored observations must be fixed CPU tensors')
    if stored.x.dtype!=torch.float32 or stored.x.shape[1]!=48:raise ValueError('Pinned stored x48 float32 required before F47 deletion')
    n_bp=config['normalization_length_bp'];all_samples={row.native_sample_node for row in roster}
    native_to_actual={row.native_sample_node:row.actual_id for row in roster};reference={row.native_sample_node:row.observed_reference_class for row in refs};actual_scored={row.native_sample_node for row in scored}
    xs,clades,node_ids,node_graph,edges,intervals,native_maps,native_trees,tree_ids,excluded,durations=[],[],[],[],[],[],[],[],[],[],[]
    physical_intervals=[]
    for tree in ts.trees():
        start=max(float(tree.interval.left)+offset_bp,left_bp);stop=min(float(tree.interval.right)+offset_bp,right_bp)
        if stop<=start:continue
        union=set()
        for sample in all_samples:
            node=sample;path=set()
            while node!=-1:
                if node in path:raise ValueError('Native sample-ancestor cycle')
                path.add(node);union.add(node);node=tree.parent(node)
        packed=list(tree.nodes(order='postorder'))
        if set(packed)!=union or len(packed)!=len(union) or tree.num_roots!=1 or set(tree.leaves())!=all_samples:raise ValueError('Pinned native postorder must equal complete sample-ancestor union')
        lookup={node:len(xs)+i for i,node in enumerate(packed)};local={node:i for i,node in enumerate(packed)};graph=len(intervals)
        parents,times,branch=[],[],[]
        for node in packed:
            ancestor=tree.parent(node);age=float(tree.time(node));dt=0. if ancestor==-1 else float(tree.time(ancestor))-age
            if not math.isfinite(age) or age<0 or dt<0:raise ValueError('Direct native float64 dates/durations invalid')
            feature=np.zeros(8,dtype=np.float32)
            if node in reference:feature[reference[node]]=1.
            feature[4]=float(node in actual_scored);feature[5]=float(node in all_samples)
            feature[6]=math.log1p(age)/math.log1p(config['time_scale']);feature[7]=math.log1p(dt)/math.log1p(config['time_scale'])
            xs.append(feature);clades.append([float(tree.is_descendant(landmark,node)) for landmark in landmarks])
            node_ids.append(node if node in all_samples else -1);node_graph.append(graph)
            if ancestor!=-1:edges.extend([(lookup[ancestor],lookup[node]),(lookup[node],lookup[ancestor])])
            parents.append(-1 if ancestor==-1 else local[ancestor]);times.append(age);branch.append(dt)
        native_trees.append(ObservedDatedTreeV1(torch.tensor(parents),torch.tensor(times,dtype=torch.float64),
            torch.tensor([row.actual_id for row in roster]),torch.tensor([local[row.native_sample_node] for row in roster])))
        native_maps.append(torch.tensor(packed));durations.append(torch.tensor(branch,dtype=torch.float64))
        intervals.append([start/n_bp,stop/n_bp]);physical_intervals.append([start,stop]);tree_ids.append(int(tree.index));excluded.append(ts.num_nodes-len(union))
    if not intervals:raise ValueError('No original trees in clipped support')
    intervals=torch.tensor(intervals,dtype=torch.float64)
    if (intervals[0,0].item()!=left_bp/n_bp or intervals[-1,1].item()!=right_bp/n_bp or not torch.equal(intervals[:-1,1],intervals[1:,0])):raise ValueError('Exact common clipped native coverage required')
    positions=torch.from_numpy(positions_bp.numpy()/n_bp)
    target=torch.tensor(np.searchsorted(intervals[:,1].numpy(),positions.numpy(),side='right'),dtype=torch.long)
    base=GraphSequence(torch.from_numpy(np.asarray(xs,dtype=np.float32)),torch.tensor(edges,dtype=torch.long).reshape(-1,2).T.contiguous(),
        torch.tensor(node_graph),torch.tensor(node_ids),torch.tensor(clades,dtype=torch.float32),intervals,
        torch.tensor([row.native_sample_node for row in dynamic]),positions,target,torch.tensor([left_bp/n_bp,right_bp/n_bp],dtype=torch.float64))
    expected=ReferenceFeatureAdapter(8,reference_columns=(0,1,2,3),time_column=6,branch_column=7,kernel_scales=config['sigmas']).precompute(base)
    for field in fields(stored):
        actual=getattr(stored,field.name);wanted=getattr(expected,field.name)
        if actual is None or wanted is None:
            equal=actual is wanted
        else:equal=actual.dtype==wanted.dtype and actual.shape==wanted.shape and actual.stride()==wanted.stride() and torch.equal(actual,wanted)
        if not equal:raise ValueError('Native/serialized exact parity failed: '+field.name)
    feature_columns=tuple(config['feature_columns']);inputs=[];row_maps=[];sample_maps=[];start_row=0
    for graph,tree in enumerate(native_trees):
        count=len(tree.parent);rows=torch.arange(start_row,start_row+count);f47=stored.x[rows][:,feature_columns].clone()
        prepared=prepare_whole_tree_encoder_inputs_v1(tree,f47)
        # Translate retained provenance back to exact native vertices and source rows.
        retained=prepared.native_vertex_index
        from dataclasses import replace
        prepared=replace(prepared,native_vertex_index=native_maps[graph][retained])
        inputs.append(prepared);row_maps.append(rows);sample_maps.append({'actual_ids':tree.sample_ids,'native_sample_nodes':torch.tensor([row.native_sample_node for row in roster]),'source_rows':rows[tree.sample_nodes]})
        start_row+=count
    bank=prepare_census_exchange_bank_v1(native_trees,intervals,positions,torch.tensor([row.actual_id for row in dynamic]),
        torch.tensor([row.actual_id for row in refs]),torch.tensor([row.actual_id for row in scored]),num_anchors=config['num_anchors'],sigmas=config['sigmas'])
    # Shared physical bp values have one division representation. Frozen anchor
    # arithmetic may differ; reject an intended collision instead of adding eps.
    anchor_bp=left_bp+(np.arange(config['num_anchors'],dtype=np.float64)+.5)*((right_bp-left_bp)/config['num_anchors'])
    event_bp=[left_bp]+[pair[1] for pair in physical_intervals]
    event_norm=[left_bp/n_bp]+intervals[:,1].tolist()
    for i,value in enumerate(anchor_bp):
        for bp,norm in zip(event_bp+positions_bp.tolist(),event_norm+positions.tolist()):
            if float(value)==bp and bank.layout.anchors[i].item()!=norm:raise ValueError('Physical anchor collision has unequal canonical normalized values')
    forest=prepare_retained_observed_forest_v1(inputs)
    provenance={'parent_id':parent_id,'split':split,'tile_id':tile_id,'source_bindings':bindings,'original_tree_ids':tree_ids,
        'native_vertex_by_source_row':native_maps,'source_rows_by_tree':row_maps,'sample_joins_by_tree':sample_maps,
        'native_endpoint_duration_by_source_row':durations,'landmark_native_nodes':landmarks,'excluded_unrelated_nodes_by_tree':excluded,
        'original_grid_indexes':original_grid_indexes.clone(),'positions_bp':positions_bp.clone(),'physical_bounds_bp':[left_bp,right_bp],
        'physical_intervals_bp':physical_intervals,'offset_bp':offset_bp,'feature_columns':feature_columns,'full_x48_and_all_GraphSequence_fields_exact_parity':True,
        'age_source':'direct_native_float64_never_inverted','unary_policy':'only_source-derived_non_sample_nonroot_unary_rows_disposable_after_full_parity',
        'source_native_float64_tree_fingerprint':fingerprint_sha256_v1(native_trees)}
    cert=CertifiedCensusBiologicalBankV1(tuple(inputs),bank,forest,tuple(native_trees),stored,provenance,config,'')
    from dataclasses import replace
    return replace(cert,certificate_sha256=fingerprint_sha256_v1(_content(cert)))


def forward_certified_census_biological_v1(model,cert,*,expected_certificate_sha256,control='proposed',group_permutation=None,readout_chunk_size=128):
    """Fail closed on stale source/geometry before calling the unchanged shared GNN."""
    validate_census_biological_certificate_v1(cert,expected_certificate_sha256)
    bank=cert.bank
    if control=='permuted_groups':
        if group_permutation is None:raise ValueError('Fixed global group permutation required')
        bank=permuted_group_control_v1(bank,group_permutation)
    elif group_permutation is not None:raise ValueError('Membership permutation belongs only to its declared contrast')
    return model(cert.inputs,bank,control=control,readout_chunk_size=readout_chunk_size)


def select_census_numerical_tiles_v1(metadata,*,expected_tiles=32):
    """Prespecify TRAIN min/max original-tree-count cells without outcomes."""
    allowed={'parent_id','split','tile_id','num_trees'}
    if len(metadata)!=expected_tiles or any(set(row)!=allowed for row in metadata):raise ValueError('All structural tile metadata only, no outcomes')
    identities=[(row['parent_id'],row['tile_id']) for row in metadata]
    if len(set(identities))!=len(identities) or any(row['split'] not in {'train','val'} or row['tile_id'] not in range(8) or type(row['num_trees']) is not int or row['num_trees']<1 for row in metadata):raise ValueError('Invalid development tile inventory')
    if expected_tiles==32 and (len({row['parent_id'] for row in metadata})!=4 or any({row['tile_id'] for row in metadata if row['parent_id']==parent}!=set(range(8)) for parent,_ in identities)):raise ValueError('Require all32 tiles of four parents')
    parent_splits={parent:{row['split'] for row in metadata if row['parent_id']==parent} for parent,_ in identities}
    if any(len(splits)!=1 for splits in parent_splits.values()) or (expected_tiles==32 and sum(splits=={'train'} for splits in parent_splits.values())!=2):raise ValueError('Require consistent twoTRAIN/twoVAL parent splits')
    train=[row for row in metadata if row['split']=='train']
    if not train:raise ValueError('TRAIN tiles required')
    return {'minimum':min(train,key=lambda row:(row['num_trees'],row['parent_id'],row['tile_id'])),
            'maximum':min(train,key=lambda row:(-row['num_trees'],row['parent_id'],row['tile_id']))}


def collect_census_biological_support_v1(certificates,expected_hashes,*,expected_tiles=32,hidden_dim=32):
    """Callable later after root array admission; structural observations only."""
    if len(certificates)!=len(expected_hashes):raise ValueError('Every tile requires its external certificate hash')
    output=[]
    for cert,digest in zip(certificates,expected_hashes):
        validate_census_biological_certificate_v1(cert,digest)
        spans=[right-left for left,right in cert.provenance['physical_intervals_bp']]
        weighted_size={};weighted_rho={};group_counts=[];single=0.;total=sum(spans)
        for graph,(part,span) in enumerate(zip(cert.bank.partitions,spans)):
            sizes=part.group_size_by_query.tolist();group_counts.append(len(part.group_members))
            for size in sizes:weighted_size[str(int(size))]=weighted_size.get(str(int(size)),0.)+span/len(sizes)/total
            single+=span/total*sum(size==1 for size in sizes)/len(sizes)
            if graph:
                for rho in clade_turnover_rho_v1(cert.bank.partitions[graph-1],part).tolist():weighted_rho[str(rho)]=weighted_rho.get(str(rho),0.)+span/len(sizes)
        normalizer=sum(weighted_rho.values())
        if normalizer:weighted_rho={key:value/normalizer for key,value in weighted_rho.items()}
        n=len(cert.forest.node);height=len(cert.forest.upward_levels)-1;depth=len(cert.forest.downward_levels)-1
        output.append({'parent_id':cert.provenance['parent_id'],'split':cert.provenance['split'],'tile_id':cert.provenance['tile_id'],
            'num_trees':len(cert.native_trees),'source_certificate_sha256':digest,'sample_count':len(cert.native_trees[0].sample_ids),
            'native_connected_node_counts':[len(tree.parent) for tree in cert.native_trees],'native_edge_counts':[len(tree.parent)-1 for tree in cert.native_trees],
            'excluded_unrelated_nodes':cert.provenance['excluded_unrelated_nodes_by_tree'],'older_R_proxy_count':len(cert.bank.partitions[0].older_reference_ids),
            'group_count_by_tree':group_counts,'physical_span_weighted_query_group_size':weighted_size,'physical_span_weighted_singleton_Q_fraction':single,
            'cross_query_exchange_possible_fraction':1-single,'right_hold_span_weighted_boundary_rho':weighted_rho,
            'maximum_retained_height':height,'maximum_retained_depth':depth,'forest_float32_state_copy_budget_bytes':2*(height+depth+2)*n*hidden_dim*4,
            'budget_scope':'conservative source estimate only, excludes other activations/allocator/donor/optimizer; no memory measurement',
            'limited_mechanism_opportunity':single>=cert.config['limited_singleton_Q_fraction'] or (bool(weighted_rho) and set(weighted_rho)=={'1.0'}),
            'mechanism_opportunity_scope':'singleton threshold>=0.95 or all boundary rho1; one-tree/no-boundary turnover is unmeasured; not a benefit finding'})
    selection=select_census_numerical_tiles_v1([{key:row[key] for key in ('parent_id','split','tile_id','num_trees')} for row in output],expected_tiles=expected_tiles)
    return {'schema':'census_biological_support_v1','scope':'observed-only structural data, no outcomes','tiles':output,'prespecified_TRAIN_numerical_cells':selection}
