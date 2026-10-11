"""Exact parent-context core. Standard-library source report by default.

No array reader, simulation, fit, prediction writer, device selector, budget or
execution admission exists. Actual data/GPU execution needs separate root gates.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/anchor_parent_context_v3.json'
CONFIG_SHA256='e8a2afdfd61495ed0acacbfe5376ed8785878b0fe417ba74c6947bd965ea38d5'

def source_report_v3():
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest()!=CONFIG_SHA256:
        raise ValueError('Pinned inactive parent-context configuration changed')
    cfg=json.loads(CONFIG.read_text())
    if cfg['execution_enabled'] is not False:raise ValueError('Configuration never admits actual execution')
    for p,h in cfg['source_pins'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Original source changed: '+p)
    for record in cfg['predecessor'].values():
        if hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Sealed predecessor/rejection changed: '+record['path'])
    return {'kind':'anchor_parent_context_v3_source_report','status':'source_only',
      'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'config_sha256':CONFIG_SHA256,'original_sources':cfg['source_pins'],
      'actual_arrays_fit_GPU_budget_or_installation':False}

def _dependencies():
    source_report_v3()
    import torch
    from . import anchor_kernel as legacy
    if Path(legacy.__file__).resolve()!=ROOT/'src/arggnn/anchor_kernel.py':
        raise ValueError('Exact original kernel module origin required')
    return torch,legacy

@dataclass(frozen=True)
class TileContextV3:
    layout: object
    operators: object
    query_ids: object
    reference_ids: object
    original_tree_ids: tuple[int,...]
    geometry: str = 'age'
    feature_fingerprints: tuple[str,...] | None = None
    parent_id: str | None = None
    split: str | None = None
    namespace: str | None = None

@dataclass(frozen=True)
class ParentContextV3:
    layout: object
    operators: object
    query_ids: object
    reference_ids: object
    geometry: str
    parent_id: str
    split: str
    namespace: str
    original_tree_ids_by_cache_graph: tuple[int,...]
    tile_anchor_ranges: tuple[tuple[int,int],...]
    tile_graph_ranges: tuple[tuple[int,int],...]
    tile_evaluation_coordinates: tuple[object,...]
    boundary_events: tuple[int,...]
    target_graph: object
    tile_contexts: tuple[TileContextV3,...]

@dataclass(frozen=True)
class ParentActionV3:
    values: object
    evaluation_ids: object
    forward_boundary_states: tuple[object,...]
    reverse_boundary_states: tuple[object,...]

@dataclass(frozen=True)
class ParentLatentContextV3:
    anchors: object
    targets: object
    target_indexes: object
    layers: tuple[object,...]

def _tile_evaluations(tile):
    torch,_=_dependencies();layout=tile.layout
    if len(layout.output_at_event)!=len(layout.coordinates) or len(layout.source_at_event)!=len(layout.coordinates):
        raise ValueError('Complete explicit event ownership required')
    out=torch.empty(layout.num_evaluations,dtype=torch.float64);seen=set();sources=set()
    for i,(s,o) in enumerate(zip(layout.source_at_event,layout.output_at_event)):
        if s>=0:
            if s in sources or s>=len(layout.anchors) or layout.coordinates[i]!=layout.anchors[s]:
                raise ValueError('Each cached source event must be owned exactly once')
            sources.add(s)
        elif s!=-1:raise ValueError('Invalid source ownership')
        if o>=0:
            if o in seen or o>=layout.num_evaluations:raise ValueError('Each evaluation event must be owned exactly once')
            out[o]=layout.coordinates[i];seen.add(o)
        elif o!=-1:raise ValueError('Invalid evaluation ownership')
    if seen!=set(range(layout.num_evaluations)) or sources!=set(range(len(layout.anchors))):
        raise ValueError('No missing/duplicate source or evaluation allowed')
    if not torch.equal(out[layout.anchor_output],layout.anchors):
        raise ValueError('Original cached anchor-output correspondence changed')
    return out

def stitch_context_v3(tiles,*,fixture=False):
    """Union exact cached layouts; never relocate anchors or deduplicate trees."""
    torch,legacy=_dependencies();tiles=tuple(tiles)
    if not tiles or (not fixture and len(tiles)!=8):raise ValueError('Complete eight-tile parent context required')
    identity=(tiles[0].parent_id,tiles[0].split,tiles[0].namespace)
    if any(not isinstance(x,str) or not x for x in identity) or identity[1] not in ('train','val'):
        raise ValueError('Explicit same-parent development namespace required')
    if not fixture and (not re.fullmatch(r'bio_paths_pilot_(train|val)_00[01]',identity[0])
      or identity[1] not in identity[0] or not re.fullmatch(re.escape(identity[0])+r':[0-9a-f]{64}',identity[2])):
        raise ValueError('Original parent/split/native-tree digest namespace required')
    if any((t.parent_id,t.split,t.namespace)!=identity for t in tiles):
        raise ValueError('Never stitch across parents, splits or native ARG namespaces')
    q=tiles[0].query_ids;r=tiles[0].reference_ids
    if q.dtype!=torch.long or r.dtype!=torch.long or q.device.type!='cpu' or r.device.type!='cpu':
        raise ValueError('Original CPU integer sample joins required')
    if q.ndim!=1 or r.ndim!=1 or not len(q) or not len(r) or len(set(q.tolist()+r.tolist()))!=len(q)+len(r):
        raise ValueError('Disjoint unique Q/R sample identities required')
    if not fixture and (len(q)!=184 or len(r)!=96):raise ValueError('Original Q184/R96 contract required')
    graph_offset=anchor_offset=0;graph_ranges=[];anchor_ranges=[];tree_ids=[];evaluations=[];boundaries=[];feature_by_tree={}
    arrays={k:[] for k in ('cqq','cqr','eigenvalues','eigenvectors')}
    for index,tile in enumerate(tiles):
        L=tile.layout;O=tile.operators
        if tile.geometry not in ('age','plain') or tile.geometry!=tiles[0].geometry:
            raise ValueError('One explicit original geometry per parent required')
        if not torch.equal(tile.query_ids,q) or not torch.equal(tile.reference_ids,r):
            raise ValueError('Persistent sampled-ID order differs across tiles')
        for field in ('anchors','weights','coordinates','segment_duration'):
            value=getattr(L,field)
            if value.device.type!='cpu' or value.dtype!=torch.float64 or value.requires_grad or value.ndim!=1:
                raise ValueError('Exact fixed CPU float64 coordinate metadata required')
        if len(L.coordinates)<2 or not bool((L.coordinates[1:]>L.coordinates[:-1]).all()):
            raise ValueError('Strict chronological coordinates required')
        if len(L.segment_graph)!=len(L.coordinates)-1 or not torch.equal(L.segment_duration,L.coordinates[1:]-L.coordinates[:-1]):
            raise ValueError('Original physical subhold durations required')
        if not len(L.anchors) or L.weights.shape!=L.anchors.shape or not bool((L.weights>0).all()):
            raise ValueError('Original positive anchor masses required')
        if not bool(((L.anchors>L.coordinates[0])&(L.anchors<L.coordinates[-1])).all()) or not bool((L.anchors[1:]>L.anchors[:-1]).all()):
            raise ValueError('Sources are strictly interior; artificial cut owns no anchor')
        if not fixture and len(L.anchors)!=32:raise ValueError('Original32 anchors per tile required')
        if index and tiles[index-1].layout.coordinates[-1]!=L.coordinates[0]:
            raise ValueError('Exact contiguous parent tile support required')
        if index:boundaries.append(L.coordinates[0])
        graphs=len(O.eigenvalues);scales=O.eigenvalues.shape[1]
        if any(getattr(O,f).requires_grad for f in ('cqq','cqr','eigenvalues','eigenvectors')):
            raise ValueError('Original fixed cached operators must not become learned tensors')
        legacy._check_operators(O,graphs,scales,len(q),len(r))
        if len(tile.original_tree_ids)!=graphs or any(type(t) is not int or t<0 for t in tile.original_tree_ids):
            raise ValueError('All original cache tree identities required')
        fingerprints=tile.feature_fingerprints
        if fingerprints is None:
            if not fixture:raise ValueError('Original fixed per-tree feature fingerprints required before actual integration')
        else:
            if len(fingerprints)!=graphs or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in fingerprints):
                raise ValueError('Exact original per-tree feature fingerprints required')
            for tree_id,fingerprint in zip(tile.original_tree_ids,fingerprints):
                if tree_id in feature_by_tree and feature_by_tree[tree_id]!=fingerprint:
                    raise ValueError('Duplicated clipped original-tree fixed features are incompatible')
                feature_by_tree[tree_id]=fingerprint
        if L.segment_graph.dtype!=torch.long or L.anchor_graph.dtype!=torch.long or L.anchor_output.dtype!=torch.long or L.target_output.dtype!=torch.long:
            raise ValueError('Integer cache/event joins required')
        if L.segment_graph.device.type!='cpu' or L.anchor_graph.device.type!='cpu' or bool((L.segment_graph<0).any()) or bool((L.segment_graph>=graphs).any()):
            raise ValueError('Original hold graph ownership required')
        if L.anchor_graph.shape!=L.anchors.shape or not torch.equal(L.anchor_graph,L.segment_graph[torch.searchsorted(L.coordinates,L.anchors,right=True)-1]):
            raise ValueError('Original anchor half-open graph ownership changed')
        if any(a>=b for a,b in zip(tile.original_tree_ids,tile.original_tree_ids[1:])):
            raise ValueError('Original tree IDs retain chronological order')
        ev=_tile_evaluations(tile)
        if bool(((ev[L.target_output]<L.coordinates[0])|(ev[L.target_output]>=L.coordinates[-1])).any()):
            raise ValueError('Half-open target ownership belongs to the right tile at a cut')
        # Same original tree straddling a cut must retain compatible operator data.
        if index and tiles[index-1].original_tree_ids[int(tiles[index-1].layout.segment_graph[-1])]==tile.original_tree_ids[int(L.segment_graph[0])]:
            prev=tiles[index-1];a=int(prev.layout.segment_graph[-1]);b=int(L.segment_graph[0])
            for field in ('cqq','cqr'):
                if not torch.equal(getattr(prev.operators,field)[a],getattr(O,field)[b]):
                    raise ValueError('Clipped occurrences of one original tree have incompatible operators')
        evaluations.append(ev);graph_ranges.append((graph_offset,graph_offset+graphs));anchor_ranges.append((anchor_offset,anchor_offset+len(L.anchors)))
        for field in arrays:arrays[field].append(getattr(O,field))
        tree_ids.extend(tile.original_tree_ids);graph_offset+=graphs;anchor_offset+=len(L.anchors)
    anchors=torch.cat([t.layout.anchors for t in tiles]);weights=torch.cat([t.layout.weights for t in tiles])
    targets=torch.cat([e[t.layout.target_output] for t,e in zip(tiles,evaluations)])
    ev=torch.unique(torch.cat((anchors,targets)),sorted=True)
    coords=torch.unique(torch.cat([t.layout.coordinates for t in tiles]),sorted=True)
    owners=torch.searchsorted(torch.stack([t.layout.coordinates[-1] for t in tiles]),coords[:-1],right=True)
    graphs=torch.empty(len(coords)-1,dtype=torch.long)
    for i,tile in enumerate(tiles):
        selected=owners==i
        local=torch.searchsorted(tile.layout.coordinates,coords[:-1][selected],right=True)-1
        graphs[selected]=tile.layout.segment_graph[local]+graph_ranges[i][0]
    source=[-1]*len(coords);output=[-1]*len(coords)
    for i,event in enumerate(torch.searchsorted(coords,anchors).tolist()):source[event]=i
    for i,event in enumerate(torch.searchsorted(coords,ev).tolist()):output[event]=i
    anchor_graph=torch.cat([t.layout.anchor_graph+rg[0] for t,rg in zip(tiles,graph_ranges)])
    layout=legacy.AnchorKernelLayout(anchors,weights,anchor_graph,coords,graphs,coords[1:]-coords[:-1],
      torch.searchsorted(ev,anchors),torch.searchsorted(ev,targets),tuple(source),tuple(output),len(ev))
    O=legacy.AnchorOperators(**{k:torch.cat(v) for k,v in arrays.items()})
    if any(a>b for a,b in zip(tree_ids,tree_ids[1:])):raise ValueError('Original parent tree chronology changed')
    target_graph=graphs[torch.searchsorted(coords,targets,right=True)-1]
    return ParentContextV3(layout,O,q.clone(),r.clone(),tiles[0].geometry,*identity,tuple(tree_ids),tuple(anchor_ranges),tuple(graph_ranges),
      tuple(evaluations),tuple(torch.searchsorted(coords,torch.stack(boundaries)).tolist()) if boundaries else (),target_graph,tiles)

def streamed_parent_action_v3(parent,sources,rates,damping,*,mode='carry',evaluation_ids=None,
        event_block_size=64,checkpoint_blocks=False,return_boundary_states=False):
    """Exact finite sum with ordinary autograd, fresh zero exterior states per call."""
    torch,legacy=_dependencies();L=parent.layout;O=parent.operators
    if mode not in ('carry','reset'):raise ValueError('Explicit matched reset/carry mode required')
    if type(event_block_size) is not int or event_block_size<1:raise ValueError('Positive finite event block required')
    if sources.ndim!=5 or sources.shape[0]!=len(L.anchors) or sources.shape[1:3]!=O.eigenvalues.shape[1:] or sources.shape[-1]!=2:
        raise ValueError('Carry per-scale unnormalized numerator AND matching mass [A,S,Q,H,2]')
    A,S,Q,H,V=sources.shape
    if rates.shape!=(S,H) or damping.shape!=(S,H) or not bool(torch.isfinite(sources).all()) or not bool(torch.isfinite(rates).all()) or not bool(torch.isfinite(damping).all()) or bool((rates<=0).any()) or bool((damping<=0).any()):
        raise ValueError('Finite sources and positive parameter-dependent rates/damping required')
    if any(x.device!=sources.device or x.dtype!=sources.dtype for x in (rates,damping)):
        raise ValueError('Explicit common differentiable source/rate device and dtype required')
    if sources.dtype not in (torch.float32,torch.float64):raise ValueError('Qualified float32/float64 only')
    if O.eigenvalues.requires_grad or O.eigenvectors.requires_grad:raise ValueError('Original cached operators are fixed')
    ids=torch.arange(L.num_evaluations,dtype=torch.long) if evaluation_ids is None else evaluation_ids
    if ids.device.type!='cpu' or ids.dtype!=torch.long or ids.ndim!=1 or len(set(ids.tolist()))!=len(ids) or bool((ids<0).any()) or bool((ids>=L.num_evaluations).any()):
        raise ValueError('Unique explicit original evaluation IDs required')
    selected=set(ids.tolist());boundary=set(parent.boundary_events)
    zero=sources.new_zeros((S,Q,H,2));n_events=len(L.coordinates)
    def scan(reverse):
        state=zero;emitted={};snapshots={}
        events=list(range(n_events-1,-1,-1)) if reverse else list(range(n_events))
        for offset in range(0,n_events,event_block_size):
            block=events[offset:offset+event_block_size]
            emitted_ids=[L.output_at_event[e] for e in block if L.output_at_event[e] in selected]
            snapshot_ids=[e for e in block if return_boundary_states and e in boundary]
            def run(incoming,signal,lam,delta,block=tuple(block)):
                current=incoming;values=[];snaps=[];fixed={}
                def hold(state,graph,dt):
                    if graph not in fixed:
                        fixed[graph]=(O.eigenvalues[graph].to(signal),O.eigenvectors[graph].to(signal))
                    eigenvalues,vectors=fixed[graph]
                    decay=eigenvalues[...,None]*lam[:,None,:]+delta[:,None,:]
                    if not bool(torch.isfinite(decay).all()) or bool((decay<=0).any()):
                        raise ValueError('Kernel decay is not numerically positive')
                    factors=torch.exp(-dt*decay)
                    return legacy._hold_action(state,vectors,factors)
                for event in block:
                    source,output=L.source_at_event[event],L.output_at_event[event]
                    if not reverse and event:
                        graph=int(L.segment_graph[event-1]);dt=L.segment_duration[event-1].to(signal)
                        current=hold(current,graph,dt)
                    if not reverse:
                        if return_boundary_states and event in boundary:snaps.append(current)
                        if mode=='reset' and event in boundary:current=zero
                        if source>=0:current=current+signal[source]*L.weights[source].to(signal)
                        if output in selected:values.append(current)
                    else:
                        if output in selected:values.append(current)  # reverse exclusive
                        if source>=0:current=current+signal[source]*L.weights[source].to(signal)
                        if return_boundary_states and event in boundary:snaps.append(current)
                        if mode=='reset' and event in boundary:current=zero  # cut output is right-owned
                        if event:
                            graph=int(L.segment_graph[event-1]);dt=L.segment_duration[event-1].to(signal)
                            current=hold(current,graph,dt)
                return current,torch.stack(values) if values else signal.new_empty((0,S,Q,H,2)),torch.stack(snaps) if snaps else signal.new_empty((0,S,Q,H,2))
            if checkpoint_blocks and torch.is_grad_enabled() and any(x.requires_grad for x in (state,sources,rates,damping)):
                from torch.utils.checkpoint import checkpoint
                state,values,snaps=checkpoint(run,state,sources,rates,damping,use_reentrant=False,preserve_rng_state=True)
            else:state,values,snaps=run(state,sources,rates,damping)
            emitted.update(zip(emitted_ids,values.unbind(0)));snapshots.update(zip(snapshot_ids,snaps.unbind(0)))
        return torch.stack([emitted[int(i)] for i in ids]),tuple(snapshots[e] for e in parent.boundary_events) if return_boundary_states else ()
    forward,fs=scan(False);reverse,rs=scan(True)
    return ParentActionV3(forward+reverse,ids,fs,rs)

def parent_layer_messages_v3(parent,query_anchor,reference_anchor,layer,*,mode='carry',
        target_indexes=None,event_block_size=64,checkpoint_blocks=False):
    """Same projected source1_Q-lambda*Cqr*1_R and normalization, no detach/floor."""
    torch,legacy=_dependencies();L=parent.layout;O=parent.operators
    if query_anchor.shape[0]!=len(L.anchors) or reference_anchor.shape[0]!=len(L.anchors):
        raise ValueError('Global exact union anchor sources required')
    target=torch.arange(len(L.target_output),dtype=torch.long) if target_indexes is None else target_indexes
    if target.dtype!=torch.long or target.device.type!='cpu' or target.ndim!=1 or bool((target<0).any()) or bool((target>=len(L.target_output)).any()):
        raise ValueError('Original target indexes required')
    q=layer.query_projection(query_anchor).tanh();r=layer.reference_projection(reference_anchor).tanh()
    lam,delta=layer.rates,layer.damping;cqr=O.cqr[L.anchor_graph.to(O.cqr.device)].to(q)
    sources=q[:,None]-(cqr@r[:,None])*lam[None,:,None,:]
    mass=torch.ones_like(sources)-cqr.sum(-1)[...,None]*lam[None,:,None,:]
    wanted=torch.unique(torch.cat((L.anchor_output,L.target_output[target])),sorted=True)
    action=streamed_parent_action_v3(parent,torch.stack((sources,mass),-1),lam,delta,mode=mode,
      evaluation_ids=wanted,event_block_size=event_block_size,checkpoint_blocks=checkpoint_blocks)
    mixed=(action.values*layer.mixture[None,:,None,:,None]).sum(1);numerator,denominator=mixed.unbind(-1)
    if not bool(torch.isfinite(mixed).all()) or bool((denominator<=0).any()):
        raise ValueError('Matching kernel mass is nonpositive/nonfinite; no floor')
    normalized=numerator/denominator
    ai=torch.searchsorted(wanted,L.anchor_output).to(normalized.device);ti=torch.searchsorted(wanted,L.target_output[target]).to(normalized.device)
    return legacy.AnchorLayerMessages(normalized[ai],normalized[ti],numerator,denominator,sources,mass)


@dataclass(frozen=True)
class ParentEncodedSourcesV3:
    query: object
    reference: object
    unique_original_tree_ids: tuple[int,...]
    first_cache_graph_indexes: tuple[int,...]
    inverse_cache_graph_indexes: tuple[int,...]

def _original_row_plan_v3(parent,arrays):
    """Compare only fixed inputs or initial encoder rows, never evolved states."""
    torch,_=_dependencies();ids=parent.original_tree_ids_by_cache_graph
    graphs=len(parent.operators.eigenvalues)
    if len(ids)!=graphs or any(type(t) is not int or t<0 for t in ids):
        raise ValueError('Original cache-graph tree correspondence required')
    if any(x.shape[0]!=graphs for x in arrays):raise ValueError('Complete original graph rows required')
    first={};indexes=[];inverse=[]
    for graph,tree in enumerate(ids):
        if tree not in first:
            first[tree]=len(indexes);indexes.append(graph)
        else:
            prior=indexes[first[tree]]
            if any(not torch.equal(x[prior],x[graph]) for x in arrays):
                raise ValueError('Repeated original-tree initial source rows are incompatible')
        inverse.append(first[tree])
    return tuple(ids[i] for i in indexes),tuple(indexes),tuple(inverse)

def encode_parent_sources_v3(parent,model,leaf_by_cache_graph,response_by_cache_graph,*,leaf_ids,fixture=False):
    """Encode each unique original tree once in one batch per call, then reuse rows.

    Fixed original inputs have explicit sampled-ID joins. Every cache graph and
    hold occurrence remains in the parent. No encoding state persists between
    calls or optimizer steps. Differentiable input fixtures need fixture=True.
    """
    torch,_=_dependencies();weight=model.encoder.input_linear.weight
    leaf=leaf_by_cache_graph;response=response_by_cache_graph
    graphs=len(parent.operators.eigenvalues);N=len(parent.query_ids)+len(parent.reference_ids)
    if not isinstance(leaf,torch.Tensor) or not isinstance(response,torch.Tensor) or leaf.shape!=(graphs,N,model.input_dim) or response.shape!=(graphs,N,N):
        raise ValueError('Complete original fixed leaf/response bank dimensions required')
    if any(x.dtype!=weight.dtype or x.device!=weight.device or not bool(torch.isfinite(x).all()) for x in (leaf,response)):
        raise ValueError('Finite fixed original model input device/dtype required')
    if type(fixture) is not bool:raise ValueError('Exact handmade fixture selector required')
    if not fixture and (leaf.requires_grad or response.requires_grad):
        raise ValueError('Production original feature/response bank must remain fixed')
    if not isinstance(leaf_ids,torch.Tensor) or leaf_ids.dtype!=torch.long or leaf_ids.device.type!='cpu' or leaf_ids.ndim!=1 or len(leaf_ids)!=N or len(set(leaf_ids.tolist()))!=N:
        raise ValueError('Explicit unique CPU integer leaf IDs required')
    if set(leaf_ids.tolist())!=set(parent.query_ids.tolist()+parent.reference_ids.tolist()):
        raise ValueError('Original Q/R sample identity join differs')
    tree_ids,first,inverse=_original_row_plan_v3(parent,(leaf,response))
    take=torch.tensor(first,dtype=torch.long,device=leaf.device)
    encoded=model.encoder(leaf.index_select(0,take),response.index_select(0,take))
    if encoded.shape!=(len(first),N,model.hidden_dim) or not bool(torch.isfinite(encoded).all()):
        raise ValueError('Original unique-tree encoder output shape/finite domain required')
    lookup={node:i for i,node in enumerate(leaf_ids.tolist())}
    qi=torch.tensor([lookup[node] for node in parent.query_ids.tolist()],dtype=torch.long,device=encoded.device)
    ri=torch.tensor([lookup[node] for node in parent.reference_ids.tolist()],dtype=torch.long,device=encoded.device)
    reuse=torch.tensor(inverse,dtype=torch.long,device=encoded.device)
    return ParentEncodedSourcesV3(encoded[:,qi].index_select(0,reuse),encoded[:,ri].index_select(0,reuse),tree_ids,first,inverse)

def parent_context_v3(parent,model,leaf_by_cache_graph,response_by_cache_graph,*,leaf_ids,fixture=False,
        mode='carry',target_indexes=None,event_block_size=64,checkpoint_blocks=False):
    """Production semantic entry: original-tree unique encoding before context."""
    encoded=encode_parent_sources_v3(parent,model,leaf_by_cache_graph,response_by_cache_graph,leaf_ids=leaf_ids,fixture=fixture)
    return _parent_context_encoded_v3(parent,model,encoded.query,encoded.reference,mode=mode,
      target_indexes=target_indexes,event_block_size=event_block_size,checkpoint_blocks=checkpoint_blocks)

def parent_encoded_context_v3(parent,model,query_by_cache_graph,reference_by_cache_graph,*,fixture=False,
        mode='carry',target_indexes=None,event_block_size=64,checkpoint_blocks=False):
    """Handmade-only entry for independently exercising initial-row rejection."""
    if fixture is not True:raise ValueError('Caller encoded tensors are restricted to explicit handmade fixtures')
    return _parent_context_encoded_v3(parent,model,query_by_cache_graph,reference_by_cache_graph,mode=mode,
      target_indexes=target_indexes,event_block_size=event_block_size,checkpoint_blocks=checkpoint_blocks)

def _optimizer_paths_v3(value):
    """Reachable optimizer leaf identities, without a second gradient traversal."""
    pending=[value.grad_fn] if value.grad_fn is not None else [];seen=set();ids=set()
    if value.is_leaf and value.requires_grad:ids.add(id(value))
    while pending:
        node=pending.pop()
        if node is None or node in seen:continue
        seen.add(node);variable=getattr(node,'variable',None)
        if variable is not None:ids.add(id(variable))
        pending.extend(next_node for next_node,_ in node.next_functions if next_node is not None)
    return ids

def _parent_context_encoded_v3(parent,model,query_by_cache_graph,reference_by_cache_graph,*,mode='carry',
        target_indexes=None,event_block_size=64,checkpoint_blocks=False):
    """Explicit parent metadata entry point; original model.encode_context is untouched.

    Inputs are same encoder outputs per original cache graph. Target states never
    become sources; each nonlinear stage uses the global anchor contexts.
    """
    torch,_=_dependencies();L=parent.layout
    if model.geometry!=parent.geometry or model.coupling!='full':
        raise ValueError('Explicit parent geometry and original full coupling required')
    if model.num_scales!=parent.operators.eigenvalues.shape[1]:raise ValueError('Original scale count differs')
    if model.num_anchors!=len(parent.tile_contexts[0].layout.anchors):
        raise ValueError('Original model per-tile anchor contract differs')
    if len(L.anchors)!=sum(len(t.layout.anchors) for t in parent.tile_contexts):
        raise ValueError('Explicit parent union anchor count required')
    if query_by_cache_graph.shape[:2]!=(len(parent.operators.eigenvalues),len(parent.query_ids)) or reference_by_cache_graph.shape[:2]!=(len(parent.operators.eigenvalues),len(parent.reference_ids)):
        raise ValueError('Original graph and sampled-ID source correspondence required')
    weight=model.encoder.input_linear.weight
    if any(x.ndim!=3 or x.shape[2]!=model.hidden_dim or x.dtype!=weight.dtype or x.device!=weight.device or not bool(torch.isfinite(x).all()) for x in (query_by_cache_graph,reference_by_cache_graph)):
        raise ValueError('Finite original initial encoder rows with model hidden/device/dtype required')
    _original_row_plan_v3(parent,(query_by_cache_graph,reference_by_cache_graph))
    target=torch.arange(len(L.target_output),dtype=torch.long) if target_indexes is None else target_indexes
    if target.dtype!=torch.long or target.device.type!='cpu' or target.ndim!=1 or bool((target<0).any()) or bool((target>=len(L.target_output)).any()):
        raise ValueError('Original target indexes required')
    anchors=query_by_cache_graph[L.anchor_graph.to(query_by_cache_graph.device)]
    targets=query_by_cache_graph[parent.target_graph[target].to(query_by_cache_graph.device)]
    fixed=reference_by_cache_graph[L.anchor_graph.to(reference_by_cache_graph.device)];messages=[]
    for layer in model.layers:
        m=parent_layer_messages_v3(parent,anchors,fixed,layer,mode=mode,target_indexes=target,
          event_block_size=event_block_size,checkpoint_blocks=checkpoint_blocks)
        anchors=layer.update(anchors,m.anchors);targets=layer.update(targets,m.targets);messages.append(m)
    return ParentLatentContextV3(anchors,targets,target,tuple(messages))

def matched_parent_step_v3(optimizer,parameters,loss_cells,*,gradient_clip_norm=1.0,inactive_parameters=()):
    """Finite caller-supplied parent step, not a fit/data/admission runner.

    Each callable returns (connected scalar loss numerator, fixed positive
    query-bp denominator). All cells use unchanged parameters, followed by
    one aggregate backward, one clip and one update. No source/target IO.
    """
    torch,_=_dependencies();parameters=tuple(parameters);loss_cells=tuple(loss_cells)
    ids=[id(p) for p in parameters];optim_ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    if not parameters or len(set(ids))!=len(ids) or len(set(optim_ids))!=len(optim_ids) or set(ids)!=set(optim_ids):
        raise ValueError('Exact unique complete optimizer parameter set required')
    inactive_parameters=tuple(inactive_parameters);inactive_ids={id(p) for p in inactive_parameters}
    if len(inactive_ids)!=len(inactive_parameters) or not inactive_ids<=set(ids):
        raise ValueError('Exact unique declared inactive optimizer subset required')
    active_ids=set(ids)-inactive_ids
    if not active_ids:raise ValueError('At least one intended active optimizer parameter required')
    if any(not isinstance(p,torch.Tensor) or not p.is_leaf or not bool(torch.isfinite(p).all()) or (id(p) in active_ids and not p.requires_grad) for p in parameters):
        raise ValueError('Finite leaf optimizer parameters and differentiable active parameters required')
    binding=tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups)
    if not loss_cells or any(not callable(f) for f in loss_cells):raise ValueError('Finite explicit parent loss cells required')
    if gradient_clip_norm!=1.0:raise ValueError('Original single aggregate clipping norm1 required')
    optimizer.zero_grad(set_to_none=True);numerators=[];denominators=[]
    version=[p._version for p in parameters]
    for cell in loss_cells:
        numerator,denominator=cell()
        if any(p.grad is not None for p in parameters):
            raise ValueError('Parent loss callbacks must leave optimizer gradients unset')
        if not isinstance(numerator,torch.Tensor) or numerator.ndim!=0 or not numerator.requires_grad or not bool(torch.isfinite(numerator)):
            raise ValueError('Finite connected scalar loss numerator required')
        if type(denominator) not in (int,float) or denominator<=0 or not math.isfinite(denominator):
            raise ValueError('Positive finite fixed query-bp denominator required')
        if any(p._version!=v for p,v in zip(parameters,version)):raise ValueError('Parameters changed during parent loss evaluation')
        if tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups)!=binding:
            raise ValueError('Optimizer parameter binding changed during parent loss evaluation')
        if not _optimizer_paths_v3(numerator)&active_ids:
            raise ValueError('Parent loss cell has no gradient path to intended active optimizer parameters')
        numerators.append(numerator);denominators.append(denominator)
    loss=torch.stack(numerators).sum()/sum(denominators)
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite exact parent-normalized loss')
    connected=_optimizer_paths_v3(loss)&set(ids)
    if connected!=active_ids:
        raise ValueError('Aggregate gradient paths differ from intended active optimizer parameters; declare individually inactive parameters')
    if any(p.grad is not None for p in parameters):
        raise ValueError('Optimizer gradients must be unset immediately before aggregate backward')
    loss.backward()
    if tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups)!=binding:
        raise ValueError('Optimizer parameter binding changed during parent backward')
    if any((id(p) in active_ids and p.grad is None) or (id(p) in inactive_ids and p.grad is not None) for p in parameters):
        raise ValueError('Actual connected optimizer gradients differ from declared active/inactive set')
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('Nonfinite full-parent gradient')
    norm=torch.nn.utils.clip_grad_norm_(parameters,gradient_clip_norm,error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):raise ValueError('Nonfinite finite parent update')
    return {'loss':float(loss.detach()),'query_bp':sum(denominators),'loss_cells':len(loss_cells),
      'gradient_norm_before_single_clip':float(norm),'backward_calls':1,'clip_calls':1,'optimizer_steps':1,
      'parameters_fixed_across_loss_cells':True,'connected_optimizer_parameters':len(connected),
      'declared_inactive_parameters':len(inactive_ids),'optimizer_parameter_binding_verified':True,'actual_fit_or_data_admission':False}

if __name__=='__main__':print(json.dumps(source_report_v3(),indent=2))
