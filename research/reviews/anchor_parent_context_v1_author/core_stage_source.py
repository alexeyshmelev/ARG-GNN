"""Exact parent-context core. Standard-library source report by default.

No array reader, simulation, fit, prediction writer, device selector, budget or
execution admission exists. Actual data/GPU execution needs separate root gates.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/anchor_parent_context_v1.json'
CONFIG_SHA256='4a365c3bd9c05f85e70192f3c39b97b08fd709ce77de5c19b2c3630effa75fdb'

def source_report_v1():
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest()!=CONFIG_SHA256:
        raise ValueError('Pinned inactive parent-context configuration changed')
    cfg=json.loads(CONFIG.read_text())
    if cfg['execution_enabled'] is not False:raise ValueError('Configuration never admits actual execution')
    for p,h in cfg['source_pins'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Original source changed: '+p)
    return {'kind':'anchor_parent_context_v1_source_report','status':'source_only',
      'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'config_sha256':CONFIG_SHA256,'original_sources':cfg['source_pins'],
      'actual_arrays_fit_GPU_budget_or_installation':False}

def _dependencies():
    source_report_v1()
    import torch
    from . import anchor_kernel as legacy
    if Path(legacy.__file__).resolve()!=ROOT/'src/arggnn/anchor_kernel.py':
        raise ValueError('Exact original kernel module origin required')
    return torch,legacy

@dataclass(frozen=True)
class TileContextV1:
    layout: object
    operators: object
    query_ids: object
    reference_ids: object
    original_tree_ids: tuple[int,...]
    geometry: str = 'age'

@dataclass(frozen=True)
class ParentContextV1:
    layout: object
    operators: object
    query_ids: object
    reference_ids: object
    geometry: str
    original_tree_ids_by_cache_graph: tuple[int,...]
    tile_anchor_ranges: tuple[tuple[int,int],...]
    tile_graph_ranges: tuple[tuple[int,int],...]
    tile_evaluation_coordinates: tuple[object,...]
    boundary_events: tuple[int,...]
    target_graph: object
    tile_contexts: tuple[TileContextV1,...]

@dataclass(frozen=True)
class ParentActionV1:
    values: object
    evaluation_ids: object
    forward_boundary_states: tuple[object,...]
    reverse_boundary_states: tuple[object,...]

@dataclass(frozen=True)
class ParentLatentContextV1:
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

def stitch_context_v1(tiles,*,fixture=False):
    """Union exact cached layouts; never relocate anchors or deduplicate trees."""
    torch,legacy=_dependencies();tiles=tuple(tiles)
    if not tiles or (not fixture and len(tiles)!=8):raise ValueError('Complete eight-tile parent context required')
    q=tiles[0].query_ids;r=tiles[0].reference_ids
    if q.dtype!=torch.long or r.dtype!=torch.long or q.device.type!='cpu' or r.device.type!='cpu':
        raise ValueError('Original CPU integer sample joins required')
    if q.ndim!=1 or r.ndim!=1 or not len(q) or not len(r) or len(set(q.tolist()+r.tolist()))!=len(q)+len(r):
        raise ValueError('Disjoint unique Q/R sample identities required')
    if not fixture and (len(q)!=184 or len(r)!=96):raise ValueError('Original Q184/R96 contract required')
    graph_offset=anchor_offset=0;graph_ranges=[];anchor_ranges=[];tree_ids=[];evaluations=[];boundaries=[]
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
        legacy._check_operators(O,graphs,scales,len(q),len(r))
        if len(tile.original_tree_ids)!=graphs or any(type(t) is not int or t<0 for t in tile.original_tree_ids):
            raise ValueError('All original cache tree identities required')
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
    return ParentContextV1(layout,O,q.clone(),r.clone(),tiles[0].geometry,tuple(tree_ids),tuple(anchor_ranges),tuple(graph_ranges),
      tuple(evaluations),tuple(torch.searchsorted(coords,torch.stack(boundaries)).tolist()) if boundaries else (),target_graph,tiles)

def streamed_parent_action_v1(parent,sources,rates,damping,*,mode='carry',evaluation_ids=None,
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
    if any(x.device!=sources.device or x.dtype!=sources.dtype for x in (rates,damping,O.eigenvalues,O.eigenvectors)):
        raise ValueError('Explicit common source/operator/rate device and dtype required')
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
                current=incoming;values=[];snaps=[]
                for event in block:
                    source,output=L.source_at_event[event],L.output_at_event[event]
                    if not reverse and event:
                        graph=int(L.segment_graph[event-1]);dt=L.segment_duration[event-1].to(signal)
                        factors=torch.exp(-dt*(O.eigenvalues[graph][...,None]*lam[:,None,:]+delta[:,None,:]))
                        current=legacy._hold_action(current,O.eigenvectors[graph],factors)
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
                            factors=torch.exp(-dt*(O.eigenvalues[graph][...,None]*lam[:,None,:]+delta[:,None,:]))
                            current=legacy._hold_action(current,O.eigenvectors[graph],factors)
                return current,torch.stack(values) if values else signal.new_empty((0,S,Q,H,2)),torch.stack(snaps) if snaps else signal.new_empty((0,S,Q,H,2))
            if checkpoint_blocks and torch.is_grad_enabled() and any(x.requires_grad for x in (state,sources,rates,damping)):
                from torch.utils.checkpoint import checkpoint
                state,values,snaps=checkpoint(run,state,sources,rates,damping,use_reentrant=False,preserve_rng_state=True)
            else:state,values,snaps=run(state,sources,rates,damping)
            emitted.update(zip(emitted_ids,values.unbind(0)));snapshots.update(zip(snapshot_ids,snaps.unbind(0)))
        return torch.stack([emitted[int(i)] for i in ids]),tuple(snapshots[e] for e in parent.boundary_events) if return_boundary_states else ()
    forward,fs=scan(False);reverse,rs=scan(True)
    return ParentActionV1(forward+reverse,ids,fs,rs)

def parent_layer_messages_v1(parent,query_anchor,reference_anchor,layer,*,mode='carry',
        target_indexes=None,event_block_size=64,checkpoint_blocks=False):
    """Same projected source1_Q-lambda*Cqr*1_R and normalization, no detach/floor."""
    torch,legacy=_dependencies();L=parent.layout;O=parent.operators
    if query_anchor.shape[0]!=len(L.anchors) or reference_anchor.shape[0]!=len(L.anchors):
        raise ValueError('Global exact union anchor sources required')
    target=torch.arange(len(L.target_output),dtype=torch.long) if target_indexes is None else target_indexes
    if target.dtype!=torch.long or target.device.type!='cpu' or target.ndim!=1 or bool((target<0).any()) or bool((target>=len(L.target_output)).any()):
        raise ValueError('Original target indexes required')
    q=layer.query_projection(query_anchor).tanh();r=layer.reference_projection(reference_anchor).tanh()
    lam,delta=layer.rates,layer.damping;cqr=O.cqr[L.anchor_graph.to(O.cqr.device)]
    sources=q[:,None]-(cqr@r[:,None])*lam[None,:,None,:]
    mass=torch.ones_like(sources)-cqr.sum(-1)[...,None]*lam[None,:,None,:]
    wanted=torch.unique(torch.cat((L.anchor_output,L.target_output[target])),sorted=True)
    action=streamed_parent_action_v1(parent,torch.stack((sources,mass),-1),lam,delta,mode=mode,
      evaluation_ids=wanted,event_block_size=event_block_size,checkpoint_blocks=checkpoint_blocks)
    mixed=(action.values*layer.mixture[None,:,None,:,None]).sum(1);numerator,denominator=mixed.unbind(-1)
    if not bool(torch.isfinite(mixed).all()) or bool((denominator<=0).any()):
        raise ValueError('Matching kernel mass is nonpositive/nonfinite; no floor')
    normalized=numerator/denominator
    ai=torch.searchsorted(wanted,L.anchor_output).to(normalized.device);ti=torch.searchsorted(wanted,L.target_output[target]).to(normalized.device)
    return legacy.AnchorLayerMessages(normalized[ai],normalized[ti],numerator,denominator,sources,mass)

def parent_context_v1(parent,model,query_by_cache_graph,reference_by_cache_graph,*,mode='carry',
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
    target=torch.arange(len(L.target_output),dtype=torch.long) if target_indexes is None else target_indexes
    anchors=query_by_cache_graph[L.anchor_graph.to(query_by_cache_graph.device)]
    targets=query_by_cache_graph[parent.target_graph[target].to(query_by_cache_graph.device)]
    fixed=reference_by_cache_graph[L.anchor_graph.to(reference_by_cache_graph.device)];messages=[]
    for layer in model.layers:
        m=parent_layer_messages_v1(parent,anchors,fixed,layer,mode=mode,target_indexes=target,
          event_block_size=event_block_size,checkpoint_blocks=checkpoint_blocks)
        anchors=layer.update(anchors,m.anchors);targets=layer.update(targets,m.targets);messages.append(m)
    return ParentLatentContextV1(anchors,targets,target,tuple(messages))

if __name__=='__main__':print(json.dumps(source_report_v1(),indent=2))
