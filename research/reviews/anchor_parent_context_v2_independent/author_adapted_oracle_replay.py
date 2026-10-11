"""Independent handmade CPU review: matrix-exponential impulse oracle and counterexamples."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib,json,os,types
import torch
import torch.nn.functional as F
from arggnn import anchor_kernel as old
from arggnn import anchor_parent_context_v2 as core

torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
ROOT=core.ROOT
DIR=ROOT/'research/reviews/anchor_parent_context_v2_independent'
cfg=json.loads(core.CONFIG.read_text())
checks=[];details={};findings=[]
def check(name,condition=True):
    if not bool(condition):raise AssertionError(name)
    checks.append(name)
def close(name,a,b,dtype,gradient=False):
    t=cfg['qualification'][str(dtype).split('.')[-1]]
    torch.testing.assert_close(a,b,atol=t['vjp_atol' if gradient else 'output_atol'],
                              rtol=t['vjp_rtol' if gradient else 'output_rtol'])
    check(name)
def gradients(name,a,b,dtype):
    for x,y in zip(a,b):
        if x is None or y is None:assert x is y
        else:
            t=cfg['qualification'][str(dtype).split('.')[-1]]
            torch.testing.assert_close(x,y,atol=t['vjp_atol'],rtol=t['vjp_rtol'])
    check(name)
def rejects(call):
    try:call()
    except (ValueError,RuntimeError,IndexError,TypeError):return True
    return False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={
 'src/arggnn/anchor_parent_context_v1.py':'3934a7b0e2afb105b42384ec562d509374cd336f9764a429d52ad40a8e4dbe05',
 'configs/anchor_parent_context_v1.json':'4a365c3bd9c05f85e70192f3c39b97b08fd709ce77de5c19b2c3630effa75fdb',
 'research/reviews/anchor_parent_context_v1_author/author_receipt.json':'76c54ba05e80e913f49ea15a660e0bbd3ceafe8e9b0344e660c2d3f5eb021843',
 **cfg['source_pins']}
for p,h in pins.items():check('assigned_pin_'+p,sha(ROOT/p)==h)
receipt=json.loads((ROOT/'research/reviews/anchor_parent_context_v1_author/author_receipt.json').read_text())
for item in receipt['files']:check('author_sealed_file_'+item['path'],sha(ROOT/item['path'])==item['sha256'])
check('successor_adapted_author_parity_152',json.loads((DIR/'core_checks.json').read_text())['count']==152)
torch.manual_seed(24011)
Q,R,S,H=2,1,2,2
fields=('cqq','cqr','eigenvalues','eigenvectors')

def fixture(dtype=torch.float64):
    matrices=torch.tensor([[[1.1,-.15],[-.15,1.8]],
                           [[2.0,-.45],[-.45,.8]],
                           [[.9,-.22],[-.22,1.5]],
                           [[1.8,-.08],[-.08,1.0]]],dtype=torch.float64)
    cqq=torch.stack((matrices,matrices*1.7),1)
    values,vectors=torch.linalg.eigh(cqq)
    cqr=-torch.tensor([.12,.35,.21,.08,.09,.18,.27,.16,
                      .17,.23,.31,.11,.05,.19,.24,.14],dtype=torch.float64).reshape(4,S,Q,R)
    ops=old.AnchorOperators(*(v.to(dtype) for v in (cqq,cqr,values,vectors)))
    tiles=[]
    for i,bounds in enumerate(([0.,.3,.7],[.7,1.1,1.4],[1.4,1.9,2.2])):
        left,middle,right=bounds
        anchor0=left+((right-left)/3)*.5
        positions=torch.tensor([left,anchor0,middle,right-.011],dtype=torch.float64)
        intervals=torch.tensor([[left,middle],[middle,right]],dtype=torch.float64)
        graph=torch.searchsorted(intervals[:,0].contiguous(),positions,right=True)-1
        bank=types.SimpleNamespace(intervals=intervals,positions=positions,target_graph=graph,
                                  window_bounds=torch.tensor([left,right],dtype=torch.float64))
        layout=old._layout(bank,3)
        tree_ids=(i+20,i+21)
        subset=old.AnchorOperators(*(getattr(ops,f)[i:i+2] for f in fields))
        tiles.append(core.TileContextV2(layout,subset,torch.tensor([7,11]),torch.tensor([42]),tree_ids,
                     feature_fingerprints=tuple(hashlib.sha256(str(t).encode()).hexdigest() for t in tree_ids),
                     parent_id='handmade_independent_train',split='train',namespace='independent_ARG'))
    tiles=tuple(tiles)
    return tiles,core.stitch_context_v2(tiles,fixture=True)

def output_events(parent):
    result=[None]*parent.layout.num_evaluations
    for event,output in enumerate(parent.layout.output_at_event):
        if output>=0:result[output]=event
    return result

def direct_impulses(parent,sources,rates,damping,mode='carry',events=None,allow=None):
    """Independent finite impulse sum using Cqq matrix_exp and explicit ordered products.
    Never calls legacy eigen-hold/action or streamed recurrence.
    """
    L=parent.layout
    events=output_events(parent) if events is None else list(events)
    C=parent.operators.cqq[L.segment_graph].to(sources)
    eye=torch.eye(Q,dtype=sources.dtype)
    K=C[:,:,None]*rates[None,:,:,None,None]+eye*damping[None,:,:,None,None]
    exp=torch.matrix_exp(-L.segment_duration.to(sources)[:,None,None,None,None]*K)
    anchor_events=torch.searchsorted(L.coordinates,L.anchors).tolist()
    rights=torch.stack([t.layout.coordinates[-1] for t in parent.tile_contexts])
    anchor_owner=torch.searchsorted(rights,L.anchors,right=True).tolist()
    result=[]
    for out_index,event in enumerate(events):
        coordinate=L.coordinates[event]
        owner=int(torch.searchsorted(rights,coordinate,right=True))
        terms=[]
        for a,source_event in enumerate(anchor_events):
            if mode=='reset' and anchor_owner[a]!=owner:continue
            if allow is not None and not allow(a,out_index):continue
            term=(sources[a]*L.weights[a].to(sources)).permute(0,2,1,3)
            segments=range(source_event,event) if source_event<=event else range(source_event-1,event-1,-1)
            for segment in segments:term=exp[segment]@term
            terms.append(term.permute(0,2,1,3))
        result.append(torch.stack(terms).sum(0) if terms else sources.new_zeros((S,Q,H,2)))
    return torch.stack(result)

for dtype in (torch.float64,torch.float32):
    tiles,parent=fixture(dtype);L=parent.layout
    check('three_tile_exact_anchor_mass_union_'+str(dtype),
          torch.equal(L.anchors,torch.cat([t.layout.anchors for t in tiles])) and
          torch.equal(L.weights,torch.cat([t.layout.weights for t in tiles])))
    check('two_duplicate_tree_occurrences_preserved_'+str(dtype),
          parent.original_tree_ids_by_cache_graph==(20,21,21,22,22,23))
    check('two_artificial_cuts_and_source_coincidence_'+str(dtype),
          len(parent.boundary_events)==2 and all(int(t.layout.target_output[1])==int(t.layout.anchor_output[0]) for t in tiles))
    check('noncommuting_fixture_independent_'+str(dtype),
          not torch.allclose(parent.operators.cqq[0]@parent.operators.cqq[1],
                             parent.operators.cqq[1]@parent.operators.cqq[0]))
    sources=torch.randn(len(L.anchors),S,Q,H,2,dtype=dtype,requires_grad=True)
    rates=torch.tensor([[.7,1.3],[1.1,.9]],dtype=dtype,requires_grad=True)
    damping=torch.tensor([[.2,.4],[.3,.5]],dtype=dtype,requires_grad=True)
    coefficient=torch.randn(L.num_evaluations,S,Q,H,2,dtype=dtype)
    inputs=(sources,rates,damping)
    for mode in ('carry','reset'):
        exact=direct_impulses(parent,*inputs,mode=mode)
        eg=torch.autograd.grad((exact*coefficient).sum(),inputs,retain_graph=True)
        actual=core.streamed_parent_action_v2(parent,*inputs,mode=mode,event_block_size=5)
        close('matrix_exponential_finite_impulse_values_'+str(dtype)+'_'+mode,actual.values,exact,dtype)
        ag=torch.autograd.grad((actual.values*coefficient).sum(),inputs,retain_graph=True)
        gradients('matrix_exponential_finite_impulse_VJP_'+str(dtype)+'_'+mode,ag,eg,dtype)
        ids=torch.tensor([L.num_evaluations-1,2,0,7])
        for block in (1,2,3,5,7,64):
            selective=core.streamed_parent_action_v2(parent,*inputs,mode=mode,evaluation_ids=ids,
                        event_block_size=block,checkpoint_blocks=True,return_boundary_states=True)
            close('checkpoint_selected_order_'+str(dtype)+'_'+mode+'_'+str(block),selective.values,exact[ids],dtype)
            coeff=coefficient[ids]
            sg=torch.autograd.grad((selective.values*coeff).sum(),inputs,retain_graph=True)
            qg=torch.autograd.grad((exact[ids]*coeff).sum(),inputs,retain_graph=True)
            gradients('checkpoint_selected_VJP_'+str(dtype)+'_'+mode+'_'+str(block),sg,qg,dtype)
        boundary=core.streamed_parent_action_v2(parent,*inputs,mode=mode,event_block_size=2,
                                                checkpoint_blocks=True,return_boundary_states=True)
        anchor_event=torch.searchsorted(L.coordinates,L.anchors).tolist()
        owners=[i for i,(a,b) in enumerate(parent.tile_anchor_ranges) for _ in range(a,b)]
        fs=direct_impulses(parent,*inputs,events=parent.boundary_events,
             allow=lambda a,j:anchor_event[a]<=parent.boundary_events[j] and (mode=='carry' or owners[a]==j))
        rs=direct_impulses(parent,*inputs,events=parent.boundary_events,
             allow=lambda a,j:anchor_event[a]>parent.boundary_events[j] and (mode=='carry' or owners[a]==j+1))
        close('forward_incoming_cut_snapshot_'+str(dtype)+'_'+mode,torch.stack(boundary.forward_boundary_states),fs,dtype)
        close('reverse_incoming_cut_snapshot_'+str(dtype)+'_'+mode,torch.stack(boundary.reverse_boundary_states),rs,dtype)
        bc=torch.randn_like(fs)
        objective=((torch.stack(boundary.forward_boundary_states)+torch.stack(boundary.reverse_boundary_states))*bc).sum()
        bg=torch.autograd.grad(objective,inputs,retain_graph=True)
        og=torch.autograd.grad(((fs+rs)*bc).sum(),inputs,retain_graph=True)
        gradients('checkpoint_boundary_state_VJP_'+str(dtype)+'_'+mode,bg,og,dtype)
    # Both numerators and mass depend on lambda, and normalized values use their ratio.
    q=torch.randn(len(L.anchors),Q,H,dtype=dtype,requires_grad=True)
    r=torch.randn(len(L.anchors),R,H,dtype=dtype,requires_grad=True)
    cqr=parent.operators.cqr[L.anchor_graph]
    numerator=q[:,None].tanh()-(cqr@r[:,None].tanh())*rates[None,:,None,:]
    mass=torch.ones_like(numerator)-cqr.sum(-1)[...,None]*rates[None,:,None,:]
    bundled=torch.stack((numerator,mass),-1)
    scanned=core.streamed_parent_action_v2(parent,bundled,rates,damping,event_block_size=2,checkpoint_blocks=True).values
    oracle=direct_impulses(parent,bundled,rates,damping)
    mixture=torch.tensor([[.3,.6],[.7,.4]],dtype=dtype)
    mixed=(scanned*mixture[None,:,None,:,None]).sum(1)
    omixed=(oracle*mixture[None,:,None,:,None]).sum(1)
    norm=mixed[...,0]/mixed[...,1];onorm=omixed[...,0]/omixed[...,1]
    close('parameter_dependent_numerator_mass_normalized_oracle_'+str(dtype),norm,onorm,dtype)
    nc=torch.randn_like(norm)
    ng=torch.autograd.grad((norm*nc).sum(),(q,r,rates,damping),retain_graph=True)
    ong=torch.autograd.grad((onorm*nc).sum(),(q,r,rates,damping),retain_graph=True)
    gradients('parameter_dependent_numerator_mass_normalized_VJP_'+str(dtype),ng,ong,dtype)
    detached_mass=torch.stack((numerator,mass.detach()),-1)
    dvalues=core.streamed_parent_action_v2(parent,detached_mass,rates,damping).values
    dmixed=(dvalues*mixture[None,:,None,:,None]).sum(1)
    dg=torch.autograd.grad(((dmixed[...,0]/dmixed[...,1])*nc).sum(),rates,retain_graph=True)[0]
    check('mass_detachment_counterexample_changes_rate_gradient_'+str(dtype),(dg-ng[2]).abs().max()>1e-5)
    details['mass_detachment_gradient_max_difference_'+str(dtype)]=float((dg-ng[2]).abs().max())
    # Fresh parent identity and call order cannot import prior carry states.
    other=core.stitch_context_v2(tuple(replace(t,parent_id='handmade_other_train',namespace='other_ARG') for t in tiles),fixture=True)
    first=core.streamed_parent_action_v2(parent,*inputs).values
    core.streamed_parent_action_v2(other,sources*100,rates,damping)
    after=core.streamed_parent_action_v2(parent,*inputs).values
    close('parent_call_interleaving_has_no_state_leak_'+str(dtype),first,after,dtype)
    check('other_parent_zero_sources_zero_every_readout_'+str(dtype),
          torch.count_nonzero(core.streamed_parent_action_v2(other,torch.zeros_like(sources),rates,damping).values)==0)
    check('cross_parent_stitch_rejected_'+str(dtype),
          rejects(lambda:core.stitch_context_v2((tiles[0],replace(tiles[1],parent_id='other'),tiles[2]),fixture=True)))

# Independent aggregate reset objective uses weighted classification losses and
# original nonlinear source encoder / tile model, plus realistic AdamW groups.
for dtype in (torch.float64,torch.float32):
    tiles,parent=fixture(dtype)
    model=old.AnchorKernelModel(input_dim=4,num_classes=3,hidden_dim=5,latent_dim=H,num_layers=2,
         kernel_depth=2,num_scales=S,num_anchors=3,geometry='age',initial_coupling=(.7,1.1),
         initial_damping=(.3,.5)).to(dtype)
    ref=deepcopy(model)
    unique=torch.randn(4,Q+R,4,dtype=dtype)
    leaf=unique[torch.tensor([0,1,1,2,2,3])]
    response=torch.tensor([[1.,-.13,-.05],[-.13,1.2,-.08],[-.05,-.08,.9]],dtype=dtype).expand(6,3,3)
    labels=torch.arange(12*Q).reshape(12,Q)%3
    bp=torch.arange(1,12*Q+1,dtype=dtype).reshape(12,Q)
    parameters=tuple(model.parameters());rp=tuple(ref.parameters())
    def numerator_for(logits,index):
        ce=F.cross_entropy(logits.reshape(-1,3),labels[index].reshape(-1),reduction='none').reshape(-1,Q)
        return (ce*bp[index]).sum(),float(bp[index].sum())
    def parent_cell(index):
        encoded=model.encoder(leaf,response)
        context=core.parent_context_v2(parent,model,leaf,response,leaf_ids=torch.cat((parent.query_ids,parent.reference_ids)),fixture=True,mode='reset',
                   target_indexes=index,event_block_size=2,checkpoint_blocks=True)
        return numerator_for(model.head(context.targets),index)
    def tile_cells():
        cells=[]
        for i,(tile,(a,b)) in enumerate(zip(tiles,parent.tile_graph_ranges)):
            pos=parent.tile_evaluation_coordinates[i][tile.layout.target_output]
            target_graph=tile.layout.segment_graph[torch.searchsorted(tile.layout.coordinates,pos,right=True)-1]
            bank=types.SimpleNamespace(leaf_x=leaf[a:b],query_cache=types.SimpleNamespace(normalized_response=response[a:b],
                 query_indices=torch.arange(Q),reference_indices=torch.arange(Q,Q+R)),target_graph=target_graph)
            prepared=types.SimpleNamespace(bank=bank,layout=tile.layout,operators={'age':tile.operators},num_scales=S)
            cells.append(numerator_for(ref(prepared),torch.arange(i*4,i*4+4)))
        return cells
    all_ids=torch.arange(12)
    pn,pd=parent_cell(all_ids);tc=tile_cells()
    parent_loss=pn/pd
    reference_loss=torch.stack([n for n,d in tc]).sum()/sum(d for n,d in tc)
    close('weighted_crossentropy_reset_parent_tile_loss_'+str(dtype),parent_loss,reference_loss,dtype)
    gradients('weighted_crossentropy_reset_all_parameter_VJP_'+str(dtype),
          torch.autograd.grad(parent_loss,parameters,allow_unused=True),
          torch.autograd.grad(reference_loss,rp,allow_unused=True),dtype)
    def optimizer(which):
        rates=list(which.rate_and_mixture_parameters());rate_ids={id(p) for p in rates}
        ordinary=[p for p in which.parameters() if id(p) not in rate_ids]
        return torch.optim.AdamW([{'params':ordinary,'weight_decay':.01},{'params':rates,'weight_decay':0.}],
                                lr=.002,betas=(.8,.95))
    opt=optimizer(model);oref=optimizer(ref)
    select=torch.tensor([11,2,6,0])
    remain=torch.tensor([i for i in range(12) if i not in select.tolist()])
    result=core.matched_parent_step_v2(opt,parameters,[lambda:parent_cell(select),lambda:parent_cell(remain)])
    oref.zero_grad(set_to_none=True)
    cells=tile_cells()
    loss=torch.stack([n for n,d in cells]).sum()/sum(d for n,d in cells)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(rp,1.,error_if_nonfinite=True)
    oref.step()
    close('weighted_crossentropy_reset_aggregate_loss_after_step_'+str(dtype),torch.tensor(result['loss'],dtype=dtype),loss.detach(),dtype)
    gradients('weighted_crossentropy_single_clip_gradients_'+str(dtype),[p.grad for p in parameters],[p.grad for p in rp],dtype)
    close('weighted_crossentropy_single_AdamW_parameters_'+str(dtype),
          torch.cat([p.detach().reshape(-1) for p in parameters]),torch.cat([p.detach().reshape(-1) for p in rp]),dtype)
    for pi,(p,r) in enumerate(zip(parameters,rp)):
        check('AdamW_state_keys_'+str(dtype)+'_'+str(pi),opt.state[p].keys()==oref.state[r].keys())
        for key in opt.state[p]:
            close('AdamW_state_'+str(dtype)+'_'+str(pi)+'_'+key,opt.state[p][key],oref.state[r][key],dtype,gradient=True)
    check('one_aggregate_backward_clip_optimizer_step_'+str(dtype),
          result['backward_calls']==result['clip_calls']==result['optimizer_steps']==1 and result['query_bp']==float(bp.sum()))
    details['classification_step_'+str(dtype)]={'loss':result['loss'],'reference_loss':float(loss.detach()),
                    'gradient_norm':result['gradient_norm_before_single_clip'],'reference_gradient_norm':float(norm)}

report={'kind':'anchor_parent_context_v2_author_adapted_reviewer_oracle','review_role':'author_not_independent_reviewer','status':'PASS','count':len(checks),'checks':checks,'details':details,
 'original_oracle_source':'research/reviews/anchor_parent_context_v1_independent/independent_checks.py','original_oracle_sha256':sha(ROOT/'research/reviews/anchor_parent_context_v1_independent/independent_checks.py'),
 'adaptation':'unchanged matrix-exponential impulse/reference math; versioned core/production unique source entry; own output; preserved v1 pins; rejection-specific cases handled separately',
 'source_sha256':sha(ROOT/'src/arggnn/anchor_parent_context_v2.py'),'config_sha256':sha(core.CONFIG),'test_source_sha256':sha(__file__),
 'actual_arrays_simulations_fits_GPU_installations_budgets_or_Git':False}
(DIR/'adapted_oracle_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)},indent=2))
