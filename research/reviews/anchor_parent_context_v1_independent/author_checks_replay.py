"""Core handmade CPU qualification. No actual bank/target arrays or biological fits."""
from copy import deepcopy
from dataclasses import replace
import hashlib,json,os
from pathlib import Path
import subprocess,sys,types
import torch
from arggnn import anchor_kernel as old
from arggnn import anchor_parent_context_v1 as m
torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
ROOT=m.ROOT;DIR=ROOT/'research/reviews/anchor_parent_context_v1_independent'
checks=[];details={};cfg=json.loads(m.CONFIG.read_text())
def check(name,value=True):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,call):
 try:call()
 except (ValueError,RuntimeError,IndexError,TypeError):check(name);return
 raise AssertionError('unexpected acceptance: '+name)
def close(name,a,b,dtype):
 t=cfg['qualification'][str(dtype).split('.')[-1]]
 torch.testing.assert_close(a,b,rtol=t['output_rtol'],atol=t['output_atol']);check(name)
def grads_close(name,a,b,dtype):
 t=cfg['qualification'][str(dtype).split('.')[-1]]
 for x,y in zip(a,b):
  if x is None or y is None:assert x is y
  else:torch.testing.assert_close(x,y,rtol=t['vjp_rtol'],atol=t['vjp_atol'])
 check(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Whole-source import and report stay standard-library only.
r=subprocess.run([sys.executable,'-B','-c',"import sys; from arggnn.anchor_parent_context_v1 import source_report_v1; source_report_v1(); assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
check('default_source_report_no_scientific_import',r.returncode==0)
for p,h in cfg['source_pins'].items():check('original_source_preserved_'+p,sha(ROOT/p)==h)
torch.manual_seed(17)
Q,R,S,H,V=4,3,2,3,2
def operators(dtype):
 records=[]
 for graph in range(3):
  records.append([])
  for scale in range(S):
   edge=torch.tensor([[0.,.13+.03*graph,.06,.02],[.13+.03*graph,0.,.01,.09+.01*graph],[.06,.01,0.,.11],[.02,.09+.01*graph,.11,0.]],dtype=torch.float64)*(scale+1)
   M=torch.diag(edge.sum(1)+torch.tensor([.7,.9,1.1,1.3],dtype=torch.float64)+graph*.07)-edge
   records[-1].append(M)
 cqq=torch.stack([torch.stack(x) for x in records]);values,vectors=torch.linalg.eigh(cqq)
 cqr=-torch.arange(1.,3*S*Q*R+1.,dtype=torch.float64).reshape(3,S,Q,R)/300.
 return old.AnchorOperators(cqq.to(dtype),cqr.to(dtype),values.to(dtype),vectors.to(dtype))
def fixtures(dtype=torch.float64):
 O=operators(dtype);q=torch.tensor([11,3,27,8]);r=torch.tensor([91,54,70]);tiles=[]
 for tile_id,(bounds,ids,graphs,positions) in enumerate([
  ([0.,.05,.125],(10,11),[0,1],[0.,.025,.05,.12]),
  ([.125,.18,.25],(11,12),[1,2],[.125,.13,.2,.249])]):
  intervals=torch.tensor(list(zip(bounds[:-1],bounds[1:])),dtype=torch.float64)
  pos=torch.tensor(positions,dtype=torch.float64);target=torch.searchsorted(intervals[:,0],pos,right=True)-1
  bank=types.SimpleNamespace(intervals=intervals,positions=pos,target_graph=target,window_bounds=torch.tensor([bounds[0],bounds[-1]],dtype=torch.float64))
  L=old._layout(bank,32)
  ops=old.AnchorOperators(*(getattr(O,f)[graphs] for f in ('cqq','cqr','eigenvalues','eigenvectors')))
  tiles.append(m.TileContextV1(L,ops,q,r,ids,feature_fingerprints=tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in ids),parent_id='handmade_train0',split='train',namespace='handmade_same_ARG'))
 return tuple(tiles),m.stitch_context_v1(tiles,fixture=True)
def dense_reset(parent,sources,lam,delta):
 result={}
 for tile,(a,b),ev in zip(parent.tile_contexts,parent.tile_anchor_ranges,parent.tile_evaluation_coordinates):
  values=old.anchor_kernel_action(tile.layout,tile.operators,sources[a:b],lam,delta)
  for x,value in zip(ev.tolist(),values):result[x]=value
 # Target/anchor coordinates are right-owned; each cached tile has no right-end readout.
 ev=parent.layout.coordinates[[i for i,o in enumerate(parent.layout.output_at_event) if o>=0]]
 return torch.stack([result[float(x)] for x in ev])
for dtype in (torch.float64,torch.float32):
 tiles,parent=fixtures(dtype);L=parent.layout
 check('anchor_union_bit_exact_'+str(dtype),torch.equal(L.anchors,torch.cat([t.layout.anchors for t in tiles])) and torch.equal(L.weights,torch.cat([t.layout.weights for t in tiles])))
 check('cut_inside_same_original_tree_retained_'+str(dtype),parent.original_tree_ids_by_cache_graph==(10,11,11,12))
 check('noncommuting_original_holds_'+str(dtype),not torch.allclose(parent.operators.cqq[0]@parent.operators.cqq[1],parent.operators.cqq[1]@parent.operators.cqq[0]))
 sources=torch.randn(len(L.anchors),S,Q,H,2,dtype=dtype,requires_grad=True)
 lam=torch.full((S,H),2.,dtype=dtype,requires_grad=True);delta=torch.full((S,H),.8,dtype=dtype,requires_grad=True)
 coefficient=torch.randn(L.num_evaluations,S,Q,H,2,dtype=dtype)
 for mode in ('carry','reset'):
  reference=old.anchor_kernel_action(L,parent.operators,sources,lam,delta) if mode=='carry' else dense_reset(parent,sources,lam,delta)
  rg=torch.autograd.grad((reference*coefficient).sum(),(sources,lam,delta),retain_graph=True)
  for block,cp in ((1,False),(17,False),(64,True),(4096,True)):
   actual=m.streamed_parent_action_v1(parent,sources,lam,delta,mode=mode,event_block_size=block,checkpoint_blocks=cp,return_boundary_states=True)
   close(f'{dtype}_{mode}_values_block{block}_checkpoint{cp}',actual.values,reference,dtype)
   ag=torch.autograd.grad((actual.values*coefficient).sum(),(sources,lam,delta),retain_graph=True)
   grads_close(f'{dtype}_{mode}_VJP_block{block}_checkpoint{cp}',ag,rg,dtype)
   check(f'{dtype}_{mode}_boundary_states_stay_differentiable_{block}',all(x.grad_fn is not None for x in actual.forward_boundary_states+actual.reverse_boundary_states))
 ids=torch.tensor([L.num_evaluations-1,0,5])
 selective=m.streamed_parent_action_v1(parent,sources,lam,delta,evaluation_ids=ids,event_block_size=17,checkpoint_blocks=True).values
 close('selective_readout_order_'+str(dtype),selective,old.anchor_kernel_action(L,parent.operators,sources,lam,delta)[ids],dtype)
 if dtype==torch.float32:
  _,double_parent=fixtures(torch.float64)
  transferred=m.streamed_parent_action_v1(double_parent,sources,lam,delta,event_block_size=17,checkpoint_blocks=True).values
  expected=old.anchor_kernel_action(L,parent.operators,sources,lam,delta)
  close('fixed_CPU_float64_cache_block_cast_matches_original_float32',transferred,expected,dtype)
  tg=torch.autograd.grad((transferred*coefficient).sum(),(sources,lam,delta),retain_graph=True)
  eg=torch.autograd.grad((expected*coefficient).sum(),(sources,lam,delta),retain_graph=True)
  grads_close('fixed_CPU_float64_block_cast_VJP',tg,eg,dtype)
 # Both directional influence crosses the cut; detached/reset fallback is visible.
 for side,event_id,other_anchor in [('late_loss_early_source',int(L.target_output[-1]),0),('early_loss_late_source',int(L.target_output[0]),len(L.anchors)-1)]:
  for mode in ('carry','reset'):
   output=m.streamed_parent_action_v1(parent,sources,lam,delta,mode=mode,evaluation_ids=torch.tensor([event_id]),checkpoint_blocks=True,event_block_size=17).values
   g=torch.autograd.grad(output.sum(),sources,retain_graph=True)[0][other_anchor]
   check(str(dtype)+'_'+mode+'_'+side,bool((g.abs()>1e-9).any()) if mode=='carry' else torch.count_nonzero(g)==0)
 # Fresh call state is zero at exterior endpoints, no prior call/parent mutation.
 one=m.streamed_parent_action_v1(parent,sources,lam,delta).values
 zero=m.streamed_parent_action_v1(parent,torch.zeros_like(sources),lam,delta).values
 check('fresh_zero_exterior_each_call_'+str(dtype),torch.count_nonzero(zero)==0)
 close('fresh_call_replay_'+str(dtype),one,m.streamed_parent_action_v1(parent,sources,lam,delta).values,dtype)
 if dtype==torch.float64:
  directions=[torch.randn_like(x)/100 for x in (sources,lam,delta)];epsilon=cfg['qualification']['directional_epsilon']
  output=m.streamed_parent_action_v1(parent,sources,lam,delta,event_block_size=17,checkpoint_blocks=True).values
  grads=torch.autograd.grad((output*coefficient).sum(),(sources,lam,delta))
  analytic=sum((g*d).sum() for g,d in zip(grads,directions))
  values=[]
  for sign in (-1,1):
   args=[x+sign*epsilon*d for x,d in zip((sources,lam,delta),directions)]
   values.append((m.streamed_parent_action_v1(parent,*args).values*coefficient).sum())
  finite=(values[1]-values[0])/(2*epsilon)
  torch.testing.assert_close(analytic,finite,rtol=1e-8,atol=1e-10);check('float64_action_directional_derivative')
  details['action_directional']={'analytic':float(analytic),'central':float(finite)}
 # Full nonlinear layer sources include lambda-dependent normalization mass.
 layer=old.AnchorKernelLayer(6,H,S,(2.,3.),(1.,1.)).to(dtype)
 q=torch.randn(len(L.anchors),Q,6,dtype=dtype,requires_grad=True);r=torch.randn(len(L.anchors),R,6,dtype=dtype,requires_grad=True)
 prepared=types.SimpleNamespace(layout=L,operators={'age':parent.operators})
 expected=layer.messages(prepared,q,r,'age','full')
 actual=m.parent_layer_messages_v1(parent,q,r,layer,checkpoint_blocks=True,event_block_size=17)
 for field in ('anchors','targets','numerator','mass','sources','mass_sources'):close('layer_'+field+'_'+str(dtype),getattr(actual,field),getattr(expected,field),dtype)
 expected_mass=1.-parent.operators.cqr[L.anchor_graph].sum(-1)[...,None]*layer.rates[None,:,None,:]
 check('mass_parameter_dependent_exact_formula_'+str(dtype),torch.equal(actual.mass_sources,expected_mass))
 parameters=tuple(layer.parameters());coeff=torch.randn_like(expected.targets)
 eg=torch.autograd.grad((expected.targets*coeff).sum(),(q,r,*parameters),retain_graph=True,allow_unused=True)
 ag=torch.autograd.grad((actual.targets*coeff).sum(),(q,r,*parameters),retain_graph=True,allow_unused=True)
 grads_close('layer_source_and_all_parameter_VJP_'+str(dtype),ag,eg,dtype)
 mass_grad=torch.autograd.grad(actual.mass_sources.sum(),layer.raw_coupling,retain_graph=True)[0]
 check('mass_keeps_rate_gradient_'+str(dtype),bool((mass_grad.abs()>0).all()))
 # Two-layer parent adapter uses global anchor contexts and no target sources.
 model=old.AnchorKernelModel(input_dim=47,hidden_dim=6,latent_dim=H,num_layers=2,kernel_depth=2,num_scales=S,num_anchors=32,geometry='age',initial_coupling=(2.,3.)).to(dtype)
 full_model=deepcopy(model);full_model.num_anchors=64 # explicit small dense reference only
 G=len(parent.operators.eigenvalues);base=torch.randn(3,Q+R,47,dtype=dtype,requires_grad=True);leaf=torch.stack((base[0],base[1],base[1],base[2]))
 response=torch.eye(Q+R,dtype=dtype).expand(G,Q+R,Q+R)
 cache=types.SimpleNamespace(normalized_response=response,query_indices=torch.arange(Q),reference_indices=torch.arange(Q,Q+R))
 bank=types.SimpleNamespace(leaf_x=leaf,query_cache=cache,target_graph=parent.target_graph)
 dense=types.SimpleNamespace(bank=bank,layout=L,operators={'age':parent.operators},num_scales=S)
 reject('original_model_global_anchor_contract_still_rejects_'+str(dtype),lambda:model.encode_context(dense))
 enc=model.encoder(leaf,response)
 ctx=m.parent_context_v1(parent,model,enc[:,:Q],enc[:,Q:],checkpoint_blocks=True,event_block_size=17)
 reference=full_model.encode_context(dense)
 close('two_layer_global_anchor_context_'+str(dtype),ctx.anchors,reference.anchors,dtype)
 close('two_layer_global_target_context_'+str(dtype),ctx.targets,reference.targets,dtype)
 subset=torch.tensor([7,0,4])
 ctx2=m.parent_context_v1(parent,model,enc[:,:Q],enc[:,Q:],target_indexes=subset,checkpoint_blocks=True,event_block_size=17)
 close('target_chunk_updates_every_layer_'+str(dtype),ctx2.targets,ctx.targets[subset],dtype)
 params=tuple(model.parameters());params_ref=tuple(full_model.parameters())
 coeff=torch.randn_like(ctx.targets)
 actual_logits=model.head(ctx.targets);reference_logits=full_model.head(reference.targets)
 close('two_layer_head_logits_'+str(dtype),actual_logits,reference_logits,dtype)
 c=torch.randn_like(actual_logits)
 ag=torch.autograd.grad((actual_logits*c).sum(),(leaf,*params),allow_unused=True,retain_graph=True)
 eg=torch.autograd.grad((reference_logits*c).sum(),(leaf,*params_ref),allow_unused=True,retain_graph=True)
 grads_close('two_layer_encoder_input_and_all_parameter_VJP_'+str(dtype),ag,eg,dtype)

 for mode in ('carry','reset'):
  e=model.encoder(leaf,response)
  context=m.parent_context_v1(parent,model,e[:,:Q],e[:,Q:],mode=mode,checkpoint_blocks=True,event_block_size=17)
  for side,target_index,other_graph in [('late_loss_early_source',-1,0),('early_loss_late_source',0,2)]:
   scalar=model.head(context.targets[target_index]).square().sum()
   influence=torch.autograd.grad(scalar,base,retain_graph=True)[0][other_graph]
   check('two_layer_'+str(dtype)+'_'+mode+'_'+side,bool((influence.abs()>1e-10).any()) if mode=='carry' else torch.count_nonzero(influence)==0)
 if dtype==torch.float64:
  direction=[torch.randn_like(p)/100 for p in params]
  analytic=sum((g*d).sum() for g,d in zip(ag[1:],direction) if g is not None)
  state=[p.detach().clone() for p in params];values=[]
  for sign in (-1,1):
   with torch.no_grad():
    for p,v,d in zip(params,state,direction):p.copy_(v+sign*1e-5*d)
   encoded=model.encoder(leaf,response);trial=m.parent_context_v1(parent,model,encoded[:,:Q],encoded[:,Q:],event_block_size=17)
   values.append((model.head(trial.targets)*c).sum().detach())
  with torch.no_grad():
   for p,v in zip(params,state):p.copy_(v)
  finite=(values[1]-values[0])/2e-5
  torch.testing.assert_close(analytic,finite,rtol=1e-8,atol=1e-10);check('two_layer_encoder_rates_mass_gate_head_directional')
  details['model_directional']={'analytic':float(analytic),'central':float(finite)}

# Reset parent-step parity: all original tile losses evaluated at fixed parameters.
for dtype in (torch.float64,torch.float32):
 tiles,parent=fixtures(dtype);G=len(parent.operators.eigenvalues)
 base=torch.randn(3,Q+R,47,dtype=dtype);leaf=torch.stack((base[0],base[1],base[1],base[2]))
 response=torch.eye(Q+R,dtype=dtype).expand(G,Q+R,Q+R)
 model=old.AnchorKernelModel(input_dim=47,hidden_dim=6,latent_dim=H,num_layers=2,kernel_depth=2,num_scales=S,num_anchors=32,geometry='age',initial_coupling=(2.,3.)).to(dtype)
 reference=deepcopy(model);parameters=tuple(model.parameters());params_ref=tuple(reference.parameters())
 weights=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.],dtype=dtype)
 def whole_loss(which):
  encoded=which.encoder(leaf,response)
  context=m.parent_context_v1(parent,which,encoded[:,:Q],encoded[:,Q:],mode='reset',event_block_size=17,checkpoint_blocks=True)
  logits=which.head(context.targets)
  return (logits.square().mean((1,2))*weights).sum()/float(weights.sum())
 reference_cells=[]
 for tile_i,(tile,(g0,g1)) in enumerate(zip(tiles,parent.tile_graph_ranges)):
  def cell(which,tile=tile,g0=g0,g1=g1,tile_i=tile_i):
   bank=types.SimpleNamespace(leaf_x=leaf[g0:g1],query_cache=types.SimpleNamespace(normalized_response=response[g0:g1],query_indices=torch.arange(Q),reference_indices=torch.arange(Q,Q+R)),target_graph=tile.layout.segment_graph[torch.searchsorted(tile.layout.coordinates,parent.tile_evaluation_coordinates[tile_i][tile.layout.target_output],right=True)-1])
   prepared=types.SimpleNamespace(bank=bank,layout=tile.layout,operators={'age':tile.operators},num_scales=S)
   logits=which(prepared);w=weights[tile_i*4:(tile_i+1)*4]
   return (logits.square().mean((1,2))*w).sum(),float(w.sum())
  reference_cells.append(cell)
 reset_loss=whole_loss(model)
 original_sum=torch.stack([f(reference)[0] for f in reference_cells]).sum()/float(weights.sum())
 close('reset_parent_loss_equals_original_fixed_parameter_tile_sum_'+str(dtype),reset_loss,original_sum,dtype)
 grad=torch.autograd.grad(reset_loss,parameters,allow_unused=True)
 refgrad=torch.autograd.grad(original_sum,params_ref,allow_unused=True)
 grads_close('reset_parent_VJP_equals_original_aggregate_'+str(dtype),grad,refgrad,dtype)
 # Runner supports recomputed chunk/cell objectives with global graph intact.
 def parent_cell(which,selected):
  encoded=which.encoder(leaf,response)
  context=m.parent_context_v1(parent,which,encoded[:,:Q],encoded[:,Q:],mode='reset',target_indexes=selected,event_block_size=17,checkpoint_blocks=True)
  logits=which.head(context.targets);w=weights[selected]
  return (logits.square().mean((1,2))*w).sum(),float(w.sum())
 opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=.0001)
 optref=torch.optim.AdamW(params_ref,lr=.001,weight_decay=.0001)
 before=[p.detach().clone() for p in parameters]
 result=m.matched_parent_step_v1(opt,parameters,[lambda:parent_cell(model,torch.arange(4)),lambda:parent_cell(model,torch.arange(4,8))])
 optref.zero_grad(set_to_none=True)
 loss=torch.stack([f(reference)[0] for f in reference_cells]).sum()/float(weights.sum())
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(params_ref,1.,error_if_nonfinite=True);optref.step()
 grads_close('reset_single_clip_parameter_gradient_parity_'+str(dtype),[p.grad for p in parameters],[p.grad for p in params_ref],dtype)
 close('reset_single_AdamW_update_parameter_parity_'+str(dtype),torch.cat([p.flatten() for p in parameters]),torch.cat([p.flatten() for p in params_ref]),dtype)
 check('parent_runner_single_backward_clip_update_'+str(dtype),result['backward_calls']==result['clip_calls']==result['optimizer_steps']==1 and result['query_bp']==36. and result['loss_cells']==2)
 check('parent_runner_actual_handmade_weight_change_'+str(dtype),any(not torch.equal(a,p) for a,p in zip(before,parameters)))
 # Check the AdamW internal step counter, avoiding a fake bookkeeping claim.
 check('all_active_AdamW_states_have_one_step_'+str(dtype),all(float(x['step'])==1. for x in opt.state.values()))
 details['reset_optimizer_'+str(dtype)]={'loss':result['loss'],'reference_loss':float(loss.detach()),'gradient_norm':result['gradient_norm_before_single_clip'],'reference_gradient_norm':float(norm),'single_clip_update':True}
 bad_tiles=list(tiles);finger=list(bad_tiles[1].feature_fingerprints);finger[0]='0'*64;bad_tiles[1]=replace(bad_tiles[1],feature_fingerprints=tuple(finger))
 reject('duplicate_clipped_tree_feature_incompatibility_'+str(dtype),lambda:m.stitch_context_v1(bad_tiles,fixture=True))
 def mutation():
  with torch.no_grad():parameters[0].add_(.0001)
  return parameters[0].square().sum(),1.
 reject('matched_parent_runner_rejects_per_cell_parameter_update_'+str(dtype),lambda:m.matched_parent_step_v1(opt,parameters,[mutation]))
 reject('matched_parent_runner_rejects_per_tile_clipping_policy_'+str(dtype),lambda:m.matched_parent_step_v1(opt,parameters,[lambda:parent_cell(model,torch.arange(4))],gradient_clip_norm=.5))


# Persistent numeric IDs are joins, not learned identities: restore permutations.
for dtype in (torch.float64,torch.float32):
 tiles,parent=fixtures(dtype);pq=torch.tensor([2,0,3,1]);pr=torch.tensor([2,0,1]);permuted=[]
 for t in tiles:
  O=t.operators
  changed=old.AnchorOperators(O.cqq[:,:,pq][:,:,:,pq],O.cqr[:,:,pq][:,:,:,pr],O.eigenvalues,O.eigenvectors[:,:,pq,:])
  permuted.append(replace(t,operators=changed,query_ids=t.query_ids[pq],reference_ids=t.reference_ids[pr]))
 other=m.stitch_context_v1(permuted,fixture=True)
 sources=torch.randn(len(parent.layout.anchors),S,Q,H,2,dtype=dtype,requires_grad=True)
 lam=torch.full((S,H),2.,dtype=dtype,requires_grad=True);delta=torch.full((S,H),.8,dtype=dtype,requires_grad=True)
 original=m.streamed_parent_action_v1(parent,sources,lam,delta,checkpoint_blocks=True,event_block_size=17).values
 changed=m.streamed_parent_action_v1(other,sources[:,:,pq],lam,delta,checkpoint_blocks=True,event_block_size=17).values[:,:,torch.argsort(pq)]
 close('restored_sample_order_permutation_'+str(dtype),changed,original,dtype)
 coefficient=torch.randn_like(original)
 a=torch.autograd.grad((original*coefficient).sum(),(sources,lam,delta),retain_graph=True)
 b=torch.autograd.grad((changed*coefficient).sum(),(sources,lam,delta),retain_graph=True)
 grads_close('restored_sample_order_VJP_'+str(dtype),a,b,dtype)

# Admission-relevant metadata counterexamples use only handmade cached layouts.
tiles,parent=fixtures()
for fault in ('sample_order','overlap','missing_source','duplicate_evaluation','wrong_anchor_graph','tree_incompatibility','geometry','tree_chronology'):
 bad=list(tiles)
 if fault=='sample_order':bad[1]=replace(bad[1],query_ids=bad[1].query_ids.flip(0))
 elif fault=='overlap':
  L=replace(bad[1].layout,coordinates=bad[1].layout.coordinates+.001);bad[1]=replace(bad[1],layout=L)
 elif fault=='missing_source':
  sources=list(bad[0].layout.source_at_event);sources[sources.index(0)]=-1;bad[0]=replace(bad[0],layout=replace(bad[0].layout,source_at_event=tuple(sources)))
 elif fault=='duplicate_evaluation':
  outputs=list(bad[0].layout.output_at_event);idx=[i for i,v in enumerate(outputs) if v>=0];outputs[idx[1]]=outputs[idx[0]]
  bad[0]=replace(bad[0],layout=replace(bad[0].layout,output_at_event=tuple(outputs)))
 elif fault=='wrong_anchor_graph':bad[0]=replace(bad[0],layout=replace(bad[0].layout,anchor_graph=bad[0].layout.anchor_graph.flip(0)))
 elif fault=='tree_incompatibility':
  O=bad[1].operators;changed=replace(O,cqr=O.cqr.clone());changed.cqr[0]*=2.;bad[1]=replace(bad[1],operators=changed)
 elif fault=='geometry':bad[1]=replace(bad[1],geometry='plain')
 elif fault=='tree_chronology':bad[1]=replace(bad[1],original_tree_ids=(12,11))
 reject('metadata_reject_'+fault,lambda bad=bad:m.stitch_context_v1(bad,fixture=True))
reject('handmade_dimensions_cannot_impersonate_production',lambda:m.stitch_context_v1(tiles))
bad=list(tiles);O=bad[0].operators;learned=replace(O,cqr=O.cqr.clone().requires_grad_());bad[0]=replace(bad[0],operators=learned)
reject('fixed_operator_cannot_become_parameter',lambda:m.stitch_context_v1(bad,fixture=True))
bad=list(tiles);bad[0]=replace(bad[0],feature_fingerprints=('not_a_hash','not_a_hash'))
reject('feature_fingerprint_exact_schema',lambda:m.stitch_context_v1(bad,fixture=True))


for fault in ('parent','split','native_namespace'):
 bad=list(tiles)
 if fault=='parent':bad[1]=replace(bad[1],parent_id='handmade_train1')
 elif fault=='split':bad[1]=replace(bad[1],split='val')
 else:bad[1]=replace(bad[1],namespace='handmade_other_ARG')
 reject('cross_'+fault+'_carry_impossible',lambda bad=bad:m.stitch_context_v1(bad,fixture=True))

check('all_original_sources_preserved_after',all(sha(ROOT/p)==h for p,h in cfg['source_pins'].items()))
report={'kind':'anchor_parent_context_v1_core_handmade_checks','status':'passed','count':len(checks),'checks':checks,'details':details,
 'source_sha256':sha(ROOT/'src/arggnn/anchor_parent_context_v1.py'),'config_sha256':sha(m.CONFIG),'test_source_sha256':sha(__file__),
 'actual_arrays_simulations_fits_GPU_installations_budgets_or_Git':False,'original_sources_preserved':True}
(DIR/'core_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','checks':len(checks),'source_sha256':report['source_sha256']},indent=2))
