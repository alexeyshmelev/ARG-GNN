"""Core handmade CPU qualification. No actual bank/target arrays or biological fits."""
from copy import deepcopy
from dataclasses import replace
import hashlib,json,os
from pathlib import Path
import subprocess,sys,types
import torch
from arggnn import anchor_kernel as old
from arggnn import anchor_parent_context_v2 as m
torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
ROOT=m.ROOT;DIR=ROOT/'research/reviews/anchor_parent_context_v2_author'
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
r=subprocess.run([sys.executable,'-B','-c',"import sys; from arggnn.anchor_parent_context_v2 import source_report_v2; source_report_v2(); assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
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
  tiles.append(m.TileContextV2(L,ops,q,r,ids,feature_fingerprints=tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in ids),parent_id='handmade_train0',split='train',namespace='handmade_same_ARG'))
 return tuple(tiles),m.stitch_context_v2(tiles,fixture=True)
def dense_reset(parent,sources,lam,delta):
 result={}
 for tile,(a,b),ev in zip(parent.tile_contexts,parent.tile_anchor_ranges,parent.tile_evaluation_coordinates):
  values=old.anchor_kernel_action(tile.layout,tile.operators,sources[a:b],lam,delta)
  for x,value in zip(ev.tolist(),values):result[x]=value
 # Target/anchor coordinates are right-owned; each cached tile has no right-end readout.
 ev=parent.layout.coordinates[[i for i,o in enumerate(parent.layout.output_at_event) if o>=0]]
 return torch.stack([result[float(x)] for x in ev])

import builtins,contextlib,stat,traceback
from unittest.mock import patch
GUARD=[]
def audit(event,args):
 if event=='open' and not isinstance(args[0],int):
  try:p=Path(os.path.abspath(os.fsdecode(args[0])))
  except (TypeError,ValueError):return
  mode,flags=args[1:3];writing=isinstance(mode,str) and any(x in mode for x in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
  owned=p==DIR or DIR in p.parents
  if writing and not owned:GUARD.append('outside ownership write '+str(p));raise RuntimeError(GUARD[-1])
  if not writing and not owned:
   for base in (ROOT/'data',ROOT/'experiments/results',ROOT/'.runtime/gpup',ROOT/'.runtime/locks',ROOT/'.runtime/storage',Path('/proc')):
    if p==base or base in p.parents:GUARD.append('actual state read '+str(p));raise RuntimeError(GUARD[-1])
sys.addaudithook(audit)
baseline=json.loads((DIR/'baseline_pins.json').read_text())
for p,h in baseline.items():check('predecessor_pin_before_'+p,sha(ROOT/p)==h)

for dtype in (torch.float64,torch.float32):
 tiles,parent=fixtures(dtype);N=Q+R;G=len(parent.operators.eigenvalues)
 model=old.AnchorKernelModel(input_dim=47,hidden_dim=6,latent_dim=H,num_layers=2,kernel_depth=2,num_scales=S,num_anchors=32,geometry='age',initial_coupling=(2.,3.)).to(dtype)
 base=torch.randn(3,N,47,dtype=dtype);leaf=base[torch.tensor([0,1,1,2])]
 response=torch.eye(N,dtype=dtype).expand(G,N,N)
 ids=torch.cat((parent.query_ids,parent.reference_ids));calls=[]
 handle=model.encoder.register_forward_pre_hook(lambda module,inputs:calls.append(inputs[0].shape[0]))
 encoded=m.encode_parent_sources_v2(parent,model,leaf,response,leaf_ids=ids);handle.remove()
 check('F1_unique_encoder_batch_once_'+str(dtype),calls==[3])
 check('F1_all_cache_occurrences_retained_'+str(dtype),encoded.unique_original_tree_ids==(10,11,12) and encoded.first_cache_graph_indexes==(0,1,3) and encoded.inverse_cache_graph_indexes==(0,1,1,2) and encoded.query.shape[0]==4 and parent.original_tree_ids_by_cache_graph==(10,11,11,12))
 check('F1_exact_initial_reused_query_reference_'+str(dtype),torch.equal(encoded.query[1],encoded.query[2]) and torch.equal(encoded.reference[1],encoded.reference[2]))
 actual=m.parent_context_v2(parent,model,leaf,response,leaf_ids=ids,checkpoint_blocks=True,event_block_size=17)
 diagnostic=m.parent_encoded_context_v2(parent,model,encoded.query,encoded.reference,fixture=True,checkpoint_blocks=True,event_block_size=17)
 close('F1_production_equals_valid_encoded_diagnostic_'+str(dtype),actual.targets,diagnostic.targets,dtype)
 check('F1_evolved_anchor_states_legitimately_diverge_'+str(dtype),not torch.allclose(actual.anchors[parent.layout.anchor_graph==1][0],actual.anchors[parent.layout.anchor_graph==2][-1]))
 for which,step in [('query',2.),('reference',-2.),('query',1e-13 if dtype==torch.float64 else 1e-5)]:
  wrongq=encoded.query.clone();wrongr=encoded.reference.clone()
  (wrongq if which=='query' else wrongr)[2,0,0]+=step
  reject('F1_reject_initial_'+which+'_delta'+str(step)+'_'+str(dtype),lambda:m.parent_encoded_context_v2(parent,model,wrongq,wrongr,fixture=True))
 reject('F1_arbitrary_encoded_entry_not_production_'+str(dtype),lambda:m.parent_encoded_context_v2(parent,model,encoded.query,encoded.reference))
 changed=leaf.clone();changed[2,0,0]+=1.
 reject('F1_reject_duplicate_original_feature_rows_'+str(dtype),lambda:m.parent_context_v2(parent,model,changed,response,leaf_ids=ids))
 changed_response=response.clone();changed_response[2,0,0]+=.1
 reject('F1_reject_duplicate_original_response_rows_'+str(dtype),lambda:m.parent_context_v2(parent,model,leaf,changed_response,leaf_ids=ids))
 reject('F1_explicit_leaf_ID_wrong_join_'+str(dtype),lambda:m.parent_context_v2(parent,model,leaf,response,leaf_ids=ids+1))
 reject('F1_explicit_leaf_ID_duplicate_'+str(dtype),lambda:m.parent_context_v2(parent,model,leaf,response,leaf_ids=ids.clone().fill_(1)))
 reject('F1_production_fixed_inputs_'+str(dtype),lambda:m.parent_context_v2(parent,model,leaf.clone().requires_grad_(),response,leaf_ids=ids))
 reject('F1_typed_fixture_selector_'+str(dtype),lambda:m.parent_context_v2(parent,model,leaf,response,leaf_ids=ids,fixture=1))
 permutation=torch.tensor([6,0,4,2,5,1,3])
 permuted=m.parent_context_v2(parent,model,leaf[:,permutation],response[:,permutation][:,:,permutation],leaf_ids=ids[permutation])
 close('F1_explicit_permuted_leaf_join_context_'+str(dtype),permuted.targets,actual.targets,dtype)
 coefficients=torch.randn_like(actual.targets)
 ga=torch.autograd.grad((actual.targets*coefficients).sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
 gb=torch.autograd.grad((permuted.targets*coefficients).sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
 grads_close('F1_explicit_permuted_leaf_join_parameter_VJP_'+str(dtype),ga,gb,dtype)
 check('F1_initial_encoder_not_detached_'+str(dtype),encoded.query.grad_fn is not None and encoded.reference.grad_fn is not None)
 details['F1_'+str(dtype)]={'unique_encoder_calls':calls,'cache_graphs':G,'unique_original_trees':3,'initial_Q_R_exact_reuse':True,'evolved_anchor_rows_differ':True,'production_raw_fixed_feature_join':True}
 # Guard exactly the original computed decay domain across modes/checkpoints.
 small=[]
 for tile in tiles:
  cqq=torch.diag(torch.tensor([-1e-15,1.,1.,1.],dtype=dtype)).expand(2,S,Q,Q).clone()
  eig=torch.tensor([-1e-15,1.,1.,1.],dtype=dtype).expand(2,S,Q).clone()
  vectors=torch.eye(Q,dtype=dtype).expand(2,S,Q,Q).clone()
  small.append(replace(tile,operators=old.AnchorOperators(cqq,tile.operators.cqr,eig,vectors)))
 negative=m.stitch_context_v2(small,fixture=True)
 sources=torch.ones(len(parent.layout.anchors),S,Q,H,2,dtype=dtype,requires_grad=True)
 delta=torch.ones((S,H),dtype=dtype)
 huge=torch.full((S,H),1e308 if dtype==torch.float64 else torch.finfo(dtype).max*.9,dtype=dtype)
 rates=torch.full((S,H),1e16,dtype=dtype)
 zero_rates=torch.full((S,H),1e15,dtype=dtype)
 for label,P,lam in [('overflow',parent,huge),('negative',negative,rates),('zero',negative,zero_rates)]:
  reject('F2_original_baseline_reject_'+label+'_'+str(dtype),lambda P=P,lam=lam:old.anchor_kernel_action(P.layout,P.operators,sources,lam,delta))
  for mode in ('carry','reset'):
   for cp in (False,True):
    reject('F2_stream_reject_'+label+'_'+str(dtype)+'_'+mode+'_cp'+str(cp),lambda P=P,lam=lam,mode=mode,cp=cp:m.streamed_parent_action_v2(P,sources,lam,delta,mode=mode,checkpoint_blocks=cp,event_block_size=17))
 positive=torch.full((S,H),1e100 if dtype==torch.float64 else 1e20,dtype=dtype)
 close('F2_finite_large_positive_underflow_matches_baseline_'+str(dtype),m.streamed_parent_action_v2(parent,sources,positive,delta).values,old.anchor_kernel_action(parent.layout,parent.operators,sources,positive,delta),dtype)

def step_reject(name,make):
 p,optimizer,parameters,cells,kwargs=make();before=[q.detach().clone() for q in parameters]
 try:m.matched_parent_step_v2(optimizer,parameters,cells,**kwargs)
 except (ValueError,RuntimeError,TypeError) as error:
  check(name)
  check(name+'_no_optimizer_state_or_parameter_change',not optimizer.state and all(torch.equal(a,q) for a,q in zip(before,parameters)))
  details[name]={'error':str(error),'optimizer_state_entries':0,'no_update':True};return
 raise AssertionError('unexpected step acceptance: '+name)
def one(kind):
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.01,weight_decay=0.)
 leaf=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
 if kind=='disconnected':cells=[lambda:(leaf.square(),3.)]
 elif kind=='foreign_optimizer_leaf':other=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64));cells=[lambda:(other.square(),3.)]
 elif kind=='mixed_disconnected_cell':cells=[lambda:(p.square(),1.),lambda:(leaf.square(),1.)]
 elif kind=='fabricated_grad':
  def fake():p.grad=torch.ones_like(p);return leaf.square(),1.
  cells=[fake]
 else:raise AssertionError(kind)
 return p,opt,[p],cells,{}
for kind in ('disconnected','foreign_optimizer_leaf','mixed_disconnected_cell','fabricated_grad'):step_reject('F3_'+kind,lambda k=kind:one(k))
def omitted():
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));q=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64))
 return p,torch.optim.AdamW([p,q],lr=.01),[p,q],[lambda:(p.square(),1.)],{}
step_reject('F3_undeclared_inactive_parameter',omitted)
def false_inactive():
 p,opt,params,cells,kw=omitted();return p,opt,params,[lambda:(sum(q.square() for q in params),1.)],{'inactive_parameters':[params[1]]}
step_reject('F3_falsely_declared_inactive_connected_parameter',false_inactive)
for frozen in (False,True):
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));q=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64),requires_grad=not frozen)
 opt=torch.optim.AdamW([p,q],lr=.01,weight_decay=0.)
 result=m.matched_parent_step_v2(opt,[p,q],[lambda:(p.square(),1.)],inactive_parameters=[q])
 check('F3_declared_individual_inactive_frozen'+str(frozen),p.grad is not None and q.grad is None and q not in opt.state and p in opt.state and result['connected_optimizer_parameters']==1 and result['declared_inactive_parameters']==1)
p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.01,weight_decay=0.)
result=m.matched_parent_step_v2(opt,[p],[lambda:((p*0).sum(),1.)])
check('F3_connected_legitimate_zero_gradient',p.grad is not None and p.grad==0 and result['gradient_norm_before_single_clip']==0. and p in opt.state and opt.state[p]['step']==1)
class NoneGradient(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):return x*2
 @staticmethod
 def backward(ctx,gradient):return None
def no_actual_grad():
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64))
 return p,torch.optim.AdamW([p],lr=.01),[p],[lambda:(NoneGradient.apply(p),1.)],{}
step_reject('F3_graph_path_without_actual_gradient',no_actual_grad)
def infinite_grad():
 p=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
 return p,torch.optim.AdamW([p],lr=.01),[p],[lambda:(p.sqrt(),1.)],{}
step_reject('F3_finite_loss_nonfinite_actual_gradient',infinite_grad)
def changed_binding():
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));q=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.01)
 def cell():opt.param_groups[0]['params']=[q];return p.square(),1.
 return p,opt,[p],[cell],{}
step_reject('F3_optimizer_binding_mutated_inside_cell',changed_binding)
def backward_changed_binding():
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));q=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.01)
 def hook(gradient):opt.param_groups[0]['params']=[q];return gradient
 p.register_hook(hook)
 return p,opt,[p],[lambda:(p.square(),1.)],{}
step_reject('F3_optimizer_binding_mutated_during_backward',backward_changed_binding)
def missing_binding():
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));q=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64));return p,torch.optim.AdamW([p],lr=.01),[q],[lambda:(q.square(),1.)],{}
step_reject('F3_wrong_parameter_optimizer_binding',missing_binding)
def frozen_active():
 p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64),requires_grad=False);other=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
 return p,torch.optim.AdamW([p],lr=.01),[p],[lambda:(other.square(),1.)],{}
step_reject('F3_frozen_intended_active_parameter',frozen_active)
def invalid_inactive(kind):
 p,opt,params,cells,kw=one('disconnected')
 return p,opt,params,cells,{'inactive_parameters':[p,p] if kind=='duplicate' else [torch.nn.Parameter(torch.tensor(1.))]}
for kind in ('duplicate','foreign'):step_reject('F3_inactive_subset_'+kind,lambda k=kind:invalid_inactive(k))
for p,h in baseline.items():check('predecessor_pin_after_'+p,sha(ROOT/p)==h)
check('unchanged_qualification_tolerances',cfg['qualification']==json.loads((ROOT/'configs/anchor_parent_context_v1.json').read_text())['qualification'])
check('zero_forbidden_actual_access_events',not GUARD)
report={'kind':'anchor_parent_context_v2_author_correction_checks','status':'PASS','count':len(checks),'checks':checks,'details':details,'guard_events':GUARD,
 'source_sha256':sha(ROOT/'src/arggnn/anchor_parent_context_v2.py'),'config_sha256':sha(m.CONFIG),'test_source_sha256':sha(__file__),
 'actual_arrays_simulations_fits_GPU_installations_budgets_or_Git':False,'tolerances_unchanged':True,'original_sources_seals_and_REJECT_preserved':True}
(DIR/'correction_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'guard_events':GUARD},indent=2))
