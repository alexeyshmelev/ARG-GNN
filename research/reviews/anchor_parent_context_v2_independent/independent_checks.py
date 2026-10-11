"""Fresh independent source and handmade CPU checks of H024v2 corrections."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import ast,hashlib,json,os,types
import torch
from arggnn import anchor_kernel as legacy
from arggnn import anchor_parent_context_v2 as m
torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
ROOT=m.ROOT;DIR=ROOT/'research/reviews/anchor_parent_context_v2_independent'
assert Path(os.environ['TMPDIR']).is_relative_to(DIR)
assert Path(os.environ['XDG_CACHE_HOME']).is_relative_to(DIR)
checks=[];details={};cfg=json.loads(m.CONFIG.read_text())
def check(name,condition=True):
 if not bool(condition):raise AssertionError(name)
 checks.append(name)
def reject(name,call):
 try:call()
 except (ValueError,RuntimeError,TypeError,IndexError,KeyError):check(name);return
 raise AssertionError('unexpected acceptance: '+name)
def close(name,a,b,dtype,grad=False):
 c=cfg['qualification'][str(dtype).split('.')[-1]]
 torch.testing.assert_close(a,b,atol=c['vjp_atol' if grad else 'output_atol'],rtol=c['vjp_rtol' if grad else 'output_rtol']);check(name)
def grads(name,a,b,dtype):
 for x,y in zip(a,b):
  if x is None or y is None:assert x is y
  else:
   c=cfg['qualification'][str(dtype).split('.')[-1]];torch.testing.assert_close(x,y,atol=c['vjp_atol'],rtol=c['vjp_rtol'])
 check(name)
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
pins={
 'src/arggnn/anchor_parent_context_v2.py':'e93fe2a6cc064ccfc69379ee89de8512ebec8d6dc59231027c734474b8a551ad',
 'configs/anchor_parent_context_v2.json':'62a1b8a87f47eed0db5ce7dd2013cbf7d45a42090198de0be9f92be69709ffdc',
 'research/reviews/anchor_parent_context_v2_author/author_receipt.json':'b597514dc6fd826474f3d5508f34ec5e28be86bb78e996a7a183a95d789153a8',
 'research/reviews/anchor_parent_context_v1_independent/receipt.json':'2f77aba99fb2c9c82b7ec7d05528eb09bc380bf70675032abab55d9db97d6f2c',
 **cfg['source_pins']}
for p,h in pins.items():check('exact_assigned_pin_'+p,sha(p)==h)
check('unrelaxed_predecessor_tolerances',cfg['qualification']==json.loads((ROOT/'configs/anchor_parent_context_v1.json').read_text())['qualification'])
for file,count in [('core_checks.json',152),('adapted_oracle_checks.json',427),('correction_checks.json',120)]:
 check('author_replay_'+file,json.loads((DIR/file).read_text())['count']==count)
# Verify unchanged stitch/message scientific AST after only V1->V2 naming.
old=ast.parse((ROOT/'src/arggnn/anchor_parent_context_v1.py').read_text())
new=ast.parse((ROOT/'src/arggnn/anchor_parent_context_v2.py').read_text())
class Normalize(ast.NodeTransformer):
 def visit_Name(self,n):n.id=n.id.replace('_v2','_v1').replace('V2','V1');return n
 def visit_FunctionDef(self,n):n.name=n.name.replace('_v2','_v1');return self.generic_visit(n)
 def visit_ClassDef(self,n):n.name=n.name.replace('V2','V1');return self.generic_visit(n)
for name in ['_tile_evaluations','stitch_context_v1','parent_layer_messages_v1','TileContextV1','ParentContextV1']:
 a=next(n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 b=next(n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name.replace('_v1','_v2').replace('V1','V2'))
 b=Normalize().visit(deepcopy(b))
 check('predecessor_scientific_AST_unchanged_'+name,ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
torch.manual_seed(9021)
Q,R,S,H=3,2,2,2
fields=('cqq','cqr','eigenvalues','eigenvectors')
def fixture(dtype):
 cs=[]
 for i in range(5):
  edge=torch.tensor([[0.,.13+i*.02,.05],[.13+i*.02,0.,.07],[.05,.07,0.]],dtype=torch.float64)
  cs.append(torch.diag(edge.sum(1)+torch.tensor([.8,1.2,1.5])+i*.03)-edge)
 matrices=torch.stack(cs);cqq=torch.stack((matrices,matrices*1.4),1)
 ev,evec=torch.linalg.eigh(cqq);cqr=-torch.arange(1.,5*S*Q*R+1.).double().reshape(5,S,Q,R)/500.
 ops=legacy.AnchorOperators(*(v.to(dtype) for v in (cqq,cqr,ev,evec)))
 tiles=[]
 for i,bounds in enumerate(([0.,.03,.07],[.07,.11,.16],[.16,.19,.21],[.21,.25,.29])):
  left,middle,right=bounds;positions=torch.tensor([left,left+(right-left)/10,middle,right-.001],dtype=torch.float64)
  intervals=torch.tensor([[left,middle],[middle,right]],dtype=torch.float64)
  bank=types.SimpleNamespace(intervals=intervals,positions=positions,target_graph=torch.searchsorted(intervals[:,0].contiguous(),positions,right=True)-1,
       window_bounds=torch.tensor([left,right],dtype=torch.float64))
  layout=legacy._layout(bank,5);trees=(30+i,31+i)
  tiles.append(m.TileContextV2(layout,legacy.AnchorOperators(*(getattr(ops,f)[i:i+2] for f in fields)),
   torch.tensor([17,4,91]),torch.tensor([8,10]),trees,feature_fingerprints=tuple(hashlib.sha256(str(t).encode()).hexdigest() for t in trees),
   parent_id='handmade_independent_train',split='train',namespace='independent_native'))
 return tuple(tiles),m.stitch_context_v2(tiles,fixture=True)
for dtype in (torch.float64,torch.float32):
 tiles,parent=fixture(dtype);G=8;N=Q+R
 model=legacy.AnchorKernelModel(input_dim=4,hidden_dim=5,num_layers=2,latent_dim=H,kernel_depth=2,num_scales=S,
             geometry='age',num_anchors=5,initial_coupling=(2.,3.),initial_damping=(.7,1.)).to(dtype)
 unique=torch.randn(5,N,4,dtype=dtype)
 inverse=torch.tensor([0,1,1,2,2,3,3,4]);leaf=unique[inverse]
 response=torch.tensor([[1.,-.03,-.07,-.1,-.02],[-.03,1.1,-.08,-.02,-.04],[-.07,-.08,.9,-.03,-.02],
                        [-.1,-.02,-.03,1.2,-.01],[-.02,-.04,-.02,-.01,.8]],dtype=dtype).expand(G,N,N)
 ids=torch.tensor([8,4,10,17,91]);calls=[]
 hook=model.encoder.register_forward_pre_hook(lambda module,args:calls.append(args[0].shape[0]))
 encoded=m.encode_parent_sources_v2(parent,model,leaf,response,leaf_ids=ids);hook.remove()
 check('unique_original_encoder_one_batch_'+str(dtype),calls==[5])
 check('exact_unique_tree_row_plan_'+str(dtype),encoded.unique_original_tree_ids==(30,31,32,33,34) and
       encoded.first_cache_graph_indexes==(0,1,3,5,7) and encoded.inverse_cache_graph_indexes==tuple(inverse.tolist()))
 check('cache_occurrences_anchors_masses_not_deduplicated_'+str(dtype),parent.original_tree_ids_by_cache_graph==(30,31,31,32,32,33,33,34) and len(parent.layout.anchors)==20 and
       torch.equal(parent.layout.weights,torch.cat([t.layout.weights for t in tiles])))
 for a,b in [(1,2),(3,4),(5,6)]:
  check('exact_initial_QR_reuse_'+str(dtype)+'_'+str(a),torch.equal(encoded.query[a],encoded.query[b]) and torch.equal(encoded.reference[a],encoded.reference[b]))
  wrongq=encoded.query.clone();wrongr=encoded.reference.clone()
  wrongq[b,0,0]=torch.nextafter(wrongq[b,0,0],torch.tensor(float('inf'),dtype=dtype))
  reject('one_ULP_initial_Q_mismatch_rejected_'+str(dtype)+'_'+str(a),lambda:m.parent_encoded_context_v2(parent,model,wrongq,wrongr,fixture=True))
  wrongq=encoded.query.clone();wrongr[b,0,1]=torch.nextafter(wrongr[b,0,1],torch.tensor(float('-inf'),dtype=dtype))
  reject('one_ULP_initial_R_mismatch_rejected_'+str(dtype)+'_'+str(a),lambda:m.parent_encoded_context_v2(parent,model,wrongq,wrongr,fixture=True))
 context=m.parent_context_v2(parent,model,leaf,response,leaf_ids=ids,event_block_size=3,checkpoint_blocks=True)
 diagnostic=m.parent_encoded_context_v2(parent,model,encoded.query,encoded.reference,fixture=True,event_block_size=3,checkpoint_blocks=True)
 close('production_exact_valid_diagnostic_'+str(dtype),context.targets,diagnostic.targets,dtype)
 check('evolved_same_tree_contexts_may_diverge_'+str(dtype),
       not torch.equal(context.anchors[parent.layout.anchor_graph==1][0],context.anchors[parent.layout.anchor_graph==2][-1]))
 reject('encoded_diagnostic_requires_fixture_'+str(dtype),lambda:m.parent_encoded_context_v2(parent,model,encoded.query,encoded.reference))
 for field in ['leaf','response']:
  wrongleaf=leaf.clone();wrongresponse=response.clone()
  target=wrongleaf if field=='leaf' else wrongresponse
  target[2,0,0]=torch.nextafter(target[2,0,0],torch.tensor(float('inf'),dtype=dtype))
  reject('one_ULP_fixed_original_'+field+'_mismatch_rejected_'+str(dtype),lambda:m.parent_context_v2(parent,model,wrongleaf,wrongresponse,leaf_ids=ids))
 reject('fixed_production_leaf_disallows_grad_'+str(dtype),lambda:m.parent_context_v2(parent,model,leaf.clone().requires_grad_(),response,leaf_ids=ids))
 reject('fixed_production_response_disallows_grad_'+str(dtype),lambda:m.parent_context_v2(parent,model,leaf,response.clone().requires_grad_(),leaf_ids=ids))
 for invalid in [ids.double(),ids[:4],torch.tensor([8,4,10,17,17]),torch.tensor([8,4,10,17,92])]:
  reject('sample_join_rejection_'+str(dtype)+'_'+str(invalid.tolist()),lambda invalid=invalid:m.parent_context_v2(parent,model,leaf,response,leaf_ids=invalid))
 permutation=torch.tensor([3,0,4,2,1])
 changed=m.parent_context_v2(parent,model,leaf[:,permutation],response[:,permutation][:,:,permutation],leaf_ids=ids[permutation],event_block_size=3,checkpoint_blocks=True)
 close('sample_permutation_join_preserves_context_'+str(dtype),changed.targets,context.targets,dtype)
 coefficient=torch.randn_like(context.targets)
 ga=torch.autograd.grad((context.targets*coefficient).sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
 gb=torch.autograd.grad((changed.targets*coefficient).sum(),tuple(model.parameters()),retain_graph=True,allow_unused=True)
 grads('sample_permutation_join_preserves_all_model_VJP_'+str(dtype),ga,gb,dtype)
 selected=torch.tensor([15,0,7,3])
 chunk=m.parent_context_v2(parent,model,leaf,response,leaf_ids=ids,target_indexes=selected,event_block_size=1,checkpoint_blocks=True)
 close('selected_target_order_full_layers_'+str(dtype),chunk.targets,context.targets[selected],dtype)
 # The encoder is rerun after an actual handmade parameter change.
 oldbias=model.encoder.input_linear.bias.detach().clone()
 with torch.no_grad():model.encoder.input_linear.bias[0]+=.4
 next_encoded=m.encode_parent_sources_v2(parent,model,leaf,response,leaf_ids=ids)
 check('unique_encoding_not_stale_after_parameter_update_'+str(dtype),not torch.equal(encoded.query,next_encoded.query))
 with torch.no_grad():model.encoder.input_linear.bias.copy_(oldbias)
 # Same legacy nonlinear context using independently prepared dense bank.
 dense_model=deepcopy(model);dense_model.num_anchors=20
 lookup={node:i for i,node in enumerate(ids.tolist())}
 cache=types.SimpleNamespace(normalized_response=response,query_indices=torch.tensor([lookup[i] for i in parent.query_ids.tolist()]),
                             reference_indices=torch.tensor([lookup[i] for i in parent.reference_ids.tolist()]))
 prepared=types.SimpleNamespace(bank=types.SimpleNamespace(leaf_x=leaf,query_cache=cache,target_graph=parent.target_graph),
                         layout=parent.layout,operators={'age':parent.operators},num_scales=S)
 dense=dense_model.encode_context(prepared)
 close('unique_initial_encoder_legacy_dense_nonlinear_targets_'+str(dtype),context.targets,dense.targets,dtype)
 details['unique_encoder_'+str(dtype)]={'unique_trees':5,'cache_occurrences':8,'retained_anchors':20,'encoder_batches':calls}
# Exact decay-domain guard reproductions including zero, in both ordinary/checkpoint entry.
_,parent=fixture(torch.float64);L=parent.layout
signal=torch.ones(len(L.anchors),S,Q,H,2,dtype=torch.float64)
for kind in ['overflow','negative','zero']:
 which=parent
 if kind=='overflow':lam=torch.full((S,H),1e308,dtype=torch.float64);delta=torch.ones((S,H),dtype=torch.float64)
 else:
  tiles=[]
  for t in parent.tile_contexts:
   C=torch.diag(torch.tensor([-1e-15,1.,2.],dtype=torch.float64)).expand(2,S,Q,Q).clone()
   E=torch.tensor([-1e-15,1.,2.],dtype=torch.float64).expand(2,S,Q).clone()
   V=torch.eye(Q,dtype=torch.float64).expand(2,S,Q,Q).clone()
   tiles.append(replace(t,operators=legacy.AnchorOperators(C,t.operators.cqr,E,V)))
  which=m.stitch_context_v2(tiles,fixture=True)
  lam=torch.full((S,H),1e16 if kind=='negative' else 1e15,dtype=torch.float64);delta=torch.ones((S,H),dtype=torch.float64)
 reject('legacy_rejects_'+kind+'_decay',lambda:legacy.anchor_kernel_action(which.layout,which.operators,signal,lam,delta))
 for cp in [False,True]:reject('v2_rejects_'+kind+'_decay_checkpoint'+str(cp),lambda:m.streamed_parent_action_v2(which,signal.clone().requires_grad_(cp),lam,delta,event_block_size=1,checkpoint_blocks=cp))
# Real finite zero gradients are permitted; individual inactive parameters explicit.
for dtype in [torch.float64,torch.float32]:
 for frozen in [False,True]:
  p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));unused=torch.nn.Parameter(torch.tensor([1.],dtype=dtype),requires_grad=not frozen)
  opt=torch.optim.SGD([p,unused],lr=.1)
  before=p.detach().clone();ub=unused.detach().clone()
  result=m.matched_parent_step_v2(opt,[p,unused],[lambda:(p.square().sum()*0.,2.)],inactive_parameters=[unused])
  check('connected_zero_grad_permitted_'+str(dtype)+'_'+str(frozen),torch.count_nonzero(p.grad)==0 and torch.equal(p,before) and result['connected_optimizer_parameters']==1)
  check('explicit_unused_inactive_preserved_'+str(dtype)+'_'+str(frozen),unused.grad is None and torch.equal(unused,ub))
 p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));q=torch.nn.Parameter(p.detach().clone())
 opt=torch.optim.SGD([p],lr=.1);ref=torch.optim.SGD([q],lr=.1)
 result=m.matched_parent_step_v2(opt,[p],[lambda:(p[0].square(),2.),lambda:(p[1].square(),3.)])
 ((q[0].square()+q[1].square())/5).backward();torch.nn.utils.clip_grad_norm_([q],1.);ref.step()
 close('separate_cell_paths_one_aggregate_step_'+str(dtype),p,q,dtype)
 grads('separate_cell_paths_one_aggregate_gradient_'+str(dtype),[p.grad],[q.grad],dtype)
# Disconnection and binding failures stop before optimizer step.
def parameters():
 p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64));q=torch.nn.Parameter(torch.tensor([.1],dtype=torch.float64))
 return p,q,torch.optim.SGD([p,q],lr=.1)
p,q,opt=parameters();foreign=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
reject('per_cell_foreign_leaf_rejected',lambda:m.matched_parent_step_v2(opt,[p,q],[lambda:((p.square().sum()+q.square().sum()),1.),lambda:(foreign.square(),1.)]))
check('foreign_rejection_no_parameter_gradients',p.grad is None and q.grad is None)
p,q,opt=parameters()
reject('aggregate_missing_undeclared_parameter_rejected',lambda:m.matched_parent_step_v2(opt,[p,q],[lambda:(p.square().sum(),1.)]))
p,q,opt=parameters()
reject('declared_inactive_but_connected_rejected',lambda:m.matched_parent_step_v2(opt,[p,q],[lambda:((p.square().sum()+q.square().sum()),1.)],inactive_parameters=[q]))
p,q,opt=parameters()
def mutate_binding():
 opt.param_groups[0]['params']=[p]
 return p.square().sum()+q.square().sum(),1.
reject('binding_mutation_during_cell_rejected',lambda:m.matched_parent_step_v2(opt,[p,q],[mutate_binding]))
p,q,opt=parameters()
handle=p.register_hook(lambda gradient:(opt.param_groups[0].update(params=[q]) or gradient))
reject('binding_mutation_during_backward_rejected',lambda:m.matched_parent_step_v2(opt,[p,q],[lambda:((p.square().sum()+q.square().sum()),1.)]))
handle.remove()
class NoGradient(torch.autograd.Function):
 @staticmethod
 def forward(ctx,p):return p.sum()
 @staticmethod
 def backward(ctx,g):return None
class NanGradient(torch.autograd.Function):
 @staticmethod
 def forward(ctx,p):ctx.shape=p.shape;return p.sum()
 @staticmethod
 def backward(ctx,g):return g.new_full(ctx.shape,float('nan'))
for cls in [NoGradient,NanGradient]:
 p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64));opt=torch.optim.SGD([p],lr=.1);before=p.detach().clone()
 reject('actual_missing_or_nonfinite_gradient_'+cls.__name__,lambda:m.matched_parent_step_v2(opt,[p],[lambda:(cls.apply(p),1.)]))
 check('no_step_on_bad_gradient_'+cls.__name__,torch.equal(p,before))
# New decisive counterexample: callbacks can contaminate accumulated gradients.
for kind in ['preloaded_gradient','callback_backward']:
 p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64));q=torch.nn.Parameter(p.detach().clone())
 opt=torch.optim.SGD([p],lr=.1);ref=torch.optim.SGD([q],lr=.1);backwards=[]
 h=p.register_hook(lambda gradient:(backwards.append(gradient.detach().clone()) or gradient))
 def cell():
  loss=p.square().sum()
  if kind=='preloaded_gradient':p.grad=torch.tensor([10.,0.],dtype=torch.float64)
  else:loss.backward(retain_graph=True)
  return loss,1.
 result=m.matched_parent_step_v2(opt,[p],[cell]);h.remove()
 q.square().sum().backward();torch.nn.utils.clip_grad_norm_([q],1.);ref.step()
 difference=float((p-q).abs().max())
 check('counterexample_'+kind+'_accepted_and_changes_matched_update',difference>1e-5)
 check('counterexample_'+kind+'_reports_single_backward',result['backward_calls']==1)
 if kind=='callback_backward':check('callback_and_aggregate_two_actual_VJPs',len(backwards)==2)
 details[kind]={'result':result,'actual_parameter':p.detach().tolist(),'clean_reference_parameter':q.detach().tolist(),
               'maximum_parameter_difference':difference,'actual_backward_hook_calls':len(backwards)}
for p,h in pins.items():check('all_pins_preserved_after_'+p,sha(p)==h)
report={'kind':'anchor_parent_context_v2_independent_checks','status':'completed_with_counterexamples','count':len(checks),'checks':checks,'details':details,
 'source_sha256':pins['src/arggnn/anchor_parent_context_v2.py'],'config_sha256':pins['configs/anchor_parent_context_v2.json'],
 'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'author_checks_replayed_once':699,
 'actual_arrays_GPU_live_budgets_locks_installs_or_Git':False,'temporary_and_cache_writes_confined_to_ownership':True}
(DIR/'independent_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'counterexamples':{k:v for k,v in details.items() if 'gradient' in k or 'backward' in k}},indent=2))
