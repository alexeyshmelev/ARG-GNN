"""Concrete callback contamination regression and exact clean aggregate updates."""
import contextlib,hashlib,json,os,sys
from pathlib import Path
from unittest.mock import patch
import torch
from arggnn import anchor_parent_context_v2 as previous,anchor_parent_context_v3 as core
torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
R=core.ROOT;O=R/'research/reviews/anchor_parent_context_v3_independent'
checks=[];details={};guard=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,value=True):
 if not bool(value):raise AssertionError(name)
 checks.append(name)
def equal(name,a,b,dtype):
 cfg=json.loads(core.CONFIG.read_text())['qualification'][str(dtype).split('.')[-1]]
 torch.testing.assert_close(a,b,rtol=cfg['vjp_rtol'],atol=cfg['vjp_atol']);check(name)
def audit(event,args):
 if event=='open' and not isinstance(args[0],int):
  try:p=Path(os.path.abspath(os.fsdecode(args[0])))
  except (TypeError,ValueError):return
  mode,flags=args[1:3];writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
  own=p==O or O in p.parents
  if writing and not own:guard.append('outside ownership write '+str(p));raise RuntimeError(guard[-1])
  if not writing and not own:
   for base in (R/'data',R/'experiments/results',R/'.runtime/gpup',R/'.runtime/locks',R/'.runtime/storage',Path('/proc')):
    if p==base or base in p.parents:guard.append('actual state read '+str(p));raise RuntimeError(guard[-1])
sys.addaudithook(audit)
pins=json.loads((O/'baseline_pins.json').read_text())
for p,h in pins.items():check('pin_before_'+p,sha(R/p)==h)
# Reproduce the sealed two cases unchanged, then require v3 rejection.
for dtype in (torch.float64,torch.float32):
 for mode in ('manual','callback_backward','zero_manual','inactive_manual'):
  p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype))
  q=torch.nn.Parameter(torch.tensor([.7],dtype=dtype))
  params=[p,q] if mode=='inactive_manual' else [p]
  opt=torch.optim.SGD(params,lr=.1)
  before=[x.detach().clone() for x in params];hooks=[];cells=[]
  p.register_hook(lambda g:hooks.append(g.detach().clone()))
  def callback():
   cells.append(True);loss=p.square().sum()
   if mode=='manual':p.grad=torch.tensor([10.,0.],dtype=dtype)
   elif mode=='zero_manual':p.grad=torch.zeros_like(p)
   elif mode=='callback_backward':loss.backward(retain_graph=True)
   else:q.grad=torch.tensor([10.],dtype=dtype)
   return loss,1.
  kwargs={'inactive_parameters':[q]} if mode=='inactive_manual' else {}
  try:core.matched_parent_step_v3(opt,params,[callback],**kwargs)
  except ValueError as error:
   check('F4_reject_'+mode+'_'+str(dtype),'callbacks must leave optimizer gradients unset' in str(error))
  else:raise AssertionError('contaminating callback accepted')
  check('F4_no_step_'+mode+'_'+str(dtype),all(torch.equal(a,b) for a,b in zip(before,params)) and not opt.state)
  check('F4_grads_preserved_not_silently_cleared_'+mode+'_'+str(dtype),(q if mode=='inactive_manual' else p).grad is not None)
  check('F4_callback_and_aggregate_hook_counts_'+mode+'_'+str(dtype),len(cells)==1 and len(hooks)==(1 if mode=='callback_backward' else 0))
  details['rejected_'+mode+'_'+str(dtype)]={'callback_calls':len(cells),'actual_backward_hook_calls':len(hooks),'parameter_unchanged':True,'grad_preserved':True}
 for mode in ('manual','callback_backward'):
  p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));opt=torch.optim.SGD([p],lr=.1);hooks=[]
  p.register_hook(lambda g:hooks.append(g.detach().clone()))
  def previous_callback():
   loss=p.square().sum()
   if mode=='manual':p.grad=torch.tensor([10.,0.],dtype=dtype)
   else:loss.backward(retain_graph=True)
   return loss,1.
  result=previous.matched_parent_step_v2(opt,[p],[previous_callback])
  clean=torch.tensor([.16,-.24],dtype=dtype)
  check('F4_preserved_v2_counterexample_'+mode+'_'+str(dtype),not torch.allclose(p,clean,atol=1e-5,rtol=1e-5))
  check('F4_preserved_v2_actual_extra_hook_'+mode+'_'+str(dtype),len(hooks)==(2 if mode=='callback_backward' else 1) and result['backward_calls']==1)
  details['v2_'+mode+'_'+str(dtype)]={'actual_parameter':p.detach().tolist(),'clean_reference_parameter':clean.tolist(),'actual_backward_hook_calls':len(hooks),'reported_backward_calls':result['backward_calls']}
 # The second requested guard is distinct from the per-cell guard.
 p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));opt=torch.optim.SGD([p],lr=.1);before=p.detach().clone();hooks=[]
 p.register_hook(lambda g:hooks.append(g.detach().clone()));original_stack=torch.stack
 def injected_stack(*args,**kwargs):
  value=original_stack(*args,**kwargs);p.grad=torch.tensor([1.,2.],dtype=dtype);return value
 with patch.object(torch,'stack',side_effect=injected_stack):
  try:core.matched_parent_step_v3(opt,[p],[lambda:(p.square().sum(),1.)])
  except ValueError as error:check('F4_immediate_pre_backward_guard_'+str(dtype),'immediately before aggregate backward' in str(error))
  else:raise AssertionError('pre-backward contamination accepted')
 check('F4_pre_backward_no_actual_backward_or_step_'+str(dtype),not hooks and torch.equal(p,before) and p.grad is not None)
 # Initial old-step grads are cleared by the original zero_grad, before callbacks.
 p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));opt=torch.optim.SGD([p],lr=.1);p.grad=torch.tensor([10.,0.],dtype=dtype);hooks=[]
 p.register_hook(lambda g:hooks.append(g.detach().clone()))
 result=core.matched_parent_step_v3(opt,[p],[lambda:(p.square().sum(),1.)])
 equal('F4_clean_SGD_update_after_original_initial_zero_grad_'+str(dtype),p,torch.tensor([.16,-.24],dtype=dtype),dtype)
 check('F4_clean_SGD_one_actual_hook_'+str(dtype),len(hooks)==1 and result['backward_calls']==result['clip_calls']==result['optimizer_steps']==1)
 # Same complete objective / single clip / real optimizer state over two steps.
 for family in ('SGD','AdamW'):
  p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));q=torch.nn.Parameter(torch.tensor([.6],dtype=dtype));inactive=torch.nn.Parameter(torch.tensor([.7],dtype=dtype),requires_grad=False)
  rp=torch.nn.Parameter(p.detach().clone());rq=torch.nn.Parameter(q.detach().clone());ri=torch.nn.Parameter(inactive.detach().clone(),requires_grad=False)
  params=[p,q,inactive];refs=[rp,rq,ri]
  def optimizer(parameters):
   return torch.optim.SGD(parameters,lr=.1,momentum=.8,weight_decay=.01) if family=='SGD' else torch.optim.AdamW(parameters,lr=.001,betas=(.8,.95),weight_decay=.0001)
  opt=optimizer(params);refopt=optimizer(refs);hooks=[]
  p.register_hook(lambda g:hooks.append(g.detach().clone()))
  for iteration in range(2):
   cells=[lambda:(20.*p.square().sum()+q.square().sum(),1.),lambda:((p.sum()-q.sum()).square(),2.)]
   calls=[];original_clip=torch.nn.utils.clip_grad_norm_;step=opt.step
   with patch.object(torch.nn.utils,'clip_grad_norm_',wraps=original_clip) as clipped,patch.object(opt,'step',wraps=step) as stepped:
    result=core.matched_parent_step_v3(opt,params,cells,inactive_parameters=[inactive])
    check('F4_one_clip_and_step_'+family+'_'+str(dtype)+'_'+str(iteration),clipped.call_count==1 and stepped.call_count==1)
   refopt.zero_grad(set_to_none=True)
   reference_loss=(20.*rp.square().sum()+rq.square().sum()+(rp.sum()-rq.sum()).square())/3.
   reference_loss.backward();reference_norm=original_clip(refs,1.,error_if_nonfinite=True);refopt.step()
   check('F4_single_actual_backward_hook_'+family+'_'+str(dtype)+'_'+str(iteration),len(hooks)==iteration+1)
   equal('F4_loss_'+family+'_'+str(dtype)+'_'+str(iteration),torch.tensor(result['loss'],dtype=dtype),reference_loss.detach(),dtype)
   equal('F4_clip_norm_'+family+'_'+str(dtype)+'_'+str(iteration),torch.tensor(result['gradient_norm_before_single_clip'],dtype=dtype),reference_norm,dtype)
   for index,(a,b) in enumerate(zip(params,refs)):
    equal('F4_parameter_'+family+'_'+str(dtype)+'_'+str(iteration)+'_'+str(index),a,b,dtype)
    if a.grad is None or b.grad is None:check('F4_inactive_grad_None_'+family+'_'+str(dtype)+'_'+str(iteration),a.grad is b.grad is None)
    else:equal('F4_clipped_grad_'+family+'_'+str(dtype)+'_'+str(iteration)+'_'+str(index),a.grad,b.grad,dtype)
    check('F4_optimizer_state_keys_'+family+'_'+str(dtype)+'_'+str(iteration)+'_'+str(index),opt.state.get(a,{}).keys()==refopt.state.get(b,{}).keys())
    for key in opt.state.get(a,{}):
     av,bv=opt.state[a][key],refopt.state[b][key]
     if isinstance(av,torch.Tensor):equal('F4_state_'+family+'_'+str(dtype)+'_'+str(iteration)+'_'+str(index)+'_'+key,av,bv,dtype)
     else:check('F4_state_'+key,av==bv)
   check('F4_declared_inactive_policy_'+family+'_'+str(dtype)+'_'+str(iteration),result['connected_optimizer_parameters']==2 and result['declared_inactive_parameters']==1)
 # Genuine zero gradients occur after, not inside, the callbacks.
 p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));opt=torch.optim.AdamW([p],lr=.001,weight_decay=0.);hooks=[]
 p.register_hook(lambda g:hooks.append(g.detach().clone()))
 result=core.matched_parent_step_v3(opt,[p],[lambda:((p*0).sum(),1.)])
 check('F4_connected_zero_post_backward_permitted_'+str(dtype),p.grad is not None and torch.count_nonzero(p.grad)==0 and len(hooks)==1 and result['gradient_norm_before_single_clip']==0. and opt.state[p]['step']==1)
for p,h in pins.items():check('pin_after_'+p,sha(R/p)==h)
check('zero_forbidden_access_events',not guard)
report={'kind':'anchor_parent_context_v3_author_F4_handmade_checks','status':'PASS','count':len(checks),'checks':checks,'details':details,'guard_events':guard,
 'source_sha256':sha(R/'src/arggnn/anchor_parent_context_v3.py'),'config_sha256':sha(core.CONFIG),'test_source_sha256':sha(__file__),'actual_arrays_fits_GPU_live_resources_installation_or_Git':False}
(O/'author_F4_replay_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'guard_events':guard},indent=2))
