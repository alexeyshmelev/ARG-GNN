"""Independent narrow H024 v3 gradient purity and whole-module preservation checks."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import ast,hashlib,json,os,sys
import torch
from arggnn import anchor_parent_context_v2 as old, anchor_parent_context_v3 as new
torch.set_num_threads(1)
ROOT=new.ROOT
OWN=ROOT/"research/reviews/anchor_parent_context_v3_independent"
checks=[];details={};events=[]
assert os.environ["CUDA_VISIBLE_DEVICES"]==""
def check(name,value=True):
    if not value:raise AssertionError(name)
    checks.append(name)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
def audit(event,args):
    if event!="open" or isinstance(args[0],int):return
    try:p=Path(os.path.abspath(os.fsdecode(args[0])))
    except (TypeError,ValueError):return
    mode,flags=args[1:3]
    writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    owned=p==OWN or OWN in p.parents
    if writing and not owned:events.append("outside ownership write "+str(p));raise RuntimeError(events[-1])
    if not writing and not owned:
        for base in (ROOT/"data",ROOT/"experiments/results",ROOT/".runtime/gpup",ROOT/".runtime/locks",ROOT/".runtime/storage",Path("/proc")):
            if p==base or base in p.parents:events.append("actual state read "+str(p));raise RuntimeError(events[-1])
sys.addaudithook(audit)
baseline=json.loads((OWN/"baseline_pins.json").read_text())
baseline.update({
 "src/arggnn/anchor_parent_context_v3.py":"1b9f8f8148ed494373f9f76de677b847dff8ed256db15f6ff9b83ee0755f66e7",
 "configs/anchor_parent_context_v3.json":"e8a2afdfd61495ed0acacbfe5376ed8785878b0fe417ba74c6947bd965ea38d5",
 "research/reviews/anchor_parent_context_v3_author/author_receipt.json":"d83d8b6e531ad50ae221a00974666140868cb6b3aaa46eefe7cc32c535dbd1ab",
 "research/reviews/anchor_parent_context_v3_author/source_delta_evidence.json":"ee68140b10f8c7eb6316767a7c9b4ed074d724507dfaee657c20613b60b16bb9"})
for path,h in baseline.items():check("pin_before_"+path,sha(ROOT/path)==h)
source=(ROOT/"src/arggnn/anchor_parent_context_v3.py").read_text()
pred=(ROOT/"src/arggnn/anchor_parent_context_v2.py").read_text()
callback_guard="        if any(p.grad is not None for p in parameters):\n            raise ValueError('Parent loss callbacks must leave optimizer gradients unset')\n"
backward_guard="    if any(p.grad is not None for p in parameters):\n        raise ValueError('Optimizer gradients must be unset immediately before aggregate backward')\n"
check("exact_one_post_callback_guard",source.count(callback_guard)==1)
check("exact_one_immediate_pre_backward_guard",source.count(backward_guard)==1)
inverse=source.replace(callback_guard,"").replace(backward_guard,"").replace("_v3","_v2").replace("V3","V2").replace(new.CONFIG_SHA256,old.CONFIG_SHA256)
check("inverse_whole_module_AST_identical_to_sealed_v2",ast.dump(ast.parse(inverse),include_attributes=False)==ast.dump(ast.parse(pred),include_attributes=False))
cfg3=json.loads(new.CONFIG.read_text());cfg2=json.loads(old.CONFIG.read_text())
changed={k for k in set(cfg2)|set(cfg3) if cfg2.get(k)!=cfg3.get(k)}
check("config_only_registered_version_lineage_callback_policy_changes",changed=={"kind","predecessor","optimizer_connection_policy"})
check("original_tolerances_production_streaming_training_source_pins_preserved",all(cfg3[k]==cfg2[k] for k in ("qualification","production_contract","streaming","training_design","source_pins")))
check("v3_stays_inactive_and_actual_execution_unauthorized",cfg3["execution_enabled"] is False and cfg3["actual_arrays_models_fits_GPU_budget_or_installation_authorized"] is False)
helper=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="matched_parent_step_v3")
calls=[n.func.attr for n in ast.walk(helper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
check("single_initial_clear_single_backward_clip_step",all(calls.count(k)==1 for k in ("zero_grad","backward","clip_grad_norm_","step")))
def close(name,a,b,dtype):
    tol=cfg3["qualification"][str(dtype).split(".")[-1]]
    torch.testing.assert_close(a,b,atol=tol["vjp_atol"],rtol=tol["vjp_rtol"])
    check(name)
def optimizer(family,parameters):
    return torch.optim.SGD(parameters,lr=.13,momentum=.71,weight_decay=.03) if family=="SGD" else torch.optim.AdamW(parameters,lr=.017,betas=(.73,.91),weight_decay=.021)
def seed_state(opt,params):
    opt.zero_grad(set_to_none=True)
    (7*params[0].square().sum()+params[1].square().sum()).backward()
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    opt.step()
for dtype in (torch.float64,torch.float32):
 for family in ("SGD","AdamW"):
  # Existing optimizer state must survive a rejected callback unchanged.
  for scenario in ("manual_active","zero_active","backward_first","backward_later","zero_inactive","frozen_inactive"):
   p=torch.nn.Parameter(torch.tensor([.31,-.41],dtype=dtype));q=torch.nn.Parameter(torch.tensor([.19],dtype=dtype))
   inactive=torch.nn.Parameter(torch.tensor([.83],dtype=dtype),requires_grad=scenario!="frozen_inactive")
   params=[p,q,inactive];opt=optimizer(family,params);seed_state(opt,params)
   params_before=[x.detach().clone() for x in params];state_before=deepcopy(opt.state_dict());hooks=[];callback_count=[]
   p.register_hook(lambda g:hooks.append(g.detach().clone()))
   def callback(index):
    def cell():
     callback_count.append(index);loss=11*p.square().sum()+q.square().sum()
     if scenario in ("backward_first","backward_later"):loss.backward(retain_graph=True)
     elif scenario=="manual_active":p.grad=torch.tensor([3.,-2.],dtype=dtype)
     elif scenario=="zero_active":p.grad=torch.zeros_like(p)
     else:inactive.grad=torch.zeros_like(inactive)
     return loss,2.
    return cell
   cells=[callback(0)]
   if scenario=="backward_later":cells=[lambda:(p.square().sum()+q.square().sum(),1.),callback(1)]
   with patch.object(opt,"step",wraps=opt.step) as steps,patch.object(torch.nn.utils,"clip_grad_norm_",wraps=torch.nn.utils.clip_grad_norm_) as clips:
    try:new.matched_parent_step_v3(opt,params,cells,inactive_parameters=[inactive])
    except ValueError as e:check("callback_reject_"+family+scenario+str(dtype),"callbacks must leave optimizer gradients unset" in str(e))
    else:raise AssertionError("Callback gradient accepted")
    check("no_helper_update_clip_"+family+scenario+str(dtype),steps.call_count==clips.call_count==0)
   check("parameters_preserved_"+family+scenario+str(dtype),all(torch.equal(x,y) for x,y in zip(params,params_before)))
   check("existing_optimizer_state_preserved_"+family+scenario+str(dtype),same(opt.state_dict(),state_before))
   dirty=inactive if scenario in ("zero_inactive","frozen_inactive") else p
   check("contamination_remains_visible_"+family+scenario+str(dtype),dirty.grad is not None)
   check("no_extra_aggregate_hook_"+family+scenario+str(dtype),len(hooks)==int(scenario.startswith("backward")))
   details[family+scenario+str(dtype)]={"callback_count":len(callback_count),"actual_backward_hooks":len(hooks),"parameters_and_existing_state_unchanged":True}
  # Inject contamination after all callback guards, while connectivity is checked.
  p=torch.nn.Parameter(torch.tensor([.31,-.41],dtype=dtype));opt=optimizer(family,[p])
  before=p.detach().clone();hooks=[];p.register_hook(lambda g:hooks.append(g.detach().clone()))
  original_paths=new._optimizer_paths_v3;calls_seen=[]
  def dirty_paths(loss):
   result=original_paths(loss);calls_seen.append(True)
   if len(calls_seen)==2:p.grad=torch.zeros_like(p)
   return result
  with patch.object(new,"_optimizer_paths_v3",side_effect=dirty_paths),patch.object(opt,"step",wraps=opt.step) as steps,patch.object(torch.nn.utils,"clip_grad_norm_",wraps=torch.nn.utils.clip_grad_norm_) as clips:
   try:new.matched_parent_step_v3(opt,[p],[lambda:(p.square().sum(),1.)])
   except ValueError as e:check("immediate_pre_backward_reject_"+family+str(dtype),"immediately before aggregate backward" in str(e))
   else:raise AssertionError("late gradient accepted")
   check("late_no_backward_clip_or_step_"+family+str(dtype),not hooks and steps.call_count==clips.call_count==0 and torch.equal(p,before) and p.grad is not None)
  # Fresh complete aggregate reference, including clipped gradients and all state.
  p=torch.nn.Parameter(torch.tensor([.31,-.41],dtype=dtype));q=torch.nn.Parameter(torch.tensor([.19],dtype=dtype));inactive=torch.nn.Parameter(torch.tensor([.83],dtype=dtype),requires_grad=False)
  params=[p,q,inactive];reference=[torch.nn.Parameter(x.detach().clone(),requires_grad=x.requires_grad) for x in params]
  opt=optimizer(family,params);refopt=optimizer(family,reference);rp,rq,ri=reference
  hooks=[];p.register_hook(lambda g:hooks.append(g.detach().clone()))
  for iteration in range(2):
   # Old-step grads are deliberately present; only the original initial clear handles them.
   p.grad=torch.full_like(p,8.)
   result=new.matched_parent_step_v3(opt,params,[lambda:(19*p.square().sum()+q.square().sum(),2.),lambda:((p.sum()+3*q.sum()).square(),3.)],inactive_parameters=[inactive])
   refopt.zero_grad(set_to_none=True);loss=(19*rp.square().sum()+rq.square().sum()+(rp.sum()+3*rq.sum()).square())/5.
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(reference,1.,error_if_nonfinite=True);refopt.step()
   prefix=family+str(dtype)+str(iteration)
   check("one_actual_aggregate_hook_"+prefix,len(hooks)==iteration+1)
   check("one_reported_backward_clip_step_"+prefix,result["backward_calls"]==result["clip_calls"]==result["optimizer_steps"]==1)
   close("reference_loss_"+prefix,torch.tensor(result["loss"],dtype=dtype),loss.detach(),dtype)
   close("reference_pre_clip_norm_"+prefix,torch.tensor(result["gradient_norm_before_single_clip"],dtype=dtype),norm,dtype)
   for index,(a,b) in enumerate(zip(params,reference)):
    close("reference_parameter_"+prefix+str(index),a,b,dtype)
    if a.grad is None:check("inactive_gradient_None_"+prefix,b.grad is None)
    else:close("reference_clipped_gradient_"+prefix+str(index),a.grad,b.grad,dtype)
    check("reference_optimizer_state_keys_"+prefix+str(index),opt.state.get(a,{}).keys()==refopt.state.get(b,{}).keys())
    for key,value in opt.state.get(a,{}).items():
     expected=refopt.state[b][key]
     if isinstance(value,torch.Tensor):close("reference_optimizer_state_"+prefix+str(index)+key,value,expected,dtype)
     else:check("reference_optimizer_state_"+prefix+str(index)+key,value==expected)
   check("explicit_inactive_and_full_active_connectivity_"+prefix,result["declared_inactive_parameters"]==1 and result["connected_optimizer_parameters"]==2)
  # Both active parameters can have connected legitimate zero aggregate grads.
  p=torch.nn.Parameter(torch.tensor([.31,-.41],dtype=dtype));q=torch.nn.Parameter(torch.tensor([.19],dtype=dtype));opt=optimizer(family,[p,q])
  hooks=[];p.register_hook(lambda g:hooks.append(g.detach().clone()))
  result=new.matched_parent_step_v3(opt,[p,q],[lambda:((p*0).sum()+(q*0).sum(),4.)])
  check("connected_zero_gradients_accepted_"+family+str(dtype),len(hooks)==1 and p.grad is not None and q.grad is not None and torch.count_nonzero(p.grad)==torch.count_nonzero(q.grad)==0 and result["gradient_norm_before_single_clip"]==0.)
 # Preserve the exact rejected v2 counterexample as independent evidence.
 for mode in ("manual","backward"):
  p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=dtype));opt=torch.optim.SGD([p],lr=.1);hooks=[];p.register_hook(lambda g:hooks.append(g.detach().clone()))
  def old_cell():
   loss=p.square().sum()
   if mode=="manual":p.grad=torch.tensor([10.,0.],dtype=dtype)
   else:loss.backward(retain_graph=True)
   return loss,1.
  report=old.matched_parent_step_v2(opt,[p],[old_cell])
  check("sealed_v2_counterexample_reproduced_"+mode+str(dtype),not torch.allclose(p,torch.tensor([.16,-.24],dtype=dtype),atol=1e-5,rtol=1e-5))
  check("sealed_v2_hook_count_reproduced_"+mode+str(dtype),len(hooks)==(1 if mode=="manual" else 2) and report["backward_calls"]==1)
  details["v2_"+mode+str(dtype)]={"actual_parameter":p.detach().tolist(),"clean_parameter":[.16,-.24],"actual_hooks":len(hooks)}
for path,h in baseline.items():check("pin_after_"+path,sha(ROOT/path)==h)
check("no_forbidden_actual_state_access_or_write",not events)
report={"kind":"anchor_parent_context_v3_fresh_independent_gradient_purity_checks","verdict":"PASS","count":len(checks),"checks":checks,"details":details,"audit_events":events,"source_sha256":sha(ROOT/"src/arggnn/anchor_parent_context_v3.py"),"config_sha256":sha(new.CONFIG),"author_receipt_sha256":sha(ROOT/"research/reviews/anchor_parent_context_v3_author/author_receipt.json"),"test_source_sha256":sha(__file__),"whole_module_inverse_delta_AST_identical":True,"actual_arrays_targets_TEST_simulation_models_fits_GPU_budgets_or_leases":False}
(OWN/"fresh_gradient_checks.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"verdict":"PASS","count":len(checks),"audit_events":events}))
