"""Independent finite original-size model/adapter panel on handmade sources."""
import os
from pathlib import Path
import tempfile
import sys
import json
import hashlib
import ast
import copy
import importlib.util
from dataclasses import fields, replace
import gc
import weakref
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
(OUT/'_tmp').mkdir(exist_ok=True);os.environ['TMPDIR']=str(OUT/'_tmp');tempfile.tempdir=str(OUT/'_tmp')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np
import torch
from arggnn import census_certified_forest_reuse_v1 as reuse
from arggnn import census_biological_fit_v2 as fit
from arggnn import census_biological_bank_v1 as bank
from arggnn import census_tree_encoder_v1 as tree
from arggnn.census_cut_exchange_v1 import permuted_group_control_v1

torch.set_num_threads(1);torch.set_num_interop_threads(1)
forbidden=[];checks=[];cells=[]
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if ROOT in p.parents:
   rel=p.relative_to(ROOT)
   if rel.parts[0] in ('data','results') or str(rel).startswith('experiments/results/'):
    forbidden.append(str(p));raise PermissionError('Actual scientific input/output forbidden')
   mode,flags=args[1:3]
   write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
   if write and OUT not in p.parents:
    forbidden.append(str(p));raise PermissionError('Review writes restricted to own evidence')
 if event in ('subprocess.Popen','os.system'):
  forbidden.append(event);raise PermissionError('No external jobs')
sys.addaudithook(audit)

def ck(name,value):
 if not bool(value):raise AssertionError(name)
 checks.append(name)
def reject(name,call):
 try:call()
 except (ValueError,TypeError,RuntimeError):checks.append(name)
 else:raise AssertionError('Did not reject '+name)
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def certified(c):return replace(c,certificate_sha256=bank.fingerprint_sha256_v1(bank._content(c)))
def run(adapter,model,cert,control='proposed',permutation=None):
 kw={'group_permutation':permutation} if control=='permuted_groups' else {}
 return adapter(model,cert,expected_certificate_sha256=cert.certificate_sha256,control=control,readout_chunk_size=4,**kw)

def original_model(dtype):
 torch.manual_seed(2027)
 return tree.CensusTreeGNNV1(**fit.load_config_v1()['model']).to(dtype=dtype)

def state(model):
 cfg=fit.load_config_v1()
 return {'model':model,'optimizer':torch.optim.AdamW(fit.optimizer_groups_v1(model,cfg),lr=cfg['training']['learning_rate'],betas=tuple(cfg['training']['betas']),eps=cfg['training']['epsilon']),'steps':0}

def optimizer_equal(a,b):
 a=a.state_dict();b=b.state_dict()
 return a['param_groups']==b['param_groups'] and a['state'].keys()==b['state'].keys() and all(x.keys()==b['state'][key].keys() and all(torch.equal(val,b['state'][key][name]) if isinstance(val,torch.Tensor) else val==b['state'][key][name] for name,val in x.items()) for key,x in a['state'].items())

spec=importlib.util.spec_from_file_location('independent_original_handmade_fixture',ROOT/'tests/test_census_biological_bank_v1.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
cfg=reuse.load_config_v1()
for p,sha in cfg['original_source_sha256'].items():ck('original source pin '+p,h(ROOT/p)==sha)
# Independent AST confirmation of unchanged call, arguments and control branch.
source=ast.parse((ROOT/reuse.SOURCE_PATH).read_text());baseline=ast.parse((ROOT/'src/arggnn/census_biological_bank_v1.py').read_text())
original=next(n for n in baseline.body if isinstance(n,ast.FunctionDef) and n.name=='forward_certified_census_biological_v1')
cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='CertifiedForestReuseV1');adapt=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
def dump(n):return ast.dump(n,include_attributes=False)
call_a=next(n for n in ast.walk(original) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model')
call_b=next(n for n in ast.walk(adapt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model')
ck('exact original model call AST',dump(call_a)==dump(call_b))
branch_a=next(n for n in original.body if isinstance(n,ast.If));branch_b=next(n for n in adapt.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
ck('exact original control/permutation branch AST',dump(branch_a)==dump(branch_b))
validation=next(n for n in ast.walk(adapt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='validate_census_biological_certificate_v1')
ck('same original per-use certificate arguments',dump(ast.Tuple(elts=validation.args,ctx=ast.Load()))==dump(ast.Tuple(elts=original.body[1].value.args,ctx=ast.Load())))
cert=fixture.prepare(unary=True);permutation=torch.tensor([1,3,0,2]);fitcfg=fit.load_config_v1()
ck('new handmade unary source retains original suppressed nodes',len(cert.native_trees[0].parent)>len(cert.inputs[0].parent))
for dtype in (torch.float32,torch.float64):
 for control in reuse.CONTROLS:
  clean=original_model(dtype);candidate=copy.deepcopy(clean);adapter=reuse.CertifiedForestReuseV1();kwargs={'group_permutation':permutation} if control=='permuted_groups' else {}
  parameters=tuple(id(p) for p in candidate.parameters());rng=torch.get_rng_state().clone()
  expected=bank.forward_certified_census_biological_v1(clean,cert,expected_certificate_sha256=cert.certificate_sha256,control=control,readout_chunk_size=4,**kwargs)
  actual=run(adapter,candidate,cert,control,permutation)
  label=str(dtype)+' '+control
  ck(label+' all output/ID/context arrays bitwise equal',all(torch.equal(getattr(expected,f.name),getattr(actual,f.name)) for f in fields(expected)))
  ck(label+' same original parameter objects and RNG',parameters==tuple(id(p) for p in candidate.parameters()) and torch.equal(rng,torch.get_rng_state()))
  mass=torch.arange(1,expected.logits.numel()+1,dtype=torch.float64).reshape_as(expected.logits)/17.
  states=[state(clean),state(candidate)]
  for step in (1,2):
   old=fit.train_step_v1(states[0],cert,cert.certificate_sha256,mass,control,permutation,fitcfg,'cpu')
   with reuse.installed_fit_forward_v1(adapter):new=fit.train_step_v1(states[1],cert,cert.certificate_sha256,mass,control,permutation,fitcfg,'cpu')
   ck(label+f' step{step} complete loss/norm/count equal',old==new)
   ck(label+f' step{step} every parameter/gradient/inactive state equal',all(torch.equal(a,b) and (a.grad is None)==(b.grad is None) and (a.grad is None or torch.equal(a.grad,b.grad)) for a,b in zip(clean.parameters(),candidate.parameters())))
   ck(label+f' step{step} complete AdamW state equal',optimizer_equal(states[0]['optimizer'],states[1]['optimizer']))
  old=fit.raw_prediction_v1(states[0],cert,cert.certificate_sha256,control,permutation,fitcfg)
  with reuse.installed_fit_forward_v1(adapter):new=fit.raw_prediction_v1(states[1],cert,cert.certificate_sha256,control,permutation,fitcfg)
  ck(label+' raw VAL logits/probabilities/IDs bitwise equal',all(np.array_equal(x,y) for x,y in zip(old,new)))
  ck(label+' one weak binding and every later call reused',adapter.bindings==1 and adapter.reuses==adapter.calls-1 and adapter.calls==4)
  ck(label+' original methods restored',fit.forward_certified_census_biological_v1 is reuse.ORIGINAL_FORWARD and 'forward' not in candidate.encoder.__dict__)
  cells.append({'dtype':str(dtype),'control':control,'original_hidden':32,'latent':8,'steps':2,'all_output_loss_gradient_clip_AdamW_VAL_arrays_bitwise_equal':True})
# Original builder and input validator calls observed, not inferred from counters.
builder=tree.prepare_retained_observed_forest_v1;inputcheck=tree._checked_inputs;count={'build':0,'input':0}
def measured_builder(*a,**k):count['build']+=1;return builder(*a,**k)
def measured_input(*a,**k):count['input']+=1;return inputcheck(*a,**k)
tree.prepare_retained_observed_forest_v1=measured_builder;tree._checked_inputs=measured_input
try:
 adapter=reuse.CertifiedForestReuseV1();model=original_model(torch.float32)
 for control in reuse.CONTROLS:
  with torch.no_grad():run(adapter,model,cert,control,permutation)
 ck('one original forest reconstruction for five calls',count['build']==1)
 ck('all original retained checks every graph every call',count['input']==5*len(cert.inputs))
finally:tree.prepare_retained_observed_forest_v1=builder;tree._checked_inputs=inputcheck
# Certificate mutation, dtypes, source roles, learned inputs and recertified
# structural incoherence must all fail before accepting a different forest.
for name,mutation in [
 ('feature',lambda c:c.inputs[0].features.__setitem__((0,10),c.inputs[0].features[0,10]+.2)),
 ('forest parent',lambda c:c.forest.parent.__setitem__(0,0)),
 ('forest level',lambda c:c.forest.upward_levels[0].__setitem__(0,1)),
 ('query age',lambda c:c.bank.partitions[0].query_ages.__setitem__(0,c.bank.partitions[0].query_ages[0]+1)),
 ('sample ID',lambda c:c.inputs[0].sample_ids.__setitem__(0,c.inputs[0].sample_ids[0]+1)),
 ('donor',lambda c:c.bank.donor_weights.__setitem__((0,0,0,0),c.bank.donor_weights[0,0,0,0]+.1)),
 ('learned input',lambda c:c.inputs[0].features.requires_grad_()),
]:
 c=copy.deepcopy(cert);adapter=reuse.CertifiedForestReuseV1();model=original_model(torch.float32)
 with torch.no_grad():run(adapter,model,c)
 mutation(c);reject('cached mutated '+name,lambda:run(adapter,model,c))
 ck('mutated '+name+' leaves methods untouched','forward' not in model.encoder.__dict__)
c=copy.deepcopy(cert);c=replace(c,forest=replace(c.forest,node=c.forest.node.double()));c=certified(c)
reject('new recertified wrong forest dtype',lambda:run(reuse.CertifiedForestReuseV1(),original_model(torch.float32),c))
c=copy.deepcopy(cert);c=replace(c,inputs=(replace(c.inputs[0],upward_order=tuple(reversed(c.inputs[0].upward_order))),*c.inputs[1:]));c=certified(c)
reject('new recertified invalid tree sweep retains original check',lambda:run(reuse.CertifiedForestReuseV1(),original_model(torch.float32),c))
c=copy.deepcopy(cert);c.bank.partitions[0].query_ages[0]+=1;c=certified(c)
reject('new recertified query/native age retains original GNN check',lambda:run(reuse.CertifiedForestReuseV1(),original_model(torch.float32),c))
# Original-size full280 panel, no target/objective or optimizer.
full=fixture.prepare(full=True);adapter=reuse.CertifiedForestReuseV1();model=original_model(torch.float32);fullperm=torch.randperm(184,generator=torch.Generator().manual_seed(73))
for control in reuse.CONTROLS:
 kwargs={'group_permutation':fullperm} if control=='permuted_groups' else {}
 with torch.no_grad():
  a=bank.forward_certified_census_biological_v1(model,full,expected_certificate_sha256=full.certificate_sha256,control=control,readout_chunk_size=4,**kwargs)
  b=run(adapter,model,full,control,fullperm)
 ck('full280 Q184/R96/scored120 '+control+' bitwise outputs and IDs',len(full.inputs[0].parent)==559 and len(full.bank.query_ids)==184 and len(full.bank.reference_ids)==96 and len(a.scored_query_ids)==120 and all(torch.equal(getattr(a,f.name),getattr(b,f.name)) for f in fields(a)))
# Exact restoration on an exception before the original model body consumes its encoder.
model=original_model(torch.float32);adapter=reuse.CertifiedForestReuseV1()
def fail_hook(*args):raise RuntimeError('handmade original model pre-hook exception')
hook=model.register_forward_pre_hook(fail_hook)
try:reject('exception inside model call propagates',lambda:run(adapter,model,cert))
finally:hook.remove()
ck('encoder route restored on exception','forward' not in model.encoder.__dict__ and adapter._routing is False)
with torch.no_grad():run(adapter,model,cert)
ck('route usable after exception',adapter.calls==1)
try:
 with reuse.installed_fit_forward_v1(adapter):raise RuntimeError('handmade scoped exception')
except RuntimeError:pass
ck('fit helper restored on scoped exception',fit.forward_certified_census_biological_v1 is reuse.ORIGINAL_FORWARD)
model=original_model(torch.float32);model.encoder.time_scale=10001.
reject('different encoder time scale rejected',lambda:run(reuse.CertifiedForestReuseV1(),model,cert))
model=original_model(torch.float32);model.encoder.forward=lambda _:None
reject('existing encoder override rejected',lambda:run(reuse.CertifiedForestReuseV1(),model,cert))
# Timing helper numerical-only boundary is checked with an explicit handmade
# admission-hook stub; no genuine root capability, reader or fit is executed.
admit=reuse.admit_reuse_v1;admission_calls=[]
def handmade_admit(*args,**kwargs):admission_calls.append(kwargs);return {'handmade_hook_stub':True}
reuse.admit_reuse_v1=handmade_admit
try:
 model=original_model(torch.float32);before={n:p.detach().clone() for n,p in model.named_parameters()};events=[]
 def prepare():events.append('prepare');return cert
 report=reuse.observed_timing_v1(model,prepare,expected_certificate_sha256=cert.certificate_sha256,base_admission_ref={'handmade':True},reuse_admission_ref={'handmade':True})
 ck('observed timing admits declared profile phase before one preparation',admission_calls[0]['phase']=='observed_profile' and events==['prepare'])
 ck('timing clears synthetic grads without parameter update',all(p.grad is None and torch.equal(p,before[n]) for n,p in model.named_parameters()))
 ck('timing no fit/optimizer or unmeasured speed claim',report['TRAIN_loss_or_fit'] is False and report['speedup_claim'] is False and report['calls']==1 and report['forest_bindings']==1)
 reject('external timing expected SHA cannot derive from returned object',lambda:reuse.observed_timing_v1(model,lambda:cert,expected_certificate_sha256='f'*64,base_admission_ref={'handmade':True},reuse_admission_ref={'handmade':True}))
 ck('wrong external SHA exception restores model grads and route',all(p.grad is None for p in model.parameters()) and 'forward' not in model.encoder.__dict__)
finally:reuse.admit_reuse_v1=admit
for p,sha in cfg['original_source_sha256'].items():ck('original source remains sealed '+p,h(ROOT/p)==sha)
ck('no forbidden actual input/target/profile/job access',forbidden==[])
result={'kind':'census_certified_forest_reuse_v1_fresh_independent_handmade_checks','status':'PASS','checks':checks,'checks_passed':len(checks),'cells':cells,'actual_data_or_targets_opened':False,'original_fit_body_executed':False,'actual_profile_executed':False,'GPU_used':False,'forbidden_access_events':forbidden,'source_sha256':h(ROOT/reuse.SOURCE_PATH),'config_sha256':h(ROOT/reuse.CONFIG_PATH),'observed_timing_admission_hook_stub_only':True,'unmeasured_speed_claim':False}
(OUT/'fresh_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','checks_passed':len(checks),'parity_cells':len(cells),'actual_data_opened':False},sort_keys=True))
