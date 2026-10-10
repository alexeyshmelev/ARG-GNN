"""Independent handmade CPU qualification, no data loading or fitting."""
from pathlib import Path
from dataclasses import replace
import json, math, os, traceback
import torch
from torch.func import functional_call
from arggnn.census_cut_exchange_v1 import (ObservedDatedTreeV1, prepare_whole_tree_encoder_inputs_v1,
 build_census_cut_partition_v1, clade_turnover_rho_v1, observed_tmrca_donor_weights_v1,
 prepare_census_exchange_bank_v1, permuted_group_control_v1, census_cut_action_v1,
 census_cut_messages_v1)
from arggnn.census_tree_encoder_v1 import (CensusWholeTreeEncoderV1, CensusTreeGNNV1,
 connected_observed_tree_v1, prepare_connected_whole_tree_inputs_v1,
 retained_observed_features_v1, prepare_retained_observed_forest_v1, encode_whole_tree_contexts_v1)
OUT=Path(/workspace-SR003.nfs2/ARG-GNN/research/reviews/census_cut_encoder_v1_independent)
assert os.environ.get(CUDA_VISIBLE_DEVICES)==
assert os.environ.get(PYTHONDONTWRITEBYTECODE)==1
torch.set_num_threads(1)
Q=torch.tensor([89,3001,404,27,605,909]);R=torch.tensor([777,88,10101]);SCORED=Q[torch.tensor([0,2,4])]
LEFT,RIGHT=.125,.925
BOUNDARY=LEFT+1.5*((RIGHT-LEFT)/5)
INT=torch.tensor([[LEFT,BOUNDARY],[BOUNDARY,.665],[.665,RIGHT]],dtype=torch.float64)
POS=torch.tensor([BOUNDARY,LEFT,.221,BOUNDARY,.664999,.665,.911],dtype=torch.float64)
SIGMAS=(.07,.18)
GROUPS=[[[0,1,2],[3,4],[5]],[[0,3],[1,2,4],[5]],[[0,2],[1,3],[4,5]]]

def tree(g):
 parents=[-1]*16
 for group,members in enumerate(GROUPS[g]):
  for q in members: parents[q]=9+group
 for j in range(3): parents[9+j]=12+j;parents[6+j]=12+j;parents[12+j]=15
 times=[0.,15.,80.,140.,201.,349.,350.,420.,620.]+[[300.,350.,349.8],[220.,330.,349.8],[300.,260.,349.5]][g]+[700.,780.,850.,1000.]
 if g==1: parents[15]=16;parents.append(-1);times.append(1100.)
 return ObservedDatedTreeV1(torch.tensor(parents),torch.tensor(times,dtype=torch.float64),torch.cat((Q,R)),torch.arange(9))
TREES=[tree(g) for g in range(3)]

def features(t,dtype=torch.float64,grad=False):
 n=len(t.parent)
 f=(.17*torch.sin(torch.arange(n*47,dtype=torch.float64).reshape(n,47)*.43)).to(dtype)
 f[:,6]=(torch.log1p(t.node_times)/math.log1p(10000.)).to(dtype)
 return f.requires_grad_(grad)

def bank(dtype=torch.float64,trees=TREES,intervals=INT,positions=POS):
 return prepare_census_exchange_bank_v1(trees,intervals,positions,Q,R,SCORED,num_anchors=5,sigmas=SIGMAS).to(cpu,dtype)

def mr(t,q,r):
 lookup=dict(zip(t.sample_ids.tolist(),t.sample_nodes.tolist()));path=set();v=lookup[q]
 while v>=0: path.add(v);v=int(t.parent[v])
 v=lookup[r]
 while v not in path:v=int(t.parent[v])
 return float(t.node_times[v])

def independent_members(t):
 return [frozenset(q2 for q2 in Q.tolist() if mr(t,q1,q2)<=350.) for q1 in Q.tolist()]

def geometry(ts,dtype):
 members=[independent_members(t) for t in ts]
 projections=[torch.tensor([[float(r in a)/len(a) for r in Q.tolist()] for a in row],dtype=dtype) for row in members]
 rho=[torch.ones(len(Q),dtype=dtype)]
 for previous,current in zip(members[:-1],members[1:]):
  rho.append(torch.tensor([len(a&b)/len(a|b) for a,b in zip(previous,current)],dtype=dtype))
 return projections,rho

def dense_exp(m):
 # Independent dense Taylor polynomial after fixed scaling, not matrix_exp or block averaging.
 z=m/8.;term=torch.eye(len(m),dtype=m.dtype);answer=term
 for k in range(1,24):term=term@z/k;answer=answer+term
 for _ in range(3):answer=answer@answer
 return answer

def path(x,y,ex,da,eta,projections,rho,intervals,control,s,h):
 forward=torch.eye(len(Q),dtype=ex.dtype);lo,hi=min(x,y),max(x,y)
 for rank,(left,right) in enumerate(intervals.tolist()):
  if rank and lo<left<=hi:
   power=0.*eta[s,h] if control==eta0 else eta[s,h]
   forward=torch.diag(rho[rank]**power)@forward
  length=max(0.,min(hi,right)-max(lo,left))
  if length:
   gen=da[s,h]*torch.eye(len(Q),dtype=ex.dtype)+ex[s,h]*(torch.eye(len(Q),dtype=ex.dtype)-projections[rank])
   if control==raw_diagonal:gen=torch.diag(gen.diagonal())
   forward=dense_exp(-length*gen)@forward
 return forward if x>=y else forward.T

def dense_action(b,source,ex,da,eta,control,projections,rho,intervals=INT):
 rows=[]
 for x in b.layout.evaluations.tolist():
  scales=[]
  for s in range(2):
   channels=[]
   for h in range(2):
    channels.append(sum(float(b.layout.weights[a])*path(x,y,ex,da,eta,projections,rho,intervals,control,s,h)@source[a,s,:,h] for a,y in enumerate(b.layout.anchors.tolist())))
   scales.append(torch.stack(channels,1))
  rows.append(torch.stack(scales))
 return torch.stack(rows)

def leaves(dtype):
 shapes=[(5,6,2),(5,3,2),(2,2),(2,2),(2,2),(2,2)]
 return tuple((.27*torch.sin(torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)*(.31+i*.11)+i)).to(dtype).requires_grad_() for i,shape in enumerate(shapes))

def params(v):
 q,r,le,ld,lh,lm=v
 return q.tanh(),r.tanh(),le.exp()*.9,ld.exp()*.6,lh.exp()*.8,lm.softmax(0)

def bundle(b,q,r,ex):
 w=b.donor_weights[b.layout.anchor_graph]
 source=q[:,None]+ex[None,:,None,:]*(w@r[:,None])
 mass=torch.ones_like(source)+ex[None,:,None,:]*w.sum(-1)[...,None]
 return torch.stack((source,mass),-1)

def parts(raw,mix):
 n,m=(raw*mix[None,:,None,:,None]).sum(1).unbind(-1)
 return n,m,n/m

def objective(parts_):
 n,m,z=parts_;w=torch.cos(torch.arange(z.numel(),dtype=z.dtype)*.3).reshape_as(z)
 return (z*w).sum()+.031*m.log().sum()

def close(a,b,dtype=torch.float64,grad=False):
 rt,at=((2e-8,2e-10) if grad else (2e-10,2e-12)) if dtype==torch.float64 else ((2e-3,3e-5) if grad else (3e-4,3e-5))
 torch.testing.assert_close(a,b,rtol=rt,atol=at)

def error(a,b):return float((a-b).abs().max())
METRICS={};FAILURES=[]
def run(name,fn):
 try:METRICS[name]=fn();print(PASS,name,flush=True)
 except Exception as e:
  item={name:name,type:type(e).__name__,message:str(e),traceback:traceback.format_exc()};FAILURES.append(item);print(FAIL,name,item[traceback],flush=True)

def contracts():
 observed=[]
 for t,groups in zip(TREES,GROUPS):
  p=build_census_cut_partition_v1(t,Q,R,SCORED)
  assert set(p.group_members)=={frozenset(Q[g].tolist()) for g in groups}
  assert p.older_reference_ids==tuple(R.tolist())
  expected=torch.tensor([[mr(t,q,r) for r in R.tolist()] for q in Q.tolist()],dtype=torch.float64)
  donor=torch.softmax(-torch.log1p(expected)[None]/math.log1p(10000.)/torch.tensor(SIGMAS,dtype=torch.float64)[:,None,None],-1)
  close(observed_tmrca_donor_weights_v1(t,Q,R,sigmas=SIGMAS),donor)
  observed.append([sorted(a) for a in p.group_members])
 failures=[]
 for slot,age in [(0,350.),(1,350.),(1,351.)]:
  t=TREES[0];times=t.node_times.clone();times[slot]=age;times[9]=max(age,float(times[9]));bad=replace(t,node_times=times)
  try:build_census_cut_partition_v1(bad,Q,R,SCORED)
  except ValueError as e:failures.append(str(e))
  else:raise AssertionError(Equal/older Q admitted)
 assert len(failures)==3
 p=build_census_cut_partition_v1(TREES[0],Q,R,SCORED);times=TREES[0].node_times.clone();times[10]=350.0001
 changed=build_census_cut_partition_v1(replace(TREES[0],node_times=times),Q,R,SCORED)
 assert any(frozenset(Q[[3,4]].tolist())==a for a in p.group_members)
 assert not any(frozenset(Q[[3,4]].tolist())==a for a in changed.group_members)
 return {groups:observed,date_rejections:failures,older_R_dates:[350.,420.,620.],MRCA_equal_350_included:True}
run(cut_dates_donors,contracts)

def numerics(dtype,control):
 b=bank(dtype);ps,rho=geometry(TREES,dtype)
 permutation=torch.tensor([5,2,0,4,1,3])
 if control==permuted_groups:
  b=permuted_group_control_v1(b,permutation)
  ps=[p[permutation][:,permutation] for p in ps];rho=[r[permutation] for r in rho]
 v=leaves(dtype);q,r,ex,da,eta,mix=params(v);source=bundle(b,q,r,ex)
 actual=census_cut_action_v1(b,source,ex,da,eta,control=control,readout_chunk_size=2)
 expected=dense_action(b,source,ex,da,eta,control,ps,rho)
 close(actual,expected,dtype);pa,pe=parts(actual,mix),parts(expected,mix)
 for a,e in zip(pa,pe):close(a,e,dtype)
 assert (pa[1]>0).all() and pa[2].abs().max()<=max(float(q.abs().max()),float(r.abs().max()))+3e-5
 ga=torch.autograd.grad(objective(pa),v,retain_graph=True,allow_unused=True)
 ge=torch.autograd.grad(objective(pe),v,retain_graph=True,allow_unused=True)
 errors={}
 for i,(a,e) in enumerate(zip(ga,ge)):
  if i==4 and control==eta0:assert a is None and (e is None or bool((e==0).all()));continue
  assert a is not None and e is not None;close(a,e,dtype,True);errors[str(i)]=error(a,e)
 wrapped=census_cut_messages_v1(b,q,r,ex,da,eta,mix,control=control,readout_chunk_size=2)
 close(wrapped.targets,pa[2][b.layout.target_output],dtype)
 return {raw_error:error(actual,expected),numerator_error:error(pa[0],pe[0]),mass_error:error(pa[1],pe[1]),normalized_error:error(pa[2],pe[2]),vjp_error:errors,min_mass:float(pa[1].min()),max_normalized_abs:float(pa[2].abs().max())}
for dtype in [torch.float64,torch.float32]:
 for control in [proposed,eta0,permuted_groups,raw_diagonal]:run(str(dtype)+_+control,lambda d=dtype,c=control:numerics(d,c))

def direction():
 b=bank();v=leaves(torch.float64)
 def evaluate(vv):
  q,r,ex,da,eta,mix=params(vv);return objective(parts(census_cut_action_v1(b,bundle(b,q,r,ex),ex,da,eta),mix))
 g=torch.autograd.grad(evaluate(v),v);directions=[torch.cos(torch.arange(x.numel(),dtype=torch.float64).reshape_as(x)*.19+i) for i,x in enumerate(v)]
 analytic=sum((a*d).sum() for a,d in zip(g,directions));step=1e-4
 points={k:evaluate(tuple(x.detach()+k*step*d for x,d in zip(v,directions))) for k in [-2,-1,1,2]}
 fd=(points[-2]-8*points[-1]+8*points[1]-points[2])/(12*step)
 torch.testing.assert_close(analytic,fd,rtol=2e-8,atol=2e-9)
 return {analytic:float(analytic),five_point:float(fd),absolute_error:error(analytic,fd),step:step}
run(exchange_directional_derivative,direction)

def ordering_covariance():
 b=bank();ps,rho=geometry(TREES,torch.float64);q,r,ex,da,eta,mix=params(leaves(torch.float64));source=bundle(b,q,r,ex)
 value=census_cut_action_v1(b,source,ex,da,eta)
 serialized=bank(trees=[TREES[i] for i in [2,0,1]],intervals=INT[torch.tensor([2,0,1])])
 close(census_cut_action_v1(serialized,source,ex,da,eta),value)
 split=bank(trees=[TREES[0],TREES[0],TREES[1],TREES[2]],intervals=torch.tensor([[LEFT,.24],[.24,BOUNDARY],[BOUNDARY,.665],[.665,RIGHT]],dtype=torch.float64))
 close(census_cut_action_v1(split,source,ex,da,eta),value)
 shift=2.13;scaled=1.7
 shifted=bank(intervals=INT+shift,positions=POS+shift)
 close(census_cut_action_v1(shifted,source,ex,da,eta),value)
 enlarged=bank(intervals=LEFT+(INT-LEFT)*scaled,positions=LEFT+(POS-LEFT)*scaled)
 close(census_cut_action_v1(enlarged,source,ex/scaled,da/scaled,eta),scaled*value)
 assert b.target_graph[0]==1 and b.target_graph[3]==1
 impulses=torch.zeros_like(source);impulses[1]=source[1]
 selfvalue=census_cut_action_v1(b,impulses,ex,da,eta)[b.layout.anchor_output[1]]
 close(selfvalue,b.layout.weights[1]*source[1]);assert b.layout.target_output[0]==b.layout.anchor_output[1]
 points=[LEFT,.22,BOUNDARY,.5,.665,.8,.91]
 gram=torch.cat([torch.cat([path(x,y,ex,da,eta,ps,rho,INT,proposed,0,0) for y in points],1) for x in points],0)
 close(gram,gram.T);minimum=float(torch.linalg.eigvalsh(gram).min());assert minimum>=-1e-11
 jump=torch.diag(rho[1]**eta[0,0]);comm=float(torch.linalg.matrix_norm(ps[0]@jump-jump@ps[0]));assert comm>.01
 E=dense_exp(-.18*(da[0,0]*torch.eye(6)+ex[0,0]*(torch.eye(6)-ps[0])))
 noise=torch.eye(6)-E@E.T;boundary_noise=torch.eye(6)-jump@jump.T
 assert torch.linalg.eigvalsh(noise).min()>0 and torch.linalg.eigvalsh(boundary_noise).min()>=0
 wrong=census_cut_action_v1(bank(trees=list(reversed(TREES))),source,ex,da,eta)
 difference=error(wrong,value);assert difference>1e-3
 return {gram_min_eigenvalue:minimum,jump_projection_commutator:comm,reversed_tree_output_difference:difference,self_count_once:True,half_open_collision:True,serialization_subdivision_translation_physical_scaling:True}
run(ordering_covariance,ordering_covariance)

def manual_encoder(encoder,inputs):
 children=[[] for _ in inputs.parent]
 for child,parent in enumerate(inputs.parent.tolist()):
  if parent>=0:children[parent].append(child)
 age=torch.log1p(inputs.node_times)/math.log1p(10000.)
 flags=torch.tensor([[float(p<0),float(not c),float(len(c)>1)] for p,c in zip(inputs.parent.tolist(),children)],dtype=inputs.features.dtype)
 base=encoder.input_projection(torch.cat([inputs.features,age[:,None],flags],1)).tanh()
 edge=(torch.log1p(inputs.branch_duration)/math.log1p(10000.))[:,None]
 up={};down={}
 def ascend(node):
  aggregate=base.new_zeros(encoder.hidden_dim)
  if children[node]:aggregate=torch.stack([encoder.up_message(ascend(c))+encoder.up_edge(edge[c]) for c in children[node]]).mean(0)
  up[node]=encoder.up_update(torch.cat([base[node],aggregate])).tanh();return up[node]
 root=inputs.parent.tolist().index(-1);ascend(root)
 def descend(node):
  p=int(inputs.parent[node]);context=base.new_zeros(encoder.hidden_dim) if p<0 else encoder.down_message(down[p])+encoder.down_edge(edge[node])
  down[node]=encoder.down_update(torch.cat([up[node],context])).tanh()
  for c in children[node]:descend(c)
 descend(root)
 return torch.stack([encoder.output_projection(torch.cat([up[i],down[i]])).tanh() for i in range(len(base))])

def whole_tree():
 fs=[features(t,grad=True) for t in TREES];inputs=[prepare_whole_tree_encoder_inputs_v1(t,f) for t,f in zip(TREES,fs)]
 torch.manual_seed(843);encoder=CensusWholeTreeEncoderV1(6).double();forest=prepare_retained_observed_forest_v1(inputs)
 actual=encoder(forest);expected=[manual_encoder(encoder,i) for i in inputs]
 for a,e in zip(actual,expected):close(a,e)
 ga=torch.autograd.grad(sum(x.square().sum() for x in actual),(*fs,*encoder.parameters()),retain_graph=True)
 ge=torch.autograd.grad(sum(x.square().sum() for x in expected),(*fs,*encoder.parameters()),retain_graph=True)
 for a,e in zip(ga,ge):close(a,e,grad=True)
 assert sum(len(x) for x in forest.upward_levels)==sum(len(i.parent) for i in inputs)
 assert sum(len(x) for x in forest.upward_children)==sum(len(i.parent)-1 for i in inputs)
 minimum=min(float(g[i.native_vertex_index].abs().sum(1).min()) for g,i in zip(ga,inputs));assert minimum>0
 t=TREES[0];f=features(t);n=len(t.parent)
 unused=replace(t,parent=torch.cat([t.parent,torch.tensor([-1,999,17])]),node_times=torch.cat([t.node_times,torch.tensor([float(nan),-2.,float(inf)],dtype=torch.float64)]))
 extra=torch.full((3,47),float(nan),dtype=torch.float64);selected=prepare_connected_whole_tree_inputs_v1(unused,torch.cat([f,extra]));base=prepare_whole_tree_encoder_inputs_v1(t,f)
 torch.testing.assert_close(encoder(base),encoder(selected),rtol=0,atol=0)
 assert selected.native_vertex_index.tolist()==base.native_vertex_index.tolist()
 p=t.parent.clone();p[6]=-1
 try:connected_observed_tree_v1(replace(t,parent=p),f)
 except ValueError:pass
 else:raise AssertionError(Sample-bearing second root admitted)
 parents=torch.cat([t.parent,torch.tensor([9,16])]);parents[0]=17
 unary=replace(t,parent=parents,node_times=torch.cat([t.node_times,torch.tensor([200.,100.],dtype=torch.float64)]))
 uf=torch.cat([f,torch.full((2,47),11.,dtype=torch.float64)]);uf[-2:,6]=torch.log1p(unary.node_times[-2:])/math.log1p(10000.)
 ui=prepare_whole_tree_encoder_inputs_v1(unary,uf)
 torch.testing.assert_close(encoder(base),encoder(ui),rtol=0,atol=0)
 assert float(ui.branch_duration[int(ui.sample_nodes[0])])==300.
 renamed=replace(base,sample_ids=base.sample_ids+45678)
 qa,ra=encode_whole_tree_contexts_v1(encoder,[base],Q,R);qb,rb=encode_whole_tree_contexts_v1(encoder,[renamed],Q+45678,R+45678)
 torch.testing.assert_close(qa,qb,rtol=0,atol=0);torch.testing.assert_close(ra,rb,rtol=0,atol=0)
 old=base.features.clone();old[8,2]+=2.;qc,_=encode_whole_tree_contexts_v1(encoder,[replace(base,features=old)],Q,R)
 influence=error(qa,qc);assert influence>1e-8
 return {retained_nodes:[len(i.parent) for i in inputs],level_heights:[len(x) for x in forest.upward_levels],minimum_retained_feature_gradient_L1:minimum,max_oracle_output_error:max(error(a,e) for a,e in zip(actual,expected)),max_oracle_vjp_error:max(error(a,e) for a,e in zip(ga,ge)),older_reference_context_influence:influence,irrelevant_invalid_rows_ignored:True,serial_unary_exact:True,sample_ID_rename_exact:True}
run(independent_whole_tree_oracle_union_unary,whole_tree)

def gnn_controls():
 b=bank();torch.manual_seed(931);gnn=CensusTreeGNNV1(6,2,4,2,2,initial_exchange=(.8,1.4),initial_damping=.5).double()
 results={};base_context=None
 for control in [proposed,eta0,permuted_groups,raw_diagonal,local]:
  fs=[features(t,grad=True) for t in TREES];inputs=[prepare_whole_tree_encoder_inputs_v1(t,f) for t,f in zip(TREES,fs)]
  work=permuted_group_control_v1(b,torch.tensor([5,2,0,4,1,3])) if control==permuted_groups else b
  gnn.zero_grad(set_to_none=True);out=gnn(inputs,work,control=control)
  if base_context is None:base_context=out.encoder_query_context.detach()
  else:torch.testing.assert_close(out.encoder_query_context,base_context,rtol=0,atol=0)
  loss=(out.logits*torch.cos(torch.arange(out.logits.numel(),dtype=torch.float64).reshape_as(out.logits)*.11)).sum()+.1*out.logits.square().sum();loss.backward()
  inactive=[];active=[]
  for name,p in gnn.named_parameters():
   skip=(name.endswith(raw_eta) and control in [eta0,local]) or (name.endswith(raw_damping) and control==local)
   if skip:assert p.grad is None;inactive.append(name)
   else:assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0;active.append(name)
  minimum=min(float(f.grad[i.native_vertex_index].abs().sum(1).min()) for f,i in zip(fs,inputs));assert minimum>0
  assert out.scored_query_ids.tolist()==SCORED.tolist()
  results[control]={active_parameter_tensors:len(active),inactive:inactive,min_retained_node_gradient:minimum,logits_shape:list(out.logits.shape)}
 inputs=[prepare_whole_tree_encoder_inputs_v1(t,features(t)) for t in TREES]
 changed=inputs[2].features.clone();changed[8,2]+=3.;modified=[*inputs[:2],replace(inputs[2],features=changed)]
 loc0=gnn(inputs,b,control=local).logits;loc1=gnn(modified,b,control=local).logits;mask=b.target_graph==0
 torch.testing.assert_close(loc0[mask],loc1[mask],rtol=0,atol=0)
 a=gnn(inputs,b).logits;z=gnn(modified,b).logits;difference=error(a[mask],z[mask]);assert difference>1e-9
 results[locality]={local_far_hold_change:error(loc0[mask],loc1[mask]),finite_far_hold_change:difference}
 return results
run(GNN_controls_full_node_gradients_locality,gnn_controls)

def gnn_direction():
 torch.manual_seed(931);gnn=CensusTreeGNNV1(6,2,4,2,2,initial_exchange=(.8,1.4),initial_damping=.5).double();b=bank()
 fs=[features(t,grad=True) for t in TREES];inputs=[prepare_whole_tree_encoder_inputs_v1(t,f) for t,f in zip(TREES,fs)];named=list(gnn.named_parameters())
 def objective(z):return (torch.sin(torch.arange(z.numel(),dtype=torch.float64)*.12).reshape_as(z)*z).sum()+.11*z.square().sum()
 grad=torch.autograd.grad(objective(gnn(inputs,b).logits),(*fs,*(p for _,p in named)))
 dirs=[.2*torch.cos(torch.arange(x.numel(),dtype=torch.float64).reshape_as(x)*.071+i*.3) for i,x in enumerate((*fs,*(p for _,p in named)))]
 for d in dirs[:3]:d[:,6]=0.
 expected=sum((a*d).sum() for a,d in zip(grad,dirs));step=1e-4;vals={}
 for k in [-2,-1,1,2]:
  perturbed=[prepare_whole_tree_encoder_inputs_v1(t,f.detach()+k*step*d) for t,f,d in zip(TREES,fs,dirs[:3])]
  parameters={name:p.detach()+k*step*d for (name,p),d in zip(named,dirs[3:])}
  vals[k]=objective(functional_call(gnn,parameters,(perturbed,b)).logits)
 fd=(vals[-2]-8*vals[-1]+8*vals[1]-vals[2])/(12*step)
 torch.testing.assert_close(expected,fd,rtol=3e-8,atol=2e-9)
 return {analytic:float(expected),five_point:float(fd),absolute_error:error(expected,fd),step:step,active_parameter_tensors:len(named)}
run(whole_GNN_directional_derivative,gnn_direction)

def prequalification_boundary():
 b=bank();torch.manual_seed(931);gnn=CensusTreeGNNV1(6,2,4,2,2,initial_exchange=(.8,1.4),initial_damping=.5).double()
 changed=[]
 for t in TREES:
  p=t.parent.clone();p[6],p[7]=t.parent[7],t.parent[6];changed.append(replace(t,parent=p))
 inputs=[prepare_whole_tree_encoder_inputs_v1(t,features(t)) for t in changed]
 stale=gnn(inputs,b).logits
 corrected=bank(trees=changed);fresh=gnn(inputs,corrected).logits
 donor_difference=error(b.donor_weights,corrected.donor_weights);assert donor_difference>.01
 output_difference=error(stale,fresh);assert output_difference>1e-9
 changed_dates=[]
 for t in TREES:
  times=t.node_times.clone();times[6]+=1.;changed_dates.append(replace(t,node_times=times))
 assert bool(torch.isfinite(gnn([prepare_whole_tree_encoder_inputs_v1(t,features(t)) for t in changed_dates],b).logits).all())
 base_inputs=[prepare_whole_tree_encoder_inputs_v1(t,features(t)) for t in TREES]
 marker=base_inputs[0].features.clone();marker[:,0]=torch.arange(len(marker),dtype=torch.float64)%4
 marker_output=gnn([replace(base_inputs[0],features=marker),*base_inputs[1:]],b).logits
 assert error(marker_output,gnn(base_inputs,b).logits)>0
 return {stale_donor_bank_accepted:True,max_donor_error:donor_difference,stale_vs_corrected_logits_error:output_difference,uniformly_changed_reference_dates_accepted:True,arbitrary_semantic_F47_marker_accepted:True,status:Expected exposed caller-prequalification boundary, future integration must bind donor and F47 provenance before fitting}
run(prequalification_boundary_exposure,prequalification_boundary)
report={schema:h021_independent_handmade_CPU_metrics,torch_version:torch.__version__,checks:METRICS,failures:FAILURES,passed_checks:len(METRICS),failed_checks:len(FAILURES),actions:{actual_arrays:False,stochastic_simulation:False,native_inference:False,model_fit:False,CUDA_work:False,packages_installed:False,author_sources_edited:False}}
path_=OUT/independent_metrics_attempt_01.json
assert not path_.exists();path_.write_text(json.dumps(report,indent=2)+n)
print(json.dumps({passed:len(METRICS),failed:len(FAILURES)}))
raise SystemExit(bool(FAILURES))
