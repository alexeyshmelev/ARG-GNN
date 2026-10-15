"""Focused CPU synthetic algebra/shape/gradient/complexity fixtures only."""
from pathlib import Path
from dataclasses import fields
import importlib.util,json,math,torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_H051_prototype",HERE/"conditional_innovation_v1.py")
core=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=core;spec.loader.exec_module(core)
provenance=dict(schema="H051_observation_only_cut_source_v1",observed_trees_sha256="a"*64,
    builder_sha256="b"*64,targets_or_outcomes_used=False,identity_namespace_sha256="a"*64)
def plan(a,b,c,provenance=provenance):
    return core.make_plan_v1(*(torch.tensor(x,dtype=torch.long) for x in (a,b,c)),provenance=provenance)
def close(a,b,atol=2e-5):return torch.allclose(a,b,rtol=2e-5,atol=atol)
results=[]
def passed(name,ok=True):assert ok,name;results.append(dict(case=name,passed=True))
def rejects(name,call):
    try:call()
    except (ValueError,TypeError):passed(name)
    else:raise AssertionError(name)
def projector(index):
    incidence=torch.nn.functional.one_hot(index).to(torch.float32)
    return incidence/incidence.sum(0)[None,:]@incidence.T
p=plan([0,0,1,1],[0,0,0,0],[0,0,1,1])
old=torch.tensor([[2.],[-2.]],dtype=torch.float32)
triple=core.ConditionalInnovationTransportV1(alpha=1.)
marginal=core.ConditionalInnovationTransportV1(alpha=0.)
parts=triple.components(old,p)
passed("handmade_conditional_innovation",close(parts["innovation"],old) and close(parts["marginal"],old*0))
independent=plan([0,0,1,1],[0]*4,[0,1,0,1])
passed("conditional_independence_zero_for_every_old_basis",close(triple.components(torch.eye(2,dtype=torch.float32),independent)["innovation"],torch.zeros(2,2,dtype=torch.float32)))
generator=torch.Generator(device="cpu").manual_seed(51001)
a=torch.tensor([3,3,3,9,9,9,2,2,2,8,8,8]);b=torch.tensor([1,1,2,1,2,2,1,1,2,2,2,1]);c=torch.tensor([4,5,4,5,5,4,4,5,5,4,5,4])
rp=core.make_plan_v1(a,b,c,provenance=provenance)
x=torch.randn(12,5,dtype=torch.float32,generator=generator)
pab,pb,pbc=(projector(v) for v in (rp.leaf_ab,rp.leaf_b,rp.leaf_bc))
j=(pbc-pb)@pab
got=triple.components(x,rp,basis="leaf")
passed("occupied_result_equals_dense_J",close(got["innovation"][rp.leaf_bc],j@x))
passed("triple_equals_dense_conditional_mean",close(got["triple"][rp.leaf_bc],pbc@pab@x))
passed("marginal_equals_dense_middle_projection",close(got["marginal"][rp.leaf_bc],pb@x))
passed("B_mean_innovation_exactly_zero",close(pb@got["innovation"][rp.leaf_bc],torch.zeros_like(x)))
passed("J_annihilates_B_measurable_inputs",close(j@pb,torch.zeros_like(j)))
passed("J_spectral_contraction",float(torch.linalg.matrix_norm(j,ord=2))<=1+2e-5)
for label,model in [
 ("zero",marginal),("one",triple),("interior",core.ConditionalInnovationTransportV1(alpha=.37)),
 ("support",core.ConditionalInnovationTransportV1(gate="count",prior_count=16.))]:
    value=model(x,rp,basis="leaf")[rp.leaf_bc]
    passed("contraction_"+label,float(value.norm())<=float(x.norm())+2e-5)
    passed("B_mean_preservation_"+label,close(pb@value,pb@x))
before=torch.tensor([[13,12]]*4,dtype=torch.long)
after=torch.tensor([[13,12],[13,12],[13,14],[13,14]],dtype=torch.long)
w=core.EdgeWitnessV1(before,after,("a"*64,))
identity=core.ConditionalInnovationTransportV1(gate="edge_continuity",prior_count=0.)
alpha,receipt=core.edge_continuity_gate_v1(p,w,prior_count=0.)
passed("observed_edge_continuity_gate_and_no_extra_parameters",close(alpha,torch.tensor([.5],dtype=torch.float64)) and sum(z.numel() for z in identity.parameters())==0)
passed("same_cut_plan_different_shared_identity_channel",
    close(identity(old,p,witness=w),old*.5)
    and close(identity(old,p,witness=core.EdgeWitnessV1(before,after+100,("a"*64,))),old*0))
missing=core.EdgeWitnessV1(torch.full_like(before,-1),torch.full_like(after,-1),("a"*64,))
passed("missing_and_root_edges_do_not_fabricate_continuity",close(identity(old,p,witness=missing),old*0))
renamed_provenance=dict(provenance,identity_namespace_sha256="c"*64)
renamed_plan=plan([0,0,1,1],[0]*4,[0,0,1,1],provenance=renamed_provenance)
rename=lambda z:(z*7+3)%101
renamed=core.EdgeWitnessV1(rename(before),rename(after),("c"*64,))
passed("global_node_key_bijection_equivariance",close(identity(old,p,witness=w),identity(old,renamed_plan,witness=renamed)))
passed("no_edge_row_ID_field_for_fragmentation_artifacts",set(f.name for f in fields(core.EdgeWitnessV1))=={"before","after","namespace_sha256"})
perm=torch.tensor([3,0,2,1])
pp=plan([1,0,1,0],[0]*4,[1,0,1,0])
wp=core.EdgeWitnessV1(before[perm],after[perm],("a"*64,))
leaf=torch.tensor([[2.],[2.],[-2.],[-2.]],dtype=torch.float32)
v=identity(leaf,p,basis="leaf",witness=w)[p.leaf_bc]
vp=identity(leaf[perm],pp,basis="leaf",witness=wp)[pp.leaf_bc]
passed("leaf_identity_equivariance_including_identity_sidecar",close(vp,v[perm]))
relabel=plan([102,102,7,7],[25]*4,[8,8,99,99])
passed("anonymous_cut_key_bijection_equivariance",close(triple(leaf,p,basis="leaf")[p.leaf_bc],triple(leaf,relabel,basis="leaf")[relabel.leaf_bc]))
p2=plan([0,1,2],[0,1,1],[4,4,4])
batched=core.batch_plans_v1([p,p2])
s1=torch.randn(p.old_cells,2,3,dtype=torch.float32,generator=generator)
s2=torch.randn(p2.old_cells,2,3,dtype=torch.float32,generator=generator)
out=triple(torch.cat((s1,s2)),batched)
passed("ragged_disjoint_batch_equals_separate",close(out,torch.cat((triple(s1,p),triple(s2,p2)))))
passed("ragged_event_counts_no_padding",batched.event_sizes==p.event_sizes+p2.event_sizes and batched.h==7 and batched.old_cells==p.old_cells+p2.old_cells)
passed("arbitrary_trailing_feature_shape",tuple(out.shape)==(batched.new_cells,2,3))
passed("scalar_feature_state",triple(old[:,0],p).shape==(p.new_cells,))
single=plan([999],[88],[777])
passed("single_leaf_degenerate_J_zero",close(triple.components(torch.ones(1,7,dtype=torch.float32),single)["innovation"],torch.zeros(1,7,dtype=torch.float32)))
saved=[z.clone() for z in (a,b,c,x,before,after)]+[z.clone() for z in rp.tensors()]
triple(x,rp,basis="leaf");identity(old,p,witness=w)
passed("no_caller_or_plan_mutation",all(torch.equal(u,v) for u,v in zip(saved,[a,b,c,x,before,after]+list(rp.tensors()))))
reused=rp.to("cpu")
passed("explicit_device_transfer_preserves_plan",close(triple(x,reused,basis="leaf"),triple(x,rp,basis="leaf")))
gstate=x.clone().requires_grad_()
direction=torch.randn(gstate.shape,dtype=torch.float32,generator=generator)
count_model=core.ConditionalInnovationTransportV1(gate="count",prior_count=3.)
grad_a=torch.autograd.grad(count_model(gstate,rp,basis="leaf").square().sum(),gstate)[0]
epsilon=1e-3
observed=(count_model(x+epsilon*direction,rp,basis="leaf").square().sum()-count_model(x-epsilon*direction,rp,basis="leaf").square().sum())/(2*epsilon)
expected=(grad_a*direction).sum()
passed("FP32_input_directional_finite_difference",torch.allclose(observed,expected,rtol=3e-3,atol=3e-3))
passed("finite_FP32_full_state_gradients",bool(torch.isfinite(grad_a).all()) and bool((grad_a!=0).any()))
pyg=core.ConditionalInnovationTransportV1(alpha=.37,backend="pyg")
torch_model=core.ConditionalInnovationTransportV1(alpha=.37)
sx=x.clone().requires_grad_();sy=x.clone().requires_grad_()
y1=pyg(sx,rp,basis="leaf");y2=torch_model(sy,rp,basis="leaf")
gx=torch.autograd.grad(y1.square().sum(),sx)[0];gy=torch.autograd.grad(y2.square().sum(),sy)[0]
passed("native_PyG_backend_output_and_gradient_equivalence",close(y1,y2) and close(gx,gy))
passed("FP32_feature_outputs_preserved",torch_model(x,rp,basis="leaf").dtype==torch.float32)
z=torch.randn(3,4,dtype=torch.float32,generator=generator);f=torch.randn(3,4,dtype=torch.float32,generator=generator)
whole=core.relax_interval_v1(z,f,100.,500.,distance_unit="bp",scale_unit="bp")
split=core.relax_interval_v1(core.relax_interval_v1(z,f,37.,500.,distance_unit="bp",scale_unit="bp"),f,63.,500.,distance_unit="bp",scale_unit="bp")
passed("fixed_carrier_constant_forcing_subdivision_semigroup",close(whole,split))
passed("zero_distance_no_relaxation",torch.equal(core.relax_interval_v1(z,f,0.,500.,distance_unit="bp",scale_unit="bp"),z))
scale=torch.tensor(500.,dtype=torch.float32,requires_grad=True)
zz=z.clone().requires_grad_();ff=f.clone().requires_grad_()
relaxed=core.relax_interval_v1(zz,ff,100.,scale,distance_unit="bp",scale_unit="bp")
gz,gf,gl=torch.autograd.grad(relaxed.sum(),(zz,ff,scale))
rho=torch.exp(-torch.tensor(.2,dtype=torch.float32))
expected_scale=(rho*100./scale.detach().square()*(z-f)).sum()
passed("FP32_forcing_state_and_length_scale_gradients",close(gz,torch.full_like(z,rho)) and close(gf,torch.full_like(f,1-rho)) and close(gl,expected_scale))
map_bp=torch.tensor([0.,100.,200.,300.],dtype=torch.float64);map_m=torch.tensor([0.,.01,.01,.03],dtype=torch.float64)
intervals=torch.tensor([[0.,300.],[0.,150.],[150.,300.],[100.,200.]],dtype=torch.float64)
d=core.interval_morgan_distances_v1(map_bp,map_m,intervals)
passed("nonuniform_genetic_map_integrated_and_subdivision_additive",close(d,torch.tensor([.03,.01,.02,0.],dtype=torch.float64)) and close(d[0],d[1]+d[2]))
rejects("no_silent_bp_Morgan_unit_mismatch",lambda:core.relax_interval_v1(z,f,1.,1.,distance_unit="morgan",scale_unit="bp"))
rejects("no_silent_centimorgan_unit_guess",lambda:core.relax_interval_v1(z,f,1.,1.,distance_unit="cM",scale_unit="cM"))
rejects("negative_genomic_distance",lambda:core.relax_interval_v1(z,f,-1.,1.,distance_unit="bp",scale_unit="bp"))
rejects("zero_length_scale",lambda:core.relax_interval_v1(z,f,1.,0.,distance_unit="bp",scale_unit="bp"))
rejects("map_extrapolation_not_allowed",lambda:core.interval_morgan_distances_v1(map_bp,map_m,torch.tensor([[-1.,100.]],dtype=torch.float64)))
rejects("nonmonotone_cumulative_map",lambda:core.interval_morgan_distances_v1(map_bp,torch.tensor([0.,.02,.01,.03],dtype=torch.float64),intervals))
rejects("target_or_outcome_provenance_rejected",lambda:plan([0],[0],[0],provenance=dict(provenance,targets_or_outcomes_used=True)))
rejects("extra_target_field_rejected",lambda:plan([0],[0],[0],provenance=dict(provenance,target_tensor="not accepted")))
rejects("foreign_identity_namespace_rejected",lambda:identity(old,p,witness=core.EdgeWitnessV1(before,after,("d"*64,))))
rejects("partial_missing_edge_not_counted_as_valid",lambda:identity(old,p,witness=core.EdgeWitnessV1(torch.tensor([[-1,12]]*4),after,("a"*64,))))
rejects("unavailable_shared_ID_namespace_blocks_identity_arm",lambda:identity(old,plan([0,0,1,1],[0]*4,[0,0,1,1],provenance=dict(provenance,identity_namespace_sha256=None)),witness=w))
rejects("hidden_witness_side_input_in_fixed_arm",lambda:triple(old,p,witness=w))
rejects("state_shape_mismatch",lambda:triple(torch.ones(3,2),p))
rejects("alpha_outside_contraction_range",lambda:core.ConditionalInnovationTransportV1(alpha=1.1))
changed=plan([0,0,1,1],[0]*4,[0,0,1,1]);changed.edge_count.add_(1)
rejects("mutated_immutable_recipe_rejected",changed.validate)
for h in (5,17,65,1024):
    ids=torch.arange(h)
    pp=core.make_plan_v1(ids,ids,ids,provenance=provenance)
    passed("linear_occupied_storage_bound_H"+str(h),pp.tensor_bytes()<=88*h and pp.occupied_triples<=h
        and pp.old_cells<=h and pp.new_cells<=h and all(t.numel()<=h for t in pp.tensors()))
gident=old.clone().requires_grad_()
gi=torch.autograd.grad(identity(gident,p,witness=w).square().sum(),gident)[0]
passed("FP32_identity_gate_state_gradient",close(gi,old*.5))
per_event=core.new_cell_event_distances_v1(batched,[37.,63.])
passed("ragged_per_event_clock_has_no_padding",per_event.shape==(batched.new_cells,) and torch.equal(per_event[:p.new_cells],torch.full((p.new_cells,),37.,dtype=torch.float64)))
forcing=torch.zeros_like(out)
vector_clock=core.relax_interval_v1(out,forcing,per_event,500.,distance_unit="bp",scale_unit="bp")
separate_clock=torch.cat((core.relax_interval_v1(out[:p.new_cells],forcing[:p.new_cells],37.,500.,distance_unit="bp",scale_unit="bp"),
    core.relax_interval_v1(out[p.new_cells:],forcing[p.new_cells:],63.,500.,distance_unit="bp",scale_unit="bp")))
passed("ragged_vector_clock_equals_separate_intervals",close(vector_clock,separate_clock))
rejects("wrong_per_cell_distance_shape",lambda:core.relax_interval_v1(z,f,torch.ones(2),500.,distance_unit="bp",scale_unit="bp"))
print(json.dumps(dict(schema="H051_focused_synthetic_CPU_fixtures_v1",cases=results,cases_passed=len(results),
    data_truth_checkpoint_reads=0,GPU_or_training_jobs=0,registered_or_admitted_sources_changed=0,
    fixtures_only=True,feature_and_autograd_dtype="float32",observed_map_and_count_metadata_dtype="float64",
    tolerances=dict(default_rtol=2e-5,default_atol=2e-5,directional_rtol=3e-3,directional_atol=3e-3)),indent=2,sort_keys=True))
