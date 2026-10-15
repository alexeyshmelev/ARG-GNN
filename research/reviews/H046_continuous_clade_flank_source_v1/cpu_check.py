"""Focused CPU/FP32 H046 checks. No production inputs or execution entry point."""
from pathlib import Path
from dataclasses import replace
import ast,hashlib,json,math,os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
import numpy as np
import torch
from arggnn.continuous_clade_flank_gnn_v1 import (
    ContinuousCladeFlankGNNV1,LocalReferencePrefixV1,segmented_flank_integrals_v1,
    prepare_continuous_clade_bank_v1,make_matched_flank_family_v1,
    build_clade_flank_arrays_v1,ContinuousCladeFlanksV1,logical_flank_bounds_v1,
    MODEL,READOUT,
)
from arggnn.marker_resolution_readout_v1 import (
    MarkerCoordinatesV1,prepare_marker_evidence_v1,bind_marker_bank_v1,
    tensor_identity_v1,ORIGINAL_LOADER_REF,build_marker_evidence_arrays_v1,
    original_grid_span_nll_v1,
)
from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagControlGNNV1,build_clade_flag_arrays_v1
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
torch.set_num_threads(1);torch.manual_seed(101)
checks=[]
def check(name,ok):
    if not bool(ok):raise AssertionError(name)
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError):checks.append(name)
    else:raise AssertionError("Expected rejection: "+name)

def literal_scan(f,w,starts,ends,tau):
    left=[];state=f.new_zeros(5)
    for i in range(len(f)):
        if starts[i]:state=f.new_zeros(5)
        left.append(state)
        a=torch.exp(-w[i]/tau);state=a*state+(-torch.expm1(-w[i]/tau))*torch.cat((f[i],f.new_ones(1)))
    right=[None]*len(f);state=f.new_zeros(5)
    for i in reversed(range(len(f))):
        if ends[i]:state=f.new_zeros(5)
        right[i]=state
        a=torch.exp(-w[i]/tau);state=a*state+(-torch.expm1(-w[i]/tau))*torch.cat((f[i],f.new_ones(1)))
    return torch.stack(left),torch.stack(right)
f=torch.randn(9,4,dtype=torch.float32,requires_grad=True)
w=torch.tensor([1,2,5,3,1,4,2,7,6],dtype=torch.float32)
starts=torch.tensor([1,0,0,1,1,0,0,0,1],dtype=torch.bool)
ends=torch.tensor([0,0,1,1,0,0,0,1,1],dtype=torch.bool)
tau=torch.tensor(2.5,dtype=torch.float32,requires_grad=True)
actual=segmented_flank_integrals_v1(f,w,starts,ends,tau,checkpoint_scan=False)
reference=literal_scan(f,w,starts,ends,tau)
check("two_segmented_scans_equal_literal_FP32",all(torch.allclose(a,b,atol=3e-7,rtol=2e-6) for a,b in zip(actual,reference)))
g1=torch.autograd.grad(sum(x.square().sum() for x in actual),(f,tau),retain_graph=True)
g2=torch.autograd.grad(sum(x.square().sum() for x in reference),(f,tau),retain_graph=True)
check("scan_state_and_tau_gradients_equal_literal",all(torch.allclose(a,b,atol=3e-6,rtol=3e-5) for a,b in zip(g1,g2)))
delta=.002
def value(t):return sum(x.square().sum() for x in segmented_flank_integrals_v1(f.detach(),w,starts,ends,t,checkpoint_scan=False))
finite=(value(tau.detach()+delta)-value(tau.detach()-delta))/(2*delta)
check("tau_FP32_finite_difference",torch.allclose(g1[1],finite,atol=.002,rtol=.004))
checkpointed=segmented_flank_integrals_v1(f,w,starts,ends,tau,checkpoint_scan=True)
g3=torch.autograd.grad(sum(x.square().sum() for x in checkpointed),(f,tau),retain_graph=True)
check("checkpointed_scan_same_values_and_gradients",all(torch.equal(a,b) for a,b in zip(actual,checkpointed))
      and all(torch.allclose(a,b,atol=1e-6) for a,b in zip(g1,g3)))
check("singleton_episodes_exactly_zero",all(torch.count_nonzero(x[[3,8]])==0 for x in actual))
# Split a non-focal integration cell while preserving the original focal support.
ff=torch.tensor([[1.,2,3,4],[4.,2,1,3],[2.,3,1,5]],dtype=torch.float32,requires_grad=True)
ww=torch.tensor([2.,3.,4.],dtype=torch.float32)
aa=torch.tensor([1,0,0],dtype=torch.bool);zz=torch.tensor([0,0,1],dtype=torch.bool)
tt=torch.tensor(4.,dtype=torch.float32,requires_grad=True)
la,ra=segmented_flank_integrals_v1(ff,ww,aa,zz,tt,checkpoint_scan=False)
splitf=torch.cat((ff[:1],ff[:1],ff[1:]))
splitw=torch.tensor([.5,1.5,3.,4.],dtype=torch.float32)
lb,rb=segmented_flank_integrals_v1(splitf,splitw,torch.tensor([1,0,0,0],dtype=torch.bool),
    torch.tensor([0,0,0,1],dtype=torch.bool),tt,checkpoint_scan=False)
check("fixed_focal_support_neighbor_refinement_invariant",torch.allclose(la[1],lb[2],atol=1e-7)
      and torch.allclose(ra[1],rb[2],atol=1e-7))
ga=torch.autograd.grad((la[1]+ra[1]).sum(),(ff,tt),retain_graph=True)
gb=torch.autograd.grad((lb[2]+rb[2]).sum(),(ff,tt),retain_graph=True)
check("fixed_support_refinement_gradients_invariant",all(torch.allclose(a,b,atol=3e-7) for a,b in zip(ga,gb)))
check("strict_focal_state_has_no_context_gradient",torch.count_nonzero(ga[0][1])==0)
check("flank_state_gradient_reaches_other_holds",ga[0][0].abs().sum()>0 and ga[0][2].abs().sum()>0)
check("physical_distance_attenuation_retained",torch.all(la[1]*torch.exp(torch.tensor(-1./4.)) != la[1]))

fixture=ast.parse((ROOT/"research/reviews/H045_clade_episode_source_v1/cpu_synthetic_check.py").read_text())
exec(compile(ast.Module(body=[n for n in fixture.body if isinstance(n,ast.FunctionDef) and n.name in
    ("forest","synthetic_bank")],type_ignores=[]),"<synthetic_fixture>","exec"),globals())
bank=synthetic_bank(("A","A","A"))
reference_class=bank.reference_class.clone()
reference_class[686:]=torch.arange(952-686)%4
genotype=bank.genotype.clone()
genotype[:,:686]=(torch.arange(6)%2)[:,None].to(genotype.dtype)
genotype[:,686:]=(reference_class[686:]%2)[None,:].to(genotype.dtype)
bank=replace(bank,spans=torch.full((3,),200.,dtype=torch.float32),readout_hold=torch.tensor([0,0,1,1,2,2]),
             reference_class=reference_class,genotype=genotype)
check("full952_synthetic_bank_valid",validate_inputs_v1(bank,expected_haplotypes=952)["queries"]==686)
positions=torch.tensor([25,175,225,375,425,575],dtype=torch.float64)
centers=torch.arange(50,600,100,dtype=torch.float64)
edges=torch.arange(0,601,100,dtype=torch.float64)
intervals=torch.tensor([[0,200],[200,400],[400,600]],dtype=torch.float64)
card=torch.full((6,),2,dtype=torch.long)
arrays=dict(genotype=bank.genotype,reference_class=bank.reference_class,query_column=bank.query_column,
    site_hold=bank.site_hold,readout_hold=bank.readout_hold,positions_bp0=positions,centers_bp=centers,
    bin_edges_bp=edges,intervals_bp=intervals,allele_cardinality=card)
binding=dict(schema="H044_original_observed_marker_coordinate_binding_v1",
    original_loader_source_ref=ORIGINAL_LOADER_REF,
    original_observed_entry_ref=dict(path="synthetic/observed.json",sha256="0"*64),
    parent_id="ap_expanded_train_000",support_rule="original_allele_cardinality",
    array_identities={k:tensor_identity_v1(v) for k,v in arrays.items()})
sidecar=MarkerCoordinatesV1(positions,centers,edges,intervals,card,binding)
markers=prepare_marker_evidence_v1(bank,sidecar)
marker_bundle=bind_marker_bank_v1(bank,markers)
bundle,meta=prepare_continuous_clade_bank_v1(marker_bundle,intervals)
flags,_=build_clade_flag_arrays_v1(bank.parent.numpy(),bank.sample_index.numpy())
check("complete_H042_flags_preserved",np.array_equal(flags,bundle.flanks.flags.node_flags.numpy()))
check("deepest_eligible_clade_query_mapping",bundle.flanks.node_member[bank.sample_index[:,0]].min()>=0
      and bool((bundle.flanks.node_member[bank.sample_index[:,2:]]==-1).all()))
check("payload_resource_formula_exact",bundle.flanks.tensor_bytes()==6*len(bank.parent)+14*meta["member_count"]
      +4*meta["episode_count"]+4*len(bank.spans)+40*len(centers))
prefix=LocalReferencePrefixV1(markers,0,2)
local=prefix.features(bundle.flanks.window_index)
check("local_contrast_gate_nonzero",bool((local[...,9]>0).all()) and local.dtype==torch.float32)
# Long observation sequence exercises both sides of a1024-site block boundary.
s=2100
g=torch.stack((torch.arange(s)%2,(torch.arange(s)//3)%2,torch.arange(s)%2,
    torch.zeros(s,dtype=torch.long),torch.ones(s,dtype=torch.long),torch.zeros(s,dtype=torch.long),
    torch.ones(s,dtype=torch.long),torch.full((s,),3,dtype=torch.long)),-1).to(torch.int16)
g[1000,0]=-1;g[1025,3]=-1
mk=build_marker_evidence_arrays_v1(g,torch.tensor([-1,-1,-1,0,1,2,3,-1]),torch.tensor([0,1,7]),
    torch.zeros(s,dtype=torch.long),torch.zeros(22,dtype=torch.long),torch.arange(s,dtype=torch.float64),
    torch.arange(50,2200,100,dtype=torch.float64),torch.arange(0,2201,100,dtype=torch.float64),
    torch.tensor([[0.,2200.]],dtype=torch.float64),allele_cardinality=torch.full((s,),4,dtype=torch.long))
pref=LocalReferencePrefixV1(mk,0,3)
li=torch.tensor([0,1000,1020,1024,2000,2100]);ri=torch.tensor([1,1100,1120,1030,2100,2100])
got=pref.ranges(li,ri)
direct=[]
for l,r in zip(li,ri):
    q=mk.genotype[l:r][:,mk.query_column].long()
    active=(q>=0)&mk.reference_called[l:r].all(-1)[:,None]
    ev=mk.log_evidence[l:r].gather(2,q.clamp_min(0)[:,None,:].expand(-1,4,-1)).permute(0,2,1)
    ev=(ev-ev.mean(-1,keepdim=True)).masked_fill(~active[:,:,None],0.)
    direct.append(torch.cat((ev.sum(0),active.float().sum(0)[:,None]),-1))
check("blocked_FP32_ranges_match_direct_multiallelic_missingness",
    torch.allclose(got,torch.stack(direct),atol=1e-4,rtol=1e-5))
check("empty_range_neutral",torch.count_nonzero(got[-1])==0)
check("FP32_prefix_tables",pref.prefix.dtype==torch.float32)

base=AnonymousCladeFlagControlGNNV1()
with torch.no_grad():base.clade_coupling.O.weight.normal_(0,.03)
payload=dict(schema="anonymous_clade_flag_checkpoint_v1",hypothesis_id="H042",arm="exact_clade_flags",
    pass_index=30,additional_optimizer_steps=960,parameter_count=30788,seed=101,teacher_used=False,
    model_kwargs=MODEL,model_state={k:v.clone() for k,v in base.state_dict().items()})
models,optimizers,proof=make_matched_flank_family_v1(payload,**READOUT)
check("377_added_31165_total",all(sum(p.numel() for p in m.parameters())==31165 for m in models.values()))
check("tau_initial5000_FP32",all(m.tau.dtype==torch.float32 and abs(m.tau.item()-5000)<.002 for m in models.values()))
check("identical_full_initial_states_independent_storage",all(
    torch.equal(a,b) and a.data_ptr()!=b.data_ptr() for a,b in
    zip(models["foreign_flank"].parameters(),models["own_state"].parameters())))
check("independent_reset_AdamW",all(not o.state for o in optimizers.values())
      and optimizers["foreign_flank"] is not optimizers["own_state"])
check("all_complete_H042_learned_weights_preserved",all(torch.equal(m.backbone.state_dict()[k],v)
    for m in models.values() for k,v in payload["model_state"].items()))
with torch.no_grad():
    wanted=base(bank,flags=bundle.flanks.flags)
    outputs={a:m(bundle) for a,m in models.items()}
check("exact_initial_H042_original_grid_identity",all(torch.equal(v,wanted) for v in outputs.values()))
model=models["foreign_flank"]
with torch.no_grad():model.O.weight.normal_(0,.08)
up_calls=[];down_calls=[]
h1=model.backbone.pyg_upward.register_forward_hook(lambda *args:up_calls.append(1))
h2=model.backbone.pyg_downward.register_forward_hook(lambda *args:down_calls.append(1))
logits=model(bundle)
mass=torch.zeros_like(logits);mass[:,:,0]=100.
loss=original_grid_span_nll_v1(logits,mass,markers.bin_widths_bp)
loss.backward();h1.remove();h2.remove()
check("full_grid_finite_FP32_loss_gradient",logits.shape==(6,686,4) and logits.dtype==torch.float32
      and bool(torch.isfinite(loss)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
check("actual_two_complete_PyG_sweeps",len(up_calls)==2*sum(bool(x.numel()) for x in bank.upward_children)
      and len(down_calls)==2*len(bank.downward_levels))
check("body_SNP_BiGRU_and_context_trainable",all(next(module.parameters()).grad.abs().sum()>0
    for module in (model.backbone.site_mlp,model.backbone.genomic_gru,model.backbone.up_message,model.A,model.H,model.O)))
check("TRAIN_scale_gradient_nonzero",model.theta.grad.abs()>0)
check("bounded_grid_correction",bool((logits-wanted).abs().max()<=1.000001))
# Zero local contrast gate, even with a trained nonzero output, gives exact baseline.
localzero=local.clone();localzero[...,9]=0.
up=torch.randn(3,4,requires_grad=True)
width=torch.full((3,),200.)
left,right=segmented_flank_integrals_v1(up,width,torch.tensor([1,0,0],dtype=torch.bool),
    torch.tensor([0,0,1],dtype=torch.bool),model.tau,checkpoint_scan=False)
smallbase=torch.randn(3,2,4)
smallmap=torch.tensor([0,1,2],dtype=torch.int32)
samples=torch.tensor([[0,0],[1,1],[2,2]])
arg=(smallbase,up,left,right,smallmap,samples,torch.tensor([0,1]),
    torch.tensor([0,0,1,1,2,2]),bundle.flanks.distance_left,bundle.flanks.distance_right,localzero,model.tau)
read=model._readout_block(*arg)
check("zero_local_gate_exactly_no_correction",torch.equal(read,smallbase[torch.tensor([0,0,1,1,2,2])]))
localtest=local.detach()
grads={}
for arm in ("foreign_flank","own_state"):
    model.arm=arm
    read=model._readout_block(*(*arg[:10],localtest,arg[-1]))
    grads[arm]=torch.autograd.grad(read[2].sum(),up,retain_graph=True)[0]
check("foreign_readout_cross_hold_gradient",grads["foreign_flank"][[0,2]].abs().sum()>0)
check("own_readout_zero_foreign_state_gradient",torch.count_nonzero(grads["own_state"][[0,2]])==0)
model.arm="foreign_flank"
rejects("changed_bound_intervals_rejected",lambda:prepare_continuous_clade_bank_v1(marker_bundle,intervals+1))
bounds=logical_flank_bounds_v1(6,3,len(bank.parent),meta["member_count"],meta["episode_count"],6)
check("logical_tensor_counts_match_payload",bounds["compact_flank_payload_bytes"]==bundle.flanks.tensor_bytes()
      and bounds["GPU_flank_payload_bytes"]==bundle.flanks.device_bytes())
receipt=dict(schema="H046_focused_CPU_FP32_source_checks_v1",status="passed",checks=checks,check_count=len(checks),
    synthetic_only=True,model_device="cpu",dtype="float32",real_data_or_checkpoint_reads=False,
    CUDA_operations=False,source_sha256=hashlib.sha256((ROOT/"src/arggnn/continuous_clade_flank_gnn_v1.py").read_bytes()).hexdigest(),
    check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameters=31165,added_parameters=377,
    invariance_scope="fixed field, original focal physical supports, episode membership and positions; no source-hold or whole-body claim")
print(json.dumps(receipt,indent=2))
if "--write-receipt" in sys.argv:Path(__file__).with_name("cpu_check_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
