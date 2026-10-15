"""One bounded synthetic CPU FP32 algebra/shape check, no actual corpus/model."""
from pathlib import Path
import importlib.util,sys,math,json,hashlib
import torch
torch.set_num_threads(1)
path=Path("src/arggnn/marker_resolution_readout_v1.py")
spec=importlib.util.spec_from_file_location("_H044_synthetic_only",path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
g=torch.tensor([[0,1,1,0,1,-1],[1,1,0,-1,0,-1],
    [0,2,1,2,2,-1],[3,0,2,1,3,-1],[0,0,1,1,-1,-1]],dtype=torch.int8)
classes=torch.tensor([0,0,1,2,-1,-1])
query=torch.tensor([4,5])
pos=torch.tensor([100.,900.,1700.,3300.,5100.],dtype=torch.float64)
centers=50.+100.*torch.arange(60,dtype=torch.float64)
edges=100.*torch.arange(61,dtype=torch.float64)
intervals=torch.tensor([[0.,3000.],[3000.,6000.]],dtype=torch.float64)
site_hold=torch.searchsorted(intervals[:,1].contiguous(),pos,right=True)
readout=torch.searchsorted(intervals[:,1].contiguous(),centers,right=True)
card=torch.tensor([2,2,3,4,2])
data=m.build_marker_evidence_arrays_v1(g,classes,query,site_hold,readout,pos,centers,edges,intervals,
    allele_cardinality=card)
feature=data.features_for_queries(0,2)
assert feature.dtype==torch.float32 and feature.shape==(60,2,4,6)
expected=torch.zeros_like(feature)
for pi,x in enumerate(centers.tolist()):
 for qi,column in enumerate(query.tolist()):
  for c in range(4):
   for ri,radius in enumerate(m.RADII_BP):
    values=[]
    for si,location in enumerate(pos.tolist()):
     if abs(location-x)>radius or int(g[si,column])<0:continue
     refs=[int(g[si,j]) for j in range(g.shape[1]) if int(classes[j])==c and int(g[si,j])>=0]
     if not refs:continue
     allele=int(g[si,column]);a=int(card[si])
     values.append(math.log((refs.count(allele)+.5)/(len(refs)+.5*a))+math.log(a))
    expected[pi,qi,c,ri]=sum(values)/len(values) if values else 0.
    expected[pi,qi,c,ri+3]=math.log1p(len(values))
torch.testing.assert_close(feature,expected,rtol=1e-5,atol=1e-6)
assert not feature[:,1].count_nonzero()                 # missing queries zero/countzero
assert not feature[:,:,3].count_nonzero()               # absent reference class zero/countzero
assert not torch.equal(feature[0,0],feature[29,0])       # within-hold evidence can change
pooled=data.pool_within_trees(feature)
for hold in range(2):
 rows=pooled[readout==hold]
 torch.testing.assert_close(rows,rows[:1].expand_as(rows),rtol=0,atol=0)
torch.testing.assert_close((pooled*data.bin_widths_bp[:,None,None,None]).sum(0),
    (feature*data.bin_widths_bp[:,None,None,None]).sum(0),rtol=1e-5,atol=1e-3)
derived=m.build_marker_evidence_arrays_v1(g,classes,query,site_hold,readout,pos,centers,edges,intervals)
assert derived.support_rule=="all_observed_panel_alleles"
torch.testing.assert_close(derived.features_for_queries(0,2),feature,rtol=1e-5,atol=1e-6)
assert torch.isfinite(data.log_evidence).all() and data.log_evidence.shape==(5,4,4)
# The binary row agrees exactly with Bernoulli p=.75 for class1/query allele1.
torch.testing.assert_close(data.log_evidence[0,1,1],torch.tensor(math.log(.75)+math.log(2.)))
# Native allele3 remains present; no conversion into a binary "other" bucket.
torch.testing.assert_close(data.log_evidence[3,0,3],torch.tensor(math.log(1.5/4.)+math.log(4.)))
empty=m.build_marker_evidence_arrays_v1(g[:0],classes,query,site_hold[:0],readout,pos[:0],
    centers,edges,intervals,allele_cardinality=card[:0])
assert not empty.features_for_queries(0,2).count_nonzero()
torch.manual_seed(101)
decoder=m.SharedMarkerResidualV1()
assert sum(p.numel() for p in decoder.parameters())==640
context=torch.randn(60,2,32,dtype=torch.float32,requires_grad=True)
zero=decoder(context,feature)
assert zero.shape==(60,2,4) and not zero.count_nonzero()
assert not decoder(context,pooled).count_nonzero()
base=torch.randn(60,2,4,dtype=torch.float32)
mass=torch.zeros_like(base);mass[...,1]=100.
loss=m.original_grid_span_nll_v1(base+zero,mass,data.bin_widths_bp)
torch.testing.assert_close(loss,-base.log_softmax(-1)[...,1].mean())
loss.backward()
assert decoder.output.weight.grad.isfinite().all() and decoder.output.weight.grad.count_nonzero()
assert decoder.input.weight.grad is not None and not decoder.input.weight.grad.count_nonzero()
# A fixed nonzero shared output shows the localized/pooled functional distinction.
with torch.no_grad():decoder.output.weight.fill_(.05)
assert not torch.allclose(decoder(context,feature),decoder(context,pooled))
assert not decoder(context,feature)[:,1].count_nonzero()
bounds=m.logical_bounds_v1(5,2,20000)
assert bounds["full_grid_logits_or_probabilities_bytes"]==219520000
assert bounds["all96_sealed_probability_bytes"]==21073920000
assert bounds["one_transient_FP64_target_mass_bytes"]==439040000
# Coordinate/encoding mismatches reject, rather than dropping sites or changing support.
rejected=0
for bad in (torch.tensor([2,2,2,4,2]),torch.tensor([2,2,3,3,2])):
 try:
  m.build_marker_evidence_arrays_v1(g,classes,query,site_hold,readout,pos,centers,edges,intervals,
   allele_cardinality=bad)
 except ValueError:rejected+=1
assert rejected==2
# Isolate cumulative-sum cancellation with bounded synthetic finite evidence.
# This is accumulator algebra, not a fabricated qualified biological bank.
from dataclasses import replace
long_s=100000
long_g=torch.zeros((long_s,6),dtype=torch.int8)
long_g[98000:,4]=-1
long_g[99500,4]=0
long_pos=torch.arange(long_s,dtype=torch.float64)
long_center=torch.tensor([99550.],dtype=torch.float64)
radii=long_pos.new_tensor(m.RADII_BP)
long_evidence=torch.full((long_s,4,4),.5,dtype=torch.float32)
weak=math.log((134.5/267.)*2.) # a valid266-reference binary Jeffreys evidence magnitude
long_evidence[99500,0,0]=weak
late=replace(data,genotype=long_g,query_column=torch.tensor([4]),
 positions_bp0=long_pos,centers_bp=long_center,bin_widths_bp=torch.tensor([100.]),
 readout_hold=torch.tensor([0]),log_evidence=long_evidence,
 reference_called=torch.ones((long_s,4),dtype=torch.bool),
 window_left=torch.searchsorted(long_pos,(long_center[:,None]-radii).contiguous(),right=False),
 window_right=torch.searchsorted(long_pos,(long_center[:,None]+radii).contiguous(),right=True),
 pooled_hold_index=torch.tensor([0]),n_scored_holds=1,n_holds=1)
late_feature=late.features_for_queries(0,1)
torch.testing.assert_close(late_feature[0,0,0,0],torch.tensor(weak),rtol=1e-6,atol=1e-7)
old_values=torch.where(long_g[:,4]>=0,long_evidence[:,0,0],torch.tensor(0.))
old_prefix=torch.cat((torch.zeros(1),old_values.cumsum(0)))
old_late=old_prefix[int(late.window_right[0,0])]-old_prefix[int(late.window_left[0,0])]
assert abs(float(old_late)-weak)>1e-4
assert late_feature.dtype==torch.float32
assert m.logical_bounds_v1(2000000,2,20000)["mixed_evidence_count_prefix_bytes"]==6144003072
# Storage/version stamps detect in-place mutation without any GPU payload hash.
stamps=m._tensor_stamps_v1(data)
data.genotype[0,0]=1-data.genotype[0,0]
assert stamps!=m._tensor_stamps_v1(data)
data.genotype[0,0]=1-data.genotype[0,0]
result=dict(schema="H044_tiny_synthetic_CPU_FP32_checks_v1",status="passed",
 source_ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
 scope=dict(sites=5,haplotypes=6,queries=2,classes=4,scoring_centers=60,native_holds=2,
  native_H042_model_constructed=False,actual_observation_target_checkpoint_runtime_or_GPU_queries=False,
  additional_prefix_algebra_sites=100000,additional_prefix_haplotypes=6),
 checks=["categorical direct-sum parity","Bernoulli binary equivalence","allele3 preserved",
 "missing query and absent class zero","within-tree localized variation",
 "tree-pooled constancy and weighted-mean conservation","explicit observed-support derivation",
 "empty sites","640parameter zero-init matched skip","finite output projection gradient",
 "expected zero first-layer gradient at zero final projection","localized-pooled functional distinction",
 "physical-grid span NLL","exact prediction/target RAM bounds","invalid encodings reject",
 "long-prefix FP64 cancellation correction","immutable tensor mutation stamps"])
out=Path("research/reviews/H044_marker_resolution_source_v1/synthetic_checks_receipt.json")
out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(result,sort_keys=True))
