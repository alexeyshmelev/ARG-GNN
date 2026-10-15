"""CPU-only synthetic cache interface checks, never a production loader/input."""
from pathlib import Path
from dataclasses import dataclass,replace
from datetime import datetime,timezone
import hashlib,json,time,torch
from arggnn import marker_resolution_readout_v1 as core
from arggnn import marker_resolution_input_cache_v2 as cache
HERE=Path("/workspace-SR003.nfs2/ARG-GNN/research/reviews/H044_cached_evaluation_root_checks_20261015_v1")
def main():
 torch.set_num_threads(1);torch.manual_seed(6600)
 started=time.monotonic()
 h,q,p,s=952,686,20,12
 geno=torch.randint(-1,2,(s,h),dtype=torch.int8)
 query=torch.arange(q);classes=torch.full((h,),-1,dtype=torch.long)
 classes[q:]=torch.arange(h-q)%4
 positions=torch.tensor([10,90,250,400,550,700,850,1000,1200,1400,1700,1900],dtype=torch.float64)
 intervals=torch.tensor([[0,600],[600,1300],[1300,2000]],dtype=torch.float64)
 centers=torch.arange(50,2000,100,dtype=torch.float64)
 edges=torch.arange(0,2001,100,dtype=torch.float64)
 site_hold=torch.searchsorted(intervals[:,1].contiguous(),positions,right=True)
 readout_hold=torch.searchsorted(intervals[:,1].contiguous(),centers,right=True)
 markers=core.build_marker_evidence_arrays_v1(geno,classes,query,site_hold,readout_hold,
     positions,centers,edges,intervals,allele_cardinality=torch.full((s,),2,dtype=torch.long))
 # Explicit synthetic interface object. It is never sent to a model or the
 # production bind_marker_bank_v1/observation/context/launcher path.
 markers=replace(markers,original_binding={"parent_id":"ap_expanded_val_000","synthetic_cache_unit_fixture":True})
 @dataclass(frozen=True)
 class FixtureObservations:
  genotype:object
  query_column:object
  readout_hold:object
 bank=FixtureObservations(geno,query,readout_hold)
 stamps=core._tensor_stamps_v1(bank)
 bundle=core.H044BoundMarkerBankV1(bank,bank,markers,stamps,stamps,core._tensor_stamps_v1(markers))
 store=cache.EvaluationMarkerBlockStoreV2(maximum_feature_bytes=cache.pair_feature_bytes_v2(p),maximum_parents=1)
 blocks=store.get("ap_expanded_val_000",bundle)
 paired=cache.bind_cached_bundle_v2(bundle,blocks)
 _,view=paired.validate()
 checks={}
 for first,last in blocks.ranges:
  raw=markers.features_for_queries(first,last)
  pooled=markers.pool_within_trees(raw)
  got=view.features_for_queries(first,last)
  assert torch.equal(got,raw)
  assert torch.equal(view.pool_within_trees(got),pooled)
  assert got.is_contiguous() and not torch.is_inference(got)
 checks["all_11_original_query_ranges_bitwise_equal_final_FP32"]=True
 assert store.get("ap_expanded_val_000",bundle) is blocks
 checks["cache_hit_reuses_original_block_objects"]=True
 saved=blocks.localized[0].clone()
 blocks.localized[0].add_(1)
 try:blocks.validate()
 except ValueError:checks["in_place_cache_mutation_rejected"]=True
 else:raise AssertionError("Mutation accepted")
 # Do not attempt a fake restoration of the version-stamped immutable cache.
 del blocks,store,paired,view
 with torch.inference_mode():
  try:cache.build_marker_blocks_v2("ap_expanded_val_000",bundle,remaining_feature_bytes=10**9)
  except ValueError:checks["inference_mode_construction_rejected"]=True
  else:raise AssertionError("Inference-mode cache accepted")
 try:cache.build_marker_blocks_v2("ap_expanded_val_000",bundle,remaining_feature_bytes=1)
 except ValueError:checks["preallocation_byte_cap_rejected"]=True
 else:raise AssertionError("Byte cap ignored")
 receipt=dict(schema="H044_synthetic_cache_CPU_checks_v1",status="passed",checks=checks,
    synthetic=True,production_loader_or_model_run=False,real_data_or_checkpoints_read=False,
    learned_operations_or_optimizer_run=False,GPU_used=False,dtype="final_features_FP32",
    original_observation_prefix_precision_unchanged=True,elapsed_seconds=time.monotonic()-started,
    fixture_dimensions=dict(centers=p,haplotypes=h,queries=q,sites=s),
    source_refs={str(Path(m.__file__).relative_to(HERE.parents[2])):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (core,cache)},
    recorded_utc=datetime.now(timezone.utc).isoformat())
 with (HERE/"result.json").open("x") as f:json.dump(receipt,f,indent=2);f.write("\n")
 print(json.dumps(receipt))
if __name__=="__main__":main()
