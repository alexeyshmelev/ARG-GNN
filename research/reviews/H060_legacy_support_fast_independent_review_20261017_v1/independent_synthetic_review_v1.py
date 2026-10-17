"""Independent tiny synthetic source tests only. No production loader or model call."""
from pathlib import Path
from dataclasses import fields
import ast,copy,hashlib,json,sys,unittest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
import torch
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1,_layout,branch_concordance_support_v1
from research.reviews.level_batched_observed_preparation_source_20261017_v1.level_batched_support_v1 import prepare_branch_level_layout_v1,descendant_allele_chunks_v1
from research.reviews.H060_throughput_source_analysis_20261017_v1.legacy_support_exact_v1 import branch_concordance_support_exact_v1
from research.reviews.H060_throughput_source_analysis_20261017_v1 import observations_legacy_exact_v1 as candidate
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import observations_v1 as original
from arggnn.american_c3_gnn_dataset_isolated_v2 import AmericanC3BlockV1
from arggnn.american_c3_gnn_stream_train_isolated_v2 import _feature_identity
from research.reviews.H060_throughput_source_analysis_20261017_v1 import test_legacy_exact_v1 as supplied
HERE=ROOT/"research/reviews/H060_legacy_support_fast_independent_review_20261017_v1"
stats=dict(integer_count_chunks=0,summary_byte_pairs=0,encoder_tensor_pairs=0,feature_digest_pairs=0,representation_domain_notes=[])
digests=hashlib.sha256()
def tree_of(parent,times,sn,sid):return NativeObservedTreeV1(np.array(parent,dtype=np.int64),np.array(times,dtype=np.float64),np.array(sn,dtype=np.int64),np.array(sid,dtype=np.int64))
def topology(rng,h,ancestor=False):
 parent=[-1]*h;times=[0.]*h;active=list(range(h))
 while len(active)>1:
  size=int(rng.integers(2,min(len(active),5)+1));at=sorted(rng.choice(len(active),size,replace=False).tolist())
  children=[active[i]for i in at];active=[u for i,u in enumerate(active)if i not in at]
  p=len(parent);parent.append(-1);times.append(max(times[u]for u in children)+1.)
  for u in children:parent[u]=p
  active.append(p)
  if rng.integers(2):
   u=active.pop();p=len(parent);parent.append(-1);times.append(times[u]+1.);parent[u]=p;active.append(p)
 samples=list(range(h))
 if ancestor:samples.append(active[0])
 ids=np.arange(len(samples),dtype=np.int64)*19+5
 order=rng.permutation(len(samples));return tree_of(parent,times,np.asarray(samples)[order],ids[order])
def integer_oracle(tree,g,k,ids):
 p,_,sn,sid,post,*_=_layout(tree);at={int(v):i for i,v in enumerate(ids)}
 out=[]
 for site in range(len(g)):
  values=g[site,[at[int(x)]for x in sid]];count=np.zeros((len(p),int(k.max())),dtype=np.int64)
  for i,u in enumerate(sn):
   if values[i]>=0:count[u,int(values[i])]=1
  for node in post:
   if p[node]>=0:count[p[node]]+=count[node]
  out.append(count)
 return out
class IndependentReviewTests(unittest.TestCase):
 def test_integer_counts_against_independent_site_postorder_oracle(self):
  rng=np.random.default_rng(2026101701)
  for h in(2,3,6,13):
   for sampled_ancestor in(False,True):
    t=topology(rng,h,sampled_ancestor);ids=rng.permutation(t.sample_id);k=rng.integers(1,5,19,dtype=np.int8)
    g=np.array([rng.integers(-1,int(a),len(ids))for a in k],dtype=np.int8)
    expected=integer_oracle(t,g,k,ids);layout=prepare_branch_level_layout_v1(_layout(t))
    for chunk in(1,2,7,32):
     for start,stop,counts,total,called,proper in descendant_allele_chunks_v1(layout,g,k,ids,marker_chunk_size=chunk,max_count_elements=len(t.parent)*int(k.max())*3):
      self.assertEqual(counts.dtype,torch.int64);self.assertLessEqual(counts.numel(),len(t.parent)*int(k.max())*3)
      for j,site in enumerate(range(start,stop)):
       self.assertTrue(np.array_equal(counts[:,j,:].numpy(),expected[site]))
       want=np.bincount(g[site,g[site]>=0],minlength=int(k.max()))
       self.assertTrue(np.array_equal(total[j].numpy(),want))
       nc=int((g[site]>=0).sum());self.assertEqual(called[j].item(),nc)
       self.assertTrue(np.array_equal(proper[j].numpy(),(want>0)&(want<nc)&(np.arange(int(k.max()))<k[site])&(nc>=2)))
      stats["integer_count_chunks"]+=1
 def test_ordered_summary_byte_identity_more_topologies(self):
  rng=np.random.default_rng(2026101702)
  for h in(2,3,5,9,17):
   for repeat in range(4):
    t=topology(rng,h,repeat%2==1);ids=rng.permutation(t.sample_id);k=rng.integers(1,5,23,dtype=np.int8)
    g=np.array([rng.integers(-1,int(a),len(ids))for a in k],dtype=np.int8)
    expected=branch_concordance_support_v1(t,g,k,ids)
    for chunk in(1,3,8,32):
     got=branch_concordance_support_exact_v1(t,g,k,ids,marker_chunk_size=chunk,max_count_elements=len(t.parent)*int(k.max())*4)
     self.assertEqual(got.dtype,expected.dtype);self.assertEqual(got.shape,expected.shape);self.assertEqual(got.tobytes(),expected.tobytes())
     digests.update(got.tobytes());stats["summary_byte_pairs"]+=1
 def test_empty_missing_monomorphic_unary_ties_and_invalid_domains(self):
  t=tree_of([2,2,3,-1],[0,0,1,2],[0,1],[11,7]);ids=np.array([7,11])
  for g,k in((np.empty((0,2),dtype=np.int8),np.empty(0,dtype=np.int8)),(np.array([[-1,-1],[0,-1],[0,0],[0,1],[1,0]],dtype=np.int8),np.array([4,4,1,2,2],dtype=np.int8))):
   a=branch_concordance_support_v1(t,g,k,ids);b=branch_concordance_support_exact_v1(t,g,k,ids)
   self.assertEqual(a.tobytes(),b.tobytes());stats["summary_byte_pairs"]+=1
  single=tree_of([-1],[0],[0],[37])
  for g,k in((np.empty((0,1),dtype=np.int8),np.empty(0,dtype=np.int8)),(np.array([[0],[-1]],dtype=np.int8),np.array([1,4],dtype=np.int8))):
   self.assertEqual(branch_concordance_support_v1(single,g,k,np.array([37])).tobytes(),branch_concordance_support_exact_v1(single,g,k,np.array([37])).tobytes());stats["summary_byte_pairs"]+=1
  for controls in(dict(max_count_elements=1),dict(marker_chunk_size=0),dict(marker_chunk_size=True),dict(max_count_elements=-1)):
   with self.assertRaises(ValueError):branch_concordance_support_exact_v1(t,np.array([[0,1]],dtype=np.int8),np.array([2],dtype=np.int8),ids,**controls)
  for f in(branch_concordance_support_v1,branch_concordance_support_exact_v1):
   with self.assertRaises(ValueError):f(t,np.array([[0,2]],dtype=np.int8),np.array([2],dtype=np.int8),ids)
   with self.assertRaises(ValueError):f(t,np.array([[0,1]],dtype=np.int8),np.array([2],dtype=np.int8),np.array([7,7]))
 def test_encoder_tensor_bytes_and_digest_connected_isolated_empty(self):
  t=tree_of([4,4,5,5,6,6,-1],[0,0,0,0,1,1,2],[0,1,2,3],[0,1,2,3])
  u=tree_of([4,4,5,5,6,7,7,-1],[0,0,0,0,1,1,2,3],[0,1,2,3],[0,1,2,3])
  iso=tree_of([-1]*4,[0]*4,[3,0,2,1],[0,1,2,3])
  for trees,intervals,g,k,x in [
   ([t,u],[[0,5],[5,10]],[[0,1,2,3],[0,1,-1,0],[0,0,1,1]],[4,2,2],[1.,4.,9.]),
   ([t,iso,u],[[0,5],[5,10],[10,15]],[[0,1,2,3],[0,-1,1,1]],[4,2],[1.,11.]),
   ([t],[[0,5]],[],[],[]),([iso],[[0,5]],[],[],[])]:
   args=(trees,np.array(intervals,dtype=np.float64),np.array(g,dtype=np.int8).reshape(-1,4),np.array(k,dtype=np.int8),np.array(x,dtype=np.float64),np.array([0,1,-1,-1]),np.zeros(4))
   a,ma=original.encode_block(*args,num_classes=3);b,mb=candidate.encode_block(*args,num_classes=3);self.assertEqual(ma,mb)
   for field in fields(a):
    av=getattr(a,field.name);bv=getattr(b,field.name)
    for aa,bb in zip(av if isinstance(av,tuple)else(av,),bv if isinstance(bv,tuple)else(bv,)):
     self.assertEqual(aa.dtype,bb.dtype);self.assertEqual(aa.shape,bb.shape);self.assertEqual(aa.numpy().tobytes(),bb.numpy().tobytes());stats["encoder_tensor_pairs"]+=1
   meta=dict(parent_id="independent_synthetic",block_index=0,start_tree=0,stop_tree=len(trees),sequence_length_bp=intervals[-1][1],complete_parent=True,genomic_geometry=torch.zeros(len(trees),3),focal_rows=torch.tensor([0,1]),focal_hold=torch.tensor([0,len(trees)-1]),focal_sample_column=torch.tensor([2,3]))
   self.assertEqual(_feature_identity(AmericanC3BlockV1(a,**meta)),_feature_identity(AmericanC3BlockV1(b,**meta)));stats["feature_digest_pairs"]+=1
  bad=( [iso],np.array([[0,5]],dtype=np.float64),np.array([[0,1,0,1]],dtype=np.int8),np.array([2],dtype=np.int8),np.array([1.]),np.array([0,1,-1,-1]),np.zeros(4))
  for f in(original.encode_block,candidate.encode_block):
   with self.assertRaises(ValueError):f(*bad,num_classes=3)
 def test_representation_domain_limits_recorded(self):
  t=tree_of([2,2,-1],[0,0,1],[0,1],[0,1]);g=np.array([[0,1],[1,0]],dtype=np.int8);k=np.array([2,2],dtype=np.int8);ids=np.array([0,1])
  for name,gg,kk,ii in [("negative_stride_sites",g[::-1],k[::-1],ids),("unsigned16_calls",g.astype(np.uint16),k,ids)]:
   expected=branch_concordance_support_v1(t,gg,kk,ii)
   try:got=branch_concordance_support_exact_v1(t,gg,kk,ii)
   except (ValueError,TypeError) as exc:stats["representation_domain_notes"].append(dict(case=name,original_accepts=True,candidate_rejects=type(exc).__name__,reason=str(exc)[:400]))
   else:self.assertEqual(got.tobytes(),expected.tobytes())
if __name__=="__main__":
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(IndependentReviewTests),unittest.defaultTestLoader.loadTestsFromTestCase(supplied.ExactLegacyTests)])
 result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(suite)
 receipt=dict(schema="H060_legacy_support_fast_independent_synthetic_receipt_v1",success=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),fixture_summary_sha256=digests.hexdigest(),stats=stats,CUDA_VISIBLE_DEVICES=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),production_data_truth_checkpoints_models_or_profile_read=False,production_speed_or_fit_claim=False)
 (HERE/"synthetic_receipt.v1.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
 print(json.dumps(receipt,sort_keys=True))
 raise SystemExit(0 if result.wasSuccessful()else 1)
