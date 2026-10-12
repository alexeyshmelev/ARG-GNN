"""Small old/new acceptance parity and one authorized published TRAIN000 bank timing."""
from dataclasses import fields,replace
import json,time
from pathlib import Path
import numpy as np
import torch
from arggnn import mutation_routed_full_panel_fast_reader_v1 as fast
from arggnn import mutation_routed_whole_tree_gnn_v1 as old
from arggnn import mutation_routed_full_panel_observations_v1 as full


def fixture(h=4):
 if h==4:
  p=np.array([4,4,4,4,5,-1,10,10,10,10,11,-1],dtype=np.int64)
  sample=np.array([[0,1,2,3],[6,7,8,9]],dtype=np.int64);clock=np.array([0,0,0,0,1,2,0,0,0,0,1,2],dtype=np.float32)
  up=(sample.ravel(),np.array([4,10]),np.array([5,11]));ch=(np.empty(0,np.int64),sample.ravel(),np.array([4,10]));rank=(np.empty(0,np.int64),np.repeat([0,1],4),np.array([0,1]));down=(np.array([4,10]),sample.ravel())
  ref=np.full(h,-1,np.int64);query=np.array([0,1],np.int64)
 else:
  p=np.r_[np.full(h,h,dtype=np.int64),-1];sample=np.arange(h,dtype=np.int64)[None];clock=np.r_[np.zeros(h,np.float32),np.float32(2)]
  up=(sample.ravel(),np.array([h]));ch=(np.empty(0,np.int64),sample.ravel());rank=(np.empty(0,np.int64),np.zeros(h,np.int64));down=(sample.ravel(),)
  ref=np.r_[np.full(686,-1,np.int64),np.repeat(np.arange(4,dtype=np.int64),full.PROXY_COUNTS)];query=np.arange(686,dtype=np.int64)
 n=len(p);t=len(sample);roots=p<0;geo=np.ones((n,4),np.float32);geo[roots]=0
 support=np.zeros((n,5),np.float32);support[~roots,3]=1;support[roots,4]=1
 arr=dict(parent=p,node_time=clock,sample_index=sample,upward_levels=up,upward_children=ch,upward_parent_rank=rank,downward_levels=down,
  edge_geometry=geo,branch_support_true=support,branch_support_sham=support.copy(),genotype=np.zeros((2,h),np.int16),
  log_donor_allele_frequency=np.zeros((2,2,8),np.float32),donor_call_mask=np.ones((2,8),bool),panel_allele_frequency=np.full((2,2),.5,np.float32),
  site_entropy=np.ones(2,np.float32),site_hold=np.array([0,t-1],np.int64),spans=np.ones(t,np.float32),readout_hold=np.arange(t,dtype=np.int64),
  query_column=query,reference_class=ref,sample_age=np.zeros(h,np.float32))
 return old.MutationRoutedWholeTreeInputsV1.from_arrays(arr)


def edit(bank,name,index,value):
 tensor=getattr(bank,name).clone();tensor[index]=value;return replace(bank,**{name:tensor})


def accepts(call,bank,h):
 try:result=call(bank,expected_haplotypes=h);return True,result
 except (ValueError,TypeError,IndexError,RuntimeError):return False,None


def main():
 torch.set_num_threads(1);fast._source_guard();base=fixture()
 cases={'valid':base,'equal_rounded_clocks':edit(base,'node_time',[4,10],0),
  'parent_self':edit(base,'parent',0,0),'reversed_clock':edit(base,'node_time',0,3),
  'nonpositive_duration':edit(base,'edge_geometry',(0,2),0),'negative_parent':edit(base,'parent',0,-2),
  'wrong_root_count':edit(base,'parent',5,11),'duplicate_samples':edit(base,'sample_index',(0,0),1),
  'sampled_interior':edit(base,'sample_index',(0,0),4),'persistent_age_changed':edit(base,'sample_age',0,1),
  'invalid_genotype':edit(base,'genotype',(0,0),2),'bad_called_donor':edit(base,'log_donor_allele_frequency',(0,0,0),float('nan')),
  'bad_called_panel':edit(base,'panel_allele_frequency',(0,0),0),'bad_query_label':edit(base,'reference_class',0,0),
  'duplicate_query':edit(base,'query_column',1,0),'unordered_site_hold':edit(base,'site_hold',slice(None),torch.tensor([1,0])),
  'root_geometry':edit(base,'edge_geometry',(5,0),1),'root_support':edit(base,'branch_support_true',(5,4),0),
  'empty_nonzero_support':edit(base,'branch_support_true',(0,4),1),'support_above_one':edit(base,'branch_support_true',(0,0),2),
  'bad_empty_flag':edit(base,'branch_support_sham',(0,4),.5),'nonfinite_geometry':edit(base,'edge_geometry',(0,0),float('inf')),
  'downward_order':replace(base,downward_levels=tuple(reversed(base.downward_levels))),
  'upward_order':replace(base,upward_levels=tuple(reversed(base.upward_levels)),upward_children=tuple(reversed(base.upward_children)),upward_parent_rank=tuple(reversed(base.upward_parent_rank))),
  'missing_edge':replace(base,upward_children=(base.upward_children[0],base.upward_children[1][:-1],base.upward_children[2]),upward_parent_rank=(base.upward_parent_rank[0],base.upward_parent_rank[1][:-1],base.upward_parent_rank[2])),
  'rank_mismatch':replace(base,upward_parent_rank=(base.upward_parent_rank[0],base.upward_parent_rank[1].roll(1),base.upward_parent_rank[2]))}
 unused=base.log_donor_allele_frequency.clone();unused[:,1,:]=float('nan');cases['unused_allele_nan_allowed']=replace(base,log_donor_allele_frequency=unused)
 allnan=torch.full_like(base.log_donor_allele_frequency,float('nan'));cases['unavailable_donor_nan_allowed']=replace(base,log_donor_allele_frequency=allnan,donor_call_mask=torch.zeros_like(base.donor_call_mask))
 checks=[]
 for name,bank in cases.items():
  before=accepts(old.validate_inputs_v1,bank,4);after=accepts(fast.validate_inputs_v1,bank,4)
  assert before==after,(name,before,after);checks.append(dict(case=name,accepted=after[0]))
 bank=fixture(952);full.validate_full_bank_v1(bank);fast.validate_full_bank_v1(bank)
 bad=edit(bank,'reference_class',686,1)
 for call in (full.validate_full_bank_v1,fast.validate_full_bank_v1):
  try:call(bad)
  except ValueError:pass
  else:raise AssertionError('Wrong full-panel proxy counts accepted')
 # Only this existing published observation bank is read. No truth/model execution.
 path=Path('data/expanded_full_panel_pyg_cache_v4/cache/ap_expanded_train_000.npz');start=time.perf_counter()
 with np.load(path,allow_pickle=False) as archive:
  arrays={}
  for field in fields(old.MutationRoutedWholeTreeInputsV1):
   prefix='model.'+field.name
   if field.name in old._LEVEL_FIELDS:
    keys=sorted((key for key in archive.files if key.startswith(prefix+'.')),key=lambda key:int(key.rsplit('.',1)[1]))
    arrays[field.name]=tuple(archive[key].copy() for key in keys)
   else:arrays[field.name]=archive[prefix].copy()
  scoring={key[len('scoring.'):]:archive[key].copy() for key in archive.files if key.startswith('scoring.')}
  wanted=set(archive.files)
 bank=old.MutationRoutedWholeTreeInputsV1.from_arrays(arrays);decode_seconds=time.perf_counter()-start
 start=time.perf_counter();summary=fast.validate_inputs_v1(bank,expected_haplotypes=952);fast.validate_full_bank_v1(bank)
 validation_seconds=time.perf_counter()-start
 start=time.perf_counter();payload,record=fast.encode_parent_v1(dict(model_inputs=bank,scoring=scoring,observations_receipt={}),expected_haplotypes=952)
 compression_seconds=time.perf_counter()-start
 import io
 with np.load(io.BytesIO(payload),allow_pickle=False) as rebuilt:
  assert set(rebuilt.files)==wanted
  for name,value in arrays.items():
   if isinstance(value,tuple):
    pairs=[('model.'+name+'.'+str(i),array) for i,array in enumerate(value)]
   else:pairs=[('model.'+name,value)]
   for key,array in pairs:
    assert rebuilt[key].dtype==array.dtype and rebuilt[key].shape==array.shape
    assert np.array_equal(rebuilt[key],array,equal_nan=array.dtype.kind in 'fc'),key
  for name,array in scoring.items():
   assert rebuilt['scoring.'+name].dtype==array.dtype
   assert np.array_equal(rebuilt['scoring.'+name],array,equal_nan=array.dtype.kind in 'fc'),name
 result=dict(status='PASS_small_malformed_parity_and_one_published_bank',source_ref={'path':fast.SOURCE,'sha256':__import__('hashlib').sha256((fast.ROOT/fast.SOURCE).read_bytes()).hexdigest()},
  small_parity_cases=checks,full952_proxy_contract_parity=True,published_bank_path=str(path),compressed_bytes=path.stat().st_size,
  published_bank_summary=summary,decode_seconds=decode_seconds,two_fast_validation_calls_seconds=validation_seconds,fast_original_encoder_seconds=compression_seconds,
  actual_published_array_roundtrip_parity=True,roundtrip_compressed_bytes=len(payload),
  actual_parent_banks_read=1,actual_model_fits=0,GPU_used=False,admitted_sources_modified=False)
 Path('research/reviews/full_panel_fast_reader_v1_author/receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
