"""Vectorized preparation validation and unchanged-format full-panel cache reader.

No per-node Torch scalar reads. Original decode/provenance functions run in
isolated namespaces with only the structural-validation callback replaced.
This consumer helper never changes cached bytes or producer source_refs.
"""
from __future__ import annotations
from dataclasses import fields
import hashlib
from pathlib import Path
from types import FunctionType
import numpy as np
import torch
from . import mutation_routed_whole_tree_gnn_v1 as model
from . import mutation_routed_full_panel_cache_v1 as original
from . import mutation_routed_full_panel_observations_v1 as observations

ROOT=original.ROOT
SOURCE='src/arggnn/mutation_routed_full_panel_fast_reader_v1.py'
SOURCE_PINS={
 'src/arggnn/mutation_routed_whole_tree_gnn_v1.py':'254f5155e31f81ae675e2bd08d38c97c9bf130d47e98efa2deea4c7344d9e073',
 'src/arggnn/mutation_routed_full_panel_cache_v1.py':'1c1f1e5222df0e09e83b470d5375b8946cb3a89db1bc55e9d2c293a09037cb38',
 'src/arggnn/mutation_routed_full_panel_observations_v1.py':'a7184be9a735d95b80975f68666d51e9ed0316cc535d9f7b54baafea532a1107',
 'src/arggnn/mutation_routed_pyg_dataset_v1.py':'f7488ae9794c301d47ad66bab9caed5ff87ef09afcf96c5e48c9c00d350706ca',
 'src/arggnn/mutation_routed_observation_cache_v1.py':'f4adf6832b6d80bdf3233083e0aacc87ae27818314ca185021a9f4c8b0e97150'}
_PIN_CACHE=None


def _source_guard():
 global _PIN_CACHE
 before=[]
 for path in SOURCE_PINS:
  p=ROOT/path;s=p.lstat()
  if p.resolve()!=p or not p.is_file() or s.st_nlink!=1:raise ValueError('Canonical singly linked original source required')
  before.append((path,s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
 before=tuple(before)
 if before==_PIN_CACHE:return
 for path,sha in SOURCE_PINS.items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('Original reader/validation source differs from pin')
 after=[]
 for path in SOURCE_PINS:
  s=(ROOT/path).lstat();after.append((path,s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
 if before!=tuple(after):raise ValueError('Original source changed during binding')
 _PIN_CACHE=before


def _numpy(value):return value.detach().resolve_conj().resolve_neg().numpy()
def _cat(levels):return np.concatenate(levels) if levels else np.empty(0,dtype=np.int64)


def validate_inputs_v1(bank,*,expected_haplotypes=280):
 """Same CPU representation acceptance checks using NumPy views and level sweeps."""
 if not isinstance(bank,model.MutationRoutedWholeTreeInputsV1):raise TypeError('Expected the observation-only dataclass')
 for field in fields(bank):
  value=getattr(bank,field.name);tensors=value if field.name in model._LEVEL_FIELDS else (value,)
  if any(not isinstance(x,torch.Tensor) or x.device.type!='cpu' for x in tensors):raise ValueError('Preparation validation requires CPU tensors')
 if any(getattr(bank,name).dtype!=torch.float32 for name in model._FLOAT_FIELDS):raise ValueError('Prototype floating observations must be FP32')
 if any(getattr(bank,name).dtype!=torch.long for name in model._INDEX_FIELDS):raise ValueError('Topology/identity lookup tensors must be int64')
 if bank.genotype.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64):raise ValueError('Categorical integer genotypes required')
 if bank.donor_call_mask.dtype!=torch.bool:raise ValueError('Call mask must be boolean')
 values={field.name:tuple(_numpy(x) for x in getattr(bank,field.name)) if field.name in model._LEVEL_FIELDS
  else _numpy(getattr(bank,field.name)) for field in fields(bank)}
 p=values['parent'];clock=values['node_time'];sample=values['sample_index'];geo=values['edge_geometry'];gen=values['genotype']
 n=p.size
 if p.ndim!=1 or not n or clock.shape!=(n,):raise ValueError('Nonempty node geometry required')
 if sample.ndim!=2:raise ValueError('sample_index must be [T,H]')
 t,h=sample.shape
 if not t or h!=expected_haplotypes:raise ValueError('Every original interval must retain the expected full sampled panel')
 if (geo.shape!=(n,4) or values['branch_support_true'].shape!=(n,5) or values['branch_support_sham'].shape!=(n,5)
  or values['spans'].shape!=(t,) or values['reference_class'].shape!=(h,) or values['sample_age'].shape!=(h,)):
  raise ValueError('Geometry/support/sample field shapes differ')
 if gen.ndim!=2 or gen.shape[1]!=h:raise ValueError('Genotype columns must follow persistent sample order')
 s=gen.shape[0];donor=values['log_donor_allele_frequency'];panel=values['panel_allele_frequency'];mask=values['donor_call_mask']
 if donor.ndim!=3:raise ValueError('Donor log frequencies must be [S,A,8]')
 a=donor.shape[1]
 if (not 1<=a<=4 or donor.shape!=(s,a,8) or panel.shape!=(s,a) or mask.shape!=(s,8)
  or values['site_entropy'].shape!=(s,) or values['site_hold'].shape!=(s,)):
  raise ValueError('All original categorical site arrays must agree')
 readout=values['readout_hold'];query=values['query_column'];reference=values['reference_class']
 if readout.ndim!=1 or query.ndim!=1 or not readout.size or not query.size:raise ValueError('Exact nonempty readout/query index vectors required')
 for name in model._FLOAT_FIELDS:
  if name!='log_donor_allele_frequency' and not np.isfinite(values[name]).all():raise ValueError('Nonfinite '+name)
 if np.any(clock<0) or np.any(values['sample_age']<0) or np.any(values['spans']<=0) or np.any(values['site_entropy']<0):
  raise ValueError('Observed ages/entropy must be nonnegative and spans positive')
 if np.any(p< -1) or np.any(p>=n) or np.any(sample<0) or np.any(sample>=n):raise ValueError('Invalid node/sample index')
 sampled=np.bincount(sample.reshape(-1),minlength=n)
 if np.any(sampled>1):raise ValueError('Every sample must have a unique per-tree native vertex')
 site_hold=values['site_hold']
 if (np.any(gen< -1) or np.any(gen>=a) or np.any(site_hold<0) or np.any(site_hold>=t) or np.any(site_hold[1:]<site_hold[:-1])
  or np.any(readout<0) or np.any(readout>=t) or np.any(query<0) or np.any(query>=h)
  or np.unique(query).size!=query.size or np.any(reference< -1) or np.any(reference>3)):
  raise ValueError('Invalid categorical, interval, query or proxy-class lookup')
 if h==280 and (query.size!=120 or tuple(np.count_nonzero(reference==c) for c in range(4))!=(24,24,24,24)):
  raise ValueError('The declared 280-panel requires actual query120 and pure R96, 24 per class')
 if np.any(reference[query]!= -1):raise ValueError('Scored queries may not carry input reference labels')
 if np.any(panel<0) or np.any(panel>1):raise ValueError('Panel frequencies must be probabilities')
 # Donor/panel values depend only on site and allele. Checking presence of an
 # allele in each site is equivalent to the original S*H*8 gather/mask check,
 # without materializing a repeated haplotype-by-donor temporary.
 for allele in range(a):
  present=np.any(gen==allele,axis=1);active=present[:,None]&mask
  called_donor=donor[:,allele,:][active]
  if not np.isfinite(called_donor).all() or np.any(called_donor>np.float32(1e-6)):
   raise ValueError('Called/available donor log frequencies must be finite and <=0')
  if np.any(present&(panel[:,allele]<=0)):raise ValueError('Called allele must have positive observed panel frequency')
 nonroot=np.flatnonzero(p>=0);roots=np.flatnonzero(p== -1)
 if (np.any(p[nonroot]==nonroot) or np.any(clock[p[nonroot]]<clock[nonroot]) or np.any(geo[nonroot,2]<=0)):
  raise ValueError('Rounded FP32 chronology/raw-derived duration geometry differs')
 if roots.size!=t:raise ValueError('One complete connected tree is required per interval')
 up=values['upward_levels'];children=values['upward_children'];ranks=values['upward_parent_rank'];down=values['downward_levels']
 if len(up)!=len(children) or len(up)!=len(ranks):raise ValueError('Upward level/child/rank tuples must agree')
 for arrays in zip(up,children,ranks):
  if any(x.dtype!=np.int64 or x.ndim!=1 for x in arrays) or arrays[1].shape!=arrays[2].shape:
   raise ValueError('Level arrays must be one-dimensional int64')
 if any(x.dtype!=np.int64 or x.ndim!=1 for x in down):raise ValueError('Downward levels must be int64 vectors')
 up_flat=_cat(up);down_flat=_cat(down)
 if up_flat.size!=n or np.any(up_flat<0) or np.any(up_flat>=n) or np.any(np.bincount(up_flat,minlength=n)!=1):
  raise ValueError('Upward sweep must retain every node exactly once')
 if (down_flat.size!=nonroot.size or np.any(down_flat<0) or np.any(down_flat>=n)
  or np.any(p[down_flat]<0) or np.any(np.bincount(down_flat,minlength=n)[nonroot]!=1)):
  raise ValueError('Downward sweep must retain every nonroot exactly once')
 up_rank=np.empty(n,dtype=np.int64);down_rank=np.full(n,-1,dtype=np.int64)
 for index,level in enumerate(up):up_rank[level]=index
 for index,level in enumerate(down):down_rank[level]=index
 if np.any(up_rank[nonroot]>=up_rank[p[nonroot]]):raise ValueError('Upward order must put each child before its parent')
 for level,ch,rank in zip(up,children,ranks):
  if (np.any(ch<0) or np.any(ch>=n) or np.any(rank<0) or np.any(rank>=level.size)
   or np.any(p[ch]!=level[rank])):raise ValueError('Upward children/ranks must match the original edges')
 ch_flat=_cat(children)
 if ch_flat.size!=nonroot.size or np.any(np.bincount(ch_flat,minlength=n)!=np.bincount(nonroot,minlength=n)):
  raise ValueError('Upward children/ranks must enumerate every original edge exactly once')
 interior=nonroot[p[p[nonroot]]>=0]
 if np.any(down_rank[p[interior]]>=down_rank[interior]):raise ValueError('Downward order must put each parent before its child')
 root_of=np.full(n,-1,dtype=np.int64);root_of[roots]=roots
 # Each level contains no internal parent-child edge, as verified above.
 # Reversing levels therefore propagates complete root identity in bulk.
 for level in reversed(up):
  nodes=level[p[level]>=0];root_of[nodes]=root_of[p[nodes]]
 if np.any(root_of<0):raise ValueError('Parent structure is cyclic or sweep order invalid')
 sample_roots=root_of[sample]
 if np.any(sample_roots!=sample_roots[:,:1]) or np.unique(sample_roots[:,0]).size!=t:
  raise ValueError('Sample columns must belong to their own complete interval tree')
 counts=np.bincount(p[nonroot],minlength=n);leaves=counts==0
 if np.any(~leaves[sample]) or np.count_nonzero(leaves)!=sample.size:raise ValueError('Every leaf must be a retained sampled haplotype')
 if not np.array_equal(clock[sample],np.broadcast_to(values['sample_age'],sample.shape)):
  raise ValueError('Persistent sampled ages changed across intervals')
 if np.any(geo[roots]!=0):raise ValueError('Root edge geometry must be zero')
 for name in ('branch_support_true','branch_support_sham'):
  support=values[name]
  if np.any(support<0) or np.any(support[:,:3]>np.float32(1+1e-6)):raise ValueError('Nonnegative support/counts and unit-scaled summaries required')
  if np.any((support[:,4]!=0)&(support[:,4]!=1)):raise ValueError('Empty support flag must be binary')
  if not np.array_equal(support[roots],np.broadcast_to(np.array([0,0,0,0,1],dtype=np.float32),(t,5))):
   raise ValueError('Root support must be [0,0,0,0,1] in both arms')
  if np.any(support[support[:,4]==1,:4]!=0):raise ValueError('Empty branch evidence must have zero support/count fields')
 return dict(nodes=n,trees=t,haplotypes=h,sites=s,queries=query.size,readouts=readout.size)


def validate_full_bank_v1(bank):
 validate_inputs_v1(bank,expected_haplotypes=952)
 reference=_numpy(bank.reference_class)
 if (bank.query_column.numel()!=686 or tuple(np.count_nonzero(reference==c) for c in range(4))!=observations.PROXY_COUNTS
  or np.count_nonzero(reference<0)!=686):raise ValueError('Exact full952 query/proxy count contract required')


def _clone(method,**updates):
 value=FunctionType(method.__code__,dict(method.__globals__,**updates),method.__name__,method.__defaults__,method.__closure__)
 value.__kwdefaults__=method.__kwdefaults__;return value


_CHECK_ITEM=_clone(original._check_item_v1,validate_full_bank_v1=validate_full_bank_v1)
_READ_PARENT=_clone(original.read_full_cache_parent_v1,_check_item_v1=_CHECK_ITEM)
_ENCODE_PARENT=_clone(original.old._encode_parent,validate_inputs_v1=validate_inputs_v1)

def check_item_v1(*args,**kwargs):
 _source_guard();return _CHECK_ITEM(*args,**kwargs)

def encode_parent_v1(item,*,expected_haplotypes=952):
 _source_guard();return _ENCODE_PARENT(item,expected_haplotypes=expected_haplotypes)


def read_full_cache_index_v1(*args,**kwargs):
 _source_guard();return original.read_full_cache_index_v1(*args,**kwargs)


def read_full_cache_parent_v1(*args,**kwargs):
 _source_guard();return _READ_PARENT(*args,**kwargs)


class FullPanelParentDatasetV1(original.FullPanelParentDatasetV1):
 """Original CPU spawn Dataset, with one fast check during each parent decode."""
 get=_clone(original.FullPanelParentDatasetV1.get,read_full_cache_parent_v1=read_full_cache_parent_v1)


make_parent_dataloader_v1=original.make_parent_dataloader_v1
shutdown_parent_dataloader_v1=original.shutdown_parent_dataloader_v1
