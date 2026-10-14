"""Observation-only H040 sample-anchored reassociation incidence. Inactive API."""
from __future__ import annotations
from dataclasses import dataclass,fields
from itertools import combinations
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
SOURCE="src/arggnn/anonymous_clade_witness_v1.py"
SCHEMA="anonymous_clade_reassociation_sidecar_v1"
PARENTS=tuple("ap_expanded_%s_%03d"%(s,i) for s,n in (("train",32),("val",8)) for i in range(n))
ARRAY_FIELDS=("boundary_index","old_node","new_node","shared_sample","weight","triple_index","old_pair","new_pair",
 "landmark_columns","triple_columns")
PAIR_SLOTS=((0,1),(0,2),(1,2))
SHARED_SLOT=np.asarray([[0,0,1],[0,0,2],[1,2,0]],dtype=np.int64)
EVENT_INDEX_FIELDS=("boundary_index","old_node","new_node","shared_sample")


def _np(value):
 if value.device.type!="cpu" or value.requires_grad:raise ValueError("Read-only CPU observed tensors required")
 return value.detach().resolve_conj().resolve_neg().numpy()


def _sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def choose_landmarks_v1(keys,parent_id,*,seed=101):
 """Uniform hash priorities over persistent keys; class/date/outcomes unused."""
 if seed!=101 or not isinstance(parent_id,str) or not parent_id or len(keys)<4 or len(set(keys))!=len(keys):
  raise ValueError("Fixed seed101, parent namespace and distinct observed sample keys required")
 for key in keys:
  if not isinstance(key,tuple) or len(key)!=2 or not isinstance(key[0],str) or type(key[1]) is not int or key[1] not in (0,1):
   raise ValueError("Actual (sample_id,haplotype) keys required")
 priority=lambda i:hashlib.sha256(json.dumps(["anonymous_clade_K4_uniform_v1",seed,parent_id,keys[i]],
  separators=(",",":"),ensure_ascii=True).encode()).digest()
 chosen=sorted(range(len(keys)),key=lambda i:(priority(i),keys[i]))[:4]
 return np.asarray(sorted(chosen,key=lambda i:keys[i]),dtype=np.int32)


def candidate_triples_v1(keys,landmarks):
 """Every unordered triple with >=2 landmarks, exactly once and key-order invariant."""
 ordered=sorted(range(len(keys)),key=lambda i:keys[i]);rank={column:i for i,column in enumerate(ordered)}
 landmark_set=set(map(int,landmarks));lm=sorted(landmark_set,key=lambda i:rank[i])
 triples=[tuple(sorted((q,a,b),key=lambda i:rank[i])) for q in ordered if q not in landmark_set
  for a,b in combinations(lm,2)]
 triples.extend(tuple(c) for c in combinations(lm,3))
 return np.asarray(triples,dtype=np.int32)


class ForestLCAV1:
 """Structural binary lifting; disjoint observed trees have no shared internal identity."""
 def __init__(self,parent,downward_levels):
  p=np.asarray(parent)
  if p.ndim!=1 or not p.size or p.size>=2**31 or np.any(p< -1) or np.any(p>=p.size):
   raise ValueError("Bounded native disjoint observed forest required")
  self.depth=np.zeros(p.size,dtype=np.int32);roots=np.flatnonzero(p<0);seen=np.zeros(p.size,dtype=bool);seen[roots]=True
  for level in downward_levels:
   nodes=np.asarray(level,dtype=np.int64)
   if nodes.size and (np.any(nodes<0) or np.any(nodes>=p.size) or np.any(p[nodes]<0)
    or np.any(seen[nodes]) or not np.all(seen[p[nodes]])):
    raise ValueError("Complete strict parent-before-child native levels required")
   self.depth[nodes]=self.depth[p[nodes]]+1;seen[nodes]=True
  if not np.all(seen):raise ValueError("Observed forest omitted vertices")
  first=p.astype(np.int32,copy=True);first[roots]=roots.astype(np.int32);self.up=[first]
  for _ in range(int(self.depth.max()).bit_length()):self.up.append(self.up[-1][self.up[-1]])
  self.root=self.up[-1];self.nodes=p.size
  self.sampled_leaf=np.bincount(p[p>=0],minlength=p.size)==0
 def lca(self,left,right):
  a,b=np.broadcast_arrays(left,right);shape=a.shape
  a=np.asarray(a,dtype=np.int32).reshape(-1).copy();b=np.asarray(b,dtype=np.int32).reshape(-1).copy()
  if np.any(a<0) or np.any(b<0) or np.any(a>=self.nodes) or np.any(b>=self.nodes):
   raise ValueError("Bounded actual local observed node indexes required")
  if np.any(self.root[a]!=self.root[b]):raise ValueError("No LCA across native trees")
  swap=self.depth[a]<self.depth[b];old=a.copy();a=np.where(swap,b,a);b=np.where(swap,old,b)
  delta=self.depth[a]-self.depth[b]
  for level,up in enumerate(self.up):a=np.where((delta>>level)&1,up[a],a)
  for up in reversed(self.up):
   aa,bb=up[a],up[b];move=(a!=b)&(aa!=bb);a=np.where(move,aa,a);b=np.where(move,bb,b)
  return np.where(a==b,a,self.up[0][a]).reshape(shape)


def resolve_triplets_v1(lca,nodes,pair_lcas=None):
 """Strict closest-pair topology; ties/sample interiors unresolved, no age threshold."""
 a,b,c=np.moveaxis(nodes,-1,0)
 x,y,z=(lca.lca(a,b),lca.lca(a,c),lca.lca(b,c)) if pair_lcas is None else pair_lcas
 interior=(~lca.sampled_leaf[a])|(~lca.sampled_leaf[b])|(~lca.sampled_leaf[c])
 for ancestor in (x,y,z):
  for sample in (a,b,c):interior|=ancestor==sample
 state=np.full(a.shape,3,dtype=np.uint8);receiver=np.full(a.shape,-1,dtype=np.int32)
 for code,(winner,left,right) in enumerate(((x,y,z),(y,x,z),(z,x,y))):
  valid=(~interior)&(left==right)&(lca.depth[winner]>lca.depth[left])
  state[valid]=code;receiver[valid]=winner[valid]
 return state,receiver


def _block_states(lca,samples,landmarks,triples):
 # Compact [B,H,4] cache, not HxH or all H^3 triples.
 cache=lca.lca(samples[:,:,None],samples[:,landmarks][:,None,:])
 lm_rank={int(column):i for i,column in enumerate(landmarks)}
 pair_lcas=[]
 for left,right in PAIR_SLOTS:
  a,b=triples[:,left],triples[:,right]
  b_landmark=np.isin(b,landmarks)
  sample=np.where(b_landmark,a,b)
  landmark=np.where(b_landmark,b,a)
  ranks=np.asarray([lm_rank[int(i)] for i in landmark],dtype=np.int64)
  pair_lcas.append(cache[:,sample,ranks])
 return resolve_triplets_v1(lca,samples[:,triples],tuple(pair_lcas))


def _original_bank(bank):
 from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
 if type(bank) is not MutationRoutedWholeTreeInputsV1:raise TypeError("Exact original observation-only bank required")
 samples=_np(bank.sample_index);parent=_np(bank.parent)
 if parent.ndim!=1 or not parent.size or parent.size>=2**31 or np.any(parent< -1) or np.any(parent>=parent.size):
  raise ValueError('Bounded native observed parent forest required')
 if samples.ndim!=2 or samples.shape[0]<1 or samples.shape[1]!=952 or _np(bank.query_column).shape!=(686,):
  raise ValueError("Complete native holds/all952 observed haplotypes/all686queries required")
 if np.any(samples<0) or np.any(samples>=parent.size):raise ValueError("Bounded actual sampled vertices required")
 return parent,samples


def build_event_arrays_v1(bank,keys,parent_id,*,hold_block_size=16):
 """All eligible changes, blocked structural LCA preparation; no sparsity assumption."""
 parent,samples=_original_bank(bank);t,h=samples.shape
 if parent_id not in PARENTS or len(keys)!=h or type(hold_block_size) is not int or not 1<=hold_block_size<=64:
  raise ValueError("Registered32TRAIN/eightVAL, full stable keys and bounded hold block required")
 landmarks=choose_landmarks_v1(keys,parent_id);triples=candidate_triples_v1(keys,landmarks)
 if triples.shape!=(5692,3):raise ValueError("All5692 eligible unordered sample triples required")
 lca=ForestLCAV1(parent,tuple(_np(v) for v in bank.downward_levels))
 roots=lca.root[samples]
 if np.any(roots!=roots[:,:1]) or np.unique(roots[:,0]).size!=t:
  raise ValueError("Every hold is one distinct native tree containing all952 sampled leaves")
 chunks={name:[] for name in (*EVENT_INDEX_FIELDS,"triple_index","old_pair","new_pair")}
 previous=None;both_count=0;unchanged_count=0
 for start in range(0,t,hold_block_size):
  end=min(t,start+hold_block_size);state,receiver=_block_states(lca,samples[start:end],landmarks,triples)
  if start==0:old,new=state[:-1],state[1:];oldnode,newnode=receiver[:-1],receiver[1:];offset=0
  else:
   old=np.concatenate((previous[0][None],state[:-1]),0);new=state
   oldnode=np.concatenate((previous[1][None],receiver[:-1]),0);newnode=receiver;offset=start-1
  both=(old<3)&(new<3);changed=both&(old!=new)
  both_count+=int(both.sum());unchanged_count+=int((both&~changed).sum())
  rows,ids=np.nonzero(changed)
  if ids.size:
   oldpair=old[rows,ids];newpair=new[rows,ids]
   shared=triples[ids,SHARED_SLOT[oldpair,newpair]]
   values=dict(boundary_index=rows+offset,old_node=oldnode[rows,ids],new_node=newnode[rows,ids],
    shared_sample=shared,triple_index=ids,old_pair=oldpair,new_pair=newpair)
   for name,value in values.items():chunks[name].append(value.astype(np.uint8 if name.endswith("pair") else np.int32,copy=False))
  previous=(state[-1].copy(),receiver[-1].copy())
 arrays={name:np.concatenate(value) if value else np.empty(0,dtype=np.uint8 if name.endswith("pair") else np.int32)
  for name,value in chunks.items()}
 # A global native vertex can occur in only one boundary in each endpoint role,
 # because the forest has distinct node blocks. Thus these counts are per-boundary.
 old_degree=np.bincount(arrays["old_node"],minlength=parent.size)
 new_degree=np.bincount(arrays["new_node"],minlength=parent.size)
 arrays["weight"]=(1./np.sqrt(old_degree[arrays["old_node"]].astype(np.float64)*
  new_degree[arrays["new_node"]].astype(np.float64))).astype(np.float32)
 arrays.update(landmark_columns=landmarks,triple_columns=triples)
 total=(t-1)*5692
 meta=dict(native_holds=t,haplotypes=h,queries=686,native_nodes=int(parent.size),event_count=int(arrays["weight"].size),
  candidates_per_boundary=5692,candidate_boundaries=total,resolved_both=both_count,
  unresolved_either=total-both_count,unchanged_resolved=unchanged_count,seed=101,K=4,
  landmark_records=[dict(column=int(i),key=list(keys[i])) for i in landmarks],
  sample_keys_sha256=hashlib.sha256(json.dumps(keys,separators=(",",":")).encode()).hexdigest(),
  topology_sha256=_sha(parent),sample_index_sha256=_sha(samples),query_column_sha256=_sha(_np(bank.query_column)),
  complete_observed_panel=True,truth_or_teacher_inputs=False,internal_ancestor_correspondence_claimed=False,
  landmark_selection_independent_of_class_age_outcome=True,unsupported_sampled_interior_triplets_unresolved=True,raw_array_bytes=sum(a.nbytes for a in arrays.values()))
 return arrays,meta


def verify_event_arrays_v1(arrays,bank,*,block_size=16384):
 """Read-time bounded proof of within-tree winning LCAs, shared sample and eta."""
 parent,samples=_original_bank(bank);t,h=samples.shape
 if type(block_size) is not int or not 1<=block_size<=65536:raise ValueError("Bounded CPU verification block required")
 if set(arrays)!=set(ARRAY_FIELDS):raise ValueError("Exact event/audit-only sidecar fields required")
 e=arrays["weight"].size
 for name in (*EVENT_INDEX_FIELDS,"triple_index"):
  if arrays[name].dtype!=np.int32 or arrays[name].shape!=(e,):raise ValueError("Compact int32 event indexes required")
 for name in ("old_pair","new_pair"):
  if arrays[name].dtype!=np.uint8 or arrays[name].shape!=(e,):raise ValueError("Actual compact structural pair codes required")
 lm,triples=arrays["landmark_columns"],arrays["triple_columns"]
 if (lm.dtype!=np.int32 or lm.shape!=(4,) or np.unique(lm).size!=4 or np.any(lm<0) or np.any(lm>=h)
  or triples.dtype!=np.int32 or triples.shape!=(5692,3) or np.any(triples<0) or np.any(triples>=h)
  or np.any(np.diff(np.sort(triples,axis=-1),axis=-1)==0)
  or np.unique(np.sort(triples,axis=-1),axis=0).shape[0]!=5692
  or np.any(np.isin(triples,lm).sum(-1)<2)):
  raise ValueError("Exactly four landmarks/all5692 unique eligible triples required")
 bounds={"boundary_index":t-1,"old_node":parent.size,"new_node":parent.size,"shared_sample":h,"triple_index":5692}
 for name,limit in bounds.items():
  if np.any(arrays[name]<0) or np.any(arrays[name]>=limit):raise ValueError("Bounded native incidence indexes required")
 if (arrays["weight"].dtype!=np.float32 or arrays["weight"].shape!=(e,)
  or not np.isfinite(arrays["weight"]).all() or np.any(arrays["weight"]<=0) or np.any(arrays["weight"]>1)
  or np.any(arrays["old_pair"]>=3) or np.any(arrays["new_pair"]>=3) or np.any(arrays["old_pair"]==arrays["new_pair"])):
  raise ValueError("Strict changed resolved rooted-triplet events and positive observed eta required")
 if np.unique(arrays["boundary_index"].astype(np.int64)*5692+arrays["triple_index"]).size!=e:
  raise ValueError("No duplicate boundary/triple incidences")
 lca=ForestLCAV1(parent,tuple(_np(v) for v in bank.downward_levels))
 roots=lca.root[samples]
 if np.any(roots!=roots[:,:1]) or np.unique(roots[:,0]).size!=t:
  raise ValueError("Distinct complete native trees per hold required")
 for start in range(0,e,block_size):
  part=slice(start,min(e,start+block_size));boundary=arrays["boundary_index"][part]
  cols=triples[arrays["triple_index"][part]]
  os,oc=resolve_triplets_v1(lca,samples[boundary[:,None],cols])
  ns,nc=resolve_triplets_v1(lca,samples[(boundary+1)[:,None],cols])
  shared=cols[np.arange(cols.shape[0]),SHARED_SLOT[os.clip(0,2),ns.clip(0,2)]]
  if (not np.array_equal(os,arrays["old_pair"][part]) or not np.array_equal(ns,arrays["new_pair"][part])
   or not np.array_equal(oc,arrays["old_node"][part]) or not np.array_equal(nc,arrays["new_node"][part])
   or not np.array_equal(shared,arrays["shared_sample"][part])):
   raise ValueError("Each receiver must be its actual within-tree winning-pair LCA through the shared sample")
 old_degree=np.bincount(arrays["old_node"],minlength=parent.size)
 new_degree=np.bincount(arrays["new_node"],minlength=parent.size)
 expected=(1./np.sqrt(old_degree[arrays["old_node"]].astype(np.float64)*
  new_degree[arrays["new_node"]].astype(np.float64))).astype(np.float32)
 if not np.array_equal(expected,arrays["weight"]):raise ValueError("Exact observed endpoint degree normalization required")
 return arrays


@dataclass(frozen=True)
class AnonymousCladeEventsV1:
 boundary_index: torch.Tensor
 old_node: torch.Tensor
 new_node: torch.Tensor
 shared_sample: torch.Tensor
 weight: torch.Tensor
 native_holds: int
 haplotypes: int
 native_nodes: int
 @classmethod
 def from_verified_arrays(cls,arrays,meta):
  return cls(**{name:torch.from_numpy(arrays[name]) for name in (*EVENT_INDEX_FIELDS,"weight")},
   native_holds=int(meta["native_holds"]),haplotypes=int(meta["haplotypes"]),native_nodes=int(meta["native_nodes"]))
 def to(self,device,*,non_blocking=False):
  return type(self)(**{name:getattr(self,name).to(device,non_blocking=non_blocking)
   for name in (*EVENT_INDEX_FIELDS,"weight")},native_holds=self.native_holds,
   haplotypes=self.haplotypes,native_nodes=self.native_nodes)
 def tensor_bytes(self):return sum(getattr(self,n).numel()*getattr(self,n).element_size() for n in (*EVENT_INDEX_FIELDS,"weight"))


def prepare_parent_sidecar_v1(parent_id,bank,entry_ref,reader,output,*,cache_parent_ref,hold_block_size=16):
 """Existing root guarded output.save owns writes; source arrays/old caches unchanged."""
 from .mutation_routed_parent_loader_v1 import ObservationOnlyParentReaderV1
 from .mutation_routed_full_panel_observations_v1 import validate_full_roster_v1
 boundary=reader if isinstance(reader,ObservationOnlyParentReaderV1) else ObservationOnlyParentReaderV1(reader)
 if parent_id not in PARENTS or not callable(getattr(output,"save",None)):raise ValueError("Root-owned guarded observation writer required")
 entry=boundary.decode_json(entry_ref)
 if (entry.get("parent_id")!=parent_id or entry.get("schema")!="mutation_routed_full_panel_observed_entry_v1"
  or entry.get("observations_only") is not True or entry.get("truth_or_oracle_read") is not False):
  raise ValueError("Original same-parent observation-only entry required")
 rows=boundary.decode_json(entry["refs"]["source_roster"]);validate_full_roster_v1(rows)
 keys=tuple((row["sample_id"],int(row["haplotype"])) for row in rows)
 classes=np.asarray([row["observed_reference_class"] if row["role"]=="pure_source_reference" else -1 for row in rows],dtype=np.int64)
 if not np.array_equal(classes,_np(bank.reference_class)):raise ValueError("Original stable observed roster column join required")
 started=time.monotonic();arrays,meta=build_event_arrays_v1(bank,keys,parent_id,hold_block_size=hold_block_size)
 array_ref=output.save(parent_id+".anonymous_clade_events.npz",arrays,"npz")
 record=dict(schema=SCHEMA,parent_id=parent_id,observation_entry_ref=boundary.record(entry_ref),
  cache_parent_ref=boundary.record(cache_parent_ref),roster_ref=entry["refs"]["source_roster"],
  builder_ref={"path":SOURCE,"sha256":hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()},
  array_ref=array_ref,array_fields=list(ARRAY_FIELDS),metadata=meta,targets_opened=False,outer_included=False,
  preprocessing_seconds=time.monotonic()-started)
 record_ref=output.save(parent_id+".anonymous_clade_events.json",record)
 return record_ref,AnonymousCladeEventsV1.from_verified_arrays(arrays,meta),record


def read_parent_sidecar_v1(record_ref,reader,*,parent_id,entry_ref,cache_parent_ref,bank):
 import io
 from .mutation_routed_parent_loader_v1 import ObservationOnlyParentReaderV1
 boundary=reader if isinstance(reader,ObservationOnlyParentReaderV1) else ObservationOnlyParentReaderV1(reader)
 record=boundary.decode_json(record_ref);meta=record["metadata"]
 if (parent_id not in PARENTS or record.get("schema")!=SCHEMA or record.get("parent_id")!=parent_id
  or record.get("observation_entry_ref")!=boundary.record(entry_ref)
  or record.get("cache_parent_ref")!=boundary.record(cache_parent_ref)
  or record.get("builder_ref")!={"path":SOURCE,"sha256":hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()}
  or record.get("targets_opened") is not False or record.get("outer_included") is not False
  or meta.get("truth_or_teacher_inputs") is not False or meta.get("internal_ancestor_correspondence_claimed") is not False):
  raise ValueError("Exact original parent/cache/source observation-only incidence lineage required")
 for name,key in (("parent","topology_sha256"),("sample_index","sample_index_sha256"),("query_column","query_column_sha256")):
  if _sha(_np(getattr(bank,name)))!=meta[key]:raise ValueError("Local observed bank/sidecar identity differs")
 with np.load(io.BytesIO(boundary.read(record["array_ref"])),allow_pickle=False) as archive:
  if set(archive.files)!=set(ARRAY_FIELDS):raise ValueError("No target/teacher/extra fields in event archive")
  arrays={name:archive[name].copy() for name in ARRAY_FIELDS}
 from .mutation_routed_full_panel_observations_v1 import validate_full_roster_v1
 rows=boundary.decode_json(record["roster_ref"]);validate_full_roster_v1(rows)
 keys=tuple((row["sample_id"],int(row["haplotype"])) for row in rows)
 if (hashlib.sha256(json.dumps(keys,separators=(",",":")).encode()).hexdigest()!=meta["sample_keys_sha256"]
  or not np.array_equal(choose_landmarks_v1(keys,parent_id),arrays["landmark_columns"])
  or not np.array_equal(candidate_triples_v1(keys,arrays["landmark_columns"]),arrays["triple_columns"])):
  raise ValueError("Exact class/date-independent stable-key landmark and triple enumeration required")
 verify_event_arrays_v1(arrays,bank)
 if (meta["native_holds"]!=bank.sample_index.shape[0] or meta["native_nodes"]!=bank.parent.numel()
  or meta["haplotypes"]!=952 or meta["queries"]!=686 or meta["event_count"]!=arrays["weight"].size
  or meta["K"]!=4 or meta["seed"]!=101 or meta["candidate_boundaries"]!=5692*(meta["native_holds"]-1)
  or meta["resolved_both"]+meta["unresolved_either"]!=meta["candidate_boundaries"]
  or meta["resolved_both"]-meta["unchanged_resolved"]!=meta["event_count"]):
  raise ValueError("Complete observed incidence dimensions required")
 return AnonymousCladeEventsV1.from_verified_arrays(arrays,meta),record


def prepare_all_sidecars_v1(cache_ref,reader,*,registry,entry_refs,source_refs,output,execution_guard):
 """One original cache decode/parent, all40 observations; no target or GPU factory."""
 from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
 from .mutation_routed_full_panel_observations_v1 import validate_registry_v1
 roles=validate_registry_v1(registry)
 if tuple(roles["train"]+roles["val"])!=PARENTS or set(entry_refs)!=set(PARENTS) or not callable(execution_guard):
  raise ValueError("Root-bound original32TRAIN/eightVAL observation-only collection required")
 index=read_full_cache_index_v1(cache_ref,reader,registry=registry,entry_refs=entry_refs,source_refs=source_refs)
 records={};raw_bytes=0;events=0;started=time.monotonic()
 for pid in PARENTS:
  execution_guard();item=read_full_cache_parent_v1(pid,index,reader)
  ref,boundary,record=prepare_parent_sidecar_v1(pid,item["model_inputs"],entry_refs[pid],reader,output,
   cache_parent_ref=index["parents"][pid]["array_ref"])
  records[pid]=ref;raw_bytes+=record["metadata"]["raw_array_bytes"];events+=record["metadata"]["event_count"]
  del item,boundary,record
 manifest=output.save("all40_anonymous_clade_observation_sidecars.json",dict(
  schema="anonymous_clade_full_observation_sidecars_v1",parent_ids=list(PARENTS),parents=records,
  observation_cache_ref=cache_ref,entry_refs=entry_refs,source_refs=source_refs,targets_opened=False,outer_included=False,
  raw_array_bytes=raw_bytes,event_count=events,preprocessing_seconds=time.monotonic()-started))
 return manifest


def sidecar_byte_bound_v1(native_holds,*,haplotypes=952):
 """Raw array bound assumes every eligible triple changes; compression uncredited."""
 if type(native_holds) is not int or native_holds<1 or haplotypes!=952:raise ValueError("Full952/native holds required")
 triples=(952-4)*6+4;events=(native_holds-1)*triples
 return dict(max_events=events,event_bytes=26*events,static_audit_bytes=triples*3*4+4*4,
  raw_array_bytes=26*events+triples*3*4+4*4,compression_assumed=False)
