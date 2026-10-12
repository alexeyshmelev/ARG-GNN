"""Observation-only H039 rooted-triplet witnesses. No execution/admission on import."""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/directed_clade_exchange_witness_v1.py"
PARENTS=tuple("ap_expanded_%s_%03d"%(split,i) for split,n in (("train",32),("val",8)) for i in range(n))
OFF_DIAGONAL=tuple((a,b) for a in range(4) for b in range(4) if a!=b)
ARRAY_FIELDS=("witness_counts","turnover","unresolved","donor_counts","sentinel_columns")
SCHEMA="directed_clade_exchange_observation_sidecar_v1"


def _np(tensor):
 if tensor.device.type!="cpu" or tensor.requires_grad:raise ValueError("Read-only CPU observation tensors required")
 return tensor.detach().resolve_conj().resolve_neg().numpy()


def select_sentinels_v1(keys,ages,classes):
 """At most two distinct observed individuals per class, fixed along genome."""
 if len(keys)!=952 or len(set(keys))!=952 or any(not isinstance(k,tuple) or len(k)!=2 for k in keys):
  raise ValueError("All952 unique stable (sample_id,haplotype) keys required")
 selected=[];record=[]
 for cls in range(4):
  candidates=[i for i,c in enumerate(classes) if int(c)==cls]
  # One actual haplotype per actual individual, deterministically chosen.
  by_person={}
  for i in candidates:
   key=keys[i]
   if not isinstance(key[0],str) or type(key[1]) is not int or key[1] not in (0,1):raise ValueError("Stable observed phased identity required")
   old=by_person.get(key[0])
   if old is None or key[1]<keys[old][1]:by_person[key[0]]=i
  people=list(by_person.values())
  tie=lambda i:hashlib.sha256(json.dumps(keys[i],separators=(",",":")).encode()).hexdigest()
  people.sort(key=lambda i:(float(ages[i]),tie(i)))
  if people:
   first=people[0];chosen=[first]
   rest=[i for i in people if keys[i][0]!=keys[first][0]]
   if rest:chosen.append(sorted(rest,key=lambda i:(-float(ages[i]),tie(i)))[0])
   for i in chosen:selected.append(i);record.append(dict(column=i,source_class=cls,key=list(keys[i]),age_generations=float(ages[i])))
 return np.asarray(selected,dtype=np.int64),record


class BatchedForestLCAV1:
 """Binary lifting on disjoint observed trees, never cross-interval identities."""
 def __init__(self,parent,downward_levels):
  p=np.asarray(parent)
  if p.ndim!=1 or not p.size or p.size>=np.iinfo(np.int32).max or np.any(p< -1) or np.any(p>=p.size):
   raise ValueError("Bounded observed forest required")
  self.depth=np.zeros(p.size,dtype=np.int32)
  roots=np.flatnonzero(p<0);seen=np.zeros(p.size,dtype=bool);seen[roots]=True
  for nodes in downward_levels:
   nodes=np.asarray(nodes,dtype=np.int64)
   if nodes.size and (np.any(p[nodes]<0) or not np.all(seen[p[nodes]]) or np.any(seen[nodes])):
    raise ValueError("Complete strict parent-before-child observed levels required")
   self.depth[nodes]=self.depth[p[nodes]]+1;seen[nodes]=True
  if not np.all(seen):raise ValueError("Observed forest omitted vertices")
  first=p.astype(np.int32,copy=True);first[roots]=roots.astype(np.int32)
  self.up=[first]
  for _ in range(int(self.depth.max()).bit_length()):self.up.append(self.up[-1][self.up[-1]])
  self.nodes=p.size

 def lca(self,left,right):
  a,b=np.broadcast_arrays(left,right);shape=a.shape
  a=np.asarray(a,dtype=np.int32).reshape(-1).copy();b=np.asarray(b,dtype=np.int32).reshape(-1).copy()
  if np.any(a<0) or np.any(b<0) or np.any(a>=self.nodes) or np.any(b>=self.nodes):raise ValueError("Actual local node indexes required")
  if np.any(self.up[-1][a]!=self.up[-1][b]):raise ValueError("LCA requests cannot cross observed tree roots")
  swap=self.depth[a]<self.depth[b];old=a.copy();a=np.where(swap,b,a);b=np.where(swap,old,b)
  delta=self.depth[a]-self.depth[b]
  for level,up in enumerate(self.up):a=np.where((delta>>level)&1,up[a],a)
  for up in reversed(self.up):
   aa,bb=up[a],up[b];move=(a!=b)&(aa!=bb);a=np.where(move,aa,a);b=np.where(move,bb,b)
  return np.where(a==b,a,self.up[0][a]).reshape(shape)


def _triplets(lca,samples,queries,sentinels,pr,ps):
 q=samples[:,queries];r=samples[:,sentinels]
 qr=lca.lca(q[:,:,None],r[:,None,:]);rs=lca.lca(r[:,:,None],r[:,None,:])
 x,y,z=qr[:,:,pr],qr[:,:,ps],rs[:,pr,ps][:,None,:]
 qnode=q[:,:,None];rnode=r[:,pr][:,None,:];snode=r[:,ps][:,None,:]
 ancestors=(x==qnode)|(x==rnode)|(y==qnode)|(y==snode)|(z==rnode)|(z==snode)
 state=np.full(x.shape,3,dtype=np.uint8) # unresolved/tied/sampled-ancestor
 resolved=~ancestors;dx,dy,dz=lca.depth[x],lca.depth[y],lca.depth[z]
 state[resolved&(x==y)&(dz>dx)]=0 # donor pair closest, q outside
 state[resolved&(y==z)&(dx>dy)]=1 # (q,r)|s
 state[resolved&(x==z)&(dy>dx)]=2 # (q,s)|r
 return state


def build_witness_arrays_v1(bank,keys,raw_ages,*,hold_block_size=32):
 """Complete native holds, all686 queries, compact q-to-eight-sentinel LCAs."""
 from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
 if type(bank) is not MutationRoutedWholeTreeInputsV1:raise TypeError("Original exact observation bank required")
 if type(hold_block_size) is not int or hold_block_size<1:raise ValueError("Positive bounded hold block required")
 samples=_np(bank.sample_index);classes=_np(bank.reference_class);queries=_np(bank.query_column)
 ages=np.asarray(raw_ages,dtype=np.float64)
 t,h=samples.shape;q=queries.size
 if t<1 or h!=952 or q!=686 or ages.shape!=(952,) or not np.isfinite(ages).all() or np.any(ages<0):
  raise ValueError("Full actual952/686 observations and real dates required")
 if not np.array_equal(ages.astype(np.float32),_np(bank.sample_age)) or np.any(classes[queries]!=-1):
  raise ValueError("Original sample age/class join required")
 sentinels,sentinel_record=select_sentinels_v1(keys,ages,classes)
 donor_counts=np.asarray([np.count_nonzero(classes==c) for c in range(4)],dtype=np.int64)
 if tuple(donor_counts)!=(50,28,102,86) or sentinels.size!=8:
  raise ValueError("Original full266 observed donors and eight distinct-individual sentinels required")
 sc=classes[sentinels]
 pairs=[(r,s) for r in range(8) for s in range(8) if sc[r]!=sc[s]]
 pr=np.asarray([r for r,s in pairs],dtype=np.int64);ps=np.asarray([s for r,s in pairs],dtype=np.int64)
 pair_class=np.asarray([OFF_DIAGONAL.index((int(sc[r]),int(sc[s]))) for r,s in pairs],dtype=np.int64)
 lca=BatchedForestLCAV1(_np(bank.parent),tuple(_np(x) for x in bank.downward_levels))
 counts=np.zeros((max(t-1,0),q,12),dtype=np.float32)
 turnover=np.zeros((max(t-1,0),q),dtype=np.float32);unresolved=np.zeros_like(turnover)
 previous=None
 for start in range(0,t,hold_block_size):
  end=min(t,start+hold_block_size)
  state=_triplets(lca,samples[start:end],queries,sentinels,pr,ps)
  if start==0:old,new=state[:-1],state[1:];destination=slice(0,end-1)
  else:
   old=np.concatenate((previous[None],state[:-1]),axis=0);new=state;destination=slice(start-1,end-1)
  if new.shape[0]:
   both=(old!=3)&(new!=3);changed=(old!=new)&both
   valid=both.sum(-1)
   np.divide(changed.sum(-1),valid,out=turnover[destination],where=valid>0)
   unresolved[destination]=1.-both.mean(-1,dtype=np.float64)
   witness=(old==1)&(new==2)
   for j in range(12):
    mask=pair_class==j
    counts[destination,:,j]=witness[:,:,mask].mean(-1,dtype=np.float64)
  previous=state[-1].copy()
 return dict(witness_counts=counts,turnover=turnover,unresolved=unresolved,
  donor_counts=donor_counts,sentinel_columns=sentinels),dict(
  native_holds=int(t),queries=int(q),haplotypes=952,donors=266,
  sentinel_records=sentinel_record,ordered_pairs=len(pairs),hold_block_size=hold_block_size,
  sample_keys_sha256=hashlib.sha256(json.dumps(keys,separators=(",",":")).encode()).hexdigest(),
  query_column_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
  topology_sha256=hashlib.sha256(_np(bank.parent).tobytes()).hexdigest(),
  sample_index_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
  sample_identity_only=True,internal_ancestor_correspondence_claimed=False,
  sampled_ancestor_triplets_unresolved=True,truth_or_teacher_inputs=False)


def prepare_parent_sidecar_v1(parent_id,bank,entry_ref,reader,output,*,cache_parent_ref,guard,hold_block_size=32):
 """Root passes a bank decoded once by the frozen observation-only cache reader."""
 from .mutation_routed_parent_loader_v1 import ObservationOnlyParentReaderV1
 from .mutation_routed_full_panel_observations_v1 import validate_full_roster_v1
 if parent_id not in PARENTS:raise ValueError("Exactly32TRAIN/eight VAL; no outer")
 if not callable(guard):raise ValueError("Admitted root write guard required")
 boundary=reader if isinstance(reader,ObservationOnlyParentReaderV1) else ObservationOnlyParentReaderV1(reader)
 entry=boundary.decode_json(entry_ref)
 if entry.get("parent_id")!=parent_id:raise ValueError("Original observed parent entry required")
 rows=boundary.decode_json(entry["refs"]["source_roster"]);validate_full_roster_v1(rows)
 keys=tuple((row["sample_id"],int(row["haplotype"])) for row in rows)
 age=np.asarray([row["age_generations"] for row in rows],dtype=np.float64)
 observed=np.asarray([row["observed_reference_class"] if row["role"]=="pure_source_reference" else -1 for row in rows],dtype=np.int64)
 if not np.array_equal(observed,_np(bank.reference_class)):raise ValueError("Original observed proxy-class column join required")
 started=time.monotonic();arrays,meta=build_witness_arrays_v1(bank,keys,age,hold_block_size=hold_block_size)
 output=Path(output)
 if not output.is_absolute():output=ROOT/output
 output.relative_to(ROOT)
 if output.exists() or ".." in output.parts or output.resolve()!=output:raise ValueError("Fresh canonical project parent-sidecar directory required")
 guard(sum(x.nbytes for x in arrays.values())+65536);output.mkdir()
 archive=output/"observed_witnesses.npz"
 with archive.open("xb") as stream:np.savez(stream,**arrays)
 def ref(p):
  digest=hashlib.sha256()
  with p.open("rb") as stream:
   for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
  return dict(path=p.relative_to(ROOT).as_posix(),sha256=digest.hexdigest())
 record=dict(schema=SCHEMA,parent_id=parent_id,split="train" if "_train_" in parent_id else "val",
  observation_entry_ref=boundary.record(entry_ref),cache_parent_ref=boundary.record(cache_parent_ref),
  roster_ref=entry["refs"]["source_roster"],builder_ref=ref(ROOT/SOURCE),array_ref=ref(archive),array_fields=list(ARRAY_FIELDS),
  metadata=meta,preprocessing_seconds=time.monotonic()-started,targets_opened=False,outer_included=False)
 raw=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode();guard(len(raw)+8192)
 receipt=output/"sidecar.json"
 with receipt.open("xb") as stream:stream.write(raw)
 return ref(receipt)


def read_parent_sidecar_v1(record_ref,reader,*,parent_id,observation_entry_ref,cache_parent_ref):
 from .mutation_routed_parent_loader_v1 import ObservationOnlyParentReaderV1
 boundary=reader if isinstance(reader,ObservationOnlyParentReaderV1) else ObservationOnlyParentReaderV1(reader)
 record=boundary.decode_json(record_ref)
 if (parent_id not in PARENTS or record.get("schema")!=SCHEMA or record["parent_id"]!=parent_id
  or record["observation_entry_ref"]!=boundary.record(observation_entry_ref)
  or record["cache_parent_ref"]!=boundary.record(cache_parent_ref)
  or record["targets_opened"] is not False or record["outer_included"] is not False
  or record.get("builder_ref")!={"path":SOURCE,"sha256":hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()}):
  raise ValueError("Exact observation-only sidecar/source-parent lineage required")
 import io
 with np.load(io.BytesIO(boundary.read(record["array_ref"])),allow_pickle=False) as archive:
  if set(archive.files)!=set(ARRAY_FIELDS):raise ValueError("No extra witness/target/teacher fields")
  arrays={name:archive[name].copy() for name in ARRAY_FIELDS}
 t=record["metadata"]["native_holds"]
 if (arrays["witness_counts"].shape!=(t-1,686,12) or arrays["turnover"].shape!=(t-1,686)
  or arrays["unresolved"].shape!=(t-1,686) or arrays["donor_counts"].shape!=(4,)
  or arrays["sentinel_columns"].shape!=(8,)):raise ValueError("Complete sidecar shapes required")
 for name in ("witness_counts","turnover","unresolved"):
  if arrays[name].dtype!=np.float32 or not np.isfinite(arrays[name]).all() or np.any(arrays[name]<0) or np.any(arrays[name]>1):
   raise ValueError("Bounded FP32 observed witness fields required")
 if arrays["donor_counts"].dtype!=np.int64 or tuple(arrays["donor_counts"])!=(50,28,102,86):
  raise ValueError("All266 original donor availability required")
 if (arrays["sentinel_columns"].dtype!=np.int64 or np.unique(arrays["sentinel_columns"]).size!=8
  or np.any(arrays["sentinel_columns"]<0) or np.any(arrays["sentinel_columns"]>=952)):
  raise ValueError("Eight actual distinct sentinel columns required")
 return arrays,record


def prepare_full_sidecars_v1(cache_manifest_ref,reader,*,registry,entry_refs,source_refs,
 output,execution_guard,hold_block_size=32):
 """Decode each original observed bank once; one admitted writer owns all outputs."""
 from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
 from .mutation_routed_full_panel_observations_v1 import validate_registry_v1,validate_full_roster_v1
 from .mutation_routed_parent_loader_v1 import ObservationOnlyParentReaderV1
 from .directed_clade_exchange_crf_v1 import CladeExchangeInputsV1
 roles=validate_registry_v1(registry)
 if tuple(roles["train"]+roles["val"])!=PARENTS or set(entry_refs)!=set(PARENTS):
  raise ValueError("Exact original32TRAIN/eightVAL, no outer observations")
 if not callable(execution_guard) or not callable(getattr(output,"save",None)):
  raise ValueError("Existing admitted root output writer and execution guard required")
 boundary=reader if isinstance(reader,ObservationOnlyParentReaderV1) else ObservationOnlyParentReaderV1(reader)
 index=read_full_cache_index_v1(cache_manifest_ref,reader,registry=registry,entry_refs=entry_refs,source_refs=source_refs)
 builder_ref={"path":SOURCE,"sha256":hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()}
 banks,parents,records={},{},{};raw_witness_bytes=0;started=time.monotonic()
 for pid in PARENTS:
  execution_guard();item=read_full_cache_parent_v1(pid,index,reader);base=item["model_inputs"]
  entry=boundary.decode_json(entry_refs[pid]);rows=boundary.decode_json(entry["refs"]["source_roster"])
  validate_full_roster_v1(rows)
  keys=tuple((r["sample_id"],int(r["haplotype"])) for r in rows)
  ages=np.asarray([r["age_generations"] for r in rows],dtype=np.float64)
  observed=np.asarray([r["observed_reference_class"] if r["role"]=="pure_source_reference" else -1 for r in rows],dtype=np.int64)
  if entry.get("parent_id")!=pid or not np.array_equal(observed,_np(base.reference_class)):
   raise ValueError("Original observed roster/class column join required")
  prep=time.monotonic();arrays,meta=build_witness_arrays_v1(base,keys,ages,hold_block_size=hold_block_size)
  raw_witness_bytes+=sum(a.nbytes for a in arrays.values())
  array_ref=output.save(pid+".observed_witnesses.npz",arrays,"npz")
  sidecar=dict(schema=SCHEMA,parent_id=pid,split="train" if pid in PARENTS[:32] else "val",
   observation_entry_ref=boundary.record(entry_refs[pid]),cache_parent_ref=boundary.record(index["parents"][pid]["array_ref"]),
   roster_ref=entry["refs"]["source_roster"],builder_ref=builder_ref,array_ref=array_ref,array_fields=list(ARRAY_FIELDS),
   metadata=meta,preprocessing_seconds=time.monotonic()-prep,targets_opened=False,outer_included=False)
  records[pid]=output.save(pid+".sidecar.json",sidecar)
  bank=CladeExchangeInputsV1.attach(base,arrays,record=sidecar)
  values={}
  import torch
  from dataclasses import replace
  for f in fields(bank):
   value=getattr(bank,f.name)
   values[f.name]=tuple(v.pin_memory() for v in value) if isinstance(value,tuple) else value.pin_memory()
  banks[pid]=replace(bank,**values);parents[pid]=item["scoring"]
  print(json.dumps(dict(stage="H039_observation_witness_parent",parent_id=pid,native_holds=meta["native_holds"],
   preprocessing_seconds=sidecar["preprocessing_seconds"],owned_output_bytes=output.used)),flush=True)
  del item,base,bank,arrays,values,rows
  execution_guard()
 manifest=output.save("all40_observation_witness_sidecars.json",dict(
  schema="directed_clade_exchange_full_observation_sidecars_v1",builder_ref=builder_ref,
  observation_cache_ref=boundary.record(cache_manifest_ref),entry_refs=entry_refs,source_refs=source_refs,
  parent_ids=list(PARENTS),parents=records,all40_observed_banks_complete=True,targets_opened=False,
  outer_included=False,raw_witness_array_bytes=raw_witness_bytes,preprocessing_seconds=time.monotonic()-started))
 return banks,parents,manifest,dict(raw_witness_array_bytes=raw_witness_bytes,
  preprocessing_seconds=time.monotonic()-started,
  parent_native_holds={p:int(b.sample_index.shape[0]) for p,b in banks.items()},
  parent_readout_rows={p:int(b.readout_hold.numel()) for p,b in banks.items()})


def main():
 p=argparse.ArgumentParser(description=__doc__);p.parse_args()
 print(json.dumps(dict(status="source_only_root_callable_witness_builder",source=SOURCE,TRAIN=32,VAL=8,
  sentinel_count=8,actual_arrays_read=False,execution_authorized=False)))
if __name__=="__main__":main()
