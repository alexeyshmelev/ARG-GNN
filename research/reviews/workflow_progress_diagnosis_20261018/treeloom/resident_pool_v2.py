"""Lossless prepare-once compact RAM pool; no disk cache or launch API."""
from __future__ import annotations
from dataclasses import fields,replace
import copy
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from . import cache_v1 as codec

GIB=2**30
COMPACT_POOL_BYTES=352*GIB
WORKING_BYTES=128*GIB
MAX_RSS_BYTES=512*GIB
OUTER_TENSORS=("focal_hold","focal_sample_column")
STAMPS=("bank_stamps","episode_stamps","focal_stamps")


def need(ok,message):
    if not ok:raise ValueError(message)


def array_hash(array,chunk_bytes=8*2**20):
    result=hashlib.sha256()
    result.update(array.dtype.str.encode());result.update(str(array.shape).encode())
    for _,block in codec.chunks(array,chunk_bytes):
        result.update(memoryview(block).cast("B"))
    return result.hexdigest()


def exact_signature(bundle,metadata,resource_check=lambda stage:None):
    arrays,structures=codec.flatten(bundle.bank,bundle.episodes,bundle.cut)
    for name in OUTER_TENSORS:arrays["outer."+name]=getattr(bundle,name).numpy()
    outer={f.name:codec._plain(getattr(bundle,f.name)) for f in fields(bundle)
           if f.name not in ("bank","episodes","cut",*OUTER_TENSORS,*STAMPS)}
    hashes={}
    for key,array in arrays.items():
        resource_check("signature_"+key)
        hashes[key]=array_hash(array)
    # Structures retain scalars, including all cut source fingerprints.
    return dict(arrays=hashes,structures=structures,outer=outer,metadata=codec._plain(metadata))


def pack_bundle(bundle,metadata,*,input_refs,external_arrays=None,resource_check=lambda stage:None):
    """Independent compact buffers: no original expanded bank is retained."""
    plan=codec.plan_cache(bundle.bank,bundle.episodes,bundle.cut,
        external_arrays=external_arrays,rebuild_indices=True)
    arrays,_=codec.flatten(bundle.bank,bundle.episodes,bundle.cut)
    codec.integer_recipe(bundle.bank,arrays,{"bank":{}})
    structures=plan["structures"]
    outer={}
    for f in fields(bundle):
        if f.name in ("bank","episodes","cut",*STAMPS):continue
        value=getattr(bundle,f.name)
        if f.name in OUTER_TENSORS:
            key="outer."+f.name;arrays[key]=value.detach().numpy()
            dtype=codec.packed_dtype(key,arrays[key],plan["io_chunk_bytes"])
            plan["arrays"][key]=dict(shape=list(arrays[key].shape),dtype=arrays[key].dtype.str,
                                    stored_dtype=dtype.str,expanded_bytes=arrays[key].nbytes)
            outer[f.name]={"array":key}
        else:outer[f.name]={"scalar":codec._plain(value)}
    stored={};digests={}
    for key,entry in plan["arrays"].items():
        if "alias" in entry or "external_ref" in entry:continue
        resource_check("pack_"+key)
        source=arrays[key];target=np.empty(source.shape,dtype=entry["stored_dtype"])
        flat=target.reshape(-1)
        for first,block in codec.chunks(source,plan["io_chunk_bytes"]):
            resource_check("pack_"+key)
            packed=block.astype(target.dtype,copy=False)
            need(np.array_equal(packed.astype(source.dtype),block),"Every compact RAM value must round-trip exactly")
            flat[first:first+len(block)]=packed
        target.flags.writeable=False
        stored[key]=target;digests[key]=array_hash(target,plan["io_chunk_bytes"])
    expanded=plan["expanded_tensor_bytes"]+sum(arrays["outer."+name].nbytes for name in OUTER_TENSORS)
    charged=sum(a.nbytes for a in stored.values())+64*2**20
    record=dict(plan=plan,stored=stored,hashes=digests,outer=outer,bundle_type=type(bundle),
                metadata=codec._plain(metadata),input_refs=copy.deepcopy(input_refs),
                buffer_ids={k:id(v) for k,v in stored.items()},
                external_stamps={k:(Path(v["external_ref"]["path"]).stat().st_dev,
                    Path(v["external_ref"]["path"]).stat().st_ino,
                    Path(v["external_ref"]["path"]).stat().st_size,
                    Path(v["external_ref"]["path"]).stat().st_mtime_ns)
                    for k,v in plan["arrays"].items() if "external_ref" in v},
                charged_bytes=charged,payload_bytes=sum(a.nbytes for a in stored.values()),
                expanded_tensor_bytes=expanded,
                read_peak_upper_bound_bytes=expanded+256*bundle.bank.parent.numel()+64*2**20)
    # Explicitly discard source dictionaries; the record contains compact copies only.
    del arrays,structures,source,target,flat
    return record


def restore_bundle(record,*,input_refs,resource_check=lambda stage:None,verify_integrity=False):
    import torch
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1
    from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagsV1
    from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
    need(record["input_refs"]==input_refs,"Exact original complete-parent observed input refs required")
    plan=record["plan"];tensors={}
    for key,entry in plan["arrays"].items():
        if "alias" in entry:continue
        resource_check("restore_"+key)
        if "external_ref" in entry:
            ref=entry["external_ref"];path=Path(ref["path"])
            stat=path.stat()
            need(path.is_file() and not path.is_symlink() and
                 (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==record["external_stamps"][key],
                 "Authenticated external observed genotype file identity must remain unchanged")
            if verify_integrity:
                need(codec.sha(path)==ref["sha256"],"External observed genotype must authenticate")
            source=np.load(path,mmap_mode="r" if np.prod(entry["shape"]) else None,allow_pickle=False)
        else:
            source=record["stored"][key]
            if verify_integrity:
                need(array_hash(source)==record["hashes"][key],"Immutable compact RAM buffers must authenticate")
            need(id(source)==record["buffer_ids"][key] and not source.flags.writeable,
                 "Routine reuse requires the same private readonly compact master")
        need(list(source.shape)==entry["shape"] and source.dtype.str==entry["stored_dtype"],
             "Exact compact RAM shape/dtype")
        target=np.empty(source.shape,dtype=entry["dtype"]);flat=target.reshape(-1)
        for first,block in codec.chunks(source,plan["io_chunk_bytes"]):
            resource_check("restore_"+key);flat[first:first+len(block)]=block
        tensors[key]=torch.from_numpy(target)
    for key,entry in plan["arrays"].items():
        if "alias" in entry:tensors[key]=tensors[entry["alias"]]
    def restore(desc):
        if "array" in desc:return tensors[desc["array"]]
        if "tuple" in desc:return tuple(restore(v) for v in desc["tuple"])
        if "derived_integer" in desc:return desc
        return codec._restore(desc["scalar"])
    structures={name:{key:restore(desc) for key,desc in values.items()}
                for name,values in plan["structures"].items()}
    resource_check("restore_integer_index_families")
    structures["bank"].update(codec.rebuild_integer_indices(tensors["bank.parent"].numpy(),
        tensors["recipe.height"].numpy(),tensors["recipe.depth"].numpy()))
    bank=MutationRoutedWholeTreeInputsV1(**structures["bank"])
    flags=AnonymousCladeFlagsV1(**structures["flags"])
    episodes=OrderedCladeEpisodesV1(flags=flags,**structures["episodes"])
    cut=core.CutPlanV1(**structures["cut"],source_stamps=core._bank_stamps(bank),tensor_stamps=())
    cut=replace(cut,tensor_stamps=core._stamps(cut.tensors()))
    outer={key:restore(desc) for key,desc in record["outer"].items()}
    from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_preparation_v1 as original
    bundle=record["bundle_type"](bank=bank,episodes=episodes,cut=cut,**outer,
        bank_stamps=core._stamps(original._bank_tensors(bank)),
        episode_stamps=core._stamps(original._episode_tensors(episodes)),
        focal_stamps=core._stamps((outer["focal_hold"],outer["focal_sample_column"])))
    bundle.validate()
    metadata=codec._restore(record["metadata"])
    del restore,tensors,structures,source,target,flat
    return bundle,metadata


class CompactParentPool:
    """No eviction or cold rebuild: each expected complete parent prepares once."""
    def __init__(self,*,parent_bounds,input_refs,external_arrays,max_cache_bytes,
                 max_rss_bytes,preparation_headroom_bytes,rss_bytes,resource_check,event,
                 pack=pack_bundle,restore=restore_bundle):
        need(type(max_cache_bytes) is int and sum(parent_bounds.values())<=max_cache_bytes,
             "Every complete parent compact bound must fit the RAM pool before preparation")
        need(max_cache_bytes+preparation_headroom_bytes<max_rss_bytes,
             "Compact pool plus full parent working reserve must fit process RSS cap")
        self.bounds=dict(parent_bounds);self.inputs=copy.deepcopy(input_refs)
        self.external=copy.deepcopy(external_arrays);self.max_cache_bytes=max_cache_bytes
        self.max_rss_bytes=max_rss_bytes;self.reserve=preparation_headroom_bytes
        self.rss=rss_bytes;self.check=resource_check;self.event=event
        self.pack=pack;self.restore=restore;self.records={}
        self.preparations={};self.restorations={};self.active_expanded_bytes={};self.charged_bytes=0

    def admit(self,reserve,stage):
        self.check(stage)
        current=self.rss()
        need(type(current) is int and current>=0 and current+reserve<=self.max_rss_bytes,
             "Complete parent RAM headroom unavailable; never evict or rebuild the compact pool")

    def load(self,parent,prepare):
        need(parent in self.bounds,"Only exact registered complete parents may enter the pool")
        if parent not in self.records:
            need(parent not in self.preparations,"Failed once-only preparation cannot be silently retried")
            need(self.charged_bytes+self.bounds[parent]<=self.max_cache_bytes,
                 "Complete new parent reservation must fit compact pool before preparation")
            self.admit(self.reserve,"compact_pool_before_once_only_prepare")
            self.preparations[parent]=1
            bundle,metadata=prepare()
            record=self.pack(bundle,metadata,input_refs=self.inputs[parent],
                external_arrays=self.external.get(parent),resource_check=self.check)
            need(record["charged_bytes"]<=self.bounds[parent],
                 "Actual complete parent compact record exceeds its explicit metadata-only bound")
            need(self.rss()<=self.max_rss_bytes,"Actual prepare-and-pack RSS cap exceeded")
            self.records[parent]=record;self.charged_bytes+=record["charged_bytes"]
            self.event("compact_parent_sealed",dict(parent_id=parent,parents=len(self.records),
                       compact_charged_bytes=record["charged_bytes"],pool_bytes=self.charged_bytes,
                       preparation_count=1,disk_cache_bytes=0))
            return bundle,metadata
        record=self.records[parent]
        if self.rss()+record["read_peak_upper_bound_bytes"]>self.max_rss_bytes:gc.collect()
        self.admit(record["read_peak_upper_bound_bytes"],"compact_pool_before_exact_restore")
        bundle,metadata=self.restore(record,input_refs=self.inputs[parent],resource_check=self.check)
        need(self.rss()<=self.max_rss_bytes,"Actual complete-parent restore RSS cap exceeded")
        import torch
        from arggnn.benchmarking.treeloom_training import _cpu_bytes
        self.active_expanded_bytes[parent]=_cpu_bytes(bundle,torch)
        self.restorations[parent]=self.restorations.get(parent,0)+1
        return bundle,metadata

    def require_all_prepared(self):
        need(set(self.records)==set(self.bounds) and set(self.preparations)==set(self.bounds)
             and set(self.preparations.values())=={1},"All original observations must prepare and seal exactly once")

    def receipt(self):
        return dict(schema="exact_compact_RAM_pool_v2",parents=len(self.records),
                    charged_bytes=self.charged_bytes,cap_bytes=self.max_cache_bytes,
                    preparations=dict(self.preparations),restorations=dict(self.restorations),
                    all_original_data_retained=True,cold_rebuilds=0,disk_cache_bytes=0)
