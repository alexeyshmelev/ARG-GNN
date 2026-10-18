"""Bounded once-only lossless NPY cache candidate; no real bank generation.

Inputs are already prepared observed CPU tensors. Integers may be stored in a
smaller exact dtype and expanded in bounded copies when admitted for reuse.
Branch support sham aliases aligned storage only when all original bytes agree.
"""
from __future__ import annotations
from dataclasses import fields,replace
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA="review_only_exact_observed_preparation_cache_v1"


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path,chunk_bytes=8*2**20):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(chunk_bytes),b""):h.update(block)
    return h.hexdigest()


def chunks(array,chunk_bytes):
    flat=array.reshape(-1)
    width=max(1,chunk_bytes//array.dtype.itemsize)
    for first in range(0,len(flat),width):
        yield first,flat[first:first+width]


def packed_dtype(key,array,chunk_bytes):
    if array.size==0:return array.dtype
    if array.dtype.kind in "iu" and array.dtype.itemsize>2:
        lo=min(int(block.min()) for _,block in chunks(array,chunk_bytes))
        hi=max(int(block.max()) for _,block in chunks(array,chunk_bytes))
        if 0<=lo and hi<=65535:return np.dtype("<u2")
        if -(2**31)<=lo and hi<2**31:return np.dtype("<i4")
    if key.startswith("cut.") and "count" in key and array.dtype==np.float32:
        if all(bool(np.isfinite(block).all() and (block>=0).all() and (block<=65535).all()
                    and np.array_equal(block,block.astype(np.uint16).astype(np.float32)))
               for _,block in chunks(array,chunk_bytes)):
            return np.dtype("<u2")
    return array.dtype


def _plain(value):
    if isinstance(value,tuple):return {"__tuple__":[_plain(v) for v in value]}
    if isinstance(value,list):return [_plain(v) for v in value]
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,np.generic):return value.item()
    need(value is None or type(value) in (str,int,float,bool),"Only bounded scalar cache metadata")
    return value


def _restore(value):
    if isinstance(value,dict):
        if set(value)=={"__tuple__"}:return tuple(_restore(v) for v in value["__tuple__"])
        return {k:_restore(v) for k,v in value.items()}
    if isinstance(value,list):return [_restore(v) for v in value]
    return value


def flatten(bank,episodes,cut):
    import torch
    arrays={};structures={}
    def describe(value,key):
        if torch.is_tensor(value):
            need(value.device.type=="cpu" and not value.requires_grad and value.is_contiguous(),
                 "Only immutable complete contiguous observed CPU tensors")
            arrays[key]=value.detach().numpy()
            return {"array":key}
        if isinstance(value,tuple):
            return {"tuple":[describe(v,key+"."+str(i)) for i,v in enumerate(value)]}
        return {"scalar":_plain(value)}
    structures["bank"]={f.name:describe(getattr(bank,f.name),"bank."+f.name) for f in fields(bank)}
    structures["episodes"]={f.name:describe(getattr(episodes,f.name),"episodes."+f.name)
                            for f in fields(episodes) if f.name!="flags"}
    structures["flags"]={f.name:describe(getattr(episodes.flags,f.name),"flags."+f.name)
                         for f in fields(episodes.flags)}
    structures["cut"]={f.name:describe(getattr(cut,f.name),"cut."+f.name)
                       for f in fields(cut) if f.name not in ("source_stamps","tensor_stamps")}
    del describe
    return arrays,structures


def integer_recipe(bank,arrays,structures):
    """Store height/depth labels instead of four expanded integer index families."""
    n=bank.parent.numel()
    height=np.empty(n,dtype=np.uint16)
    depth=np.zeros(n,dtype=np.uint16)
    need(len(bank.upward_levels)<=65536 and len(bank.downward_levels)<65536,
         "Lossless uint16 level labels need their explicit bound")
    for at,nodes in enumerate(bank.upward_levels):height[nodes.numpy()]=at
    for at,nodes in enumerate(bank.downward_levels,1):depth[nodes.numpy()]=at
    names=("upward_levels","upward_children","upward_parent_rank","downward_levels")
    for name in names:
        for key in tuple(arrays):
            if key.startswith("bank."+name+"."):del arrays[key]
        structures["bank"][name]={"derived_integer":name}
    arrays["recipe.height"]=height;arrays["recipe.depth"]=depth


def rebuild_integer_indices(parent,height,depth):
    import torch
    from . import indices_v1 as ix
    levels=ix.stable_levels(height)
    rank=np.empty(len(parent),dtype=np.int64)
    for level in levels:rank[level]=np.arange(len(level),dtype=np.int64)
    child=np.flatnonzero(parent>=0)
    groups=[None]*len(levels)
    ix.fill_groups(groups,child,height[parent[child]])
    return dict(upward_levels=tuple(torch.from_numpy(v) for v in levels),
        upward_children=tuple(torch.from_numpy(v) for v in groups),
        upward_parent_rank=tuple(torch.from_numpy(rank[parent[v]]) for v in groups),
        downward_levels=tuple(torch.from_numpy(v) for v in ix.stable_levels(depth)[1:]))


def plan_cache(bank,episodes,cut,*,external_arrays=None,chunk_bytes=8*2**20,rebuild_indices=True):
    arrays,structures=flatten(bank,episodes,cut)
    expanded=sum(a.nbytes for a in arrays.values())
    if rebuild_indices:integer_recipe(bank,arrays,structures)
    need(type(chunk_bytes) is int and 0<chunk_bytes<=64*2**20,"Finite cache IO scratch")
    external_arrays=external_arrays or {}
    need(set(external_arrays)<=set(arrays),"Only explicitly existing matching arrays may stay external")
    entries={};payload=0
    for key,a in arrays.items():
        if key=="bank.branch_support_sham":
            true=arrays["bank.branch_support_true"]
            need(a.shape==true.shape and a.dtype==true.dtype and all(
                np.array_equal(block,true.reshape(-1)[first:first+len(block)])
                for first,block in chunks(a,chunk_bytes)),"Only exact equal immutable support storage may alias")
            entries[key]={"alias":"bank.branch_support_true"}
            continue
        dtype=packed_dtype(key,a,chunk_bytes)
        entry=dict(shape=list(a.shape),dtype=a.dtype.str,stored_dtype=dtype.str,
                   payload_bytes=a.size*dtype.itemsize,expanded_bytes=a.nbytes)
        if key in external_arrays:
            path=Path(external_arrays[key]["path"])
            need(path.is_file() and not path.is_symlink() and sha(path,chunk_bytes)==external_arrays[key]["sha256"],
                 "Exact existing external observed NPY required")
            existing=np.load(path,mmap_mode="r",allow_pickle=False)
            need(existing.shape==a.shape and existing.dtype==a.dtype and all(
                np.array_equal(block,existing.reshape(-1)[first:first+len(block)])
                for first,block in chunks(a,chunk_bytes)),"External values must equal every original observed tensor")
            entry.update(external_ref=dict(external_arrays[key]),stored_dtype=a.dtype.str)
        else:
            payload+=entry["payload_bytes"]+4096
        entries[key]=entry
    # Fixed allowance includes file headers, full structure/episode/cut metadata.
    return dict(schema=SCHEMA,arrays=entries,structures=structures,
        expanded_tensor_bytes=expanded,stored_upper_bound_bytes=payload+64*2**20,
        io_chunk_bytes=chunk_bytes,complete_parent=True,arrays_omitted=False,
        rebuild_integer_indices=rebuild_indices,
        read_peak_upper_bound_bytes=expanded+256*bank.parent.numel()+64*2**20,
        support_storage_alias_only=True,source_float_math_changed=False)


def write_cache(directory,bank,episodes,cut,metadata,*,input_refs,max_output_bytes,
                external_arrays=None,chunk_bytes=8*2**20,rebuild_indices=True):
    directory=Path(directory)
    need(not directory.exists() and not directory.is_symlink(),"Once-only fresh cache directory")
    plan=plan_cache(bank,episodes,cut,external_arrays=external_arrays,chunk_bytes=chunk_bytes,rebuild_indices=rebuild_indices)
    need(type(max_output_bytes) is int and plan["stored_upper_bound_bytes"]<=max_output_bytes,
         "Complete parent cache must fit a caller reservation before any output")
    arrays,structures=flatten(bank,episodes,cut)
    if rebuild_indices:integer_recipe(bank,arrays,structures)
    directory.mkdir()
    for key,entry in plan["arrays"].items():
        if "alias" in entry or "external_ref" in entry:continue
        name=key+".npy";path=directory/name
        a=arrays[key];dtype=np.dtype(entry["stored_dtype"])
        if a.size:
            mapped=np.lib.format.open_memmap(path,mode="w+",dtype=dtype,shape=a.shape)
            flat=mapped.reshape(-1)
            for first,block in chunks(a,chunk_bytes):
                stored=block.astype(dtype,copy=False)
                need(np.array_equal(stored.astype(a.dtype),block),"Every packed value must round-trip exactly")
                flat[first:first+len(block)]=stored
            mapped.flush();del flat,mapped
        else:
            with path.open("xb") as stream:np.lib.format.write_array(stream,a.astype(dtype),allow_pickle=False)
        entry["ref"]=dict(path=name,sha256=sha(path,chunk_bytes))
    plan.update(input_refs=input_refs,metadata=_plain(metadata),max_output_bytes=max_output_bytes)
    encoded=(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(encoded)<=64*2**20,"Complete cache metadata cap")
    (directory/"cache.json").write_bytes(encoded)
    need(sum(p.stat().st_size for p in directory.iterdir())<=max_output_bytes,"Actual complete cache output cap")
    return dict(path=str(directory/"cache.json"),sha256=sha(directory/"cache.json",chunk_bytes))


def read_cache(ref,*,max_expanded_bytes,max_stored_bytes,input_refs):
    import torch
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1
    from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagsV1
    from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
    path=Path(ref["path"])
    need(path.stat().st_size<=64*2**20 and sha(path)==ref["sha256"],"Exact bounded cache receipt")
    plan=json.loads(path.read_text())
    need(plan["schema"]==SCHEMA and plan["input_refs"]==input_refs
        and plan["read_peak_upper_bound_bytes"]<=max_expanded_bytes
        and plan["stored_upper_bound_bytes"]<=max_stored_bytes,
        "Full same-input cache requires explicit current RAM/disk admission")
    tensors={};chunk_bytes=plan["io_chunk_bytes"]
    for key,entry in plan["arrays"].items():
        if "alias" in entry:continue
        record=entry.get("external_ref",entry.get("ref"))
        source=Path(record["path"]) if "external_ref" in entry else path.parent/record["path"]
        need(source.is_file() and not source.is_symlink() and sha(source,chunk_bytes)==record["sha256"],
             "Every original or cached observed array must authenticate")
        mapped=np.load(source,mmap_mode="c" if np.prod(entry["shape"]) else None,allow_pickle=False)
        need(list(mapped.shape)==entry["shape"] and mapped.dtype.str==entry["stored_dtype"],
             "Exact original stored array axes/dtype")
        dtype=np.dtype(entry["dtype"])
        if dtype!=mapped.dtype:
            expanded=np.empty(mapped.shape,dtype=dtype)
            target=expanded.reshape(-1)
            for first,block in chunks(mapped,chunk_bytes):
                target[first:first+len(block)]=block
            mapped=expanded
        tensors[key]=torch.from_numpy(mapped)
    for key,entry in plan["arrays"].items():
        if "alias" in entry:tensors[key]=tensors[entry["alias"]]
    def restore(desc):
        if "array" in desc:return tensors[desc["array"]]
        if "tuple" in desc:return tuple(restore(v) for v in desc["tuple"])
        if "derived_integer" in desc:return desc
        return _restore(desc["scalar"])
    structures={name:{k:restore(v) for k,v in values.items()} for name,values in plan["structures"].items()}
    if plan["rebuild_integer_indices"]:
        structures["bank"].update(rebuild_integer_indices(
            tensors["bank.parent"].numpy(),tensors["recipe.height"].numpy(),tensors["recipe.depth"].numpy()))
    bank=MutationRoutedWholeTreeInputsV1(**structures["bank"])
    flags=AnonymousCladeFlagsV1(**structures["flags"])
    episodes=OrderedCladeEpisodesV1(flags=flags,**structures["episodes"])
    cut=core.CutPlanV1(**structures["cut"],source_stamps=core._bank_stamps(bank),tensor_stamps=())
    cut=replace(cut,tensor_stamps=core._stamps(cut.tensors()))
    cut.validate(bank)
    metadata=_restore(plan["metadata"])
    del restore,tensors,structures
    return bank,episodes,cut,metadata


def storage_upper_bound(holds,sites,*,haplotypes=1600,genotype_external=True,rebuild_indices=True):
    """Unary-collapsed complete forest upper bound, without reading real banks."""
    t=int(holds);s=int(sites);h=int(haplotypes)
    need(t>0 and s>=0 and 3<=h<=65535,"Declared complete metadata-only dimensions")
    n=t*(2*h-1);members=t*(h-2);sampled=t*h
    bank=(48 if rebuild_indices else 60)*n+4*sampled+130*s+(0 if genotype_external else s*h)
    episodes=2*n+16*members+4*t
    cut=31*sampled+h
    return dict(holds=t,sites=s,haplotypes=h,packed_node_upper_bound=n,
        compact_bank_bytes=bank,episode_bytes=episodes,cut_bytes=cut,
        metadata_and_header_margin_bytes=64*2**20,
        complete_compact_cache_upper_bound_bytes=bank+episodes+cut+64*2**20,
        # Full original widths, separate support fields and genotype corpus.
        original_width_cache_upper_bound_bytes=100*n+8*sampled+130*s+s*h+
            episodes+89*sampled+64*2**20,
        condition="all original sampled leaves; keep_unary=False; no sampled interiors",
        storage_only_lossless_integer_packing=True,genotype_external=genotype_external,
        rebuild_integer_indices=rebuild_indices)
