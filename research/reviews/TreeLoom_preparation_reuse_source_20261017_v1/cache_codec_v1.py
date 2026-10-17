"""Pickle-free exact CPU observation-bundle archive; no truth or model state.

ZIP_STORED entries contain owned C-order NPY tensors plus bounded JSON. The
publication controller supplies the actual charged Writer/guard; this module
never reserves, closes, launches or silently evicts an artifact.
"""
from __future__ import annotations
from dataclasses import fields, replace
import hashlib
import io
import json
import math
import zipfile
import numpy as np
import torch
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as f
from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagsV1
from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1, _LEVEL_FIELDS

SCHEMA="TreeLoom_exact_observed_preparation_archive_v1"
MAX_ARRAYS=16384
MAX_JSON=64*2**20
ALLOWED_DTYPES={"|b1","|i1","|u1","<i2","<u2","<i4","<u4","<i8","<u8","<f4","<f8"}
BUNDLE_FIELDS=("parent_id","split","sequence_length_bp","native_holds","original_sites",
 "focal_identity_sha256","observation_ref","focal_registry_ref","native_preparation_ref")
CUT_META=("group_offsets","episode_of_hold","native_holds","haplotypes","native_nodes","source_fingerprints")


def need(ok,message):
    if not ok:raise ValueError(message)


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def digest_array(array):
    need(array.flags.c_contiguous and not array.dtype.hasobject,"Owned numeric C-order tensor bytes")
    h=hashlib.sha256()
    h.update(memoryview(array.reshape(-1).view(np.uint8)))
    return h.hexdigest()


def identity_key_v1(parent_id,item,preparation_source_manifest_ref):
    need(parent_id in f.TRAIN,"Original TRAIN parent only")
    value=dict(schema=SCHEMA,parent_id=parent_id,
        observation_ref=item["observation_ref"],focal_registry_ref=item["focal_registry_ref"],
        native_preparation_ref=item["native_preparation_ref"],
        preparation_source_manifest_ref=preparation_source_manifest_ref,
        frozen_genealogy=f.GENEALOGY,haplotypes=1600,focals=10000,
        reference_counts=[200,200,200],class_order=["EUR","EAS","AFR"],
        age_split_generations=350.,max_nodes=400000000,max_holds=100000,
        max_episode_payload_bytes=8*2**30)
    return hashlib.sha256(encoded(value)).hexdigest(),value


def tensors_v1(bundle):
    values={}
    level_counts={}
    for field in fields(bundle.bank):
        value=getattr(bundle.bank,field.name)
        if field.name in _LEVEL_FIELDS:
            level_counts[field.name]=len(value)
            for at,tensor in enumerate(value):values["bank."+field.name+"."+str(at)]=tensor
        else:values["bank."+field.name]=value
    values["episodes.node_flags"]=bundle.episodes.flags.node_flags
    for name in ("member_node","member_hold","member_episode","episode_size","hold_span"):
        values["episodes."+name]=getattr(bundle.episodes,name)
    cut_tensor_fields=[]
    for field in fields(bundle.cut):
        value=getattr(bundle.cut,field.name)
        if isinstance(value,torch.Tensor):
            cut_tensor_fields.append(field.name);values["cut."+field.name]=value
    values["focal_hold"]=bundle.focal_hold
    values["focal_sample_column"]=bundle.focal_sample_column
    need(0<len(values)<=MAX_ARRAYS,"Bounded original tensor field count")
    for value in values.values():
        need(isinstance(value,torch.Tensor) and value.device.type=="cpu"
            and not value.requires_grad and value.is_contiguous(),"Only immutable prepared CPU tensors")
    return values,level_counts,cut_tensor_fields


def describe_v1(bundle,metadata,key,identity):
    bundle.validate()
    need(key==hashlib.sha256(encoded(identity)).hexdigest()
        and bundle.parent_id==identity["parent_id"]
        and all(getattr(bundle,name)==identity[name]for name in
            ("observation_ref","focal_registry_ref","native_preparation_ref")),
        "Bundle exactly belongs to immutable observation/focal/source key")
    need(metadata["observation_only"]is True and metadata["private_truth_opened"]is False
        and metadata["model_executed"]is False,"Observation preparation only; no target/model payload")
    values,levels,cut_fields=tensors_v1(bundle)
    arrays={name:tensor.detach().numpy()for name,tensor in values.items()}
    records=[]
    for at,(name,array)in enumerate(arrays.items()):
        need(array.dtype.str in ALLOWED_DTYPES and array.ndim<=8,"Original numeric dtype and bounded rank")
        records.append(dict(name=name,entry="a%05d.npy"%at,dtype=array.dtype.str,
            shape=list(array.shape),nbytes=int(array.nbytes),sha256=digest_array(array)))
    manifest=dict(schema=SCHEMA,key=key,identity=identity,
        bundle={name:getattr(bundle,name)for name in BUNDLE_FIELDS},
        bank_level_counts=levels,cut_tensor_fields=cut_fields,
        cut_metadata={name:getattr(bundle.cut,name)for name in CUT_META},
        metadata=metadata,arrays=records,truth_payload_included=False,
        model_optimizer_rng_or_checkpoint_included=False)
    raw=encoded(manifest)
    need(len(raw)<=MAX_JSON,"Bounded complete recipe JSON")
    nbytes=sum(record["nbytes"]for record in records)
    # Each NPY header+two ZIP directory records+name fits4096 bytes. Manifest
    # shares that allowance; EOCD has an additional64KiB. ZIP64 is supported.
    bound=nbytes+len(raw)+4096*(len(records)+1)+65536
    return manifest,arrays,dict(tensor_nbytes=nbytes,recipe_json_bytes=len(raw),
        array_count=len(records),serialized_upper_bound_bytes=bound,
        per_entry_charge_upper_bound_bytes=((bound+4095)//4096)*4096+2**20)


def write_archive_v1(fileobj,bundle,metadata,key,identity):
    manifest,arrays,measurement=describe_v1(bundle,metadata,key,identity)
    with zipfile.ZipFile(fileobj,"w",compression=zipfile.ZIP_STORED,allowZip64=True)as archive:
        archive.writestr("manifest.json",encoded(manifest))
        for record in manifest["arrays"]:
            array=arrays[record["name"]]
            header=io.BytesIO()
            np.lib.format.write_array_header_1_0(header,np.lib.format.header_data_from_array_1_0(array))
            need(header.tell()<=1024 and len(record["entry"])<=256,"Conservative NPY/ZIP header bound")
            with archive.open(record["entry"],"w",force_zip64=True)as stream:
                np.lib.format.write_array(stream,array,allow_pickle=False)
    need(fileobj.tell()<=measurement["serialized_upper_bound_bytes"],"Actual complete archive within proved bound")
    return measurement


def restore_v1(fileobj,key,identity,*,max_tensor_bytes,max_archive_bytes):
    need(type(max_tensor_bytes)is int and max_tensor_bytes>0
        and type(max_archive_bytes)is int and max_archive_bytes>0,"Explicit measured replay bounds")
    fileobj.seek(0,2);actual_size=fileobj.tell();fileobj.seek(0)
    need(actual_size<=max_archive_bytes,"Bounded immutable archive bytes")
    with zipfile.ZipFile(fileobj,"r")as archive:
        info=archive.infolist();names=[entry.filename for entry in info]
        need(len(info)==len(set(names)) and 1<len(info)<=MAX_ARRAYS+1
            and all(entry.compress_type==zipfile.ZIP_STORED for entry in info),
            "Exact uncompressed archive, no duplicates or decompression expansion")
        manifest_info=archive.getinfo("manifest.json")
        need(manifest_info.file_size<=MAX_JSON,"Bounded recipe JSON before read")
        manifest=json.loads(archive.read("manifest.json"))
        need(manifest["schema"]==SCHEMA and manifest["key"]==key
            and manifest["identity"]==identity and manifest["truth_payload_included"]is False
            and manifest["model_optimizer_rng_or_checkpoint_included"]is False,
            "Exact observation/focal/source identity; no truth/model state")
        records=manifest["arrays"]
        need(0<len(records)<=MAX_ARRAYS and len({r["name"]for r in records})==len(records)
            and set(names)=={"manifest.json",*(r["entry"]for r in records)},"Exact unique tensor manifest")
        total=0
        for record in records:
            shape=record["shape"];dtype=np.dtype(record["dtype"])
            need(dtype.str in ALLOWED_DTYPES and type(shape)is list and len(shape)<=8
                and all(type(x)is int and x>=0 for x in shape),"Bounded numeric replay array")
            size=math.prod(shape)*dtype.itemsize
            need(type(record["nbytes"])is int and record["nbytes"]==size
                and archive.getinfo(record["entry"]).file_size<=size+4096,"Exact declared tensor bytes")
            total+=size
        need(total<=max_tensor_bytes,"Measured immutable tensor payload fits explicit replay cap")
        values={}
        for record in records:
            with archive.open(record["entry"],"r")as stream:
                # Compare NPY header to authenticated record before allocating.
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError("Unsupported NPY header")
                need(not fortran and list(shape)==record["shape"] and dtype.str==record["dtype"]
                    and archive.getinfo(record["entry"]).file_size==record["nbytes"]+stream.tell(),
                    "Header shape/dtype/length exactly matches measured manifest")
            with archive.open(record["entry"],"r")as stream:
                array=np.lib.format.read_array(stream,allow_pickle=False)
                need(array.nbytes==record["nbytes"] and digest_array(array)==record["sha256"],
                    "Every replayed tensor byte verified before use")
            values[record["name"]]=torch.from_numpy(array)
    bank={}
    for field in fields(MutationRoutedWholeTreeInputsV1):
        if field.name in _LEVEL_FIELDS:
            bank[field.name]=tuple(values["bank."+field.name+"."+str(at)]
                for at in range(manifest["bank_level_counts"][field.name]))
        else:bank[field.name]=values["bank."+field.name]
    bank=MutationRoutedWholeTreeInputsV1(**bank)
    b=manifest["bundle"]
    need(key==hashlib.sha256(encoded(identity)).hexdigest()
        and b["parent_id"]==identity["parent_id"]
        and all(b[name]==identity[name]for name in
            ("observation_ref","focal_registry_ref","native_preparation_ref")),
        "Replay bundle exactly belongs to immutable source key")
    flags=AnonymousCladeFlagsV1(values["episodes.node_flags"],b["native_holds"],1600,bank.parent.numel())
    episodes=OrderedCladeEpisodesV1(flags,*(values["episodes."+name]for name in
        ("member_node","member_hold","member_episode","episode_size","hold_span")))
    cut_meta=manifest["cut_metadata"]
    cut_meta["group_offsets"]=tuple(cut_meta["group_offsets"])
    cut_meta["episode_of_hold"]=tuple(cut_meta["episode_of_hold"])
    cut_meta["source_fingerprints"]=tuple(tuple(pair)for pair in cut_meta["source_fingerprints"])
    cut=f.memory_core.CutPlanV1(**{name:values["cut."+name]for name in manifest["cut_tensor_fields"]},
        **cut_meta,source_stamps=f.memory_core._bank_stamps(bank),tensor_stamps=())
    cut=replace(cut,tensor_stamps=f.memory_core._stamps(cut.tensors()))
    focal_hold,focal_sample=values["focal_hold"],values["focal_sample_column"]
    bundle=f.AmericanTreeLoomWholeParentV1(bank,episodes,cut,b["parent_id"],b["split"],
        b["sequence_length_bp"],b["native_holds"],b["original_sites"],focal_hold,focal_sample,
        b["focal_identity_sha256"],b["observation_ref"],b["focal_registry_ref"],b["native_preparation_ref"],
        f.memory_core._stamps(f._bank_tensors(bank)),f.memory_core._stamps(f._episode_tensors(episodes)),
        f.memory_core._stamps((focal_hold,focal_sample)))
    bundle.validate()
    replay_values,_,_=tensors_v1(bundle)
    need(set(replay_values)=={record["name"]for record in records},"No omitted/extra replay tensor")
    # torch.from_numpy and direct dataclass construction retain the very same
    # already verified bytes. No conversion/reduction occurs during restamping.
    return bundle,manifest["metadata"]
