"""Zero-copy logical archive concatenation for the unchanged author trainer.

Stock native Rust archives contain uncompressed numeric NPY members. Their
original bytes are memory mapped in place. Only indexed training rows are copied
and cast exactly as the released dense loader does. TensorDataset and the full
original trainer bytecode are retained, including global shuffle order.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import redirect_stdout, redirect_stderr
import logging
from pathlib import Path
import struct
import sys
import time
import types
import zipfile
from . import european_native_models as m
from .. import expanded_native_argmix_v1 as native_archives

SCHEMA="european_release_argmix_sharded_archive_v1"


def stored_npy_mapping(path, member):
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(member+".npy")
        m._need(info.compress_type == zipfile.ZIP_STORED
            and info.compress_size == info.file_size and not info.flag_bits & 1,
            "Original uncompressed unencrypted native NPY members required")
    with Path(path).open("rb") as stream:
        stream.seek(info.header_offset)
        header=stream.read(30)
        m._need(len(header)==30,"Complete native ZIP local header required")
        fields=struct.unpack("<4s5H3I2H",header)
        m._need(fields[0] == b"PK\x03\x04","Actual native ZIP member header required")
        data_start=info.header_offset+30+fields[-2]+fields[-1]
        stream.seek(data_start)
        version=np.lib.format.read_magic(stream)
        if version==(1,0):
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):
            shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Original native numeric NPY v1/v2 required")
        offset=stream.tell()
    m._need(not fortran and not dtype.hasobject
        and offset+int(np.prod(shape))*dtype.itemsize == data_start+info.file_size,
        "Exact contiguous native numeric member payload required")
    return np.memmap(path,dtype=dtype,mode="r",offset=offset,shape=shape,order="C")


class LogicalTensor:
    """Minimal exact TensorDataset tensor protocol, no tensor corpus duplication."""
    def __init__(self,arrays,dtype):
        import torch
        self.arrays=tuple(arrays)
        self.ends=[]
        total=0
        for array in arrays:
            total+=array.shape[0]
            self.ends.append(total)
        self.shape=torch.Size((total,*arrays[0].shape[1:]))
        self.dtype=dtype
    def size(self,dim=None):
        return self.shape if dim is None else self.shape[dim]
    def __len__(self):
        return self.shape[0]
    def __getitem__(self,index):
        import operator
        import numpy as np
        import torch
        index=operator.index(index)
        if index<0:
            index+=len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard=bisect_right(self.ends,index)
        previous=0 if shard==0 else self.ends[shard-1]
        # The author dense loader holds these same native values in RAM and
        # converts row_mask to bool. Copy only this row for a writable tensor.
        array=np.array(self.arrays[shard][index-previous],copy=True,order="C")
        return torch.from_numpy(array).to(dtype=self.dtype)


def load_sharded_split(path,*,expected_split_code,guard,execution_check):
    import torch
    manifest=m._read(m._ref(m._path(path)))
    m._need(manifest.get("schema")==SCHEMA
        and manifest["split_code"]==expected_split_code
        and manifest["archive_refs"] and manifest["parent_order"]
        and len(manifest["parent_order"])==len(manifest["archive_refs"]),
        "Exact complete logical native split manifest required")
    layouts=[]
    arrays={key:[] for key in native_archives.ARCHIVE_FIELDS}
    for ref in manifest["archive_refs"]:
        execution_check()
        guard(0)
        actual=m._checked(ref)
        layout=native_archives.archive_layout_v1(actual)
        m._need(layout["metadata"][1]==expected_split_code
            and (not layouts or layout["metadata"]==layouts[0]["metadata"]),
            "Every native shard keeps identical source split and token metadata")
        layouts.append(layout)
        for key in arrays:
            arrays[key].append(stored_npy_mapping(actual,key))
    m._need(manifest["total_chunks"]==sum(v["chunks"] for v in layouts),
            "Logical concatenation must retain every native chunk")
    metadata=torch.as_tensor(layouts[0]["metadata"],dtype=torch.long)
    vocab_size=int(metadata[11])
    # Released loader checks the full input-ID range. Preserve that check with
    # sequential bounded views, without materializing all arrays in RAM.
    for shard in arrays["input_ids"]:
        for first in range(0,len(shard),64):
            execution_check()
            guard(0)
            block=shard[first:first+64]
            m._need(block.size>0 and int(block.min())>=0 and int(block.max())<vocab_size,
                    "Native input token IDs must remain in the original vocabulary")
    dtypes=dict(input_ids=torch.long,tmrca=torch.int16,labels=torch.long,
                row_mask=torch.bool,cm_features=torch.float32)
    result={key:LogicalTensor(value,dtypes[key]) for key,value in arrays.items()}
    return dict(result,metadata=metadata,pad_id=int(metadata[7]),vocab_size=vocab_size,
        format_version=int(metadata[0]),base_encoder="tmrca_transformer",
        num_chunks=manifest["total_chunks"])


def write_manifest(path,parent_order,archive_refs,*,split_code,guard):
    layouts=[native_archives.archive_layout_v1(m._checked(ref)) for ref in archive_refs]
    m._need(layouts and len(archive_refs)==len(parent_order)
        and all(v["metadata"]==layouts[0]["metadata"] for v in layouts)
        and layouts[0]["metadata"][1]==split_code,
        "All complete native shards in frozen parent/global chunk order required")
    return m._json(path,dict(schema=SCHEMA,parent_order=list(parent_order),
        archive_refs=archive_refs,split_code=split_code,
        total_chunks=sum(v["chunks"] for v in layouts),
        member_layouts=layouts,global_row_order="parent registry order, then original native chunk order",
        arrays_rewritten=False,training_examples_removed=False),guard)


def run_original_main(author_module,argv,output,*,guard,execution_check):
    """Private unchanged main code object, with only the archive IO name rebound."""
    namespace=dict(author_module.main.__globals__)
    namespace["_load_gt_gru_split_pt"]=lambda path,expected_split_code:load_sharded_split(
        path,expected_split_code=expected_split_code,guard=guard,
        execution_check=execution_check)
    main=types.FunctionType(author_module.main.__code__,namespace,
        author_module.main.__name__,author_module.main.__defaults__,
        author_module.main.__closure__)
    m._need(main.__code__ is author_module.main.__code__,
            "Original native trainer main bytecode must remain unchanged")
    old=sys.argv[:]
    sys.argv=[str(Path(author_module.__file__)),*map(str,argv)]
    loggers=[logging.getLogger()]+[value for value in logging.Logger.manager.loggerDict.values()
                                  if isinstance(value,logging.Logger)]
    saved={value:(value.handlers[:],value.level,value.propagate,value.disabled)
           for value in loggers}
    old_handlers={handler for value in loggers for handler in value.handlers}
    for value in loggers:
        value.handlers=[]
    started=time.monotonic()
    code=0
    log=output/"train_joint_model.log"
    try:
        with log.open("x") as stream,redirect_stdout(stream),redirect_stderr(stream):
            try:
                main()
            except SystemExit as error:
                code=error.code if isinstance(error.code,int) else (0 if error.code is None else 1)
                if code:
                    raise
    except BaseException:
        code=1
        raise
    finally:
        sys.argv=old
        active=[logging.getLogger()]+[value for value in logging.Logger.manager.loggerDict.values()
                                     if isinstance(value,logging.Logger)]
        for value in active:
            for handler in value.handlers[:]:
                value.removeHandler(handler)
                if handler not in old_handlers:
                    target = getattr(handler,"stream",None)
                    if target is None or not getattr(target,"closed",False):
                        handler.flush()
                    handler.close()
            if value in saved:
                value.handlers,value.level,value.propagate,value.disabled=saved[value]
        m._json(output/"train_joint_model.command.json",dict(
            source_ref=m._ref(author_module.__file__),argv=list(map(str,argv)),
            exit_code=code,elapsed_seconds=time.monotonic()-started,
            original_main_code_object_unchanged=True,
            replaced_global_names=["_load_gt_gru_split_pt"],
            adaptation="memory mapped original stored NPZ shards, logical concatenation",
            dense_merged_or_extracted_tensor_corpus_written=False,
            original_TensorDataset_DataLoader_seed_shuffle_loss_optimizer_selection=True),guard)
