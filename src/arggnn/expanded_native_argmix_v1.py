"""H038 full952 native current-release ARGMix; guarded callable stages, inactive default."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
from . import official_argmix_biological as native
from .full_panel_native_baseline_contracts_v1 import validate_observed_roster
from .mutation_routed_full_panel_observations_v1 import validate_registry_v1,join_full_rosters_v1
from .mutation_routed_parent_loader_v1 import _scoring_grid_v1

ROOT=native.ROOT
SOURCE="src/arggnn/expanded_native_argmix_v1.py"
DRAFT="research/reviews/expanded_native_argmix_v1_author/configdraft.json"
GENERATION_REF={"path":"configs/expanded_full_panel_v1.json","sha256":"662ba8e4613ce85b9aaec9909e873c9cddf9c62672b895e7d9e02f851d518d08"}
REGISTRATION_REF={"path":"research/reviews/expanded_full_panel_v1_root/registration.json","sha256":"3e3f5fab53cc9fb6142e0b11926deb9a2f8e91ddabafc9b5e2f9c5d19c2db9ac"}
PARENT_SPLITS={**{"ap_expanded_train_%03d"%i:"train" for i in range(32)},
               **{"ap_expanded_val_%03d"%i:"val" for i in range(8)}}
SOURCE_PATHS=(SOURCE,"src/arggnn/official_argmix_biological.py",
    "src/arggnn/full_panel_native_baseline_contracts_v1.py",
    "src/arggnn/mutation_routed_full_panel_observations_v1.py",
    "src/arggnn/mutation_routed_parent_loader_v1.py")
ARCHIVE_FIELDS=("input_ids","tmrca","labels","row_mask","cm_features")
CLASS_ORDER=["ANA","CHG","WHG","EHG"]


def record_v1(path):
    path=native.local(path)
    return {"path":str(path.relative_to(ROOT)),"sha256":native.sha(path)}


class ReaderV1:
    def read(self,record,*,observed=False):
        return native.checked(record,observed=observed).read_bytes()
    def decode_json(self,record,*,observed=False):
        return json.loads(self.read(record,observed=observed))


def load_config_v1(record,reader=None):
    reader=reader or ReaderV1();cfg=reader.decode_json(record)
    if (cfg.get("schema")!="expanded_native_argmix_v1" or cfg.get("hypothesis_id")!="H038"
            or cfg.get("parent_splits")!=PARENT_SPLITS or cfg.get("class_order")!=CLASS_ORDER
            or cfg.get("native_model")!=native.NATIVE_MODEL or cfg.get("joint_model")!=native.JOINT_MODEL
            or cfg.get("source_seed")!=50 or cfg.get("published_April_paper_reproduction") is not False
            or cfg.get("generation_ref")!=GENERATION_REF or cfg.get("registration_ref")!=REGISTRATION_REF
            or cfg.get("runtime_prefix")!=str(native.PREFIX.relative_to(ROOT))
            or cfg.get("source_copy")!=str(native.SOURCE.relative_to(ROOT))
            or cfg.get("source_tree_sha256")!=native.SOURCE_TREE_SHA
            or cfg.get("inference_only_exclude_edge_trees") is not False
            or cfg.get("normalization_length_bp")!=2000000):
        raise ValueError("Exact full32TRAIN/8VAL native current-release ARGMix policy required")
    sources=cfg.get("owned_code_sha256")
    if not isinstance(sources,dict) or set(sources)!=set(SOURCE_PATHS):
        raise ValueError("Exact reviewed adapter/helper Python source closure required")
    for path,digest in sources.items():
        native.checked({"path":path,"sha256":digest},observed=True)
    generation=reader.decode_json(GENERATION_REF);registration=reader.decode_json(REGISTRATION_REF)
    roles=validate_registry_v1(generation["parent_registry"])
    if (registration["parent_registry"]!=generation["parent_registry"]
            or registration["config_ref"]!=GENERATION_REF
            or roles["train"]!=[p for p,s in PARENT_SPLITS.items() if s=="train"]
            or roles["val"]!=[p for p,s in PARENT_SPLITS.items() if s=="val"]):
        raise ValueError("Frozen H038 development partition differs")
    cfg=dict(cfg,config_ref=record,output_root=generation["output_root"])
    return cfg


def _ready(cfg,guard):
    if cfg.get("execution_guard_ready") is not True or not callable(guard):
        raise ValueError("Root-admitted execution config and write/boundary guard required")
    guard(0)


def _new_output(output):
    path=native.local(output,observed=True);path.mkdir(parents=True,exist_ok=False)
    return path


def _write(path,payload,guard):
    guard(len(payload)+8192)
    with Path(path).open("xb") as stream:stream.write(payload)
    return record_v1(path)


def _json(path,value,guard):
    return _write(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode(),guard)


def _npz(path,value,guard):
    import numpy as np
    stream=io.BytesIO();np.savez_compressed(stream,**value)
    return _write(path,stream.getvalue(),guard)


def validate_slot_metadata_v1(rows):
    checked=validate_observed_roster(rows,require_argmix_native_ids=True)
    return [(row["inferred_tree_node_id"],
        CLASS_ORDER[row["observed_reference_class"]] if row["role"]=="pure_source_reference" else "QUERY",
        float(row["age_generations"])) for row in rows],checked


def prepare_observed_parent_v1(parent_id,entry_ref,output,cfg,*,guard,reader=None):
    """One completed registered TRAIN/VAL; full observed sample/age/native-node join."""
    _ready(cfg,guard);reader=reader or ReaderV1()
    if parent_id not in PARENT_SPLITS:raise ValueError("Outer/unknown parents are not accepted")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Observed/native parent preparation is CPU-only")
    expected=cfg["output_root"]+"/"+parent_id+"/full_panel_observed_entry.json"
    if entry_ref["path"]!=expected:raise ValueError("Exact registered observed entry path required")
    entry=reader.decode_json(entry_ref,observed=True);refs=entry["refs"]
    if (entry["schema"]!="mutation_routed_full_panel_observed_entry_v1" or entry["parent_id"]!=parent_id
            or entry["split"]!=PARENT_SPLITS[parent_id] or entry["observations_only"] is not True
            or entry["truth_or_oracle_read"] is not False or entry["samples"]!=952
            or entry["query_haplotypes"]!=686 or entry["pure_reference_haplotypes"]!=266):
        raise ValueError("Full952 observed-only parent entry required")
    source=reader.decode_json(refs["source_roster"],observed=True)
    rows=reader.decode_json(refs["derived_roster"],observed=True)
    inference=reader.decode_json(refs["inference_receipt"],observed=True)
    original=reader.decode_json(entry["original_native_inference_receipt"],observed=True)
    if (inference["parent_id"]!=parent_id or inference["truth_or_oracle_read"] is not False
            or inference["original_observed_sample_manifest"]!=refs["source_roster"]
            or inference["observed_sample_manifest"]!=refs["derived_roster"]
            or inference["inferred_tree"]!=refs["native_tree"]
            or inference["native_inference_receipt"]!=entry["original_native_inference_receipt"]
            or original["parent_id"]!=parent_id or original["status"]!="post_five_iterations_inferred"
            or original["truth_or_oracle_read"] is not False
            or original["inferred_tree"]!=refs["native_tree"]
            or original["post_five_iterations"]["condition"]!="post_five_size_iterations_primary"
            or inference["coordinate_contract"]!=entry["coordinate_contract"]
            or inference["native_sample_dates_unchanged"] is not True):
        raise ValueError("Bound post-five-size-iteration observed lineage required")
    import numpy as np
    import tskit
    ts=tskit.load(native.checked(refs["native_tree"],observed=True))
    contract=entry["coordinate_contract"];coverage=np.asarray(contract["coverage_original_bp0"],dtype=np.float64)
    if (coverage.shape!=(2,) or not 0<=coverage[0]<coverage[1]<=2000000
            or contract["local_to_original_offset_bp"]!=coverage[0]
            or contract["normalization_length_bp"]!=2000000
            or ts.sequence_length!=coverage[1]-coverage[0]):
        raise ValueError("Native local/original physical coordinate identity required")
    tables=ts.tables
    if (tables.metadata or ts.num_individuals or ts.num_populations or ts.num_migrations
            or np.any(tables.nodes.population!=-1) or np.any(tables.nodes.individual!=-1)
            or any(len(t.metadata) for t in (tables.nodes,tables.edges,tables.sites,tables.mutations,
                                           tables.individuals,tables.populations,tables.migrations))):
        raise ValueError("Native observed tree must contain no ancestry/population annotations")
    join_full_rosters_v1(source,rows,inference["sample_index_to_inferred_tree_node_id"],
        np.asarray(ts.samples(),dtype=np.int64),np.asarray(ts.nodes_time)[ts.samples()])
    slot_rows,roster=validate_slot_metadata_v1(rows)
    positions,edges,retained=_scoring_grid_v1(reader.read(refs["scoring_coordinates"],observed=True),contract)
    local_sites=np.asarray(ts.tables.sites.position,dtype=np.float64)
    if (not len(local_sites) or not np.array_equal(local_sites,local_sites.astype(np.int64))
            or np.any(np.diff(local_sites)<=0) or np.any(local_sites<0) or np.any(local_sites>=ts.sequence_length)):
        raise ValueError("Native truth SNP positions must be exact retained local integer bp0")
    offset=float(coverage[0]);queries=[r for r in rows if r["role"]=="query"]
    site_join=original["post_five_iterations"]["site_join"]
    if (len(site_join)!=len(local_sites)
            or [row["native_site_id"] for row in site_join]!=list(range(len(local_sites)))
            or not np.array_equal([row["local_position_0based"] for row in site_join],local_sites)
            or not np.array_equal([row["original_position_0based"] for row in site_join],local_sites+offset)
            or any(row["VCF_position_1based"]-1!=row["original_position_0based"] for row in site_join)):
        raise ValueError("Native sites must equal the receipt's exact VCF/local/original SNP join")
    output=_new_output(output)
    references=_write(output/"references.tsv","".join("%d %s %s\n"%r for r in slot_rows).encode(),guard)
    sample="ID_1 ID_2 missing\n0 0 0\n"+"".join(
        "%s %s 0\n"%(rows[i]["sample_id"],rows[i]["sample_id"]) for i in range(0,952,2))
    native_sample=_write(output/"native_sample.sample",sample.encode(),guard)
    coordinates=_npz(output/"observed_coordinates.npz",dict(
        grid_indexes=retained,positions_bp=positions,local_positions_bp=positions-offset,bin_edges_bp=edges,
        local_site_positions_bp=local_sites.astype(np.int64),original_site_positions_bp=local_sites+offset,
        query_nodes=np.asarray(roster.query_nodes,dtype=np.int32)),guard)
    copied_roster=_json(output/"observed_roster.json",rows,guard)
    local_map=_write(output/"local_uniform_map.tsv",b"position Genetic_Map(cM)\n0 0\n2000000 2\n",guard)
    frame=dict(kind="expanded_native_argmix_observed_parent_v1",parent_id=parent_id,split=PARENT_SPLITS[parent_id],
        tree=refs["native_tree"],references=references,native_sample=native_sample,coordinates=coordinates,
        roster=copied_roster,source_observed_entry_ref=entry_ref,local_map=local_map,
        local_offset_bp=offset,coverage_original_bp0=coverage.tolist(),normalization_length_bp=2000000,
        pure_refs=266,queries=686,native_tree_count=ts.num_trees,stock_native_ID_correspondence_verified=True,
        config_sha256=cfg["config_ref"]["sha256"],owned_code_sha256=cfg["owned_code_sha256"],truth_opened=False)
    return _json(output/"observed_parent.json",frame,guard)


def snp_truth_v1(intervals,labels,actual_nodes,queries,local_sites,original_sites):
    """External exact original SNP join; native first_snp resolver stays unchanged."""
    import numpy as np
    raw=np.asarray([r["source_observed_node_id"] for r in queries])
    if (intervals.dtype!=np.float64 or intervals.ndim!=2 or intervals.shape[1:]!=(2,)
            or not len(intervals) or intervals[0,0]!=0 or intervals[-1,1]!=2000000
            or not np.array_equal(intervals[:-1,1],intervals[1:,0])
            or not np.isfinite(intervals).all() or np.any(intervals[:,1]<=intervals[:,0])
            or labels.shape!=(len(intervals),686) or labels.dtype.kind not in "iu"
            or np.any((labels<0)|(labels>3)) or not np.array_equal(actual_nodes,raw)
            or len(queries)!=686 or any(r["role"]!="query" for r in queries)):
        raise ValueError("External complete C4 truth must join the original686 query identities")
    indexes=np.searchsorted(intervals[:,1],original_sites,side="right")
    if np.any(indexes>=len(intervals)) or np.any(original_sites<0):
        raise ValueError("Observed original SNP lies outside complete truth domain")
    return dict(variant_pos=local_sites.astype(np.int64),truth__hap_labels=labels[indexes].astype(np.int16),
        truth__sample_ids=np.asarray([r["sample_id"] for r in queries[::2]],dtype=str),
        ancestry_names=np.asarray(CLASS_ORDER,dtype=str))


def _frame(record,cfg):
    frame=json.loads(native.checked(record,observed=True).read_text())
    if (frame.get("kind")!="expanded_native_argmix_observed_parent_v1"
            or frame.get("parent_id") not in PARENT_SPLITS or frame["split"]!=PARENT_SPLITS[frame["parent_id"]]
            or frame["config_sha256"]!=cfg["config_ref"]["sha256"]
            or frame["owned_code_sha256"]!=cfg["owned_code_sha256"] or frame["truth_opened"] is not False):
        raise ValueError("Bound full-panel observed parent frame required")
    for key in ("tree","references","native_sample","coordinates","roster","local_map"):
        native.checked(frame[key],observed=True)
    validate_slot_metadata_v1(json.loads(native.checked(frame["roster"],observed=True).read_text()))
    return frame


def admit_native_v1(build_record,cpu_record,cfg):
    """Reuse exact unchanged build/library/source qualification; no environment changes."""
    native.runtime_admission(build_record,cpu_record,cfg)
    torch,ancpath,joint,trainer,loader,libraries=native.source_imports()
    return torch,ancpath,joint,trainer,loader


def prepare_native_parent_v1(frame_record,target_binding,output,cfg,*,guard,reader=None):
    """One native Rust archive now, while other development parents are finishing."""
    _ready(cfg,guard)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or str(native.CLI) not in native.ADMITTED_NATIVE_FILES:
        raise ValueError("CPU-only qualified native build/library admission required before supervision")
    reader=reader or ReaderV1();frame=_frame(frame_record,cfg)
    parent_id=frame["parent_id"];split=frame["split"]
    expected=cfg["output_root"]+"/"+parent_id+"/truth/full_truth_intervals.npz"
    if (target_binding.get("split")!=split or target_binding["truth_npz"]["path"]!=expected
            or not target_binding["generation_receipt"]["path"].endswith(".json")
            or "/truth/" in target_binding["generation_receipt"]["path"]):
        raise ValueError("Root-bound same-parent TRAIN/VAL external truth required; no outer")
    generation=reader.decode_json(target_binding["generation_receipt"])
    if (generation["parent_id"]!=parent_id
            or generation["files"]["truth/full_truth_intervals.npz"]["sha256"]!=target_binding["truth_npz"]["sha256"]):
        raise ValueError("Original same-parent generation truth hash required")
    import numpy as np
    import yaml
    rows=json.loads(native.checked(frame["roster"],observed=True).read_text())
    queries=[r for r in rows if r["role"]=="query"]
    with np.load(io.BytesIO(reader.read(target_binding["truth_npz"])),allow_pickle=False) as z:
        intervals=z["intervals_bp"].copy();labels=z["source_classes_0based"].copy();nodes=z["query_raw_node_ids"].copy()
    with np.load(native.checked(frame["coordinates"],observed=True),allow_pickle=False) as z:
        truth=snp_truth_v1(intervals,labels,nodes,queries,z["local_site_positions_bp"],z["original_site_positions_bp"])
    output=_new_output(output);truth_ref=_npz(output/(parent_id+".native_truth.npz"),truth,guard)
    pair=dict(rel=str(native.checked(frame["tree"],observed=True)),
        references=str(native.checked(frame["references"],observed=True)),
        relate_sample=str(native.checked(frame["native_sample"],observed=True)),
        truth_npz=str(native.checked(truth_ref)))
    pairs={}
    for role in ("train","val"):
        pairs[role]=_write(output/(role+"_pairs.yaml"),yaml.safe_dump([pair] if role==split else [],sort_keys=False).encode(),guard)
    _write(output/"local_uniform_map.tsv",native.checked(frame["local_map"],observed=True).read_bytes(),guard)
    backend=native.backend_config(cfg,output,native.checked(pairs["train"]),native.checked(pairs["val"]))
    backend_ref=_write(output/"backend.yaml",yaml.safe_dump(backend,sort_keys=False).encode(),guard)
    archive=output/(split+".pt");log=output/(split+".Rust.log")
    # Full native chunks bounded by all native trees/query groups, including overlap.
    chunk_bound=686*(math.ceil(frame["native_tree_count"]/512)+1)
    archive_bound=chunk_bound*1024*(16*8+16*16*2+8+1+2*4)+8_000_000
    guard(archive_bound)
    args=[str(native.CLI),"--config",str(native.checked(backend_ref)),"--split",split,"--output",str(archive)]
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=""
    started=time.monotonic()
    with log.open("xb") as stream:
        done=subprocess.run(args,cwd=native.SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT)
    guard(0)
    command=dict(argv=args,exit_code=done.returncode,elapsed_seconds=time.monotonic()-started,
        stock_binary_sha256=native.sha(native.CLI),log=record_v1(log))
    _json(output/"native_command.json",command,guard)
    if done.returncode or not archive.is_file():raise RuntimeError("Unchanged native Rust parent preparation failed")
    info=archive_layout_v1(archive)
    if info["metadata"][:7]!=[2,0 if split=="train" else 1,1024,512,15,0,14]:
        raise ValueError("Native Rust archive changed source split/chunk/reference contract")
    product=dict(kind="expanded_native_argmix_parent_archive_v1",parent_id=parent_id,split=split,
        frame=frame_record,archive=record_v1(archive),archive_layout=info,pair=pair,
        native_truth=truth_ref,truth_binding=target_binding,command=command,
        config_sha256=cfg["config_ref"]["sha256"],owned_code_sha256=cfg["owned_code_sha256"])
    return _json(output/"native_parent.json",product,guard)



def _array_header_v1(np,stream):
    version=np.lib.format.read_magic(stream)
    if version==(1,0):return np.lib.format.read_array_header_1_0(stream)
    if version==(2,0):return np.lib.format.read_array_header_2_0(stream)
    raise ValueError("Native numeric NPY v1/v2 headers required")

def archive_layout_v1(path):
    """Read only stored NPY headers plus tiny metadata, without large tensor decoding."""
    import numpy as np
    headers={}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist())!={key+".npy" for key in (*ARCHIVE_FIELDS,"metadata")}:
            raise ValueError("Exact stock native Rust NPZ schema required")
        for key in ARCHIVE_FIELDS:
            with archive.open(key+".npy") as stream:
                shape,fortran,dtype=_array_header_v1(np,stream)
                if fortran or dtype.hasobject:raise ValueError("Native contiguous numeric arrays required")
                headers[key]=dict(shape=list(shape),dtype=dtype.str)
        with archive.open("metadata.npy") as stream:
            metadata=np.lib.format.read_array(stream,allow_pickle=False).tolist()
    n=headers["input_ids"]["shape"][0]
    if (len(metadata)!=12 or metadata[:1]!=[2] or metadata[1] not in (0,1)
            or metadata[2:7]!=[1024,512,15,0,14] or metadata[7:]!=[0,1,2,3,9]):
        raise ValueError("Native source metadata/token vocabulary/class remap contract differs")
    if (headers["input_ids"]!=dict(shape=[n,1024,16],dtype="<i8")
            or headers["tmrca"]!=dict(shape=[n,1024,16,16],dtype="<i2")
            or headers["labels"]!=dict(shape=[n,1024],dtype="<i8")
            or headers["row_mask"]!=dict(shape=[n,1024],dtype="|u1")
            or headers["cm_features"]!=dict(shape=[n,1024,2],dtype="<f4") or not n):
        raise ValueError("Native archive tensor widths/dtypes changed")
    return dict(chunks=n,arrays=headers,metadata=metadata)


def concatenate_native_archives_v1(paths,output,guard):
    """Preserve each native tensor byte and chunk; bounded streaming NPZ assembly."""
    import numpy as np
    layouts=[archive_layout_v1(path) for path in paths]
    if not layouts or any(row["metadata"]!=layouts[0]["metadata"] for row in layouts):
        raise ValueError("All native split archives must have identical source metadata")
    arrays=layouts[0]["arrays"];chunks=sum(row["chunks"] for row in layouts)
    for row in layouts:
        if any(row["arrays"][key]["shape"][1:]!=arrays[key]["shape"][1:]
               or row["arrays"][key]["dtype"]!=arrays[key]["dtype"] for key in ARCHIVE_FIELDS):
            raise ValueError("Native archive tensor shape/dtype identity required")
    guard(sum(Path(path).stat().st_size for path in paths)+1_000_000)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_STORED,allowZip64=True) as merged:
        for key in ARCHIVE_FIELDS:
            shape=(chunks,*arrays[key]["shape"][1:]);dtype=np.dtype(arrays[key]["dtype"])
            header=dict(descr=np.lib.format.dtype_to_descr(dtype),fortran_order=False,shape=shape)
            with merged.open(key+".npy","w",force_zip64=True) as destination:
                np.lib.format.write_array_header_2_0(destination,header)
                for path in paths:
                    with zipfile.ZipFile(path) as archive,archive.open(key+".npy") as source:
                        _array_header_v1(np,source)
                        shutil.copyfileobj(source,destination,length=16*1024*1024)
        with merged.open("metadata.npy","w",force_zip64=True) as stream:
            np.lib.format.write_array(stream,np.asarray(layouts[0]["metadata"],dtype=np.int64),allow_pickle=False)
    guard(0)
    return record_v1(output)


def assemble_development_v1(parent_products,output,cfg,*,guard):
    _ready(cfg,guard)
    if set(parent_products)!=set(PARENT_SPLITS):raise ValueError("Exactly32TRAIN+8VAL native products required; no outer")
    import yaml
    products={};frames={};pairs={"train":[],"val":[]};provenance={}
    for parent_id,split in PARENT_SPLITS.items():
        product=json.loads(native.checked(parent_products[parent_id]).read_text())
        if (product["kind"]!="expanded_native_argmix_parent_archive_v1" or product["parent_id"]!=parent_id
                or product["split"]!=split or product["config_sha256"]!=cfg["config_ref"]["sha256"]
                or product["owned_code_sha256"]!=cfg["owned_code_sha256"]):
            raise ValueError("Same-parent native preparation/config lineage required")
        frames[parent_id]=_frame(product["frame"],cfg)
        for key in ("archive","native_truth"):native.checked(product[key])
        products[parent_id]=product;pairs[split].append(product["pair"])
        provenance[parent_id]=dict(product["truth_binding"]["truth_npz"],native_truth=product["native_truth"],
            purpose="TRAIN_gradients" if split=="train" else "native_VAL_selection_only")
    output=_new_output(output)
    chosen=next(iter(PARENT_SPLITS));map_ref=frames[chosen]["local_map"]
    if any(frame["local_map"]["sha256"]!=map_ref["sha256"] for frame in frames.values()):
        raise ValueError("Uniform native local recombination map differs across parents")
    local_map=_write(output/"local_uniform_map.tsv",native.checked(map_ref,observed=True).read_bytes(),guard)
    frames_ref=_json(output/"observed_frames.json",dict(kind="expanded_native_argmix_observed_frames_v1",
        parents=frames,local_map=local_map,config_sha256=cfg["config_ref"]["sha256"],
        owned_code_sha256=cfg["owned_code_sha256"],truth_opened=False),guard)
    pair_refs={split:_write(output/(split+"_pairs.yaml"),yaml.safe_dump(pairs[split],sort_keys=False).encode(),guard)
               for split in ("train","val")}
    backend=native.backend_config(cfg,output,native.checked(pair_refs["train"]),native.checked(pair_refs["val"]))
    backend_ref=_write(output/"backend.yaml",yaml.safe_dump(backend,sort_keys=False).encode(),guard)
    archives={split:concatenate_native_archives_v1(
        [native.checked(products[p]["archive"]) for p,s in PARENT_SPLITS.items() if s==split],
        output/(split+".pt"),guard) for split in ("train","val")}
    return _json(output/"native_prepared.json",dict(kind="expanded_native_argmix_archives_v1",
        frames=frames_ref,backend=backend_ref,archives=archives,train_pairs=pair_refs["train"],
        val_pairs=pair_refs["val"],local_map=local_map,truth_provenance=provenance,
        parent_products=parent_products,config_sha256=cfg["config_ref"]["sha256"],
        owned_code_sha256=cfg["owned_code_sha256"],stock_source_unchanged=True,
        archive_order="preserved native per-parent chunks concatenated in frozen registry order",
        monolithic_Rust_chunk_order_or_bit_identical_optimization_claimed=False),guard)


def _stage(record,kind,cfg,observed=False):
    result=json.loads(native.checked(record,observed=observed).read_text())
    if (result.get("kind")!=kind or result.get("config_sha256")!=cfg["config_ref"]["sha256"]
            or result.get("owned_code_sha256")!=cfg["owned_code_sha256"]):
        raise ValueError("Exact expanded native stage/source/config record required")
    return result


def _verify_frames(record,cfg):
    frames=_stage(record,"expanded_native_argmix_observed_frames_v1",cfg,observed=True)
    if set(frames["parents"])!=set(PARENT_SPLITS):raise ValueError("Exactly40development frames required")
    for parent_id,frame in frames["parents"].items():
        if frame["parent_id"]!=parent_id or frame["split"]!=PARENT_SPLITS[parent_id]:
            raise ValueError("Frozen development frame identities differ")
        for key in ("tree","references","native_sample","coordinates","roster"):native.checked(frame[key],observed=True)
        validate_slot_metadata_v1(json.loads(native.checked(frame["roster"],observed=True).read_text()))
    native.checked(frames["local_map"],observed=True)
    return frames


def _verify_prepared(record,cfg):
    prepared=_stage(record,"expanded_native_argmix_archives_v1",cfg)
    _verify_frames(prepared["frames"],cfg)
    if set(prepared["parent_products"])!=set(PARENT_SPLITS):raise ValueError("Full development lineage required")
    for key in ("backend","train_pairs","val_pairs","local_map"):native.checked(prepared[key])
    for split in ("train","val"):native.checked(prepared["archives"][split])
    for row in prepared["truth_provenance"].values():native.checked(row["native_truth"])
    return prepared


def _reference_metadata(prepared,cfg):
    """Stock available-query counting pointer; each parent retains its actual ages/roles."""
    import yaml
    frames=_verify_frames(prepared["frames"],cfg);summaries={};signatures=[]
    expected_counts=dict(ANA=50,CHG=28,WHG=102,EHG=86,QUERY=686)
    for parent_id,frame in frames["parents"].items():
        rows=json.loads(native.checked(frame["roster"],observed=True).read_text())
        expected,_=validate_slot_metadata_v1(rows)
        actual=[(int(p[0]),p[1],float(p[2])) for p in
                (line.split() for line in native.checked(frame["references"],observed=True).read_text().splitlines())]
        counts={name:sum(row[1]==name for row in actual) for name in expected_counts}
        if actual!=expected or counts!=expected_counts:raise ValueError("Exact actual slot metadata/ages required")
        signatures.append(hashlib.sha256(native.canonical([[n,p] for n,p,a in actual])).hexdigest())
        summaries[parent_id]=dict(reference_record=frame["references"],counts=counts,rows=952,
            exact_observed_roster_join=True,all266_donors_including34older_retained=True)
    if len(set(signatures))!=1:raise ValueError("Stock reference counting eligibility must match across development")
    for split in ("train","val"):
        pairs=yaml.safe_load(native.checked(prepared[split+"_pairs"]).read_text())
        expected=[]
        for parent_id,role in PARENT_SPLITS.items():
            if role!=split:continue
            frame=frames["parents"][parent_id]
            expected.append(dict(rel=str(native.checked(frame["tree"],observed=True)),
                references=str(native.checked(frame["references"],observed=True)),
                relate_sample=str(native.checked(frame["native_sample"],observed=True)),
                truth_npz=str(native.checked(prepared["truth_provenance"][parent_id]["native_truth"]))))
        if pairs!=expected:raise ValueError("Native pairs cannot change parent/reference/sample/truth bindings")
    selected=next(iter(PARENT_SPLITS))
    return dict(kind="expanded_native_argmix_slot_metadata_v1",selected_parent=selected,
        selected_reference=frames["parents"][selected]["references"],parents=summaries,
        stock_available_predict_haplotypes=686,
        scope="stock startup counting only; actual parent metadata stays in every native pair",
        inherited_products=dict(frames=dict(record=prepared["frames"],original_config_sha256=cfg["config_ref"]["sha256"],
            original_owned_code_sha256=cfg["owned_code_sha256"]),
            native_prepared=dict(original_config_sha256=cfg["config_ref"]["sha256"],
                original_owned_code_sha256=cfg["owned_code_sha256"])))


def _source_clone(function,cfg,replacements):
    text=inspect.getsource(function)
    for before,after in replacements:
        if text.count(before)!=1:raise ValueError("Frozen native stage adapter literal changed")
        text=text.replace(before,after)
    namespace=dict(function.__globals__)
    namespace.update(CONFIG=native.checked(cfg["config_ref"]),verify_frames=_verify_frames,
        verify_prepared=_verify_prepared,reference_metadata_amendment=_reference_metadata,stage=_stage)
    exec(compile(text,function.__code__.co_filename+":H038_panel_IO","exec"),namespace)
    return namespace[function.__name__]


def fit_v1(prepared_record,output,cfg,*,guard,modules):
    """Root-owned GPU lease/storage admission; unchanged author native optimizer/selection."""
    _ready(cfg,guard);output=_new_output(output)
    function=_source_clone(native.stock_train,cfg,[
        ('"official_argmix_biological_stock_fit_v1"','"expanded_native_argmix_stock_fit_v1"')])
    result=function(prepared_record,output,cfg,modules)
    guard(0)
    return _json(output/"native_fit.json",result,guard)


def infer_VAL_v1(fit_record,output,cfg,*,guard,modules):
    """Eight VAL seals; original stock exporter/Rust/GRU/chunk-softmax overlap averaging."""
    _ready(cfg,guard);output=_new_output(output)
    function=_source_clone(native.infer_and_seal,cfg,[
        ('"official_argmix_biological_stock_fit_v1"','"expanded_native_argmix_stock_fit_v1"'),
        ('len(coords["grid_indexes"]), 120, 4','len(coords["grid_indexes"]), 686, 4'),
        ('if len(seals) != 2:','if len(seals) != 8:'),
        ('Both observed VAL parents must be sealed','All eight observed VAL parents must be sealed')])
    seals=function(fit_record,output,cfg,modules)
    guard(0)
    return seals


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args(argv)
    print(json.dumps(dict(status="source_only_inactive_H038_native_ARGMix",
        source=record_v1(ROOT/SOURCE),TRAIN=32,VAL=8,outer_accepted=False,
        haplotypes=952,queries=686,donors=266,source_seed=50,
        native_optimizer="AdamW",native_max_epochs=20,native_patience=2,
        published_April_paper_reproduction=False,root_storage_GPU_guard_integration_pending=True),indent=2))


if __name__=="__main__":main()
