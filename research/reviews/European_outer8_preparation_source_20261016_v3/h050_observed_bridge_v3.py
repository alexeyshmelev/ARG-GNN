"""Truth-free full952 H050 preparation/reconstruction bridge. No work at import."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,os,stat
from .outer_prediction_validator_v2 import (
 ROOT,HERE,ObservedReaderV2,need,record,ref,same,archive,observed_support,completed_wait,validate_h050_bank_geometry_v3,h050_preparation_wait_v3)
def _array(tensor):
    return tensor.detach().cpu().numpy()
class _CacheReaderV3(ObservedReaderV2):
    def __init__(self,guard,array_ref,byte_count,cap):
        super().__init__(guard);need(type(byte_count) is int and 0<byte_count<=cap,"Actual full cache under ROOT CPU cap")
        self.array_ref,self.byte_count=array_ref,byte_count;self.extra_arrays={}
    def read(self,r,*,maximum=512*2**20):
        bound=self.byte_count if r==self.array_ref else self.extra_arrays.get((r["path"],r["sha256"]),maximum)
        return super().read(r,maximum=bound)
class _CutWriterV3:
    """Bind genuine writer publication to this cell and register native readback bounds."""
    def __init__(self,base,reader,output_directory,cap,guard):
        self.base,self.reader,self.directory,self.cap,self.guard=base,reader,output_directory+"/cuts",cap,guard
    def save(self,name,value,kind=None):
        need(Path(name).name==name and name not in (".",".."),"Canonical native cut publication leaf")
        self.guard()
        result=self.base.save(name,value) if kind is None else self.base.save(name,value,kind)
        record(result,self.directory+"/"+name)
        if kind=="npz":
            path=ROOT/result["path"];size=path.stat().st_size
            need(0<size<=self.cap,"Actual native compact cut archive under admitted ROOT cap")
            self.reader.extra_arrays[result["path"],result["sha256"]]=size
        self.guard();return result

class _TreeReaderV3(ObservedReaderV2):
    """The original loader may read only this exact observed tree, never a target."""
    def __init__(self,guard,tree_ref,cap):
        super().__init__(guard);record(tree_ref);need(tree_ref["path"].endswith("/relate/post_size.observed.trees"),"Exact original observed tree")
        self.tree_ref,self.cap=tree_ref,cap
    def read(self,r,*,maximum=512*2**20):
        if r!=self.tree_ref:return super().read(r,maximum=maximum)
        path=ROOT/record(r);need(path.resolve()==path,"Canonical original observed tree");self.guard()
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        fields=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        with os.fdopen(fd,"rb") as f:
            before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and 0<before.st_size<=self.cap,"Bounded observed tree")
            need(fields(path.lstat())==fields(before),"Exact opened observed tree");data=f.read(self.cap+1)
            need(len(data)==before.st_size and fields(os.fstat(f.fileno()))==fields(before) and fields(path.lstat())==fields(before),"Immutable observed tree")
        need(hashlib.sha256(data).hexdigest()==r["sha256"],"Actual original observed tree bytes")
        self.guard();return data
def _fixed_TRAIN_cache_v3(identity,reader):
    """Read compact frozen TRAIN provenance, never a target or checkpoint."""
    train=reader.decode_json(identity["training_completion_ref"])
    seal=reader.decode_json(train["observations_seal"]);cache_ref=seal["observation_cache_ref"]
    raw=reader.decode_json(cache_ref)
    need(raw["targets_cached"] is False and raw["observations_only"] is True and raw["receipt"]["targets_read"] is False,"Genuine target-free TRAIN cache provenance")
    return cache_ref,raw
def prepare_h050_outer_bank_v3(request_ref,ctx,*,guard):
    from .outer_adapters_v2 import _authority,json_parent,_pin_module,_publish_npz,_publish_json
    q,reader=_authority(request_ref,ctx["method_id"],json_parent(request_ref,guard),guard)
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only original bank preparation")
    need(ctx["method_id"] in ("H050_stable_leaf_memory","H050_h045_continued_training"),"Fixed H050 pair preparation only")
    need(q["stage"]=="prepare_bank","Explicit full observation preparation admission")
    import numpy as np
    from arggnn import mutation_routed_full_panel_observations_v1 as full
    from arggnn import mutation_routed_full_panel_cache_v1 as cache
    from arggnn import mutation_routed_full_panel_fast_reader_v1 as fast
    _pin_module(full);_pin_module(cache);_pin_module(fast)
    inventory=reader.decode_json(ref(HERE+"/frozen_method_ids.v1.json"));identity=inventory["methods"][ctx["method_id"]]
    frozen_ref,raw=_fixed_TRAIN_cache_v3(identity,reader)
    generation=reader.decode_json(dict(path="configs/expanded_full_panel_v1.json",sha256="662ba8e4613ce85b9aaec9909e873c9cddf9c62672b895e7d9e02f851d518d08"))
    frozen=fast.read_full_cache_index_v1(frozen_ref,reader,registry=generation["parent_registry"],entry_refs=raw["entry_refs"],source_refs=raw["source_refs"])
    expected=observed_support(q["parent_id"],ctx["observation_entry_ref"],reader)
    directory=ROOT/q["output_directory"]/"observation";guard(32768);directory.mkdir()
    tree_reader=_TreeReaderV3(guard,expected["entry"]["refs"]["native_tree"],ctx["cfg"]["CPU_bank_cap_bytes"])
    row=next(x for x in generation["parent_registry"] if x["parent_id"]==q["parent_id"])
    observed=full.load_full_parent_v1(row,ctx["observation_entry_ref"],tree_reader,scratch_directory=directory/"scratch")
    bins=full.old.ShamBinEdgesV1(**frozen["sham_bin_edges"])
    original=full._full_prepare(observed.trees,observed.intervals_bp,observed.positions_bp0,observed.genotype,observed.allele_count,
        observed.sample_ids,observed.reference_class,observed.sample_age,observed.query_ids,observed.readout_x_bp,sham_bin_edges=bins,fixture=False)
    bank=full.MutationRoutedWholeTreeInputsV1.from_arrays(original.model_arrays());fast.validate_full_bank_v1(bank)
    validate_h050_bank_geometry_v3(_array(bank.query_column),_array(bank.readout_hold),_array(bank.spans),bank.sample_index.shape,observed.intervals_bp,expected)
    item=dict(model_inputs=bank,scoring=observed.scoring,observations_receipt=dict(observed.receipt,preprocessing=original.metadata))
    collection=dict(parents={q["parent_id"]:item},registry=generation["parent_registry"],entry_refs={q["parent_id"]:ctx["observation_entry_ref"]},
        sham_bin_edges=asdict(bins),receipt=dict(schema="European_outer_observed_bank_preparation_v3",targets_read=False,observations_only=True,
            sham_bins_inherited_from_TRAIN_cache_ref=frozen_ref,original_native_holds_preserved=True))
    cache_ref=cache.write_full_cache_v1(collection,directory/"cache",source_refs=raw["source_refs"],guard_exact_bytes=guard)
    geometry_ref=_publish_npz(directory/"native_hold_geometry.npz",dict(intervals_bp0=np.asarray(observed.intervals_bp,dtype=np.float64),
        positions_bp=expected["positions"],query_sample_indexes=expected["query_slots"],readout_hold=_array(bank.readout_hold)),guard)
    return _publish_json(directory/"bank_preparation.json",dict(schema="European_outer_H050_bank_preparation_v3",parent_id=q["parent_id"],
        original_observed_entry_ref=ctx["observation_entry_ref"],cache_ref=cache_ref,native_hold_geometry_ref=geometry_ref,
        frozen_TRAIN_cache_ref=frozen_ref,producer_source_ref=ref(HERE+"/h050_observed_bridge_v3.py"),
        input_ref=request_ref,targets_read=False),guard)
def reconstruct_h050_bank_v3(q,ctx,expected,reader,guard):
    """Decode genuine cache and verify original query/native-hold geometry."""
    from arggnn.mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
    base=q["output_directory"];record(ctx["bank_preparation_ref"],base+"/observation/bank_preparation.json")
    prep=reader.decode_json(ctx["bank_preparation_ref"])
    need(prep["schema"]=="European_outer_H050_bank_preparation_v3" and prep["parent_id"]==q["parent_id"]
        and prep["original_observed_entry_ref"]==ctx["observation_entry_ref"] and prep["targets_read"] is False
        and prep["producer_source_ref"]==ref(HERE+"/h050_observed_bridge_v3.py"),"Genuine source-bound original outer bank preparation")
    record(prep["cache_ref"],base+"/observation/cache/manifest.json");record(prep["native_hold_geometry_ref"],base+"/observation/native_hold_geometry.npz")
    cell=dict(run_id=q["run_id"],method_id=q["method_id"],parent_id=q["parent_id"],actual_inference_adapter_ref=q["adapter_ref"])
    h050_preparation_wait_v3(ctx["bank_execution"],cell,ctx["bank_preparation_ref"],"prepare_bank",reader)
    identity=reader.decode_json(ref(HERE+"/frozen_method_ids.v1.json"))["methods"][q["method_id"]]
    frozen_ref,_=_fixed_TRAIN_cache_v3(identity,reader);need(prep["frozen_TRAIN_cache_ref"]==frozen_ref,"Same frozen TRAIN preprocessing")
    raw=reader.decode_json(prep["cache_ref"]);meta=raw["parents"][q["parent_id"]]
    cache_reader=_CacheReaderV3(guard,meta["array_ref"],meta["compressed_bytes"],ctx["cfg"]["CPU_bank_cap_bytes"])
    need(set(raw["entry_refs"])=={q["parent_id"]} and raw["entry_refs"][q["parent_id"]]==ctx["observation_entry_ref"],"Exact single outer bank entry")
    index=read_full_cache_index_v1(prep["cache_ref"],cache_reader,registry=ctx["registry"],entry_refs=raw["entry_refs"],source_refs=raw["source_refs"])
    item=read_full_cache_parent_v1(q["parent_id"],index,cache_reader);bank,parent=item["model_inputs"],item["scoring"]
    for k in ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts"):same(parent[k],expected[k],"Actual decoded bank scoring identity")
    geometry=archive(prep["native_hold_geometry_ref"],reader)
    need(set(geometry)=={"intervals_bp0","positions_bp","query_sample_indexes","readout_hold"},"Exact genuine native-hold snapshot schema")
    same(geometry["positions_bp"],expected["positions"],"Original snapshot grid");same(geometry["query_sample_indexes"],expected["query_slots"],"Original snapshot query order")
    same(geometry["readout_hold"],_array(bank.readout_hold),"Snapshot/actual bank readout identity")
    validate_h050_bank_geometry_v3(_array(bank.query_column),_array(bank.readout_hold),_array(bank.spans),bank.sample_index.shape,geometry["intervals_bp0"],expected)
    return bank,parent,prep,meta,cache_reader
def reconstruct_h050_bundle_v3(q,ctx,expected,producer,reader,guard):
    """Caller cannot inject a bank, query permutation, readout map or cut tensor."""
    bank,parent,prep,meta,cache_reader=reconstruct_h050_bank_v3(q,ctx,expected,reader,guard)
    cuts=reader.decode_json(ctx["cut_preparation_completion_ref"])
    need(cuts["schema"]=="H049_order2_observation_preparation_v1" and cuts["observation_cache_ref"]==prep["cache_ref"]
        and cuts["entry_refs"][q["parent_id"]]==ctx["observation_entry_ref"] and cuts["targets_read"] is False
        and cuts["observations_only"] is True,"Actual same-bank compact cut preparation")
    cell=dict(run_id=q["run_id"],method_id=q["method_id"],parent_id=q["parent_id"],actual_inference_adapter_ref=q["adapter_ref"])
    h050_preparation_wait_v3(ctx["cut_execution"],cell,ctx["cut_preparation_completion_ref"],"prepare_cuts",reader)
    product=cuts["parents"][q["parent_id"]];receipt=reader.decode_json(product["receipt_ref"])
    need(receipt["original_cache_array_ref"]==meta["array_ref"] and receipt["original_observed_entry_ref"]==ctx["observation_entry_ref"]
        and receipt["arrays_ref"]==product["arrays_ref"],"Actual cut recipe bound to exact original bank archive")
    limit=receipt["bounds"]["compressed_NPZ_upper_bytes"]
    need(type(limit) is int and 0<product["compressed_archive_bytes"]<=limit,"Original codec exact archive bound")
    cache_reader.extra_arrays[product["arrays_ref"]["path"],product["arrays_ref"]["sha256"]]=product["compressed_archive_bytes"]
    local=dict(ctx,guard=guard,entry_refs={q["parent_id"]:ctx["observation_entry_ref"]},cut_preparation=cuts)
    episodes=producer.h45.EpisodeCache(local["cfg"],guard,local["before_episode_build"])
    keys={q["parent_id"]:producer.shared.stable_keys(bank,ctx["observation_entry_ref"],cache_reader)}
    bundle=producer.BundleCache(local,cache_reader,episodes,keys).get(q["parent_id"],bank)
    bundle.validate(bank)
    provenance=dict(bank_preparation_ref=ctx["bank_preparation_ref"],cache_ref=prep["cache_ref"],
        native_hold_geometry_ref=prep["native_hold_geometry_ref"],cache_array_ref=meta["array_ref"],
        cut_preparation_completion_ref=ctx["cut_preparation_completion_ref"],bank_execution=ctx["bank_execution"],cut_execution=ctx["cut_execution"])
    return bundle,parent,provenance
def prepare_h050_outer_cuts_v3(request_ref,ctx,*,guard):
    from .outer_adapters_v2 import _authority,json_parent,_clone
    import importlib
    q,reader=_authority(request_ref,ctx["method_id"],json_parent(request_ref,guard),guard)
    need(ctx["method_id"] in ("H050_stable_leaf_memory","H050_h045_continued_training"),"Fixed H050 pair cut preparation only")
    need(q["stage"]=="prepare_cuts","Explicit original compact-cut preparation stage")
    expected=observed_support(q["parent_id"],ctx["observation_entry_ref"],reader)
    bank,parent,prep,meta,cache_reader=reconstruct_h050_bank_v3(q,ctx,expected,reader,guard)
    native=importlib.import_module("research.reviews.H049_order2_cut_memory_prepare_source_20261015_v2.prepare_v2")
    run=_clone(native,"run_context_v1",[("completed_all40_compact_cut_recipes","completed_outer_compact_cut_recipe")],{"PARENTS":(q["parent_id"],)})
    writer=_CutWriterV3(ctx["output"],cache_reader,q["output_directory"],ctx["cfg"]["cut_recipe_archive_cap_bytes"],guard)
    local=dict(ctx,reader=cache_reader,output=writer,guard=guard,phase="prepare",admitted=True,root_authorizer="root_agent",
        observation_cache_ref=prep["cache_ref"],entry_refs={q["parent_id"]:ctx["observation_entry_ref"]})
    result=run(local,request_ref)
    need(result["parents"][q["parent_id"]]["lossless_roundtrip_verified"] is True,"Original cut encode/decode parity")
    return result
