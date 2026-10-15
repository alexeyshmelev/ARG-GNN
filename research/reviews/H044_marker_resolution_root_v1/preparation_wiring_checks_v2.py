"""Bounded actual run_context wiring qualification with mock I/O and tiny arrays.

The real v2 run_context and coordinate join execute. Original-reader functions,
writer/storage guards and strict full952 wrapper are mocked. No original assets,
native model, target, checkpoint, guard/runtime or CUDA context is touched.
"""
import contextlib,dataclasses,hashlib,io,json,sys,types
from pathlib import Path
import numpy as np
import torch
import arggnn
from arggnn import prepare_marker_resolution_observations_v2 as prep
from arggnn import marker_resolution_readout_v1 as real_core
from arggnn.mutation_routed_parent_loader_v1 import LoadedMutationRoutedParentV1
from arggnn.mutation_routed_full_panel_observations_v1 import validate_registry_v1

events=[];loaded_ids=[];saved={};scratch_calls=[]
def reference(name):return dict(path=name,sha256="0"*64)
entries={p:reference("synthetic/"+p+".json") for p in prep.PARENTS}
registry=[dict(parent_id=p,split="train" if p in prep.TRAIN else "val") for p in prep.PARENTS]
registry += [dict(parent_id="synthetic_outer_%d"%i,split="outer") for i in range(8)]
for i,row in enumerate(registry):row["seeds"]=dict(ancestry=2*i+1,mutations=2*i+2)
positions=np.array([10,90,110,210,290.],dtype=np.float64)
centers=np.array([50.,150.,250.]);edges=np.array([0.,100.,200.,300.])
intervals=np.array([[0.,200.],[200.,300.]])
genotype=np.array([[0,1,0,1,0,1]]*5,dtype=np.int8)
reference_class=np.array([0,1,2,3,-1,-1],dtype=np.int64);query=np.array([4,5],dtype=np.int64)
def score(pid):return dict(parent_id=pid,split="train" if pid in prep.TRAIN else "val",
    positions=centers,edges=edges,query_slots=query,raw_query_nodes=np.array([40,50]),
    cohorts=np.array(["present_day","Neo"]),retained_indexes=np.arange(3))
def bank():
    return types.SimpleNamespace(genotype=torch.from_numpy(genotype.copy()),
        reference_class=torch.from_numpy(reference_class.copy()),query_column=torch.from_numpy(query.copy()),
        sample_age=torch.zeros(6),site_hold=torch.tensor([0,0,0,1,1]),
        readout_hold=torch.tensor([0,0,1]),spans=torch.tensor([200.,100.]))
class Reader:
    root=Path("/workspace-SR003.nfs2/ARG-GNN")
    def read(self,ref,**kwargs):
        assert ref in entries.values(),"Only synthetic observed entry reads"
        return json.dumps(dict(refs=dict(native_tree=reference("AGENTS.md")))).encode()
class Output:
    def save(self,name,value,kind="json"):
        assert name not in saved
        saved[name]=value
        return dict(path="synthetic_output/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
def cache_index(ref,reader,*,registry,entry_refs,source_refs):
    roles=validate_registry_v1(registry)
    assert [len(roles[k]) for k in ("train","val","outer")]==[32,8,8]
    assert len(registry)==48 and set(entry_refs)==set(prep.PARENTS)
    events.append(("original_cache_validation",48))
    return dict(parents={p:dict(array_ref=reference("synthetic/"+p+".npz")) for p in prep.PARENTS})
def cache_parent(pid,index,reader):
    assert events.index(("original_cache_validation",48))<events.index(("selected_development_rows",40))
    assert pid in prep.PARENTS
    return dict(model_inputs=bank(),scoring=score(pid))
def load_parent(row,entry_ref,reader,*,scratch_directory):
    pid=row["parent_id"];assert row["split"]!="outer";loaded_ids.append(pid)
    return LoadedMutationRoutedParentV1(pid,row["split"],[],intervals,positions,genotype,
        np.full(5,2,dtype=np.int64),np.arange(6),reference_class,np.zeros(6),query,centers,edges,score(pid),
        dict(entry_ref=entry_ref,observations_only=True,targets_read=False))
def evidence(bank,sidecar):
    return real_core.build_marker_evidence_arrays_v1(bank.genotype,bank.reference_class,bank.query_column,
        bank.site_hold,bank.readout_hold,sidecar.positions_bp0,sidecar.centers_bp,sidecar.bin_edges_bp,
        sidecar.intervals_bp,allele_cardinality=sidecar.allele_cardinality)
def bind(bank,markers):
    assert markers.genotype is bank.genotype and markers.query_column is bank.query_column
    return types.SimpleNamespace(validate=lambda:(bank,markers))
original_selection=prep.selected_rows_v1
def selected(registry):
    result=original_selection(registry)
    events.append(("selected_development_rows",len(result)));return result
prep.selected_rows_v1=selected
fake_h42=types.ModuleType("arggnn.train_anonymous_clade_flag_control_v1")
fake_h42.science=lambda:None;fake_h42.fit=types.SimpleNamespace(_jsonable_v1=lambda x:x);fake_h42.h038=types.SimpleNamespace(tensor_bytes_v1=lambda bank:10000)
fake_fast=types.ModuleType("arggnn.mutation_routed_full_panel_fast_reader_v1")
fake_fast.read_full_cache_index_v1=cache_index;fake_fast.read_full_cache_parent_v1=cache_parent
fake_observed=types.ModuleType("arggnn.mutation_routed_full_panel_observations_v1")
fake_observed.load_full_parent_v1=load_parent
fake_core=types.ModuleType("arggnn.marker_resolution_readout_v1")
for name in ("MarkerCoordinatesV1","tensor_identity_v1","ORIGINAL_LOADER_REF","logical_bounds_v1"):
    setattr(fake_core,name,getattr(real_core,name))
fake_core.prepare_marker_evidence_v1=evidence;fake_core.bind_marker_bank_v1=bind
stubs={m.__name__:m for m in (fake_h42,fake_fast,fake_observed,fake_core)}
previous={};missing=object()
for name,module in stubs.items():
    previous[name]=(sys.modules.get(name,missing),getattr(arggnn,name.split(".")[-1],missing))
    sys.modules[name]=module;setattr(arggnn,name.split(".")[-1],module)
ctx=dict(root_authorizer="root_agent",admitted=True,phase="prepare",reader=Reader(),output=Output(),
    cfg=dict(CPU_bank_cap_bytes=100000000,CPU_scratch_cap_bytes=100000000,marker_cache_cap_bytes=100000000,
        CPU_coordinate_bulk_cap_bytes=1000000,CPU_binding_bulk_cap_bytes=1000000),
    guard=lambda:None,scratch_guard=lambda size:scratch_calls.append(size),scratch_directory="synthetic_unused_scratch",
    entry_refs=entries,registry=registry,observation_cache_ref=reference("synthetic/cache.json"),
    observation_source_refs={},config_ref=reference("synthetic/config.json"),source_manifest_ref=reference("synthetic/source.json"))
try:
    with contextlib.redirect_stdout(io.StringIO()):
        result=prep.run_context_v1(ctx,reference("synthetic/input.json"))
finally:
    for name,(module,attr) in previous.items():
        if module is missing:sys.modules.pop(name,None)
        else:sys.modules[name]=module
        key=name.split(".")[-1]
        if attr is missing:delattr(arggnn,key)
        else:setattr(arggnn,key,attr)
    prep.selected_rows_v1=original_selection
assert events==[("original_cache_validation",48),("selected_development_rows",40)]
assert tuple(loaded_ids)==prep.PARENTS and len(result["parents"])==40
assert result["targets_read"] is False and result["checkpoint_payloads_read"] is False
assert set(saved)=={"all40_original_marker_coordinates.npz","all40_original_marker_bindings.json","completed.json"}
assert result["runner_durable_product_count"]==3
assert scratch_calls==[100000000]
assert set(saved["all40_original_marker_coordinates.npz"])=={p+"__"+k for p in prep.PARENTS for k in prep.COORDINATE_KEYS}
for pid in prep.PARENTS:
    entry=result["parents"][pid]
    assert entry["binding_member_selector"]==pid
    assert entry["coordinates_member_selector"]=={k:pid+"__"+k for k in prep.COORDINATE_KEYS}
    binding=saved["all40_original_marker_bindings.json"]["parents"][pid]
    for key,member in entry["coordinates_member_selector"].items():
        value=torch.from_numpy(saved["all40_original_marker_coordinates.npz"][member])
        assert real_core.tensor_identity_v1(value)==binding["array_identities"][key]
source=Path("src/arggnn/prepare_marker_resolution_observations_v2.py")
receipt=dict(schema="H044_preparation_actual_run_context_wiring_checks_v2",status="passed",
    source_ref=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    checks=["actual run_context_v1 branch executed","unchanged cache interface receives original48 registry","real original validate_registry_v1 accepts48 unique seed streams",
        "development40 selected only after cache validation","exact40 non-outer loader calls",
        "actual original dataclass coordinate join on tiny arrays","exact three batched durable products","one scratch-growth preflight before sequential temporary files","all40 member selectors and full ordered identities verified",
        "no target capability or target/checkpoint access"],
    events=events,prepared_parents=40,original_registry_count=48,sites_per_parent=5,haplotypes=6,queries=2,
    native_holds=2,centers=3,original_IO_and_strict952_wrapper_mocked=True,
    actual_observation_assets_targets_checkpoint_runtime_guards_GPU_queries=False,
    CUDA_context_created=False,native_model_constructed=False,all_products_RAM_only=True,batched_product_count=3,coordinate_archive_members=200,scratch_preflight_calls=len(scratch_calls))
print(json.dumps(receipt,sort_keys=True))
