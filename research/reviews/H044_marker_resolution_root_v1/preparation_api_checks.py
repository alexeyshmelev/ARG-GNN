"""One bounded source-interface qualification; no actual assets/native model."""
import dataclasses,hashlib,json
from pathlib import Path
import numpy as np
import torch
from arggnn.mutation_routed_parent_loader_v1 import LoadedMutationRoutedParentV1
from arggnn.prepare_marker_resolution_observations_v1 import coordinate_sidecar_v1,selected_rows_v1,TRAIN,VAL
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from arggnn.marker_resolution_readout_v1 import build_marker_evidence_arrays_v1,tensor_identity_v1
from types import SimpleNamespace

pid=TRAIN[0];ref=dict(path="synthetic/observed.json",sha256="0"*64)
positions=np.array([10,90,110,210,290.],dtype=np.float64)
centers=np.array([50.,150.,250.]);edges=np.array([0.,100.,200.,300.])
intervals=np.array([[0.,200.],[200.,300.]])
genotype=np.array([[0,1,0,1,0,1]]*5,dtype=np.int8)
reference=np.array([0,1,2,3,-1,-1],dtype=np.int64);query=np.array([4,5],dtype=np.int64)
score=dict(positions=centers,edges=edges,query_slots=query,
    raw_query_nodes=np.array([40,50]),cohorts=np.array(["present_day","Neo"]),retained_indexes=np.arange(3))
loaded=LoadedMutationRoutedParentV1(pid,"train",[],intervals,positions,genotype,
    np.full(5,2,dtype=np.int64),np.arange(6),reference,np.zeros(6),query,centers,edges,score,
    dict(observations_only=True,targets_read=False))
# Exact real dataclass field contract, with tiny tensor attributes for pure join.
bank=SimpleNamespace(genotype=torch.from_numpy(genotype.copy()),reference_class=torch.from_numpy(reference.copy()),
    query_column=torch.from_numpy(query.copy()),sample_age=torch.zeros(6),
    site_hold=torch.tensor([0,0,0,1,1]),readout_hold=torch.tensor([0,0,1]))
ctx=dict(entry_refs={pid:ref},observation_cache_ref=ref,observation_source_refs={})
index=dict(parents={pid:dict(array_ref=ref)})
sidecar=coordinate_sidecar_v1(pid,loaded,bank,score,index,ctx)
assert sidecar.binding["original_loader_cache_checks"] and all(sidecar.binding["original_loader_cache_checks"].values())
assert sidecar.binding["array_identities"]["genotype"]==tensor_identity_v1(bank.genotype)
data=build_marker_evidence_arrays_v1(bank.genotype,bank.reference_class,bank.query_column,
    bank.site_hold,bank.readout_hold,sidecar.positions_bp0,sidecar.centers_bp,
    sidecar.bin_edges_bp,sidecar.intervals_bp,allele_cardinality=sidecar.allele_cardinality)
assert data.features_for_queries(0,2).shape==(3,2,4,6)
registry=[dict(parent_id=p,split="train" if p in TRAIN else "val") for p in TRAIN+VAL]
registry += [dict(parent_id="synthetic_outer_%d"%i,split="outer") for i in range(8)]
assert set(selected_rows_v1(registry))==set(TRAIN+VAL)
loaded=dataclasses.replace(loaded,genotype=genotype.copy());loaded.genotype[0,0]=1
try:coordinate_sidecar_v1(pid,loaded,bank,score,index,ctx)
except ValueError:pass
else:raise AssertionError("Original/cache mismatch was accepted")
source="src/arggnn/prepare_marker_resolution_observations_v1.py"
result=dict(schema="H044_preparation_tiny_dataclass_API_checks_v1",status="passed",
    source_ref=dict(path=source,sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest()),
    loaded_dataclass_fields=[f.name for f in dataclasses.fields(LoadedMutationRoutedParentV1)],
    checks=["actual original dataclass field access","original ordered loader/cache join",
        "exact sidecar identities","pure marker preparation from joined sidecar",
        "original48 registry retained with40 selected locally","changed genotype rejected"],
    actual_assets_targets_checkpoint_runtime_GPU_queries=False,native_model_constructed=False,
    shape=dict(sites=5,haplotypes=6,queries=2,centers=3,holds=2))
print(json.dumps(result,sort_keys=True))
