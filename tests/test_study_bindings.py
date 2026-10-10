"""Reject mixed parents even when independently hashed files are valid."""
import copy
from pathlib import Path
import pytest
import torch
from arggnn.ancient_simulate import AncientSimulationConfig,simulate_ancient_replicate,save_ancient_replicate
from arggnn.reference_encoder import ReferenceFeatureAdapter
from arggnn.train_study import load_cases,sha


@pytest.fixture
def manifest(tmp_path):
    config=AncientSimulationConfig(sequence_length=1000,grid_points=4,
        reference_diploids_per_source=1,query_diploids=1,landmarks_per_source=1)
    options=dict(reference_columns=[0,1,2,3],time_column=6,branch_column=7)
    result={"protocol":{"reference_adapter":options},"replicates":[]}
    for rep in range(2):
        case=simulate_ancient_replicate(config,master_seed=991,split="train",replicate=rep)
        record=save_ancient_replicate(case,tmp_path/f"rep{rep:03d}")
        adapter=ReferenceFeatureAdapter(8,**options)
        record["tensor_sha256"]=sha(record["tensor_file"])
        record["metadata_sha256"]=sha(record["metadata"])
        prepared=Path(record["directory"])/"prepared.pt"
        provenance=adapter.manifest_provenance(case.metadata["feature_names"],record["tensor_sha256"])
        torch.save({"sequence":adapter.precompute(case.sequence).as_dict(),"adapter":provenance},prepared)
        record.update(prepared_tensor_file=str(prepared),prepared_tensor_sha256=sha(prepared),
                      adapter_configuration_sha256=provenance["configuration_sha256"])
        result["replicates"].append(record)
    return result


def test_correct_parents_load_and_mixed_directory_is_rejected(manifest):
    torch.set_num_threads(1)
    assert [c.metadata["replicate"] for c in load_cases(manifest,"train")]==[0,1]
    corrupt=copy.deepcopy(manifest)
    corrupt["replicates"][0]["directory"]=corrupt["replicates"][1]["directory"]
    with pytest.raises(RuntimeError,match="Directory is not bound"):
        load_cases(corrupt,"train")


def test_valid_prepared_file_from_other_parent_is_rejected(manifest):
    corrupt=copy.deepcopy(manifest)
    for field in ("prepared_tensor_file","prepared_tensor_sha256"):
        corrupt["replicates"][0][field]=corrupt["replicates"][1][field]
    with pytest.raises(RuntimeError,match="not derived from this raw parent"):
        load_cases(corrupt,"train")
