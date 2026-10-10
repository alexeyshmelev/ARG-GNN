"""Age-cache isolation, supervision counterfactuals and integrity rejection."""
import gzip
import json
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from test_boundary_dataset import fixture_source
from arggnn.reference_encoder import ReferenceFeatureAdapter
from arggnn.boundary_dataset import prepare_boundary_dataset, sha256_file, load_boundary_dataset
from arggnn.age_dataset import prepare_age_dataset, load_age_caches

@pytest.fixture(autouse=True)
def bounded_cpu_storage():
    torch.set_num_threads(1)
    with patch("arggnn.boundary_dataset.scan_repository_storage", return_value=0):
        yield

def create(root, flip_labels=False):
    manifest, protocol = fixture_source(root/"source")
    if flip_labels:
        source = json.loads(manifest.read_text())
        adapter = ReferenceFeatureAdapter(6, **source["protocol"]["reference_adapter"])
        for record in source["replicates"]:
            raw = Path(record["tensor_file"])
            with gzip.open(raw,"rb") as handle:
                payload = torch.load(handle, weights_only=True)
            payload["labels"] = 1-payload["labels"]
            with gzip.open(raw,"wb") as handle:
                torch.save(payload, handle)
            record["tensor_sha256"] = sha256_file(raw)
            prepared_path = Path(record["prepared_tensor_file"])
            prepared = torch.load(prepared_path, weights_only=True)
            metadata = json.loads(Path(record["metadata"]).read_text())
            prepared["adapter"] = adapter.manifest_provenance(metadata["feature_names"],
                record["tensor_sha256"])
            torch.save(prepared,prepared_path)
            record["prepared_tensor_sha256"] = sha256_file(prepared_path)
        manifest.write_text(json.dumps(source,indent=2)+"\n")
    boundary = prepare_boundary_dataset(manifest, root/"boundary",
        source_protocol_path=protocol)
    ages = root/"ages"
    preparation = prepare_age_dataset(boundary.directory, ages,
        expected_boundary_manifest_sha256=boundary.manifest_sha256,
        expected_boundary_preparation_sha256=boundary.preparation_sha256)
    return boundary, ages, preparation

def load(boundary, ages, preparation):
    return load_age_caches(ages, boundary,
        expected_manifest_sha256=sha256_file(ages/"manifest.json"),
        expected_preparation_sha256=preparation["preparation_sha256"])

def test_label_counterfactual_changes_supervision_and_preserves_all_age_operators(tmp_path):
    original, first, fp = create(tmp_path/"original")
    changed, second, sp = create(tmp_path/"changed", flip_labels=True)
    a,b = load(original,first,fp),load(changed,second,sp)
    assert len(a)==len(b)==4
    for left,right in zip(original.parent_cases,changed.parent_cases):
        assert torch.equal(left.labels,1-right.labels)
        for tl,tr in zip(left.tiles,right.tiles):
            ca,cb = a[tl.provenance["tile_id"]],b[tr.provenance["tile_id"]]
            assert torch.equal(tl.bank.leaf_x,tr.bank.leaf_x)
            for name in ("cqq","cqr","scalar_cqq","scalar_cqr","eigenvalues",
                         "mean_diagonal_ratios","normalization_scale"):
                torch.testing.assert_close(getattr(ca,name),getattr(cb,name),atol=0,rtol=0)
    artifact = first/fp["records"][0]["operator_file"]
    envelope = torch.load(artifact,weights_only=True)
    assert set(envelope)=={"schema_version","operator_cache","provenance"}
    assert not any(key in {"labels", "query_labels", "targets", "baseline_probabilities", "ancestry_truth"} for key in envelope["operator_cache"])

@pytest.mark.parametrize("fault", ["bytes","extra_label","query_order","tile_identity","dependency","missing"])
def test_age_loader_rejects_mutated_or_unbound_artifacts(tmp_path,fault):
    boundary, ages, preparation = create(tmp_path)
    manifest_path = ages/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["records"][0]
    artifact = ages/record["operator_file"]
    if fault == "bytes":
        with artifact.open("ab") as handle:
            handle.write(b"untrusted")
    elif fault == "dependency":
        manifest["implementation_sha256"]["age_interface.py"]="0"*64
    elif fault == "missing":
        manifest["records"].pop()
    else:
        payload = torch.load(artifact,weights_only=True)
        if fault=="extra_label":
            payload["labels"]=torch.zeros(1,dtype=torch.long)
        elif fault=="query_order":
            payload["operator_cache"]["query_ids"]=payload["operator_cache"]["query_ids"].flip(0)
        else:
            payload["provenance"]["tile_id"]="different-source"
        torch.save(payload,artifact)
        record.update(operator_sha256=sha256_file(artifact),
                      operator_bytes=artifact.stat().st_size)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    with pytest.raises((RuntimeError,ValueError)):
        load(boundary,ages,preparation)

def test_age_loader_rejects_another_plain_dataset_even_with_matching_geometry(tmp_path):
    a,ages,preparation=create(tmp_path/"a")
    b,_,_=create(tmp_path/"b")
    with pytest.raises(RuntimeError,match="boundary"):
        load(b,ages,preparation)
