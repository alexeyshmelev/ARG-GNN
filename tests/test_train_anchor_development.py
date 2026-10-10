"""Synthetic contracts and tiny CPU fitting for the fresh-development runner."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from arggnn.train_anchor_development import checked_development_protocol, evaluate, main
from arggnn.confirmatory_metrics import logit_metrics
from arggnn.boundary_dataset import prepare_boundary_dataset, sha256_file
from test_boundary_dataset import fixture_source
from test_train_anchor import protocol


def full_protocol():
    p = protocol()
    p.update(development_namespace="development-v1",
        splits={"train":64,"val":16,"test":0}, training_seeds=[101,202,303],
        hidden_dim=32,num_layers=2,latent_dim=8,num_anchors=32,num_scales=3,
        num_classes=4,epochs=48,patience=8,initial_coupling=[5.,20.,80.],
        initial_damping=1.,age_dataset_manifest_sha256="d"*64,
        age_preparation_sha256="e"*64,models=[],model_specs={},model_learning_rates={})
    for geometry in ("plain","age"):
        for depth in (1,2):
            for lr in (.0003,.001):
                name=f"{geometry}_d{depth}_lr{lr}"
                p["models"].append(name)
                p["model_specs"][name]=dict(geometry=geometry,coupling="full",
                    kernel_depth=depth,rate_parameterization="log")
                p["model_learning_rates"][name]=lr
    return p


@pytest.mark.parametrize("fault",["test","pilot","seed","duplicate","lr","geometry","budget","sourcehash"])
def test_exact_grid_cannot_silently_change(fault,tmp_path):
    p=full_protocol()
    if fault=="test":p["splits"]["test"]=1
    elif fault=="pilot":p["splits"]["train"]=16
    elif fault=="seed":p["training_seeds"].append(404)
    elif fault=="duplicate":p["model_specs"][p["models"][2]]=p["model_specs"][p["models"][0]].copy()
    elif fault=="lr":p["model_learning_rates"][p["models"][0]]=.03
    elif fault=="geometry":p["model_specs"][p["models"][0]]["geometry"]="matched_scalar_age"
    elif fault=="budget":p["epochs"]=49
    else:p["source_dataset_manifest_sha256"]="not_frozen"
    path=tmp_path/"p.json"
    path.write_text(json.dumps(p))
    with pytest.raises(RuntimeError):checked_development_protocol(path)


def test_complete_eight_configuration_grid_is_admitted(tmp_path):
    p=full_protocol()
    path=tmp_path/"p.json"
    path.write_text(json.dumps(p))
    assert checked_development_protocol(path)==p


def test_validation_uses_raw_logits_and_keeps_scatter_correspondence():
    class Fixed(torch.nn.Module):
        def forward(self,bank):
            return torch.tensor([[[1000.,-1000.]], [[-1000.,1000.]]])
    labels=torch.tensor([[0],[1]])
    positions=torch.tensor([.1,.2],dtype=torch.float64)
    ids=torch.tensor([42])
    tile=SimpleNamespace(labels=labels.flip(0),parent_position_indices=torch.tensor([1,0]),
        bank=SimpleNamespace(positions=positions.flip(0),query_node_ids=ids))
    case=SimpleNamespace(labels=labels,positions=positions,query_node_ids=ids,
        metadata={"replicate":"synthetic"},tiles=[tile])
    records,raw=evaluate(Fixed(),[case],{id(tile):tile.bank})
    assert records[0]["nll_uniform_grid"]==2000.
    assert records[0]["historical_floor_1e12_nll"]==pytest.approx(-torch.log(torch.tensor(1e-12)).item(),rel=1e-7)
    assert raw[0].dtype==torch.float64
    assert raw[0][0,0,0]==-1000.
    tile.bank.query_node_ids=torch.tensor([99])
    with pytest.raises(RuntimeError):evaluate(Fixed(),[case],{id(tile):tile.bank})


def test_tiny_cpu_fit_stores_hash_bound_raw_logit_predictions(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    with patch("arggnn.boundary_dataset.scan_repository_storage",return_value=0):
        manifest,source_protocol=fixture_source(tmp_path/"source")
        dataset=prepare_boundary_dataset(manifest,tmp_path/"boundary",
            source_protocol_path=source_protocol)
    p=protocol()
    p.update(num_classes=2,num_scales=1,initial_coupling=20.,models=["plain"],
        model_specs={"plain":p["model_specs"]["plain"]},model_learning_rates={"plain":.001},
        training_seeds=[101,202,303],boundary_dataset_manifest_sha256=dataset.manifest_sha256,
        boundary_preparation_sha256=dataset.preparation_sha256,
        source_dataset_manifest_sha256=sha256_file(manifest))
    config=tmp_path/"p.json"
    config.write_text(json.dumps(p))
    output=tmp_path/"run"
    monkeypatch.setattr("sys.argv",["train_anchor_development","--config",str(config),
        "--expected-config-sha256",sha256_file(config),"--data",str(dataset.directory),
        "--output",str(output),"--device","cpu"])
    with patch("arggnn.train_anchor_development.require_storage",return_value=0), patch(
        "arggnn.train_anchor_development.checked_development_protocol",return_value=p):
        main()
    r=json.loads((output/"results.json").read_text())
    assert r["evaluation_split"]=="fresh_development_validation_only"
    record=r["models"]["plain"]
    payload=torch.load(record["validation_predictions"],weights_only=True)
    assert "probabilities" not in payload
    val=dataset.cases("val")[0]
    assert torch.equal(payload["positions"][0],val.positions)
    assert torch.equal(payload["query_node_ids"][0],val.query_node_ids)
    loss=logit_metrics(payload["logits"][0].numpy(),val.labels.numpy())
    assert loss["nll"]==pytest.approx(record["validation_nll"])
    assert record["learning_rate"]==.001
    binding=json.loads((output/"dataset_binding.json").read_text())
    assert binding["boundary_manifest_sha256"]==dataset.manifest_sha256
    assert not (output/"dataset_manifest.json").exists()


def test_wrong_config_freeze_rejected_before_loading_data(tmp_path,monkeypatch):
    p=tmp_path/"p.json"
    p.write_text("{}")
    monkeypatch.setattr("sys.argv",["runner","--config",str(p),
        "--expected-config-sha256","a"*64,"--data",str(tmp_path),
        "--output",str(tmp_path/"run"),"--device","cpu"])
    with pytest.raises(RuntimeError,match="exact supplied freeze"):main()
