"""Frozen anchor attribution and tiny end-to-end pilot evidence."""
from pathlib import Path
import json
from unittest.mock import patch
import pytest
import torch
from test_boundary_dataset import fixture_source
from test_boundary_memory import sequence, cached
from arggnn.boundary_dataset import prepare_boundary_dataset, sha256_file
from arggnn.anchor_kernel import prepare_anchor_kernel_bank
from arggnn.train_anchor import (checked_anchor_protocol, make_anchor_model,
    optimizer_groups, main)

def protocol():
    names=["plain","scalar","softplus"]
    return {"stage":"tiny_validation_fixture", "interpretation":"synthetic integration",
        "splits":{"train":1,"val":1,"test":0}, "num_classes":3, "hidden_dim":8,
        "num_layers":1,"latent_dim":3,"num_scales":3,"num_anchors":8,
        "epochs":1,"patience":1,"learning_rate":.001,"weight_decay":.0001,
        "rate_weight_decay":0.,"initial_coupling":[5.,20.,80.],"initial_damping":1.,
        "boundary_dataset_manifest_sha256":"a"*64,
        "boundary_preparation_sha256":"b"*64,
        "source_dataset_manifest_sha256":"c"*64,
        "selection":"minimum mean parent-validation NLL","models":names,
        "model_specs":{name:{"geometry":"plain", "coupling":"scalar" if name=="scalar" else "full",
            "kernel_depth":2,"rate_parameterization":"softplus" if name=="softplus" else "log"}
            for name in names}}

@pytest.mark.parametrize("fault",["test","missing_model","rate_decay","zero_anchors","invalid_mode"])
def test_protocol_rejects_invalid_attribution_or_test_settings(tmp_path,fault):
    settings=protocol()
    if fault=="test": settings["splits"]["test"]=1
    elif fault=="missing_model": del settings["model_specs"]["scalar"]
    elif fault=="rate_decay": settings["rate_weight_decay"]=.01
    elif fault=="zero_anchors": settings["num_anchors"]=0
    else: settings["model_specs"]["plain"]["geometry"]="fitted_truth"
    path=tmp_path/"config.json";path.write_text(json.dumps(settings))
    with pytest.raises(RuntimeError): checked_anchor_protocol(path)

def test_log_softplus_controls_have_equal_initial_model_outputs_and_optimizer_coverage():
    torch.set_num_threads(1)
    bank=cached(sequence())
    settings=protocol()
    pack=prepare_anchor_kernel_bank(bank,num_anchors=settings["num_anchors"],num_scales=3)
    torch.manual_seed(42)
    log=make_anchor_model("plain",bank.leaf_x.shape[-1],3,settings).double()
    soft=make_anchor_model("softplus",bank.leaf_x.shape[-1],3,settings).double()
    soft.load_state_dict(log.state_dict())
    torch.testing.assert_close(log(pack),soft(pack),atol=1e-10,rtol=1e-10)
    groups=optimizer_groups(log,.1)
    all_ids=[id(p) for group in groups for p in group["params"]]
    assert len(all_ids)==len(set(all_ids))==len(list(log.parameters()))
    assert groups[1]["weight_decay"]==0.
    assert set(all_ids)=={id(p) for p in log.parameters()}

def test_tiny_cpu_study_emits_exact_parent_predictions_and_source_receipts(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    with patch("arggnn.boundary_dataset.scan_repository_storage",return_value=0):
        manifest,source_protocol=fixture_source(tmp_path/"source")
        dataset=prepare_boundary_dataset(manifest,tmp_path/"boundary",
            source_protocol_path=source_protocol)
    settings=protocol()
    settings.update(num_classes=2, num_scales=1, initial_coupling=20.,
        models=["plain"],model_specs={"plain":settings["model_specs"]["plain"]},
        boundary_dataset_manifest_sha256=dataset.manifest_sha256,
        boundary_preparation_sha256=dataset.preparation_sha256,
        source_dataset_manifest_sha256=sha256_file(manifest))
    config=tmp_path/"protocol.json";config.write_text(json.dumps(settings))
    output=tmp_path/"run"
    monkeypatch.setattr("sys.argv",["train_anchor","--config",str(config),
        "--data",str(dataset.directory),"--output",str(output),"--device","cpu"])
    with patch("arggnn.train_anchor.require_storage",return_value=0):
        main()
    result=json.loads((output/"results.json").read_text())
    assert result["evaluation_split"]=="validation_only"
    assert result["protocol_sha256"]==sha256_file(config)
    record=result["models"]["plain"]
    prediction=Path(record["validation_predictions"])
    assert sha256_file(prediction)==record["validation_predictions_sha256"]
    payload=torch.load(prediction,weights_only=True)
    val=dataset.cases("val")[0]
    assert torch.equal(payload["positions"][0],val.positions)
    assert torch.equal(payload["query_node_ids"][0],val.query_node_ids)
    assert payload["probabilities"][0].shape==(*val.labels.shape,2)
    torch.testing.assert_close(payload["probabilities"][0].sum(-1),
        torch.ones(val.labels.shape),atol=1e-6,rtol=1e-6)
    assert record["per_replicate"][0]["observations"]==val.labels.numel()
