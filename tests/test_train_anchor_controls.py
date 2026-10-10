"""Separate runner receipts for matched noncontextual/static controls."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from test_train_anchor import protocol
from test_boundary_dataset import fixture_source
from arggnn.boundary_dataset import prepare_boundary_dataset, sha256_file
from arggnn.train_anchor_controls import checked_anchor_protocol, main

@pytest.mark.parametrize("fault",["missing","unknown","static_scalar"])
def test_control_protocol_rejects_unscoped_control_modes(tmp_path,fault):
    settings=protocol()
    settings["control_modes"]={name:"noncontextual" for name in settings["models"]}
    if fault=="missing": settings["control_modes"].pop("plain")
    elif fault=="unknown":settings["control_modes"]["plain"]="different-information"
    else:
        settings["control_modes"]["scalar"]="static_local"
    path=tmp_path/"protocol.json";path.write_text(json.dumps(settings))
    with pytest.raises(RuntimeError):checked_anchor_protocol(path)

@pytest.mark.parametrize("mode",["noncontextual","static_local"])
def test_tiny_control_run_preserves_fixed_parent_predictions_and_mode_receipt(tmp_path,monkeypatch,mode):
    torch.set_num_threads(1)
    with patch("arggnn.boundary_dataset.scan_repository_storage",return_value=0):
        manifest,source_protocol=fixture_source(tmp_path/"source")
        dataset=prepare_boundary_dataset(manifest,tmp_path/"boundary",
            source_protocol_path=source_protocol)
    settings=protocol()
    settings.update(num_classes=2,num_scales=1,initial_coupling=20.,
        models=["plain"],model_specs={"plain":settings["model_specs"]["plain"]},
        control_modes={"plain":mode},
        boundary_dataset_manifest_sha256=dataset.manifest_sha256,
        boundary_preparation_sha256=dataset.preparation_sha256,
        source_dataset_manifest_sha256=sha256_file(manifest))
    config=tmp_path/"protocol.json";config.write_text(json.dumps(settings))
    output=tmp_path/"run"
    monkeypatch.setattr("sys.argv",["train_anchor_controls","--config",str(config),
        "--data",str(dataset.directory),"--output",str(output),"--device","cpu"])
    with patch("arggnn.train_anchor_controls.require_storage",return_value=0):
        main()
    result=json.loads((output/"results.json").read_text())
    model=result["models"]["plain"]
    assert result["evaluation_split"]=="validation_only"
    assert result["protocol_sha256"]==sha256_file(config)
    payload=torch.load(model["validation_predictions"],weights_only=True)
    val=dataset.cases("val")[0]
    assert torch.equal(payload["positions"][0],val.positions)
    assert torch.equal(payload["query_node_ids"][0],val.query_node_ids)
    assert payload["probabilities"][0].shape==(*val.labels.shape,2)
    assert model["history"][0]["kernel_optimization"]["gradient_norm_before_clip_mean"]["coupling"]>=0
    assert model["control_mode"]==mode
