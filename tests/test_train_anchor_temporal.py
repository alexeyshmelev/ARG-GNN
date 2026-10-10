"""Protocol, external target joining, and optimizer checks without a dataset run."""
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from arggnn.resources import ROOT
from arggnn.train_anchor_temporal import checked_temporal_protocol,make_temporal_model,optimizer_groups,evaluate_temporal
from test_anchor_temporal_baselines import prepared,readouts,model


def config():
    return json.loads((ROOT/'configs/anchor_inferred_cohort64_temporal_baselines_pilot_v1.json').read_text())


def test_every_pilot_factory_and_source_rate_weight_decay():
    settings=config()
    for name in settings['models']:
        m=make_temporal_model(name,47,settings)
        parameters=list(m.parameters())
        if parameters:
            groups=optimizer_groups(m,.0001)
            assert groups[1]['weight_decay']==0
            assert {id(p) for group in groups for p in group['params']}=={id(p) for p in parameters}
            assert sum(len(group['params']) for group in groups)==len(parameters)
        else: assert settings['model_specs'][name]['family']=='reference_smoother'
    assert len([s for s in settings['model_specs'].values() if s['family']=='reference_smoother'])==8


def test_validated_pilot_protocol_and_invalid_test_or_selection_rejected(tmp_path):
    for mutate in ('test','selection','rate_decay','hash'):
        settings=config()
        if mutate=='test': settings['splits']['test']=1
        elif mutate=='selection':settings['selection']='test-best checkpoint'
        elif mutate=='rate_decay':settings['source_rate_weight_decay']=.001
        else:settings['source_dataset_manifest_sha256']='incorrect'
        path=tmp_path/(mutate+'.json');path.write_text(json.dumps(settings))
        with pytest.raises(RuntimeError):checked_temporal_protocol(path)
    settings=checked_temporal_protocol(ROOT/'configs/anchor_inferred_cohort64_temporal_baselines_pilot_v1.json')
    assert settings['hidden_dim']==32 and settings['num_anchors']==32


@pytest.mark.parametrize('family',['gru','transformer'])
def test_parent_scatter_alignment_and_target_join_after_forward(family):
    pack=prepared();m=model(family).eval();expected=m(pack).detach()
    indexes=[torch.tensor([3,4,5,6]),torch.tensor([0,1,2])]
    tiles=[];banks={}
    for ix in indexes:
        bank=readouts(pack,pack.bank.positions[ix])
        tile=SimpleNamespace(bank=bank.bank,parent_position_indices=ix)
        tiles.append(tile);banks[id(tile)]=bank
    labels=torch.zeros(7,2,dtype=torch.long)
    case=SimpleNamespace(tiles=tiles,positions=pack.bank.positions,query_node_ids=pack.bank.query_node_ids,
        labels=labels,metadata={'replicate':0},parent_id='synthetic-parent')
    first,prediction=evaluate_temporal(m,[case],banks)
    torch.testing.assert_close(prediction[0],expected,rtol=2e-12,atol=2e-12)
    case.labels=torch.ones_like(labels)
    second,prediction2=evaluate_temporal(m,[case],banks)
    torch.testing.assert_close(prediction2[0],prediction[0],rtol=0,atol=0)
    assert first[0]['nll']!=second[0]['nll']
    tiles[1].parent_position_indices=torch.tensor([3,1,2])
    with pytest.raises(RuntimeError):evaluate_temporal(m,[case],banks)


def test_raw_float64_logit_nll_selection_does_not_hide_probability_floor():
    pack=prepared()
    class Fixed(torch.nn.Module):
        def forward(self,bank):
            value=torch.zeros(len(bank.bank.positions),2,4,dtype=torch.float64)
            value[:,:,0]=1000.
            return value
    tile=SimpleNamespace(bank=pack.bank,parent_position_indices=torch.arange(7))
    case=SimpleNamespace(tiles=[tile],positions=pack.bank.positions,query_node_ids=pack.bank.query_node_ids,
        labels=torch.ones(7,2,dtype=torch.long),metadata={'replicate':0},parent_id='synthetic-parent')
    records,_=evaluate_temporal(Fixed(),[case],{id(tile):pack})
    assert records[0]['nll']==1000.
    assert records[0]['historical_floor_1e12_nll']<30.


@pytest.mark.parametrize("unsafe", ["/tmp/outside_project_no_write", "../outside", "x/y", "x"+chr(92)+"y", "x"+chr(10)+"control", "..", "", "x"*129])
def test_unsafe_model_ids_rejected_before_forming_output_paths(tmp_path,unsafe):
    settings=config()
    old=settings["models"][0]
    settings["models"][0]=unsafe
    settings["model_specs"][unsafe]=settings["model_specs"].pop(old)
    path=tmp_path/"config.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(RuntimeError,match="safe file-name"):
        checked_temporal_protocol(path)


def test_config_edit_during_synthetic_validation_cannot_seal_a_run(tmp_path,monkeypatch):
    from unittest.mock import patch
    from test_boundary_dataset import fixture_source
    from arggnn.boundary_dataset import prepare_boundary_dataset,sha256_file
    from arggnn import train_anchor_temporal as runner
    torch.set_num_threads(1)
    with patch("arggnn.boundary_dataset.scan_repository_storage",return_value=0):
        manifest,source_protocol=fixture_source(tmp_path/"source")
        dataset=prepare_boundary_dataset(manifest,tmp_path/"boundary",
            source_protocol_path=source_protocol)
    settings=config()
    settings.update(splits={"train":1,"val":1,"test":0},num_classes=2,
        models=["synthetic_smoother"],model_specs={"synthetic_smoother":{
        "family":"reference_smoother","bandwidth_bp":10000.,"pseudocount":.01}},
        boundary_dataset_manifest_sha256=dataset.manifest_sha256,
        boundary_preparation_sha256=dataset.preparation_sha256,
        source_dataset_manifest_sha256=sha256_file(manifest))
    path=tmp_path/"config.json"
    path.write_text(json.dumps(settings))
    initial_sha=sha256_file(path)
    output=tmp_path/"run"
    class Fixed(torch.nn.Module):
        def forward(self,prepared):
            return torch.zeros(len(prepared.bank.positions),len(prepared.bank.query_node_ids),2)
        def configuration_manifest(self):
            return {"synthetic_fixture":True}
    original_eval=runner.evaluate_temporal
    def mutate_during_validation(*args,**kwargs):
        result=original_eval(*args,**kwargs)
        path.write_text(json.dumps({**settings,"learning_rate":.003}))
        return result
    monkeypatch.setattr(runner,"make_temporal_model",lambda *args:Fixed())
    monkeypatch.setattr(runner,"evaluate_temporal",mutate_during_validation)
    monkeypatch.setattr("sys.argv",["runner","--config",str(path),"--data",str(dataset.directory),
        "--output",str(output),"--device","cpu"])
    with patch("arggnn.train_anchor_temporal.require_storage",return_value=0):
        with pytest.raises(RuntimeError,match="Frozen configuration changed during training"):
            runner.main()
    assert not (output/"results.json").exists()
    assert sha256_file(output/"protocol.json")==initial_sha
    payload=torch.load(output/"synthetic_smoother_validation_logits.pt",weights_only=True)
    assert payload["protocol_sha256"]==initial_sha
