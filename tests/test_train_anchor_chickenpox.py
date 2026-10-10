"""Direct new-architecture forecast tests with synthetic counts and no real scoring."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arggnn.anchor_kernel import AnchorKernelModel, prepare_anchor_kernel_bank
from arggnn.boundary_chickenpox import bind_chickenpox_history, precompute_chickenpox_geometry
from arggnn.chickenpox import ARCHIVE_SHA256, ChickenpoxData, ForecastProtocol, TrainingScaler
from arggnn.resources import ROOT
import arggnn.train_anchor_chickenpox as runner
import arggnn.train_boundary_chickenpox as predecessor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic(counties=20, observations=522, randomized=False):
    counts = 1.+2.*np.arange(observations)[:,None]+.1*np.arange(counties)[None]
    if randomized:
        counts = counts + np.random.default_rng(404).uniform(0,20,counts.shape)
    edges = [(i,i) for i in range(counties)]
    edges += [(i,i+1) for i in range(counties-1)]
    edges += [(i+1,i) for i in range(counties-1)]
    return ChickenpoxData(counts, tuple(f"synthetic-week-{t}" for t in range(observations)),
        tuple(f"county{i}" for i in range(counties)), np.asarray(edges,dtype=np.int64).T, ARCHIVE_SHA256)


def small_settings():
    return dict(runner.FROZEN_POLICY, hidden_dim=8, num_layers=1, latent_dim=3,
                model_specs=runner.FROZEN_SPECS)


def one_prepared(target=365, no_geography=False):
    data = synthetic(counties=4, randomized=True)
    data = runner.restrict_validation_data(data, ForecastProtocol())
    scaler = TrainingScaler.fit(data.counts[:365],365)
    geometry = precompute_chickenpox_geometry(data,no_geography=no_geography)
    template = prepare_anchor_kernel_bank(geometry.template,num_anchors=52,num_scales=3).to("cpu",torch.float32)
    geometry = replace(geometry,template=template.bank)
    bank = bind_chickenpox_history(geometry,data,scaler,target)
    return data,scaler,geometry,replace(template,bank=bank)


@pytest.fixture(autouse=True)
def cpu_scope(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(runner,"require_storage",lambda **kwargs:0)


@pytest.fixture
def frozen_inputs(tmp_path,monkeypatch):
    original = ROOT/"configs/chickenpox_anchor_pilot_v1.json"
    if not original.exists():
        pytest.skip("Frozen direct-anchor config unavailable")
    settings = json.loads(original.read_text())
    manifest = json.loads((ROOT/settings["source_manifest"]).read_text())
    data = replace(synthetic(),counties=tuple(manifest["counties"]))
    monkeypatch.setattr(runner,"load_chickenpox_archive",lambda path:data)
    native = runner.project_path
    def scoped(value):
        value = Path(value)
        if not value.is_absolute() and value.parts and value.parts[0]=="checkpoints":
            value = tmp_path/value
        return native(value)
    monkeypatch.setattr(runner,"project_path",scoped)
    config = tmp_path/"config.json"
    config.write_text(json.dumps(settings,indent=2)+"\n")
    return config,sha(config),settings,data


def rewrite(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")
    return sha(path)


@pytest.mark.parametrize("name",runner.FROZEN_MODELS)
def test_uses_actual_core_constructor_and_all_observed_sources(name):
    data,scaler,geometry,prepared = one_prepared(no_geography=name.endswith("_no_geography"))
    model = runner.make_forecaster(name,small_settings())
    assert type(model) is AnchorKernelModel
    assert model.input_dim == model.num_classes == 1
    assert model.num_anchors == prepared.layout.anchors.numel() == 52
    assert model.num_scales == prepared.num_scales == 3
    assert prepared.layout.anchor_graph.tolist()==list(range(52))
    assert prepared.bank.target_graph.tolist()==[51]
    assert prepared.bank.positions.item()==prepared.bank.intervals[-1].mean().item()
    assert sum(s>=0 for s in prepared.layout.source_at_event)==52
    assert prepared.bank.query_cache.reference_ids.numel()==0
    assert model.geometry=="plain" and model.rate_parameterization=="log"
    np.testing.assert_array_equal(prepared.bank.leaf_x[...,0],
        scaler.transform(data.counts[313:365]).astype(np.float32))
    for scale in range(3):
        torch.testing.assert_close(prepared.operators["plain"].cqq[:,scale],prepared.bank.query_cache.cqq)
    output = model(prepared)
    assert output.shape==(1,4,1) and torch.isfinite(output).all()
    manifest=model.configuration_manifest(prepared)
    assert manifest["implementation_sha256"]==sha(Path(runner.__file__).with_name("anchor_kernel.py"))
    assert manifest["self_term"]=="inclusive forward plus exclusive reverse counts once"


@pytest.mark.parametrize("name",runner.FROZEN_MODELS)
def test_next_week_and_future_poisoning_cannot_change_actual_core_output(name):
    data,scaler,geometry,prepared=one_prepared(no_geography=name.endswith("_no_geography"))
    poisoned=data.counts.copy()
    poisoned[365:]=1e8
    other=bind_chickenpox_history(geometry,replace(data,counts=poisoned),scaler,365)
    changed=replace(prepared,bank=other)
    torch.testing.assert_close(other.leaf_x,prepared.bank.leaf_x,rtol=0,atol=0)
    model=runner.make_forecaster(name,small_settings()).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(changed),model(prepared),rtol=0,atol=0)
    assert not any(hasattr(prepared.bank,n) for n in ("label","labels","truth","target_counts"))
    # The final reserved tail remains masked independently of the source poison.
    masked=runner.restrict_validation_data(replace(data,counts=poisoned),ForecastProtocol())
    assert np.isnan(masked.counts[443:]).all()


@pytest.mark.parametrize("name",["anchor_full_d2","anchor_diagonal_d2","anchor_gain_matched_d2","anchor_scalar_d2"])
def test_gradient_of_forecast_is_zero_to_raw_counts_at_or_after_target(name):
    data,scaler,geometry,prepared=one_prepared()
    # This independent differentiable source audit uses frozen scaler constants.
    # Its value is checked against the actual non-differentiable archive binding.
    raw=torch.tensor(np.nan_to_num(data.counts,nan=1000.),dtype=torch.float64,requires_grad=True)
    mean=torch.tensor(scaler.mean,dtype=torch.float64)
    scale=torch.tensor(scaler.scale,dtype=torch.float64)
    features=((raw[313:365]-mean)/scale).float()[...,None]
    torch.testing.assert_close(features,prepared.bank.leaf_x,rtol=0,atol=0)
    differentiable=replace(prepared,bank=replace(prepared.bank,leaf_x=features))
    model=runner.make_forecaster(name,small_settings())
    model(differentiable).square().sum().backward()
    assert torch.count_nonzero(raw.grad[365:])==0
    assert torch.count_nonzero(raw.grad[:313])==0
    assert torch.count_nonzero(raw.grad[313:365])>0
    assert torch.isfinite(raw.grad).all()


def test_completed_bidirectional_known_history_context_is_intentionally_allowed():
    _,_,_,prepared=one_prepared()
    features=prepared.bank.leaf_x.clone().requires_grad_()
    model=runner.make_forecaster("anchor_full_d2",small_settings())
    context=model.encode_context(replace(prepared,bank=replace(prepared.bank,leaf_x=features)))
    # A message at the FIRST available historical anchor depends on a LATER
    # available observed source. This is legitimate history encoding, and would
    # contradict an internal earlier-anchor causal claim.
    context.layers[0].anchors[0].square().sum().backward()
    assert torch.count_nonzero(features.grad[-1])>0
    assert torch.isfinite(features.grad).all()
    assert len(context.layers)==2


def test_static_preparation_is_reused_and_mask_checked_before_geometry(monkeypatch):
    data=runner.restrict_validation_data(synthetic(counties=4),ForecastProtocol())
    scaler=TrainingScaler.fit(data.counts[:365],365)
    calls=[]
    original=runner.prepare_anchor_kernel_bank
    def counted(*args,**kwargs):
        calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,"prepare_anchor_kernel_bank",counted)
    banks,manifest=runner.prepare_histories(data,ForecastProtocol(),scaler,small_settings(),"cpu")
    assert calls==[{"num_anchors":52,"num_scales":3}]
    assert set(banks)==set(range(52,443))
    assert len({id(b.layout) for b in banks.values()})==1
    assert len({id(b.operators) for b in banks.values()})==1
    assert len({id(b.bank.query_cache) for b in banks.values()})==1
    assert manifest["anchor_graph_indexes"]==list(range(52))
    assert manifest["cached_model_dtype"]=="torch.float32"
    with pytest.raises(ValueError,match="masked before"):
        runner.prepare_histories(synthetic(counties=4),ForecastProtocol(),scaler,small_settings(),"cpu")


def test_rate_groups_and_structural_parameter_activity_are_disclosed():
    _,_,_,prepared=one_prepared()
    settings=dict(runner.FROZEN_POLICY,model_specs=runner.FROZEN_SPECS)
    full=runner.make_forecaster("anchor_full_d2",settings)
    groups=runner.optimizer_groups(full,.0001)
    special=list(full.rate_and_mixture_parameters())
    assert groups[1]["weight_decay"]==0
    assert {id(p) for p in groups[1]["params"]}=={id(p) for p in special}
    assert {id(p) for p in groups[0]["params"]}.isdisjoint({id(p) for p in special})
    assert sum(len(g["params"]) for g in groups)==len(list(full.parameters()))
    activity=runner.parameter_activity(full,prepared,False)
    assert activity["parameters"]==8849
    assert activity["structurally_inactive_reference_parameters"]==528
    assert activity["structurally_active_parameters"]==8321
    _,_,_,zero=one_prepared(no_geography=True)
    no_geo=runner.make_forecaster("anchor_full_d2_no_geography",settings)
    result=runner.parameter_activity(no_geo,zero,True)
    assert result["structurally_inactive_zero_geography_coupling_parameters"]==48
    assert result["structurally_inactive_zero_geography_neighbor_parameters"]==2048
    assert result["structurally_active_parameters"]==6225
    assert torch.count_nonzero(zero.operators["plain"].cqq)==0
    for layer in full.layers:
        torch.testing.assert_close(layer.rates[:,0],torch.tensor([5.,20.,80.]))
        torch.testing.assert_close(layer.mixture.sum(0),torch.ones(8))


def test_runner_pins_masks_and_complete_raw_receipts(frozen_inputs,tmp_path,monkeypatch):
    config,pin,settings,data=frozen_inputs
    original=runner.prepare_histories
    preparation_calls=[]
    def checked(restricted,*args,**kwargs):
        assert np.isnan(restricted.counts[443:]).all()
        preparation_calls.append(kwargs["no_geography"])
        return original(restricted,*args,**kwargs)
    monkeypatch.setattr(runner,"prepare_histories",checked)
    fit_calls=[]
    def fit(model,banks,restricted,scaler,protocol,settings,checkpoint,config_sha):
        assert type(model) is AnchorKernelModel
        assert np.isnan(restricted.counts[443:]).all()
        assert scaler.fit_end==365 and config_sha==pin
        np.testing.assert_array_equal(scaler.mean,data.counts[:365].mean(0))
        assert set(banks)==set(range(52,443))
        for target,prepared in banks.items():
            np.testing.assert_array_equal(prepared.bank.leaf_x[...,0].cpu().numpy(),
                scaler.transform(data.counts[target-52:target]).astype(np.float32))
            assert prepared.layout.anchor_graph.tolist()==list(range(52))
            assert prepared.bank.target_graph.tolist()==[51]
        # Change actual core rates so final selected-state receipt must differ
        # from initialization rather than recording only initial diagnostics.
        with torch.no_grad():
            model.layers[0].raw_damping.add_(.05)
        torch.save({"model":model.state_dict(),"study_config_sha256":pin},checkpoint)
        fit_calls.append(settings["active_model"])
        prediction=data.counts[364:442].copy()
        return prediction,{"best_epoch":2,"history":[{"epoch":2,"mae_cases":2.}],
            "training_seconds":0.,**runner.regression_metrics(prediction,data.counts[365:443])}
    monkeypatch.setattr(runner,"_fit_model",fit)
    output=tmp_path/"receipts"
    requested=["anchor_full_d1","anchor_full_d2_no_geography"]
    result=runner.run_study(config,output,pin,models=requested)
    assert fit_calls==requested and preparation_calls==[False,True]
    assert result["status"]=="complete" and result["final_test_evaluated"] is False
    assert result["internal_earlier_anchor_causal"] is False
    assert result["all_predeclared_models_completed_in_this_run"] is False
    assert result["reserved_numeric_rows_masked_from"]==443
    assert result["study_config_sha256"]==pin
    assert result["protocol_config_sha256"]==settings["protocol_config_sha256"]
    assert result["implementation_sha256"]["anchor_kernel.py"]==sha(Path(runner.__file__).with_name("anchor_kernel.py"))
    for name,record in result["models"].items():
        assert record["mae_cases"]==record["rmse_cases"]==pytest.approx(2.)
        assert record["initial_model_configuration"]["current_damping"]!=record["model_configuration"]["current_damping"]
        assert record["model_configuration"]["num_anchors"]==52
        assert len(record["weekly_validation_metrics"])==78
        assert record["weekly_validation_metrics"][0]["history_range"]==[313,365]
        assert record["weekly_validation_metrics"][-1]["history_range"]==[390,442]
        assert record["checkpoint_sha256"]==sha(record["checkpoint"])
        saved=Path(record["validation_predictions"])
        assert sha(saved)==record["validation_predictions_sha256"]
        with np.load(saved,allow_pickle=False) as artifact:
            assert artifact["prediction"].shape==artifact["truth"].shape==(78,20)
            np.testing.assert_array_equal(artifact["truth"],data.counts[365:443])
            np.testing.assert_array_equal(artifact["target_indices"],np.arange(365,443))
            np.testing.assert_array_equal(artifact["target_dates"],data.dates[365:443])
            np.testing.assert_array_equal(artifact["counties"],data.counties)
            assert artifact["study_config_sha256"].item()==pin
    assert (output/"study_config.json").read_bytes()==config.read_bytes()
    assert json.loads((output/"results.json").read_text())==result
    assert json.loads((output/"progress.json").read_text())==result


def test_real_small_cpu_fit_selected_checkpoint_and_rate_diagnostics(tmp_path):
    protocol=ForecastProtocol(history=52,train_end=54,validation_end=56,observations=58)
    data=runner.restrict_validation_data(synthetic(counties=3,observations=58,randomized=True),protocol)
    scaler=TrainingScaler.fit(data.counts[:54],54)
    settings=dict(small_settings(),epochs=2,active_model="synthetic_actual_core_d2")
    banks,_=runner.prepare_histories(data,protocol,scaler,settings,"cpu")
    model=runner.make_forecaster("anchor_full_d2",settings)
    before={key:p.detach().clone() for key,p in model.named_parameters()}
    checkpoint=tmp_path/"synthetic_core.pt"
    prediction,result=runner._fit_model(model,banks,data,scaler,protocol,settings,checkpoint,"a"*64)
    assert prediction.shape==(2,3) and np.isfinite(prediction).all()
    assert any(not torch.equal(before[key],p) for key,p in model.named_parameters())
    chosen=torch.load(checkpoint,weights_only=True)
    assert chosen["epoch"]==result["best_epoch"]
    assert result["best_epoch"]==min(result["history"],key=lambda r:r["mae_cases"])["epoch"]
    assert chosen["validation_mae"]==pytest.approx(result["mae_cases"])
    assert chosen["study_config_sha256"]=="a"*64
    for record in result["history"]:
        diagnostics=record["kernel_optimization"]
        assert set(diagnostics["gradient_norm_before_clip_mean"])=={"coupling","damping","mixture"}
        assert all(np.isfinite(v) for v in diagnostics["gradient_norm_before_clip_mean"].values())
        assert np.shape(diagnostics["current_rates"])==(2,3,3)
        assert np.shape(diagnostics["current_mixture"])==(2,3,3)
    np.testing.assert_allclose(result["per_county_mae"],np.abs(prediction-data.counts[54:56]).mean(0))


@pytest.mark.parametrize("field",["protocol_config_sha256","source_archive_sha256","source_manifest_sha256"])
def test_pin_failure_prevents_all_preparation(frozen_inputs,tmp_path,monkeypatch,field):
    config,_,settings,_=frozen_inputs
    settings[field]="0"*64
    pin=rewrite(config,settings)
    def forbidden(*args,**kwargs):
        raise AssertionError("No preparation after invalid pin")
    monkeypatch.setattr(runner,"prepare_histories",forbidden)
    with pytest.raises(ValueError,match="SHA256|archive pin"):
        runner.run_study(config,tmp_path/"invalid",pin)
    assert not (tmp_path/"invalid").exists()


@pytest.mark.parametrize("mutation",[
    {"models":["anchor_full_d1"]},{"epochs":1},{"patience":2},{"num_anchors":51},
    {"num_scales":1},{"initial_coupling":[5.,20.,81.]},{"rate_weight_decay":.0001},
    {"history_encoder":"causal"},{"final_test_evaluated":True},{"training_seed":True},
])
def test_frozen_training_and_direct_core_policy_cannot_drift(frozen_inputs,tmp_path,mutation):
    config,_,settings,_=frozen_inputs
    settings.update(mutation)
    pin=rewrite(config,settings)
    with pytest.raises(ValueError,match="Frozen"):
        runner.run_study(config,tmp_path/"invalid",pin)


def test_frozen_model_specification_cannot_replace_architecture(frozen_inputs,tmp_path):
    config,_,settings,_=frozen_inputs
    settings["model_specs"]["anchor_full_d2"]["rate_parameterization"]="softplus"
    pin=rewrite(config,settings)
    with pytest.raises(ValueError,match="specifications"):
        runner.run_study(config,tmp_path/"invalid",pin)


@pytest.mark.parametrize("mutation",[{"history":53},{"validation_end":442},{"epochs":39},{"final_test_evaluated":True}])
def test_common_protocol_drift_rejected(frozen_inputs,tmp_path,mutation):
    config,_,settings,_=frozen_inputs
    common=json.loads((ROOT/settings["protocol_config"]).read_text())
    common.update(mutation)
    changed=tmp_path/"changed_common.json"
    settings["protocol_config_sha256"]=rewrite(changed,common)
    settings["protocol_config"]=str(changed)
    pin=rewrite(config,settings)
    with pytest.raises(ValueError,match="Common protocol"):
        runner.run_study(config,tmp_path/"invalid",pin)


def test_fit_failure_never_emits_complete(frozen_inputs,tmp_path,monkeypatch):
    config,pin,_,_=frozen_inputs
    def failed(*args,**kwargs):
        raise RuntimeError("synthetic core optimization failure")
    monkeypatch.setattr(runner,"_fit_model",failed)
    output=tmp_path/"failed"
    with pytest.raises(RuntimeError,match="synthetic core optimization"):
        runner.run_study(config,output,pin,models=["anchor_full_d1"])
    result=json.loads((output/"progress.json").read_text())
    assert result["status"]=="failed" and result["models"]=={}
    assert result["final_test_evaluated"] is False
    assert not (output/"results.json").exists()


def test_title_before_cuda_guards_reused_exactly(monkeypatch):
    events=[]
    monkeypatch.setattr(predecessor,"set_research_title",lambda:events.append("title"))
    monkeypatch.delenv("ARGGNN_PHYSICAL_GPU",raising=False)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:events.append("cuda"))
    with pytest.raises(RuntimeError,match="project GPU launcher"):
        runner.initialize_worker("cuda")
    assert events==["title"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(RuntimeError,match="empty CUDA"):
        runner.initialize_worker("cpu")
    assert events==["title","title"]


@pytest.mark.parametrize("models",[[],["anchor_full_d1","anchor_full_d1"],["boundary_full"]])
def test_only_distinct_direct_anchor_specs_may_run(frozen_inputs,tmp_path,models):
    config,pin,_,_=frozen_inputs
    with pytest.raises(ValueError,match="distinct nonempty"):
        runner.run_study(config,tmp_path/"invalid",pin,models=models)


def test_existing_output_and_paths_outside_project_are_rejected(frozen_inputs,tmp_path):
    config,pin,_,_=frozen_inputs
    output=tmp_path/"existing"
    output.mkdir()
    sentinel=output/"keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(RuntimeError,match="fresh"):
        runner.run_study(config,output,pin)
    assert sentinel.read_text()=="untouched"
    with pytest.raises(ValueError,match="outside project"):
        runner.run_study(config,"/tmp/forbidden_anchor_output",pin)


def test_cli_has_no_final_test_budget_or_seed_override():
    parser=runner.make_parser()
    options={option for action in parser._actions for option in action.option_strings}
    assert options=={"-h","--help","--config","--expected-config-sha256","--output","--device","--models"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--config","frozen.json","--output","fresh"])




def test_frozen_config_hash_mismatch_prevents_outputs(frozen_inputs,tmp_path):
    config,_,_,_=frozen_inputs
    output=tmp_path/"invalid_pin"
    with pytest.raises(ValueError,match="Anchor Chickenpox config SHA256"):
        runner.run_study(config,output,"0"*64)
    assert not output.exists()


def test_no_geography_structural_inactive_blocks_really_have_zero_gradients():
    _,_,_,prepared=one_prepared(no_geography=True)
    model=runner.make_forecaster("anchor_full_d2_no_geography",small_settings())
    model(prepared).square().sum().backward()
    for layer in model.layers:
        assert layer.raw_coupling.grad is not None
        assert torch.count_nonzero(layer.raw_coupling.grad)==0
        assert torch.count_nonzero(layer.reference_projection.weight.grad)==0
        assert torch.count_nonzero(layer.reference_projection.bias.grad)==0
    for layer in model.encoder.layers:
        assert torch.count_nonzero(layer.neighbor_linear.weight.grad)==0
    assert all(np.isfinite(value) for value in runner._gradient_norms(model).values())


def test_dependency_change_during_fit_is_recorded_as_failed(frozen_inputs,tmp_path,monkeypatch):
    config,pin,_,data=frozen_inputs
    original=runner._implementation_hashes
    calls=[]
    def hashes():
        digest=original()
        calls.append(None)
        if len(calls)>1:
            digest["anchor_kernel.py"]="0"*64
        return digest
    def fit(model,banks,restricted,scaler,protocol,settings,checkpoint,config_sha):
        torch.save({"model":model.state_dict()},checkpoint)
        prediction=data.counts[364:442]
        return prediction,{"best_epoch":0,"history":[],"training_seconds":0.,
            **runner.regression_metrics(prediction,data.counts[365:443])}
    monkeypatch.setattr(runner,"_implementation_hashes",hashes)
    monkeypatch.setattr(runner,"_fit_model",fit)
    output=tmp_path/"dependency_changed"
    with pytest.raises(RuntimeError,match="implementation changed"):
        runner.run_study(config,output,pin,models=["anchor_full_d1"])
    record=json.loads((output/"progress.json").read_text())
    assert record["status"]=="failed" and record["models"]=={}
    assert "implementation changed" in record["failure"]["message"]
    assert not (output/"results.json").exists()



def test_actual_core_regression_output_is_inverse_scaled_without_clipping():
    data,scaler,_,prepared=one_prepared()
    model=runner.make_forecaster("anchor_full_d1",small_settings())
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-100.)
    prediction,metrics=runner.evaluate(model,{365:prepared},np.asarray([365]),scaler,data.counts[[365]])
    expected=scaler.inverse(np.full((1,4),-100.,dtype=np.float32))
    np.testing.assert_array_equal(prediction,expected)
    assert (prediction<0).all()
    assert metrics["mae_cases"]==pytest.approx(np.abs(prediction-data.counts[[365]]).mean())
