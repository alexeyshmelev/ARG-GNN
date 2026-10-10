"""Source guards and handmade semantic contracts, never biological arrays."""
from copy import deepcopy
from dataclasses import replace
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arggnn import biological_mechanism_controls_v1 as controls
from arggnn import anchor_kernel
from arggnn.anchor_kernel import AnchorKernelModel, AnchorKernelLayer, prepare_anchor_kernel_bank
from arggnn.biological_candidate import optimizer_groups
from test_anchor_kernel import prepared, sequence, cached, synthetic_age_cache, dense_action


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(1)


def model(name):
    torch.manual_seed(2910)
    return controls.make_control(name, input_dim=4, num_classes=3, hidden_dim=8,
        num_layers=2, latent_dim=3, num_anchors=5,
        initial_coupling=(.8, 2., 5.), initial_damping=.6).double()


def protocol():
    path = controls.frozen.path("configs/biological_mechanism_controls_v1.json")
    return controls.load_protocol(path, controls.frozen.sha(path), controls.BASE_CONFIG_SHA256)


@pytest.mark.parametrize("name", controls.CONTROL_NAMES)
def test_factory_fixed_biological_parameter_counts_and_same_parameter_schema(name):
    cfg, _, _ = protocol()
    actual = controls.model_factory(name, cfg)
    count = sum(p.numel() for p in actual.parameters())
    assert count == controls.CONTROL_SPECS[name]["parameters"]
    depth = 1 if name == "bio_mc_v1_age_d1" else 2
    baseline = AnchorKernelModel(input_dim=47, num_classes=4, hidden_dim=32,
        num_layers=2, latent_dim=8, num_scales=3, num_anchors=32,
        kernel_depth=depth, geometry="age", initial_coupling=(5., 20., 80.))
    assert [(n,tuple(p.shape)) for n,p in actual.named_parameters()] == [
        (n,tuple(p.shape)) for n,p in baseline.named_parameters()]
    assert list(actual.state_dict()) == list(baseline.state_dict())


@pytest.mark.parametrize("name", controls.CONTROL_NAMES)
def test_all_control_groups_are_finite_active_and_optimizer_excludes_rates_from_decay(name):
    net = model(name)
    output = net(prepared())
    assert output.shape == (7, 3, 3)
    loss = (output * torch.arange(output.numel(),dtype=output.dtype).reshape_as(output)/100).sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(torch.count_nonzero(p.grad) > 0 for p in net.parameters())
    cfg = {"training": {"weight_decay": .0001}}
    groups = optimizer_groups(net, cfg)
    assert groups[1]["weight_decay"] == 0
    assert {id(p) for p in groups[1]["params"]} == {id(p) for p in net.rate_and_mixture_parameters()}
    before = {n:p.detach().clone() for n,p in net.named_parameters()}
    torch.optim.AdamW(groups,lr=.001).step()
    assert all(torch.count_nonzero(p-before[n]) > 0 for n,p in net.named_parameters())


@pytest.mark.parametrize("name", controls.CONTROL_NAMES)
def test_full_model_unary_interval_serialization_and_readout_contracts(name):
    net = model(name).eval()
    original = prepared()
    reference = net(original)
    changed = prepared(sequence(unary=True,split=True,serialize=True))
    torch.testing.assert_close(net(changed),reference,rtol=3e-11,atol=3e-11)
    bank = original.bank
    order = torch.arange(len(bank.positions)-1,-1,-1)
    permuted = replace(bank,positions=bank.positions[order],target_graph=bank.target_graph[order])
    age = synthetic_age_cache(permuted)
    pack = prepare_anchor_kernel_bank(permuted,age,num_anchors=5,num_scales=3)
    torch.testing.assert_close(net(pack),reference[order],rtol=3e-12,atol=3e-12)
    extra = replace(bank,positions=torch.cat((bank.positions,bank.positions[[2]])),
        target_graph=torch.cat((bank.target_graph,bank.target_graph[[2]])))
    pack = prepare_anchor_kernel_bank(extra,synthetic_age_cache(extra),num_anchors=5,num_scales=3)
    torch.testing.assert_close(net(pack)[:-1],reference,rtol=3e-12,atol=3e-12)
    torch.testing.assert_close(net(pack)[-1],reference[2],rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("name", [
    "bio_mc_v1_scalar_age_d2",
    "bio_mc_v1_age_propagation_only_d2",
    "bio_mc_v1_age_source_only_d2",
    "bio_mc_v1_age_gainmatched_d2",
    "bio_mc_v1_noncontext_age_d2",
    "bio_mc_v1_age_d1",
])
def test_selected_source_and_matching_ones_and_every_action_match_dense_oracle(name):
    pack, net = prepared(), model(name)
    if isinstance(net, controls.SplitAgeAnchorKernelModel):
        operators = net.selected_operators(pack)
    else:
        operators = pack.operators[net.geometry]
    h = net.encoder(pack.bank.leaf_x,pack.bank.query_cache.normalized_response)
    query = h[:,pack.bank.query_cache.query_indices]
    refs = h[:,pack.bank.query_cache.reference_indices][pack.layout.anchor_graph]
    original_anchors = query[pack.layout.anchor_graph]
    anchors = original_anchors
    targets = query[pack.bank.target_graph]
    result = net.encode_context(pack)
    for layer, actual in zip(net.layers,result.layers):
        incoming = original_anchors if isinstance(net,controls.NoncontextualAnchorKernelModel) else anchors
        q,r = layer.query_projection(incoming).tanh(),layer.reference_projection(refs).tanh()
        cqr = operators.cqr[pack.layout.anchor_graph]
        source = q[:,None]-(cqr@r[:,None])*layer.rates[None,:,None,:]
        mass = torch.ones_like(source)-cqr.sum(-1)[...,None]*layer.rates[None,:,None,:]
        torch.testing.assert_close(actual.sources,source,rtol=3e-11,atol=3e-11)
        torch.testing.assert_close(actual.mass_sources,mass,rtol=0,atol=0)
        values = dense_action(pack,operators,layer.rates,layer.damping,
            torch.stack((source,mass),dim=-1),net.coupling)
        mixed = (values*layer.mixture[None,:,None,:,None]).sum(1)
        torch.testing.assert_close(actual.numerator,mixed[...,0],rtol=3e-11,atol=3e-11)
        torch.testing.assert_close(actual.mass,mixed[...,1],rtol=3e-11,atol=3e-11)
        msg = mixed[...,0]/mixed[...,1]
        assert msg.abs().max() <= 1+1e-12
        anchors = layer.update(anchors,msg[pack.layout.anchor_output])
        targets = layer.update(targets,msg[pack.layout.target_output])
    torch.testing.assert_close(result.targets,targets,rtol=3e-11,atol=3e-11)


def test_split_blocks_alias_exact_selected_arrays_and_do_not_mutate_bank():
    pack = prepared()
    for name,qq,qr in [
        ("bio_mc_v1_age_source_only_d2","plain","age"),
        ("bio_mc_v1_age_propagation_only_d2","age","plain")]:
        net = model(name)
        selected = net.selected_operators(pack)
        assert selected.cqq is pack.operators[qq].cqq
        assert selected.eigenvectors is pack.operators[qq].eigenvectors
        assert selected.cqr is pack.operators[qr].cqr
        before = controls.frozen.fingerprint_sha(pack)
        net(pack)
        assert controls.frozen.fingerprint_sha(pack) == before
        assert net.configuration_manifest()["physical_full_boundary_Kron_response"] is False


def test_noncontextual_age_second_sources_ignore_first_message_feedback():
    pack = prepared()
    net = model("bio_mc_v1_noncontext_age_d2")
    before = net.encode_context(pack)
    with torch.no_grad():
        net.layers[0].message_projection.weight.mul_(20)
        net.layers[0].gate.bias.add_(8)
    after = net.encode_context(pack)
    torch.testing.assert_close(before.layers[1].sources,after.layers[1].sources,rtol=0,atol=0)
    assert (before.targets-after.targets).abs().max() > 1e-5


def test_pointwise_matches_independent_scalar_loop_and_uses_same_qr_mass():
    torch.manual_seed(793)
    layer = AnchorKernelLayer(8,3,3,[.8,2.,5.],.6).double()
    query = torch.randn(2,3,8,dtype=torch.float64)
    refs = torch.randn(2,2,8,dtype=torch.float64)
    cqr = -torch.rand(2,3,3,2,dtype=torch.float64)
    actual = controls.pointwise_reference_message(layer,cqr,query,refs)
    q,r = layer.query_projection(query).tanh(),layer.reference_projection(refs).tanh()
    expected = torch.zeros(2,3,3,dtype=torch.float64)
    for g in range(2):
        for u in range(3):
            for h in range(3):
                for s in range(3):
                    source = q[g,u,h] - layer.rates[s,h] * sum(cqr[g,s,u,v]*r[g,v,h] for v in range(2))
                    mass = 1 - layer.rates[s,h] * sum(cqr[g,s,u,v] for v in range(2))
                    torch.testing.assert_close(actual.source[g,s,u,h],source,rtol=3e-12,atol=3e-12)
                    torch.testing.assert_close(actual.mass_source[g,s,u,h],mass,rtol=3e-12,atol=3e-12)
                    expected[g,u,h] += layer.mixture[s,h]*layer.damping[s,h]/(1+layer.damping[s,h])*source/mass
    torch.testing.assert_close(actual.message,expected,rtol=3e-12,atol=3e-12)
    assert actual.message.abs().max() <= 1
    assert ((actual.bounded_gain>0)&(actual.bounded_gain<1)).all()


def test_pointwise_no_query_context_or_coordinate_exchange():
    torch.manual_seed(817)
    layer = AnchorKernelLayer(8,3,3,[.8,2.,5.],.6).double()
    query = torch.randn(3,3,8,dtype=torch.float64)
    refs = torch.randn(3,2,8,dtype=torch.float64)
    cqr = -torch.rand(3,3,3,2,dtype=torch.float64)
    before = controls.pointwise_reference_message(layer,cqr,query,refs).message
    perturbed = query.clone()
    perturbed[1,1] += 30
    after = controls.pointwise_reference_message(layer,cqr,perturbed,refs).message
    mask = torch.ones(3,3,dtype=torch.bool)
    mask[1,1] = False
    torch.testing.assert_close(before[mask],after[mask],rtol=0,atol=0)
    assert (before[1,1]-after[1,1]).abs().max()>1e-5


def test_pointwise_first_kernel_is_weight_copied_d1_and_second_kernel_is_never_called(monkeypatch):
    pack = prepared()
    net = model("bio_mc_v1_age_d1_plus_pointwise")
    d1 = model("bio_mc_v1_age_d1")
    d1.load_state_dict({k:v for k,v in net.state_dict().items() if not k.startswith("layers.1.")})
    called = []
    original = anchor_kernel.anchor_kernel_action
    def observed_action(*args,**kwargs):
        called.append(kwargs.get("coupling"))
        return original(*args,**kwargs)
    monkeypatch.setattr(anchor_kernel,"anchor_kernel_action",observed_action)
    def poison(*args,**kwargs):
        raise AssertionError("No second global action")
    monkeypatch.setattr(net.layers[1],"messages",poison)
    actual = net.encode_context(pack)
    assert called == ["full"]
    first = d1.encode_context(pack)
    torch.testing.assert_close(actual.kernel_message.numerator,first.layers[0].numerator,rtol=0,atol=0)
    torch.testing.assert_close(actual.kernel_message.mass,first.layers[0].mass,rtol=0,atol=0)
    manifest = net.configuration_manifest(pack)
    assert manifest["kernel_depth"]==1 and manifest["pointwise_depth"]==1
    assert manifest["second_layer_kernel_called"] is False
    assert manifest["pointwise_parameter_padding"] is False


@pytest.mark.parametrize("field", ["raw_coupling","raw_damping"])
def test_pointwise_nonfinite_rates_reject_without_floor(field):
    layer = AnchorKernelLayer(8,3,3,[.8,2.,5.],.6).double()
    with torch.no_grad():
        getattr(layer,field).fill_(1000)
    with pytest.raises(ValueError,match="finite and positive"):
        controls.pointwise_reference_message(layer,-torch.ones(1,3,3,1,dtype=torch.float64),
            torch.ones(1,3,8,dtype=torch.float64),torch.ones(1,1,8,dtype=torch.float64))


def test_pointwise_invalid_qr_and_constructor_rejected():
    layer = AnchorKernelLayer(8,3,3,[.8,2.,5.],.6).double()
    with pytest.raises(ValueError,match="QR nonpositive"):
        controls.pointwise_reference_message(layer,torch.ones(1,3,3,1,dtype=torch.float64),
            torch.ones(1,3,8,dtype=torch.float64),torch.ones(1,1,8,dtype=torch.float64))
    with pytest.raises(ValueError,match="age/full"):
        controls.OneKernelPointwiseModel(geometry="plain")
    with pytest.raises(ValueError,match="exactly two"):
        controls.OneKernelPointwiseModel(geometry="age",kernel_depth=1)
    with pytest.raises(ValueError,match="Unknown"):
        controls.make_control("candidate_age_d2")


@pytest.mark.parametrize("change", ["models","geometry","training","source","reference","spec"])
def test_frozen_extension_config_mutations_rejected_before_any_observations(tmp_path,change):
    cfg, _, _ = protocol()
    bad = deepcopy(cfg)
    if change=="models": bad["models"].append("candidate_age_d2")
    elif change=="geometry": bad["geometry"]["num_anchors"]=64
    elif change=="training": bad["training"]["epochs"]=31
    elif change=="source": bad["source_sha256"]["src/arggnn/anchor_controls.py"]="0"*64
    elif change=="reference": bad["reference_config"]["sha256"]="0"*64
    else: bad["model_specs"]["bio_mc_v1_age_d1"]["kernel_depth"]=2
    path=tmp_path/"bad-config.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        controls.load_protocol(path,controls.frozen.sha(path),controls.BASE_CONFIG_SHA256)


def test_source_only_protocol_checks_never_open_bank_truth_or_observation(monkeypatch):
    def poison(*args,**kwargs):
        raise AssertionError("No observed/truth/bank operation in source check")
    for name in ("observed_contract","load_bank","prepare","train","ExternalTruthCapability"):
        monkeypatch.setattr(controls.frozen,name,poison)
    monkeypatch.setattr(torch,"load",poison)
    cfg, base, _ = protocol()
    assert base["models"] == ["candidate_age_d2","candidate_plain_d2","matched_gru_d2"]
    assert cfg["models"] == list(controls.CONTROL_NAMES)
    assert cfg["geometry"] == base["geometry"]
    assert cfg["training"] == base["training"]
    assert cfg["metrics"] == base["metrics"]
    assert cfg["external_truth_manifest"] == base["external_truth_manifest"]


@pytest.mark.parametrize("path", [
    "data/biological_candidate_pilot_v1/new",
    "results/biological_candidate_pilot_v1_seed101",
    "results/biological_mechanism_controls_v1_new/nested",
    "../outside"])
def test_fresh_output_scope_rejects_frozen_or_nonextension_names(path):
    with pytest.raises(ValueError):
        controls.require_output_namespace(controls.ROOT/path)


def test_fresh_output_scope_and_reuse_rejection(tmp_path,monkeypatch):
    monkeypatch.setattr(controls,"ROOT",tmp_path)
    (tmp_path/"results").mkdir()
    output=tmp_path/"results/biological_mechanism_controls_v1_fixture"
    controls.require_output_namespace(output)
    output.mkdir()
    with pytest.raises(ValueError,match="fresh"):
        controls.require_output_namespace(output)


def test_fake_prepared_metadata_binds_original_config_and_rejects_new_extraction(monkeypatch):
    _,base,_=protocol()
    nodes=np.arange(280)+1000
    parents=[]
    rows=[]
    for parent_id in controls.frozen.PARENTS:
        parent=dict(parent_id=parent_id,split="train" if "_train_" in parent_id else "val",
            observation={"entry":{"kind":"handmade_observed_only"},"nodes":nodes},
            query_slots=np.arange(120),dynamics_slots=np.arange(184),
            output_indexes=np.arange(120),positions=np.array([50.,150.]),
            edges=np.array([0.,100.,200.]),retained_indexes=np.arange(2))
        parents.append(parent)
        rows.append(dict(parent_id=parent_id,split=parent["split"],relate_entry=parent["observation"]["entry"],
            query_node_ids=nodes[:120].tolist(),dynamics_query_node_ids=nodes[:184].tolist(),
            positions_bp=[50.,150.],bin_edges_bp=[0.,100.,200.],
            actual_query_output_indexes=list(range(120)),actual_query_sample_indexes=list(range(120)),
            dynamics_sample_indexes=list(range(184)),retained_fixed100bp_indexes=[0,1],tiles=[]))
    prepared_metadata=dict(status="complete_observed_only_Q184_R96_actualQ120",
        config_sha256=controls.BASE_CONFIG_SHA256,observed_manifest_sha256="a"*64,
        source_sha256=base["source_sha256"],truth_opened=False,parents=rows)
    records=controls.validate_prepared_records(prepared_metadata,base,"a"*64,parents)
    assert len(records)==4
    for key,new in [
        ("config_sha256","b"*64),
        ("source_sha256",{}),
        ("truth_opened",True)]:
        changed=deepcopy(prepared_metadata)
        changed[key]=new
        with pytest.raises(ValueError,match="Original observed prepared"):
            controls.validate_prepared_records(changed,base,"a"*64,parents)
    bad=deepcopy(prepared_metadata)
    bad["parents"][0]["actual_query_output_indexes"][0]=184
    with pytest.raises(ValueError,match="joins changed"):
        controls.validate_prepared_records(bad,base,"a"*64,parents)


def test_runner_source_has_pre_torch_gpu_guard_no_prepare_mode_and_unchanged_truth_lease_calls():
    module=Path(controls.__file__).read_text()
    script=controls.frozen.path("scripts/train_biological_mechanism_controls_v1.py").read_text()
    assert module.index("validate_lease()") < module.index("import torch")
    assert script.index("validate_lease()") < script.index("from arggnn import biological_mechanism_controls")
    assert 'add_argument("--prepare"' not in script
    assert "--prepare-observed-cpu" not in script
    tree=ast.parse(module)
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
    text=ast.get_source_segment(module,train)
    assert text.index("configure_fit(") < text.index("ExternalTruthCapability(")
    assert text.index("seal_prediction(") < text.index('"VAL_after_sealed_prediction"')
    assert "frozen.observed_contract(" in text
    assert "load_bank(" in text
    assert "validate_prepared_records(" in text
    assert text.count("verify_bindings(")>=4
    assert "check_fit_lease(" in text
    assert "prepared.bank" not in text  # all controls consume the original fixed prepared object
    assert "precompute_" not in text
