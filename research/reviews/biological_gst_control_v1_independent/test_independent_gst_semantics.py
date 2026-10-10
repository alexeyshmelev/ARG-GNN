"""Independent GST audit: handmade CPU tensors and source AST only."""
from dataclasses import replace
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
from arggnn import biological_gst_control_v1 as control
from arggnn import biological_candidate as h020
from arggnn.gst_baseline import GSTAdaptation

spec = importlib.util.spec_from_file_location(
    "_handmade_gst_review_fixtures", ROOT / "tests/test_biological_gst_control_v1.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
torch.set_num_threads(1)


def cfg():
    return json.loads((ROOT / "configs/biological_gst_control_v1.json").read_text())


def refined_unscored_graph(sequence):
    # Reproduce the artificial middle graph twice with split physical width.
    choices = [0, 1, 1, 2]
    xs, ids, groups, clades, edges = [], [], [], [], []
    offset = 0
    for group, source_graph in enumerate(choices):
        old = (sequence.node_graph == source_graph).nonzero().flatten()
        mapping = torch.full((len(sequence.x),), -1, dtype=torch.long)
        mapping[old] = torch.arange(len(old)) + offset
        selected = sequence.edge_index[:, sequence.node_graph[sequence.edge_index[0]] == source_graph]
        edges.append(mapping[selected])
        xs.append(sequence.x[old])
        ids.append(sequence.node_ids[old])
        clades.append(sequence.clades[old])
        groups.append(torch.full((len(old),), group, dtype=torch.long))
        offset += len(old)
    return replace(sequence, x=torch.cat(xs), node_ids=torch.cat(ids),
                   clades=torch.cat(clades), node_graph=torch.cat(groups),
                   edge_index=torch.cat(edges, dim=1),
                   intervals=torch.tensor([[.25,.35],[.35,.375],[.375,.4],[.4,.5]],
                                          dtype=torch.float64),
                   target_graph=torch.tensor([0,3], dtype=torch.long))


def test_full280_node_span_control_refinement_and_record_difference():
    sequence, _, _ = fixtures.handmade()
    original = control.source47(sequence)
    refined = refined_unscored_graph(original)
    torch.manual_seed(6719)
    span = control.model_factory("gst32_primary", cfg()).eval()
    record_settings = dict(cfg()["architecture"], token_weighting="record")
    record = GSTAdaptation(hidden_dim=32, **record_settings).eval()
    record.load_state_dict(span.state_dict())
    with torch.no_grad():
        span_before, span_after = span(original), span(refined)
        record_before, record_after = record(original), record(refined)
    assert torch.allclose(span_before, span_after, atol=3e-6, rtol=3e-6)
    assert not torch.allclose(record_before, record_after, atol=1e-6, rtol=1e-6)
    assert original.query_node_ids.numel() == refined.query_node_ids.numel() == 184
    assert refined.x.shape == (1124,47)


def test_all_node_branch_projection_and_arbitrary_id_renaming_preserve_logits():
    sequence, _, _ = fixtures.handmade()
    torch.manual_seed(381)
    model = control.model_factory("gst32_primary", cfg()).eval()
    branch_changed = sequence.x.clone()
    branch_changed[:,7] += torch.arange(len(sequence.x)).float()*100
    changed = replace(sequence,x=branch_changed,
                      node_ids=torch.where(sequence.node_ids>=0,sequence.node_ids+12345,
                                           sequence.node_ids),
                      query_node_ids=sequence.query_node_ids+12345)
    with torch.no_grad():
        original = model(control.source47(sequence))
        revised = model(control.source47(changed))
    assert torch.equal(control.source47(sequence).x,control.source47(changed).x)
    assert torch.equal(original,revised)


def test_span_loss_and_raw_parent_nll_agree_exactly_with_h020_definitions():
    torch.manual_seed(810)
    full = torch.randn((2,184,4),dtype=torch.float64,requires_grad=True)
    _, parent, _ = fixtures.handmade()
    selected = control.actual_query_logits(full,parent["output_indexes"])
    mass = torch.zeros_like(selected)
    mass[0,:,0] = 30.0
    mass[0,:,2] = 70.0
    mass[1,:,1] = 12.5
    mass[1,:,3] = 27.5
    observed = control.exact_span_loss(selected,mass)
    reference = -(mass*(selected-torch.logsumexp(selected,dim=-1,keepdim=True))).sum()/mass.sum()
    assert torch.allclose(observed,reference,atol=1e-14,rtol=1e-14)
    observed.backward()
    mixed = sorted(set(range(184))-set(parent["output_indexes"]))
    assert torch.count_nonzero(full.grad[:,mixed])==0
    p = torch.softmax(selected.detach(),dim=-1).numpy().astype(np.float32)
    m = mass.numpy()
    for columns in [np.arange(64),np.arange(64,120),np.arange(120)]:
        assert control.exact_probability_nll(p,m,columns)==h020.exact_probability_nll(p,m,columns)
    p[0,0,0]=0.0
    assert control.exact_probability_nll(p,m,[0])==h020.exact_probability_nll(p,m,[0])
    assert np.isinf(control.exact_probability_nll(p,m,[0])[0])


def test_frozen_loss_selection_support_and_capacity_configuration():
    gst_cfg = cfg()
    base = json.loads((ROOT/"configs/biological_candidate_pilot_v1.json").read_text())
    for name in ["epochs","patience","learning_rate","weight_decay","gradient_clip_norm",
                 "selection","supervision","VAL_calibration_fit"]:
        assert gst_cfg["training"][name]==base["training"][name]
    assert gst_cfg["metrics"]==base["metrics"]
    assert gst_cfg["seeds"]==base["seeds"]==[101,202,303]
    assert gst_cfg["models"]==["gst32_primary","gst64_capacity_sensitivity"]
    assert gst_cfg["model_widths"]=={"gst32_primary":32,"gst64_capacity_sensitivity":64}
    assert gst_cfg["analytic_parameters"]=={"gst32_primary":43300,
                                          "gst64_capacity_sensitivity":166468}
    assert gst_cfg["input"]["feature_columns"]==[i for i in range(48) if i!=7]
    assert gst_cfg["input"]["one_complete_existing_tile_per_forward"] is True
    assert gst_cfg["input"]["graph_subset_batching"] is False
    assert gst_cfg["input"]["Kron_bank_loading_or_reconstruction"] is False


def test_source_seals_all_epoch_predictions_before_score_and_no_kron_loading():
    tree=ast.parse((ROOT/"src/arggnn/biological_gst_control_v1.py").read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
    epoch=next(n for n in ast.walk(train) if isinstance(n,ast.For)
               and isinstance(n.target,ast.Name) and n.target.id=="epoch")
    blocks=[n for n in epoch.body if isinstance(n,ast.For)
            and isinstance(n.target,ast.Name) and n.target.id=="name"]
    assert len(blocks)==2
    calls=lambda n: {a.func.id for a in ast.walk(n) if isinstance(a,ast.Call)
                     and isinstance(a.func,ast.Name)}
    assert "sealed_prediction" in calls(blocks[0])
    assert "ExternalTruthCapability" not in calls(blocks[0])
    assert "exact_probability_nll" not in calls(blocks[0])
    assert {"ExternalTruthCapability","exact_probability_nll"}.issubset(calls(blocks[1]))
    assert blocks[0].end_lineno<blocks[1].lineno
    assert not any(isinstance(n,ast.Name) and n.id=="load_bank" for n in ast.walk(tree))
    whole=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="load_whole")
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=="load" and isinstance(n.func.value,ast.Name)
               and n.func.value.id=="torch" for n in ast.walk(whole))
    selected=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                  and n.name=="sealed_prediction")
    assert "actual_query_logits" in calls(selected)
