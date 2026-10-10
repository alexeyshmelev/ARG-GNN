"""Handmade CPU semantic checks only, without biological arrays or fitting."""
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from arggnn.data import GraphSequence
from arggnn import biological_gst_control_v1 as control

torch.set_num_threads(1)
ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")


def handmade():
    """Three complete star trees with artificial F48 and H020 role counts."""
    nodes = np.arange(280, dtype=np.int64) + 700
    actual = np.r_[0:64, 80:96, 104:112, 144:160, 168:184]
    dynamic = np.arange(184)
    reference = list(range(184, 280))
    samples = [dict(observed_reference_class=(i - 184) // 24 if i >= 184 else None)
               for i in range(280)]
    generator = torch.Generator().manual_seed(918)
    x = torch.randn((843, 48), generator=generator, dtype=torch.float32) * 0.1
    x[:, :6] = 0
    node_ids = torch.tensor((nodes.tolist() + [-1]) * 3, dtype=torch.long)
    for graph in range(3):
        start = graph * 281
        x[start:start + 280, 5] = 1
        x[start + actual, 4] = 1
        for klass in range(4):
            x[start + 184 + klass * 24:start + 208 + klass * 24, klass] = 1
        x[start + 280, 6] = 2
    edges = []
    for graph in range(3):
        start = graph * 281
        root = start + 280
        edges.extend((root, start + leaf) for leaf in range(280))
        edges.extend((start + leaf, root) for leaf in range(280))
    positions = np.array([500050.0, 900050.0], dtype=np.float64)
    sequence = GraphSequence(x=x, edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
        node_graph=torch.arange(3).repeat_interleave(281), node_ids=node_ids,
        clades=torch.zeros((843, 16), dtype=torch.float32),
        intervals=torch.tensor([[0.25, 0.35], [0.35, 0.4], [0.4, 0.5]], dtype=torch.float64),
        query_node_ids=torch.as_tensor(nodes[dynamic].copy()),
        positions=torch.from_numpy(positions / 2000000),
        target_graph=torch.tensor([0, 2], dtype=torch.long),
        window_bounds=torch.tensor([0.25, 0.5], dtype=torch.float64))
    parent = dict(observation=dict(nodes=nodes), dynamics_slots=dynamic, query_slots=actual,
        reference_slots=reference, positions=positions, samples=samples, output_indexes=actual)
    tile = dict(position_indexes=[0, 1], original_tree_ids=[12, 13, 14],
        normalized_interval=[0.25, 0.5], all_tree_nodes_retained=True,
        all280_leaves_retained=True, exact_serialization_roundtrip_checked=True)
    return sequence, parent, tile


def settings():
    return json.loads((ROOT / "configs/biological_gst_control_v1.json").read_text())


def test_all_node_projection_preserves_complete_graph_and_original_coordinates():
    sequence, parent, tile = handmade()
    control.validate_whole_tile(sequence, tile, parent)
    projected = control.source47(sequence)
    assert projected.x.shape == (843, 47)
    assert torch.equal(projected.x, sequence.x[:, control.FEATURE_COLUMNS])
    for name in sequence.as_dict():
        if name != "x":
            assert getattr(projected, name) is getattr(sequence, name)
    changed = sequence.x.clone()
    changed[:, 7] += 1000
    assert torch.equal(control.source47(replace(sequence, x=changed)).x, projected.x)
    assert projected.positions.dtype == projected.intervals.dtype == torch.float64
    assert projected.positions.tolist() == [500050 / 2000000, 900050 / 2000000]


@pytest.mark.parametrize("arm, expected", [("gst32_primary", 43300), ("gst64_capacity_sensitivity", 166468)])
def test_unchanged_gst_factory_and_actual_parameter_counts(arm, expected):
    model = control.model_factory(arm, settings())
    assert sum(p.numel() for p in model.parameters()) == expected == control.parameter_count(model.hidden_dim)
    manifest = model.configuration_provenance()
    assert manifest["control_name"] == "gst_ca_span"
    assert manifest["input_dim"] == 47
    assert manifest["routing"] == "cross_attention"
    assert manifest["gate_variant"] == "paper"
    assert manifest["num_layers"] == 2 and manifest["num_heads"] == 4
    assert not manifest["position_encodings"] and not manifest["learned_numerical_id_embeddings"]


def test_full184_forward_and_actual120_only_direct_loss():
    sequence, parent, _ = handmade()
    torch.manual_seed(101)
    model = control.model_factory("gst32_primary", settings())
    logits = model(control.source47(sequence))
    assert logits.shape == (2, 184, 4)
    logits.retain_grad()
    selected = control.actual_query_logits(logits, parent["output_indexes"])
    mass = torch.zeros_like(selected)
    mass[..., 1] = 100.0
    mass[1, :, 1] = 30.0
    mass[1, :, 2] = 70.0
    loss = control.exact_span_loss(selected, mass)
    loss.backward()
    mixed = sorted(set(range(184)) - set(parent["output_indexes"]))
    assert torch.count_nonzero(logits.grad[:, mixed]) == 0
    assert torch.count_nonzero(logits.grad[:, parent["output_indexes"]]) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(sequence.x[torch.isin(sequence.node_ids, sequence.query_node_ids), 4]) == 360


def test_unscored_complete_graph_influences_readout_through_set_context():
    sequence, _, _ = handmade()
    projected = control.source47(sequence)
    torch.manual_seed(119)
    model = control.model_factory("gst32_primary", settings()).eval()
    x = projected.x.clone()
    x[projected.node_graph == 1, 6:] += 2.5
    with torch.no_grad():
        before = model(projected)
        after = model(replace(projected, x=x))
    assert not torch.allclose(before, after, atol=1e-7, rtol=1e-7)


def test_mixed_flags_source_classes_or_thinned_leaves_are_rejected():
    sequence, parent, tile = handmade()
    changed = sequence.x.clone()
    changed[64, 4] = 1
    with pytest.raises(ValueError, match="query flags"):
        control.validate_whole_tile(replace(sequence, x=changed), tile, parent)
    changed = sequence.x.clone()
    changed[184, 0] = 0
    changed[184, 1] = 1
    with pytest.raises(ValueError, match="class identities"):
        control.validate_whole_tile(replace(sequence, x=changed), tile, parent)
    # Remove one pure source leaf from graph0 and mechanically reindex topology.
    keep = torch.ones(843, dtype=torch.bool)
    keep[184] = False
    mapping = torch.full((843,), -1, dtype=torch.long)
    mapping[keep] = torch.arange(842)
    edge_keep = keep[sequence.edge_index].all(0)
    thinned = replace(sequence, x=sequence.x[keep], node_ids=sequence.node_ids[keep],
        node_graph=sequence.node_graph[keep], clades=sequence.clades[keep],
        edge_index=mapping[sequence.edge_index[:, edge_keep]])
    with pytest.raises(ValueError, match="all280"):
        control.validate_whole_tile(thinned, tile, parent)


def test_raw_zero_probabilities_have_infinite_exact_span_nll():
    probabilities = np.array([[[1., 0., 0., 0.]]], dtype=np.float32)
    mass = np.array([[[0., 30., 0., 70.]]], dtype=np.float64)
    nll, infinite_bp = control.exact_probability_nll(probabilities, mass, [0])
    assert math.isinf(nll) and infinite_bp == 100
    mass = np.array([[[100., 0., 0., 0.]]], dtype=np.float64)
    assert control.exact_probability_nll(probabilities, mass, [0]) == (0.0, 0.0)


def test_source_preflight_imports_neither_torch_nor_numpy():
    code = """
import importlib.abc, sys
class ForbidArrays(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy'):
            raise AssertionError('source preflight attempted array import: '+fullname)
sys.meta_path.insert(0, ForbidArrays())
from arggnn.biological_gst_control_v1 import load_protocol, source_only_receipt, sha, path
p=path('configs/biological_gst_control_v1.json')
cfg=load_protocol(p,sha(p))
print(source_only_receipt(cfg,sha(p))['status'])
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
        text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "frozen_source_contract_checked_only"


def test_handmade_prepared_manifest_preserves_all3718_records_and_exact_grid(tmp_path, monkeypatch):
    """Exercise serializer metadata semantics with artificial records and joins."""
    cfg = settings()
    h020 = json.loads((ROOT / cfg["H020_config"]["path"]).read_text())
    query = np.r_[0:64, 80:96, 104:112, 144:160, 168:184]
    parents = []
    records = []
    for ordinal, (name, trees) in enumerate(zip(control.PARENTS, [955, 897, 946, 920])):
        nodes = np.arange(280, dtype=np.int64) + 1000 + ordinal * 1000
        parent = dict(parent_id=name, split="train" if ordinal < 2 else "val",
            observation=dict(nodes=nodes, entry=dict(handmade_parent=name)),
            query_slots=query, dynamics_slots=np.arange(184), reference_slots=list(range(184, 280)),
            positions=np.arange(8, dtype=np.float64) * 100 + 50,
            edges=np.arange(9, dtype=np.float64) * 100,
            output_indexes=query, retained_indexes=np.arange(8))
        parents.append(parent)
        tiles = []
        for number, ids in enumerate(np.array_split(np.arange(trees), 8)):
            tiles.append(dict(position_indexes=[number], original_tree_ids=ids.tolist(),
                all_tree_nodes_retained=True, all280_leaves_retained=True, query_dynamics=184,
                pure_reference_sources=96, supervised_outputs=120, mixed_latent_outputs=64,
                mixed_query_flag=0, whole_observed_graph=dict(path=f"handmade_{ordinal}_{number}.pt", sha256="unused")))
        records.append(dict(parent_id=name, split=parent["split"], relate_entry=parent["observation"]["entry"],
            query_node_ids=nodes[query].tolist(), dynamics_query_node_ids=nodes[:184].tolist(),
            positions_bp=parent["positions"].tolist(), bin_edges_bp=parent["edges"].tolist(),
            actual_query_output_indexes=query.tolist(), actual_query_sample_indexes=query.tolist(),
            dynamics_sample_indexes=list(range(184)), reference_sample_indexes=list(range(184, 280)),
            retained_fixed100bp_indexes=list(range(8)), tiles=tiles, original_trees_preserved=trees))
    observed_sha = "handmade_observed_hash"
    prepared = dict(status="complete_observed_only_Q184_R96_actualQ120",
        config_sha256=cfg["H020_config"]["sha256"], observed_manifest_sha256=observed_sha,
        source_sha256=h020["source_sha256"], truth_opened=False, parents=records)
    fixture_file = tmp_path / "handmade_prepared.json"
    # File admission is replaced only for the handmade metadata unit test.
    monkeypatch.setattr(control, "checked", lambda record: control.path(record["path"]))

    def admit():
        fixture_file.write_text(json.dumps(prepared))
        digest = control.sha(fixture_file)
        cfg["prepared_manifest"] = dict(path=str(fixture_file), sha256=digest)
        return control.prepared_contract(cfg, parents, fixture_file, digest, observed_sha)

    assert set(admit()) == set(control.PARENTS)
    records[0]["tiles"][0]["original_tree_ids"].pop(0)
    with pytest.raises(ValueError, match="Every original tree ID"):
        admit()
    records[0]["tiles"][0]["original_tree_ids"].insert(0, 119)
    # The inserted ID is a duplicate within the same tile, still leaving original0 absent.
    with pytest.raises(ValueError, match="Every original tree ID"):
        admit()
    records[0]["tiles"][0]["original_tree_ids"][0] = 0
    records[0]["tiles"][0]["position_indexes"] = [1]
    with pytest.raises(ValueError, match="exactly once"):
        admit()
