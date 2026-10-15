"""Prospective H038-only fixed-weight H40 factory, inactive/unexecuted here.

This does not relax the full952 loader and does not implement a real-data loader.
Root must first qualify a separate H018 observation bank with the same feature
definitions and mapping. H040/H042/H043 require separate reviewed count guards.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OWN = ROOT / "research/reviews/H018_genealogy_next_source_v1"
SOURCE_KEYS = ("core_model", "pyg_model", "full_model", "observation_features")


def _verify_sources():
    pins = json.loads((OWN/"pins.json").read_text())
    for key in SOURCE_KEYS:
        r = pins[key]; p = ROOT/r["path"]
        if p.resolve() != p or hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Frozen source feature/model drift: "+key)


def make_frozen_h038_h40_v1(selected_model_state):
    """Root supplies an already frozen CPU state mapping, never a checkpoint path.

    All 38 learned state tensors are loaded strictly into the unchanged class.
    H changes only the plain expected_haplotypes attribute. No real observations,
    training labels, optimizer, calibration, padding or new parameters enter.
    """
    _verify_sources()
    from arggnn.mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
    model = MutationRoutedFullPanelPyGGNNV1(hidden_dim=32, expected_haplotypes=40,
        site_chunk_size=256, checkpoint_sites=True)
    if set(model.state_dict()) != set(selected_model_state):
        raise ValueError("Exact frozen H038 state keys required; no missing/new layers")
    for name, tensor in model.state_dict().items():
        source = selected_model_state[name]
        if source.device.type != "cpu" or source.dtype != tensor.dtype or source.shape != tensor.shape:
            raise ValueError("Exact original CPU state shape/dtype required: "+name)
    model.load_state_dict(selected_model_state, strict=True)
    if any(not model.state_dict()[k].equal(v) for k, v in selected_model_state.items()):
        raise ValueError("Strict frozen weight values differ")
    model.eval()
    model.requires_grad_(False)
    if len(model.state_dict()) != 38 or sum(p.numel() for p in model.parameters()) != 29764:
        raise ValueError("Original H038 parameter identity differs")
    return model


def validate_h018_h40_bank_v1(bank, *, qualified_lineage):
    """Future observation bridge must establish provenance before this shape gate."""
    _verify_sources()
    import torch
    from arggnn.mutation_routed_whole_tree_gnn_v1 import validate_inputs_v1
    if (qualified_lineage.get("schema") != "H018_R16_Q24_h40_observation_prepared_v1"
            or qualified_lineage.get("condition_id") != "H018_R16_20_30Mb_extension_v1"
            or qualified_lineage.get("original_GT_or_coherent_HS_unchanged") is not True
            or qualified_lineage.get("all_original_native_holds_preserved") is not True
            or qualified_lineage.get("source_age_and_haplotype_join_qualified") is not True
            or qualified_lineage.get("feature_semantics_unchanged") is not True
            or qualified_lineage.get("truth_or_AP_read") is not False):
        raise ValueError("Separate qualified H018 observed-only bridge required")
    validate_inputs_v1(bank, expected_haplotypes=40)
    expected = torch.tensor([c for c in range(4) for _ in range(4)]+[-1]*24, dtype=torch.long)
    if (not torch.equal(bank.reference_class, expected)
            or not torch.equal(bank.query_column, torch.arange(16, 40, dtype=torch.long))):
        raise ValueError("Exact real R16/Q24 order/classes required; no duplicate/padded rows")
    return dict(haplotypes=40, references=16, queries=24, new_parameters=0,
                feature_semantics_asserted_by_separate_observed_bridge=True,
                real_data_accuracy_qualified=False)
