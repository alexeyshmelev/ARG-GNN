"""Scientific protocol and scattered parent-coordinate evaluation contracts."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
import torch
from test_boundary_memory import sequence, cached
from arggnn.train_boundary import checked_protocol, evaluate


def protocol():
    return {"splits": {"train": 1, "val": 1, "test": 0},
            "boundary_dataset_manifest_sha256": "a"*64,
            "boundary_preparation_sha256": "b"*64,
            "selection": "minimum mean parent-validation NLL"}


@pytest.mark.parametrize("change", ["test", "manifest", "preparation", "selection"])
def test_mechanism_runner_rejects_unbound_or_test_protocol(tmp_path, change):
    settings = protocol()
    if change == "test":
        settings["splits"]["test"] = 1
    elif change == "manifest":
        del settings["boundary_dataset_manifest_sha256"]
    elif change == "preparation":
        settings["boundary_preparation_sha256"] = "Z"*64
    else:
        settings["selection"] = "maximum observed validation accuracy"
    path = tmp_path/"config.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(RuntimeError):
        checked_protocol(path)


class UniformModel(torch.nn.Module):
    def forward(self, bank):
        return torch.zeros(bank.positions.numel(), bank.query_node_ids.numel(), 3)


def case_and_tiles():
    bank = cached(sequence())
    labels = torch.zeros(4, 2, dtype=torch.long)
    tiles = []
    for indices in (torch.tensor([0, 1]), torch.tensor([2, 3])):
        tile_bank = replace(bank, positions=bank.positions[indices],
                            target_graph=bank.target_graph[indices])
        tiles.append(SimpleNamespace(bank=tile_bank, labels=labels[indices],
                                     parent_position_indices=indices))
    case = SimpleNamespace(labels=labels, positions=bank.positions,
                           query_node_ids=bank.query_node_ids, tiles=tiles[::-1],
                           metadata={"replicate": 0})
    return case, tiles


def test_evaluation_scatters_reordered_tiles_at_exact_parent_coordinates():
    case, tiles = case_and_tiles()
    banks = {id(t): t.bank for t in tiles}
    records, probabilities = evaluate(UniformModel(), [case], banks)
    torch.testing.assert_close(probabilities[0], torch.full((4, 2, 3), 1/3))
    assert records[0]["observations"] == 8
    assert records[0]["accuracy_uniform_grid"] == 1.


@pytest.mark.parametrize("fault", ["duplicate", "negative", "missing", "labels", "coordinates", "query"])
def test_evaluation_rejects_misaligned_or_incomplete_parent_targets(fault):
    case, tiles = case_and_tiles()
    if fault == "duplicate":
        tiles[0].parent_position_indices = torch.tensor([0, 0])
    elif fault == "negative":
        tiles[0].parent_position_indices = torch.tensor([-1, 1])
    elif fault == "missing":
        case.tiles = [tiles[0]]
    elif fault == "labels":
        tiles[0].labels = torch.ones_like(tiles[0].labels)
    elif fault == "coordinates":
        tiles[0].bank = replace(tiles[0].bank, positions=tiles[0].bank.positions+.01)
    else:
        tiles[0].bank = replace(tiles[0].bank, query_node_ids=tiles[0].bank.query_node_ids.flip(0))
    with pytest.raises(RuntimeError):
        evaluate(UniformModel(), [case], {id(t): t.bank for t in tiles})
