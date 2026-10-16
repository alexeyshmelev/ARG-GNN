"""CPU-only six-candidate scheduling over one immutable block per direction.

This callable neither loads checkpoints nor admits or starts jobs. The guarded
caller supplies six independently loaded models, original loaders and registry.
Learned graph operations and per-model recurrent scan order stay unchanged.
"""
from __future__ import annotations
import numpy as np
import torch
from arggnn.american_c3_common_v1 import require
from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from arggnn.american_c3_gnn_dataset_isolated_v2 import (
    AmericanC3WholeTreeDatasetV1, shutdown_dataloader)
from arggnn.american_c3_evaluate_v1 import verify_prediction

IDENTITY = ("record_ids", "sample_nodes", "sample_indices",
            "positions_bp0", "parent_id", "class_order")


def _models_v1(models, class_order):
    require(len(models) == 6 and len({id(m) for m in models}) == 6,
            "Six separate original candidate models required")
    stores = set()
    for model in models:
        require(type(model) is AmericanC3WholeTreeGNNV1
                and model.class_order == tuple(class_order)
                and model.hidden_dim == models[0].hidden_dim,
                "Same original C3 model family and widths")
        model.eval()
        require(not any(m.training for m in model.modules()),
                "Every module is in deterministic eval mode")
        for module in model.modules():
            if isinstance(module, torch.nn.GRU):
                require(module.num_layers == 1 and module.dropout == 0,
                        "Original one-layer zero-dropout recurrent modules")
        for parameter in model.parameters():
            require(parameter.device.type == "cpu"
                    and parameter.dtype == torch.float32,
                    "Original CPU FP32 parameters only")
            address = parameter.untyped_storage().data_ptr()
            require(address not in stores, "Candidate parameter storage must not alias")
            stores.add(address)


def _drain_v1(loaders):
    drains, errors = [], []
    for loader in loaders:
        try:
            result = shutdown_dataloader(loader)
            require(result["drained"], "Every original loader must drain")
            drains.append(result)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise errors[0]
    return drains


@torch.inference_mode()
def interleaved_six_focal_predictions_v1(
        models, forward_loader, reverse_loader, *, records, runtime_guard):
    """Keep six independent states while sharing only observed block features."""
    models = tuple(models)
    require(callable(runtime_guard), "Original live deadline/RSS guard required")
    _models_v1(models, records["class_order"].tolist())
    fd, rd = forward_loader.dataset, reverse_loader.dataset
    require(isinstance(fd, AmericanC3WholeTreeDatasetV1)
            and isinstance(rd, AmericanC3WholeTreeDatasetV1)
            and fd.direction == "forward" and rd.direction == "reverse"
            and fd.plan_ref == rd.plan_ref
            and fd.plan["parent_id"] == rd.plan["parent_id"],
            "Same original observation plan in opposite complete orders")
    require(tuple(fd.plan["class_order"]) == models[0].class_order
            and getattr(reverse_loader, "_iterator", None) is None,
            "Exact class order and reverse loader must not prestart")
    require(all(loader.num_workers == 0 and loader.generator is None
                for loader in (forward_loader, reverse_loader)),
            "Exact original zero-worker loaders and default CPU generator")
    n, d = len(records["record_ids"]), models[0].hidden_dim
    require(n > 0, "All original focal records required")
    saved = [torch.empty((n, d), dtype=torch.float32, device="cpu")
             for _ in models]
    probabilities = [np.empty((n, model.num_classes), dtype=np.float32)
                     for model in models]
    forward_seen = np.zeros(n, dtype=bool)
    reverse_seen = np.zeros(n, dtype=bool)
    forward_blocks = reverse_blocks = 0
    failure = None
    try:
        expected_tree, hidden = 0, [None] * 6
        runtime_guard("before_original_loader_iteration")
        for block in forward_loader:
            runtime_guard("before_original_whole_tree_inference_block")
            require(block.parent_id == fd.plan["parent_id"]
                    and block.plan_ref == fd.plan_ref
                    and block.start_tree == expected_tree,
                    "Every original forward block in order")
            expected_tree = block.stop_tree
            rows = block.focal_rows.numpy().copy()
            require(((rows >= 0) & (rows < n)).all()
                    and len(np.unique(rows)) == len(rows)
                    and not forward_seen[rows].any(),
                    "Unique exact focal forward support")
            block = block.to(torch.device("cpu"), non_blocking=True)
            bank = block.to_model_inputs()
            for k, model in enumerate(models):
                runtime_guard("before_original_candidate_block")
                context = model.encode_bank(bank)
                history, hidden[k] = model.scan_context(
                    context, block.genomic_geometry,
                    direction="forward", hidden=hidden[k])
                if len(rows):
                    saved[k][rows] = history[
                        block.focal_hold, block.focal_sample_column].cpu()
                del context, history
                runtime_guard("after_original_candidate_block")
            forward_seen[rows] = True
            forward_blocks += 1
            del bank, block
            runtime_guard("after_original_whole_tree_inference_block")
        require(expected_tree == fd.plan["trees"]
                and forward_blocks == len(fd) and forward_seen.all(),
                "Complete original forward scan before any reverse scan")
        require(shutdown_dataloader(forward_loader)["drained"],
                "Forward loader drains before reverse starts")
        expected_tree, hidden = fd.plan["trees"], [None] * 6
        runtime_guard("before_original_loader_iteration")
        for block in reverse_loader:
            runtime_guard("before_original_whole_tree_inference_block")
            require(block.parent_id == fd.plan["parent_id"]
                    and block.plan_ref == fd.plan_ref
                    and block.stop_tree == expected_tree,
                    "Every original reverse block in order")
            expected_tree = block.start_tree
            rows = block.focal_rows.numpy().copy()
            require(((rows >= 0) & (rows < n)).all()
                    and len(np.unique(rows)) == len(rows)
                    and not reverse_seen[rows].any(),
                    "Unique exact focal reverse support")
            block = block.to(torch.device("cpu"), non_blocking=True)
            bank = block.to_model_inputs()
            for k, model in enumerate(models):
                runtime_guard("before_original_candidate_block")
                context = model.encode_bank(bank)
                history, hidden[k] = model.scan_context(
                    context, block.genomic_geometry,
                    direction="reverse", hidden=hidden[k])
                if len(rows):
                    local = context[block.focal_hold, block.focal_sample_column]
                    reverse = history[block.focal_hold, block.focal_sample_column]
                    forward = saved[k][rows].to(torch.device("cpu"), non_blocking=True)
                    logits = model.classify_context(local, forward, reverse)
                    probabilities[k][rows] = logits.softmax(-1).cpu().numpy()
                    del local, reverse, forward, logits
                del context, history
                runtime_guard("after_original_candidate_block")
            reverse_seen[rows] = True
            reverse_blocks += 1
            del bank, block
            runtime_guard("after_original_whole_tree_inference_block")
        require(expected_tree == 0 and reverse_blocks == len(rd)
                and reverse_seen.all(),
                "Complete original reverse scan and every focal")
        # PyTorch2.6 zero-worker DataLoader still consumes one default-generator
        # base-seed scalar per iterator. The original six serial pairs used12
        # draws. Shared forward/reverse iterators used2. Reproduce the omitted10
        # draws after deterministic eval, preserving successful-call RNG state.
        # No model/dataset function uses RNG, and no workers or sampler change.
        for _ in range(10):
            torch.empty((), dtype=torch.int64).random_().item()
        results = []
        for k, model in enumerate(models):
            prediction = {name: records[name] for name in IDENTITY}
            prediction["probabilities"] = probabilities[k]
            verify_prediction(prediction, records)
            receipt = dict(
                schema="american_c3_exact_streaming_focal_prediction_v1",
                parent_id=fd.plan["parent_id"], class_order=list(model.class_order),
                block_index_ref=fd.plan_ref,
                focal_registry_ref=fd.plan["focal_registry_ref"],
                observation_ref=fd.plan["observation_ref"],
                focal_records=n, trees=fd.plan["trees"],
                forward_blocks=forward_blocks, reverse_blocks=reverse_blocks,
                tree_encoding_passes=2,
                recurrent_state_reset_only_at_true_chromosome_endpoints=True,
                geometry_uses_global_neighbor_spans=True,
                finite_halo_approximation=False, CPU_focal_state_bytes=n*d*4,
                full_parent_GPU_graph_or_history=False, private_truth_opened=False,
                checkpoint_contract=model.checkpoint_contract(),
                author_Table1_exact_reproduction=False)
            results.append((prediction, receipt))
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            drains = _drain_v1((forward_loader, reverse_loader))
        except BaseException as error:
            if failure is None:
                raise
            if hasattr(failure, "add_note"):
                failure.add_note("Inference drain failed: " + str(error)[:500])
    for _, receipt in results:
        receipt["worker_drain"] = dict(forward=drains[0], reverse=drains[1])
    return results
