"""Bind the fresh static executor to one authenticated fresh-cohort runtime."""
import types

def _train_step(self, model, bundle, targets, optimizer, metadata, *,
                resource_check, profile=None):
    self._check_bundle(model, bundle, metadata, "train")
    if (targets.receipt.get("parent_registry_ref") != self.registry.reference
            or targets.receipt.get("split") != "train"):
        raise ValueError("Separate registered TRAIN target join required")
    return self.execution.train_step_v1(
        model, bundle, targets, optimizer, metadata, resource_check=resource_check,
        holds_per_block=64, haplotypes_per_gru=64, history_holds_per_window=1024,
        profile=profile)

def _predict_logits(self, model, bundle, metadata, *, resource_check, profile=None):
    split = self._check_bundle(model, bundle, metadata)
    if split not in ("validation", "test") or not callable(resource_check):
        raise ValueError("Registered observation-only held-out prediction required")
    import torch
    model.eval()
    resource_check("fresh_TreeLoom_prediction")
    with torch.no_grad():
        executor = self.execution.BoundedExecutionV1(
            model, bundle, metadata, resource_check=resource_check,
            holds_per_block=64, haplotypes_per_gru=64, history_holds_per_window=1024,
            profile=profile)
        return executor.forward()

def install_into_runtime(runtime):
    from . import bounded_execution_v2 as execution
    if not hasattr(runtime, "frozen") or not hasattr(runtime, "registry"):
        raise ValueError("Existing authenticated FreshTreeLoomRuntime required")
    # One task owns this interpreter. No historical module or live worker changes.
    execution.frozen = runtime.frozen
    runtime.execution = execution
    runtime.train_step = types.MethodType(_train_step, runtime)
    runtime.predict_logits = types.MethodType(_predict_logits, runtime)
    return dict(execution_variant="full_parent_parallel_history_v2",
                GNN_holds_per_block=64, GRU_haplotypes_per_tile=64,
                history_holds_per_window=1024, gradient_detach=False,
                optimizer_or_model_weights_changed=False)
