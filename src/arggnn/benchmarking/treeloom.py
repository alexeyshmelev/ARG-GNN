"""Public TreeLoom facade over the pinned corrected implementation.

Importing this module imports no Torch/PyG and reads no data. Model construction
is CPU-only. Numerical stages are called only inside the caller's separately
admitted worker, with its real resource callback. No launch or checkpoint reader
is provided here.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_MODULE = "research.reviews.frozen_benchmark_training_ready_20261017_v1.american_full30mb_frozen_v1"
PREPARATION_MODULE = "research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_preparation_v1"
EXECUTION_MODULE = "research.reviews.TreeLoom_GPU_history_source_20261017_v1.bounded_execution_v1"
MANIFEST_REF = {
    "path": "research/reviews/TreeLoom_GPU_history_source_20261017_v1/source_manifest.v1.json",
    "sha256": "0b0013f0907ef1b9a2b7b37dda1e59230111761d19a55821e9c6489adde5c4eb",
}
HISTORY_HOLDS_PER_WINDOW = 128


def verify_implementation() -> dict:
    """Authenticate the exact corrected source closure, without scientific I/O."""
    path = ROOT / MANIFEST_REF["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_REF["sha256"]:
        raise ValueError("TreeLoom corrected implementation manifest changed")
    manifest = json.loads(data)
    for relative, expected in manifest["source_sha256"].items():
        source = ROOT / relative
        if (".." in Path(relative).parts or not source.is_relative_to(ROOT)
                or source.resolve() != source or source.stat().st_size > 32 * 2**20
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
            raise ValueError("TreeLoom implementation source changed: " + relative)
    return manifest


def build_model():
    """Construct the genuine frozen H050 FP32 model on CPU with seed70001."""
    verify_implementation()
    import torch
    frozen = importlib.import_module(MODEL_MODULE)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70001)
        model = frozen.AmericanTreeLoomC3FullV1()
    if (len(list(model.parameters())) != 49
            or sum(parameter.numel() for parameter in model.parameters()) != 32132
            or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                   for parameter in model.parameters())):
        raise ValueError("Original49-tensor32132-parameter FP32 CPU model required")
    return model


def prepare_observation(state, *, resource_check, **kwargs):
    """Delegate complete sanitized observations to the immutable fast preparer.

    The caller supplies the existing original signature's split, references and
    whole-parent limits. Limits reject the full workload rather than crop it.
    """
    if not callable(resource_check):
        raise ValueError("The admitted worker's resource callback is required")
    verify_implementation()
    preparation = importlib.import_module(PREPARATION_MODULE)
    return preparation.prepare_from_loaded_observation_v1(
        state, resource_check=resource_check, **kwargs)


def train_step(model, bundle, targets, optimizer, metadata, *, resource_check, profile=None):
    """Delegate one complete-parent update to the corrected CUDA/PyG executor.

    This is an in-worker numerical API, not a submission or authorization API.
    The exact original model, all10000focals, FP32, clipping and AdamW checks run
    in the implementation. The parent retains fresh GPU0/storage admissions.
    """
    if not callable(resource_check):
        raise ValueError("The admitted worker's resource callback is required")
    verify_implementation()
    resource_check("benchmark_TreeLoom_train_step")
    execution = importlib.import_module(EXECUTION_MODULE)
    return execution.train_step_v1(
        model, bundle, targets, optimizer, metadata, resource_check=resource_check,
        history_holds_per_window=HISTORY_HOLDS_PER_WINDOW, profile=profile)


def predict_logits(model, bundle, metadata, *, resource_check, profile=None):
    """Compute every original focal logit with the same corrected executor.

    The caller owns the sealed observation, selected model and output writer.
    No truth resolver or checkpoint selection is performed here.
    """
    if not callable(resource_check):
        raise ValueError("The admitted worker's resource callback is required")
    verify_implementation()
    import torch
    execution = importlib.import_module(EXECUTION_MODULE)
    model.eval()
    resource_check("benchmark_TreeLoom_prediction")
    with torch.no_grad():
        executor = execution.BoundedExecutionV1(
            model, bundle, metadata, resource_check=resource_check,
            history_holds_per_window=HISTORY_HOLDS_PER_WINDOW, profile=profile)
        return executor.forward()


EUROPEAN_MODEL_REF = {
    "path": "research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py",
    "sha256": "e591760c2bec9e468e8bc636b40e48879686a66e5981db805f46f288e6734073",
}
EUROPEAN_CONDITIONS = {
    "european_c4_prospective1611": "ARGMix_prospective1611",
    "european_c4_expanded": "benchmark_european_c4_expanded",
    "ancestralpaths_paper200mb_native6": "AP_paper_scale_200Mb_native6",
}
_european_implementation = None


def _european_module():
    """Load unchanged European equations with explicit endpoint/length bindings."""
    global _european_implementation
    verify_implementation()
    if _european_implementation is None:
        import importlib.util
        import sys
        source = ROOT / EUROPEAN_MODEL_REF["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != EUROPEAN_MODEL_REF["sha256"]:
            raise ValueError("Frozen European TreeLoom source changed")
        name = "arggnn.benchmarking._bound_european_treeloom"
        spec = importlib.util.spec_from_file_location(name, source)
        implementation = importlib.util.module_from_spec(spec)
        sys.modules[name] = implementation
        spec.loader.exec_module(implementation)
        # No public imported module or historical file is changed. Both C4
        # conditions use the original32325-parameter model and four donor axes.
        implementation.CONDITIONS = dict(
            implementation.CONDITIONS,
            benchmark_european_c4_expanded={
                "haplotypes": 952, "queries": 686, "length_bp": 2000000,
                "output_classes": ("Ana", "CHG", "WHG", "EHG"),
                "parameters": 32325,
            })
        _european_implementation = implementation
    return _european_implementation


def _install_european_bounded_history(memory, *, resource_check, profile):
    """Reuse the exact pinned window VJP and original CPU group pooling.

    The original European model otherwise retains its full-bank tree encoder.
    This bounds history recurrence/VJP storage, not complete-parent GPU memory.
    A fresh European full-parent fit qualification remains mandatory.
    """
    from types import MethodType, SimpleNamespace
    import torch
    history = importlib.import_module(
        "research.reviews.TreeLoom_GPU_history_source_20261017_v1.stable_leaf_history_v1")
    original_history = type(memory)._history
    profile = profile or history.NullProfileV1()

    def bounded(self, forcing, spans, plan):
        offsets = tuple(plan.group_offsets)
        if (forcing.dtype != torch.float32 or forcing.shape[1] != 8
                or plan.haplotypes not in (952, 3222)
                or offsets[0] != 0 or offsets[-1] != len(forcing)
                or len(offsets) != plan.native_holds + 1
                or any(not 0 < b-a <= plan.haplotypes
                       for a, b in zip(offsets, offsets[1:]))):
            raise ValueError("Complete European FP32 cut-group partition required")
        cpu_forcing = forcing.to("cpu")
        cpu_spans = spans.to("cpu")
        cpu_plan = SimpleNamespace(leaf_groups=plan.leaf_groups.to("cpu"),
                                  group_counts=plan.group_counts.to("cpu"))
        # Use the original learned scale property. The coefficient evaluation
        # and pooling move to CPU, as in the qualified-design C3 execution.
        ell = self.ell_bp.to("cpu")
        def direction(values, physical_spans, ignored_plan, reverse):
            rho = torch.exp(-physical_spans / ell)
            source = -torch.expm1(-physical_spans / ell)
            return history._Direction.apply(
                values, rho, source, cpu_plan.leaf_groups, offsets, reverse,
                HISTORY_HOLDS_PER_WINDOW, forcing.device, resource_check, profile)
        proxy = SimpleNamespace(_direction=direction, _mean=self._mean)
        return original_history(proxy, cpu_forcing, cpu_spans, cpu_plan).to(forcing.device)

    memory._history = MethodType(bounded, memory)
    # The imported custom Function performs bounded recomputation itself.
    memory.checkpoint_memory = False


def build_european_model(protocol_id, *, execution_mode="bounded_history",
                         resource_check=None, profile=None):
    """Construct frozen European H050 math, including the native six-path head.

    bounded_history reuses the pinned class-independent recurrence/VJP. It does
    not claim bounded observation/tree storage or measured European GPU fit.
    """
    if protocol_id not in EUROPEAN_CONDITIONS:
        raise ValueError("Explicit supported European protocol required")
    if execution_mode not in ("bounded_history", "reference"):
        raise ValueError("Unknown European execution mode")
    if execution_mode == "bounded_history" and not callable(resource_check):
        raise ValueError("Future admitted worker resource callback required for bounded history")
    implementation = _european_module()
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        model = implementation.FrozenEuropeanTreeLoomV1(
            condition=EUROPEAN_CONDITIONS[protocol_id])
    if execution_mode == "bounded_history":
        _install_european_bounded_history(model.memory, resource_check=resource_check, profile=profile)
    return model


def make_european_bundle(bank, episodes, cut, *, protocol_id, parent_id, split, source_refs):
    """Bind complete observations to the frozen European model API."""
    if protocol_id not in EUROPEAN_CONDITIONS:
        raise ValueError("Explicit supported European protocol required")
    implementation = _european_module()
    memory = implementation.memory_core
    native_split = {"train": "train", "validation": "val_prediction",
                    "test": "evaluation_prediction"}.get(split, split)
    bundle = implementation.FrozenEuropeanBundleV1(
        bank, episodes, cut, EUROPEAN_CONDITIONS[protocol_id], parent_id, native_split,
        dict(source_refs), memory._stamps(implementation._bank_tensors(bank)),
        memory._stamps(implementation._episode_tensors(episodes)))
    bundle.validate()
    return bundle


def european_forward(model, bundle, *, resource_check):
    """Original complete European forward under a future admitted worker."""
    if not callable(resource_check):
        raise ValueError("The admitted worker resource callback is required")
    implementation = _european_module()
    if (type(model) is not implementation.FrozenEuropeanTreeLoomV1
            or type(bundle) is not implementation.FrozenEuropeanBundleV1):
        raise ValueError("The original bound European model and bundle are required")
    bundle.validate()
    resource_check("benchmark_European_TreeLoom_forward")
    return model(bundle.bank, episodes=bundle)


def train_european_step(model, bundle, target_mass, optimizer, *, resource_check):
    """Retain the historical exact-span objective for its declared protocols."""
    from arggnn.biological_gst_control_v1 import exact_span_loss
    return _train_european_update(model, bundle, target_mass, optimizer,
                                  resource_check=resource_check, loss_function=exact_span_loss)


def train_european_sampled_step(model, bundle, targets, optimizer, *, resource_check, sampled_registry_ref):
    """Paper-style sampled-example objective on the same frozen TreeLoom model."""
    from .targets import SampledFocalTargets, sampled_focal_loss
    if (not isinstance(targets, SampledFocalTargets)
            or tuple(model.output_classes) != targets.class_order):
        raise ValueError("Training label classes must match the native task output head")
    targets.validate_binding(parent_id=bundle.parent_id, split=bundle.split,
                             sampled_registry_ref=sampled_registry_ref)
    return _train_european_update(model, bundle, targets, optimizer,
                                  resource_check=resource_check, loss_function=sampled_focal_loss)


def _train_european_update(model, bundle, targets, optimizer, *, resource_check, loss_function):
    """One whole-parent update with its explicitly selected supervision objective.

    The caller separately joins TRAIN targets and owns the finite protocol,
    parent order, checkpoint barrier and actual GPU0 admission. No labels enter
    make_european_bundle. This function performs no file or checkpoint reads.
    """
    import torch
    if bundle.split != "train" or type(optimizer) is not torch.optim.AdamW:
        raise ValueError("Genuine TRAIN parent and native AdamW required")
    params = list(model.parameters())
    if len(optimizer.param_groups) != 1:
        raise ValueError("Single original AdamW parameter group required")
    group = optimizer.param_groups[0]
    if (group["lr"] != .001 or group["weight_decay"] != .0001
            or group["betas"] != (.9, .999) or group["eps"] != 1e-8
            or {id(p) for p in group["params"]} != {id(p) for p in params}
            or any(p.dtype != torch.float32 for p in params)):
        raise ValueError("Original H050 FP32 AdamW settings required")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=next(model.parameters()).device.type, enabled=False):
        logits = european_forward(model, bundle, resource_check=resource_check)
        loss = loss_function(logits, targets)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Finite whole-parent training loss required")
    resource_check("benchmark_European_TreeLoom_backward")
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError("All original European parameters need connected finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    resource_check("benchmark_European_TreeLoom_optimizer")
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm": float(norm),
            "one_complete_parent_update": True, "ancestry_accuracy": None}
