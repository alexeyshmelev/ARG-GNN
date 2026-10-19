"""Private exact integer preparation for AP native-six and its separate C4 fit."""
from __future__ import annotations
import ast
import copy
import importlib
import types
from types import SimpleNamespace
from arggnn.benchmarking import european_training as ap
from arggnn.benchmarking import european_c4_training as c4
from arggnn.benchmarking.execution import GPU0_UUID
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import candidate_v1 as candidate
from research.reviews.benchmark_repairs_20261019.treeloom.cut_hoist_v1 import prepare_cut

AP = "ancestralpaths_paper200mb_native6"
C4 = "ancestralpaths_paper200mb_c4"
PROTOCOLS = (AP, C4)
_BOUND = {}


def clone(function, **bindings):
    result = types.FunctionType(function.__code__, dict(function.__globals__, **bindings),
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


def prepared_function(function):
    observations = importlib.import_module(
        "research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_observations_v1")
    observed = SimpleNamespace(**vars(observations))
    observed.encode_block = candidate.encode_block
    def transform(node):
        changed = {"observations": 0, "cut_preparation": 0}
        for at, item in enumerate(node.body):
            if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                name = item.targets[0].id
                if name in changed and isinstance(item.value, ast.Call) and \
                        ast.unparse(item.value.func) == "importlib.import_module":
                    item.value = ast.Name(id="_optimized_" + name, ctx=ast.Load())
                    changed[name] += 1
        candidate.need(set(changed.values()) == {1}, "Exact private European observation/cut imports required")
        starts = [at for at, item in enumerate(node.body) if isinstance(item, ast.Assign)
                  and any(isinstance(target, ast.Tuple) and [getattr(e, "id", None) for e in target.elts]
                          == ["arrays", "episode_metadata"] for target in item.targets)]
        ends = [at for at, item in enumerate(node.body) if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "episodes" for target in item.targets)]
        candidate.need(len(starts) == len(ends) == 1 and ends[0] == starts[0] + 1,
                       "Exact original European episode build/load pair required")
        node.body[starts[0]:ends[0] + 1] = ast.parse(
            "episodes,episode_metadata=_optimized_episodes(bank,max_payload_bytes=max_episode_payload_bytes,"
            "isolated_hold_indices=isolated)").body
    return candidate.compile_function(function, transform, _optimized_observations=observed,
        _optimized_cut_preparation=SimpleNamespace(prepare_cut_plan_v1=prepare_cut),
        _optimized_episodes=candidate.build_episodes)



def _cpu_setup_function(function):
    def transform(node):
        first = node.body[1] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else node.body[0]
        candidate.need(isinstance(first, ast.Expr) and "Current supervised physicalGPU0" in ast.unparse(first),
                       "Exact original GPU entry guard required")
        at = node.body.index(first)
        node.body[at:at+1] = ast.parse(
            "_need(callable(resource_check) and os.environ.get('CUDA_VISIBLE_DEVICES')==_GPU0_UUID "
            "and os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK')=='1' "
            "and 'ARGGNN_GPU_LEASE_FD' not in os.environ and 'ARGGNN_PHYSICAL_GPU' not in os.environ,"
            "'Deferred preparation needs fixed GPU0 visibility, NVML availability checks and no acquired lease')").body
        removed = []
        for item in tuple(node.body):
            text = ast.unparse(item)
            if ("torch.cuda." in text or "allow_tf32" in text or
                (isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "cap"
                                                    for target in item.targets)) or
                (isinstance(item, ast.Expr) and "Finite current GPU0 allocator cap required" in text)):
                node.body.remove(item);removed.append(text)
        candidate.need(len(removed) == 7, "Exact original CUDA-only setup statements required")
        imports = [at for at, item in enumerate(node.body) if isinstance(item, ast.Import)
                   and any(alias.name == "torch" for alias in item.names)]
        candidate.need(len(imports) == 1, "Exact original Torch import required")
        node.body[imports[0]+1:imports[0]+1] = ast.parse(
            "_need(not torch.cuda.is_initialized(),'Deferred preparation cannot inherit initialized CUDA')").body
    return candidate.compile_function(function, transform, _GPU0_UUID=GPU0_UUID)


def finish_gpu_setup(torch):
    import os
    candidate.need(os.environ.get("ARGGNN_PHYSICAL_GPU") == "0", "Admitted physicalGPU0 handoff required")
    candidate.need(torch.cuda.device_count() == 1, "One admitted visible GPU0 required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = int(os.environ.get("ARGGNN_MAX_CUDA_RESERVED_BYTES", "0"))
    total = torch.cuda.get_device_properties(0).total_memory
    candidate.need(0 < cap <= min(74*2**30, total), "Finite current GPU0 allocator cap required")
    torch.cuda.set_per_process_memory_fraction(cap/total, 0)


def _fit_function(function, *, loader, gpu):
    def transform(node):
        node.args.kwonlyargs.append(ast.arg(arg="acquire_gpu0"))
        node.args.kw_defaults.append(None)
        anchors = [at for at, item in enumerate(node.body) if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "barrier_ref"
                           for target in item.targets)]
        candidate.need(len(anchors) == 1, "Exact original complete observed barrier write required")
        node.body[anchors[0]+1:anchors[0]+1] = ast.parse(
            "_need(callable(acquire_gpu0),'Current supervised GPU0 acquisition callback required')\n"
            "acquire_gpu0()\n_finish_gpu_setup(torch)").body
    return candidate.compile_function(function, transform, _observed_loader=loader,
                                      gpu=gpu, _finish_gpu_setup=finish_gpu_setup)


def backend(protocol_id):
    candidate.need(protocol_id in PROTOCOLS, "Only exact AP native-six and separate C4 protocols required")
    if protocol_id not in _BOUND:
        # C4's verified private namespace retains its own registry class view,
        # parent-plan validator and TRAIN target loader; never substitute AP6 I/O.
        core = ap if protocol_id == AP else c4._bound()
        view = SimpleNamespace(**vars(core.european))
        view.prepare_observation = prepared_function(core.european.prepare_observation)
        loader = clone(core._observed_loader, european=view)
        gpu = SimpleNamespace(**vars(core.gpu))
        gpu._setup_gpu = _cpu_setup_function(core.gpu._setup_gpu)
        _BOUND[protocol_id] = SimpleNamespace(core=core, european=view,
            parent_plan=core._parent_plan, build_fit_spec=core.build_fit_spec,
            build_prediction_spec=core.build_prediction_spec,
            run_fit=_fit_function(core.run_fit, loader=loader, gpu=gpu),
            run_predictions=clone(core.run_predictions, _observed_loader=loader))
    return _BOUND[protocol_id]
