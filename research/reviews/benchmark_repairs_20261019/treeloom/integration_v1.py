"""Private qualification/fit integration for the exact integer root hoist."""
from __future__ import annotations
import ast
import copy
import types
from types import SimpleNamespace
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import (
    candidate_v1 as compiler, prepare_integration_v2 as frozen,
    qualify_parent_v2 as frozen_qualifier)
from .cut_hoist_v1 import prepare_cut


def _clone(function, **bindings):
    call = types.FunctionType(function.__code__, dict(function.__globals__, **bindings),
                              function.__name__, function.__defaults__, function.__closure__)
    call.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return call


def install_runtime_preparation(runtime):
    # Start from the actual fresh private class/registry namespace, as in v2.
    compiled = frozen.compiled_complete(runtime.preparation.prepare_from_loaded_observation_v1)
    compiled = _clone(compiled, prepare_cut_plan_v1=prepare_cut)
    runtime.preparation = SimpleNamespace(prepare_from_loaded_observation_v1=compiled)
    return runtime


run_training = _clone(frozen.run_training, install_runtime_preparation=install_runtime_preparation)


def _qualification_transform(node):
    matches = [at for at, item in enumerate(node.body)
               if isinstance(item, ast.ImportFrom) and item.level == 1
               and item.module == "prepare_integration_v2"
               and [(alias.name, alias.asname) for alias in item.names]
                   == [("install_runtime_preparation", None)]]
    compiler.need(len(matches) == 1, "One frozen qualification preparation import required")
    del node.body[matches[0]]


qualify = compiler.compile_function(frozen_qualifier.qualify, _qualification_transform,
                                   install_runtime_preparation=install_runtime_preparation)
