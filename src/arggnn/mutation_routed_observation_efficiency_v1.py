"""Isolated equivalent-observation efficiency helper for H036 continuation.

Frozen sources are imported and never edited or monkeypatched. Only the
null_constant_fields assignment in the frozen sham helper is replaced. All
other function code, RNG, scientific arrays and observation metadata survive.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
from types import FunctionType

import numpy as np

from . import mutation_routed_tree_observations_v1 as original_preprocessing
from . import mutation_routed_parent_loader_v1 as original_loader

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/mutation_routed_observation_efficiency_v1.py"

_EXPECTED_CONSTANT_ASSIGNMENT = """constant_fields = [SUPPORT_FIELDS[f] for f in range(5)
    if all(len(np.unique(value[(hold == h) & branches, f])) <= 1 for h in range(len(span)))]"""


def null_constant_fields_v1(parent, support, node_hold):
    """Exact per-hold finite-value constancy, with contiguous slices.

    Original validation already guarantees finite support and valid holds.
    Missing holds and holds containing zero/one branches are constant, matching
    the original unique-count test. Root rows are excluded. Field order stays
    unchanged. Noncontiguous holds use a stable local sort without RNG access.
    """
    value, p, hold = np.asarray(support), np.asarray(parent), np.asarray(node_hold)
    if len(hold) == 0:
        return list(original_preprocessing.SUPPORT_FIELDS)
    if np.any(hold[1:] < hold[:-1]):
        order = np.argsort(hold, kind="stable")
        value, p, hold = value[order], p[order], hold[order]
    starts = np.r_[0, np.flatnonzero(hold[1:] != hold[:-1]) + 1]
    ends = np.r_[starts[1:], len(hold)]
    constant = np.ones(5, dtype=bool)
    for lo, hi in zip(starts, ends):
        rows = value[lo:hi][p[lo:hi] >= 0]
        if len(rows):
            constant &= np.all(rows == rows[0], axis=0)
    return [name for name, active in zip(original_preprocessing.SUPPORT_FIELDS, constant) if active]


def _clone_with_globals_v1(function, **overrides):
    """Independent function namespace, retaining original code and defaults."""
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__doc__ = function.__doc__
    return clone


def _efficient_shuffle_v1():
    """Replace exactly one recognized frozen assignment; retain the rest AST."""
    original = original_preprocessing.shuffle_branch_support_v1
    module = ast.parse(inspect.getsource(original))
    function = module.body[0]
    expected = ast.parse(_EXPECTED_CONSTANT_ASSIGNMENT).body[0]
    replacements = 0
    for index, node in enumerate(function.body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "constant_fields"):
            if ast.dump(node, include_attributes=False) != ast.dump(expected, include_attributes=False):
                raise ValueError("Frozen constant-field expression changed; isolated optimization must be reviewed")
            replacement = ast.parse("constant_fields = null_constant_fields_v1(p, value, hold)").body[0]
            function.body[index] = ast.copy_location(replacement, node)
            replacements += 1
    if replacements != 1:
        raise ValueError("Exactly the one frozen constant-field assignment must be replaced")
    ast.fix_missing_locations(module)
    namespace = dict(original.__globals__, null_constant_fields_v1=null_constant_fields_v1)
    exec(compile(module, original.__code__.co_filename + ":null_constant_fields_efficiency_v1", "exec"), namespace)
    return namespace[original.__name__]


shuffle_branch_support_v1 = _efficient_shuffle_v1()
prepare_mutation_routed_tree_observations_v1 = _clone_with_globals_v1(
    original_preprocessing.prepare_mutation_routed_tree_observations_v1,
    shuffle_branch_support_v1=shuffle_branch_support_v1)
_prepare_parent_collection_v1 = _clone_with_globals_v1(
    original_loader.prepare_parents_v1,
    prepare_mutation_routed_tree_observations_v1=prepare_mutation_routed_tree_observations_v1)


def source_provenance_v1():
    return dict(schema="mutation_routed_observation_efficiency_v1", change="null_constant_fields_scan_only",
        preparation_equivalence="same_scientific_arrays_sham_RNG_and_original_metadata",
        original_source_modules_mutated=False, source_files_edited=False,
        constant_scan="contiguous_per_hold_slices;stable_sort_fallback_for_noncontiguous_holds",
        constant_scan_complexity="O(N*5)_for_ordered_holds;O(NlogN+N*5)_fallback",
        new_observation_cache=False, source_refs={name: {
            "path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
            for name, path in (("efficiency_helper", SOURCE),
                ("frozen_preprocessing", "src/arggnn/mutation_routed_tree_observations_v1.py"),
                ("frozen_parent_loader", "src/arggnn/mutation_routed_parent_loader_v1.py"))})


def prepare_parents_v1(parent_refs, reader, *, scratch_directory=None):
    """Same observation bundle and scratch contract, with efficient constancy.

    The root must supply output.path / observation_tmp for worker isolation.
    Original observations_receipt and collection receipt are unchanged. A
    separate top-level efficiency_provenance records the implementation route.
    No ancestry target or actual-array authorization is added by this wrapper.
    """
    bundle = _prepare_parent_collection_v1(parent_refs, reader, scratch_directory=scratch_directory)
    bundle["efficiency_provenance"] = source_provenance_v1()
    return bundle
