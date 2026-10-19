"""Lossless integer-only root lookup hoist; frozen preparation stays immutable."""
from __future__ import annotations
import ast
import copy
import types
from research.reviews.workflow_progress_diagnosis_20261018.treeloom import candidate_v1 as frozen


def _hoist(node, *, instrument=False):
    expected = ast.parse(
        "ordered=[classify_sample_roots(p,head,roots[owner],row) for row in samples]").body[0]
    matches = [(at, item) for at, item in enumerate(node.body)
               if ast.dump(item, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    frozen.need(len(matches) == 1, "Exactly one original full-root gather inside hold comprehension required")
    frozen.need(not any(isinstance(item, ast.Name) and item.id == "_hoisted_root_of_node"
                        for item in ast.walk(node)), "Fresh local root lookup name required")
    at, _ = matches[0]
    gather = "_root_gather(roots,owner)" if instrument else "roots[owner]"
    node.body[at:at + 1] = ast.parse(
        "_hoisted_root_of_node=" + gather + "\n"
        "ordered=[classify_sample_roots(p,head,_hoisted_root_of_node,row) for row in samples]\n"
        "del _hoisted_root_of_node").body


def make_prepare_cut(*, root_gather=None):
    """Private original candidate with exactly one extra hoist; hook is CPU-test-only."""
    def compile_hoisted(function, transform=None, **bindings):
        def combined(node):
            if transform is not None:
                transform(node)
            _hoist(node, instrument=root_gather is not None)
        if root_gather is not None:
            bindings["_root_gather"] = root_gather
        return frozen.compile_function(function, combined, **bindings)
    original = frozen.prepare_cut
    call = types.FunctionType(original.__code__,
        dict(original.__globals__, compile_function=compile_hoisted),
        original.__name__, original.__defaults__, original.__closure__)
    call.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    return call


prepare_cut = make_prepare_cut()
