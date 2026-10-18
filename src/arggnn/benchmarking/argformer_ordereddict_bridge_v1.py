"""One checkpoint-container compatibility correction over the immutable bridge.

Only the bridge guard changes. The loaded Composer object and tensor references
are returned unchanged to the original author initializer. No Torch import,
checkpoint load, forward, optimizer step or source-file mutation occurs here.
"""
from __future__ import annotations
import ast
from collections import OrderedDict
import hashlib
from pathlib import Path
from types import ModuleType

ORIGINAL_PATH = "research/reviews/native_VAL000_saved_MLM_acceptance_bridge_source_20261017_v1/saved_mlm_bridge_v1.py"
ORIGINAL_SHA256 = "5810041797592fa414d57747607f646a3eee1050d0373df5412b7dd4b1b63d97"


def supported_model_state(value, torch):
    """Permit exactly native dict/OrderedDict maps of named tensors."""
    return (type(value) in (dict, OrderedDict) and bool(value)
            and all(isinstance(key, str) and torch.is_tensor(item)
                    for key, item in value.items()))


def compatible_bridge_v1(original):
    """Return a private bridge namespace with one authenticated AST replacement."""
    path = Path(original.__file__)
    if (not path.as_posix().endswith(ORIGINAL_PATH)
            or hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA256):
        raise ValueError("Exact immutable original initialization bridge required")
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "_instrument_training"]
    if len(functions) != 1:
        raise ValueError("One original instrumented initializer required")
    function = functions[0]
    anchor = ast.parse('type(value["state"].get("model")) is dict', mode="eval").body
    replacement = ast.parse('_supported_model_state(value["state"].get("model"), torch)',
                            mode="eval").body
    count = [0]
    class CorrectModelContainer(ast.NodeTransformer):
        def visit_Compare(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(anchor, include_attributes=False):
                count[0] += 1
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)
    function = CorrectModelContainer().visit(function)
    if count != [1]:
        raise ValueError("Exactly one original model-container type check must change")
    private = ModuleType("_benchmark_ordereddict_compatible_saved_mlm_bridge_v1")
    private.__dict__.update(original.__dict__)
    private.__dict__["_supported_model_state"] = supported_model_state
    compiled = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(compiled)
    exec(compile(compiled, str(Path(__file__).resolve())+":dict_or_OrderedDict_only", "exec"),
         private.__dict__)
    return private
