"""Versioned metadata-only scalar normalization for observed prefix preparation.

The frozen v1 preparer still produces every numerical object. Only the returned
scope receipt receives a new recursively copied tree with native JSON scalars.
NumPy arrays and tensors are deliberately not converted or modified.
"""
import numpy as np
from research.efficiency_20261027.execution.prepare_large_slice import (
    prepare_large_prefix as _prepare_large_prefix)


def plain_scope_metadata(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: plain_scope_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain_scope_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(plain_scope_metadata(item) for item in value)
    return value


def prepare_large_prefix(args, rt, saved, check):
    bundle, metadata, labels, frozen, scope = _prepare_large_prefix(args, rt, saved, check)
    return bundle, metadata, labels, frozen, plain_scope_metadata(scope)
