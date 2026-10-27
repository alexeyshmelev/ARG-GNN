"""Exact short-plan delegation to the immutable length-aware generator."""
from __future__ import annotations
import json
from pathlib import Path
from .short_plan_v1 import validate_recipe, validate_parent, plan_ref


def generate_parent(recipe, parent, output, *, guard, execution_check, write_headroom_bytes):
    from .aggregate_guard_v1 import AggregateOwner
    if type(guard) is not AggregateOwner or execution_check is not guard.execution_check:
        raise ValueError('Same aggregate owner and admitted execution callback required')
    output = guard.path(output)
    validate_recipe(recipe)
    validate_parent(parent)
    if not callable(guard) or not callable(execution_check):
        raise ValueError('Aggregate owner guard and current execution check required')
    # The immutable helper requires exactly these two simulation seed keys.
    simulation_parent = dict(parent_id=parent['parent_id'], split=parent['split'],
                             seeds={key: parent['seeds'][key] for key in ('ancestry', 'mutations')})
    from research.efficiency_20261027.architecture.european_cohort_candidate.validation_generation_v2 import generate_validation_parent
    receipt = generate_validation_parent(recipe, simulation_parent, output,
        guard=guard, execution_check=execution_check, write_headroom_bytes=write_headroom_bytes,
        require_paper_match=False)
    combined = dict(schema='AP_short_development_generation_v1', short_development_plan_ref=plan_ref(),
        parent_id=parent['parent_id'], split=parent['split'], seeds=dict(parent['seeds']),
        sequence_length_bp=10_000_000, development_model_selection_only=True,
        paper_reproduction_claimed=False, independent_replication_per_linked_row=False,
        generation=receipt)
    destination = Path(output) / 'short_generation.json'
    raw = (json.dumps(combined, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    guard(len(raw) + 4096)
    with destination.open('xb') as stream:
        stream.write(raw)
    guard(0)
    return combined
