"""Harmless actual python -m callback-contract probe; no scientific execution."""
from pathlib import Path
import json
from arggnn.benchmarking.paper_campaign import runtime as canonical
from .restore_entrypoint_v3 import validate_resource_callback
from .entrypoint_v3 import need


def rejects(function):
    try:function()
    except ValueError:return
    raise AssertionError('Expected invalid callback or scope rejection')


def run_task(request,*,resource_check):
    need(type(request) is dict and set(request)=={'expected_task_contract'},'Exact bounded CPU probe request required')
    prior=request['expected_task_contract']
    task=validate_resource_callback(resource_check,prior)
    need(type(resource_check).__module__=='__main__','Actual python -m runtime identity required')
    need(not isinstance(resource_check,canonical.ResourceCheck),'Fixture must reproduce the previous canonical-class mismatch')
    rejects(lambda:validate_resource_callback(lambda stage:None,prior))
    for key in ('max_rss_bytes','max_output_bytes','max_seconds','minimum_free_gpu_bytes'):
        wrong=dict(prior)
        wrong[key]+=1
        rejects(lambda:validate_resource_callback(resource_check,wrong))
    rejects(lambda:validate_resource_callback(resource_check,dict(prior,output_directory='wrong/scope')))
    resource_check(4096)
    import torch
    need(not torch.cuda.is_initialized(),'CPU-only callback probe must not initialize CUDA')
    return dict(schema='TreeLoom_actual_module_callback_contract_v3',status='pass',
        callback_class_module=type(resource_check).__module__,
        callback_class_name=type(resource_check).__name__,
        canonical_isinstance=False,actual_module_identity_gap_reproduced=True,
        callable_and_exact_admitted_scope_contract_passed=True,invalid_scope_and_callback_rejected=True,
        CUDA_initialized=False,scientific_executor_called=False,history_restored=False,
        production_modified=False)
