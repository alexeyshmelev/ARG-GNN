"""CPU-only v5 integration:exact math body,route reporting and release binding."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch
import torch
from arggnn.benchmarking import contracts as c
from . import qualification_combined_v5 as gate
from .configuration_combined_v5 import configuration,EXPECTED,FACTORY,GIB,PINS
from research.efficiency_20261027.architecture.qualified_residency_router import router_v1 as router

HERE=Path(__file__).parent

def function(path,name):
    return next(node for node in ast.parse(Path(path).read_text()).body if isinstance(node,ast.FunctionDef) and node.name==name)


def fails(call):
    try:call()
    except (ValueError,KeyError,TypeError):return
    raise AssertionError('Expected router gate rejection')


def main():
    assert not torch.cuda.is_initialized()
    expected=configuration(dict(EXPECTED))
    assert expected['resident_soft_bytes']==74*GIB and expected['resident_scratch_bytes']==16*GIB
    assert expected['executor_factory']==router.__name__+':build_executor'==FACTORY
    assert expected['qualified_fallback_factory']==router.FACTORIES[router.EXPLICIT]
    assert expected['capacity_selection']=='actual_prepared_nodes_and_live_capacity_before_forward'
    original=function(HERE/'step_combined_v1.py','train_step');candidate=function(HERE/'step_combined_v2.py','train_step')
    factory_calls=0;receipt_fields=0
    for node in ast.walk(candidate):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='build_executor':
            extra={kw.arg:kw.value for kw in node.keywords if kw.arg in ('resident_soft_bytes','resident_scratch_bytes')}
            assert set(extra)=={'resident_soft_bytes','resident_scratch_bytes'}
            for name,value in extra.items():
                assert isinstance(value,ast.Subscript) and isinstance(value.value,ast.Name) and value.value.id=='EXPECTED'
                assert isinstance(value.slice,ast.Constant) and value.slice.value==name
            node.keywords=[kw for kw in node.keywords if kw.arg not in extra];factory_calls+=1
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
            fields=[kw for kw in node.keywords if kw.arg=='capacity_route_receipt']
            for field in fields:
                assert ast.unparse(field.value)=='dict(executor.capacity_route_receipt)'
                node.keywords.remove(field);receipt_fields+=1
    assert factory_calls==receipt_fields==1 and ast.dump(original)==ast.dump(candidate)
    for name in ('run_training','train_loop','build_spec','validate_saved_state','validate_resume_checkpoint',
                 'validate_resume_metadata','schedule_identity','random_state','restore_random_state'):
        assert ast.dump(function(HERE/'training_combined_v4.py',name))==ast.dump(function(HERE/'training_combined_v5.py',name)),name
    old=(HERE/'runtime_combined_v1.py').read_text().replace('from .step_combined_v1 import','from .step_combined_v2 import')
    assert (HERE/'runtime_combined_v2.py').read_text()==old
    nodes=205369272;holds=77472;allocated=GIB;required=2*nodes*128+16*GIB
    common=dict(native_nodes=nodes,native_holds=holds,allocated_bytes=allocated,reserved_bytes=allocated,
        total_bytes=100*GIB,free_bytes=90*GIB,resident_soft_bytes=allocated+required)
    assert router.select_capacity(**common)['selected_route']==router.RESIDENT
    assert router.select_capacity(**dict(common,resident_soft_bytes=common['resident_soft_bytes']-1))['selected_route']==router.EXPLICIT
    assert router.select_capacity(**dict(common,free_bytes=required-1))['selected_route']==router.EXPLICIT
    for key,value in [('resident_soft_bytes',75*GIB),('resident_scratch_bytes',15*GIB),('no_mid_update_fallback',False),
        ('qualified_fallback_factory','unreviewed'),('capacity_selection','after_OOM')]:
        fails(lambda key=key,value=value:configuration(dict(expected,**{key:value})))
    evidence=dict(source_manifest_ref=c.file_ref(gate.ROUTER_MANIFEST),CPU_contracts_ref=c.file_ref(gate.ROUTER_CPU),
        independent_source_review_ref=dict(path='synthetic_review_only',sha256='a'*64))
    original_read=gate.read;review=dict(passed=True,source_manifest_ref=evidence['source_manifest_ref'])
    def read(ref):return review if ref==evidence['independent_source_review_ref'] else original_read(ref)
    with patch.object(gate,'read',side_effect=read):
        assert gate.validate_router(evidence)==evidence
        review['passed']=False;fails(lambda:gate.validate_router(evidence));review['passed']=True
        review['source_manifest_ref']={};fails(lambda:gate.validate_router(evidence));review['source_manifest_ref']=evidence['source_manifest_ref']
        altered=copy.deepcopy(evidence);altered['CPU_contracts_ref']['sha256']='0'*64;fails(lambda:gate.validate_router(altered))
        altered=copy.deepcopy(evidence);altered['source_manifest_ref']['sha256']='0'*64;fails(lambda:gate.validate_router(altered))
    result=dict(schema='TreeLoom_combined_v5_router_integration_CPU_v1',passed=True,candidate_source_pins_ref=c.file_ref(PINS),
        numerical_train_step_AST_equal_after_exact_two_capacity_keywords_and_one_receipt_field=True,
        complete_run_training_build_spec_schedule_checkpoint_RNG_functions_AST_equal=True,
        runtime_identical_except_step_import=True,exact_factory_and74GiB16GiB_arguments=True,
        exact_capacity_boundary_and_one_byte_excess_routes_verified=True,
        no_OOM_retry_or_new_math_or_context_policy=True,actual_per_parent_route_receipt_required=True,
        router_evidence_and_independent_review_tamper_rejected=True,
        router_CPU_evidence_ref=c.file_ref(gate.ROUTER_CPU),CPU_only=True,CUDA_initialized=False,
        scientific_arrays_opened=False,jobs_started=0,production_modified=False)
    path=HERE/'ROUTING_COMBINED_CPU_v5.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,evidence=c.file_ref(path))))

if __name__=='__main__':main()
