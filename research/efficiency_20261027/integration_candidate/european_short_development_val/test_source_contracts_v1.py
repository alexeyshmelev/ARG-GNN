"""Metadata, AST and tiny synthetic filesystem checks; no native/model execution."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
from . import short_plan_v1 as short
from .sampling_v1 import build_validation_registry
from . import aggregate_guard_v1 as limits
from .native_supervisor_v1 import owned_family, NativeSupervisor, LOG_LIMIT, DRAIN_LIMIT
from . import execution_stages_v1 as stages

BASE = Path(__file__).absolute().parent


def fails(call):
    try:
        call()
    except (ValueError, TypeError, KeyError, limits.GuardFailure, FileExistsError, BlockingIOError):
        return
    raise AssertionError('Required rejection did not occur')


def nodes(path):
    return ast.parse(Path(path).read_text())


def function(path, name, class_name=None):
    tree=nodes(path)
    body=tree.body if class_name is None else next(n for n in tree.body if getattr(n,'name',None)==class_name).body
    return copy.deepcopy(next(n for n in body if getattr(n,'name',None)==name))


def dump(node):
    return ast.dump(node, include_attributes=False)


def source_checks():
    root=short.ROOT
    source=root/'src/arggnn/benchmarking/european_execution.py'
    for name in ('prepare_relate_inputs','clean_converted_tree'):
        old=function(source,name);new=function(BASE/'execution_stages_v1.py',name)
        # The sole added statement confines the destination to the aggregate.
        assert isinstance(new.body[1],ast.Assign) and ast.unparse(new.body[1])=='destination = guard.path(destination)'
        del new.body[1]
        assert dump(old)==dump(new),name
    native_old=function(root/'src/arggnn/benchmarking/european_native_relate.py','infer_parent')
    native_new=function(BASE/'native_relate_v1.py','infer_parent')
    def run_calls(node):
        return [dump(n) for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run']
    assert run_calls(native_old)==run_calls(native_new)
    assert len(run_calls(native_new))==5
    assert any(isinstance(n,ast.Constant) and n.value==10_000_000 for n in ast.walk(native_new))
    generation=nodes(BASE/'generation_v1.py')
    calls=[n for n in ast.walk(generation) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='generate_validation_parent']
    assert len(calls)==1 and next(k.value.value for k in calls[0].keywords if k.arg=='require_paper_match') is False
    assert "('ancestry', 'mutations')" in (BASE/'generation_v1.py').read_text()
    sampler_old=function(root/'research/efficiency_20261027/architecture/european_cohort_candidate/validation_sampling.py','build_validation_registry')
    sampler_new=function(BASE/'sampling_v1.py','build_validation_registry')
    old_loop=next(n for n in sampler_old.body if isinstance(n,ast.For) and ast.unparse(n.target)=='cohort')
    new_loop=next(n for n in sampler_new.body if isinstance(n,ast.For) and ast.unparse(n.target)=='cohort')
    assert dump(old_loop)==dump(new_loop)
    frozen=root/'research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py'
    new_frozen=BASE/'model_binding_v1.py'
    for name in ('forward_holds','forward_train_holds','forward'):
        assert dump(function(frozen,name,'FrozenEuropeanTreeLoomV1'))==dump(function(new_frozen,name,'ShortDevelopmentTreeLoomV1'))
    old=function(frozen,'encode_trees','FrozenEuropeanTreeLoomV1')
    new=function(new_frozen,'encode_trees','ShortDevelopmentTreeLoomV1')
    for node in ast.walk(new):
        if isinstance(node,ast.Name) and node.id=='ShortDevelopmentBundleV1':node.id='FrozenEuropeanBundleV1'
    assert dump(old)==dump(new)
    cohort=root/'research/efficiency_20261027/architecture/european_cohort_candidate'
    for name in ('__init__','base_from_sites','prepare_tree_base_v1'):
        assert dump(function(cohort/'model.py',name,'EuropeanCohortTreeLoom'))==dump(function(BASE/'cohort_model_binding_v1.py',name,'ShortDevelopmentCohortTreeLoom'))
    assert dump(function(cohort/'observations.py','cohort_tensor'))==dump(function(BASE/'cohort_observations_v1.py','cohort_tensor'))
    old=function(cohort/'execution.py','predict_validation')
    new=function(BASE/'consumer_execution_binding_v1.py','predict_validation')
    for node in ast.walk(new):
        if isinstance(node,ast.Name) and node.id=='ShortDevelopmentCohortStagedExecution':node.id='CohortStagedExecution'
    assert dump(old)==dump(new)
    for path in BASE.glob('*.py'):
        ast.parse(path.read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'exec','eval','compile'} for n in ast.walk(nodes(path)))
    return dict(map_filter_cleaner_AST_equal_except_output_confinement=True,
        five_native_argv_AST_identical=True,sampling_inner_algorithm_AST_identical=True,
        model_equations_cohort_tensor_and_prediction_AST_identical=True)


def metadata_checks():
    loaded=short.validate_plan();recipe=loaded['recipe'];parent=short.frozen_parent()
    short.validate_recipe(recipe);short.validate_parent(parent)
    for key in recipe['parameters']:
        changed=copy.deepcopy(recipe);changed['parameters'][key]=None
        fails(lambda:short.validate_recipe(changed))
    changed=copy.deepcopy(parent);changed['seeds']['ancestry']+=1
    fails(lambda:short.validate_parent(changed))
    rows=[]
    for cohort in recipe['parameters']['cohorts']:
        for individual in range(cohort['diploids']):
            for hap in (0,1):
                index=len(rows)
                rows.append(dict(sample_index=index,observed_node_id=index,sample_id=cohort['id']+str(individual),
                    haplotype=hap,observed_cohort=cohort['id'],age_generations=0 if cohort['id']=='present_day' else 100,
                    role='query' if cohort['role']=='query' else 'pure_source_reference',
                    observed_reference_class=None if cohort['role']=='query' else {'Ana':0,'CHG':1,'WHG':2,'EHG':3}[cohort['id']]))
    assert len(rows)==952
    sites=[dict(native_site_id=i,VCF_position_1based=101+1000*i,
                local_position_0based=1000*i,original_position_0based=100+1000*i) for i in range(256)]
    inference=dict(coverage_original_bp0=[100,255101],local_to_original_offset_bp=100,site_join=sites,
        sample_index_to_inferred_tree_node_id=[dict(sample_index=i,inferred_tree_node_id=i) for i in range(952)])
    registry=build_validation_registry(loaded['sampling_plan'],parent['parent_id'],rows,inference,
        observation_ref={'path':'synthetic/inference.json','sha256':'a'*64})
    assert registry['sequence_length_bp']==10_000_000 and registry['examples']==10000
    assert registry['short_development_plan_ref']==short.plan_ref()
    counts={cohort:sum(r['cohort']==cohort for r in registry['records']) for cohort in loaded['sampling_plan']['parents'][parent['parent_id']]['cohort_quotas']}
    assert set(counts.values())=={2000}
    assert build_validation_registry(loaded['sampling_plan'],parent['parent_id'],rows,inference,
        observation_ref={'path':'synthetic/inference.json','sha256':'a'*64})==registry
    sparse=dict(inference,site_join=sites[:1])
    fails(lambda:build_validation_registry(loaded['sampling_plan'],parent['parent_id'],rows,sparse,
        observation_ref={'path':'synthetic/inference.json','sha256':'a'*64}))
    invalid=dict(inference,coverage_original_bp0=[100,10_000_001])
    fails(lambda:build_validation_registry(loaded['sampling_plan'],parent['parent_id'],rows,invalid,
        observation_ref={'path':'synthetic/inference.json','sha256':'a'*64}))
    from arggnn.benchmarking.european_sampling import registry_arrays
    arrays=registry_arrays(registry)
    assert len(arrays['record_ids'])==10000 and arrays['positions_bp0'].max()<short.LENGTH
    return loaded,rows,dict(exact_recipe_and_seed_proof=True,reject_changed_parameter_count=len(recipe['parameters']),
        synthetic_samples=952,synthetic_examples=10000,per_cohort=counts,insufficient_sites_fail_no_redraw=True,
        original_registry_arrays_consumes_truthful_short_registry=True)


def file_checks(loaded,rows):
    fixture=BASE/'synthetic_files_v1';fixture.mkdir(exist_ok=False)
    scope=fixture/'aggregate';scope.mkdir()
    check=lambda:None
    owner=limits.AggregateOwner(scope,execution_check=check,max_family_rss_bytes=2**30,max_seconds=60)
    owner(4096)
    data=scope/'small.data';data.write_bytes(b'abc')
    measured=limits.measure(scope)
    assert measured['logical_bytes']>=3 and measured['charged_bytes']==max(measured['logical_bytes'],measured['allocated_bytes'])
    fails(lambda:owner.path(fixture/'outside'))
    link=scope/'symlink';link.symlink_to(data)
    fails(lambda:limits.measure(scope))
    # Preserve rejection fixtures separately, outside the aggregate used below.
    link.rename(fixture/'retained_symlink_fixture')
    hard=scope/'hardlink';os.link(data,hard)
    fails(lambda:limits.measure(scope))
    hard.rename(fixture/'retained_hardlink_fixture')
    data.rename(fixture/'retained_hardlink_source')
    owner.refresh()
    saved=copy.deepcopy(owner._last)
    owner._last['started_monotonic']=time.monotonic()-6
    fails(owner.check)
    owner._last=saved
    owner._error='synthetic monitor exception'
    fails(owner.check)
    owner._error=None
    with patch.object(limits,'OPERATING',owner._last['charged_bytes']+100):
        fails(lambda:owner(100))
    log=scope/'previous.stdout.log';log.write_bytes(b'x'*4096)
    owner.refresh()
    assert owner._last['captured_log_bytes']==4096
    with patch.object(limits,'METADATA',owner._last['supervisor_metadata_bytes']+10):
        fails(lambda:owner.reserve_stream_write(100))
    owner(2**20)
    with patch.object(limits,'measure',side_effect=OSError('synthetic scan failure')):
        fails(lambda:owner.check()) if owner._error else None
        try:owner.refresh()
        except OSError:pass
        else:raise AssertionError('Scan failure was ignored')
    inputs=scope/'observed';inputs.mkdir()
    paths={name:inputs/name for name in ('genotypes.vcf','sample_ages.generations','population_labels.txt')}
    for path in paths.values():path.write_text('synthetic metadata\n')
    assets=dict(rows=rows,names=[r['sample_id'] for r in rows[::2]],paths=paths,
                position_join={101:dict(VCF_position_1based=101,tree_position_0based=100)},source_sha256={})
    def text_write(path,value,guard):
        raw=value.encode();guard(len(raw)+4096);path.write_bytes(raw)
    full=SimpleNamespace(confined=owner.path,roster=lambda:rows,text_write=text_write,
        json_write=lambda path,value,guard:text_write(path,json.dumps(value),guard))
    helper=SimpleNamespace(filter_vcf=lambda stream,names,positions:(['header'],[dict(fields=['synthetic'])],{}))
    with patch.object(stages,'_runtime',return_value=(None,None,full,helper)):
        stages.prepare_relate_inputs(loaded['recipe'],assets,scope/'prepared',guard=owner,
            execution_check=check,write_headroom_bytes=2**20)
    map_rows=(scope/'prepared/input.map').read_text().splitlines()
    assert map_rows==['pos COMBINED_rate Genetic_Map','1 1 0','10000001 1 10']
    owner.write_json(scope/'complete.terminal.json',dict(status='synthetic_only'))
    sample=limits.measure(scope)
    owner.close()
    assert LOG_LIMIT<DRAIN_LIMIT<limits.METADATA and limits.OPERATING+4*2**30==limits.RESERVATION
    leader=dict(pid=101,start=1000,session=101,ppid=1,state='S',rss_bytes=1,pgrp=101)
    child=dict(pid=102,start=1001,session=101,ppid=101,state='S',rss_bytes=1,pgrp=101)
    unrelated=dict(pid=999,start=99,session=999,ppid=1,state='S',rss_bytes=1,pgrp=999)
    family=owned_family({101:leader,102:child,999:unrelated},leader,{101:leader})
    assert set(family)=={101,102}
    recycled=dict(child,start=2000,session=999,ppid=1)
    assert 102 not in owned_family({102:recycled},leader,{102:child})
    return dict(tiny_file_bytes=sample['logical_bytes'],logical_allocated_max=True,
        traversal_symlink_hardlink_rejected=True,stale_and_error_rejected=True,
        prospective_write_and_aggregate_log_bound_rejected=True,exact_10cM_map=True,
        synthetic_PID_start_session_family_selection=True,owned_child_fixture_executed=False,
        metadata_only_native_and_runtime_launches=0)


def main():
    source=source_checks()
    loaded,rows,metadata=metadata_checks()
    files=file_checks(loaded,rows)
    receipt=dict(status='passed',schema='European_short_development_source_contracts_v1',source=source,
        metadata=metadata,files=files,plan_ref=short.plan_ref(),science_executed=False,models_constructed=False,
        simulations=0,native_processes=0,model_predictions=0,target_joins=0,production_modified=False,
        limitations=['Native supervision is source-reviewed and synthetic-tested, not a live child/process guarantee.',
                    'Periodic monitoring plus 4 GiB is provisional; no filesystem hard-write guarantee.',
                    'ROOT live host/global storage admission and independent source approval remain required.'])
    with (BASE/'SOURCE_CONTRACTS_CPU_v1.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
