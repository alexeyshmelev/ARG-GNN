"""Small operational and mock-stage checks; no simulation or native tool execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from . import contracts_v2 as c,aggregate_guard_v2 as guard,worker_v2 as worker

HERE=Path(__file__).parent
CHECKS=[]


def check(name,ok=True):
    assert ok,name
    CHECKS.append(name)


def rejected(name,function):
    try:function()
    except (ValueError,guard.GuardFailure):check(name)
    else:raise AssertionError(name+' must reject')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--receipt',default=str(HERE/'CONTRACTS_CPU_v2.json'))
    args=parser.parse_args()
    destination=Path(args.receipt)
    assert destination.is_relative_to(HERE)
    frozen=c.validate_plan()
    check('original_full200Mb_parameters_seeds_and_sampling_authenticated')
    check('original_total_and_stage_resource_budgets',c.RESERVATION==128983236608
          and c.HOST_CAP==128*2**30 and c.OPERATING==c.RESERVATION-4*2**30
          and c.MAX_SECONDS==691200 and c.RETAINED_SHORT_RESERVATION==16*2**30)
    auth=c.owner_authorization()
    check('explicit_owner_authorization_is1p2TB',c.read(auth)['authorized_limit_bytes']==1200000000000)
    from research.efficiency_20261027.architecture.european_cohort_candidate import validation_generation_v2,validation_sampling
    from arggnn.benchmarking import european_native_relate
    check('scientific_generation_native_and_sampling_are_original_imports',
        worker.generation.generate_validation_parent is validation_generation_v2.generate_validation_parent
        and worker.native_relate.infer_parent is european_native_relate.infer_parent
        and worker.sampling.build_validation_registry is validation_sampling.build_validation_registry)
    with tempfile.TemporaryDirectory(prefix='CPU_full_val_',dir=HERE) as temporary:
        scope=Path(temporary);trace=[]
        def execution_check():trace.append('guard')
        owner=guard.AggregateOwner(scope,execution_check=execution_check,
            max_family_rss_bytes=c.HOST_CAP,max_seconds=30)
        def save(path,value):
            path.write_text(json.dumps(value,sort_keys=True)+'\n')
            return c.ref(path)
        def generate(recipe,parent,output,**kwargs):
            check('generator_receives_full_recipe_and_original_simulation_seeds',recipe is frozen['recipe']
                  and parent['parent_id']==frozen['parent']['parent_id'] and parent['split']=='validation'
                  and parent['seeds']=={key:frozen['parent']['seeds'][key] for key in ('ancestry','mutations')}
                  and kwargs['require_paper_match'] is True and kwargs['guard'] is owner
                  and kwargs['execution_check'] is execution_check and owner.stage=='generation')
            trace.append('generate');(output/'observed').mkdir()
            files={}
            for name in worker.native_relate.OBSERVED_FILES:
                path=output/'observed'/name;path.write_text('Synthetic observed fixture\n')
                files['observed/'+name]=dict(sha256=c.ref(path)['sha256'],bytes=path.stat().st_size)
            receipt=dict(files=files);save(output/'generation.json',receipt);return receipt
        fake_supervisor=object()
        def infer(recipe,parent,observed,destination,**kwargs):
            check('native_receives_original_full_parent_and_same_owner',recipe is frozen['recipe']
                  and parent is frozen['parent'] and kwargs['guard'] is owner
                  and kwargs['native_step'] is fake_supervisor and kwargs['execution_check'] is execution_check
                  and owner.stage=='native' and set(kwargs['expected_input_refs'])==set(worker.native_relate.OBSERVED_FILES))
            trace.append('infer');destination.mkdir()
            rows_ref=save(destination/'observed_sample_manifest.json',[dict(synthetic=True)])
            tree_ref=save(destination/'synthetic_observed_tree.json',{'synthetic':True})
            commands_ref=save(destination/'commands.json',{'mock_only':True})
            result=dict(observed_sample_manifest=rows_ref,inferred_tree=tree_ref,commands=commands_ref,
                post_five_iterations={'mock_only':True},sequence_length_bp=200000000)
            save(destination/'inference.json',result);return result
        def sample(plan,parent,rows,inferred,**kwargs):
            check('sampler_receives_original_full_plan_and_observation_only_metadata',plan is frozen['sampling_plan']
                  and parent==frozen['parent']['parent_id'] and rows==[dict(synthetic=True)]
                  and owner.stage is None and kwargs['observation_ref']==c.ref(scope/'native/inference.json'))
            trace.append('sample');return dict(examples=10000,labels_present=False,query_truth_read=False,split='validation')
        try:
            with patch.object(worker,'AGGREGATE',scope),patch.object(worker.generation,'generate_validation_parent',side_effect=generate),\
                 patch.object(worker.native_relate,'infer_parent',side_effect=infer),\
                 patch.object(worker.sampling,'build_validation_registry',side_effect=sample),\
                 patch.object(worker,'NativeSupervisor',return_value=fake_supervisor):
                result=worker.run_stages(owner,frozen,c.read(c.NATIVE_CONFIG_REF),{'mock_tool':'mock_hash'})
            check('serial_original_stage_order', [x for x in trace if x!='guard']==['generate','infer','sample'])
            check('full_size_observed_only_result',result['sequence_length_bp']==200000000
                  and result['sampled_examples']==10000 and result['full_independent_validation_parent'] is True
                  and result['model_fit_prediction_or_scoring_performed'] is False
                  and result['original_TRAIN_TEST_unchanged'] is True)
            owner.set_stage('generation')
            rejected('generation_prospective24GiB_cap',lambda:owner(24*2**30))
            owner.set_stage('native')
            rejected('native_prospective96GiB_cap',lambda:owner(96*2**30))
            owner.set_stage(None)
            rejected('aggregate_prospective_operating_cap',lambda:owner(c.OPERATING))
            owner.set_stage('generation');owner.stage_started-=86401
            rejected('original_generation24hour_deadline',owner.check)
            owner.stage_started+=86401;owner.set_stage('native');owner.stage_started-=604801
            rejected('original_native7day_deadline',owner.check)
        finally:owner.close()
    manifest_ref=dict(path='synthetic/source_pins.json',sha256='0'*64)
    review_ref=dict(path='synthetic/review.json',sha256='1'*64)
    release=dict(schema='European_full_VAL_ROOT_execution_release_v1',passed=True,execution_authorized=True,
        source_manifest_ref=manifest_ref,independent_source_review_ref=review_ref,plan_ref=c.PLAN_REF,
        owner_storage_authorization_ref=auth,aggregate_directory=str(c.AGGREGATE.relative_to(c.ROOT)),
        host_cap_bytes=c.HOST_CAP,reservation_bytes=c.RESERVATION,operating_stop_bytes=c.OPERATING,
        max_seconds=c.MAX_SECONDS,CUDA_VISIBLE_DEVICES='',peer_directories=[],global_project_ceiling_bytes=c.PROJECT_CAP,
        ROOT_live_global_storage_and_peer_admission_required=True,model_fit_prediction_or_scoring_authorized=False,
        ROOT_status_max_age_seconds=c.ROOT_STATUS_MAX_AGE_SECONDS,growth_refresh_interval_seconds=c.GROWTH_INTERVAL_SECONDS,
        retained_attempts_ref=c.ref(c.RETAINED_PATH),retained_short_reservation_in_other_bytes=c.RETAINED_SHORT_RESERVATION)
    c.validate_release(release,manifest_ref,review_ref,[]);check('exact_ROOT_full_size_release_contract')
    for key,value in (('global_project_ceiling_bytes',1000000000000),('reservation_bytes',16*2**30),
                      ('model_fit_prediction_or_scoring_authorized',True),('retained_short_reservation_in_other_bytes',0)):
        rejected('altered_release_'+key,lambda key=key,value=value:c.validate_release(dict(release,**{key:value}),manifest_ref,review_ref,[]))
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    check('all_full_runner_sources_parse')
    report=dict(schema='European_full_VAL_ROOT_runner_CPU_contracts_v2',passed=True,checks=CHECKS,check_count=len(CHECKS),
        original_plan_ref=c.PLAN_REF,owner_storage_authorization_ref=auth,
        source_refs=[c.ref(HERE/name) for name in ('contracts_v2.py','aggregate_guard_v2.py','worker_v2.py','guardian_v2.py','native_supervisor_v2.py','test_contracts_v2.py')],
        simulation_native_model_GPU_executions=0,real_scientific_payloads_opened=False,
        source_algorithms_reused_without_changes=True,retained_evidence_unchanged=True)
    with destination.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,checks=len(CHECKS),receipt=c.ref(destination))))


if __name__=='__main__':main()
