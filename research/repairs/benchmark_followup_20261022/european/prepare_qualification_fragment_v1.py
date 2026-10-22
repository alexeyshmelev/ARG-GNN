"""Prepare two fresh CPU qualification tasks and fit rebindings, without dispatch.

The historical qualification receipts remain unchanged. No policy, current plan,
queue state, scientific output, process or pinned source is modified.
"""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from research.repairs.benchmark_resume_20261021.gpu import readable_workflow_v1 as workflow
from research.repairs.benchmark_resume_20261021.gpu.queue_adapters import common, european_stage as stage

ROOT = workflow.c.ROOT
HERE = ROOT/'research/repairs/benchmark_followup_20261022/european'
FACTORY = 'research.repairs.benchmark_resume_20261021.gpu.queue_adapters.european_stage'
EXPECTED_PLAN = dict(path='configs/benchmarks/executable_workflow_20261021_gpu5_v1.json',
    sha256='ae644a8df467b85d9b029d48fc9391b86ebd99e663f4c82e7000228166978bd3')
PAIRS = [('AP__train_000__treeloom_preparation_qualification','AP__all_train__treeloom_fit'),
         ('C4__train_000__treeloom_preparation_qualification','C4__all_train__c4_treeloom_fit')]


def need(value,message):
    if not value:raise ValueError(message)


def main():
    policy=workflow.c.read_json(workflow.c.POLICY_PATH)
    need(policy['approved_readable_workflow_ref']==EXPECTED_PLAN,'Diagnosed exact current GPU5 plan required')
    plan=workflow.c.read_ref(EXPECTED_PLAN)
    state_path=ROOT/plan['records_directory']/'state.json'
    state=workflow.c.read_json(state_path)
    by_id={task['id']:task for task in plan['tasks']}
    expected=common.source_refs()
    need(len(expected)==21,'Retain the exact current21-source qualification contract')
    need(FACTORY in plan['readable_factory_modules'],'Current approved factory allowlist must already include qualifier')
    qualifications=[];rebindings=[];diagnoses=[];calls=[]
    for previous_id,fit_id in PAIRS:
        previous=by_id[previous_id];fit=by_id[fit_id]
        entry=state['tasks'][fit_id]
        need(entry['status']=='failed' and 'job_ref' not in entry and 'reservation_bytes' not in entry,
             'Only the diagnosed pre-materialization failure may be rebound')
        need(not (ROOT/fit['output_directory']).exists(),'Failed fit must have no created output scope')
        fresh=copy.deepcopy(previous)
        fresh['id']=previous_id+'_gpu5_20261022_v1'
        fresh['factory_module']=FACTORY
        fresh['output_directory']='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_gpu5_qualifications/'+fresh['id']
        need(fresh['id'] not in by_id and not (ROOT/fresh['output_directory']).exists(),'Fresh task ID and output scope required')
        kwargs=workflow.resolve(fresh['factory_arguments'],state)
        call=stage.qualification_factory(output_directory=fresh['output_directory']+'/native',**kwargs)
        need(call['spec']['candidate_source_refs']==expected,'Qualify exactly the source list expected by the unchanged fits')
        need(call['module']==FACTORY and call['function']=='run_qualification','Current ordinary GPU5 qualifier required')
        rebound=copy.deepcopy(fit)
        rebound['dependencies']=[fresh['id'] if dep==previous_id else dep for dep in fit['dependencies']]
        rebound['factory_arguments']['qualification_ref']={'$task':fresh['id']}
        resolved_fit=workflow.resolve(fit['factory_arguments'],state)
        old_ref=resolved_fit.pop('qualification_ref')
        spec=stage.build_fit_spec(output_directory=fit['output_directory']+'/native',**resolved_fit)
        first=spec['train_inputs'][kwargs['parent_id']]
        observed={key:first[key] for key in stage.INPUTS}
        need(call['spec']['observed_inputs']==observed,'Identical actual first-parent observation refs required')
        need(all(call['spec'][key]==spec[key] for key in ['protocol_id','sampling_plan_ref','max_sites','max_metadata_bytes','preparation_limits']),
             'Identical scientific preparation parameters required')
        historical=workflow.c.read_ref(old_ref)
        diagnoses.append(dict(fit_task_id=fit_id,historical_qualification_ref=old_ref,
            historical_schema=historical['schema'],historical_candidate_source_count=len(historical['candidate_source_refs']),
            required_schema='european_readable_complete_parent_CPU_qualification_v1',required_candidate_source_count=len(expected),
            observed_inputs_unchanged=historical['observed_inputs']==observed,
            preparation_contract_unchanged=historical['preparation_contract']=={key:spec[key] for key in ['max_sites','max_metadata_bytes','preparation_limits']}))
        qualifications.append(fresh)
        calls.append(dict(task_id=fresh['id'],descriptor=call))
        rebindings.append(dict(task_id=fit_id,previous_task_fingerprint=workflow.c.fingerprint(fit),
            previous_failed_state=entry,replacement_task=rebound,
            reconciliation='ROOT preserves the original materialization failure and may mark this exact never-spawned task pending only in a separately approved successor state'))
    refs={ref['path']:ref for call in calls for ref in call['descriptor']['source_refs']}
    fragment=dict(schema='two_fresh_European_GPU5_CPU_qualifications_plan_fragment_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),base_plan_ref=EXPECTED_PLAN,
        purpose='Same scientific inputs after the ordinary-source/GPU5 operational migration, with genuine new full CPU qualification evidence',
        qualification_tasks=qualifications,fit_rebindings=rebindings,qualification_descriptors=calls,
        exact_candidate_source_refs=expected,required_source_refs=list(refs.values()),diagnoses=diagnoses,
        no_execution_authority=True,canonical_policy_modified=False,current_plan_or_state_modified=False,
        benchmark_launched=False,processes_signalled=False,
        integration_notes=['Append both fresh qualification tasks; keep both historical qualification tasks and receipts unchanged.',
            'Use fit replacement tasks only after source/plan/state reconciliation by ROOT.',
            'Successful qualification returns its actual qualification.json FileRef through result.json; fit dependencies resolve only succeeded task outputs.',
            'Current _require_pass schema, all21 exact source refs, preparation/input checks and scientific bodies remain unchanged.',
            'Preserve the healthy running ARGformer MLM and its exact release/plan authority; this fragment does not authorize a controller or policy transition.'])
    path=HERE/'qualification_plan_fragment_v1.json'
    with path.open('x') as stream:json.dump(fragment,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(fragment_ref=workflow.c.file_ref(path),new_tasks=[task['id'] for task in qualifications],
        exact_candidate_sources=len(expected),required_sources=len(refs)),sort_keys=True),flush=True)


if __name__=='__main__':main()
