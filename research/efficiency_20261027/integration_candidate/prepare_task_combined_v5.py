"""Build an inert scheduler-compatible replacement after ROOT archive; never submit."""
import argparse
import copy
import json
from pathlib import Path
from arggnn.benchmarking import contracts as c
from . import entrypoint_combined_v5 as entry
from .envelope_combined_v5 import predecessor_request, validate_task
from . import restore_entrypoint_combined_v5 as restore


def build_task(*, transition_ref, execution_release_ref, preserved_history_ref, attempt_id):
    release=entry.validate_release(execution_release_ref)
    transition=entry.read(transition_ref)
    prior=entry.read(transition['predecessor_job_ref'])
    task=copy.deepcopy(prior)
    old=predecessor_request(prior)
    arguments=copy.deepcopy(old['arguments'])
    arguments.update(resume_checkpoint_ref=transition['checkpoint_ref'],
        resume_evidence_ref=transition['evidence_ref'],execution_configuration=release['configuration'],
        max_rss_bytes=release['configuration']['max_rss_bytes'])
    preprocessing_ref=old.get('release_ref',old.get('preprocessing_release_ref'))
    successor=dict(stage='fit',arguments=arguments,preprocessing_release_ref=preprocessing_ref,
        execution_release_ref=execution_release_ref,transition_ref=transition_ref,attempt_id=attempt_id)
    request=dict(successor=successor,preserved_history_ref=preserved_history_ref)
    checked=restore.preflight(request)
    spec=checked['spec']
    task['cpu_threads']=8
    task['max_rss_bytes'] = release['configuration']['max_rss_bytes']
    task['minimum_free_gpu_bytes'] = release['configuration']['minimum_free_gpu_bytes']
    task['call']=dict(module='research.efficiency_20261027.integration_candidate.restore_entrypoint_combined_v5',
        function='run_task',spec=request)
    refs={ref['path']:ref for ref in task['source_refs']}
    for ref in spec['requirements']['source_refs']+[transition_ref,execution_release_ref,
        preprocessing_ref,preserved_history_ref,c.file_ref(Path(__file__).resolve())]:
        if ref['path'] in refs:
            entry.need(refs[ref['path']]==ref,'Existing immutable source reference must remain exact')
        refs[ref['path']]=ref
    task['source_refs']=[refs[key] for key in sorted(refs)]
    validate_task(task,prior,release)
    return task


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--transition',required=True)
    parser.add_argument('--execution-release',required=True)
    parser.add_argument('--preserved-history',required=True)
    parser.add_argument('--attempt-id',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    task=build_task(transition_ref=c.file_ref(args.transition),execution_release_ref=c.file_ref(args.execution_release),
        preserved_history_ref=c.file_ref(args.preserved_history),attempt_id=args.attempt_id)
    output=entry.previous._ordinary_path(args.output,allow_missing=True)
    entry.need(output.is_relative_to(c.ROOT/'research/efficiency_20261027/integration_candidate')
               and not output.exists(),'Fresh inert candidate metadata output only')
    with output.open('x') as stream:
        json.dump(task,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(task_ref=c.file_ref(output),jobs_started=0,production_modified=False)))


if __name__=='__main__':main()
