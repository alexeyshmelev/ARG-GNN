"""CPU fixture of unchanged repair validation, runtime and admitted history restore.

Every write stays inside the candidate's synthetic project. Numerical training
and independently tested release authentication are mocked; archive validation,
transition checks, scheduler repair validation, ResourceCheck and restoration run
unchanged. No scheduler loop, subprocess launch, GPU or production path is used.
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import torch
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking.paper_campaign import runtime,repairs,scheduler
from . import entrypoint_combined_v4 as entry
from . import restore_entrypoint_combined_v4 as restore
from . import prepare_task_combined_v4 as builder
from .configuration_combined_v4 import configuration, EXPECTED
from .envelope_combined_v4 import resource_declaration

BASE=c.ROOT/'research/efficiency_20261027/integration_candidate'
JOB='.runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt7.json'


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2)
        stream.write('\n')
    return c.file_ref(path)


def fails(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError('Expected contract rejection')


def main():
    assert not torch.cuda.is_initialized()
    original=c.read_json(JOB)
    directory=BASE/'scheduler_restore_combined_fixture_v4'
    project=directory/'project'
    project.mkdir(parents=True,exist_ok=False)
    config=configuration(dict(EXPECTED))
    original_builder=Path(builder.__file__).read_text()
    original_repairs=Path(repairs.__file__).read_text()
    with patch.object(c,'ROOT',project),patch.object(runtime,'ROOT',project):
        work=project/'research/efficiency_20261027/capture'
        work.mkdir(parents=True)
        outer=project/original['output_directory']
        native=outer/'native'
        previous=native/'previous_attempt'
        archive=work/'archive'
        saved=archive/'native/previous_attempt/checkpoints/latest.pt'
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b'synthetic checkpoint only; no numerical resume assertion')
        captured=work/'captured.pt'
        captured.write_bytes(saved.read_bytes())
        target_checkpoint=dict(path=str((previous/'checkpoints/latest.pt').relative_to(project)),
            sha256=c.file_ref(saved)['sha256'])
        put(saved.parent.parent/'latest_checkpoint.json',dict(checkpoint_ref=target_checkpoint,complete_parent_updates=266))
        (saved.parent.parent/'progress.jsonl').write_text('{"event":"synthetic complete update history"}\n')
        put(archive/'actual_wait.json',dict(historical_outer_metadata=True))
        history_files=[]
        declared=[]
        for source in sorted((archive/'native').rglob('*')):
            if source.is_file():
                name=str(source.relative_to(archive/'native'))
                history_files.append(dict(relative_path=name,source_ref=c.file_ref(source),size_bytes=source.stat().st_size))
                declared.append(dict(path=str((native/name).relative_to(project)),sha256=c.file_ref(source)['sha256']))
        history=dict(schema='exact_preprocessing_preserved_history_v1',destination=entry.OUTPUT,
            files=history_files,file_count=len(history_files),total_bytes=sum(row['size_bytes'] for row in history_files))
        history_ref=put(work/'preserved_history.json',history)
        evidence_ref=put(work/'evidence.json',dict(complete_parent_updates=266,checkpoint_ref=c.file_ref(captured)))
        wait_ref=put(work/'actual_wait.json',dict(actual_wait=True,owned_group_absent=True,
            pid=12345,process_start_ticks='987',actual_returncode=-15))
        preproc_ref=put(work/'preprocessing_release.json',dict(synthetic=True))
        release_ref=put(work/'execution_release.json',dict(synthetic=True))
        prior=copy.deepcopy(original)
        prior['dependencies']=[]
        prior['python']='.runtime/env/bin/python'
        python=project/prior['python']
        python.parent.mkdir(parents=True)
        python.write_text('Synthetic interpreter marker; never executed.\n')
        source=work/'pinned_source.txt'
        source.write_text('Immutable synthetic source evidence.\n')
        prior['source_refs']=[c.file_ref(source)]
        prior['call']['spec']['successor']['preprocessing_release_ref']=preproc_ref
        job_ref=put(work/'prior_job.json',prior)
        approved=dict(configuration=config,resource_declaration=resource_declaration(prior,job_ref))
        transition=dict(schema='TreeLoom_existing_native_successor_boundary_v1',native_output_directory=entry.OUTPUT,
            predecessor_output_directory=str(previous.relative_to(project)),warm_observation_cache_transfer=False,
            ROOT_confirmed_no_other_writer=True,predecessor_pid=12345,predecessor_start_ticks='987',
            predecessor_actual_wait_ref=wait_ref,checkpoint_ref=c.file_ref(captured),evidence_ref=evidence_ref,
            predecessor_job_ref=job_ref,preserved_history_refs=declared)
        transition_ref=put(work/'transition.json',transition)
        fake_builder=work/'prepare_task_combined_v4.py'
        fake_builder.write_text(original_builder)
        fake_repairs=work/'repairs.py'
        fake_repairs.write_text(original_repairs)
        def inert_spec(**arguments):
            assert not (project/arguments['output_directory']).exists()
            return dict(output_directory=arguments['output_directory'],
                requirements=dict(output_scope_cap_bytes=64*2**20,source_refs=[c.file_ref(source)]))
        def fake_training(spec,*,resource_check):
            leaf=project/spec['output_directory']
            assert all(c.file_ref(project/ref['path'])==ref for ref in declared)
            leaf.mkdir()
            result=dict(complete_parent_updates=600,resumed_from_complete_parent_updates=266)
            result['completion_ref']=put(leaf/'training_completed.json',result)
            return result
        with patch.object(entry,'identity',return_value=None),patch.object(entry,'owned_group_members',return_value=[]), \
             patch.object(entry,'validate_release',return_value=approved), \
             patch.object(entry.previous,'authenticate_release',return_value=dict(encoder_workers=4)), \
             patch.object(restore.training,'build_spec',side_effect=inert_spec), \
             patch.object(builder,'__file__',str(fake_builder)),patch.object(repairs,'__file__',str(fake_repairs)), \
             patch.object(entry.training,'run_training',side_effect=fake_training):
            replacement=builder.build_task(transition_ref=transition_ref,execution_release_ref=release_ref,
                preserved_history_ref=history_ref,attempt_id='CPU_fixture')
            assert replacement['minimum_free_gpu_bytes']==76*2**30 and replacement['preferred_gpus']==[3]
            request=replacement['call']['spec']
            checked=restore.preflight(request)
            budget=checked['budget']
            assert not outer.exists()
            assert replacement['max_output_bytes']==original['max_output_bytes']
            original_read=entry.read
            def reject_transition(change):
                modified=copy.deepcopy(transition)
                change(modified)
                def read(ref):
                    return modified if ref==transition_ref else original_read(ref)
                with patch.object(entry,'read',side_effect=read):
                    fails(lambda:restore.preflight(request))
                assert not outer.exists()
            reject_transition(lambda value:value.update(preserved_history_refs=declared[:-1]))
            reject_transition(lambda value:value['preserved_history_refs'][0].update(sha256='0'*64))
            reject_transition(lambda value:value.update(predecessor_actual_wait_ref=dict(path=original['output_directory']+'/actual_wait.json',sha256='0'*64)))
            with patch.object(entry,'identity',return_value='987'):
                fails(lambda:restore.preflight(request))
            with patch.object(entry,'owned_group_members',return_value=[12346]):
                fails(lambda:restore.preflight(request))
            denied=copy.deepcopy(request)
            denied['successor']['arguments']['max_rss_bytes']+=1
            fails(lambda:restore.preflight(denied))
            fails(lambda:restore.output_budget(dict(prior,max_output_bytes=budget['required_output_bytes']-1),history,checked['spec']))
            plan=dict(schema='paper_campaign_v1',physical_gpus=[3,4,5],tasks=[prior],storage_ceiling_bytes=995000000000)
            plan_ref=put(work/'plan.json',plan)
            state=dict(plan_ref=plan_ref,plan_revision=7,archived_attempt_bytes=1234,
                tasks={prior['id']:dict(status='failed',attempt=5)})
            records=work/'controller'
            archive_bytes=runtime.scope_bytes(archive)
            manifest_ref=put(work/'archive_manifest.json',dict(archived_bytes=archive_bytes))
            repair=dict(schema=repairs.SCHEMA,request_id='synthetic_restore',expected_plan_ref=plan_ref,
                expected_plan_revision=7,expected_attempts={prior['id']:5},reset_task_ids=[],
                replacement_tasks=[replacement],archived_bytes=archive_bytes,archive_manifest_ref=manifest_ref,
                reason='CPU fixture: unchanged repair requires absent outer output; admitted runtime restores native history.')
            repair_ref=put(records/'repair_requests/synthetic_restore.request.json',repair)
            submission=records/'repair_requests/synthetic_restore.ref.json'
            put(submission,repair_ref)
            candidate,receipt=repairs._prepare(plan,state,records,submission)
            assert receipt['archived_bytes']==archive_bytes and receipt['resulting_attempts'][prior['id']]==6
            assert archive_bytes>history['total_bytes']
            outer.mkdir(parents=True)
            fails(lambda:repairs._prepare(plan,state,records,submission))
            outer.rmdir()
            applied=repairs.apply_pending(plan,state,records)
            assert len(applied)==1 and state['tasks'][prior['id']]['status']=='pending'
            assert state['archived_attempt_bytes']==1234+archive_bytes
            assert not outer.exists()
            # This is the scheduler's unchanged mkdir-before-runtime behavior.
            outer.mkdir()
            task_ref=put(records/'jobs/synthetic_restore.attempt6.json',replacement)
            runtime.run_task(task_ref['path'])
            result=runtime.read(runtime.ref(outer/'result.json'))
            assert result['scientific_success'] is True and result['value']['complete_parent_updates']==600
            assert all(c.file_ref(project/ref['path'])==ref for ref in declared)
            assert runtime.scope_bytes(outer)<=replacement['max_output_bytes']
            assert runtime.ref(saved)==history_files[0]['source_ref'] or all(
                c.file_ref(row['source_ref']['path'])==row['source_ref'] for row in history_files)
            restored_bytes=runtime.scope_bytes(native)
    receipt=dict(status='passed',test_ref=c.file_ref(Path(__file__).resolve()),
        source_pins_ref=c.file_ref(BASE/'source_pins_combined_v4.json'),CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),
        actual_repairs_prepare_and_apply_pending_used=True,actual_runtime_run_task_and_ResourceCheck_used=True,
        actual_validate_history_restore_history_and_live_transition_used=True,
        original_outer_scope_absent_for_repair=True,outer_created_by_simulated_scheduler_before_runtime=True,
        native_restored_inside_admitted_runtime=True,archive_bytes_include_outer_metadata=True,
        archive_ledger_increment_exact=True,all_history_digests_preserved=True,
        output_budget=budget,synthetic_restored_native_bytes=restored_bytes,
        exact_592_to_768GiB_task_and_argument_delta=True,
        exact_scheduler_56_to_76GiB_free_GPU_delta=True,preferred_GPUs_unchanged=[3],actual_attempt7_nested_shape_used=True,
        metadata_tamper_missing_history_live_predecessor_undeclared_cap_budget_overflow_rejected=True,
        release_authentication_mocked_and_separately_tested=True,numerical_training_mocked=True,
        no_training_or_scheduler_loop_or_process_launch=True,production_modified=False,jobs_started=0)
    put(BASE/'SCHEDULER_RESTORE_COMBINED_CPU_v4.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
