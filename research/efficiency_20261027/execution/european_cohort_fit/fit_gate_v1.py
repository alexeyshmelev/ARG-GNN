"""Exact successful200Mb qualification and actual process completion gate."""
from research.efficiency_20261027.architecture.european_cohort_train.qualification_gate_v1 import ROOT,RUNS,GPU_UUID,need,contained,reference,read
from research.efficiency_20261027.execution.deferred_gpu_gate import validate_completion,owned_group_members
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity

SOURCE_MANIFEST_REF=dict(path='research/efficiency_20261027/architecture/european_cohort_train/source_pins_v2.json',sha256='56203f33d6acd884b4ee2221745956ac2592e6b47160ff577607adc6a9b1eaa0')
SOURCE_REVIEW_REF=dict(path='research/efficiency_20261027/fidelity/EUROPEAN_COHORT8_TRAIN_V2_SOURCE_REVIEW.json',sha256='2e044448dce28bd9138efde0f9225a3ae7c75487494c9ecb0f2c0c6352164f37')
QUALIFICATION_MODULE='research.efficiency_20261027.architecture.european_cohort_train.qualify_cli_v1'


def require_release(release,qualification_ref):
    need(release.get('schema')=='European_cohort8_TRAIN_release_v1' and release.get('passed') is True
         and release.get('source_manifest_ref')==SOURCE_MANIFEST_REF
         and release.get('source_review_ref')==SOURCE_REVIEW_REF
         and release.get('qualification_ref')==qualification_ref
         and release.get('qualification_actual_wait_returncode')==0
         and release.get('qualification_process_family_absent') is True
         and release.get('live_process_family_host_GPU_storage_admission_required') is True,
         'Exact ROOT successful qualification release, retaining separate live admission')


def require_qualification(value):
    from research.efficiency_20261027.architecture.european_cohort_train import driver_v2 as driver
    scientific=driver.read_small(driver.ORIGINAL_JOB_REF)['call']['spec']['scientific_spec']
    plan=driver.original._parent_plan(scientific['sampling_plan_ref'],scientific['protocol_id'])
    need(value.get('schema')=='European_cohort8_full_parent_qualification_v2' and value.get('passed') is True
         and value.get('source_manifest_ref')==SOURCE_MANIFEST_REF and value.get('scientific_condition')==driver.CONDITION
         and value.get('all_original_rows_used') is True and value.get('qualified_parent')==plan['train_parent_ids'][0]
         and value.get('prepared_parents')==plan['train_parent_ids'] and value.get('tolerances')==driver.TOLERANCES
         and value.get('gradient_tensors')==49 and value.get('parameter_tensors')==49 and value.get('Adam_tensors')==147
         and value.get('sampled_rows')==16667 and value.get('samples')==952 and value.get('argmax_changes')==0
         and value.get('TEST_truth_read') is False and value.get('no_qualification_checkpoint_used_for_fit') is True
         and 0<value.get('worker_peak_RSS_bytes',0)<=512*2**30,
         'Complete original200Mb qualification with all49 parameters, gradients and147 Adam tensors')


def qualification_release(qualification_path,qualification_sha256,release_path,release_sha256):
    qualification,qualification_ref=read(qualification_path,qualification_sha256)
    path=contained(qualification_path);directory=path.parent.parent
    need(directory.parent==RUNS and path.parent.name=='data' and path.name=='qualification.json',
         'Exact successful qualification in one ROOT run data directory')
    release,release_ref=read(release_path,release_sha256)
    need(contained(release_path).is_relative_to(ROOT/'research/efficiency_20261027/root'),
         'Explicit ROOT-owned FIT release')
    require_release(release,qualification_ref);require_qualification(qualification)
    request,request_ref=read(directory/'request.json');launch,launch_ref=read(directory/'launch.json')
    wait,wait_ref=read(directory/'actual_wait.json')
    command=request.get('command')
    need(type(command) is list and command==launch.get('actual_argv')
         and command[:4]==['.runtime/env/bin/python3.11','-B','-m',QUALIFICATION_MODULE]
         and request.get('physical_gpu')==5 and request.get('gpu_uuid')==GPU_UUID
         and type(launch.get('pid')) is int and launch['pid']>1
         and type(launch.get('process_start_ticks')) is str,
         'Exact visible successful qualification worker and admitted identity')
    for flag,expected in (('--output',str(path.parent)),('--source-pins-sha256',SOURCE_MANIFEST_REF['sha256']),
                          ('--source-review-sha256',SOURCE_REVIEW_REF['sha256'])):
        need(command.count(flag)==1 and command.index(flag)+1<len(command),'Unique qualification argument '+flag)
        actual=command[command.index(flag)+1]
        need((str(contained(actual)) if flag=='--output' else actual)==expected,'Exact qualification argument '+flag)
    validate_completion(launch,wait,identity(launch['pid']),owned_group_members(launch['pid']))
    completed,completed_ref=read(path.parent/'qualification_wrapper_completed.json')
    need(completed.get('schema')=='European_cohort8_qualification_CLI_completion_v1' and completed.get('passed') is True
         and completed.get('qualification_ref')==qualification_ref and completed.get('source_manifest_ref')==SOURCE_MANIFEST_REF
         and completed.get('actual_ROOT_request_ref')==request_ref and completed.get('actual_ROOT_launch_ref')==launch_ref,
         'Wrapper success binds the exact qualification and actual admitted process')
    return dict(qualification_ref=qualification_ref,fit_release_ref=release_ref,request_ref=request_ref,
        launch_ref=launch_ref,actual_wait_ref=wait_ref,wrapper_completion_ref=completed_ref,
        actual_wait=True,actual_returncode=0,owned_group_absent=True)
