"""Seal reviewed source candidates and CPU evidence;this does not launch work."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/efficiency_20261027/execution/european_cohort_fit'
ARCH=ROOT/'research/efficiency_20261027/architecture/european_cohort_train'

def ref(path):
    path=ROOT/Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def put(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return ref(path)


def main():
    inherited=json.loads((ARCH/'source_pins_v2.json').read_text())['source_refs']
    for row in inherited:assert ref(row['path'])==row
    refs={row['path']:row for row in inherited}
    for path in (ARCH/'source_pins_v2.json',ARCH/'driver_v3.py',HERE/'test_restore_v3.py'):
        value=ref(path);refs[value['path']]=value
    driver_ref=put(ARCH/'source_pins_v3.json',dict(schema='European_cohort8_TRAIN_resume_sources_v3',
        source_refs=[refs[key] for key in sorted(refs)],numerical_driver_v2_unchanged=True,
        original_3x200Mb_TRAIN_and90update_protocol_unchanged=True,production_modified=False))
    closure=json.loads((ARCH/'qualification_cli_pins_v1.json').read_text())['source_refs']
    refs={row['path']:row for row in closure}
    for row in closure:assert ref(row['path'])==row
    for path in [*HERE.glob('*.py'),ARCH/'source_pins_v3.json',ARCH/'driver_v3.py',
                 HERE/'CONTRACTS_CPU_v1.json',HERE/'CONTRACTS_CPU_v2.json',HERE/'RESTORE_EQUIVALENCE_CPU_v3.json']:
        value=ref(path);refs[value['path']]=value
    wrapper_ref=put(HERE/'fit_cli_pins_v2.json',dict(schema='European_cohort8_FIT_CLI_sources_v2',
        source_refs=[refs[key] for key in sorted(refs)],resume_driver_source_manifest_ref=driver_ref,
        no_launch_authorization=True,production_modified=False))
    status=dict(schema='European_cohort8_supervised_FIT_candidate_v2',status='Awaiting independent source review and exact successful qualification release',
        wrapper_manifest_ref=wrapper_ref,resume_driver_manifest_ref=driver_ref,
        CPU_contracts_ref=ref(HERE/'CONTRACTS_CPU_v2.json'),CPU_restore_ref=ref(HERE/'RESTORE_EQUIVALENCE_CPU_v3.json'),
        exact_v2_driver_review_ref=ref(ROOT/'research/efficiency_20261027/fidelity/EUROPEAN_COHORT8_TRAIN_V2_SOURCE_REVIEW.json'),
        fresh_fit_supported=True,exact_continuation_supported=True,fixed_updates=90,fixed_epochs=30,seed=101,
        original_scientific_writer_bytes=64*2**20,resume_writer_bytes=768*2**20,whole_run_output_allowance_bytes=2*2**30,
        per_checkpoint_bytes=8*2**20,CPU_synthetic_checkpoint_bytes=463080,
        worker_RSS_bytes=512*2**30,minimum_outer_family_cap_bytes=512*2**30,
        GPU_policy='Original full device allocator;ROOT finite measured soft guard chosen after actual European qualification;no American74GiB inference',
        free_GPU_required='max(ROOT soft guard +2GiB,original64GiB)',physical_gpu=5,gpu_uuid='GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
        checkpoint_policy='Immutable recovery checkpoint after every complete parent update;original epoch5,10,15,20,25,30 scientific checkpoints unchanged;final30 fixed',
        resume_policy='Exact selected v3 recovery checkpoint plus prior actual_wait and absent family;revalidate all3observations;restore all49weights,147Adam tensors,Python/NumPy/TorchCPU/CUDA RNG and nextparent;continue only remaining budget in fresh run',
        limitations=['CPU equality test uses actual cohort model with stochastic synthetic gradients;no GPU/native/scientific data executed.',
            'Fresh observation preparation repeats after interruption;this does not repeat accepted optimizer updates.',
            'Periodic whole-run output monitoring is not a filesystem quota;ROOT still owns live global1TB/project,host,peer and GPU admission.',
            'Immutable per-update recovery artifacts and failed run evidence are retained;no old files or jobs are removed.'],
        launched=False,production_modified=False)
    status_ref=put(HERE/'IMPLEMENTATION_STATUS_v2.json',status)
    plan=dict(schema='European_cohort8_FIT_ROOT_launch_contract_v2',launchable=False,
        reason='Fill exact terminal qualification/release/review and current admission before ROOT-only launch',
        guardian_module='research.efficiency_20261027.execution.european_cohort_fit.run_guarded_fit_v1',
        worker_module='research.efficiency_20261027.execution.european_cohort_fit.fit_cli_v2',
        wrapper_manifest_ref=wrapper_ref,resume_driver_manifest_ref=driver_ref,
        required_guardian_arguments=['--directory ROOT/runs/FRESH_RUN','--peer-directory EACH_CURRENT_ROOT_PEER','--gpu 5','--host-gib >=512',
            '--device-gib ROOT_SELECTED_EUROPEAN_SOFT_GUARD','--wait-seconds FINITE','--run-seconds FINITE_<=864000','-- VISIBLE_PROJECT_PYTHON -B -m WORKER_MODULE WORKER_ARGS'],
        required_worker_arguments=['--output SAME_FRESH_RUN/data','--source-pins-sha256 56203f33d6acd884b4ee2221745956ac2592e6b47160ff577607adc6a9b1eaa0',
            '--source-review-sha256 2e044448dce28bd9138efde0f9225a3ae7c75487494c9ecb0f2c0c6352164f37',
            '--wrapper-pins-sha256 '+wrapper_ref['sha256'],'--wrapper-review-path EXACT_REVIEW','--wrapper-review-sha256 EXACT_REVIEW_SHA',
            '--resume-driver-pins-sha256 '+driver_ref['sha256'],'--qualification-path EXACT_SUCCESSFUL_ROOT_RUN/data/qualification.json',
            '--qualification-sha256 EXACT_QUALIFICATION_SHA','--fit-release-path EXACT_ROOT_RELEASE','--fit-release-sha256 EXACT_RELEASE_SHA',
            '--family-cap-bytes SAME_GUARDIAN_BYTES','--device-soft-bytes SAME_GUARDIAN_BYTES','--output-cap-bytes 2147483648',
            '--max-seconds SAME_RUN_SECONDS','--gpu-wait-seconds FINITE_<=MAX','--required-peer-directory EACH_SAME_ROOT_PEER'],
        optional_resume_arguments=['--resume-checkpoint-path EXACT_REAPED_FIT/data/resume/update_NNN.pt','--resume-checkpoint-sha256 EXACT_HASH'],
        expected_review_schema='European_cohort8_FIT_CLI_review_v2',expected_review_fields=dict(passed=True,
            wrapper_manifest_ref=wrapper_ref,resume_restore_CPU_equivalence_passed=True),
        expected_release_schema='European_cohort8_TRAIN_release_v1',
        final_required_receipts=['data/training_completed.json','data/fit_wrapper_completed.json','actual_wait.json returncode0 and absent family'],
        progress_file='data/fit_progress.json',resume_checkpoint_hint='latest_resume_checkpoint_ref;ROOT verifies exact hash and previous actual wait before selecting')
    plan_ref=put(HERE/'ROOT_LAUNCH_CONTRACT_v2.json',plan)
    print(json.dumps(dict(driver_manifest=driver_ref,wrapper_manifest=wrapper_ref,status=status_ref,launch_contract=plan_ref,wrapper_refs=len(refs))))

if __name__=='__main__':main()
