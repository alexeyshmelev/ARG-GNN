"""Seal one plaintext full200Mb runner assembly; never launch its worker."""
from pathlib import Path
import ast
import difflib
import importlib.metadata
import json
import platform
from . import contracts_v1 as c

HERE=Path(__file__).parent


def save(name,value):
    path=HERE/name
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    return c.ref(path)


def main():
    need=c.need
    dependencies={}
    inherited=[]
    for relative,keys,expected in (
        ('research/efficiency_20261027/execution/native_terminal_pins_v5.json',('source_refs',),'a01db04b59f34a97445746c0e11d7a8de8df4a41be1874dabdd33742a05f2131'),
        ('research/efficiency_20261027/execution/european_short_val_runner/source_pins_v3.json',('source_refs',),'55d1befb175122ce198a505e11398edc8f422ff8709134216902b819d5bcc36f'),
        ('research/efficiency_20261027/architecture/european_cohort_candidate/source_manifest_v1.json',('sources','inherited_dependencies','selected_CPU_evidence_refs','selected_plan_refs'),'e2a5a951d4674270faaaaa98b62c2ae3006ed3bb36876aa6fe8af2caa31c4a7a')):
        binding=dict(path=relative,sha256=expected);manifest=c.read(binding);inherited.append(binding)
        dependencies[relative]=binding
        for key in keys:
            for item in manifest[key]:
                need(c.ref(item['path'])==item,'Inherited source changed: '+item['path'])
                need(item['path'] not in dependencies or dependencies[item['path']]==item,'Conflicting inherited pin')
                dependencies[item['path']]=item
    frozen=c.validate_plan();config,tools,chain_ref=c.native_bindings()
    owner_ref=c.owner_authorization();retained_ref,retained_bytes=c.retained_attempts()
    native_review_ref=c.ref(c.ROOT/'research/efficiency_20261027/fidelity/NATIVE_TERMINAL_V5_SOURCE_REVIEW.json')
    native_review=c.read(native_review_ref)
    need(native_review['passed'] is True and native_review['authenticated_source_refs']==219,'Independently reviewed native v5')
    checks=c.read(c.ref(HERE/'CONTRACTS_CPU_v1.json'));native_checks=c.read(c.ref(HERE/'NATIVE_BINDING_CPU_v1.json'))
    need(checks['passed'] is True and checks['check_count']==20 and native_checks['passed'] is True
        and native_checks['actual_tiny_shell_checks_passed']==3,'Completed mock and actual tiny-shell checks')
    sources=('contracts_v1.py','aggregate_guard_v1.py','worker_v1.py','guardian_v1.py','native_supervisor_v1.py',
             'test_contracts_v1.py','test_native_binding_v1.py','seal_sources_v1.py')
    for name in sources:ast.parse((HERE/name).read_text(),filename=name)
    prior=c.ROOT/'research/efficiency_20261027'
    pairs=(
        (prior/'execution/european_short_val_runner/contracts_v3.py',HERE/'contracts_v1.py'),
        (prior/'integration_candidate/european_short_development_val/aggregate_guard_v3.py',HERE/'aggregate_guard_v1.py'),
        (prior/'execution/european_short_val_runner/worker_v3.py',HERE/'worker_v1.py'),
        (prior/'execution/european_short_val_runner/guardian_v3.py',HERE/'guardian_v1.py'),
        (prior/'integration_candidate/european_short_development_val/native_supervisor_v5.py',HERE/'native_supervisor_v1.py'))
    with (HERE/'FULL_VAL_PLAINTEXT_v1.diff').open('x') as stream:
        for old,new in pairs:
            stream.writelines(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),
                fromfile=str(old.relative_to(c.ROOT)),tofile=str(new.relative_to(c.ROOT))))
    runtime_ref=save('SCIENTIFIC_RUNTIME_v1.json',dict(schema='European_full_VAL_runtime_versions_v1',
        python=platform.python_version(),implementation=platform.python_implementation(),
        distributions={name:importlib.metadata.version(name) for name in ('msprime','tskit','numpy','scipy')},
        interpreter='.runtime/env/bin/python3.11',native_config_ref=c.NATIVE_CONFIG_REF,
        installed_native_chain_review_ref=chain_ref,record_only=True,
        ROOT_must_verify_current_versions_at_launch=True))
    status_ref=save('IMPLEMENTATION_STATUS_v1.json',dict(schema='European_full_VAL_runner_implementation_v1',
        original_complete_200Mb_registration_ref=c.PLAN_REF,owner_storage_authorization_ref=owner_ref,
        exact_original_scientific_callables=[frozen['plan'][role]['source_ref'] for role in ('generator','inference','registry')],
        scientific_source_algorithms_changed=False,original_TRAIN_TEST_and_failed_short_attempts_unchanged=True,
        native_terminal_v5_review_ref=native_review_ref,
        contract_CPU_ref=c.ref(HERE/'CONTRACTS_CPU_v1.json'),native_binding_CPU_ref=c.ref(HERE/'NATIVE_BINDING_CPU_v1.json'),
        retained_attempts_ref=retained_ref,retained_short_charged_bytes_at_seal=retained_bytes,
        draft_repairs=['Before sealing, removed an indentation error from a copied emergency receipt block.',
            'The first contract run and recorded attempt2 rejected an overly narrow authorization scope comparison; the gate now matches the owner receipt exact project-plus-host scope. Attempt3 passed20 checks.'],
        log_evidence=[c.ref(HERE/name) for name in ('CONTRACTS_CPU_attempt2.log','CONTRACTS_CPU_attempt3.log','NATIVE_BINDING_CPU_attempt1.log')],
        completed_scope='Source assembly and tiny CPU operational checks only.',
        launch_authorization=False,scientific_native_model_GPU_executions=0,
        limitations=['One independent validation parent cannot establish across-parent variance.',
            'Periodic external-write sampling plus4GiB reserved margin is provisional, not a filesystem quota.',
            'Full-size scientific execution, model fitting, prediction and scoring have not been performed by this assembly.']))
    extra=[owner_ref,retained_ref,native_review_ref,runtime_ref,status_ref,c.PLAN_REF,c.NATIVE_CONFIG_REF,chain_ref,
           frozen['plan']['canonical_recipe_ref'],frozen['plan']['sampling_plan_ref'],native_review['independent_actual_CPU_receipt_ref']]
    extra.extend(c.ref(HERE/name) for name in (*sources,'CONTRACTS_CPU_v1.json','NATIVE_BINDING_CPU_v1.json',
        'FULL_VAL_PLAINTEXT_v1.diff','CONTRACTS_CPU_attempt2.log','CONTRACTS_CPU_attempt3.log','NATIVE_BINDING_CPU_attempt1.log'))
    for attempt in c.read(retained_ref)['attempts']:extra.extend(attempt['evidence_refs'])
    for item in extra:
        need(c.ref(item['path'])==item,'Local assembly pin changed');dependencies[item['path']]=item
    resource=dict(aggregate_directory=str(c.AGGREGATE.relative_to(c.ROOT)),host_cap_bytes=c.HOST_CAP,
        reservation_bytes=c.RESERVATION,operating_stop_bytes=c.OPERATING,max_seconds=c.MAX_SECONDS,
        generation_output_cap_bytes=24*c.GIB,native_output_cap_bytes=96*c.GIB,registry_max_bytes=128*2**20,
        generation_max_seconds=86400,native_max_seconds=604800,native_step_max_seconds=172800,
        native_threads=8,native_chunk_memory_gb=2,CUDA_VISIBLE_DEVICES='',
        global_project_ceiling_bytes=c.PROJECT_CAP,project_admission_ceiling_bytes=c.PROJECT_ADMISSION_CAP,
        retained_short_reservation_in_other_bytes=c.RETAINED_SHORT_RESERVATION,
        retained_short_charged_bytes_at_seal=retained_bytes,ROOT_status_max_age_seconds=c.ROOT_STATUS_MAX_AGE_SECONDS,
        growth_refresh_interval_seconds=c.GROWTH_INTERVAL_SECONDS,
        external_write_monitor_is_provisional_not_filesystem_quota=True)
    manifest_ref=save('source_pins_v1.json',dict(schema='European_full_VAL_ROOT_runner_sources_v1',
        source_refs=[dependencies[name] for name in sorted(dependencies)],inherited_manifest_refs=inherited,
        original_registration_ref=c.PLAN_REF,owner_storage_authorization_ref=owner_ref,
        native_terminal_v5_review_ref=native_review_ref,resource_contract=resource,
        original_scientific_algorithms_unchanged=True,registered_parent_count=1,
        failed_short_attempts_preserved=True,no_heavy_execution_performed=True,launch_authorization=False))
    review_placeholder=dict(path='ROOT_REPLACE_WITH_INDEPENDENT_REVIEW_PATH',sha256='ROOT_REPLACE_WITH_REVIEW_SHA256')
    release=dict(schema='European_full_VAL_ROOT_execution_release_v1',passed=False,execution_authorized=False,
        source_manifest_ref=manifest_ref,independent_source_review_ref=review_placeholder,plan_ref=c.PLAN_REF,
        owner_storage_authorization_ref=owner_ref,aggregate_directory=resource['aggregate_directory'],
        host_cap_bytes=c.HOST_CAP,reservation_bytes=c.RESERVATION,operating_stop_bytes=c.OPERATING,max_seconds=c.MAX_SECONDS,
        CUDA_VISIBLE_DEVICES='',peer_directories=[],other_project_reserved_bytes=None,
        reservation_scope_excluded_from_other_bytes=True,global_project_ceiling_bytes=c.PROJECT_CAP,
        ROOT_live_global_storage_and_peer_admission_required=True,model_fit_prediction_or_scoring_authorized=False,
        ROOT_status_max_age_seconds=c.ROOT_STATUS_MAX_AGE_SECONDS,growth_refresh_interval_seconds=c.GROWTH_INTERVAL_SECONDS,
        retained_attempts_ref=retained_ref,retained_short_reservation_in_other_bytes=c.RETAINED_SHORT_RESERVATION)
    template_ref=save('ROOT_RELEASE_TEMPLATE_v1.json',release)
    contract_ref=save('ROOT_LAUNCH_CONTRACT_v1.json',dict(schema='European_full_VAL_ROOT_launch_contract_v1',
        source_manifest_ref=manifest_ref,resource_contract=resource,release_template_ref=template_ref,
        expected_independent_review_schema='European_full_VAL_ROOT_runner_review_v1',
        required_review_fields=dict(passed=True,source_manifest_ref=manifest_ref,native_terminal_race_fix_reviewed=True),
        exact_working_directory=str(c.ROOT),runtime_versions_ref=runtime_ref,
        argv=['.runtime/env/bin/python3.11','-B','-m','research.efficiency_20261027.execution.european_full_val_runner.guardian_v1',
            '--source-pins-sha256',manifest_ref['sha256'],'--source-review-path','ROOT_REPLACE_WITH_REVIEW_PATH',
            '--source-review-sha256','ROOT_REPLACE_WITH_REVIEW_SHA256','--execution-release-path','ROOT_REPLACE_WITH_RELEASE_PATH',
            '--execution-release-sha256','ROOT_REPLACE_WITH_RELEASE_SHA256'],
        environment=dict(PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),
        required_ROOT_steps=['Verify exact independent source review and current package versions.',
            'Recompute current campaign commitments, all peer output allowances and retained completed artifacts under1.2TB, preserving5GB admission margin.',
            'Include one conservative16GiB combined allowance for failedshortVALv1/v2 inside other_project_reserved_bytes; do not pass those retained scopes as peers.',
            'Write a separate ROOT execution release from the non-authorizing template with exact current peers and other-project bytes excluding the new128983236608-byte reservation.',
            'Append one --peer-directory argument for every current explicitly admitted ROOT peer; the exact same list is bound in the release.',
            'Only ROOT launches the reviewed guardian. Preserve failed evidence; no retry into the fixed fresh output scope.'],
        no_model_fit_prediction_scoring_authorized=True,launch_performed=False))
    print(json.dumps(dict(source_manifest_ref=manifest_ref,source_refs=len(dependencies),launch_contract_ref=contract_ref,
        release_template_ref=template_ref,owner_authorization_ref=owner_ref)))


if __name__=='__main__':main()
