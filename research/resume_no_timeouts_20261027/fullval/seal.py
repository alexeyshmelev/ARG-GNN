"""Seal prepared successor source and a non-authorizing ROOT launch contract."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
from . import contracts as c

HERE=Path(__file__).resolve().parent
OLD=c.ROOT/'research/efficiency_20261027/execution/european_full_val_runner'

def save(name,value):
    path=HERE/name
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return c.ref(path)

def function(path,name):
    return next(ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node,ast.FunctionDef) and node.name==name)

def main():
    assert Path.cwd()==c.ROOT
    assert not c.AGGREGATE.exists()
    tests=json.loads((HERE/'CPU_TESTS.json').read_text())
    assert tests['passed'] and tests['check_count']>=16
    for reference in tests['source_refs']:assert c.ref(reference['path'])==reference
    transition=json.loads((HERE/'TRANSITION_TESTS.json').read_text())
    assert transition['passed'] and transition['check_count']>=6
    for reference in transition['source_refs']:assert c.ref(reference['path'])==reference
    assert function(OLD/'worker_v2.py','run_stages')==function(HERE/'worker.py','run_stages')
    predecessor=c.read(c.ref(OLD/'source_pins_v2.json'))
    for reference in predecessor['source_refs']:assert c.ref(reference['path'])==reference
    c.validate_plan()
    c.native_bindings()
    retained_ref,retained_bytes=c.retained_attempts()
    differences=[]
    for name in ('aggregate_guard','native_supervisor','contracts','worker','guardian'):
        differences.extend(difflib.unified_diff((OLD/(name+'_v2.py')).read_text().splitlines(True),
            (HERE/(name+'.py')).read_text().splitlines(True),
            fromfile=str((OLD/(name+'_v2.py')).relative_to(c.ROOT)),
            tofile=str((HERE/(name+'.py')).relative_to(c.ROOT))))
    (HERE/'CHANGES.diff').write_text(''.join(differences))
    resource=dict(predecessor['resource_contract'])
    resource.update(aggregate_directory=str(c.AGGREGATE.relative_to(c.ROOT)),
        max_seconds=None,generation_max_seconds=None,native_max_seconds=None,native_step_max_seconds=None,
        elapsed_time_cutoffs_enabled=False,
        retained_attempts_reservation_in_other_bytes=c.RETAINED_ATTEMPTS_RESERVATION,
        retained_attempts_charged_bytes_at_seal=retained_bytes)
    resource.pop('retained_short_reservation_in_other_bytes',None)
    resource.pop('retained_short_charged_bytes_at_seal',None)
    own_names=('aggregate_guard.py','native_supervisor.py','contracts.py','worker.py','guardian.py',
        'test_cpu.py','CPU_TESTS.json','CPU_TESTS.log','test_transition.py','TRANSITION_TESTS.json','TRANSITION_TESTS.log','DIAGNOSIS.md','RETAINED_ATTEMPTS.json','CHANGES.diff','seal.py')
    references={row['path']:row for row in predecessor['source_refs']}
    for path in [OLD/'source_pins_v2.json',*[HERE/name for name in own_names]]:
        reference=c.ref(path);references[reference['path']]=reference
    manifest=dict(schema='European_full_VAL_ROOT_runner_sources_v1',successor_revision=4,
        predecessor_manifest_ref=c.ref(OLD/'source_pins_v2.json'),source_refs=list(references.values()),
        original_registration_ref=c.PLAN_REF,owner_storage_authorization_ref=c.owner_authorization(),
        original_scientific_algorithms_unchanged=True,scientific_stage_AST_unchanged=True,
        simulation_checkpoint_reusable=False,predecessor_evidence_preserved=True,
        original_parent_dates_parameters_and_seeds_preserved=True,
        elapsed_time_cutoffs_enabled=False,liveness_resource_and_owned_process_safeguards_retained=True,
        cause='Reproduced status.tmp atomic-rename race; exact historical filename unavailable in repr-only predecessor error.',
        fix='Account the canonical ordinary status.json replacement only for vanished root status.tmp; other errors fail.',
        tests_ref=c.ref(HERE/'CPU_TESTS.json'),transition_tests_ref=c.ref(HERE/'TRANSITION_TESTS.json'),
        campaign_transition_retry_seconds=60,resource_contract=resource,
        retained_attempts_ref=retained_ref,launch_authorization=False,no_heavy_execution_performed=True)
    manifest_ref=save('source_pins.json',manifest)
    release=json.loads((OLD/'ROOT_RELEASE_TEMPLATE_v2.json').read_text())
    release.update(source_manifest_ref=manifest_ref,max_seconds=None,
        aggregate_directory=str(c.AGGREGATE.relative_to(c.ROOT)),
        retained_attempts_ref=retained_ref,
        retained_attempts_reservation_in_other_bytes=c.RETAINED_ATTEMPTS_RESERVATION)
    release.pop('retained_short_reservation_in_other_bytes',None)
    release_ref=save('ROOT_RELEASE_TEMPLATE.json',release)
    contract=json.loads((OLD/'ROOT_LAUNCH_CONTRACT_v2.json').read_text())
    contract.update(source_manifest_ref=manifest_ref,resource_contract=resource,release_template_ref=release_ref,
        checkpoint_reuse=dict(predecessor_generation_empty=True,reusable_scientific_checkpoint=False,
            action='Repeat exact original preregistered parent and seeds in fresh successor scope; preserve predecessor.'),
        current_campaign_state_path='Required existing helper binding: .runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json',
        downstream_evaluator_scope_rebinding_required=True)
    contract['argv'][3]='research.resume_no_timeouts_20261027.fullval.guardian'
    contract['argv'][5]=manifest_ref['sha256']
    contract['required_review_fields'].update(source_manifest_ref=manifest_ref,
        atomic_status_race_fix_reviewed=True,elapsed_time_cutoffs_removed_reviewed=True,
        campaign_transition_retry_reviewed=True)
    contract['required_ROOT_steps']=[
        'Review CHANGES.diff, CPU_TESTS.json and DIAGNOSIS.md and write the exact new source review with required fields.',
        'Verify the queue state path still matches the helper binding; if queue state moves, bind the successor helper before launching.',
        'Recompute current campaign commitments, all peer output allowances and retained artifacts under 1.2TB, preserving 5GB admission margin.',
        'Count both failed shortVAL scopes and failed fullVAL v1 once in other-project commitments with a combined 16GiB + 128MiB allowance; do not pass these as active peers.',
        'Create a separate ROOT execution release from the non-authorizing template with current peers and other-project bytes excluding the new 128983236608-byte reservation.',
        'Append one --peer-directory argument for every exact current peer and bind the same list in the release.',
        'Only ROOT launches the reviewed guardian once into the fixed fresh scope; preserve all predecessor evidence.',
        'Bind the downstream evaluator to successor observed artifacts explicitly before evaluation; original scientific protocol stays pinned.'
    ]
    contract_ref=save('ROOT_LAUNCH_CONTRACT.json',contract)
    print(json.dumps(dict(source_manifest_ref=manifest_ref,launch_contract_ref=contract_ref,
        tests_ref=c.ref(HERE/'CPU_TESTS.json'),transition_tests_ref=c.ref(HERE/'TRANSITION_TESTS.json'),source_refs=len(references),retained_bytes=retained_bytes,
        launch_performed=False)))

if __name__=='__main__':main()
