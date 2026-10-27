"""Seal isolated exact preprocessing integration; no launch or policy change."""
import difflib
import hashlib
import json
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/efficiency_20261027/execution/european_cohort_fit'
BRIDGE=ROOT/'research/efficiency_20261027/fidelity/european_fit_preprocessing_v1'


def ref(path):
    path=ROOT/Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def put(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return ref(path)


def main():
    inherited=json.loads((HERE/'fit_cli_pins_v2.json').read_text())
    refs={row['path']:row for row in inherited['source_refs']}
    for row in refs.values():assert ref(row['path'])==row
    bridge=json.loads((BRIDGE/'source_pins_v1.json').read_text())
    for row in bridge['source_refs']:
        assert ref(row['path'])==row
        assert row['path'] not in refs or refs[row['path']]==row
        refs[row['path']]=row
    for path in [HERE/'fit_cli_pins_v2.json',HERE/'fit_cli_v3.py',HERE/'fit_gate_v3.py',
                 HERE/'test_contracts_v3.py',HERE/'seal_sources_v3.py',HERE/'CONTRACTS_CPU_v3.json',
                 BRIDGE/'source_pins_v1.json',BRIDGE/'CPU_TEST_RESULTS_v1.json',
                 ROOT/'research/efficiency_20261027/fidelity/EUROPEAN_COHORT8_PREPROCESSING_ROUTE_REVIEW_v1.json']:
        value=ref(path);refs[value['path']]=value
    wrapper_ref=put(HERE/'fit_cli_pins_v3.json',dict(schema='European_cohort8_FIT_CLI_sources_v3',
        source_refs=[refs[key] for key in sorted(refs)],
        inherited_wrapper_manifest_ref=ref(HERE/'fit_cli_pins_v2.json'),
        resume_driver_source_manifest_ref=inherited['resume_driver_source_manifest_ref'],
        preprocessing_source_manifest_ref=ref(BRIDGE/'source_pins_v1.json'),
        original_numerical_driver_unchanged=True,original_qualification_reused=True,
        frozen_source_files_changed=False,no_launch_authorization=True))
    diff=''.join(difflib.unified_diff((HERE/'fit_cli_v2.py').read_text().splitlines(True),
        (HERE/'fit_cli_v3.py').read_text().splitlines(True),fromfile='fit_cli_v2.py',tofile='fit_cli_v3.py'))
    diff+=''.join(difflib.unified_diff((HERE/'fit_gate_v2.py').read_text().splitlines(True),
        (HERE/'fit_gate_v3.py').read_text().splitlines(True),fromfile='fit_gate_v2.py',tofile='fit_gate_v3.py'))
    with (HERE/'FIT_V3_PLAINTEXT.diff').open('x') as stream:stream.write(diff)
    plan=json.loads((HERE/'ROOT_LAUNCH_CONTRACT_v2.json').read_text())
    plan.update(schema='European_cohort8_FIT_ROOT_launch_contract_v3',
        worker_module='research.efficiency_20261027.execution.european_cohort_fit.fit_cli_v3',
        wrapper_manifest_ref=wrapper_ref,preprocessing_source_manifest_ref=ref(BRIDGE/'source_pins_v1.json'),
        expected_review_schema='European_cohort8_FIT_CLI_review_v3',
        expected_review_fields=dict(passed=True,wrapper_manifest_ref=wrapper_ref,
            resume_restore_CPU_equivalence_passed=True,preprocessing_actual_route_CPU_equivalence_passed=True,
            original_numerical_driver_unchanged=True),
        reason='Exact successful original model qualification, independent v3 review and current ROOT admission remain required',
        preprocessing_receipt='preprocessing_bindings.json is written before driver.run_fit under the original whole-run allowance',
        prior_full_tree_evidence_domain='American TRAIN000; exact generic sources plus tiny European actual-route parity',
        no_additional_full_GPU_qualification_required_by_preprocessing_bridge=True)
    plan['required_worker_arguments']=[('--wrapper-pins-sha256 '+wrapper_ref['sha256'])
        if argument.startswith('--wrapper-pins-sha256 ') else argument for argument in plan['required_worker_arguments']]
    plan['final_required_receipts'].insert(0,'preprocessing_bindings.json with exact four actual-route bindings')
    contract_ref=put(HERE/'ROOT_LAUNCH_CONTRACT_v3.json',plan)
    result=dict(schema='European_cohort8_FIT_preprocessing_candidate_v3',status='Ready for independent review; no launch',
        wrapper_manifest_ref=wrapper_ref,preprocessing_source_manifest_ref=ref(BRIDGE/'source_pins_v1.json'),
        bridge_CPU_ref=ref(BRIDGE/'CPU_TEST_RESULTS_v1.json'),wrapper_CPU_ref=ref(HERE/'CONTRACTS_CPU_v3.json'),
        launch_contract_ref=contract_ref,plaintext_diff_ref=ref(HERE/'FIT_V3_PLAINTEXT.diff'),
        exact_original_model_qualification_reused=True,scientific_driver_v3_unchanged=True,
        fixed_epochs=30,fixed_updates=90,all_original_TRAIN_inputs_preserved=True,
        no_active_runs_modified=True,no_launches=True)
    print(json.dumps(dict(status_ref=put(HERE/'IMPLEMENTATION_STATUS_v3.json',result),
        wrapper_manifest_ref=wrapper_ref,launch_contract_ref=contract_ref,source_count=len(refs))))


if __name__=='__main__':main()
