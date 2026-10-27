"""Seal a narrow monitor-close successor and preserve the complete v1 closure."""
import ast
import copy
import json
from pathlib import Path
from . import contracts_v2 as c

HERE = Path(__file__).parent


def save(name, value):
    path = HERE/name
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return c.ref(path)


def main():
    predecessor_ref = c.ref(HERE/'source_pins_v1.json')
    assert predecessor_ref['sha256'] == '4587125200ec09562004916cb08340f1e638ce2557181e9b6f524d867edfa6ec'
    predecessor = c.read(predecessor_ref)
    dependencies = {}
    for item in predecessor['source_refs']:
        assert c.ref(item['path']) == item
        assert item['path'] not in dependencies
        dependencies[item['path']] = item
    assert len(dependencies) == 279
    for name, field, count in (('CONTRACTS_CPU_v2.json', 'check_count', 20),
                              ('NATIVE_BINDING_CPU_v2.json', 'actual_tiny_shell_checks_passed', 3),
                              ('MONITOR_CLOSE_CPU_v2.json', 'check_count', 6)):
        receipt = c.read(c.ref(HERE/name))
        assert receipt['passed'] is True and receipt[field] == count
        for item in receipt['source_refs']:
            assert c.ref(item['path']) == item
    modules = ('aggregate_guard', 'contracts', 'worker', 'guardian', 'native_supervisor',
               'test_contracts', 'test_native_binding', 'test_monitor_close')
    sources = tuple(name+'_v2.py' for name in modules) + ('prepare_successor_v2.py', 'seal_sources_v2.py')
    for name in sources:
        ast.parse((HERE/name).read_text(), filename=name)
    from arggnn.benchmarking.european_recipes import load_recipe, material_blockers
    frozen = c.validate_plan()
    canonical = load_recipe(frozen['recipe']['id'])
    fields = ('parameters', 'generator', 'evidence', 'source_refs')
    assert {key:frozen['recipe'][key] for key in fields} == {key:canonical[key] for key in fields}
    assert material_blockers(frozen['recipe']) == []
    baseline_review_ref = save('INDEPENDENT_V1_SOURCE_REVIEW.json', dict(
        schema='European_full_VAL_ROOT_runner_review_v1', passed=False,
        source_manifest_ref=predecessor_ref, authenticated_source_refs=279,
        native_terminal_race_fix_reviewed=True,
        independent_contract_ref=c.ref(HERE/'INDEPENDENT_CONTRACTS_CPU_v1.json'),
        independent_native_binding_ref=c.ref(HERE/'INDEPENDENT_NATIVE_BINDING_CPU_v1.json'),
        blocker_ref=c.ref(HERE.parent/'FULL_VAL_V1_GUARD_CLOSE_REVIEW.json'),
        original_scientific_imports_and_full200Mb_seeds_preserved=True,
        original_five_native_commands_and_five_size_iterations_preserved=True,
        full10000_label_free_validation_rows_and_original_sampling=True,
        exact_strict_recipe_metadata_equal=True, material_blockers=[],
        finite_stage_aggregate_host_wall_and_ROOT_binding_reviewed=True,
        canonical_paths_temporary_cache_and_truth_charged=True,
        only_required_correction='Propagate late monitor failure after final close join.',
        review_scope='Independent full v1 source and bounded lifecycle review; v2 correction requires Fidelity independent delta review.',
        scientific_native_model_GPU_executions=0, real_scientific_payloads_opened=False))
    extra = [predecessor_ref, baseline_review_ref,
        c.ref(HERE.parent/'FULL_VAL_V1_GUARD_CLOSE_REVIEW.json'),
        c.ref(HERE.parent/'review_fullval_guard_v1.py')]
    extra.extend(c.ref(HERE/name) for name in (*sources, 'MONITOR_CLOSE_PLAINTEXT_v2.diff',
        'CONTRACTS_CPU_v2.json', 'NATIVE_BINDING_CPU_v2.json', 'MONITOR_CLOSE_CPU_v2.json',
        'INDEPENDENT_CONTRACTS_CPU_v1.json', 'INDEPENDENT_NATIVE_BINDING_CPU_v1.json'))
    for item in extra:
        assert c.ref(item['path']) == item
        assert item['path'] not in dependencies or dependencies[item['path']] == item
        dependencies[item['path']] = item
    manifest = copy.deepcopy(predecessor)
    manifest.update(source_refs=[dependencies[key] for key in sorted(dependencies)],
        successor_revision=2, predecessor_manifest_ref=predecessor_ref,
        independent_predecessor_review_ref=baseline_review_ref,
        monitor_close_fix='Join monitor, propagate its stored final error, always release lease.',
        monitor_close_regression_ref=c.ref(HERE/'MONITOR_CLOSE_CPU_v2.json'),
        predecessor_evidence_preserved=True)
    manifest_ref = save('source_pins_v2.json', manifest)
    template = c.read(c.ref(HERE/'ROOT_RELEASE_TEMPLATE_v1.json'))
    assert template['passed'] is False and template['execution_authorized'] is False
    template['source_manifest_ref'] = manifest_ref
    template_ref = save('ROOT_RELEASE_TEMPLATE_v2.json', template)
    launch = c.read(c.ref(HERE/'ROOT_LAUNCH_CONTRACT_v1.json'))
    launch['source_manifest_ref'] = manifest_ref
    launch['release_template_ref'] = template_ref
    launch['required_review_fields']['source_manifest_ref'] = manifest_ref
    assert launch['argv'][3].endswith('.guardian_v1')
    launch['argv'][3] = launch['argv'][3].replace('.guardian_v1', '.guardian_v2')
    assert launch['argv'][4] == '--source-pins-sha256'
    launch['argv'][5] = manifest_ref['sha256']
    launch_ref = save('ROOT_LAUNCH_CONTRACT_v2.json', launch)
    print(json.dumps(dict(source_manifest_ref=manifest_ref, source_refs=len(dependencies),
        launch_contract_ref=launch_ref, release_template_ref=template_ref,
        independent_predecessor_review_ref=baseline_review_ref, no_launch=True)))

if __name__ == '__main__':
    main()
