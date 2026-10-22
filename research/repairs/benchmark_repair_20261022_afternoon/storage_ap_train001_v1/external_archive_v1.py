"""One supplementary AP TRAIN001 archive; frozen recovery receipts stay intact.

The unchanged v2 codec owns compression, decompression verification, durable
sidecars, and raw replacement. This adapter authenticates one external producer
and credits its frozen predecessor charge only after a new successful receipt.
"""
from pathlib import Path
import json

from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v2 as closed

ROOT = closed.ROOT
SOURCE = 'research/repairs/benchmark_repair_20261022_afternoon/storage_ap_train001_v1/external_archive_v1.py'
SHIM = 'src/arggnn/benchmarking/storage_recovery_ap_train001_20261022.py'
MODULE = 'arggnn.benchmarking.storage_recovery_ap_train001_20261022'
FIELD = 'supplemental_external_ap_closed_recovery'
SCHEMA = 'external_ap_train001_closed_recovery_20261022_v1'
OWNER = 'AP__ancestralpaths_european_train_001__ap_infer'
TASK = OWNER + '__closed_lossless_archive_20261022_v1'
TARGET = 'closed_ap_train001_20261022_v1'
SCOPE = 'experiments/results/benchmark_campaign_20261018_workflow/release_v9/inherited/' + OWNER
RETAINED = 32458678324
CANDIDATES = ('inputs/input.annot', 'inputs/input.haps', 'inputs/input.vcf',
              'post_size.pairwise.bin', 'post_size.trees', 'pre_size.anc', 'pre_size.mut')
need, path, read, reference, record = closed.need, closed.path, closed.read, closed.reference, closed.record


def _load(inventory_ref, target_key, repo=ROOT):
    need(target_key == TARGET, 'Only the named AP TRAIN001 target is eligible')
    inventory, target = closed.load_target(inventory_ref, target_key, repo)
    need(len(inventory['targets']) == 1 and not inventory['preserved_failed_attempts']
         and not target['prior_archives'], 'One fresh supplementary target and no historical relabelling')
    original = target['original']
    binding = inventory['external_predecessor_binding']
    need(original['scope'] == SCOPE and original['key'] == TARGET
         and original['before_scope_bytes'] == target['start_scope_bytes'] == RETAINED
         and original['accounting'] == {'kind': 'workflow_predecessor', 'source_task_id': OWNER}
         and binding['source_task_id'] == OWNER
         and path(repo, binding['output_directory']) == path(repo, SCOPE)
         and binding['original_job_ref'] == original['producer']['job_ref']
         and binding['source_plan_ref'] == original['producer']['source_plan_ref'],
         'Exact external producer, original charged scope and original lineage required')
    proof = original['producer']
    wait = read(repo, proof['wait_ref'], 2**20)
    need(proof['task_id'] == OWNER and proof['entry']['status'] == 'succeeded'
         and type(proof['entry']['actual_controller_wait']) is int
         and proof['entry']['actual_controller_wait'] == 0
         and proof['entry']['owned_process_group_absent'] is True
         and proof['entry']['retained_bytes'] == RETAINED
         and wait.get('actual_direct_wait') is True
         and type(wait.get('actual_direct_returncode')) is int
         and wait['actual_direct_returncode'] == 0
         and wait.get('owned_process_group_absent') is True
         and wait.get('supervisor_error') is None,
         'Original genuinely successful direct wait and group drain required')
    names = sorted(str(path(repo, row['path']).relative_to(path(repo, SCOPE) / 'native'))
                   for row in original['candidates'])
    need(names == list(CANDIDATES), 'Only the seven frozen codec allowlisted intermediates')
    protected = {row['path'] for row in original['protected_files']}
    need(all(SCOPE + '/' + name in protected for name in (
        'actual_wait.json', 'result.json', 'native/inference.json',
        'native/post_size.observed.trees', 'native/observed_sample_manifest.json',
        'native/commands.json', 'native/02_relate_initial.log',
        'native/03_five_population_size_iterations.log')),
        'Protected observed science, logs and original completion evidence must be explicit')
    return inventory, target


def _authority(inventory, repo=ROOT):
    policy = json.loads(path(repo, 'configs/benchmarks/execution_policy.json').read_text())
    source_ref = inventory['external_predecessor_binding']['source_plan_ref']
    need(policy.get('resource_continuation_drain_ref') == source_ref,
         'Original v9 future dispatch must remain explicitly drained')


def validate_plan(plan, repo=ROOT):
    rows = [task for task in plan['tasks'] if task.get('factory_module') == MODULE]
    policy = plan.get(FIELD)
    if policy is None:
        need(not rows, 'Supplementary factory needs its exact zero-credit plan binding')
        return None
    need(set(policy) == {'schema', 'task_id', 'inventory_ref', 'initial_credit_bytes'}
         and policy['schema'] == SCHEMA and policy['task_id'] == TASK
         and policy['initial_credit_bytes'] == 0 and len(rows) == 1,
         'One exact zero-initial-credit supplementary recovery binding required')
    inventory, target = _load(policy['inventory_ref'], TARGET, repo)
    task = rows[0]
    original = target['original']
    owners = [item for item in plan['tasks'] if item['id'] == OWNER]
    bindings = [item for item in plan['workflow_external_predecessors'] if item['source_task_id'] == OWNER]
    need(len(owners) == len(bindings) == 1
         and owners[0]['resource'] == 'external'
         and owners[0]['workflow_predecessor'] == bindings[0] == inventory['external_predecessor_binding'],
         'Original external ownership and immutable predecessor accounting must stay exact')
    need(task['id'] == TASK and task['factory_function'] == 'factory'
         and task['resource'] == 'cpu' and task['max_rss_bytes'] == 2**30
         and task['max_output_bytes'] == original['max_output_bytes']
         and task['max_seconds'] == 86400 and task['dependencies'] == [OWNER]
         and task['factory_arguments'] == {'inventory_ref': policy['inventory_ref'], 'target_key': TARGET}
         and not task.get('remeasure_succeeded_tasks'),
         'Exact supplementary CPU task; no historical or external state remeasurement')
    output = path(repo, task['output_directory'])
    owner = path(repo, SCOPE)
    need(not output.is_relative_to(owner) and not owner.is_relative_to(output),
         'Archive output must be separately charged outside its original owner')
    primary = plan.get('closed_storage_recovery')
    if primary:
        primary_inventory = read(repo, primary['inventory_ref'])
        need(all(path(repo, row['original']['scope']) != owner for row in primary_inventory['targets']),
             'An original scope may receive credit in only one recovery group')
    return inventory, target, task


def factory(*, inventory_ref, target_key, output_directory):
    inventory, target = _load(inventory_ref, target_key)
    _authority(inventory)
    call = closed.factory(inventory_ref=inventory_ref, target_key=target_key,
                          output_directory=output_directory)
    call['module'] = MODULE
    additions = [reference(ROOT, SOURCE), reference(ROOT, SHIM), inventory_ref,
                 inventory['original_inventory_ref']]
    for item in additions:
        if item not in call['source_refs']:
            call['source_refs'].append(item)
    return call


def run(spec, *, resource_check, execution_check=None, bounded_runner=None):
    need(set(spec) == {'schema', 'inventory_ref', 'target_key', 'output_directory'}
         and spec['schema'] == closed.SCHEMA, 'Exact unchanged v2 codec specification required')
    inventory, target = _load(spec['inventory_ref'], spec['target_key'])
    def authority():
        if execution_check is not None:
            execution_check()
        _authority(inventory)
    authority()
    return closed.run(spec, resource_check=resource_check,
                      execution_check=authority, bounded_runner=bounded_runner)


def completed_storage_reduction(plan, state, repo=ROOT):
    validated = validate_plan(plan, repo)
    if validated is None:
        return 0
    inventory, target, task = validated
    entry = state['tasks'][task['id']]
    if entry['status'] != 'succeeded':
        return 0
    owner_entry = state['tasks'][OWNER]
    need(owner_entry['status'] == 'succeeded' and owner_entry.get('external') is True
         and 'reservation_bytes' not in owner_entry and 'retained_bytes' not in owner_entry,
         'External original remains charged by its unchanged predecessor exactly once')
    need(type(entry.get('actual_controller_wait')) is int
         and entry['actual_controller_wait'] == 0 and entry.get('owned_process_group_absent') is True,
         'New successful controller wait and owned-group drain required before any credit')
    job = read(repo, entry['job_ref'])
    output = path(repo, task['output_directory'])
    native = output / 'native'
    expected_spec = dict(schema=closed.SCHEMA,
                         inventory_ref=plan[FIELD]['inventory_ref'], target_key=TARGET,
                         output_directory=str(native))
    need(job.get('plan_ref') == state['plan_ref']
         and job['task'] == task and job['call']['module'] == MODULE
         and job['call']['function'] == 'run' and job['call']['spec'] == expected_spec,
         'New job must bind the exact reviewed task and unchanged codec specification')
    wait = read(repo, reference(repo, output / 'actual_wait.json'), 2**20)
    need(wait.get('job_ref') == entry['job_ref'] and wait.get('actual_direct_wait') is True
         and type(wait.get('actual_direct_returncode')) is int
         and wait['actual_direct_returncode'] == 0
         and wait.get('owned_process_group_absent') is True and wait.get('supervisor_error') is None,
         'New genuine successful direct wait and actual drain required')
    envelope = read(repo, entry['result_ref'])
    need(envelope.get('job_ref') == entry['job_ref']
         and path(repo, envelope['value']['path']) == native / 'restore_manifest.json',
         'Exact new restoration manifest result envelope required')
    manifest = read(repo, envelope['value'])
    original = target['original']
    need(manifest['schema'] == closed.SCHEMA
         and manifest['inventory_ref'] == plan[FIELD]['inventory_ref']
         and manifest['target_key'] == TARGET and manifest['owner_scope'] == SCOPE
         and path(repo, manifest['archive_output_directory']) == native
         and manifest['original_wait_ref'] == original['producer']['wait_ref']
         and manifest['original_status_preserved'] == 'succeeded'
         and manifest['every_original_byte_restorable'] is True
         and manifest['protected_science_and_evidence_unchanged'] is True
         and manifest['raw_replacement_after_verified_durable_archive'] is True
         and manifest['archives_in_new_or_exact_separately_charged_failed_scopes'] is True
         and manifest['preserved_failed_attempts'] == []
         and manifest['old_failure_relabelled'] is False,
         'Complete new lossless restoration proof; original completion remains unchanged')
    originals = {row['path']: row for row in original['candidates']}
    need(len(manifest['entries']) == len(originals), 'Exactly one terminal receipt per original')
    seen = set()
    for row in manifest['entries']:
        source = path(repo, row['original_path'])
        key = str(source.relative_to(Path(repo).resolve()))
        need(key in originals and key not in seen
             and row.get('decompressed_hash_and_size_verified') is True
             and row['original_bytes'] == originals[key]['identity'][2],
             'Exact original bytes and fresh full decompression verification required')
        seen.add(key)
        if row['state'] == 'archive_committed':
            archive = path(repo, row['archive_path'])
            need(not source.exists() and archive.is_relative_to(native)
                 and record(archive)['identity'] == row['archive_identity']
                 and archive.stat().st_size == row['archive_bytes'],
                 'Current exact durable gzip inside the separately charged new output required')
        else:
            need(row['state'] == 'original_kept_no_storage_gain' and row['credited_bytes'] == 0
                 and record(source)['identity'] == originals[key]['identity'],
                 'Unchanged originals receive no speculative credit')
    for row in original['protected_files']:
        need(record(path(repo, row['path']))['identity'] == row['identity'],
             'Every protected original file must remain exact')
    current = closed.old.scope_bytes(path(repo, SCOPE))
    need(manifest['before_scope_bytes'] == manifest['repair_start_scope_bytes'] == RETAINED
         and manifest['after_scope_bytes'] == current and 0 <= current <= RETAINED
         and manifest['original_scope_reduction_bytes'] == RETAINED - current,
         'Only the exact current original-scope reduction can offset the frozen predecessor charge')
    retained = closed.old.scope_bytes(output)
    need(type(entry.get('retained_bytes')) is int and entry['retained_bytes'] == retained
         and retained <= task['max_output_bytes'],
         'The complete new archive output must remain separately charged at its measured size')
    return RETAINED - current
