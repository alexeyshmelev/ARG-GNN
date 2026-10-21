"""Plain metadata adoption, routing, capacity and storage for the readable queue."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from . import contracts_gpu5_v1 as c
from arggnn.benchmarking import workflow as original
from arggnn.benchmarking import workflow_v4 as lifecycle
from arggnn.benchmarking import workflow_predecessor_v5 as predecessor
from research.reviews.benchmark_repairs_20261019.resources_v9 import resource_accounting_v1 as accounting
from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v1 as recovery
from . import queue_gpu5_v1 as gpu
from research.repairs.benchmark_recovery_20261020.scheduler import queue_adoption_v1 as adoption
from research.repairs.benchmark_recovery_20261020.scheduler.native_continuation_v1 import managed_RAM, remaining_growth

UNSAFE_FACTORIES={
 'arggnn.benchmarking.european_preparation_v9',
 'arggnn.benchmarking.treeloom_preparation_v8',
 'arggnn.benchmarking.empirical_treeloom',
}
UNSAFE_SCIENCE={
 'arggnn.benchmarking.treeloom_fresh',
 'arggnn.benchmarking.treeloom_preparation_v8',
 'arggnn.benchmarking.european_preparation_v9',
 'arggnn.benchmarking.empirical_treeloom',
}


def need(ok,message):
    if not ok:raise ValueError(message)


def route_key(module,function):return module+':'+function


def factory_route(plan,task):
    original_route=route_key(task['factory_module'],task['factory_function'])
    route=plan.get('readable_factory_routes',{}).get(original_route)
    if route is None:return task['factory_module'],task['factory_function']
    return route['module'],route['function']


def scientific_route(plan,call):
    key=route_key(call['module'],call['function'])
    route=plan.get('readable_scientific_routes',{}).get(key)
    if route is None:
        need(call['module'] not in UNSAFE_SCIENCE,'Scientific callable still needs a readable adapter: '+key)
        return call['module'],call['function']
    return route['module'],route['function']


def route_block_reason(plan,task):
    if task['resource']=='external':return None
    archive=adoption.archive_dispatch_reason(plan,task)
    if archive:return archive
    module,function=factory_route(plan,task)
    if module in UNSAFE_FACTORIES:return 'Awaiting readable scientific adapter: '+module
    if task['factory_module']=='arggnn.benchmarking.american_workflow' and task['factory_function']=='prediction_factory':
        if route_key(task['factory_module'],task['factory_function']) not in plan.get('readable_factory_routes',{}):
            return 'Awaiting readable TreeLoom prediction adapter'
    return None


def validate_plan_routes(plan):
    need(plan.get('readable_execution') is True,'Explicit readable successor plan required')
    for key in ('readable_factory_routes','readable_scientific_routes'):
        need(isinstance(plan.get(key,{}),dict),'Readable routes must be explicit mappings')
        for route in plan.get(key,{}).values():
            need(set(route)=={'module','function'} and not route['function'].startswith('_'),
                 'Only explicit named readable callables may be routed')


def validate_native_command(argv):
    need(isinstance(argv,(list,tuple)) and argv and all(isinstance(value,str) for value in argv),'Plain native argv required')
    denied={'-c','-EncodedCommand','-EncodedArguments','-Command','--eval','--command','-e'}
    need(not any(value in denied for value in argv),'Native command uses a hidden interpreter payload')
    need(not any('base64' in value.lower() or 'Invoke-Expression' in value for value in argv),
         'Encoded or expression-based native transport forbidden')


def validate_settlements(plan):
    accounting.validate_static_settlements(plan,c.ROOT)
    required=plan.get('required_policy_remove_approved_closed_storage_recovery_refs',[])
    if required:
        policy=c.check_execution_release(c.file_ref(c.POLICY_PATH))
        approved=policy.get('approved_closed_storage_recovery_refs',[])
        need(all(ref not in approved for ref in required),'ROOT must revoke old standalone archive dispatch before successor writes')


def _hex(value):
    return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')


def validate_retirement(task, entry, retirement, plan, state):
    """Authenticate the new receipt before original local-owner remeasurement."""
    if task.get("factory_module") != 'arggnn.benchmarking.european_retention_v9':
        need(retirement.get("successful_new_scratch_only") is True,
             "Actual retirement completion required")
        return
    owner_ids = task.get("remeasure_succeeded_tasks")
    need(task.get("factory_function") == "factory" and task["resource"] == "cpu"
         and isinstance(owner_ids, list) and len(owner_ids) == 1,
         "One explicit local AP lossless retention owner required")
    owner_id = owner_ids[0]
    need(owner_id.startswith("AP__") and owner_id.endswith("__ap_infer")
         and task["id"] == owner_id + "__lossless_archive_v9"
         and task["dependencies"] == [owner_id],
         "Only the exact AP inference archive route is eligible")
    owners = [item for item in plan["tasks"] if item["id"] == owner_id]
    need(len(owners) == 1 and owners[0]["resource"] != "external"
         and state["tasks"][owner_id]["status"] == "succeeded"
         and not state["tasks"][owner_id].get("external"),
         "Only a succeeded local observed inference scope may be remeasured")
    owner = owners[0]
    scope = c.contained_path(owner["output_directory"])
    output = c.contained_path(task["output_directory"])
    need(not output.is_relative_to(scope) and not scope.is_relative_to(output),
         "Archive output must be separately accounted from the owner")
    need(entry.get("status") == "succeeded" and entry.get("actual_controller_wait") == 0
         and entry.get("owned_process_group_absent") is True,
         "Lossless archive needs its actual successful wait and group drain")
    job = c.read_ref(entry["job_ref"])
    need(job.get("task") == task and job.get("task_fingerprint") == c.fingerprint(task)
         and job.get("call", {}).get("module") == 'arggnn.benchmarking.european_retention_v9'
         and job["call"].get("function") == "run",
         "Actual archive job must bind the exact reviewed task")
    spec = job["call"]["spec"]
    need(spec.get("schema") == 'closed_european_lossless_archive_v1' and spec.get("owner_task_id") == owner_id
         and c.contained_path(spec["owner_scope"]) == scope
         and c.contained_path(spec["output_directory"]) == output / "native",
         "Actual archive job must bind exact owner and separately charged output")
    envelope = c.read_ref(entry["result_ref"])
    receipt_ref = envelope.get("value")
    need(envelope.get("job_ref") == entry["job_ref"]
         and isinstance(receipt_ref, dict)
         and c.contained_path(receipt_ref["path"]) == output / "native/restore_manifest.json",
         "Actual lossless manifest must be the archive worker result")
    need(retirement.get("schema") == 'closed_european_lossless_archive_v1'
         and retirement.get("inference_ref") == spec.get("inference_ref")
         and retirement.get("actual_wait_ref") == spec.get("actual_wait_ref")
         and c.contained_path(retirement["owner_scope"]) == scope
         and retirement.get("owner_task_id") == owner_id
         and c.contained_path(retirement["archive_output_directory"]) == output / "native"
         and retirement.get("archives_inside_current_task_output") is True
         and retirement.get("compressed_content_preserves_every_original_byte") is True
         and retirement.get("final_observed_tree_and_inputs_unchanged") is True
         and retirement.get("raw_path_replacement_only_after_verified_durable_archive") is True,
         "Verified durable lossless retention receipt required")
    from research.reviews.benchmark_repairs_20261019.resources_v9.european_archive_v1 import CANDIDATES
    rows = retirement.get("entries")
    need(isinstance(rows, list) and rows, "Actual verified archive entries required")
    seen = set()
    saved = 0
    for row in rows:
        original = c.contained_path(row["original_path"])
        need(original.is_relative_to(scope / "native")
             and str(original.relative_to(scope / "native")) in CANDIDATES
             and original not in seen,
             "Each archive entry must name one allowlisted original intermediate")
        seen.add(original)
        need(row.get("decompressed_hash_and_size_verified") is True
             and _hex(row.get("original_sha256")) and _hex(row.get("archive_sha256"))
             and type(row.get("original_bytes")) is int and row["original_bytes"] >= 0
             and type(row.get("archive_bytes")) is int and row["archive_bytes"] >= 0
             and type(row.get("credited_bytes")) is int and row["credited_bytes"] >= 0,
             "Verified size, digest and nonnegative archive accounting required")
        if row.get("state") == "archive_committed":
            archive = c.contained_path(row["archive_path"])
            need(archive.is_relative_to(output / "native") and archive.is_file()
                 and not archive.is_symlink() and archive.stat().st_size == row["archive_bytes"]
                 and not original.exists(),
                 "Committed gzip must remain in its charged new output scope")
            before, after = row.get("original_conservative_charge_bytes"), row.get("archive_conservative_charge_bytes")
            need(type(before) is int and type(after) is int
                 and before >= after >= 0 and row["credited_bytes"] == before - after,
                 "Committed archive credit must match its verified charge difference")
        else:
            need(row.get("state") == "original_kept_no_storage_gain"
                 and row["credited_bytes"] == 0 and original.is_file(),
                 "Unchanged original files cannot receive storage credit")
        saved += row["credited_bytes"]
    need(type(retirement.get("observed_saved_bytes")) is int
         and retirement["observed_saved_bytes"] == saved,
         "Lossless archive total must match its per-file verified accounting")
    # The unchanged scheduler performs current scope_bytes() remeasurement next.
    # Per-file estimates do not directly reduce the active storage projection.


def standalone_result(binding):
    if binding.get('kind')=='treeloom_candidate_fit':return adoption.candidate_result(binding)
    if not adoption.drained(binding):return None
    output=c.contained_path(binding['output_directory'])
    wait_path=output/'actual_wait.json'
    if not wait_path.is_file():return None
    wait_ref=c.file_ref(wait_path)
    wait=c.read_ref(wait_ref)
    if wait.get('actual_direct_wait') is not True or wait.get('owned_process_group_absent') is not True:return None
    if wait.get('actual_direct_returncode')!=0 or wait.get('supervisor_error') is not None:
        return dict(status='failed',error='Standalone scientific child failed',actual_wait_ref=wait_ref)
    expected=binding['supervisor_identity']
    try:current=gpu.identity(expected['pid'])
    except (FileNotFoundError,ProcessLookupError):current=None
    if current==expected:return None
    request=c.read_ref(binding['request_ref'])
    launch=c.read_ref(binding['launch_ref'])
    need(launch['request_ref']==binding['request_ref'] and launch['identity']==binding['supervisor_identity'],'Native launch binding changed')
    job=c.read_ref(wait['job_ref'])
    need(job['task']==request['task'] and job['shared_native_request_ref']==binding['request_ref']
         and job['call']==request['call'] and job['task']['id']==binding['task_id'],
         'Standalone exact request/job/science changed')
    result_ref=c.file_ref(output/'result.json')
    envelope=c.read_ref(result_ref)
    need(envelope['job_ref']==wait['job_ref'],'Standalone result envelope changed')
    science=c.read_ref(envelope['value'])
    spec=request['call']['spec']
    need(science.get('actual_optimizer_updates')==spec['expected_updates']
         and science.get('optimizer_unique_step_values')==[spec['expected_updates']]
         and science.get('completed_epochs')==1 and science.get('stage')=='contrastive'
         and science.get('parent_id')==spec['parent_id']
         and science.get('preparation_ref')==spec['preparation_ref']
         and science.get('focal_registry_ref')==spec['focal_registry_ref']
         and science.get('query_ancestry_truth_opened') is False,'Standalone full unchanged budget and provenance required')
    return dict(status='succeeded',value=envelope['value'],external_actual_wait_ref=wait_ref,
        external_source_result_ref=result_ref,external_original_job_ref=wait['job_ref'],
        controller_completion_evidence=dict(kind='scientific_direct_wait_and_exact_supervisor_generation_absence',
            supervisor_identity=expected,observed_current_generation=current,actual_outer_OS_wait_known=False,
            actual_outer_returncode=None,scientific_direct_returncode=0))


def external_result(task):
    if 'standalone_predecessor' in task:return standalone_result(task['standalone_predecessor'])
    if 'workflow_predecessor' in task:return predecessor.external_result(task)
    return original.external_result(task)


def initial_state(plan,reference):
    state=dict(plan_ref=reference,tasks={task['id']:dict(status='pending',external=task['resource']=='external') for task in plan['tasks']})
    if plan.get('adopted_state_snapshot_ref'):
        snapshot=c.read_ref(plan['adopted_state_snapshot_ref'])
        for identifier,fingerprint in plan.get('adopted_completed_task_fingerprints',{}).items():
            task=next(task for task in plan['tasks'] if task['id']==identifier)
            need(c.fingerprint(task)==fingerprint,'Adopted completed task descriptor changed')
            old=snapshot['tasks'][identifier]
            need(old['status']=='succeeded' and 'result_ref' in old,'Only genuine recorded successes may be adopted')
            c.read_ref(old['result_ref'])
            state['tasks'][identifier]=dict(copy.deepcopy(old),adopted_snapshot_ref=plan['adopted_state_snapshot_ref'])
    if plan.get('adopted_archive_state_snapshot_ref'):
        snapshot=c.read_ref(plan['adopted_archive_state_snapshot_ref'])
        for identifier in plan.get('adopted_archive_task_ids',[]):
            old=snapshot['tasks'][identifier]
            need(old['status']=='succeeded' and old.get('actual_controller_wait')==0 and old.get('owned_process_group_absent') is True,
                 'Only actual completed standalone archives may be adopted')
            state['tasks'][identifier]=dict(copy.deepcopy(old),adopted_archive_snapshot_ref=plan['adopted_archive_state_snapshot_ref'])
    adoption.refresh_archives(plan,state)
    return state


def _standalone_rows(plan):
    return [task['standalone_predecessor'] for task in plan['tasks'] if 'standalone_predecessor' in task]


def external_resources(plan):
    cpu,rss=original.external_resources(plan)
    extra_cpu,extra_rss=predecessor.external_resources(plan)
    cpu+=extra_cpu;rss+=extra_rss
    for binding in _standalone_rows(plan):
        result=standalone_result(binding)
        if result is None:rss+=binding['max_rss_bytes']
    archive=adoption.archive_state(plan)
    if archive:
        tasks={task['id']:task for task in plan['tasks']}
        for identifier,entry in archive['tasks'].items():
            if entry['status']=='running':cpu+=1;rss+=tasks[identifier]['max_rss_bytes']
    return cpu,rss


def storage_projection(plan,state,new_cap,base_projection):
    total=base_projection(plan,state,new_cap)+predecessor.storage_reservation_bytes(plan)
    if plan.get('closed_storage_recovery',{}).get('recovery_helper_module')=='arggnn.benchmarking.storage_recovery_20261020_v2':
        from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v2
        total-=closed_archives_v2.completed_storage_reduction(plan,state,c.ROOT)
    else:
        total-=recovery.completed_storage_reduction(plan,state,c.ROOT)
    for binding in _standalone_rows(plan):
        result=standalone_result(binding)
        output=c.contained_path(binding['output_directory'])
        if not adoption.drained(binding):
            total+=binding['max_output_bytes']
        else:
            retained=lifecycle.scope_bytes(output)
            need(retained<=binding['max_output_bytes'],'Standalone retained output exceeds its cap')
            total+=retained
    return total


def managed_ram_rows(plan,state):
    rows=managed_RAM(plan,state)
    for binding in _standalone_rows(plan):
        if adoption.drained(binding):continue
        sample=dict(complete=False,enforcement_PSS_bytes=0)
        try:
            receipt=dict(child_identity=adoption.scientific_root(binding))
            from research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1 import repaired_owned_group_class
            tracker=repaired_owned_group_class(lifecycle)(receipt['child_identity']['pid'])
            need(all(tracker.root[key]==receipt['child_identity'][key] for key in ('pid','start_ticks','pgid','session')),
                 'Standalone managed RAM root changed')
            tracker.measure();sample=tracker.latest
        except (OSError,ValueError,RuntimeError,KeyError):pass
        rows.append(dict(full_cap_bytes=binding['max_rss_bytes'],remaining_growth_bytes=remaining_growth(binding['max_rss_bytes'],sample)))
    return rows


def ram_allows(plan,state,task):
    rows=managed_ram_rows(plan,state)
    full=sum(row['full_cap_bytes'] for row in rows)
    growth=sum(row['remaining_growth_bytes'] for row in rows)
    new=task['max_rss_bytes']
    return (full+new<=plan['max_parallel_rss_bytes'] and
            accounting.kernel_available_ram_bytes()-growth>=new+plan['host_ram_margin_bytes'])


@contextmanager
def admission_lane():
    context=gpu.capacity_lock(blocking=False)
    try:context.__enter__()
    except BlockingIOError:
        yield False
        return
    try:yield True
    finally:context.__exit__(None,None,None)


def archive_controller_live(plan):
    owner=plan.get('standalone_archive_owner')
    return owner is not None and not adoption.generation_absent(owner['supervisor_identity'])


def gpu_dispatch_needed(task):
    """CPU observation preparation reserves RAM first and acquires GPU later."""
    return task['resource']=='gpu0'
