"""Explicit RAM-only authority for unchanged frozen ARGMix stage functions.

The original 64 GiB admission remains immutable storage provenance. A separately
pinned root amendment supersedes only CPU family RSS authority. No model,
observation, label, window, minibatch, ordering, cache or scoring function changes.
"""
import argparse
import json
from pathlib import Path
import psutil
from arggnn.benchmarking.paper_campaign import runtime as records, processes
from research.joint_evaluation_20261029.argmix import pipeline, runtime
from research.joint_evaluation_20261029.argmix.common import require

ORIGINAL_RAM_CAP = 64 * 2**30
EFFECTIVE_RAM_CAP = 384 * 2**30
MINIMUM_HOST_BYTES = EFFECTIVE_RAM_CAP + 64 * 2**30
STAGES = {
    'prepare-cache': pipeline.prepare_cache,
    'train': pipeline.train,
    'predict': pipeline.predict,
    'barrier': pipeline.prediction_barrier,
    'score': pipeline.score,
    'validation-score': pipeline.validation_score,
}


def measured_bound(profile):
    require(profile['passed'] is True and profile['heldout_inputs_or_truth_opened'] is False
            and profile['GPU_initialized'] is False and profile['cache_files_written'] == 0,
            'Read-only successful TRAIN source profile required')
    rows = profile['observations']
    require(any(row['stage'] == 'largest_TRAIN_source_loaded' for row in rows)
            and any(row['stage'] == 'second_large_parent_replaced' for row in rows),
            'Both cold source and source-replacement transient were measured')
    peak = max(row['peak_rss_bytes'] for row in rows)
    return peak, 17 * peak


def validate_amendment(reference, freeze, admission):
    amendment = records.read(reference)
    require(amendment['schema'] == 'joint_argmix_cpu_ram_amendment_v1'
            and amendment['root_authorized'] is True
            and amendment['original_science_unchanged'] is True,
            'Pinned root-authorized RAM-only amendment required')
    require(amendment['prior_owned_RAM_max_bytes'] == ORIGINAL_RAM_CAP
            and amendment['owned_RAM_max_bytes'] == EFFECTIVE_RAM_CAP
            and amendment['minimum_host_available_bytes'] == MINIMUM_HOST_BYTES,
            'Exact reviewed CPU RAM amendment and host headroom')
    require(amendment['workers'] == 16 and amendment['pending_window_jobs'] == 32,
            'Original sixteen workers and ordered prefetch remain unchanged')
    freeze_ref = records.ref(freeze)
    require(amendment['original_freeze_ref'] == freeze_ref
            and amendment['original_admission_ref'] == records.ref(admission),
            'Exact frozen science and historical storage admission')
    seal, protocol = runtime.frozen(freeze)
    require(amendment['original_protocol_ref'] == seal['protocol_ref'] and protocol['workers'] == 16,
            'Exact original protocol and worker count')
    historical = runtime.cache_admission(admission, protocol)
    require(runtime.RAM_CAP == ORIGINAL_RAM_CAP
            and historical['owned_RAM_max_bytes'] == ORIGINAL_RAM_CAP,
            'Original admission remains unchanged historical storage provenance')
    require(amendment['storage_and_GPU_limits_unchanged'] is True
            and amendment['cache_max_bytes'] == historical['cache_max_bytes']
            and amendment['results_max_bytes'] == historical['results_max_bytes']
            and amendment['owner_storage_ceiling_bytes'] == historical['owner_ceiling_bytes'],
            'Storage and GPU permissions cannot change through a CPU RAM amendment')
    require(amendment['worker_source_ref'] == records.ref(__file__), 'Exact reviewed wrapper source')
    for source in amendment['supporting_source_refs']:
        require(records.ref(source['path']) == source, 'Reviewed supporting source changed')
    profile = records.read(amendment['memory_profile_ref'])
    require(profile['original_freeze_sha256'] == freeze_ref['sha256'], 'Profile used original frozen science')
    peak, bound = measured_bound(profile)
    require(amendment['single_process_measured_peak_bytes'] == peak
            and amendment['conservative_17_process_peak_bytes'] == bound
            and bound < EFFECTIVE_RAM_CAP, 'Measured cold/replacement profile must fit effective bound')
    failed = records.read(amendment['failed_cache_wait_ref'])
    require(failed['task_id'] == 'joint_argmix.train_cache' and failed['actual_wait'] is True
            and failed['actual_returncode'] == -15 and failed['owned_process_group_absent'] is True
            and failed['failure'] == 'Registered owned64GiB RAM limit exceeded'
            and failed['freeze_ref'] == freeze_ref, 'Preserved genuine failed cache attempt required')
    require_old_attempt_drained(failed)
    return amendment, protocol, freeze_ref


def require_old_attempt_drained(failed):
    stat = Path('/proc') / str(failed['pid']) / 'stat'
    try:
        fields = stat.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        # The former leader may be absent while a descendant still owns its session.
        require(not processes.session_members(failed['pid']), 'Failed cache session still owns processes')
        return
    require(fields[19] != str(failed['process_start_ticks']),
            'Failed cache original process identity is still present')
    # A new process with the same PID cannot be the old waited owner. Do not
    # inspect or adopt an unrelated new session solely because its number matches.


def effective_check_ram():
    """Same original family-RSS sum, with explicit amended authority and message."""
    process = psutil.Process()
    total = 0
    for child in [process] + process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    require(total <= EFFECTIVE_RAM_CAP,
            'Owned parent/worker RSS exceeds explicitly amended384GiB CPU cap; no truncation or retry')
    return total


def apply_ram_authority():
    require(pipeline.RAM_CAP == ORIGINAL_RAM_CAP, 'No prior unreviewed RAM override')
    pipeline.RAM_CAP = EFFECTIVE_RAM_CAP
    pipeline.check_ram = effective_check_ram


def dispatch(args, protocol, freeze_ref):
    if args.stage == 'qualify-gpu':
        # Import after the explicit guard assignment, so its imported check_ram
        # uses the same amended guard. All qualification science is unchanged.
        from research.joint_evaluation_20261029.argmix.gpu_qualification import qualify
        qualify(args, protocol, freeze_ref)
    else:
        STAGES[args.stage](args, protocol, freeze_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=list(STAGES) + ['qualify-gpu'])
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--gpu', type=int, choices=[3, 4, 5])
    parser.add_argument('--parent')
    parser.add_argument('--admission', default=str(records.ROOT / 'research/joint_evaluation_20261029/root/argmix_joint_production_admission_v1.json'))
    parser.add_argument('--resource-amendment', required=True)
    parser.add_argument('--resource-amendment-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    reference = {'path': str(records.path(args.resource_amendment).relative_to(records.ROOT)),
                 'sha256': args.resource_amendment_sha256}
    amendment, protocol, freeze_ref = validate_amendment(reference, args.freeze, args.admission)
    require(args.stage not in ('train', 'predict', 'qualify-gpu') or args.gpu is not None,
            'Assigned physical GPU required')
    require(args.stage != 'predict' or args.parent is not None, 'One exact held-out parent required')
    if args.check_only:
        print(json.dumps({'valid': True, 'stage': args.stage, 'resource_amendment_ref': reference,
                          'effective_owned_RAM_max_bytes': EFFECTIVE_RAM_CAP,
                          'workers': 16, 'production_stage_executed': False}, sort_keys=True), flush=True)
        return
    if args.stage == 'prepare-cache':
        require(not (runtime.RESULTS / 'cache_build').exists(),
                'Root must preserve the empty failed cache_build directory before this new attempt')
    apply_ram_authority()
    pipeline.output_guard()
    print(json.dumps({'resource_amendment_ref': reference, 'effective_owned_RAM_max_bytes': EFFECTIVE_RAM_CAP,
                      'historical_storage_admission_ref': amendment['original_admission_ref'],
                      'original_freeze_ref': freeze_ref, 'stage': args.stage,
                      'original_workers': 16, 'scientific_functions_unchanged': True}, sort_keys=True), flush=True)
    dispatch(args, protocol, freeze_ref)


if __name__ == '__main__':
    main()
