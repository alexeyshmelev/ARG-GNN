"""Root-priority GPU admission and a lifetime-shared GPU3 flock."""
import fcntl
import json
import os
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue.successor_support import need

LOCK_PATH = '.runtime/benchmarking/joint_root_successor_gpu3.lock'


def root_release(plan):
    path = r.path(plan['root_gpu_priority_release_path'])
    if not path.is_file():
        return None
    release_ref = r.ref(path)
    release = r.read(release_ref)
    need(release['schema'] == 'root_joint_models_gpu_release_v1', 'Unknown root GPU release schema')
    need(release['successor_config_ref'] == plan['handoff_config_ref'], 'Root release belongs to a different successor')
    need(set(release['completed_models']) == set(plan['required_root_model_chains']), 'All three root model GPU chains have priority')
    for name, wait_ref in release['completed_models'].items():
        wait = r.read(wait_ref)
        need(wait.get('actual_wait') is True and wait.get('actual_returncode') == 0
             and wait.get('owned_process_group_absent') is True, 'Root model fit needs genuine successful wait: ' + name)
        need(not processes.session_members(wait['pid']), 'Root model worker session remains live')
    return release_ref


def allowed(plan):
    # Both new root fits retain priority even before their prepared inputs exist.
    if root_release(plan) is None:
        return []
    config = r.read(plan['handoff_config_ref'])
    native = config['sidecars'][0]
    state = json.loads(r.path(native['state_path']).read_text())
    need(state['plan_ref'] == native['plan_ref'], 'Native GPU owner plan changed')
    if not all(row['status'] in ('succeeded', 'failed', 'blocked') for row in state['tasks'].values()):
        return [3]
    for row in state['tasks'].values():
        if 'pid' in row:
            need(row.get('actual_wait') is True and row.get('owned_process_group_absent') is True, 'Terminal sidecar row lacks a real wait')
            if processes.session_members(row['pid']):
                return [3]
    return [3, 4, 5]


def acquire(gpu):
    if gpu != 3:
        return None
    path = r.path(LOCK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    return fd


def release(fd):
    if fd is not None and fd is not False:
        os.close(fd)
