"""Preserve a durable optimizer boundary and replace only the owned TreeLoom job.

The owner requested an efficient rebuilt comparison queue. This operation targets
one verified task session. Other sessions, GPUs and historical files are untouched.
Run once after the replacement's source review and CPU continuation checks pass.
"""
from pathlib import Path
import json
import os
import shutil
import signal
import sys
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes

sys.stderr = sys.stdout
ROOT = r.ROOT
TASK_ID = 'AM_treeloom_fit'
JOB = '.runtime/benchmarking/workflow_20261022_gpu345_v2/jobs/AM_treeloom_fit.json'
OUTPUT = ROOT / 'experiments/results/benchmark_campaign_20261018_workflow/release_20261022_gpu345_v2/other/AM_treeloom_fit'
CAPTURE = ROOT / 'experiments/results/paper_campaign_20261024/treeloom_migration_capture'
STATE = ROOT / '.runtime/benchmarking/workflow_20261022_gpu345_v2/state.json'


def main():
    if CAPTURE.exists():
        raise RuntimeError('Capture namespace already exists. Inspect it, never replay process control.')
    job = json.loads((ROOT / JOB).read_text())
    identity = json.loads((OUTPUT / 'supervision_identity.json').read_text())
    row = json.loads(STATE.read_text())['tasks'][TASK_ID]
    assert row['status'] == 'running' and row['job_ref'] == r.ref(JOB) == identity['job_ref']
    assert job['task']['id'] == TASK_ID and job['call']['spec']['physical_gpu'] == 4
    child = identity['child_identity']
    leader = child['pid']
    assert child['session'] == child['pgid'] == leader and child['ppid'] == row['pid']
    proc = Path('/proc') / str(leader)
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    assert int(stat[19]) == child['start_ticks'] and int(stat[3]) == leader
    assert (proc / 'cwd').resolve().is_relative_to(ROOT)
    assert (proc / 'cmdline').read_bytes().startswith(b'AI Auto Research Pipeline Test')
    supervisor = Path('/proc') / str(row['pid'])
    assert JOB.encode() in (supervisor / 'cmdline').read_bytes()
    # Authenticate every saved source before any process is changed.
    for reference in job['call']['source_refs']:
        assert r.ref(reference['path']) == reference, reference['path']
    manifest_path = 'research/benchmark_rebuild_20261024/american/treeloom_speed/source_pins.json'
    assert r.ref(manifest_path)['sha256'] == 'b0d53c2c054a63c1ec7a6b8438130b22e2b2990f352c4d7eba1ee581f84bcf2f'
    from research.benchmark_rebuild_20261024.american.treeloom_speed import training_v2 as training
    training.requirements()
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    import torch
    original = json.loads((ROOT / '.runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json').read_text())['call']['spec']['training_spec']
    registry = fresh.FreshParentRegistry.from_ref(original['parent_registry_ref'])
    model = fresh.FreshTreeLoom(registry).build_model()
    source = OUTPUT / 'native/checkpoints/latest.pt'
    preview = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    training.validate_saved_state(preview, model, registry)
    del preview
    members = processes.session_members(leader)
    assert any(m['pid'] == leader and int(m['start_ticks']) == child['start_ticks'] for m in members)
    CAPTURE.mkdir(parents=True)
    r.save(CAPTURE / 'control_intent.json', dict(task_id=TASK_ID, job_ref=r.ref(JOB),
        identity=identity, exact_session_members=members, signal='SIGTERM',
        purpose='Resume the same optimizer state in qualified faster full-parent execution',
        other_processes_authorized=False, started_unix=time.time()))
    for member in members:
        processes.signal_member(member, signal.SIGTERM)
    deadline = time.monotonic() + 25
    while processes.session_members(leader) and time.monotonic() < deadline:
        time.sleep(.5)
    remaining = processes.session_members(leader)
    if remaining:
        for member in remaining:
            processes.signal_member(member, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while processes.session_members(leader) and time.monotonic() < deadline:
            time.sleep(.5)
    assert not processes.session_members(leader), 'Owned numerical session must exit before capture'
    checkpoint = CAPTURE / 'checkpoint.pt'
    with source.open('rb') as src, checkpoint.open('xb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    copied = r.ref(checkpoint)
    assert copied['sha256'] == r.ref(source)['sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    steps = training.validate_saved_state(saved, model, registry)
    evidence = dict(schema='treeloom_durable_optimizer_boundary_v2', checkpoint_ref=copied,
        original_checkpoint_ref=r.ref(source), parent_registry_ref=registry.reference,
        complete_parent_updates=steps, durable_complete_parent_optimizer_boundary=True,
        no_inflight_update_in_serialized_state=True, VAL_truth_read=False, TEST_truth_read=False,
        source_refs=job['call']['source_refs'], original_job_ref=r.ref(JOB),
        observation_barrier_ref=saved['observation_barrier_ref'],
        original_last_completed_parent=saved['last_completed_parent'],
        next_schedule_index=saved['next_schedule_index'],
        owned_numerical_session_absent=True, unrelated_sessions_signalled=False,
        control_intent_ref=r.ref(CAPTURE / 'control_intent.json'), captured_unix=time.time())
    evidence_ref = r.save(CAPTURE / 'evidence.json', evidence)
    validation_spec = dict(train_inputs=original['train_inputs'],
        resume_checkpoint_ref=copied, resume_evidence_ref=evidence_ref)
    assert training.validate_resume_checkpoint(saved, model, registry, validation_spec) == steps
    del saved
    for name in ('latest_checkpoint.json', 'training_live.json'):
        old = OUTPUT / 'native' / name
        if old.is_file():
            shutil.copyfile(old, CAPTURE / ('original_' + name))
    result = dict(checkpoint_ref=copied, evidence_ref=evidence_ref, completed_updates_preserved=steps,
                  next_update=steps+1, total_budget=600, owned_session_absent=True,
                  old_supervisor_actual_wait_path=str((OUTPUT / 'actual_wait.json').relative_to(ROOT)))
    r.save(CAPTURE / 'completed.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
