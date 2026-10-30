"""Pause owned CPU descendants and release the verified project CUDA worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/joint_evaluation_20261029/queue_pause_20261031'
GPU_PID = 3204575
GPU_TICKS = 607005234
controllers = json.loads((OUT / 'controllers.json').read_text())['controllers']
controller_ids = {row['pid'] for row in controllers}

def inspect(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        cwd = Path(os.readlink(p / 'cwd'))
        return {'pid': pid, 'ppid': int(stat[1]), 'state': stat[0],
                'pgid': int(stat[2]), 'start_ticks': int(stat[19]), 'cwd': str(cwd)}
    except (OSError, ProcessLookupError):
        return None

def snapshot():
    rows = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            row = inspect(int(p.name))
            if row:
                rows[row['pid']] = row
    return rows

def descendants(rows, roots):
    selected = set(roots)
    while True:
        expanded = selected | {pid for pid, row in rows.items() if row['ppid'] in selected}
        if expanded == selected:
            return selected
        selected = expanded

def send(row, sig):
    try:
        fd = os.pidfd_open(row['pid'])
        try:
            current = inspect(row['pid'])
            if not current or current['start_ticks'] != row['start_ticks']:
                return 'already_exited'
            cwd = Path(current['cwd'])
            assert cwd == ROOT or ROOT in cwd.parents, 'Outside project boundary'
            signal.pidfd_send_signal(fd, sig)
            return sig.name
        finally:
            os.close(fd)
    except ProcessLookupError:
        return 'already_exited'

rows = snapshot()
assert rows[GPU_PID]['start_ticks'] == GPU_TICKS
assert rows[GPU_PID]['ppid'] == 1981120
assert rows[GPU_PID]['pgid'] == GPU_PID
owned = descendants(rows, controller_ids)
gpu_members = descendants(rows, {GPU_PID})
assert gpu_members <= owned
receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'reason': 'User requested queue pause and GPU release',
           'cpu_signals': [], 'gpu_signals': [], 'actual_child_wait_performed': False,
           'resume_note': 'Controllers and CPU reconstruction are suspended. GPU training was interrupted; validate and explicitly resume the saved checkpoint before continuing the controller. Do not blindly SIGCONT all controllers.'}
checkpoint_dir = ROOT / 'experiments/results/paper_american_20261024/argformer_american_test_009/mlm/native/checkpoints'
receipt['existing_checkpoints'] = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                                    'mtime_ns': p.stat().st_mtime_ns, 'is_symlink': p.is_symlink()}
                                   for p in checkpoint_dir.iterdir()]
for pid in sorted(owned - gpu_members - controller_ids):
    row = rows[pid]
    receipt['cpu_signals'].append({**row, 'signal': send(row, signal.SIGSTOP)})
receipt['gpu_signals'].append({**rows[GPU_PID], 'signal': send(rows[GPU_PID], signal.SIGINT)})
(OUT / 'workers.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('SIGINT sent to verified ARGformer worker; CPU queue descendants suspended.', flush=True)
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    current = inspect(GPU_PID)
    if current is None or current['state'] == 'Z':
        break
    time.sleep(0.5)
# Release any remaining children of this exact CUDA worker; never signal by name.
for sig, grace in [(signal.SIGTERM, 10), (signal.SIGKILL, 3)]:
    remaining = [row for pid in gpu_members if (row := inspect(pid))
                 and row['start_ticks'] == rows[pid]['start_ticks'] and row['state'] != 'Z']
    if not remaining:
        break
    for row in remaining:
        receipt['gpu_signals'].append({**row, 'signal': send(row, sig)})
    time.sleep(grace)
receipt['controller_states'] = [inspect(row['pid']) for row in controllers]
receipt['cpu_states'] = [inspect(row['pid']) for row in receipt['cpu_signals']]
receipt['gpu_worker_states'] = [inspect(pid) for pid in sorted(gpu_members)]
receipt['gpu_memory'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader'], text=True)
receipt['compute_processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader'], text=True)
receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
(OUT / 'workers.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'controllers': receipt['controller_states'], 'cpu_states': receipt['cpu_states'],
                  'gpu_memory': receipt['gpu_memory'], 'compute_processes': receipt['compute_processes']}, indent=2), flush=True)
