"""End the exact previously paused project processes for folder retirement."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import signal

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
PAUSE = ROOT / 'research/joint_evaluation_20261029/queue_pause_20261031'
records = json.loads((PAUSE / 'controllers.json').read_text())['controllers']
records += json.loads((PAUSE / 'workers.json').read_text())['cpu_signals']
# Historical project-only searches and the project dataset smoke-check family.
records += [{'pid': pid, 'start_ticks': ticks} for pid, ticks in {
    1844431: 441824086, 1844434: 441824087, 1844436: 441824087,
    2060822: 442630324, 2060998: 442630937, 2060999: 442630938,
    2061000: 442630938, 4095907: 489812003,
}.items()]
receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'reason': 'User authorized deletion of ARG-GNN server checkout', 'processes': []}
for row in records:
    result = {'pid': row['pid'], 'start_ticks': row['start_ticks']}
    try:
        fd = os.pidfd_open(row['pid'])
        try:
            p = Path('/proc') / str(row['pid'])
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[19]) != row['start_ticks']:
                result['action'] = 'not_signalled_identity_changed'
            else:
                cwd = Path(os.readlink(p / 'cwd'))
                assert cwd == ROOT or ROOT in cwd.parents
                signal.pidfd_send_signal(fd, signal.SIGKILL)
                result['action'] = 'SIGKILL_exact_project_process'
        finally:
            os.close(fd)
    except (ProcessLookupError, FileNotFoundError):
        result['action'] = 'already_exited'
    receipt['processes'].append(result)
(ROOT / 'research/decommission_20261031/process_retirement.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
