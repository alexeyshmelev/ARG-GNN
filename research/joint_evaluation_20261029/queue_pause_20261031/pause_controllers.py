"""Suspend verified project controllers before releasing their GPU worker."""
import json
import os
from pathlib import Path
import signal
from datetime import datetime, timezone

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
TARGETS = {
    1981120: 591807927,
    2031398: 592703029,
    2767954: 603486940,
    2968580: 605198446,
    2969549: 605205847,
    2970583: 605212785,
    2970611: 605212926,
}
receipt = {'request': 'Pause current queue and release GPUs', 'utc': datetime.now(timezone.utc).isoformat(), 'controllers': []}
for pid, expected_ticks in TARGETS.items():
    entry = {'pid': pid, 'start_ticks': expected_ticks}
    try:
        fd = os.pidfd_open(pid)
        try:
            proc = Path('/proc') / str(pid)
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            assert int(stat[19]) == expected_ticks, 'PID identity changed'
            assert Path(os.readlink(proc / 'cwd')) == ROOT, 'Unexpected working directory'
            signal.pidfd_send_signal(fd, signal.SIGSTOP)
            entry['signal'] = 'SIGSTOP'
        finally:
            os.close(fd)
    except ProcessLookupError:
        entry['signal'] = 'already_exited'
    receipt['controllers'].append(entry)
    print(entry, flush=True)
(ROOT / 'research/joint_evaluation_20261029/queue_pause_20261031/controllers.json').write_text(json.dumps(receipt, indent=2) + '\n')
