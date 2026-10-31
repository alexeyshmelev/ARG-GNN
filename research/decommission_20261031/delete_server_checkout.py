"""Delete only the user-authorized checkout after verifying its GitHub backup."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/decommission_20261031'
assert ROOT.is_dir() and not ROOT.is_symlink() and ROOT.resolve() == ROOT
assert Path.cwd() == ROOT.parent, 'Run from the parent, outside the deletion target'
guard = json.loads((OUT / 'deletion_guard.json').read_text())
info = ROOT.stat()
assert (guard['exact_path'], guard['device'], guard['inode']) == (str(ROOT), info.st_dev, info.st_ino)
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    mountpoint = line.split()[4]
    assert not mountpoint.startswith(str(ROOT) + '/'), 'Nested mount must not be deleted'
summary = json.loads((OUT / 'archive_summary.json').read_text())
assert summary['status'] == 'local_archive_integrity_verified_remote_push_pending'
for row in summary['archives']:
    assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256']
# Refuse removal if tracked changes appeared after publication.
subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet'], check=True)
subprocess.run(['git', '-C', str(ROOT), 'diff', '--cached', '--quiet'], check=True)
head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
env = dict(os.environ, SSH_AUTH_SOCK=str(ROOT / '.runtime/ssh/agent.sock'))
remote = subprocess.check_output(['git', '-C', str(ROOT), 'ls-remote', 'origin', 'refs/heads/main'], env=env, text=True).strip().split()[0]
assert remote == head, 'GitHub main does not contain the exact local commit'
print('VERIFIED_GITHUB_COMMIT', head, flush=True)
# All project workers were retired separately using PID and start-time checks.
# Refuse deletion if another process still has its working directory in the repo.
private_agents = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid():
        continue
    try:
        cmd = (p / 'cmdline').read_bytes().split(b'\0')
        cwd = Path(os.readlink(p / 'cwd'))
        assert cwd != ROOT and ROOT not in cwd.parents, f'Process {p.name} still uses project cwd'
        if cmd and Path(os.fsdecode(cmd[0])).name == 'ssh-agent' and os.fsencode(str(ROOT / '.runtime/ssh/agent.sock')) in cmd:
            ticks = int((p / 'stat').read_text().rsplit(')', 1)[1].split()[19])
            private_agents.append((int(p.name), ticks))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
for pid, ticks in private_agents:
    fd = os.pidfd_open(pid)
    try:
        p = Path('/proc') / str(pid)
        assert int((p / 'stat').read_text().rsplit(')', 1)[1].split()[19]) == ticks
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
print('DELETION_START', str(ROOT), datetime.now(timezone.utc).isoformat(), flush=True)
command = ['rm', '--recursive', '--force', '--one-file-system', '--', str(ROOT)]
result = subprocess.run(command)
assert result.returncode == 0 and not ROOT.exists() and not ROOT.is_symlink(), 'Deletion is incomplete'
print('DELETION_CONFIRMED', json.dumps({'path': str(ROOT), 'exists': False, 'github_commit': head,
                                     'utc': datetime.now(timezone.utc).isoformat(), 'outside_folder_deleted': False}), flush=True)
