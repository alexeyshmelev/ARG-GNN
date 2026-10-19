"""Publish the owner's requested progress updates from the server repository.

The caller must hold workflow_reporting/reporting.lock across report writes and
this call. Only the two public progress files are staged or committed here.
Publication failure is reported locally and never interrupts scientific jobs.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
PATHS = ['README.md', 'research/benchmark_readable_current.json']
RECEIPT = ROOT / '.runtime/benchmarking/workflow_reporting/readable_git_publication.json'


def command(arguments, environment):
    result = subprocess.run(['git', *arguments], cwd=ROOT, env=environment,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('git '+arguments[0]+' failed: '+result.stderr[-1600:])
    return result.stdout.strip()


def _push_reports(minimum_interval_seconds=900):
    now = time.time()
    if RECEIPT.exists():
        previous = json.loads(RECEIPT.read_text())
        if now-previous['attempt_unix'] < minimum_interval_seconds:
            return
    receipt = dict(attempt_unix=now, observed_utc=datetime.now(timezone.utc).isoformat(),
                   host='anogena', repository=str(ROOT), paths=PATHS, success=False)
    try:
        for relative in PATHS:
            item = ROOT / relative
            if not item.is_file() or item.is_symlink() or item.stat().st_size > 2**20:
                raise RuntimeError('Expected a bounded ordinary public progress file: '+relative)
        environment = dict(os.environ, GIT_TERMINAL_PROMPT='0',
            GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
        if command(['rev-parse', '--show-toplevel'], environment) != str(ROOT):
            raise RuntimeError('Wrong progress repository')
        if command(['symbolic-ref', '--short', 'HEAD'], environment) != 'main':
            raise RuntimeError('Progress publication requires the existing main branch')
        changed = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *PATHS],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, timeout=30).returncode
        if changed not in (0, 1):
            raise RuntimeError('Unable to inspect progress changes')
        if changed:
            command(['add', '--', *PATHS], environment)
            command(['commit', '--only', '-m', 'Update active benchmark progress', '--', *PATHS], environment)
        receipt['commit'] = command(['rev-parse', 'HEAD'], environment)
        command(['push', 'origin', 'HEAD:refs/heads/main'], environment)
        receipt['success'] = True
    except Exception as error:
        receipt['error'] = type(error).__name__+': '+str(error)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_name(RECEIPT.name+'.next')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    os.replace(temporary, RECEIPT)
    return receipt


def push_reports(minimum_interval_seconds=900):
    try:
        return _push_reports(minimum_interval_seconds)
    except Exception as error:
        return dict(success=False, error=type(error).__name__+': '+str(error),
                    scientific_jobs_interrupted=False)
