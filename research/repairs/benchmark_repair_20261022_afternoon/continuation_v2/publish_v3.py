"""Commit the verified repair and attempt one GitHub push from anogena."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=Path(__file__).parent
PATHS=['README.md','configs/benchmarks/execution_policy.json',
    'configs/benchmarks/executable_workflow_20261022_afternoon_v2.json',
    'configs/benchmarks/executable_workflow_20261022_afternoon_v3.json',
    'research/repairs/benchmark_repair_20261022_afternoon',
    'src/arggnn/benchmarking/storage_recovery_ap_train001_20261022.py',
    'research/benchmark_recovery_afternoon_v2_current.json','research/benchmark_current_verified_status.json',
    'research/current_scoreboard.json','research/current_scoreboard.md','research/experiment_decisions.json',
    'research/experiment_ledger.json','research/experiment_ledger.md']


def publish():
    environment=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
    def git(arguments):
        result=subprocess.run(['git',*arguments],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,timeout=40)
        if result.returncode:raise RuntimeError(result.stdout[-1200:]+result.stderr[-2000:])
        return result.stdout.strip()
    with (ROOT/'.runtime/benchmarking/workflow_reporting/reporting.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        git(['add','--',*PATHS])
        # Preserve exact unified-diff evidence, including its space-prefixed context.
        excluded=':(glob,exclude)research/repairs/benchmark_repair_20261022_afternoon/**/*.diff'
        git(['diff','--cached','--check','--',*PATHS,excluded])
        print(git(['diff','--cached','--shortstat','--',*PATHS]),flush=True)
        print(git(['commit','--only','-m','Continue benchmark with verified external storage recovery and European memory repair','--',*PATHS]),flush=True)
        commit=git(['rev-parse','HEAD'])
    try:
        output=git(['push','origin','HEAD:refs/heads/main'])
        receipt=dict(success=True,commit=commit,output=output)
    except Exception as error:
        receipt=dict(success=False,commit=commit,error=str(error))
    receipt.update(observed_utc=datetime.now(timezone.utc).isoformat(),host='anogena',
        benchmark_jobs_interrupted=False,Mac_git_used=False)
    (HERE/'actual_v3/git_publication.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':publish()
