"""Commit selected repair files on anogena and attempt one ordinary Git push."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=Path(__file__).parent
PATHS=['README.md','configs/benchmarks/execution_policy.json','configs/benchmarks/executable_workflow_20261022_afternoon_v1.json',
    'research/repairs/benchmark_repair_20261022_afternoon','research/benchmark_recovery_afternoon_current.json',
    'research/benchmark_repair_status_20261022.json','research/current_scoreboard.json','research/current_scoreboard.md',
    'research/experiment_decisions.json','research/experiment_ledger.json','research/experiment_ledger.md']


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
        # Historical unified diffs include significant space-prefixed blank context
        # lines. Keep them exact and check source, JSON and prose independently.
        excluded=[':(exclude)research/repairs/benchmark_repair_20261022_afternoon/runtime/lineage/*.diff',
                  ':(exclude)research/repairs/benchmark_repair_20261022_afternoon/training/fit_v4.diff']
        git(['diff','--cached','--check','--',*PATHS,*excluded])
        print(git(['diff','--cached','--shortstat','--',*PATHS]),flush=True)
        print(git(['commit','--only','-m','Recover full benchmark queue with adequate American GPU memory allowance','--',*PATHS]),flush=True)
        commit=git(['rev-parse','HEAD'])
    try:
        output=git(['push','origin','HEAD:refs/heads/main'])
        receipt=dict(success=True,commit=commit,output=output)
    except Exception as error:
        receipt=dict(success=False,commit=commit,error=str(error))
    receipt.update(observed_utc=datetime.now(timezone.utc).isoformat(),all_git_commands_host='anogena',
        immutable_diff_context_preserved=True,other_selected_files_passed_whitespace_check=True)
    (HERE/'actual_v1/git_publication.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':publish()
