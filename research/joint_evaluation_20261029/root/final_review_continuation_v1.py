"""Run the independent CPU audit once after all comparisons, then publish it.

This continuation never launches training, changes science or signals processes.
It preserves a failed audit and requires genuine completion before publication.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes

BASE = 'research/joint_evaluation_20261029'
RECORDS = BASE + '/root/final_review_records_v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def immutable(path, value):
    path = r.path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return r.ref(path)


def sources(plan):
    for reference in plan['source_refs']:
        need(r.ref(reference['path']) == reference, 'Reviewed final-audit source changed')
    for reference in plan['european_review_refs']:
        r.read(reference)
    r.read(plan['scientific_contract_ref']) if plan['scientific_contract_ref']['path'].endswith('.json') else need(
        r.ref(plan['scientific_contract_ref']['path']) == plan['scientific_contract_ref'], 'Scientific contract changed')


def readiness(plan):
    path = r.path(plan['trigger_path'])
    if not path.exists():
        return None
    trigger = r.read(r.ref(path))
    need(trigger['plan_ref'] == plan['finish_plan_ref'] and trigger['all_computations_complete'] is True,
         'All reviewed comparisons must actually finish')
    need(trigger['publication']['pushed'] is True, 'Completed comparisons must already be saved on GitHub')
    need(set(trigger['completed_comparison_artifacts']) == set(plan['comparison_paths']), 'Exact complete comparison set')
    return r.ref(path)


def audit_once(plan, folder, trigger_ref, plan_ref=None):
    wait_path = folder/'actual_wait.json'
    if wait_path.exists():
        wait = r.read(r.ref(wait_path))
        need(wait['actual_wait'] is True and wait['actual_returncode'] == 0 and wait['owned_process_group_absent'] is True,
             'Previous audit failed or was interrupted, inspect rather than replay')
        need(wait['command'] == plan['audit_argv'] and wait['trigger_ref'] == trigger_ref, 'Exact previous audit binding')
        return wait['result_ref']
    need(not (folder/'started.json').exists(), 'Interrupted audit must not be replayed')
    output = r.path(plan['audit_output'])
    need(not output.exists(), 'Existing unaffiliated audit output must not be overwritten')
    while int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))) * 1024 < 64*2**30:
        time.sleep(30)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (folder/'worker.log').open('xb') as log:
        child = subprocess.Popen(plan['audit_argv'], cwd=r.ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
        row=dict(pid=child.pid,process_start_ticks=ticks,command=plan['audit_argv'],trigger_ref=trigger_ref,started_unix=time.time())
        immutable(folder/'started.json',row)
        r.save(folder/'status.json',dict(status='running_independent_audit',plan_ref=plan_ref,
            pid=os.getpid(),audit_pid=child.pid,audit_process_start_ticks=ticks,trigger_ref=trigger_ref,
            started_ref=r.ref(folder/'started.json')))
        code=child.wait()
        while processes.session_members(child.pid):
            time.sleep(1)
    row.update(actual_wait=True,actual_returncode=code,owned_process_group_absent=True,finished_unix=time.time())
    if code==0 and output.is_file():
        row['result_ref']=r.ref(output)
    immutable(wait_path,row)
    need(code==0 and 'result_ref' in row, 'Independent audit did not pass, preserve its failure')
    return row['result_ref']


def publication_paths(plan, folder):
    paths=['README.md',plan['audit_output'],plan['scientific_contract_ref']['path'],
           str((folder/'benchmark_completion.json').relative_to(r.ROOT)),
           str((folder/'actual_wait.json').relative_to(r.ROOT))]
    for path in paths:
        need(r.path(path).resolve().is_relative_to(r.ROOT), 'Publication stays inside the project')
        need(r.path(path).is_file() and r.path(path).stat().st_size<=8*2**20, 'Only small registered reports')
    return paths


def publish(plan, folder):
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=r.ROOT).returncode!=0:
        return False
    readme=r.ROOT/'README.md'
    text=readme.read_text()
    begin='<!-- joint-finish-operations:start -->'
    end='<!-- joint-finish-operations:end -->'
    need(text.count(begin)==text.count(end)==1,'Exact existing operational report block')
    replacement=begin+'\nAll registered American, ARGMix and European comparisons have completed. The final American and ARGMix products passed independent numerical checks. Scientific reconstruction limits and mixed method rankings remain explicit. This completes the registered comparison run, not the manuscript.\n\n[Independent final audit]('+plan['audit_output']+') · [Scientific interpretation and limitations]('+plan['scientific_contract_ref']['path']+')\n'+end
    readme.write_text(text[:text.index(begin)]+replacement+text[text.index(end)+len(end):])
    paths=publication_paths(plan,folder)
    subprocess.run(['git','add','--',*paths],cwd=r.ROOT,check=True)
    staged=set(filter(None,subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=r.ROOT).decode().split('\0')))
    if not staged<=set(paths):
        return False
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=r.ROOT).returncode
    need(changed in (0,1),'Git status check failed')
    if changed:
        subprocess.run(['git','commit','--only','-m','Publish independent final benchmark audit','--',*paths],cwd=r.ROOT,check=True)
    env=os.environ.copy()
    env['SSH_AUTH_SOCK']=str(r.ROOT/'.runtime/ssh/agent.sock')
    if subprocess.run(['git','push','origin','HEAD'],cwd=r.ROOT,env=env).returncode!=0:
        return False
    r.save(folder/'publication.json',dict(pushed=True,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip(),published_unix=time.time(),paths=paths))
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    plan_ref=r.ref(args.plan)
    plan=r.read(plan_ref)
    need(plan['schema']=='final_review_continuation_plan_v1','Reviewed plan required')
    sources(plan)
    if args.inspect:
        print(json.dumps(dict(valid=True,ready=readiness(plan) is not None,workers_launched=0,truth_opened=False)))
        return
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    folder=r.path(RECORDS)
    folder.mkdir(exist_ok=True)
    with (folder/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not (folder/'publication.json').exists(),'Final audit already published')
        r.save(folder/'status.json',dict(status='waiting_for_completed_comparisons',plan_ref=plan_ref,pid=os.getpid()))
        trigger=None
        while trigger is None:
            sources(plan)
            trigger=readiness(plan)
            if trigger is None:
                time.sleep(60)
        try:
            result_ref=audit_once(plan,folder,trigger,plan_ref)
            result=r.read(result_ref)
            need(result['passed'] is True and result['schema']=='independent_final_benchmark_audit_v1' and
                 set(result['reports'])=={'american','argmix'} and result['source_ref']==plan['audit_source_ref'],
                 'Exact independent numerical audit must pass')
        except Exception as error:
            failure=dict(status='audit_failed',plan_ref=plan_ref,pid=os.getpid(),trigger_ref=trigger,
                         error=str(error),finished_unix=time.time())
            for name in ('started.json','actual_wait.json'):
                path=folder/name
                if path.is_file():
                    failure[name.removesuffix('.json')+'_ref']=r.ref(path)
            r.save(folder/'status.json',failure)
            raise
        completion=dict(schema='registered_benchmark_completion_v1',passed=True,plan_ref=plan_ref,
            trigger_ref=trigger,independent_audit_ref=result_ref,european_review_refs=plan['european_review_refs'],
            scientific_contract_ref=plan['scientific_contract_ref'],paper_complete=False,
            exact_published_table_replication_claimed=False,comparison_refs=[r.ref(path) for path in plan['comparison_paths']])
        if (folder/'benchmark_completion.json').exists():
            need(r.read(r.ref(folder/'benchmark_completion.json'))==completion,'Existing final completion differs')
        else:
            immutable(folder/'benchmark_completion.json',completion)
        r.save(folder/'status.json',dict(status='audit_passed_publication_pending',plan_ref=plan_ref,
            independent_audit_ref=result_ref))
        while not publish(plan,folder):
            r.save(folder/'status.json',dict(status='audit_passed_publication_pending',plan_ref=plan_ref))
            time.sleep(60)
        r.save(folder/'status.json',dict(status='registered_benchmark_complete_and_pushed',plan_ref=plan_ref,independent_audit_ref=result_ref))
        print('Registered comparisons independently checked and pushed. Manuscript completion remains separate.',flush=True)


if __name__=='__main__':
    main()
