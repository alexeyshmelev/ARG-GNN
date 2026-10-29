"""Finish reviewed benchmark dependencies without changing any model or dataset.

Authorize the exact ARGMix cache only when full storage accounting fits, release
native GPU work after all three model chains really finish, and publish completed
comparisons. All execution uses explicit plaintext argv. Existing jobs are never
signalled. A completed computation remains pending independent final review.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_v2_support as support
from research.joint_evaluation_20261029.queue import successor_v2_gpu
from research.joint_evaluation_20261029.root.result_sections import completed_sections

BASE = 'research/joint_evaluation_20261029'
RECORDS = BASE + '/root/finish_records_v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def immutable(path, value):
    path = r.path(path)
    need(not path.exists(), 'Preserve existing immutable receipt: ' + str(path))
    return r.save(path, value)


def check_reference(reference):
    observed = r.ref(reference['path'])
    need(observed['sha256'] == reference['sha256'], 'Product content changed')
    if 'bytes' in reference:
        need(r.path(reference['path']).stat().st_size == reference['bytes'], 'Product size changed')
    return observed


def verify_reference(reference):
    observed = check_reference(reference)
    return r.read(observed)


def waited(reference):
    row = verify_reference(reference)
    need(row.get('actual_wait') is True and row.get('actual_returncode') == 0 and
         row.get('owned_process_group_absent') is True and row.get('products_complete', True) is True,
         'A genuine successful process wait is required')
    need(not processes.session_members(row['pid']), 'Producer process session is not drained')
    for product in row.get('products', []):
        check_reference(product)
    if row.get('result_ref'):
        verify_reference(row['result_ref'])
    return row


def chain_completions(plan):
    completed = {}
    for name, spec in plan['model_chains'].items():
        path = r.path(spec['wait_path'])
        if not path.exists():
            continue
        reference = r.ref(path)
        row = waited(reference)
        need(row[spec['binding_key']] == spec['binding_ref'], 'Exact frozen chain ownership: ' + name)
        if name == 'european_four_joint':
            product = verify_reference(row['result_ref'])
            need(product['complete'] and product['all_five_TEST_predictions_complete'], 'All five European predictions')
            stage_refs = product['stage_actual_wait_refs']
        else:
            stage_refs = row['stage_actual_wait_refs']
        need(len(stage_refs) == spec['GPU_stage_count'], 'Entire registered GPU chain: ' + name)
        expected = spec['stages']
        need(len(stage_refs) == len(expected) and len({v['path'] for v in stage_refs}) == len(expected), 'Unique complete stage waits')
        for stage_ref, stage in zip(stage_refs, expected):
            need(r.path(stage_ref['path']) == r.path(stage['wait_path']), 'Exact ordered stage wait path')
            saved = waited(stage_ref)
            need(saved['plan_ref'] == stage['plan_ref'] and saved['command'] == stage['command'], 'Exact stage owner and command')
            if 'task_id' in stage:
                need(saved['task_id'] == stage['task_id'], 'Exact original stage task identity')
        completed[name] = reference
    return completed


def maybe_admit_cache(plan, template, folder):
    config = r.read(plan['successor_config_ref'])
    path = r.path(config['argmix_production_admission_path'])
    if path.exists():
        admission = r.read(r.ref(path))
        need(admission['protocol_ref'] == plan['argmix_protocol_ref'], 'Exact frozen ARGMix protocol admission')
        support.argmix_reserved_bytes(template, config, r.read(config['argmix_joint_allocation_ref']))
        return dict(admitted=True, admission_ref=r.ref(path))
    accounting = support.commitment(template, argmix_override_bytes=25*2**30)
    if not accounting['admitted']:
        return dict(admitted=False, reserved_bytes=accounting['total_bytes'], ceiling_bytes=accounting['ceiling_bytes'])
    verify_reference(plan['argmix_protocol_ref'])
    ledger = dict(schema='argmix_production_actual_ledger_v1', created_unix=time.time(), admitted=True,
        successor_config_ref=plan['successor_config_ref'], cache_scope='experiments/cache/joint_argmix_20261029',
        cache_max_bytes=24*2**30, combined_reserved_bytes=accounting['total_bytes'], accounting=accounting,
        root_authorization_performed=True)
    ledger_path = folder / 'argmix_admission_ledger.json'
    if ledger_path.exists():
        # An interrupted publication does not authorize replay from stale accounting.
        raise ValueError('Admission ledger exists without root receipt, inspect interrupted publication')
    ledger_ref = immutable(ledger_path, ledger)
    admission = dict(schema='joint_argmix_cache_admission_v1', admitted=True, root_confirmed=True,
        protocol_ref=plan['argmix_protocol_ref'], prior_allocation_ref=config['argmix_joint_allocation_ref'],
        successor_config_ref=plan['successor_config_ref'], cache_scope=ledger['cache_scope'],
        results_scope='experiments/results/joint_argmix_20261029', cache_max_bytes=24*2**30,
        results_max_bytes=2**30, owned_RAM_max_bytes=64*2**30, combined_reserved_bytes=accounting['total_bytes'],
        owner_ceiling_bytes=1_200_000_000_000, actual_ledger_receipt_ref=ledger_ref)
    admission_ref = immutable(path, admission)
    support.argmix_reserved_bytes(template, config, r.read(config['argmix_joint_allocation_ref']))
    return dict(admitted=True, admission_ref=admission_ref)


def maybe_release(plan, template, completed):
    path = r.path(template['root_gpu_priority_release_path'])
    if path.exists():
        return successor_v2_gpu.root_release(template)
    if set(completed) != set(template['required_root_model_chains']):
        return None
    immutable(path, dict(schema='root_joint_models_gpu_release_v1',
        successor_v2_config_ref=plan['successor_config_ref'], completed_models=completed))
    return successor_v2_gpu.root_release(template)


def maybe_compare_american(plan, folder):
    from research.joint_evaluation_20261029.root.american_comparison import readiness
    stage = folder / 'american_comparison'
    terminal = stage / 'actual_wait.json'
    if terminal.exists():
        row = waited(r.ref(terminal))
        need(row['command'] == plan['american_comparison_argv'], 'Exact comparison command')
        return row['result_ref']
    if readiness(plan['american_plan_ref']['path']) is None:
        return None
    need(not (stage/'started.json').exists(), 'Interrupted comparison preserved, no automatic replay')
    product_path = r.path(plan['american_comparison_product_path'])
    need(not product_path.exists(), 'Unattributed comparison product must not be overwritten')
    stage.mkdir(exist_ok=True)
    with (stage/'worker.log').open('xb') as log:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        child = subprocess.Popen(plan['american_comparison_argv'], cwd=r.ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
        row=dict(pid=child.pid, process_start_ticks=ticks, command=plan['american_comparison_argv'], started_unix=time.time())
        immutable(stage/'started.json', row)
        code=child.wait()
        while not processes.drain(child):
            time.sleep(1)
    complete=code==0 and product_path.is_file()
    row.update(actual_wait=True, actual_returncode=code, owned_process_group_absent=True,
               products_complete=complete, finished_unix=time.time())
    if complete:
        row['result_ref']=r.ref(product_path)
    reference=immutable(terminal,row)
    waited(reference)
    return row['result_ref']


def publish(plan, folder, artifacts):
    """Commit only explicitly listed reports when the index is otherwise clean."""
    receipt_path=folder/'last_publication.json'
    allowed = set(plan['publication_artifact_paths'])
    normalized = [str(r.path(path).resolve().relative_to(r.ROOT)) for path in artifacts]
    need(set(normalized) <= allowed and len(normalized) == len(set(normalized)), 'Only exact registered comparison artifacts may be published')
    refs=[r.ref(path) for path in normalized]
    if receipt_path.exists():
        previous=r.read(r.ref(receipt_path))
        if previous['result_refs']==refs and previous['pushed']:
            return previous
    index=subprocess.run(['git','diff','--cached','--quiet'],cwd=r.ROOT)
    if index.returncode!=0:
        return dict(pushed=False, reason='Existing staged work belongs to root, publication deferred')
    subprocess.run(plan['publisher_argv'],cwd=r.ROOT,check=True)
    paths=['README.md',BASE+'/root/CURRENT_STATUS.json']+normalized
    for path in paths:
        need(r.path(path).stat().st_size <= 8*2**20, 'Publication limited to small reports')
    subprocess.run(['git','add','--',*paths],cwd=r.ROOT,check=True)
    staged = subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=r.ROOT).decode().split('\0')
    if not set(filter(None,staged)) <= set(paths):
        return dict(pushed=False, reason='Concurrent staged changes detected, preserve them for root review')
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=r.ROOT).returncode
    if changed==1:
        subprocess.run(['git','commit','--only','-m','Report completed joint-model benchmark comparisons','--',*paths],cwd=r.ROOT,check=True)
    else:
        need(changed==0,'Git index comparison failed')
    env=os.environ.copy()
    env['SSH_AUTH_SOCK']=str(r.ROOT/'.runtime/ssh/agent.sock')
    pushed=subprocess.run(['git','push','origin','HEAD'],cwd=r.ROOT,env=env).returncode==0
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip()
    receipt=dict(result_refs=refs,commit=commit,pushed=pushed,updated_unix=time.time())
    r.save(receipt_path,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    plan_ref=r.ref(args.plan)
    plan=r.read(plan_ref)
    need(plan['schema']=='joint_benchmark_finish_plan_v1','Exact reviewed finishing plan')
    for source in plan['source_refs']:
        need(r.ref(source['path'])==source,'Reviewed finishing source changed')
    config=r.read(plan['successor_config_ref'])
    template=dict(r.read(config['successor_template_ref']),handoff_config_ref=plan['successor_config_ref'])
    need(set(plan['model_chains'])==set(template['required_root_model_chains']),'All three GPU chains')
    if args.inspect:
        accounting=support.commitment(template,argmix_override_bytes=25*2**30)
        print(json.dumps(dict(valid=True,plan_ref=plan_ref,ready_GPU_chains=list(chain_completions(plan)),
            production_storage_fits=accounting['admitted'],production_commitment_bytes=accounting['total_bytes'],
            workers_launched=0,signals_sent=0,admissions_written=0)))
        return
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    folder=r.path(RECORDS)
    folder.mkdir(exist_ok=True)
    with (folder/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not (folder/'ready_for_final_review.json').exists(),'All computations already ready for review')
        previous=None
        while True:
            for source in plan['source_refs']:
                need(r.ref(source['path'])==source,'Finishing source changed while active')
            admission=maybe_admit_cache(plan,template,folder)
            completed=chain_completions(plan)
            release=maybe_release(plan,template,completed)
            comparison=maybe_compare_american(plan,folder)
            _,artifacts=completed_sections()
            publication=publish(plan,folder,artifacts) if artifacts else None
            all_scores=(comparison is not None and
                (r.ROOT/'experiments/results/joint_argmix_20261029/scores/actual_wait.json').exists() and
                (r.path(BASE)/'european_four/chain_records_v1/paired_score/actual_wait.json').exists())
            state=dict(schema='joint_benchmark_finishing_state_v1',plan_ref=plan_ref,
                admitted_ARGMix=admission,completed_GPU_chains=completed,native_GPU_release_ref=release,
                completed_comparison_artifacts=artifacts,publication=publication,
                all_computations_complete=False,independent_final_review_complete=False)
            argmix_complete_path = r.path(plan['argmix_completed_chain_path'])
            if all_scores and argmix_complete_path.exists() and publication and publication.get('pushed') is True:
                need(set(artifacts)==set(plan['publication_artifact_paths']), 'All three complete comparisons and native comparator retained')
                final_argmix = verify_reference(r.ref(argmix_complete_path))
                need(final_argmix['products_complete'] is True and final_argmix['plan_ref'] == plan['argmix_task_plan_ref'], 'Complete ARGMix execution including final validation diagnostic')
                argmix_tasks = r.read(plan['argmix_task_plan_ref'])['tasks']
                final_waits = final_argmix['stage_actual_wait_refs']
                need(len(final_waits) == len(argmix_tasks) == 15 and len({v['path'] for v in final_waits}) == 15, 'All fifteen genuine ARGMix stage waits')
                for reference, task in zip(final_waits, argmix_tasks):
                    stage = waited(reference)
                    need(stage['task_id'] == task['id'] and stage['plan_ref'] == plan['argmix_task_plan_ref'], 'All final ARGMix stage identities')
                waited(r.ref(r.ROOT/'experiments/results/joint_argmix_20261029/scores/actual_wait.json'))
                waited(r.ref(r.path(BASE)/'european_four/chain_records_v1/paired_score/actual_wait.json'))
                state['all_computations_complete']=True
                immutable(folder/'ready_for_final_review.json',state)
                r.save(folder/'status.json',state)
                print('All three comparisons computed and saved. Independent final review remains required.',flush=True)
                return
            compact=json.dumps(state,sort_keys=True)
            if compact!=previous:
                r.save(folder/'status.json',state)
                print(json.dumps(dict(ARGMix_admitted=admission['admitted'],GPU_chains=list(completed),comparisons=len(artifacts))),flush=True)
                previous=compact
            time.sleep(60)


if __name__=='__main__':main()
