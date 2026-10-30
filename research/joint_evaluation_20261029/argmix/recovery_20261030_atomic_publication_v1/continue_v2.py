"""Continue unchanged remaining stages with atomic receipt gates after a childless handoff."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from research.joint_evaluation_20261029.argmix.recovery_20261030_gpu_capacity_v1 import continue_v1 as prior
from research.joint_evaluation_20261029.argmix.recovery_20261030_atomic_publication_v1.atomic_v1 import atomic_record
from research.joint_evaluation_20261029.argmix.recovery_20261030_atomic_publication_v1.stage_v2 import stage

c=prior.c
HERE=c.BASE/'recovery_20261030_atomic_publication_v1'
PREFIX_IDS=['joint_argmix.train_cache','joint_argmix.gpu_qualification','joint_argmix.train']


def validate(reference, *, require_stopped=False):
    spec=c.read(reference)
    c.require(spec['schema']=='argmix_atomic_publication_continuation_v1' and spec['root_authorized'] is True,
              'Explicit atomic publication continuation authority')
    c.require(spec['automatic_retries_authorized'] is False and spec['scientific_changes'] is False,
              'Only publication mechanics may change')
    for source in spec['source_refs']:c.verify(source)
    old_spec,manifest,plan,prefix,old_binding,host_minimum=prior.validate(spec['prior_capacity_continuation_ref'])
    c.require([task['id'] for task in plan['tasks'][:3]]==PREFIX_IDS and len(plan['tasks'])==15,
              'Exact original completed three-stage prefix and fifteen-stage plan')
    c.require(spec['prefix_actual_wait_refs'][:2]==prefix,'Original cache and qualification wait references remain identical')
    fit_ref=spec['prefix_actual_wait_refs'][2]
    task=plan['tasks'][2]
    folder=Path(manifest['records_directory'])
    c.require(fit_ref==c.ref(folder/task['id']/'actual_wait.json'),'Exact original fit wait')
    fit=c.read(fit_ref)
    c.require(fit['task_id']==task['id'] and fit['command']==task['argv_by_physical_gpu']['3'] and
              fit['plan_ref']==manifest['plan_ref'] and fit['freeze_ref']==manifest['freeze_ref'] and
              fit['recovery_binding']==old_binding and fit['actual_wait'] is True and fit['actual_returncode']==0 and
              fit['owned_process_group_absent'] is True and fit['status']=='succeeded' and fit['products_complete'] is True and
              c.historical_session_absent(fit['pid'],fit['process_start_ticks']), 'Exact genuine completed native fit prefix')
    product=c.completion_path(task)
    c.require(fit['result_ref']==c.ref(product) and c.read(c.ref(product.parent/'actual_wait.json'))==fit,
              'Fit product and colocated successful wait are unchanged')
    c.require_wait(product,task['id'])
    previous=c.read(spec['predecessor_started_ref'])
    c.require(previous['continuation_ref']==spec['prior_capacity_continuation_ref'] and
              {key:previous[key] for key in ('pid','process_start_ticks')}==spec['predecessor_controller'],
              'Exact capacity controller predecessor identity')
    preserved=c.read(spec['preserved_predecessor_state_ref'])
    c.require(preserved['controller_pid']==previous['pid'] and preserved['status']=='waiting' and
              preserved['operational_continuation_ref']==spec['prior_capacity_continuation_ref'] and
              preserved['plan_ref']==manifest['plan_ref'], 'Preserved childless waiting predecessor state')
    if require_stopped:
        c.require(c.historical_session_absent(previous['pid'],previous['process_start_ticks']),
                  'Root must stop and drain only the verified childless predecessor first')
        c.require(c.ref(folder/'state.json')['sha256']==spec['preserved_predecessor_state_ref']['sha256'],
                  'Original waiting state changed; re-audit any intervening work')
    for remaining in plan['tasks'][3:]:
        c.require(not (folder/remaining['id']).exists() and not c.completion_path(remaining).exists(),
                  'A remaining stage already started or produced an unowned product; no automatic replay')
    c.require(not (folder/'completed.json').exists() and not (folder/'gpu_chain_completed.json').exists(),
              'No existing chain receipt may be overwritten')
    binding=dict(old_binding,atomic_publication_continuation_ref=reference)
    return spec,old_spec,manifest,plan,list(spec['prefix_actual_wait_refs']),binding,host_minimum


def chain_record(manifest,spec,reference,binding,waits,tasks,*,gpu=False):
    value=prior.chain_record(manifest,spec['prior_capacity_continuation_ref'],binding,waits,tasks,gpu=gpu)
    value['atomic_publication_continuation_ref']=reference
    return value


def run(reference):
    spec,old_spec,manifest,plan,prefix,binding,host_minimum=validate(reference,require_stopped=True)
    folder=Path(manifest['records_directory'])
    with (folder/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate(reference,require_stopped=True)
        c.require(not (HERE/'started.json').exists(),'No automatic continuation replay')
        ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19]
        atomic_record(HERE/'started.json',dict(pid=os.getpid(),process_start_ticks=ticks,
            continuation_ref=reference,predecessor_controller=spec['predecessor_controller'],
            preserved_predecessor_state_ref=spec['preserved_predecessor_state_ref'],started_utc=c.now()),HERE,immutable=True)
        previous=None
        def status(state,reason,missing=None):
            nonlocal previous
            key=(state,reason,tuple(missing or []))
            if key!=previous:
                atomic_record(folder/'state.json',dict(status=state,reason=reason,
                    manifest_ref=binding['manifest_ref'],plan_ref=manifest['plan_ref'],model_id=c.MODEL_ID,
                    controller_pid=os.getpid(),operational_continuation_ref=spec['prior_capacity_continuation_ref'],
                    atomic_publication_continuation_ref=reference,missing_dependencies=missing or [],updated_utc=c.now()),folder)
                print(state+': '+reason,flush=True)
                previous=key
        _,protocol=c.frozen(manifest['freeze_ref']['path'])
        admission_path=c.ROOT/protocol['production_admission_path']
        admission=c.cache_admission(admission_path,protocol)
        admission_ref=c.ref(admission_path)
        external={task['id']:task for task in c.read(manifest['original_effective_plan_ref'])['tasks']}
        internal={task['id']:task for task in plan['tasks']}
        all_waits,gpu_waits=list(prefix),list(prefix[1:])
        gpu_tasks=[task for task in plan['tasks'] if task['kind']=='gpu']
        gpu_lock=None
        try:
            for task in plan['tasks'][3:]:
                while True:
                    missing=c.dependency_state(task,internal,external)
                    if not missing:break
                    status('waiting','Successful actual producer waits pending',missing);time.sleep(20)
                c.verify(admission_ref)
                for source in spec['source_refs']:c.verify(source)
                while c.host_available_bytes()<host_minimum:
                    status('waiting','Amended family allowance plus64GiB host headroom pending');time.sleep(20)
                if task['kind']=='gpu':
                    gpu_lock=Path(plan['GPU_mutex_path']).open('a+b')
                    while True:
                        try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:
                            status('waiting','Another authorized root GPU chain holds the shared mutex');time.sleep(20)
                    while True:
                        uuid,available=c.gpu_capacity(3,prior.gpu_requirement(old_spec,task))
                        if available and c.host_available_bytes()>=host_minimum:break
                        status('waiting','Empty GPU3 with measured reservation plus2GiB free headroom pending');time.sleep(20)
                    command=task['argv_by_physical_gpu']['3']
                    environment=dict(CUDA_VISIBLE_DEVICES=uuid,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                                     MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                else:command,environment=task['argv'],task['environment']
                status('running',task['id'])
                waited=stage(task,command,environment,folder,manifest['plan_ref'],manifest['freeze_ref'],
                             admission['cache_max_bytes'],gpu_lock,binding)
                all_waits.append(waited)
                if task['kind']=='gpu':
                    gpu_lock.close();gpu_lock=None
                    gpu_waits.append(waited)
                    if task['id']==gpu_tasks[-1]['id']:
                        atomic_record(folder/'gpu_chain_completed.json',
                            chain_record(manifest,spec,reference,binding,gpu_waits,gpu_tasks,gpu=True),folder,immutable=True)
            atomic_record(folder/'completed.json',
                chain_record(manifest,spec,reference,binding,all_waits,plan['tasks']),folder,immutable=True)
            status('completed','All original fifteen stages and genuine waits complete')
        except BaseException as error:
            status('failed',str(error));raise
        finally:
            if gpu_lock is not None:gpu_lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuation',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    reference=c.ref(args.continuation)
    if args.check_only:
        spec,old_spec,manifest,plan,prefix,binding,host_minimum=validate(reference)
        print(json.dumps(dict(valid=True,prefix_waits=prefix,remaining_stages=len(plan['tasks'][3:]),
            plan_ref=manifest['plan_ref'],manifest_ref=binding['manifest_ref'],
            host_minimum_bytes=host_minimum,minimum_free_gpu_bytes=old_spec['minimum_free_gpu_bytes'],
            predecessor_stop_still_required=not c.historical_session_absent(spec['predecessor_controller']['pid'],
                spec['predecessor_controller']['process_start_ticks']),production_launches=0,signals=0)),flush=True)
    else:
        import setproctitle
        setproctitle.setproctitle('AI Auto Research Pipeline Test')
        run(reference)


if __name__=='__main__':main()
