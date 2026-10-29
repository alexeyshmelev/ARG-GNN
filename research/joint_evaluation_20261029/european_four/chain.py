"""Root-owned direct4 chain with actual process waits, capacity checks and CPU score continuation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil
from arggnn.benchmarking.paper_campaign import processes
from .common import BASE,ROOT,RESULTS,MODEL_ID,GPUS,freeze,ref,read,write,check_disk,require


def gpu_snapshot():
    value=subprocess.check_output(['nvidia-smi','--id='+GPUS[4],'--query-gpu=uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    uuid,total,free=[s.strip() for s in value.strip().split(',')]
    require(uuid==GPUS[4],'Physical GPU4 identity')
    return dict(uuid=uuid,total_bytes=int(total)*2**20,free_bytes=int(free)*2**20)


def family_rss(pid):
    try:
        process=psutil.Process(pid);members=[process]+process.children(recursive=True)
    except psutil.NoSuchProcess:return 0
    total=0
    for member in members:
        try:total+=member.memory_info().rss
        except psutil.NoSuchProcess:pass
    return total


def verify_plan(filename):
    plan=read(ref(filename))
    require(plan['schema']=='european_direct4_chain_plan_v1' and plan['physical_gpu']==4,'Exact root chain plan')
    for r in plan['source_refs']:require(ref(ROOT/r['path'])==r,'Chain source changed')
    require(ref(BASE/'freeze_v1.json')==plan['freeze_ref'],'Exact reviewed scientific freeze')
    freeze()
    require(plan['family_ram_cap_bytes']==64*2**30 and plan['output_cap_bytes']==128*2**20,'Reserved family and output bounds')
    return plan,ref(filename)


def capacity(plan,gpu):
    ram=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if ram<plan['family_ram_cap_bytes']+plan['host_headroom_bytes']:return False
    return not gpu or gpu_snapshot()['free_bytes']>=plan['gpu_peak_budget_bytes']+plan['shared_gpu_headroom_bytes']


def wait_capacity(plan,folder,gpu):
    while not capacity(plan,gpu):
        write(folder/'capacity_wait.json',dict(waiting=True,gpu=gpu,pid=os.getpid(),time_unix=time.time()),replace=(folder/'capacity_wait.json').exists())
        time.sleep(20)


def stage(plan,plan_ref,folder,name,command,products,gpu,lock_fd):
    require(ref(ROOT/plan_ref['path'])==plan_ref,'Immutable chain plan during execution')
    for source in plan['source_refs']:require(ref(ROOT/source['path'])==source,'Immutable chain source during execution')
    directory=folder/name;directory.mkdir(parents=True,exist_ok=True)
    terminal=directory/'actual_wait.json'
    if terminal.exists():
        row=read(ref(terminal))
        require(row['plan_ref']==plan_ref and row['command']==command and row['actual_wait'] and row['actual_returncode']==0 and row['products_complete'] and row['owned_process_group_absent'],'Prior exact successful wait required')
        require(not processes.session_members(row['pid']),'Prior worker remains live')
        require([r['path'] for r in row['products']]==[str(Path(p).relative_to(ROOT)) for p in products],'Prior product list identity')
        for r in row['products']:require(ref(ROOT/r['path'])==r,'Prior product changed')
        return ref(terminal)
    require(not (directory/'started.json').exists(),'Prior unresolved start may not be replayed')
    for path in products:require(not Path(path).exists(),'Fresh stage output required')
    wait_capacity(plan,folder,gpu);check_disk()
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=GPUS[4] if gpu else '',PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    worker=None;code=None;problem=None;peak=0
    with (directory/'worker.log').open('xb') as log:
        try:
            worker=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
            ticks=(Path('/proc')/str(worker.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            row=dict(plan_ref=plan_ref,pid=worker.pid,process_start_ticks=ticks,command=command,started_unix=time.time(),task_id='european_four_joint.'+name,model_id=MODEL_ID)
            write(directory/'started.json',row)
            while True:
                try:
                    code=worker.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    peak=max(peak,family_rss(worker.pid))
                    require(peak<=plan['family_ram_cap_bytes'],'Owned stage family exceeds64GiB')
                    check_disk()
            while not processes.drain(worker):time.sleep(1)
        except BaseException as error:
            problem=repr(error)
            if worker is not None:
                processes.drain(worker,terminate=True);code=worker.wait()
            else:raise
    product_refs=[]
    if code==0 and problem is None:
        try:product_refs=[ref(path) for path in products]
        except BaseException as error:problem=repr(error)
    row.update(actual_wait=True,actual_returncode=code,owned_process_group_absent=not processes.session_members(worker.pid),
        products=product_refs,products_complete=code==0 and problem is None,observed_peak_family_rss_bytes=peak,finished_unix=time.time())
    if len(product_refs)==1:row['result_ref']=product_refs[0]
    if problem:row['error']=problem
    result=write(terminal,row)
    require(row['products_complete'] and row['owned_process_group_absent'],'Stage failed; preserve receipt and stop: '+name)
    return result


def prefix(module):return [str(ROOT/'.runtime/env/bin/python3.11'),'-B','-s','-u','-m','research.joint_evaluation_20261029.european_four.'+module]


def gpu_chain(plan,plan_ref,folder,lock_fd):
    waits=[]
    for spec in plan['GPU_stages']:
        command=prefix(spec['module'])+spec['arguments']
        receipt=stage(plan,plan_ref,folder,spec['name'],command,[ROOT/p for p in spec['products']],True,lock_fd);waits.append(receipt)
        if spec['name']=='qualification':
            q=read(ref(RESULTS/'gpu_qualification_v1.json'))
            require(q['passed'] and q['freeze_ref']==plan['freeze_ref'] and q['physical_gpu']==4,'Exact GPU4 qualification')
            require(max(r['peak_reserved_bytes'] for r in q['groups'])<=plan['gpu_peak_budget_bytes'],'Largest TRAIN batches exceed admitted16GiB')
    return write(folder/'gpu_chain_product.json',dict(complete=True,model_id=MODEL_ID,completion_key='european_four_joint',plan_ref=plan_ref,
        scientific_freeze_ref=plan['freeze_ref'],stage_actual_wait_refs=waits,all_five_TEST_predictions_complete=True,pairedscore_pending=True,completed_unix=time.time()))


def cnn_products(plan):
    state_path=ROOT/plan['cnn_state_path']
    if not state_path.exists():return None
    state=json.loads(state_path.read_text())
    require(state['plan_ref']['path']==plan['cnn_plan_path'],'Exact CNN4 successor plan path')
    concrete=read(state['plan_ref'])
    require(concrete['handoff_config_ref']==plan['cnn_handoff_config_ref'],'Exact authenticated CNN4 idle handoff')
    read(plan['cnn_handoff_config_ref'])
    products={}
    for parent,spec in plan['CNN_predictions'].items():
        row=state['tasks'][spec['task_id']]
        require(row['status'] not in ('failed','blocked','unknown'),'CNN4 predecessor failed; preserve dependency failure')
        if row['status']!='succeeded':return None
        wait_path=ROOT/spec['actual_wait_path'];require(wait_path.exists(),'CNN4 stage needs genuine wait receipt')
        wait=read(ref(wait_path))
        require(wait['actual_wait'] and wait['actual_returncode']==0 and wait['owned_process_group_absent'] and wait['result_ref']==row['result_ref'] and not processes.session_members(wait['pid']),'CNN4 producer successfully waited and drained')
        result=read(wait['result_ref']);require(result['schema']=='paper_campaign_result_v1' and result['scientific_success'],'CNN4 scientific success')
        product=read(result['value'])
        require(product['schema']=='ap_four_predictions_v1' and product['parent_id']==parent and product['rows']==8000 and product['condition']=='ancestralpaths_historical_cnn_four_no_baa' and product['probability_merging'] is False,'Complete direct CNN4 prediction product')
        products[parent]=result['value']
    return products


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--gpu-chain-only',action='store_true');parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args();plan,plan_ref=verify_plan(args.plan);folder=ROOT/plan['records_directory']
    if args.inspect:
        print(json.dumps(dict(plan_ref=plan_ref,GPU4=gpu_snapshot(),host_capacity_available=capacity(plan,False),stages=len(plan['GPU_stages']),workers_launched=0,signals_sent=0)));return
    folder.mkdir(parents=True,exist_ok=True)
    def interrupted(signum,frame):raise RuntimeError('Explicit controller termination')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (folder/('gpu_chain.lock' if args.gpu_chain_only else 'controller.lock')).open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.gpu_chain_only:
            gpu_chain(plan,plan_ref,folder,lock.fileno());return
        require(not (folder/'completed.json').exists(),'Chain already completed')
        command=prefix('chain')+['--plan',args.plan,'--gpu-chain-only']
        gpu_wait=stage(plan,plan_ref,folder,'gpu_chain',command,[folder/'gpu_chain_product.json'],False,lock.fileno())
        completed_gpu=read(ref(folder/'gpu_chain_product.json'))
        require(completed_gpu['complete'] and completed_gpu['plan_ref']==plan_ref and completed_gpu['all_five_TEST_predictions_complete'],'Exact complete GPU chain before release')
        while True:
            products=cnn_products(plan)
            if products is not None:break
            write(folder/'status.json',dict(status='waiting_for_CNN4',GPU_chain_actual_wait_ref=gpu_wait,pid=os.getpid(),time_unix=time.time()),replace=(folder/'status.json').exists())
            time.sleep(20)
        mapping=folder/'cnn_prediction_products.json'
        if mapping.exists():require(json.loads(mapping.read_text())==products,'Immutable CNN4 product mapping')
        else:write(mapping,products)
        score_wait=stage(plan,plan_ref,folder,'paired_score',prefix('score')+['--cnn-predictions',str(mapping)],
            [RESULTS/'paired_score_disclosure.json'],False,lock.fileno())
        write(folder/'completed.json',dict(complete=True,model_id=MODEL_ID,plan_ref=plan_ref,GPU_chain_actual_wait_ref=gpu_wait,paired_score_actual_wait_ref=score_wait))
        print(json.dumps(dict(complete=True,GPU_chain_actual_wait_ref=gpu_wait,paired_score_actual_wait_ref=score_wait)),flush=True)


if __name__=='__main__':main()
