"""Concise status from scheduler metadata, with no scientific result inference."""
from datetime import datetime,timezone
import os
import fcntl
from pathlib import Path
from arggnn.benchmarking import contracts as c

START='<!-- readable-parallel-queue:start -->'
END='<!-- readable-parallel-queue:end -->'


def tree_progress(output):
    """Use the newest bounded progress file, including real training stages."""
    choices=[]
    for base in (output,output/'native'):
        for filename,phase in (('training_live.json','training'),('preparation_live.json','preparation')):
            path=base/filename
            try:
                info=path.stat()
                if path.is_symlink() or info.st_size>65536:continue
                sample=c.read_json(path)
                if not isinstance(sample,dict) or sample.get('schema')!='treeloom_stage_progress_v1':continue
                choices.append((info.st_mtime_ns,phase,path,sample))
            except (OSError,ValueError,KeyError):continue
    if not choices:return dict(stage='live; bounded progress metadata pending')
    modified,phase,path,sample=max(choices,key=lambda item:item[0])
    latest=sample.get('latest',{})
    active=sample.get('active_stages',[])
    mark=active[-1].get('stage') if active else latest.get('preparation_stage',latest.get('last_mark','running'))
    result=dict(stage='TreeLoom '+phase+': '+str(mark),phase=phase,parent_id=sample.get('parent_id'),
        latest=latest,active_stages=active,progress_path=str(path.relative_to(c.ROOT)),
        progress_modified_ns=modified,peak_RSS_bytes=sample.get('peak_RSS_bytes'))
    for base in (output,output/'native'):
        checkpoint=base/'latest_checkpoint.json'
        try:
            if checkpoint.is_symlink() or checkpoint.stat().st_size>65536:continue
            current=c.read_json(checkpoint)
            result['completed_parent_updates']=current['complete_parent_updates']
            result['checkpoint_epoch']=current.get('epoch')
            result['checkpoint_parent_id']=current.get('parent_id')
        except (OSError,ValueError,KeyError):continue
    return result


def local_status(plan,state):
    rows={}
    for task in plan['tasks']:
        if state['tasks'][task['id']]['status']!='running' or task['resource']=='external':continue
        output=c.contained_path(task['output_directory'])
        row=tree_progress(output)
        if row.get('phase'):rows[task['id']]=row
    return rows


def standalone_status(plan):
    from research.repairs.benchmark_recovery_20261020.scheduler.queue_adoption_v1 import generation_absent
    rows={}
    for task in plan['tasks']:
        binding=task.get('standalone_predecessor')
        if not binding or generation_absent(binding['supervisor_identity']):continue
        output=c.contained_path(binding['output_directory'])
        row=dict(kind=binding.get('kind'),supervisor_generation_present=True)
        try:
            if binding.get('kind')=='native_contrastive':
                sample=c.read_json(output/'native/latest_batch_memory.json')
                row.update(stage='full native contrastive batches',latest_batch=sample.get('batch'),
                    expected_updates=c.read_ref(binding['request_ref'])['call']['spec']['expected_updates'])
            else:
                row.update(tree_progress(output))
        except (OSError,ValueError,KeyError):row['stage']='live; bounded progress metadata pending'
        rows[task['id']]=row
    return rows


def publish(plan,reference,state,save):
    counts={status:sum(row['status']==status for row in state['tasks'].values())
            for status in ('pending','running','succeeded','failed','blocked','unknown')}
    updated=datetime.now(timezone.utc).isoformat()
    score_refs={task['id']:state['tasks'][task['id']]['result_ref'] for task in plan['tasks']
        if 'score' in task['id'].lower() and state['tasks'][task['id']]['status']=='succeeded'
        and state['tasks'][task['id']].get('result_ref')}
    for result_ref in score_refs.values():c.read_ref(result_ref)
    standalones=standalone_status(plan)
    local=local_status(plan,state)
    deferred=[dict(task_id=item['task']['id'],reason=item['reason']) for item in plan.get('deferred_optional_archive_tasks',[])]
    receipt=dict(live_local_stages=local,live_standalone_stages=standalones,deferred_optional_archives=deferred,completed_score_result_refs=score_refs,plan_ref=reference,updated_utc=updated,run_status=state.get('run_status'),counts=counts,
        waiting_reasons=state.get('waiting_reasons',{}),new_ancestry_accuracy=None,scientific_scores_claimed=False)
    save(c.contained_path(plan['records_directory'])/'readable_progress.json',receipt)
    active=[identifier for identifier,row in state['tasks'].items() if row['status']=='running']
    content='\n'.join([START,'### Current parallel queue',
        'Updated '+updated+'. Status: '+str(state.get('run_status'))+'.',
        ', '.join(str(value)+' '+key for key,value in counts.items())+'.',
        'Local active: '+(', '.join(active) if active else 'waiting for dependencies or capacity')+'.',
        'Live training stages: '+('; '.join(key+': '+row['stage']+((' ('+str(row['completed_parent_updates'])+'/600 saved parent updates)') if 'completed_parent_updates' in row else '') for key,row in local.items()) if local else 'bounded stage metadata pending')+'.',
        'Standalone work: '+('; '.join(key+': '+row['stage']+((' '+str(row['latest_batch'])+'/'+str(row['expected_updates'])) if 'latest_batch' in row else '') for key,row in standalones.items()) if standalones else 'no live standalone generations')+'.',
        'The earlier v9 queue block is a historical snapshot. This block describes the active successor.',
        ('Verified completed score artifacts: '+', '.join('['+key+']('+ref['path']+')' for key,ref in score_refs.items())+'.'
         if score_refs else 'Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.'),
        ('One optional external AP001 archive is deferred with its full retained storage charge and zero credit.' if deferred else ''),
        '[Current queue metadata](research/benchmark_parallel_current.json).',END])
    lock_path=c.ROOT/'.runtime/benchmarking/workflow_reporting/reporting.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        save(c.ROOT/'research/benchmark_parallel_current.json',receipt)
        readme=c.ROOT/'README.md';text=readme.read_text()
        if START in text:
            begin=text.index(START);end=text.index(END,begin)+len(END)
            text=text[:begin]+content+text[end:]
        else:
            first=text.find('\n')
            text=text[:first+1]+'\n'+content+'\n'+text[first+1:]
        temporary=readme.with_name('README.md.readable.'+str(os.getpid()))
        with temporary.open('x') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,readme)
        from research.repairs.benchmark_recovery_20261020.reporting.git_progress_v1 import push_reports
        push_reports()
    return receipt
