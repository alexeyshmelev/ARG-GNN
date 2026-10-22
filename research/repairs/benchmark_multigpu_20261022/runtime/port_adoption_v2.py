"""Create the narrow completed-adoption/runtime/reporting v2 chain once."""
from pathlib import Path

HERE=Path('research/repairs/benchmark_multigpu_20261022/runtime')
PACKAGE='research.repairs.benchmark_multigpu_20261022.runtime'


def save(name,text):
    with (HERE/name).open('x') as handle:handle.write(text)


source=(HERE/'readable_workflow_v1.py').read_text()
source=source.replace(PACKAGE+'.readable_workflow_v1',PACKAGE+'.readable_workflow_v2')
source=source.replace(str(HERE/'readable_workflow_v1.py'),str(HERE/'readable_workflow_v2.py'))
source=source.replace('from . import queue_execute_v1 as runtime','from . import queue_execute_v2 as runtime')
source=source.replace('from research.repairs.benchmark_resume_20261021.gpu.queue_reporting_v1 import publish','from .queue_reporting_v2 import publish')
helper='''

def authenticated_adopted_completed(plan):
    """Bind each historical exemption to its immutable succeeded source state."""
    mapping=plan.get('adopted_completed_task_fingerprints',{})
    need(isinstance(mapping,dict),'Explicit adopted completion fingerprint mapping required')
    if not mapping:return {}
    snapshot=c.read_ref(plan['adopted_state_snapshot_ref'])
    original=c.read_ref(snapshot['plan_ref'])
    current={task['id']:task for task in plan['tasks']}
    previous={task['id']:task for task in original['tasks']}
    authenticated={}
    for identifier,fingerprint in mapping.items():
        need(identifier in current and identifier in previous
             and c.fingerprint(current[identifier])==fingerprint==c.fingerprint(previous[identifier]),
             'Historical exemption requires exact original/current task fingerprint')
        entry=snapshot['tasks'][identifier]
        need(entry.get('status')=='succeeded' and isinstance(entry.get('result_ref'),dict),
             'Historical exemption requires a genuinely recorded succeeded snapshot entry')
        authenticated[identifier]=entry
    return authenticated


def validate_adopted_state(plan,state):
    for identifier,original in authenticated_adopted_completed(plan).items():
        entry=state['tasks'][identifier]
        need(entry.get('status')=='succeeded' and entry.get('result_ref')==original['result_ref'],
             'An adopted completed task cannot become pending or change its result')


def initial_state(plan,reference):
    authenticated_adopted_completed(plan)
    state=support.initial_state(plan,reference)
    validate_adopted_state(plan,state)
    return state
'''
anchor='\n\ndef save(path, value, *, exclusive=False):'
if anchor not in source:raise ValueError('Missing helper anchor')
source=source.replace(anchor,helper+anchor,1)
source=source.replace('    need(len(ids) == len(set(ids)), "Unique workflow task IDs required")',
    '    need(len(ids) == len(set(ids)), "Unique workflow task IDs required")\n    adopted=authenticated_adopted_completed(plan)',1)
source=source.replace('                shared_gpu.target(task)','                if task["id"] not in adopted:shared_gpu.target(task)',1)
source=source.replace('    task = job["task"]\n    need(task in plan["tasks"]',
    '    task = job["task"]\n    need(task["id"] not in authenticated_adopted_completed(plan), "Historical completed tasks cannot execute again")\n    need(task in plan["tasks"]',1)
source=source.replace('        state=support.initial_state(plan,ref)','        state=initial_state(plan,ref)',1)
source=source.replace('    if not execute_jobs:return state','    validate_adopted_state(plan,state)\n    if not execute_jobs:return state',1)
save('readable_workflow_v2.py',source)
source=(HERE/'queue_execute_v1.py').read_text().replace(PACKAGE+'.queue_native_worker_v1',PACKAGE+'.queue_native_worker_v2')
save('queue_execute_v2.py',source)
source=(HERE/'queue_native_worker_v1.py').read_text().replace('from .queue_native_admission_v1 import validate_worker_lease','from .queue_native_admission_v2 import validate_worker_lease')
save('queue_native_worker_v2.py',source)
source=(HERE/'queue_native_admission_v1.py').read_text().replace('from . import readable_workflow_v1 as workflow','from . import readable_workflow_v2 as workflow')
save('queue_native_admission_v2.py',source)
source=Path('research/repairs/benchmark_resume_20261021/gpu/queue_reporting_v1.py').read_text()
source=source.replace('readable-recovery-queue:','readable-parallel-queue:')
source=source.replace('benchmark_readable_current.json','benchmark_parallel_current.json')
source=source.replace('### Current recovery queue','### Current parallel queue')
save('queue_reporting_v2.py',source)
print('Created five fresh v2 operational files; every v1 file remains unchanged')
