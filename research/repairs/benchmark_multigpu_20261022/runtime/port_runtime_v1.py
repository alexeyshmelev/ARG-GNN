"""Prepare ordinary-source parallel runtime successors without dispatch.

All original GPU5 runtime and scientific source files stay immutable.
"""
from pathlib import Path
import difflib
import hashlib
import json

OLD=Path('research/repairs/benchmark_resume_20261021/gpu')
NEW=Path('research/repairs/benchmark_multigpu_20261022/runtime')
PACKAGE='research.repairs.benchmark_multigpu_20261022.runtime'
lineage=[]


def port(name,replacements):
    source=OLD/name
    text=source.read_text()
    for before,after in replacements:
        if before not in text:raise ValueError('Missing port anchor '+name+': '+before)
        text=text.replace(before,after)
    destination=NEW/name
    with destination.open('x') as stream:stream.write(text)
    difference=''.join(difflib.unified_diff(source.read_text().splitlines(True),text.splitlines(True),fromfile=str(source),tofile=str(destination)))
    diff=NEW/'lineage'/(name+'.diff');diff.parent.mkdir(exist_ok=True)
    with diff.open('x') as stream:stream.write(difference)
    lineage.append(dict(original=str(source),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),diff_path=str(diff)))


port('readable_workflow_v1.py',[
    ('contracts_gpu5_v1','contracts_parallel_v1'),('queue_gpu5_v1','queue_gpu_v1'),
    ('research.repairs.benchmark_resume_20261021.gpu.readable_workflow_v1',PACKAGE+'.readable_workflow_v1'),
    ('research/repairs/benchmark_resume_20261021/gpu/readable_workflow_v1.py',str(NEW/'readable_workflow_v1.py')),
    ('approved_readable_workflow_ref','approved_parallel_workflow_ref'),
    ('readable_workflow_drain_ref','parallel_workflow_drain_ref'),
    ('plan.get("schema") == SCHEMA and plan.get("physical_gpu") == 5',
     'plan.get("schema") == SCHEMA and plan.get("physical_gpu") == 5 and plan.get("physical_gpus") == [2,3,5]'),
    ('"Frozen one-GPU workflow required"','"Frozen explicit GPU2/3/5 parallel workflow required"'),
    ('            if task["resource"]=="gpu0" or task.get("gpu_after_preparation") is True:\n',
     '            if task["resource"]=="gpu0" or task.get("gpu_after_preparation") is True:\n                shared_gpu.target(task)\n'),
    ('physicalGPU5','physicalGPU2/3/5'),('"GPU5 lease or memory is occupied"','"Assigned GPU lease or memory is occupied"'),
    ('from .queue_reporting_v1 import publish','from research.repairs.benchmark_resume_20261021.gpu.queue_reporting_v1 import publish'),
    ('                            deps=[state["tasks"][i]["status"] for i in task["dependencies"]]\n',
     '                            entry.pop("waiting_reason",None)\n                            deps=[state["tasks"][i]["status"] for i in task["dependencies"]]\n'),
    ('                            if not all(s=="succeeded" for s in deps):continue',
     '                            if not all(s=="succeeded" for s in deps):\n                                entry["waiting_reason"]="Required predecessors have not completed";continue'),
])
port('queue_support_v1.py',[('contracts_gpu5_v1','contracts_parallel_v1'),('queue_gpu5_v1','queue_gpu_v1')])
port('queue_execute_v1.py',[
    ('contracts_gpu5_v1','contracts_parallel_v1'),('queue_gpu5_v1','queue_gpu_v1'),
    ('gpu.GPU5_UUID','gpu.target(task)[1]'),
    ('research.repairs.benchmark_resume_20261021.gpu.queue_native_worker_v1',PACKAGE+'.queue_native_worker_v1'),
    ("ARGGNN_PHYSICAL_GPU='5'","ARGGNN_PHYSICAL_GPU=str(gpu.target(task)[0])"),
    ('dict(physical_gpu=5,gpu_uuid=', 'dict(physical_gpu=gpu.target(task)[0],gpu_uuid='),
])
port('queue_native_admission_v1.py',[('queue_gpu5_v1','queue_gpu_v1')])
port('queue_native_worker_v1.py',[
    ('shared GPU5 route','per-task GPU2/3/5 route'),
    ('from .shared_gpu5_v1 import GPU5_UUID','from .shared_gpu_v1 import validate_visible_target'),
    ('    api.need(os.environ.get("CUDA_VISIBLE_DEVICES")==GPU5_UUID and os.environ.get("ARGGNN_PHYSICAL_GPU")=="5",\n             "Only admitted physical GPU5 may become visible")',
     '    validate_visible_target()'),
])
with (NEW/'initial_port_lineage_v1.json').open('x') as stream:json.dump(lineage,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(created=len(lineage),directory=str(NEW)),sort_keys=True))
