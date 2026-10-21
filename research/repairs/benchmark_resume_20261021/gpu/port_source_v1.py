"""Create reviewed, ordinary-source GPU5 operational successors once.

This source migration retains frozen science modules and records a plain diff.
It does not execute any generated source or launch a benchmark.
"""
from pathlib import Path
import hashlib
import difflib
import json

OLD = Path('research/repairs/benchmark_recovery_20261020')
NEW = Path('research/repairs/benchmark_resume_20261021/gpu')
PACKAGE = 'research.repairs.benchmark_resume_20261021.gpu'
lineage = []

def write(name, text, original=None):
    path = NEW / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(text)
    row = dict(path=str(path), sha256=hashlib.sha256(text.encode()).hexdigest())
    if original is not None:
        source = original.read_text()
        row.update(original_path=str(original), original_sha256=hashlib.sha256(source.encode()).hexdigest())
        diff = ''.join(difflib.unified_diff(source.splitlines(True), text.splitlines(True),
                                          fromfile=str(original), tofile=str(path)))
        diff_path = NEW / 'lineage' / (name.replace('/', '__') + '.diff')
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with diff_path.open('x') as stream:
            stream.write(diff)
        row['diff_path'] = str(diff_path)
    lineage.append(row)

def port(source, target, replacements):
    text = source.read_text()
    for before, after in replacements:
        if before not in text:
            raise ValueError('Missing migration anchor in ' + str(source) + ': ' + before)
        text = text.replace(before, after)
    write(target, text, source)

contracts_import = ('from arggnn.benchmarking import contracts as c', 'from . import contracts_gpu5_v1 as c')
port(OLD/'scheduler/readable_workflow_v1.py', 'readable_workflow_v1.py', [
    contracts_import,
    ('from arggnn.benchmarking.execution import GPU0_UUID, TITLE', 'from arggnn.benchmarking.execution import TITLE'),
    ('queue_gpu0_v1', 'queue_gpu5_v1'),
    ('research.repairs.benchmark_recovery_20261020.scheduler.readable_workflow_v1', PACKAGE+'.readable_workflow_v1'),
    ('research/repairs/benchmark_recovery_20261020/scheduler/readable_workflow_v1.py', str(NEW/'readable_workflow_v1.py')),
    ('plan.get("physical_gpu") == 0', 'plan.get("physical_gpu") == 5'),
    ('physicalGPU0', 'physicalGPU5'), ('GPU0 lease or memory', 'GPU5 lease or memory'),
    ('from .queue_reporting_v1 import publish', 'from research.repairs.benchmark_recovery_20261020.scheduler.queue_reporting_v1 import publish'),
])
port(OLD/'scheduler/queue_execute_v1.py', 'queue_execute_v1.py', [
    contracts_import, ('queue_gpu0_v1', 'queue_gpu5_v1'), ('gpu.GPU0_UUID', 'gpu.GPU5_UUID'),
    ('research.repairs.benchmark_recovery_20261020.scheduler.queue_native_worker_v1', PACKAGE+'.queue_native_worker_v1'),
    ("ARGGNN_PHYSICAL_GPU='0'", "ARGGNN_PHYSICAL_GPU='5'"), ('physical_gpu=0', 'physical_gpu=5'),
])
port(OLD/'scheduler/queue_support_v1.py', 'queue_support_v1.py', [
    contracts_import, ('queue_gpu0_v1', 'queue_gpu5_v1'),
    ('from . import queue_adoption_v1 as adoption', 'from research.repairs.benchmark_recovery_20261020.scheduler import queue_adoption_v1 as adoption'),
    ('from .native_continuation_v1 import managed_RAM, remaining_growth', 'from research.repairs.benchmark_recovery_20261020.scheduler.native_continuation_v1 import managed_RAM, remaining_growth'),
])
port(OLD/'scheduler/queue_gpu0_v1.py', 'queue_gpu5_v1.py', [
    contracts_import,
    ('from .shared_gpu0_v1 import GPU0_UUID, gpu_sample, identity', 'from .shared_gpu5_v1 import GPU5_UUID, PHYSICAL_GPU, gpu_sample, identity'),
    ('GPU0', 'GPU5'), ('--id=0', '--id=5'), ("readable_gpu0'", "readable_gpu5_resume_20261021'"),
    ('gpu_0.lock', 'gpu_5.lock'), ("=='0'", "=='5'"),
    ('from . import queue_adoption_v1 as adoption', 'from research.repairs.benchmark_recovery_20261020.scheduler import queue_adoption_v1 as adoption'),
    ("def _admission_reason_locked(plan, task, exclude_ref=None):\n", "def _admission_reason_locked(plan, task, exclude_ref=None):\n    need(plan.get('physical_gpu') == PHYSICAL_GPU, 'Only the physical GPU5 plan may enter this pool')\n"),
    ("    task = job['task']\n    POOL.mkdir", "    task = job['task']\n    validate_job_target(job)\n    POOL.mkdir"),
    ("cap_bytes=task['max_cuda_reserved_bytes'], members=[], physical_fd=physical)", "cap_bytes=task['max_cuda_reserved_bytes'], members=[], physical_fd=physical,\n                       physical_gpu=PHYSICAL_GPU, gpu_uuid=GPU5_UUID)"),
    ("    need(os.environ.get('CUDA_VISIBLE_DEVICES')", "    validate_job_target(job)\n    need(os.environ.get('CUDA_VISIBLE_DEVICES')"),
    ("row = json.loads(os.pread(fd, 65536, 0))\n", "row = json.loads(os.pread(fd, 65536, 0))\n                validate_record_target(row)\n"),
    ("        held,named=os.fstat(fd)", "        validate_record_target(row)\n        held,named=os.fstat(fd)"),
    ("        need(row['job_ref']==ref and row['supervisor_identity']", "        validate_record_target(row)\n        need(row['job_ref']==ref and row['supervisor_identity']"),
])
port(OLD/'scheduler/queue_native_admission_v1.py', 'queue_native_admission_v1.py', [('queue_gpu0_v1','queue_gpu5_v1')])
port(OLD/'scheduler/queue_native_worker_v1.py', 'queue_native_worker_v1.py', [
    ('shared GPU0 route', 'shared GPU5 route'),
    ('from arggnn.benchmarking import american_competitors as api', 'from arggnn.benchmarking import american_competitors as api\nfrom .shared_gpu5_v1 import GPU5_UUID'),
    ('api.GPU0_UUID', 'GPU5_UUID'), ('get("ARGGNN_PHYSICAL_GPU")=="0"', 'get("ARGGNN_PHYSICAL_GPU")=="5"'),
    ('physical GPU0', 'physical GPU5'),
    ('from .native_telemetry_v1 import memory_callback', 'from research.repairs.benchmark_recovery_20261020.scheduler.native_telemetry_v1 import memory_callback'),
    ('from .activation_checkpoint_v1 import install_native', 'from research.repairs.benchmark_recovery_20261020.scheduler.activation_checkpoint_v1 import install_native'),
    ('from .native_seal_v1 import native_top1_seal', 'from research.repairs.benchmark_recovery_20261020.scheduler.native_seal_v1 import native_top1_seal'),
])
for name in ('common.py', 'european_observations.py', 'european_execution.py', 'european_gru.py',
             'european_stage.py', 'qualification.py', 'american_prediction.py', 'empirical.py', 'american_workflow_v2.py'):
    source=OLD/'training/queue_adapters'/name
    replacements=[]
    if name=='common.py':
        replacements += [
            ('from arggnn.benchmarking.execution import GPU0_UUID', 'from ..shared_gpu5_v1 import GPU5_UUID'),
            ('research.repairs.benchmark_recovery_20261020.training.queue_adapters', PACKAGE+'.queue_adapters'),
            ('research/repairs/benchmark_recovery_20261020/training/queue_adapters', str(NEW/'queue_adapters')),
            ('GPU0_UUID','GPU5_UUID'), ('== "0"','== "5"'), ('physical GPU0','physical GPU5'),
        ]
    if name=='european_execution.py':
        replacements += [
            ('from ..layout import block_indices', 'from research.repairs.benchmark_recovery_20261020.training.layout import block_indices'),
            ('from ..decoder import geometry_from_spans', 'from research.repairs.benchmark_recovery_20261020.training.decoder import geometry_from_spans'),
            ('get("ARGGNN_PHYSICAL_GPU") == "0"', 'get("ARGGNN_PHYSICAL_GPU") == "5"'), ('physical GPU0','physical GPU5'),
        ]
    if name=='american_prediction.py':
        replacements += [
            ('from ..candidate_gru import install_chunked_gru', 'from research.repairs.benchmark_recovery_20261020.training.candidate_gru import install_chunked_gru'),
            ('from ..execution import StagedExecutionV1', 'from research.repairs.benchmark_recovery_20261020.training.execution import StagedExecutionV1'),
            ('from ..decoder import geometry_from_spans', 'from research.repairs.benchmark_recovery_20261020.training.decoder import geometry_from_spans'),
            ('from ...runtime.binding import ReadableTreeLoom', 'from research.repairs.benchmark_recovery_20261020.runtime.binding import ReadableTreeLoom'),
            ('GPU0 lease','GPU5 lease'),
        ]
    if name=='european_stage.py':replacements += [('GPU0 lease','GPU5 lease'),('shared GPU0 acquisition','shared GPU5 acquisition')]
    if replacements:port(source,'queue_adapters/'+name,replacements)
    else:write('queue_adapters/'+name,source.read_text(),source)
with (NEW/'port_lineage_v1.json').open('x') as stream:
    json.dump(dict(schema='gpu5_operational_source_port_v1', sources=lineage),stream,indent=2,sort_keys=True)
    stream.write('\n')
print(json.dumps(dict(created=len(lineage), directory=str(NEW)),sort_keys=True))
