"""Create a separate, static recovery runtime without editing any predecessor."""
from pathlib import Path
import json
import hashlib
import difflib

OLD=Path('research/repairs/benchmark_multigpu_20261022/runtime')
NEW=Path('research/repairs/benchmark_repair_20261022_afternoon/runtime')
PACKAGE='research.repairs.benchmark_repair_20261022_afternoon.runtime'
lineage=[]


def port(source,name,replacements):
    text=source.read_text()
    for before,after in replacements:
        if before not in text:raise ValueError('Missing anchor '+str(source)+': '+before)
        text=text.replace(before,after)
    destination=NEW/name
    with destination.open('x') as handle:handle.write(text)
    diff=NEW/'lineage'/(name+'.diff');diff.parent.mkdir(exist_ok=True)
    with diff.open('x') as handle:handle.write(''.join(difflib.unified_diff(source.read_text().splitlines(True),text.splitlines(True),fromfile=str(source),tofile=str(destination))))
    lineage.append(dict(original=str(source),original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))


port(OLD/'contracts_parallel_v1.py','contracts_recovery_v1.py',[
    ('Separate parallel release validation preserving active GPU5 primary authority.','Separate recovery release validation preserving all healthy predecessor approvals.'),
    ("('approved_readable_workflow_ref','approved_parallel_workflow_ref')","('approved_readable_workflow_ref','approved_parallel_workflow_ref','approved_recovery_workflow_ref')"),
])
port(OLD/'readable_workflow_v2.py','readable_workflow_v1.py',[
    ('contracts_parallel_v1','contracts_recovery_v1'),
    ('from . import queue_support_v1 as support','from research.repairs.benchmark_multigpu_20261022.runtime import queue_support_v1 as support'),
    ('from . import queue_execute_v2 as runtime','from . import queue_execute_v1 as runtime'),
    ('research.repairs.benchmark_multigpu_20261022.runtime.readable_workflow_v2',PACKAGE+'.readable_workflow_v1'),
    ('research/repairs/benchmark_multigpu_20261022/runtime/readable_workflow_v2.py',str(NEW/'readable_workflow_v1.py')),
    ('approved_parallel_workflow_ref','approved_recovery_workflow_ref'),
    ('parallel_workflow_drain_ref','recovery_workflow_drain_ref'),
    ('from .queue_reporting_v2 import publish','from .queue_reporting_v1 import publish'),
    ('    by_id={t["id"]:t for t in plan["tasks"]}',
     '    total_ram=int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")))*1024\n    need(plan["host_ram_margin_bytes"]>=64*2**30 and plan["max_parallel_rss_bytes"]<=1792*2**30\n         and plan["max_parallel_rss_bytes"]+plan["host_ram_margin_bytes"]<=total_ram,"Physical RAM and original64GiB margin bound required")\n    by_id={t["id"]:t for t in plan["tasks"]}'),
])
port(OLD/'queue_execute_v2.py','queue_execute_v1.py',[
    ('contracts_parallel_v1','contracts_recovery_v1'),
    ('research.repairs.benchmark_multigpu_20261022.runtime.queue_native_worker_v2',PACKAGE+'.queue_native_worker_v1'),
])
port(OLD/'queue_native_worker_v2.py','queue_native_worker_v1.py',[
    ('from .shared_gpu_v1 import validate_visible_target','from research.repairs.benchmark_multigpu_20261022.runtime.shared_gpu_v1 import validate_visible_target'),
    ('from .queue_native_admission_v2 import validate_worker_lease','from .queue_native_admission_v1 import validate_worker_lease'),
])
port(OLD/'queue_native_admission_v2.py','queue_native_admission_v1.py',[
    ('from . import readable_workflow_v2 as workflow','from . import readable_workflow_v1 as workflow'),
])
port(OLD/'queue_gpu_v1.py','queue_gpu_v1.py',[
    ('from .shared_gpu_v1 import GPU_UUIDS, target, validate_visible_target, gpu_sample, identity',
     'from research.repairs.benchmark_multigpu_20261022.runtime.shared_gpu_v1 import GPU_UUIDS, target, validate_visible_target, gpu_sample, identity'),
    ('contracts_parallel_v1','contracts_recovery_v1'),
    ('approved_parallel_workflow_ref','approved_recovery_workflow_ref'),
])
port(OLD/'queue_reporting_v2.py','queue_reporting_v1.py',[
    ('readable-parallel-queue:','readable-recovery-afternoon-queue:'),
    ('benchmark_parallel_current.json','benchmark_recovery_afternoon_current.json'),
    ('### Current parallel queue','### Current recovery queue'),
])
with (NEW/'source_lineage_initial_v1.json').open('x') as handle:json.dump(lineage,handle,indent=2,sort_keys=True);handle.write('\n')
print(json.dumps(dict(created=len(lineage),directory=str(NEW)),sort_keys=True))
