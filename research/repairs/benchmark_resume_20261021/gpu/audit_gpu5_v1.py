"""Audit new GPU5 source, CPU mocks and read-only target-device inventory.

No benchmark, process signal, canonical policy mutation or CUDA context occurs.
"""
import ast
from datetime import datetime, timezone
import difflib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

from . import contracts_gpu5_v1 as c
from . import shared_gpu5_v1 as hardware
from . import queue_gpu5_v1 as pool
from . import test_gpu5_v1

HERE = Path(__file__).parent
OLD = c.ROOT/'research/repairs/benchmark_recovery_20261020'
PACKAGE = 'research.repairs.benchmark_resume_20261021.gpu'


def main():
    started = datetime.now(timezone.utc).isoformat()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_gpu5_v1)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    transcript = stream.getvalue()
    print(transcript, flush=True)
    with (HERE/'cpu_tests_v1.txt').open('x') as handle:
        handle.write(transcript)
    if not result.wasSuccessful():
        raise SystemExit('GPU5 CPU guard tests failed')
    modules = ['queue_reporting_v1','queue_native_admission_v1','queue_native_worker_v1',
        'queue_execute_v1','queue_support_v1','readable_workflow_v1',
        'native_scientific_gpu5_v1','queue_adapters.european_stage',
        'queue_adapters.american_prediction','queue_adapters.american_workflow_v2',
        'queue_adapters.empirical','queue_adapters.qualification']
    imported = []
    for name in modules:
        module = importlib.import_module(PACKAGE+'.'+name)
        imported.append(dict(module=module.__name__,source_ref=c.file_ref(module.__file__)))
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError('CPU audit must not initialize CUDA')
    inventory = hardware.gpu_sample()
    processes = pool.process_bytes()
    migration = json.loads((HERE/'port_lineage_v1.json').read_text())
    ancestors = {row['path']:row['original_path'] for row in migration['sources']}
    ancestors[str((HERE/'queue_reporting_v1.py').relative_to(c.ROOT))] = str((OLD/'scheduler/queue_reporting_v1.py').relative_to(c.ROOT))
    final = []
    for path in sorted(HERE.rglob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
        row = dict(source_ref=c.file_ref(path))
        ancestor = ancestors.get(str(path.relative_to(c.ROOT)))
        if ancestor:
            original = c.ROOT/ancestor
            row['original_ref'] = c.file_ref(original)
            diff = ''.join(difflib.unified_diff(original.read_text().splitlines(True),path.read_text().splitlines(True),
                fromfile=ancestor,tofile=str(path.relative_to(c.ROOT))))
            diff_path = HERE/'final_lineage'/(str(path.relative_to(HERE)).replace('/','__')+'.diff')
            diff_path.parent.mkdir(parents=True,exist_ok=True)
            with diff_path.open('x') as handle:handle.write(diff)
            row['diff_ref'] = c.file_ref(diff_path)
        final.append(row)
    operational = [row['source_ref'] for row in final
        if Path(row['source_ref']['path']).name not in ('test_gpu5_v1.py','audit_gpu5_v1.py','port_source_v1.py')]
    receipt = dict(schema='gpu5_operational_stack_CPU_inventory_audit_v1',started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(),status='PASS',
        tests_run=result.testsRun,test_failures=len(result.failures),test_errors=len(result.errors),
        test_transcript_ref=c.file_ref(HERE/'cpu_tests_v1.txt'),
        CUDA_initialized=False,benchmark_launched=False,processes_signalled=False,
        physical_gpu=hardware.PHYSICAL_GPU,gpu_uuid=hardware.GPU5_UUID,
        readonly_inventory=inventory,readonly_gpu_process_bytes=processes,imported=imported,
        operational_source_refs=operational,source_lineage=final,
        preserved_shared_capacity_lock=str(pool.CAPACITY.relative_to(c.ROOT)),
        physical_lease_path='.runtime/locks/gpu_5.lock',
        logical_pool_path=str(pool.POOL.relative_to(c.ROOT)),
        required_workflow_fields=dict(physical_gpu=5,logical_gpu_resource_name='gpu0'),
        required_policy_fields=dict(physical_gpus=[5],approved_readable_workflow_ref='exact current plan FileRef'),
        native_scientific_route=dict(original='arggnn.benchmarking.american_competitors:run_argformer',
            module=PACKAGE+'.native_scientific_gpu5_v1',function='run_argformer'),
        scope='Scheduler, native physical binding and TreeLoom queue adapters only; C4 ARGMix uses separate static successor')
    with (HERE/'audit_receipt_v1.json').open('x') as handle:
        json.dump(receipt,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(status='PASS',tests=result.testsRun,receipt_ref=c.file_ref(HERE/'audit_receipt_v1.json'),
        operational_sources=len(operational),readonly_inventory=inventory),sort_keys=True),flush=True)


if __name__=='__main__':main()
