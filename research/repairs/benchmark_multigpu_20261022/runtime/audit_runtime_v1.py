"""Run CPU mocks and read-only device inventory, then freeze source receipts.

No benchmark dispatch, CUDA call, active lock probe, policy mutation or signal.
"""
from datetime import datetime,timezone
import difflib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

from . import contracts_parallel_v1 as c
from . import shared_gpu_v1 as hardware
from . import queue_gpu_v1 as pool
from . import test_runtime_v1

HERE=Path(__file__).parent
OLD=c.ROOT/'research/repairs/benchmark_resume_20261021/gpu'


def main():
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_runtime_v1))
    transcript=stream.getvalue();print(transcript,flush=True)
    with (HERE/'cpu_tests_v1.txt').open('x') as handle:handle.write(transcript)
    if not result.wasSuccessful():raise SystemExit('Parallel runtime CPU checks failed')
    before=c.read_json(OLD/'audit_receipt_v1.json')
    for reference in before['operational_source_refs']:
        if c.file_ref(reference['path'])!=reference:raise ValueError('Original GPU5 operational source changed')
    inventory={str(index):hardware.gpu_sample(index) for index in (2,3,5)}
    device_processes={str(index):pool.process_bytes(index) for index in (2,3,5)}
    modules=['contracts_parallel_v1','shared_gpu_v1','queue_gpu_v1','queue_support_v1',
        'queue_execute_v1','queue_native_admission_v1','queue_native_worker_v1','readable_workflow_v1']
    references=[]
    for name in modules:
        module=importlib.import_module('research.repairs.benchmark_multigpu_20261022.runtime.'+name)
        references.append(c.file_ref(module.__file__))
    lineage=[]
    for name in ['queue_support_v1.py','queue_execute_v1.py','queue_native_admission_v1.py',
                 'queue_native_worker_v1.py','readable_workflow_v1.py']:
        source=OLD/name;destination=HERE/name
        diff=''.join(difflib.unified_diff(source.read_text().splitlines(True),destination.read_text().splitlines(True),
            fromfile=str(source.relative_to(c.ROOT)),tofile=str(destination.relative_to(c.ROOT))))
        path=HERE/'final_lineage'/(name+'.diff');path.parent.mkdir(exist_ok=True)
        with path.open('x') as handle:handle.write(diff)
        lineage.append(dict(original_ref=c.file_ref(source),successor_ref=c.file_ref(destination),diff_ref=c.file_ref(path)))
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():raise ValueError('CPU audit must not initialize CUDA')
    receipt=dict(schema='parallel_GPU2_3_5_runtime_CPU_inventory_audit_v1',status='PASS',
        completed_utc=datetime.now(timezone.utc).isoformat(),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),
        transcript_ref=c.file_ref(HERE/'cpu_tests_v1.txt'),operational_source_refs=references,
        validation_source_refs=[c.file_ref(HERE/'test_runtime_v1.py'),c.file_ref(__file__)],
        source_lineage=lineage,original_GPU5_source_refs_unchanged=True,
        readonly_inventory=inventory,readonly_device_process_bytes=device_processes,
        CUDA_initialized=False,benchmark_launched=False,processes_signalled=False,
        canonical_policy_modified=False,active_lock_probed=False,
        main_module='research.repairs.benchmark_multigpu_20261022.runtime.readable_workflow_v1',
        plan_fields=dict(physical_gpu=5,physical_gpus=[2,3,5]),task_device_fields=['physical_gpu','gpu_uuid'],
        preserved_primary_policy_fields=['physical_gpus=[5]','approved_readable_workflow_ref'],
        required_parallel_policy_fields=['additional_physical_gpus=[2,3]','user_authorized_physical_gpus=[2,3,5]','approved_parallel_workflow_ref'],
        capacity_lock=str(pool.CAPACITY.relative_to(c.ROOT)),
        physical_lease_paths={str(index):str(pool.physical_path(index).relative_to(c.ROOT)) for index in (2,3,5)},
        accounting_pool_paths={str(index):[str(path.relative_to(c.ROOT)) for path in pool.accounting_pool_directories(index)] for index in (2,3,5)},
        external_predecessor_contract='Reuse unchanged arggnn.benchmarking.workflow_predecessor_v5 for live GPU5 MLM: exact source plan/task/job/state/supervisor/session and caps; genuine original waits required before result adoption',
        source_only=True)
    path=HERE/'audit_receipt_v1.json'
    with path.open('x') as handle:json.dump(receipt,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(status='PASS',tests=result.testsRun,receipt_ref=c.file_ref(path),
        operational_sources=len(references),readonly_inventory=inventory),sort_keys=True),flush=True)


if __name__=='__main__':main()
