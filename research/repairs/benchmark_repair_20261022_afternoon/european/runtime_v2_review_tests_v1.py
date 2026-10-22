"""Independent metadata-only authority and source regression checks; no launch."""
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone,timedelta
import copy
import hashlib
import json
import re
import sys
from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import contracts_recovery_v1 as c
from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import readable_workflow_v1 as workflow
from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import queue_gpu_v1 as gpu
from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import queue_execute_v1 as execution
from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import queue_native_admission_v1 as admission


def main():
    root=c.ROOT
    manifest_path=root/'research/repairs/benchmark_repair_20261022_afternoon/continuation_v2/runtime_manifest.json'
    expected_sha='9a19e8f6d9ba5fdf6d938f940cdd6d71a545d5310b7170234ec607289d4a8bae'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==expected_sha
    manifest=json.loads(manifest_path.read_text())
    canonical=c.read_json(c.POLICY_PATH)
    original_plan=c.read_ref(canonical['approved_recovery_workflow_ref'])
    pins={ref['path']:ref for ref in original_plan['sources']}
    for row in manifest['source_lineage']:
        assert pins[row['original_ref']['path']]==row['original_ref']
        for key in ('original_ref','successor_ref','diff_ref'):assert c.file_ref(row[key]['path'])==row[key]
    assert len(manifest['source_lineage'])==8
    fixture=copy.deepcopy(canonical)
    fixture.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        approved_recovery_continuation_workflow_ref=dict(path='synthetic/continuation.json',sha256='a'*64))
    fixture.pop('recovery_continuation_workflow_drain_ref',None)
    policy_ref=dict(path='synthetic/policy.json',sha256='b'*64)
    checks=[]
    with patch.object(c,'read_json',side_effect=lambda path:fixture),patch.object(c,'read_ref',side_effect=lambda ref:fixture),patch.object(c,'file_ref',return_value=policy_ref):
        assert c.check_execution_release(policy_ref)==fixture
        for key in ('approved_readable_workflow_ref','approved_parallel_workflow_ref','approved_recovery_workflow_ref','approved_recovery_continuation_workflow_ref'):
            saved=fixture.pop(key)
            try:c.check_execution_release(policy_ref)
            except c.ContractError:pass
            else:raise AssertionError('Missing approval accepted: '+key)
            fixture[key]=saved
    checks.append('All four approvals required, including the predecessor recovery approval')
    with patch.object(c,'check_execution_release',side_effect=lambda ref:fixture),patch.object(c,'file_ref',return_value=policy_ref):
        fixture['workflow_drain_ref']=fixture['approved_recovery_workflow_ref']
        fixture['recovery_workflow_drain_ref']=fixture['approved_recovery_workflow_ref']
        result=workflow.current_policy()
        assert result['approved_workflow_ref']==fixture['approved_recovery_continuation_workflow_ref']
        assert result['approved_recovery_workflow_ref']==canonical['approved_recovery_workflow_ref']
        assert 'workflow_drain_ref' not in result
        fixture['recovery_continuation_workflow_drain_ref']=fixture['approved_recovery_continuation_workflow_ref']
        assert workflow.current_policy()['workflow_drain_ref']==fixture['approved_recovery_continuation_workflow_ref']
    checks.append('Separate continuation authority and drain preserve predecessor fields')
    job=dict(plan_ref=fixture['approved_recovery_continuation_workflow_ref'],task=dict(physical_gpu=5,gpu_uuid='GPU-3db85505-d061-e45d-d154-e7e8c1cad497'))
    plan=dict(physical_gpus=[2,3,5])
    with patch.object(c,'read_ref',return_value=plan),patch.object(c,'file_ref',return_value=policy_ref),patch.object(c,'check_execution_release',return_value=fixture):
        assert gpu.validate_job_target(job)==plan
        wrong=copy.deepcopy(job);wrong['plan_ref']=canonical['approved_recovery_workflow_ref']
        try:gpu.validate_job_target(wrong)
        except RuntimeError:pass
        else:raise AssertionError('Predecessor authority accepted for continuation GPU job')
    checks.append('GPU authority requires exact continuation plan reference')
    assert admission.workflow is workflow and admission.gpu is gpu
    assert workflow.c is c and workflow.shared_gpu is gpu and workflow.runtime is execution
    assert execution.c is c and execution.gpu is gpu
    assert workflow.MODULE=='research.repairs.benchmark_repair_20261022_afternoon.runtime_v2.readable_workflow_v1'
    assert root/workflow.SOURCE==Path(workflow.__file__)
    native='research.repairs.benchmark_repair_20261022_afternoon.runtime_v2.queue_native_worker_v1'
    assert native in Path(execution.__file__).read_text()
    workflow.support.validate_native_command([str(root/'.runtime/env/bin/python'),'-B','-s','-m',native])
    directory=Path(workflow.__file__).parent
    assert 'from .queue_native_admission_v1 import validate_worker_lease' in (directory/'queue_native_worker_v1.py').read_text()
    checks.append('Native admission, authority modules and self source path bind the new runtime')
    for row in manifest['source_lineage']:
        text=(root/row['successor_ref']['path']).read_text()
        assert re.search(r'\b(?:exec|eval|compile)\s*\(',text) is None
        assert all(token not in text for token in ('b64decode','b64encode','marshal.loads','-EncodedCommand'))
    checks.append('No dynamic executable-source or encoded loader in the eight files')
    previous=directory.parent/'runtime'
    for name in ('queue_native_worker_v1.py','queue_native_admission_v1.py','overlay_predecessor_v1.py'):
        assert (previous/name).read_bytes()==(directory/name).read_bytes()
    support_ref=c.file_ref(workflow.support.__file__)
    assert pins[support_ref['path']]==support_ref
    support_source=Path(workflow.support.__file__).read_text()
    assert "full+new<=plan['max_parallel_rss_bytes']" in support_source
    assert "accounting.kernel_available_ram_bytes()-growth>=new+plan['host_ram_margin_bytes']" in support_source
    checks.append('Canonical support source and full cap/remaining growth RAM guards unchanged')
    assert not ('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==expected_sha
    receipt=dict(schema='independent_recovery_runtime_v2_source_review_v1',status='passed_no_concrete_blocker',
        runtime_manifest_ref=c.file_ref(manifest_path),old_runtime_approval_ref=canonical['approved_recovery_workflow_ref'],
        verified_source_lineage=manifest['source_lineage'],focused_metadata_checks=checks,test_source_ref=c.file_ref(__file__),
        eight_old_sources_match_canonical_pins=True,RAM_guard_code_unchanged=True,
        observed_delta_scope=['required continuation approval key added','continuation GPU authority key','module/self source/native worker path','continuation drain mapping','reporter markers and metadata filename'],
        canonical_policy_modified=False,runtime_source_files_modified=False,worker_or_controller_launched=False,
        scientific_stage_executed=False,CUDA_initialized=False,
        scope_limit='Source and mocked metadata review only; does not authorize launch or verify final plan adoption/storage reconciliation.')
    out=Path(__file__).parent/'runtime_v2_source_review_v1.json'
    with out.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(status=receipt['status'],receipt_ref=c.file_ref(out),checks=len(checks),sources=8),sort_keys=True),flush=True)


if __name__=='__main__':main()
