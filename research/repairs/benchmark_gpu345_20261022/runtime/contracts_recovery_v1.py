"""Separate recovery release validation preserving all healthy predecessor approvals."""
from datetime import datetime, timezone
from arggnn.benchmarking.contracts import (
    ROOT, POLICY_PATH, ContractError, contained_path, file_ref, fingerprint,
    read_json, read_ref,
)


def check_execution_release(ref):
    canonical=read_json(POLICY_PATH)
    if canonical.get('benchmark_execution_approved') is not True:
        raise ContractError('Benchmarking is paused pending explicit user approval')
    if ref != file_ref(POLICY_PATH):
        raise ContractError('Use the current canonical execution policy')
    policy=read_ref(ref)
    if policy.get('status')!='approved':
        raise ContractError('The canonical execution policy must remain approved')
    if policy.get('currently_authorized_physical_gpus')!=[3,4,5]:
        raise ContractError('Only current explicit physical GPU3/4/5 authority is accepted')
    if policy.get('legacy_gpu_authority_status')!='preserved_for_running_workers_only':
        raise ContractError('Legacy GPU fields must remain explicit running-worker compatibility metadata')
    if policy.get('approval_origin')!='explicit_user_go_ahead':
        raise ContractError('Explicit user go-ahead is required')
    end=datetime.fromisoformat(policy['expires_at_utc'])
    if end.tzinfo is None or datetime.now(timezone.utc)>=end:
        raise ContractError('Execution approval has expired')
    if policy.get('smoke_tests_only') is not False:
        raise ContractError('Release smoke-only mode explicitly before benchmarking')
    for key in ('approved_readable_workflow_ref','approved_parallel_workflow_ref','approved_recovery_workflow_ref','approved_recovery_continuation_workflow_ref','approved_full_vram_workflow_ref','approved_gpu345_workflow_ref'):
        ref_value=policy.get(key)
        if not isinstance(ref_value,dict) or set(ref_value)!={'path','sha256'}:
            raise ContractError('Exact primary and parallel workflow approvals required')
    if policy.get('gpu_memory_policy') != 'full_device':
        raise ContractError('Explicit user full-device VRAM permission required')
    return policy
