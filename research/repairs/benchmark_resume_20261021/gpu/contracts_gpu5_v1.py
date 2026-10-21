"""Canonical release validation for the explicitly authorized GPU5 successor.

Metadata contracts are reused unchanged. Historical contracts and policies remain
immutable; this module grants no release and requires the current canonical one.
"""
from datetime import datetime, timezone
from arggnn.benchmarking.contracts import (
    ROOT, POLICY_PATH, ContractError, contained_path, file_ref, fingerprint,
    read_json, read_ref,
)


def check_execution_release(ref: dict) -> dict:
    """Keep every canonical approval gate and require physical GPU5 alone."""
    canonical = read_json(POLICY_PATH)
    if canonical.get("benchmark_execution_approved") is not True:
        raise ContractError("Benchmarking is paused pending explicit user approval")
    if ref != file_ref(POLICY_PATH):
        raise ContractError("Use the current canonical execution policy")
    policy = read_ref(ref)
    if policy.get("status") != "approved" or policy.get("physical_gpus") != [5]:
        raise ContractError("The execution policy must explicitly approve GPU5 only")
    if policy.get("approval_origin") != "explicit_user_go_ahead":
        raise ContractError("A new explicit user go-ahead is required")
    end = datetime.fromisoformat(policy["expires_at_utc"])
    if end.tzinfo is None or datetime.now(timezone.utc) >= end:
        raise ContractError("Execution approval has expired")
    if policy.get("smoke_tests_only") is not False:
        raise ContractError("Release smoke-only mode explicitly before benchmarking")
    return policy
