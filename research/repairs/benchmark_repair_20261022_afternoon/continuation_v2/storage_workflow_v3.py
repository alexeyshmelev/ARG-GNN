"""Explicit workflow entry with authenticated supplemental lossless-storage credit.

Original runtime files and scientific code remain unchanged. Supervisor and
scientific child commands re-enter this named module with visible JSON arguments.
Native numerical children never mutate workflow storage accounting.
"""
from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import readable_workflow_v1 as queue
from research.repairs.benchmark_repair_20261022_afternoon.storage_ap_train001_v1 import external_archive_v1 as archive

MODULE='research.repairs.benchmark_repair_20261022_afternoon.continuation_v2.storage_workflow_v3'
SOURCE='research/repairs/benchmark_repair_20261022_afternoon/continuation_v2/storage_workflow_v3.py'
_BASE_LOAD=queue.load_plan
_BASE_PROJECTION=queue.storage_projection


def load_plan(reference,*,approved=False):
    plan=_BASE_LOAD(reference,approved=approved)
    archive.validate_plan(plan)
    return plan


def storage_projection(plan,state,new_cap=0):
    original=_BASE_PROJECTION(plan,state,new_cap)
    credit=archive.completed_storage_reduction(plan,state)
    queue.need(type(credit) is int and 0<=credit<=original,'Only exact authenticated completed storage reduction is permitted')
    return original-credit


queue.MODULE=MODULE
queue.SOURCE=SOURCE
queue.load_plan=load_plan
queue.storage_projection=storage_projection


if __name__=='__main__':queue.main()
