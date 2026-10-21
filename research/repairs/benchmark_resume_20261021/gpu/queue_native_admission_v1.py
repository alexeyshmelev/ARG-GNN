"""Ordinary admission adapter for the queue's static native numerical worker."""
import json
import os
from . import readable_workflow_v1 as workflow
from . import queue_gpu5_v1 as gpu


def validate_worker_lease(first_cuda=False):
    reference=json.loads(os.environ['ARGGNN_READABLE_JOB_REF'])
    job=workflow.job_spec(reference)
    gpu.validate_child(job,reference,first_cuda=first_cuda)
    return job
