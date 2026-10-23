"""Current campaign GPU identity checks; Root separately owns actual leases."""
from __future__ import annotations

import ctypes
import os

ALLOWED_PHYSICAL_GPUS=(3,4,5)


def nvml_uuid(index):
    library=ctypes.CDLL('libnvidia-ml.so.1')
    def need(code):
        if code!=0:
            raise RuntimeError('NVML identity query failed with code '+str(code))
    need(library.nvmlInit_v2())
    try:
        handle=ctypes.c_void_p()
        need(library.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index),ctypes.byref(handle)))
        value=ctypes.create_string_buffer(96)
        need(library.nvmlDeviceGetUUID(handle,value,ctypes.c_uint(len(value))))
        return value.value.decode('ascii')
    finally:
        need(library.nvmlShutdown())


def validate_gpu_binding(*,gpu_uuid=None,gpu_uuid_map=None):
    raw=os.environ.get('ARGGNN_PHYSICAL_GPU','')
    if raw not in tuple(map(str,ALLOWED_PHYSICAL_GPUS)):
        raise ValueError('Current campaign permits only physical GPU3,4,5')
    physical=int(raw)
    if os.environ.get('ARGGNN_GPU_MEMORY_POLICY')!='full_device':
        raise ValueError('Current campaign full-device GPU memory policy required')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    expected=gpu_uuid or visible
    if not (expected.startswith('GPU-') and visible==expected):
        raise ValueError('Expose one explicit campaign GPU UUID as logical cuda0')
    if gpu_uuid_map is not None and gpu_uuid_map.get(str(physical))!=expected:
        raise ValueError('Requested physical GPU and campaign UUID mapping differ')
    actual=nvml_uuid(physical)
    if actual!=expected:
        raise ValueError('Live NVML physical GPU UUID differs from campaign binding')
    return dict(physical_gpu=physical,gpu_uuid=actual,logical_cuda_device=0,
                lease_authority='Root campaign executor; UUID check does not create a lease')
