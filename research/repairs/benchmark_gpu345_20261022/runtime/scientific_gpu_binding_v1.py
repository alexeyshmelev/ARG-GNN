"""Explicit authorized per-job GPU identity checks before CUDA initialization."""
import os
import subprocess

GPU_UUIDS = {
    4: 'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
    3: 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
    5: 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
}


def target(physical_gpu, gpu_uuid):
    if type(physical_gpu) is not int or GPU_UUIDS.get(physical_gpu) != gpu_uuid:
        raise ValueError('Exact authorized physical GPU 3, 4 or 5 and UUID required')
    return dict(physical_gpu=physical_gpu, gpu_uuid=gpu_uuid)


def validate_env(spec=None, *, acquired=True):
    if spec is None:
        value=os.environ.get('ARGGNN_PHYSICAL_GPU', '')
        if value not in ('3', '4', '5'):
            raise ValueError('An explicit admitted physical GPU is required')
        spec=target(int(value), os.environ.get('CUDA_VISIBLE_DEVICES'))
    expected=target(spec['physical_gpu'], spec['gpu_uuid'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected['gpu_uuid']:
        raise ValueError('Visible CUDA UUID must match the exact per-job GPU target')
    if acquired:
        if os.environ.get('ARGGNN_PHYSICAL_GPU') != str(expected['physical_gpu']):
            raise ValueError('Acquired physical GPU must match the exact per-job target')
        descriptor=os.environ.get('ARGGNN_READABLE_GPU_FD', '')
        if not descriptor.isdigit():
            raise ValueError('Actual inherited shared GPU lease descriptor required')
        os.fstat(int(descriptor))
    elif ('ARGGNN_PHYSICAL_GPU' in os.environ or 'ARGGNN_READABLE_GPU_FD' in os.environ
          or os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK') != '1'):
        raise ValueError('Deferred observed preparation must start before GPU acquisition')
    return expected


def verify_before_cuda(torch, spec):
    expected=validate_env(spec)
    if torch.cuda.is_initialized():
        raise ValueError('GPU identity must be verified before CUDA initialization')
    result=subprocess.run(['nvidia-smi', '-i', str(expected['physical_gpu']),
        '--query-gpu=uuid', '--format=csv,noheader,nounits'],
        check=True, text=True, capture_output=True, timeout=15)
    if result.stdout.strip() != expected['gpu_uuid']:
        raise ValueError('Actual physical GPU UUID does not match the per-job target')
    return expected
