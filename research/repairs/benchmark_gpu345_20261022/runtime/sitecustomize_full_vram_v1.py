"""Fixed plaintext startup for admitted native children with full-device VRAM."""
import os
import sys
import traceback

TITLE='AI Auto Research Pipeline Test'
SOURCE_ROOT='/workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/cu129/source/src'
CHILD_NAMES=('export_torchscript.py','run_inference.py','run_joint_inference.py')


def admit_child():
    import setproctitle
    setproctitle.setproctitle(TITLE)
    name=os.path.basename(sys.argv[0])
    if name not in CHILD_NAMES:return
    if os.path.realpath(sys.argv[0])!=os.path.join(SOURCE_ROOT,name):
        raise RuntimeError('Native child script must retain its admitted source origin')
    from research.repairs.benchmark_gpu345_20261022.runtime.queue_gpu_v1 import validate_policy_environment
    from research.repairs.benchmark_gpu345_20261022.runtime.queue_native_admission_v1 import validate_worker_lease
    validate_policy_environment()
    validate_worker_lease()
    device=sys.argv[sys.argv.index('--device')+1]
    if name=='export_torchscript.py':
        if device!='cpu':raise RuntimeError('Native TorchScript export must stay CPU-only')
        return
    if device!='cuda':raise RuntimeError('Native GPU inference must retain its declared device')
    validate_worker_lease(first_cuda=True)
    import torch
    if torch.cuda.device_count()!=1:
        raise RuntimeError('One inherited admitted CUDA device required')
    if torch.cuda.get_device_properties(0).total_memory<=0:
        raise RuntimeError('Actual admitted physical CUDA memory required')


if __name__=='sitecustomize':
    try:admit_child()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        os._exit(78)
