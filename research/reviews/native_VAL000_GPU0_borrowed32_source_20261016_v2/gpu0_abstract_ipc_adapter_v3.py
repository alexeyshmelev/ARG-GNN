"""Exact private GPU0 eligibility adaptation for unchanged native abstract IPC.
No IPC installation, lease check, CUDA or project import happens on import.
"""
from pathlib import Path
import hashlib,importlib.util,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/gpu0_abstract_ipc_adapter_v3.py'
ORIGINAL_REF={'path':'research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py','sha256':'4550ead44f36eeb0fda8a817a2002082c3d067cf1a84fd817301532577d8b014'}
GPU0_UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667'
GUARD_SOURCE='research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py'
_PRIVATE_ALIAS='_native_GPU0_private_original_abstract_IPC_native32_v1'
def require(ok,message):
    if not ok:raise ValueError(message)
def _reference(path):
    p=ROOT/path
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=1048576,'Exact bounded canonical source')
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def validate_root_gpu_choice_native32_v1(*,physical_gpu,gpu_uuid):
    require(type(physical_gpu) is int and physical_gpu==0 and gpu_uuid==GPU0_UUID,'Only canonical physical GPU0 UUID')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    require(visible==GPU0_UUID and os.environ.get('ARGGNN_PHYSICAL_GPU')=='0','Original canonical GPU0 UUID visibility gate')
    guard=sys.modules.get('arggnn.native_argformer_exec')
    require(guard is not None and getattr(guard,'GPU0_ADAPTER_SOURCE',None)==GUARD_SOURCE,
        'Original exact GPU0 title/lease adapter must already be installed by native child')
    lease=guard.validate_lease();guard.apply_and_validate_title()
    require(lease['physical_gpu']==0 and lease['visible_gpu_uuid']==GPU0_UUID,'Actual inherited canonical GPU0 lease unchanged')
    return dict(schema='native_root_operational_GPU_choice_v1',physical_gpu=physical_gpu,
        root_authorized_gpu_uuid=gpu_uuid,CUDA_VISIBLE_DEVICES=visible,GPU_probe_performed=False,
        existing_actual_lease_and_CUDA_context_UUID_validation_required=True,
        original_allocation_config_science_and_charge_changed=False)
def install_private_module_before_native_child_native32_v1():
    require(sys.platform=='linux' and 'torch' not in sys.modules and 'torch_geometric' not in sys.modules,
        'Linux native interpreter only, before all Torch/author imports')
    require(Path(__file__).resolve()==ROOT/SOURCE and _reference(ORIGINAL_REF['path'])==ORIGINAL_REF,
        'Exact immutable qualified IPC source and actual adapter path')
    require('abstract_unix_bootstrap_v1' not in sys.modules and _PRIVATE_ALIAS not in sys.modules,
        'No stale IPC alias or duplicate installation')
    spec=importlib.util.spec_from_file_location(_PRIVATE_ALIAS,ROOT/ORIGINAL_REF['path'])
    original=importlib.util.module_from_spec(spec);sys.modules[_PRIVATE_ALIAS]=original;spec.loader.exec_module(original)
    # Only this private root-choice binding changes. Original bootstrap and
    # install_abstract_unix_v1 keep their original function code/globals.
    require(original.validate_root_gpu_choice_v1.__module__==_PRIVATE_ALIAS and original._ORIGINAL is None,
        'Unmodified private original eligibility and never-installed IPC state')
    original.validate_root_gpu_choice_v1=validate_root_gpu_choice_native32_v1
    sys.modules['abstract_unix_bootstrap_v1']=original
    require(original.bootstrap_native_execution_v1.__globals__['validate_root_gpu_choice_v1'] is validate_root_gpu_choice_native32_v1,
        'Actual unchanged bootstrap resolves only the reviewed GPU0 eligibility function')
    return dict(schema='native_GPU0_private_abstract_IPC_source_route_native32_v1',adapter_ref=_reference(SOURCE),
        original_source_ref=ORIGINAL_REF,actual_IPC_installation=False,
        only_private_root_choice_replaced=True,original_abstract_address_authkey_Listener_Client_and_fork_unchanged=True)
